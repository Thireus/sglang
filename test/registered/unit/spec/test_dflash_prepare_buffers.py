"""DFlash spec-v2's prepare buffers on a draft input that lives across rounds.

Upstream's decode builds a fresh DFlashDraftInputV2 every step, so its prepare
buffers are allocated once and dropped. DSpark on the pipeline keeps one for a
micro-batch's life (DSparkPPRelayInput.alloc_state), and prepare_for_decode reaches
the same buffers every round. The batch names its device without an index ("cuda",
the runner's device) while a buffer allocated there reports one ("cuda:0"): compared
as strings they differed, every round reallocated both device buffers at twice the
size, and the d48 boot row's stage 0 asked for 2 GiB on its 25th round.

CPU only. "cpu:0" asked against a tensor that reports "cpu" is the same pair of
names for one device; a stand-in buffer that reports cuda:0 checks the CUDA names
without a GPU.
"""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import dflash_info_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_RELAY = "sglang.srt.speculative.dspark_components.dspark_pp_relay"
_HAS_PP_DSPARK = importlib.util.find_spec(_RELAY) is not None


def _fresh():
    return DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))


def _device_buffers(info):
    return info._prepare_cur_kv_lens_gpu_buf, info._prepare_nxt_kv_lens_gpu_buf


class _StandIn:
    """What the prepare reads of a device buffer: its size and its device."""

    def __init__(self, numel, device):
        self._numel = numel
        self.device = torch.device(device)

    def numel(self):
        return self._numel


class TestSameDevice(CustomTestCase):
    def test_a_name_without_index_is_the_current_device(self):
        same = dflash_info_v2._same_device
        with mock.patch.object(torch.cuda, "current_device", return_value=0):
            self.assertTrue(same(torch.device("cuda", 0), "cuda"))
            self.assertTrue(same("cuda", torch.device("cuda", 0)))
            self.assertFalse(same(torch.device("cuda", 1), "cuda"))
        with mock.patch.object(torch.cuda, "current_device", return_value=1):
            self.assertTrue(same(torch.device("cuda", 1), "cuda"))
            self.assertFalse(same(torch.device("cuda", 0), "cuda"))

    def test_explicit_indexes_and_types_need_no_current_device(self):
        same = dflash_info_v2._same_device
        unused = AssertionError("the current device was asked for")
        with mock.patch.object(torch.cuda, "current_device", side_effect=unused):
            self.assertTrue(same(torch.device("cuda", 0), "cuda:0"))
            self.assertTrue(same("cuda", "cuda"))
            self.assertFalse(same(torch.device("cuda", 0), "cuda:1"))
            self.assertFalse(same(torch.device("cpu"), "cuda"))
            self.assertFalse(same(torch.device("cuda", 0), "cpu"))
            self.assertTrue(same(torch.device("cpu"), "cpu:0"))
            self.assertTrue(same(torch.empty(1).device, "cpu"))


class TestPrepareLengthBuffers(CustomTestCase):
    def test_a_name_without_index_keeps_the_buffers(self):
        info = _fresh()
        info._ensure_prepare_length_buffers(4, "cpu:0")
        first = _device_buffers(info)
        self.assertEqual(str(first[0].device), "cpu")
        self.assertEqual(first[0].numel(), 32)
        for _ in range(40):
            info._ensure_prepare_length_buffers(4, "cpu:0")
            self.assertIs(info._prepare_cur_kv_lens_gpu_buf, first[0])
            self.assertIs(info._prepare_nxt_kv_lens_gpu_buf, first[1])

    def test_the_batch_cuda_name_against_buffers_on_cuda_0(self):
        info = _fresh()
        info._ensure_prepare_length_buffers(4, "cpu")  # the host buffers, 32 each
        cur, nxt = _StandIn(32, "cuda:0"), _StandIn(32, "cuda:0")
        info._prepare_cur_kv_lens_gpu_buf = cur
        info._prepare_nxt_kv_lens_gpu_buf = nxt
        no_pin = mock.patch.object(
            dflash_info_v2, "is_pin_memory_available", return_value=False
        )
        with mock.patch.object(torch.cuda, "current_device", return_value=0), no_pin:
            for _ in range(40):
                info._ensure_prepare_length_buffers(4, "cuda")
        self.assertIs(info._prepare_cur_kv_lens_gpu_buf, cur)
        self.assertIs(info._prepare_nxt_kv_lens_gpu_buf, nxt)

    def test_a_device_move_reallocates_what_the_batch_needs(self):
        info = _fresh()
        info._ensure_prepare_length_buffers(4, "cpu")
        info._prepare_cur_kv_lens_gpu_buf = _StandIn(128, "cuda:1")
        info._prepare_nxt_kv_lens_gpu_buf = _StandIn(128, "cuda:1")
        info._ensure_prepare_length_buffers(4, "cpu")
        for buf in _device_buffers(info):
            self.assertEqual((buf.device.type, buf.numel()), ("cpu", 32))

    def test_only_the_batch_size_grows_the_buffers(self):
        sizes = [4, 32, 33, 64, 65, 200, 7, 200]
        want = [32, 32, 64, 64, 128, 256, 256, 256]
        for name in ("cpu", "cpu:0"):
            with self.subTest(device=name):
                info = _fresh()
                device_caps, host_caps = [], []
                for bs in sizes:
                    info._ensure_prepare_length_buffers(bs, name)
                    cur, nxt = _device_buffers(info)
                    self.assertEqual(cur.numel(), nxt.numel())
                    device_caps.append(cur.numel())
                    host_caps.append(info._prepare_batch_seq_lens_cpu_buf.numel())
                self.assertEqual(device_caps, want)
                self.assertEqual(host_caps, want)


@unittest.skipUnless(_HAS_PP_DSPARK, "no DSpark under pipeline parallelism here")
class TestRelayRoundsOnOneDraftInput(CustomTestCase):
    """The relay's rounds as the d48 row ran them: one alloc_state through adopt,
    reindex, merge and filter, the real prepare_for_decode on it every round, and
    the batch's device named without an index. Only the KV sizing and allocation
    are stand-ins."""

    WIDTH = 6

    def _rows(self, rids, fill):
        from sglang.srt.speculative.dspark_components.dspark_pp_relay import (
            DSparkPPRelayInput,
        )

        return DSparkPPRelayInput(
            rids=list(rids),
            tokens=torch.full((len(rids), self.WIDTH), fill, dtype=torch.int64),
            has_chain=[True] * len(rids),
        )

    def _batch(self, reqs):
        batch = SimpleNamespace(
            reqs=reqs,
            device="cpu:0",
            maybe_evict_swa=lambda: None,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
            tree_cache=None,
            req_to_token_pool=None,
            req_pool_indices=None,
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )
        batch.batch_size = lambda: len(batch.reqs)
        return batch

    def test_forty_rounds_keep_one_pair_of_buffers(self):
        allocated = []

        def alloc_lens(reqs, *, reserve, page_size):
            cur = [int(r.kv.kv_committed_len) for r in reqs]
            return cur, [c + reserve for c in cur], 0

        def alloc_for_spec_decode(tree_cache, req_to_token_pool, **kw):
            allocated.append((len(kw["reqs"]), kw["cur_kv_lens"].numel()))

        spec = SimpleNamespace(speculative_num_draft_tokens=self.WIDTH)
        for target, value in (
            ("get_spec", lambda: spec),
            ("page_aligned_decode_alloc_lens", alloc_lens),
            ("alloc_for_spec_decode", alloc_for_spec_decode),
        ):
            patcher = mock.patch.object(dflash_info_v2, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        reqs = {
            rid: SimpleNamespace(
                rid=rid,
                kv=SimpleNamespace(kv_committed_len=100 + i),
                sampling_params=SimpleNamespace(top_k=1),
                decode_batch_idx=0,
            )
            for i, rid in enumerate("abc")
        }
        live = ["a", "b"]
        relay = self._rows(live, 1)
        state = relay.alloc_state
        first = None
        for round_ in range(40):
            relay.prepare_for_decode(self._batch([reqs[r] for r in live]))
            self.assertIs(relay.alloc_state, state)
            buffers = _device_buffers(state)
            first = first or buffers
            self.assertIs(buffers[0], first[0], f"round {round_}")
            self.assertIs(buffers[1], first[1], f"round {round_}")
            relay.adopt(self._rows(live, round_ + 2))  # the next chain, round the ring
            if round_ == 10:
                live = ["b", "a"]
                relay = relay.reindex(live)
            elif round_ == 20:
                relay.merge_batch(self._rows(["c"], 0))
                live = ["b", "a", "c"]
            elif round_ == 30:
                relay.filter_batch(torch.tensor([0, 2]), [0, 2])
                live = ["b", "c"]
        self.assertEqual(first[0].numel(), 32)
        self.assertEqual(len(allocated), 40)
        self.assertEqual(reqs["b"].decode_batch_idx, 40)


if __name__ == "__main__":
    unittest.main()
