"""The verify window of a relayed round, on the real request table and allocator.

Prepare once per round from the committed length; the verify's window and the tail
draft's come out of the same reserve on every stage, at any accept length.
"""

import importlib.util
import random
import unittest
from array import array
from pathlib import Path
from unittest import mock

import torch
from triton.runtime.jit import JITFunction

from sglang.kernels.ops.speculative import cache_locs
from sglang.kernels.ops.speculative.reject_sampling import (
    speculative_sampling_classic_kernel,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dspark_components.dspark_planner import (
    VerifyWindow,
    alloc_verify_window,
)
from sglang.srt.speculative.dspark_components.dspark_pp_relay import (
    DSparkPPRelayInput,
    clamp_window_to_context,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

WIDTH = 6
PAGE = 256


def _engine_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "mem_cache"
        / "test_swa_passed_tail_demotion.py"
    )
    spec = importlib.util.spec_from_file_location("dsv41_tail_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stage:
    """One stage's request table and allocator, through the calls a stage makes."""

    def __init__(self, module, prompt_len):
        self.engine = module._Engine({}, swa_slots=65536)
        self.prompt_len = prompt_len

    def admit(self, rid):
        engine = self.engine
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", range(1, self.prompt_len + 1)),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=4096),
        )
        req.output_ids = array("q", [11])
        req.extra_key = None
        req.lock = None
        req.init_next_round_input(engine.cache)
        req.extend_end = self.prompt_len
        req.lock = engine.cache.lock(req.last_node)
        engine.pool.alloc([req])
        engine._extend(req, req.prefix_len, self.prompt_len)
        return req

    def batch(self, reqs):
        engine = self.engine
        batch = ScheduleBatch(reqs=list(reqs))
        batch.device = "cpu"
        batch.spec_algorithm = SpeculativeAlgorithm.DSPARK
        batch.tree_cache = engine.cache
        batch.req_to_token_pool = engine.pool
        batch.token_to_kv_pool_allocator = engine.alloc
        batch.hisparse_coordinator = None
        batch.forward_mode = None
        batch.req_pool_indices = torch.tensor(
            [r.kv.req_pool_idx for r in reqs], dtype=torch.int64
        )
        batch.seq_lens = torch.tensor(
            [r.kv.kv_committed_len for r in reqs], dtype=torch.int64
        )
        batch.maybe_evict_swa = lambda: None
        return batch

    def runner_ids(self):
        engine = self.engine
        return RunnerIds(
            req_to_token=engine.pool.req_to_token,
            token_to_kv_pool_allocator=engine.alloc,
            token_to_kv_pool=engine.alloc.get_kvcache(),
            page_size=PAGE,
            device="cpu",
        )


class _Runner:
    def __init__(self, runner_ids):
        self.runner_ids = runner_ids


def _window(batch, runner_ids, seq_lens_cpu):
    bs = batch.seq_lens.numel()
    return alloc_verify_window(
        batch=batch,
        bs=bs,
        device="cpu",
        verify_num_draft_tokens=WIDTH,
        block_pos_offsets=torch.arange(WIDTH, dtype=torch.int64),
        model_runner=_Runner(runner_ids),
        seq_lens_cpu=seq_lens_cpu,
    )


@unittest.skipUnless(
    torch.cuda.device_count() > 0
    or not isinstance(speculative_sampling_classic_kernel, JITFunction),
    "the table's gathers are Triton: needs CUDA or TRITON_INTERPRET=1 from the start",
)
class TestRelayWindow(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _engine_module()

    def setUp(self):
        # The gather's Triton kernel, under the interpreter when no device is visible.
        patcher = mock.patch.object(cache_locs, "_is_cuda", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _publish_dspark(self):
        """After the engines: each publishes its own plain config when it is built."""
        get_context().override(
            "test.dspark_window",
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=WIDTH,
            speculative_num_steps=1,
            speculative_eagle_topk=1,
        )

    def _rounds(self, *, seed, prompt_len, rounds):
        """Three stages walk the same accepts; returns the windows they planned."""
        rng = random.Random(seed)
        stages = [_Stage(self.module, prompt_len) for _ in range(3)]
        self._publish_dspark()
        reqs = [s.admit("a") for s in stages]
        batches = [s.batch([r]) for s, r in zip(stages, reqs)]
        ids = [s.runner_ids() for s in stages]
        relays = [
            DSparkPPRelayInput.degenerate(["a"], torch.tensor([11]), WIDTH)
            for _ in stages
        ]
        for _ in range(rounds):
            planned = []
            for batch, relay, runner_ids, req in zip(batches, relays, ids, reqs):
                relay.prepare_for_decode(batch)
                self.assertEqual(int(batch.seq_lens_cpu[0]), req.kv.kv_committed_len)
                planned.append(_window(batch, runner_ids, batch.seq_lens_cpu))
            for other in planned[1:]:
                self.assertTrue(
                    torch.equal(planned[0].verify_cache_loc, other.verify_cache_loc)
                )
                self.assertTrue(
                    torch.equal(planned[0].positions_2d, other.positions_2d)
                )
            yield stages, batches, reqs, ids, planned, rng.randrange(1, WIDTH + 1)
            for batch, req in zip(batches, reqs):
                batch.seq_lens = batch.seq_lens + self._k
                req.kv.commit_decode(self._k)

    def test_same_seq_lens_same_slots_and_tail_draft_fits_the_reserve(self):
        """Invariant 8: real table, real allocator, across page boundaries."""
        for seed, prompt_len in ((0, 500), (1, 251), (2, 255), (3, 256), (4, 4000)):
            for stages, batches, reqs, ids, planned, k in self._rounds(
                seed=seed, prompt_len=prompt_len, rounds=120
            ):
                self._k = k
                req, batch = reqs[0], batches[0]
                p = int(batch.seq_lens[0])
                self.assertEqual(
                    planned[0].positions_2d[0].tolist(), list(range(p, p + WIDTH))
                )
                self.assertEqual(
                    planned[0].verify_cache_loc.tolist(),
                    stages[0]
                    .engine.pool.req_to_token[req.kv.req_pool_idx, p : p + WIDTH]
                    .tolist(),
                )
                # The tail draft's window after this accept, before the next prepare:
                # all of it allocated, whatever the accept length was.
                for accept in range(1, WIDTH + 1):
                    self.assertLessEqual(p + accept + WIDTH, req.kv.kv_allocated_len)
                    batch_after = stages[0].batch([req])
                    batch_after.seq_lens = torch.tensor([p + accept], dtype=torch.int64)
                    tail = _window(
                        batch_after,
                        ids[0],
                        torch.tensor([p + accept], dtype=torch.int64),
                    )
                    self.assertTrue((tail.verify_cache_loc >= 0).all())
                    self.assertEqual(
                        len(set(tail.verify_cache_loc.tolist())),
                        WIDTH,
                        "a row of the tail draft shares a slot",
                    )
                self.assertTrue((planned[0].verify_cache_loc >= 0).all())

    def test_relay_prepare_for_decode_survives_filter_and_merge(self):
        """Invariant 9: the state the DFlash family's prep keeps, through composition."""
        stage = _Stage(self.module, 100)
        self._publish_dspark()
        a, b = stage.admit("a"), stage.admit("b")
        batch = stage.batch([a, b])
        relay = DSparkPPRelayInput(
            ["a", "b"],
            torch.zeros((2, WIDTH), dtype=torch.int64),
            has_chain=[True, False],
        )
        relay.prepare_for_decode(batch)
        self.assertEqual(relay.alloc_state.nxt_kv_lens_cpu.tolist(), [256, 256])
        self.assertEqual(a.kv.kv_allocated_len, 256)
        relay.filter_batch(torch.tensor([1]), [1])
        self.assertIsNone(relay.alloc_state.nxt_kv_lens_cpu)
        self.assertEqual(relay.rids, ["b"])
        c = stage.admit("c")
        relay.merge_batch(
            DSparkPPRelayInput.degenerate(["c"], torch.tensor([3]), WIDTH)
        )
        batch2 = stage.batch([b, c])
        relay.prepare_for_decode(batch2)
        self.assertEqual(batch2.seq_lens_cpu.tolist(), [100, 100])
        self.assertEqual(relay.alloc_state.nxt_kv_lens_cpu.tolist(), [256, 256])
        self.assertEqual(c.kv.kv_allocated_len, 256)

    def test_context_end(self):
        """Invariant 24: rows at context_len - 1 stay inside req_to_token and the
        position table. Extra rows are the headroom the pool gives every request."""
        _Stage(self.module, 1)
        self._publish_dspark()
        extra = get_req_to_token_extra_context_len()
        self.assertGreaterEqual(extra, 2 * WIDTH + PAGE - 1)
        limit = 1024
        table_rows = limit + extra
        for p in range(limit - 8, limit):
            for offset in (WIDTH, 2 * WIDTH - 1):
                self.assertLessEqual(p + offset, table_rows)
        positions = torch.tensor([[limit - 2 + i for i in range(WIDTH)]])
        window = VerifyWindow(
            positions_2d=positions,
            verify_cache_loc=positions,
            verify_cache_loc_2d=positions,
            iteration_ids=None,
        )
        clamped = clamp_window_to_context(
            window, torch.tensor([limit - 2]), WIDTH, limit
        )
        self.assertEqual(int(clamped.positions_2d.max()), limit - 1)
        self.assertEqual(clamped.positions_2d[0, :2].tolist(), [limit - 2, limit - 1])
        # Well inside, the same object: no copy, no write, on every normal round.
        self.assertIs(
            clamp_window_to_context(window, torch.tensor([limit - 8]), WIDTH, limit),
            window,
        )
        self.assertFalse(envs.SGLANG_PP_DSPARK_LAUNCH_FIRST.get())


if __name__ == "__main__":
    unittest.main()
