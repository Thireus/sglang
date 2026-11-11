"""A plain round under DSpark: what it hands the forward, what it commits, what it relays."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.dspark_components.dspark_pp_relay import (
    DSparkPPRelayInput,
    prepare_plain_round,
)
from sglang.srt.speculative.dspark_components.dspark_pp_round import DSparkPPLastStage
from sglang.srt.speculative.dspark_components.dspark_pp_stash import DsparkPPStash
from sglang.srt.speculative.dspark_components.dspark_switch import DsparkPPSwitch
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BS, WIDTH, HIDDEN, VOCAB = 2, 6, 8, 16


class _Plan:
    def __init__(self, write_virtual):
        self.write_virtual = write_virtual

    def write_ids(self, runner_ids, cols=None):
        return self.write_virtual


class _RunnerIds:
    def __init__(self):
        self.calls = []

    def plan_uniform_extend(self, *, req_pool_indices, seq_lens, seq_lens_cpu, width):
        self.calls.append(width)
        return _Plan(seq_lens.to(torch.int64) + 1000)


def _batch(**extra):
    fields = dict(
        reqs=[SimpleNamespace(rid="a"), SimpleNamespace(rid="b")],
        seq_lens=torch.tensor([10, 20], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int64),
        seq_lens_sum=30,
        orig_seq_lens=torch.tensor([10, 20], dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
        input_ids=None,
        out_cache_loc=None,
        forward_mode=ForwardMode.DECODE,
        spec_info="relay",
        sampling_info=None,
        return_logprob=False,
        has_grammar=False,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def _relay():
    return DSparkPPRelayInput(
        ["a", "b"],
        torch.tensor([[5, 0, 0, 0, 0, 0], [6, 7, 8, 9, 1, 2]], dtype=torch.int64),
        has_chain=[False, True],
    )


class _Injector:
    swa_ring_size = None

    def __init__(self):
        self.calls = []

    def ids_for(self, plan, cols=None):
        return plan.write_ids(None)

    def inject_target_hidden(self, **kwargs):
        self.calls.append(kwargs)


class _Engram:
    def __init__(self):
        self.commits = 0

    def update_after_verify(self, **kwargs):
        self.commits += 1


def _stage(warm=True):
    injector, engram = _Injector(), _Engram()
    seen = {}

    def forward(batch, pp_proxy_tensors=None, capture_hidden_mode=None):
        seen.update(
            mode=batch.forward_mode,
            input_ids=batch.input_ids.clone(),
            seq_lens=batch.seq_lens.clone(),
            seq_lens_cpu=batch.seq_lens_cpu.clone(),
            out_cache_loc=batch.out_cache_loc.clone(),
            spec_info=batch.spec_info,
            proxies=pp_proxy_tensors,
            capture=capture_hidden_mode,
        )
        return SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=torch.ones(BS, HIDDEN),
                next_token_logprobs=torch.tensor([-0.5, -1.5]),
            ),
            next_token_ids=torch.tensor([3, 4], dtype=torch.int64),
            can_run_cuda_graph=True,
        )

    worker = SimpleNamespace(
        verify_num_draft_tokens=WIDTH,
        gamma=WIDTH - 1,
        model_runner=SimpleNamespace(
            runner_ids=_RunnerIds(),
            ngram_embedding_manager=engram,
            model_config=SimpleNamespace(
                context_len=1 << 20,
                hf_text_config=SimpleNamespace(max_position_embeddings=1 << 20),
            ),
        ),
        target_worker=SimpleNamespace(forward_batch_generation=forward),
        _tp_sync=SpecTpSync(SimpleNamespace(world_size=1)),
        _kv_injector=injector,
        _draft_context=nullcontext,
        device="cpu",
    )
    stage = DSparkPPLastStage(
        worker,
        stash=DsparkPPStash(capacity=4),
        switch=DsparkPPSwitch(mode="off"),
    )
    stage.warm_when_off = warm
    return stage, injector, engram, seen


class TestPlainRound(CustomTestCase):
    def test_prepare_plain_round_is_the_decode_shape(self):
        batch = _batch()
        runner_ids = _RunnerIds()
        plan = prepare_plain_round(batch, _relay(), runner_ids)
        self.assertEqual(runner_ids.calls, [1])
        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        self.assertEqual(batch.input_ids.tolist(), [5, 6])
        self.assertEqual(batch.seq_lens.tolist(), [11, 21])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [11, 21])
        self.assertEqual(batch.orig_seq_lens.tolist(), [11, 21])
        self.assertIsNone(batch.seq_lens_sum)
        self.assertIsNone(batch.spec_info)
        self.assertEqual(batch.out_cache_loc.tolist(), [1010, 1020])
        self.assertIs(batch.out_cache_loc, plan.write_virtual)

    def test_engram_commits_once_and_window_one_row(self):
        """Invariant 28: the forward committed the Engram history, so the relay does
        not; the draft window commits one row, cache_loc_2d of shape (bs, 1)."""
        stage, injector, engram, seen = _stage()
        relay = _relay()
        proxies = object()
        result = stage.forward_round(
            _batch(),
            relay,
            plain=True,
            pp_proxy_tensors=proxies,
            mb_id=0,
            decoding_requests=2,
        )
        self.assertEqual(engram.commits, 0)
        self.assertEqual(seen["mode"], ForwardMode.DECODE)
        self.assertEqual(seen["capture"], CaptureHiddenMode.FULL)
        self.assertIs(seen["proxies"], proxies)
        self.assertIsNone(seen["spec_info"])
        self.assertEqual(len(injector.calls), 1)
        call = injector.calls[0]
        self.assertEqual(call["cache_loc_2d"].shape, (BS, 1))
        self.assertEqual(call["commit_lens"].tolist(), [1, 1])
        self.assertEqual(call["positions"].tolist(), [10, 20])
        self.assertEqual(call["target_hidden"].shape, (BS, HIDDEN))
        self.assertEqual(result.accept_lens.tolist(), [1, 1])
        self.assertEqual(result.next_token_ids.tolist(), [3, 4])
        self.assertEqual(result.new_seq_lens.tolist(), [11, 21])
        self.assertEqual(result.speculative_output_stride, 1)
        self.assertEqual(result.speculative_num_draft_tokens, WIDTH)
        self.assertIsNone(result.next_verify_chain)
        self.assertEqual(result.next_draft_input.bonus_tokens.tolist(), [3, 4])
        self.assertTrue(result.can_run_cuda_graph)

    def test_cold_window_commits_nothing(self):
        stage, injector, _, seen = _stage(warm=False)
        stage.forward_round(
            _batch(),
            _relay(),
            plain=True,
            pp_proxy_tensors=None,
            mb_id=0,
            decoding_requests=2,
        )
        self.assertEqual(injector.calls, [])
        self.assertEqual(seen["capture"], CaptureHiddenMode.NULL)

    def test_logprobs_take_the_speculative_row_shape(self):
        """The result processor reads a list per request under a speculative algorithm."""
        stage, *_ = _stage()
        result = stage.forward_round(
            _batch(return_logprob=True),
            _relay(),
            plain=True,
            pp_proxy_tensors=None,
            mb_id=0,
            decoding_requests=2,
        )
        self.assertEqual(
            result.logits_output.next_token_logprobs.tolist(), [[-0.5], [-1.5]]
        )

    def test_no_chain_rows_run_plain_and_logprob_batches_too(self):
        batch = _batch()
        none = DSparkPPRelayInput.degenerate(["a", "b"], torch.tensor([1, 2]), WIDTH)
        self.assertTrue(none.runs_plain(batch))
        self.assertFalse(_relay().runs_plain(batch))
        self.assertTrue(_relay().runs_plain(_batch(return_logprob=True)))
        self.assertTrue(_relay().runs_plain(batch, verify_max_bs=1))
        self.assertFalse(_relay().runs_plain(batch, verify_max_bs=2))


if __name__ == "__main__":
    unittest.main()
