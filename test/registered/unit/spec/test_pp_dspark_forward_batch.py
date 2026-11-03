"""The two forward shapes of a relayed round, through the real ScheduleBatch,
ForwardBatch.init_new and DFlashVerifyInput.prepare_for_verify."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_pp_relay import (
    DSparkPPRelayInput,
    prepare_plain_round,
    relay_verify_input,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

WIDTH = 6


class _Plan:
    def __init__(self, write_virtual):
        self.write_virtual = write_virtual
        self.bound = []

    def bind(self, forward_batch, runner_ids, cols=None):
        self.bound.append(cols)
        forward_batch.iteration_ids = self

    def write_ids(self, runner_ids, cols=None):
        return self.write_virtual


class _RunnerIds:
    def plan_uniform_extend(self, *, req_pool_indices, seq_lens, seq_lens_cpu, width):
        rows = seq_lens.to(torch.int64).unsqueeze(1) + torch.arange(width)
        return _Plan((req_pool_indices.unsqueeze(1) * 10000 + rows).reshape(-1))

    def own_plan(self, forward_batch):
        return _Plan(forward_batch.out_cache_loc)


class _Req:
    """The request fields init_new reads; the rest are off."""

    def __init__(self, rid, committed):
        self.rid = rid
        self.kv = SimpleNamespace(kv_committed_len=committed)

    def __getattr__(self, name):
        return None


def _req(rid, committed):
    return _Req(rid, committed)


def _batch():
    batch = ScheduleBatch(reqs=[_req("a", 40), _req("b", 90)])
    batch.device = "cpu"
    batch.spec_algorithm = SpeculativeAlgorithm.DSPARK
    batch.forward_mode = ForwardMode.DECODE
    batch.req_pool_indices = torch.tensor([3, 5], dtype=torch.int64)
    batch.seq_lens = torch.tensor([40, 90], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([40, 90], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([40, 90], dtype=torch.int32)
    batch.seq_lens_sum = 130
    batch.sampling_info = SimpleNamespace(has_watermark_candidates=False)
    return batch


def _runner():
    runner = MagicMock()
    runner.device = "cpu"
    runner.is_draft_worker = False
    runner.runner_ids = _RunnerIds()
    runner.model_config = SimpleNamespace(
        model_is_mrope=False,
        is_encoder_decoder=False,
        requires_mm_token_modalities=False,
        context_len=1 << 20,
        hf_text_config=SimpleNamespace(max_position_embeddings=1 << 20),
    )
    runner.ngram_embedding_manager = SimpleNamespace(enabled=False)
    runner.lora_manager = None
    runner.prefill_attention_backend_str = "flashinfer"
    runner.watermark_state = None
    runner.decode_cuda_graph_runner = None
    return runner


class TestRealBatchShapes(CustomTestCase):
    def test_plain_round_reads_as_a_decode_of_the_next_row(self):
        with published_topology(pp_size=3):
            batch, runner = _batch(), _runner()
            relay = DSparkPPRelayInput.degenerate(
                ["a", "b"], torch.tensor([7, 8]), WIDTH
            )
            prepare_plain_round(batch, relay, runner.runner_ids)
            fb = ForwardBatch.init_new(
                batch,
                runner,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_hidden_states_before_norm=False,
            )
        self.assertEqual(fb.forward_mode, ForwardMode.DECODE)
        self.assertEqual(fb.input_ids.tolist(), [7, 8])
        self.assertEqual(fb.positions.tolist(), [40, 90])
        self.assertEqual(fb.seq_lens.tolist(), [41, 91])
        self.assertEqual(fb.seq_lens_cpu.tolist(), [41, 91])
        self.assertEqual(fb.seq_lens_sum, 132)
        self.assertEqual(fb.out_cache_loc.tolist(), [30040, 50090])
        self.assertIsNone(fb.spec_info)
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)

    def test_verify_round_reads_as_a_static_verify(self):
        with published_topology(pp_size=3):
            batch, runner = _batch(), _runner()
            ids = torch.arange(12, dtype=torch.int64).view(2, WIDTH)
            positions = batch.seq_lens.unsqueeze(1) + torch.arange(WIDTH)
            plan = runner.runner_ids.plan_uniform_extend(
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
                width=WIDTH,
            )
            window = VerifyWindow(
                positions_2d=positions,
                verify_cache_loc=plan.write_virtual,
                verify_cache_loc_2d=plan.write_virtual.view(2, WIDTH),
                iteration_ids=plan,
            )
            verify_input = relay_verify_input(batch, ids, window, WIDTH)
            target_worker = SimpleNamespace(model_runner=runner)
            fb, can_run = verify_input.prepare_for_verify(batch, target_worker)
        self.assertFalse(can_run)
        self.assertEqual(fb.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(fb.input_ids.tolist(), ids.reshape(-1).tolist())
        self.assertEqual(fb.positions.tolist(), positions.reshape(-1).tolist())
        # A verify's seq_lens do not count its window; the backend adds it.
        self.assertEqual(fb.seq_lens.tolist(), [40, 90])
        self.assertEqual(fb.out_cache_loc.tolist(), plan.write_virtual.tolist())
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertEqual(fb.spec_info.draft_token_num, WIDTH)
        self.assertEqual(fb.spec_info.num_tokens_per_req, WIDTH)
        self.assertIsNone(fb.spec_info.custom_mask)
        runner.attn_backend.init_forward_metadata.assert_called_once()


if __name__ == "__main__":
    unittest.main()
