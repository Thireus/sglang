"""What crosses the stages on a verify round, and the graph buffers that hold it."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import deepseek_v41_pp_sources as pp
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.decode_family_dispatch import DecodeFamilyDispatch
from sglang.srt.model_executor.runner_utils.buffers import allocate_pp_proxy_tensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

WIDTH = 6
HIDDEN, Q_LORA, HC = 16, 8, 4
RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
V41 = SimpleNamespace(
    compress_ratios=RATIOS,
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    num_hidden_layers=40,
)


def _stages(*sizes):
    out, start = [], 0
    for size in sizes:
        out.append(range(start, start + size))
        start += size
    return out


class TestStageHandoff(CustomTestCase):
    def test_pack_and_unpack_round_trip_at_verify_rows(self):
        """Invariant 15: hidden, pre-mix and producer inputs at 6 x bs rows."""
        for bs in (1, 3, 9):
            rows = WIDTH * bs
            x = torch.rand(rows, HIDDEN).bfloat16()
            q = torch.rand(rows, Q_LORA).bfloat16()
            pre = torch.rand(rows, HC)
            tensors = {
                "hidden_states": torch.zeros(rows, HC * HIDDEN, dtype=torch.bfloat16)
            }
            pp.pack_stage_outputs(
                tensors,
                pre=pre,
                sources={20: (x, q), 24: (x, q)},
                num_rows=rows,
                hidden_size=HIDDEN,
                q_lora_rank=Q_LORA,
                dtype=torch.bfloat16,
                device="cpu",
            )
            got_pre, got = pp.unpack_stage_inputs(tensors, (20, 24))
            self.assertTrue(torch.equal(got_pre, pre))
            for layer in (20, 24):
                self.assertTrue(torch.equal(got[layer][0], x), layer)
                self.assertTrue(torch.equal(got[layer][1], q), layer)

    def test_source_tap_of_a_verify_forward(self):
        """Stage 2 replays 20 and 24 from the verify's rows, this forward's alone."""
        tap = pp.SourceTap()
        x = torch.ones(WIDTH * 2, HIDDEN)
        q = torch.full((WIDTH * 2, Q_LORA), 2.0)
        tap.record(x, q)
        got_x, got_q = tap.take()
        self.assertEqual(got_x.shape[0], WIDTH * 2)
        self.assertEqual(got_q.shape[0], WIDTH * 2)
        self.assertEqual(tap.take(), (None, None))

    def test_plan_of_the_split(self):
        """Invariant 16: 14/13/13: stage 1 replays nothing, stage 2 replays 20 and 24."""
        stages = _stages(14, 13, 13)
        plans = [pp.stage_sources(V41, stages, r) for r in range(3)]
        self.assertEqual([p.replayed for p in plans], [(), (), (20, 24)])
        self.assertEqual(plans[0].sent, ())
        self.assertEqual(plans[1].sent, (20, 24))
        self.assertEqual(plans[2].received, (20, 24))

    def test_draft_target_layers_are_on_the_last_stage(self):
        for split in ((14, 13, 13), (13, 13, 14), (13, 14, 13)):
            self.assertTrue(all(i in _stages(*split)[2] for i in (37, 38, 39)), split)


class TestProxyBuffers(CustomTestCase):
    def test_buffers_fit_the_largest_verify(self):
        """Invariant 17: 9 x 6 rows of every key a stage receives."""
        extra = {
            pp.PRE_KEY: (HC, torch.float32),
            pp.source_input_key(20): (HIDDEN, None),
            pp.source_q_lora_key(20): (Q_LORA, None),
        }
        buffers = allocate_pp_proxy_tensors(
            max_num_tokens=9 * WIDTH,
            max_hidden_tokens=9 * WIDTH,
            hidden_size=HIDDEN,
            dtype=torch.bfloat16,
            hc_hidden_size=HC * HIDDEN,
            pp_proxy_extra=extra,
        )
        self.assertEqual(set(buffers), {"hidden_states", *extra})
        for key, tensor in buffers.items():
            self.assertEqual(tensor.shape[0], 54, key)
        self.assertEqual(buffers[pp.PRE_KEY].dtype, torch.float32)

    def test_a_preplanned_replay_refreshes_every_proxy_key_and_skips_markers(self):
        """Invariant 17: prepare_for_verify plans without the proxies; execute brings
        them. __msg_type__ is not a tensor buffer, so it is not copied."""
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.dllm_uses_input_embeds = False
        runner.dllm_input_preparation = None
        runner.ragged_verify_mode = False
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.bs = 2
        runner.raw_bs = 2  # read by the replay key since upstream #47676
        runner.require_mlp_tp_gather = False
        runner._with_logits_variant = lambda key, variant: key
        runner._resolve_logits_variant = lambda fb, padded: None
        runner.captured_req_width = WIDTH
        runner.raw_num_token = 2 * WIDTH
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
            is_draft_worker=False,
        )
        buffers = {
            "hidden_states": torch.zeros(54, HC * HIDDEN),
            pp.PRE_KEY: torch.zeros(54, HC),
            pp.source_input_key(20): torch.zeros(54, HIDDEN),
        }
        runner.buffers = SimpleNamespace(pp_proxy_tensors=buffers, num_token_non_padded=None)
        runner._set_dp_real_num_tokens = lambda n: None
        runner._copy_replay_inputs = lambda fb: None
        runner._capture_graph_size = lambda bs, num_tokens: bs
        runner._resolve_lora_variant = lambda fb: None
        runner._resolve_attention_variant = lambda fb, bs: None
        runner._make_graph_key = lambda *a: a
        rows = 2 * WIDTH
        proxies = PPProxyTensors(
            {
                "hidden_states": torch.ones(rows, HC * HIDDEN),
                pp.PRE_KEY: torch.full((rows, HC), 2.0),
                pp.source_input_key(20): torch.full((rows, HIDDEN), 3.0),
                "__msg_type__": "proxy",
            }
        )
        forward_batch = SimpleNamespace(
            ragged_verify_layout=None,
            needs_forward_metadata_init=lambda: False,
            input_embeds=None,
        )
        runner.load_batch(forward_batch, proxies)
        self.assertTrue((buffers["hidden_states"][:rows] == 1).all())
        self.assertTrue((buffers["hidden_states"][rows:] == 0).all())
        self.assertTrue((buffers[pp.PRE_KEY][:rows] == 2).all())
        self.assertTrue((buffers[pp.source_input_key(20)][:rows] == 3).all())


class _Runner:
    def __init__(self, name, max_bs):
        self.name, self.max_bs = name, max_bs
        self.capture_bs = list(range(1, max_bs + 1))
        self.loaded = []

    def can_run_graph(self, forward_batch):
        return forward_batch.batch_size <= self.max_bs

    def load_batch(self, forward_batch, proxies=None):
        self.loaded.append(("load", forward_batch.forward_mode))

    def execute(self, forward_batch, pp_proxy_tensors=None):
        return self.name


def _fb(mode, batch_size=1):
    return SimpleNamespace(forward_mode=mode, batch_size=batch_size)


class TestGraphFamilies(CustomTestCase):
    def setUp(self):
        self.verify = _Runner("verify", 4)
        self.plain = _Runner("plain", 9)
        self.dispatch = DecodeFamilyDispatch(verify=self.verify, plain=self.plain)

    def test_mode_picks_the_family(self):
        """Invariant 18: both families on every stage, chosen by the forward's mode."""
        self.assertEqual(
            self.dispatch.execute(_fb(ForwardMode.TARGET_VERIFY)), "verify"
        )
        self.assertEqual(self.dispatch.execute(_fb(ForwardMode.DECODE)), "plain")
        self.dispatch.load_batch(_fb(ForwardMode.TARGET_VERIFY))
        self.assertEqual(self.verify.loaded, [("load", ForwardMode.TARGET_VERIFY)])
        self.assertEqual(self.plain.loaded, [])

    def test_a_micro_batch_above_a_list_does_not_reach_that_graph(self):
        self.assertFalse(self.dispatch.can_run_graph(_fb(ForwardMode.TARGET_VERIFY, 5)))
        self.assertTrue(self.dispatch.can_run_graph(_fb(ForwardMode.DECODE, 5)))
        self.assertFalse(self.dispatch.can_run_graph(_fb(ForwardMode.DECODE, 10)))

    def test_everything_else_reads_the_verify_runner(self):
        self.assertEqual(self.dispatch.capture_bs, self.verify.capture_bs)
        self.assertEqual(self.dispatch.max_bs, 4)

    def test_padded_rows_address_slot_zero(self):
        """Padded verify rows read a ring slot before their no-write check, so the graph
        buffers must zero both tails: the policy the verify family inherits."""
        from sglang.srt.model_executor.cuda_graph_buffer_registry import PaddingPolicy

        self.assertTrue(hasattr(mod, "build_decode_registry"))
        registry = mod.build_decode_registry(
            device="cpu",
            max_bs=9,
            max_num_token=54,
            seq_len_fill_value=1,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            is_encoder_decoder=False,
            encoder_len_fill_value=0,
            enable_num_token_non_padded=False,
            require_gathered_buffer=False,
            enable_prefill_cp=False,
            require_mlp_tp_gather=False,
            attn_tp_sharded_fn=lambda n: False,
            num_dp_ranks=1,
            source=None,
        )
        for name in ("positions", "req_pool_indices", "out_cache_loc"):
            self.assertIs(
                registry.get_slot(name).padding_policy, PaddingPolicy.ZERO, name
            )

    def test_plain_family_has_no_spec_info_and_a_verify_family_does(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.plain_decode_family = True
        self.assertIsNone(runner.get_spec_info(9))
        self.assertFalse(DecodeCudaGraphRunner.plain_decode_family)

    def test_capture_dispatch_is_only_for_the_relayed_launch(self):
        from sglang.srt.model_executor.model_runner_components import cuda_graph_setup

        runner = object()
        draft = SimpleNamespace(is_draft_worker=True)
        self.assertIs(
            cuda_graph_setup._maybe_add_pp_dspark_plain_family(draft, runner), runner
        )
        built = []

        def plain_family(model_runner, *, plain_decode_family):
            built.append(plain_decode_family)
            return SimpleNamespace(capture_bs=[1, 2, 4, 8, 9])

        target = SimpleNamespace(
            is_draft_worker=False, _decode_cuda_graph_runner_cls=lambda: plain_family
        )

        def launch(algorithm, flag):
            with (
                mock.patch.object(
                    cuda_graph_setup, "get_parallel", lambda: SimpleNamespace(pp_size=3)
                ),
                mock.patch.object(
                    cuda_graph_setup,
                    "get_spec",
                    lambda: SimpleNamespace(speculative_algorithm=algorithm),
                ),
                envs.SGLANG_ENABLE_PP_SPEC.override(flag),
            ):
                return cuda_graph_setup._maybe_add_pp_dspark_plain_family(target, runner)

        # The flag on, another algorithm; the algorithm, no flag: the one runner.
        self.assertIs(launch("EAGLE", True), runner)
        self.assertIs(launch("DSPARK", False), runner)
        self.assertEqual(built, [])
        both = launch("DSPARK", True)
        self.assertEqual(built, [True])
        self.assertIs(both.verify, runner)


if __name__ == "__main__":
    unittest.main()
