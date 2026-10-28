"""What crosses a DeepSeek-V4.1 pipeline-stage boundary besides the residual
streams: the mHC pre-mix and the attention inputs of the producers a later stage
replays. CPU only: the helpers, the runner buffers, the weight-name routing and
the replayed module itself."""

import json
import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache import deepseek_v41_pp_sources as pp
from sglang.srt.model_executor.model_runner_components.misc_utils import (
    resolve_pp_proxy_extra_tensors,
)
from sglang.srt.model_executor.runner_utils.buffers import (
    allocate_pp_proxy_tensors,
    pp_proxy_extra_of,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN, Q_LORA, HC = 5120, 1280, 4


class TestSourceTap(CustomTestCase):
    def test_decode_records_x_then_q_lora(self):
        tap = pp.SourceTap()
        x, q = torch.ones(3, 8), torch.full((3, 4), 2.0)
        tap.record(x, None)  # compressor call
        tap.record(x, q)  # indexer call
        x.zero_()  # a later layer reusing the buffer must not reach the copy
        got_x, got_q = tap.take()
        self.assertTrue(torch.equal(got_x, torch.ones(3, 8)))
        self.assertTrue(torch.equal(got_q, q))
        self.assertIsNot(got_q, q)
        self.assertEqual(tap.take(), (None, None))


class TestStageBoundaryTensors(CustomTestCase):
    def _pack(self, sources, pre=None, rows=5):
        tensors = {"hidden_states": torch.zeros(rows, HC * 8, dtype=torch.bfloat16)}
        return pp.pack_stage_outputs(
            tensors,
            pre=pre,
            sources=sources,
            num_rows=rows,
            hidden_size=8,
            q_lora_rank=4,
            dtype=torch.bfloat16,
            device="cpu",
        )

    def test_pre_and_sources_are_added(self):
        pre = torch.rand(5, HC).t().contiguous().t()  # a non-contiguous view
        x = torch.rand(5, 8).bfloat16()
        q = torch.rand(5, 4).bfloat16()
        out = self._pack({20: (x, q)}, pre=pre)
        self.assertTrue(out[pp.PRE_KEY].is_contiguous())
        self.assertEqual(out[pp.PRE_KEY].dtype, torch.float32)
        self.assertTrue(torch.equal(out[pp.PRE_KEY], pre))
        self.assertTrue(torch.equal(out["dsv41_pp_x_20"], x))
        self.assertTrue(torch.equal(out["dsv41_pp_q_20"], q))

    def test_short_or_missing_inputs_are_padded_with_zeros(self):
        x = torch.ones(3, 8, dtype=torch.bfloat16)
        out = self._pack({8: (x, None)})
        self.assertEqual(tuple(out["dsv41_pp_x_8"].shape), (5, 8))
        self.assertTrue(torch.equal(out["dsv41_pp_x_8"][:3], x))
        self.assertEqual(out["dsv41_pp_x_8"][3:].abs().sum().item(), 0)
        self.assertEqual(tuple(out["dsv41_pp_q_8"].shape), (5, 4))
        self.assertNotIn(pp.PRE_KEY, out)

    def test_unpack_reads_what_pack_wrote(self):
        pre = torch.rand(5, HC)
        x, q = torch.rand(5, 8).bfloat16(), torch.rand(5, 4).bfloat16()
        out = self._pack({20: (x, q), 24: (x, q)}, pre=pre)
        got_pre, sources = pp.unpack_stage_inputs(out, (20, 24))
        self.assertTrue(torch.equal(got_pre, pre))
        self.assertTrue(torch.equal(sources[24][1], q))

    def test_a_dummy_input_carries_the_residual_alone(self):
        got_pre, sources = pp.unpack_stage_inputs({"hidden_states": None}, (20,))
        self.assertIsNone(got_pre)
        self.assertEqual(sources, {20: (None, None)})

    def test_input_specs(self):
        self.assertEqual(
            pp.stage_input_specs(
                is_first_stage=True,
                pre_width=HC,
                received=(),
                hidden_size=HIDDEN,
                q_lora_rank=Q_LORA,
            ),
            {},
        )
        specs = pp.stage_input_specs(
            is_first_stage=False,
            pre_width=HC,
            received=(20, 24),
            hidden_size=HIDDEN,
            q_lora_rank=Q_LORA,
        )
        self.assertEqual(
            specs,
            {
                pp.PRE_KEY: (HC, torch.float32),
                "dsv41_pp_x_20": (HIDDEN, None),
                "dsv41_pp_q_20": (Q_LORA, None),
                "dsv41_pp_x_24": (HIDDEN, None),
                "dsv41_pp_q_24": (Q_LORA, None),
            },
        )


class TestRunnerBuffers(CustomTestCase):
    def test_extra_keys_are_allocated_beside_hidden_states(self):
        bufs = allocate_pp_proxy_tensors(
            max_num_tokens=16,
            max_hidden_tokens=16,
            hidden_size=HIDDEN,
            dtype=torch.bfloat16,
            hc_hidden_size=HC * HIDDEN,
            pp_proxy_extra={
                pp.PRE_KEY: (HC, torch.float32),
                "dsv41_pp_x_20": (HIDDEN, None),
            },
        )
        self.assertEqual(tuple(bufs["hidden_states"].shape), (16, HC * HIDDEN))
        self.assertEqual(tuple(bufs[pp.PRE_KEY].shape), (16, HC))
        self.assertEqual(bufs[pp.PRE_KEY].dtype, torch.float32)
        self.assertEqual(bufs["dsv41_pp_x_20"].dtype, torch.bfloat16)
        self.assertNotIn("residual", bufs)

    def test_resolver(self):
        model = SimpleNamespace(get_pp_proxy_extra_tensors=lambda: {"k": (1, None)})
        self.assertEqual(
            resolve_pp_proxy_extra_tensors(model=model, pp_size=3, pp_rank=1),
            {"k": (1, None)},
        )
        self.assertEqual(
            resolve_pp_proxy_extra_tensors(model=model, pp_size=3, pp_rank=0), {}
        )
        self.assertEqual(
            resolve_pp_proxy_extra_tensors(model=object(), pp_size=3, pp_rank=2), {}
        )
        # A runner without the hook (a test stub, another backend) allocates none.
        self.assertEqual(pp_proxy_extra_of(SimpleNamespace()), {})
        self.assertEqual(
            pp_proxy_extra_of(
                SimpleNamespace(get_pp_proxy_extra_tensors=lambda: {"k": (1, None)})
            ),
            {"k": (1, None)},
        )


def _v41_text_config():
    from sglang.srt.configs.deepseek_v41 import DeepseekV41TextConfig

    path = os.environ.get("DSV41_CONFIG_JSON")
    if path:
        return DeepseekV41TextConfig(**json.load(open(path))["text_config"])
    ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    return DeepseekV41TextConfig(
        vocab_size=129280,
        hidden_size=HIDDEN,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=Q_LORA,
        o_lora_rank=1024,
        max_position_embeddings=1048576,
        rope_theta=10000,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        compress_ratios=ratios,
        compress_rope_theta=160000,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        rms_norm_eps=1e-20,
    )


class TestReplayedSourceModule(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sglang.srt.models import deepseek_v4

        cls.m = deepseek_v4
        cls.config = _v41_text_config()

    def test_parameters_match_the_checkpoint_names(self):
        remap = self.m.DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        sources = torch.nn.ModuleDict(
            {
                str(i): self.m.DeepseekV41ReplayedSource(
                    self.config, i, None, prefix=f"model.layers.{i}.self_attn"
                )
                for i in (8, 20, 24)
            }
        )
        checkpoint = [
            "layers.8.attn.compressor.norm.weight",
            "layers.8.attn.compressor.wgate.weight",
            "layers.8.attn.compressor.wkv.weight",
            "layers.8.attn.indexer.k_norm.weight",
            "layers.8.attn.indexer.weights_proj.weight",
            "layers.8.attn.indexer.wk.weight",
            "layers.8.attn.indexer.wq_b.weight",
            "layers.20.attn.compressor.norm.weight",
            "layers.20.attn.compressor.wkv.weight",
            "layers.20.attn.indexer.k_norm.weight",
            "layers.20.attn.indexer.weights_proj.weight",
            "layers.20.attn.indexer.wk.weight",
            "layers.20.attn.indexer.wq_b.weight",
            "layers.24.attn.indexer.weights_proj.weight",
            "layers.24.attn.indexer.wq_b.weight",
        ]
        params = {f"model.pp_sources.{n}" for n, _ in sources.named_parameters()}
        routed = set()
        for name in checkpoint:
            name = remap(name, num_hidden_layers=40)
            layer_id = int(name.split(".")[2])
            routed.add(
                self.m._replayed_source_param_name(name, layer_id, sources)
            )
        # A ratio-2 compressor either keeps wkv and wgate (this CPU build) or
        # fuses them into wkv_gate at load, which the loader handles by name.
        params = {p for p in params if not p.endswith(".wkv_gate.weight")}
        self.assertEqual(routed, params)

    def test_other_tensors_are_not_routed(self):
        sources = torch.nn.ModuleDict(
            {"20": self.m.DeepseekV41ReplayedSource(self.config, 20, None, "p")}
        )
        route = self.m._replayed_source_param_name
        self.assertIsNone(route("model.layers.20.self_attn.wq_a.weight", 20, sources))
        self.assertIsNone(route("model.layers.20.mlp.gate.weight", 20, sources))
        self.assertIsNone(
            route("model.layers.21.self_attn.indexer.wk.weight", 21, sources)
        )
        self.assertIsNone(route("model.norm.weight", None, sources))
        self.assertIsNone(
            route("model.layers.20.self_attn.indexer.wk.weight", 20, None)
        )

    def test_what_the_backend_reads(self):
        source = self.m.DeepseekV41ReplayedSource(self.config, 20, None, "p")
        self.assertEqual((source.layer_id, source.compress_ratio), (20, 1))
        self.assertEqual(source.rope_head_dim, 64)
        self.assertIsNotNone(source.compressor)
        self.assertTrue(source.indexer.owns_k)
        self.assertTrue(source.indexer.is_candidate_source)
        only_index = self.m.DeepseekV41ReplayedSource(self.config, 24, None, "p")
        self.assertIsNone(only_index.compressor)
        self.assertTrue(only_index.indexer.uses_candidates)
        # The compressed-layer table of MqaAttentionBase (deepseek_v4.py:1135-1164):
        # YaRN on compress_rope_theta, cached, so the very tensor the layer holds.
        from sglang.kernels.ops.attention.dsv4.rope import precompute_freqs_cis

        expected = precompute_freqs_cis(
            dim=64,
            seqlen=1048576,
            original_seq_len=65536,
            base=160000,
            factor=16,
            beta_fast=32,
            beta_slow=1,
        )
        self.assertTrue(torch.equal(source.freqs_cis, expected))

    def test_stage_methods(self):
        Model = self.m.DeepseekV4Model
        tap = pp.SourceTap()
        x8, q8 = torch.rand(4, 8).bfloat16(), torch.rand(4, 4).bfloat16()
        tap.record(x8, q8)
        x2, q2 = torch.rand(4, 8).bfloat16(), torch.rand(4, 4).bfloat16()
        stub = SimpleNamespace(
            pp_source_plan=pp.StageSources(
                replayed=(2,), received=(2,), sent=(2, 8), captured=(8,)
            ),
            layers={8: SimpleNamespace(self_attn=SimpleNamespace(pp_source_tap=tap))},
            hidden_size=8,
            config=SimpleNamespace(q_lora_rank=4),
            hc_mult=HC,
            hc_pre_from_prev_sublayer=True,
            pp_group=SimpleNamespace(is_first_rank=False),
        )
        tensors = {"hidden_states": torch.zeros(4, HC * 8, dtype=torch.bfloat16)}
        pre = torch.rand(4, HC)
        Model._pack_stage_outputs(stub, tensors, pre, {2: (x2, q2)})
        self.assertTrue(torch.equal(tensors["dsv41_pp_x_8"], x8))
        self.assertTrue(torch.equal(tensors["dsv41_pp_q_2"], q2))
        self.assertTrue(torch.equal(tensors[pp.PRE_KEY], pre))
        self.assertEqual(tap.take(), (None, None))  # taken once, then cleared
        self.assertEqual(
            set(Model.pp_input_specs(stub)),
            {pp.PRE_KEY, "dsv41_pp_x_2", "dsv41_pp_q_2"},
        )


if __name__ == "__main__":
    unittest.main()
