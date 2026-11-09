"""The two-tier MoE of DeepSeek-V4 hot/cold checkpoints, on CPU.

The layer is production's DeepseekV2MoE built from the toy checkpoint of
sglang/test/hotcold_fixture.py (hot tier: the real FusedMoE + Fp8Config ->
Mxfp4FlashinferCutlassMoEMethod parameters; cold tier: ColdGGUFExperts), loaded
through HotColdWeightMap.take. What runs on CPU:

* the route split (remap tables, sentinels), the sum of the two tiers, the clamp,
  the router weights, padded routes, the delegation and post-load contract, the
  imatrix collector, and cold_moe's kernel switch and chunking with the kernels
  replaced by float stand-ins;
* the hot tier's FlashInfer call is replaced by a float model of it (ids outside
  [0, H) add nothing) and the cold kernels by the IQ2_S float dequant, so the
  numbers compared are float references of both tiers. The kernels' own numerics
  (q8_1 activations, FlashInfer MXFP8 x MXFP4, the sentinel skip inside
  FlashInfer, CUDA-graph replay) are left to the GPU tests.
"""

import atexit
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput, StandardTopKOutputPacked
from sglang.srt.layers.quantization import hotcold as hc
from sglang.test import hotcold_fixture as fx
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

from types import SimpleNamespace

from torch import nn

from sglang.srt.model_loader.hotcold_manifest import HotColdManifestError
from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

N, HID, INTER = 8, 256, 256


HOT0, COLD0 = [0, 2, 5, 7], [1, 3, 4, 6]
LIMIT = 10.0


class _Base(CustomTestCase):
    def setUp(self):
        self.stack = self.enterContext(fx.cpu_moe_context())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))
        from sglang.srt.models import deepseek_v2

        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=fx.toy_hf_config(self.ckpt.path),
            layer_id=0,
            quant_config=fx.fp4_quant_config(),
            prefix="model.layers.0.mlp",
            is_deepseek_v4=True,
        )
        fx.load_toy_layer(self.mlp, self.ckpt, 0)
        self.experts = self.mlp.experts
        self.experts.hot.forward = fx.hot_reference_forward(self.experts.hot)
        self.gen = torch.Generator().manual_seed(1)

    def routes(self, pool, tokens=7, top_k=3):
        ids = torch.stack(
            [
                torch.tensor(pool)[
                    torch.randperm(len(pool), generator=self.gen)[:top_k]
                ]
                for _ in range(tokens)
            ]
        )
        weights = torch.rand(tokens, top_k, generator=self.gen) + 0.1
        return ids.to(torch.int32), weights

    def run_layer(self, x, ids, weights, packed=False):
        logits = torch.zeros(x.shape[0], 8)
        if packed:
            out = StandardTopKOutputPacked(weights, ids, logits, torch.zeros_like(ids))
        else:
            out = StandardTopKOutput(weights, ids, logits)
        with torch.no_grad():
            return self.experts(x, out)


class TestTwoTierForward(_Base):
    def test_split_routes(self):
        ids = torch.tensor([[0, 1, -1], [7, 6, 5], [-1, -1, -1]], dtype=torch.int32)
        hot, cold = self.experts.split_routes(ids)
        self.assertEqual(hot.tolist(), [[0, 4, 4], [3, 4, 2], [4, 4, 4]])
        self.assertEqual(cold.tolist(), [[-1, 0, -1], [-1, 3, -1], [-1, -1, -1]])
        self.assertEqual(hot.dtype, torch.int32)
        hot64, _ = self.experts.split_routes(ids.long())
        self.assertEqual(hot64.dtype, torch.int64)

    def test_matches_dense_reference_fp32(self):
        x = torch.randn(7, 256, generator=self.gen) * 1.5
        cases = {"all hot": HOT0, "all cold": COLD0, "mixed": list(range(8))}
        for label, pool in cases.items():
            with self.subTest(label):
                ids, w = self.routes(pool)
                got = self.run_layer(x, ids, w)
                ref, clamped = fx.dense_moe_reference(self.ckpt, 0, x, ids, w, LIMIT)
                scale = ref.abs().max().item()
                torch.testing.assert_close(got.double(), ref, rtol=0, atol=1e-5 * scale)
                self.assertGreater(clamped, 0, "the clamp must bite in this case")
                unclamped, _ = fx.dense_moe_reference(self.ckpt, 0, x, ids, w, None)
                self.assertGreater((unclamped - ref).abs().max().item(), 1e-3 * scale)

    def test_padded_routes_give_zero(self):
        x = torch.randn(6, 256, generator=self.gen)
        ids, w = self.routes(list(range(8)), tokens=6)
        ids[1] = -1  # a padded row
        ids[3, 1] = -1  # one padded route
        got = self.run_layer(x, ids, w)
        self.assertTrue(torch.equal(got[1], torch.zeros(256)))
        ref, _ = fx.dense_moe_reference(self.ckpt, 0, x, ids, w, LIMIT)
        torch.testing.assert_close(
            got.double(), ref, rtol=0, atol=1e-5 * ref.abs().max().item()
        )

    def test_bf16(self):
        x = (torch.randn(5, 256, generator=self.gen)).to(torch.bfloat16)
        ids, w = self.routes(list(range(8)), tokens=5)
        got = self.run_layer(x, ids, w)
        self.assertEqual(got.dtype, torch.bfloat16)
        ref, _ = fx.dense_moe_reference(self.ckpt, 0, x.float(), ids, w, LIMIT)
        err = (got.double() - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(err, 3e-2)

    def test_packed_topk_output_is_accepted(self):
        x = torch.randn(4, 256, generator=self.gen)
        ids, w = self.routes(list(range(8)), tokens=4)
        torch.testing.assert_close(
            self.run_layer(x, ids, w, packed=True), self.run_layer(x, ids, w)
        )


class TestModuleContract(_Base):
    def test_delegation(self):
        e, hot = self.experts, self.experts.hot
        self.assertIs(
            e.should_fuse_routed_scaling_factor_in_topk,
            hot.should_fuse_routed_scaling_factor_in_topk,
        )
        self.assertTrue(e.should_fuse_routed_scaling_factor_in_topk)
        self.assertEqual(e.runner_backend, hot.runner_backend)
        self.assertIs(e.moe_runner_config, hot.moe_runner_config)
        self.assertEqual(e.moe_runner_config.swiglu_limit, LIMIT)
        self.assertFalse(e.supports_deferred_finalize)
        self.assertFalse(e.supports_routed_deferred_finalize)
        for name in (
            "set_overlap_args",
            "clear_overlap_args",
            "run_moe_core",
            "forward_deferred_finalize",
            "forward_impl",
        ):
            with self.subTest(name):
                with self.assertRaises(AttributeError):
                    getattr(e, name)

    def test_post_load_walk_processes_the_hot_tier_once(self):
        from sglang.srt.layers.moe.hotcold import HotColdQuantMethod
        from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
            Mxfp4FlashinferCutlassMoEMethod,
        )

        e = self.experts
        self.assertIsInstance(e.quant_method, HotColdQuantMethod)
        self.assertNotIsInstance(e.quant_method, Mxfp4FlashinferCutlassMoEMethod)
        self.assertIsInstance(e.hot.quant_method, Mxfp4FlashinferCutlassMoEMethod)
        hot_process = MagicMock()
        cold_check = MagicMock()
        with (
            patch.object(
                e.hot.quant_method, "process_weights_after_loading", hot_process
            ),
            patch.object(e.cold, "check_kernels", cold_check),
        ):
            # the loop of model_loader/loader.py's post-load pass
            for _, module in self.mlp.named_modules():
                qm = getattr(module, "quant_method", None)
                if qm is not None:
                    qm.process_weights_after_loading(module)
        hot_process.assert_called_once_with(e.hot)
        cold_check.assert_called_once_with()

    def test_cold_weight_loader_refusals(self):
        cold = self.experts.cold
        good = self.ckpt.cold[(0, "w1")]
        with self.assertRaises(TypeError):
            cold.weight_loader(cold.w13_qweight, good.to(torch.int8), 0)
        with self.assertRaises(ValueError):
            cold.weight_loader(cold.w13_qweight, good[:, :, :80], 0)
        with self.assertRaises(ValueError):
            cold.weight_loader(cold.w13_qweight, good, 2)

    def test_cpu_reference_type_limit(self):
        with self.assertRaises(NotImplementedError):
            hc.dequantize_cold_rows(torch.zeros(2, 1402, dtype=torch.uint8), 145)


class TestColdKernelPath(_Base):
    """cold_moe's CUDA branch with the kernels swapped for float stand-ins."""

    def _seams(self, calls):
        cold = self.experts.cold
        w13_f, w2_f = cold.reference_weights()

        def vec(x, weight, ids, top_k, qtype, rows, tokens):
            calls.append(("vec", tokens, top_k, rows))
            self.assertEqual(ids.dtype, torch.int32)
            self.assertLessEqual(tokens * top_k, hc.VEC_MAX_ROWS)
            return hc._reference_project(
                x, w13_f if weight is cold.w13_qweight else w2_f, ids, top_k, rows
            )

        def mmq(
            x,
            weight,
            sorted_ids,
            expert_ids,
            post_padded,
            qtype,
            rows,
            top_k,
            tokens,
            block,
        ):
            calls.append(("mmq", tokens, top_k, rows, block))
            return hc._reference_project(
                x,
                w13_f if weight is cold.w13_qweight else w2_f,
                sorted_ids,
                top_k,
                rows,
            )

        def weight_and_sum(y, topk_weights, ids, tokens, top_k):
            valid = (ids >= 0).reshape(-1, 1)
            wts = topk_weights.reshape(-1, 1).float().masked_fill(~valid, 0)
            y = (y.float() * wts).to(y.dtype)
            return y.view(tokens, top_k, -1).float().sum(1).to(y.dtype)

        return [
            patch.object(hc, "_vec_project", vec),
            patch.object(hc, "_mmq_project", mmq),
            patch.object(hc, "_mmq_block", lambda t: 32),
            patch.object(hc, "_mmq_align", lambda ids, block, n: (ids, None, None)),
            patch.object(hc, "_clamped_swiglu", hc._clamped_swiglu_reference),
            patch.object(hc, "_weight_and_sum", weight_and_sum),
        ]

    def _cold_call(self, x, ids, w, **kw):
        cold = self.experts.cold
        _, cold_ids = self.experts.split_routes(ids)
        return hc.cold_moe(
            x, cold.w13_qweight, cold.w2_qweight, w, cold_ids, 22, 22, LIMIT, **kw
        )

    def test_vec_chunks_above_the_grid_limit(self):
        x = torch.randn(7, 256, generator=self.gen)
        ids, w = self.routes(list(range(8)))
        expect = self._cold_call(
            x,
            ids,
            w,
            reference_weights=self.experts.cold.reference_weights(),
            use_kernels=False,
        )
        calls = []
        with patch.object(hc, "VEC_MAX_ROWS", 2 * 3 + 1):
            ctx = self._seams(calls)
            for c in ctx:
                c.start()
            try:
                got = self._cold_call(x, ids, w, use_kernels=True)
            finally:
                for c in ctx:
                    c.stop()
        # chunked bmm sums in another order: float32 rounding only
        torch.testing.assert_close(
            got, expect, rtol=0, atol=1e-6 * expect.abs().max().item()
        )
        self.assertEqual(
            [c[1] for c in calls if c[2] == 3], [2, 2, 2, 1]
        )  # w13 chunks of <= 2 tokens
        self.assertEqual(
            [c[1] for c in calls if c[2] == 1], [6, 6, 6, 3]
        )  # w2 rows per chunk

    def test_mmq_above_threshold_when_built(self):
        x = torch.randn(7, 256, generator=self.gen)
        ids, w = self.routes(list(range(8)))
        expect = self._cold_call(
            x,
            ids,
            w,
            reference_weights=self.experts.cold.reference_weights(),
            use_kernels=False,
        )
        calls = []
        env = {
            "SGLANG_HOTCOLD_MMQ_MIN_TOKENS": "4",
            "SGLANG_HOTCOLD_COLD_KERNEL": "auto",
        }
        with (
            patch.dict(os.environ, env),
            patch.object(hc, "_mmq_types", lambda: frozenset({22, 144, 145})),
        ):
            ctx = self._seams(calls)
            for c in ctx:
                c.start()
            try:
                got = self._cold_call(x, ids, w, use_kernels=True)
                big = list(calls)
                calls[:] = []
                self._cold_call(x[:4], ids[:4], w[:4], use_kernels=True)
                calls_small = list(calls)
            finally:
                for c in ctx:
                    c.stop()
        torch.testing.assert_close(got, expect)
        # (kernel, tokens, top_k, rows, block): w13 over the tokens, w2 over the routes
        self.assertEqual(big, [("mmq", 7, 3, 512, 32), ("mmq", 21, 1, 256, 32)])
        self.assertEqual(
            [c[0] for c in calls_small], ["vec", "vec"]
        )  # 4 tokens: not above the threshold

    def test_kernel_switch(self):
        sel = hc.select_cold_kernel
        with patch.object(hc, "_mmq_types", lambda: frozenset({22, 144, 145})):
            with patch.dict(os.environ, {"SGLANG_HOTCOLD_COLD_KERNEL": "auto"}):
                self.assertEqual(sel(64, 22, 22), "vec")
                self.assertEqual(sel(65, 22, 22), "mmq")
                self.assertEqual(
                    sel(54, 145, 145), "vec"
                )  # every captured batch (<= 54) stays on vec
            with patch.dict(os.environ, {"SGLANG_HOTCOLD_COLD_KERNEL": "vec"}):
                self.assertEqual(sel(4096, 22, 22), "vec")
        with patch.object(hc, "_mmq_types", lambda: frozenset()):
            with patch.dict(os.environ, {"SGLANG_HOTCOLD_COLD_KERNEL": "auto"}):
                self.assertEqual(sel(4096, 22, 22), "vec")  # without the MMQ kernels
            with patch.dict(os.environ, {"SGLANG_HOTCOLD_COLD_KERNEL": "mmq"}):
                with self.assertRaises(RuntimeError):
                    sel(4096, 22, 22)
        with patch.dict(os.environ, {"SGLANG_HOTCOLD_COLD_KERNEL": "fast"}):
            with self.assertRaises(ValueError):
                sel(1, 22, 22)


class TestImatrixCollector(_Base):
    def test_counts_and_sums(self):
        hc.HotColdImatrixCollector._instance = None
        self.addCleanup(setattr, hc.HotColdImatrixCollector, "_instance", None)
        out_dir = os.path.join(self.tmp.name, "imatrix")
        x = torch.randn(6, 256, generator=self.gen)
        ids, w = self.routes(list(range(8)), tokens=6)
        ids[2] = -1  # padded row: never counted
        with (
            patch.dict(os.environ, {"SGLANG_DSV41_HOTCOLD_IMATRIX_DIR": out_dir}),
            patch.object(hc.signal, "signal", lambda *a: None),
        ):
            self.run_layer(x, ids, w)
            inst = hc.HotColdImatrixCollector._instance
            atexit.unregister(inst.dump)
        count = torch.zeros(4, dtype=torch.int64)
        sum_x2 = torch.zeros(4, 256, dtype=torch.float64)
        sum_a2 = torch.zeros(4, 256, dtype=torch.float64)
        for t in range(6):
            for k in range(3):
                e = int(ids[t, k])
                if e in COLD0:
                    s = COLD0.index(e)
                    count[s] += 1
                    sum_x2[s] += x[t].double() ** 2
                    g_w, u_w, _ = self.ckpt.dense_expert(0, e)
                    g, u = (
                        (g_w @ x[t].double()).clamp(max=LIMIT),
                        (u_w @ x[t].double()).clamp(-LIMIT, LIMIT),
                    )
                    sum_a2[s] += (g * torch.sigmoid(g) * u) ** 2
        self.assertEqual(inst.count[0].tolist(), count.tolist())
        torch.testing.assert_close(inst.sum_x2[0], sum_x2, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(inst.sum_a2[0], sum_a2, rtol=1e-4, atol=1e-6)
        path = inst.dump()
        data = np.load(path)
        self.assertEqual(sorted(data.files), ["l0.count", "l0.sum_a2", "l0.sum_x2"])
        self.assertEqual(data["l0.count"].tolist(), count.tolist())


class TestDiagnostics(_Base):
    """SGLANG_DSV41_HOTCOLD_ZERO_COLD and the post-load recheck."""

    def test_zero_cold_returns_the_hot_tier_only(self):
        with patch.dict(os.environ, {"SGLANG_DSV41_HOTCOLD_ZERO_COLD": "1"}):
            from sglang.srt.models import deepseek_v2

            mlp = deepseek_v2.DeepseekV2MoE(
                config=fx.toy_hf_config(self.ckpt.path),
                layer_id=0,
                quant_config=fx.fp4_quant_config(),
                prefix="model.layers.0.mlp",
                is_deepseek_v4=True,
            )
        self.assertTrue(mlp.experts.zero_cold)
        self.assertFalse(self.experts.zero_cold)
        x = torch.randn(7, HID, generator=self.gen)
        ids, weights = self.routes(HOT0 + COLD0)
        full = self.run_layer(x, ids, weights)
        self.experts.zero_cold = True
        self.experts.cold.forward = MagicMock(
            side_effect=AssertionError("the cold tier ran")
        )
        dropped = self.run_layer(x, ids, weights)
        hot_ids, _ = self.experts.split_routes(ids)
        with torch.no_grad():
            hot_only = self.experts.hot(
                x, StandardTopKOutput(weights, hot_ids, torch.zeros(7, 8))
            )
        self.assertTrue(torch.equal(dropped, hot_only))
        self.assertFalse(torch.allclose(dropped, full))

    def test_digest_sees_one_byte_and_a_swap(self):
        from sglang.srt.layers.moe.hotcold import _digest

        t = torch.arange(4096, dtype=torch.int32).view(torch.uint8)[:16381].clone()
        ref = _digest([t])
        one = t.clone()
        one[16380] ^= 1
        swap = t.clone()
        swap[0:4], swap[4:8] = t[4:8].clone(), t[0:4].clone()
        self.assertFalse(torch.equal(_digest([one]), ref))
        self.assertFalse(torch.equal(_digest([swap]), ref))
        self.assertTrue(torch.equal(_digest([t.clone()]), ref))

    def test_recheck_logs_unchanged_then_changed(self):
        x = torch.randn(5, HID, generator=self.gen)
        ids, weights = self.routes(HOT0 + COLD0, tokens=5)
        self.experts.arm_recheck()
        with self.assertLogs("sglang.srt.layers.moe.hotcold", level="INFO") as logs:
            self.run_layer(x, ids, weights)
            self.run_layer(x, ids, weights)  # RECHECK_EVERY 0: the first forward only
        self.assertEqual(len(logs.records), 1)
        self.assertIn(
            "recheck layer 0, eager forward 1 (5 tokens): cold rows unchanged, hot weights unchanged",
            logs.output[0],
        )
        self.experts._recheck_every = 3
        self.experts.cold.w13_qweight.data[2, 7, 11] ^= 0x40
        with self.assertLogs("sglang.srt.layers.moe.hotcold", level="ERROR") as logs:
            self.run_layer(x, ids, weights)
        self.assertIn(
            "eager forward 3 (5 tokens): cold rows CHANGED, hot weights unchanged",
            logs.output[0],
        )


def build_model(ckpt_path, layer_range=(0, 2)):
    """DeepseekV4ForCausalLM's loader over production's DeepseekV2MoE layers."""
    from sglang.srt.models import deepseek_v2
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    config = fx.toy_hf_config(ckpt_path)
    quant_config = fx.fp4_quant_config()
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.quant_config = quant_config
    model.wo_a_fp8 = False
    model.use_npu_arch35_mxfp8_wo_a = False
    model.num_fused_shared_experts = 0
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.model = nn.Module()
    model.model.start_layer, model.model.end_layer = layer_range
    layers = []
    for layer_id in range(2):
        layer = nn.Module()
        layer.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"model.layers.{layer_id}.mlp",
            is_deepseek_v4=True,
        )
        layers.append(layer)
    model.model.layers = nn.ModuleList(layers)
    return model


def checkpoint_stream(path):
    files = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")
    )
    return safetensors_weights_iterator(files)


class TestHybridLoad(CustomTestCase):
    """The toy checkpoint read by the production safetensors iterator into production's
    DeepseekV2MoE layers under DeepseekV4ForCausalLM.load_weights itself (the hot/cold
    loader and the two-tier module)."""

    def setUp(self):
        self.stack = self.enterContext(fx.cpu_moe_context())
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        self.stack.enter_context(
            patch.object(
                DeepseekV4ForCausalLM, "post_load_weights", lambda self, **kw: None
            )
        )
        self.stack.enter_context(
            patch.object(
                DeepseekV4ForCausalLM, "_prewarm_mhc_kernels", lambda self: None
            )
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))

    def _assert_loaded(self, model, layer_id):
        ck = self.ckpt
        experts = model.model.layers[layer_id].mlp.experts
        hot, cold = experts.hot, experts.cold
        for slot, e in enumerate(ck.hot_sets[layer_id]):
            w1, s1 = ck.native[(layer_id, e, "w1")]
            w3, s3 = ck.native[(layer_id, e, "w3")]
            w2, s2 = ck.native[(layer_id, e, "w2")]
            # FlashInfer CUTLASS order: w13 = [up; gate]
            self.assertTrue(torch.equal(hot.w13_weight.data[slot, :INTER], w3))
            self.assertTrue(torch.equal(hot.w13_weight.data[slot, INTER:], w1))
            self.assertTrue(torch.equal(hot.w2_weight.data[slot], w2))
            self.assertTrue(
                torch.equal(
                    hot.w13_weight_scale_inv.data[slot, :INTER].view(torch.uint8),
                    s3.view(torch.uint8),
                )
            )
            self.assertTrue(
                torch.equal(
                    hot.w13_weight_scale_inv.data[slot, INTER:].view(torch.uint8),
                    s1.view(torch.uint8),
                )
            )
            self.assertTrue(
                torch.equal(
                    hot.w2_weight_scale_inv.data[slot].view(torch.uint8),
                    s2.view(torch.uint8),
                )
            )
        # silu_and_mul order: w13 = [gate; up]
        self.assertTrue(
            torch.equal(cold.w13_qweight.data[:, :INTER], ck.cold[(layer_id, "w1")])
        )
        self.assertTrue(
            torch.equal(cold.w13_qweight.data[:, INTER:], ck.cold[(layer_id, "w3")])
        )
        self.assertTrue(torch.equal(cold.w2_qweight.data, ck.cold[(layer_id, "w2")]))

    def test_load_through_deepseek_v4_load_weights(self):
        with torch.no_grad():
            model = build_model(self.ckpt.path)
            for layer in model.model.layers:
                for p in layer.mlp.experts.parameters():
                    p.data.zero_()
            model.load_weights(checkpoint_stream(self.ckpt.path))
        for layer_id in range(2):
            self._assert_loaded(model, layer_id)

    def test_pipeline_stage_loads_its_layers_only(self):
        with torch.no_grad():
            model = build_model(self.ckpt.path, layer_range=(1, 2))
            before = model.model.layers[0].mlp.experts.cold.w13_qweight.data.clone()
            model.load_weights(checkpoint_stream(self.ckpt.path))
        self._assert_loaded(model, 1)
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.experts.cold.w13_qweight.data, before)
        )

    def test_verify_load_logs_every_layer_and_arms_the_recheck(self):
        with (
            torch.no_grad(),
            patch.dict(os.environ, {"SGLANG_DSV41_HOTCOLD_VERIFY_LOAD": "1"}),
        ):
            model = build_model(self.ckpt.path)
            with self.assertLogs(
                "sglang.srt.model_loader.hotcold_manifest", level="INFO"
            ) as logs:
                model.load_weights(checkpoint_stream(self.ckpt.path))
        lines = [
            r.getMessage() for r in logs.records if "verify layer" in r.getMessage()
        ]
        self.assertEqual(
            lines,
            [
                f"hot/cold verify layer {l}: hot 24/24 and cold 3/3 tensors byte-equal to the shards"
                for l in (0, 1)
            ],
        )
        for layer in model.model.layers:
            self.assertIsNotNone(layer.mlp.experts._recheck_ref)

    def test_verify_load_is_off_by_default(self):
        with torch.no_grad():
            model = build_model(self.ckpt.path)
            model.load_weights(checkpoint_stream(self.ckpt.path))
        for layer in model.model.layers:
            self.assertIsNone(layer.mlp.experts._recheck_ref)

    def test_verify_names_the_tensors_that_differ(self):
        from sglang.srt.model_loader.hotcold_manifest import (
            HotColdWeightMap,
            manifest_of_config,
        )

        with torch.no_grad():
            model = build_model(self.ckpt.path)
            model.load_weights(checkpoint_stream(self.ckpt.path))
            experts = model.model.layers[0].mlp.experts
            experts.hot.w2_weight.data[1, 3, 5] ^= 0x11  # slot 1 = expert 2
            experts.hot.w13_weight_scale_inv.data.view(torch.uint8)[3, 0, 0] ^= (
                1  # slot 3 = expert 7, up rows
            )
            experts.cold.w13_qweight.data[0, INTER + 2, 9] ^= 0x80  # cold up rows (w3)
            # gate and up swapped in hot slot 0 (expert 0): both halves differ
            w13 = experts.hot.w13_weight.data[0]
            w13.copy_(torch.cat([w13[INTER:].clone(), w13[:INTER].clone()]))
            wm = HotColdWeightMap(manifest_of_config(model.config), (0, 2))
            params = dict(model.named_parameters())
            with self.assertLogs(
                "sglang.srt.model_loader.hotcold_manifest", level="ERROR"
            ) as logs:
                report = wm.verify_against_shards(params, self.ckpt.path)
        self.assertEqual(
            sorted(report[0]["mismatch"]),
            sorted(
                [
                    "layers.0.ffn.experts.2.w2.weight",
                    "layers.0.ffn.experts.7.w3.scale",
                    "layers.0.ffn.experts_cold.w3.qweight",
                    "layers.0.ffn.experts.0.w1.weight",
                    "layers.0.ffn.experts.0.w3.weight",
                ]
            ),
        )
        self.assertEqual(report[0]["hot"], (20, 24))
        self.assertEqual(report[0]["cold"], (2, 3))
        self.assertEqual(report[1]["mismatch"], [])
        self.assertIn(
            "hot/cold verify layer 0: hot 20/24 and cold 2/3 tensors byte-equal to the shards; MISMATCH 5",
            logs.output[0],
        )

    def test_refusals(self):
        ck = self.ckpt
        cold_native = ("layers.1.ffn.experts.0.w2.weight", ck.native[(0, 0, "w2")][0])
        cases = {
            "native tensor of cold expert 0": dict(extra=[cold_native]),
            "23 hot tensors (expected 24)": dict(
                drop={"layers.0.ffn.experts.5.w3.scale"}
            ),
            "0 cold tensors": dict(
                drop={
                    f"layers.1.ffn.experts_cold.{p}.qweight" for p in ("w1", "w2", "w3")
                }
            ),
            "seen twice": dict(
                extra=[("layers.0.ffn.experts.2.w1.weight", ck.native[(0, 2, "w1")][0])]
            ),
            "shape": dict(
                drop={"layers.0.ffn.experts_cold.w2.qweight"},
                extra=[
                    ("layers.0.ffn.experts_cold.w2.qweight", ck.cold[(0, "w2")][:3])
                ],
            ),
        }
        for needle, kw in cases.items():
            with self.subTest(needle):
                with torch.no_grad():
                    model = build_model(ck.path)
                    with self.assertRaises((HotColdManifestError, ValueError)) as ctx:
                        model.load_weights(iter(ck.weights(**kw)))
                self.assertIn(
                    needle,
                    str(ctx.exception)
                    + "".join(getattr(ctx.exception, "__notes__", [])),
                )


class TestModuleSwitch(CustomTestCase):
    def setUp(self):
        self.stack = self.enterContext(fx.cpu_moe_context())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))

    def _mlp(self, config, **kw):
        from sglang.srt.models import deepseek_v2

        return deepseek_v2.DeepseekV2MoE(
            config=config,
            layer_id=0,
            quant_config=fx.fp4_quant_config(),
            prefix="model.layers.0.mlp",
            is_deepseek_v4=True,
            **kw,
        )

    def test_without_manifest_the_tree_is_stock(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mlp = self._mlp(fx.toy_hf_config())
        self.assertIs(type(mlp.experts), FusedMoE)
        shapes = {n: tuple(p.shape) for n, p in mlp.experts.named_parameters()}
        self.assertEqual(
            shapes,
            {
                "w13_weight": (8, 512, 128),
                "w2_weight": (8, 256, 128),
                "w13_weight_scale_inv": (8, 512, 8),
                "w2_weight_scale_inv": (8, 256, 8),
            },
        )

    def test_manifest_builds_two_tiers_and_the_draft_stays_plain(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.hotcold import HotColdFusedMoE

        config = fx.toy_hf_config(self.ckpt.path)
        mlp = self._mlp(config)
        self.assertIsInstance(mlp.experts, HotColdFusedMoE)
        self.assertEqual(mlp.experts.hot.num_experts, 4)
        self.assertFalse(mlp.experts.hot.uses_expert_location)
        self.assertEqual(mlp.experts.hot.moe_runner_config.swiglu_limit, 10.0)
        draft = self._mlp(config, is_nextn=True)
        self.assertIs(type(draft.experts), FusedMoE)
        self.assertEqual(draft.experts.num_experts, 8)


if __name__ == "__main__":
    unittest.main()
