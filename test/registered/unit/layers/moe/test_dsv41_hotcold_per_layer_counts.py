"""DeepSeek-V4 hot/cold checkpoints whose hot count differs between layers, on CPU.

A pipeline stage with free memory can hold more hot experts than another: the
manifest then gives each layer its own (H, C), the hot FusedMoE of a layer has
that layer's H experts and its cold tensors that layer's C slots. The toy here
has three layers with three sizes (3, 5 and 4 hot of 8). What is checked:

* the manifest: per-layer counts, remap tables per layer, the declared counts
  ("H": int or "per-layer" with "H_by_layer") against the lists, the boot lines;
* a uniform manifest reads exactly as before (num_hot / num_cold, the remap
  tables, the module shapes and the load log line);
* the load through DeepseekV4ForCausalLM.load_weights into production's
  DeepseekV2MoE layers (pipeline stages included) and the two-tier forward of
  every layer against a float64 reference from the checkpoint's own tensors
  (hot tier on the float stand-in of hotcold_fixture, as the other CPU tests).
"""

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_loader.hotcold_manifest import (
    HotColdManifestError,
    parse_manifest,
)
from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator
from sglang.test import hotcold_fixture as fx
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

N, HID, INTER = 8, 256, 256
MIXED = {0: [0, 2, 5], 1: [0, 1, 2, 3, 6], 2: [1, 2, 3, 6]}
LIMIT = 10.0


def parse(raw, layers=3):
    return parse_manifest(
        raw,
        n_routed_experts=N,
        hidden_size=HID,
        moe_intermediate_size=INTER,
        num_hidden_layers=layers,
    )


class TestPerLayerManifest(CustomTestCase):
    def setUp(self):
        self.mixed = fx.manifest_dict(MIXED, N, HID, INTER)
        self.uniform = fx.manifest_dict(fx.DEFAULT_HOT, N, HID, INTER)

    def test_three_sizes_parse(self):
        m = parse(self.mixed)
        self.assertIsNone(m.num_hot)
        self.assertIsNone(m.num_cold)
        self.assertFalse(m.uniform)
        self.assertEqual([m.num_hot_of(i) for i in range(3)], [3, 5, 4])
        self.assertEqual([m.num_cold_of(i) for i in range(3)], [5, 3, 4])
        self.assertEqual(
            m.counts_text(),
            "layers 0-0: 3 hot + 5 cold, layers 1-1: 5 hot + 3 cold, layers 2-2: 4 hot + 4 cold",
        )
        self.assertEqual(m.counts_text([1]), "5 hot + 3 cold")

    def test_remap_tables_use_the_layers_own_count(self):
        m = parse(self.mixed)
        for layer, hot in MIXED.items():
            h, c = m.remap_tables(layer)
            H = len(hot)
            cold = [e for e in range(N) if e not in hot]
            self.assertEqual(
                int(h[0]), H
            )  # the padded route -1 maps out of this layer's hot range
            self.assertEqual(int(c[0]), -1)
            for e in range(N):
                if e in hot:
                    self.assertEqual((int(h[e + 1]), int(c[e + 1])), (hot.index(e), -1))
                else:
                    self.assertEqual((int(h[e + 1]), int(c[e + 1])), (H, cold.index(e)))

    def test_declared_counts(self):
        ok = copy.deepcopy(self.mixed)
        ok.update(
            H="per-layer",
            C="per-layer",
            H_by_layer={"0": 3, "1": 5, "2": 4},
            C_by_layer={"0": 5, "1": 3, "2": 4},
        )
        parse(ok)
        cases = {
            "H_by_layer says 4": lambda r: r.update(
                H="per-layer", H_by_layer={"0": 3, "1": 4, "2": 4}
            ),
            "H_by_layer is missing": lambda r: r.update(H="per-layer"),
            "H is 4 but layers [0, 1]": lambda r: r.update(H=4),
            'C must be an integer or "per-layer"': lambda r: r.update(C="4"),
        }
        for needle, fn in cases.items():
            with self.subTest(needle):
                raw = copy.deepcopy(self.mixed)
                fn(raw)
                with self.assertRaises(HotColdManifestError) as ctx:
                    parse(raw)
                self.assertIn(needle, str(ctx.exception))

    def test_uniform_reads_as_before(self):
        m = parse(self.uniform, layers=2)
        self.assertEqual((m.num_hot, m.num_cold), (4, 4))
        self.assertTrue(m.uniform)
        self.assertEqual(m.counts_text(), "4 hot + 4 cold")
        # the tables of the one-H code: hot table filled with num_hot, then the slots
        for layer, hot in fx.DEFAULT_HOT.items():
            h, c = m.remap_tables(layer)
            old_h = torch.full((N + 1,), m.num_hot, dtype=torch.int32)
            old_c = torch.full((N + 1,), -1, dtype=torch.int32)
            for e, s in m.layer(layer).hot_slot.items():
                old_h[e + 1] = s
            for e, s in m.layer(layer).cold_slot.items():
                old_c[e + 1] = s
            self.assertTrue(torch.equal(h, old_h))
            self.assertTrue(torch.equal(c, old_c))
        declared = dict(copy.deepcopy(self.uniform), H=4, C=4)
        self.assertEqual(parse(declared, layers=2).num_hot, 4)


def write_config_layers(ckpt_path, layers):
    p = os.path.join(ckpt_path, "config.json")
    with open(p) as f:
        cfg = json.load(f)
    cfg["num_hidden_layers"] = layers
    with open(p, "w") as f:
        json.dump(cfg, f)


def build_model(ckpt_path, layers, layer_range):
    """DeepseekV4ForCausalLM's loader over production's DeepseekV2MoE layers."""
    from sglang.srt.models import deepseek_v2
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    config = fx.toy_hf_config(ckpt_path, num_hidden_layers=layers)
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
    mods = []
    for layer_id in range(layers):
        layer = nn.Module()
        layer.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"model.layers.{layer_id}.mlp",
            is_deepseek_v4=True,
        )
        mods.append(layer)
    model.model.layers = nn.ModuleList(mods)
    return model


def checkpoint_stream(path):
    files = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")
    )
    return safetensors_weights_iterator(files)


class TestPerLayerLoadAndForward(CustomTestCase):
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
        self.ckpt = fx.build_toy_checkpoint(
            os.path.join(self.tmp.name, "ckpt"), hot_sets=MIXED
        )
        write_config_layers(self.ckpt.path, 3)
        self.gen = torch.Generator().manual_seed(5)

    def _load(self, layer_range=(0, 3)):
        with torch.no_grad():
            model = build_model(self.ckpt.path, 3, layer_range)
            for layer in model.model.layers:
                for p in layer.mlp.experts.parameters():
                    p.data.zero_()
            with self.assertLogs(
                "sglang.srt.model_loader.hotcold_manifest", level="INFO"
            ) as logs:
                model.load_weights(checkpoint_stream(self.ckpt.path))
        return model, [
            r.getMessage()
            for r in logs.records
            if r.getMessage().startswith("hot/cold load:")
        ]

    def test_modules_have_each_layers_size(self):
        model, _ = self._load()
        for layer_id, hot in MIXED.items():
            ex = model.model.layers[layer_id].mlp.experts
            H, C = len(hot), N - len(hot)
            self.assertEqual((ex.num_hot, ex.num_cold), (H, C))
            self.assertEqual(ex.hot.num_experts, H)
            self.assertEqual(tuple(ex.hot.w13_weight.shape), (H, 2 * INTER, HID // 2))
            self.assertEqual(tuple(ex.hot.w2_weight.shape), (H, HID, INTER // 2))
            self.assertEqual(tuple(ex.cold.w13_qweight.shape[:2]), (C, 2 * INTER))
            self.assertEqual(tuple(ex.cold.w2_qweight.shape[:2]), (C, HID))

    def test_loads_every_layer_byte_for_byte(self):
        model, lines = self._load()
        ck = self.ckpt
        for layer_id, hot in MIXED.items():
            ex = model.model.layers[layer_id].mlp.experts
            for slot, e in enumerate(hot):
                w1, _ = ck.native[(layer_id, e, "w1")]
                w3, _ = ck.native[(layer_id, e, "w3")]
                w2, _ = ck.native[(layer_id, e, "w2")]
                self.assertTrue(torch.equal(ex.hot.w13_weight.data[slot, :INTER], w3))
                self.assertTrue(torch.equal(ex.hot.w13_weight.data[slot, INTER:], w1))
                self.assertTrue(torch.equal(ex.hot.w2_weight.data[slot], w2))
            self.assertTrue(
                torch.equal(
                    ex.cold.w13_qweight.data[:, :INTER], ck.cold[(layer_id, "w1")]
                )
            )
            self.assertTrue(
                torch.equal(
                    ex.cold.w13_qweight.data[:, INTER:], ck.cold[(layer_id, "w3")]
                )
            )
            self.assertTrue(
                torch.equal(ex.cold.w2_qweight.data, ck.cold[(layer_id, "w2")])
            )
        self.assertEqual(
            lines,
            [
                "hot/cold load: layers 0..2 (3), 72 hot tensors, 9 cold tensors (3 a layer); "
                "layers 0-0: 3 hot + 5 cold, layers 1-1: 5 hot + 3 cold, layers 2-2: 4 hot + 4 cold"
            ],
        )

    def test_a_stage_with_one_size_logs_the_old_line(self):
        model, lines = self._load(layer_range=(1, 2))
        self.assertEqual(
            lines,
            [
                "hot/cold load: layers 1..1 (1), 30 hot tensors (30 a layer, 5 slots), "
                "3 cold tensors (3 a layer, 3 slots)"
            ],
        )

    def test_a_missing_hot_tensor_names_its_layers_count(self):
        drop = "layers.1.ffn.experts.6.w2.scale"
        with torch.no_grad():
            model = build_model(self.ckpt.path, 3, (0, 3))
            with self.assertRaises(HotColdManifestError) as ctx:
                model.load_weights(iter(self.ckpt.weights(drop={drop})))
        self.assertIn("layer 1: 29 hot tensors (expected 30)", str(ctx.exception))

    def test_forward_matches_the_dense_reference_on_every_layer(self):
        model, _ = self._load()
        x = torch.randn(9, HID, generator=self.gen) * 1.5
        for layer_id, hot in MIXED.items():
            ex = model.model.layers[layer_id].mlp.experts
            ex.hot.forward = fx.hot_reference_forward(ex.hot)
            cold = [e for e in range(N) if e not in hot]
            for label, pool in {
                "all hot": hot,
                "all cold": cold,
                "mixed": list(range(N)),
            }.items():
                with self.subTest(layer=layer_id, case=label):
                    ids = torch.stack(
                        [
                            torch.tensor(pool)[
                                torch.randperm(len(pool), generator=self.gen)[:3]
                            ]
                            for _ in range(9)
                        ]
                    )
                    ids = ids.to(torch.int32)
                    ids[4, 1] = -1  # one padded route
                    w = torch.rand(9, 3, generator=self.gen) + 0.1
                    with torch.no_grad():
                        got = ex(x, StandardTopKOutput(w, ids, torch.zeros(9, N)))
                    ref, _ = fx.dense_moe_reference(
                        self.ckpt, layer_id, x, ids, w, LIMIT
                    )
                    torch.testing.assert_close(
                        got.double(), ref, rtol=0, atol=1e-5 * ref.abs().max().item()
                    )


class TestUniformModuleUnchanged(CustomTestCase):
    """A uniform manifest builds the modules the one-H code built: same counts, shapes, tables, log line."""

    def setUp(self):
        self.stack = self.enterContext(fx.cpu_moe_context())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))

    def test_uniform(self):
        from sglang.srt.models import deepseek_v2

        config = fx.toy_hf_config(self.ckpt.path)
        for layer_id, hot in fx.DEFAULT_HOT.items():
            mlp = deepseek_v2.DeepseekV2MoE(
                config=config,
                layer_id=layer_id,
                quant_config=fx.fp4_quant_config(),
                prefix=f"model.layers.{layer_id}.mlp",
                is_deepseek_v4=True,
            )
            ex = mlp.experts
            self.assertEqual((ex.num_hot, ex.num_cold, ex.hot.num_experts), (4, 4, 4))
            self.assertEqual(
                {n: tuple(p.shape) for n, p in ex.named_parameters()},
                {
                    "hot.w13_weight": (4, 512, 128),
                    "hot.w2_weight": (4, 256, 128),
                    "hot.w13_weight_scale_inv": (4, 512, 8),
                    "hot.w2_weight_scale_inv": (4, 256, 8),
                    "cold.w13_qweight": (4, 512, 82),
                    "cold.w2_qweight": (4, 256, 82),
                },
            )
            self.assertEqual(int(ex.remap_hot[0]), 4)
            with self.assertLogs(
                "sglang.srt.model_loader.hotcold_manifest", level="INFO"
            ) as logs:
                fx.load_toy_layer(mlp, self.ckpt, layer_id)
            self.assertEqual(
                [
                    r.getMessage()
                    for r in logs.records
                    if r.getMessage().startswith("hot/cold load:")
                ],
                [
                    f"hot/cold load: layers {layer_id}..{layer_id} (1), 24 hot tensors (24 a layer, 4 slots), "
                    "3 cold tensors (3 a layer, 4 slots)"
                ],
            )


if __name__ == "__main__":
    unittest.main()
