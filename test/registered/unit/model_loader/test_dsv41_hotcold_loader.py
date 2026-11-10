"""DeepSeek-V4 hot/cold checkpoints: manifest rules, index checks, the weight map, the hook.

CPU only. The checkpoint is the toy of sglang/test/hotcold_fixture.py, written
by its own safetensors writer. The load into production's DeepseekV2MoE through
DeepseekV4ForCausalLM.load_weights needs the two-tier module and is tested with
it (test/registered/unit/layers/moe/test_dsv41_hotcold_moe.py).
"""

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.hotcold_manifest import (
    HotColdManifestError,
    HotColdWeightMap,
    attach_hotcold_manifest,
    check_checkpoint_index,
    cold_row_bytes,
    parse_manifest,
)
from sglang.test import hotcold_fixture as fx
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

N, HID, INTER = 8, 256, 256


def parse(raw, **kw):
    args = dict(
        n_routed_experts=N,
        hidden_size=HID,
        moe_intermediate_size=INTER,
        num_hidden_layers=2,
    )
    args.update(kw)
    return parse_manifest(raw, **args)


class TestManifestRules(CustomTestCase):
    def setUp(self):
        self.good = fx.manifest_dict(fx.DEFAULT_HOT, N, HID, INTER)

    def test_good_manifest_parses(self):
        m = parse(self.good)
        self.assertEqual((m.num_hot, m.num_cold), (4, 4))
        self.assertEqual(m.layer(0).hot, (0, 2, 5, 7))
        self.assertEqual(m.layer(1).cold_slot, {0: 0, 4: 1, 5: 2, 7: 3})
        self.assertEqual(m.row_bytes, {"w1": 82, "w3": 82, "w2": 82})
        self.assertEqual((m.w13_type, m.w2_type), (22, 22))

    def test_row_bytes_match_ggml_row_size(self):
        # DeepSeek-V4.1-Flash's widths: hidden 5120, moe_intermediate 2304.
        self.assertEqual(cold_row_bytes(22, 5120), 1640)
        self.assertEqual(cold_row_bytes(22, 2304), 738)
        self.assertEqual(cold_row_bytes(145, 5120), 1402)
        self.assertEqual(cold_row_bytes(145, 2304), 632)
        self.assertEqual(cold_row_bytes(144, 5120), 4 + 20 * 136)
        with self.assertRaises(HotColdManifestError):
            cold_row_bytes(23, 5120)
        with self.assertRaises(HotColdManifestError):
            cold_row_bytes(22, 64)

    def test_every_refusal(self):
        def mutate(fn):
            raw = copy.deepcopy(self.good)
            fn(raw)
            return raw

        cases = {
            "not strictly ascending": lambda r: r["layers"]["0"].update(
                hot=[2, 0, 5, 7]
            ),
            "overlap": lambda r: r["layers"]["0"].update(cold=[1, 2, 3, 4, 6]),
            "missing [1]": lambda r: r["layers"]["0"].update(cold=[3, 4, 6]),
            # layers may differ in (H, C) since the per-stage counts; a declared H must still hold for every layer
            "H is 4 but layers [1]": lambda r: (
                r["layers"]["1"].update(hot=[1, 2, 3], cold=[0, 4, 5, 6, 7]),
                r["layers"]["1"].update(forced_hot=[1]),
                r.update(H=4),
            ),
            "row_bytes.w2": lambda r: r["cold_quant"]["row_bytes"].update(w2=81),
            "one ggml type": lambda r: (
                r["cold_quant"]["ggml_type"].update(w3=145),
                r["cold_quant"]["row_bytes"].update(w3=cold_row_bytes(145, HID)),
            ),
            "forced_hot": lambda r: r["layers"]["0"].update(forced_hot=[1]),
            "n_routed_experts": lambda r: r.update(n_routed_experts=16),
            "ggml_type.w1": lambda r: r["cold_quant"]["ggml_type"].update(w1=23),
            "outside [0, 2)": lambda r: r["layers"].update({"5": r["layers"]["0"]}),
            "format": lambda r: r.update(format="dsv41-hotcold/0"),
            "at least one": lambda r: r["layers"]["0"].update(
                hot=list(range(8)), cold=[]
            ),
        }
        for needle, fn in cases.items():
            with self.subTest(needle):
                with self.assertRaises(HotColdManifestError) as ctx:
                    parse(mutate(fn))
                self.assertIn(needle, str(ctx.exception))

    def test_errors_are_reported_together(self):
        raw = copy.deepcopy(self.good)
        raw["format"] = "x"
        raw["layers"]["0"]["hot"] = [2, 0, 5, 7]
        with self.assertRaises(HotColdManifestError) as ctx:
            parse(raw)
        self.assertIn("format", str(ctx.exception))
        self.assertIn("ascending", str(ctx.exception))

    def test_remap_tables(self):
        m = parse(self.good)
        hot, cold = m.remap_tables(0)
        self.assertEqual(hot.dtype, torch.int32)
        self.assertEqual(hot.shape, (N + 1,))
        # entry 0 is the padded route -1
        self.assertEqual((int(hot[0]), int(cold[0])), (4, -1))
        for e in range(N):
            if e in (0, 2, 5, 7):
                self.assertEqual(int(hot[e + 1]), [0, 2, 5, 7].index(e))
                self.assertEqual(int(cold[e + 1]), -1)
            else:
                self.assertEqual(int(hot[e + 1]), 4)
                self.assertEqual(int(cold[e + 1]), [1, 3, 4, 6].index(e))


class TestCheckpointIndex(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _manifest(self, path):
        with open(os.path.join(path, "hotcold.json")) as f:
            return parse(json.load(f))

    def test_good_index_passes_and_attaches(self):
        path = os.path.join(self.tmp.name, "ok")
        fx.build_toy_checkpoint(path)
        check_checkpoint_index(path, self._manifest(path))
        configs = [
            SimpleNamespace(
                hotcold={"manifest": "hotcold.json"},
                n_routed_experts=N,
                hidden_size=HID,
                moe_intermediate_size=INTER,
                num_hidden_layers=2,
            )
        ]
        manifest = attach_hotcold_manifest(configs, path)
        self.assertEqual(manifest.num_hot, 4)
        self.assertEqual(
            configs[0].hotcold_manifest_path, os.path.join(path, "hotcold.json")
        )
        self.assertIsNone(attach_hotcold_manifest([SimpleNamespace()], path))

    def test_refusals(self):
        cases = {
            "native tensor of cold expert 1": dict(
                extra_index={
                    "layers.0.ffn.experts.1.w1.weight": "model-00001-of-00002.safetensors"
                }
            ),
            "shard type tag": dict(tag_types=False),
            "missing from the index": dict(),
        }
        for needle, kw in cases.items():
            with self.subTest(needle):
                path = os.path.join(self.tmp.name, needle.replace(" ", "_"))
                fx.build_toy_checkpoint(path, **kw)
                if needle == "missing from the index":
                    index = os.path.join(path, "model.safetensors.index.json")
                    with open(index) as f:
                        data = json.load(f)
                    del data["weight_map"]["layers.1.ffn.experts_cold.w2.qweight"]
                    with open(index, "w") as f:
                        json.dump(data, f)
                with self.assertRaises(HotColdManifestError) as ctx:
                    check_checkpoint_index(path, self._manifest(path))
                self.assertIn(needle, str(ctx.exception))

    def test_wrong_cold_shape_is_refused(self):
        path = os.path.join(self.tmp.name, "shape")
        fx.build_toy_checkpoint(path)
        manifest = parse(
            fx.manifest_dict(fx.DEFAULT_HOT, N, 512, INTER), hidden_size=512
        )
        with self.assertRaises(HotColdManifestError) as ctx:
            check_checkpoint_index(path, manifest)
        self.assertIn("shape", str(ctx.exception))


class TestWeightMap(CustomTestCase):
    """HotColdWeightMap.take on renamed names, with recording parameters."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))
        with open(os.path.join(self.ckpt.path, "hotcold.json")) as f:
            self.manifest = parse(json.load(f))
        self.calls = []
        self.params = {}
        for layer in range(2):
            for leaf in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale_inv",
                "w2_weight_scale_inv",
            ):
                self._param(f"model.layers.{layer}.mlp.experts.hot.{leaf}")
            for leaf in ("w13_qweight", "w2_qweight"):
                self._param(f"model.layers.{layer}.mlp.experts.cold.{leaf}")

    def _param(self, name):
        p = SimpleNamespace()
        p.weight_loader = lambda param, tensor, *args, **kw: self.calls.append(
            (name, args, kw)
        )
        self.params[name] = p

    def _names(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        for raw, tensor in self.ckpt.weights():
            yield (
                DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
                    raw, num_hidden_layers=2
                ),
                tensor,
            )

    def test_take_counts_and_slots(self):
        wm = HotColdWeightMap(self.manifest, (0, 2))
        taken = [wm.take(n, t, self.params) for n, t in self._names()]
        self.assertEqual(len(taken), 2 * (6 * 4 + 3))
        self.assertEqual(sum(".experts.hot." in t for t in taken), 2 * 6 * 4)
        wm.check_complete()
        # layer 1 expert 6 is hot slot 3; its up projection is w13 shard w3
        self.assertIn(
            (
                "model.layers.1.mlp.experts.hot.w13_weight",
                ("model.layers.1.mlp.experts.hot.w13_weight",),
                {"shard_id": "w3", "expert_id": 3},
            ),
            self.calls,
        )
        self.assertIn(
            ("model.layers.0.mlp.experts.cold.w13_qweight", (1,), {}), self.calls
        )
        for other in (
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_experts.down_proj.weight",
        ):
            self.assertIsNone(wm.take(other, torch.zeros(1), self.params))

    def test_stage_filter_and_refusals(self):
        off = HotColdWeightMap(self.manifest, (1, 2))
        self.assertIsNone(
            off.take(
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                torch.zeros(1),
                self.params,
            )
        )
        self.assertIsNone(
            off.take(
                "model.layers.0.mlp.experts_cold.gate_proj.qweight",
                torch.zeros(1),
                self.params,
            )
        )
        wm = HotColdWeightMap(self.manifest, (0, 2))
        with self.assertRaises(HotColdManifestError) as ctx:
            wm.take(
                "model.layers.0.mlp.experts.1.down_proj.weight",
                torch.zeros(1),
                self.params,
            )
        self.assertIn("native tensor of cold expert 1", str(ctx.exception))
        wm.take(
            "model.layers.0.mlp.experts.0.down_proj.weight", torch.zeros(1), self.params
        )
        with self.assertRaises(HotColdManifestError):
            wm.take(
                "model.layers.0.mlp.experts.0.down_proj.weight",
                torch.zeros(1),
                self.params,
            )
        with self.assertRaises(HotColdManifestError) as ctx:
            wm.check_complete()
        self.assertIn("1 hot tensors (expected 24)", str(ctx.exception))
        with self.assertRaises(HotColdManifestError) as ctx:
            wm.take("model.layers.0.mlp.experts.2.gate_proj.weight", torch.zeros(1), {})
        self.assertIn("not built from this manifest", str(ctx.exception))


class TestHook(CustomTestCase):
    def test_hook(self):
        from sglang.srt.arg_groups import hotcold_hook

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt = fx.build_toy_checkpoint(os.path.join(tmp.name, "ckpt"))
        good = dict(
            model_path=ckpt.path,
            tp_size=1,
            dp_size=1,
            attn_dp_size=1,
            ep_size=1,
            moe_a2a_backend="none",
            ep_num_redundant_experts=0,
            enable_eplb=False,
            moe_runner_backend="flashinfer_mxfp4",
        )
        with patch.object(hotcold_hook, "resolving_view", side_effect=lambda a: a):
            hotcold_hook.handle_hotcold(SimpleNamespace(**good))
            plain = dict(good, model_path=tmp.name, tp_size=8)
            hotcold_hook.handle_hotcold(
                SimpleNamespace(**plain)
            )  # no hotcold key: inert
            bad = dict(
                good,
                tp_size=2,
                ep_size=2,
                moe_a2a_backend="deepep",
                ep_num_redundant_experts=4,
                enable_eplb=True,
                moe_runner_backend="triton",
            )
            with self.assertRaises(ValueError) as ctx:
                hotcold_hook.handle_hotcold(SimpleNamespace(**bad))
            for needle in (
                "--tp-size",
                "--ep-size",
                "--moe-a2a-backend",
                "--ep-num-redundant-experts",
                "--enable-eplb",
                "flashinfer_mxfp4",
            ):
                self.assertIn(needle, str(ctx.exception))
            with patch.dict(os.environ, {"SGLANG_DSV4_FP4_DEQUANT": "1"}):
                with self.assertRaises(ValueError) as ctx:
                    hotcold_hook.handle_hotcold(SimpleNamespace(**good))
                self.assertIn("SGLANG_DSV4_FP4_DEQUANT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
