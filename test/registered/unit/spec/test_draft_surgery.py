"""A checkpoint whose draft keeps E experts a stage loads through the real name mapping:
invariant 37. Meta tensors from safetensors headers; nothing is read but the headers."""

import json
import logging
import os
import re
import struct
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.models import deepseek_v4_dspark as dspark
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.")


def _synthetic_names(experts):
    names = []
    for stage in range(3):
        for e in range(experts):
            for part in ("w1", "w2", "w3"):
                names.append((f"mtp.{stage}.ffn.experts.{e}.{part}.weight", (8, 8)))
                names.append((f"mtp.{stage}.ffn.experts.{e}.{part}.scale", (8, 1)))
        names.append((f"mtp.{stage}.ffn.gate.weight", (experts, 16)))
        names.append((f"mtp.{stage}.ffn.gate.bias", (experts,)))
        names.append((f"mtp.{stage}.ffn.gate.bias_vl", (experts,)))
        for part in ("w1", "w2", "w3"):
            names.append((f"mtp.{stage}.ffn.shared_experts.{part}.weight", (8, 8)))
            names.append((f"mtp.{stage}.ffn.shared_experts.{part}.scale", (8, 1)))
        names.append((f"mtp.{stage}.attn.wkv.weight", (8, 16)))
        names.append((f"mtp.{stage}.attn_norm.weight", (16,)))
    names.append(("mtp.2.markov_head.embed.weight", (32, 8)))
    names.append(("mtp.2.markov_head.head.weight", (32, 8)))
    names.append(("mtp.2.confidence_head.proj.weight", (1, 8)))
    return names


def _header_names(folder):
    with open(os.path.join(folder, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    out = []
    for shard in sorted({v for k, v in weight_map.items() if k.startswith("mtp.")}):
        with open(os.path.join(folder, shard), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        out.extend((name, tuple(t["shape"])) for name, t in header.items())
    return out


class _Param:
    def __init__(self, sink, name):
        self.sink, self.name = sink, name
        self.weight_loader = self._load

    def _load(self, param, weight, *args, shard_id=None, expert_id=None):
        self.sink.append((self.name, expert_id, tuple(weight.shape)))


def _load(names, experts, fused):
    """DeepseekV4ForCausalLMDSpark.load_weights over meta tensors of these names."""
    draft = dspark.DeepseekV4ForCausalLMDSpark.__new__(dspark.DeepseekV4ForCausalLMDSpark)
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(
        dspark_n_routed_experts=experts,
        dspark_num_experts_per_tok=3,
        n_routed_experts=320,
        num_experts_per_tok=6,
        vision_n_layers=0,
    )
    draft.num_fused_shared_experts = fused
    draft.confidence_head = None
    draft.num_stages = 3
    draft.post_load_weights = lambda: None
    sink = []
    mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=experts + fused,
    )
    params = {}
    for name, _ in names:
        mapped = draft._remap_dspark_weight_name(name)
        if mapped is None:
            continue
        if fused and ".mlp.shared_experts." in mapped:
            mapped = mapped.replace(".mlp.shared_experts.", f".mlp.experts.{experts}.")
        for param_name, weight_name, _expert, _shard in mapping:
            if weight_name in mapped:
                candidate = mapped.replace(weight_name, param_name)
                params[candidate] = _Param(sink, candidate)
                break
        else:
            # A real draft has no parameter for an expert past its E + fused slots.
            if not EXPERT.search(mapped):
                params[mapped] = _Param(sink, mapped)
    draft.named_parameters = lambda: list(params.items())
    weights = [(n, torch.empty(shape, device="meta")) for n, shape in names]
    with mock.patch.object(dspark, "_dequant_fp8_wo_a_streaming", lambda w: w):
        with mock.patch.object(dspark, "_is_npu", False):
            draft.load_weights(weights)
    return sink


class TestDraftSurgeryLoads(CustomTestCase):
    E = 4

    def _check(self, names, experts, fused):
        with self.assertNoLogs(dspark.logger, level=logging.WARNING):
            sink = _load(names, experts, fused)
        expert_ids = {e for _, e, _ in sink if e is not None}
        self.assertEqual(expert_ids, set(range(experts + fused)))
        gates = [s for name, _, s in sink if name.endswith(".mlp.gate.weight")]
        self.assertEqual(len(gates), 3)
        self.assertTrue(all(s[0] == experts for s in gates), gates)
        return sink

    def test_small_draft_without_and_with_fused_shared_experts(self):
        names = _synthetic_names(self.E)
        self._check(names, self.E, 0)
        self._check(names, self.E, 1)

    def test_an_expert_past_e_is_not_a_parameter(self):
        names = _synthetic_names(self.E)
        names.append(("mtp.0.ffn.experts.9.w1.weight", (8, 8)))
        with self.assertRaises(AssertionError):
            self._check(names, self.E, 0)

    def test_the_stage_config_reads_e_and_the_target_keeps_k(self):
        config = SimpleNamespace(
            dspark_n_routed_experts=64,
            dspark_num_experts_per_tok=3,
            n_routed_experts=320,
            num_experts_per_tok=6,
            vision_n_layers=0,
        )
        stage = dspark._dspark_stage_config(config)
        self.assertEqual((stage.n_routed_experts, stage.num_experts_per_tok), (64, 3))
        self.assertEqual(config.n_routed_experts, 320)
        self.assertGreaterEqual(stage.n_routed_experts, stage.num_experts_per_tok)

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_DSV41_DRAFT_CKPT"),
        "needs SGLANG_TEST_DSV41_DRAFT_CKPT, a directory prune_draft.py wrote",
    )
    def test_the_written_checkpoint(self):
        folder = os.environ["SGLANG_TEST_DSV41_DRAFT_CKPT"]
        with open(os.path.join(folder, "config.json")) as fh:
            text = json.load(fh)["text_config"]
        experts = int(text["dspark_n_routed_experts"])
        self.assertLess(experts, 128)
        self.assertEqual(text["n_routed_experts"], 320)
        names = _header_names(folder)
        ids = {
            int(m.group(1))
            for m in (re.search(r"\.experts\.(\d+)\.", n) for n, _ in names)
            if m
        }
        self.assertEqual(ids, set(range(experts)))
        self._check(names, experts, 0)
        self._check(names, experts, 1)
        parsed = parse_dspark_draft_config(
            draft_hf_config=SimpleNamespace(**text, architectures=["DeepseekV41ForCausalLM"])
        )
        self.assertEqual(int(parsed.markov_rank), 256)
        self.assertEqual(int(parsed.resolve_gamma(default=None)), 5)


if __name__ == "__main__":
    unittest.main()
