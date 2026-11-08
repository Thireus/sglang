"""A toy DeepSeek-V4 hot/cold checkpoint for CPU tests, written from scratch.

Nothing here reads a real checkpoint. ``write_safetensors`` is a minimal
safetensors writer (header JSON + raw little-endian bytes). The hot tier is
random MXFP4 (E2M1 nibbles, low nibble first, one E8M0 scale per 32); the cold
tier is random valid IQ2_S rows (82-byte blocks: fp16 d, 64 qs, 8 qh, 8 scales;
every byte pattern decodes, d is set to a small positive fp16). The float
references dequantise both: MXFP4 here, IQ2_S through gguf-py.

The toy is 2 layers x 8 experts (H 4 / C 4), hidden 256 and moe_intermediate
256: a ggml k-block holds 256 values, so 256 is the smallest width a cold row
can have (hidden 64 cannot be quantised), and FlashInfer's CUTLASS MoE needs
multiples of 128.
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_DTYPES = {
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.float8_e8m0fnu: "F8_E8M0",
    torch.float32: "F32",
    torch.bfloat16: "BF16",
}
E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)
IQ2_S = 22
IQ2_S_BLOCK = 82


def write_safetensors(
    path: str,
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    header = {}
    blobs: List[bytes] = []
    offset = 0
    for name, t in tensors.items():
        t = t.contiguous()
        data = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
        header[name] = {
            "dtype": _DTYPES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


def random_mxfp4(
    gen: torch.Generator, n: int, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(weight int8 [n, k/2], scale e8m0 [n, k/32]); scales 2^-3..2^-1."""
    w = (
        torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.int32)
        .to(torch.uint8)
        .view(torch.int8)
    )
    e = torch.randint(124, 127, (n, k // 32), generator=gen, dtype=torch.int32).to(
        torch.uint8
    )
    return w, e.view(torch.float8_e8m0fnu)


def dequant_mxfp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """float64 [..., k] from int8/uint8 [..., k/2] and e8m0 [..., k/32]."""
    b = w.contiguous().view(torch.uint8).long()
    lo, hi = b & 0xF, b >> 4
    vals = torch.stack([E2M1.double()[lo], E2M1.double()[hi]], dim=-1).flatten(-2)
    exp = s.contiguous().view(torch.uint8).double() - 127.0
    return vals * torch.pow(2.0, exp).repeat_interleave(32, dim=-1)


def random_iq2s_rows(
    gen: torch.Generator, shape: Tuple[int, ...], k: int, d_range=(0.003, 0.006)
) -> torch.Tensor:
    """uint8 [*shape, k/256*82]: random valid IQ2_S rows."""
    nblk = k // 256
    data = torch.randint(
        0, 256, (*shape, nblk, IQ2_S_BLOCK), generator=gen, dtype=torch.int32
    ).to(torch.uint8)
    d = torch.empty(*shape, nblk).uniform_(*d_range, generator=gen).to(torch.float16)
    data[..., 0:2] = d.unsqueeze(-1).view(torch.uint8).view(*shape, nblk, 2)
    return data.reshape(*shape, nblk * IQ2_S_BLOCK)


def dequant_iq2s(rows: torch.Tensor) -> torch.Tensor:
    import gguf

    flat = rows.reshape(-1, rows.shape[-1]).numpy()
    out = gguf.quants.dequantize(flat, gguf.GGMLQuantizationType.IQ2_S)
    return (
        torch.from_numpy(np.ascontiguousarray(out))
        .double()
        .reshape(*rows.shape[:-1], -1)
    )


def iq2s_row_bytes(k: int) -> int:
    return k // 256 * IQ2_S_BLOCK


@dataclasses.dataclass
class ToyCheckpoint:
    path: str
    manifest: dict
    hot_sets: Dict[int, List[int]]
    # (layer, expert, "w1"|"w2"|"w3") -> (weight, scale), hot experts only
    native: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]]
    # (layer, "w1"|"w2"|"w3") -> uint8 [C, N, row_bytes]
    cold: Dict[Tuple[int, str], torch.Tensor]
    hidden: int
    inter: int
    n_experts: int

    def cold_sets(self) -> Dict[int, List[int]]:
        return {
            l: [e for e in range(self.n_experts) if e not in h]
            for l, h in self.hot_sets.items()
        }

    def dense_expert(self, layer: int, expert: int):
        """float64 (gate [I, hidden], up [I, hidden], down [hidden, I])."""
        if expert in self.hot_sets[layer]:
            return tuple(
                dequant_mxfp4(*self.native[(layer, expert, p)])
                for p in ("w1", "w3", "w2")
            )
        slot = self.cold_sets()[layer].index(expert)
        return tuple(
            dequant_iq2s(self.cold[(layer, p)][slot]) for p in ("w1", "w3", "w2")
        )

    def weights(self, *, drop=(), extra=()):
        """The raw checkpoint stream (name, tensor), as the safetensors iterator yields it."""
        out = []
        for (l, e, p), (w, s) in sorted(self.native.items()):
            out.append((f"layers.{l}.ffn.experts.{e}.{p}.weight", w))
            out.append((f"layers.{l}.ffn.experts.{e}.{p}.scale", s))
        for (l, p), rows in sorted(self.cold.items()):
            out.append((f"layers.{l}.ffn.experts_cold.{p}.qweight", rows))
        out = [kv for kv in out if kv[0] not in drop]
        return out + list(extra)


DEFAULT_HOT = {0: [0, 2, 5, 7], 1: [1, 2, 3, 6]}


def manifest_dict(
    hot_sets, n_experts, hidden, inter, cold_type=IQ2_S, forced=None
) -> dict:
    from sglang.srt.model_loader.hotcold_manifest import cold_row_bytes

    layers = {}
    for l, hot in hot_sets.items():
        layers[str(l)] = {
            "hot": list(hot),
            "cold": [e for e in range(n_experts) if e not in hot],
            "forced_hot": list((forced or {}).get(l, hot[:1])),
        }
    return {
        "format": "dsv41-hotcold/1",
        "source": "toy",
        "n_routed_experts": n_experts,
        "hot_rule": {"selector": "toy"},
        "cold_quant": {
            "ggml_type": {"w1": cold_type, "w3": cold_type, "w2": cold_type},
            "row_bytes": {
                "w1": cold_row_bytes(cold_type, hidden),
                "w3": cold_row_bytes(cold_type, hidden),
                "w2": cold_row_bytes(cold_type, inter),
            },
        },
        "layers": layers,
    }


def toy_config_dict(
    n_experts=8, hidden=256, inter=256, top_k=3, layers=2, swiglu_limit=10.0
) -> dict:
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "hidden_size": hidden,
        "moe_intermediate_size": inter,
        "intermediate_size": inter,
        "n_routed_experts": n_experts,
        "num_experts_per_tok": top_k,
        "n_shared_experts": 1,
        "num_hidden_layers": layers,
        "first_k_dense_replace": 0,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": swiglu_limit,
        "hidden_act": "silu",
        "vocab_size": 1024,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "vision_n_layers": 0,
    }


def build_toy_checkpoint(
    root: str,
    *,
    hot_sets=None,
    n_experts: int = 8,
    hidden: int = 256,
    inter: int = 256,
    seed: int = 0,
    tag_types: bool = True,
    extra_index: Optional[Dict[str, str]] = None,
) -> ToyCheckpoint:
    hot_sets = {int(k): list(v) for k, v in (hot_sets or DEFAULT_HOT).items()}
    gen = torch.Generator().manual_seed(seed)
    os.makedirs(root, exist_ok=True)
    native = {}
    cold = {}
    shapes = {"w1": (inter, hidden), "w3": (inter, hidden), "w2": (hidden, inter)}
    for l, hot in sorted(hot_sets.items()):
        for e in hot:
            for p, (n, k) in shapes.items():
                native[(l, e, p)] = random_mxfp4(gen, n, k)
        c = n_experts - len(hot)
        for p, (n, k) in shapes.items():
            cold[(l, p)] = random_iq2s_rows(gen, (c, n), k)
    manifest = manifest_dict(hot_sets, n_experts, hidden, inter)
    hot_file, cold_file = (
        "model-00001-of-00002.safetensors",
        "model-cold-00001-of-00001.safetensors",
    )
    hot_tensors = {}
    for (l, e, p), (w, s) in sorted(native.items()):
        hot_tensors[f"layers.{l}.ffn.experts.{e}.{p}.weight"] = w
        hot_tensors[f"layers.{l}.ffn.experts.{e}.{p}.scale"] = s
    cold_tensors = {
        f"layers.{l}.ffn.experts_cold.{p}.qweight": rows
        for (l, p), rows in sorted(cold.items())
    }
    meta = {f"hotcold.ggml_type.{n}": IQ2_S for n in cold_tensors} if tag_types else {}
    write_safetensors(os.path.join(root, hot_file), hot_tensors)
    write_safetensors(os.path.join(root, cold_file), cold_tensors, meta)
    weight_map = {n: hot_file for n in hot_tensors}
    weight_map.update({n: cold_file for n in cold_tensors})
    weight_map.update(extra_index or {})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with open(os.path.join(root, "hotcold.json"), "w") as f:
        json.dump(manifest, f)
    config = toy_config_dict(n_experts=n_experts, hidden=hidden, inter=inter)
    config["hotcold"] = {"manifest": "hotcold.json"}
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config, f)
    return ToyCheckpoint(
        root, manifest, hot_sets, native, cold, hidden, inter, n_experts
    )


def dense_moe_reference(
    ckpt: ToyCheckpoint,
    layer: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: Optional[float],
) -> Tuple[torch.Tensor, int]:
    """float64 [T, hidden] from the checkpoint's own tensors, and the count of
    gate/up values the clamp changed. Negative ids add nothing."""
    out = torch.zeros(x.shape[0], ckpt.hidden, dtype=torch.float64)
    clamped = 0
    cache = {}
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            if e < 0:
                continue
            if e not in cache:
                cache[e] = ckpt.dense_expert(layer, e)
            gate_w, up_w, down_w = cache[e]
            xd = x[t].double()
            g, u = gate_w @ xd, up_w @ xd
            if swiglu_limit is not None:
                clamped += int(
                    (g > swiglu_limit).sum() + (u.abs() > swiglu_limit).sum()
                )
                g, u = g.clamp(max=swiglu_limit), u.clamp(-swiglu_limit, swiglu_limit)
            a = g * torch.sigmoid(g) * u
            out[t] += float(topk_weights[t, k]) * (down_w @ a)
    return out, clamped


def hot_reference_forward(hot):
    """A float stand-in for the hot FusedMoE's FlashInfer call, from its loaded
    parameters (w13 is [up; gate], FlashInfer's order): routes with an id outside
    [0, H) add nothing, which is what FlashInfer's CUTLASS MoE does with a
    non-local id (proved on the GPU, not here)."""

    def forward(hidden_states, topk_output, pre_quant_input=None):
        num = hot.w13_weight.shape[0]
        inter = hot.w13_weight.shape[1] // 2
        w13 = dequant_mxfp4(hot.w13_weight.data, hot.w13_weight_scale_inv.data)
        w2 = dequant_mxfp4(hot.w2_weight.data, hot.w2_weight_scale_inv.data)
        limit = hot.moe_runner_config.swiglu_limit
        ids, weights = topk_output.topk_ids, topk_output.topk_weights
        out = torch.zeros(hidden_states.shape[0], w2.shape[1], dtype=torch.float64)
        for t in range(ids.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k])
                if not 0 <= e < num:
                    continue
                xd = hidden_states[t].double()
                u, g = w13[e, :inter] @ xd, w13[e, inter:] @ xd
                if limit is not None:
                    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
                out[t] += float(weights[t, k]) * (w2[e] @ (g * torch.sigmoid(g) * u))
        return out.to(hidden_states.dtype)

    return forward


def cpu_moe_context():
    """An ExitStack where production's DeepSeek-V4 MoE construction runs on CPU:
    server args on cpu, the FlashInfer MXFP4 runner backend, an SM120 platform
    (so Fp8Config picks Mxfp4FlashinferCutlassMoEMethod and its real
    create_weights), the runner object replaced (the pattern of
    test_fp8_moe_runner_ownership.TestMxfp4MoEMethodContract)."""
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.layers.quantization import fp8 as fp8_module
    from sglang.srt.layers.quantization import (
        mxfp4_flashinfer_cutlass_moe as cutlass_mxfp4,
    )
    from sglang.srt.runtime_context import get_context, get_flags, get_parallel

    stack = ExitStack()
    stack.enter_context(
        get_context().override_server_args(model_path="dummy", device="cpu")
    )
    stack.enter_context(
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE,
            runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4,
        )
    )
    stack.enter_context(
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )
    )
    platform = SimpleNamespace(
        is_sm90=False, is_sm100=False, is_sm120=True, is_hip=False
    )
    stack.enter_context(patch.object(fp8_module, "get_platform", return_value=platform))
    stack.enter_context(
        patch.object(cutlass_mxfp4, "get_platform", return_value=platform)
    )
    stack.enter_context(
        patch.object(cutlass_mxfp4, "is_flashinfer_available", return_value=True)
    )
    for module in (
        "sglang.srt.layers.moe.moe_runner",
        "sglang.srt.layers.moe.moe_runner.runner",
    ):
        stack.enter_context(
            patch(
                f"{module}.MoeRunner",
                side_effect=lambda b, c: SimpleNamespace(runner_backend=b, config=c),
            )
        )
    return stack


def fp4_quant_config():
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        is_fp4_experts=True,
    )


def toy_hf_config(ckpt_path: Optional[str] = None, **overrides):
    """A DeepseekV41Config for the toy; with ckpt_path it names that manifest."""
    from sglang.srt.configs.deepseek_v41 import DeepseekV41Config

    values = toy_config_dict()
    values.update(overrides)
    config = DeepseekV41Config(**values)
    if ckpt_path is not None:
        config.hotcold = {"manifest": "hotcold.json"}
        config.hotcold_manifest_path = os.path.join(ckpt_path, "hotcold.json")
    return config


def load_toy_layer(mlp, ckpt: ToyCheckpoint, layer_id: int) -> None:
    """Load one layer's experts through the production name path: the model's
    renames, then HotColdWeightMap.take (what DeepseekV4ForCausalLM.load_weights runs)."""
    from sglang.srt.model_loader.hotcold_manifest import (
        HotColdWeightMap,
        manifest_of_config,
    )
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    prefix = f"model.layers.{layer_id}.mlp."
    params = {prefix + n: p for n, p in mlp.named_parameters()}
    wm = HotColdWeightMap(manifest_of_config(mlp.config), (layer_id, layer_id + 1))
    with torch.no_grad():
        for raw, tensor in ckpt.weights():
            name = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
                raw, num_hidden_layers=2
            )
            wm.take(name, tensor, params)
    wm.check_complete()
