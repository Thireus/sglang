# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 hot/cold checkpoints: the manifest and the loader's name map.

A hot/cold checkpoint keeps every routed expert of a layer, split in two tiers:

* hot experts keep their native MXFP4 safetensors tensors, under their original
  names and expert ids (``layers.L.ffn.experts.E.{w1,w2,w3}.{weight,scale}``);
* cold experts are stored as ggml-quantised rows, one uint8 tensor per layer and
  projection, slot-major: ``layers.L.ffn.experts_cold.{w1,w3}.qweight``
  ``[C, moe_intermediate, row_bytes(hidden)]`` and ``.w2.qweight``
  ``[C, hidden, row_bytes(moe_intermediate)]``. A row is what
  ``ggml_quantize_chunk`` writes for it: the ik row header (IQ2_KS 2 bytes,
  IQ4_KS 4 bytes, IQ2_S none) and then the blocks.

``config.json`` names the manifest (``"hotcold": {"manifest": "hotcold.json"}``)
and keeps ``n_routed_experts`` at the full count, so the router is unchanged.
The slot of an expert is its position in its tier's list.

The hot count may differ between layers (more hot experts on a pipeline stage
with room): each layer's count is the length of its lists. ``num_hot`` /
``num_cold`` are then None and ``num_hot_of(layer)`` / ``num_cold_of(layer)``
give a layer's counts; a manifest with one (H, C) reads as before. Optional
top-level ``"H"`` / ``"C"`` are checked against the lists: an integer for every
layer, or ``"per-layer"`` with ``"H_by_layer"`` / ``"C_by_layer"``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

MANIFEST_FORMAT = "dsv41-hotcold/1"

# ggml type id -> (row header bytes, bytes per 256-value block).
COLD_GGML_TYPES: Dict[int, Tuple[int, int]] = {
    22: (0, 82),  # IQ2_S
    144: (4, 136),  # IQ4_KS
    145: (2, 70),  # IQ2_KS
}
COLD_PROJECTIONS = ("w1", "w3", "w2")

# Raw checkpoint names, before the model's renames.
_NATIVE_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
_COLD_RAW_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts_cold\.(w[123])\.qweight$")
# Names after DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format.
_COLD_HF_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts_cold\.(gate_proj|up_proj|down_proj)\.qweight$"
)
_HOT_HF_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)

SHARD_TYPE_KEY = "hotcold.ggml_type.{name}"


class HotColdManifestError(ValueError):
    """The checkpoint does not satisfy the hot/cold contract."""


def cold_row_bytes(ggml_type: int, k: int) -> int:
    """``ggml_row_size(type, k)`` for the cold types, row header included."""
    if int(ggml_type) not in COLD_GGML_TYPES:
        raise HotColdManifestError(
            f"unsupported cold ggml type {ggml_type}; supported: {sorted(COLD_GGML_TYPES)}"
        )
    if k % 256:
        raise HotColdManifestError(f"cold rows need K % 256 == 0, got K={k}")
    meta, block = COLD_GGML_TYPES[int(ggml_type)]
    return meta + k // 256 * block


def cold_tensor_name(layer_id: int, projection: str) -> str:
    return f"layers.{layer_id}.ffn.experts_cold.{projection}.qweight"


@dataclasses.dataclass(frozen=True)
class HotColdLayer:
    hot: Tuple[int, ...]
    cold: Tuple[int, ...]
    forced_hot: Tuple[int, ...]
    hot_slot: Dict[int, int]
    cold_slot: Dict[int, int]


@dataclasses.dataclass(frozen=True)
class HotColdManifest:
    path: str
    n_routed_experts: int
    # The one (H, C) of a uniform manifest; None when the counts differ between layers.
    num_hot: Optional[int]
    num_cold: Optional[int]
    ggml_type: Dict[str, int]
    row_bytes: Dict[str, int]
    layers: Dict[int, HotColdLayer]
    raw: dict = dataclasses.field(repr=False, compare=False)

    @property
    def w13_type(self) -> int:
        return self.ggml_type["w1"]

    @property
    def w2_type(self) -> int:
        return self.ggml_type["w2"]

    def layer(self, layer_id: int) -> Optional[HotColdLayer]:
        return self.layers.get(int(layer_id))

    def num_hot_of(self, layer_id: int) -> int:
        """The hot slots of one layer (the hot FusedMoE's expert count there)."""
        return len(self.layers[int(layer_id)].hot)

    def num_cold_of(self, layer_id: int) -> int:
        """The cold slots of one layer (the leading dim of its cold tensors)."""
        return len(self.layers[int(layer_id)].cold)

    @property
    def uniform(self) -> bool:
        return self.num_hot is not None

    def counts_text(self, layer_ids: Optional[Iterable[int]] = None) -> str:
        """ "222 hot + 162 cold" for one size, else each size with its layer ranges."""
        ids = sorted(
            self.layers
            if layer_ids is None
            else (int(x) for x in layer_ids if int(x) in self.layers)
        )
        runs: List[Tuple[int, int, int, int]] = []  # (H, C, first, last)
        for i in ids:
            h, c = self.num_hot_of(i), self.num_cold_of(i)
            if runs and runs[-1][:2] == (h, c) and runs[-1][3] == i - 1:
                runs[-1] = (h, c, runs[-1][2], i)
            else:
                runs.append((h, c, i, i))
        if len({r[:2] for r in runs}) <= 1 and runs:
            return f"{runs[0][0]} hot + {runs[0][1]} cold"
        return ", ".join(f"layers {a}-{b}: {h} hot + {c} cold" for h, c, a, b in runs)

    def remap_tables(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(remap_hot, remap_cold)``: int32 [n_routed_experts + 1], indexed by
        ``topk_ids + 1``. Entry 0 is the padded route (-1): hot -> H, cold -> -1.
        A cold expert maps to H in the hot table (out of the hot range), a hot
        expert to -1 in the cold table."""
        entry = self.layers[int(layer_id)]
        n = self.n_routed_experts
        remap_hot = torch.full((n + 1,), len(entry.hot), dtype=torch.int32)
        remap_cold = torch.full((n + 1,), -1, dtype=torch.int32)
        for expert, slot in entry.hot_slot.items():
            remap_hot[expert + 1] = slot
        for expert, slot in entry.cold_slot.items():
            remap_cold[expert + 1] = slot
        return remap_hot, remap_cold


def _strict_ascending(values: Sequence[int]) -> bool:
    return all(b > a for a, b in zip(values, values[1:]))


def parse_manifest(
    raw: dict,
    *,
    path: str = "<memory>",
    n_routed_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    num_hidden_layers: Optional[int] = None,
) -> HotColdManifest:
    """Check the manifest against the model's shapes; every error is reported together."""
    errors: List[str] = []
    if not isinstance(raw, dict):
        raise HotColdManifestError(f"{path}: the manifest must be a JSON object")
    if raw.get("format") != MANIFEST_FORMAT:
        errors.append(f"format must be {MANIFEST_FORMAT!r}, got {raw.get('format')!r}")
    if raw.get("n_routed_experts") != n_routed_experts:
        errors.append(
            f"n_routed_experts {raw.get('n_routed_experts')!r} differs from the "
            f"config's {n_routed_experts}"
        )
    quant = raw.get("cold_quant") or {}
    types = quant.get("ggml_type") or {}
    rows = quant.get("row_bytes") or {}
    ggml_type: Dict[str, int] = {}
    row_bytes: Dict[str, int] = {}
    k_of = {"w1": hidden_size, "w3": hidden_size, "w2": moe_intermediate_size}
    for proj in COLD_PROJECTIONS:
        t = types.get(proj)
        if not isinstance(t, int) or t not in COLD_GGML_TYPES:
            errors.append(
                f"cold_quant.ggml_type.{proj} must be one of {sorted(COLD_GGML_TYPES)}, got {t!r}"
            )
            continue
        ggml_type[proj] = t
        expected = cold_row_bytes(t, k_of[proj])
        if rows.get(proj) != expected:
            errors.append(
                f"cold_quant.row_bytes.{proj} {rows.get(proj)!r} differs from "
                f"ggml_row_size({t}, {k_of[proj]}) = {expected}"
            )
        row_bytes[proj] = expected
    if "w1" in ggml_type and "w3" in ggml_type and ggml_type["w1"] != ggml_type["w3"]:
        errors.append(
            "cold w1 and w3 must share one ggml type (they are one fused w13 tensor), "
            f"got {ggml_type['w1']} and {ggml_type['w3']}"
        )

    layers_raw = raw.get("layers")
    layers: Dict[int, HotColdLayer] = {}
    sizes = set()
    if not isinstance(layers_raw, dict) or not layers_raw:
        errors.append("layers must be a non-empty object keyed by layer id")
        layers_raw = {}
    full = set(range(n_routed_experts))
    for key, entry in sorted(layers_raw.items(), key=lambda kv: str(kv[0])):
        where = f"layers[{key}]"
        try:
            layer_id = int(key)
        except (TypeError, ValueError):
            errors.append(f"{where}: the key is not a layer id")
            continue
        if num_hidden_layers is not None and not 0 <= layer_id < num_hidden_layers:
            errors.append(f"{where}: layer id outside [0, {num_hidden_layers})")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{where}: must be an object")
            continue
        hot = entry.get("hot")
        cold = entry.get("cold")
        forced = entry.get("forced_hot", [])
        bad = [
            n
            for n, v in (("hot", hot), ("cold", cold), ("forced_hot", forced))
            if not isinstance(v, list) or not all(isinstance(x, int) for x in v)
        ]
        if bad:
            errors.append(f"{where}: {', '.join(bad)} must be lists of expert ids")
            continue
        if not _strict_ascending(hot):
            errors.append(f"{where}: hot is not strictly ascending")
        if not _strict_ascending(cold):
            errors.append(f"{where}: cold is not strictly ascending")
        overlap = sorted(set(hot) & set(cold))
        if overlap:
            errors.append(f"{where}: hot and cold overlap at {overlap[:8]}")
        union = set(hot) | set(cold)
        if union != full:
            missing = sorted(full - union)[:8]
            extra = sorted(union - full)[:8]
            errors.append(
                f"{where}: hot + cold must be range({n_routed_experts}); "
                f"missing {missing}, outside {extra}"
            )
        if not set(forced) <= set(hot):
            errors.append(
                f"{where}: forced_hot {sorted(set(forced) - set(hot))[:8]} not in hot"
            )
        if not hot or not cold:
            errors.append(f"{where}: both tiers need at least one expert")
        sizes.add((len(hot), len(cold)))
        layers[layer_id] = HotColdLayer(
            hot=tuple(hot),
            cold=tuple(cold),
            forced_hot=tuple(forced),
            hot_slot={e: s for s, e in enumerate(hot)},
            cold_slot={e: s for s, e in enumerate(cold)},
        )
    _check_declared_counts(raw, layers, errors)
    if errors:
        details = "\n".join(f"- {e}" for e in errors)
        raise HotColdManifestError(f"Invalid hot/cold manifest {path}:\n{details}")
    (num_hot, num_cold) = next(iter(sizes)) if len(sizes) == 1 else (None, None)
    return HotColdManifest(
        path=path,
        n_routed_experts=n_routed_experts,
        num_hot=num_hot,
        num_cold=num_cold,
        ggml_type=ggml_type,
        row_bytes=row_bytes,
        layers=layers,
        raw=raw,
    )


def _check_declared_counts(
    raw: dict, layers: Dict[int, HotColdLayer], errors: List[str]
) -> None:
    """Top-level "H" / "C" (optional, as the builder writes them) must agree with
    the layers' lists: an integer for every layer, or "per-layer" with
    "H_by_layer" / "C_by_layer" giving each layer's count."""
    for key, by_key, tier in (("H", "H_by_layer", "hot"), ("C", "C_by_layer", "cold")):
        declared = raw.get(key)
        if declared is None:
            continue
        if declared == "per-layer":
            by_layer = raw.get(by_key)
            if not isinstance(by_layer, dict):
                errors.append(f'{key} is "per-layer" but {by_key} is missing')
                continue
            for layer_id, entry in layers.items():
                want = by_layer.get(str(layer_id))
                if want != len(getattr(entry, tier)):
                    errors.append(
                        f"layers[{layer_id}]: {len(getattr(entry, tier))} {tier} experts, "
                        f"{by_key} says {want!r}"
                    )
        elif isinstance(declared, int) and not isinstance(declared, bool):
            bad = sorted(
                i for i, e in layers.items() if len(getattr(e, tier)) != declared
            )
            if bad:
                errors.append(
                    f"{key} is {declared} but layers {bad[:8]} hold "
                    f"{sorted({len(getattr(layers[i], tier)) for i in bad})} {tier} experts"
                )
        else:
            errors.append(f'{key} must be an integer or "per-layer", got {declared!r}')


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def check_checkpoint_index(model_dir: str, manifest: HotColdManifest) -> None:
    """Index-level checks that need no tensor data: every cold tensor is present
    with its type tag in its shard's metadata and its shape, and no native tensor
    of a cold expert is listed."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise HotColdManifestError(f"{index_path} is missing")
    with open(index_path) as f:
        weight_map = json.load(f).get("weight_map") or {}
    errors: List[str] = []
    for name in weight_map:
        m = _NATIVE_EXPERT_RE.match(name)
        if m is None:
            continue
        entry = manifest.layer(int(m.group(1)))
        if entry is not None and int(m.group(2)) in entry.cold_slot:
            errors.append(f"{name}: a native tensor of cold expert {m.group(2)}")
            if len(errors) > 16:
                break
    headers: Dict[str, dict] = {}
    for layer_id in sorted(manifest.layers):
        for proj in COLD_PROJECTIONS:
            name = cold_tensor_name(layer_id, proj)
            shard = weight_map.get(name)
            if shard is None:
                errors.append(f"{name}: missing from the index")
                continue
            if shard not in headers:
                headers[shard] = _read_safetensors_header(
                    os.path.join(model_dir, shard)
                )
            header = headers[shard]
            meta = header.get(name) or {}
            tag = (header.get("__metadata__") or {}).get(
                SHARD_TYPE_KEY.format(name=name)
            )
            if tag is None or str(tag) != str(manifest.ggml_type[proj]):
                errors.append(
                    f"{name}: shard type tag {tag!r} differs from the manifest's "
                    f"{manifest.ggml_type[proj]}"
                )
            if meta.get("dtype") != "U8":
                errors.append(f"{name}: dtype {meta.get('dtype')!r}, expected U8")
            rows = None
            shape = meta.get("shape")
            if isinstance(shape, list) and len(shape) == 3:
                rows = shape[2]
            num_cold = manifest.num_cold_of(layer_id)
            if (
                shape is None
                or shape[0] != num_cold
                or rows != manifest.row_bytes[proj]
            ):
                errors.append(
                    f"{name}: shape {shape} needs [{num_cold}, *, "
                    f"{manifest.row_bytes[proj]}]"
                )
    if errors:
        details = "\n".join(f"- {e}" for e in errors[:32])
        raise HotColdManifestError(
            f"Hot/cold checkpoint {model_dir} refused:\n{details}"
        )


_CACHE: Dict[Tuple[str, int, int, int, Optional[int]], HotColdManifest] = {}


def load_manifest(
    path: str,
    *,
    n_routed_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    num_hidden_layers: Optional[int] = None,
) -> HotColdManifest:
    key = (
        os.path.realpath(path),
        n_routed_experts,
        hidden_size,
        moe_intermediate_size,
        num_hidden_layers,
    )
    if key not in _CACHE:
        with open(path) as f:
            raw = json.load(f)
        _CACHE[key] = parse_manifest(
            raw,
            path=path,
            n_routed_experts=n_routed_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
        )
    return _CACHE[key]


def hotcold_manifest_path(config, model_path: Optional[str] = None) -> Optional[str]:
    """The manifest path a config names, or None for an ordinary checkpoint."""
    resolved = getattr(config, "hotcold_manifest_path", None)
    if resolved:
        return resolved
    spec = getattr(config, "hotcold", None)
    if not spec:
        return None
    if not isinstance(spec, dict) or not isinstance(spec.get("manifest"), str):
        raise HotColdManifestError(
            f'config "hotcold" must be {{"manifest": "<file>"}}, got {spec!r}'
        )
    path = spec["manifest"]
    if not os.path.isabs(path):
        base = model_path or getattr(config, "_name_or_path", None) or ""
        path = os.path.join(base, path)
    return path


def manifest_of_config(
    config, model_path: Optional[str] = None
) -> Optional[HotColdManifest]:
    path = hotcold_manifest_path(config, model_path)
    if path is None:
        return None
    return load_manifest(
        path,
        n_routed_experts=int(config.n_routed_experts),
        hidden_size=int(config.hidden_size),
        moe_intermediate_size=int(config.moe_intermediate_size),
        num_hidden_layers=int(config.num_hidden_layers),
    )


def attach_hotcold_manifest(
    configs: Iterable, model_path: str
) -> Optional[HotColdManifest]:
    """Validate a hot/cold checkpoint at config time and record the resolved
    manifest path on every config object (they may be one object)."""
    configs = [c for c in configs if c is not None]
    owner = next((c for c in configs if getattr(c, "hotcold", None)), None)
    if owner is None:
        return None
    path = hotcold_manifest_path(owner, model_path)
    if not os.path.exists(path):
        raise HotColdManifestError(f"hot/cold manifest {path} does not exist")
    text = next(
        (c for c in configs if getattr(c, "n_routed_experts", None) is not None), owner
    )
    manifest = load_manifest(
        path,
        n_routed_experts=int(text.n_routed_experts),
        hidden_size=int(text.hidden_size),
        moe_intermediate_size=int(text.moe_intermediate_size),
        num_hidden_layers=int(text.num_hidden_layers),
    )
    indexed = os.path.isdir(model_path)
    if indexed:
        check_checkpoint_index(model_path, manifest)
    for c in configs:
        c.hotcold_manifest_path = path
    logger.info(
        "hot/cold manifest %s: format %s, %d layers, %s of %d routed "
        "experts a layer, cold ggml types w1 %d w3 %d w2 %d, row bytes %d / %d / %d, "
        "index %s",
        path,
        MANIFEST_FORMAT,
        len(manifest.layers),
        manifest.counts_text(),
        manifest.n_routed_experts,
        manifest.ggml_type["w1"],
        manifest.ggml_type["w3"],
        manifest.ggml_type["w2"],
        manifest.row_bytes["w1"],
        manifest.row_bytes["w3"],
        manifest.row_bytes["w2"],
        "checked" if indexed else "not checked",
    )
    return manifest


class HotColdWeightMap:
    """The loader side of a hot/cold checkpoint for one model (not the draft).

    ``take`` sees each tensor after the model's renames and the stage filter:
    a hot expert ``model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,
    weight_scale_inv}`` loads into slot S of the hot FusedMoE
    (``mlp.experts.hot.w13_*`` / ``w2_*``) through its own weight_loader; a
    native tensor of a cold expert is refused; the cold tensors
    ``model.layers.L.mlp.experts_cold.{gate,up,down}_proj.qweight`` load into
    ``mlp.experts.cold.{w13,w2}_qweight``. Anything else is not taken.
    ``check_complete`` proves the count: 6·H distinct hot tensors (H of that
    layer) and 3 cold ones per layer of this stage.
    """

    _COLD_SHARD = {
        "gate_proj": ("w13_qweight", 0),
        "up_proj": ("w13_qweight", 1),
        "down_proj": ("w2_qweight", None),
    }
    _HOT_SHARD = {
        "gate_proj": ("w13_", "w1"),
        "up_proj": ("w13_", "w3"),
        "down_proj": ("w2_", "w2"),
    }

    def __init__(
        self, manifest: HotColdManifest, layer_range: Optional[Tuple[int, int]] = None
    ):
        self.manifest = manifest
        self.layer_range = layer_range
        self.hot_seen: Dict[int, set] = {}
        self.cold_loaded: Dict[int, set] = {}

    def _on_stage(self, layer_id: int) -> bool:
        if self.layer_range is None:
            return True
        start, end = self.layer_range
        return start <= layer_id < end

    def _param(self, params_dict: dict, param_name: str, name: str, layer_id: int):
        param = params_dict.get(param_name)
        if param is None:
            raise HotColdManifestError(
                f"{name}: layer {layer_id} has no {param_name}; the model was not "
                "built from this manifest"
            )
        return param

    def take(self, name: str, tensor: torch.Tensor, params_dict: dict) -> Optional[str]:
        """Load ``name`` if it belongs to a hot/cold layer of this stage; returns
        the parameter name it loaded, or None when the tensor is not taken."""
        m = _HOT_HF_RE.match(name)
        if m is not None:
            layer_id, expert = int(m.group(1)), int(m.group(2))
            entry = self.manifest.layer(layer_id)
            if entry is None or not self._on_stage(layer_id):
                return None
            if expert in entry.cold_slot:
                raise HotColdManifestError(
                    f"{name}: the checkpoint holds a native tensor of cold expert "
                    f"{expert} of layer {layer_id}"
                )
            if expert not in entry.hot_slot:
                raise HotColdManifestError(
                    f"{name}: expert {expert} is in neither tier"
                )
            slot = entry.hot_slot[expert]
            key = (slot, m.group(3), m.group(4))
            seen = self.hot_seen.setdefault(layer_id, set())
            if key in seen:
                raise HotColdManifestError(f"{name}: seen twice")
            seen.add(key)
            prefix, shard_id = self._HOT_SHARD[m.group(3)]
            param_name = f"model.layers.{layer_id}.mlp.experts.hot.{prefix}{m.group(4)}"
            param = self._param(params_dict, param_name, name, layer_id)
            param.weight_loader(
                param, tensor, param_name, shard_id=shard_id, expert_id=slot
            )
            return param_name
        m = _COLD_HF_RE.match(name)
        if m is not None:
            layer_id, proj = int(m.group(1)), m.group(2)
            if self.manifest.layer(layer_id) is None:
                raise HotColdManifestError(
                    f"{name}: layer {layer_id} is not in the manifest"
                )
            if not self._on_stage(layer_id):
                return None
            param_leaf, shard_id = self._COLD_SHARD[proj]
            param_name = f"model.layers.{layer_id}.mlp.experts.cold.{param_leaf}"
            param = self._param(params_dict, param_name, name, layer_id)
            seen = self.cold_loaded.setdefault(layer_id, set())
            if proj in seen:
                raise HotColdManifestError(f"{name}: seen twice")
            param.weight_loader(param, tensor, shard_id)
            seen.add(proj)
            return param_name
        return None

    def check_complete(self) -> None:
        errors = []
        stage, hot_total, cold_total = [], 0, 0
        for layer_id in sorted(self.manifest.layers):
            if not self._on_stage(layer_id):
                continue
            expect_hot = 6 * self.manifest.num_hot_of(layer_id)
            hot = len(self.hot_seen.get(layer_id, ()))
            cold = len(self.cold_loaded.get(layer_id, ()))
            stage.append(layer_id)
            hot_total += hot
            cold_total += cold
            if hot != expect_hot or cold != 3:
                errors.append(
                    f"layer {layer_id}: {hot} hot tensors (expected {expect_hot}), "
                    f"{cold} cold tensors (expected 3)"
                )
        if errors:
            details = "\n".join(f"- {e}" for e in errors)
            raise HotColdManifestError(f"Hot/cold load incomplete:\n{details}")
        if not stage or len({self.manifest.num_hot_of(i) for i in stage}) == 1:
            # one size on this stage: the line the boot proof reads, unchanged
            h = self.manifest.num_hot_of(stage[0]) if stage else 0
            c = self.manifest.num_cold_of(stage[0]) if stage else 0
            logger.info(
                "hot/cold load: layers %s..%s (%d), %d hot tensors (%d a layer, %d slots), "
                "%d cold tensors (3 a layer, %d slots)",
                stage[0] if stage else "none",
                stage[-1] if stage else "none",
                len(stage),
                hot_total,
                6 * h,
                h,
                cold_total,
                c,
            )
        else:
            logger.info(
                "hot/cold load: layers %s..%s (%d), %d hot tensors, %d cold tensors (3 a layer); %s",
                stage[0],
                stage[-1],
                len(stage),
                hot_total,
                cold_total,
                self.manifest.counts_text(stage),
            )

    def _loaded_slice(self, params_dict: dict, name: str) -> Optional[torch.Tensor]:
        """The parameter slice a raw checkpoint tensor was loaded into: hot
        ``w13_*[slot]`` in FlashInfer's [up; gate] order (w3 rows, then w1),
        ``w2_*[slot]``; cold ``w13_qweight[:, :I]`` (w1), ``[:, I:]`` (w3),
        ``w2_qweight``. None for a tensor this map does not own."""
        m = _NATIVE_EXPERT_RE.match(name)
        if m is not None:
            layer_id, expert, proj, kind = (
                int(m.group(1)),
                int(m.group(2)),
                m.group(3),
                m.group(4),
            )
            entry = self.manifest.layer(layer_id)
            if entry is None or expert not in entry.hot_slot:
                return None
            leaf = "weight" if kind == "weight" else "weight_scale_inv"
            prefix = "w2_" if proj == "w2" else "w13_"
            param = params_dict.get(
                f"model.layers.{layer_id}.mlp.experts.hot.{prefix}{leaf}"
            )
            if param is None:
                return None
            rows = param.data[entry.hot_slot[expert]]
            if proj == "w2":
                return rows
            half = rows.shape[0] // 2
            return rows[:half] if proj == "w3" else rows[half:]
        m = _COLD_RAW_RE.match(name)
        if m is not None:
            layer_id, proj = int(m.group(1)), m.group(2)
            leaf = "w2_qweight" if proj == "w2" else "w13_qweight"
            param = params_dict.get(f"model.layers.{layer_id}.mlp.experts.cold.{leaf}")
            if param is None:
                return None
            if proj == "w2":
                return param.data
            half = param.data.shape[1] // 2
            return param.data[:, :half] if proj == "w1" else param.data[:, half:]
        return None

    def verify_against_shards(
        self, params_dict: dict, model_dir: str
    ) -> Dict[int, dict]:
        """Re-read every hot and cold tensor of this stage's hot/cold layers from
        the checkpoint's shards and compare its bytes with the parameter slice it
        was loaded into (``_loaded_slice``). Run after the load and before
        ``process_weights_after_loading``: the SM120 hot method interleaves its
        scales in place. One log line per layer (an error line on a mismatch);
        returns {layer: {"hot": (equal, total), "cold": (equal, total),
        "mismatch": [names]}}. Diagnostics (SGLANG_DSV41_HOTCOLD_VERIFY_LOAD)."""
        from safetensors import safe_open

        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            weight_map = json.load(f).get("weight_map") or {}
        by_layer: Dict[int, List[Tuple[str, str]]] = {}
        for name, shard in weight_map.items():
            m = _NATIVE_EXPERT_RE.match(name) or _COLD_RAW_RE.match(name)
            if m is None:
                continue
            layer_id = int(m.group(1))
            if self.manifest.layer(layer_id) is None or not self._on_stage(layer_id):
                continue
            by_layer.setdefault(layer_id, []).append((shard, name))
        handles = {}
        report: Dict[int, dict] = {}
        for layer_id in sorted(by_layer):
            counts = {"hot": [0, 0], "cold": [0, 0]}
            mismatch: List[str] = []
            for shard, name in sorted(by_layer[layer_id]):
                tier = "cold" if _COLD_RAW_RE.match(name) else "hot"
                counts[tier][1] += 1
                if shard not in handles:
                    handles[shard] = safe_open(
                        os.path.join(model_dir, shard), framework="pt", device="cpu"
                    )
                dest = self._loaded_slice(params_dict, name)
                src = handles[shard].get_tensor(name)
                if dest is not None and _same_bytes(dest, src):
                    counts[tier][0] += 1
                else:
                    mismatch.append(
                        name if dest is not None else f"{name} (no parameter)"
                    )
            report[layer_id] = {
                "hot": tuple(counts["hot"]),
                "cold": tuple(counts["cold"]),
                "mismatch": mismatch,
            }
            line = (
                f"hot/cold verify layer {layer_id}: hot {counts['hot'][0]}/{counts['hot'][1]} and cold "
                f"{counts['cold'][0]}/{counts['cold'][1]} tensors byte-equal to the shards"
            )
            if mismatch:
                logger.error(
                    "%s; MISMATCH %d: %s", line, len(mismatch), ", ".join(mismatch[:8])
                )
            else:
                logger.info("%s", line)
        handles.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return report


def _same_bytes(dest: torch.Tensor, src: torch.Tensor) -> bool:
    """Whether a loaded parameter slice holds exactly the bytes of the checkpoint
    tensor (dtype ignored: E8M0 scales and packed FP4 compare as bytes)."""
    if (
        tuple(dest.shape) != tuple(src.shape)
        or dest.element_size() != src.element_size()
    ):
        return False
    if dest.element_size() == 1:
        a, b = dest.view(torch.uint8), src.view(torch.uint8)
    else:
        a, b = (
            dest.contiguous().view(-1).view(torch.uint8),
            src.contiguous().view(-1).view(torch.uint8),
        )
    return bool(torch.equal(a, b.to(a.device)))
