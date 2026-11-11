# SPDX-License-Identifier: Apache-2.0
"""Two-tier routed experts for DeepSeek-V4 hot/cold checkpoints.

The router is unchanged: TopK picks among all ``n_routed_experts``. After top-k
each route is sent to one tier by two gathers on [T, top_k] with static shapes
and no host sync, indexed by ``topk_ids + 1`` so the padded route (-1) has its
own entry:

* ``hot_ids = remap_hot[ids + 1]``: the hot slot, or H for a cold or padded route;
  FlashInfer's CUTLASS MoE treats an id outside [0, H) as non-local (no GEMM rows,
  skipped at finalize), so such a route adds 0;
* ``cold_ids = remap_cold[ids + 1]``: the cold slot, or -1 for a hot or padded
  route; the GGUF kernels write 0 for a negative id without reading weights.

Both tiers take the full [T, top_k] tensor with sentinels, so neither is padded
or counted per tier, both sit in the same CUDA graph, and the output is
``hot(x) + cold(x)`` with the router weights (the routed scaling factor fused in
them, as the hot method asks). Both tiers clamp the SwiGLU at ``swiglu_limit``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.hotcold import ColdGGUFExperts
from sglang.srt.model_loader.hotcold_manifest import HotColdManifest

# Members of the fused-MoE interface a hot/cold layer does not offer in v1:
# all-to-all / overlap / deferred finalize. Reading one raises.
_UNSUPPORTED = frozenset(
    {
        "set_overlap_args",
        "clear_overlap_args",
        "run_moe_core",
        "forward_deferred_finalize",
        "forward_impl",
        "get_moe_weights",
    }
)


class HotColdQuantMethod:
    """``layer.quant_method`` of a HotColdFusedMoE.

    The loader runs ``process_weights_after_loading`` on every module that has a
    ``quant_method``; the hot FusedMoE gets its own call through its own
    attribute, so this one must not touch it again (the SM120 path interleaves
    the scales in place). It checks the cold tier's kernels. Other reads, such
    as ``isinstance`` probes, see a method of its own type; attribute reads fall
    through to the hot tier's method."""

    def __init__(self, hot_method):
        self._hot_method = hot_method

    def process_weights_after_loading(self, layer: HotColdFusedMoE) -> None:
        layer.cold.check_kernels()

    def __getattr__(self, name):
        if name.startswith("__") or name == "_hot_method":
            raise AttributeError(name)
        return getattr(self._hot_method, name)


class HotColdFusedMoE(nn.Module):
    supports_deferred_finalize = False
    supports_routed_deferred_finalize = False

    def __init__(
        self,
        *,
        manifest: HotColdManifest,
        layer_id: int,
        hot: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: Optional[float],
    ):
        super().__init__()
        entry = manifest.layer(layer_id)
        if entry is None:
            raise ValueError(f"layer {layer_id} is not in the hot/cold manifest")
        self.layer_id = layer_id
        self.num_hot = manifest.num_hot
        self.num_cold = manifest.num_cold
        self.hot = hot
        self.cold = ColdGGUFExperts(
            num_cold=manifest.num_cold,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_type=manifest.w13_type,
            w2_type=manifest.w2_type,
            swiglu_limit=swiglu_limit,
            layer_id=layer_id,
        )
        remap_hot, remap_cold = manifest.remap_tables(layer_id)
        self.register_buffer("remap_hot", remap_hot, persistent=False)
        self.register_buffer("remap_cold", remap_cold, persistent=False)
        self.quant_method = HotColdQuantMethod(getattr(hot, "quant_method", None))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name in _UNSUPPORTED:
            raise AttributeError(f"HotColdFusedMoE does not support {name}")
        modules = self.__dict__.get("_modules", {})
        hot = modules.get("hot")
        if hot is None:
            raise AttributeError(name)
        return getattr(hot, name)

    def split_routes(self, topk_ids: torch.Tensor):
        """(hot_ids, cold_ids), each [T, top_k] in topk_ids' dtype (cold int32)."""
        index = (topk_ids + 1).reshape(-1)
        hot_ids = self.remap_hot.index_select(0, index).view(topk_ids.shape)
        cold_ids = self.remap_cold.index_select(0, index).view(topk_ids.shape)
        if hot_ids.dtype != topk_ids.dtype:
            hot_ids = hot_ids.to(topk_ids.dtype)
        return hot_ids, cold_ids

    def forward(self, hidden_states: torch.Tensor, topk_output, pre_quant_input=None):
        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError(
                f"HotColdFusedMoE needs standard top-k output, got {topk_output.format!r}"
            )
        topk_weights = topk_output.topk_weights
        hot_ids, cold_ids = self.split_routes(topk_output.topk_ids)
        # The packed-id field (FlashInfer routed MoE) encodes logical ids: dropped.
        hot_topk = StandardTopKOutput(topk_weights, hot_ids, topk_output.router_logits)
        if pre_quant_input is None:
            hot_out = self.hot(hidden_states, hot_topk)
        else:
            hot_out = self.hot(hidden_states, hot_topk, pre_quant_input=pre_quant_input)
        cold_out = self.cold(hidden_states, topk_weights, cold_ids)
        return hot_out.add_(cold_out)


def hotcold_layer_enabled(
    config, layer_id: int, *, is_nextn: bool, is_deepseek_v4: bool
) -> Optional[HotColdManifest]:
    """The manifest when this layer's routed experts are two-tier, else None."""
    if is_nextn or not is_deepseek_v4:
        return None
    from sglang.srt.model_loader.hotcold_manifest import manifest_of_config

    manifest = manifest_of_config(config)
    if manifest is None or manifest.layer(layer_id) is None:
        return None
    return manifest
