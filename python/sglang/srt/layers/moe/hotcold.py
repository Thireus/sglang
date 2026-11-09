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

Diagnostics (environ.py): ``SGLANG_DSV41_HOTCOLD_ZERO_COLD`` drops the cold
tier's output; ``SGLANG_DSV41_HOTCOLD_VERIFY_LOAD`` arms ``arm_recheck`` after the
loader's shard check, and the first eager forward of the layer (then every
``SGLANG_DSV41_HOTCOLD_RECHECK_EVERY`` eager forwards) compares a checksum of the
cold rows and of the hot weights (not the scales, which the SM120 post-load
interleaves) with the one taken at load.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.hotcold import ColdGGUFExperts
from sglang.srt.model_loader.hotcold_manifest import HotColdManifest

logger = logging.getLogger(__name__)

_ZERO_COLD_LOGGED = False
_DIGEST_CHUNK = 1 << 24


def _digest(tensors) -> torch.Tensor:
    """int64 [2]: the sum and a position-weighted sum of the 32-bit words of the
    tensors' bytes (a trailing partial word summed by byte), on their device."""
    acc = None
    for t in tensors:
        b = t.detach().reshape(-1).view(torch.uint8)
        if acc is None:
            acc = torch.zeros(2, dtype=torch.int64, device=b.device)
        n4 = b.numel() // 4 * 4
        words = b[:n4].view(torch.int32)
        for start in range(0, words.numel(), _DIGEST_CHUNK):
            c = words[start : start + _DIGEST_CHUNK].to(torch.int64)
            pos = (
                torch.arange(
                    start, start + c.numel(), dtype=torch.int64, device=c.device
                )
                % 65521
                + 1
            )
            acc[0] += c.sum()
            acc[1] += (c * pos).sum()
        if n4 < b.numel():
            acc[0] += b[n4:].to(torch.int64).sum()
    return acc


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
        self.zero_cold = envs.SGLANG_DSV41_HOTCOLD_ZERO_COLD.get()
        self._recheck_every = envs.SGLANG_DSV41_HOTCOLD_RECHECK_EVERY.get()
        self._recheck_ref = None
        self._eager_forwards = 0
        global _ZERO_COLD_LOGGED
        if self.zero_cold and not _ZERO_COLD_LOGGED:
            _ZERO_COLD_LOGGED = True
            logger.warning(
                "hot/cold: SGLANG_DSV41_HOTCOLD_ZERO_COLD=1, the cold tier's output is "
                "dropped in every hot/cold layer (diagnostics only)"
            )

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

    def param_digest(self) -> torch.Tensor:
        """int64 [2, 2]: _digest of the hot weights (w13, w2; not the scales) and
        of the cold rows."""
        return torch.stack(
            [
                _digest([self.hot.w13_weight, self.hot.w2_weight]),
                _digest([self.cold.w13_qweight, self.cold.w2_qweight]),
            ]
        )

    def arm_recheck(self) -> None:
        """Take the reference checksum (the loader calls this after its shard check)."""
        self._recheck_ref = self.param_digest()

    def _maybe_recheck(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        self._eager_forwards += 1
        n = self._eager_forwards
        if n != 1 and (self._recheck_every <= 0 or n % self._recheck_every):
            return
        now = self.param_digest()
        hot_ok = bool(torch.equal(now[0], self._recheck_ref[0]))
        cold_ok = bool(torch.equal(now[1], self._recheck_ref[1]))
        line = (
            f"hot/cold recheck layer {self.layer_id}, eager forward {n} "
            f"({hidden_states.shape[0]} tokens): cold rows "
            f"{'unchanged' if cold_ok else 'CHANGED'}, hot weights "
            f"{'unchanged' if hot_ok else 'CHANGED'} since the shard check"
        )
        if hot_ok and cold_ok:
            logger.info("%s", line)
        else:
            logger.error("%s", line)

    def forward(self, hidden_states: torch.Tensor, topk_output, pre_quant_input=None):
        if self._recheck_ref is not None:
            self._maybe_recheck(hidden_states)
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
        if self.zero_cold:
            return hot_out
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
