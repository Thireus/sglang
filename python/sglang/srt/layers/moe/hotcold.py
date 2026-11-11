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

On CUDA both gathers are one kernel (``SGLANG_HOTCOLD_SPLIT_KERNEL``); on a decode
batch that kernel also groups the cold routes per cold expert for the cold tier's
multi-route kernel (``SGLANG_HOTCOLD_DECODE_ALIGN``). When nothing reads the cold
tier's own output the cold tier adds its sum into the hot tier's output itself:
on a decode batch the kernel that applies the route weights and sums over top-k
does it (``SGLANG_HOTCOLD_DECODE_SUM_FOLD``), else ``add_``. The same output bits
either way.

Both tiers take the full [T, top_k] tensor with sentinels, so neither is padded
or counted per tier, both sit in the same CUDA graph, and the output is
``hot(x) + cold(x)`` with the router weights (the routed scaling factor fused in
them, as the hot method asks). Both tiers clamp the SwiGLU at ``swiglu_limit``.

Diagnostics (environ.py): ``SGLANG_DSV41_HOTCOLD_ZERO_COLD`` drops the cold
tier's output; ``SGLANG_DSV41_HOTCOLD_VERIFY_LOAD`` arms ``arm_recheck`` after the
loader's shard check, and the first eager forward of the layer (then every
``SGLANG_DSV41_HOTCOLD_RECHECK_EVERY`` eager forwards) compares a checksum of the
cold rows and of the hot weights (not the scales, which the SM120 post-load
interleaves) with the one taken at load. ``SGLANG_DSV41_HOTCOLD_STATS`` logs the
first real forwards: the model sets ``set_real_forward`` around each forward
(deepseek_v4.py, the filter the per-layer dump uses too), and while a dump forward
runs it sets ``_DUMP_SINK``, which receives each layer's route split and tier
outputs before they are summed.
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
# Set by the model around each forward (deepseek_v4.py _Dsv41Diag): whether it is a
# real extend forward, and the dump sink while one is dumped. Unset: stats never log.
_REAL_FORWARD = False
_DUMP_SINK = None


def set_real_forward(real: bool) -> None:
    global _REAL_FORWARD
    _REAL_FORWARD = bool(real)


def set_dump_sink(sink) -> None:
    """``sink(layer_id, **tensors)`` copies what it keeps; None to stop."""
    global _DUMP_SINK
    _DUMP_SINK = sink


# Kernel seams: the CPU tests replace them to drive split_routes' CUDA branch.
def _split_kernel_serves(topk_ids) -> bool:
    """Whether the one-launch route split takes these ids (the engine's: contiguous int32 on CUDA)."""
    return (
        topk_ids.is_cuda and topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
    )


def _split_kernel(topk_ids, remap_hot, remap_cold):
    from sglang.kernels.ops.moe.gguf_moe_split import gguf_moe_split

    return gguf_moe_split(topk_ids, remap_hot, remap_cold)


def _split_align_kernel(topk_ids, remap_hot, remap_cold, num_cold, block):
    """(hot_ids, cold_ids, sorted_ids, expert_ids, num_post_padded): the split and the
    cold routes grouped per cold expert in blocks, one launch."""
    from sglang.kernels.ops.moe.gguf_moe_split import gguf_moe_split_align

    return gguf_moe_split_align(topk_ids, remap_hot, remap_cold, num_cold, block)


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
        # this layer's counts: a manifest may give each layer its own H (per pipeline stage)
        self.num_hot = manifest.num_hot_of(layer_id)
        self.num_cold = manifest.num_cold_of(layer_id)
        if getattr(hot, "num_experts", self.num_hot) != self.num_hot:
            raise ValueError(
                f"hot/cold layer {layer_id}: the hot FusedMoE has {hot.num_experts} experts, "
                f"the manifest {self.num_hot} hot slots"
            )
        self.hot = hot
        self.cold = ColdGGUFExperts(
            num_cold=self.num_cold,
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
        self._stats_left = envs.SGLANG_DSV41_HOTCOLD_STATS.get()
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
        if _split_kernel_serves(topk_ids) and envs.SGLANG_HOTCOLD_SPLIT_KERNEL.get():
            # the add and the two gathers below in one launch: the same integers
            return _split_kernel(topk_ids, self.remap_hot, self.remap_cold)
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
        topk_ids = topk_output.topk_ids
        # On a decode batch the split's launch also groups the cold routes for the
        # cold tier's multi-route kernel (SGLANG_HOTCOLD_DECODE_ALIGN); else the cold
        # tier aligns them itself. The same routes in every block's expert either way.
        aligned = None
        block = 0
        if (
            not self.zero_cold
            and _split_kernel_serves(topk_ids)
            and envs.SGLANG_HOTCOLD_SPLIT_KERNEL.get()
        ):
            block = self.cold.decode_align_block(hidden_states.shape[0])
        if block:
            hot_ids, cold_ids, sorted_ids, expert_ids, post_padded = (
                _split_align_kernel(
                    topk_ids, self.remap_hot, self.remap_cold, self.num_cold, block
                )
            )
            aligned = (block, sorted_ids, expert_ids, post_padded)
        else:
            hot_ids, cold_ids = self.split_routes(topk_ids)
        # The packed-id field (FlashInfer routed MoE) encodes logical ids: dropped.
        hot_topk = StandardTopKOutput(topk_weights, hot_ids, topk_output.router_logits)
        if pre_quant_input is None:
            hot_out = self.hot(hidden_states, hot_topk)
        else:
            hot_out = self.hot(hidden_states, hot_topk, pre_quant_input=pre_quant_input)
        sink = _DUMP_SINK
        if self.zero_cold:
            if sink is not None:
                sink(self.layer_id, hot_ids=hot_ids, cold_ids=cold_ids, hot_out=hot_out)
            return hot_out
        stats = (
            self._stats_left > 0
            and _REAL_FORWARD
            and hidden_states.shape[0] > 0
            and not (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing())
        )
        if sink is None and not stats:
            # Nothing reads the cold tier's own output: the cold tier adds its sum into
            # the hot tier's output itself (add_, or on a decode batch the kernel that
            # makes the sum; the same bits: quantization/hotcold.py).
            return self.cold(
                hidden_states,
                topk_weights,
                cold_ids,
                fold_into=hot_out,
                aligned=aligned,
            )
        cold_out = self.cold(hidden_states, topk_weights, cold_ids, aligned=aligned)
        if sink is not None:
            # Before the in-place sum below: the sink copies both tiers.
            sink(
                self.layer_id,
                hot_ids=hot_ids,
                cold_ids=cold_ids,
                hot_out=hot_out,
                cold_out=cold_out,
            )
        if stats:
            self._stats_left -= 1
            self._log_stats(
                hidden_states,
                topk_weights,
                hot_ids,
                cold_ids,
                hot_topk,
                hot_out,
                cold_out,
            )
        return hot_out.add_(cold_out)

    def _log_stats(
        self, x, topk_weights, hot_ids, cold_ids, hot_topk, hot_out, cold_out
    ) -> None:
        """Diagnostics (SGLANG_DSV41_HOTCOLD_STATS): one line of routing and output
        sizes, and both tiers recomputed on a copy of the input, the cold one with
        its other kernel when that kernel is built for its types."""
        from sglang.srt.layers.quantization.hotcold import (
            _mmq_types,
            select_cold_kernel,
        )

        tokens = x.shape[0]
        hot_route = hot_ids < self.num_hot
        weights = topk_weights.float()
        mass = float((weights * hot_route).sum() / weights.sum().clamp_min(1e-30))
        all_cold = float((~hot_route).all(dim=1).float().mean())
        xf = x.float()
        rms = lambda t: float(t.float().pow(2).mean().sqrt())
        kernel = select_cold_kernel(tokens, self.cold.w13_type, self.cold.w2_type)
        other = "vec" if kernel == "mmq" else "mmq"
        if (
            other == "mmq"
            and not {self.cold.w13_type, self.cold.w2_type} <= _mmq_types()
        ):
            other = None
        x2 = x.clone()
        hot2 = self.hot(x2, hot_topk)
        hot_diff = float((hot2.float() - hot_out.float()).abs().max())
        if other is not None and x.is_cuda:
            cold2 = self.cold(x2, topk_weights, cold_ids, kernel=other)
            cold_rel = float(
                (cold2.float() - cold_out.float()).norm()
                / cold_out.float().norm().clamp_min(1e-30)
            )
            other_txt = f"cold({other}) vs cold({kernel}) rel {cold_rel:.2e}"
        else:
            other_txt = f"cold kernel {kernel}"
        nonfinite = (
            int((~torch.isfinite(hot_out)).sum()),
            int((~torch.isfinite(cold_out)).sum()),
        )
        logger.info(
            "hot/cold stats layer %d: %d tokens, x %s %s rms %.3g max %.3g, weight sum/token %.4f, "
            "hot routes %.1f %% (weight mass %.1f %%), all-cold tokens %.1f %%, out rms hot %.3g cold %.3g, "
            "nonfinite hot/cold %d/%d; recompute on a copy: hot max|diff| %.2e, %s",
            self.layer_id,
            tokens,
            str(x.dtype).replace("torch.", ""),
            "contiguous" if x.is_contiguous() else f"strides {tuple(x.stride())}",
            rms(xf),
            float(xf.abs().max()),
            float(weights.sum(dim=1).mean()),
            100 * float(hot_route.float().mean()),
            100 * mass,
            100 * all_cold,
            rms(hot_out),
            rms(cold_out),
            nonfinite[0],
            nonfinite[1],
            hot_diff,
            other_txt,
        )


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
