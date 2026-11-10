# SPDX-License-Identifier: Apache-2.0
"""The cold tier of a DeepSeek-V4 hot/cold MoE layer: ggml-quantised experts.

``ColdGGUFExperts`` holds the layer's cold experts as packed ggml rows,
preallocated from the manifest (no lazy GGUF parameter, nothing to materialise):
``w13_qweight`` uint8 [C, 2·I, row_bytes(hidden)], gate rows then up rows, and
``w2_qweight`` uint8 [C, hidden, row_bytes(I)].

``cold_moe`` runs the routes whose cold id is >= 0 (every other route, the
padded ones included, gives exactly 0): gate/up projection, the clamped SwiGLU
(``swiglu_limit``, the same clamp the hot tier applies inside FlashInfer), down
projection, route weights, sum over top-k. On CUDA the projections are the
GGUF vector kernel or, above ``SGLANG_HOTCOLD_MMQ_MIN_TOKENS`` tokens, the MoE
MMQ kernel when it is built for the cold types (``SGLANG_HOTCOLD_COLD_KERNEL``
switches). An eager MMQ call (a prefill) keeps the rows of the cold routes only
(``SGLANG_HOTCOLD_PREFILL_COMPACT``): same bits, a fraction of the memory
traffic. Off CUDA the projections are a float reference that dequantises the
rows (IQ2_S through gguf-py); it is what the CPU tests run and it does not
reproduce the kernels' q8_1 activation rounding.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.model_loader.hotcold_manifest import cold_row_bytes

logger = logging.getLogger(__name__)

# The vector kernel's grid limit: tokens * top_k <= 65535 (gguf_moe_vec.cuh).
VEC_MAX_ROWS = 65535

# check_kernels logs the cold tier's kernels once a process.
_KERNELS_LOGGED = False


def _mmq_types():
    try:
        from sglang.kernels.ops.moe.gguf_moe_mmq import GGUF_MOE_MMQ_TYPES
    except ImportError:
        return frozenset()
    return frozenset(GGUF_MOE_MMQ_TYPES)


_COLD_KERNEL_MODES = ("auto", "vec", "vecm", "mmq", "legacy")
_IQ2_S = 22
# Routes of one expert per CTA the wide-block MMQ kernel is built for (gguf_moe_mmq_wide.py).
_MMQ_WIDE_BLOCKS = (16, 32, 64, 128)
# The same for the staged MMQ kernel (gguf_moe_mmq_staged.py).
_MMQ_STAGED_BLOCKS = (32, 64)


def _cold_kernel_mode() -> str:
    mode = envs.SGLANG_HOTCOLD_COLD_KERNEL.get()
    if mode not in _COLD_KERNEL_MODES:
        raise ValueError(
            f"SGLANG_HOTCOLD_COLD_KERNEL must be auto, vec, vecm, mmq or legacy, got {mode!r}"
        )
    return mode


def select_cold_kernel(tokens: int, w13_type: int, w2_type: int) -> str:
    """'mmq', 'vecm' or 'vec' for a batch of ``tokens``; the same answer for a given
    (tokens, types), so a captured graph and its replay agree. The choice never
    looks at the routes: a captured graph has no data-dependent branch.

    auto: MMQ above SGLANG_HOTCOLD_MMQ_MIN_TOKENS tokens when it is built for the
    types; up to that many tokens, for IQ2_S, the multi-route vector kernel from
    SGLANG_HOTCOLD_VEC_MULTI_MIN_TOKENS tokens (SGLANG_HOTCOLD_VEC_MULTI=0 turns it
    off), else the vector kernel. legacy: the earlier kernels (the vector
    kernel at one row per CTA, MMQ in blocks of 16)."""
    mode = _cold_kernel_mode()
    if mode == "vec":
        return "vec"
    if mode == "vecm":
        if {int(w13_type), int(w2_type)} != {22}:
            raise RuntimeError("SGLANG_HOTCOLD_COLD_KERNEL=vecm serves IQ2_S (22) only")
        return "vecm"
    supported = {int(w13_type), int(w2_type)} <= _mmq_types()
    if mode == "mmq":
        if not supported:
            raise RuntimeError(
                f"SGLANG_HOTCOLD_COLD_KERNEL=mmq but gguf_moe_mmq does not serve types "
                f"{w13_type}/{w2_type}"
            )
        return "mmq"
    if supported and tokens > envs.SGLANG_HOTCOLD_MMQ_MIN_TOKENS.get():
        return "mmq"
    if (
        mode == "auto"
        and {int(w13_type), int(w2_type)} == {_IQ2_S}
        and envs.SGLANG_HOTCOLD_VEC_MULTI.get()
        and envs.SGLANG_HOTCOLD_VEC_MULTI_MIN_TOKENS.get()
        <= tokens
        <= envs.SGLANG_HOTCOLD_MMQ_MIN_TOKENS.get()
    ):
        return "vecm"
    return "vec"


def select_mmq_wide(tokens: int, w13_type: int, w2_type: int) -> int:
    """Routes of one expert per CTA of the wide-block MMQ kernel for this batch
    (SGLANG_HOTCOLD_MMQ_WIDE; IQ2_S only), or 0 for the 16-route kernel. Same bits
    either way; one answer per (tokens, types)."""
    if _cold_kernel_mode() == "legacy" or {int(w13_type), int(w2_type)} != {_IQ2_S}:
        return 0
    wide = envs.SGLANG_HOTCOLD_MMQ_WIDE.get()
    if wide <= 0:
        return 0
    if wide not in _MMQ_WIDE_BLOCKS:
        raise ValueError(
            f"SGLANG_HOTCOLD_MMQ_WIDE must be 0 or one of {_MMQ_WIDE_BLOCKS}, got {wide}"
        )
    return wide


def select_mmq_staged(tokens: int, w13_type: int, w2_type: int) -> int:
    """Routes of one expert per CTA of the staged MMQ kernel for this batch
    (SGLANG_HOTCOLD_MMQ_STAGED; IQ2_S only), or 0 for the wide-block or 16-route
    kernel. Same bits either way; one answer per (tokens, types)."""
    if _cold_kernel_mode() == "legacy" or {int(w13_type), int(w2_type)} != {_IQ2_S}:
        return 0
    staged = envs.SGLANG_HOTCOLD_MMQ_STAGED.get()
    if staged <= 0:
        return 0
    if staged not in _MMQ_STAGED_BLOCKS:
        raise ValueError(
            f"SGLANG_HOTCOLD_MMQ_STAGED must be 0 or one of {_MMQ_STAGED_BLOCKS}, got {staged}"
        )
    return staged


def select_prefill_float(tokens: int, w13_type: int, w2_type: int) -> bool:
    """Whether the staged kernel's products run in half-float arithmetic on the
    unquantised activations (SGLANG_HOTCOLD_PREFILL_FLOAT; IQ2_S on the staged
    kernel only). It changes the output bits of the calls above the MMQ threshold
    (closer to the fp32 product of the dequantised weights: no 8-bit rounding of
    the activations); one answer per (tokens, types)."""
    return (
        bool(envs.SGLANG_HOTCOLD_PREFILL_FLOAT.get())
        and select_mmq_staged(tokens, w13_type, w2_type) > 0
    )


def select_prefill_compact(x: torch.Tensor, top_k: int, collector) -> bool:
    """Whether an MMQ call keeps the rows of the cold routes only (h, a and y of
    [cold routes, .] instead of [tokens * top_k, .], and one weighted sum over
    them). Eager calls only: the row count is read on the host, which a captured
    graph cannot do, and no captured batch reaches the MMQ threshold. The imatrix
    collector reads the dense layout, and moe_sum has kernels of its own for a
    top-k of 2 to 4 that the compact sum does not reproduce. The host read sits
    behind the gate/up kernel (cold_moe), so the device does not idle for it."""
    if not envs.SGLANG_HOTCOLD_PREFILL_COMPACT.get() or collector is not None:
        return False
    if _cold_kernel_mode() == "legacy":
        return False
    if x.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    return _compact_sum_serves(x.dtype, top_k)


def select_mmq_block(tokens: int, qtype: int) -> int:
    """BLOCK of the 16-route MMQ kernel for a batch of ``tokens``: 32 from
    ``SGLANG_HOTCOLD_MMQ_WIDE_MIN_TOKENS`` tokens (0 = never), else the type's
    default; one answer per token count, so both projections of a call agree."""
    wide = envs.SGLANG_HOTCOLD_MMQ_WIDE_MIN_TOKENS.get()
    if wide > 0 and tokens >= wide and _cold_kernel_mode() != "legacy":
        return 32
    return _mmq_block(qtype)


def _clamped_swiglu_reference(h: torch.Tensor, limit: Optional[float]) -> torch.Tensor:
    """The SiluAndMulClamp kernel's arithmetic: clamp in bf16 (gate above, up both
    sides), silu * up in fp32, rounded to the input dtype."""
    half = h.shape[-1] // 2
    gate, up = h[..., :half], h[..., half:]
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    gate = gate.float()
    return (gate * torch.sigmoid(gate) * up.float()).to(h.dtype)


def dequantize_cold_rows(packed: torch.Tensor, ggml_type: int) -> torch.Tensor:
    """float32 [..., K] from packed rows [..., row_bytes]; CPU, gguf-py types only."""
    import gguf
    import numpy as np

    try:
        qtype = gguf.GGMLQuantizationType(int(ggml_type))
        data = packed.detach().to("cpu").contiguous().numpy()
        out = gguf.quants.dequantize(data, qtype)
    except (ValueError, NotImplementedError) as e:
        raise NotImplementedError(
            f"gguf-py cannot dequantise ggml type {ggml_type} ({e}); the CPU "
            "reference serves the types gguf-py knows, IQ2_S (22) among them"
        ) from e
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))


class HotColdImatrixCollector:
    """Per (layer, cold slot): Σx² over the gate/up input channels and Σa² over the
    clamped SwiGLU output channels of the routes the cold tier ran, and the route
    count. Routes with a negative cold id (hot or padded) are never counted.
    Eager only; dumped as one ``.npz`` per rank at exit and on SIGUSR1."""

    _instance: Optional[HotColdImatrixCollector] = None
    _lock = threading.Lock()

    def __init__(self, directory: str):
        self.directory = directory
        self.sum_x2: Dict[int, torch.Tensor] = {}
        self.sum_a2: Dict[int, torch.Tensor] = {}
        self.count: Dict[int, torch.Tensor] = {}

    @classmethod
    def get(cls) -> Optional[HotColdImatrixCollector]:
        directory = envs.SGLANG_DSV41_HOTCOLD_IMATRIX_DIR.get()
        if not directory:
            return None
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(directory)
                atexit.register(cls._instance.dump)
                try:
                    signal.signal(signal.SIGUSR1, lambda *_: cls._instance.dump())
                except ValueError:  # not the main thread
                    logger.warning("hot/cold imatrix: SIGUSR1 dump not installed")
            return cls._instance

    def accumulate(
        self, layer_id: int, x: torch.Tensor, a: torch.Tensor, cold_ids: torch.Tensor
    ) -> None:
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the hot/cold imatrix collector is eager only; run with CUDA graphs disabled"
            )
        if layer_id not in self.count:
            raise RuntimeError(f"hot/cold imatrix: layer {layer_id} has no slot table")
        top_k = cold_ids.shape[1]
        flat = cold_ids.reshape(-1).long()
        valid = flat >= 0
        slots = flat[valid]
        x2 = x.float().pow(2).repeat_interleave(top_k, dim=0)[valid]
        a2 = a.float().pow(2)[valid]
        self.sum_x2[layer_id].index_add_(0, slots, x2.double())
        self.sum_a2[layer_id].index_add_(0, slots, a2.double())
        self.count[layer_id].index_add_(0, slots, torch.ones_like(slots))

    def set_num_slots(
        self, layer_id: int, num_slots: int, x_width: int, a_width: int, device
    ) -> None:
        if layer_id not in self.sum_x2:
            self.sum_x2[layer_id] = torch.zeros(
                num_slots, x_width, dtype=torch.float64, device=device
            )
            self.sum_a2[layer_id] = torch.zeros(
                num_slots, a_width, dtype=torch.float64, device=device
            )
            self.count[layer_id] = torch.zeros(
                num_slots, dtype=torch.int64, device=device
            )

    def dump(self) -> Optional[str]:
        import numpy as np

        if not self.count:
            return None
        rank = (
            os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or str(os.getpid())
        )
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"hotcold-imatrix-rank{rank}.npz")
        arrays = {}
        for layer_id in sorted(self.count):
            arrays[f"l{layer_id}.sum_x2"] = self.sum_x2[layer_id].cpu().numpy()
            arrays[f"l{layer_id}.sum_a2"] = self.sum_a2[layer_id].cpu().numpy()
            arrays[f"l{layer_id}.count"] = self.count[layer_id].cpu().numpy()
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        logger.info("hot/cold imatrix: wrote %s (%d layers)", path, len(self.count))
        return path


# Kernel seams: the CPU tests replace these to drive cold_moe's CUDA branch.
def _vec_rows_per_block():
    """ggml's one row per CTA under "legacy"; else None (the op's own default)."""
    return 1 if envs.SGLANG_HOTCOLD_COLD_KERNEL.get() == "legacy" else None


def _vec_project(x, weight, ids, top_k, qtype, rows, tokens):
    from sglang.kernels.ops.moe.gguf_moe_vec import gguf_moe_a8_vec

    return gguf_moe_a8_vec(
        x, weight, ids, top_k, qtype, rows, tokens, rows_per_block=_vec_rows_per_block()
    )


def _mmq_block(qtype):
    from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq_block_size

    return gguf_moe_mmq_block_size(qtype)


def _mmq_align(ids, block, num_experts):
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(ids, block, num_experts, pad_expert_ids=True)


def _mmq_project(
    x, weight, sorted_ids, expert_ids, post_padded, qtype, rows, top_k, tokens, block
):
    from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq

    return gguf_moe_mmq(
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
    )


def _mmq_wide_project(
    x,
    weight,
    sorted_ids,
    expert_ids,
    post_padded,
    qtype,
    rows,
    top_k,
    tokens,
    wide,
    sorted_rows=None,
    out_rows=None,
):
    from sglang.kernels.ops.moe.gguf_moe_mmq_wide import gguf_moe_mmq_wide

    # No zero fill of the output: see cold_moe (only the rows of the cold routes are read).
    return gguf_moe_mmq_wide(
        x,
        weight,
        sorted_ids,
        expert_ids,
        post_padded,
        qtype,
        rows,
        top_k,
        tokens,
        wide,
        zero_init=False,
        sorted_rows=sorted_rows,
        out_rows=out_rows,
    )


def _mmq_staged_project(
    x,
    weight,
    sorted_ids,
    expert_ids,
    post_padded,
    qtype,
    rows,
    top_k,
    tokens,
    wide,
    sorted_rows=None,
    out_rows=None,
):
    from sglang.kernels.ops.moe.gguf_moe_mmq_staged import gguf_moe_mmq_staged

    return gguf_moe_mmq_staged(
        x,
        weight,
        sorted_ids,
        expert_ids,
        post_padded,
        qtype,
        rows,
        top_k,
        tokens,
        wide,
        zero_init=False,
        sorted_rows=sorted_rows,
        out_rows=out_rows,
    )


def _f16_staged_project(
    x,
    weight,
    sorted_ids,
    expert_ids,
    post_padded,
    qtype,
    rows,
    top_k,
    tokens,
    wide,
    sorted_rows=None,
    out_rows=None,
):
    from sglang.kernels.ops.moe.gguf_moe_f16_staged import gguf_moe_f16_staged

    return gguf_moe_f16_staged(
        x,
        weight,
        sorted_ids,
        expert_ids,
        post_padded,
        qtype,
        rows,
        top_k,
        tokens,
        wide,
        zero_init=False,
        sorted_rows=sorted_rows,
        out_rows=out_rows,
    )


def _compact_sum_serves(dtype, top_k):
    from sglang.kernels.ops.moe.gguf_moe_cold_sum import gguf_moe_cold_sum_serves

    return gguf_moe_cold_sum_serves(dtype, top_k)


def _compact_rows(ids):
    """(rows, count): the row of every cold route in a compact [count, .] layout
    (int32 [tokens, top_k], route order, -1 for the other routes) and the number
    of cold routes as a 0-d int32 tensor on the device (no host read here)."""
    active = ids.reshape(-1) >= 0
    rows = torch.cumsum(active, dim=0, dtype=torch.int32) - 1
    count = rows[-1] + 1
    return rows.masked_fill(~active, -1).view(ids.shape), count


def _compact_sorted_rows(sorted_ids, rows, count):
    """The compact row of every sorted column of the alignment: padding (ids past
    the routes) maps to ``count`` (a 0-d tensor), an inactive route to -1; both
    are skipped by the kernels, as outputs and as the down projection's sorted ids."""
    numel = rows.numel()
    picked = rows.reshape(-1).index_select(0, sorted_ids.clamp(0, numel - 1))
    return torch.where((sorted_ids >= 0) & (sorted_ids < numel), picked, count)


def _compact_weight_and_sum(y, topk_weights, rows):
    from sglang.kernels.ops.moe.gguf_moe_cold_sum import gguf_moe_cold_sum

    return gguf_moe_cold_sum(y, rows, topk_weights)


def _vecm_project(
    x, weight, sorted_ids, expert_ids, post_padded, qtype, rows, top_k, tokens, routes
):
    from sglang.kernels.ops.moe.gguf_moe_vec_multi import gguf_moe_vec_multi

    return gguf_moe_vec_multi(
        x,
        weight,
        sorted_ids,
        expert_ids,
        post_padded,
        qtype,
        rows,
        top_k,
        tokens,
        routes,
        lanes=envs.SGLANG_HOTCOLD_VECM_LANES.get(),
        zero_init=False,
    )


def _clamped_swiglu(h, limit):
    if limit is None:
        from sglang.kernels.ops.activation.activation import silu_and_mul

        return silu_and_mul(h)
    from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp

    a = torch.empty((h.shape[0], h.shape[1] // 2), dtype=h.dtype, device=h.device)
    silu_and_mul_clamp(h, a, float(limit))
    return a


def _weight_and_sum(y, topk_weights, ids, tokens, top_k):
    from sgl_kernel import moe_sum

    from sglang.kernels.ops.moe.gguf_moe_weights import apply_gguf_moe_weights

    apply_gguf_moe_weights(y, topk_weights, ids)
    out = torch.empty((tokens, y.shape[1]), dtype=y.dtype, device=y.device)
    moe_sum(y.view(tokens, top_k, -1), out)
    return out


def _reference_project(x, dequantized, ids, top_k, rows):
    """[tokens * top_k, rows] in x's dtype; zero for negative ids."""
    flat = ids.reshape(-1).long()
    src = x.repeat_interleave(top_k, dim=0) if top_k > 1 else x
    out = torch.zeros(flat.numel(), rows, dtype=torch.float32, device=x.device)
    valid = flat >= 0
    if valid.any():
        w = dequantized[flat[valid]]  # [n, rows, K]
        out[valid] = torch.bmm(w, src[valid].float().unsqueeze(-1)).squeeze(-1)
    return out.to(x.dtype)


def cold_moe(
    x: torch.Tensor,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    topk_weights: torch.Tensor,
    cold_ids: torch.Tensor,
    w13_type: int,
    w2_type: int,
    swiglu_limit: Optional[float],
    *,
    layer_id: int = -1,
    reference_weights: Optional[tuple] = None,
    collector: Optional[HotColdImatrixCollector] = None,
    use_kernels: Optional[bool] = None,
    kernel: Optional[str] = None,
) -> torch.Tensor:
    """Sum over the cold routes of ``topk_weights * expert(x)``; [tokens, hidden]."""
    tokens, hidden = x.shape
    top_k = cold_ids.shape[1]
    inter2 = w13_qweight.shape[1]
    if tokens == 0:
        return x.new_zeros((0, w2_qweight.shape[1]))
    if use_kernels is None:
        use_kernels = x.is_cuda
    if not use_kernels:
        if reference_weights is None:
            raise RuntimeError("the CPU cold path needs dequantised reference weights")
        w13_f, w2_f = reference_weights
        h = _reference_project(x, w13_f, cold_ids, top_k, inter2)
        a = _clamped_swiglu_reference(h, swiglu_limit)
        if collector is not None:
            collector.accumulate(layer_id, x, a, cold_ids)
        y = _reference_project(a, w2_f, cold_ids, 1, w2_qweight.shape[1])
        valid = (cold_ids >= 0).reshape(-1, 1)
        weights = topk_weights.reshape(-1, 1).float().masked_fill(~valid, 0.0)
        y = (y.float() * weights).to(x.dtype).masked_fill(~valid, 0)
        return y.view(tokens, top_k, -1).float().sum(dim=1).to(x.dtype)

    ids = (
        cold_ids
        if cold_ids.dtype == torch.int32
        else cold_ids.clamp_min(-1).to(torch.int32)
    )
    ids = ids.contiguous()
    rows2 = w2_qweight.shape[1]
    if kernel is None:
        kernel = select_cold_kernel(tokens, w13_type, w2_type)
    if kernel in ("mmq", "vecm"):
        # Both take the routes grouped per expert in blocks (moe_align_block_size):
        # a CTA reads or unpacks its weight rows once for the routes of its block.
        # The grouped kernels write the rows of the cold routes only and their
        # outputs are not zero-filled: the rows of the other routes are undefined in
        # h, a and y. Nothing reads them: the down projection gathers the rows of
        # its routes (a padded column reads row 0 and its product is dropped), the
        # collector indexes the cold routes, and _weight_and_sum writes zero for an
        # inactive route without reading its row.
        if kernel == "vecm":
            # the vector kernel's arithmetic, `block` routes of one expert per CTA
            block, project = envs.SGLANG_HOTCOLD_VECM_ROUTES.get(), _vecm_project
        else:
            compact = False
            block, project = (
                select_mmq_staged(tokens, w13_type, w2_type),
                _mmq_staged_project,
            )
            if block and select_prefill_float(tokens, w13_type, w2_type):
                project = (
                    _f16_staged_project  # not the int8 kernels' bits: behind its switch
                )
            if block == 0:
                block, project = (
                    select_mmq_wide(tokens, w13_type, w2_type),
                    _mmq_wide_project,
                )
            if block == 0:
                block, project = select_mmq_block(tokens, w13_type), _mmq_project
                if select_mmq_block(tokens, w2_type) != block:
                    raise RuntimeError(
                        "the cold w13 and w2 types need one MMQ block size"
                    )
            else:
                compact = select_prefill_compact(x, top_k, collector)
            if compact:
                # The same arithmetic per route on [cold routes, .] tensors: the kernels
                # write each route's row at its compact position, the SwiGLU and the down
                # projection's activation quantisation run over those rows only, and one
                # kernel applies the route weights and sums in the dense path's rounding
                # and order (gguf_moe_cold_sum.py). Same output bits.
                rows, count = _compact_rows(ids)
                sorted_ids, expert_ids, post_padded = _mmq_align(
                    ids, block, w13_qweight.shape[0]
                )
                sorted_rows = _compact_sorted_rows(sorted_ids, rows, count)
                # The gate/up kernel is launched into a buffer of the static bound (only its first
                # rows are written) BEFORE the cold route count is read on the host: the read then
                # waits behind that kernel instead of leaving the device idle while the next
                # launches are prepared.
                h = project(
                    x,
                    w13_qweight,
                    sorted_ids,
                    expert_ids,
                    post_padded,
                    w13_type,
                    inter2,
                    top_k,
                    tokens,
                    block,
                    sorted_rows=sorted_rows,
                    out_rows=tokens * top_k,
                )
                count = int(count)
                if count == 0:
                    return x.new_zeros((tokens, rows2))
                h = h[:count]
                a = _clamped_swiglu(h, swiglu_limit)
                # the down projection's input is one row per cold route: its sorted ids are the compact rows
                y = project(
                    a,
                    w2_qweight,
                    sorted_rows,
                    expert_ids,
                    post_padded,
                    w2_type,
                    rows2,
                    1,
                    count,
                    block,
                )
                return _compact_weight_and_sum(y, topk_weights, rows)
        sorted_ids, expert_ids, post_padded = _mmq_align(
            ids, block, w13_qweight.shape[0]
        )
        h = project(
            x,
            w13_qweight,
            sorted_ids,
            expert_ids,
            post_padded,
            w13_type,
            inter2,
            top_k,
            tokens,
            block,
        )
        a = _clamped_swiglu(h, swiglu_limit)
        if collector is not None:
            collector.accumulate(layer_id, x, a, cold_ids)
        y = project(
            a,
            w2_qweight,
            sorted_ids,
            expert_ids,
            post_padded,
            w2_type,
            rows2,
            1,
            tokens * top_k,
            block,
        )
    else:
        # The vector kernel's grid holds tokens * top_k <= 65535 rows: an eager
        # prefill above that runs in chunks (no captured batch comes near it).
        step = VEC_MAX_ROWS // top_k
        parts = []
        for start in range(0, tokens, step):
            end = min(tokens, start + step)
            xs = x if (start, end) == (0, tokens) else x[start:end].contiguous()
            part_ids = (
                ids if (start, end) == (0, tokens) else ids[start:end].contiguous()
            )
            h = _vec_project(
                xs, w13_qweight, part_ids, top_k, w13_type, inter2, end - start
            )
            a = _clamped_swiglu(h, swiglu_limit)
            if collector is not None:
                collector.accumulate(layer_id, xs, a, cold_ids[start:end])
            parts.append(
                _vec_project(
                    a, w2_qweight, part_ids, 1, w2_type, rows2, (end - start) * top_k
                )
            )
        y = parts[0] if len(parts) == 1 else torch.cat(parts)
    return _weight_and_sum(y, topk_weights, ids, tokens, top_k)


class ColdGGUFExperts(nn.Module):
    """The cold experts of one layer, packed ggml rows by slot."""

    def __init__(
        self,
        *,
        num_cold: int,
        hidden_size: int,
        intermediate_size: int,
        w13_type: int,
        w2_type: int,
        swiglu_limit: Optional[float],
        layer_id: int,
    ):
        super().__init__()
        self.num_cold = num_cold
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w13_type = int(w13_type)
        self.w2_type = int(w2_type)
        self.swiglu_limit = swiglu_limit
        self.layer_id = layer_id
        self.w13_qweight = Parameter(
            torch.empty(
                num_cold,
                2 * intermediate_size,
                cold_row_bytes(w13_type, hidden_size),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w2_qweight = Parameter(
            torch.empty(
                num_cold,
                hidden_size,
                cold_row_bytes(w2_type, intermediate_size),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        for p in (self.w13_qweight, self.w2_qweight):
            p.weight_loader = self.weight_loader
        self._reference = None
        self._loaded = {"w13": set(), "w2": False}

    def weight_loader(
        self, param: Parameter, loaded: torch.Tensor, shard_id: Optional[int]
    ) -> None:
        if loaded.dtype != torch.uint8:
            raise TypeError(
                f"cold tensor dtype {loaded.dtype}, expected uint8 (layer {self.layer_id})"
            )
        inter = self.intermediate_size
        if param is self.w13_qweight:
            if shard_id not in (0, 1):
                raise ValueError(
                    f"w13 shard id must be 0 (gate) or 1 (up), got {shard_id}"
                )
            target = param.data[:, shard_id * inter : (shard_id + 1) * inter]
            self._loaded["w13"].add(shard_id)
        elif param is self.w2_qweight:
            target = param.data
            self._loaded["w2"] = True
        else:
            raise ValueError("not a cold parameter of this layer")
        if tuple(loaded.shape) != tuple(target.shape):
            raise ValueError(
                f"cold tensor of layer {self.layer_id} has shape {tuple(loaded.shape)}, "
                f"expected {tuple(target.shape)} (ggml types {self.w13_type}/{self.w2_type})"
            )
        target.copy_(loaded)
        self._reference = None

    def check_kernels(self) -> None:
        """At load on CUDA: build the kernels this layer will run (the JIT compiles
        here instead of at the first request) and refuse a type they lack."""
        if not self.w13_qweight.is_cuda:
            return
        from sglang.kernels.ops.moe.gguf_moe_vec import _jit_gguf_moe_vec_module

        types = sorted({self.w13_type, self.w2_type})
        for t in types:
            try:
                _jit_gguf_moe_vec_module(torch.bfloat16, t, _vec_rows_per_block())
            except ValueError as e:
                raise RuntimeError(
                    f"hot/cold layer {self.layer_id}: the GGUF vector kernel does not "
                    f"serve cold type {t} (IQ4_KS = 144 and IQ2_KS = 145 need the GGUF MoE"
                    f" kernels that add those types): {e}"
                ) from e
        # The MMQ build too, when auto or mmq can take it: a prefill above the
        # token threshold then runs a kernel compiled at load, not at its first call.
        mode = _cold_kernel_mode()
        multi = types == [_IQ2_S] and (
            mode == "vecm" or (mode == "auto" and envs.SGLANG_HOTCOLD_VEC_MULTI.get())
        )
        if multi:
            from sglang.kernels.ops.moe.gguf_moe_vec_multi import (
                GGUF_MOE_VEC_MULTI_DEFAULT_ROWS,
                _jit_gguf_moe_vec_multi_module,
            )

            _jit_gguf_moe_vec_multi_module(
                torch.bfloat16,
                _IQ2_S,
                envs.SGLANG_HOTCOLD_VECM_ROUTES.get(),
                GGUF_MOE_VEC_MULTI_DEFAULT_ROWS,
                True,
            )
        mmq = mode not in ("vec", "vecm") and set(types) <= _mmq_types()
        # one answer per types: the token count does not enter
        staged = select_mmq_staged(0, self.w13_type, self.w2_type) if mmq else 0
        wide = (
            select_mmq_wide(0, self.w13_type, self.w2_type) if mmq and not staged else 0
        )
        compact = bool(staged or wide) and envs.SGLANG_HOTCOLD_PREFILL_COMPACT.get()
        if compact:
            from sglang.kernels.ops.moe.gguf_moe_cold_sum import (
                _jit_gguf_moe_cold_sum_module,
            )

            # the routed top-k is not known here: built for DeepSeek-V4's six; another top-k builds at its first call
            _jit_gguf_moe_cold_sum_module(torch.bfloat16, 6)
        half_float = bool(staged) and select_prefill_float(
            0, self.w13_type, self.w2_type
        )
        if half_float:
            from sglang.kernels.ops.moe.gguf_moe_f16_staged import (
                _jit_gguf_moe_f16_staged_module,
            )

            _jit_gguf_moe_f16_staged_module(torch.bfloat16, _IQ2_S, staged)
        elif staged:
            from sglang.kernels.ops.moe.gguf_moe_mmq_staged import (
                _jit_gguf_moe_mmq_staged_module,
            )

            _jit_gguf_moe_mmq_staged_module(torch.bfloat16, _IQ2_S, staged)
        elif wide:
            from sglang.kernels.ops.moe.gguf_moe_mmq_wide import (
                _jit_gguf_moe_mmq_wide_module,
            )

            _jit_gguf_moe_mmq_wide_module(torch.bfloat16, _IQ2_S, wide, True)
        elif mmq:
            from sglang.kernels.ops.moe.gguf_moe_mmq import _jit_gguf_moe_mmq_module

            blocks = {_mmq_block(t) for t in types}
            if envs.SGLANG_HOTCOLD_MMQ_WIDE_MIN_TOKENS.get() > 0 and mode != "legacy":
                blocks.add(32)
            for t in types:
                for block in sorted(blocks):
                    _jit_gguf_moe_mmq_module(torch.bfloat16, t, block)
        global _KERNELS_LOGGED
        if not _KERNELS_LOGGED:
            _KERNELS_LOGGED = True
            if multi:
                decode = (
                    f"multi-route vector kernel ({envs.SGLANG_HOTCOLD_VECM_ROUTES.get()} routes a block, "
                    f"{envs.SGLANG_HOTCOLD_VECM_LANES.get()} lanes)"
                    + (
                        ""
                        if mode == "vecm"
                        else f" from {envs.SGLANG_HOTCOLD_VEC_MULTI_MIN_TOKENS.get()} tokens"
                    )
                )
            else:
                decode = (
                    "vector kernel at 1 row per CTA"
                    if mode == "legacy"
                    else "vector kernel"
                )
            if staged:
                build = f"staged block {staged}"
            elif wide:
                build = f"wide block {wide}"
            else:
                build = f"block {_mmq_block(types[0])}" if mmq else ""
            if compact:
                build += ", compact rows"
            if half_float:
                # SGLANG_HOTCOLD_PREFILL_FLOAT: other output bits than the int8 kernels above the threshold
                build += ", half-float products"
            # The line keeps its first three fields (the boot proof reads them); the rest names the builds.
            logger.info(
                "hot/cold cold tier: kernel mode %s, vector kernel built for ggml type %s, "
                "MMQ %s; decode: %s",
                mode,
                "/".join(str(t) for t in types),
                (
                    f"built ({build}) above {envs.SGLANG_HOTCOLD_MMQ_MIN_TOKENS.get()} tokens"
                    if mmq
                    else "off"
                ),
                decode,
            )

    def reference_weights(self):
        if self._reference is None:
            self._reference = (
                dequantize_cold_rows(self.w13_qweight.data, self.w13_type).to(
                    self.w13_qweight.device
                ),
                dequantize_cold_rows(self.w2_qweight.data, self.w2_type).to(
                    self.w2_qweight.device
                ),
            )
        return self._reference

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        cold_ids: torch.Tensor,
        kernel: Optional[str] = None,
    ) -> torch.Tensor:
        collector = HotColdImatrixCollector.get()
        if collector is not None:
            collector.set_num_slots(
                self.layer_id,
                self.num_cold,
                self.hidden_size,
                self.intermediate_size,
                x.device,
            )
        return cold_moe(
            x,
            self.w13_qweight,
            self.w2_qweight,
            topk_weights,
            cold_ids,
            self.w13_type,
            self.w2_type,
            self.swiglu_limit,
            layer_id=self.layer_id,
            reference_weights=None if x.is_cuda else self.reference_weights(),
            collector=collector,
            kernel=kernel,
        )
