"""Route weights and the sum over top-k for a compact expert output.

``y`` holds one row per ACTIVE route only; ``rows[token, slot]`` is that row or a
negative value. The result equals ``apply_gguf_moe_weights`` followed by
``moe_sum`` on the dense ``[tokens * top_k, hidden]`` layout bit for bit, for a
top-k outside 2..4 (where ``moe_sum`` is ``torch.sum`` over the top-k axis): the
weighted value is rounded to the dtype, then summed in fp32 through torch's four
interleaved accumulators. A dense layout reads and writes every route row twice
for that; this reads the active rows once.

For a decode batch the same kernel takes the dense layout (``dense=True``: one
row per route, only the sign of ``rows`` is used, e.g. the cold ids) and adds
its sum into a tensor that already holds the other tier's output (``out=``: the
value ``out.add_(sum)`` gives, the sum rounded to the dtype first), so the route
weights, the sum over top-k and the tier add are one launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GGUF_MOE_COLD_SUM_MAX_TOPK = 16


def gguf_moe_cold_sum_serves(dtype: torch.dtype, top_k: int) -> bool:
    """Whether the compact sum reproduces the dense path for this dtype and top-k
    (moe_sum has kernels of its own for top-k 2, 3 and 4: sums in the dtype)."""
    return (
        dtype in (torch.float16, torch.bfloat16)
        and 4 < top_k <= GGUF_MOE_COLD_SUM_MAX_TOPK
    )


@cache_once
def _jit_gguf_moe_cold_sum_module(
    dtype: torch.dtype, top_k: int, dense: bool = False, fold: bool = False
) -> Module:
    if not gguf_moe_cold_sum_serves(dtype, top_k):
        raise ValueError(
            f"gguf_moe_cold_sum does not serve dtype {dtype} with top-k {top_k}"
        )
    args = make_cpp_args(dtype, int(top_k), bool(dense), bool(fold))
    return load_jit(
        "gguf_moe_cold_sum",
        *args,
        cuda_files=["moe/gguf_moe_cold_sum.cuh"],
        cuda_wrappers=[("run", f"gguf_moe_cold_sum<{args}>")],
    )


@debug_kernel_api
def gguf_moe_cold_sum(
    y: torch.Tensor,
    rows: torch.Tensor,
    weights: torch.Tensor,
    *,
    dense: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[tokens, hidden]: per token the weighted sum of its active routes' rows of ``y``.

    ``y`` [active, hidden] (bf16 / fp16), ``rows`` int32 [tokens, top_k] (negative =
    inactive), ``weights`` [tokens, top_k] (taken as fp32). Every output row is
    written; a token with no active route gets zeros. ``dense``: ``y`` is
    [tokens * top_k, hidden], route (token, slot) in row ``token * top_k + slot``,
    and ``rows`` gives the activity only. ``out`` [tokens, hidden], contiguous, in
    ``y``'s dtype: the sum is added to it in place (``out.add_(sum)``'s bits) and
    it is returned.
    """
    if not gguf_moe_cold_sum_serves(y.dtype, rows.shape[1]):
        raise ValueError(
            f"gguf_moe_cold_sum does not serve dtype {y.dtype} with top-k {rows.shape[1]}"
        )
    if weights.dtype != torch.float32 or not weights.is_contiguous():
        weights = weights.float().contiguous()
    if not rows.is_contiguous():
        rows = rows.contiguous()
    fold = out is not None
    if fold:
        if (
            out.dtype != y.dtype
            or tuple(out.shape) != (rows.shape[0], y.shape[1])
            or not out.is_contiguous()
        ):
            raise ValueError(
                "gguf_moe_cold_sum: out must be contiguous [tokens, hidden] in y's dtype"
            )
    else:
        out = torch.empty((rows.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
    _jit_gguf_moe_cold_sum_module(y.dtype, rows.shape[1], bool(dense), fold).run(
        y, rows, weights, out
    )
    return out
