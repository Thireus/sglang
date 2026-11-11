"""Route weights and the sum over top-k for a compact expert output.

``y`` holds one row per ACTIVE route only; ``rows[token, slot]`` is that row or a
negative value. The result equals ``apply_gguf_moe_weights`` followed by
``moe_sum`` on the dense ``[tokens * top_k, hidden]`` layout bit for bit, for a
top-k outside 2..4 (where ``moe_sum`` is ``torch.sum`` over the top-k axis): the
weighted value is rounded to the dtype, then summed in fp32 through torch's four
interleaved accumulators. A dense layout reads and writes every route row twice
for that; this reads the active rows once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
def _jit_gguf_moe_cold_sum_module(dtype: torch.dtype, top_k: int) -> Module:
    if not gguf_moe_cold_sum_serves(dtype, top_k):
        raise ValueError(
            f"gguf_moe_cold_sum does not serve dtype {dtype} with top-k {top_k}"
        )
    args = make_cpp_args(dtype, int(top_k))
    return load_jit(
        "gguf_moe_cold_sum",
        *args,
        cuda_files=["moe/gguf_moe_cold_sum.cuh"],
        cuda_wrappers=[("run", f"gguf_moe_cold_sum<{args}>")],
    )


@debug_kernel_api
def gguf_moe_cold_sum(
    y: torch.Tensor, rows: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """[tokens, hidden]: per token the weighted sum of its active routes' rows of ``y``.

    ``y`` [active, hidden] (bf16 / fp16), ``rows`` int32 [tokens, top_k] (negative =
    inactive), ``weights`` [tokens, top_k] (taken as fp32). Every output row is
    written; a token with no active route gets zeros.
    """
    if not gguf_moe_cold_sum_serves(y.dtype, rows.shape[1]):
        raise ValueError(
            f"gguf_moe_cold_sum does not serve dtype {y.dtype} with top-k {rows.shape[1]}"
        )
    if weights.dtype != torch.float32 or not weights.is_contiguous():
        weights = weights.float().contiguous()
    if not rows.is_contiguous():
        rows = rows.contiguous()
    out = torch.empty((rows.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
    _jit_gguf_moe_cold_sum_module(y.dtype, rows.shape[1]).run(y, rows, weights, out)
    return out
