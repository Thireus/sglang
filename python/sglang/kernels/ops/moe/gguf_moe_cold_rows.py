"""The row map of a compact cold-tier call, in one launch and without a host read.

A compact call keeps one row per ACTIVE route (an id >= 0) in buffers sized for
every route. This op numbers the active routes in route order, maps the sorted
columns of ``moe_align_block_size`` to those rows, and leaves the number of
active routes in a device tensor: the kernels that follow read it there, so the
host never waits for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gguf_moe_cold_rows_module() -> Module:
    return load_jit(
        "gguf_moe_cold_rows",
        cuda_files=["moe/gguf_moe_cold_rows.cuh"],
        cuda_wrappers=[("run", "gguf_moe_cold_rows")],
    )


@debug_kernel_api
def gguf_moe_cold_rows(
    ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rows, sorted_rows, count) for ``ids`` int32 [tokens, top_k] (negative = inactive).

    ``rows`` int32 [tokens, top_k]: 0, 1, 2, ... over the active routes in route
    order, -1 for an inactive route. ``sorted_rows`` int32 like
    ``sorted_token_ids``: the row of each sorted column below
    ``num_tokens_post_padded`` (``tokens * top_k`` for a padding column; the
    columns past it are not written). ``count`` int32 [1]: the active routes.
    """
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        raise ValueError("gguf_moe_cold_rows needs contiguous int32 ids")
    rows = torch.empty_like(ids)
    sorted_rows = torch.empty_like(sorted_token_ids)
    count = torch.empty((1,), dtype=torch.int32, device=ids.device)
    _jit_gguf_moe_cold_rows_module().run(
        ids, sorted_token_ids, num_tokens_post_padded, rows, sorted_rows, count
    )
    return rows, sorted_rows, count
