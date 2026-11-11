"""The route split of a hot/cold MoE layer in one launch.

``hot_ids = remap_hot[ids + 1]`` and ``cold_ids = remap_cold[ids + 1]`` (the
tables of ``HotColdManifest.remap_tables``: entry 0 is the padded route, -1).
The same integers as the add and the two ``index_select`` calls it replaces,
in one kernel instead of three; an id outside the tables takes the padded
route's entry.

``gguf_moe_split_align`` is the decode batches' variant: the same split and, in
the same launch, the cold routes grouped per cold expert in blocks, i.e. what
``moe_align_block_size(cold_ids, block, num_cold, pad_expert_ids=True)`` hands
the multi-route vector kernel in two more launches, without the blocks of the
routes that are not cold (which that kernel skips).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gguf_moe_split_module() -> Module:
    return load_jit(
        "gguf_moe_split",
        cuda_files=["moe/gguf_moe_split.cuh"],
        cuda_wrappers=[("run", "gguf_moe_split"), ("align", "gguf_moe_split_align")],
    )


# One scan thread per cold expert in gguf_moe_split_align's single CTA.
GGUF_MOE_SPLIT_ALIGN_MAX_EXPERTS = 1024


def gguf_moe_split_align_serves(num_cold: int, block: int) -> bool:
    """Whether the split can group the cold routes too: the layer's cold experts fit its scan."""
    return 1 <= num_cold <= GGUF_MOE_SPLIT_ALIGN_MAX_EXPERTS and 1 <= block <= 1024


def gguf_moe_split_serves(ids: torch.Tensor, remap_hot: torch.Tensor) -> bool:
    """Whether the kernel takes these ids: contiguous int32 on CUDA, tables beside them."""
    return (
        ids.is_cuda
        and ids.dtype == torch.int32
        and ids.is_contiguous()
        and remap_hot.dtype == torch.int32
        and remap_hot.device == ids.device
    )


@debug_kernel_api
def gguf_moe_split(
    ids: torch.Tensor, remap_hot: torch.Tensor, remap_cold: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(hot_ids, cold_ids), each int32 in the shape of ``ids`` (contiguous int32, CUDA)."""
    if not gguf_moe_split_serves(ids, remap_hot):
        raise ValueError("gguf_moe_split needs contiguous int32 ids on CUDA")
    hot_ids = torch.empty_like(ids)
    cold_ids = torch.empty_like(ids)
    _jit_gguf_moe_split_module().run(
        ids.view(-1), remap_hot, remap_cold, hot_ids.view(-1), cold_ids.view(-1)
    )
    return hot_ids, cold_ids


@debug_kernel_api
def gguf_moe_split_align(
    ids: torch.Tensor,
    remap_hot: torch.Tensor,
    remap_cold: torch.Tensor,
    num_cold: int,
    block: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(hot_ids, cold_ids, sorted_ids, expert_ids, num_post_padded) in one launch.

    The first two as ``gguf_moe_split``. The last three group the cold routes
    (``cold_ids >= 0``) per cold expert in blocks of ``block`` routes, in the
    shapes ``moe_align_block_size(cold_ids, block, num_cold, pad_expert_ids=True)``
    returns: ``sorted_ids`` holds each expert's routes followed by padding (an id
    past the routes), ``expert_ids`` the expert of each block and -1 past the
    blocks in use, ``num_post_padded`` int32 [1] the columns in use. The routes
    that are not cold get no block.
    """
    if not gguf_moe_split_serves(ids, remap_hot):
        raise ValueError("gguf_moe_split_align needs contiguous int32 ids on CUDA")
    if not gguf_moe_split_align_serves(num_cold, block):
        raise ValueError(
            f"gguf_moe_split_align does not serve {num_cold} cold experts in blocks of {block}"
        )
    numel = ids.numel()
    # moe_align_block_size's capacities, so the kernels that follow see the same shapes
    if numel < num_cold + 1:
        capacity = numel * block
    else:
        capacity = numel + (num_cold + 1) * (block - 1)
    capacity = max((capacity + 3) // 4 * 4, 4)
    hot_ids = torch.empty_like(ids)
    cold_ids = torch.empty_like(ids)
    sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=ids.device)
    expert_ids = torch.empty(
        (-(-capacity // block),), dtype=torch.int32, device=ids.device
    )
    num_post_padded = torch.empty((1,), dtype=torch.int32, device=ids.device)
    _jit_gguf_moe_split_module().align(
        ids.view(-1),
        remap_hot,
        remap_cold,
        hot_ids.view(-1),
        cold_ids.view(-1),
        sorted_ids,
        expert_ids,
        num_post_padded,
        int(num_cold),
        int(block),
    )
    return hot_ids, cold_ids, sorted_ids, expert_ids, num_post_padded
