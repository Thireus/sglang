"""Wide-block GGUF MoE int8 tensor-core matmul (MMQ) for IQ2_S experts.

Same result as ``gguf_moe_mmq`` (the same tile loader, products and float
expression per element; the output bits are equal), but a CTA owns up to ``wide``
sorted routes of one expert instead of 16 and multiplies only the groups of 8
columns that hold a route: the weight tile of an expert is unpacked once per
``wide`` routes, and a partly filled block does not pay for its empty columns.
The routing comes from ``moe_align_block_size(ids, wide, E, pad_expert_ids=True)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GGUF_MOE_MMQ_WIDE_TYPES = (22,)
GGUF_MOE_MMQ_WIDE_BLOCKS = (16, 32, 64, 128)
GGUF_MOE_MMQ_WIDE_DEFAULT_BLOCK = 64


@cache_once
def _jit_gguf_moe_mmq_wide_module(
    dtype: torch.dtype, quant_type: int, wide: int, skip: bool = True
) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF input dtype: {dtype}")
    if quant_type not in GGUF_MOE_MMQ_WIDE_TYPES:
        raise ValueError(f"Unsupported GGUF wide MMQ quantization type: {quant_type}")
    if wide not in GGUF_MOE_MMQ_WIDE_BLOCKS:
        raise ValueError(f"Unsupported GGUF wide MMQ block size: {wide}")
    args = make_cpp_args(dtype, quant_type, wide, bool(skip))
    return load_jit(
        "gguf_moe_mmq_wide",
        *args,
        cuda_files=["moe/gguf_moe_mmq_wide.cuh"],
        cuda_wrappers=[
            ("project", f"gguf_moe_mmq_wide<{args}>"),
            ("project_rows", f"gguf_moe_mmq_wide_rows<{args}>"),
        ],
    )


@debug_kernel_api
def gguf_moe_mmq_wide(
    input: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    type: int,
    row: int,
    top_k: int,
    tokens: int,
    wide: int = GGUF_MOE_MMQ_WIDE_DEFAULT_BLOCK,
    skip: bool = True,
    zero_init: bool = True,
    sorted_rows: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
) -> torch.Tensor:
    """Project aligned routes through packed IQ2_S experts; inactive routes stay zero.

    ``input`` is [tokens, K] (gate/up) or [tokens * top_k, K] with top_k=1 (down);
    ``weight`` uint8 [E, row, row_bytes]. The routing tensors come from
    ``moe_align_block_size(ids, wide, E, pad_expert_ids=True)``; negative ids, padding
    and blocks past ``num_tokens_post_padded`` are skipped on the device, so the call
    is CUDA-graph safe. ``skip=False`` multiplies every column group of every block (the
    16-route kernel's work at ``wide=16``; for measurements). Returns [tokens * top_k, row]
    in the input dtype. The kernel
    writes the rows of the active routes only: ``zero_init=False`` skips the zero fill
    for a caller that never reads another row (those rows are then undefined).

    ``sorted_rows`` (int32, aligned with ``sorted_token_ids``) with ``out_rows``
    writes sorted column ``c`` to row ``sorted_rows[c]`` of an ``[out_rows, row]``
    output instead of row ``sorted_token_ids[c]``: a caller that keeps the rows of
    its active routes only. The arithmetic of a (route, row) is the same.
    """
    k = input.shape[1]
    alloc = torch.zeros if zero_init else torch.empty
    module = _jit_gguf_moe_mmq_wide_module(
        input.dtype, int(type), int(wide), bool(skip)
    )
    if sorted_rows is None:
        output = alloc((tokens * top_k, row), dtype=input.dtype, device=input.device)
        quantized_input = torch.empty(
            (k // 128, tokens, 36), dtype=torch.int32, device=input.device
        )
        module.project(
            input,
            weight,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            output,
            quantized_input,
            top_k,
            row,
            tokens,
        )
        return output
    if out_rows is None:
        raise ValueError("sorted_rows needs out_rows")
    output = alloc((int(out_rows), row), dtype=input.dtype, device=input.device)
    quantized_input = torch.empty(
        (k // 128, tokens, 36), dtype=torch.int32, device=input.device
    )
    module.project_rows(
        input,
        weight,
        sorted_token_ids,
        sorted_rows,
        expert_ids,
        num_tokens_post_padded,
        output,
        quantized_input,
        top_k,
        row,
        tokens,
    )
    return output
