"""GGUF MoE vector projection that serves several routes of one expert per CTA (IQ2_S).

Same result as ``gguf_moe_a8_vec`` (the same Q8_1 activations, integer sums and float
expression per row), but the routes come grouped per expert from
``moe_align_block_size(ids, routes, E, pad_expert_ids=True)`` and a CTA unpacks its
weight rows once for all of its routes: the weights of an expert are read once per
block of ``routes`` routes instead of once per route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GGUF_MOE_VEC_MULTI_TYPES = (22,)
GGUF_MOE_VEC_MULTI_ROUTES = (2, 4, 8)
GGUF_MOE_VEC_MULTI_DEFAULT_ROUTES = 4
GGUF_MOE_VEC_MULTI_DEFAULT_ROWS = 8


@cache_once
def _jit_gguf_moe_vec_multi_module(
    dtype: torch.dtype, quant_type: int, routes: int, rows_per_block: int
) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF input dtype: {dtype}")
    if quant_type not in GGUF_MOE_VEC_MULTI_TYPES:
        raise ValueError(
            f"Unsupported GGUF multi-route vector quantization type: {quant_type}"
        )
    if routes not in GGUF_MOE_VEC_MULTI_ROUTES:
        raise ValueError(f"Unsupported GGUF multi-route block size: {routes}")
    if not 1 <= rows_per_block <= 32:
        raise ValueError(f"rows per CTA must be 1..32, got {rows_per_block}")
    args = make_cpp_args(dtype, quant_type, routes, rows_per_block)
    return load_jit(
        "gguf_moe_vec_multi",
        *args,
        cuda_files=["moe/gguf_moe_vec_multi.cuh"],
        cuda_wrappers=[("project", f"gguf_moe_vec_multi<{args}>")],
    )


@debug_kernel_api
def gguf_moe_vec_multi(
    input: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    type: int,
    row: int,
    top_k: int,
    tokens: int,
    routes: int = GGUF_MOE_VEC_MULTI_DEFAULT_ROUTES,
    rows_per_block: int = GGUF_MOE_VEC_MULTI_DEFAULT_ROWS,
) -> torch.Tensor:
    """Project aligned routes through packed IQ2_S experts; inactive routes stay zero.

    ``input`` is [tokens, K] (gate/up) or [tokens * top_k, K] with top_k=1 (down);
    ``weight`` uint8 [E, row, row_bytes]. The routing tensors come from
    ``moe_align_block_size(ids, routes, E, pad_expert_ids=True)``; negative ids, padding
    and blocks past ``num_tokens_post_padded`` are skipped on the device, so the call is
    CUDA-graph safe. Returns [tokens * top_k, row] in the input dtype.
    """
    output = torch.zeros((tokens * top_k, row), dtype=input.dtype, device=input.device)
    padded = (input.shape[1] + 511) // 512 * 512
    quantized_input = torch.empty(
        (tokens, padded // 32 * 9), dtype=torch.int32, device=input.device
    )
    _jit_gguf_moe_vec_multi_module(
        input.dtype, int(type), int(routes), int(rows_per_block)
    ).project(
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
