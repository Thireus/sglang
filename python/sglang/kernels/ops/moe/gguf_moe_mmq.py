"""GGUF MoE int8 tensor-core matmul (MMQ) for IQ2_S, IQ4_KS and IQ2_KS experts.

The device code is ported from ik_llama.cpp's MoE MMQ (``mmq_id``) path, see
``sgl_kernel/quantization/gguf_mma/mmq_id.cuh``. The op is a drop-in for the
``ggml_moe_a8`` call shape: it takes ``moe_align_block_size`` routing built with
``gguf_moe_mmq_block_size(type)`` and quantises the activations itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# ggml type id -> row header bytes, block bytes per 256 values (ggml_row_size).
GGUF_MOE_MMQ_TYPES = {22: (0, 82), 144: (4, 136), 145: (2, 70)}
GGUF_MOE_MMQ_BLOCKS = (16, 32)
# BLOCK 16 beat 32 on SM120 at every measured shape (RTX 5090, 3 types x 2 projections x 7 loads).
GGUF_MOE_MMQ_DEFAULT_BLOCK = 16


def gguf_moe_mmq_row_bytes(quant_type: int, k: int) -> int:
    """Bytes of one packed row of ``k`` values, row header included."""
    meta, block = GGUF_MOE_MMQ_TYPES[int(quant_type)]
    if k % 256:
        raise ValueError(f"GGUF MMQ needs K % 256 == 0, got {k}")
    return meta + k // 256 * block


def gguf_moe_mmq_block_size(quant_type: int) -> int:
    """Rows per expert block for ``moe_align_block_size`` (the kernel's mmq_x)."""
    if int(quant_type) not in GGUF_MOE_MMQ_TYPES:
        raise ValueError(f"Unsupported GGUF MMQ quantization type: {quant_type}")
    return GGUF_MOE_MMQ_DEFAULT_BLOCK


@cache_once
def _jit_gguf_moe_mmq_module(dtype: torch.dtype, quant_type: int, block: int) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF input dtype: {dtype}")
    if quant_type not in GGUF_MOE_MMQ_TYPES:
        raise ValueError(f"Unsupported GGUF MMQ quantization type: {quant_type}")
    if block not in GGUF_MOE_MMQ_BLOCKS:
        raise ValueError(f"Unsupported GGUF MMQ block size: {block}")
    args = make_cpp_args(dtype, quant_type, block)
    return load_jit(
        "gguf_moe_mmq",
        *args,
        cuda_files=["moe/gguf_moe_mmq.cuh"],
        cuda_wrappers=[("project", f"gguf_moe_mmq<{args}>")],
    )


@debug_kernel_api
def gguf_moe_mmq(
    input: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    type: int,
    row: int,
    top_k: int,
    tokens: int,
    block: int = GGUF_MOE_MMQ_DEFAULT_BLOCK,
) -> torch.Tensor:
    """Project aligned routes through packed GGUF experts; inactive routes stay zero.

    ``input`` is [tokens, K] (gate/up) or [tokens * top_k, K] with top_k=1 (down).
    ``weight`` is uint8 [E, row, row_bytes]. The routing tensors come from
    ``moe_align_block_size(ids, block, E, pad_expert_ids=True)``; negative ids,
    padding and blocks past ``num_tokens_post_padded`` are skipped on the device,
    so the call is CUDA-graph safe. Returns [tokens * top_k, row] in input dtype.
    """
    k = input.shape[1]
    output = torch.zeros((tokens * top_k, row), dtype=input.dtype, device=input.device)
    quantized_input = torch.empty(
        (k // 128, tokens, 36), dtype=torch.int32, device=input.device
    )
    _jit_gguf_moe_mmq_module(input.dtype, int(type), int(block)).project(
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
