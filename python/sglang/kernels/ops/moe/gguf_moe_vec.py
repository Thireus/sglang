"""GGUF vector expert projections with consumer-owned inactive routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def gguf_moe_vec_rows_per_block() -> int:
    """Rows (warps) per CTA of the vector kernel: SGLANG_GGUF_MOE_VEC_ROWS_PER_BLOCK, 1..32."""
    rows = envs.SGLANG_GGUF_MOE_VEC_ROWS_PER_BLOCK.get()
    if not 1 <= rows <= 32:
        raise ValueError(
            f"SGLANG_GGUF_MOE_VEC_ROWS_PER_BLOCK must be 1..32 (1024 threads a CTA), got {rows}"
        )
    return rows


@cache_once
def _jit_gguf_moe_vec_module(
    dtype: torch.dtype, quant_type: int, rows_per_block: int | None = None
) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF input dtype: {dtype}")
    if quant_type not in (
        2,
        3,
        6,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        29,
        144,  # IQ4_KS (ik_llama.cpp), 4-byte row header
        145,  # IQ2_KS (ik_llama.cpp), 2-byte row header
    ):
        raise ValueError(f"Unsupported GGUF vector quantization type: {quant_type}")
    args = make_cpp_args(dtype, quant_type)
    # Match the AOT common_ops SM90 fast-math build; its other build is precise.
    sm90 = current_arch().is_sm(9, 0)
    flags = ["--use_fast_math"] if sm90 else []
    if rows_per_block is None:
        rows_per_block = gguf_moe_vec_rows_per_block()
    # The grid is (rows / rows_per_block, 1, routes) CTAs of rows_per_block warps: the
    # same dot product per row, fewer CTAs for the routes that only write zeros.
    flags.append(f"-DGGML_CUDA_MMV_Y={int(rows_per_block)}")
    return load_jit(
        "gguf_moe_vec",
        *args,
        cuda_files=["moe/gguf_moe_vec.cuh"],
        cuda_wrappers=[("project", f"gguf_moe_a8_vec<{args}>")],
        extra_cuda_cflags=flags,
    )


@debug_kernel_api
def gguf_moe_a8_vec(
    input: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """Project packed GGUF experts, writing zero for every negative route.

    Inputs and uint8 packed weights [E, row, row_bytes] are contiguous; row_bytes
    is ggml_row_size (IQ4_KS and IQ2_KS rows carry a 4- / 2-byte header). IDs are contiguous int32
    local expert indices or negative sentinels; their total size is tokens * top_k.
    The down projection retains the original route shape with top_k=1.
    Quantization, vector reduction and output rounding match the AOT operation.
    """
    output = torch.zeros((tokens * top_k, row), dtype=input.dtype, device=input.device)
    padded = (input.shape[1] + 511) // 512 * 512
    quantized_input = torch.empty(
        (tokens, padded // 32 * 9), dtype=torch.int32, device=input.device
    )
    _jit_gguf_moe_vec_module(input.dtype, int(type)).project(
        input, weight, topk_ids, output, quantized_input, top_k, row, tokens
    )
    return output
