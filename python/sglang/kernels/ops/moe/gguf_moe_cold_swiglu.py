"""The clamped SwiGLU over the first rows of a buffer, the row count read on the device.

``silu_and_mul_clamp``'s result (the same expression built with the same flags:
bit-equal per row) for the rows below ``row_count[0]`` of a ``[rows, 2 * dim]``
input; the other rows of the output are left as they are. A compact cold-tier
call sizes its buffers for every route and keeps the number of cold routes on
the device: the host launches this kernel without waiting for that number.

``gguf_moe_cold_swiglu_q8`` is the decode batches' variant: the same SwiGLU for
the rows of the cold routes only (by their ids), with the down projection's Q8_1
activation quantisation as a second output of the same launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gguf_moe_cold_swiglu_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"gguf_moe_cold_swiglu does not serve dtype {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "gguf_moe_cold_swiglu",
        *args,
        cuda_files=["moe/gguf_moe_cold_swiglu.cuh"],
        cuda_wrappers=[("run", f"ColdSwigluKernel<{args}>::run")],
        # silu_and_mul_clamp's flags: the two kernels must round alike
        extra_cuda_cflags=["-use_fast_math"],
    )


@debug_kernel_api
def gguf_moe_cold_swiglu(
    input: torch.Tensor, row_count: torch.Tensor, swiglu_limit: float
) -> torch.Tensor:
    """[rows, dim] in the input dtype: the clamped SwiGLU of the first ``row_count[0]``
    rows of ``input`` [rows, 2 * dim] (gate then up); the rows past them are undefined."""
    output = torch.empty(
        (input.shape[0], input.shape[1] // 2), dtype=input.dtype, device=input.device
    )
    _jit_gguf_moe_cold_swiglu_module(input.dtype).run(
        input, output, row_count, float(swiglu_limit)
    )
    return output


@cache_once
def _jit_gguf_moe_cold_swiglu_q8_module(dtype: torch.dtype) -> Module:
    if dtype != torch.bfloat16:
        raise ValueError(f"gguf_moe_cold_swiglu_q8 does not serve dtype {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "gguf_moe_cold_swiglu_q8",
        *args,
        cuda_files=["moe/gguf_moe_cold_swiglu_q8.cuh"],
        cuda_wrappers=[("run", f"ColdSwigluQ8Kernel<{args}>::run")],
        # silu_and_mul_clamp's flags: the two kernels must round alike (the
        # quantisation half asks for IEEE arithmetic by intrinsic)
        extra_cuda_cflags=["-use_fast_math"],
    )


def gguf_moe_cold_swiglu_q8_serves(dtype: torch.dtype, dim: int) -> bool:
    """Whether the one-launch SwiGLU with its Q8_1 quantisation takes this output:
    bf16 (the dtype whose clamp the SwiGLU kernel defines) in whole Q8_1 blocks."""
    return dtype == torch.bfloat16 and dim > 0 and dim % 32 == 0


@debug_kernel_api
def gguf_moe_cold_swiglu_q8(
    input: torch.Tensor, ids: torch.Tensor, swiglu_limit: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(output, quantized) for the rows of ``input`` [rows, 2 * dim] whose id is >= 0.

    ``output`` [rows, dim]: ``silu_and_mul_clamp``'s value for those rows (bit-equal).
    ``quantized`` int32 [rows, padded dim / 32 * 9]: the Q8_1 blocks the GGUF vector
    kernels make of ``output`` (``gguf_moe_vec_multi`` takes them as
    ``quantized_input`` and then skips its own quantisation launch). ``ids`` int32
    [rows] (a view of the cold ids [tokens, top_k]); the rows of a negative id are
    undefined in both outputs.
    """
    rows, dim = input.shape[0], input.shape[1] // 2
    if not gguf_moe_cold_swiglu_q8_serves(input.dtype, dim):
        raise ValueError(
            f"gguf_moe_cold_swiglu_q8 does not serve dtype {input.dtype} with {dim} columns"
        )
    if ids.dtype != torch.int32 or not ids.is_contiguous() or ids.numel() != rows:
        raise ValueError(
            "gguf_moe_cold_swiglu_q8 needs one contiguous int32 id per row"
        )
    output = torch.empty((rows, dim), dtype=input.dtype, device=input.device)
    padded = (dim + 511) // 512 * 512
    quantized = torch.empty(
        (rows, padded // 32 * 9), dtype=torch.int32, device=input.device
    )
    _jit_gguf_moe_cold_swiglu_q8_module(input.dtype).run(
        input, output, quantized, ids.view(-1), float(swiglu_limit)
    )
    return output, quantized
