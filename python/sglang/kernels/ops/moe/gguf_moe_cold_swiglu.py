"""The clamped SwiGLU over the first rows of a buffer, the row count read on the device.

``silu_and_mul_clamp``'s result (the same expression built with the same flags:
bit-equal per row) for the rows below ``row_count[0]`` of a ``[rows, 2 * dim]``
input; the other rows of the output are left as they are. A compact cold-tier
call sizes its buffers for every route and keeps the number of cold routes on
the device: the host launches this kernel without waiting for that number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
