"""Staged GGUF MoE half-float tensor-core matmul for IQ2_S experts.

The staged MMQ kernel's memory traffic (``gguf_moe_mmq_staged``) with the hot
tier's arithmetic: the weights of a tile are unpacked to fp16 with their 16-value
scales folded in (exact: an integer of at most 1,333 times the block scale, which
is applied once per 128 values in fp32), the activations are taken as fp16
WITHOUT the int8 quantisation (a bf16 value converts exactly; a row that would
overflow fp16 is scaled by a power of two and scaled back), and m16n8k16
half-float products accumulate in fp32 inside the tensor cores.

The result is the fp32 matrix product of the dequantised weights up to the order
and rounding of fp32 additions, rounded to the input dtype. It is closer to that
product than the int8 kernels (which round every activation to 8 bits per 32-value
group) and therefore NOT bit-equal to them: the cold tier takes it only under
``SGLANG_HOTCOLD_PREFILL_FLOAT``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GGUF_MOE_F16_STAGED_TYPES = (22,)
GGUF_MOE_F16_STAGED_BLOCKS = (32, 64)
GGUF_MOE_F16_STAGED_DEFAULT_BLOCK = 64


@cache_once
def _jit_gguf_moe_f16_staged_module(
    dtype: torch.dtype, quant_type: int, wide: int
) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF input dtype: {dtype}")
    if quant_type not in GGUF_MOE_F16_STAGED_TYPES:
        raise ValueError(
            f"Unsupported GGUF staged half-float quantization type: {quant_type}"
        )
    if wide not in GGUF_MOE_F16_STAGED_BLOCKS:
        raise ValueError(f"Unsupported GGUF staged half-float block size: {wide}")
    args = make_cpp_args(dtype, quant_type, wide)
    return load_jit(
        "gguf_moe_f16_staged",
        *args,
        cuda_files=["moe/gguf_moe_f16_staged.cuh"],
        cuda_wrappers=[
            ("project", f"gguf_moe_f16_staged<{args}>"),
            ("project_rows", f"gguf_moe_f16_staged_rows<{args}>"),
        ],
    )


@debug_kernel_api
def gguf_moe_f16_staged(
    input: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    type: int,
    row: int,
    top_k: int,
    tokens: int,
    wide: int = GGUF_MOE_F16_STAGED_DEFAULT_BLOCK,
    zero_init: bool = True,
    sorted_rows: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
) -> torch.Tensor:
    """Project aligned routes through packed IQ2_S experts in half-float arithmetic.

    The arguments and the output layout are those of ``gguf_moe_mmq_staged``
    (``sorted_rows`` / ``out_rows`` for a compact output; negative ids, padding and
    blocks past ``num_tokens_post_padded`` are skipped on the device; only the rows
    of the active routes are written).
    """
    k = input.shape[1]
    alloc = torch.zeros if zero_init else torch.empty
    module = _jit_gguf_moe_f16_staged_module(input.dtype, int(type), int(wide))
    if sorted_rows is None:
        output = alloc((tokens * top_k, row), dtype=input.dtype, device=input.device)
        half_input = torch.empty((tokens, k), dtype=torch.float16, device=input.device)
        input_scale = torch.empty((tokens,), dtype=torch.float32, device=input.device)
        module.project(
            input,
            weight,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            output,
            half_input,
            input_scale,
            top_k,
            row,
            tokens,
        )
        return output
    if out_rows is None:
        raise ValueError("sorted_rows needs out_rows")
    output = alloc((int(out_rows), row), dtype=input.dtype, device=input.device)
    half_input = torch.empty((tokens, k), dtype=torch.float16, device=input.device)
    input_scale = torch.empty((tokens,), dtype=torch.float32, device=input.device)
    module.project_rows(
        input,
        weight,
        sorted_token_ids,
        sorted_rows,
        expert_ids,
        num_tokens_post_padded,
        output,
        half_input,
        input_scale,
        top_k,
        row,
        tokens,
    )
    return output
