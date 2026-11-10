"""Staged GGUF MoE int8 tensor-core matmul (MMQ) for IQ2_S experts.

Same result as ``gguf_moe_mmq`` and ``gguf_moe_mmq_wide`` (the same tile values,
products and float expression per element; the output bits are equal). What
differs is the memory traffic. The wide-block kernel spends about three quarters
of its time waiting for the scattered global loads of its unpack (two CTAs of
48 KiB of shared memory per multiprocessor leave the first-level cache almost no
room). Here one CTA of 16 warps per multiprocessor copies the packed rows of a
tile into shared memory with asynchronous copies (cp.async) one tile ahead,
unpacks them there against a grid table held in shared memory (no global load in
the unpack), and keeps the activations of all the block's columns in shared
memory; a multiprocessor unpacks and multiplies one tile while the next one's
bytes are on their way from memory.
The routing comes from ``moe_align_block_size(ids, wide, E, pad_expert_ids=True)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GGUF_MOE_MMQ_STAGED_TYPES = (22,)
GGUF_MOE_MMQ_STAGED_BLOCKS = (32, 64)
GGUF_MOE_MMQ_STAGED_DEFAULT_BLOCK = 64


@cache_once
def _jit_gguf_moe_mmq_staged_module(
    dtype: torch.dtype, quant_type: int, wide: int
) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF input dtype: {dtype}")
    if quant_type not in GGUF_MOE_MMQ_STAGED_TYPES:
        raise ValueError(f"Unsupported GGUF staged MMQ quantization type: {quant_type}")
    if wide not in GGUF_MOE_MMQ_STAGED_BLOCKS:
        raise ValueError(f"Unsupported GGUF staged MMQ block size: {wide}")
    args = make_cpp_args(dtype, quant_type, wide)
    return load_jit(
        "gguf_moe_mmq_staged",
        *args,
        cuda_files=["moe/gguf_moe_mmq_staged.cuh"],
        cuda_wrappers=[
            ("project", f"gguf_moe_mmq_staged<{args}>"),
            ("project_rows", f"gguf_moe_mmq_staged_rows<{args}>"),
            ("project_counted", f"gguf_moe_mmq_staged_counted<{args}>"),
        ],
    )


@debug_kernel_api
def gguf_moe_mmq_staged(
    input: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    type: int,
    row: int,
    top_k: int,
    tokens: int,
    wide: int = GGUF_MOE_MMQ_STAGED_DEFAULT_BLOCK,
    zero_init: bool = True,
    sorted_rows: Optional[torch.Tensor] = None,
    out_rows: Optional[int] = None,
    row_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project aligned routes through packed IQ2_S experts; inactive routes stay zero.

    The arguments are those of ``gguf_moe_mmq_wide``: ``input`` [tokens, K] (gate/up)
    or [tokens * top_k, K] with top_k=1 (down), ``weight`` uint8 [E, row, row_bytes]
    (16-byte aligned, every expert a multiple of 16 bytes), the routing tensors of ``moe_align_block_size(ids, wide, E,
    pad_expert_ids=True)``. Negative ids, padding and blocks past
    ``num_tokens_post_padded`` are skipped on the device (CUDA-graph safe). Returns
    [tokens * top_k, row] in the input dtype; only the rows of the active routes are
    written (``zero_init=False`` leaves the others undefined).

    ``sorted_rows`` (int32, aligned with ``sorted_token_ids``) with ``out_rows``
    writes sorted column ``c`` to row ``sorted_rows[c]`` of an ``[out_rows, row]``
    output: a caller that keeps the rows of its active routes only.

    ``row_count`` (int32 [1] on the device; not with ``sorted_rows``): only the
    first ``row_count[0]`` rows of ``input`` hold activations, in a buffer of
    ``tokens`` rows. The activation quantisation covers those rows and reads the
    count on the device; no id may point past them.
    """
    k = input.shape[1]
    alloc = torch.zeros if zero_init else torch.empty
    module = _jit_gguf_moe_mmq_staged_module(input.dtype, int(type), int(wide))
    if row_count is not None:
        if sorted_rows is not None:
            raise ValueError(
                "row_count is for a compact input: its sorted ids are the rows already"
            )
        output = alloc((tokens * top_k, row), dtype=input.dtype, device=input.device)
        quantized_input = torch.empty(
            (k // 128, tokens, 36), dtype=torch.int32, device=input.device
        )
        module.project_counted(
            input,
            weight,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            output,
            quantized_input,
            row_count,
            top_k,
            row,
            tokens,
        )
        return output
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
