"""Dequantise packed ggml / ik_llama.cpp rows on the GPU, and multiply with the result.

For the quant types that have no matmul kernel of their own in SGLang. The packed
rows stay resident; ``gguf_dequantize`` turns them into dense weights (fp32, fp16
or bf16) whose fp32 value is bit-identical to ik_llama.cpp's CPU dequantisation,
rounded once to a 16-bit dtype. ``gguf_dequantize_matmul`` is ``x @ W.T`` through
a scratch buffer that is reused by every call, so serving such a type costs the
packed bytes plus one bounded scratch buffer, not a dense copy of the model.

A row is ``row header + K / block * block bytes`` long (ggml_row_size). Three
types are stored by ik_llama.cpp as interleaved groups of 4 or 8 rows
(IQ1_S_R4, IQ1_M_R4, Q8_K_R8); the kernels read them regrouped into one row
after the other, which is what the GGUF loader hands over
(``sglang.srt.model_loader.gguf_ik.rowgroups``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    parallel_compile,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# ggml type id -> (name, values a block, bytes a block, row header bytes): the
# types jit/csrc/gemm/gguf_dequantize.cuh has a Format for.
GGUF_DEQUANTIZE_TYPES: Dict[int, Tuple[str, int, int, int]] = {
    133: ("Q6_0", 32, 26, 0),
    134: ("IQ1_BN", 64, 13, 2),
    135: ("IQ2_BN", 64, 16, 4),
    137: ("IQ2_K", 256, 76, 0),
    138: ("IQ3_K", 256, 110, 0),
    139: ("IQ4_K", 256, 144, 0),
    140: ("IQ5_K", 256, 176, 0),
    141: ("IQ6_K", 256, 212, 0),
    144: ("IQ4_KS", 256, 136, 4),
    145: ("IQ2_KS", 256, 70, 2),
    146: ("IQ4_KSS", 256, 128, 4),
    151: ("Q8_KV", 32, 32, 8),
    152: ("IQ5_KS", 256, 168, 4),
    153: ("IQ2_KT", 256, 68, 4),
    154: ("IQ3_KT", 256, 100, 4),
    155: ("IQ4_KT", 256, 128, 4),
    156: ("IQ3_KS", 256, 102, 2),
    157: ("IQ2_KL", 256, 86, 2),
    158: ("IQ1_KT", 256, 56, 4),
    219: ("IQ1_S_R4", 32, 6, 2),
    229: ("IQ1_M_R4", 32, 7, 2),
    399: ("Q8_K_R8", 256, 258, 0),
}

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def gguf_dequantize_columns(quant_type: int, row_bytes: int) -> int:
    """Values in a row of ``row_bytes`` packed bytes of ``quant_type``."""
    try:
        name, block, block_bytes, meta = GGUF_DEQUANTIZE_TYPES[int(quant_type)]
    except KeyError:
        raise ValueError(
            f"No GPU dequantise kernel for ggml type id {int(quant_type)}"
        ) from None
    blocks, rest = divmod(row_bytes - meta, block_bytes)
    if blocks <= 0 or rest:
        raise ValueError(
            f"{name}: a row of {row_bytes} bytes is not a {meta}-byte row header "
            f"plus whole blocks of {block_bytes} bytes"
        )
    return blocks * block


@cache_once
def _jit_gguf_dequantize_module(dtype: torch.dtype, quant_type: int) -> Module:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported GGUF dequantise output dtype: {dtype}")
    if quant_type not in GGUF_DEQUANTIZE_TYPES:
        raise ValueError(f"No GPU dequantise kernel for ggml type id {quant_type}")
    args = make_cpp_args(dtype, quant_type)
    return load_jit(
        "gguf_dequantize",
        *args,
        cuda_files=["gemm/gguf_dequantize.cuh"],
        cuda_wrappers=[("run", f"gguf_dequantize<{args}>")],
    )


def gguf_dequantize_warmup(dtype: torch.dtype, quant_types: Iterable[int]) -> None:
    """Build (or load from the JIT cache) the kernels of ``quant_types`` for ``dtype``.

    A model loader calls it once the types of a layer are known, so that no
    kernel is compiled inside the first forward or a CUDA graph capture.
    """
    types = sorted({int(t) for t in quant_types})
    parallel_compile(
        [(lambda t=t: _jit_gguf_dequantize_module(dtype, t)) for t in types]
    )


@debug_kernel_api
def gguf_dequantize(
    qweight: torch.Tensor,
    quant_type: int,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Packed rows uint8 [rows, row_bytes] -> dense [rows, K] in ``dtype``.

    ``out`` (contiguous, [rows, K], ``dtype``) is written in full when given;
    otherwise a new tensor is returned.
    """
    quant_type = int(quant_type)
    k = gguf_dequantize_columns(quant_type, qweight.shape[-1])
    if qweight.dim() != 2 or qweight.dtype != torch.uint8:
        raise ValueError(
            f"GGUF dequantise expects uint8 [rows, row_bytes], got "
            f"{qweight.dtype} {tuple(qweight.shape)}"
        )
    if not qweight.is_contiguous():
        qweight = qweight.contiguous()
    if out is None:
        out = torch.empty((qweight.shape[0], k), dtype=dtype, device=qweight.device)
    elif out.dtype != dtype:
        raise ValueError(f"GGUF dequantise: out is {out.dtype}, asked for {dtype}")
    _jit_gguf_dequantize_module(dtype, quant_type).run(qweight, out)
    return out


# One scratch buffer a (device, dtype), shared by every dequantise-and-multiply
# call of the process. A buffer that is outgrown is kept, never freed: a CUDA
# graph captured while it was current still writes into it at replay.
_SCRATCH: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
_RETIRED: List[torch.Tensor] = []


def gguf_dequantize_scratch_values(rows: int, columns: int, dtype: torch.dtype) -> int:
    """Values of scratch a [rows, columns] matrix needs: whole rows, at most
    SGLANG_GGUF_DEQUANT_SCRATCH_MB of them (one row at least)."""
    cap = max(1, envs.SGLANG_GGUF_DEQUANT_SCRATCH_MB.get()) << 20
    per_row = columns * dtype.itemsize
    return max(1, min(rows, cap // per_row)) * columns


def gguf_dequantize_reserve(
    device: torch.device, dtype: torch.dtype, rows: int, columns: int
) -> int:
    """Make the scratch buffer of (device, dtype) large enough for a
    [rows, columns] matrix and return its size in bytes.

    Loaders call it for every tensor they will serve this way, before the first
    forward, so that the buffer has its final size before any CUDA graph is
    captured.
    """
    device = torch.device(device)
    need = gguf_dequantize_scratch_values(rows, columns, dtype)
    key = (device, dtype)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < need:
        if buf is not None:
            _RETIRED.append(buf)
        buf = torch.empty(need, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf.numel() * dtype.itemsize


def gguf_dequantize_scratch_bytes() -> int:
    """Bytes the scratch buffers of this process hold (retired ones included)."""
    return sum(b.numel() * b.dtype.itemsize for b in list(_SCRATCH.values()) + _RETIRED)


def gguf_dequantize_matmul(
    x: torch.Tensor, qweight: torch.Tensor, quant_type: int
) -> torch.Tensor:
    """``x @ W.T`` for packed rows ``qweight`` [rows, row_bytes] of ``quant_type``.

    ``W`` is dequantised into the shared scratch buffer in ``x.dtype`` and
    multiplied dense. A matrix larger than the scratch cap is done in runs of
    rows, each multiplied before the next overwrites the buffer; the runs
    depend on the matrix and the cap only, never on what the buffer held
    before. Not reentrant: one forward thread a device, as the engine runs.
    """
    quant_type = int(quant_type)
    rows = qweight.shape[0]
    k = gguf_dequantize_columns(quant_type, qweight.shape[-1])
    dtype = x.dtype
    if rows == 0:
        return x.new_empty((*x.shape[:-1], 0))
    gguf_dequantize_reserve(qweight.device, dtype, rows, k)
    buf = _SCRATCH[(qweight.device, dtype)]
    step = gguf_dequantize_scratch_values(rows, k, dtype) // k
    if step >= rows:
        w = gguf_dequantize(
            qweight, quant_type, dtype, out=buf[: rows * k].view(rows, k)
        )
        return x @ w.T
    parts = []
    for r0 in range(0, rows, step):
        n = min(step, rows - r0)
        w = gguf_dequantize(
            qweight[r0 : r0 + n], quant_type, dtype, out=buf[: n * k].view(n, k)
        )
        parts.append(x @ w.T)
    return torch.cat(parts, dim=-1)
