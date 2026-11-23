# SPDX-License-Identifier: Apache-2.0
"""IQ1_S_R4, IQ1_M_R4 and Q8_K_R8: interleaved row groups to self-contained rows.

Three ik_llama.cpp types store 4 or 8 rows together and are not a base type
regrouped (``types.NOT_REPACKS``): IQ1_S_R4 and IQ1_M_R4 are quantisers of
their own, Q8_K_R8 keeps a half scale where Q8_K has a float. On disk a group
is the scales of its rows, then blocks that hold the bytes of all its rows:

* IQ1_S_R4: 4 half scales; a block of 32 values is ``qs[4 * i + k]`` (group of
  8 ``i``, row ``k``) and ``qh[k]`` (u16), 24 bytes for the 4 rows;
* IQ1_M_R4: 4 half scales; a block of 32 values is ``qs[8 * i + 4 * h + k]``
  (run of 16 ``i``, half ``h``, row ``k``), ``qh[4 * i + k]`` and ``scales[k]``,
  28 bytes for the 4 rows;
* Q8_K_R8: no row header; a block of 256 values is 8 half scales and
  ``qs[32 * ib + 4 * k + i]`` (4 values ``i`` of the run ``ib`` of row ``k``).

The rest of the GGUF path moves, selects and splits ROWS (an embedding lookup,
the head reorder of an architecture, tensor-parallel shards, MoE experts), so
``split_row_groups`` rewrites a group as its rows one after the other: each row
gets its own scale and the bytes of its own blocks, in the order ik reads them.
A row then has exactly ``ggml_row_size`` bytes, as the group's share had, and
no bit changes meaning: it is a byte permutation, and ``join_row_groups`` is
its inverse. The GPU dequantisers of these types read the row layout
(kernels/jit/include/sgl_kernel/quantization/gguf/dequant).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from sglang.srt.model_loader.gguf_ik.types import (
    ROW_GROUPS,
    GGMLType,
    GGMLTypeError,
    by_id,
)

_CHUNK_BYTES = 8 << 20


def _iq1_s_r4() -> np.ndarray:
    # block_iq1_s_r4 { uint8_t qs[16]; uint16_t qh[4]; } -> row k: qs of its 4 groups, its qh
    return np.array(
        [[4 * i + k for i in range(4)] + [16 + 2 * k, 17 + 2 * k] for k in range(4)]
    )


def _iq1_m_r4() -> np.ndarray:
    # block_iq1_m_r4 { uint8_t qs[16]; uint8_t qh[8]; uint8_t scales[4]; }
    # -> row k: qs of (run 0: half 0, half 1), (run 1: half 0, half 1), qh of run 0 and 1, scales
    return np.array(
        [[k, 4 + k, 8 + k, 12 + k, 16 + k, 20 + k, 24 + k] for k in range(4)]
    )


def _q8_k_r8() -> np.ndarray:
    # block_q8_k_r8 { ggml_half d[8]; int8_t qs[8 * 256]; } -> row k: its d, its 256 values
    return np.array(
        [
            [2 * k, 2 * k + 1]
            + [16 + 32 * ib + 4 * k + i for ib in range(64) for i in range(4)]
            for k in range(8)
        ]
    )


# name -> (row header bytes, bytes a block of ONE row, source offset of every
# byte of a row's block inside the group's block [rows, block bytes])
_LAYOUT: Dict[str, Tuple[int, int, np.ndarray]] = {}


def _layout(name: str) -> Tuple[int, int, np.ndarray]:
    if name not in _LAYOUT:
        hdr, src = {
            "IQ1_S_R4": (2, _iq1_s_r4),
            "IQ1_M_R4": (2, _iq1_m_r4),
            "Q8_K_R8": (0, _q8_k_r8),
        }[name]
        src = src()
        rows, blk = src.shape
        assert rows == ROW_GROUPS[name]
        assert sorted(src.reshape(-1).tolist()) == list(range(rows * blk)), name
        _LAYOUT[name] = (hdr, blk, src)
    return _LAYOUT[name]


def _plan(tensor_type, data: np.ndarray):
    t = tensor_type if isinstance(tensor_type, GGMLType) else by_id(int(tensor_type))
    if t.name not in ROW_GROUPS:
        raise GGMLTypeError(f"{t.name} is not stored as interleaved row groups")
    hdr, blk, src = _layout(t.name)
    R = ROW_GROUPS[t.name]
    data = np.ascontiguousarray(data, dtype=np.uint8)
    shape = data.shape
    if len(shape) < 2:
        raise GGMLTypeError(f"{t.name}: expected [rows, row_bytes], got {shape}")
    nrows, row_bytes = int(np.prod(shape[:-1])), shape[-1]
    if (row_bytes - hdr) % blk or nrows % R or shape[-2] % R:
        raise GGMLTypeError(
            f"{t.name}: {shape[-2]} rows of {row_bytes} bytes do not form whole "
            f"groups of {R} rows with blocks of {blk} bytes a row"
        )
    return t, hdr, blk, src, R, data, shape, nrows // R, (row_bytes - hdr) // blk


def split_row_groups(tensor_type, data: np.ndarray) -> np.ndarray:
    """Interleaved groups as ik stores them -> self-contained rows.

    ``data`` is uint8 ``[..., rows, row_bytes]`` (any leading dims, e.g. experts);
    returns a new uint8 array of the same shape.
    """
    _t, hdr, blk, src, R, data, shape, G, nb = _plan(tensor_type, data)
    row_bytes = shape[-1]
    d = data.reshape(G, R * row_bytes)
    out = np.empty((G, R, row_bytes), np.uint8)
    step = max(1, _CHUNK_BYTES // (R * row_bytes))
    for g0 in range(0, G, step):
        c = d[g0 : g0 + step]
        n = c.shape[0]
        if hdr:
            out[g0 : g0 + n, :, :hdr] = c[:, : R * hdr].reshape(n, R, hdr)
        blocks = c[:, R * hdr :].reshape(n, nb, R * blk)[:, :, src]  # [n, nb, R, blk]
        out[g0 : g0 + n, :, hdr:] = blocks.transpose(0, 2, 1, 3).reshape(n, R, nb * blk)
    return out.reshape(shape)


def join_row_groups(tensor_type, data: np.ndarray) -> np.ndarray:
    """The inverse of ``split_row_groups``: rows -> groups as ik stores them."""
    _t, hdr, blk, src, R, data, shape, G, nb = _plan(tensor_type, data)
    row_bytes = shape[-1]
    d = data.reshape(G, R, row_bytes)
    out = np.empty((G, R * row_bytes), np.uint8)
    if hdr:
        out[:, : R * hdr] = d[:, :, :hdr].reshape(G, R * hdr)
    blocks = (
        d[:, :, hdr:].reshape(G, R, nb, blk).transpose(0, 2, 1, 3)
    )  # [G, nb, R, blk]
    body = np.empty((G, nb, R * blk), np.uint8)
    body[:, :, src.reshape(-1)] = blocks.reshape(G, nb, R * blk)
    out[:, R * hdr :] = body.reshape(G, nb * R * blk)
    return out.reshape(shape)
