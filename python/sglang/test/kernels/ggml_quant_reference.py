# SPDX-License-Identifier: Apache-2.0
"""Bit-exact numpy references for the ggml / ik_llama.cpp quant types.

``dequantize(type_id, data, k)`` turns packed rows into float32 values that
equal, bit for bit, what ik_llama.cpp's CPU ``to_float`` (the
``dequantize_row_<type>`` named in ggml.c's type_traits) writes for the same
bytes. Every type is written from the ik_llama.cpp source (ggml-quants.c and
iqk/iqk_quantize.cpp at 7c1deaf3) and keeps its float operations in the same
order, so that each product rounds exactly where the C code rounds.

Covered: every base type of the ik_llama.cpp quant collection (legacy Q4_0 to
Q8_0 and Q6_0, Q8_KV, the K-quants, the IQ codebook types, ik's K / KS / KL /
KSS non-linear types, the KT trellis types, the BN ternary types) and the two
row-interleaved types that are quantisers of their own, IQ1_S_R4 and IQ1_M_R4.
The other _R4 / _R8 types are repacks of a base type and are not covered here.

Three facts a reader of the C code needs:

* Multiply-adds. A compiler that is allowed to (GCC's default for GNU C on a
  CPU with FMA) contracts ``a * b + c`` into one fused multiply-add, which
  rounds once. ``fused=True`` (default) rounds that way, ``fused=False``
  rounds the product and the sum separately. Only IQ6_K's value polynomial
  depends on it; in ``d * q - m`` (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K) the product
  of a half scale and a small integer is exact, so both give the same bits.
* Row headers. Several ik types start each row with a float or half scale
  (``TYPES[t].header`` bytes); IQ1_S_R4 / IQ1_M_R4 store four rows as one
  group, four half scales first, then the interleaved blocks.
* BN ternary types. libggml's CPU ``dequantize_row_iq1_bn / _iq2_bn`` take a
  pointer to the blocks and return the ternary codes without the row scale
  (the scale is applied in the dot product; ik's CUDA dequantiser applies it).
  ``dequantize`` returns the scaled weights ``d * code``;
  ``dequantize(..., bn_unscaled=True)`` returns the codes as libggml's CPU
  function does.

``random_rows`` builds valid packed rows (random bytes, with finite scales
written into every float / half field) so that every code, sign, grid index
and sub-scale of a type occurs. ``LibGGML`` wraps the few libggml calls the
tests compare against (ctypes; the library path comes from ``IK_LIBGGML``).
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
from typing import Callable, Dict, Optional

import numpy as np

QK_K = 256
F32 = np.float32
LIB_ENV = "IK_LIBGGML"


@dataclasses.dataclass(frozen=True)
class QType:
    type_id: int
    name: str
    block: int  # values per block
    block_bytes: int  # bytes per block (for the R4 types: per block of ONE row)
    header: int = 0  # bytes before the first block of a row (per row)
    rows: int = 1  # rows stored together as one group (IQ1_S_R4 / IQ1_M_R4: 4)
    fused: bool = False  # has a multiply-add whose rounding depends on FMA contraction

    def row_bytes(self, k: int) -> int:
        if k % self.block:
            raise ValueError(f"{self.name}: k = {k} is not a multiple of {self.block}")
        return self.header + k // self.block * self.block_bytes


_TYPE_LIST = (
    QType(2, "Q4_0", 32, 18),
    QType(3, "Q4_1", 32, 20, fused=True),
    QType(6, "Q5_0", 32, 22),
    QType(7, "Q5_1", 32, 24, fused=True),
    QType(8, "Q8_0", 32, 34),
    QType(10, "Q2_K", QK_K, 84, fused=True),
    QType(11, "Q3_K", QK_K, 110),
    QType(12, "Q4_K", QK_K, 144, fused=True),
    QType(13, "Q5_K", QK_K, 176, fused=True),
    QType(14, "Q6_K", QK_K, 210),
    QType(16, "IQ2_XXS", QK_K, 66),
    QType(17, "IQ2_XS", QK_K, 74),
    QType(18, "IQ3_XXS", QK_K, 98),
    QType(19, "IQ1_S", QK_K, 50),
    QType(20, "IQ4_NL", 32, 18),
    QType(21, "IQ3_S", QK_K, 110),
    QType(22, "IQ2_S", QK_K, 82),
    QType(23, "IQ4_XS", QK_K, 136),
    QType(29, "IQ1_M", QK_K, 56),
    QType(133, "Q6_0", 32, 26),
    QType(134, "IQ1_BN", 64, 13, header=2),
    QType(135, "IQ2_BN", 64, 16, header=4),
    QType(137, "IQ2_K", QK_K, 76),
    QType(138, "IQ3_K", QK_K, 110),
    QType(139, "IQ4_K", QK_K, 144),
    QType(140, "IQ5_K", QK_K, 176),
    QType(141, "IQ6_K", QK_K, 212, fused=True),
    QType(144, "IQ4_KS", QK_K, 136, header=4),
    QType(145, "IQ2_KS", QK_K, 70, header=2),
    QType(146, "IQ4_KSS", QK_K, 128, header=4),
    QType(151, "Q8_KV", 32, 32, header=8),
    QType(152, "IQ5_KS", QK_K, 168, header=4),
    QType(153, "IQ2_KT", QK_K, 68, header=4),
    QType(154, "IQ3_KT", QK_K, 100, header=4),
    QType(155, "IQ4_KT", QK_K, 128, header=4),
    QType(156, "IQ3_KS", QK_K, 102, header=2),
    QType(157, "IQ2_KL", QK_K, 86, header=2),
    QType(158, "IQ1_KT", QK_K, 56, header=4),
    QType(219, "IQ1_S_R4", 32, 6, header=2, rows=4),
    QType(229, "IQ1_M_R4", 32, 7, header=2, rows=4),
)
TYPES: Dict[int, QType] = {t.type_id: t for t in _TYPE_LIST}
TYPE_IDS: Dict[str, int] = {t.name: t.type_id for t in _TYPE_LIST}


def row_bytes(type_id: int, k: int) -> int:
    return TYPES[type_id].row_bytes(k)


# --------------------------------------------------------------------------- tables
# From ggml-common.h (ik_llama.cpp). The large codebook grids are the ones
# gguf-py ships for the same types; the tests prove them against libggml.
KMASK = np.array([1, 2, 4, 8, 16, 32, 64, 128], np.int64)
KSIGNS = np.array([i | ((bin(i).count("1") & 1) << 7) for i in range(128)], np.int64)
IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    np.int64,
)
IQ2NL = np.array([-31, -13, 1, 17, -26, -8, 6, 22], np.int64)
IQ2KL = np.array(
    [
        0xE9C1, 0x0DC1, 0xC1D8, 0xF6D8, 0x0DD8, 0x2FD8, 0xD8E9, 0xE9E9,
        0x01E9, 0x0DE9, 0x1CE9, 0xC1F6, 0x01F6, 0x0DF6, 0x2FF6, 0xE901,
        0xF601, 0x0101, 0x0D01, 0x1C01, 0xD80D, 0xE90D, 0xF60D, 0x010D,
        0x0D0D, 0xC11C, 0xE91C, 0x011C, 0x1C1C, 0x2F1C, 0xE92F, 0x0D2F,
    ],
    np.uint16,
).view(np.int8).astype(np.int64).reshape(32, 2)  # fmt: skip
IQ3NL = np.array(
    [-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51], np.int64
)
IQ4K = np.concatenate(
    [IQ4NL, [-123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117]]
).astype(np.int64)
IQ5NL = np.array(
    [
        -126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6,
        -1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121,
        -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4,
        1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123,
    ],
    np.int64,
)  # fmt: skip
# IQ6_K: y = dl * (A + q * (B + q * (-C + q * D)) + m), q in 0..63
A_IQ6K, B_IQ6K, C_IQ6K, D_IQ6K = F32(-127.0), F32(6.2568), F32(0.11218), F32(0.0011972)
IQ1S_DELTA = F32(0.125)
BN_MULT = np.array([81, 27, 9, 3, 1], np.int64)

_GRIDS: Dict[str, np.ndarray] = {}


def _grid(name: str) -> np.ndarray:
    """Codebook grid as float32 [entries, values]: gguf-py's decoding of ggml's tables."""
    if name not in _GRIDS:
        from gguf import quants

        cls = getattr(quants, name)
        cls.init_grid()
        _GRIDS[name] = np.ascontiguousarray(
            cls.grid.reshape(cls.grid_shape).astype(F32)
        )
    return _GRIDS[name]


# --------------------------------------------------------------------------- helpers
def _f16(b: np.ndarray) -> np.ndarray:
    """uint8 [..., 2] -> float32 [...] (exact, as GGML_FP16_TO_FP32)."""
    return np.ascontiguousarray(b).view("<f2")[..., 0].astype(F32)


def _f32(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view("<f4")[..., 0].astype(F32)


def _u16(b: np.ndarray) -> np.ndarray:
    """uint8 [..., 2n] -> int64 [..., n] little-endian."""
    return np.ascontiguousarray(b).view("<u2").astype(np.int64)


def _u32(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view("<u4").astype(np.int64)


def _i8(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.int8).astype(np.int64)


def _fl(x) -> np.ndarray:
    """Integer values -> float32, exact (they are small), as C's int -> float."""
    return np.asarray(x).astype(F32)


def _fma32(a, b, c) -> np.ndarray:
    """float32 fma(a, b, c) rounded once, like the hardware instruction.

    a * b is exact in float64; the float64 sum may round, so the error of the
    sum is carried (TwoSum) and decides the one case where rounding the float64
    sum to float32 differs from rounding the exact value: a float32 midpoint.
    """
    a64 = np.asarray(a, F32).astype(np.float64)
    b64 = np.asarray(b, F32).astype(np.float64)
    c64 = np.asarray(c, F32).astype(np.float64)
    p = a64 * b64
    s = p + c64
    bb = s - p
    e = (p - (s - bb)) + (c64 - bb)
    r = s.astype(F32)
    r64 = r.astype(np.float64)
    other = np.where(
        s > r64, np.nextafter(r, F32(np.inf)), np.nextafter(r, F32(-np.inf))
    )
    mid = (s != r64) & (e != 0) & ((r64 + other.astype(np.float64)) * 0.5 == s)
    if np.any(mid):
        hi = np.maximum(r, other)
        lo = np.minimum(r, other)
        r = np.where(mid & (e > 0), hi, np.where(mid & (e < 0), lo, r))
    return r


def _madd(a, b, c, fused: bool) -> np.ndarray:
    """a * b + c in float32: one rounding when fused, two otherwise."""
    if fused:
        return _fma32(a, b, c)
    return (np.asarray(a, F32) * np.asarray(b, F32)).astype(F32) + np.asarray(c, F32)


def _neg_where(v: np.ndarray, neg) -> np.ndarray:
    """v * (neg ? -1.f : 1.f): an exact sign flip (also of zeros)."""
    return np.where(neg, -v, v).astype(F32)


def _get_scale_min_k4(sc: np.ndarray):
    """Q4_K / Q5_K: 12 scale bytes [..., 12] -> (scales, mins) int64 [..., 8]."""
    q = sc.astype(np.int64)
    d = np.empty(q.shape[:-1] + (8,), np.int64)
    m = np.empty_like(d)
    d[..., :4] = q[..., 0:4] & 63
    m[..., :4] = q[..., 4:8] & 63
    d[..., 4:] = (q[..., 8:12] & 0xF) | ((q[..., 0:4] >> 6) << 4)
    m[..., 4:] = (q[..., 8:12] >> 4) | ((q[..., 4:8] >> 6) << 4)
    return d, m


# --------------------------------------------------------------------------- legacy 32-value blocks
def _q4_0(b, fused):
    d = _f16(b[..., 0:2])[..., None]
    qs = b[..., 2:18].astype(np.int64)
    q = np.concatenate([(qs & 15) - 8, (qs >> 4) - 8], -1)
    return _fl(q) * d


def _q4_1(b, fused):
    d = _f16(b[..., 0:2])[..., None]
    m = _f16(b[..., 2:4])[..., None]
    qs = b[..., 4:20].astype(np.int64)
    q = np.concatenate([qs & 15, qs >> 4], -1)
    return _madd(_fl(q), d, m, fused)


def _q5_qh(qh, qs):
    j = np.arange(16)
    xh0 = ((qh[..., None] >> j) << 4) & 0x10
    xh1 = (qh[..., None] >> (j + 12)) & 0x10
    return np.concatenate([(qs & 15) | xh0, (qs >> 4) | xh1], -1)


def _q5_0(b, fused):
    d = _f16(b[..., 0:2])[..., None]
    q = _q5_qh(_u32(b[..., 2:6])[..., 0], b[..., 6:22].astype(np.int64)) - 16
    return _fl(q) * d


def _q5_1(b, fused):
    d = _f16(b[..., 0:2])[..., None]
    m = _f16(b[..., 2:4])[..., None]
    q = _q5_qh(_u32(b[..., 4:8])[..., 0], b[..., 8:24].astype(np.int64))
    return _madd(_fl(q), d, m, fused)


def _q6_0(b, fused):
    d = _f16(b[..., 0:2])[..., None]
    qh = b[..., 2:10].astype(np.int64)
    qs = b[..., 10:26].astype(np.int64)
    j = np.arange(16)
    h = qh[..., j % 8] >> (4 * (j // 8))
    x0 = ((qs & 15) | ((h << 4) & 0x30)) - 32
    x1 = ((qs >> 4) | ((h << 2) & 0x30)) - 32
    return _fl(np.concatenate([x0, x1], -1)) * d


def _q8_0(b, fused):
    return _fl(_i8(b[..., 2:34])) * _f16(b[..., 0:2])[..., None]


# --------------------------------------------------------------------------- K-quants
def _q2_k(b, fused):
    sc = b[..., 0:16].astype(np.int64)
    qs = b[..., 16:80].astype(np.int64).reshape(*b.shape[:-1], 2, 32)
    d = _f16(b[..., 80:82])[..., None, None, None]
    dmin = _f16(b[..., 82:84])[..., None, None, None]
    j = np.arange(4)
    q = (qs[..., :, None, :] >> (2 * j)[:, None]) & 3  # [.., h, j, l]
    s = sc.reshape(*b.shape[:-1], 2, 4, 2)  # [.., h, j, first/second 16]
    s = np.repeat(s, 16, axis=-1)  # [.., h, j, 32]
    dl = d * _fl(s & 0xF)
    ml = dmin * _fl(s >> 4)
    y = _madd(dl, _fl(q), -ml, fused)
    return y.reshape(*b.shape[:-1], QK_K)


def _q3_k(b, fused):
    hm = b[..., 0:32].astype(np.int64)
    qs = b[..., 32:96].astype(np.int64).reshape(*b.shape[:-1], 2, 32)
    aux = _u32(b[..., 96:108])  # [.., 3]
    d = _f16(b[..., 108:110])[..., None, None, None]
    k1, k2 = 0x03030303, 0x0F0F0F0F
    a0, a1, tmp = aux[..., 0], aux[..., 1], aux[..., 2]
    s2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    s3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    s0 = (a0 & k2) | (((tmp >> 0) & k1) << 4)
    s1 = (a1 & k2) | (((tmp >> 2) & k1) << 4)
    words = np.stack([s0, s1, s2, s3], -1).astype("<u4")
    scales = words.view(np.int8).astype(np.int64)  # [.., 16]
    h = np.arange(2)[:, None, None]
    j = np.arange(4)[None, :, None]
    q = (qs[..., :, None, :] >> (2 * j)) & 3  # [.., h, j, l]
    hbit = (hm[..., None, None, :] >> (4 * h + j)) & 1
    q = q - np.where(hbit == 1, 0, 4)
    s = np.repeat(scales.reshape(*b.shape[:-1], 2, 4, 2), 16, axis=-1)
    dl = d * _fl(s - 32)
    return (dl * _fl(q)).reshape(*b.shape[:-1], QK_K)


def _q45_k(b, qs_off, with_qh, fused):
    d = _f16(b[..., 0:2])[..., None, None]
    dmin = _f16(b[..., 2:4])[..., None, None]
    sc, mn = _get_scale_min_k4(b[..., 4:16])  # [.., 8]
    qs = b[..., qs_off : qs_off + 128].astype(np.int64).reshape(*b.shape[:-1], 4, 32)
    q = np.stack([qs & 15, qs >> 4], -2)  # [.., c, 2, 32]
    if with_qh:
        qh = b[..., 16:48].astype(np.int64)
        c = np.arange(4)[:, None, None]
        half = np.arange(2)[None, :, None]
        q = q + 16 * ((qh[..., None, None, :] >> (2 * c + half)) & 1)
    sc = sc.reshape(*b.shape[:-1], 4, 2)[..., None]
    mn = mn.reshape(*b.shape[:-1], 4, 2)[..., None]
    dl = d[..., None] * _fl(sc)
    ml = dmin[..., None] * _fl(mn)
    return _madd(dl, _fl(q), -ml, fused).reshape(*b.shape[:-1], QK_K)


def _q4_k(b, fused):
    return _q45_k(b, 16, False, fused)


def _q5_k(b, fused):
    return _q45_k(b, 48, True, fused)


def _q6_k(b, fused):
    ql = b[..., 0:128].astype(np.int64).reshape(*b.shape[:-1], 2, 64)
    qh = b[..., 128:192].astype(np.int64).reshape(*b.shape[:-1], 2, 32)
    sc = _i8(b[..., 192:208]).reshape(*b.shape[:-1], 2, 8)
    d = _f16(b[..., 208:210])[..., None, None, None]
    lo, hi = ql[..., :32], ql[..., 32:]
    q1 = ((lo & 15) | (((qh >> 0) & 3) << 4)) - 32
    q2 = ((hi & 15) | (((qh >> 2) & 3) << 4)) - 32
    q3 = ((lo >> 4) | (((qh >> 4) & 3) << 4)) - 32
    q4 = ((hi >> 4) | (((qh >> 6) & 3) << 4)) - 32
    q = np.stack([q1, q2, q3, q4], -2)  # [.., h, 4, 32]
    isl = np.arange(32) // 16
    s = sc[..., (isl[None, :] + 2 * np.arange(4)[:, None])]  # [.., h, 4, 32]
    return ((d * _fl(s)) * _fl(q)).reshape(*b.shape[:-1], QK_K)


# --------------------------------------------------------------------------- IQ codebook types
def _db_iq2(d, ls, half: F32):
    # d * (0.5f + ls) * 0.25f (IQ2_*) or * 0.5f (IQ3_XXS), left to right
    return ((d * (F32(0.5) + _fl(ls))).astype(F32) * half).astype(F32)


def _iq2_xxs(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    q = b[..., 2:66].reshape(*lead, 8, 8)
    aux1 = _u32(q[..., 4:8])[..., 0]  # [.., 8]
    db = _db_iq2(d, aux1 >> 28, F32(0.25))  # [.., 8]
    grid = _grid("IQ2_XXS")[q[..., 0:4].astype(np.int64)]  # [.., 8, 4, 8]
    signs = KSIGNS[(aux1[..., None] >> (7 * np.arange(4))) & 127]  # [.., 8, 4]
    neg = (signs[..., None] & KMASK) != 0
    v = (db[..., None, None] * grid).astype(F32)
    return _neg_where(v, neg).reshape(*lead, QK_K)


def _iq2_xs(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    qs = _u16(b[..., 2:66]).reshape(*lead, 8, 4)
    sc = b[..., 66:74].astype(np.int64)
    db = np.stack([_db_iq2(d, sc & 15, F32(0.25)), _db_iq2(d, sc >> 4, F32(0.25))], -1)
    db = db[..., np.arange(4) // 2]  # [.., 8, 4]
    grid = _grid("IQ2_XS")[qs & 511]
    neg = (KSIGNS[qs >> 9][..., None] & KMASK) != 0
    v = (db[..., None] * grid).astype(F32)
    return _neg_where(v, neg).reshape(*lead, QK_K)


def _iq2_s(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    qs = b[..., 2:34].astype(np.int64).reshape(*lead, 8, 4)
    sg = b[..., 34:66].astype(np.int64).reshape(*lead, 8, 4)
    qh = b[..., 66:74].astype(np.int64)
    sc = b[..., 74:82].astype(np.int64)
    db = np.stack([_db_iq2(d, sc & 15, F32(0.25)), _db_iq2(d, sc >> 4, F32(0.25))], -1)
    db = db[..., np.arange(4) // 2]
    idx = qs | ((qh[..., None] << (8 - 2 * np.arange(4))) & 0x300)
    grid = _grid("IQ2_S")[idx]
    neg = (sg[..., None] & KMASK) != 0
    v = (db[..., None] * grid).astype(F32)
    return _neg_where(v, neg).reshape(*lead, QK_K)


def _iq3_xxs(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    qs = b[..., 2:66].astype(np.int64).reshape(*lead, 8, 4, 2)
    aux = _u32(b[..., 66:98])  # [.., 8]
    db = _db_iq2(d, aux >> 28, F32(0.5))
    signs = KSIGNS[(aux[..., None] >> (7 * np.arange(4))) & 127]  # [.., 8, 4]
    grid = _grid("IQ3_XXS")[qs].reshape(*lead, 8, 4, 8)
    neg = (signs[..., None] & KMASK) != 0
    v = (db[..., None, None] * grid).astype(F32)
    return _neg_where(v, neg).reshape(*lead, QK_K)


def _iq3_s(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    qs = b[..., 2:66].astype(np.int64).reshape(*lead, 8, 4, 2)
    qh = b[..., 66:74].astype(np.int64)
    sg = b[..., 74:106].astype(np.int64).reshape(*lead, 8, 4)
    sc = b[..., 106:110].astype(np.int64)
    db = np.stack([d * _fl(1 + 2 * (sc & 15)), d * _fl(1 + 2 * (sc >> 4))], -1)
    db = db.reshape(*lead, 8)
    l2 = 2 * np.arange(4)
    i1 = qs[..., 0] | ((qh[..., None] << (8 - l2)) & 256)
    i2 = qs[..., 1] | ((qh[..., None] << (7 - l2)) & 256)
    grid = np.concatenate([_grid("IQ3_S")[i1], _grid("IQ3_S")[i2]], -1)
    neg = (sg[..., None] & KMASK) != 0
    v = (db[..., None, None] * grid).astype(F32)
    return _neg_where(v, neg).reshape(*lead, QK_K)


def _iq1_s(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    qs = b[..., 2:34].astype(np.int64).reshape(*lead, 8, 4)
    qh = _u16(b[..., 34:50])  # [.., 8]
    dl = d * _fl(2 * ((qh >> 12) & 7) + 1)
    delta = np.where(qh & 0x8000, -IQ1S_DELTA, IQ1S_DELTA).astype(F32)
    idx = qs | (((qh[..., None] >> (3 * np.arange(4))) & 7) << 8)
    grid = _grid("IQ1_S")[idx]  # [.., 8, 4, 8]
    v = dl[..., None, None] * (grid + delta[..., None, None])
    return v.astype(F32).reshape(*lead, QK_K)


def _iq1_m(b, fused):
    lead = b.shape[:-1]
    qs = b[..., 0:32].astype(np.int64).reshape(*lead, 8, 4)
    qh = b[..., 32:48].astype(np.int64).reshape(*lead, 8, 2)
    sc = _u16(b[..., 48:56])  # [.., 4]
    u = (sc[..., 0] >> 12) | ((sc[..., 1] >> 8) & 0xF0) | ((sc[..., 2] >> 4) & 0xF00)
    u = u | (sc[..., 3] & 0xF000)
    d = u.astype("<u2").view("<f2").astype(F32)[..., None]
    ib = np.arange(8)
    s = sc[..., ib // 2]  # [.., 8]
    sh = 6 * (ib % 2)
    dl1 = d * _fl(2 * ((s >> sh) & 7) + 1)
    dl2 = d * _fl(2 * ((s >> (sh + 3)) & 7) + 1)
    h = qh[..., [0, 0, 1, 1]]  # [.., 8, 4]
    idx = qs | ((h << np.array([8, 4, 8, 4])) & 0x700)
    dbit = h & np.array([0x08, 0x80, 0x08, 0x80])
    delta = np.where(dbit != 0, -IQ1S_DELTA, IQ1S_DELTA).astype(F32)
    dl = np.stack([dl1, dl1, dl2, dl2], -1)  # [.., 8, 4]
    grid = _grid("IQ1_S")[idx]
    v = dl[..., None] * (grid + delta[..., None])
    return v.astype(F32).reshape(*lead, QK_K)


def _iq4_nl(b, fused):
    d = _f16(b[..., 0:2])[..., None]
    qs = b[..., 2:18].astype(np.int64)
    q = np.concatenate([qs & 15, qs >> 4], -1)
    return d * _fl(IQ4NL[q])


def _iq4_xs(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    sh = _u16(b[..., 2:4])  # [.., 1]
    sl = b[..., 4:8].astype(np.int64)
    qs = b[..., 8:136].astype(np.int64).reshape(*lead, 8, 16)
    ib = np.arange(8)
    ls = ((sl[..., ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
    dl = d * _fl(ls - 32)
    q = np.concatenate([qs & 15, qs >> 4], -1)
    return (dl[..., None] * _fl(IQ4NL[q])).reshape(*lead, QK_K)


# --------------------------------------------------------------------------- ik K types (block-scaled)
def _iq2_k(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    extra = _u16(b[..., 2:4])  # [.., 1]
    sc = b[..., 4:12].astype(np.int64)
    qs = b[..., 12:76].astype(np.int64).reshape(*lead, 2, 32)
    ib = np.arange(8)
    dl = np.stack([d * _fl((sc & 15) - 8), d * _fl((sc >> 4) - 8)], -1)  # [.., 8, 2]
    vsel = (
        (extra[..., None] >> (2 * ib[:, None] + np.arange(2))) & 1
    ) * 4  # [.., 8, 2]
    q = (qs[..., ib // 4, :] >> (2 * (ib % 4))[:, None]) & 3  # [.., 8, 32]
    q = q.reshape(*lead, 8, 2, 16)
    v = IQ2NL[q + vsel[..., None]]
    return (dl[..., None] * _fl(v)).reshape(*lead, QK_K)


def _iq2_ks(b, d, fused):
    lead = b.shape[:-1]
    extra = _u16(b[..., 0:2])  # [.., 1]
    sc = b[..., 2:6].astype(np.int64)  # [.., 4]
    qs = b[..., 6:70].astype(np.int64).reshape(*lead, 2, 32)
    i = np.arange(4)
    ex = extra >> (2 * i)  # [.., 4]
    ls1 = ((sc & 15) | ((ex >> 4) & 0x10)) - 16
    ls2 = ((sc >> 4) | ((ex >> 5) & 0x10)) - 16
    dl = np.stack([d * _fl(ls1), d * _fl(ls2)], -1)  # [.., 4, 2]
    vsel = np.stack([ex & 1, (ex >> 1) & 1], -1) * 4
    half = qs[..., i // 2, :]  # [.., 4, 32]
    sh = (4 * (i % 2))[:, None]
    q = np.stack([(half >> sh) & 3, (half >> (sh + 2)) & 3], -2)  # [.., 4, 2, 32]
    v = IQ2NL[q + vsel[..., None]]
    return (dl[..., None] * _fl(v)).reshape(*lead, QK_K)


def _iq2_kl(b, d, fused):
    lead = b.shape[:-1]
    sh = _u16(b[..., 0:2])  # [.., 1]
    sl = b[..., 2:6].astype(np.int64)
    qs = b[..., 6:70].astype(np.int64).reshape(*lead, 4, 16)
    qh = b[..., 70:86].astype(np.int64)  # [.., 16]
    i = np.arange(4)
    ls1 = ((sl[..., (2 * i) % 4] >> (4 * (i // 2))) & 0xF) | (
        ((sh >> (4 * i)) & 3) << 4
    )
    ls2 = ((sl[..., (2 * i + 1) % 4] >> (4 * (i // 2))) & 0xF) | (
        ((sh >> (4 * i + 2)) & 3) << 4
    )
    dl1 = d * _fl(ls1 - 32)
    dl2 = d * _fl(ls2 - 32)
    i1 = (qs & 15) | (((qh[..., None, :] >> (2 * i)[:, None]) & 1) << 4)  # [.., 4, 16]
    i2 = (qs >> 4) | (((qh[..., None, :] >> (2 * i + 1)[:, None]) & 1) << 4)
    v1 = _fl(IQ2KL[i1]).reshape(*lead, 4, 32)  # pairs -> y[2j], y[2j+1]
    v2 = _fl(IQ2KL[i2]).reshape(*lead, 4, 32)
    y = np.concatenate([dl1[..., None] * v1, dl2[..., None] * v2], -1)
    return y.astype(F32).reshape(*lead, QK_K)


def _iq3_k(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    extra = _u16(b[..., 2:4])
    sh = _u16(b[..., 4:6])
    sl = b[..., 6:14].astype(np.int64)
    qs = b[..., 14:78].astype(np.int64).reshape(*lead, 2, 2, 16)
    qh = b[..., 78:110].astype(np.int64).reshape(*lead, 2, 16)
    ib = np.arange(8)
    k2 = np.arange(2)
    bit = 2 * ib[:, None] + k2  # [8, 2]
    mag = np.stack([2 * (sl & 15) + 1, 2 * (sl >> 4) + 1], -1)  # [.., 8, 2]
    sgn = np.where((sh[..., None] >> bit) & 1, -1, 1)
    dl = d[..., None] * _fl(mag * sgn)
    vsel = ((extra[..., None] >> bit) & 1) * 8
    q = (qs[..., ib // 4, :, :] >> (2 * (ib % 4))[:, None, None]) & 3  # [.., 8, 2, 16]
    h = (qh[..., None, :, :] >> ib[:, None, None]) & 1  # [.., 8, 2, 16]
    v = IQ3NL[(q | (h << 2)) + vsel[..., None]]
    return (dl[..., None] * _fl(v)).reshape(*lead, QK_K)


def _iq3_ks(b, d, fused):
    lead = b.shape[:-1]
    extra = _u16(b[..., 0:2])
    sc = b[..., 2:6].astype(np.int64)
    qs = b[..., 6:70].astype(np.int64).reshape(*lead, 2, 32)
    qh = b[..., 70:102].astype(np.int64)  # [.., 32]
    j = np.arange(4)
    ls1 = (sc & 15) | (((extra >> j) & 1) << 4)
    ls2 = (sc >> 4) | (((extra >> (j + 4)) & 1) << 4)
    dl = np.concatenate([d * _fl(ls1 - 16), d * _fl(ls2 - 16)], -1)  # [.., 8]
    blk = np.arange(8)  # 4 * i128 + ib
    i128, ibb = blk // 4, blk % 4
    vsel = ((extra >> (8 + blk)) & 1) << 3  # [.., 8]
    q = (qs[..., i128, :] >> (2 * ibb)[:, None]) & 3  # [.., 8, 32]
    h = (qh[..., None, :] >> blk[:, None]) & 1
    v = IQ3NL[(q | (h << 2)) + vsel[..., None]]
    return (dl[..., None] * _fl(v)).reshape(*lead, QK_K)


def _iq4_k(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    extra = _u16(b[..., 2:4])
    shb = b[..., 4:8].astype(np.int64)
    sl = b[..., 8:16].astype(np.int64)
    qs = b[..., 16:144].astype(np.int64).reshape(*lead, 8, 16)
    ib = np.arange(8)
    sh = (shb[..., ib // 2] >> (4 * (ib % 2))) & 0xFF
    ls1 = ((sl & 15) | ((sh << 4) & 0x30)) - 32
    ls2 = ((sl >> 4) | ((sh << 2) & 0x30)) - 32
    dl = np.stack([d * _fl(ls1), d * _fl(ls2)], -1)  # [.., 8, 2]
    vsel = ((extra[..., None] >> (2 * ib[:, None] + np.arange(2))) & 1) * 16
    q = np.stack([qs & 15, qs >> 4], -2)  # [.., 8, 2, 16]
    v = IQ4K[q + vsel[..., None]]
    return (dl[..., None] * _fl(v)).reshape(*lead, QK_K)


def _iq5_k(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None]
    extra = _u16(b[..., 2:4])
    sh = b[..., 4:8].astype(np.int64)  # [.., 4]
    sl = b[..., 8:16].astype(np.int64).reshape(*lead, 4, 2)
    qs = b[..., 16:144].astype(np.int64).reshape(*lead, 4, 2, 16)
    qh = b[..., 144:176].astype(np.int64).reshape(*lead, 2, 16)
    i = np.arange(4)
    s0, s1 = sl[..., 0], sl[..., 1]
    ls = np.stack(
        [
            (s0 & 15) | ((sh << 4) & 0x30),
            (s0 >> 4) | ((sh << 2) & 0x30),
            (s1 & 15) | ((sh >> 0) & 0x30),
            (s1 >> 4) | ((sh >> 2) & 0x30),
        ],
        -1,
    )  # [.., 4, 4]
    dl = d[..., None] * _fl(ls - 32)
    vsel = (
        (extra[..., None] >> (4 * i[:, None] + np.arange(4))) & 1
    ) * 32  # [.., 4, 4]
    h = qh[..., None, :, :] >> (2 * i)[:, None, None]  # [.., 4, 2, 16]
    lo = qs & 15
    hi = qs >> 4
    q = np.stack(
        [
            lo[..., 0, :] | ((h[..., 0, :] & 1) << 4),
            lo[..., 1, :] | ((h[..., 1, :] & 1) << 4),
            hi[..., 0, :] | ((h[..., 0, :] & 2) << 3),
            hi[..., 1, :] | ((h[..., 1, :] & 2) << 3),
        ],
        -2,
    )  # [.., 4, 4, 16]
    v = IQ5NL[q + vsel[..., None]]
    return (dl[..., None] * _fl(v)).reshape(*lead, QK_K)


def _iq6_k(b, fused):
    lead = b.shape[:-1]
    d = _f16(b[..., 0:2])[..., None, None]
    extra = _u16(b[..., 2:4])
    sc = _i8(b[..., 4:20]).reshape(*lead, 4, 4)
    qs = b[..., 20:148].astype(np.int64).reshape(*lead, 4, 2, 16)
    qh = b[..., 148:212].astype(np.int64).reshape(*lead, 2, 2, 16)
    i = np.arange(4)
    hq = qh[..., i // 2, :, :] >> (4 * (i % 2))[:, None, None]  # [.., 4, 2, 16]
    lo = qs & 15
    hi = qs >> 4
    q = np.stack(
        [
            lo[..., 0, :] | ((hq[..., 0, :] & 3) << 4),
            lo[..., 1, :] | ((hq[..., 1, :] & 3) << 4),
            hi[..., 0, :] | ((hq[..., 0, :] & 0x0C) << 2),
            hi[..., 1, :] | ((hq[..., 1, :] & 0x0C) << 2),
        ],
        -2,
    )  # [.., 4, 4, 16]
    dl = (d * _fl(sc))[..., None]
    m = np.where((extra[..., None] >> (4 * i[:, None] + np.arange(4))) & 1, 1.0, 0.0)
    m = m.astype(F32)[..., None]
    qf = _fl(q)
    t = _madd(qf, D_IQ6K, -C_IQ6K, fused)
    t = _madd(qf, t, B_IQ6K, fused)
    t = _madd(qf, t, A_IQ6K, fused)
    return (dl * (t + m)).astype(F32).reshape(*lead, QK_K)


def _iq4_ks(b, d, fused):
    lead = b.shape[:-1]
    sc = b[..., 0:8].astype(np.int64)
    qs = b[..., 8:136].astype(np.int64).reshape(*lead, 8, 16)
    dl = d * _fl((sc & 254) - 127)
    q = np.concatenate([qs & 15, qs >> 4], -1) + ((sc & 1) << 4)[..., None]
    return (dl[..., None] * _fl(IQ4K[q])).reshape(*lead, QK_K)


def _iq5_ks(b, d, fused):
    lead = b.shape[:-1]
    sc = b[..., 0:8].astype(np.int64).reshape(*lead, 4, 2)
    qs = b[..., 8:136].astype(np.int64).reshape(*lead, 4, 32)
    qh = b[..., 136:168].astype(np.int64)
    i = np.arange(4)
    dl = d[..., None] * _fl((sc & 254) - 127)  # [.., 4, 2]
    q1 = (qs & 15) | (((qh[..., None, :] >> (2 * i)[:, None]) & 1) << 4)
    q2 = (qs >> 4) | (((qh[..., None, :] >> (2 * i + 1)[:, None]) & 1) << 4)
    q = np.stack([q1, q2], -2) + ((sc & 1) << 5)[..., None]
    return (dl[..., None] * _fl(IQ5NL[q])).reshape(*lead, QK_K)


def _iq4_kss(b, d, fused):
    lead = b.shape[:-1]
    qs = _u16(b[..., 0:128]).reshape(*lead, 8, 8)
    a = qs & 0xFFFE
    a = a ^ (a >> 1)
    ls = ((qs & 1) << np.arange(8)).sum(-1)  # [.., 8]
    aux8 = a.astype("<u2").view(np.uint8).astype(np.int64)  # [.., 8, 16]
    q = np.concatenate([aux8 & 15, aux8 >> 4], -1) + ((ls & 1) << 4)[..., None]
    dl = d * _fl((ls & 254) - 127)
    return (dl[..., None] * _fl(IQ4K[q])).reshape(*lead, QK_K)


# --------------------------------------------------------------------------- ik KT trellis types
def _kt_values(idx: np.ndarray, group: int, offset, absolute: bool = False):
    """QuantizerIQKT<..., is_int=true>::set_values without the scale: float32 [..., group]."""
    x = (idx.astype(np.int64) + offset).astype(np.uint32)
    out = []
    for _ in range(group):
        x = (x * np.uint32(0xCBAC1FED)).astype(np.uint32)
        s = (x & np.uint32(0x3F3F3F3F)).astype(np.int64)
        tot = (s & 0xFF) + ((s >> 8) & 0xFF) + ((s >> 16) & 0xFF) + (s >> 24)
        v = _fl(tot) - F32(126.0)
        out.append(np.abs(v) if absolute else v)
    return np.stack(out, -1).astype(F32)


def _iq1_kt(b, d, fused):
    lead = b.shape[:-1]
    sh = b[..., 0:8].astype(np.int64)
    ql = b[..., 8:40].astype(np.int64).reshape(*lead, 8, 4)
    qh = b[..., 40:56].astype(np.int64).reshape(*lead, 4, 4)
    ib = np.arange(8)
    ig = np.arange(4)
    hq = qh[..., ib % 4, :] << (8 - 4 * (ib // 4))[:, None]  # [.., 8, 4]
    idx = ql | (hq & 0xF00) | ((sh[..., None] << (8 - ig)) & 0x1000)
    sl = d * _fl(IQ4K[sh & 15])  # [.., 8]
    v = _kt_values(idx, 8, 4096)
    return (sl[..., None, None] * v).astype(F32).reshape(*lead, QK_K)


def _iq2_kt(b, d, fused):
    lead = b.shape[:-1]
    sc = b[..., 0:4].astype(np.int64)
    ql = _u16(b[..., 4:68]).reshape(*lead, 2, 4, 4)  # [.., half, ib, ig]
    sl = np.stack([d * _fl(IQ4K[sc & 15]), d * _fl(IQ4K[sc >> 4])], -2)  # [.., 2, 4]
    v = _kt_values(ql, 8, 4096)
    return (sl[..., None, None] * v).astype(F32).reshape(*lead, QK_K)


def _iq3_kt(b, d, fused):
    lead = b.shape[:-1]
    d = (d * F32(1.01)).astype(F32)
    sc = b[..., 0:4].astype(np.int64)
    ql = _u16(b[..., 4:68]).reshape(*lead, 2, 4, 4)
    qh = b[..., 68:100].astype(np.int64)  # [.., 32]: byte w % 32, bit w // 32
    sl = np.stack([d * _fl(sc & 15), d * _fl(sc >> 4)], -2)
    v = (sl[..., None, None] * _kt_values(ql, 8, 4096, absolute=True)).astype(F32)
    v = v.reshape(*lead, QK_K)
    w = np.arange(QK_K)
    neg = (qh[..., w % 32] >> (w // 32)) & 1
    return _neg_where(v, neg != 0)


def _iq4_kt(b, d, fused):
    lead = b.shape[:-1]
    shb = _u32(b[..., 0:32])  # [.., 8]
    ql = b[..., 32:96].astype(np.int64).reshape(*lead, 8, 8)
    qh = b[..., 96:128].astype(np.int64)
    jj = np.arange(64).reshape(8, 8)
    ig = np.arange(8)
    idx = ql | ((qh[..., jj % 32] << (8 - 4 * (jj // 32))) & 0xF00)
    idx = idx | (((shb[..., None] >> (8 + 3 * ig)) & 7) << 12)
    offset = np.where(shb & 1, 32768 + 4096, 4096)[..., None]
    ls = ((shb & 0xFF) >> 1) - 64
    sl = d * _fl(ls)
    v = _kt_values(idx, 4, offset)
    return (sl[..., None, None] * v).astype(F32).reshape(*lead, QK_K)


# --------------------------------------------------------------------------- BN ternary types (codes)
def _iq1_bn_codes(b):
    lead = b.shape[:-1]
    ql = b[..., 0:12].astype(np.int64).reshape(*lead, 4, 3)
    extra = b[..., 12].astype(np.int64)

    def tern(q, mult):
        v = (q * mult) & 0xFF
        return ((v + (v >> 1)) >> 7) - 1

    main = tern(ql[..., None], BN_MULT).reshape(*lead, 4, 15)
    last = tern(extra[..., None], BN_MULT[:4])  # [.., 4]
    return _fl(np.concatenate([main, last[..., None]], -1)).reshape(*lead, 64)


def _iq2_bn_codes(b):
    lead = b.shape[:-1]
    qs = b[..., 0:16].astype(np.int64)
    q = np.stack([(qs >> (2 * s)) & 3 for s in range(4)], -2)  # [.., 4, 16]
    return (_fl(q) - F32(1.0)).reshape(*lead, 64)


# --------------------------------------------------------------------------- R4 group types
def _iq1_s_r4(g, k):
    """g: uint8 [G, 8 + nb * 24] (four rows) -> float32 [G, 4, k]."""
    lead = g.shape[0]
    nb = k // 32
    d = _f16(g[:, 0:8].reshape(lead, 4, 2))  # [G, 4]
    blk = g[:, 8:].reshape(lead, nb, 24)
    qs = blk[..., 0:16].astype(np.int64).reshape(lead, nb, 4, 4)  # [G, nb, i, k]
    qh = _u16(blk[..., 16:24])  # [G, nb, 4 (k)]
    shift = np.where(qh & 0x8000, -IQ1S_DELTA, IQ1S_DELTA).astype(F32)
    dl = d[:, None, :] * _fl(2 * ((qh >> 12) & 7) + 1)  # [G, nb, k]
    qsk = np.swapaxes(qs, -1, -2)  # [G, nb, k, i]
    idx = qsk | (((qh[..., None] >> (3 * np.arange(4))) & 7) << 8)
    grid = _grid("IQ1_S")[idx]  # [G, nb, k, i, 8]
    v = dl[..., None, None] * (grid + shift[..., None, None])
    return np.moveaxis(v.astype(F32).reshape(lead, nb, 4, 32), 2, 1).reshape(lead, 4, k)


def _iq1_m_r4(g, k):
    lead = g.shape[0]
    nb = k // 32
    d = _f16(g[:, 0:8].reshape(lead, 4, 2))
    blk = g[:, 8:].reshape(lead, nb, 28)
    qs = blk[..., 0:16].astype(np.int64).reshape(lead, nb, 2, 2, 4)  # [.., i, half, k]
    qh = blk[..., 16:24].astype(np.int64).reshape(lead, nb, 2, 4)  # [.., i, k]
    sc = blk[..., 24:28].astype(np.int64)  # [.., k]
    dl = np.stack([d[:, None, :] * _fl(sc & 15), d[:, None, :] * _fl(sc >> 4)], -2)
    i1 = qs[..., 0, :] | ((qh & 0x07) << 8)  # [.., i, k]
    i2 = qs[..., 1, :] | ((qh & 0x70) << 4)
    dl1 = np.where(qh & 0x08, -IQ1S_DELTA, IQ1S_DELTA).astype(F32)
    dl2 = np.where(qh & 0x80, -IQ1S_DELTA, IQ1S_DELTA).astype(F32)
    g1 = _grid("IQ1_S")[i1] + dl1[..., None]  # [.., i, k, 8]
    g2 = _grid("IQ1_S")[i2] + dl2[..., None]
    v = dl[..., None] * np.concatenate([g1, g2], -1)  # [.., i, k, 16]
    v = np.moveaxis(v.astype(F32), -2, 2)  # [G, nb, k, i, 16]
    return np.moveaxis(v.reshape(lead, nb, 4, 32), 2, 1).reshape(lead, 4, k)


# --------------------------------------------------------------------------- dispatch
_BLOCK_FN: Dict[int, Callable] = {
    2: _q4_0, 3: _q4_1, 6: _q5_0, 7: _q5_1, 8: _q8_0, 133: _q6_0,
    10: _q2_k, 11: _q3_k, 12: _q4_k, 13: _q5_k, 14: _q6_k,
    16: _iq2_xxs, 17: _iq2_xs, 18: _iq3_xxs, 19: _iq1_s, 20: _iq4_nl,
    21: _iq3_s, 22: _iq2_s, 23: _iq4_xs, 29: _iq1_m,
    137: _iq2_k, 138: _iq3_k, 139: _iq4_k, 140: _iq5_k, 141: _iq6_k,
}  # fmt: skip
# types whose row header is a scale for the whole row: (header decoder, block fn)
_ROW_FN: Dict[int, Callable] = {
    144: _iq4_ks, 145: _iq2_ks, 146: _iq4_kss, 152: _iq5_ks, 156: _iq3_ks, 157: _iq2_kl,
    153: _iq2_kt, 154: _iq3_kt, 155: _iq4_kt, 158: _iq1_kt,
}  # fmt: skip


def _as_rows(t: QType, data: np.ndarray, k: int, nrows: Optional[int]) -> np.ndarray:
    rb = t.row_bytes(k)
    a = np.ascontiguousarray(np.asarray(data, dtype=np.uint8)).reshape(-1)
    if nrows is None:
        nrows = a.size // rb
    if a.size != nrows * rb:
        raise ValueError(f"{t.name}: {a.size} bytes for {nrows} rows of {rb}")
    if nrows % t.rows:
        raise ValueError(f"{t.name}: rows come in groups of {t.rows}, got {nrows}")
    return a.reshape(nrows, rb)


def dequantize(
    type_id: int,
    data: np.ndarray,
    k: int,
    nrows: Optional[int] = None,
    *,
    fused: Optional[bool] = None,
    bn_unscaled: bool = False,
) -> np.ndarray:
    """Packed rows (uint8, flat or [nrows, row_bytes]) -> float32 [nrows, k]."""
    t = TYPES[int(type_id)]
    rows = _as_rows(t, data, k, nrows)
    n = rows.shape[0]
    fused = t.fused if fused is None else fused
    if t.rows == 4:
        groups = rows.reshape(n // 4, 4 * rows.shape[1])
        fn = _iq1_s_r4 if t.type_id == 219 else _iq1_m_r4
        return fn(groups, k).reshape(n, k)
    nb = k // t.block
    hdr = rows[:, : t.header]
    blk = rows[:, t.header :].reshape(n, nb, t.block_bytes)
    if t.type_id in (134, 135):
        codes = (_iq1_bn_codes if t.type_id == 134 else _iq2_bn_codes)(blk)
        codes = codes.reshape(n, k)
        if bn_unscaled:
            return codes
        d = _f16(hdr) if t.type_id == 134 else _f32(hdr)
        return (d[:, None] * codes).astype(F32)
    if t.type_id == 151:  # Q8_KV: float d, 4 more header bytes, k int8
        return (_f32(hdr[:, 0:4])[:, None] * _fl(_i8(rows[:, 8:]))).astype(F32)
    if t.type_id in _ROW_FN:
        d = (_f32(hdr) if t.header == 4 else _f16(hdr))[:, None, None]
        y = _ROW_FN[t.type_id](blk, d, fused)
    else:
        y = _BLOCK_FN[t.type_id](blk, fused)
    return np.ascontiguousarray(y, dtype=F32).reshape(n, k)


# --------------------------------------------------------------------------- random valid rows
def _put_f16(dst: np.ndarray, vals: np.ndarray) -> None:
    dst[...] = vals.astype("<f2").reshape(*vals.shape, 1).view(np.uint8)


def _put_f32(dst: np.ndarray, vals: np.ndarray) -> None:
    dst[...] = vals.astype("<f4").reshape(*vals.shape, 1).view(np.uint8)


# offsets of the half-precision scale fields inside a block
_F16_FIELDS = {
    2: (0,), 3: (0, 2), 6: (0,), 7: (0, 2), 8: (0,), 133: (0,),
    10: (80, 82), 11: (108,), 12: (0, 2), 13: (0, 2), 14: (208,),
    16: (0,), 17: (0,), 18: (0,), 19: (0,), 20: (0,), 21: (0,), 22: (0,), 23: (0,),
    137: (0,), 138: (0,), 139: (0,), 140: (0,), 141: (0,),
}  # fmt: skip


def _scales(rng: np.random.Generator, shape, lo: float, hi: float) -> np.ndarray:
    """Finite scales with random sign, log-uniform in [lo, hi)."""
    mag = np.exp(rng.uniform(np.log(lo), np.log(hi), size=shape))
    return (mag * rng.choice([-1.0, 1.0], size=shape)).astype(F32)


def random_rows(
    type_id: int, nrows: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Valid packed rows uint8 [nrows, row_bytes]: random bytes, finite scales."""
    t = TYPES[int(type_id)]
    rb = t.row_bytes(k)
    if nrows % t.rows:
        raise ValueError(f"{t.name}: rows come in groups of {t.rows}")
    a = rng.integers(0, 256, size=(nrows, rb), dtype=np.uint8)
    nb = k // t.block
    if t.rows == 4:
        g = a.reshape(nrows // 4, 4 * rb)
        d = _scales(rng, (nrows // 4, 4), 1e-3, 1e-2)
        for i in range(4):
            _put_f16(g[:, 2 * i : 2 * i + 2], d[:, i])
        return a
    if t.header == 2:
        _put_f16(a[:, 0:2], _scales(rng, (nrows,), 1e-3, 1e-2))
    elif t.header == 4:
        _put_f32(a[:, 0:4], _scales(rng, (nrows,), 1e-4, 1e-2))
    elif t.header == 8:
        _put_f32(a[:, 0:4], _scales(rng, (nrows,), 1e-4, 1e-2))
    blk = a[:, t.header :].reshape(nrows, nb, t.block_bytes)
    for off in _F16_FIELDS.get(t.type_id, ()):
        _put_f16(blk[..., off : off + 2], _scales(rng, (nrows, nb), 1e-3, 1e-2))
    if t.type_id == 29:  # IQ1_M: the half scale is the top nibble of 4 u16 scale words
        h = (
            _scales(rng, (nrows, nb), 1e-3, 1e-2)
            .astype("<f2")
            .view("<u2")
            .astype(np.int64)
        )
        sc = _u16(blk[..., 48:56])
        for i in range(4):
            sc[..., i] = (sc[..., i] & 0x0FFF) | (((h >> (4 * i)) & 0xF) << 12)
        blk[..., 48:56] = sc.astype("<u2").view(np.uint8).reshape(nrows, nb, 8)
    return a


def nmse(y: np.ndarray, ref: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    ref = np.asarray(ref, np.float64)
    den = float((ref**2).sum())
    return float(((y - ref) ** 2).sum() / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- libggml (tests)
_LIB_NAMES = {f"q{i}_k": f"q{i}_K" for i in range(2, 7)} | {"q8_kv": "q8_KV"}


class LibGGML:
    """The libggml calls the tests compare against, through ctypes.

    ``dequant`` calls the type's CPU ``dequantize_row_<name>`` exactly as ggml
    does for one row (one group of four rows for the R4 types); for the BN
    types it is given the pointer past the row header and returns the codes.
    """

    def __init__(self, path: Optional[str] = None):
        p = path or os.environ.get(LIB_ENV)
        if not p or not os.path.isfile(p):
            raise FileNotFoundError(f"set {LIB_ENV} to ik_llama.cpp's libggml.so")
        g = ctypes.CDLL(p)

        class InitParams(ctypes.Structure):
            _fields_ = [
                ("mem_size", ctypes.c_size_t),
                ("mem_buffer", ctypes.c_void_p),
                ("no_alloc", ctypes.c_bool),
            ]

        g.ggml_init.restype = ctypes.c_void_p
        g.ggml_init.argtypes = [InitParams]
        g.ggml_free.argtypes = [ctypes.c_void_p]
        # ggml_init fills the fp16 table the CPU (de)quantisers read
        g.ggml_free(g.ggml_init(InitParams(1 << 20, None, False)))
        g.ggml_row_size.restype = ctypes.c_size_t
        g.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        g.ggml_quantize_chunk.restype = ctypes.c_size_t
        g.ggml_quantize_chunk.argtypes = (
            [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            + [ctypes.c_int64] * 3
            + [ctypes.c_void_p] * 2
        )
        self.g = g
        self._fns: Dict[int, Callable] = {}

    def row_size(self, type_id: int, k: int) -> int:
        return int(self.g.ggml_row_size(int(type_id), int(k)))

    def _fn(self, type_id: int):
        if type_id not in self._fns:
            name = TYPES[type_id].name.lower()
            name = _LIB_NAMES.get(name, name)
            f = getattr(self.g, "dequantize_row_" + name)
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            f.restype = None
            self._fns[type_id] = f
        return self._fns[type_id]

    def dequant(self, type_id: int, data: np.ndarray, k: int) -> np.ndarray:
        t = TYPES[int(type_id)]
        rows = _as_rows(t, data, k, None)
        n, rb = rows.shape
        if rb != self.row_size(t.type_id, k):
            raise ValueError(
                f"{t.name}: row size {rb} != libggml {self.row_size(t.type_id, k)}"
            )
        y = np.empty((n, k), F32)
        f = self._fn(t.type_id)
        base, out = rows.ctypes.data, y.ctypes.data
        skip = t.header if t.type_id in (134, 135) else 0
        step = t.rows
        for r in range(0, n, step):
            f(base + r * rb + skip, out + r * k * 4, k * step)
        return y

    def quantize(self, type_id: int, x: np.ndarray, imatrix=None) -> np.ndarray:
        """float32 [n, k] -> uint8 [n, row_bytes] as ggml_quantize_chunk writes it."""
        x = np.ascontiguousarray(x, F32)
        n, k = x.shape
        rb = self.row_size(type_id, k)
        out = np.zeros((n, rb), np.uint8)
        im = None if imatrix is None else np.ascontiguousarray(imatrix, F32)
        got = self.g.ggml_quantize_chunk(
            int(type_id), x.ctypes.data, out.ctypes.data, 0, n, k,
            None if im is None else im.ctypes.data, None,
        )  # fmt: skip
        if got != n * rb:
            raise RuntimeError(
                f"ggml_quantize_chunk wrote {got} bytes, expected {n * rb}"
            )
        return out
