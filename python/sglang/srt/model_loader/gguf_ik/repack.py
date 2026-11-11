# SPDX-License-Identifier: Apache-2.0
"""Undo ik_llama.cpp's row-interleaved repacks (``_R4`` / ``_R8``) at load.

ik_llama.cpp makes a type such as IQ4_KS_R4 by quantising with the base type
(IQ4_KS) and then regrouping each run of 4 (or 8) rows so that a CPU kernel
reads the same block of 4 rows at once (iqk_quantize.cpp ``repack_<type>``).
Nothing is re-quantised: every bit of the base rows lands somewhere in the
regrouped rows. A GPU engine serves the base type, so the loader moves the
bits back.

How the inverse is obtained. Each ``_fwd_<type>`` below is a line-by-line
transcription of ik's ``repack_<type>`` for one group of rows holding one block
each. It runs once, on symbolic bytes (``_Sym``) whose bits carry the index of
the input bit they came from, so the result is the permutation itself: output
bit -> input bit. ``_Sym`` refuses any step that would merge two bits or drop
one, so a transcription that is not a pure regrouping fails loudly instead of
producing a wrong inverse. The one lookup ik applies (``scrambled_sign``, a
bijection of 7-bit sign codes, used by IQ2_XXS / IQ2_XS / IQ3_XXS) is carried
as a field and inverted with the inverse table.

Applying the inverse to a tensor is a handful of vectorised steps, one per
(shift, mask) with which bits travel between bytes: copy, mask, shift, OR. A
byte that moved whole is one such step; nothing works bit by bit except the
sign codes of the three types that scramble them. The unit tests prove each type three ways: inverse of the
forward on random bytes, against ik's own ``iqk_repack_tensor`` through
libggml when it is available, and against the base type's own shard of the
same tensor in a SPECIAL_SPLIT shard collection when one is mounted.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from sglang.srt.model_loader.gguf_ik.types import (
    BY_NAME,
    NOT_REPACKS,
    REPACKS,
    GGMLType,
    GGMLTypeError,
    by_id,
)

QK_K = 256


# --------------------------------------------------------------------- symbols
class _Sym:
    """An unsigned integer whose bits are labels (input bit indices) or zero."""

    __slots__ = ("bits",)

    def __init__(self, bits=None):
        self.bits: Dict[int, int] = bits or {}

    def __and__(self, mask: int) -> _Sym:
        return _Sym({p: v for p, v in self.bits.items() if (mask >> p) & 1})

    def shr(self, n: int) -> _Sym:
        """A right shift that may drop low bits (C's ``x >> n`` before a mask)."""
        return _Sym({p - n: v for p, v in self.bits.items() if p >= n})

    def __lshift__(self, n: int) -> _Sym:
        return _Sym({p + n: v for p, v in self.bits.items()})

    def __or__(self, other) -> _Sym:
        if isinstance(other, int):
            if other != 0:
                raise GGMLTypeError("repack transcription ORs a constant")
            return self
        if self.bits.keys() & other.bits.keys():
            raise GGMLTypeError("repack transcription merges two bits")
        return _Sym({**self.bits, **other.bits})

    __ror__ = __or__
    __add__ = __or__
    __radd__ = __or__

    def width(self) -> int:
        return max(self.bits) + 1 if self.bits else 0


def _u(b: List[_Sym], off: int, n: int) -> _Sym:
    """Little-endian integer of ``n`` bytes at ``off``."""
    v = _Sym()
    for i in range(n):
        v = v | (b[off + i] << (8 * i))
    return v


def _put(y: List[_Sym], off: int, v: _Sym, n: int = 1) -> None:
    """``y[off:off+n] |= v`` as an ``n``-byte little-endian store."""
    if v.width() > 8 * n:
        raise GGMLTypeError("repack transcription stores bits beyond the field")
    for i in range(n):
        y[off + i] = y[off + i] | (v.shr(8 * i) & 0xFF)


def _copy(y, yoff, x, xoff, n):
    for i in range(n):
        _put(y, yoff + i, x[xoff + i])


# fmt: off
_SCRAMBLED_SIGN = (
    0x00, 0x7F, 0x7E, 0x01, 0x7C, 0x03, 0x02, 0x7D, 0x78, 0x07, 0x06, 0x79, 0x04, 0x7B, 0x7A, 0x05,
    0x70, 0x0F, 0x0E, 0x71, 0x0C, 0x73, 0x72, 0x0D, 0x08, 0x77, 0x76, 0x09, 0x74, 0x0B, 0x0A, 0x75,
    0x60, 0x1F, 0x1E, 0x61, 0x1C, 0x63, 0x62, 0x1D, 0x18, 0x67, 0x66, 0x19, 0x64, 0x1B, 0x1A, 0x65,
    0x10, 0x6F, 0x6E, 0x11, 0x6C, 0x13, 0x12, 0x6D, 0x68, 0x17, 0x16, 0x69, 0x14, 0x6B, 0x6A, 0x15,
    0x40, 0x3F, 0x3E, 0x41, 0x3C, 0x43, 0x42, 0x3D, 0x38, 0x47, 0x46, 0x39, 0x44, 0x3B, 0x3A, 0x45,
    0x30, 0x4F, 0x4E, 0x31, 0x4C, 0x33, 0x32, 0x4D, 0x48, 0x37, 0x36, 0x49, 0x34, 0x4B, 0x4A, 0x35,
    0x20, 0x5F, 0x5E, 0x21, 0x5C, 0x23, 0x22, 0x5D, 0x58, 0x27, 0x26, 0x59, 0x24, 0x5B, 0x5A, 0x25,
    0x50, 0x2F, 0x2E, 0x51, 0x2C, 0x53, 0x52, 0x2D, 0x28, 0x57, 0x56, 0x29, 0x54, 0x2B, 0x2A, 0x55,
)  # iqk_quantize.cpp scrambled_sign k_table
# fmt: on


class _Trace:
    """Collects the 7-bit fields that went through ``scrambled_sign``."""

    def __init__(self):
        self.fields: List[Tuple[int, ...]] = []

    def scrambled_sign(self, s: _Sym) -> _Sym:
        if s.width() > 7 or len(s.bits) != 7:
            raise GGMLTypeError("scrambled_sign takes a full 7-bit field")
        self.fields.append(tuple(s.bits[p] for p in range(7)))
        return s


# --------------------------------------------------- transcriptions of ik's repack_<type>
# Each takes x: R lists of the base row's bytes (row header, then one block), and
# t: the _Trace, and returns the R rows' bytes as the repacked type stores them.


def _fwd_iq4_nl(x, t, R=4):  # repack_iq4_nl -> block_iq4_nl_r4 {d[4], qs[64]}
    y = [_Sym() for _ in range(R * 18)]
    qs = 2 * R
    for k in range(R):
        _copy(y, 2 * k, x[k], 0, 2)
        q = lambda j: x[k][2 + j]  # noqa: E731
        for i in range(4):
            _put(y, qs + 4 * k + i + 0, (q(i + 0) & 0xF) | ((q(i + 8) & 0x0F) << 4))
            _put(y, qs + 4 * k + i + 16, q(i + 0).shr(4) | (q(i + 8) & 0xF0))
            _put(y, qs + 4 * k + i + 32, (q(i + 4) & 0xF) | ((q(i + 12) & 0x0F) << 4))
            _put(y, qs + 4 * k + i + 48, q(i + 4).shr(4) | (q(i + 12) & 0xF0))
    return y


def _fwd_q4_0(x, t, R=8):  # repack_q4_0 -> block_iq4_nl_r8 {d[8], qs[128]}
    y = [_Sym() for _ in range(R * 18)]
    for k in range(R):
        _copy(y, 2 * k, x[k], 0, 2)
        for l in range(4):
            for i in range(4):
                _put(y, 16 + 32 * l + 4 * k + i, x[k][2 + 4 * l + i])
    return y


def _fwd_q8_0(x, t, R=8):  # repack_q8_0 -> block_q8_0_r8 {d[8], qs[256]}
    y = [_Sym() for _ in range(R * 34)]
    for k in range(R):
        _copy(y, 2 * k, x[k], 0, 2)
    for l in range(4):
        for k in range(R):
            for i in range(4):
                _put(y, 16 + 32 * l + 4 * k + i + 0, x[k][2 + i + 4 * l + 0])
                _put(y, 16 + 32 * l + 4 * k + i + 128, x[k][2 + i + 4 * l + 16])
    return y


def _convert_q5_0(xk):  # block_q5_0 {d, qh[4], qs[16]}
    qh = _u(xk, 2, 4)
    L = [None] * 32
    for j in range(16):
        xh_0 = (qh.shr(j) << 4) & 0x10
        xh_1 = qh.shr(j + 12) & 0x10
        L[j] = (xk[6 + j] & 0x0F) | xh_0
        L[j + 16] = xk[6 + j].shr(4) | xh_1
    return L


def _fwd_q5_0(x, t, R=4):  # repack_q5_0 -> block_q5_0_r4 {d[4], qh[16], qs[64]}
    y = [_Sym() for _ in range(R * 22)]
    QH, QS = 8, 24
    for k in range(R):
        _copy(y, 2 * k, x[k], 0, 2)
        L = _convert_q5_0(x[k])
        for l in range(4):
            l1 = 4 * (l // 2) + 16 * (l % 2)
            l2 = l1 + 8
            for i in range(4):
                _put(
                    y,
                    QS + 4 * k + i + 16 * l,
                    (L[i + l1] & 0xF) | ((L[i + l2] & 0xF) << 4),
                )
                _put(
                    y, QH + 4 * k + i, (L[i + l1].shr(4) | (L[i + l2].shr(4) << 4)) << l
                )
    return y


def _convert_q6_0(xk):  # block_q6_0 {d, qh[8], qs[16]}
    L = [None] * 32
    for j in range(16):
        h = xk[2 + j % 8].shr(4 * (j // 8))
        L[j] = (xk[10 + j] & 0x0F) | ((h << 4) & 0x30)
        L[j + 16] = xk[10 + j].shr(4) | ((h << 2) & 0x30)
    return L


def _fwd_q6_0(x, t, R=4):  # repack_q6_0 -> block_q6_0_r4 {d[4], qh[32], qs[64]}
    y = [_Sym() for _ in range(R * 26)]
    QH, QS = 8, 40
    for k in range(R):
        _copy(y, 2 * k, x[k], 0, 2)
        L = _convert_q6_0(x[k])
        for l in range(4):
            l1 = 4 * (l // 2) + 16 * (l % 2)
            l2 = l1 + 8
            for i in range(4):
                _put(
                    y,
                    QS + 4 * k + i + 16 * l,
                    (L[i + l1] & 0xF) | ((L[i + l2] & 0xF) << 4),
                )
                _put(
                    y,
                    QH + 4 * k + i + 16 * (l % 2),
                    (L[i + l1].shr(4) | (L[i + l2].shr(4) << 4)) << 2 * (l // 2),
                )
    return y


def _fwd_iq4_xs(
    x, t, R=8
):  # repack_iq4_xs -> block_iq4_xs_r8 {d[8], scales_h[16], scales_l[32], qs[1024]}
    y = [_Sym() for _ in range(R * 136)]
    SH, SL, QS = 16, 32, 64
    for k in range(R):
        _copy(y, 2 * k, x[k], 0, 2)
        scales_h = _u(x[k], 2, 2)
        q = lambda j: x[k][8 + j]  # noqa: E731
        for ib in range(8):
            sl = x[k][4 + ib // 2].shr(4 * (ib % 2)) & 0xF
            sh = scales_h.shr(2 * ib) & 3
            i = 8 * ib + k
            _put(y, SL + i % 32, sl << 4 * (i // 32))
            _put(y, SH + i % 16, sh << 2 * (i // 16))
            for i in range(4):
                _put(
                    y,
                    QS + 128 * ib + 4 * k + i + 0,
                    (q(16 * ib + i + 0) & 0xF) | ((q(16 * ib + i + 4) & 0xF) << 4),
                )
                _put(
                    y,
                    QS + 128 * ib + 4 * k + i + 32,
                    (q(16 * ib + i + 8) & 0xF) | ((q(16 * ib + i + 12) & 0xF) << 4),
                )
                _put(
                    y,
                    QS + 128 * ib + 4 * k + i + 64,
                    q(16 * ib + i + 0).shr(4) | (q(16 * ib + i + 4).shr(4) << 4),
                )
                _put(
                    y,
                    QS + 128 * ib + 4 * k + i + 96,
                    q(16 * ib + i + 8).shr(4) | (q(16 * ib + i + 12).shr(4) << 4),
                )
    return y


def _nibble_rows_qs(y, QS, k, q):
    """The qs regrouping shared by IQ4_KS / IQ4_K: 16 bytes of a block of 32 -> 4 x 4 bytes."""
    for ib in range(8):
        for i in range(4):
            _put(
                y,
                QS + 64 * ib + 4 * k + i + 0,
                (q(16 * ib + i + 0) & 0xF) | ((q(16 * ib + i + 8) & 0x0F) << 4),
            )
            _put(
                y,
                QS + 64 * ib + 4 * k + i + 16,
                q(16 * ib + i + 0).shr(4) | (q(16 * ib + i + 8) & 0xF0),
            )
            _put(
                y,
                QS + 64 * ib + 4 * k + i + 32,
                (q(16 * ib + i + 4) & 0xF) | ((q(16 * ib + i + 12) & 0x0F) << 4),
            )
            _put(
                y,
                QS + 64 * ib + 4 * k + i + 48,
                q(16 * ib + i + 4).shr(4) | (q(16 * ib + i + 12) & 0xF0),
            )


def _fwd_iq4_ks(
    x, t, R=4
):  # repack_iq4_ks: [float d[4]] block_iq4_ks_r4 {scales[32], qs[512]}
    y = [_Sym() for _ in range(R * 140)]
    SC, QS = 16, 48
    for k in range(R):
        _copy(y, 4 * k, x[k], 0, 4)
        for ib in range(8):
            _put(y, SC + 4 * ib + k, x[k][4 + ib])
        _nibble_rows_qs(y, QS, k, lambda j, k=k: x[k][12 + j])
    return y


def _convert_iq5_k(xk, qs0, qh0):
    L = [None] * 256
    shift, qs, qh, base = 0, qs0, qh0, 0
    for _ in range(4):
        for j in range(16):
            L[base + j + 0] = (xk[qs + j + 0] & 0xF) | (
                (xk[qh + j + 0].shr(shift) & 1) << 4
            )
            L[base + j + 16] = (xk[qs + j + 16] & 0xF) | (
                (xk[qh + j + 16].shr(shift) & 1) << 4
            )
            L[base + j + 32] = xk[qs + j + 0].shr(4) | (
                (xk[qh + j + 0].shr(shift) & 2) << 3
            )
            L[base + j + 48] = xk[qs + j + 16].shr(4) | (
                (xk[qh + j + 16].shr(shift) & 2) << 3
            )
        base += 64
        qs += 32
        shift += 2
        if shift == 8:
            qh += 32
            shift = 0
    return L


def _iq5_rows(y, QS, QH, k, L):
    for ib in range(8):
        for i in range(4):
            a = lambda o: L[32 * ib + i + o]  # noqa: E731
            _put(y, QS + 64 * ib + 4 * k + i + 0, (a(0) & 0xF) | ((a(8) & 0xF) << 4))
            _put(y, QS + 64 * ib + 4 * k + i + 16, (a(16) & 0xF) | ((a(24) & 0xF) << 4))
            _put(y, QS + 64 * ib + 4 * k + i + 32, (a(4) & 0xF) | ((a(12) & 0xF) << 4))
            _put(y, QS + 64 * ib + 4 * k + i + 48, (a(20) & 0xF) | ((a(28) & 0xF) << 4))
            _put(
                y,
                QH + 16 * ib + 4 * k + i,
                (a(0).shr(4) << 0)
                | (a(8).shr(4) << 1)
                | (a(16).shr(4) << 2)
                | (a(24).shr(4) << 3)
                | (a(4).shr(4) << 4)
                | (a(12).shr(4) << 5)
                | (a(20).shr(4) << 6)
                | (a(28).shr(4) << 7),
            )


def _fwd_iq5_ks(
    x, t, R=4
):  # repack_iq5_ks: [float d[4]] block_iq5_ks_r4 {scales[32], qs[512], qh[128]}
    y = [_Sym() for _ in range(R * 172)]
    SC, QS, QH = 16, 48, 560
    for k in range(R):
        _copy(y, 4 * k, x[k], 0, 4)
        L = _convert_iq5_k(x[k], 4 + 8, 4 + 8 + 128)
        for ib in range(8):
            _put(y, SC + 4 * ib + k, x[k][4 + ib])
        _iq5_rows(y, QS, QH, k, L)
    return y


def _k_extra_scales(y, EX, SH, SL, k, extra, sl_of, sh_of, sh_bits, sh_stride):
    """extra bits and 4+N-bit scales of the IQ*_K types, as their repacks store them."""
    for ib in range(8):
        _put(y, EX + k + 0, (extra & 1) << ib)
        _put(y, EX + k + 4, (extra.shr(1) & 1) << ib)
        extra = extra.shr(2)
        sl1, sl2 = sl_of(ib)
        sh1, sh2 = sh_of(ib)
        i = 8 * ib + k
        _put(y, SL + i % 32, sl1 << 4 * (i // 32))
        _put(y, SH + i % sh_stride, sh1 << sh_bits * (i // sh_stride))
        i += 4
        _put(y, SL + i % 32, sl2 << 4 * (i // 32))
        _put(y, SH + i % sh_stride, sh2 << sh_bits * (i // sh_stride))


def _fwd_iq4_k(
    x, t, R=4
):  # block_iq4_k {d, extra, scales_h[4], scales_l[8], qs[128]} -> _r4
    y = [_Sym() for _ in range(R * 144)]
    EX, SH, SL, QS = 8, 16, 32, 64
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        _k_extra_scales(
            y,
            EX,
            SH,
            SL,
            k,
            _u(xk, 2, 2),
            lambda ib: (xk[8 + ib] & 0xF, xk[8 + ib].shr(4)),
            lambda ib: (
                xk[4 + ib // 2].shr(4 * (ib % 2)) & 3,
                xk[4 + ib // 2].shr(4 * (ib % 2)).shr(2) & 3,
            ),
            2,
            16,
        )
    for k in range(R):
        _nibble_rows_qs(y, QS, k, lambda j, k=k: x[k][16 + j])
    return y


def _fwd_iq5_k(
    x, t, R=4
):  # block_iq5_k {d, extra, scales_h[4], scales_l[8], qs[128], qh[32]} -> _r4
    y = [_Sym() for _ in range(R * 176)]
    EX, SH, SL, QS, QH = 8, 16, 32, 64, 576
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        L = _convert_iq5_k(xk, 16, 144)
        _k_extra_scales(
            y,
            EX,
            SH,
            SL,
            k,
            _u(xk, 2, 2),
            lambda ib: (xk[8 + ib] & 0xF, xk[8 + ib].shr(4)),
            lambda ib: (
                xk[4 + ib // 2].shr(4 * (ib % 2)) & 3,
                xk[4 + ib // 2].shr(4 * (ib % 2)).shr(2) & 3,
            ),
            2,
            16,
        )
        _iq5_rows(y, QS, QH, k, L)
    return y


def _convert_iq3_k(xk, qs0, qh0):
    L = [None] * 256
    qs = qs0
    for ib32 in range(8):
        shift_l = 2 * (ib32 % 4)
        shift_h = ib32 % 8
        for j in range(16):
            L[32 * ib32 + j + 0] = (xk[qs + j + 0].shr(shift_l) & 3) | (
                (xk[qh0 + j + 0].shr(shift_h) & 1) << 2
            )
            L[32 * ib32 + j + 16] = (xk[qs + j + 16].shr(shift_l) & 3) | (
                (xk[qh0 + j + 16].shr(shift_h) & 1) << 2
            )
        if shift_l == 6:
            qs += 32
    return L


def _convert_iq2_k(xk, qs0):
    L = [None] * 256
    qs = qs0
    for ib32 in range(8):
        shift_l = 2 * (ib32 % 4)
        for j in range(16):
            L[32 * ib32 + j + 0] = xk[qs + j + 0].shr(shift_l) & 3
            L[32 * ib32 + j + 16] = xk[qs + j + 16].shr(shift_l) & 3
        if shift_l == 6:
            qs += 32
    return L


def _crumb_rows(y, QS, k, L, QH=None):
    """2-bit low quants (and the third bit) as the Q2_K / Q3_K / IQ2_K / IQ3_K repacks store them."""
    for ib in range(8):
        for i in range(4):
            a = lambda o: L[32 * ib + i + o]  # noqa: E731
            _put(
                y,
                QS + 32 * ib + 4 * k + i + 0,
                ((a(0) & 3) << 0)
                | ((a(4) & 3) << 2)
                | ((a(8) & 3) << 4)
                | ((a(12) & 3) << 6),
            )
            _put(
                y,
                QS + 32 * ib + 4 * k + i + 16,
                ((a(16) & 3) << 0)
                | ((a(20) & 3) << 2)
                | ((a(24) & 3) << 4)
                | ((a(28) & 3) << 6),
            )
            if QH is not None:
                _put(
                    y,
                    QH + 16 * ib + 4 * k + i,
                    (a(0).shr(2) << 0)
                    | (a(4).shr(2) << 1)
                    | (a(8).shr(2) << 2)
                    | (a(12).shr(2) << 3)
                    | (a(16).shr(2) << 4)
                    | (a(20).shr(2) << 5)
                    | (a(24).shr(2) << 6)
                    | (a(28).shr(2) << 7),
                )


def _fwd_iq3_k(
    x, t, R=4
):  # block_iq3_k {d, extra, scales_h, scales_l[8], qs[64], qh[32]} -> _r4
    y = [_Sym() for _ in range(R * 110)]
    EX, SH, SL, QS, QH = 8, 16, 24, 56, 312
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        sh = _u(xk, 4, 2)
        L = _convert_iq3_k(xk, 14, 78)
        _k_extra_scales(
            y,
            EX,
            SH,
            SL,
            k,
            _u(xk, 2, 2),
            lambda ib: (xk[6 + ib] & 0xF, xk[6 + ib].shr(4)),
            lambda ib: (sh.shr(2 * ib) & 1, sh.shr(2 * ib + 1) & 1),
            1,
            8,
        )
        _crumb_rows(y, QS, k, L, QH)
    return y


def _fwd_iq2_k(
    x, t, R=4
):  # block_iq2_k {d, extra, scales[8], qs[64]} -> {d[4], extra[8], scales[32], qs[256]}
    y = [_Sym() for _ in range(R * 76)]
    EX, SC, QS = 8, 16, 48
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        extra = _u(xk, 2, 2)
        L = _convert_iq2_k(xk, 12)
        for ib in range(8):
            _put(y, EX + k + 0, (extra & 1) << ib)
            _put(y, EX + k + 4, (extra.shr(1) & 1) << ib)
            extra = extra.shr(2)
            i = 8 * ib + k
            _put(y, SC + i % 32, (xk[4 + ib] & 0xF) << 4 * (i // 32))
            i += 4
            _put(y, SC + i % 32, xk[4 + ib].shr(4) << 4 * (i // 32))
        _crumb_rows(y, QS, k, L)
    return y


def _convert_q2_k(xk, qs0):
    L = [None] * 256
    qs = qs0
    for n in (0, 128):
        for j in range(32):
            L[n + j + 0] = xk[qs + j] & 3
            L[n + j + 32] = xk[qs + j].shr(2) & 3
            L[n + j + 64] = xk[qs + j].shr(4) & 3
            L[n + j + 96] = xk[qs + j].shr(6) & 3
        qs += 32
    return L


def _fwd_q2_k(
    x, t, R=4
):  # block_q2_K {scales[16], qs[64], d, dmin} -> {d[8], scales[64], qs[256]}
    y = [_Sym() for _ in range(R * 84)]
    SC, QS = 16, 80
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 80, 2)
        _copy(y, 2 * (k + 4), xk, 82, 2)
        for ib in range(16):
            _put(y, SC + 4 * ib + k, xk[ib])
        _crumb_rows(y, QS, k, _convert_q2_k(xk, 16))
    return y


def _convert_q3_k(xk):  # block_q3_K {hmask[32], qs[64], scales[12], d}
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    a0, a1, tmp = _u(xk, 96, 4), _u(xk, 100, 4), _u(xk, 104, 4)
    a2 = (a0.shr(4) & kmask2) | ((tmp.shr(4) & kmask1) << 4)
    a3 = (a1.shr(4) & kmask2) | ((tmp.shr(6) & kmask1) << 4)
    a0 = (a0 & kmask2) | ((tmp & kmask1) << 4)
    a1 = (a1 & kmask2) | ((tmp.shr(2) & kmask1) << 4)
    Ld = [a.shr(8 * b) & 0xFF for a in (a0, a1, a2, a3) for b in range(4)]
    L, q, mbit = [], 32, 0
    for _ in (0, 128):
        shift = 0
        for _j in range(4):
            for l in range(32):
                L.append((xk[q + l].shr(shift) & 3) | ((xk[l].shr(mbit) & 1) << 2))
            shift += 2
            mbit += 1
        q += 32
    return L, Ld


def _fwd_q3_k(
    x, t, R=4
):  # -> block_q3_k_r4 {d[4], scales_h[16], scales_l[32], qh[128], qs[256]}
    y = [_Sym() for _ in range(R * 110)]
    SH, SL, QH, QS = 8, 24, 56, 184
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 108, 2)
        L, Ld = _convert_q3_k(xk)
        for ib in range(8):
            s = 8 * ib + k
            _put(y, SL + s % 32, (Ld[2 * ib + 0] & 0xF) << 4 * (s // 32))
            _put(y, SH + s % 16, Ld[2 * ib + 0].shr(4) << 2 * (s // 16))
            s += 4
            _put(y, SL + s % 32, (Ld[2 * ib + 1] & 0xF) << 4 * (s // 32))
            _put(y, SH + s % 16, Ld[2 * ib + 1].shr(4) << 2 * (s // 16))
        _crumb_rows(y, QS, k, L, QH)
    return y


def _scale_min_k4(j, q):
    if j < 4:
        return q(j) & 63, q(j + 4) & 63
    return (
        (q(j + 4) & 0xF) | (q(j - 4).shr(6) << 4),
        q(j + 4).shr(4) | (q(j - 0).shr(6) << 4),
    )


def _q45_k_scales(y, SH, SL, k, xk):
    q = lambda j: xk[4 + j]  # noqa: E731  (scales[12] after d, dmin)
    for ib in range(8):
        ld, lm = _scale_min_k4(ib, q)
        _put(y, SL + 4 * ib + k, (ld & 0xF) | ((lm & 0xF) << 4))
        h = ld.shr(4) | (lm.shr(4) << 2)
        _put(y, SH + (4 * ib + k) % 16, h << 4 * ((4 * ib + k) // 16))


def _fwd_q4_k(
    x, t, R=4
):  # block_q4_K {d, dmin, scales[12], qs[128]} -> {d[8], scales_h[16], scales_l[32], qs[512]}
    y = [_Sym() for _ in range(R * 144)]
    SH, SL, QS = 16, 32, 64
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        _copy(y, 2 * (k + 4), xk, 2, 2)
        _q45_k_scales(y, SH, SL, k, xk)
        L = [None] * 256
        for ib64 in range(4):
            for j in range(32):
                L[64 * ib64 + j + 0] = xk[16 + 32 * ib64 + j] & 0xF
                L[64 * ib64 + j + 32] = xk[16 + 32 * ib64 + j].shr(4)
        for ib in range(8):
            for i in range(4):
                a = lambda o: L[32 * ib + i + o]  # noqa: E731
                _put(y, QS + 64 * ib + 4 * k + i + 0, a(0) | (a(8) << 4))
                _put(y, QS + 64 * ib + 4 * k + i + 16, a(16) | (a(24) << 4))
                _put(y, QS + 64 * ib + 4 * k + i + 32, a(4) | (a(12) << 4))
                _put(y, QS + 64 * ib + 4 * k + i + 48, a(20) | (a(28) << 4))
    return y


def _fwd_q5_k(
    x, t, R=4
):  # block_q5_K {d, dmin, scales[12], qh[32], qs[128]} -> {d[8], scales_h[16], scales_l[32], qh[128], qs[512]}
    y = [_Sym() for _ in range(R * 176)]
    SH, SL, QH, QS = 16, 32, 64, 192
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        _copy(y, 2 * (k + 4), xk, 2, 2)
        _q45_k_scales(y, SH, SL, k, xk)
        L = [None] * 256
        for ib64 in range(4):
            for j in range(32):
                L[64 * ib64 + j + 0] = (xk[48 + 32 * ib64 + j] & 0xF) | (
                    (xk[16 + j].shr(2 * ib64 + 0) & 1) << 4
                )
                L[64 * ib64 + j + 32] = xk[48 + 32 * ib64 + j].shr(4) | (
                    (xk[16 + j].shr(2 * ib64 + 1) & 1) << 4
                )
        for ib in range(8):
            for i in range(4):
                a = lambda o: L[32 * ib + i + o]  # noqa: E731
                _put(
                    y, QS + 64 * ib + 4 * k + i + 0, (a(0) & 0xF) | ((a(8) & 0xF) << 4)
                )
                _put(
                    y,
                    QS + 64 * ib + 4 * k + i + 16,
                    (a(16) & 0xF) | ((a(24) & 0xF) << 4),
                )
                _put(
                    y,
                    QS + 64 * ib + 4 * k + i + 32,
                    (a(4) & 0xF) | ((a(12) & 0xF) << 4),
                )
                _put(
                    y,
                    QS + 64 * ib + 4 * k + i + 48,
                    (a(20) & 0xF) | ((a(28) & 0xF) << 4),
                )
                _put(
                    y,
                    QH + 16 * ib + 4 * k + i,
                    (a(0).shr(4) << 0)
                    | (a(8).shr(4) << 1)
                    | (a(4).shr(4) << 2)
                    | (a(12).shr(4) << 3)
                    | (a(16).shr(4) << 4)
                    | (a(24).shr(4) << 5)
                    | (a(20).shr(4) << 6)
                    | (a(28).shr(4) << 7),
                )
    return y


def _fwd_q6_k(
    x, t, R=4
):  # block_q6_K {ql[128], qh[64], scales[16], d} -> {d[4], scales[64], qh[256], ql[512]}
    y = [_Sym() for _ in range(R * 210)]
    SC, QH, QL = 8, 72, 328
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 208, 2)
        L = [None] * 256
        ql, qh = 0, 128
        for n in (0, 128):
            for l in range(32):
                L[n + l + 0] = (xk[ql + l + 0] & 0xF) | ((xk[qh + l] & 3) << 4)
                L[n + l + 32] = (xk[ql + l + 32] & 0xF) | ((xk[qh + l].shr(2) & 3) << 4)
                L[n + l + 64] = xk[ql + l + 0].shr(4) | ((xk[qh + l].shr(4) & 3) << 4)
                L[n + l + 96] = xk[ql + l + 32].shr(4) | ((xk[qh + l].shr(6) & 3) << 4)
            ql += 64
            qh += 32
        for ib in range(8):
            _put(y, SC + 8 * ib + k + 0, xk[192 + 2 * ib + 0])
            _put(y, SC + 8 * ib + k + 4, xk[192 + 2 * ib + 1])
            for i in range(4):
                a = lambda o: L[32 * ib + i + o]  # noqa: E731
                _put(
                    y, QL + 64 * ib + 4 * k + i + 0, (a(0) & 0xF) | ((a(8) & 0xF) << 4)
                )
                _put(
                    y,
                    QL + 64 * ib + 4 * k + i + 16,
                    (a(16) & 0xF) | ((a(24) & 0xF) << 4),
                )
                _put(
                    y,
                    QL + 64 * ib + 4 * k + i + 32,
                    (a(4) & 0xF) | ((a(12) & 0xF) << 4),
                )
                _put(
                    y,
                    QL + 64 * ib + 4 * k + i + 48,
                    (a(20) & 0xF) | ((a(28) & 0xF) << 4),
                )
                _put(
                    y,
                    QH + 32 * ib + 4 * k + i + 0,
                    a(0).shr(4)
                    | (a(8).shr(4) << 2)
                    | (a(4).shr(4) << 4)
                    | (a(12).shr(4) << 6),
                )
                _put(
                    y,
                    QH + 32 * ib + 4 * k + i + 16,
                    a(16).shr(4)
                    | (a(24).shr(4) << 2)
                    | (a(20).shr(4) << 4)
                    | (a(28).shr(4) << 6),
                )
    return y


def _sas(t, aux, scale):
    """The four sign+scale bytes IQ2_XXS / IQ3_XXS repacks build from one 32-bit word."""
    out = _Sym()
    for m in range(4):
        s = (t.scrambled_sign(aux.shr(7 * m) & 127) << 1) | (scale.shr(m) & 1)
        out = out | (s << 8 * m)
    return out


def _fwd_iq2_xxs(
    x, t, R=4
):  # block_iq2_xxs {d, qs[32] u16} -> {d[4], sas[128], qs[128]}
    y = [_Sym() for _ in range(R * 66)]
    SAS, QS = 8, 136
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        for ib in range(8):
            for i in range(4):
                _put(y, QS + 16 * ib + 4 * k + i, xk[2 + 8 * ib + i])
            a1 = _u(xk, 2 + 8 * ib + 4, 4)
            _put(y, SAS + 4 * (4 * ib + k), _sas(t, a1, a1.shr(28)), 4)
    return y


def _fwd_iq2_xs(
    x, t, R=4
):  # block_iq2_xs {d, qs[32] u16, scales[8]} -> {d[4], qs[128] u16, scales[32]}
    y = [_Sym() for _ in range(R * 74)]
    QS, SC = 8, 264
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        for ib in range(8):
            for i in range(4):
                v = _u(xk, 2 + 2 * (4 * ib + i), 2)
                _put(
                    y,
                    QS + 2 * (16 * ib + 4 * k + i),
                    (v & 511) | (t.scrambled_sign(v.shr(9)) << 9),
                    2,
                )
            _put(y, SC + 4 * ib + k, xk[66 + ib])
    return y


def _fwd_iq2_s(
    x, t, R=4
):  # block_iq2_s {d, qs[64], qh[8], scales[8]} -> {d[4], qs[128], qh[32], signs[128], scales[32]}
    y = [_Sym() for _ in range(R * 82)]
    QS, QH, SG, SC = 8, 136, 168, 296
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        for ib in range(8):
            _put(y, SC + 4 * ib + k, xk[74 + ib])
            for i in range(4):
                _put(y, QS + 16 * ib + 4 * k + i, xk[2 + 4 * ib + i])
                _put(y, SG + 16 * ib + 4 * k + i, xk[2 + 32 + 4 * ib + i])
            _put(y, QH + 4 * ib + k, xk[66 + ib])
    return y


def _fwd_iq3_xxs(x, t, R=4):  # block_iq3_xxs {d, qs[96]} -> {d[4], sas[128], qs[256]}
    y = [_Sym() for _ in range(R * 98)]
    SAS, QS = 8, 136
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        for ib in range(8):
            for i in range(8):
                _put(y, QS + 32 * ib + 8 * k + i, xk[2 + 8 * ib + i])
            a = _u(xk, 2 + 64 + 4 * ib, 4)
            _put(y, SAS + 4 * (4 * ib + k), _sas(t, a, a.shr(28)), 4)
    return y


def _fwd_iq3_s(
    x, t, R=4
):  # block_iq3_s {d, qs[64], qh[8], signs[32], scales[4]} -> {d[4], qs[256], qh[32], signs[128], scales[16]}
    y = [_Sym() for _ in range(R * 110)]
    QS, QH, SG, SC = 8, 264, 296, 424
    for k in range(R):
        xk = x[k]
        _copy(y, 2 * k, xk, 0, 2)
        for ib in range(4):
            j = 8 * ib + k
            _put(y, SC + (j + 0) % 16, (xk[106 + ib] & 0xF) << 4 * ((j + 0) // 16))
            _put(y, SC + (j + 4) % 16, xk[106 + ib].shr(4) << 4 * ((j + 4) // 16))
        for ib in range(8):
            _put(y, QH + 4 * ib + k, xk[66 + ib])
            for i in range(4):
                _put(y, QS + 32 * ib + k + 8 * i + 0, xk[2 + 8 * ib + i + 0])
                _put(y, QS + 32 * ib + k + 8 * i + 4, xk[2 + 8 * ib + i + 4])
            for i in range(4):
                s = lambda m: xk[74 + 4 * ib + m]  # noqa: E731
                _put(
                    y,
                    SG + 16 * ib + 4 * k + i,
                    ((s(0).shr(i) & 1) << 0)
                    | ((s(0).shr(4 + i) & 1) << 1)
                    | ((s(1).shr(i) & 1) << 2)
                    | ((s(1).shr(4 + i) & 1) << 3)
                    | ((s(2).shr(i) & 1) << 4)
                    | ((s(2).shr(4 + i) & 1) << 5)
                    | ((s(3).shr(i) & 1) << 6)
                    | ((s(3).shr(4 + i) & 1) << 7),
                )
    return y


def _fwd_iq2_bn(
    x, t, R=4
):  # repack_iq2_bn: [float d[4]] then 64 bytes for 4 rows x 64 values
    y = [_Sym() for _ in range(R * 20)]
    for k in range(R):
        _copy(y, 4 * k, x[k], 0, 4)
        q = lambda j, k=k: x[k][4 + j]  # noqa: E731
        for l in range(4):
            for i in range(4):
                _put(
                    y,
                    16 + 4 * k + i + 16 * l,
                    ((q(i + 0).shr(2 * l) & 3) << 0)
                    | ((q(i + 4).shr(2 * l) & 3) << 2)
                    | ((q(i + 8).shr(2 * l) & 3) << 4)
                    | ((q(i + 12).shr(2 * l) & 3) << 6),
                )
    return y


_FORWARD: Dict[str, Callable] = {
    "Q4_0_R8": _fwd_q4_0,
    "Q5_0_R4": _fwd_q5_0,
    "Q6_0_R4": _fwd_q6_0,
    "Q8_0_R8": _fwd_q8_0,
    "Q2_K_R4": _fwd_q2_k,
    "Q3_K_R4": _fwd_q3_k,
    "Q4_K_R4": _fwd_q4_k,
    "Q5_K_R4": _fwd_q5_k,
    "Q6_K_R4": _fwd_q6_k,
    "IQ2_XXS_R4": _fwd_iq2_xxs,
    "IQ2_XS_R4": _fwd_iq2_xs,
    "IQ2_S_R4": _fwd_iq2_s,
    "IQ3_XXS_R4": _fwd_iq3_xxs,
    "IQ3_S_R4": _fwd_iq3_s,
    "IQ4_NL_R4": _fwd_iq4_nl,
    "IQ4_XS_R8": _fwd_iq4_xs,
    "IQ2_K_R4": _fwd_iq2_k,
    "IQ3_K_R4": _fwd_iq3_k,
    "IQ4_K_R4": _fwd_iq4_k,
    "IQ5_K_R4": _fwd_iq5_k,
    "IQ4_KS_R4": _fwd_iq4_ks,
    "IQ5_KS_R4": _fwd_iq5_ks,
    "IQ2_BN_R4": _fwd_iq2_bn,
}


# ------------------------------------------------------------------ the plan
@dataclass(frozen=True, eq=False)
class _Gather:
    """dst[..., D] |= shift(src[..., S] & mask) for a few (shift, mask) groups.

    Every destination bit comes from one source bit. Bits that travel between the
    same two bytes with the same shift form one (destination, source, mask)
    entry; entries with the same shift and mask are one vectorised step. A byte
    that moved whole is the group (0, 0xFF).
    """

    nbytes: int
    steps: Tuple[Tuple[int, int, np.ndarray, np.ndarray], ...]  # (shift, mask, D, S)


def _gather(src_of_dst: np.ndarray) -> _Gather:
    """The gather for a bit map: destination bit -> source bit."""
    nbytes = src_of_dst.size // 8
    masks: Dict[Tuple[int, int, int], int] = {}
    for dbit, sbit in enumerate(src_of_dst.tolist()):
        key = (dbit // 8, sbit // 8, dbit % 8 - sbit % 8)
        masks[key] = masks.get(key, 0) | (1 << (sbit % 8))
    groups: Dict[Tuple[int, int], Tuple[list, list]] = {}
    for (d, sb, shift), m in masks.items():
        g = groups.setdefault((shift, m), ([], []))
        g[0].append(d)
        g[1].append(sb)
    steps = tuple(
        (shift, m, np.asarray(D, np.int64), np.asarray(S, np.int64))
        for (shift, m), (D, S) in sorted(groups.items(), key=lambda kv: -len(kv[1][0]))
    )
    return _Gather(nbytes, steps)


@dataclass(frozen=True, eq=False)
class RepackPlan:
    name: str
    base: GGMLType
    rows: int
    hdr: int  # row header bytes of the base type
    blk: int  # block bytes of the base type
    hdr_map: np.ndarray  # base header bit -> repacked header bit, [rows * hdr * 8]
    blk_map: np.ndarray  # base block bit -> repacked block bit, [rows * blk * 8]
    sign_fields: (
        np.ndarray
    )  # [F, 7] bit indices (block region, base layout) of scrambled sign codes

    @functools.cached_property
    def undo(self) -> Tuple[_Gather, _Gather]:
        return _gather(self.hdr_map), _gather(self.blk_map)

    @functools.cached_property
    def redo(self) -> Tuple[_Gather, _Gather]:
        return _gather(_invert(self.hdr_map)), _gather(_invert(self.blk_map))


def _invert(m: np.ndarray) -> np.ndarray:
    inv = np.empty_like(m)
    inv[m] = np.arange(m.size)
    return inv


@functools.lru_cache(maxsize=None)
def plan(name: str) -> RepackPlan:
    """The inverse of ik's repack for type ``name``, derived from its transcription."""
    if name in NOT_REPACKS:
        raise GGMLTypeError(f"{name} {NOT_REPACKS[name]}")
    if name not in REPACKS or name not in _FORWARD:
        raise GGMLTypeError(f"{name} is not a row-interleaved repack SGLang can undo")
    base_name, R = REPACKS[name]
    base = BY_NAME[base_name]
    hdr, blk = base.row_meta, base.type_size
    row = hdr + blk
    x = [
        [_Sym({b: (k * row + j) * 8 + b for b in range(8)}) for j in range(row)]
        for k in range(R)
    ]
    trace = _Trace()
    y = _FORWARD[name](x, trace, R)
    if len(y) != R * row:
        raise GGMLTypeError(
            f"{name}: the transcription writes {len(y)} bytes, not {R * row}"
        )
    n = R * row * 8
    src_of_out = np.full(n, -1, np.int64)
    for j, sym in enumerate(y):
        if sym.width() > 8:
            raise GGMLTypeError(f"{name}: byte {j} holds more than 8 bits")
        for pos, lab in sym.bits.items():
            src_of_out[8 * j + pos] = lab
    if (src_of_out < 0).any() or np.unique(src_of_out).size != n:
        raise GGMLTypeError(
            f"{name}: the transcription is not a permutation of the bits"
        )
    out_of_in = _invert(src_of_out)

    # Input bit -> (row k, byte in row); split the row header from the block.
    lab = np.arange(n)
    k, byte = lab // (8 * row), (lab // 8) % row
    in_hdr = byte < hdr
    if not np.array_equal(in_hdr, out_of_in < R * hdr * 8):
        raise GGMLTypeError(f"{name}: row headers and blocks are mixed by the repack")
    # base-region coordinates: header region [R, hdr], block region [R, blk]
    hdr_in = (k * hdr + byte) * 8 + lab % 8
    blk_in = (k * blk + byte - hdr) * 8 + lab % 8
    hdr_map = np.empty(R * hdr * 8, np.int64)
    hdr_map[hdr_in[in_hdr]] = out_of_in[in_hdr]
    blk_map = np.empty(R * blk * 8, np.int64)
    blk_map[blk_in[~in_hdr]] = out_of_in[~in_hdr] - R * hdr * 8
    fields = np.asarray(
        [[blk_in[b] for b in f] for f in trace.fields], np.int64
    ).reshape(-1, 7)
    return RepackPlan(name, base, R, hdr, blk, hdr_map, blk_map, fields)


# Groups are restored a few MB at a time, so the temporaries stay small.
_CHUNK_BYTES = 1 << 22

_INV_SCRAMBLED = np.zeros(128, np.uint8)
_INV_SCRAMBLED[np.asarray(_SCRAMBLED_SIGN)] = np.arange(128, dtype=np.uint8)


def _apply(src: np.ndarray, g: _Gather) -> np.ndarray:
    """src [..., M] uint8 -> [..., g.nbytes] uint8."""
    out = np.zeros(src.shape[:-1] + (g.nbytes,), np.uint8)
    for shift, mask, D, S in g.steps:
        v = src[..., S]
        if mask != 0xFF:
            v &= mask
        if shift > 0:
            v <<= shift
        elif shift < 0:
            v >>= -shift
        out[..., D] |= v
    return out


def _recode_signs(
    blocks: np.ndarray, fields: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Pass every 7-bit sign code at ``fields`` (base layout) through ``table``."""
    if fields.size == 0:
        return blocks
    bits = np.unpackbits(blocks, axis=-1, bitorder="little")
    code = (bits[..., fields] << np.arange(7, dtype=np.uint8)).sum(-1, dtype=np.uint8)
    bits[..., fields] = (table[code][..., None] >> np.arange(7, dtype=np.uint8)) & 1
    return np.packbits(bits, axis=-1, bitorder="little")


def _groups(t, data: np.ndarray):
    p = plan(t.name)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    shape = data.shape
    if len(shape) < 2:
        raise GGMLTypeError(f"{t.name}: expected [rows, row_bytes], got {shape}")
    nrows, row_bytes = int(np.prod(shape[:-1])), shape[-1]
    if (row_bytes - p.hdr) % p.blk or nrows % p.rows:
        raise GGMLTypeError(
            f"{t.name}: {nrows} rows of {row_bytes} bytes do not form whole groups "
            f"of {p.rows} rows of {p.base.name} blocks"
        )
    return p, data, shape, nrows // p.rows, (row_bytes - p.hdr) // p.blk, row_bytes


def undo_repack(tensor_type, data: np.ndarray) -> Tuple[GGMLType, np.ndarray]:
    """Rows of a ``_R4`` / ``_R8`` type -> the same rows in the base type's layout.

    ``data`` is uint8 ``[..., rows, row_bytes]`` (any leading dims, e.g. experts);
    the rows of each run of 4 (or 8) are restored together. Returns the base
    type and a new uint8 array of the same shape.
    """
    t = tensor_type if isinstance(tensor_type, GGMLType) else by_id(int(tensor_type))
    p, data, shape, G, nb, row_bytes = _groups(t, data)
    R = p.rows
    gh, gb = p.undo
    d = data.reshape(G, R * row_bytes)
    out = np.empty((G, R, row_bytes), np.uint8)
    step = max(1, _CHUNK_BYTES // (R * row_bytes))
    for g0 in range(0, G, step):
        c = d[g0 : g0 + step]
        n = c.shape[0]
        blk = _apply(c[:, R * p.hdr :].reshape(n, nb, R * p.blk), gb)
        blk = _recode_signs(blk, p.sign_fields, _INV_SCRAMBLED)
        out[g0 : g0 + n, :, p.hdr :] = (
            blk.reshape(n, nb, R, p.blk).transpose(0, 2, 1, 3).reshape(n, R, nb * p.blk)
        )
        if p.hdr:
            out[g0 : g0 + n, :, : p.hdr] = _apply(c[:, : R * p.hdr], gh).reshape(
                n, R, p.hdr
            )
    return p.base, out.reshape(shape)


def do_repack(tensor_type, data: np.ndarray) -> np.ndarray:
    """The forward direction (base rows -> ``_R4`` / ``_R8`` rows), as ik writes them."""
    t = tensor_type if isinstance(tensor_type, GGMLType) else by_id(int(tensor_type))
    p, data, shape, G, nb, row_bytes = _groups(t, data)
    R = p.rows
    gh, gb = p.redo
    d = data.reshape(G, R, row_bytes)
    blk = (
        d[:, :, p.hdr :]
        .reshape(G, R, nb, p.blk)
        .transpose(0, 2, 1, 3)
        .reshape(G, nb, R * p.blk)
    )
    blk = _recode_signs(blk, p.sign_fields, np.asarray(_SCRAMBLED_SIGN, np.uint8))
    blk = _apply(blk, gb).reshape(G, nb * R * p.blk)
    if p.hdr:
        hdr = _apply(d[:, :, : p.hdr].reshape(G, R * p.hdr), gh)
        blk = np.concatenate([hdr, blk], axis=-1)
    return blk.reshape(shape)


def repack_types() -> List[str]:
    return sorted(_FORWARD)
