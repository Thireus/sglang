# SPDX-License-Identifier: Apache-2.0
"""Zero signs of low-bit ggml quants: measure the offset, repair the sign bits.

The problem in one paragraph. A weight that is exactly 0 in the source has to
become *something* in a type that has no zero level (IQ2_S, IQ3_S, ...). The
ggml quantisers pick the sign with ``x >= 0``, so every exact zero comes out
POSITIVE. In a source with many exact zeros (weights dequantised from a 4-bit
float format such as MXFP4 or NVFP4: about 11 % of them; pruned weights) that
is a small positive bias on every row of every matrix. The matrix NMSE does not
see it, but a layer answers the mean of its input with a constant vector, and a
deep model amplifies it layer after layer.

The fix is a sign bit: each weight whose source is an exact zero gets a sign
that keeps the mean error at zero. When the source has signed zeros (MXFP4 and
NVFP4 codes 0 = +0 and 8 = -0, or IEEE -0.0 in a float tensor), that is the
source's own sign; when it has only +0, a balanced alternating rule (below).
Nothing else changes: grid indices, scales and every other bit stay as they
are, so the NMSE stays the same to the last digit.

This module has no dependency beyond the Python standard library and numpy.
ik_llama.cpp's ``libggml.so`` (a shared build of ggml) does the dequantisation
and, in the tests, the quantisation; it is found through an explicit path or
the ``IK_LIBGGML`` environment variable.

Definitions used everywhere below (x = source values, y = dequantised quant):

* offset  = mean(y - x) / mean(|x|)  (healthy: |offset| <= 0.005)
* nmse    = sum((y - x)^2) / sum(x^2)
* rows_pos = share of rows whose summed error sum_j (y - x) is positive
  (healthy: near 0.5; 1.0 means every row is biased the same way)
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import os
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LIB_ENV = "IK_LIBGGML"
MAX_OFFSET = 0.005
QK_K = 256

# ggml type ids (ik_llama.cpp's ggml.h) -> name. Only the names matter for the
# dequantiser lookup (dequantize_row_<name>) and for the reports.
TYPE_NAMES: Dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    29: "IQ1_M",
    30: "BF16",
    39: "MXFP4",
    133: "Q6_0",
    137: "IQ2_K",
    138: "IQ3_K",
    139: "IQ4_K",
    140: "IQ5_K",
    141: "IQ6_K",
    144: "IQ4_KS",
    145: "IQ2_KS",
    146: "IQ4_KSS",
    152: "IQ5_KS",
    153: "IQ2_KT",
    154: "IQ3_KT",
    155: "IQ4_KT",
    156: "IQ3_KS",
    157: "IQ2_KL",
    158: "IQ1_KT",
}
TYPE_IDS: Dict[str, int] = {v: k for k, v in TYPE_NAMES.items()}

# Types whose blocks carry one explicit sign bit per weight, at a fixed place:
# type -> (block bytes per 256 weights, offset of the 32 sign bytes in a block).
# Bit j of sign byte g is the sign of weight 8 * g + j of the block (1 = negative).
#   block_iq2_s: d (2), qs[64] = 32 grid bytes + 32 sign bytes, qh[8], scales[8]  -> 82, signs at 34
#   block_iq3_s: d (2), qs[64], qh[8], signs[32], scales[4]                       -> 110, signs at 74
SIGN_FIELD: Dict[int, Tuple[int, int]] = {22: (82, 34), 21: (110, 74)}

# Why the other types cannot be repaired by setting sign bits.
PARITY_SIGNED = (16, 17, 18)  # IQ2_XXS, IQ2_XS, IQ3_XXS
TRELLIS = (153, 154, 155, 158)  # IQ2_KT, IQ3_KT, IQ4_KT, IQ1_KT
REFUSAL = {
    "parity": "stores 7 signs per group of 8 weights; the 8th is implied by parity (an even number of "
    "negatives), so the sign of one zero cannot change without changing another weight. "
    "Requantise from a source whose zeros are given signs first, or pick another type.",
    "trellis": "has no per-weight sign field (values come from a trellis code); requantise instead.",
    "unknown": "has no sign field this tool knows. If its check passes there is nothing to repair.",
}

# Measured on DeepSeek-V4.1-Flash experts (MXFP4 source, 11.5 % exact zeros), libggml as built,
# no imatrix: the offset each type gets from the quantiser. The README carries the full table.
AFFECTED_AS_BUILT = ("IQ2_S", "IQ3_S", "IQ3_KT", "IQ2_XXS", "IQ2_XS", "IQ3_XXS")
CLEAN_AS_BUILT = ("IQ3_KS", "IQ3_K", "IQ2_KL", "IQ2_K", "IQ2_KT", "IQ1_S", "IQ2_KS")

E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)


class ZeroSignError(RuntimeError):
    """Raised when an input does not satisfy what the tool expects."""


def type_name(t: int) -> str:
    return TYPE_NAMES.get(int(t), f"type{int(t)}")


def repair_support(t: int) -> Tuple[bool, str]:
    """(can this type be repaired by setting sign bits, the reason when it cannot)."""
    t = int(t)
    if t in SIGN_FIELD:
        return True, ""
    if t in PARITY_SIGNED:
        return False, f"{type_name(t)} {REFUSAL['parity']}"
    if t in TRELLIS:
        return False, f"{type_name(t)} {REFUSAL['trellis']}"
    return False, f"{type_name(t)} {REFUSAL['unknown']}"


# ----------------------------------------------------------------------------------------------- libggml
def find_libggml(path: Optional[str] = None) -> str:
    """The libggml.so to use: ``path``, else $IK_LIBGGML. Raises ZeroSignError when neither exists."""
    p = path or os.environ.get(LIB_ENV)
    if not p:
        raise ZeroSignError(
            f"no libggml: pass --lib /path/to/libggml.so or set {LIB_ENV}"
        )
    if not os.path.isfile(p):
        raise ZeroSignError(f"libggml not found: {p}")
    return os.path.abspath(p)


class GGML:
    """The few libggml calls this tool needs, through ctypes."""

    def __init__(self, path: Optional[str] = None):
        self.path = find_libggml(path)
        g = ctypes.CDLL(self.path)

        class InitParams(ctypes.Structure):
            _fields_ = [
                ("mem_size", ctypes.c_size_t),
                ("mem_buffer", ctypes.c_void_p),
                ("no_alloc", ctypes.c_bool),
            ]

        g.ggml_init.restype = ctypes.c_void_p
        g.ggml_init.argtypes = [InitParams]
        g.ggml_free.argtypes = [ctypes.c_void_p]
        # ggml_init fills the fp16 table the CPU (de)quantisers read; without it they return zeros.
        g.ggml_free(g.ggml_init(InitParams(1 << 20, None, False)))
        g.ggml_row_size.restype = ctypes.c_size_t
        g.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        g.ggml_quantize_chunk.restype = ctypes.c_size_t
        g.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.g = g
        self._deq = {}

    def row_size(self, t: int, k: int) -> int:
        return int(self.g.ggml_row_size(int(t), int(k)))

    def row_header(self, t: int) -> int:
        """Bytes before the first block of a row (ik's per-row scales; 0 for IQ2_S / IQ3_S):
        row_size(256) = header + 1 block and row_size(512) = header + 2 blocks."""
        return 2 * self.row_size(t, QK_K) - self.row_size(t, 2 * QK_K)

    def _dequantizer(self, t: int):
        if t not in self._deq:
            base = type_name(t)
            f = None
            for cand in (base.lower(), base.lower().replace("_k", "_K"), base):
                f = getattr(self.g, "dequantize_row_" + cand, None)
                if f is not None:
                    break
            if f is None:
                raise ZeroSignError(f"libggml has no dequantize_row for {base}")
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            f.restype = None
            self._deq[t] = f
        return self._deq[t]

    def dequant(self, t: int, blob, nrows: int, k: int) -> np.ndarray:
        """blob: uint8 buffer of nrows quantised rows of k values -> float32 [nrows, k]."""
        rs = self.row_size(t, k)
        b = np.ascontiguousarray(
            np.frombuffer(blob, np.uint8) if not isinstance(blob, np.ndarray) else blob
        )
        if b.size != nrows * rs:
            raise ZeroSignError(
                f"{type_name(t)}: {b.size} bytes for {nrows} rows of {rs}"
            )
        y = np.empty((nrows, k), np.float32)
        f = self._dequantizer(t)
        if k % QK_K == 0 and self.row_header(t) == 0:
            f(
                b.ctypes.data, y.ctypes.data, nrows * k
            )  # blocks are contiguous across rows
        else:
            for r in range(nrows):
                f(b.ctypes.data + r * rs, y.ctypes.data + r * k * 4, k)
        return y

    def quantize(
        self, t: int, x: np.ndarray, imatrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """float32 [n, k] -> uint8 [n * row_size] as ggml_quantize_chunk writes it (imatrix: float32 [k] or None)."""
        x = np.ascontiguousarray(x, np.float32)
        n, k = x.shape
        rs = self.row_size(t, k)
        out = np.zeros(n * rs, np.uint8)
        im = None if imatrix is None else np.ascontiguousarray(imatrix, np.float32)
        got = self.g.ggml_quantize_chunk(
            int(t),
            x.ctypes.data,
            out.ctypes.data,
            0,
            n,
            k,
            None if im is None else im.ctypes.data,
            None,
        )
        if got != n * rs:
            raise ZeroSignError(
                f"ggml_quantize_chunk wrote {got} bytes, expected {n * rs}"
            )
        return out


# ----------------------------------------------------------------------------------------------- sources
def fp4_nibbles(packed: np.ndarray) -> np.ndarray:
    """uint8 [n, k/2], two E2M1 codes a byte, low nibble = even column (MXFP4 / NVFP4 safetensors) -> [n, k]."""
    n = packed.shape[0]
    nib = np.empty((n, packed.shape[1] * 2), np.uint8)
    nib[:, 0::2] = packed & 15
    nib[:, 1::2] = packed >> 4
    return nib


def mxfp4_dequant(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """MXFP4 safetensors: codes uint8 [n, k/2], E8M0 scales uint8 [n, k/32] -> float32 [n, k], exact, -0 kept."""
    if (scales == 255).any():
        raise ZeroSignError("E8M0 NaN scale (255) in a source tensor")
    v = E2M1[fp4_nibbles(packed)]
    sc = np.ldexp(np.float32(1), scales.astype(np.int32) - 127).astype(np.float32)
    n = packed.shape[0]
    return (v.reshape(n, -1, 32) * sc[:, :, None]).reshape(n, -1)


def e4m3_to_float(b: np.ndarray) -> np.ndarray:
    """FP8 E4M3 (fn: no infinities, 0x7f / 0xff NaN) bytes -> float32."""
    b = b.astype(np.int32)
    s = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    e = (b >> 3) & 15
    m = (b & 7).astype(np.float32)
    v = np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * np.ldexp(np.float32(1), e - 7))
    v = (s * v).astype(np.float32)
    v[(b & 0x7F) == 0x7F] = np.nan
    return v


def nvfp4_dequant(
    packed: np.ndarray, scales_e4m3: np.ndarray, global_scale: float = 1.0
) -> np.ndarray:
    """NVFP4: codes uint8 [n, k/2] (low nibble first), E4M3 scales uint8 [n, k/16], a float global scale."""
    v = E2M1[fp4_nibbles(packed)]
    sc = e4m3_to_float(scales_e4m3) * np.float32(global_scale)
    n = packed.shape[0]
    return (v.reshape(n, -1, 16) * sc[:, :, None]).reshape(n, -1).astype(np.float32)


def ggml_mxfp4_nibbles(blob: np.ndarray, nrows: int, k: int) -> np.ndarray:
    """ggml's block_mxfp4 (e8m0 1 byte + qs[16]; weight j < 16 = low nibble of qs[j], j >= 16 = high nibble
    of qs[j - 16]) -> E2M1 codes uint8 [nrows, k]."""
    blk = np.frombuffer(blob, np.uint8).reshape(nrows, k // 32, 17)[:, :, 1:]
    nib = np.empty((nrows, k // 32, 32), np.uint8)
    nib[:, :, :16] = blk & 15
    nib[:, :, 16:] = blk >> 4
    return nib.reshape(nrows, k)


def zero_masks_from_codes(nib: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """E2M1 codes -> (+0 mask, -0 mask): code 0 is +0, code 8 is -0."""
    return nib == 0, nib == 8


def zero_masks_from_float(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """float values -> (+0 mask, -0 mask) from the IEEE sign bit."""
    z = x == 0
    neg = np.signbit(x)
    return z & ~neg, z & neg


def bf16_to_float(b: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(b).view(np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


# ----------------------------------------------------------------------------------------------- the rule
RULES = ("auto", "source", "alternate")


def choose_rule(rule: str, pos0: np.ndarray, neg0: np.ndarray) -> str:
    """auto: the source's own sign when -0 makes up 25-75 % of the source's zeros (native MXFP4 / NVFP4
    checkpoints, float tensors dequantised from them), else the alternating rule (sources with +0 only:
    pruned weights, fp4 written by a quantiser that never emits -0, such as ggml's MXFP4)."""
    if rule not in RULES:
        raise ZeroSignError(f"rule {rule!r}: expected one of {RULES}")
    if rule != "auto":
        return rule
    nz = int(pos0.sum()) + int(neg0.sum())
    if nz == 0:
        return "source"
    return "source" if 0.25 <= int(neg0.sum()) / nz <= 0.75 else "alternate"


def target_negative(
    rule: str, pos0: np.ndarray, neg0: np.ndarray, row0: int = 0
) -> np.ndarray:
    """Bool [n, k]: True where a source zero must decode NEGATIVE.

    source:    exactly the -0 positions.
    alternate: along each row the zeros take turns, +, -, +, - ... and the first sign alternates from one row
               to the next (row index row0 + r), so rows with an odd number of zeros do not all lean one way.
               Deterministic: the same matrix always gets the same bits.
    """
    if rule == "source":
        return neg0.copy()
    if rule == "alternate":
        z = pos0 | neg0
        idx = np.cumsum(z, axis=1, dtype=np.int64) - 1
        rows = (np.arange(z.shape[0], dtype=np.int64) + row0)[:, None]
        return z & (((idx + rows) % 2) == 1)
    raise ZeroSignError(f"rule {rule!r} has no target")


def sign_view(
    blob: np.ndarray, t: int, nrows: int, row_bytes: int, header: int = 0
) -> np.ndarray:
    """A writable view of the sign bytes: uint8 [nrows, blocks, 32]."""
    bs, so = SIGN_FIELD[int(t)]
    nb = (row_bytes - header) // bs
    if header + nb * bs != row_bytes:
        raise ZeroSignError(
            f"{type_name(t)}: row of {row_bytes} bytes is not {header} + n x {bs}"
        )
    rows = blob.reshape(nrows, row_bytes)[:, header:]
    return rows.reshape(nrows, nb, bs)[:, :, so : so + 32]


def sign_field_mask(t: int, nrows: int, row_bytes: int, header: int = 0) -> np.ndarray:
    """Bool [nrows * row_bytes]: True on the bytes that hold sign bits."""
    m = np.zeros(nrows * row_bytes, bool)
    sign_view(m.view(np.uint8), t, nrows, row_bytes, header)[...] = True
    return m


def apply_zero_signs(
    blob: np.ndarray,
    t: int,
    nrows: int,
    k: int,
    zero: np.ndarray,
    neg: np.ndarray,
    row_bytes: int,
    header: int = 0,
) -> Tuple[int, int]:
    """Set, in place, the sign bit of every weight in ``zero``: negative where ``neg``, positive elsewhere.
    Every other bit is left as it was. Idempotent. Returns (bits set, bits cleared)."""
    if k % QK_K:
        raise ZeroSignError(f"k = {k} is not a multiple of {QK_K}")
    nb = k // QK_K
    zb = np.packbits(zero.reshape(nrows, nb, 32, 8), axis=-1, bitorder="little")[..., 0]
    nbits = np.packbits(
        (neg & zero).reshape(nrows, nb, 32, 8), axis=-1, bitorder="little"
    )[..., 0]
    s = sign_view(blob, t, nrows, row_bytes, header)
    old = s.copy()
    s[...] = (s & ~zb) | nbits
    set_ = int(np.unpackbits(s & ~old).sum())
    cleared = int(np.unpackbits(old & ~s).sum())
    return set_, cleared


# ----------------------------------------------------------------------------------------------- statistics
@dataclasses.dataclass
class Stats:
    """Running sums over one or more chunks of rows of one matrix."""

    n: int = 0
    rows: int = 0
    rows_pos: int = 0
    sum_d: float = 0.0
    sum_d2: float = 0.0
    sum_x2: float = 0.0
    sum_absx: float = 0.0
    zeros: int = 0
    neg0: int = 0
    zero_dec_pos: int = 0
    zero_dec_neg: int = 0
    zero_dec_exact: int = 0
    plus0_kept: int = 0
    minus0_kept: int = 0

    def add(
        self, x: np.ndarray, y: np.ndarray, pos0: np.ndarray, neg0: np.ndarray
    ) -> "Stats":
        d = y.astype(np.float64) - x.astype(np.float64)
        rs = d.sum(axis=1)
        z = pos0 | neg0
        yz = y[z]
        self.n += d.size
        self.rows += d.shape[0]
        self.rows_pos += int((rs > 0).sum())
        self.sum_d += float(rs.sum())
        self.sum_d2 += float((d * d).sum())
        self.sum_x2 += float((x.astype(np.float64) ** 2).sum())
        self.sum_absx += float(np.abs(x).sum(dtype=np.float64))
        self.zeros += int(z.sum())
        self.neg0 += int(neg0.sum())
        self.zero_dec_pos += int((yz > 0).sum())
        self.zero_dec_neg += int((yz < 0).sum())
        self.zero_dec_exact += int((yz == 0).sum())
        self.plus0_kept += int((y[pos0] > 0).sum())
        self.minus0_kept += int((y[neg0] < 0).sum())
        return self

    def summary(self, max_offset: float = MAX_OFFSET) -> dict:
        off = self.sum_d / self.sum_absx if self.sum_absx > 0 else 0.0
        z = max(self.zeros, 1)
        out = {
            "offset": off,
            "nmse": self.sum_d2 / self.sum_x2 if self.sum_x2 > 0 else 0.0,
            "rows_pos": self.rows_pos / max(self.rows, 1),
            "zero_share": self.zeros / max(self.n, 1),
            "neg0_of_zeros": self.neg0 / z,
            "zeros_decoded_positive": self.zero_dec_pos / z,
            "zeros_decoded_negative": self.zero_dec_neg / z,
            "zeros_decoded_exact": self.zero_dec_exact / z,
            "minus0_decoded_negative": self.minus0_kept / max(self.neg0, 1),
            "plus0_decoded_positive": self.plus0_kept / max(self.zeros - self.neg0, 1),
            "zeros": self.zeros,
            "weights": self.n,
            "rows": self.rows,
        }
        out["verdict"] = "PASS" if abs(off) <= max_offset else "OFFSET"
        return out


def matrix_stats(
    x: np.ndarray,
    y: np.ndarray,
    pos0: np.ndarray,
    neg0: np.ndarray,
    max_offset: float = MAX_OFFSET,
) -> dict:
    return Stats().add(x, y, pos0, neg0).summary(max_offset)


# ----------------------------------------------------------------------------------------------- small file helpers
def sha256_file(path: str, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_range(path: str, off: int, ln: int, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        while ln > 0:
            b = f.read(min(chunk, ln))
            if not b:
                raise ZeroSignError(f"short read in {path}")
            h.update(b)
            ln -= len(b)
    return h.hexdigest()


def pread(path: str, off: int, ln: int) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        out = bytearray()
        while len(out) < ln:
            b = os.pread(fd, ln - len(out), off + len(out))
            if not b:
                raise ZeroSignError(f"short read in {path} at {off}")
            out += b
        return bytes(out)
    finally:
        os.close(fd)


def pwrite_all(fd: int, data, off: int) -> None:
    mv = memoryview(data).cast("B")
    while mv:
        w = os.pwrite(fd, mv, off)
        mv, off = mv[w:], off + w


def write_new_file(path: str, data: bytes) -> None:
    """Write through <path>.partial and rename: a reader never sees half a file."""
    tmp = path + ".partial"
    if os.path.lexists(tmp):
        os.unlink(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        pwrite_all(fd, data, 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- safetensors
def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def safetensors_header(path: str) -> Tuple[int, dict]:
    """(header length n, header dict); tensor data of entry t lives at 8 + n + t['data_offsets'][0]."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ZeroSignError(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        return n, json.loads(f.read(n))


class SafetensorsDir:
    """Read-only access to the tensors of a sharded safetensors checkpoint (model.safetensors.index.json)."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        idx = os.path.join(self.root, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.weight_map = load_json(idx)["weight_map"]
        else:
            self.weight_map = {}
            for fn in sorted(os.listdir(self.root)):
                if fn.endswith(".safetensors"):
                    _, h = safetensors_header(os.path.join(self.root, fn))
                    for name in h:
                        if name != "__metadata__":
                            self.weight_map[name] = fn
        self._hdr = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def info(self, name: str) -> Tuple[str, int, int, dict]:
        """(file path, absolute offset, byte length, header entry)."""
        if name not in self.weight_map:
            raise ZeroSignError(f"{self.root}: no tensor {name}")
        fn = self.weight_map[name]
        if fn not in self._hdr:
            self._hdr[fn] = safetensors_header(os.path.join(self.root, fn))
        n, h = self._hdr[fn]
        t = h[name]
        a, b = t["data_offsets"]
        return os.path.join(self.root, fn), 8 + n + a, b - a, t

    def read(self, name: str) -> Tuple[np.ndarray, dict]:
        p, off, ln, t = self.info(name)
        return np.frombuffer(pread(p, off, ln), np.uint8), t


def read_source_matrix(
    src: SafetensorsDir, base: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """A source matrix named ``base`` (without .weight): (x float32 [n, k], +0 mask, -0 mask, format).

    MXFP4: ``base.weight`` uint8/int8 [n, k/2] + ``base.scale`` E8M0 [n, k/32].
    Float: ``base.weight`` BF16 / F16 / F32 [n, k] (signed zeros read from the IEEE sign bit).
    """
    raw, t = src.read(base + ".weight")
    dt, shape = t["dtype"], t["shape"]
    if dt in ("U8", "I8") and src.has(base + ".scale"):
        sraw, st = src.read(base + ".scale")
        n, half = shape
        codes = raw.reshape(n, half)
        sc = sraw.reshape(st["shape"])
        if list(st["shape"]) == [n, half * 2 // 32] and st["dtype"] in (
            "U8",
            "I8",
            "F8_E8M0",
        ):
            x = mxfp4_dequant(codes, sc)
            pos0, neg0 = zero_masks_from_codes(fp4_nibbles(codes))
            return x, pos0, neg0, "mxfp4"
        raise ZeroSignError(
            f"{base}: scale shape {st['shape']} / dtype {st['dtype']} is not MXFP4's"
        )
    if dt == "BF16":
        x = bf16_to_float(raw.view(np.uint16).reshape(shape))
    elif dt == "F16":
        x = raw.view(np.float16).reshape(shape).astype(np.float32)
    elif dt == "F32":
        x = raw.view(np.float32).reshape(shape).copy()
    else:
        raise ZeroSignError(
            f"{base}.weight: dtype {dt} is not a source this tool reads"
        )
    pos0, neg0 = zero_masks_from_float(x)
    return x, pos0, neg0, "float"


# ----------------------------------------------------------------------------------------------- GGUF
GGUF_MAGIC = b"GGUF"
_GGUF_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


@dataclasses.dataclass
class GGUFTensor:
    name: str
    ggml_type: int
    ne: Tuple[int, ...]  # ne[0] = values per row (contiguous), then rows
    offset: int  # absolute offset of the data in the file
    nbytes: int
    path: str

    @property
    def k(self) -> int:
        return self.ne[0]

    @property
    def nrows(self) -> int:
        r = 1
        for v in self.ne[1:]:
            r *= v
        return r


def _gguf_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _gguf_value(f, vt: int):
    if vt in _GGUF_SCALARS:
        fmt = _GGUF_SCALARS[vt]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vt == 8:
        return _gguf_str(f)
    if vt == 9:
        it, n = struct.unpack("<IQ", f.read(12))
        if it in _GGUF_SCALARS:  # numeric arrays: one read, kept only when short
            fmt = _GGUF_SCALARS[it]
            data = f.read(struct.calcsize(fmt) * n)
            return (
                list(struct.unpack("<" + fmt[1] * n, data))
                if n <= 4096
                else ("array", it, n)
            )
        items = [
            _gguf_value(f, it) for _ in range(n)
        ]  # strings (token lists) must be read to move on
        return items if n <= 4096 else ("array", it, n)
    raise ZeroSignError(f"GGUF value type {vt} unknown")


def gguf_row_bytes(t: int, k: int, ggml: Optional[GGML]) -> int:
    if t == 0:
        return 4 * k
    if t in (1, 30):
        return 2 * k
    if t == 39:
        return k // 32 * 17
    if ggml is None:
        raise ZeroSignError(f"{type_name(t)}: need libggml for its row size")
    return ggml.row_size(t, k)


def read_gguf(
    path: str, ggml: Optional[GGML] = None
) -> Tuple[dict, Dict[str, GGUFTensor]]:
    """Parse a GGUF v2 / v3 file: (key-value metadata, tensors by name). Only the header is read."""
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ZeroSignError(f"{path}: not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        if ver not in (2, 3):
            raise ZeroSignError(f"{path}: GGUF version {ver} not supported")
        n_t, n_kv = struct.unpack("<QQ", f.read(16))
        kv = {}
        for _ in range(n_kv):
            key = _gguf_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            kv[key] = _gguf_value(f, vt)
        infos = []
        for _ in range(n_t):
            name = _gguf_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            ne = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            t, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, t, tuple(int(v) for v in ne), off))
        align = int(kv.get("general.alignment", 32))
        pos = f.tell()
        data = (pos + align - 1) // align * align
    tensors = {}
    for name, t, ne, off in infos:
        rows = 1
        for v in ne[1:]:
            rows *= v
        tensors[name] = GGUFTensor(
            name, t, ne, data + off, gguf_row_bytes(t, ne[0], ggml) * rows, path
        )
    return kv, tensors


def write_gguf(
    path: str,
    tensors: Sequence[Tuple[str, int, Tuple[int, ...], bytes]],
    kv: Optional[dict] = None,
    alignment: int = 32,
) -> None:
    """A minimal GGUF v3 writer (string and uint32 metadata only), for tests and small samples.
    tensors: (name, ggml type, ne, raw bytes)."""
    kv = dict(kv or {})
    out = bytearray(GGUF_MAGIC + struct.pack("<IQQ", 3, len(tensors), len(kv)))

    def s(v: str):
        b = v.encode()
        out.extend(struct.pack("<Q", len(b)) + b)

    for key, v in kv.items():
        s(key)
        if isinstance(v, str):
            out.extend(struct.pack("<I", 8))
            s(v)
        elif isinstance(v, int):
            out.extend(struct.pack("<II", 4, v))
        else:
            raise ZeroSignError(f"write_gguf: metadata {key} must be str or int")
    off, offs = 0, []
    for name, t, ne, raw in tensors:
        offs.append(off)
        off += (len(raw) + alignment - 1) // alignment * alignment
    for (name, t, ne, raw), o in zip(tensors, offs):
        s(name)
        out.extend(
            struct.pack("<I", len(ne))
            + struct.pack("<" + "Q" * len(ne), *ne)
            + struct.pack("<IQ", t, o)
        )
    out.extend(b"\0" * ((-len(out)) % alignment))
    for (name, t, ne, raw), o in zip(tensors, offs):
        out.extend(raw)
        out.extend(b"\0" * ((-len(raw)) % alignment))
    write_new_file(path, bytes(out))


def gguf_tensor_map(
    paths: Iterable[str], ggml: Optional[GGML] = None
) -> Dict[str, GGUFTensor]:
    """All tensors of a set of GGUF files (the shards of one model), by name."""
    out: Dict[str, GGUFTensor] = {}
    for p in paths:
        _, ts = read_gguf(p, ggml)
        for name, t in ts.items():
            if name in out:
                raise ZeroSignError(
                    f"tensor {name} in two files: {out[name].path} and {p}"
                )
            out[name] = t
    return out


def read_gguf_source_rows(
    t: GGUFTensor, r0: int, r1: int, ggml: GGML
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Rows [r0, r1) of a source GGUF tensor: (x float32, +0 mask, -0 mask, format)."""
    k = t.k
    rb = gguf_row_bytes(t.ggml_type, k, ggml)
    raw = np.frombuffer(pread(t.path, t.offset + r0 * rb, (r1 - r0) * rb), np.uint8)
    n = r1 - r0
    if t.ggml_type == 0:
        x = raw.view(np.float32).reshape(n, k).copy()
    elif t.ggml_type == 1:
        x = raw.view(np.float16).reshape(n, k).astype(np.float32)
    elif t.ggml_type == 30:
        x = bf16_to_float(raw.view(np.uint16).reshape(n, k))
    elif t.ggml_type == 39:
        x = ggml.dequant(39, raw, n, k)
        pos0, neg0 = zero_masks_from_codes(ggml_mxfp4_nibbles(raw, n, k))
        return x, pos0, neg0, "mxfp4"
    else:
        raise ZeroSignError(
            f"{t.name}: source type {type_name(t.ggml_type)} is not F32 / F16 / BF16 / MXFP4"
        )
    pos0, neg0 = zero_masks_from_float(x)
    return x, pos0, neg0, "float"


# ----------------------------------------------------------------------------------------------- hot/cold checkpoints
HOTCOLD_MANIFEST = "hotcold.json"
COLD_PROJECTIONS = ("w1", "w3", "w2")
SOURCE_PATTERN = "layers.{L}.ffn.experts.{E}.{w}"


@dataclasses.dataclass
class ColdUnit:
    """One cold matrix: layer L, projection w, slot in the layer's cold list, expert id E."""

    L: int
    w: str
    slot: int
    E: int
    ggml_type: int
    nrows: int
    row_bytes: int
    path: str  # the cold file
    offset: int  # absolute offset of this slot's rows in that file
    source_name: str  # source tensor base name (without .weight)

    @property
    def key(self) -> str:
        return f"{self.L}.{self.w}.{self.E}"


class HotColdCheckpoint:
    """A hot/cold checkpoint (python/sglang/srt/model_loader/hotcold_manifest.py): hotcold.json lists, per
    layer, the hot and cold expert ids; the cold experts of layer L live in one safetensors file as
    ``layers.L.ffn.experts_cold.{w1,w3,w2}.qweight`` uint8 [C, N, row_bytes], slot-major."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        p = os.path.join(self.root, HOTCOLD_MANIFEST)
        if not os.path.exists(p):
            raise ZeroSignError(
                f"{self.root}: no {HOTCOLD_MANIFEST}: not a hot/cold checkpoint"
            )
        self.manifest_path = p
        self.manifest = load_json(p)
        self.layers = sorted(int(L) for L in self.manifest["layers"])
        self.types = {
            w: int(t) for w, t in self.manifest["cold_quant"]["ggml_type"].items()
        }

    def cold_file(self, L: int) -> str:
        return self.manifest["cold_files"][str(L)]["file"]

    def cold_ids(self, L: int) -> List[int]:
        return [int(e) for e in self.manifest["layers"][str(L)]["cold"]]

    @staticmethod
    def tensor_name(L: int, w: str) -> str:
        return f"layers.{L}.ffn.experts_cold.{w}.qweight"

    def tensors(
        self, L: int, path: Optional[str] = None
    ) -> Dict[str, Tuple[int, int, List[int]]]:
        """{projection: (absolute offset, byte length, shape [C, N, row_bytes])} of layer L's cold file."""
        path = path or os.path.join(self.root, self.cold_file(L))
        n, h = safetensors_header(path)
        meta = h.get("__metadata__", {}) or {}
        out = {}
        for w in COLD_PROJECTIONS:
            name = self.tensor_name(L, w)
            if name not in h:
                raise ZeroSignError(f"{path}: no tensor {name}")
            t = h[name]
            tag = meta.get(f"hotcold.ggml_type.{name}")
            if tag is not None and int(tag) != self.types[w]:
                raise ZeroSignError(
                    f"{name}: file says type {tag}, manifest {self.types[w]}"
                )
            if t["dtype"] != "U8" or len(t["shape"]) != 3:
                raise ZeroSignError(
                    f"{name}: dtype {t['dtype']} shape {t['shape']}, expected U8 [C, N, row_bytes]"
                )
            a, b = t["data_offsets"]
            out[w] = (8 + n + a, b - a, list(t["shape"]))
        return out

    def units(
        self,
        L: int,
        path: Optional[str] = None,
        slots: Optional[Sequence[int]] = None,
        source_pattern: str = SOURCE_PATTERN,
    ) -> List[ColdUnit]:
        path = path or os.path.join(self.root, self.cold_file(L))
        cold = self.cold_ids(L)
        out = []
        for w, (off, ln, (C, N, rb)) in self.tensors(L, path).items():
            if C != len(cold):
                raise ZeroSignError(
                    f"layer {L} {w}: {C} slots, manifest lists {len(cold)} cold experts"
                )
            for slot in range(C) if slots is None else slots:
                E = cold[slot]
                out.append(
                    ColdUnit(
                        L,
                        w,
                        slot,
                        E,
                        self.types[w],
                        N,
                        rb,
                        path,
                        off + slot * N * rb,
                        source_pattern.format(L=L, E=E, w=w),
                    )
                )
        return out


def pick_slots(C: int, sample: Optional[int]) -> Optional[List[int]]:
    """None (all slots) or ``sample`` slots spread evenly, first and last included; deterministic."""
    if sample is None or sample >= C:
        return None
    if sample <= 1:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, C - 1, sample)})


def _source_for_unit(src: SafetensorsDir, u: ColdUnit, ggml: GGML):
    x, pos0, neg0, fmt = read_source_matrix(src, u.source_name)
    n, k = x.shape
    if n != u.nrows or ggml.row_size(u.ggml_type, k) != u.row_bytes:
        raise ZeroSignError(
            f"{u.key}: source [{n}, {k}] does not match the cold rows [{u.nrows}, {u.row_bytes} B]"
        )
    return x, pos0, neg0, fmt, k


def check_unit(
    u: ColdUnit, src: SafetensorsDir, ggml: GGML, max_offset: float = MAX_OFFSET
) -> dict:
    x, pos0, neg0, fmt, k = _source_for_unit(src, u, ggml)
    blob = np.frombuffer(pread(u.path, u.offset, u.nrows * u.row_bytes), np.uint8)
    s = matrix_stats(
        x, ggml.dequant(u.ggml_type, blob, u.nrows, k), pos0, neg0, max_offset
    )
    return dict(
        key=u.key,
        L=u.L,
        w=u.w,
        slot=u.slot,
        E=u.E,
        type=type_name(u.ggml_type),
        source=fmt,
        **s,
    )


def repair_unit(
    u: ColdUnit,
    src: SafetensorsDir,
    ggml: GGML,
    rule: str,
    dst_fd: Optional[int],
    max_offset: float = MAX_OFFSET,
) -> dict:
    """Repair one cold matrix read from u.path and pwrite it at the same offset into dst_fd (None: dry)."""
    ok, why = repair_support(u.ggml_type)
    if not ok:
        raise ZeroSignError(f"{u.key}: {why}")
    x, pos0, neg0, fmt, k = _source_for_unit(src, u, ggml)
    used = choose_rule(rule, pos0, neg0)
    before_raw = np.frombuffer(pread(u.path, u.offset, u.nrows * u.row_bytes), np.uint8)
    blob = before_raw.copy()
    before = matrix_stats(
        x, ggml.dequant(u.ggml_type, before_raw, u.nrows, k), pos0, neg0, max_offset
    )
    hdr = ggml.row_header(u.ggml_type)
    zero = pos0 | neg0
    n_set, n_clr = apply_zero_signs(
        blob,
        u.ggml_type,
        u.nrows,
        k,
        zero,
        target_negative(used, pos0, neg0),
        u.row_bytes,
        hdr,
    )
    after = matrix_stats(
        x, ggml.dequant(u.ggml_type, blob, u.nrows, k), pos0, neg0, max_offset
    )
    outside = ~sign_field_mask(u.ggml_type, u.nrows, u.row_bytes, hdr)
    if not np.array_equal(blob[outside], before_raw[outside]):
        raise ZeroSignError(f"{u.key}: a byte outside the sign field changed")
    if after["nmse"] != before["nmse"] and abs(
        after["nmse"] - before["nmse"]
    ) > 1e-12 * max(before["nmse"], 1e-30):
        raise ZeroSignError(
            f"{u.key}: NMSE changed {before['nmse']!r} -> {after['nmse']!r}"
        )
    if dst_fd is not None:
        pwrite_all(dst_fd, blob, u.offset)
    return dict(
        key=u.key,
        L=u.L,
        w=u.w,
        slot=u.slot,
        E=u.E,
        type=type_name(u.ggml_type),
        source=fmt,
        rule=used,
        zeros=after["zeros"],
        bits_set=n_set,
        bits_cleared=n_clr,
        offset_before=before["offset"],
        offset_after=after["offset"],
        rows_pos_before=before["rows_pos"],
        rows_pos_after=after["rows_pos"],
        nmse=after["nmse"],
        verdict=after["verdict"],
    )
