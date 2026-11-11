"""Bit-exact proof of the ggml / ik_llama.cpp quant references.

The numpy references in sglang.test.kernels.ggml_quant_reference are compared,
bit for bit, with ik_llama.cpp's own CPU dequantisation (libggml through
ctypes) on three kinds of rows for every covered type:

* random valid rows (random bytes, finite scales), so that every code, sign,
  grid index and sub-scale occurs;
* rows written by ik's own quantiser from heavy-tailed data with exact zeros;
* real tensors from a GGUF quant collection (three tensors a type, leading
  rows only).

Environment: IK_LIBGGML = path of ik_llama.cpp's libggml.so (the libggml tests
skip without it); IK_QUANT_COLLECTION = a directory holding one
``*-<TYPE>-SPECIAL_SPLIT`` shard set per type (the shard tests skip without
it). The layout tests need neither.

    IK_LIBGGML=/path/to/libggml.so python -m pytest \
        test/registered/unit/layers/quantization/test_ggml_quant_reference.py -v
"""

import glob
import os
import struct
import unittest
from fractions import Fraction

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels import ggml_quant_reference as ref

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

LIB = os.environ.get(ref.LIB_ENV)
COLLECTION = os.environ.get("IK_QUANT_COLLECTION")
SHARD_BYTES = int(os.environ.get("IK_QUANT_SHARD_BYTES", str(20 << 20)))

# the row sizes the type definitions imply at K = 5120 (ggml_row_size)
ROW_BYTES_5120 = {
    "Q4_0": 2880, "Q4_1": 3200, "Q5_0": 3520, "Q5_1": 3840, "Q8_0": 5440,
    "Q6_0": 4160, "Q8_KV": 5128, "Q2_K": 1680, "Q3_K": 2200, "Q4_K": 2880,
    "Q5_K": 3520, "Q6_K": 4200, "IQ1_S": 1000, "IQ1_M": 1120, "IQ2_XXS": 1320,
    "IQ2_XS": 1480, "IQ2_S": 1640, "IQ3_XXS": 1960, "IQ3_S": 2200,
    "IQ4_NL": 2880, "IQ4_XS": 2720, "IQ2_K": 1520, "IQ3_K": 2200,
    "IQ4_K": 2880, "IQ5_K": 3520, "IQ6_K": 4240, "IQ2_KS": 1402,
    "IQ3_KS": 2042, "IQ4_KS": 2724, "IQ4_KSS": 2564, "IQ5_KS": 3364,
    "IQ2_KL": 1722, "IQ1_KT": 1124, "IQ2_KT": 1364, "IQ3_KT": 2004,
    "IQ4_KT": 2564, "IQ1_BN": 1042, "IQ2_BN": 1284, "IQ1_S_R4": 962,
    "IQ1_M_R4": 1122,
}  # fmt: skip


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a, np.float32)
    b = np.ascontiguousarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def first_mismatch(a: np.ndarray, b: np.ndarray) -> str:
    bad = np.argwhere(a.view(np.uint32) != b.view(np.uint32))
    if not len(bad):
        return "none"
    r, c = bad[0]
    return f"{len(bad)} values differ, first at row {r} col {c}: {a[r, c]!r} vs {b[r, c]!r}"


def round_f32(v: Fraction) -> np.float32:
    """Exact value -> nearest float32, ties to even."""
    r = np.float32(float(v))
    cands = [
        np.nextafter(r, np.float32(-np.inf)),
        r,
        np.nextafter(r, np.float32(np.inf)),
    ]
    best = min(
        cands,
        key=lambda c: (
            abs(Fraction(float(c)) - v),
            int(np.float32(c).view(np.uint32)) & 1,
        ),
    )
    return np.float32(best)


def heavy_tailed(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Student-t rows with per-row scales, 10 % exact zeros (half of them -0.0)."""
    x = rng.standard_t(3, size=(n, k)) * np.exp(rng.normal(-4.0, 0.5, size=(n, 1)))
    z = rng.random((n, k)) < 0.10
    x[z] = 0.0
    x[z & (rng.random((n, k)) < 0.5)] = -0.0
    return x.astype(np.float32)


class TestGgmlQuantReferenceLayout(unittest.TestCase):
    def test_row_bytes(self):
        self.assertEqual(set(ROW_BYTES_5120), {t.name for t in ref.TYPES.values()})
        for t in ref.TYPES.values():
            with self.subTest(t.name):
                self.assertEqual(t.row_bytes(5120), ROW_BYTES_5120[t.name])

    def test_random_rows_are_finite(self):
        rng = np.random.default_rng(0)
        for t in ref.TYPES.values():
            with self.subTest(t.name):
                y = ref.dequantize(
                    t.type_id, ref.random_rows(t.type_id, 8, 512, rng), 512
                )
                self.assertEqual(y.shape, (8, 512))
                self.assertTrue(np.isfinite(y).all())

    def test_bn_scale(self):
        rng = np.random.default_rng(1)
        for tid in (134, 135):
            rows = ref.random_rows(tid, 4, 256, rng)
            codes = ref.dequantize(tid, rows, 256, bn_unscaled=True)
            self.assertTrue(set(np.unique(codes)) <= {-1.0, 0.0, 1.0, 2.0})
            d = (
                rows[:, 0:2].copy().view("<f2")
                if tid == 134
                else rows[:, 0:4].copy().view("<f4")
            ).astype(np.float32)
            self.assertTrue(bits_equal(ref.dequantize(tid, rows, 256), d * codes))

    def test_fma_emulation(self):
        rng = np.random.default_rng(2)
        a, b, c = (rng.standard_normal(256).astype(np.float32) for _ in range(3))
        exact = [
            round_f32(Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z)))
            for x, y, z in zip(a, b, c)
        ]
        self.assertTrue(bits_equal(ref._fma32(a, b, c), np.array(exact, np.float32)))
        # a float64 sum that lands on a float32 tie: the exact value is just below
        # 1 + 3 * 2^-24, so it rounds to 1 + 2^-23, not to the even neighbour
        a1 = np.float32(1 + 2.0**-23)
        b1 = np.float32((1 - 2.0**-23) * 2.0**-24)
        t = ref._fma32(a1, b1, a1)
        self.assertEqual(float(t), 1 + 2.0**-23)
        self.assertEqual(
            float(
                round_f32(
                    Fraction(float(a1)) * Fraction(float(b1)) + Fraction(float(a1))
                )
            ),
            1 + 2.0**-23,
        )


class _LibMixin:
    @classmethod
    def setUpClass(cls):
        cls.lib = ref.LibGGML(LIB)

    def _check(self, tid, rows, k, label):
        t = ref.TYPES[tid]
        want = self.lib.dequant(tid, rows, k)
        if tid in (134, 135):
            got = ref.dequantize(tid, rows, k, bn_unscaled=True)
        else:
            got = ref.dequantize(tid, rows, k)
        self.assertTrue(
            bits_equal(got, want), f"{t.name} {label}: {first_mismatch(got, want)}"
        )
        if t.fused:
            alt = ref.dequantize(tid, rows, k, fused=False)
            print(
                f"{t.name} {label}: unfused multiply-add also bit-exact = "
                f"{bits_equal(alt, want)}"
            )


@unittest.skipUnless(LIB, f"set {ref.LIB_ENV} to ik_llama.cpp's libggml.so")
class TestGgmlQuantReferenceBitExact(_LibMixin, unittest.TestCase):
    def test_row_size_matches_libggml(self):
        for t in ref.TYPES.values():
            for k in (256, 2304, 5120):
                with self.subTest(t.name, k=k):
                    self.assertEqual(t.row_bytes(k), self.lib.row_size(t.type_id, k))

    def test_random_rows(self):
        rng = np.random.default_rng(22)
        for t in ref.TYPES.values():
            with self.subTest(t.name):
                k = 4096
                rows = ref.random_rows(t.type_id, 64, k, rng)
                self._check(t.type_id, rows, k, "random rows")

    def test_quantiser_rows(self):
        rng = np.random.default_rng(145)
        for t in ref.TYPES.values():
            with self.subTest(t.name):
                k = 2304
                x = heavy_tailed(rng, 8, k)
                rows = self.lib.quantize(t.type_id, x, np.ones(k, np.float32))
                self._check(t.type_id, rows, k, "quantiser rows")


# --------------------------------------------------------------------------- real shards
_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}  # fmt: skip


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(n))

    def string(self):
        (n,) = self.take("Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, vt):
        if vt == 8:
            return self.string()
        if vt == 9:
            et, n = self.take("IQ")
            return [self.value(et) for _ in range(n)]
        return self.take(_GGUF_SCALARS[vt])[0]


def gguf_tensors(path):
    """[(name, ggml type id, dims (ne0 first), absolute data offset)] of one GGUF file."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        _version, n_tensors, n_kv = r.take("IQQ")
        align = 32
        for _ in range(n_kv):
            key = r.string()
            (vt,) = r.take("I")
            v = r.value(vt)
            if key == "general.alignment":
                align = int(v)
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            (nd,) = r.take("I")
            dims = r.take("Q" * nd)
            tid, off = r.take("IQ")
            infos.append((name, tid, dims, off))
        start = (f.tell() + align - 1) // align * align
    return [(n, t, d, start + o) for n, t, d, o in infos]


def shard_set(name):
    hits = sorted(glob.glob(os.path.join(COLLECTION or "", f"*-{name}-SPECIAL_SPLIT")))
    return hits[0] if hits else None


def pick_tensors(set_dir, tid, count=3):
    """Up to ``count`` 2-D tensors of type ``tid``, from tensors.map when present."""
    shards = sorted(glob.glob(os.path.join(set_dir, "*.gguf")))
    found = []
    for path in shards:
        for name, t, dims, off in gguf_tensors(path):
            if t == tid and len(dims) >= 2:
                found.append((path, name, dims, off))
                if len(found) == count:
                    return found
    return found


@unittest.skipUnless(LIB and COLLECTION, "set IK_LIBGGML and IK_QUANT_COLLECTION")
class TestGgmlQuantReferenceShards(_LibMixin, unittest.TestCase):
    def test_real_tensors(self):
        budget = SHARD_BYTES // 3
        for t in ref.TYPES.values():
            set_dir = shard_set(t.name)
            if set_dir is None:
                print(f"{t.name}: no shard set")
                continue
            with self.subTest(t.name):
                tensors = pick_tensors(set_dir, t.type_id)
                if not tensors:
                    # e.g. a set named after one type that holds another
                    self.skipTest(f"{t.name}: no tensor of that type in {set_dir}")
                for path, name, dims, off in tensors:
                    k = int(dims[0])
                    rb = t.row_bytes(k)
                    n = int(np.prod(dims[1:]))
                    n = max(t.rows, min(n, budget // rb) // t.rows * t.rows)
                    with open(path, "rb") as f:
                        f.seek(off)
                        rows = np.frombuffer(f.read(n * rb), np.uint8).reshape(n, rb)
                    self._check(
                        t.type_id,
                        rows,
                        k,
                        f"{name} ({n} of {int(np.prod(dims[1:]))} rows)",
                    )


if __name__ == "__main__":
    unittest.main()
