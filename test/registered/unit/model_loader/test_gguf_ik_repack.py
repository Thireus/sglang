"""Undoing ik_llama.cpp's _R4 / _R8 row-interleaved repacks (gguf_ik/repack.py).

Without libggml: every plan derives as a pure bit permutation, and the inverse
undoes the forward on random bytes. With IK_LIBGGML set (ik's libggml.so): for
every repack type, rows quantised with the base type and repacked by ik's own
``iqk_repack_tensor`` are restored bit for bit, and ik's direct quantiser of the
repacked type writes exactly the base quantiser's rows repacked.
"""

import ctypes
import os
import unittest

import numpy as np

from sglang.srt.model_loader.gguf_ik import repack as RP
from sglang.srt.model_loader.gguf_ik import types as T
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# One row length that is a whole number of blocks for every base type.
NE0 = 512


class TestPlans(CustomTestCase):
    def test_every_listed_repack_has_a_plan(self):
        self.assertEqual(sorted(T.REPACKS), RP.repack_types())
        for name in T.REPACKS:
            p = RP.plan(name)
            self.assertEqual(p.base.name, T.REPACKS[name][0])
            self.assertEqual(p.rows, T.REPACKS[name][1])

    def test_sign_codes_only_where_ik_scrambles_them(self):
        with_signs = {n for n in T.REPACKS if RP.plan(n).sign_fields.size}
        self.assertEqual(with_signs, {"IQ2_XXS_R4", "IQ2_XS_R4", "IQ3_XXS_R4"})
        self.assertEqual(sorted(RP._SCRAMBLED_SIGN), list(range(128)))

    def test_not_repacks_are_refused(self):
        for name in T.NOT_REPACKS:
            with self.assertRaisesRegex(T.GGMLTypeError, name):
                RP.plan(name)

    def test_round_trip_random_bytes(self):
        rng = np.random.default_rng(1)
        for name in T.REPACKS:
            p = RP.plan(name)
            t = T.BY_NAME[name]
            x = rng.integers(
                0, 256, (3, 2 * p.rows, T.row_size(p.base, NE0)), dtype=np.uint8
            )
            y = RP.do_repack(t, x)
            self.assertFalse(np.array_equal(x, y), name)
            base, z = RP.undo_repack(t, y)
            self.assertIs(base, p.base)
            np.testing.assert_array_equal(z, x, err_msg=name)

    def test_q8_0_r8_layout(self):
        # repack_q8_0: d of the 8 rows first, then 4 bytes of each row in turn.
        x = np.arange(8 * 34, dtype=np.uint8).reshape(8, 34)
        y = RP.do_repack(T.BY_NAME["Q8_0_R8"], x).reshape(-1)
        np.testing.assert_array_equal(y[:16], x[:, :2].reshape(-1))
        np.testing.assert_array_equal(y[16:20], x[0, 2:6])
        np.testing.assert_array_equal(y[20:24], x[1, 2:6])
        np.testing.assert_array_equal(y[16 + 128 : 16 + 132], x[0, 18:22])

    def test_chunked_restore_equals_whole(self):
        p = RP.plan("Q4_K_R4")
        t = T.BY_NAME["Q4_K_R4"]
        x = np.random.default_rng(2).integers(
            0, 256, (64, T.row_size(p.base, 1024)), dtype=np.uint8
        )
        y = RP.do_repack(t, x)
        old = RP._CHUNK_BYTES
        try:
            RP._CHUNK_BYTES = 1  # one group a chunk
            _, z = RP.undo_repack(t, y)
        finally:
            RP._CHUNK_BYTES = old
        np.testing.assert_array_equal(z, x)

    def test_partial_groups_are_refused(self):
        t = T.BY_NAME["IQ4_KS_R4"]
        with self.assertRaisesRegex(T.GGMLTypeError, "whole groups of 4 rows"):
            RP.undo_repack(t, np.zeros((6, T.row_size(t, 256)), np.uint8))


class _Libggml:
    def __init__(self, path):
        g = ctypes.CDLL(path)

        class InitParams(ctypes.Structure):
            _fields_ = [
                ("mem_size", ctypes.c_size_t),
                ("mem_buffer", ctypes.c_void_p),
                ("no_alloc", ctypes.c_bool),
            ]

        self.InitParams = InitParams
        g.ggml_init.restype = ctypes.c_void_p
        g.ggml_init.argtypes = [InitParams]
        g.ggml_free.argtypes = [ctypes.c_void_p]
        g.ggml_new_tensor_2d.restype = ctypes.c_void_p
        g.ggml_new_tensor_2d.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        g.ggml_get_data.restype = ctypes.c_void_p
        g.ggml_get_data.argtypes = [ctypes.c_void_p]
        g.ggml_nbytes.restype = ctypes.c_size_t
        g.ggml_nbytes.argtypes = [ctypes.c_void_p]
        g.iqk_repack_tensor.restype = None
        g.iqk_repack_tensor.argtypes = [ctypes.c_void_p]
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
        # ggml_init fills the fp16 tables the CPU quantisers read.
        g.ggml_free(g.ggml_init(InitParams(1 << 20, None, False)))
        self.g = g

    def quantize(self, tid, x):
        n, k = x.shape
        out = np.zeros(n * self.g.ggml_row_size(tid, k), np.uint8)
        imx = np.ones(k, np.float32)
        self.g.ggml_quantize_chunk(
            tid, x.ctypes.data, out.ctypes.data, 0, n, k, imx.ctypes.data, None
        )
        return out.reshape(n, -1)

    def repack(self, tid, rows, k):
        """ik's iqk_repack_tensor on a [n, k] tensor of type ``tid`` -> (new type id, bytes)."""
        n = rows.shape[0]
        ctx = self.g.ggml_init(self.InitParams(rows.nbytes + (1 << 20), None, False))
        try:
            t = self.g.ggml_new_tensor_2d(ctx, tid, k, n)
            assert self.g.ggml_nbytes(t) == rows.nbytes
            ctypes.memmove(self.g.ggml_get_data(t), rows.ctypes.data, rows.nbytes)
            self.g.iqk_repack_tensor(t)
            new_type = ctypes.c_int.from_address(
                t
            ).value  # ggml_tensor.type is the first field
            buf = (ctypes.c_uint8 * rows.nbytes).from_address(self.g.ggml_get_data(t))
            return new_type, np.frombuffer(bytes(buf), np.uint8).reshape(rows.shape)
        finally:
            self.g.ggml_free(ctx)


@unittest.skipUnless(os.environ.get("IK_LIBGGML"), "IK_LIBGGML not set")
class TestAgainstIk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ik = _Libggml(os.environ["IK_LIBGGML"])

    def test_every_repack(self):
        rng = np.random.default_rng(3)
        for name in sorted(T.REPACKS):
            with self.subTest(type=name):
                t = T.BY_NAME[name]
                p = RP.plan(name)
                x = rng.standard_normal((2 * p.rows, NE0)).astype(np.float32)
                base_rows = self.ik.quantize(int(p.base), x)
                new_type, ik_rows = self.ik.repack(int(p.base), base_rows, NE0)
                self.assertEqual(new_type, int(t))
                self.assertFalse(np.array_equal(ik_rows, base_rows))
                np.testing.assert_array_equal(RP.do_repack(t, base_rows), ik_rows)
                _, back = RP.undo_repack(t, ik_rows)
                np.testing.assert_array_equal(back, base_rows)
                # A file of type <name> is what ik's quantiser for <name> writes:
                # the base quantiser's rows, repacked.
                np.testing.assert_array_equal(self.ik.quantize(int(t), x), ik_rows)


if __name__ == "__main__":
    unittest.main()
