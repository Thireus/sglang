"""IQ1_S_R4, IQ1_M_R4 and Q8_K_R8 rows as the GGUF loader hands them to the kernels.

ik_llama.cpp stores these three types as interleaved groups of 4 or 8 rows.
``split_row_groups`` rewrites a group as its rows one after the other. Checked
here: it is a byte permutation with ``join_row_groups`` as its inverse, a row of
the result depends on that row's values only (rows can be moved and selected),
and decoding the rows straight from ik's field definitions gives the values of
the bit-exact reference on the stored groups. The loader's view of the types
served by dequantising, the switch that names them and its refusals follow.
"""

import unittest

import numpy as np

from sglang.srt.environ import envs
from sglang.srt.model_loader.gguf_ik import types as T
from sglang.srt.model_loader.gguf_ik.rowgroups import (
    join_row_groups,
    split_row_groups,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels import ggml_quant_reference as ref
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

F32 = np.float32
GROUP_IDS = {name: int(T.BY_NAME[name]) for name in T.ROW_GROUPS}


def f16(b):
    return np.ascontiguousarray(b).view("<f2").astype(F32)[..., 0]


def decode_row(name: str, row: np.ndarray, k: int) -> np.ndarray:
    """One self-contained row -> float32 [k], from ik's field definitions."""
    grid = ref._grid("IQ1_S")
    y = np.empty(k, F32)
    if name == "Q8_K_R8":
        blk = row.reshape(k // 256, 258)
        for b in range(k // 256):
            y[256 * b : 256 * b + 256] = f16(blk[b, 0:2]) * blk[b, 2:].view(
                np.int8
            ).astype(F32)
        return y
    d = f16(row[0:2])
    blk = row[2:].reshape(k // 32, 6 if name == "IQ1_S_R4" else 7)
    for b in range(k // 32):
        if name == "IQ1_S_R4":
            qh = int(blk[b, 4]) | (int(blk[b, 5]) << 8)
            dl = d * F32(2 * ((qh >> 12) & 7) + 1)
            delta = F32(-0.125 if qh & 0x8000 else 0.125)
            for i in range(4):
                idx = int(blk[b, i]) | (((qh >> (3 * i)) & 7) << 8)
                y[32 * b + 8 * i : 32 * b + 8 * i + 8] = dl * (grid[idx] + delta)
        else:
            sc = int(blk[b, 6])
            for i in range(2):
                dl = d * F32((sc >> 4) if i else (sc & 15))
                qh = int(blk[b, 4 + i])
                for h in range(2):
                    hi = ((qh & 0x70) << 4) if h else ((qh & 0x07) << 8)
                    delta = F32(-0.125 if qh & (0x80 if h else 0x08) else 0.125)
                    o = 32 * b + 16 * i + 8 * h
                    y[o : o + 8] = dl * (grid[int(blk[b, 2 * i + h]) | hi] + delta)
    return y


class TestRowGroups(CustomTestCase):
    def test_split_is_a_byte_permutation_with_an_inverse(self):
        rng = np.random.default_rng(0)
        for name, group in T.ROW_GROUPS.items():
            t = T.BY_NAME[name]
            for k in (256, 1280):
                rb = T.row_size(t, k)
                stored = rng.integers(0, 256, (3, 2 * group, rb), dtype=np.uint8)
                rows = split_row_groups(t, stored)
                self.assertEqual(rows.shape, stored.shape)
                self.assertFalse(np.array_equal(rows, stored))
                np.testing.assert_array_equal(join_row_groups(t, rows), stored)
                np.testing.assert_array_equal(
                    split_row_groups(int(t), join_row_groups(t, stored)), stored
                )
                # every byte of a group lands in exactly one place of its rows
                one = np.arange(group * rb, dtype=np.int64)
                tag = (one % 251).astype(np.uint8).reshape(group, rb)
                np.testing.assert_array_equal(
                    np.sort(split_row_groups(t, tag).reshape(-1)),
                    np.sort(tag.reshape(-1)),
                )

    def test_rows_decode_to_the_reference(self):
        rng = np.random.default_rng(1)
        for name, group in T.ROW_GROUPS.items():
            with self.subTest(name):
                tid, k = GROUP_IDS[name], 512
                stored = ref.random_rows(tid, 2 * group, k, rng)
                want = ref.dequantize(tid, stored, k)
                rows = split_row_groups(tid, stored)
                for r in range(2 * group):
                    got = decode_row(name, rows[r], k)
                    self.assertTrue(
                        np.array_equal(got.view(np.uint32), want[r].view(np.uint32)),
                        f"{name} row {r}",
                    )

    def test_a_row_is_self_contained(self):
        """Rows of two groups can be mixed: each keeps the values it had."""
        rng = np.random.default_rng(2)
        for name, group in T.ROW_GROUPS.items():
            with self.subTest(name):
                tid, k = GROUP_IDS[name], 256
                a = ref.random_rows(tid, group, k, rng)
                b = ref.random_rows(tid, group, k, rng)
                va, vb = ref.dequantize(tid, a, k), ref.dequantize(tid, b, k)
                rows = split_row_groups(tid, a)
                pick = rng.random(group) < 0.5
                pick[0], pick[1] = True, False
                rows[pick] = split_row_groups(tid, b)[pick]
                mixed = ref.dequantize(tid, join_row_groups(tid, rows), k)
                want = np.where(pick[:, None], vb, va)
                self.assertTrue(
                    np.array_equal(mixed.view(np.uint32), want.view(np.uint32))
                )

    def test_refusals(self):
        with self.assertRaisesRegex(T.GGMLTypeError, "not stored as interleaved"):
            split_row_groups(T.BY_NAME["IQ4_KS"], np.zeros((4, 140), np.uint8))
        with self.assertRaisesRegex(T.GGMLTypeError, "whole groups of 4 rows"):
            split_row_groups(T.BY_NAME["IQ1_S_R4"], np.zeros((6, 50), np.uint8))
        with self.assertRaisesRegex(T.GGMLTypeError, "whole groups of 8 rows"):
            split_row_groups(T.BY_NAME["Q8_K_R8"], np.zeros((8, 259), np.uint8))


class TestDequantSwitch(CustomTestCase):
    def test_every_dequant_type_is_in_the_reference(self):
        for tid in T.GGUF_DEQUANT_IDS:
            t, q = T.BY_ID[tid], ref.TYPES[tid]
            self.assertEqual(
                (t.name, t.blck_size, t.type_size, t.row_meta),
                (q.name, q.block, q.block_bytes, q.header),
            )
            self.assertEqual(q.rows, T.ROW_GROUPS.get(t.name, 1))
        self.assertFalse(T.GGUF_DEQUANT_IDS & T.GGUF_SERVED_IDS)
        self.assertEqual(set(T.ROW_GROUPS), set(T.NOT_REPACKS))

    def test_kernel_table_matches(self):
        try:
            from sglang.kernels.ops.quantization.gguf_dequantize import (
                GGUF_DEQUANTIZE_TYPES,
            )
        except ImportError as e:  # pragma: no cover - kernels missing on this host
            self.skipTest(f"the kernel module does not import here: {e}")
        self.assertEqual(set(GGUF_DEQUANTIZE_TYPES), set(T.GGUF_DEQUANT_IDS))
        for tid, row in GGUF_DEQUANTIZE_TYPES.items():
            t = T.BY_ID[tid]
            self.assertEqual(row, (t.name, t.blck_size, t.type_size, t.row_meta))

    def test_switch(self):
        iq3ks, q6 = T.BY_NAME["IQ3_KS"], T.BY_NAME["Q6_0"]
        self.assertEqual(T.dequant_served_ids(), T.GGUF_DEQUANT_IDS)
        self.assertTrue(T.served_today(iq3ks))
        with envs.SGLANG_GGUF_DEQUANT_TYPES.override("none"):
            self.assertEqual(T.dequant_served_ids(), frozenset())
            self.assertFalse(T.served_today(iq3ks))
            self.assertTrue(T.served_today(T.BY_NAME["Q4_K"]))
            msg = T.why_not_served(iq3ks)
            self.assertIn("dequantising on the GPU", msg)
            self.assertIn("SGLANG_GGUF_DEQUANT_TYPES='none'", msg)
        with envs.SGLANG_GGUF_DEQUANT_TYPES.override("iq3_ks, IQ2_KL"):
            self.assertEqual(T.dequant_served_ids(), {156, 157})
            self.assertTrue(T.served_today(iq3ks))
            self.assertFalse(T.served_today(q6))
            self.assertIn("leaves it out", T.why_not_served(T.BY_NAME["Q6_0_R4"]))
        with envs.SGLANG_GGUF_DEQUANT_TYPES.override("IQ3_KS,Q4_K"):
            with self.assertRaisesRegex(T.GGMLTypeError, "'Q4_K' is not a type served"):
                T.dequant_served_ids()


if __name__ == "__main__":
    unittest.main()
