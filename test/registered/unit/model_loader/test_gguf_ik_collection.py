"""gguf_ik against real SPECIAL_SPLIT shard sets (one directory a quant type).

Set SGLANG_GGUF_SPLIT_COLLECTION to a directory holding sets named
``<model>-<TYPE>-SPECIAL_SPLIT``, with both a base type and its ``_R4`` / ``_R8``
form of the same model (made from the same source and imatrix). The test reads
each set's tensors.map and first shard, and at most ten small tensors (under
64 MB in all) from each of six sets; it skips when the variable is not set.

For each pair the repacked tensors are restored and must equal the base type's
own shard of the same tensor, bit for bit.
"""

import glob
import os
import unittest

import numpy as np

from sglang.srt.model_loader.gguf_ik import GGUFSplit
from sglang.srt.model_loader.gguf_ik import repack as RP
from sglang.srt.model_loader.gguf_ik import served_type
from sglang.srt.model_loader.gguf_ik import types as T
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROOT = os.environ.get("SGLANG_GGUF_SPLIT_COLLECTION")
PAIRS = (("IQ4_KS", "IQ4_KS_R4"), ("Q4_K", "Q4_K_R4"), ("IQ2_XXS", "IQ2_XXS_R4"))
MAX_TENSORS = 10
MAX_BYTES = 64 << 20


def find_set(qtype):
    hits = sorted(glob.glob(os.path.join(ROOT, f"*-{qtype}-SPECIAL_SPLIT")))
    return hits[0] if hits else None


@unittest.skipUnless(
    ROOT and os.path.isdir(ROOT or ""), "SGLANG_GGUF_SPLIT_COLLECTION not set"
)
class TestCollection(CustomTestCase):
    def test_sets_open_from_the_first_shard(self):
        for base, r4 in PAIRS:
            for q in (base, r4):
                d = find_set(q)
                if d is None:
                    continue
                with self.subTest(set=os.path.basename(d)):
                    sp = GGUFSplit(d)
                    self.assertIsNotNone(sp.map_path)
                    infos = sp.tensor_infos()
                    self.assertEqual(len(infos), sp.tensor_count)
                    self.assertEqual(len(sp.shards), sp.count)
                    names = {int(t.tensor_type) for t in infos}
                    self.assertIn(int(T.BY_NAME[q]), names)

    def test_repacks_equal_the_base_shards(self):
        done = 0
        for base, r4 in PAIRS:
            db, dr = find_set(base), find_set(r4)
            if db is None or dr is None:
                continue
            sb, sr = GGUFSplit(db), GGUFSplit(dr)
            bi = {t.name: t for t in sb.tensor_infos()}
            pick, total = [], 0
            for t in sorted(sr.tensor_infos(), key=lambda t: t.n_bytes):
                b = bi.get(t.name)
                if t.tensor_type.name != r4 or b is None or b.tensor_type.name != base:
                    continue
                if total + 2 * t.n_bytes > MAX_BYTES or len(pick) == MAX_TENSORS:
                    break
                pick.append(t.name)
                total += 2 * t.n_bytes
            self.assertTrue(pick, f"no {r4} tensor with a {base} twin")
            for name in pick:
                with self.subTest(pair=r4, tensor=name):
                    tr, tb = sr.read(name), sb.read(name)
                    if T.served_today(T.BY_NAME[base]):
                        self.assertEqual(served_type(tr).name, base)
                    t, rows = RP.undo_repack(tr.tensor_type, tr.data)
                    self.assertEqual(t.name, base)
                    self.assertEqual(rows.shape, tb.data.shape)
                    np.testing.assert_array_equal(rows, tb.data)
                    self.assertFalse(np.array_equal(tr.data, tb.data))
                    done += 1
        if not done:
            self.skipTest("no base / repack pair in the collection")


if __name__ == "__main__":
    unittest.main()
