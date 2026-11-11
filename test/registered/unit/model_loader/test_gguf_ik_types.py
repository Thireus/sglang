"""The ggml / ik_llama.cpp type table of srt/model_loader/gguf_ik/types.py.

Checked three ways: the table against itself and gguf-py; against the
ik_llama.cpp source it was copied from (set SGLANG_IK_LLAMA_SRC to that tree);
and ``row_size`` against ``ggml_row_size`` of ik's libggml (set IK_LIBGGML).
The last two skip when the variable is not set.
"""

import ctypes
import os
import re
import unittest

import gguf

from sglang.srt.model_loader.gguf_ik import types as T
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Ids where gguf-py and ik_llama.cpp disagree: the activation types Q8_1 (ik
# keeps fp16 d and s) and Q8_K (ik adds a float). A GGUF loader keeps gguf-py's
# meaning for every id gguf-py knows.
GGUF_PY_DIFFERS = {9, 15}
# Same id and layout, another name: ik's Q1_0_G128 is gguf-py's Q1_0.
GGUF_PY_NAMES = {41: "Q1_0"}


def parse_ik_source(src):
    """(id, name, block, bytes, row header) of every type with a type_traits entry."""
    sizeof = {
        "ggml_half": 2,
        "ggml_bf16_t": 2,
        "ggml_fp16_t": 2,
        "uint8_t": 1,
        "int8_t": 1,
        "uint16_t": 2,
        "int16_t": 2,
        "uint32_t": 4,
        "int32_t": 4,
        "float": 4,
        "double": 8,
        "int64_t": 8,
        "uint64_t": 8,
    }
    H = open(os.path.join(src, "ggml/include/ggml.h")).read()
    C = open(os.path.join(src, "ggml/src/ggml.c")).read()
    G = open(os.path.join(src, "ggml/src/ggml-common.h")).read()
    enum = H[H.index("enum ggml_type {") :]
    enum = enum[: enum.index("GGML_TYPE_COUNT")]
    ids = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"^\s*GGML_TYPE_(\w+)\s*=\s*(\d+)\s*,", enum, re.M)
    }
    defs = {
        m.group(1): m.group(2)
        for m in re.finditer(r"^#define\s+(\w+)\s+(.+?)\s*(?://.*)?$", G, re.M)
    }
    asserts = {
        m.group(1): m.group(2)
        for m in re.finditer(
            r"static_assert\(sizeof\((block_\w+)\)\s*==\s*(.+?),\s*\"", G
        )
    }

    def ev(expr, depth=0):
        def sz(mm):
            t = mm.group(1)
            return str(sizeof[t] if t in sizeof else ev(asserts[t], depth + 1))

        e = re.sub(r"sizeof\((\w+)\)", sz, expr)
        for _ in range(20):
            e2 = re.sub(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\b",
                lambda mm: (
                    "(" + defs[mm.group(1)] + ")"
                    if mm.group(1) in defs
                    else mm.group(1)
                ),
                e,
            )
            e2 = re.sub(r"sizeof\((\w+)\)", sz, e2)
            if e2 == e:
                break
            e = e2
        return int(eval(e, {"__builtins__": {}}))

    rows = []
    for name, tid in sorted(ids.items(), key=lambda kv: kv[1]):
        m = re.search(
            r"^\s*\[GGML_TYPE_" + name + r"\]\s*=\s*\{(.*?)^\s*\},", C, re.S | re.M
        )
        if m is None:
            continue

        def field(k, body=m.group(1)):
            mm = re.search(r"\." + k + r"\s*=\s*([^,\n]+),", body)
            return mm.group(1).strip() if mm else None

        bs, ts, meta = field("blck_size"), field("type_size"), field("row_meta_size")
        rows.append(
            (
                tid,
                name,
                ev(bs) if bs else 0,
                ev(ts) if ts else 0,
                ev(meta) if meta else 0,
            )
        )
    return rows


class TestTypeTable(CustomTestCase):
    def test_ids_and_names_unique(self):
        self.assertEqual(len(T.BY_ID), len(T._ROWS))
        self.assertEqual(len(T.BY_NAME), len(T._ROWS))

    def test_repacks_share_the_base_layout(self):
        for name, (base, rows) in T.REPACKS.items():
            r, b = T.BY_NAME[name], T.BY_NAME[base]
            self.assertIn(rows, (4, 8), name)
            self.assertEqual(
                (r.blck_size, r.type_size, r.row_meta),
                (b.blck_size, b.type_size, b.row_meta),
                name,
            )
            self.assertEqual(r.repack_of, (b, rows))
        for name in T.NOT_REPACKS:
            self.assertIn(name, T.BY_NAME)
            self.assertNotIn(name, T.REPACKS)

    def test_agrees_with_gguf_py(self):
        for m in gguf.GGMLQuantizationType:
            t = T.BY_ID.get(int(m))
            if t is None or int(m) in GGUF_PY_DIFFERS:
                continue
            self.assertEqual(GGUF_PY_NAMES.get(int(m), t.name), m.name)
            self.assertEqual(
                (t.blck_size, t.type_size), gguf.GGML_QUANT_SIZES[m], m.name
            )
            self.assertEqual(t.row_meta, 0, m.name)

    def test_served_ids_match_the_gguf_quant_method(self):
        try:
            from sglang.srt.layers.quantization import gguf as G
        except ImportError as e:  # pragma: no cover - kernels missing on this host
            self.skipTest(f"layers.quantization.gguf does not import here: {e}")
        want = (
            G.UNQUANTIZED_TYPES
            | G.STANDARD_QUANT_TYPES
            | G.KQUANT_TYPES
            | G.IMATRIX_QUANT_TYPES
        )
        self.assertEqual(T.GGUF_SERVED_IDS, {int(t) for t in want})

    def test_row_size(self):
        self.assertEqual(T.row_size(T.BY_NAME["IQ4_KS"], 5120), 4 + 20 * 136)
        self.assertEqual(T.row_size(T.BY_NAME["IQ2_KS"], 5120), 2 + 20 * 70)
        self.assertEqual(T.row_size(T.BY_NAME["IQ2_S"], 5120), 20 * 82)
        self.assertEqual(T.row_size(T.BY_NAME["IQ1_S_R4"], 5120), 2 + 160 * 6)
        self.assertEqual(T.row_size(T.BY_NAME["IQ2_KT"], 5120), 4 + 20 * 68)
        self.assertEqual(T.row_size(22, 512), 164)
        with self.assertRaisesRegex(T.GGMLTypeError, "not a multiple"):
            T.row_size(T.BY_NAME["IQ4_KS"], 100)

    def test_refusals_name_the_id(self):
        with self.assertRaisesRegex(T.GGMLTypeError, "142.*PQ2_0.*IQ2_TN"):
            T.by_id(142)
        with self.assertRaisesRegex(T.GGMLTypeError, "ggml type id 4000 is unknown"):
            T.by_id(4000)
        self.assertIn("own", T.why_not_served(T.BY_NAME["IQ1_S_R4"]))
        self.assertIn("cold tier", T.why_not_served(T.BY_NAME["IQ4_KS"]))
        msg = T.why_not_served(T.BY_NAME["IQ2_K_R4"])
        self.assertIn("IQ2_K regrouped by rows", msg)
        self.assertIn("no dequantise or matmul kernel", msg)

    def test_ggml_type_behaves_like_an_int(self):
        import pickle

        import torch

        t = T.BY_NAME["IQ4_KS"]
        self.assertEqual(int(t), 144)
        self.assertEqual(t.name, "IQ4_KS")
        self.assertEqual(t.value, 144)
        self.assertEqual(int(torch.tensor(t)), 144)
        self.assertIs(pickle.loads(pickle.dumps(t)), t)
        self.assertAlmostEqual(t.bits_per_weight, 4.25)


@unittest.skipUnless(
    os.environ.get("SGLANG_IK_LLAMA_SRC"), "SGLANG_IK_LLAMA_SRC not set"
)
class TestAgainstIkSource(CustomTestCase):
    """Every row of the table equals the ik_llama.cpp source it names (IK_SOURCE)."""

    def test_every_row(self):
        rows = parse_ik_source(os.environ["SGLANG_IK_LLAMA_SRC"])
        self.assertGreater(len(rows), 100)
        self.assertEqual(sorted(rows), sorted(T._ROWS))


@unittest.skipUnless(os.environ.get("IK_LIBGGML"), "IK_LIBGGML not set")
class TestAgainstLibggml(CustomTestCase):
    """row_size equals ik's ggml_row_size for every type and several row lengths."""

    def test_row_size(self):
        g = ctypes.CDLL(os.environ["IK_LIBGGML"])
        g.ggml_row_size.restype = ctypes.c_size_t
        g.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        g.ggml_blck_size.restype = ctypes.c_int64
        g.ggml_blck_size.argtypes = [ctypes.c_int]
        g.ggml_type_size.restype = ctypes.c_size_t
        g.ggml_type_size.argtypes = [ctypes.c_int]
        n = 0
        for t in T.BY_ID.values():
            self.assertEqual(g.ggml_blck_size(int(t)), t.blck_size, t.name)
            self.assertEqual(g.ggml_type_size(int(t)), t.type_size, t.name)
            unit = T.row_blck_size(t)
            for ne in sorted({t.blck_size * 8, 2048, 5120, 256 + 32, 4096 + 96}):
                if ne % unit or ne % t.blck_size and unit == t.blck_size:
                    continue
                self.assertEqual(
                    T.row_size(t, ne), g.ggml_row_size(int(t), ne), (t.name, ne)
                )
                n += 1
        self.assertGreater(n, 300)


if __name__ == "__main__":
    unittest.main()
