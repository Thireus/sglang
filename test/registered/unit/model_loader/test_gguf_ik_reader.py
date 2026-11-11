"""Reading GGUF files with ik_llama.cpp types and split sets (gguf_ik reader / split).

The files are written here, byte by byte (gguf-py's writer cannot write an id
it does not know): a classic file must read exactly as gguf-py reads it, an ik
file gets the right byte counts with its row headers, a split set is found
from any shard or its directory, and every refusal names what is wrong.
"""

import os
import struct
import tempfile
import unittest

import gguf
import numpy as np
import torch

from sglang.srt.model_loader.gguf_ik import (
    GGMLTypeError,
    GGUFFile,
    GGUFSplit,
    GGUFSplitError,
    IkGGUFReader,
    open_gguf,
)
from sglang.srt.model_loader.gguf_ik import repack as RP
from sglang.srt.model_loader.gguf_ik import (
    served_tensor,
    served_type,
)
from sglang.srt.model_loader.gguf_ik import types as T
from sglang.srt.model_loader.weight_utils import (
    get_gguf_extra_tensor_names,
    gguf_quant_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ALIGN = 32


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, kv=None):
    """tensors: [(name, ne (ggml order), type id, bytes)]; kv: {key: int (u32) | str}."""
    kv = dict(kv or {})
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv.items():
        if isinstance(v, str):
            head += _s(k) + struct.pack("<I", 8) + _s(v)
        else:
            head += _s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
    off, blobs = 0, []
    for name, ne, tid, data in tensors:
        head += (
            _s(name)
            + struct.pack("<I", len(ne))
            + b"".join(struct.pack("<Q", d) for d in ne)
        )
        head += struct.pack("<IQ", tid, off)
        blobs.append(data + b"\0" * (-len(data) % ALIGN))
        off += len(blobs[-1])
    head += b"\0" * (-len(head) % ALIGN)
    with open(path, "wb") as f:
        f.write(head + b"".join(blobs))


def rand_bytes(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


class _Tmp(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def p(self, name):
        return os.path.join(self.dir, name)


class TestClassicFilesReadAsBefore(_Tmp):
    def _classic(self):
        path = self.p("classic.gguf")
        w = gguf.GGUFWriter(path, "llama")
        w.add_uint32("llama.block_count", 1)
        w.add_tensor(
            "token_embd.weight", np.arange(64, dtype=np.float32).reshape(4, 16)
        )
        w.add_tensor("output_norm.weight", np.ones(16, np.float16))
        Q = gguf.GGMLQuantizationType
        for name, qt in (
            ("blk.0.attn_q.weight", Q.Q8_0),
            ("blk.0.ffn_up.weight", Q.Q4_K),
            ("blk.0.ffn_down.weight", Q.IQ2_S),
        ):
            bs, ts = gguf.GGML_QUANT_SIZES[qt]
            raw = np.frombuffer(
                rand_bytes(4 * 256 // bs * ts, seed=int(qt)), np.uint8
            ).reshape(4, -1)
            w.add_tensor(name, raw, raw_dtype=qt)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return path

    def test_tensors_identical_to_gguf_py(self):
        path = self._classic()
        a, b = gguf.GGUFReader(path), IkGGUFReader(path)
        self.assertEqual(list(a.fields), list(b.fields))
        self.assertEqual(len(a.tensors), len(b.tensors))
        for x, y in zip(a.tensors, b.tensors):
            self.assertEqual(x.name, y.name)
            self.assertIs(x.tensor_type, y.tensor_type)
            np.testing.assert_array_equal(x.shape, y.shape)
            self.assertEqual(
                (x.n_elements, x.n_bytes, x.data_offset),
                (y.n_elements, y.n_bytes, y.data_offset),
            )
            self.assertEqual((x.data.dtype, x.data.shape), (y.data.dtype, y.data.shape))
            np.testing.assert_array_equal(x.data, y.data)
            # The loader's view of a classic tensor is the tensor itself.
            self.assertIs(served_type(y), y.tensor_type)
            st, data = served_tensor(y)
            self.assertIs(st, y.tensor_type)
            self.assertIs(data, y.data)

    def test_single_file_source(self):
        path = self._classic()
        src = open_gguf(path)
        self.assertIsInstance(src, GGUFFile)
        self.assertEqual(
            [t.name for t in src.tensor_infos()],
            [t.name for t in gguf.GGUFReader(path).tensors],
        )

    def test_weight_iterator_unchanged_for_classic(self):
        path = self._classic()
        name_map = {
            "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
            "blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight",
            "token_embd.weight": "model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
        }
        got = dict(gguf_quant_weights_iterator(path, name_map))
        ref = gguf.GGUFReader(path)
        by = {t.name: t for t in ref.tensors}
        self.assertEqual(
            int(got["model.layers.0.self_attn.q_proj.qweight_type"]),
            int(gguf.GGMLQuantizationType.Q8_0),
        )
        np.testing.assert_array_equal(
            got["model.layers.0.mlp.up_proj.qweight"].numpy(),
            by["blk.0.ffn_up.weight"].data,
        )
        np.testing.assert_array_equal(
            got["model.embed_tokens.weight"].numpy(), by["token_embd.weight"].data
        )
        self.assertEqual(
            get_gguf_extra_tensor_names(path, name_map), ["lm_head.weight"]
        )


class TestIkTypes(_Tmp):
    def test_row_headers_are_counted(self):
        t = T.BY_NAME["IQ4_KS"]
        rb = T.row_size(t, 512)  # 4 + 2 * 136
        self.assertEqual(rb, 276)
        body = rand_bytes(3 * rb)
        write_gguf(
            self.p("ik.gguf"),
            [
                ("a.weight", (512, 3), 144, body),
                ("b.weight", (8,), 0, np.arange(8, dtype=np.float32).tobytes()),
                (
                    "c.weight",
                    (256, 2, 2),
                    157,
                    rand_bytes(4 * T.row_size(T.BY_NAME["IQ2_KL"], 256), 5),
                ),
            ],
        )
        r = IkGGUFReader(self.p("ik.gguf"))
        a, b, c = r.tensors
        self.assertIs(a.tensor_type, t)
        self.assertEqual(a.tensor_type.name, "IQ4_KS")
        self.assertEqual((a.n_bytes, a.data.shape), (3 * rb, (3, rb)))
        self.assertEqual(a.data.tobytes(), body)
        self.assertIs(b.tensor_type, gguf.GGMLQuantizationType.F32)
        np.testing.assert_array_equal(b.data, np.arange(8, dtype=np.float32))
        self.assertEqual(c.data.shape, (2, 2, 88))  # IQ2_KL: 2 + 86
        self.assertEqual(int(torch.tensor(a.tensor_type)), 144)

    def test_refusals(self):
        cases = [
            (142, "PQ2_0"),
            (4000, "ggml type id 4000 is unknown"),
            (36, "I2_S"),
        ]
        for tid, msg in cases:
            write_gguf(self.p("x.gguf"), [("w.weight", (256, 1), tid, rand_bytes(64))])
            with self.assertRaisesRegex(GGMLTypeError, "w.weight: .*" + msg):
                IkGGUFReader(self.p("x.gguf"))
        write_gguf(self.p("x.gguf"), [("w.weight", (100, 1), 144, rand_bytes(64))])
        with self.assertRaisesRegex(
            GGMLTypeError, "w.weight: IQ4_KS: a row of 100 values"
        ):
            IkGGUFReader(self.p("x.gguf"))
        write_gguf(self.p("x.gguf"), [("w.weight", (256, 4), 144, rand_bytes(32))])
        with open(self.p("x.gguf"), "r+b") as f:
            f.truncate(os.path.getsize(self.p("x.gguf")) - 32)
        with self.assertRaisesRegex(GGMLTypeError, "past the end of the file"):
            IkGGUFReader(self.p("x.gguf"))
        write_gguf(
            self.p("x.gguf"),
            [("w.weight", (256, 1), 144, rand_bytes(140))],
            kv={T.TRELLIS_REMAP_KEY: 15},
        )
        with self.assertRaisesRegex(GGMLTypeError, "trellis"):
            IkGGUFReader(self.p("x.gguf"))

    def test_served_view_of_repacks_and_refusals(self):
        r4 = T.BY_NAME["Q4_K_R4"]
        base_rows = np.frombuffer(rand_bytes(4 * 144 * 2, 9), np.uint8).reshape(4, 288)
        stored = RP.do_repack(r4, base_rows)
        write_gguf(
            self.p("r4.gguf"),
            [
                ("blk.0.ffn_up.weight", (512, 4), int(r4), stored.tobytes()),
                ("blk.0.ffn_gate.weight", (512, 4), 144, rand_bytes(4 * 276)),
                (
                    "blk.0.attn_q.weight",
                    (256, 4),
                    219,
                    rand_bytes(4 * T.row_size(T.BY_NAME["IQ1_S_R4"], 256)),
                ),
            ],
        )
        up, gate, q = IkGGUFReader(self.p("r4.gguf")).tensors
        self.assertIs(served_type(up), gguf.GGMLQuantizationType.Q4_K)
        st, data = served_tensor(up)
        self.assertIs(st, gguf.GGMLQuantizationType.Q4_K)
        np.testing.assert_array_equal(data, base_rows)
        with self.assertRaisesRegex(
            GGMLTypeError,
            r"blk.0.ffn_gate.weight: type IQ4_KS \(ggml id 144.*cold tier",
        ):
            served_type(gate)
        with self.assertRaisesRegex(GGMLTypeError, "IQ1_S_R4 .*quantiser of its own"):
            served_type(q)
        name_map = {"blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight"}
        got = dict(gguf_quant_weights_iterator(self.p("r4.gguf"), name_map))
        self.assertEqual(
            int(got["model.layers.0.mlp.up_proj.qweight_type"]),
            int(gguf.GGMLQuantizationType.Q4_K),
        )
        np.testing.assert_array_equal(
            got["model.layers.0.mlp.up_proj.qweight"].numpy(), base_rows
        )
        name_map["blk.0.ffn_gate.weight"] = "model.layers.0.mlp.gate_proj.weight"
        with self.assertRaisesRegex(GGMLTypeError, "IQ4_KS"):
            list(gguf_quant_weights_iterator(self.p("r4.gguf"), name_map))

    def test_moe_experts_of_a_repack(self):
        r4 = T.BY_NAME["IQ4_NL_R4"]
        rb = T.row_size(r4, 64)
        base = np.frombuffer(rand_bytes(2 * 4 * rb, 4), np.uint8).reshape(
            2, 4, rb
        )  # 2 experts x 4 rows
        write_gguf(
            self.p("moe.gguf"),
            [
                (
                    "blk.3.ffn_down_exps.weight",
                    (64, 4, 2),
                    int(r4),
                    RP.do_repack(r4, base).tobytes(),
                )
            ],
        )
        got = dict(gguf_quant_weights_iterator(self.p("moe.gguf"), {}))
        for e in range(2):
            pre = f"model.layers.3.mlp.experts.{e}.down_proj"
            self.assertEqual(
                int(got[pre + ".qweight_type"]), int(gguf.GGMLQuantizationType.IQ4_NL)
            )
            np.testing.assert_array_equal(got[pre + ".qweight"].numpy(), base[e])


class TestSplitSets(_Tmp):
    N = 4

    def _set(self, with_map=True, prefix="M-IQ4_KS-SPECIAL_TENSOR"):
        """A 4-shard set: shard 1 metadata only, then one tensor a shard, as GGUF Tool Suite sets are cut."""
        meta = {
            "general.architecture": "llama",
            "split.no": 0,
            "split.count": self.N,
            "split.tensors.count": self.N - 1,
        }
        tens = [
            ("token_embd.weight", (256, 2), 30, rand_bytes(2 * 512, 1)),
            ("blk.0.ffn_up.weight", (512, 4), 144, rand_bytes(4 * 276, 2)),
            ("output_norm.weight", (256,), 0, rand_bytes(1024, 3)),
        ]
        names = [f"{prefix}-{i:05d}-of-{self.N:05d}.gguf" for i in range(1, self.N + 1)]
        write_gguf(self.p(names[0]), [], kv=meta)
        lines = []
        for i, (name, ne, tid, data) in enumerate(tens, 1):
            write_gguf(
                self.p(names[i]),
                [(name, ne, tid, data)],
                kv={
                    "split.no": i,
                    "split.count": self.N,
                    "split.tensors.count": self.N - 1,
                },
            )
            dt = T.BY_ID[tid].name.lower()
            lines.append(
                f"{names[i]}:{'0' * 64}:{name}:shape=({','.join(map(str, ne))}):dtype={dt}:elements=1:bytes={len(data)}"
            )
        if with_map:
            with open(self.p("tensors.map"), "w") as f:
                f.write("\n".join(lines) + "\n")
        return names, tens

    def test_open_from_any_shard_or_the_directory(self):
        names, tens = self._set()
        for path in (self.p(names[0]), self.p(names[2]), self.dir):
            sp = open_gguf(path)
            self.assertIsInstance(sp, GGUFSplit)
            self.assertEqual(sp.first_path, self.p(names[0]))
            self.assertEqual([t.name for t in sp.tensor_infos()], [t[0] for t in tens])
            got = list(sp.tensors())
            self.assertEqual([t.name for t in got], [t[0] for t in tens])
            self.assertEqual(got[1].data.tobytes(), tens[1][3])
            self.assertEqual(sp.fields["general.architecture"].contents(), "llama")

    def test_map_and_headers_agree(self):
        names, _ = self._set()
        with_map = GGUFSplit(self.p(names[0])).tensor_infos()
        without = GGUFSplit(self.p(names[0]), use_map=False).tensor_infos()
        self.assertEqual(
            [(t.name, int(t.tensor_type), t.shape, t.n_bytes) for t in with_map],
            [(t.name, int(t.tensor_type), t.shape, t.n_bytes) for t in without],
        )
        self.assertEqual(
            GGUFSplit(self.p(names[0])).read("blk.0.ffn_up.weight").tensor_type.name,
            "IQ4_KS",
        )

    def test_missing_shard_names_the_shard_and_tensor(self):
        names, _ = self._set()
        os.remove(self.p(names[2]))
        sp = GGUFSplit(self.p(names[0]))
        with self.assertRaisesRegex(
            GGUFSplitError, "1 of the 4 shards .*" + names[2] + ".*blk.0.ffn_up.weight"
        ):
            sp.tensor_infos()
        with self.assertRaisesRegex(GGUFSplitError, names[2]):
            list(sp.tensors())
        os.remove(self.p(names[0]))
        with self.assertRaisesRegex(
            GGUFSplitError, "first shard of this split is missing"
        ):
            open_gguf(self.p(names[1]))

    def test_inconsistent_sets(self):
        names, _ = self._set()
        with open(self.p("tensors.map")) as f:
            text = f.read()
        with open(self.p("tensors.map"), "w") as f:
            f.write(text.replace("dtype=iq4_ks", "dtype=iq2_ks"))
        with self.assertRaisesRegex(
            GGUFSplitError, "IQ4_KS .* but iq2_ks .* in tensors.map"
        ):
            list(GGUFSplit(self.p(names[0])).tensors())
        with open(self.p("tensors.map"), "w") as f:
            f.write(text.splitlines()[0] + "\n")
        with self.assertRaisesRegex(
            GGUFSplitError, "lists 1 tensors but .*split.tensors.count = 3"
        ):
            GGUFSplit(self.p(names[0]))
        os.remove(self.p("tensors.map"))
        write_gguf(self.p(names[3]), [], kv={"split.no": 3, "split.count": self.N})
        with self.assertRaisesRegex(
            GGUFSplitError, "hold 2 tensors .*split.tensors.count = 3"
        ):
            GGUFSplit(self.p(names[0])).tensor_infos()

    def test_two_sets_in_one_directory(self):
        self._set()
        self._set(with_map=False, prefix="OTHER")
        with self.assertRaisesRegex(GGUFSplitError, "first shard of 2 split sets"):
            open_gguf(self.dir)

    def test_weight_iterator_reads_a_split(self):
        names, tens = self._set()
        got = dict(
            gguf_quant_weights_iterator(
                self.p(names[0]), {"output_norm.weight": "model.norm.weight"}
            )
        )
        self.assertEqual(got["model.norm.weight"].numpy().tobytes(), tens[2][3])
        with self.assertRaisesRegex(GGMLTypeError, "IQ4_KS"):
            dict(
                gguf_quant_weights_iterator(
                    self.p(names[0]), {"blk.0.ffn_up.weight": "x.weight"}
                )
            )


if __name__ == "__main__":
    unittest.main()
