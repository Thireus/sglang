# SPDX-License-Identifier: Apache-2.0
"""Zero-sign check and repair on tiny synthetic matrices. CPU only, seconds.

The tests that quantise or dequantise need ik_llama.cpp's libggml.so: set
IK_LIBGGML=/path/to/libggml.so; without it they are skipped and the numpy-only
tests still run.

    IK_LIBGGML=/path/to/libggml.so python3 tools/zero-sign-repair/test/test_zero_sign.py
"""

import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# The repository's bit-exact numpy dequantisers (no libggml needed), when the
# tool runs from a checkout of the repository.
REF_PATH = os.path.join(
    HERE,
    "..",
    "..",
    "..",
    "python",
    "sglang",
    "test",
    "kernels",
    "ggml_quant_reference.py",
)

import numpy as np  # noqa: E402
import zero_sign as zs  # noqa: E402
import zero_sign_check  # noqa: E402
import zero_sign_repair  # noqa: E402

LIB = os.environ.get(zs.LIB_ENV)
HAVE_LIB = bool(LIB) and os.path.isfile(LIB)
needs_lib = unittest.skipUnless(
    HAVE_LIB, f"set {zs.LIB_ENV} to ik_llama.cpp's libggml.so"
)


def signed_zero_matrix(n, k, zero_share=0.12, seed=0, plus_only=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float32)
    z = rng.random((n, k)) < zero_share
    neg = (rng.random((n, k)) < 0.5) & (not plus_only)
    x[z] = np.where(neg[z], np.float32(-0.0), np.float32(0.0))
    return x


def mxfp4_encode(x):
    """float [n, k] -> (codes uint8 [n, k/2], E8M0 uint8 [n, k/32]): round to the nearest E2M1 value of each
    block's power-of-two scale; the sign of every value (and of -0) is kept."""
    n, k = x.shape
    xb = x.reshape(n, k // 32, 32)
    amax = np.abs(xb).max(axis=-1)
    e = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-30) / 6.0)), 0).astype(
        np.int32
    )
    sc = np.ldexp(np.float32(1), e)[:, :, None]
    mags = zs.E2M1[:8]
    v = np.abs(xb) / sc
    idx = np.abs(v[..., None] - mags).argmin(axis=-1).astype(np.uint8)
    code = idx | (np.signbit(xb).astype(np.uint8) << 3)
    code = code.reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, (e + 127).astype(np.uint8)


def write_safetensors(path, tensors, meta=None):
    """tensors: {name: (dtype string, shape, bytes)}."""
    hdr, off, blobs = {}, 0, []
    if meta:
        hdr["__metadata__"] = meta
    for name, (dt, shape, raw) in tensors.items():
        hdr[name] = {
            "dtype": dt,
            "shape": list(shape),
            "data_offsets": [off, off + len(raw)],
        }
        off += len(raw)
        blobs.append(raw)
    hb = json.dumps(hdr, separators=(",", ":"), sort_keys=True).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def run_cli(mod, argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = mod.main(argv)
    return rc, out.getvalue() + err.getvalue()


class TestNumpyOnly(unittest.TestCase):
    def test_mxfp4_keeps_negative_zero(self):
        codes = np.array([[0x80, 0x08, 0x21] + [0] * 13], np.uint8)  # k = 32
        x = zs.mxfp4_dequant(codes, np.full((1, 1), 127, np.uint8))
        pos0, neg0 = zs.zero_masks_from_codes(zs.fp4_nibbles(codes))
        self.assertTrue(
            np.signbit(x[0, 1]) and x[0, 1] == 0
        )  # high nibble of byte 0 = column 1
        self.assertTrue(
            np.signbit(x[0, 2]) and x[0, 2] == 0
        )  # low nibble of byte 1 = column 2
        self.assertEqual((x[0, 4], x[0, 5]), (0.5, 1.0))
        self.assertEqual(int(neg0.sum()), 2)
        self.assertEqual(int(pos0.sum()), 32 - 4)

    def test_e4m3(self):
        self.assertEqual(float(zs.e4m3_to_float(np.array([0x38], np.uint8))[0]), 1.0)
        self.assertEqual(float(zs.e4m3_to_float(np.array([0x7E], np.uint8))[0]), 448.0)
        self.assertEqual(float(zs.e4m3_to_float(np.array([0xB8], np.uint8))[0]), -1.0)
        self.assertTrue(np.isnan(zs.e4m3_to_float(np.array([0x7F], np.uint8))[0]))

    def test_nvfp4_keeps_negative_zero(self):
        codes = np.zeros((1, 8), np.uint8)
        codes[0, 0] = 0x38  # col 0 = 8 (-0), col 1 = 3 (1.5)
        x = zs.nvfp4_dequant(codes, np.array([[0x38]], np.uint8), 2.0)
        self.assertTrue(np.signbit(x[0, 0]) and x[0, 0] == 0)
        self.assertEqual(float(x[0, 1]), 3.0)

    def test_alternate_rule_is_balanced_and_deterministic(self):
        x = signed_zero_matrix(64, 512, plus_only=True)
        pos0, neg0 = zs.zero_masks_from_float(x)
        self.assertEqual(int(neg0.sum()), 0)
        t1 = zs.target_negative("alternate", pos0, neg0)
        t2 = zs.target_negative("alternate", pos0, neg0)
        self.assertTrue(np.array_equal(t1, t2))
        z = int(pos0.sum())
        self.assertLessEqual(
            abs(int(t1.sum()) - z / 2), 64
        )  # at most one extra per row
        self.assertFalse((t1 & ~pos0).any())
        # chunking by rows gives the same bits when row0 is passed
        t3 = np.concatenate(
            [
                zs.target_negative("alternate", pos0[:20], neg0[:20], 0),
                zs.target_negative("alternate", pos0[20:], neg0[20:], 20),
            ]
        )
        self.assertTrue(np.array_equal(t1, t3))

    def test_auto_rule(self):
        x = signed_zero_matrix(8, 256)
        p, n = zs.zero_masks_from_float(x)
        self.assertEqual(zs.choose_rule("auto", p, n), "source")
        x = signed_zero_matrix(8, 256, plus_only=True)
        p, n = zs.zero_masks_from_float(x)
        self.assertEqual(zs.choose_rule("auto", p, n), "alternate")
        self.assertEqual(zs.choose_rule("source", p, n), "source")

    def test_apply_touches_only_sign_bits_of_zeros(self):
        n, k = 4, 512
        rb = k // 256 * 82
        rng = np.random.default_rng(3)
        blob = rng.integers(0, 256, n * rb, dtype=np.uint8)
        zero = rng.random((n, k)) < 0.2
        neg = rng.random((n, k)) < 0.5
        b2 = blob.copy()
        s1, c1 = zs.apply_zero_signs(b2, 22, n, k, zero, neg, rb)
        outside = ~zs.sign_field_mask(22, n, rb)
        self.assertTrue(np.array_equal(b2[outside], blob[outside]))
        self.assertEqual(int(outside.sum()), n * (k // 256) * (82 - 32))
        bits = (
            np.unpackbits(zs.sign_view(b2, 22, n, rb), axis=-1, bitorder="little")
            .reshape(n, k)
            .astype(bool)
        )
        old = (
            np.unpackbits(zs.sign_view(blob, 22, n, rb), axis=-1, bitorder="little")
            .reshape(n, k)
            .astype(bool)
        )
        self.assertTrue(np.array_equal(bits[zero], (neg & zero)[zero]))
        self.assertTrue(np.array_equal(bits[~zero], old[~zero]))
        b3 = b2.copy()
        self.assertEqual(
            zs.apply_zero_signs(b3, 22, n, k, zero, neg, rb), (0, 0)
        )  # idempotent
        self.assertGreater(s1 + c1, 0)

    def test_refusals(self):
        for t, word in (
            (16, "parity"),
            (17, "parity"),
            (18, "parity"),
            (153, "trellis"),
            (155, "trellis"),
            (158, "trellis"),
            (145, "no sign field"),
        ):
            ok, why = zs.repair_support(t)
            self.assertFalse(ok)
            self.assertIn(word, why)
        for t in (22, 21, 154):
            self.assertTrue(zs.repair_support(t)[0], zs.type_name(t))

    def test_iq3_kt_sign_bits_are_bit_major(self):
        # weight w of a block: qh[w % 32] (block byte 68 + w % 32), bit w // 32
        n, k = 2, 512
        rb = 4 + k // 256 * 100
        for w in (0, 1, 31, 32, 100, 255):
            zero = np.zeros((n, k), bool)
            zero[1, 256 + w] = True
            blob = np.zeros(n * rb, np.uint8)
            self.assertEqual(
                zs.apply_zero_signs(blob, 154, n, k, zero, zero, rb, 4), (1, 0)
            )
            byte = 1 * rb + 4 + 100 + 68 + w % 32
            self.assertEqual(list(np.nonzero(blob)[0]), [byte])
            self.assertEqual(int(blob[byte]), 1 << (w // 32))
            self.assertTrue(
                np.array_equal(
                    zs.unpack_sign_bytes(zs.sign_view(blob, 154, n, rb, 4), 154), zero
                )
            )
        outside = ~zs.sign_field_mask(154, n, rb, 4)
        self.assertEqual(int(outside.sum()), n * (4 + (k // 256) * (100 - 32)))

    def test_pack_and_unpack_are_inverse(self):
        rng = np.random.default_rng(5)
        for t in (22, 21, 154):
            m = rng.random((3, 768)) < 0.5
            s = zs.pack_sign_mask(m, t, 3, 768)
            self.assertEqual(s.shape, (3, 3, 32))
            self.assertTrue(np.array_equal(zs.unpack_sign_bytes(s, t), m))

    @unittest.skipUnless(
        os.path.isfile(REF_PATH), "no repository checkout around the tool"
    )
    def test_iq3_kt_layout_matches_the_reference_dequantiser(self):
        # ggml_quant_reference.py is bit-exact against libggml's dequantize_row_iq3_kt
        import importlib.util

        spec = importlib.util.spec_from_file_location("ggml_quant_reference", REF_PATH)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref  # its dataclasses look the module up
        spec.loader.exec_module(ref)
        n, k = 4, 512
        rows = ref.random_rows(154, n, k, np.random.default_rng(1))
        rb = rows.shape[1]
        y0 = ref.dequantize(154, rows, k)
        rng = np.random.default_rng(2)
        zero = rng.random((n, k)) < 0.15
        neg = rng.random((n, k)) < 0.5
        blob = rows.reshape(-1).copy()
        bits0 = zs.unpack_sign_bytes(zs.sign_view(blob, 154, n, rb, 4), 154)
        zs.apply_zero_signs(blob, 154, n, k, zero, neg, rb, 4)
        bits = zs.unpack_sign_bytes(zs.sign_view(blob, 154, n, rb, 4), 154)
        self.assertTrue(np.array_equal(bits[zero], neg[zero]))  # signs as asked
        self.assertTrue(np.array_equal(bits[~zero], bits0[~zero]))
        y = ref.dequantize(154, blob.reshape(n, rb), k)
        # exactly the weights whose bit changed change sign, nothing else moves
        flipped = bits != bits0
        self.assertGreater(int(flipped.sum()), 0)
        self.assertTrue(np.array_equal(y, np.where(flipped, -y0, y0)))

    def test_gguf_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.gguf")
            a = np.arange(64, dtype=np.float32).reshape(2, 32)
            b = (np.arange(256, dtype=np.uint16)).view(np.uint8).tobytes()
            zs.write_gguf(
                p,
                [("a", 0, (32, 2), a.tobytes()), ("blk.0.b", 30, (128, 2), b)],
                {"general.name": "t"},
            )
            kv, ts = zs.read_gguf(p)
            self.assertEqual(kv["general.name"], "t")
            self.assertEqual(
                (ts["a"].ne, ts["a"].nbytes, ts["a"].nrows), ((32, 2), 256, 2)
            )
            self.assertEqual(zs.pread(p, ts["blk.0.b"].offset, ts["blk.0.b"].nbytes), b)
            self.assertEqual(ts["a"].offset % 32, 0)


@needs_lib
class TestWithLibggml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = zs.GGML(LIB)

    def quant(self, t, x):
        return self.g.quantize(t, x, np.ones(x.shape[1], np.float32))

    def test_sign_bit_layout_matches_the_dequantiser(self):
        # flip the sign bit of weight i: exactly y[i] changes sign
        for t in (22, 21, 154):
            x = signed_zero_matrix(1, 256, zero_share=0.0, seed=t)
            q = self.quant(t, x)
            y0 = self.g.dequant(t, q, 1, 256)
            rb = self.g.row_size(t, 256)
            hdr = self.g.row_header(t)
            for i in (0, 7, 8, 31, 32, 100, 255):
                q2 = q.copy()
                one = np.zeros((1, 256), bool)
                one[0, i] = True
                zs.sign_view(q2, t, 1, rb, hdr)[...] ^= zs.pack_sign_mask(
                    one, t, 1, 256
                )
                y = self.g.dequant(t, q2, 1, 256)
                diff = np.nonzero(y[0] != y0[0])[0]
                self.assertEqual(list(diff), [i], zs.type_name(t))
                self.assertEqual(float(y[0, i]), -float(y0[0, i]))

    def test_repair_signed_zeros(self):
        for t in (22, 21, 154):
            n, k = 128, 1024
            x = signed_zero_matrix(n, k, seed=t)
            pos0, neg0 = zs.zero_masks_from_float(x)
            q = self.quant(t, x)
            y0 = self.g.dequant(t, q, n, k)
            before = zs.matrix_stats(x, y0, pos0, neg0)
            self.assertEqual(before["verdict"], "OFFSET")
            self.assertGreater(before["offset"], 0.02)
            # IQ3_KT decodes a few zeros to exactly 0; none comes out negative
            self.assertEqual(before["zeros_decoded_negative"], 0.0)
            rb, hdr = self.g.row_size(t, k), self.g.row_header(t)
            q2 = q.copy()
            zs.apply_zero_signs(
                q2,
                t,
                n,
                k,
                pos0 | neg0,
                zs.target_negative("source", pos0, neg0),
                rb,
                hdr,
            )
            y = self.g.dequant(t, q2, n, k)
            after = zs.matrix_stats(x, y, pos0, neg0)
            self.assertEqual(after["verdict"], "PASS", zs.type_name(t))
            self.assertLess(abs(after["offset"]), 0.003)
            self.assertEqual(after["nmse"], before["nmse"])
            z = pos0 | neg0
            self.assertTrue(np.array_equal(y[~z], y0[~z]))
            nz = y0 != 0
            self.assertTrue((y[neg0 & nz] < 0).all())
            self.assertTrue((y[pos0 & nz] > 0).all())
            outside = ~zs.sign_field_mask(t, n, rb, hdr)
            self.assertTrue(np.array_equal(q2[outside], q[outside]))

    def test_repair_plus_zero_only_with_the_alternating_rule(self):
        for t in (22, 21, 154):
            n, k = 128, 1024
            x = signed_zero_matrix(n, k, seed=10 + t, plus_only=True)
            pos0, neg0 = zs.zero_masks_from_float(x)
            q = self.quant(t, x)
            before = zs.matrix_stats(x, self.g.dequant(t, q, n, k), pos0, neg0)
            rb = self.g.row_size(t, k)
            zs.apply_zero_signs(
                q,
                t,
                n,
                k,
                pos0 | neg0,
                zs.target_negative("alternate", pos0, neg0),
                rb,
                self.g.row_header(t),
            )
            after = zs.matrix_stats(x, self.g.dequant(t, q, n, k), pos0, neg0)
            self.assertGreater(before["offset"], 0.02)
            self.assertLess(abs(after["offset"]), 0.003)
            dec = after["zeros_decoded_negative"] + after["zeros_decoded_positive"]
            self.assertLess(abs(after["zeros_decoded_negative"] / dec - 0.5), 0.01)
            self.assertLess(abs(after["rows_pos"] - 0.5), 0.15)
            self.assertEqual(after["nmse"], before["nmse"])

    def test_ggml_mxfp4_nibbles(self):
        n, k = 4, 256
        x = signed_zero_matrix(n, k, zero_share=0.0, seed=5)
        q = self.g.quantize(39, x)
        y = self.g.dequant(39, q, n, k)
        nib = zs.ggml_mxfp4_nibbles(q, n, k)
        nz = y != 0
        self.assertTrue(np.array_equal(np.signbit(y[nz]), (nib[nz] >= 8)))
        self.assertTrue(((nib == 0) | (nib == 8))[~nz].all())


def tiny_hotcold(d, g, ctype=22, n_experts=4, cold=(1, 3), hid=512, inter=256, seed=0):
    """A one-layer hot/cold toy: MXFP4 source experts, cold experts quantised plainly (zeros positive)."""
    src = os.path.join(d, "src")
    ck = os.path.join(d, "ck")
    os.makedirs(src)
    os.makedirs(ck)
    shapes = {"w1": (inter, hid), "w3": (inter, hid), "w2": (hid, inter)}
    tens, xs = {}, {}
    for E in range(n_experts):
        for w, (n, k) in shapes.items():
            x = signed_zero_matrix(n, k, seed=seed + 7 * E + ord(w[1]))
            packed, sc = mxfp4_encode(x)
            xs[(E, w)] = zs.mxfp4_dequant(packed, sc)
            base = f"layers.0.ffn.experts.{E}.{w}"
            tens[base + ".weight"] = ("U8", packed.shape, packed.tobytes())
            tens[base + ".scale"] = ("U8", sc.shape, sc.tobytes())
    write_safetensors(os.path.join(src, "model-00001-of-00001.safetensors"), tens)
    with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tens}}, f
        )
    cold_t, meta = {}, {}
    for w, (n, k) in shapes.items():
        rows = [g.quantize(ctype, xs[(E, w)], np.ones(k, np.float32)) for E in cold]
        rb = g.row_size(ctype, k)
        name = f"layers.0.ffn.experts_cold.{w}.qweight"
        cold_t[name] = ("U8", (len(cold), n, rb), b"".join(r.tobytes() for r in rows))
        meta[f"hotcold.ggml_type.{name}"] = str(ctype)
    fn = "model-cold-00001-of-00001.safetensors"
    write_safetensors(os.path.join(ck, fn), cold_t, meta)
    man = {
        "format": "dsv41-hotcold/1",
        "source": src,
        "n_routed_experts": n_experts,
        "cold_quant": {"ggml_type": {w: ctype for w in shapes}},
        "layers": {
            "0": {
                "hot": [e for e in range(n_experts) if e not in cold],
                "cold": list(cold),
            }
        },
        "cold_files": {
            "0": {
                "file": fn,
                "bytes": os.path.getsize(os.path.join(ck, fn)),
                "sha256": "",
            }
        },
        "cold_tensor_sha256": {},
    }
    with open(os.path.join(ck, "hotcold.json"), "w") as f:
        json.dump(man, f)
    with open(os.path.join(ck, "config.json"), "w") as f:
        f.write("{}")
    return src, ck


@needs_lib
class TestHotColdCommands(unittest.TestCase):
    def test_check_repair_check(self):
        g = zs.GGML(LIB)
        with tempfile.TemporaryDirectory() as d:
            src, ck = tiny_hotcold(d, g)
            base = ["--lib", LIB]
            rc, out = run_cli(
                zero_sign_check, ["hotcold", "--ckpt", ck, "--workers", "1"] + base
            )
            self.assertEqual(rc, 1, out)
            self.assertIn("OFFSET:2", out)
            new = os.path.join(d, "fixed")
            rc, out = run_cli(
                zero_sign_repair,
                ["hotcold", "--ckpt", ck, "--out", new, "--dry-run", "--workers", "1"]
                + base,
            )
            self.assertEqual(rc, 0, out)
            self.assertFalse(os.path.exists(new))
            rc, out = run_cli(
                zero_sign_repair,
                ["hotcold", "--ckpt", ck, "--out", new, "--workers", "1"] + base,
            )
            self.assertEqual(rc, 0, out)
            self.assertIn("repaired and verified", out)
            self.assertTrue(os.path.exists(os.path.join(new, "config.json")))
            self.assertEqual(
                os.stat(os.path.join(new, "config.json")).st_ino,
                os.stat(os.path.join(ck, "config.json")).st_ino,
            )
            rc, out = run_cli(
                zero_sign_check,
                ["hotcold", "--ckpt", new, "--source", src, "--workers", "2"] + base,
            )
            self.assertEqual(rc, 0, out)
            man = zs.load_json(os.path.join(new, "hotcold.json"))
            fp = os.path.join(new, man["cold_files"]["0"]["file"])
            self.assertEqual(zs.sha256_file(fp), man["cold_files"]["0"]["sha256"])
            ckp = zs.HotColdCheckpoint(new)
            for w, (off, ln, _) in ckp.tensors(0).items():
                self.assertEqual(
                    zs.sha256_range(fp, off, ln),
                    man["cold_tensor_sha256"][ckp.tensor_name(0, w)],
                )
            # the damaged original is untouched; repairing the repaired copy in place changes nothing
            rc, _ = run_cli(
                zero_sign_check, ["hotcold", "--ckpt", ck, "--workers", "1"] + base
            )
            self.assertEqual(rc, 1)
            before = zs.sha256_file(fp)
            rc, out = run_cli(
                zero_sign_repair,
                [
                    "hotcold",
                    "--ckpt",
                    new,
                    "--source",
                    src,
                    "--in-place",
                    "--workers",
                    "1",
                ]
                + base,
            )
            self.assertEqual(rc, 0, out)
            self.assertEqual(zs.sha256_file(fp), before)

    def test_check_repair_check_iq3_kt(self):
        # a cold tier of IQ3_KT (4-byte row header, bit-major signs) end to end
        g = zs.GGML(LIB)
        with tempfile.TemporaryDirectory() as d:
            src, ck = tiny_hotcold(d, g, ctype=154)
            base = ["--lib", LIB, "--workers", "1"]
            rc, out = run_cli(zero_sign_check, ["hotcold", "--ckpt", ck] + base)
            self.assertEqual(rc, 1, out)
            self.assertIn("IQ3_KT", out)
            new = os.path.join(d, "fixed")
            rc, out = run_cli(
                zero_sign_repair, ["hotcold", "--ckpt", ck, "--out", new] + base
            )
            self.assertEqual(rc, 0, out)
            rc, out = run_cli(
                zero_sign_check, ["hotcold", "--ckpt", new, "--source", src] + base
            )
            self.assertEqual(rc, 0, out)

    def test_refuses_parity_signed_cold_type(self):
        g = zs.GGML(LIB)
        with tempfile.TemporaryDirectory() as d:
            _, ck = tiny_hotcold(d, g, ctype=17)
            rc, out = run_cli(
                zero_sign_repair,
                ["hotcold", "--ckpt", ck, "--in-place", "--lib", LIB, "--workers", "1"],
            )
            self.assertEqual(rc, 2)
            self.assertIn("parity", out)
            rc, out = run_cli(
                zero_sign_check,
                ["hotcold", "--ckpt", ck, "--lib", LIB, "--workers", "1"],
            )
            self.assertEqual(rc, 1, out)  # IQ2_XS carries an offset too


@needs_lib
class TestGGUFCommands(unittest.TestCase):
    def test_bf16_and_mxfp4_sources(self):
        g = zs.GGML(LIB)
        with tempfile.TemporaryDirectory() as d:
            n, k = 64, 512
            x = signed_zero_matrix(n, k, seed=42)
            bf = (x.view(np.uint32) >> 16).astype(
                np.uint16
            )  # truncation keeps the sign of -0
            xb = zs.bf16_to_float(bf)  # quantise what the source holds
            mx = g.quantize(39, x)
            xm = g.dequant(39, mx, n, k)  # ggml's MXFP4 writes +0 only
            self.assertFalse((zs.ggml_mxfp4_nibbles(mx, n, k) == 8).any())
            src = os.path.join(d, "src.gguf")
            zs.write_gguf(
                src,
                [
                    ("blk.0.ffn_up.weight", 30, (k, n), bf.tobytes()),
                    ("blk.0.ffn_gate.weight", 39, (k, n), mx.tobytes()),
                    (
                        "blk.0.attn_norm.weight",
                        0,
                        (k,),
                        np.ones(k, np.float32).tobytes(),
                    ),
                ],
            )
            one = np.ones(k, np.float32)
            q = os.path.join(d, "q.gguf")
            zs.write_gguf(
                q,
                [
                    (
                        "blk.0.ffn_up.weight",
                        22,
                        (k, n),
                        g.quantize(22, xb, one).tobytes(),
                    ),
                    (
                        "blk.0.ffn_gate.weight",
                        21,
                        (k, n),
                        g.quantize(21, xm, one).tobytes(),
                    ),
                    (
                        "blk.0.attn_norm.weight",
                        0,
                        (k,),
                        np.ones(k, np.float32).tobytes(),
                    ),
                ],
            )
            rc, out = run_cli(
                zero_sign_check, ["gguf", "--quant", q, "--source", src, "--lib", LIB]
            )
            self.assertEqual(rc, 1, out)
            fixed = os.path.join(d, "fixed")
            rc, out = run_cli(
                zero_sign_repair,
                ["gguf", "--quant", q, "--source", src, "--out", fixed, "--lib", LIB],
            )
            self.assertEqual(rc, 0, out)
            self.assertIn(
                "blk.0.ffn_up.weight (IQ2_S, rule source)", out
            )  # BF16 with half -0
            self.assertIn(
                "blk.0.ffn_gate.weight (IQ3_S, rule alternate)", out
            )  # MXFP4 with +0 only
            rc, out = run_cli(
                zero_sign_check,
                [
                    "gguf",
                    "--quant",
                    os.path.join(fixed, "q.gguf"),
                    "--source",
                    src,
                    "--lib",
                    LIB,
                ],
            )
            self.assertEqual(rc, 0, out)
            self.assertEqual(
                os.path.getsize(os.path.join(fixed, "q.gguf")), os.path.getsize(q)
            )
            # in place gives the same bytes as --out
            rc, out = run_cli(
                zero_sign_repair,
                ["gguf", "--quant", q, "--source", src, "--in-place", "--lib", LIB],
            )
            self.assertEqual(rc, 0, out)
            self.assertEqual(
                zs.sha256_file(q), zs.sha256_file(os.path.join(fixed, "q.gguf"))
            )
            self.assertFalse(os.path.exists(q + zero_sign_repair.PARTIAL))

    def test_refuses_parity_type_but_reports_it(self):
        g = zs.GGML(LIB)
        with tempfile.TemporaryDirectory() as d:
            n, k = 32, 256
            x = signed_zero_matrix(n, k, seed=9)
            src, q = os.path.join(d, "s.gguf"), os.path.join(d, "q.gguf")
            zs.write_gguf(src, [("w", 0, (k, n), x.tobytes())])
            zs.write_gguf(
                q,
                [
                    (
                        "w",
                        16,
                        (k, n),
                        g.quantize(16, x, np.ones(k, np.float32)).tobytes(),
                    )
                ],
            )
            rc, out = run_cli(
                zero_sign_repair,
                ["gguf", "--quant", q, "--source", src, "--in-place", "--lib", LIB],
            )
            self.assertEqual(rc, 1, out)
            self.assertIn("NOT REPAIRED w", out)


if __name__ == "__main__":
    unittest.main()
