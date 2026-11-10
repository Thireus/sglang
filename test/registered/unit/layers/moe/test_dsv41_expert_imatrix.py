"""The DeepSeek-V4 expert importance matrix (layers/moe/expert_imatrix.py), on CPU.

* the MXFP4 numerics the a^2 recompute reads: E2M1 x E8M0 dequant against a
  scalar loop, the SWIZZLED_128x4 scale layout and its inverse;
* the x^2 accumulation against a plain per-route loop on synthetic routes
  (padded ids -1 and E, an expert routed twice by one token, experts with no
  route, chunked index_add);
* the chunked a^2 recompute against a direct float64 reference on tiny MXFP4
  experts (up-first and gate-first w13, swizzled and plain scales, one and
  several experts a chunk), the whole-output recompute of the self-check;
* the collector's forward filter, per-forward bookkeeping (layer misses, a
  layer seen twice, the ids hashes), its dump and meta block;
* the first moments (FIRST_MOMENTS): the sums of x and of a against plain
  loops, their arrays and the two checks in the dump (layer totals,
  Cauchy-Schwarz), a planted wrong sum that the check must catch, and the
  unchanged output with the switch off.

The FlashInfer swizzle check and the fused kernel are GPU-only and not run here.
"""

import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import torch

from sglang.srt.layers.moe import expert_imatrix as eim
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
LIMIT = 10.0


def _random_mxfp4(gen, n, rows, cols, scale_lo=120, scale_hi=130):
    codes = torch.randint(
        0, 256, (n, rows, cols // 2), generator=gen, dtype=torch.int64
    ).to(torch.uint8)
    scales = torch.randint(
        scale_lo, scale_hi, (n, rows, cols // 32), generator=gen, dtype=torch.int64
    ).to(torch.uint8)
    return codes, scales


def _dequant_loop(codes, scales):
    n, rows, half = codes.shape
    out = np.zeros((n, rows, half * 2), dtype=np.float64)
    c = codes.numpy()
    s = scales.numpy()
    for e in range(n):
        for r in range(rows):
            for j in range(half):
                b = int(c[e, r, j])
                out[e, r, 2 * j] = E2M1[b & 15] * 2.0 ** (
                    int(s[e, r, (2 * j) // 32]) - 127
                )
                out[e, r, 2 * j + 1] = E2M1[b >> 4] * 2.0 ** (
                    int(s[e, r, (2 * j + 1) // 32]) - 127
                )
    return out


def _tiny_layer(seed, num_experts, hidden, inter, *, up_first, swizzled):
    gen = torch.Generator().manual_seed(seed)
    w13c, w13s = _random_mxfp4(gen, num_experts, 2 * inter, hidden)
    w2c, w2s = _random_mxfp4(gen, num_experts, hidden, inter)
    dense13 = eim.dequant_mxfp4(w13c, w13s).double()
    dense2 = eim.dequant_mxfp4(w2c, w2s).double()
    if swizzled:
        w13s = eim.swizzle_128x4_reference(w13s)
        w2s = eim.swizzle_128x4_reference(w2s)
    src = eim.Mxfp4ExpertWeights(
        w13c, w13s, w2c, w2s, up_first=up_first, swizzled=swizzled
    )
    if up_first:
        gate, up = dense13[:, inter:], dense13[:, :inter]
    else:
        gate, up = dense13[:, :inter], dense13[:, inter:]
    return src, gate, up, dense2


def _routes(gen, tokens, top_k, num_experts):
    ids = torch.randint(
        0, num_experts - 2, (tokens, top_k), generator=gen, dtype=torch.int64
    ).to(torch.int32)
    ids[0, 0] = -1  # padded route
    ids[1, 1] = num_experts  # out of range: treated as padded
    ids[2, 0] = ids[2, 1] = 3  # one token routed twice to the same expert
    weights = torch.rand(tokens, top_k, generator=gen, dtype=torch.float32)
    return ids, weights  # experts E-2 and E-1 are never routed


def _swiglu64(gate, up, limit):
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return gate * torch.sigmoid(gate) * up


class TestMxfp4Numerics(CustomTestCase):
    def test_dequant_matches_scalar_loop(self):
        gen = torch.Generator().manual_seed(1)
        codes, scales = _random_mxfp4(gen, 2, 3, 64, 100, 150)
        got = eim.dequant_mxfp4(codes, scales).double().numpy()
        np.testing.assert_array_equal(got, _dequant_loop(codes, scales))

    def test_swizzle_round_trip_and_layout(self):
        gen = torch.Generator().manual_seed(2)
        sf = torch.randint(0, 256, (3, 256, 8), generator=gen, dtype=torch.int64).to(
            torch.uint8
        )
        sw = eim.swizzle_128x4_reference(sf)
        self.assertFalse(torch.equal(sw, sf))
        self.assertTrue(torch.equal(eim.unswizzle_128x4(sw), sf))
        # one element by the formula of get_sf_out_offset_128x4: m = 130, k = 5
        m, k = 130, 5
        off = (
            (m // 128) * (8 // 4 * 512)
            + (k // 4) * 512
            + (m % 32) * 16
            + ((m % 128) // 32) * 4
            + k % 4
        )
        self.assertEqual(int(sw[1].reshape(-1)[off]), int(sf[1, m, k]))

    def test_swizzle_refuses_padding_shapes(self):
        with self.assertRaises(ValueError):
            eim.unswizzle_128x4(torch.zeros(1, 100, 8, dtype=torch.uint8))


class TestAccumulateX2(CustomTestCase):
    def _loop(self, x, ids, weights, num_experts):
        h = x.shape[1]
        count = np.zeros(num_experts, np.int64)
        wsum = np.zeros(num_experts)
        s = np.zeros((num_experts, h))
        ws = np.zeros((num_experts, h))
        xd = x.double().numpy()
        for t in range(ids.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k])
                if e < 0 or e >= num_experts:
                    continue
                w = float(weights[t, k])
                count[e] += 1
                wsum[e] += w
                s[e] += xd[t] ** 2
                ws[e] += w * xd[t] ** 2
        return count, wsum, s, ws

    def test_against_plain_loop(self):
        gen = torch.Generator().manual_seed(3)
        num_experts, tokens, top_k, hidden = 12, 37, 4, 16
        x = (torch.randn(tokens, hidden, generator=gen) * 3).to(torch.bfloat16)
        ids, weights = _routes(gen, tokens, top_k, num_experts)
        for chunk in (5, 8192):
            count = torch.zeros(num_experts, dtype=torch.int64)
            wsum = torch.zeros(num_experts, dtype=torch.float64)
            s = torch.zeros(num_experts, hidden, dtype=torch.float64)
            ws = torch.zeros(num_experts, hidden, dtype=torch.float64)
            e, tok, w, invalid = eim.flat_routes(ids, weights, num_experts)
            eim.accumulate_x2(x, e, tok, w, count, wsum, s, ws, route_chunk=chunk)
            rc, rw, rs, rws = self._loop(x, ids, weights, num_experts)
            self.assertEqual(int(invalid), 2)
            np.testing.assert_array_equal(count.numpy(), rc)
            np.testing.assert_allclose(wsum.numpy(), rw, rtol=1e-12)
            np.testing.assert_allclose(s.numpy(), rs, rtol=1e-12)
            np.testing.assert_allclose(ws.numpy(), rws, rtol=1e-12)
            self.assertEqual(int(count[num_experts - 1]), 0)
            self.assertEqual(float(s[num_experts - 2].abs().sum()), 0.0)
            self.assertEqual(int(count.sum()), tokens * top_k - 2)

    def test_repeated_expert_counts_twice(self):
        x = torch.tensor([[1.0, 2.0]])
        ids = torch.tensor([[3, 3]], dtype=torch.int32)
        w = torch.tensor([[0.25, 0.5]])
        count = torch.zeros(5, dtype=torch.int64)
        wsum = torch.zeros(5, dtype=torch.float64)
        s = torch.zeros(5, 2, dtype=torch.float64)
        e, tok, ww, _ = eim.flat_routes(ids, w, 5)
        eim.accumulate_x2(x, e, tok, ww, count, wsum, s, None)
        self.assertEqual(int(count[3]), 2)
        self.assertEqual(s[3].tolist(), [2.0, 8.0])
        self.assertAlmostEqual(float(wsum[3]), 0.75)


class TestA2Recompute(CustomTestCase):
    def _reference(self, x, ids, weights, gate, up, num_experts, limit):
        inter = gate.shape[1]
        s = torch.zeros(num_experts, inter, dtype=torch.float64)
        ws = torch.zeros(num_experts, inter, dtype=torch.float64)
        xd = x.double()
        for t in range(ids.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k])
                if e < 0 or e >= num_experts:
                    continue
                a = _swiglu64(gate[e] @ xd[t], up[e] @ xd[t], limit)
                s[e] += a**2
                ws[e] += float(weights[t, k]) * a**2
        return s, ws

    def _run(
        self,
        *,
        up_first,
        swizzled,
        chunk_bytes,
        hidden=128,
        inter=128,
        num_experts=6,
        limit=LIMIT,
    ):
        src, gate, up, _ = _tiny_layer(
            7, num_experts, hidden, inter, up_first=up_first, swizzled=swizzled
        )
        gen = torch.Generator().manual_seed(8)
        x = (torch.randn(29, hidden, generator=gen) * 0.05).to(torch.bfloat16)
        ids, weights = _routes(gen, 29, 3, num_experts)
        s = torch.zeros(num_experts, inter, dtype=torch.float64)
        ws = torch.zeros(num_experts, inter, dtype=torch.float64)
        e, tok, w, _ = eim.flat_routes(ids, weights, num_experts)
        chunks = eim.accumulate_a2(x, e, tok, w, src, limit, s, ws, chunk_bytes)
        rs, rws = self._reference(x, ids, weights, gate, up, num_experts, limit)
        np.testing.assert_allclose(
            s.numpy(), rs.numpy(), rtol=2e-4, atol=1e-6 * float(rs.max())
        )
        np.testing.assert_allclose(
            ws.numpy(), rws.numpy(), rtol=2e-4, atol=1e-6 * float(rws.max())
        )
        self.assertEqual(float(s[num_experts - 1].abs().sum()), 0.0)
        return chunks, rs

    def test_one_expert_per_chunk_up_first_swizzled(self):
        chunks, rs = self._run(up_first=True, swizzled=True, chunk_bytes=1)
        self.assertEqual(chunks, 4)  # experts 0..3 routed, 4 and 5 never
        # the clamp is exercised: some pre-activation exceeds the limit
        self.assertGreater(float(rs.max()), 0.0)

    def test_two_experts_per_chunk_gate_first_plain(self):
        per = 2 * (2 * 128) * 128 * 4
        chunks, _ = self._run(up_first=False, swizzled=False, chunk_bytes=2 * per)
        self.assertEqual(chunks, 2)

    def test_clamp_binds(self):
        # a small limit makes the clamp decide most values; still equal to the reference
        self._run(up_first=True, swizzled=False, chunk_bytes=1 << 30, limit=0.05)

    def test_whole_output_recompute(self):
        num_experts, hidden, inter = 5, 128, 128
        src, gate, up, down = _tiny_layer(
            9, num_experts, hidden, inter, up_first=True, swizzled=True
        )
        gen = torch.Generator().manual_seed(10)
        x = (torch.randn(11, hidden, generator=gen) * 0.05).to(torch.bfloat16)
        ids, weights = _routes(gen, 11, 2, num_experts)
        got = eim.recompute_routed_output(x, ids, weights, src, LIMIT).double()
        ref = torch.zeros(11, hidden, dtype=torch.float64)
        for t in range(11):
            for k in range(2):
                e = int(ids[t, k])
                if 0 <= e < num_experts:
                    a = _swiglu64(gate[e] @ x[t].double(), up[e] @ x[t].double(), LIMIT)
                    ref[t] += float(weights[t, k]) * (down[e] @ a)
        rel = float((got - ref).norm() / ref.norm())
        self.assertLess(rel, 1e-5)


class TestCollector(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _collector(self, **kw):
        c = eim.ExpertImatrixCollector(
            os.path.join(self.tmp.name, "imx"),
            weighting="both",
            chunk_bytes=1 << 30,
            save_every=0,
            corpus_sha256="ab" * 32,
            pp_rank=1,
            sync_timing=False,
            **kw,
        )
        self.srcs = {}
        for layer in (4, 5):
            src, *_ = _tiny_layer(layer, 6, 128, 128, up_first=True, swizzled=False)
            self.srcs[layer] = src
            c.add_layer(layer, None, 6, 128, 128, "cpu", weights_src=src, limit=LIMIT)
        return c

    def _forward(self, c, gen, layers=(4, 5), skip=None, tokens=16):
        ids_in = torch.randint(0, 1000, (tokens,), generator=gen, dtype=torch.int64)
        fb = type("FB", (), {"extend_seq_lens_cpu": [tokens - 6, 6]})()
        c.begin(skip, ids_in, fb)
        x = (torch.randn(tokens, 128, generator=gen) * 0.05).to(torch.bfloat16)
        ids, w = _routes(gen, tokens, 3, 6)
        for layer in layers:
            c.observe(layer, x, StandardTopKOutput(w, ids, None))
        c.end()
        return ids_in, x, ids, w

    def test_counts_dump_and_meta(self):
        c = self._collector(expect_forwards=[1, 2])
        gen = torch.Generator().manual_seed(11)
        ids_in, x, ids, w = self._forward(c, gen)
        snap1 = c.path[: -len(".npz")] + ".f1.npz"
        self.assertTrue(os.path.exists(snap1))
        self._forward(c, gen, skip="mode DECODE")  # not counted
        self._forward(c, gen)  # the 2nd counted forward: the expected dump
        self.assertTrue(os.path.exists(c.path))
        self.assertTrue(os.path.exists(c.path[: -len(".npz")] + ".f2.npz"))
        self.assertEqual(
            json.loads(str(np.load(snap1, allow_pickle=False)["meta"]))["forwards"], 1
        )
        z = np.load(c.path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self.assertEqual(meta["format"], eim.FORMAT)
        self.assertEqual(meta["forwards"], 2)
        self.assertEqual(meta["tokens"], 32)
        self.assertEqual(meta["requests"], 4)
        self.assertEqual(meta["reason"], "expected")
        self.assertEqual(meta["corpus_sha256"], "ab" * 32)
        self.assertEqual(meta["layer_misses"], {})
        want = hashlib.sha256(ids_in[:10].numpy().astype("<i4").tobytes()).hexdigest()
        self.assertEqual(meta["forward_log"][0]["ids_sha256"][0], want)
        for layer in (4, 5):
            self.assertEqual(int(z[f"L{layer}.count"].sum()), 2 * (16 * 3 - 2))
            self.assertEqual(int(z[f"L{layer}.invalid_routes"]), 4)
            for name in ("sum_x2", "wsum_x2", "sum_a2", "wsum_a2", "wsum"):
                self.assertIn(f"L{layer}.{name}", z.files)
        self.assertEqual(z["L4.sum_a2"].shape, (6, 128))

    def test_layer_miss_and_double_call(self):
        c = self._collector()
        gen = torch.Generator().manual_seed(12)
        self._forward(c, gen, layers=(4,))
        self.assertEqual(c.layer_misses, {5: 1})
        c.begin(None, None, None)
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        ids, w = _routes(gen, 4, 3, 6)
        c.observe(4, x, StandardTopKOutput(w, ids, None))
        with self.assertRaises(eim.ExpertImatrixError):
            c.observe(4, x, StandardTopKOutput(w, ids, None))

    def test_x_only_and_weighting_switch(self):
        c = eim.ExpertImatrixCollector(
            self.tmp.name,
            weighting="count",
            x_only=True,
            save_every=0,
            sync_timing=False,
        )
        c.add_layer(0, None, 6, 128, 128, "cpu")
        st = c.layers[0]
        self.assertIsNone(st.wsum_x2)
        self.assertIsNone(st.sum_a2)
        self.assertIsNotNone(st.sum_x2)
        with self.assertRaises(eim.ExpertImatrixError):
            eim.ExpertImatrixCollector(self.tmp.name, weighting="mean")

    def test_self_check_passes_on_its_own_recompute(self):
        c = self._collector()
        gen = torch.Generator().manual_seed(13)
        c.begin(None, None, None)
        x = (torch.randn(8, 128, generator=gen) * 0.05).to(torch.bfloat16)
        ids, w = _routes(gen, 8, 3, 6)
        c.observe(4, x, StandardTopKOutput(w, ids, None))
        fused = eim.recompute_routed_output(x, ids, w, self.srcs[4], LIMIT).to(
            torch.bfloat16
        )
        c.check_output(4, fused)
        chk = c.layers[4].checks["output"]
        self.assertTrue(chk["pass"])
        self.assertLess(chk["rel_l2"], 1e-2)
        c.end()

    def test_batched_a2_equals_per_forward_and_dumps_flush(self):
        gen = torch.Generator().manual_seed(15)
        runs = {}
        for flush in (0, 40):
            c = eim.ExpertImatrixCollector(
                os.path.join(self.tmp.name, f"f{flush}"),
                weighting="both",
                save_every=0,
                sync_timing=False,
                a2_flush_tokens=flush,
                check_tokens=0,
            )
            src, *_ = _tiny_layer(4, 6, 128, 128, up_first=True, swizzled=False)
            c.add_layer(4, None, 6, 128, 128, "cpu", weights_src=src, limit=LIMIT)
            g = torch.Generator().manual_seed(16)
            for _ in range(
                5
            ):  # 5 forwards of 16 tokens: flushes at 48 and 96 tokens... and at the dump
                c.begin(None, None, None)
                x = (torch.randn(16, 128, generator=g) * 0.05).to(torch.bfloat16)
                ids, w = _routes(g, 16, 3, 6)
                c.observe(4, x, StandardTopKOutput(w, ids, None))
                c.end()
            if flush:
                self.assertEqual(
                    c.a2_flushes, 1
                )  # 48 tokens >= 40 once; 32 still pending
                self.assertEqual(c.layers[4].pending_tokens, 32)
                c.dump("sigusr1")
                meta = json.loads(str(np.load(c.path, allow_pickle=False)["meta"]))
                self.assertEqual(meta["a2_pending_tokens"], {"4": 32})
            c.dump("expected")
            self.assertEqual(c.layers[4].pending_tokens, 0)
            runs[flush] = c.layers[4]
        torch.testing.assert_close(runs[40].sum_a2, runs[0].sum_a2, rtol=1e-6, atol=0)
        torch.testing.assert_close(runs[40].wsum_a2, runs[0].wsum_a2, rtol=1e-6, atol=0)
        self.assertTrue(torch.equal(runs[40].count, runs[0].count))

    def test_padded_tokens_of_a_mismatched_batch_refused(self):
        c = self._collector()
        c.begin(None, None, None)
        ids, w = _routes(torch.Generator().manual_seed(14), 5, 3, 6)
        with self.assertRaises(eim.ExpertImatrixError):
            c.observe(4, torch.zeros(6, 128), StandardTopKOutput(w, ids, None))


class TestFirstMoments(CustomTestCase):
    """FIRST_MOMENTS: sum_x / sum_a (each route once), the layer totals and the checks written at a dump."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_sum_x_against_plain_loop(self):
        gen = torch.Generator().manual_seed(21)
        num_experts, tokens, top_k, hidden = 12, 37, 4, 16
        x = (torch.randn(tokens, hidden, generator=gen) * 3 + 0.7).to(torch.bfloat16)
        ids, weights = _routes(gen, tokens, top_k, num_experts)
        ref = np.zeros((num_experts, hidden))
        xd = x.double().numpy()
        for t in range(tokens):
            for k in range(top_k):
                e = int(ids[t, k])
                if 0 <= e < num_experts:
                    ref[e] += xd[t]
        for chunk in (5, 8192):
            count = torch.zeros(num_experts, dtype=torch.int64)
            wsum = torch.zeros(num_experts, dtype=torch.float64)
            s2 = torch.zeros(num_experts, hidden, dtype=torch.float64)
            s1 = torch.zeros(num_experts, hidden, dtype=torch.float64)
            e, tok, w, _ = eim.flat_routes(ids, weights, num_experts)
            eim.accumulate_x2(
                x, e, tok, w, count, wsum, s2, None, route_chunk=chunk, sum_x=s1
            )
            np.testing.assert_allclose(s1.numpy(), ref, rtol=1e-12, atol=1e-12)
            # the squares are unchanged by the extra sum
            s2b = torch.zeros(num_experts, hidden, dtype=torch.float64)
            eim.accumulate_x2(
                x,
                e,
                tok,
                w,
                torch.zeros_like(count),
                torch.zeros_like(wsum),
                s2b,
                None,
                route_chunk=chunk,
            )
            self.assertTrue(torch.equal(s2, s2b))
        self.assertEqual(
            s1[3].tolist(), (ref[3]).tolist()
        )  # expert 3: one token routed twice counts twice

    def test_sum_a_against_reference(self):
        num_experts, hidden, inter = 6, 128, 128
        src, gate, up, _ = _tiny_layer(
            22, num_experts, hidden, inter, up_first=True, swizzled=True
        )
        gen = torch.Generator().manual_seed(23)
        x = (torch.randn(29, hidden, generator=gen) * 0.05).to(torch.bfloat16)
        ids, weights = _routes(gen, 29, 3, num_experts)
        ref = torch.zeros(num_experts, inter, dtype=torch.float64)
        for t in range(29):
            for k in range(3):
                e = int(ids[t, k])
                if 0 <= e < num_experts:
                    ref[e] += _swiglu64(
                        gate[e] @ x[t].double(), up[e] @ x[t].double(), LIMIT
                    )
        for chunk_bytes in (1, 1 << 30):
            s2 = torch.zeros(num_experts, inter, dtype=torch.float64)
            s1 = torch.zeros(num_experts, inter, dtype=torch.float64)
            e, tok, w, _ = eim.flat_routes(ids, weights, num_experts)
            eim.accumulate_a2(x, e, tok, w, src, LIMIT, s2, None, chunk_bytes, sum_a=s1)
            np.testing.assert_allclose(
                s1.numpy(), ref.numpy(), rtol=2e-4, atol=1e-6 * float(ref.abs().max())
            )
            s2b = torch.zeros(num_experts, inter, dtype=torch.float64)
            eim.accumulate_a2(x, e, tok, w, src, LIMIT, s2b, None, chunk_bytes)
            self.assertTrue(torch.equal(s2, s2b))

    def _collector(self, name, **kw):
        c = eim.ExpertImatrixCollector(
            os.path.join(self.tmp.name, name),
            chunk_bytes=1 << 30,
            save_every=0,
            sync_timing=False,
            check_tokens=0,
            **kw,
        )
        src, *_ = _tiny_layer(4, 6, 128, 128, up_first=True, swizzled=False)
        c.add_layer(4, None, 6, 128, 128, "cpu", weights_src=src, limit=LIMIT)
        return c

    def _feed(self, c, forwards=3, seed=24):
        g = torch.Generator().manual_seed(seed)
        for _ in range(forwards):
            c.begin(None, None, None)
            x = (torch.randn(16, 128, generator=g) * 0.05 + 0.02).to(torch.bfloat16)
            ids, w = _routes(g, 16, 3, 6)
            c.observe(4, x, StandardTopKOutput(w, ids, None))
            c.end()

    def test_dump_arrays_and_checks_pass(self):
        for weighting, flush in (("count", 0), ("both", 40)):
            c = self._collector(
                f"fm-{weighting}",
                weighting=weighting,
                first_moments=True,
                a2_flush_tokens=flush,
            )
            self._feed(c)
            c.dump("expected")
            z = np.load(c.path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            self.assertTrue(meta["first_moments"])
            chk = meta["moment_checks"]["4"]
            self.assertTrue(chk["pass"], chk)
            self.assertLess(chk["x_total_rel"], 1e-12)
            self.assertLess(chk["x2_total_rel"], 1e-12)
            self.assertLessEqual(chk["cs_x_max"], 1.0 + 1e-9)
            self.assertLessEqual(chk["cs_a_max"], 1.0 + 1e-9)
            self.assertEqual(z["L4.sum_x"].shape, (6, 128))
            self.assertEqual(z["L4.sum_a"].shape, (6, 128))
            self.assertEqual(z["L4.tot_x"].shape, (128,))
            # the mean input is not zero here (the inputs carry an offset), and experts 4 and 5 have no route
            self.assertGreater(float(np.abs(z["L4.sum_x"][:4]).sum()), 0.0)
            self.assertEqual(float(np.abs(z["L4.sum_x"][4:]).sum()), 0.0)
            self.assertEqual(float(np.abs(z["L4.sum_a"][4:]).sum()), 0.0)

    def test_batched_sum_a_equals_per_forward(self):
        runs = {}
        for flush in (0, 40):
            c = self._collector(
                f"fl{flush}",
                weighting="count",
                first_moments=True,
                a2_flush_tokens=flush,
            )
            self._feed(c, forwards=5)
            c.dump("expected")
            runs[flush] = c.layers[4]
        torch.testing.assert_close(runs[40].sum_a, runs[0].sum_a, rtol=1e-6, atol=1e-12)
        self.assertTrue(torch.equal(runs[40].sum_x, runs[0].sum_x))

    def test_planted_wrong_sums_fail_the_check(self):
        # a sum added to another expert: the totals still agree, Cauchy-Schwarz does not
        c = self._collector("mut1", weighting="count", first_moments=True)
        self._feed(c)
        st = c.layers[4]
        st.sum_x[5] += st.sum_x[0]
        st.sum_x[0].zero_()
        chk = st.moment_check()
        self.assertFalse(chk["pass"])
        self.assertLess(chk["x_total_rel"], 1e-12)
        self.assertGreater(chk["cs_x_max"], 1.0 + 1e-6)
        # a route lost from the first moment: the totals disagree
        c = self._collector("mut2", weighting="count", first_moments=True)
        self._feed(c)
        st = c.layers[4]
        st.sum_x[1] *= 0.5
        chk = st.moment_check()
        self.assertFalse(chk["pass"])
        self.assertGreater(chk["x_total_rel"], 1e-6)
        # the same for a: a doubled sum breaks Cauchy-Schwarz
        c = self._collector("mut3", weighting="count", first_moments=True)
        self._feed(c)
        st = c.layers[4]
        st.sum_a *= 40.0
        chk = st.moment_check()
        self.assertFalse(chk["pass"])
        self.assertGreater(chk["cs_a_max"], 1.0 + 1e-6)
        # and the failed check is in the dump's meta block
        c.dump("expected")
        meta = json.loads(str(np.load(c.path, allow_pickle=False)["meta"]))
        self.assertFalse(meta["moment_checks"]["4"]["pass"])

    def test_x_only_keeps_sum_x_only_and_weight_only_is_refused(self):
        c = eim.ExpertImatrixCollector(
            os.path.join(self.tmp.name, "xo"),
            weighting="count",
            x_only=True,
            save_every=0,
            sync_timing=False,
            first_moments=True,
        )
        c.add_layer(0, None, 6, 128, 128, "cpu")
        st = c.layers[0]
        self.assertIsNotNone(st.sum_x)
        self.assertIsNone(st.sum_a)
        g = torch.Generator().manual_seed(25)
        c.begin(None, None, None)
        x = (torch.randn(16, 128, generator=g) * 0.05).to(torch.bfloat16)
        ids, w = _routes(g, 16, 3, 6)
        c.observe(0, x, StandardTopKOutput(w, ids, None))
        c.end()
        chk = st.moment_check()
        self.assertTrue(chk["pass"], chk)
        self.assertNotIn("cs_a_max", chk)
        with self.assertRaises(eim.ExpertImatrixError):
            eim.ExpertImatrixCollector(
                os.path.join(self.tmp.name, "w"), weighting="weight", first_moments=True
            )

    def test_switch_off_changes_nothing(self):
        c = self._collector("off", weighting="both")
        self._feed(c)
        c.dump("expected")
        z = np.load(c.path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self.assertFalse(meta["first_moments"])
        self.assertNotIn("moment_checks", meta)
        for name in ("sum_x", "sum_a", "tot_x", "tot_x2"):
            self.assertNotIn(f"L4.{name}", z.files)
        self.assertIsNone(c.layers[4].moment_check())
        # the squares are the same numbers with the switch on
        c2 = self._collector("on", weighting="both", first_moments=True)
        self._feed(c2)
        self.assertTrue(torch.equal(c.layers[4].sum_x2, c2.layers[4].sum_x2))
        self.assertTrue(torch.equal(c.layers[4].sum_a2, c2.layers[4].sum_a2))
        self.assertTrue(torch.equal(c.layers[4].wsum_a2, c2.layers[4].wsum_a2))


class TestPendingOnCpu(CustomTestCase):
    """PENDING_ON_CPU: the kept routed inputs are host copies; the sums are the same numbers."""

    def test_same_sums_and_no_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = {}
            for on in (False, True):
                c = eim.ExpertImatrixCollector(
                    os.path.join(tmp, f"p{int(on)}"),
                    weighting="count",
                    save_every=0,
                    sync_timing=False,
                    a2_flush_tokens=40,
                    check_tokens=0,
                    first_moments=True,
                    pending_on_cpu=on,
                )
                src, *_ = _tiny_layer(4, 6, 128, 128, up_first=True, swizzled=False)
                c.add_layer(4, None, 6, 128, 128, "cpu", weights_src=src, limit=LIMIT)
                g = torch.Generator().manual_seed(31)
                for _ in range(5):
                    c.begin(None, None, None)
                    x = (torch.randn(16, 128, generator=g) * 0.05).to(torch.bfloat16)
                    ids, w = _routes(g, 16, 3, 6)
                    c.observe(4, x, StandardTopKOutput(w, ids, None))
                    if c.layers[4].pending:
                        kept = c.layers[4].pending[-1][0]
                        self.assertEqual(kept.device.type, "cpu")
                        self.assertNotEqual(
                            kept.data_ptr(), x.data_ptr()
                        )  # a copy, not the engine's buffer
                    x.zero_()  # the engine reuses its buffer: the kept copy must not change
                    c.end()
                c.dump("expected")
                meta = json.loads(str(np.load(c.path, allow_pickle=False)["meta"]))
                self.assertEqual(meta["pending_on_cpu"], on)
                self.assertTrue(meta["moment_checks"]["4"]["pass"])
                runs[on] = c.layers[4]
            for name in ("sum_x2", "sum_a2", "sum_x", "sum_a"):
                self.assertTrue(
                    torch.equal(getattr(runs[True], name), getattr(runs[False], name)),
                    name,
                )
            self.assertGreater(float(runs[True].sum_a2.sum()), 0.0)


class TestWindowFilter(CustomTestCase):
    """WINDOWS: only forwards whose every request is a corpus window seen for the first time are counted, so a smoke
    load, the dump client's top-K probe (its first window, sent twice) and a mixed batch never shift the counts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        gen = torch.Generator().manual_seed(41)
        self.windows = [
            torch.randint(0, 129280, (16,), generator=gen, dtype=torch.int64)
            for _ in range(3)
        ]
        self.wfile = os.path.join(self.tmp.name, "w.jsonl")
        with open(self.wfile, "w") as f:
            for i, ids in enumerate(self.windows):
                f.write(
                    json.dumps({"id": f"chunk-{i:05d}", "ids": ids.tolist()}) + "\n"
                )

    def _collector(self, **kw):
        c = eim.ExpertImatrixCollector(
            os.path.join(self.tmp.name, "imx"),
            weighting="count",
            x_only=True,
            save_every=0,
            sync_timing=False,
            **kw,
        )
        c.add_layer(4, None, 6, 128, 128, "cpu")
        return c

    def _send(self, c, ids_list, gen):
        ids_in = torch.cat(ids_list)
        c.begin(
            None,
            ids_in,
            type(
                "FB", (), {"extend_seq_lens_cpu": [int(t.numel()) for t in ids_list]}
            )(),
        )
        x = (torch.randn(ids_in.numel(), 128, generator=gen) * 0.05).to(torch.bfloat16)
        ids, w = _routes(gen, ids_in.numel(), 3, 6)
        c.observe(4, x, StandardTopKOutput(w, ids, None))
        c.end()
        return x, ids

    def test_smoke_probe_and_repeat_are_skipped(self):
        c = self._collector(windows=[self.wfile], expect_forwards=[3])
        gen = torch.Generator().manual_seed(42)
        smoke = torch.randint(0, 129280, (64,), generator=gen, dtype=torch.int64)
        self._send(c, [smoke], gen)  # smoke: not a window
        self._send(c, [self.windows[0]], gen)  # the probe: window 0, counted
        self._send(c, [self.windows[0]], gen)  # window 0 again: skipped
        self._send(c, [self.windows[1], smoke[:16]], gen)  # a mixed batch: skipped
        c.begin(None, None, None)
        c.end()  # no ids: skipped
        self.assertEqual(c.forwards, 1)
        self.assertEqual(
            int(c.layers[4].count.sum()) + int(c.layers[4].invalid_routes), 16 * 3
        )  # the probe only
        self._send(c, [self.windows[1]], gen)
        self._send(c, [self.windows[2]], gen)  # the 3rd counted forward: the snapshot
        snap = c.path[: -len(".npz")] + ".f3.npz"
        self.assertTrue(os.path.exists(snap))
        meta = json.loads(str(np.load(snap, allow_pickle=False)["meta"]))
        self.assertEqual(meta["forwards"], 3)
        self.assertEqual(meta["tokens"], 48)
        self.assertEqual(meta["window_filter"]["windows"], 3)
        self.assertEqual(meta["window_filter"]["seen"], 3)
        self.assertEqual(
            meta["window_filter"]["skipped_forwards"],
            {"not a window": 2, "window seen before": 1, "no request ids": 1},
        )
        want = [
            hashlib.sha256(w.numpy().astype("<i4").tobytes()).hexdigest()
            for w in self.windows
        ]
        self.assertEqual([e["ids_sha256"][0] for e in meta["forward_log"]], want)
        self.assertEqual(c.layer_misses, {})

    def test_skipped_forward_is_not_accumulated(self):
        c = self._collector(windows=[self.wfile])
        gen = torch.Generator().manual_seed(43)
        x, ids = self._send(c, [self.windows[0]], gen)
        before = c.layers[4].sum_x2.clone(), c.layers[4].count.clone()
        self._send(c, [self.windows[0]], gen)
        self._send(c, [torch.randint(0, 9, (16,), generator=gen)], gen)
        self.assertTrue(torch.equal(c.layers[4].sum_x2, before[0]))
        self.assertTrue(torch.equal(c.layers[4].count, before[1]))

    def test_off_counts_everything_and_empty_file_refused(self):
        c = self._collector()
        gen = torch.Generator().manual_seed(44)
        self._send(c, [self.windows[0]], gen)
        self._send(c, [self.windows[0]], gen)
        self.assertEqual(c.forwards, 2)
        self.assertIsNone(c.meta()["window_filter"])
        empty = os.path.join(self.tmp.name, "empty.jsonl")
        open(empty, "w").close()
        with self.assertRaises(eim.ExpertImatrixError):
            self._collector(windows=[empty])


class TestInstallOnProductionMoE(CustomTestCase):
    """install() on production's DeepseekV2MoE built on CPU (hotcold_fixture's
    context: the real FusedMoE + Mxfp4FlashinferCutlassMoEMethod parameters and
    the real TopK module); the parameters hold random MXFP4 bytes with the
    scales in the SM120 swizzled layout, and the fused FlashInfer call is
    replaced by a float model of it built from the UNswizzled scales."""

    def setUp(self):
        from sglang.test import hotcold_fixture as fx

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(fx.cpu_moe_context())
        from sglang.srt.models import deepseek_v2

        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=fx.toy_hf_config(),
            layer_id=7,
            quant_config=fx.fp4_quant_config(),
            prefix="model.layers.7.mlp",
            is_deepseek_v4=True,
        )
        ex = self.mlp.experts
        gen = torch.Generator().manual_seed(21)
        n, two_i, half_h = ex.w13_weight.shape
        self.w13c, self.w13s = _random_mxfp4(gen, n, two_i, half_h * 2)
        self.w2c, self.w2s = _random_mxfp4(
            gen, n, ex.w2_weight.shape[1], ex.w2_weight.shape[2] * 2
        )
        with torch.no_grad():
            ex.w13_weight.data.view(torch.uint8).copy_(self.w13c)
            ex.w2_weight.data.view(torch.uint8).copy_(self.w2c)
            ex.w13_weight_scale_inv.data.view(torch.uint8).copy_(
                eim.swizzle_128x4_reference(self.w13s)
            )
            ex.w2_weight_scale_inv.data.view(torch.uint8).copy_(
                eim.swizzle_128x4_reference(self.w2s)
            )
        with (
            torch.no_grad()
        ):  # the router's parameters are torch.empty after construction
            for name, prm in self.mlp.gate.named_parameters():
                prm.copy_(
                    torch.randn(prm.shape, generator=gen).to(prm.dtype)
                    if prm.dim() > 1
                    else torch.zeros_like(prm)
                )
            for name, buf in self.mlp.gate.named_buffers():
                buf.zero_()
        ex._dsv4_mxfp4_backend = "flashinfer_cutlass_sm120"
        self.inter = two_i // 2
        self.dense13 = eim.dequant_mxfp4(self.w13c, self.w13s).double()
        self.dense2 = eim.dequant_mxfp4(self.w2c, self.w2s).double()
        self.limit = ex.moe_runner_config.swiglu_limit

        def fused_stand_in(hidden_states, topk_output, *a, **k):
            out = torch.zeros(
                hidden_states.shape[0], self.dense2.shape[1], dtype=torch.float64
            )
            for t in range(hidden_states.shape[0]):
                for j in range(topk_output.topk_ids.shape[1]):
                    e = int(topk_output.topk_ids[t, j])
                    xd = hidden_states[t].double()
                    up, gate = (
                        self.dense13[e, : self.inter] @ xd,
                        self.dense13[e, self.inter :] @ xd,
                    )
                    a = _swiglu64(gate, up, self.limit)
                    out[t] += float(topk_output.topk_weights[t, j]) * (
                        self.dense2[e] @ a
                    )
            return out.to(hidden_states.dtype)

        ex.forward = fused_stand_in
        self.model = torch.nn.ModuleDict(
            {"layer7": torch.nn.ModuleDict({"mlp": self.mlp})}
        )

    def test_hooks_collect_what_the_experts_see(self):
        c = eim.ExpertImatrixCollector(
            os.path.join(self.tmp.name, "imx"),
            weighting="both",
            save_every=0,
            sync_timing=False,
            check_tokens=4,
        )
        import atexit
        import signal

        self.addCleanup(signal.signal, signal.SIGUSR1, signal.getsignal(signal.SIGUSR1))
        eim.install(self.model, c)
        self.addCleanup(atexit.unregister, c.safe_dump)
        self.assertEqual(sorted(c.layers), [7])
        self.assertEqual(c.facts["quant_method"], "Mxfp4FlashinferCutlassMoEMethod")
        self.assertTrue(os.path.exists(c.pid_file))
        gen = torch.Generator().manual_seed(22)
        x = (torch.randn(9, self.dense13.shape[2], generator=gen) * 0.05).to(
            torch.bfloat16
        )
        c.begin(None, torch.arange(9), None)
        with torch.no_grad():
            topk_out = self.mlp.topk(x, self.mlp.gate(x))
            self.mlp.experts(x, topk_out)
        c.end()
        self.assertEqual(c.layer_misses, {})
        st = c.layers[7]
        self.assertEqual(
            st.checks["swizzle"][:11], "unavailable"
        )  # FlashInfer's swizzle is GPU-only
        self.assertTrue(st.checks["output"]["pass"], st.checks["output"])
        self.assertLess(st.checks["output"]["rel_l2"], 1e-2)
        ids, w = topk_out.topk_ids, topk_out.topk_weights
        s = torch.zeros_like(st.sum_a2)
        cnt = torch.zeros_like(st.count)
        xs = torch.zeros_like(st.sum_x2)
        for t in range(9):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                xd = x[t].double()
                a = _swiglu64(
                    self.dense13[e, self.inter :] @ xd,
                    self.dense13[e, : self.inter] @ xd,
                    self.limit,
                )
                s[e] += a**2
                xs[e] += xd**2
                cnt[e] += 1
        self.assertTrue(torch.equal(st.count, cnt))
        torch.testing.assert_close(st.sum_x2, xs, rtol=1e-12, atol=0)
        torch.testing.assert_close(st.sum_a2, s, rtol=2e-4, atol=1e-6 * float(s.max()))
        # a skipped forward (decode, warm-up, capture) adds nothing
        before = st.count.clone()
        c.begin("mode DECODE", None, None)
        with torch.no_grad():
            self.mlp.experts(x, self.mlp.topk(x, self.mlp.gate(x)))
        c.end()
        self.assertTrue(torch.equal(st.count, before))


if __name__ == "__main__":
    unittest.main()
