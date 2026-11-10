"""The prefill path of the hot/cold cold tier (IQ2_S) against the kernels it replaces, bit for bit.

* ``gguf_moe_mmq_staged`` (packed rows and activations staged through shared memory, one
  CTA of 16 warps) against ``gguf_moe_mmq`` in blocks of 16: the same tile values,
  products and float expression per element, so the output must be equal.
* The compact output of ``gguf_moe_mmq_staged`` and ``gguf_moe_mmq_wide`` (``sorted_rows``
  / ``out_rows``): the rows of the active routes, in route order, equal the dense rows.
* ``gguf_moe_cold_sum`` against ``apply_gguf_moe_weights`` + ``moe_sum`` on the dense
  layout: random values, and values chosen so that the order of the fp32 additions
  shows in the result.
* The grouped kernels with ``zero_init=False`` on allocator blocks prefilled with NaN
  against their zero-filled result: every row of an active route is written whole and no
  other row is touched.
* The compact call without a host read: ``gguf_moe_cold_rows`` (the row map and the cold
  route count, one launch) against its torch reference; ``gguf_moe_cold_swiglu`` (the
  clamped SwiGLU over the first rows, the count read on the device) against
  ``silu_and_mul_clamp``; ``gguf_moe_mmq_staged`` with ``row_count`` (the activation
  quantisation over the first rows) against the dense rows; in buffers of the dense
  sizes whose other rows must stay untouched.
* ``cold_moe`` end to end: the compact path (the default) against the dense path on the
  staged, the wide-block and the 16-route kernels, on NaN-prefilled blocks, eager and in
  a captured graph replayed on new routes, and with torch's synchronisation check armed
  (the call must not read anything on the host).
* ``gguf_moe_f16_staged`` (SGLANG_HOTCOLD_PREFILL_FLOAT: half-float products on the
  unquantised activations) against the float64 product of the dequantised rows: it is
  not bit-equal to the int8 kernels and must be closer to that product than they are.

Random valid IQ2_S rows at DeepSeek-V4.1 expert shapes; routes: random with negative
ids, none active, every route active, one expert on every route, a single token.
"""

import itertools
import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp
from sglang.kernels.ops.moe.gguf_moe_cold_rows import (
    _jit_gguf_moe_cold_rows_module,
    gguf_moe_cold_rows,
)
from sglang.kernels.ops.moe.gguf_moe_cold_sum import gguf_moe_cold_sum
from sglang.kernels.ops.moe.gguf_moe_cold_swiglu import (
    _jit_gguf_moe_cold_swiglu_module,
    gguf_moe_cold_swiglu,
)
from sglang.kernels.ops.moe.gguf_moe_f16_staged import gguf_moe_f16_staged
from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq
from sglang.kernels.ops.moe.gguf_moe_mmq_staged import (
    GGUF_MOE_MMQ_STAGED_BLOCKS,
    gguf_moe_mmq_staged,
)
from sglang.kernels.ops.moe.gguf_moe_mmq_wide import gguf_moe_mmq_wide
from sglang.kernels.ops.moe.gguf_moe_vec_multi import gguf_moe_vec_multi
from sglang.kernels.ops.moe.gguf_moe_weights import apply_gguf_moe_weights
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.layers.quantization import hotcold as hc
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.gguf_iq_reference import dequantize, random_rows
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="nightly", runner_config="1-gpu-large")

HIDDEN, INTER, TOP_K, EXPERTS = 5120, 2304, 6, 192
IQ2_S = 22
SWITCHES = (
    "SGLANG_HOTCOLD_COLD_KERNEL",
    "SGLANG_HOTCOLD_MMQ_MIN_TOKENS",
    "SGLANG_HOTCOLD_MMQ_WIDE",
    "SGLANG_HOTCOLD_MMQ_WIDE_MIN_TOKENS",
    "SGLANG_HOTCOLD_MMQ_STAGED",
    "SGLANG_HOTCOLD_PREFILL_COMPACT",
    "SGLANG_HOTCOLD_PREFILL_FLOAT",
)


def _env(**kv):
    """The switches as given, every other one at its default."""
    clean = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    clean.update({k: str(v) for k, v in kv.items()})
    return patch.dict(os.environ, clean, clear=True)


def _bits(a, b):
    """Equal as values and as bit patterns (so +0 against -0 and NaN count)."""
    return bool(torch.equal(a, b)) and bool(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
    )


def _route_cases(tokens, gen):
    """name -> int32 [tokens, TOP_K]; -1 = not a route of this tier."""
    rand = lambda: (
        torch.rand(tokens, EXPERTS, generator=gen, device="cuda")
        .argsort(1)[:, :TOP_K]
        .int()
    )
    drop = lambda p: torch.rand(tokens, TOP_K, generator=gen, device="cuda") < p
    mostly_off, third = rand(), rand()
    mostly_off[drop(0.87)] = -1
    third[drop(1 / 3)] = -1
    few = (
        torch.randint(
            0, 3, (tokens, TOP_K), generator=gen, device="cuda", dtype=torch.int32
        )
        * 95
    )  # experts 0, 95, 190
    rows = rand()
    rows[tokens // 2 :] = -1
    return {
        "87 % inactive": mostly_off,
        "a third inactive": third,
        "none active": torch.full(
            (tokens, TOP_K), -1, dtype=torch.int32, device="cuda"
        ),
        "all active": rand(),
        "one expert on every route": torch.full(
            (tokens, TOP_K), 77, dtype=torch.int32, device="cuda"
        ),
        "the last expert on every route": torch.full(
            (tokens, TOP_K), EXPERTS - 1, dtype=torch.int32, device="cuda"
        ),
        "three experts": few,
        "inactive rows": rows,
    }


def _projections(gen):
    """(name, weight, K, N, top_k of the op): gate/up reads a token row per route, down a row per route."""
    w13 = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
    yield "w13", w13, HIDDEN, 2 * INTER, TOP_K
    del w13
    w2 = random_rows(IQ2_S, (EXPERTS, HIDDEN), INTER, gen)
    yield "w2", w2, INTER, HIDDEN, 1


def _inputs(proj_top_k, ids, k, gen):
    tokens = ids.shape[0]
    if proj_top_k == TOP_K:
        return (
            torch.randn(tokens, k, generator=gen, device="cuda").bfloat16(),
            ids.contiguous(),
            tokens,
        )
    return (
        torch.randn(tokens * TOP_K, k, generator=gen, device="cuda").bfloat16(),
        ids.reshape(-1, 1).contiguous(),
        tokens * TOP_K,
    )


def _poison(shape):
    """Fill an allocator block of this shape with NaN and hand it back, so that the next
    torch.empty of the shape holds NaN; True when it does."""
    block = torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")
    del block
    probe = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    seen = bool(probe.isnan().all())
    del probe
    return seen


def _row_map(ids, sorted_ids=None, ntp=None):
    """(rows, sorted_rows, count as an int) by the torch reference of the row map."""
    if sorted_ids is None:
        sorted_ids = torch.arange(ids.numel(), dtype=torch.int32, device=ids.device)
    rows, sorted_rows, count = hc._compact_row_map_reference(ids, sorted_ids, ntp)
    return rows, sorted_rows, int(count)


def _mmq16(x, w, ids, n, top_k, rows):
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        ids, 16, w.shape[0], pad_expert_ids=True
    )
    return gguf_moe_mmq(x, w, sorted_ids, expert_ids, ntp, IQ2_S, n, top_k, rows, 16)


class TestGgufMoeMmqStaged(CustomTestCase):
    def _staged(self, x, w, ids, n, top_k, rows, wide, **kw):
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            ids, wide, w.shape[0], pad_expert_ids=True
        )
        return gguf_moe_mmq_staged(
            x, w, sorted_ids, expert_ids, ntp, IQ2_S, n, top_k, rows, wide, **kw
        )

    @torch.inference_mode()
    def test_equals_the_16_route_kernel(self):
        gen = torch.Generator(device="cuda").manual_seed(2301)
        compared = 0
        for proj, w, k, n, top_k in _projections(gen):
            for tokens in (1, 65, 300, 2048):
                for label, ids in _route_cases(tokens, gen).items():
                    x, pids, rows = _inputs(top_k, ids, k, gen)
                    ref = _mmq16(x, w, pids, n, top_k, rows)
                    self.assertEqual(int(ref[pids.reshape(-1) < 0].count_nonzero()), 0)
                    for wide in GGUF_MOE_MMQ_STAGED_BLOCKS:
                        got = self._staged(x, w, pids, n, top_k, rows, wide)
                        self.assertTrue(
                            _bits(got, ref),
                            f"{proj} T={tokens} {label} staged={wide}: "
                            f"max|d|={float((got.float() - ref.float()).abs().max()):.3e}",
                        )
                        compared += int(ref.count_nonzero() > 0)
        self.assertGreater(compared, 80)  # a pass on empty outputs would not count

    @torch.inference_mode()
    def test_an_odd_expert_offset_and_the_last_rows(self):
        # w2's rows are 738 bytes: every other row starts 2 bytes into a 32-bit word, and so does every
        # other QK_K block of a w13 row. One expert, routes on the first and the last expert only.
        gen = torch.Generator(device="cuda").manual_seed(2302)
        for proj, w, k, n, top_k in _projections(gen):
            tokens = 70
            ids = torch.where(
                torch.rand(tokens, TOP_K, generator=gen, device="cuda") < 0.5,
                torch.zeros((), dtype=torch.int32, device="cuda"),
                torch.full((), EXPERTS - 1, dtype=torch.int32, device="cuda"),
            ).int()
            x, pids, rows = _inputs(top_k, ids, k, gen)
            self.assertTrue(
                _bits(
                    self._staged(x, w, pids, n, top_k, rows, 64),
                    _mmq16(x, w, pids, n, top_k, rows),
                ),
                proj,
            )

    @torch.inference_mode()
    def test_graph_replay_on_new_routes(self):
        gen = torch.Generator(device="cuda").manual_seed(2303)
        w = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        tokens = 128
        cases = list(_route_cases(tokens, gen).values())
        x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
        ids = cases[0].clone()
        self._staged(x, w, ids, 2 * INTER, TOP_K, tokens, 64)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._staged(x, w, ids, 2 * INTER, TOP_K, tokens, 64)
        for new_ids in cases:
            x.copy_(
                torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
            )
            ids.copy_(new_ids)
            graph.replay()
            self.assertTrue(_bits(out, _mmq16(x, w, ids, 2 * INTER, TOP_K, tokens)))


class TestCompactOutputs(CustomTestCase):
    """sorted_rows / out_rows: one row per active route, in route order."""

    @torch.inference_mode()
    def test_compact_rows_equal_the_dense_rows(self):
        gen = torch.Generator(device="cuda").manual_seed(2304)
        ops = {"staged": gguf_moe_mmq_staged, "wide": gguf_moe_mmq_wide}
        compared = 0
        for proj, w, k, n, top_k in _projections(gen):
            for tokens in (1, 65, 300):
                for label, ids in _route_cases(tokens, gen).items():
                    x, pids, rows = _inputs(top_k, ids, k, gen)
                    ref = _mmq16(x, w, pids, n, top_k, rows)
                    active = pids.reshape(-1) >= 0
                    sorted_ids, expert_ids, ntp = moe_align_block_size(
                        pids, 64, w.shape[0], pad_expert_ids=True
                    )
                    crow, sorted_rows, count = _row_map(pids, sorted_ids, ntp)
                    self.assertEqual(count, int(active.sum()))
                    if count == 0:
                        continue
                    for name, op in ops.items():
                        self.assertTrue(_poison((count, n)))
                        got = op(
                            x,
                            w,
                            sorted_ids,
                            expert_ids,
                            ntp,
                            IQ2_S,
                            n,
                            top_k,
                            rows,
                            64,
                            zero_init=False,
                            sorted_rows=sorted_rows,
                            out_rows=count,
                        )
                        self.assertEqual(tuple(got.shape), (count, n))
                        self.assertTrue(
                            _bits(got, ref[active]), f"{name} {proj} T={tokens} {label}"
                        )
                        compared += 1
                        if top_k == 1:
                            # the down projection of the compact path: the input holds the active rows only and
                            # the sorted ids are the compact rows themselves
                            got = op(
                                x[active].contiguous(),
                                w,
                                sorted_rows,
                                expert_ids,
                                ntp,
                                IQ2_S,
                                n,
                                1,
                                count,
                                64,
                                zero_init=False,
                            )
                            self.assertTrue(
                                _bits(got, ref[active]),
                                f"{name} {proj} T={tokens} {label} (compact input)",
                            )
        self.assertGreater(compared, 60)


class TestNoZeroFill(CustomTestCase):
    """zero_init=False on NaN-prefilled blocks: the active rows are written whole, nothing else is."""

    def _check(self, got, ref, active, what):
        self.assertTrue(_bits(got[active], ref[active]), f"{what}: active rows")
        self.assertTrue(
            bool(got[~active].isnan().all()),
            f"{what}: a row of an inactive route was written",
        )

    @torch.inference_mode()
    def test_grouped_ops_on_poisoned_memory(self):
        gen = torch.Generator(device="cuda").manual_seed(2305)
        for proj, w, k, n, top_k in _projections(gen):
            for tokens in (6, 54, 65, 300):
                for label, ids in _route_cases(tokens, gen).items():
                    x, pids, rows = _inputs(top_k, ids, k, gen)
                    active = pids.reshape(-1) >= 0
                    shape = (rows * top_k, n)
                    for name, block, op, extra in (
                        ("vec_multi", 4, gguf_moe_vec_multi, {"lanes": 16}),
                        ("mmq_wide", 64, gguf_moe_mmq_wide, {}),
                        ("mmq_staged", 64, gguf_moe_mmq_staged, {}),
                    ):
                        sorted_ids, expert_ids, ntp = moe_align_block_size(
                            pids, block, w.shape[0], pad_expert_ids=True
                        )
                        args = (
                            x,
                            w,
                            sorted_ids,
                            expert_ids,
                            ntp,
                            IQ2_S,
                            n,
                            top_k,
                            rows,
                            block,
                        )
                        ref = op(*args, zero_init=True, **extra)
                        self.assertEqual(int(ref[~active].count_nonzero()), 0)
                        self.assertTrue(_poison(shape))
                        got = op(*args, zero_init=False, **extra)
                        self._check(
                            got, ref, active, f"{name} {proj} T={tokens} {label}"
                        )


class TestColdSum(CustomTestCase):
    def _dense(self, y_dense, weights, ids):
        from sgl_kernel import moe_sum

        tokens, top_k = ids.shape
        y = y_dense.clone()
        apply_gguf_moe_weights(y, weights, ids)
        out = torch.empty((tokens, y.shape[1]), dtype=y.dtype, device=y.device)
        moe_sum(y.view(tokens, top_k, -1), out)
        return out

    def _compact(self, y_dense, weights, ids):
        rows, _, _ = _row_map(ids)
        return gguf_moe_cold_sum(
            y_dense[ids.reshape(-1) >= 0].contiguous(), rows, weights
        )

    @torch.inference_mode()
    def test_equals_weights_and_moe_sum(self):
        gen = torch.Generator(device="cuda").manual_seed(2306)
        for tokens in (1, 7, 300):
            for label, ids in _route_cases(tokens, gen).items():
                if int((ids >= 0).sum()) == 0:
                    continue
                y = (
                    torch.randn(tokens * TOP_K, HIDDEN, generator=gen, device="cuda")
                    * 3
                ).bfloat16()
                # widen the magnitudes: the order of the fp32 additions then matters for some elements
                y = (
                    y.float()
                    * torch.pow(
                        2.0,
                        torch.randint(
                            -20, 21, y.shape, generator=gen, device="cuda"
                        ).float(),
                    )
                ).bfloat16()
                y[torch.rand(y.shape, generator=gen, device="cuda") < 0.02] = -0.0
                weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
                self.assertTrue(
                    _bits(self._compact(y, weights, ids), self._dense(y, weights, ids)),
                    f"T={tokens} {label}",
                )

    @torch.inference_mode()
    def test_the_order_of_the_additions(self):
        # 1 + 2^-8 sits half way between two bf16 values and two terms of 2^-24 tip it only when
        # they meet before they reach the large terms: every placement of the four values over
        # the six slots tells the trees apart. The weights are 1, so the products are exact.
        vals = [1.0, 2.0**-8, 2.0**-24, 2.0**-24]
        places = list(itertools.permutations(range(TOP_K), 4))
        tokens = len(places)
        y = torch.zeros(tokens, TOP_K, 8, device="cuda")
        ids = torch.full((tokens, TOP_K), -1, dtype=torch.int32, device="cuda")
        for t, slots in enumerate(places):
            for v, s in zip(vals, slots):
                y[t, s] = v
                ids[t, s] = s
        # the two slots without a value: once inactive, once active with a zero
        for fill in (False, True):
            use = ids.clone()
            if fill:
                use[use < 0] = 0
            yb = y.view(tokens * TOP_K, 8).bfloat16()
            weights = torch.ones(tokens, TOP_K, device="cuda")
            dense = self._dense(yb, weights, use)
            self.assertGreater(
                int((dense.float() > 1.0).sum()), 0
            )  # some placements round up
            self.assertGreater(int((dense.float() == 1.0).sum()), 0)  # and some do not
            self.assertTrue(
                _bits(self._compact(yb, weights, use), dense),
                f"inactive slots filled: {fill}",
            )

    @torch.inference_mode()
    def test_tokens_without_an_active_route_are_zero(self):
        gen = torch.Generator(device="cuda").manual_seed(2307)
        tokens = 9
        ids = torch.full((tokens, TOP_K), -1, dtype=torch.int32, device="cuda")
        ids[4, 2] = 5
        y = torch.randn(1, HIDDEN, generator=gen, device="cuda").bfloat16()
        rows, _, count = _row_map(ids)
        self.assertEqual(count, 1)
        self.assertTrue(_poison((tokens, HIDDEN)))
        out = gguf_moe_cold_sum(
            y, rows, torch.full((tokens, TOP_K), 0.5, device="cuda")
        )
        self.assertEqual(
            int(out[[0, 1, 2, 3, 5, 6, 7, 8]].view(torch.int16).count_nonzero()), 0
        )  # +0, not -0
        self.assertTrue(_bits(out[4], (y[0].float() * 0.5).bfloat16()))


class TestCompactWithoutAHostRead(CustomTestCase):
    """The kernels of the compact call that take the cold route count from the device."""

    @torch.inference_mode()
    def test_row_map_equals_its_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(2320)
        checked = 0
        for tokens in (1, 7, 65, 300, 2048, 8192):
            for label, ids in _route_cases(tokens, gen).items():
                for block in (32, 64):
                    sorted_ids, expert_ids, ntp = moe_align_block_size(
                        ids, block, EXPERTS, pad_expert_ids=True
                    )
                    want_rows, want_sorted, want_count = hc._compact_row_map_reference(
                        ids, sorted_ids, ntp
                    )
                    # the columns past num_tokens_post_padded are not written: they must keep what they held
                    rows, sorted_rows, count = gguf_moe_cold_rows(ids, sorted_ids, ntp)
                    used = int(ntp[0])
                    what = f"T={tokens} {label} block={block}"
                    self.assertTrue(bool(torch.equal(rows, want_rows)), what)
                    self.assertEqual(count.tolist(), want_count.tolist(), what)
                    self.assertEqual(
                        (count.dtype, tuple(count.shape)), (torch.int32, (1,))
                    )
                    self.assertTrue(
                        bool(torch.equal(sorted_rows[:used], want_sorted[:used])), what
                    )
                    self.assertEqual(int(count), int((ids >= 0).sum()), what)
                    checked += 1
        self.assertGreater(checked, 90)

    @torch.inference_mode()
    def test_row_map_leaves_the_unused_columns(self):
        # the kernel on buffers of the test's own, marked beforehand: what it does not write keeps the mark
        gen = torch.Generator(device="cuda").manual_seed(2321)
        mark = 0x5A5A5A5A
        for label in ("87 % inactive", "none active", "all active"):
            ids = _route_cases(300, gen)[label]
            sorted_ids, expert_ids, ntp = moe_align_block_size(
                ids, 64, EXPERTS, pad_expert_ids=True
            )
            used = int(ntp[0])
            self.assertLess(used, sorted_ids.numel())
            rows = torch.full_like(ids, mark)
            sorted_rows = torch.full_like(sorted_ids, mark)
            count = torch.full((1,), mark, dtype=torch.int32, device="cuda")
            _jit_gguf_moe_cold_rows_module().run(
                ids, sorted_ids, ntp, rows, sorted_rows, count
            )
            want_rows, want_sorted, want_count = hc._compact_row_map_reference(
                ids, sorted_ids, ntp
            )
            self.assertTrue(
                bool(torch.equal(rows, want_rows)), label
            )  # every row of the map is written
            self.assertEqual(count.tolist(), want_count.tolist(), label)
            self.assertTrue(
                bool(torch.equal(sorted_rows[:used], want_sorted[:used])), label
            )
            self.assertTrue(
                bool((sorted_rows[used:] == mark).all()),
                f"{label}: a column past num_tokens_post_padded was written",
            )

    @torch.inference_mode()
    def test_counted_swiglu_equals_the_clamped_swiglu(self):
        gen = torch.Generator(device="cuda").manual_seed(2322)
        limit = 10.0
        compared = 0
        run = _jit_gguf_moe_cold_swiglu_module(torch.bfloat16).run
        for rows in (
            1,
            7,
            300,
            4096,
            9000,
        ):  # more rows than the largest grid (4,096): a CTA takes several
            h = torch.randn(rows, 2 * INTER, generator=gen, device="cuda") * 4
            # magnitudes on both sides of the limit and far beyond it, exact zeros of both signs
            h = h * torch.pow(
                2.0,
                torch.randint(-12, 6, h.shape, generator=gen, device="cuda").float(),
            )
            h[torch.rand(h.shape, generator=gen, device="cuda") < 0.01] = 0.0
            h[torch.rand(h.shape, generator=gen, device="cuda") < 0.01] = -0.0
            h = h.bfloat16()
            ref = torch.empty((rows, INTER), dtype=torch.bfloat16, device="cuda")
            silu_and_mul_clamp(h, ref, limit)
            self.assertGreater(int((ref.float().abs() > 0).sum()), rows * INTER // 2)
            for count in sorted(
                {
                    0,
                    1,
                    rows // 3,
                    rows - 1,
                    rows,
                    min(rows, 4095),
                    min(rows, 4097),
                    rows + 5,
                }
            ):
                # the kernel on an output of the test's own, NaN beforehand: the rows past the count keep it
                got = torch.full(
                    (rows, INTER), float("nan"), dtype=torch.bfloat16, device="cuda"
                )
                run(
                    h,
                    got,
                    torch.tensor([count], dtype=torch.int32, device="cuda"),
                    limit,
                )
                n = min(count, rows)  # a count past the buffer is clamped to it
                self.assertTrue(_bits(got[:n], ref[:n]), f"rows={rows} count={count}")
                self.assertTrue(
                    bool(got[n:].isnan().all()),
                    f"rows={rows} count={count}: a row past the count was written",
                )
                compared += int(n > 0)
            # the op: the same rows in a buffer it allocates
            got = gguf_moe_cold_swiglu(
                h, torch.tensor([rows], dtype=torch.int32, device="cuda"), limit
            )
            self.assertTrue(_bits(got, ref), f"rows={rows}: the op")
        self.assertGreater(compared, 15)

    @torch.inference_mode()
    def test_counted_projection_equals_the_dense_rows(self):
        gen = torch.Generator(device="cuda").manual_seed(2323)
        w = random_rows(IQ2_S, (EXPERTS, HIDDEN), INTER, gen)
        compared = 0
        for tokens in (1, 65, 300, 2048):
            for label, ids in _route_cases(tokens, gen).items():
                x, pids, rows = _inputs(
                    1, ids, INTER, gen
                )  # the down projection: one input row per route
                active = pids.reshape(-1) >= 0
                ref = _mmq16(x, w, pids, HIDDEN, 1, rows)
                for wide in GGUF_MOE_MMQ_STAGED_BLOCKS:
                    sorted_ids, expert_ids, ntp = moe_align_block_size(
                        pids, wide, EXPERTS, pad_expert_ids=True
                    )
                    crow, sorted_rows, count = gguf_moe_cold_rows(pids, sorted_ids, ntp)
                    n = int(count)
                    # the compact input in a buffer of the dense size, NaN past the cold rows
                    a = torch.full(
                        (rows, INTER), float("nan"), dtype=torch.bfloat16, device="cuda"
                    )
                    a[:n] = x[active]
                    self.assertTrue(_poison((rows, HIDDEN)))
                    got = gguf_moe_mmq_staged(
                        a,
                        w,
                        sorted_rows,
                        expert_ids,
                        ntp,
                        IQ2_S,
                        HIDDEN,
                        1,
                        rows,
                        wide,
                        zero_init=False,
                        row_count=count,
                    )
                    what = f"T={tokens} {label} staged={wide}"
                    self.assertEqual(tuple(got.shape), (rows, HIDDEN))
                    self.assertTrue(_bits(got[:n], ref[active]), what)
                    self.assertTrue(
                        bool(got[n:].isnan().all()),
                        f"{what}: a row past the count was written",
                    )
                    compared += int(n > 0)
        self.assertGreater(compared, 40)
        with self.assertRaises(ValueError):
            gguf_moe_mmq_staged(
                a,
                w,
                sorted_rows,
                expert_ids,
                ntp,
                IQ2_S,
                HIDDEN,
                1,
                rows,
                64,
                sorted_rows=sorted_rows,
                out_rows=rows,
                row_count=count,
            )


class TestColdMoePrefillPaths(CustomTestCase):
    """cold_moe above the MMQ threshold: the default (staged kernel, compact rows by a device count) against the dense layouts."""

    PATHS = {
        "16-route, dense": {
            "SGLANG_HOTCOLD_MMQ_STAGED": 0,
            "SGLANG_HOTCOLD_MMQ_WIDE": 0,
        },
        "wide, dense": {"SGLANG_HOTCOLD_MMQ_STAGED": 0},
        "staged, dense": {"SGLANG_HOTCOLD_PREFILL_COMPACT": 0},
        "staged 32, compact": {"SGLANG_HOTCOLD_MMQ_STAGED": 32},
        "staged, compact (default)": {},
    }

    def _weights(self, gen):
        return random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen), random_rows(
            IQ2_S, (EXPERTS, HIDDEN), INTER, gen
        )

    @torch.inference_mode()
    def test_paths_are_bit_equal(self):
        gen = torch.Generator(device="cuda").manual_seed(2308)
        w13, w2 = self._weights(gen)
        compared = 0
        for tokens in (65, 300, 2048):
            for label, ids in _route_cases(tokens, gen).items():
                x = (
                    torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 0.02
                ).bfloat16()
                weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
                ref = None
                for name, kv in self.PATHS.items():
                    with _env(**kv):
                        for shape in (
                            (tokens * TOP_K, 2 * INTER),
                            (tokens * TOP_K, INTER),
                            (tokens * TOP_K, HIDDEN),
                            (tokens, HIDDEN),
                        ):
                            _poison(shape)
                        out = hc.cold_moe(
                            x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
                        )
                    if ref is None:
                        ref = out
                        continue
                    self.assertTrue(
                        _bits(out, ref),
                        f"T={tokens} {label}: {name}: max|d|={float((out.float() - ref.float()).abs().max()):.3e}",
                    )
                    compared += int(ref.count_nonzero() > 0)
        self.assertGreater(compared, 80)

    @torch.inference_mode()
    def test_paths_are_bit_equal_with_the_limit_inside_the_activations(self):
        # With these inputs the gate/up values stay far below test_paths_are_bit_equal's limit of 10.0, so the clamp never
        # acts there and a path that handed the SwiGLU a wrong limit would still pass. Here the limit is the median |gate/up|
        # of the active routes (the dense 16-route rows): about half of the values are clamped, the result is shown to
        # differ from the one under 10.0, and every path must still give the dense staged path's bits.
        gen = torch.Generator(device="cuda").manual_seed(2315)
        w13, w2 = self._weights(gen)
        clamped = 0
        for tokens in (65, 300, 2048):
            cases = _route_cases(tokens, gen)
            for label in (
                "87 % inactive",
                "a third inactive",
                "none active",
                "all active",
                "one expert on every route",
            ):
                ids = cases[label]
                x = (
                    torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 0.02
                ).bfloat16()
                weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
                active = ids.reshape(-1) >= 0
                limit = 0.05
                if bool(active.any()):
                    h = _mmq16(x, w13, ids.contiguous(), 2 * INTER, TOP_K, tokens)
                    limit = float(h[active].float().abs().median())
                with _env(SGLANG_HOTCOLD_PREFILL_COMPACT=0):
                    loose = hc.cold_moe(
                        x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
                    )
                    ref = hc.cold_moe(
                        x, w13, w2, weights, ids, IQ2_S, IQ2_S, limit, kernel="mmq"
                    )
                if bool(active.any()):
                    self.assertFalse(
                        _bits(ref, loose),
                        f"T={tokens} {label}: the limit {limit:.4g} does not act",
                    )
                    clamped += 1
                for name, kv in self.PATHS.items():
                    with _env(**kv):
                        for shape in (
                            (tokens * TOP_K, 2 * INTER),
                            (tokens * TOP_K, INTER),
                            (tokens * TOP_K, HIDDEN),
                            (tokens, HIDDEN),
                        ):
                            _poison(shape)
                        out = hc.cold_moe(
                            x, w13, w2, weights, ids, IQ2_S, IQ2_S, limit, kernel="mmq"
                        )
                    self.assertTrue(
                        _bits(out, ref),
                        f"T={tokens} {label} limit {limit:.4g}: {name}: max|d|={float((out.float() - ref.float()).abs().max()):.3e}",
                    )
        self.assertEqual(clamped, 12)

    @torch.inference_mode()
    def test_the_default_is_the_compact_call(self):
        gen = torch.Generator(device="cuda").manual_seed(2313)
        w13, w2 = self._weights(gen)
        tokens = 70
        ids = _route_cases(tokens, gen)["87 % inactive"]
        x = (
            torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 0.02
        ).bfloat16()
        weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
        seen = []
        spy = lambda *a, _f=hc._compact_row_map: (seen.append(1), _f(*a))[1]
        with patch.object(hc, "_compact_row_map", spy):
            with _env():
                hc.cold_moe(x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq")
                self.assertEqual(len(seen), 1)
            for kv in (
                {"SGLANG_HOTCOLD_PREFILL_COMPACT": 0},
                {"SGLANG_HOTCOLD_MMQ_STAGED": 0},
                {"SGLANG_HOTCOLD_PREFILL_FLOAT": 1},
                {"SGLANG_HOTCOLD_COLD_KERNEL": "legacy"},
            ):
                with _env(**kv):
                    hc.cold_moe(
                        x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
                    )
                    self.assertEqual(len(seen), 1, kv)  # the dense layout
            with _env():
                hc.cold_moe(
                    x, w13, w2, weights, ids, IQ2_S, IQ2_S, None, kernel="mmq"
                )  # no SwiGLU limit: dense
                self.assertEqual(len(seen), 1)
                hc.cold_moe(
                    x[:54], w13, w2, weights[:54], ids[:54], IQ2_S, IQ2_S, 10.0
                )  # a decode batch: untouched
                self.assertEqual(len(seen), 1)

    @torch.inference_mode()
    def test_the_compact_call_reads_nothing_on_the_host(self):
        # torch raises on a synchronising call (.item(), a host copy of a device tensor) while the check is armed
        gen = torch.Generator(device="cuda").manual_seed(2314)
        w13, w2 = self._weights(gen)
        for tokens in (65, 2048):
            for label in ("87 % inactive", "none active", "all active"):
                ids = _route_cases(tokens, gen)[label]
                x = (
                    torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 0.02
                ).bfloat16()
                weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
                with _env(SGLANG_HOTCOLD_PREFILL_COMPACT=0):
                    ref = hc.cold_moe(
                        x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
                    )
                with _env():
                    hc.cold_moe(
                        x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
                    )  # imports and builds first
                    before = torch.cuda.get_sync_debug_mode()
                    torch.cuda.set_sync_debug_mode("error")
                    try:
                        out = hc.cold_moe(
                            x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
                        )
                    finally:
                        torch.cuda.set_sync_debug_mode(before)
                self.assertTrue(_bits(out, ref), f"T={tokens} {label}")
        # the check itself: a host read of a device value does raise while it is armed
        torch.cuda.set_sync_debug_mode("error")
        try:
            with self.assertRaises(RuntimeError):
                int(torch.ones(1, device="cuda").sum())
        finally:
            torch.cuda.set_sync_debug_mode(before)

    @torch.inference_mode()
    def test_a_captured_call_replays_on_new_routes(self):
        # the compact call has no host read and no shape that depends on the routes: it is captured as it is
        gen = torch.Generator(device="cuda").manual_seed(2309)
        w13, w2 = self._weights(gen)
        for tokens in (96, 700):
            cases = _route_cases(tokens, gen)
            x = (
                torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 0.02
            ).bfloat16()
            weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
            ids = cases["all active"].clone()
            seen = []
            spy = lambda *a, _f=hc._compact_row_map: (seen.append(1), _f(*a))[1]
            with _env():
                call = lambda: hc.cold_moe(
                    x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
                )
                call()
                torch.cuda.synchronize()
                for shape in (
                    (tokens * TOP_K, 2 * INTER),
                    (tokens * TOP_K, INTER),
                    (tokens * TOP_K, HIDDEN),
                ):
                    _poison(shape)
                graph = torch.cuda.CUDAGraph()
                with patch.object(hc, "_compact_row_map", spy):
                    with torch.cuda.graph(graph):
                        out = call()
                self.assertEqual(len(seen), 1)  # the captured call is the compact one
                for label, new_ids in list(cases.items()) + [
                    ("all active", cases["all active"])
                ]:
                    ids.copy_(new_ids)
                    x.copy_(
                        (
                            torch.randn(tokens, HIDDEN, generator=gen, device="cuda")
                            * 0.02
                        ).bfloat16()
                    )
                    graph.replay()
                    with _env(SGLANG_HOTCOLD_PREFILL_COMPACT=0):
                        ref = call()
                    self.assertTrue(_bits(out, ref), f"T={tokens} replay on '{label}'")


class TestGgufMoeF16Staged(CustomTestCase):
    """The half-float path against the float64 product of the dequantised rows."""

    def _reference(self, x, w, pids, k, n, top_k):
        """float64 [routes, n]: W[expert] @ x[route // top_k] for the active routes, zero elsewhere."""
        flat = pids.reshape(-1).long()
        ref = torch.zeros(flat.numel(), n, dtype=torch.float64, device="cuda")
        for expert in flat[flat >= 0].unique().tolist():
            routes = (flat == expert).nonzero().view(-1)
            dense = dequantize(w[expert], IQ2_S, k)  # float64 [n, k]
            ref[routes] = x[routes // top_k].double() @ dense.T
        return ref

    def _ulp(self, got, ref):
        """|got - bf16(ref)| in bf16 spacings at the reference's magnitude, or at a sixteenth of its row's
        RMS where the element is smaller than that (a sum of 5,120 products that cancels: the fp32 error
        of the sum is relative to the terms, not to their small total)."""
        want = ref.float().bfloat16().float()
        floor = ref.pow(2).mean(dim=1, keepdim=True).sqrt().float() / 16
        spacing = torch.pow(
            2.0,
            torch.floor(torch.log2(torch.maximum(want.abs(), floor).clamp_min(1e-30)))
            - 7,
        )
        return (got.float() - want).abs() / spacing

    def _f16(self, x, w, pids, n, top_k, rows, **kw):
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            pids, 64, w.shape[0], pad_expert_ids=True
        )
        return gguf_moe_f16_staged(
            x, w, sorted_ids, expert_ids, ntp, IQ2_S, n, top_k, rows, 64, **kw
        ), (sorted_ids, expert_ids, ntp)

    @torch.inference_mode()
    def test_matches_the_float64_product(self):
        gen = torch.Generator(device="cuda").manual_seed(2310)
        checked = 0
        for proj, w, k, n, top_k in _projections(gen):
            for tokens in (1, 65, 300):
                cases = _route_cases(tokens, gen)
                for label in (
                    "87 % inactive",
                    "a third inactive",
                    "three experts",
                    "the last expert on every route",
                    "none active",
                ):
                    ids = cases[label]
                    # twelve experts at most (0, 17, .., 187): the reference dequantises every expert in use
                    ids = torch.where(ids >= 0, ids % 12 * 17, ids)
                    x, pids, rows = _inputs(top_k, ids, k, gen)
                    active = pids.reshape(-1) >= 0
                    got, _ = self._f16(x, w, pids, n, top_k, rows)
                    self.assertEqual(
                        int(got[~active].count_nonzero()),
                        0,
                        f"{proj} T={tokens} {label}",
                    )
                    if not bool(active.any()):
                        continue
                    ref = self._reference(x, w, pids, k, n, top_k)
                    ulp = self._ulp(got[active], ref[active])
                    # fp32 sums of exact products: the bf16 rounding of the float64 result, except where that
                    # result sits on a rounding boundary of bf16 to within the fp32 error of the sum
                    self.assertLessEqual(
                        float(ulp.max()), 1.0, f"{proj} T={tokens} {label}"
                    )
                    self.assertGreater(
                        float((ulp == 0).float().mean()),
                        0.99,
                        f"{proj} T={tokens} {label}",
                    )
                    rel = float(
                        (got[active].double() - ref[active]).norm() / ref[active].norm()
                    )
                    int8 = _mmq16(x, w, pids, n, top_k, rows)
                    rel8 = float(
                        (int8[active].double() - ref[active]).norm()
                        / ref[active].norm()
                    )
                    self.assertLess(
                        rel,
                        rel8,
                        f"{proj} T={tokens} {label}: half-float {rel:.2e} against int8 {rel8:.2e}",
                    )
                    self.assertFalse(
                        _bits(got, int8)
                    )  # another arithmetic: behind its switch
                    checked += 1
        self.assertGreater(checked, 20)

    @torch.inference_mode()
    def test_rows_beyond_the_fp16_range_and_the_compact_output(self):
        gen = torch.Generator(device="cuda").manual_seed(2311)
        for proj, w, k, n, top_k in _projections(gen):
            tokens = 70
            ids = _route_cases(tokens, gen)["a third inactive"]
            ids = torch.where(ids >= 0, ids % 12 * 17, ids)
            x, pids, rows = _inputs(top_k, ids, k, gen)
            # every third row far beyond the largest fp16 value, one row tiny, one row zero
            x = x.float()
            x[::3] *= 3.0e7
            x[1] *= 1.0e-9
            x[2] = 0
            x = x.bfloat16()
            active = pids.reshape(-1) >= 0
            got, (sorted_ids, expert_ids, ntp) = self._f16(x, w, pids, n, top_k, rows)
            self.assertTrue(bool(torch.isfinite(got).all()), proj)
            ref = self._reference(x, w, pids, k, n, top_k)
            big = (torch.arange(pids.numel(), device="cuda") // top_k % 3 == 0) & active
            self.assertGreater(int(big.sum()), 0)
            self.assertLessEqual(float(self._ulp(got[big], ref[big]).max()), 1.0, proj)
            rel = float(
                (got[active].double() - ref[active]).norm() / ref[active].norm()
            )
            self.assertLess(rel, 3e-3, proj)  # the bf16 rounding of the output
            crow, sorted_rows, count = _row_map(pids, sorted_ids, ntp)
            self.assertTrue(_poison((count, n)))
            compact = gguf_moe_f16_staged(
                x,
                w,
                sorted_ids,
                expert_ids,
                ntp,
                IQ2_S,
                n,
                top_k,
                rows,
                64,
                zero_init=False,
                sorted_rows=sorted_rows,
                out_rows=count,
            )
            self.assertTrue(
                _bits(compact, got[active]),
                f"{proj}: the compact output against the dense one",
            )

    @torch.inference_mode()
    def test_cold_moe_under_the_switch(self):
        gen = torch.Generator(device="cuda").manual_seed(2312)
        w13 = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        w2 = random_rows(IQ2_S, (EXPERTS, HIDDEN), INTER, gen)
        for tokens in (65, 300):
            ids = _route_cases(tokens, gen)["87 % inactive"]
            x = (
                torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 0.02
            ).bfloat16()
            weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
            call = lambda: hc.cold_moe(
                x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0, kernel="mmq"
            )
            with _env():
                off = call()
            with _env(SGLANG_HOTCOLD_PREFILL_FLOAT=1):
                on = call()
                with _env(
                    SGLANG_HOTCOLD_PREFILL_FLOAT=1, SGLANG_HOTCOLD_PREFILL_COMPACT=0
                ):
                    dense = call()
            with _env(SGLANG_HOTCOLD_PREFILL_FLOAT=1, SGLANG_HOTCOLD_MMQ_STAGED=0):
                wide = (
                    call()
                )  # the switch needs the staged kernel: without it the int8 path
            self.assertFalse(_bits(on, off))
            self.assertTrue(
                _bits(on, dense), "the half-float path: compact against dense"
            )
            self.assertTrue(_bits(wide, off))
            rel = float((on.double() - off.double()).norm() / off.double().norm())
            self.assertLess(
                rel, 0.05
            )  # the same quantity up to the int8 rounding of the activations
        # a decode batch does not see the switch
        tokens = 54
        ids = _route_cases(tokens, gen)["87 % inactive"]
        x = (
            torch.randn(tokens, HIDDEN, generator=gen, device="cuda") * 0.02
        ).bfloat16()
        weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
        with _env():
            off = hc.cold_moe(x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0)
        with _env(SGLANG_HOTCOLD_PREFILL_FLOAT=1):
            on = hc.cold_moe(x, w13, w2, weights, ids, IQ2_S, IQ2_S, 10.0)
        self.assertTrue(_bits(on, off))


if __name__ == "__main__":
    unittest.main()
