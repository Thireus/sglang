"""The grouped GGUF MoE kernels for IQ2_S against the kernels they replace, bit for bit.

* ``gguf_moe_vec_multi`` (several routes of one expert per block, lanes of CTAs) against
  ``gguf_moe_a8_vec``: the same Q8_1 activations, integer sums and float expression per
  row, so the output must be equal, at every block width and lane count, and the vector
  kernel itself must not depend on its rows per CTA (ggml's 1 against the default).
* ``gguf_moe_mmq_wide`` (up to 16 / 32 / 64 / 128 routes of one expert per CTA, only the
  column groups in use multiplied, 32 columns of activations per pass) against
  ``gguf_moe_mmq`` in blocks of 16.

* The decode pack of a hot/cold layer, each kernel against the launches it replaces:
  ``gguf_moe_split`` against the add and the two ``index_select``, and
  ``gguf_moe_split_align`` against those and ``moe_align_block_size`` (every cold
  route once in a block of its expert, and the multi-route kernel's output from
  either grouping);
  ``gguf_moe_cold_swiglu_q8`` against ``silu_and_mul_clamp`` followed by the vector
  kernel's own Q8_1 quantisation (the activations, the Q8_1 blocks against the ones
  that kernel's quantisation launch writes, and the down projection fed with them);
  ``gguf_moe_cold_sum`` with dense rows and the fold against ``apply_gguf_moe_weights``,
  ``moe_sum`` and ``add_`` (also on inputs where the order of the additions decides
  the result); and the whole decode chain of a layer, earlier launches against the
  pack, on buffers the allocator hands back filled with NaN.

Random valid IQ2_S rows at DeepSeek-V4.1 expert shapes, the cold tier's 192 experts;
routes: random with negative ids, none active, every route active, one expert on every
route (many blocks and passes of one expert), the last expert, a single token; eager and
CUDA-graph replay on new routes.
"""

import itertools
import unittest

import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp
from sglang.kernels.ops.moe.gguf_moe_cold_sum import gguf_moe_cold_sum
from sglang.kernels.ops.moe.gguf_moe_cold_swiglu import gguf_moe_cold_swiglu_q8
from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq
from sglang.kernels.ops.moe.gguf_moe_mmq_wide import (
    GGUF_MOE_MMQ_WIDE_BLOCKS,
    gguf_moe_mmq_wide,
)
from sglang.kernels.ops.moe.gguf_moe_split import (
    gguf_moe_split,
    gguf_moe_split_align,
)
from sglang.kernels.ops.moe.gguf_moe_vec import gguf_moe_a8_vec
from sglang.kernels.ops.moe.gguf_moe_vec_multi import (
    GGUF_MOE_VEC_MULTI_DEFAULT_ROWS,
    GGUF_MOE_VEC_MULTI_ROUTES,
    _jit_gguf_moe_vec_multi_module,
    gguf_moe_vec_multi,
)
from sglang.kernels.ops.moe.gguf_moe_weights import apply_gguf_moe_weights
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.gguf_iq_reference import random_rows
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="nightly", runner_config="1-gpu-large")

HIDDEN, INTER, TOP_K, EXPERTS = 5120, 2304, 6, 192
IQ2_S = 22


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


class TestGgufMoeVecMulti(CustomTestCase):
    def _multi(self, x, w, ids, n, top_k, rows, routes, lanes):
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            ids, routes, w.shape[0], pad_expert_ids=True
        )
        return gguf_moe_vec_multi(
            x,
            w,
            sorted_ids,
            expert_ids,
            ntp,
            IQ2_S,
            n,
            top_k,
            rows,
            routes,
            lanes=lanes,
        )

    @torch.inference_mode()
    def test_equals_the_vector_kernel(self):
        gen = torch.Generator(device="cuda").manual_seed(2201)
        for proj, w, k, n, top_k in _projections(gen):
            for tokens in (1, 2, 6, 54, 64):
                for label, ids in _route_cases(tokens, gen).items():
                    x, pids, rows = _inputs(top_k, ids, k, gen)
                    ref = gguf_moe_a8_vec(x, w, pids, top_k, IQ2_S, n, rows)
                    one_row = gguf_moe_a8_vec(
                        x, w, pids, top_k, IQ2_S, n, rows, rows_per_block=1
                    )
                    self.assertTrue(
                        torch.equal(one_row, ref),
                        f"{proj} T={tokens} {label}: rows per CTA 1 vs default",
                    )
                    self.assertEqual(int(ref[pids.reshape(-1) < 0].count_nonzero()), 0)
                    for routes in GGUF_MOE_VEC_MULTI_ROUTES:
                        for lanes in (1, 5, 16, 4096):
                            got = self._multi(x, w, pids, n, top_k, rows, routes, lanes)
                            self.assertTrue(
                                torch.equal(got, ref),
                                f"{proj} T={tokens} {label} routes={routes} lanes={lanes}: "
                                f"max|d|={float((got.float() - ref.float()).abs().max()):.3e}",
                            )

    @torch.inference_mode()
    def test_graph_replay_on_new_routes(self):
        gen = torch.Generator(device="cuda").manual_seed(2202)
        w = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        for tokens in (6, 54):
            cases = list(_route_cases(tokens, gen).values())
            x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
            ids = cases[0].clone()
            self._multi(x, w, ids, 2 * INTER, TOP_K, tokens, 4, 16)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._multi(x, w, ids, 2 * INTER, TOP_K, tokens, 4, 16)
            for new_ids in (
                cases
            ):  # from 87 % inactive to every route active: the grid does not change
                x.copy_(
                    torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
                )
                ids.copy_(new_ids)
                graph.replay()
                self.assertTrue(
                    torch.equal(
                        out, gguf_moe_a8_vec(x, w, ids, TOP_K, IQ2_S, 2 * INTER, tokens)
                    )
                )


class TestGgufMoeMmqWide(CustomTestCase):
    def _mmq16(self, x, w, ids, n, top_k, rows):
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            ids, 16, w.shape[0], pad_expert_ids=True
        )
        return gguf_moe_mmq(
            x, w, sorted_ids, expert_ids, ntp, IQ2_S, n, top_k, rows, 16
        )

    def _wide(self, x, w, ids, n, top_k, rows, wide):
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            ids, wide, w.shape[0], pad_expert_ids=True
        )
        return gguf_moe_mmq_wide(
            x, w, sorted_ids, expert_ids, ntp, IQ2_S, n, top_k, rows, wide
        )

    @torch.inference_mode()
    def test_equals_the_16_route_kernel(self):
        gen = torch.Generator(device="cuda").manual_seed(2203)
        for proj, w, k, n, top_k in _projections(gen):
            for tokens in (1, 65, 300, 2048):
                for label, ids in _route_cases(tokens, gen).items():
                    x, pids, rows = _inputs(top_k, ids, k, gen)
                    ref = self._mmq16(x, w, pids, n, top_k, rows)
                    self.assertEqual(int(ref[pids.reshape(-1) < 0].count_nonzero()), 0)
                    for wide in GGUF_MOE_MMQ_WIDE_BLOCKS:
                        got = self._wide(x, w, pids, n, top_k, rows, wide)
                        self.assertTrue(
                            torch.equal(got, ref),
                            f"{proj} T={tokens} {label} wide={wide}: "
                            f"max|d|={float((got.float() - ref.float()).abs().max()):.3e}",
                        )

    @torch.inference_mode()
    def test_graph_replay_on_new_routes(self):
        gen = torch.Generator(device="cuda").manual_seed(2204)
        w = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        tokens = 128
        cases = list(_route_cases(tokens, gen).values())
        x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
        ids = cases[0].clone()
        self._wide(x, w, ids, 2 * INTER, TOP_K, tokens, 64)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._wide(x, w, ids, 2 * INTER, TOP_K, tokens, 64)
        for new_ids in cases:
            x.copy_(
                torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
            )
            ids.copy_(new_ids)
            graph.replay()
            self.assertTrue(
                torch.equal(out, self._mmq16(x, w, ids, 2 * INTER, TOP_K, tokens))
            )


SWIGLU_LIMIT = 10.0


def _bits(t):
    """The bit patterns: +0 and -0 differ, a NaN equals itself."""
    return t.contiguous().view(torch.int16 if t.dtype != torch.int32 else torch.int32)


def _poison(*shapes):
    """Fill allocator blocks of these (shape, dtype) with NaN (floats) or -1 (integers) and
    hand them back: the next torch.empty of such a size holds them, so a row a kernel
    reads without having written it shows."""
    blocks = [
        torch.full(
            shape,
            float("nan") if dtype.is_floating_point else -1,
            dtype=dtype,
            device="cuda",
        )
        for shape, dtype in shapes
    ]
    del blocks


def _down_with_its_own_quantisation(a, w2, sorted_ids, expert_ids, ntp, routes=4):
    """(y, quantized): the multi-route down projection of ``a`` through the kernel's own
    quantize_q8_1 launch, with the Q8_1 scratch it filled (the op keeps that scratch to
    itself, so the JIT entry is called here with a scratch of the test's)."""
    rows = a.shape[0]
    y = torch.zeros((rows, HIDDEN), dtype=a.dtype, device=a.device)
    padded = (a.shape[1] + 511) // 512 * 512
    quantized = torch.empty(
        (rows, padded // 32 * 9), dtype=torch.int32, device=a.device
    )
    _jit_gguf_moe_vec_multi_module(
        a.dtype, IQ2_S, routes, GGUF_MOE_VEC_MULTI_DEFAULT_ROWS, True
    ).project(a, w2, sorted_ids, expert_ids, ntp, y, quantized, 1, HIDDEN, rows, 16, 0)
    return y, quantized


def _gate_up(rows, gen):
    """bf16 [rows, 2 * INTER]: gate/up values the clamp bites on, with all-zero Q8_1 blocks,
    small blocks and values at the limit among them."""
    h = torch.randn(rows, 2 * INTER, generator=gen, device="cuda") * 4.0
    h[:, 64:96] = 0.0  # gate 0: a block of zeros after the SwiGLU (amax == 0)
    h[:, INTER + 128 : INTER + 160] = 0.0  # up 0: another one
    h[:, 256:288] *= 1e-3  # a small block: a d below the half's normal range
    h[:, 320:324] = SWIGLU_LIMIT
    h[:, INTER + 320 : INTER + 324] = -SWIGLU_LIMIT
    return h.bfloat16()


class TestHotColdDecodePack(CustomTestCase):
    """The decode pack's kernels against the launches they replace, bit for bit."""

    def _tables(self, gen, experts=384, hot=195):
        perm = torch.randperm(experts, generator=gen, device="cuda")
        remap_hot = torch.full((experts + 1,), hot, dtype=torch.int32, device="cuda")
        remap_cold = torch.full((experts + 1,), -1, dtype=torch.int32, device="cuda")
        remap_hot[perm[:hot] + 1] = torch.arange(hot, dtype=torch.int32, device="cuda")
        remap_cold[perm[hot:] + 1] = torch.arange(
            experts - hot, dtype=torch.int32, device="cuda"
        )
        return remap_hot, remap_cold

    @staticmethod
    def _gathers(ids, remap_hot, remap_cold):
        index = (ids + 1).reshape(-1)
        return (
            remap_hot.index_select(0, index).view(ids.shape),
            remap_cold.index_select(0, index).view(ids.shape),
        )

    @torch.inference_mode()
    def test_route_split_equals_the_gathers(self):
        gen = torch.Generator(device="cuda").manual_seed(2301)
        remap_hot, remap_cold = self._tables(gen)
        # 50,000 tokens: more routes than the kernel's largest grid holds threads
        for tokens in (1, 6, 54, 300, 2048, 50000):
            ids = torch.randint(
                -1,
                384,
                (tokens, TOP_K),
                generator=gen,
                device="cuda",
                dtype=torch.int32,
            )
            _poison(((tokens, TOP_K), torch.int32), ((tokens, TOP_K), torch.int32))
            hot, cold = gguf_moe_split(ids, remap_hot, remap_cold)
            ref_hot, ref_cold = self._gathers(ids, remap_hot, remap_cold)
            self.assertTrue(torch.equal(hot, ref_hot), f"hot ids, {tokens} tokens")
            self.assertTrue(torch.equal(cold, ref_cold), f"cold ids, {tokens} tokens")
            self.assertEqual((hot.dtype, cold.dtype), (torch.int32, torch.int32))
            self.assertEqual((hot.shape, cold.shape), (ids.shape, ids.shape))
        padded = torch.full((5, TOP_K), -1, dtype=torch.int32, device="cuda")
        hot, cold = gguf_moe_split(padded, remap_hot, remap_cold)
        self.assertEqual((int(hot.min()), int(hot.max())), (195, 195))
        self.assertEqual(int(cold.max()), -1)
        # an id outside the tables takes the padded route's entry (the gathers would fault)
        stray = torch.tensor(
            [[384, 9999, -2, -77, 2**31 - 1, -(2**31)]],
            dtype=torch.int32,
            device="cuda",
        )
        hot, cold = gguf_moe_split(stray, remap_hot, remap_cold)
        self.assertEqual(hot.tolist(), [[195] * 6])
        self.assertEqual(cold.tolist(), [[-1] * 6])
        empty = torch.empty((0, TOP_K), dtype=torch.int32, device="cuda")
        hot, cold = gguf_moe_split(empty, remap_hot, remap_cold)
        self.assertEqual((tuple(hot.shape), tuple(cold.shape)), ((0, TOP_K),) * 2)

    @torch.inference_mode()
    def test_route_split_graph_replay(self):
        gen = torch.Generator(device="cuda").manual_seed(2302)
        remap_hot, remap_cold = self._tables(gen)
        for tokens in (6, 54):
            ids = torch.randint(
                -1,
                384,
                (tokens, TOP_K),
                generator=gen,
                device="cuda",
                dtype=torch.int32,
            )
            gguf_moe_split(ids, remap_hot, remap_cold)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                hot, cold = gguf_moe_split(ids, remap_hot, remap_cold)
            for _ in range(4):
                ids.copy_(
                    torch.randint(
                        -1,
                        384,
                        (tokens, TOP_K),
                        generator=gen,
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                graph.replay()
                ref_hot, ref_cold = self._gathers(ids, remap_hot, remap_cold)
                self.assertTrue(torch.equal(hot, ref_hot))
                self.assertTrue(torch.equal(cold, ref_cold))

    def _split_align_cases(self, tokens, hot_pool, cold_pool, gen):
        pick = lambda pool: pool[
            torch.rand(tokens, pool.numel(), generator=gen, device="cuda").argsort(1)[
                :, :TOP_K
            ]
        ].int()
        mixed = torch.randint(
            -1, 384, (tokens, TOP_K), generator=gen, device="cuda", dtype=torch.int32
        )
        mostly_hot = pick(hot_pool)
        swap = torch.rand(tokens, TOP_K, generator=gen, device="cuda") < 0.1
        mostly_hot[swap] = pick(cold_pool)[swap]
        return {
            "any expert or padding": mixed,
            "a tenth cold": mostly_hot,
            "no cold route": pick(hot_pool),
            "every route cold": pick(cold_pool),
            "one cold expert on every route": cold_pool[7]
            .int()
            .expand(tokens, TOP_K)
            .contiguous(),
            "every route padded": torch.full(
                (tokens, TOP_K), -1, dtype=torch.int32, device="cuda"
            ),
        }

    def _check_grouping(self, cold, sorted_ids, expert_ids, ntp, block, msg):
        """Every cold route once, in a block of its expert; padding last; -1 past the blocks in use."""
        flat = cold.reshape(-1).tolist()
        numel = len(flat)
        cols, experts, post = sorted_ids.tolist(), expert_ids.tolist(), int(ntp[0])
        self.assertEqual(post % block, 0, msg)
        used = post // block
        self.assertLessEqual(used, len(experts), msg)
        seen = []
        for b in range(used):
            expert = experts[b]
            self.assertTrue(
                0 <= expert < EXPERTS, f"{msg}: block {b} of expert {expert}"
            )
            chunk = cols[b * block : (b + 1) * block]
            valid = [s for s in chunk if s != numel]
            self.assertTrue(valid, f"{msg}: a block without a route")
            self.assertEqual(chunk, valid + [numel] * (block - len(valid)), msg)
            self.assertEqual({flat[s] for s in valid}, {expert}, msg)
            seen += valid
        self.assertEqual(
            experts[:used], sorted(experts[:used]), msg
        )  # the groups in expert order
        self.assertEqual(sorted(seen), [s for s, e in enumerate(flat) if e >= 0], msg)
        self.assertEqual(set(experts[used:]) - {-1}, set(), msg)
        self.assertEqual(set(cols[post:]) - {numel}, set(), msg)

    @torch.inference_mode()
    def test_split_align_groups_the_cold_routes(self):
        gen = torch.Generator(device="cuda").manual_seed(2308)
        remap_hot, remap_cold = self._tables(gen, hot=384 - EXPERTS)
        cold_pool = (remap_cold[1:] >= 0).nonzero().view(-1)
        hot_pool = (remap_cold[1:] < 0).nonzero().view(-1)
        w13 = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        # 300 tokens: more routes than the kernel's one CTA has threads
        for tokens in (1, 3, 6, 54, 64, 170, 300):
            x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
            for label, ids in self._split_align_cases(
                tokens, hot_pool, cold_pool, gen
            ).items():
                ref_hot, ref_cold = self._gathers(ids, remap_hot, remap_cold)
                for block in GGUF_MOE_VEC_MULTI_ROUTES:
                    msg = f"T={tokens} {label} block={block}"
                    ref = moe_align_block_size(
                        ref_cold, block, EXPERTS, pad_expert_ids=True
                    )
                    _poison(
                        ((tokens, TOP_K), torch.int32),
                        ((tokens, TOP_K), torch.int32),
                        (tuple(ref[0].shape), torch.int32),
                        (tuple(ref[1].shape), torch.int32),
                    )
                    hot, cold, sorted_ids, expert_ids, ntp = gguf_moe_split_align(
                        ids, remap_hot, remap_cold, EXPERTS, block
                    )
                    self.assertTrue(torch.equal(hot, ref_hot), msg)
                    self.assertTrue(torch.equal(cold, ref_cold), msg)
                    # the shapes the cold tier's own alignment has: the same grids downstream
                    self.assertEqual(
                        (sorted_ids.shape, expert_ids.shape, ntp.shape),
                        (ref[0].shape, ref[1].shape, ref[2].shape),
                        msg,
                    )
                    self._check_grouping(cold, sorted_ids, expert_ids, ntp, block, msg)
                    want = gguf_moe_vec_multi(
                        x, w13, *ref, IQ2_S, 2 * INTER, TOP_K, tokens, block
                    )
                    got = gguf_moe_vec_multi(
                        x,
                        w13,
                        sorted_ids,
                        expert_ids,
                        ntp,
                        IQ2_S,
                        2 * INTER,
                        TOP_K,
                        tokens,
                        block,
                    )
                    self.assertTrue(
                        torch.equal(_bits(got), _bits(want)),
                        f"{msg}: gate/up projection",
                    )

    @torch.inference_mode()
    def test_split_align_graph_replay_on_new_routes(self):
        gen = torch.Generator(device="cuda").manual_seed(2309)
        remap_hot, remap_cold = self._tables(gen, hot=384 - EXPERTS)
        cold_pool = (remap_cold[1:] >= 0).nonzero().view(-1)
        hot_pool = (remap_cold[1:] < 0).nonzero().view(-1)
        w13 = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        for tokens in (6, 54):
            cases = self._split_align_cases(tokens, hot_pool, cold_pool, gen)
            x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
            ids = cases["a tenth cold"].clone()

            def chain():
                hot, cold, sorted_ids, expert_ids, ntp = gguf_moe_split_align(
                    ids, remap_hot, remap_cold, EXPERTS, 4
                )
                h = gguf_moe_vec_multi(
                    x,
                    w13,
                    sorted_ids,
                    expert_ids,
                    ntp,
                    IQ2_S,
                    2 * INTER,
                    TOP_K,
                    tokens,
                    4,
                )
                return hot, cold, sorted_ids, expert_ids, ntp, h

            chain()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = chain()
            for label, new_ids in cases.items():
                x.copy_(
                    torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
                )
                ids.copy_(new_ids)
                graph.replay()
                hot, cold, sorted_ids, expert_ids, ntp, h = out
                ref_hot, ref_cold = self._gathers(ids, remap_hot, remap_cold)
                self.assertTrue(torch.equal(hot, ref_hot), label)
                self.assertTrue(torch.equal(cold, ref_cold), label)
                self._check_grouping(cold, sorted_ids, expert_ids, ntp, 4, label)
                self.assertTrue(
                    torch.equal(
                        _bits(h),
                        _bits(
                            gguf_moe_a8_vec(
                                x, w13, ref_cold, TOP_K, IQ2_S, 2 * INTER, tokens
                            )
                        ),
                    ),
                    label,
                )

    @torch.inference_mode()
    def test_split_align_refuses_more_cold_experts_than_its_scan(self):
        gen = torch.Generator(device="cuda").manual_seed(2310)
        remap_hot, remap_cold = self._tables(gen)
        ids = torch.zeros((6, TOP_K), dtype=torch.int32, device="cuda")
        with self.assertRaises(ValueError):
            gguf_moe_split_align(ids, remap_hot, remap_cold, 1025, 4)
        with self.assertRaises(ValueError):
            gguf_moe_split_align(ids, remap_hot, remap_cold, 0, 4)
        with self.assertRaises(ValueError):
            gguf_moe_split_align(ids.long(), remap_hot, remap_cold, 189, 4)

    def _align(self, ids, routes=4):
        return moe_align_block_size(
            ids.reshape(-1, 1).contiguous(), routes, EXPERTS, pad_expert_ids=True
        )

    @torch.inference_mode()
    def test_swiglu_q8_equals_the_two_launches(self):
        gen = torch.Generator(device="cuda").manual_seed(2303)
        w2 = random_rows(IQ2_S, (EXPERTS, HIDDEN), INTER, gen)
        words = (
            INTER // 32 * 9
        )  # the blocks of the columns; the row is padded to 2,560 columns
        for tokens in (1, 6, 54, 64):
            for label, ids in _route_cases(tokens, gen).items():
                rows = tokens * TOP_K
                flat = ids.reshape(-1).contiguous()
                active = flat >= 0
                h = _gate_up(rows, gen)
                h[~active] = float(
                    "nan"
                )  # the grouped kernels leave these rows unwritten
                ref_a = torch.empty((rows, INTER), dtype=h.dtype, device="cuda")
                silu_and_mul_clamp(h, ref_a, SWIGLU_LIMIT)
                _poison(
                    ((rows, INTER), torch.bfloat16),
                    ((rows, 2560 // 32 * 9), torch.int32),
                )
                a, quantized = gguf_moe_cold_swiglu_q8(h, flat, SWIGLU_LIMIT)
                msg = f"T={tokens} {label}"
                self.assertEqual(tuple(quantized.shape), (rows, 2560 // 32 * 9))
                self.assertTrue(
                    torch.equal(_bits(a[active]), _bits(ref_a[active])),
                    f"{msg}: SwiGLU",
                )
                # the down projection: its own quantisation launch against the blocks handed in
                sorted_ids, expert_ids, ntp = self._align(ids)
                ref_y, ref_quantized = _down_with_its_own_quantisation(
                    ref_a, w2, sorted_ids, expert_ids, ntp
                )
                # the blocks of the cold routes' rows as quantize_q8_1 writes them: the scale,
                # the block's sum and the 32 quants of every block of the row's columns
                self.assertTrue(
                    torch.equal(
                        quantized[active][:, :words], ref_quantized[active][:, :words]
                    ),
                    f"{msg}: Q8_1 blocks",
                )
                self.assertTrue(
                    torch.equal(
                        _bits(ref_y),
                        _bits(
                            gguf_moe_vec_multi(
                                ref_a,
                                w2,
                                sorted_ids,
                                expert_ids,
                                ntp,
                                IQ2_S,
                                HIDDEN,
                                1,
                                rows,
                                4,
                            )
                        ),
                    ),
                    f"{msg}: the JIT entry and the op",
                )
                got_y = gguf_moe_vec_multi(
                    a,
                    w2,
                    sorted_ids,
                    expert_ids,
                    ntp,
                    IQ2_S,
                    HIDDEN,
                    1,
                    rows,
                    4,
                    quantized_input=quantized,
                )
                self.assertTrue(
                    torch.equal(_bits(got_y), _bits(ref_y)),
                    f"{msg}: down projection, max|d|="
                    f"{float((got_y.float() - ref_y.float()).abs().max()):.3e}",
                )
                self.assertFalse(bool(got_y.float().isnan().any()), msg)

    @torch.inference_mode()
    def test_swiglu_q8_graph_replay_on_new_routes(self):
        gen = torch.Generator(device="cuda").manual_seed(2304)
        w2 = random_rows(IQ2_S, (EXPERTS, HIDDEN), INTER, gen)
        for tokens in (6, 54):
            rows = tokens * TOP_K
            cases = list(_route_cases(tokens, gen).values())
            ids = cases[0].clone()
            h = _gate_up(rows, gen)

            def chain():
                a, quantized = gguf_moe_cold_swiglu_q8(h, ids.view(-1), SWIGLU_LIMIT)
                sorted_ids, expert_ids, ntp = self._align(ids)
                return gguf_moe_vec_multi(
                    a,
                    w2,
                    sorted_ids,
                    expert_ids,
                    ntp,
                    IQ2_S,
                    HIDDEN,
                    1,
                    rows,
                    4,
                    quantized_input=quantized,
                )

            chain()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = chain()
            for new_ids in cases:
                h.copy_(_gate_up(rows, gen))
                ids.copy_(new_ids)
                graph.replay()
                ref_a = torch.empty((rows, INTER), dtype=h.dtype, device="cuda")
                silu_and_mul_clamp(h, ref_a, SWIGLU_LIMIT)
                sorted_ids, expert_ids, ntp = self._align(ids)
                ref = gguf_moe_vec_multi(
                    ref_a, w2, sorted_ids, expert_ids, ntp, IQ2_S, HIDDEN, 1, rows, 4
                )
                self.assertTrue(torch.equal(_bits(out), _bits(ref)))

    @staticmethod
    def _three_launches(y, ids, weights, hot):
        """The earlier launches: route weights in place, moe_sum, add_ (y is consumed)."""
        from sgl_kernel import moe_sum

        tokens = ids.shape[0]
        apply_gguf_moe_weights(y, weights, ids)
        cold = torch.empty((tokens, y.shape[1]), dtype=y.dtype, device=y.device)
        moe_sum(y.view(tokens, TOP_K, -1), cold)
        return cold, hot.clone().add_(cold)

    @staticmethod
    def _hot_output(tokens, gen):
        hot = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
        hot[0, :64] = -0.0  # add_ turns a -0 into +0 when the cold sum is +0
        hot[-1, 64:128] = 0.0
        return hot

    @torch.inference_mode()
    def test_dense_sum_fold_equals_weights_sum_add(self):
        gen = torch.Generator(device="cuda").manual_seed(2305)
        for tokens in (1, 2, 3, 6, 54, 64, 300):
            for label, ids in _route_cases(tokens, gen).items():
                rows = tokens * TOP_K
                active = ids.reshape(-1) >= 0
                y = torch.randn(rows, HIDDEN, generator=gen, device="cuda").bfloat16()
                y[~active] = float("nan")  # undefined rows: nothing may read them
                weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
                weights = (weights / weights.sum(1, keepdim=True) * 1.5).float()
                hot = self._hot_output(tokens, gen)
                ref_cold, ref = self._three_launches(y.clone(), ids, weights, hot)
                msg = f"T={tokens} {label}"
                _poison(((tokens, HIDDEN), torch.bfloat16))
                cold = gguf_moe_cold_sum(y, ids, weights, dense=True)
                self.assertTrue(
                    torch.equal(_bits(cold), _bits(ref_cold)), f"{msg}: dense sum"
                )
                target = hot.clone()
                got = gguf_moe_cold_sum(y, ids, weights, dense=True, out=target)
                self.assertIs(got, target)
                self.assertTrue(
                    torch.equal(_bits(got), _bits(ref)),
                    f"{msg}: folded sum, max|d|="
                    f"{float((got.float() - ref.float()).abs().max()):.3e}",
                )
                # the compact layout with the fold: the same rows packed in route order
                rows_map = torch.cumsum(active, 0, dtype=torch.int32) - 1
                rows_map = rows_map.masked_fill(~active, -1).view(tokens, TOP_K)
                packed = y[active].contiguous()
                if packed.shape[0] == 0:
                    packed = y[:1].contiguous()  # no active route: no row is read
                got = gguf_moe_cold_sum(packed, rows_map, weights, out=hot.clone())
                self.assertTrue(
                    torch.equal(_bits(got), _bits(ref)), f"{msg}: compact, folded"
                )

    @torch.inference_mode()
    def test_dense_sum_keeps_the_order_of_the_additions(self):
        """Inputs where the order of the fp32 additions decides the bf16 result: the
        dense rows, with and without the fold, add in moe_sum's order."""
        width = 64
        values = (1.0, 2.0**-8, 2.0**-24, 2.0**-24)
        places = list(itertools.permutations(range(TOP_K), len(values)))
        tokens = len(places)
        y = torch.zeros(tokens, TOP_K, width, device="cuda")
        ids = torch.full((tokens, TOP_K), -1, dtype=torch.int32, device="cuda")
        for t, slots in enumerate(places):
            for v, s in zip(values, slots):
                y[t, s] = v
                ids[t, s] = s
        y = y.view(tokens * TOP_K, width).bfloat16()
        weights = torch.ones(tokens, TOP_K, device="cuda")
        hot = torch.zeros(tokens, width, dtype=torch.bfloat16, device="cuda")
        for fill in (False, True):  # the two empty slots inactive, or active with 0
            use = ids.masked_fill(ids < 0, 0) if fill else ids
            ref_cold, ref = self._three_launches(y.clone(), use, weights, hot)
            # the construction tips the sum on some placements and not on others
            self.assertGreater(int((ref_cold[:, 0].float() > 1.0).sum()), 0)
            self.assertGreater(int((ref_cold[:, 0].float() == 1.0).sum()), 0)
            cold = gguf_moe_cold_sum(y, use, weights, dense=True)
            self.assertTrue(torch.equal(_bits(cold), _bits(ref_cold)), f"fill={fill}")
            got = gguf_moe_cold_sum(y, use, weights, dense=True, out=hot.clone())
            self.assertTrue(torch.equal(_bits(got), _bits(ref)), f"fill={fill}, fold")
        gen = torch.Generator(device="cuda").manual_seed(977)
        tokens = 300
        ids = torch.rand(tokens, EXPERTS, generator=gen, device="cuda").argsort(1)
        ids = ids[:, :TOP_K].int()
        ids[torch.rand(tokens, TOP_K, generator=gen, device="cuda") < 1 / 3] = -1
        y = torch.randn(tokens * TOP_K, HIDDEN, generator=gen, device="cuda") * 3
        scale = torch.randint(-20, 21, y.shape, generator=gen, device="cuda")
        y = (y.bfloat16().float() * torch.pow(2.0, scale.float())).bfloat16()
        weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
        hot = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
        ref_cold, ref = self._three_launches(y.clone(), ids, weights, hot)
        cold = gguf_moe_cold_sum(y, ids, weights, dense=True)
        self.assertTrue(torch.equal(_bits(cold), _bits(ref_cold)), "wide, dense sum")
        got = gguf_moe_cold_sum(y, ids, weights, dense=True, out=hot.clone())
        self.assertTrue(torch.equal(_bits(got), _bits(ref)), "wide, folded sum")

    @torch.inference_mode()
    def test_dense_sum_fold_refuses_what_it_cannot_reproduce(self):
        y = torch.zeros(12, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ids = torch.zeros(2, TOP_K, dtype=torch.int32, device="cuda")
        weights = torch.ones(2, TOP_K, device="cuda")
        with self.assertRaises(ValueError):  # another dtype than y's
            gguf_moe_cold_sum(
                y, ids, weights, dense=True, out=torch.zeros(2, HIDDEN, device="cuda")
            )
        with self.assertRaises(ValueError):  # another shape
            gguf_moe_cold_sum(
                y,
                ids,
                weights,
                dense=True,
                out=torch.zeros(3, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            )
        with self.assertRaises(Exception):  # a dense y holds one row per route
            gguf_moe_cold_sum(y[:6].contiguous(), ids, weights, dense=True)

    def _layer_inputs(self, tokens, gen):
        x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
        weights = torch.rand(tokens, TOP_K, generator=gen, device="cuda") + 0.05
        weights = (weights / weights.sum(1, keepdim=True) * 1.5).float()
        return x, weights, self._hot_output(tokens, gen)

    def _decode_chain(self, x, cold_ids, weights, hot, w13, w2, pack, lanes=16):
        """The cold tier of a decode batch and its add into the hot output: the earlier
        launches (pack False) or the pack's; every buffer comes from poisoned memory."""
        tokens = x.shape[0]
        rows = tokens * TOP_K
        _poison(
            ((rows, 2 * INTER), torch.bfloat16),
            ((rows, INTER), torch.bfloat16),
            ((rows, HIDDEN), torch.bfloat16),
            ((rows, 2560 // 32 * 9), torch.int32),
            ((tokens, HIDDEN), torch.bfloat16),
        )
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            cold_ids, 4, EXPERTS, pad_expert_ids=True
        )
        h = gguf_moe_vec_multi(
            x,
            w13,
            sorted_ids,
            expert_ids,
            ntp,
            IQ2_S,
            2 * INTER,
            TOP_K,
            tokens,
            4,
            lanes=lanes,
            zero_init=False,
        )
        down = dict(lanes=lanes, zero_init=False)
        if pack:
            a, quantized = gguf_moe_cold_swiglu_q8(h, cold_ids.view(-1), SWIGLU_LIMIT)
            down["quantized_input"] = quantized
        else:
            a = torch.empty((rows, INTER), dtype=h.dtype, device="cuda")
            silu_and_mul_clamp(h, a, SWIGLU_LIMIT)
        y = gguf_moe_vec_multi(
            a, w2, sorted_ids, expert_ids, ntp, IQ2_S, HIDDEN, 1, rows, 4, **down
        )
        if pack:
            return gguf_moe_cold_sum(y, cold_ids, weights, dense=True, out=hot)
        return self._three_launches(y, cold_ids, weights, hot)[1]

    @torch.inference_mode()
    def test_decode_chain_equals_the_earlier_launches(self):
        gen = torch.Generator(device="cuda").manual_seed(2306)
        w13 = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        w2 = random_rows(IQ2_S, (EXPERTS, HIDDEN), INTER, gen)
        for tokens in (3, 6, 54, 64):
            for label, ids in _route_cases(tokens, gen).items():
                x, weights, hot = self._layer_inputs(tokens, gen)
                ref = self._decode_chain(x, ids, weights, hot.clone(), w13, w2, False)
                self.assertFalse(bool(ref.float().isnan().any()), f"T={tokens} {label}")
                for lanes in (1, 2, 16):
                    got = self._decode_chain(
                        x, ids, weights, hot.clone(), w13, w2, True, lanes
                    )
                    self.assertTrue(
                        torch.equal(_bits(got), _bits(ref)),
                        f"T={tokens} {label} lanes={lanes}: max|d|="
                        f"{float((got.float() - ref.float()).abs().max()):.3e}",
                    )

    @torch.inference_mode()
    def test_decode_chain_graph_replay_on_new_routes(self):
        gen = torch.Generator(device="cuda").manual_seed(2307)
        w13 = random_rows(IQ2_S, (EXPERTS, 2 * INTER), HIDDEN, gen)
        w2 = random_rows(IQ2_S, (EXPERTS, HIDDEN), INTER, gen)
        for tokens in (6, 54):
            cases = list(_route_cases(tokens, gen).values())
            x, weights, hot = self._layer_inputs(tokens, gen)
            ids = cases[0].clone()
            target = hot.clone()

            def pack():
                target.copy_(hot)  # the hot tier writes its output inside the graph too
                return self._decode_chain(x, ids, weights, target, w13, w2, True)

            pack()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = pack()
            for new_ids in cases:
                nx, nw, nh = self._layer_inputs(tokens, gen)
                x.copy_(nx)
                weights.copy_(nw)
                hot.copy_(nh)
                ids.copy_(new_ids)
                graph.replay()
                ref = self._decode_chain(x, ids, weights, hot.clone(), w13, w2, False)
                self.assertTrue(torch.equal(_bits(out), _bits(ref)))


if __name__ == "__main__":
    unittest.main()
