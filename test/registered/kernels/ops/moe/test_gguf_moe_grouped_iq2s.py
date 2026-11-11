"""The grouped GGUF MoE kernels for IQ2_S against the kernels they replace, bit for bit.

* ``gguf_moe_vec_multi`` (several routes of one expert per block, lanes of CTAs) against
  ``gguf_moe_a8_vec``: the same Q8_1 activations, integer sums and float expression per
  row, so the output must be equal, at every block width and lane count, and the vector
  kernel itself must not depend on its rows per CTA (ggml's 1 against the default).
* ``gguf_moe_mmq_wide`` (up to 16 / 32 / 64 / 128 routes of one expert per CTA, only the
  column groups in use multiplied, 32 columns of activations per pass) against
  ``gguf_moe_mmq`` in blocks of 16.

Random valid IQ2_S rows at DeepSeek-V4.1 expert shapes, the cold tier's 192 experts;
routes: random with negative ids, none active, every route active, one expert on every
route (many blocks and passes of one expert), the last expert, a single token; eager and
CUDA-graph replay on new routes.
"""

import unittest

import torch

from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq
from sglang.kernels.ops.moe.gguf_moe_mmq_wide import (
    GGUF_MOE_MMQ_WIDE_BLOCKS,
    gguf_moe_mmq_wide,
)
from sglang.kernels.ops.moe.gguf_moe_vec import gguf_moe_a8_vec
from sglang.kernels.ops.moe.gguf_moe_vec_multi import (
    GGUF_MOE_VEC_MULTI_ROUTES,
    gguf_moe_vec_multi,
)
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


if __name__ == "__main__":
    unittest.main()
