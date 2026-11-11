"""GGUF vector expert projection for ik's IQ4_KS (144) and IQ2_KS (145): row-header
stride, fp64 dequant reference, inactive routes and CUDA-graph replay."""

import unittest

import torch

from sglang.kernels.ops.moe.gguf_moe_vec import gguf_moe_a8_vec
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.gguf_iq_reference import (
    dequantize,
    nmse,
    random_rows,
    row_bytes,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="nightly", runner_config="1-gpu-large")

HIDDEN, INTER, TOP_K, EXPERTS = 5120, 2304, 6, 320


class TestGgufMoeVecIqk(CustomTestCase):
    @torch.inference_mode()
    def test_row_header_types_match_reference(self):
        """Both projections, E=320, top-6, 30 % negative routes, decode-sized batches."""
        gen = torch.Generator(device="cuda").manual_seed(144)
        for quant_type in (144, 145, 22):
            for n, k in ((2 * INTER, HIDDEN), (HIDDEN, INTER)):
                w = random_rows(quant_type, (EXPERTS, n), k, gen)
                self.assertEqual(w.shape[-1], row_bytes(quant_type, k))
                for tokens in (1, 9, 54):
                    ids = (
                        torch.rand(tokens, EXPERTS, generator=gen, device="cuda")
                        .argsort(1)[:, :TOP_K]
                        .int()
                    )
                    ids[
                        torch.rand(tokens, TOP_K, generator=gen, device="cuda") < 0.3
                    ] = -1
                    x = torch.randn(tokens, k, generator=gen, device="cuda").bfloat16()
                    y = gguf_moe_a8_vec(
                        x, w, ids.contiguous(), TOP_K, quant_type, n, tokens
                    )
                    ref = torch.zeros(
                        tokens * TOP_K, n, dtype=torch.float64, device="cuda"
                    )
                    flat = ids.reshape(-1)
                    for e in flat[flat >= 0].unique().tolist():
                        routes = (flat == e).nonzero().squeeze(1)
                        ref[routes] = (
                            x[routes // TOP_K].double()
                            @ dequantize(w[e], quant_type, k).T
                        )
                    err = nmse(y, ref)
                    print(
                        f"vec type={quant_type} N={n} K={k} T={tokens} nmse={err:.3e}"
                    )
                    self.assertLessEqual(err, 5e-4)
                    self.assertLessEqual(
                        float((y.double() - ref).abs().max()),
                        2e-2 * float(ref.abs().max()),
                    )
                    self.assertEqual(int(y[flat < 0].count_nonzero()), 0)
                del w

    @torch.inference_mode()
    def test_graph_replay_matches_eager(self):
        gen = torch.Generator(device="cuda").manual_seed(145)
        for quant_type in (144, 145):
            w = random_rows(quant_type, (192, 2 * INTER), HIDDEN, gen)
            x = torch.randn(9, HIDDEN, generator=gen, device="cuda").bfloat16()
            ids = torch.randint(
                -1, 192, (9, TOP_K), generator=gen, device="cuda", dtype=torch.int32
            )
            gguf_moe_a8_vec(x, w, ids, TOP_K, quant_type, 2 * INTER, 9)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = gguf_moe_a8_vec(x, w, ids, TOP_K, quant_type, 2 * INTER, 9)
            x.copy_(torch.randn(9, HIDDEN, generator=gen, device="cuda").bfloat16())
            ids.copy_(
                torch.randint(
                    -1, 192, (9, TOP_K), generator=gen, device="cuda", dtype=torch.int32
                )
            )
            graph.replay()
            self.assertTrue(
                torch.equal(
                    out, gguf_moe_a8_vec(x, w, ids, TOP_K, quant_type, 2 * INTER, 9)
                )
            )

    def test_row_size_is_checked(self):
        """A weight sized as whole blocks (no row header) is refused."""
        w = torch.zeros(2, 8, 256 // 256 * 136, dtype=torch.uint8, device="cuda")
        x = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
        ids = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
        with self.assertRaises(Exception):
            gguf_moe_a8_vec(x, w, ids, 1, 144, 8, 1)


if __name__ == "__main__":
    unittest.main()
