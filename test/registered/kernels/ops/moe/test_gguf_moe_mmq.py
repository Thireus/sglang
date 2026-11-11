"""GGUF MoE MMQ (IQ2_S / IQ4_KS / IQ2_KS) against a float64 dequant reference,
the vector op and CUDA-graph replay, at DeepSeek-V4.1 expert shapes.

Weights are random valid packed rows (``gguf_iq_reference.random_rows``); the
reference dequant is cross-checked against ik_llama.cpp's ``dequantize_row_*``.
``SGLANG_TEST_GGUF_IK_FIXTURES`` may name a directory of ik-quantised experts
(``ik_<type>_E8.pt``) to add a real-quantiser case. ``SGLANG_TEST_GGUF_MMQ_SPEED=1``
prints the per-call timing table against ``gguf_moe_a8_vec``.
"""

import os
import unittest

import torch

from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq, gguf_moe_mmq_block_size
from sglang.kernels.ops.moe.gguf_moe_vec import (
    _jit_gguf_moe_vec_module,
    gguf_moe_a8_vec,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.gguf_iq_reference import dequantize, nmse, random_rows
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="nightly", runner_config="1-gpu-large")

HIDDEN, INTER, TOP_K, EXPERTS = 5120, 2304, 6, 320
TOKENS = (1, 8, 40, 256, 2048)
TYPES = (144, 145, 22)  # IQ4_KS, IQ2_KS, IQ2_S
NMSE_BOUND = 5e-4  # ik test-backend-ops MUL_MAT_ID
MAX_ERR_BOUND = 2e-2  # x max|ref|
VEC_NMSE_BOUND = 1e-6  # same q8_1 values, summation order only


def _has_vec(quant_type):
    try:
        _jit_gguf_moe_vec_module(torch.bfloat16, quant_type)
        return True
    except ValueError:
        return False


def _routes(tokens, experts, gen, inactive=0.0, hot_expert=None):
    keys = torch.rand(tokens, experts, generator=gen, device="cuda")
    if hot_expert is not None:
        keys[:, hot_expert] = -1.0  # every token routes it, in slot 0
    ids = keys.argsort(dim=1)[:, :TOP_K].to(torch.int32)
    if inactive > 0:
        drop = torch.rand(tokens, TOP_K, generator=gen, device="cuda") < inactive
        ids[drop] = -1
    return ids.contiguous()


def _mmq(x, w, ids, quant_type, top_k, tokens):
    block = gguf_moe_mmq_block_size(quant_type)
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        ids, block, w.shape[0], pad_expert_ids=True
    )
    return gguf_moe_mmq(
        x, w, sorted_ids, expert_ids, ntp, quant_type, w.shape[1], top_k, tokens, block
    )


def _references(w, quant_type, cases):
    """cases: list of (x, ids, top_k); float64 [routes, N] per case, zero for inactive."""
    k = cases[0][0].shape[1]
    outs = [
        torch.zeros(ids.numel(), w.shape[1], dtype=torch.float64, device="cuda")
        for _, ids, _ in cases
    ]
    used = torch.cat([ids.reshape(-1) for _, ids, _ in cases]).unique()
    for e in used[used >= 0].tolist():
        dense = dequantize(w[e], quant_type, k)
        for (x, ids, top_k), out in zip(cases, outs):
            routes = (ids.reshape(-1) == e).nonzero().squeeze(1)
            if routes.numel():
                out[routes] = x[routes // top_k].double() @ dense.T
        del dense
    return outs


class TestGgufMoeMmq(CustomTestCase):
    def _check(self, y, ref, ids, label):
        err = nmse(y, ref)
        max_err = float((y.double() - ref).abs().max())
        bound = MAX_ERR_BOUND * float(ref.abs().max())
        inactive = ids.reshape(-1) < 0
        print(
            f"{label} nmse={err:.3e} max_err/max_ref={max_err / max(float(ref.abs().max()), 1e-30):.3e}"
        )
        if ref.abs().max() > 0:
            self.assertLessEqual(err, NMSE_BOUND, label)
            self.assertLessEqual(max_err, bound, label)
        if inactive.any():
            self.assertEqual(int(y[inactive].count_nonzero()), 0, label)

    @torch.inference_mode()
    def test_numerics_production_shapes(self):
        """MMQ matches the fp64 dequant reference at hidden 5120 / intermediate 2304, E=320, top-6."""
        gen = torch.Generator(device="cuda").manual_seed(52)
        for quant_type in TYPES:
            with self.subTest(quant_type=quant_type):
                w13 = random_rows(quant_type, (EXPERTS, 2 * INTER), HIDDEN, gen)
                w2 = random_rows(quant_type, (EXPERTS, HIDDEN), INTER, gen)
                specs = [(t, EXPERTS, 0.0, None) for t in TOKENS]
                specs += [(54, EXPERTS, 0.5, None), (65, EXPERTS, 0.5, None)]
                specs += [(64, EXPERTS, 1.0, None), (512, EXPERTS, 0.0, 7)]
                specs += [(2048, 192, 0.0, None)]  # the cold tier's C
                for proj, w, k in (("w13", w13, HIDDEN), ("w2", w2, INTER)):
                    cases, labels, outs = [], [], []
                    for tokens, experts, inactive, hot in specs:
                        ids = _routes(tokens, experts, gen, inactive, hot)
                        wv = w[:experts]
                        if proj == "w13":
                            x = torch.randn(
                                tokens, k, generator=gen, device="cuda"
                            ).bfloat16()
                            y = _mmq(x, wv, ids, quant_type, TOP_K, tokens)
                            cases.append((x, ids, TOP_K))
                        else:
                            x = torch.randn(
                                tokens * TOP_K, k, generator=gen, device="cuda"
                            ).bfloat16()
                            flat = ids.reshape(-1, 1).contiguous()
                            y = _mmq(x, wv, flat, quant_type, 1, tokens * TOP_K)
                            cases.append((x, flat, 1))
                        outs.append(y)
                        labels.append(
                            f"type={quant_type} {proj} T={tokens} E={experts} inactive={inactive} hot={hot}"
                        )
                    refs = _references(w, quant_type, cases)
                    for (x, ids, _), y, ref, label in zip(cases, outs, refs, labels):
                        self._check(y, ref, ids, label)
                del w13, w2
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def test_partial_row_tile(self):
        """N not a multiple of the 128-row tile takes the checked loader and write-back."""
        gen = torch.Generator(device="cuda").manual_seed(53)
        for quant_type in TYPES:
            w = random_rows(quant_type, (8, 200), INTER, gen)
            ids = _routes(9, 8, gen, 0.25)
            x = torch.randn(9, INTER, generator=gen, device="cuda").bfloat16()
            y = _mmq(x, w, ids, quant_type, TOP_K, 9)
            (ref,) = _references(w, quant_type, [(x, ids, TOP_K)])
            self._check(y, ref, ids, f"type={quant_type} N=200 T=9")

    @torch.inference_mode()
    def test_matches_vector_op(self):
        """MMQ and the vector kernel see the same q8_1 activations: NMSE <= 1e-6."""
        gen = torch.Generator(device="cuda").manual_seed(54)
        for quant_type in TYPES:
            if not _has_vec(quant_type):
                print(f"type={quant_type}: no vector kernel in this build, skipped")
                continue
            w = random_rows(quant_type, (EXPERTS, 2 * INTER), HIDDEN, gen)
            for tokens in (1, 8, 40, 256, 2048):
                ids = _routes(tokens, EXPERTS, gen, 0.1)
                x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
                y = _mmq(x, w, ids, quant_type, TOP_K, tokens)
                v = gguf_moe_a8_vec(x, w, ids, TOP_K, quant_type, 2 * INTER, tokens)
                err = nmse(y, v)
                print(f"type={quant_type} T={tokens} nmse(mmq, vec)={err:.3e}")
                self.assertLessEqual(err, VEC_NMSE_BOUND)
            del w
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def test_graph_replay_matches_eager(self):
        """Captured routing + MMQ replays bit for bit on new inputs and new routes."""
        gen = torch.Generator(device="cuda").manual_seed(55)
        for quant_type in TYPES:
            w = random_rows(quant_type, (192, 2 * INTER), HIDDEN, gen)
            for tokens in (54, 256):
                x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").bfloat16()
                ids = _routes(tokens, 192, gen, 0.3)
                _mmq(x, w, ids, quant_type, TOP_K, tokens)  # build and warm
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = _mmq(x, w, ids, quant_type, TOP_K, tokens)
                for _ in range(2):
                    x.copy_(
                        torch.randn(
                            tokens, HIDDEN, generator=gen, device="cuda"
                        ).bfloat16()
                    )
                    ids.copy_(_routes(tokens, 192, gen, 0.3))
                    graph.replay()
                    eager = _mmq(x, w, ids, quant_type, TOP_K, tokens)
                    self.assertTrue(
                        torch.equal(out, eager), f"type={quant_type} T={tokens}"
                    )
            del w

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_GGUF_IK_FIXTURES"), "no ik-quantised fixtures"
    )
    @torch.inference_mode()
    def test_ik_quantised_experts(self):
        """Experts written by ik's ggml_quantize_chunk (E=8) through both projections."""
        root = os.environ["SGLANG_TEST_GGUF_IK_FIXTURES"]
        gen = torch.Generator(device="cuda").manual_seed(56)
        for quant_type in TYPES:
            fixture = torch.load(f"{root}/ik_{quant_type}_E8.pt")
            for proj, k in (("w13", HIDDEN), ("w2", INTER)):
                w = fixture[proj].cuda()
                for tokens in (8, 256):
                    ids = _routes(tokens, 8, gen)
                    if proj == "w2":
                        ids = ids.reshape(-1, 1).contiguous()
                    top_k = TOP_K if proj == "w13" else 1
                    rows = tokens if proj == "w13" else tokens * TOP_K
                    x = torch.randn(rows, k, generator=gen, device="cuda").bfloat16()
                    y = _mmq(x, w, ids, quant_type, top_k, rows)
                    (ref,) = _references(w, quant_type, [(x, ids, top_k)])
                    self._check(y, ref, ids, f"ik type={quant_type} {proj} T={tokens}")

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_GGUF_MMQ_SPEED") == "1", "speed table on request"
    )
    @torch.inference_mode()
    def test_speed_line(self):
        """us per call, MMQ (routing included) vs gguf_moe_a8_vec, E=320, top-6."""
        gen = torch.Generator(device="cuda").manual_seed(57)
        print("\ntype proj tokens mmq_us vec_us vec/mmq")
        for quant_type in TYPES:
            vec = _has_vec(quant_type)
            for proj, n, k in (("w13", 2 * INTER, HIDDEN), ("w2", HIDDEN, INTER)):
                w = random_rows(quant_type, (EXPERTS, n), k, gen)
                for tokens in TOKENS:
                    ids = _routes(tokens, EXPERTS, gen)
                    top_k, rows = (
                        (TOP_K, tokens) if proj == "w13" else (1, tokens * TOP_K)
                    )
                    if proj == "w2":
                        ids = ids.reshape(-1, 1).contiguous()
                    x = torch.randn(rows, k, generator=gen, device="cuda").bfloat16()
                    calls = {"mmq": lambda: _mmq(x, w, ids, quant_type, top_k, rows)}
                    if vec:
                        calls["vec"] = lambda: gguf_moe_a8_vec(
                            x, w, ids, top_k, quant_type, n, rows
                        )
                    us = {}
                    for name, fn in calls.items():
                        for _ in range(3):
                            fn()
                        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                        iters = 20
                        start.record()
                        for _ in range(iters):
                            fn()
                        end.record()
                        end.synchronize()
                        us[name] = start.elapsed_time(end) * 1000 / iters
                    v = us.get("vec")
                    print(
                        f"{quant_type} {proj} {tokens} {us['mmq']:.1f} "
                        + (f"{v:.1f} {v / us['mmq']:.2f}" if v else "- -")
                    )
                w = None
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
