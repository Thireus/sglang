"""The routed-experts row mask on the device: kept rows keep their bits.

A scheduled DSpark verify (speculative/dspark_components/dspark_pp_verify_schedule.py)
publishes a row mask through the forward context; a hot/cold layer gives every route
of a masked row the padded route before its route split. Here the two-tier layer of
the toy checkpoint (sglang/test/hotcold_fixture.py) runs with its real kernels: the
hot tier on FlashInfer's CUTLASS MXFP4 MoE, the cold tier on the GGUF MoE kernels.

For masks of every shape (a prefix of each request's rows kept):

* an eager call and a CUDA-graph replay give, on the kept rows, the bits of the
  unmasked call, and 0 on the masked rows;
* the same with NaN written into the masked rows' inputs and with NaN-filled blocks
  handed back to the allocator first, so that a masked row that leaked into a kept
  one, or a kernel that read a block it did not write, would show;
* a graph captured with a mask of ones equals the graph captured without a mask;
* the same in every form of the route split (the launch that also groups a decode
  batch's cold routes, the split alone, the gathers).

The CPU side (the policy, the relay, the accept, the route split's ids) is
test/registered/unit/spec/test_pp_dspark_verify_schedule.py.
"""

import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor import forward_context as fc
from sglang.test import hotcold_fixture as fx
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="nightly", runner_config="1-gpu-large")

N_EXPERTS, HIDDEN, INTER, TOP_K = 8, 512, 256, 3
NAN = float("nan")


def _device_context(model_path):
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_context, get_flags, get_parallel

    stack = contextlib.ExitStack()
    stack.enter_context(
        get_context().override_server_args(
            model_path=model_path, device="cuda", chunked_prefill_size=2048
        )
    )
    stack.enter_context(
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE,
            runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4,
        )
    )
    stack.enter_context(
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_group=None,
        )
    )
    return stack


def _masked(mask):
    fb = SimpleNamespace(moe_num_token_non_padded=None, moe_verify_row_mask=mask)
    return fc.forward_context(fc.ForwardContext(attn_backend=None), forward_batch=fb)


def _bits_equal(a, b):
    return bool(
        torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
    )


def _poison(tokens):
    """NaN-filled blocks of the tiers' intermediate sizes, back to the allocator."""
    blocks = [
        torch.full((tokens * TOP_K, r), NAN, dtype=torch.bfloat16, device="cuda")
        for r in (2 * INTER, INTER, HIDDEN)
    ]
    blocks.append(
        torch.full((tokens, HIDDEN), NAN, dtype=torch.bfloat16, device="cuda")
    )
    del blocks


def _row_mask(lens, width):
    cols = torch.arange(width, device="cuda").view(1, width)
    lens = torch.tensor(lens, device="cuda").view(-1, 1)
    return (cols < lens).to(torch.int32).reshape(-1)


def _length_cases(requests, width, gen):
    cases = [[v] * requests for v in range(1, width + 1)]
    for _ in range(6):
        cases.append(torch.randint(1, width + 1, (requests,), generator=gen).tolist())
    return cases


@unittest.skipUnless(torch.cuda.is_available(), "the kernels run on a GPU")
class TestRowMaskOnDevice(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ckpt = fx.build_toy_checkpoint(
            os.path.join(cls.tmp.name, "ckpt"),
            n_experts=N_EXPERTS,
            hidden=HIDDEN,
            inter=INTER,
        )
        cls.stack = _device_context(cls.ckpt.path)
        from sglang.srt.models import deepseek_v2

        config = fx.toy_hf_config(
            cls.ckpt.path,
            hidden_size=HIDDEN,
            moe_intermediate_size=INTER,
            intermediate_size=INTER,
        )
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                cls.mlp = deepseek_v2.DeepseekV2MoE(
                    config=config,
                    layer_id=0,
                    quant_config=fx.fp4_quant_config(),
                    prefix="model.layers.0.mlp",
                    is_deepseek_v4=True,
                )
        finally:
            torch.set_default_dtype(old)
        fx.load_toy_layer(cls.mlp, cls.ckpt, 0)
        for _, module in cls.mlp.named_modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)
        cls.experts = cls.mlp.experts

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        cls.tmp.cleanup()
        super().tearDownClass()

    def _batch(self, tokens, gen):
        x = (torch.randn(tokens, HIDDEN, generator=gen) * 1.5).to(
            "cuda", torch.bfloat16
        )
        ids = torch.stack(
            [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K] for _ in range(tokens)]
        ).to("cuda", torch.int32)
        weights = (torch.rand(tokens, TOP_K, generator=gen) + 0.1).to("cuda")
        return x, ids, weights

    def _call(self, x, ids, weights):
        logits = torch.zeros(x.shape[0], N_EXPERTS, device="cuda")
        with torch.no_grad():
            return self.experts(x, StandardTopKOutput(weights, ids, logits))

    def test_eager_kept_rows_keep_their_bits(self):
        gen = torch.Generator().manual_seed(5)
        checked = 0
        for requests, width in ((2, 6), (3, 7), (9, 6)):
            tokens = requests * width
            x, ids, weights = self._batch(tokens, gen)
            free = self._call(x, ids, weights).clone()
            self.assertTrue(bool(torch.isfinite(free.float()).all()))
            self.assertGreater(float(free.float().abs().max()), 0.0)
            for lens in _length_cases(requests, width, gen):
                mask = _row_mask(lens, width)
                keep = mask.bool()
                dirty = x.clone()
                dirty[~keep] = NAN
                _poison(tokens)
                with _masked(mask):
                    got = self._call(dirty, ids, weights)
                self.assertTrue(_bits_equal(got[keep], free[keep]), (requests, lens))
                self.assertTrue(bool((got[~keep].float() == 0).all()), (requests, lens))
                checked += 1
        self.assertGreater(checked, 30)

    def _graphs(self, tokens):
        """The static inputs of a batch, a graph captured without a mask and one
        captured with the static mask among its inputs, each with its output."""
        sx = torch.zeros(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        sids = torch.zeros(tokens, TOP_K, dtype=torch.int32, device="cuda")
        sw = torch.ones(tokens, TOP_K, device="cuda")
        smask = torch.ones(tokens, dtype=torch.int32, device="cuda")

        def capture(masked):
            def run():
                if masked:
                    with _masked(smask):
                        return self._call(sx, sids, sw)
                return self._call(sx, sids, sw)

            run()
            torch.cuda.synchronize()
            graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.cuda.graph(graph):
                out = run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            return graph, out

        return (sx, sids, sw, smask), capture(masked=False), capture(masked=True)

    def _replay_masks(self, graphs, requests, width, gen, rounds=3):
        (sx, sids, sw, smask), (free_graph, free_out), (graph, out) = graphs
        tokens = requests * width
        for _ in range(rounds):
            x, ids, weights = self._batch(tokens, gen)
            sx.copy_(x), sids.copy_(ids), sw.copy_(weights)
            smask.fill_(1)
            free_graph.replay()
            torch.cuda.synchronize()
            free = free_out.clone()
            graph.replay()
            torch.cuda.synchronize()
            # A mask of ones changes no bit.
            self.assertTrue(_bits_equal(out, free))
            for lens in _length_cases(requests, width, gen):
                mask = _row_mask(lens, width)
                keep = mask.bool()
                smask.copy_(mask)
                dirty = x.clone()
                dirty[~keep] = NAN
                sx.copy_(dirty)
                _poison(tokens)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(_bits_equal(out[keep], free[keep]), lens)
                self.assertTrue(bool((out[~keep].float() == 0).all()), lens)
                sx.copy_(x)

    def test_graph_replay_reads_the_static_mask(self):
        gen = torch.Generator().manual_seed(6)
        for requests, width in ((2, 6), (3, 7)):
            graphs = self._graphs(requests * width)
            self._replay_masks(graphs, requests, width, gen)

    def test_every_form_of_the_route_split(self):
        """A decode batch splits its routes and groups the cold ones in one launch;
        SGLANG_HOTCOLD_DECODE_ALIGN=0 leaves the split alone in that launch, and
        SGLANG_HOTCOLD_SPLIT_KERNEL=0 the gathers. The masked ids reach each form,
        in eager calls and in graph replay, and each form gives the same bits."""
        from sglang.srt.layers.moe import hotcold as hc_moe

        gen = torch.Generator().manual_seed(7)
        requests, width = 3, 7
        tokens = requests * width
        calls = []
        split, split_align = hc_moe._split_kernel, hc_moe._split_align_kernel

        def spy_split(*args):
            calls.append("split")
            return split(*args)

        def spy_split_align(*args):
            calls.append("split align")
            return split_align(*args)

        x, ids, weights = self._batch(tokens, gen)
        outs = []
        with (
            mock.patch.object(hc_moe, "_split_kernel", spy_split),
            mock.patch.object(hc_moe, "_split_align_kernel", spy_split_align),
        ):
            for switches, form in (
                ({}, "split align"),
                ({"SGLANG_HOTCOLD_DECODE_ALIGN": "0"}, "split"),
                ({"SGLANG_HOTCOLD_SPLIT_KERNEL": "0"}, None),
            ):
                want = [form] if form else []
                with mock.patch.dict(os.environ, switches):
                    calls.clear()
                    free = self._call(x, ids, weights).clone()
                    self.assertEqual(calls, want)
                    outs.append(free)
                    for lens in _length_cases(requests, width, gen):
                        mask = _row_mask(lens, width)
                        keep = mask.bool()
                        dirty = x.clone()
                        dirty[~keep] = NAN
                        _poison(tokens)
                        calls.clear()
                        with _masked(mask):
                            got = self._call(dirty, ids, weights)
                        self.assertEqual(calls, want)
                        self.assertTrue(
                            _bits_equal(got[keep], free[keep]), (form, lens)
                        )
                        self.assertTrue(
                            bool((got[~keep].float() == 0).all()), (form, lens)
                        )
                    # The graphs keep the form they were captured with.
                    graphs = self._graphs(tokens)
                self._replay_masks(graphs, requests, width, gen, rounds=1)
        for other in outs[1:]:
            self.assertTrue(_bits_equal(other, outs[0]))


if __name__ == "__main__":
    unittest.main()
