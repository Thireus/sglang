"""Block-fp8 linears served as MXFP8 on SM12x (DeepSeek-V4.1's 32-wide ue8m0
blocks): FlashInfer b12x for K % 128 == 0 when FlashInfer can run it (SM120/121,
CUDA 13+, nvidia-cutlass-dsl >= 4.6.0) and its kernel fences the pipeline slot
refill (FlashInfer's own module, or the fenced copy of the 0.7.2 module put in its
place), its CUTLASS MXFP8 GEMM otherwise, after sgl-project/sglang#45262. Every
other backend and checkpoint is unchanged."""

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from functools import partial
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend as B
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# DeepSeek-V4.1-Flash's dense ue8m0 block-fp8 linears at attention TP 1, (N, K):
# wq_a + wkv fused (SGLANG_OPT_FUSE_WQA_WKV), the shared expert, the indexer of the
# index source layers and the Engram projection of layers 1 and 14.
V41_LINEARS = {
    "attn.wqkv_a": (1792, 5120),
    "attn.wq_b": (32768, 1280),
    "attn.wo_b": (5120, 8192),
    "attn.indexer.wq_b": (4096, 1280),
    "ffn.shared_experts.gate_up_proj": (4608, 5120),
    "ffn.shared_experts.down_proj": (5120, 2304),
    "engram.wkv": (25600, 6144),
}
B12X_LOG = "Use FlashInfer b12x for block-fp8-as-MXFP8 linears"
FENCE_LOG = "FlashInfer b12x slot refill fenced after the MMA warps' reads"
ACQUIRE = "                    mainloop_pipeline.producer_acquire(mainloop_producer_state)\n"
FENCE = '                    cute.arch.fence_proxy("async.shared", space="cta")\n'


class _FakeMmMxfp8:
    """FlashInfer's mm_mxfp8 capability check: b12x on SM120/SM121 only."""

    def __init__(self):
        self.calls = []

    def is_backend_supported(self, backend, cc=None):
        self.calls.append((backend, cc))
        return backend == "b12x" and cc in (120, 121)


def _flashinfer_probes(cuda_major, dsl_ok):
    dsl = types.ModuleType("flashinfer.gemm.gemm_mm_mxfp8_cute_dsl")
    dsl._b12x_mxfp8_dsl_supported = lambda: dsl_ok
    jit = types.ModuleType("flashinfer.jit.cpp_ext")
    jit.get_cuda_version = lambda: types.SimpleNamespace(major=cuda_major)
    return {dsl.__name__: dsl, jit.__name__: jit}


class TestDsv41Sm120B12xMxfp8(CustomTestCase):
    def setUp(self):
        fp8_utils._flashinfer_b12x_mxfp8_usable.cache_clear()
        self.addCleanup(fp8_utils._flashinfer_b12x_mxfp8_usable.cache_clear)
        self.fence = "copy"
        self.linear = MagicMock(name="flashinfer_mxfp8_blockscaled_linear")
        p = patch.object(fp8_utils, "flashinfer_mxfp8_blockscaled_linear", self.linear)
        p.start()
        self.addCleanup(p.stop)

    def _dispatch(self, backend=B.FLASHINFER_CUTLASS, *, sm=120, cuda=13, dsl=True):
        mm = _FakeMmMxfp8()
        with (
            patch.object(fp8_utils, "_raw_flashinfer_mm_mxfp8", mm, create=True),
            patch.object(fp8_utils, "get_device_sm", return_value=sm),
            patch.dict(sys.modules, _flashinfer_probes(cuda, dsl)),
            patch.object(
                fp8_utils, "_install_b12x_refill_fence", return_value=self.fence
            ) as self.install,
        ):
            return fp8_utils.dispatch_block_fp8_mxfp8_linear(backend), mm

    def _run(self, fn, n, k, **kw):
        self.linear.reset_mock()
        w = torch.empty((n, k), dtype=torch.float8_e4m3fn, device="meta")
        x = torch.empty((3, k), dtype=torch.bfloat16, device="meta")
        fn(input=x, weight=w, weight_scale=None, input_scale=None, bias=None, **kw)
        self.assertEqual(self.linear.call_count, 1)
        return self.linear.call_args.kwargs

    def _assert_cutlass_partial(self, fn):
        self.assertIsInstance(fn, partial)
        self.assertEqual(fn.keywords, {"backend": "cutlass", "pin_tactic": True})

    def test_sm120_runs_every_v41_linear_on_b12x(self):
        fn, _ = self._dispatch()
        self.assertIs(fn, fp8_utils._b12x_or_cutlass_mxfp8_linear)
        for name, (n, k) in V41_LINEARS.items():
            with self.subTest(name):
                kw = self._run(fn, n, k)
                self.assertEqual(kw["backend"], "b12x")
                self.assertTrue(kw["pin_tactic"])

    def test_sm121_also_takes_b12x(self):
        fn, _ = self._dispatch(sm=121)
        self.assertIs(fn, fp8_utils._b12x_or_cutlass_mxfp8_linear)

    def test_k_not_a_multiple_of_128_stays_on_cutlass(self):
        fn, _ = self._dispatch()
        for k in (32, 96, 288, 5120 + 32):
            with self.subTest(k=k):
                kw = self._run(fn, 512, k)
                self.assertEqual(kw["backend"], "cutlass")
                self.assertTrue(kw["pin_tactic"])

    def test_pin_tactic_stays_overridable(self):
        # Fp8LinearMethod.apply passes pin_tactic when prefill autotune is armed;
        # the partials take it as an override, so must the b12x route.
        fn, _ = self._dispatch()
        self.assertFalse(self._run(fn, 1792, 5120, pin_tactic=False)["pin_tactic"])
        self.assertFalse(self._run(fn, 512, 288, pin_tactic=False)["pin_tactic"])

    def test_one_log_line_per_process(self):
        with self.assertLogs(fp8_utils.logger, level="INFO") as logs:
            self._dispatch()
            self._dispatch()
        hits = [r for r in logs.output if B12X_LOG in r]
        self.assertEqual(len(hits), 1, logs.output)
        self.assertIn("K % 128 == 0 on SM120; CUTLASS for other K.", hits[0])
        fenced = [r for r in logs.output if FENCE_LOG in r]
        self.assertEqual(len(fenced), 1, logs.output)
        self.assertIn("fenced copy, 028b", fenced[0])

    def test_missing_cuda13_or_cutlass_dsl_keeps_cutlass(self):
        for cuda, dsl in ((12, True), (13, False)):
            with self.subTest(cuda=cuda, dsl=dsl):
                fp8_utils._flashinfer_b12x_mxfp8_usable.cache_clear()
                with self.assertLogs(fp8_utils.logger, level="WARNING") as logs:
                    fn, _ = self._dispatch(cuda=cuda, dsl=dsl)
                self._assert_cutlass_partial(fn)
                self.assertIn("stay on CUTLASS", logs.output[0])
                self.assertEqual(self._run(fn, 1792, 5120)["backend"], "cutlass")

    def test_devices_without_b12x_keep_cutlass(self):
        fn, mm = self._dispatch(sm=100)
        self._assert_cutlass_partial(fn)
        self.assertEqual(mm.calls, [("b12x", 100)])

    def test_other_mxfp8_backends_are_untouched(self):
        fn, mm = self._dispatch(B.FLASHINFER_CUTEDSL)
        self.assertIsInstance(fn, partial)
        self.assertEqual(fn.keywords, {"backend": "cute-dsl", "pin_tactic": True})
        fn, _ = self._dispatch(B.UNSUPPORTED)
        self.assertIs(fn, fp8_utils._unsupported_mxfp8_linear)
        self.assertEqual(mm.calls, [])

    def test_only_32_wide_ue8m0_blocks_take_the_mxfp8_view(self):
        # 128x128 block-fp8 (GLM, DeepSeek-V3 style) and non-ue8m0 scales never
        # reach dispatch_block_fp8_mxfp8_linear.
        with patch.object(
            fp8_utils,
            "resolve_block_fp8_mxfp8_backend",
            return_value=B.FLASHINFER_CUTLASS,
        ):
            self.assertTrue(fp8_utils.can_serve_block_fp8_as_mxfp8([32, 32], "ue8m0"))
            self.assertFalse(
                fp8_utils.can_serve_block_fp8_as_mxfp8([128, 128], "ue8m0")
            )
            self.assertFalse(fp8_utils.can_serve_block_fp8_as_mxfp8([32, 32], None))
            self.assertFalse(fp8_utils.can_serve_block_fp8_as_mxfp8(None, "ue8m0"))

    def test_unfenced_kernel_keeps_cutlass(self):
        # FlashInfer's kernel neither fences the refill nor is the 0.7.2 file the copy
        # was made from: b12x would race under concurrent work, so CUTLASS stays.
        self.fence = ""
        with self.assertLogs(fp8_utils.logger, level="WARNING") as logs:
            fn, _ = self._dispatch()
        self._assert_cutlass_partial(fn)
        self.assertIn("FlashInfer b12x MXFP8 GEMM unavailable", logs.output[0])
        self.assertIn("refill a pipeline slot before reading it", logs.output[0])
        self.assertFalse([r for r in logs.output if B12X_LOG in r])
        self.assertEqual(self._run(fn, 1792, 5120)["backend"], "cutlass")

    def test_fence_is_checked_only_where_cuda_and_dsl_allow_b12x(self):
        self._dispatch(cuda=12)
        self.install.assert_not_called()
        fp8_utils._flashinfer_b12x_mxfp8_usable.cache_clear()
        self._dispatch()
        self.install.assert_called_once()

    def test_fence_detection(self):
        fenced = "x = 1\n" + ACQUIRE + FENCE + "y = 2\n"
        self.assertTrue(fp8_utils._b12x_refill_is_fenced(fenced))
        self.assertFalse(fp8_utils._b12x_refill_is_fenced("x = 1\n" + ACQUIRE + "y = 2\n"))
        # the fence must follow the producer's empty-slot wait directly, and there is one wait
        self.assertFalse(fp8_utils._b12x_refill_is_fenced(ACQUIRE + "y = 2\n" + FENCE))
        self.assertFalse(fp8_utils._b12x_refill_is_fenced(ACQUIRE + FENCE + ACQUIRE + FENCE))
        self.assertFalse(fp8_utils._b12x_refill_is_fenced(FENCE))

    def test_fenced_copy_is_flashinfer_072_plus_the_fence(self):
        here = os.path.dirname(os.path.abspath(fp8_utils.__file__))
        with open(os.path.join(here, fp8_utils._B12X_FENCED_COPY), encoding="utf-8") as f:
            copy = f.read()
        self.assertTrue(fp8_utils._b12x_refill_is_fenced(copy))
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or not spec.submodule_search_locations:
            self.skipTest("FlashInfer is not installed")
        stock_path = os.path.join(
            spec.submodule_search_locations[0],
            "gemm",
            "kernels",
            "dense_blockscaled_gemm_sm120_b12x.py",
        )
        if not os.path.isfile(stock_path):
            self.skipTest("FlashInfer has no b12x kernel module")
        with open(stock_path, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != fp8_utils._B12X_KERNEL_SHA256:
            self.skipTest("installed FlashInfer is not 0.7.2's b12x kernel")
        stock = data.decode("utf-8")
        self.assertEqual(stock.count(ACQUIRE), 1)
        # The copy is a comment header, then FlashInfer's file with the fence after the
        # producer's empty-slot wait; one comment line of it is reworded, nothing else.
        start = copy.index(stock.split("\n", 1)[0] + "\n")
        self.assertTrue(all(line.startswith("# ") for line in copy[:start].splitlines()))
        got = copy[start:].split("\n")
        want = stock.replace(ACQUIRE, ACQUIRE + FENCE).split("\n")
        self.assertEqual(len(got), len(want))
        changed = [(a, b) for a, b in zip(got, want) if a != b]
        self.assertEqual(len(changed), 1, changed)
        self.assertTrue(changed[0][0].startswith("# ") and changed[0][1].startswith("# "))


class TestInstallB12xRefillFence(CustomTestCase):
    """_install_b12x_refill_fence against stand-in kernel modules (no FlashInfer)."""

    NAME = "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x"

    def setUp(self):
        fp8_utils._install_b12x_refill_fence.cache_clear()
        self.addCleanup(fp8_utils._install_b12x_refill_fence.cache_clear)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        p = patch.dict(sys.modules)
        p.start()
        self.addCleanup(p.stop)

    def _stock(self, text):
        path = os.path.join(self.dir.name, "stock_kernel.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        module = types.ModuleType(self.NAME)
        module.__file__ = path
        module.MARK = "stock"
        package = types.ModuleType(self.NAME.rpartition(".")[0])
        setattr(package, self.NAME.rpartition(".")[2], module)
        sys.modules[package.__name__] = package
        sys.modules[self.NAME] = module
        return module, hashlib.sha256(text.encode()).hexdigest()

    def _copy(self, text):
        path = os.path.join(self.dir.name, "fenced_copy.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _install(self, sha, copy_path):
        with (
            patch.object(fp8_utils, "_B12X_KERNEL_SHA256", sha),
            patch.object(fp8_utils, "_B12X_FENCED_COPY", copy_path),
        ):
            return fp8_utils._install_b12x_refill_fence()

    def test_known_file_is_replaced_by_the_fenced_copy(self):
        stock, sha = self._stock("MARK = 'stock'\n" + ACQUIRE.strip() + "\n")
        copy = self._copy("MARK = 'fenced'\n\n\ndef _load_loop():\n" + ACQUIRE + FENCE)
        self.assertEqual(self._install(sha, copy), "copy")
        module = sys.modules[self.NAME]
        self.assertIsNot(module, stock)
        self.assertEqual(module.MARK, "fenced")
        self.assertEqual(module.__file__, copy)
        package = sys.modules[self.NAME.rpartition(".")[0]]
        self.assertIs(getattr(package, self.NAME.rpartition(".")[2]), module)

    def test_flashinfer_fence_is_used_as_is(self):
        stock, _ = self._stock(ACQUIRE + FENCE)
        self.assertEqual(self._install("0" * 64, self._copy("raise SystemExit\n")), "flashinfer")
        self.assertIs(sys.modules[self.NAME], stock)

    def test_unknown_unfenced_file_is_refused(self):
        stock, _ = self._stock(ACQUIRE)
        self.assertEqual(self._install("0" * 64, self._copy(ACQUIRE + FENCE)), "")
        self.assertIs(sys.modules[self.NAME], stock)

    def test_a_copy_that_fails_to_load_leaves_flashinfer_in_place(self):
        stock, sha = self._stock(ACQUIRE)
        copy = self._copy("def _load_loop():\n" + ACQUIRE + FENCE + "\n\nraise ImportError('no')\n")
        self.assertEqual(self._install(sha, copy), "")
        self.assertIs(sys.modules[self.NAME], stock)

    def test_an_unfenced_copy_is_refused(self):
        stock, sha = self._stock(ACQUIRE)
        self.assertEqual(self._install(sha, self._copy("MARK = 'copy'\n")), "")
        self.assertIs(sys.modules[self.NAME], stock)


if __name__ == "__main__":
    unittest.main()
