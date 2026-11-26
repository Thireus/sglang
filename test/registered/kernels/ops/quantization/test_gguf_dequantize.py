"""GPU dequantisation of the ggml / ik_llama.cpp types served by dequantise-and-multiply.

For every type of ``GGUF_DEQUANTIZE_TYPES`` the kernel's fp32 output must equal,
bit for bit, the numpy reference of sglang.test.kernels.ggml_quant_reference
(itself bit-exact against ik_llama.cpp's CPU code), and a bf16 / fp16 output must
equal that reference rounded once to the dtype. Checked on random valid rows, on
rows ik's quantiser wrote (IK_LIBGGML = ik_llama.cpp's libggml.so, else skipped),
on real tensors of a GGUF quant collection (SGLANG_GGUF_SPLIT_COLLECTION = a
directory of ``*-<TYPE>-SPECIAL_SPLIT`` sets, else skipped: three tensors a type,
under 64 MB in all), eager and in a replayed CUDA graph, into NaN-filled buffers
with guard rows, and with row counts off the launch grain. Then the wiring: the
dense matmul through the shared scratch buffer, the embedding lookup and the
GGUF MoE fallback of layers/quantization/gguf.py.
"""

import glob
import os
import unittest

import numpy as np
import torch

from sglang.kernels.ops.quantization.gguf_dequantize import (
    GGUF_DEQUANTIZE_TYPES,
    gguf_dequantize,
    gguf_dequantize_columns,
    gguf_dequantize_matmul,
    gguf_dequantize_scratch_bytes,
    gguf_dequantize_scratch_values,
    gguf_dequantize_warmup,
)
from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import gguf as gguf_layer
from sglang.srt.model_loader.gguf_ik import repack as RP
from sglang.srt.model_loader.gguf_ik import types as T
from sglang.srt.model_loader.gguf_ik.reader import IkGGUFReader
from sglang.srt.model_loader.gguf_ik.rowgroups import split_row_groups
from sglang.srt.model_loader.gguf_ik.split import read_tensors_map
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels import ggml_quant_reference as ref
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="nightly", runner_config="1-gpu-large")

TYPE_IDS = sorted(GGUF_DEQUANTIZE_TYPES)
# a 16-bit float besides bf16, for one type a family
FP16_IDS = (133, 134, 141, 156, 158, 219, 399)
LIB = os.environ.get(ref.LIB_ENV)
COLLECTION = os.environ.get("SGLANG_GGUF_SPLIT_COLLECTION")
SHARD_BUDGET = 64 << 20
SHARD_VALUES = 4 << 20  # values compared a tensor (its leading rows)


def kernel_rows(tid: int, disk_rows: np.ndarray) -> torch.Tensor:
    """Rows as a GGUF file holds them -> the packed rows the kernel reads, on the GPU."""
    t = T.BY_ID[tid]
    if t.name in T.ROW_GROUPS:
        disk_rows = split_row_groups(t, disk_rows)
    return torch.from_numpy(np.ascontiguousarray(disk_rows)).cuda()


def same_bits(y: torch.Tensor, want: torch.Tensor) -> bool:
    """Equal as bit patterns (so that -0.0 and 0.0 differ and a NaN would show)."""
    view = {4: torch.int32, 2: torch.int16}[y.dtype.itemsize]
    return y.shape == want.shape and torch.equal(
        y.contiguous().view(view).cpu(), want.contiguous().view(view).cpu()
    )


def rounded(want32: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(want32)).to(dtype)


def heavy_tailed(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Student-t rows with per-row scales, 10 % exact zeros (half of them -0.0)."""
    x = rng.standard_t(3, size=(n, k)) * np.exp(rng.normal(-4.0, 0.5, size=(n, 1)))
    z = rng.random((n, k)) < 0.10
    x[z] = 0.0
    x[z & (rng.random((n, k)) < 0.5)] = -0.0
    return x.astype(np.float32)


class TestGgufDequantize(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        gguf_dequantize_warmup(torch.float32, TYPE_IDS)
        gguf_dequantize_warmup(torch.bfloat16, TYPE_IDS)
        gguf_dequantize_warmup(torch.float16, FP16_IDS)

    def _check(self, tid, disk_rows, k, label, dtypes=(torch.float32, torch.bfloat16)):
        name = GGUF_DEQUANTIZE_TYPES[tid][0]
        want = ref.dequantize(tid, disk_rows, k)
        w = kernel_rows(tid, disk_rows)
        for dtype in dtypes:
            y = gguf_dequantize(w, tid, dtype)
            self.assertTrue(
                same_bits(y, rounded(want, dtype)),
                f"{name} {label} {dtype}: differs from the reference",
            )
        return w, want

    def test_tables_agree(self):
        """Kernel table, loader registry and reference describe the same layouts."""
        self.assertEqual(set(TYPE_IDS), set(T.GGUF_DEQUANT_IDS))
        self.assertEqual(set(TYPE_IDS), set(gguf_layer.JIT_DEQUANT_TYPES))
        for tid, (name, block, block_bytes, meta) in GGUF_DEQUANTIZE_TYPES.items():
            t, q = T.BY_ID[tid], ref.TYPES[tid]
            self.assertEqual(
                (t.name, t.blck_size, t.type_size, t.row_meta),
                (name, block, block_bytes, meta),
            )
            self.assertEqual(
                (q.name, q.block, q.block_bytes, q.header),
                (name, block, block_bytes, meta),
            )
            self.assertEqual(q.rows, T.ROW_GROUPS.get(name, 1))
            self.assertEqual(
                gguf_dequantize_columns(tid, T.row_size(t, 5120)), 5120, name
            )

    @torch.inference_mode()
    def test_random_rows_are_bit_exact(self):
        rng = np.random.default_rng(156)
        for tid in TYPE_IDS:
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                dtypes = (torch.float32, torch.bfloat16)
                if tid in FP16_IDS:
                    dtypes += (torch.float16,)
                self._check(
                    tid, ref.random_rows(tid, 16, 2304, rng), 2304, "random", dtypes
                )

    @unittest.skipUnless(LIB, f"set {ref.LIB_ENV} to ik_llama.cpp's libggml.so")
    @torch.inference_mode()
    def test_quantiser_rows_are_bit_exact(self):
        lib = ref.LibGGML(LIB)
        rng = np.random.default_rng(157)
        for tid in TYPE_IDS:
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                k = 2304
                x = heavy_tailed(rng, 8, k)
                rows = lib.quantize(tid, x, np.ones(k, np.float32))
                self._check(tid, rows, k, "quantiser rows")

    @torch.inference_mode()
    def test_nan_filled_output_and_guard_rows(self):
        """Every value of the output is written and nothing beside it."""
        rng = np.random.default_rng(158)
        for tid in TYPE_IDS:
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                n, k = 8, 768
                rows = ref.random_rows(tid, n, k, rng)
                want = rounded(ref.dequantize(tid, rows, k), torch.float32)
                w = kernel_rows(tid, rows)
                buf = torch.full((n + 2, k), float("nan"), device="cuda")
                y = gguf_dequantize(w, tid, torch.float32, out=buf[1 : n + 1])
                self.assertEqual(y.data_ptr(), buf[1].data_ptr())
                self.assertTrue(same_bits(buf[1 : n + 1], want))
                self.assertTrue(bool(buf[0].isnan().all() and buf[n + 1].isnan().all()))

    @torch.inference_mode()
    def test_row_counts_off_the_launch_grain(self):
        """A launch is cut into CTAs of 256 units of 8 values; rows never fill them evenly here."""
        rng = np.random.default_rng(159)
        for tid in TYPE_IDS:
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                for k in (256, 1280):
                    rows = ref.random_rows(tid, 16, k, rng)
                    want = rounded(ref.dequantize(tid, rows, k), torch.bfloat16)
                    w = kernel_rows(tid, rows)
                    for n in (1, 3, 13):
                        self.assertNotEqual(n * k // 8 % 256, 0)
                        y = gguf_dequantize(w[:n], tid, torch.bfloat16)
                        self.assertTrue(same_bits(y, want[:n]), f"{n} rows of {k}")
                empty = gguf_dequantize(w[:0], tid, torch.bfloat16)
                self.assertEqual(tuple(empty.shape), (0, 1280))

    @torch.inference_mode()
    def test_graph_replay_matches_the_reference(self):
        rng = np.random.default_rng(160)
        for tid in TYPE_IDS:
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                n, k = 8, 1024
                first = ref.random_rows(tid, n, k, rng)
                second = ref.random_rows(tid, n, k, rng)
                w = kernel_rows(tid, first)
                gguf_dequantize(w, tid, torch.bfloat16)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = gguf_dequantize(w, tid, torch.bfloat16)
                w.copy_(kernel_rows(tid, second))
                y.fill_(float("nan"))
                graph.replay()
                want = rounded(ref.dequantize(tid, second, k), torch.bfloat16)
                self.assertTrue(same_bits(y, want))
                self.assertTrue(same_bits(gguf_dequantize(w, tid, torch.bfloat16), y))

    @torch.inference_mode()
    def test_repacks_through_the_loader_inverse(self):
        """A row-interleaved repack of a type served here is that type's rows again."""
        rng = np.random.default_rng(161)
        seen = 0
        for name, (base, group) in sorted(T.REPACKS.items()):
            tid = int(T.BY_NAME[base])
            if tid not in GGUF_DEQUANTIZE_TYPES:
                continue
            with self.subTest(name):
                k = 512
                rows = ref.random_rows(tid, 2 * group, k, rng)
                stored = RP.do_repack(T.BY_NAME[name], rows)
                self.assertFalse(np.array_equal(stored, rows))
                back, restored = RP.undo_repack(T.BY_NAME[name], stored)
                self.assertEqual(int(back), tid)
                self._check(tid, restored, k, f"restored from {name}")
                seen += 1
        self.assertEqual(seen, 8)

    def test_wrong_row_size_is_refused(self):
        for tid in TYPE_IDS:
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                rb = ref.TYPES[tid].row_bytes(256)
                with self.assertRaises(ValueError):
                    gguf_dequantize(
                        torch.zeros(2, rb + 1, dtype=torch.uint8, device="cuda"),
                        tid,
                        torch.bfloat16,
                    )
                # the kernel checks the row size against its own layout as well
                with self.assertRaises(Exception):
                    gguf_dequantize(
                        torch.zeros(2, rb, dtype=torch.uint8, device="cuda"),
                        tid,
                        torch.bfloat16,
                        out=torch.zeros(2, 512, dtype=torch.bfloat16, device="cuda"),
                    )
        with self.assertRaises(ValueError):
            gguf_dequantize(
                torch.zeros(2, 136, dtype=torch.uint8, device="cuda"), 22, torch.float32
            )

    # ------------------------------------------------------------------ wiring
    def _float_product(self, x, want32, dtype):
        """x @ W.T in fp64, W being the reference rounded to ``dtype``."""
        w = rounded(want32, dtype).double()
        return x.double().cpu() @ w.T

    def _assert_product(self, y, exact, label):
        # the dense matmul's own accuracy: a few bf16 units of the largest output
        y = y.double().cpu()
        scale = exact.abs().max()
        err = (y - exact).abs().max()
        self.assertLessEqual(float(err), float(scale) * 2.0**-6, label)

    def _assert_dense(self, y, x, want32, label):
        """y is, bit for bit, the dense matmul of x with the reference weights."""
        dense = rounded(want32, x.dtype).cuda()
        self.assertTrue(torch.equal(y, x @ dense.T), label)

    @torch.inference_mode()
    def test_matmul_through_the_scratch_buffer(self):
        rng = np.random.default_rng(162)
        gen = torch.Generator(device="cuda").manual_seed(162)
        for tid in (133, 141, 151, 156, 157, 154, 229, 399):
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                n, k = 640, 1024
                rows = ref.random_rows(tid, n, k, rng)
                want = ref.dequantize(tid, rows, k)
                w = kernel_rows(tid, rows)
                x = torch.randn(5, k, generator=gen, device="cuda").bfloat16()
                y = gguf_dequantize_matmul(x, w, tid)
                self.assertEqual(tuple(y.shape), (5, n))
                self.assertTrue(
                    torch.equal(y, x @ gguf_dequantize(w, tid, torch.bfloat16).T)
                )
                self._assert_product(
                    y, self._float_product(x, want, torch.bfloat16), "one run"
                )
                # 1 MiB of bf16 scratch is 512 rows of 1024: two runs of rows
                with envs.SGLANG_GGUF_DEQUANT_SCRATCH_MB.override(1):
                    self.assertEqual(
                        gguf_dequantize_scratch_values(n, k, torch.bfloat16), 512 * k
                    )
                    x32 = x.float()
                    self.assertEqual(
                        gguf_dequantize_scratch_values(n, k, torch.float32), 256 * k
                    )
                    y32 = gguf_dequantize_matmul(x32, w, tid)
                full = gguf_dequantize(w, tid, torch.float32)
                parts = [x32 @ full[r : r + 256].T for r in range(0, n, 256)]
                self.assertTrue(torch.equal(y32, torch.cat(parts, dim=-1)))
                self._assert_product(
                    y32, self._float_product(x32, want, torch.float32), "runs of rows"
                )

    @torch.inference_mode()
    def test_linear_and_embedding_methods(self):
        rng = np.random.default_rng(163)
        gen = torch.Generator(device="cuda").manual_seed(163)
        for tid in TYPE_IDS:
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                n, k = 64, 512
                rows = ref.random_rows(tid, n, k, rng)
                want = ref.dequantize(tid, rows, k)
                w = kernel_rows(tid, rows)
                for tokens in (1, 3, 40):
                    x = torch.randn(tokens, k, generator=gen, device="cuda").bfloat16()
                    y = gguf_layer.fused_mul_mat_gguf(x, w, tid)
                    self.assertEqual(y.dtype, torch.bfloat16)
                    self._assert_dense(y, x, want, f"{tokens} tokens")
                    self._assert_product(
                        y, self._float_product(x, want, torch.bfloat16), f"{tokens}"
                    )
                empty = gguf_layer.fused_mul_mat_gguf(x[:0], w, tid)
                self.assertEqual(tuple(empty.shape), (0, n))
                dense = gguf_layer.dequantize_gguf_weight(w, tid, torch.bfloat16)
                self.assertTrue(same_bits(dense, rounded(want, torch.bfloat16)))
                ids = torch.tensor([[5, 0, 63], [63, 17, 5]], device="cuda")
                e = gguf_layer.apply_gguf_embedding(
                    ids, w, tid, k, dtype=torch.bfloat16
                )
                self.assertEqual(tuple(e.shape), (2, 3, k))
                self.assertTrue(
                    same_bits(
                        e.reshape(6, k),
                        rounded(want[ids.flatten().cpu().numpy()], torch.bfloat16),
                    )
                )

    @torch.inference_mode()
    def test_linear_layer_loads_and_serves(self):
        """A GGUF linear layer of such a type: type id, packed rows, the load hook, a forward."""
        rng = np.random.default_rng(165)
        gen = torch.Generator(device="cuda").manual_seed(165)
        for tid in (156, 158, 219, 399):
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                n, k = 96, 512
                rows = ref.random_rows(tid, n, k, rng)
                want = ref.dequantize(tid, rows, k)
                layer = ReplicatedLinear(
                    k,
                    n,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=gguf_layer.GGUFConfig(),
                ).cuda()
                layer.weight_loader(layer.qweight_type, torch.tensor(tid))
                layer.weight_loader(layer.qweight, kernel_rows(tid, rows).cpu())
                layer.quant_method.process_weights_after_loading(layer)
                self.assertEqual(layer.qweight_type.weight_type, tid)
                self.assertEqual(layer.qweight.device.type, "cuda")
                self.assertGreaterEqual(gguf_dequantize_scratch_bytes(), n * k * 2)
                x = torch.randn(7, k, generator=gen, device="cuda").bfloat16()
                y, _ = layer(x)
                self._assert_dense(y, x, want, "layer")
                self._assert_product(
                    y, self._float_product(x, want, torch.bfloat16), "layer"
                )

    @torch.inference_mode()
    def test_moe_fallback_reaches_the_dequantiser(self):
        """fused_moe_gguf has no fused kernel for these types: its per-expert loop multiplies dense."""
        rng = np.random.default_rng(164)
        gen = torch.Generator(device="cuda").manual_seed(164)
        E, inter, hidden, tokens, top_k = 4, 256, 512, 5, 2
        for tid in (156, 153, 219):
            with self.subTest(GGUF_DEQUANTIZE_TYPES[tid][0]):
                up = ref.random_rows(tid, E * 2 * inter, hidden, rng)
                down = ref.random_rows(tid, E * hidden, inter, rng)
                w1 = kernel_rows(tid, up).view(E, 2 * inter, -1)
                w2 = kernel_rows(tid, down).view(E, hidden, -1)
                d1 = rounded(ref.dequantize(tid, up, hidden), torch.bfloat16).double()
                d2 = rounded(ref.dequantize(tid, down, inter), torch.bfloat16).double()
                d1, d2 = d1.view(E, 2 * inter, hidden), d2.view(E, hidden, inter)
                x = torch.randn(tokens, hidden, generator=gen, device="cuda").bfloat16()
                ids = torch.tensor(
                    [[0, 1], [3, -1], [2, 2], [-1, -1], [1, 3]], device="cuda"
                )
                wts = torch.rand(tokens, top_k, generator=gen, device="cuda").bfloat16()
                y = gguf_layer.fused_moe_gguf(x, w1, w2, wts, ids, tid, tid, "silu")
                want = torch.zeros(tokens, hidden, dtype=torch.float64)
                for t in range(tokens):
                    for r in range(top_k):
                        e = int(ids[t, r])
                        if e < 0:
                            continue
                        h = (x[t].double().cpu() @ d1[e].T).to(torch.bfloat16).double()
                        act = torch.nn.functional.silu(h[:inter]) * h[inter:]
                        act = act.to(torch.bfloat16).double()
                        out = (act @ d2[e].T).to(torch.bfloat16).double()
                        want[t] += (out * wts[t, r].double().cpu()).to(torch.bfloat16)
                got = y.double().cpu()
                self.assertEqual(int(got[3].count_nonzero()), 0)
                scale = float(want.abs().max())
                self.assertLessEqual(float((got - want).abs().max()), scale * 2.0**-5)

    # ------------------------------------------------------------- real shards
    @unittest.skipUnless(
        COLLECTION and os.path.isdir(COLLECTION or ""),
        "SGLANG_GGUF_SPLIT_COLLECTION not set",
    )
    @torch.inference_mode()
    def test_real_shard_tensors(self):
        """Three tensors of each set, one a row length (the leading rows of the larger ones), read as the loader reads them."""
        names = {v[0]: k for k, v in GGUF_DEQUANTIZE_TYPES.items()}
        sets = dict(names)
        sets.update(
            {
                r: names[b]
                for r, (b, _rows) in T.REPACKS.items()
                if b in names and r not in names
            }
        )
        seen = 0
        for set_name, tid in sorted(sets.items()):
            hits = sorted(
                glob.glob(os.path.join(COLLECTION, f"*-{set_name}-SPECIAL_SPLIT"))
            )
            if not hits:
                continue
            with self.subTest(set_name):
                rows = [
                    r
                    for r in read_tensors_map(os.path.join(hits[0], "tensors.map"))
                    if r.dtype.upper() == set_name and len(r.shape) == 2
                ]
                self.assertTrue(rows, f"no {set_name} tensor in {hits[0]}")
                # the smallest tensor of each row length
                picked, lengths = [], set()
                for r in sorted(rows, key=lambda r: (r.shape[0], r.shape[1], r.name)):
                    if r.shape[0] not in lengths and len(picked) < 3:
                        lengths.add(r.shape[0])
                        picked.append(r)
                budget = SHARD_BUDGET // 3
                for r in picked:
                    reader = IkGGUFReader(os.path.join(hits[0], r.shard))
                    (tensor,) = [t for t in reader.tensors if t.name == r.name]
                    stored_type = tensor.tensor_type
                    k = int(tensor.shape[0])
                    group = (
                        8 if stored_type.name != GGUF_DEQUANTIZE_TYPES[tid][0] else 1
                    )
                    group = max(group, T.ROW_GROUPS.get(stored_type.name, 1))
                    n = min(tensor.data.shape[0], budget // tensor.data.shape[1])
                    n = min(n, SHARD_VALUES // k)
                    n = max(group, n // group * group)
                    stored = np.ascontiguousarray(tensor.data[:n])
                    if stored_type.name in T.REPACKS:
                        back, stored = RP.undo_repack(stored_type, stored)
                        self.assertEqual(int(back), tid)
                    self._check(tid, stored, k, f"{r.name} ({n} rows of {k})")
                    seen += 1
        self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
