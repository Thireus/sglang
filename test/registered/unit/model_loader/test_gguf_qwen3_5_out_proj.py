"""The Gated-DeltaNet out_proj of a Qwen3.5-family GGUF, for every quant type.

llama.cpp stores the value heads of a Gated-DeltaNet layer in its own order.
For out_proj the heads are groups of INPUT columns. When a quant block divides
the value head dim (32-value types) the loader moves the packed column groups
back to the HF order; when it does not (every 256-value type, every type with
a row header) the packed bytes stay as they are and the layer gathers its input
into the file's order at run time. These tests check, on the CPU, with the
bit-exact numpy dequantisers of sglang.test.kernels.ggml_quant_reference:

- the fast path still moves the bytes and gives the HF weights bit for bit;
- the two paths compute the same products (the same float terms, bit for bit)
  and the same output up to the order of the sum;
- a 256-value type and a row-header type now pass the rule and give the HF
  layer's output through the gather;
- the gather is set on the right layers only, refused under a split input or
  a method that cannot read it, and applied by GGUFLinearMethod.apply.
"""

import unittest

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.gguf import GGUFConfig, GGUFLinearMethod
from sglang.srt.model_loader.gguf_name_maps import Qwen3_5GGUFWeightTransform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels import ggml_quant_reference as ref
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

N_K, N_V, D_K, D_V, HIDDEN = 2, 6, 128, 128, 64
NAME = "model.layers.{}.linear_attn.out_proj.qweight"

# 32-value blocks: the packed columns can move (today's path).
FAST_TYPES = {"Q8_0": 8, "Q4_0": 2, "Q5_1": 7, "IQ4_NL": 20}
# 256-value blocks, or a row header: the input is gathered instead.
GATHER_TYPES = {"Q4_K": 12, "Q6_K": 14, "IQ4_XS": 23, "IQ2_S": 22, "IQ4_KS": 144}


def make_config(**kw):
    c = dict(
        linear_num_key_heads=N_K,
        linear_num_value_heads=N_V,
        linear_key_head_dim=D_K,
        linear_value_head_dim=D_V,
    )
    c.update(kw)
    return PretrainedConfig(**c)


def file_tensor(tid, seed):
    """Random valid packed rows in the file's head order, and their values."""
    rows = ref.random_rows(tid, HIDDEN, N_V * D_V, np.random.default_rng(seed))
    return rows, ref.dequantize(tid, rows, N_V * D_V).astype(np.float64)


def hf_weight(w_file):
    """The HF-order weight, by the loader's unquantised path (real columns
    moved by ``_permute_head_blocks``), which does not use the gather."""
    t = Qwen3_5GGUFWeightTransform(make_config())
    out = t(
        "model.layers.0.linear_attn.out_proj.weight", torch.from_numpy(w_file), None
    )
    return out.numpy()


class FakeOutProj(nn.Module):
    def __init__(self, qweight, weight_type, split=False):
        super().__init__()
        self.quant_method = GGUFLinearMethod(GGUFConfig())
        self.input_size = qweight.shape[1]
        self.input_size_per_partition = self.input_size // (2 if split else 1)
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.qweight.shard_id = []
        self.qweight_type = nn.Parameter(
            torch.zeros(1, dtype=torch.uint8), requires_grad=False
        )
        self.qweight_type.weight_type = weight_type


def fake_model(layers, make):
    model = nn.Module()
    model.layers = nn.ModuleList()
    for i in range(max(layers) + 1):
        block = nn.Module()
        block.linear_attn = nn.Module()
        block.linear_attn.out_proj = make(i)
        model.layers.append(block)
    root = nn.Module()
    root.model = model
    return root


class TestOutProjRule(CustomTestCase):
    def setUp(self):
        self.t = Qwen3_5GGUFWeightTransform(make_config())
        self.cols = self.t.out_proj_input_columns()
        self.x = np.random.default_rng(7).standard_normal((5, N_V * D_V))

    def test_columns_are_a_permutation_of_whole_heads(self):
        self.assertEqual(sorted(self.cols), list(range(N_V * D_V)))
        for p in range(N_V):
            head = self.cols[p * D_V : (p + 1) * D_V]
            h = head[0] // D_V
            self.assertEqual(head, list(range(h * D_V, (h + 1) * D_V)))
            # file head p holds HF head h exactly when gguf_index_of_hf_head[h] == p
            self.assertEqual(self.t.gguf_index_of_hf_head[h], p)

    def test_fast_path_moves_bytes_and_matches_the_gather(self):
        for name, tid in FAST_TYPES.items():
            with self.subTest(type=name):
                rows, w_file = file_tensor(tid, tid)
                w_hf = hf_weight(w_file)
                moved = self.t(NAME.format(0), torch.from_numpy(rows), tid)
                self.assertFalse(self.t.input_order_kept)
                w_moved = ref.dequantize(tid, moved.numpy(), N_V * D_V)
                # the moved bytes decode to the HF weights, bit for bit
                self.assertTrue(np.array_equal(w_moved.astype(np.float64), w_hf))
                # the gather path, on the same file bytes, multiplies the same
                # pairs of numbers: each row's terms are equal as a multiset
                xg = self.x[:, self.cols]
                for r in range(3):
                    a = np.sort((self.x * w_hf[r]).ravel())
                    b = np.sort((xg * w_file[r]).ravel())
                    self.assertTrue(np.array_equal(a, b))
                # and the sums agree up to the order of the additions
                y_fast = self.x @ w_hf.T
                y_gather = xg @ w_file.T
                scale = np.abs(self.x) @ np.abs(w_hf).T
                self.assertLess(np.max(np.abs(y_fast - y_gather) / scale), 1e-14)

    def test_256_value_and_row_header_types_pass_the_rule(self):
        for name, tid in GATHER_TYPES.items():
            with self.subTest(type=name):
                t = Qwen3_5GGUFWeightTransform(make_config())
                rows, w_file = file_tensor(tid, 100 + tid)
                kept = t(NAME.format(3), torch.from_numpy(rows), tid)
                self.assertTrue(torch.equal(kept, torch.from_numpy(rows)))
                self.assertEqual(
                    t.input_order_kept, {"model.layers.3.linear_attn.out_proj"}
                )
                self.assertEqual(t.counts.get("out_proj_input_order"), 1)
                w_hf = hf_weight(w_file)
                y_hf = self.x @ w_hf.T
                y_gather = self.x[:, self.cols] @ w_file.T
                scale = np.abs(self.x) @ np.abs(w_hf).T
                self.assertLess(np.max(np.abs(y_hf - y_gather) / scale), 1e-14)

    def test_wrong_row_bytes_are_refused(self):
        rows, _ = file_tensor(12, 1)
        with self.assertRaises(ValueError):
            self.t(NAME.format(0), torch.from_numpy(rows[:, :-144]), 12)

    def test_unquantised_columns_still_move(self):
        w_file = np.random.default_rng(3).standard_normal((HIDDEN, N_V * D_V))
        out = self.t(
            "model.layers.0.linear_attn.out_proj.weight",
            torch.from_numpy(w_file.astype(np.float32)),
            None,
        )
        # HF value head h is file head gguf_index_of_hf_head[h]
        for h, p in enumerate(self.t.gguf_index_of_hf_head):
            self.assertTrue(
                np.array_equal(
                    out.numpy()[:, h * D_V : (h + 1) * D_V],
                    w_file[:, p * D_V : (p + 1) * D_V].astype(np.float32),
                )
            )
        self.assertTrue(
            np.array_equal(out.numpy(), hf_weight(w_file.astype(np.float32)))
        )
        self.assertFalse(self.t.input_order_kept)


class TestFinishLoading(CustomTestCase):
    def kept_transform(self, layers):
        t = Qwen3_5GGUFWeightTransform(make_config())
        for i in layers:
            rows, _ = file_tensor(12, i)
            t(NAME.format(i), torch.from_numpy(rows), 12)
        return t

    def test_sets_the_gather_on_those_layers_only(self):
        t = self.kept_transform([1, 3])
        model = fake_model(
            [3],
            lambda i: FakeOutProj(torch.zeros(HIDDEN, 4 * 144, dtype=torch.uint8), 12),
        )
        t.finish_loading(model)
        for i, block in enumerate(model.model.layers):
            index = getattr(block.linear_attn.out_proj, "gguf_input_index", None)
            if i in (1, 3):
                self.assertEqual(index.dtype, torch.long)
                self.assertEqual(index.tolist(), t.out_proj_input_columns())
            else:
                self.assertIsNone(index)
        self.assertEqual(t.counts["out_proj_input_gather"], 2)

    def test_nothing_kept_nothing_set(self):
        t = Qwen3_5GGUFWeightTransform(make_config())
        model = fake_model(
            [0], lambda i: FakeOutProj(torch.zeros(HIDDEN, 8, dtype=torch.uint8), 12)
        )
        t.finish_loading(model)
        self.assertFalse(
            hasattr(model.model.layers[0].linear_attn.out_proj, "gguf_input_index")
        )

    def test_split_input_is_refused(self):
        t = self.kept_transform([0])
        model = fake_model(
            [0],
            lambda i: FakeOutProj(
                torch.zeros(HIDDEN, 8, dtype=torch.uint8), 12, split=True
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "tp-size 1"):
            t.finish_loading(model)

    def test_other_linear_method_is_refused(self):
        t = self.kept_transform([0])

        def make(i):
            m = FakeOutProj(torch.zeros(HIDDEN, 8, dtype=torch.uint8), 12)
            m.quant_method = object()
            return m

        with self.assertRaises(NotImplementedError):
            t.finish_loading(fake_model([0], make))


class TestApplyGathers(CustomTestCase):
    def test_apply_reads_the_gather(self):
        # An unquantised weight in the file's order is enough to see that
        # GGUFLinearMethod.apply gathers its input (x @ qweight.T path).
        t = Qwen3_5GGUFWeightTransform(make_config())
        cols = t.out_proj_input_columns()
        rng = np.random.default_rng(11)
        w_file = rng.standard_normal((HIDDEN, N_V * D_V)).astype(np.float32)
        x = torch.from_numpy(rng.standard_normal((4, N_V * D_V)).astype(np.float32))
        layer = FakeOutProj(torch.from_numpy(w_file), 0)
        method = layer.quant_method
        plain = method.apply(layer, x)
        layer.gguf_input_index = torch.tensor(cols, dtype=torch.long)
        gathered = method.apply(layer, x)
        self.assertTrue(torch.equal(gathered, x[:, cols] @ torch.from_numpy(w_file).T))
        self.assertFalse(torch.equal(gathered, plain))
        w_hf = hf_weight(w_file.astype(np.float64))
        want = x.double().numpy() @ w_hf.T
        self.assertLess(np.max(np.abs(gathered.double().numpy() - want)), 1e-3)


if __name__ == "__main__":
    unittest.main()
