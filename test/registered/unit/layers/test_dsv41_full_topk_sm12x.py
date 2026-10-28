import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.dsv4 import v41_indexer
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFullTopKDecodeRouting(CustomTestCase):
    """The full top-k of the ratio-1/2 index layers decodes through DeepGEMM's
    FP4 paged logits on SM100 only; SM12x takes the torch/Triton path."""

    def _make(self, **platform):
        with (
            patch.object(v41_indexer, "is_sm100_or_newer", return_value=True),
            patch.object(v41_indexer, "_use_deep_gemm_prefill", return_value=True),
            override_platform(**platform),
        ):
            return v41_indexer.make_full_topk_indexer(
                token_to_kv_pool=None, req_to_token=None
            )

    def test_sm12x_decodes_without_deep_gemm(self):
        indexer = self._make(is_sm120=True)
        self.assertFalse(indexer.use_deep_gemm_decode)
        # Prefill keeps DeepGEMM's flattened-K logits, which SM12x has.
        self.assertTrue(indexer.use_deep_gemm_prefill)

    def test_sm100_keeps_deep_gemm_decode(self):
        indexer = self._make(is_sm120=False)
        self.assertTrue(indexer.use_deep_gemm_decode)

    def test_sm12x_topk_decode_takes_torch_path(self):
        indexer = self._make(is_sm120=True)
        with (
            patch.object(indexer, "_torch_decode") as torch_decode,
            patch.object(indexer, "_deep_gemm_decode") as deep_gemm_decode,
        ):
            indexer.topk_decode(object())
        torch_decode.assert_called_once()
        deep_gemm_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
