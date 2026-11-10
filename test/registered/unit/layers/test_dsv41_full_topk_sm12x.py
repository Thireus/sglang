import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.dsv4 import v41_indexer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _arch(major):
    return SimpleNamespace(
        is_sm=lambda m, minor=None: m == major,
        is_sm_at_least=lambda m, minor=0: major >= m,
    )


class TestFullTopKRouting(CustomTestCase):
    """The ratio-1/2 index layers' full top-k: DeepGEMM's FP4 paged logits decode on
    SM100 only (patch 017), while prefill keeps DeepGEMM's flattened-K logits on
    SM100 and SM12x alike (upstream's is_sm100_or_newer)."""

    def _make(self, major):
        v41_indexer.is_sm100_or_newer.cache_clear()
        v41_indexer._use_deep_gemm_prefill.cache_clear()
        try:
            with (
                patch.object(v41_indexer, "current_arch", return_value=_arch(major)),
                patch.object(v41_indexer, "is_sm100", return_value=major == 10),
                patch.object(v41_indexer, "has_dense_fp4_indexer", return_value=True),
            ):
                return v41_indexer.make_full_topk_indexer(
                    token_to_kv_pool=None, req_to_token=None
                )
        finally:
            v41_indexer.is_sm100_or_newer.cache_clear()
            v41_indexer._use_deep_gemm_prefill.cache_clear()

    def test_sm12x_decodes_without_deep_gemm(self):
        indexer = self._make(12)
        self.assertFalse(indexer.use_deep_gemm_decode)
        # Prefill keeps DeepGEMM's flattened-K logits, which SM12x has.
        self.assertTrue(indexer.use_deep_gemm_prefill)

    def test_sm100_keeps_deep_gemm(self):
        indexer = self._make(10)
        self.assertTrue(indexer.use_deep_gemm_decode)
        self.assertTrue(indexer.use_deep_gemm_prefill)

    def test_hopper_takes_torch_both_ways(self):
        indexer = self._make(9)
        self.assertFalse(indexer.use_deep_gemm_decode)
        self.assertFalse(indexer.use_deep_gemm_prefill)

    def test_sm12x_topk_decode_takes_torch_path(self):
        indexer = self._make(12)
        with (
            patch.object(indexer, "_torch_decode") as torch_decode,
            patch.object(indexer, "_deep_gemm_decode") as deep_gemm_decode,
        ):
            indexer.topk_decode(object())
        torch_decode.assert_called_once()
        deep_gemm_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
