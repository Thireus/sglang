"""DeepSeek-V4.1's 32-wide ue8m0 block-fp8 linears default to an MXFP8 GEMM on
SM12x (FlashInfer CUTLASS), as they do on SM100 (CuTe-DSL), instead of the
Triton block-fp8 fallback that auto (cutlass) leaves them on."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides import deepseek_v4 as overrides
from sglang.srt.environ import envs
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

V41 = SimpleNamespace(
    architectures=["DeepseekV4ForCausalLM"],
    model_type="deepseek_v41",
    quantization_config={
        "quant_method": "fp8",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    },
)


def _args(**kw):
    defaults = dict(
        device="cuda",
        swa_full_tokens_ratio=None,
        moe_a2a_backend="none",
        moe_runner_backend="auto",
        fp8_gemm_runner_backend="auto",
        _model_config=SimpleNamespace(is_fp4_experts=True, nvfp4_moe_meta=None),
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class TestSm120Mxfp8GemmDefault(CustomTestCase):
    def _resolve(self, args=None, hf=V41, *, flashinfer=True, **platform):
        facts = dict(is_hip=False, is_sm90=False, is_sm100=False, is_sm120=False)
        facts.update(platform)
        with (
            patch.object(overrides, "is_flashinfer_available", return_value=flashinfer),
            envs.SGLANG_DSV4_FP4_DEQUANT.override(False),
            override_platform(**facts),
        ):
            return overrides._deepseek_v4_overrides(args or _args(), hf)

    def test_sm120_takes_flashinfer_cutlass(self):
        out = self._resolve(is_sm120=True)
        self.assertEqual(out["fp8_gemm_runner_backend"], "flashinfer_cutlass")
        self.assertEqual(out["moe_runner_backend"], "flashinfer_mxfp4")

    def test_sm100_keeps_cutedsl(self):
        out = self._resolve(is_sm100=True)
        self.assertEqual(out["fp8_gemm_runner_backend"], "flashinfer_cutedsl")

    def test_explicit_choice_and_missing_flashinfer_are_left_alone(self):
        out = self._resolve(_args(fp8_gemm_runner_backend="triton"), is_sm120=True)
        self.assertNotIn("fp8_gemm_runner_backend", out)
        out = self._resolve(is_sm120=True, flashinfer=False)
        self.assertNotIn("fp8_gemm_runner_backend", out)

    def test_other_checkpoints_are_left_alone(self):
        v4 = SimpleNamespace(**{**vars(V41), "model_type": "deepseek_v4"})
        self.assertNotIn("fp8_gemm_runner_backend", self._resolve(hf=v4, is_sm120=True))
        block128 = SimpleNamespace(
            **{
                **vars(V41),
                "quantization_config": {
                    **V41.quantization_config,
                    "weight_block_size": [128, 128],
                },
            }
        )
        self.assertNotIn(
            "fp8_gemm_runner_backend", self._resolve(hf=block128, is_sm120=True)
        )


if __name__ == "__main__":
    unittest.main()
