"""DeepSeek-V4.1 accepts pipeline parallelism, without the features that do
not cross stages, and resolves decoder SWA bounded replay off under it."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook as hook
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _cfg(**overrides):
    cfg = dict(
        dsv4_attn_backend="flashmla",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED)
        ),
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=None,
        speculative_algorithm=None,
        enable_hisparse=False,
        enable_two_batch_overlap=False,
        pp_size=3,
        tp_size=1,
        attn_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        enable_prefill_cp=False,
        disaggregation_mode="null",
        enable_return_hidden_states=False,
        moe_a2a_backend="none",
        ep_join_mode=None,
        language_model_only=True,
        mm_feature_transport=None,
        enable_unified_memory=False,
        enable_mixed_chunk=False,
        enable_lora=False,
        enable_session_radix_cache=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestDeepseekV41PipelineGate(CustomTestCase):
    def _validate(self, cfg, *, vision_n_layers=32):
        declared = {}
        hf_config = SimpleNamespace(
            model_type="deepseek_v41", vision_n_layers=vision_n_layers
        )
        with (
            patch.object(hook, "resolving_view", lambda server_args: server_args),
            patch.object(
                hook,
                "model_config_of",
                lambda server_args: SimpleNamespace(hf_config=hf_config),
            ),
            patch.object(
                hook,
                "declare_resolution",
                lambda server_args, source, **kv: declared.update(kv),
            ),
            override_platform(is_cuda=True, is_hip=False),
        ):
            hook.validate_deepseek_v41_features(cfg)
        return declared

    def test_pp_text_only_is_accepted_and_turns_the_replay_tail_off(self):
        self.assertEqual(
            self._validate(_cfg()), {"enable_decoder_swa_bounded_replay": False}
        )

    def test_single_stage_keeps_the_replay_tail(self):
        self.assertEqual(
            self._validate(_cfg(pp_size=1)),
            {"enable_decoder_swa_bounded_replay": True},
        )

    def test_pp_serves_the_vision_tower(self):
        # The first stage holds the tower; every stage routes image positions.
        self.assertEqual(
            self._validate(_cfg(language_model_only=False)),
            {"enable_decoder_swa_bounded_replay": False},
        )
        self._validate(_cfg(language_model_only=False, mm_feature_transport="cpu"))
        self._validate(_cfg(language_model_only=False), vision_n_layers=0)

    def test_pp_vision_refuses_a_gpu_feature_transport(self):
        for transport in ("cuda_ipc", "cuda_vmm"):
            with self.subTest(transport=transport):
                with self.assertRaisesRegex(ValueError, "mm-feature-transport cpu"):
                    self._validate(
                        _cfg(language_model_only=False, mm_feature_transport=transport)
                    )
                # A text-only launch maps no feature and keeps its choice.
                self._validate(_cfg(mm_feature_transport=transport))
        # The legacy switch, when no transport is named (restored even on failure).
        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "1"}):
            with self.assertRaisesRegex(ValueError, "mm-feature-transport cpu"):
                self._validate(_cfg(language_model_only=False))
            self._validate(_cfg(language_model_only=False, mm_feature_transport="cpu"))

    def test_pp_refuses_an_explicit_replay_tail(self):
        with self.assertRaisesRegex(ValueError, "decoder SWA bounded replay"):
            self._validate(_cfg(enable_decoder_swa_bounded_replay=True))
        self.assertEqual(
            self._validate(_cfg(enable_decoder_swa_bounded_replay=False)), {}
        )

    def test_pp_refuses_context_parallelism_and_encoder_replay(self):
        with self.assertRaisesRegex(ValueError, "context parallelism"):
            self._validate(_cfg(dcp_size=2))
        with self.assertRaisesRegex(ValueError, "encoder SWA bounded replay"):
            self._validate(_cfg(enable_encoder_swa_bounded_replay=True))

    def test_other_unsupported_features_still_fail_first(self):
        with self.assertRaisesRegex(ValueError, "two-batch overlap"):
            self._validate(_cfg(enable_two_batch_overlap=True))


if __name__ == "__main__":
    unittest.main()
