"""Which launches the relayed DSpark flow accepts, and what every other launch keeps."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import speculative_hook, validation_hook
from sglang.srt.arg_groups.validation_hook import check_pipeline_parallel_compat
from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components import dspark_pp_launch as launch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _cfg(**overrides):
    fields = dict(
        attn_dp_size=1,
        attn_cp_size=1,
        disable_overlap_schedule=True,
        speculative_adaptive=False,
        enable_mixed_chunk=False,
        enable_watermark=False,
        enable_lora=None,
        lora_paths=None,
        pp_size=3,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _hf(**text):
    fields = dict(dspark_target_layer_ids=[37, 38, 39], num_hidden_layers=40)
    fields.update(text)
    return SimpleNamespace(
        model_type="deepseek_v41", text_config=SimpleNamespace(**fields)
    )


class TestRefusals(CustomTestCase):
    def setUp(self):
        for field, value in ((envs.SGLANG_ENABLE_PP_SPEC, True),):
            scope = field.override(value)
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)

    def _refusal(self, cfg=None, hf=None, partition="14,13,13"):
        with patch.dict("os.environ", {"SGLANG_PP_LAYER_PARTITION": partition}):
            return launch.pp_dspark_refusal(cfg or _cfg(), hf or _hf())

    def test_the_supported_launch_passes(self):
        self.assertIsNone(self._refusal())

    def test_each_refusal_names_itself(self):
        for overrides, expect in (
            (dict(attn_dp_size=3), "attention DP"),
            (dict(attn_cp_size=2), "attention DP or CP"),
            (dict(disable_overlap_schedule=False), "overlap"),
            (dict(speculative_adaptive=True), "adaptive"),
            (dict(enable_mixed_chunk=True), "mixed-chunk"),
            (dict(enable_watermark=True), "watermark"),
            (dict(enable_lora=True), "LoRA"),
            (dict(lora_paths=["a=/x"]), "LoRA"),
        ):
            with self.subTest(expect=expect):
                self.assertIn(expect, self._refusal(_cfg(**overrides)))

    def test_only_deepseek_v41(self):
        glm = SimpleNamespace(model_type="glm5", text_config=SimpleNamespace())
        self.assertIn("DeepSeek-V4.1 only", self._refusal(hf=glm))

    def test_ragged_verify_is_refused(self):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            self.assertIn("static", self._refusal())

    def test_target_layers_must_sit_on_the_last_stage(self):
        self.assertIsNone(self._refusal(partition="13,13,14"))
        self.assertIsNone(self._refusal(partition="13,14,13"))
        off = self._refusal(partition="14,14,12")
        self.assertIsNone(off)
        off = self._refusal(partition="20,18,2")
        self.assertIn("[37]", off)
        self.assertIn("not on the last stage", off)

    def test_the_graph_owned_embedding_is_required(self):
        with envs.SGLANG_DSPARK_EMBED_IN_GRAPH.override(False):
            self.assertIn("EMBED_IN_GRAPH", self._refusal())

    def test_flag_values(self):
        with envs.SGLANG_PP_DSPARK_EMBED.override("nvme"):
            self.assertIn("host or gpu", self._refusal())
        with envs.SGLANG_PP_DSPARK_SWITCH.override("maybe"):
            self.assertIn("auto, on or off", self._refusal())

    def test_validate_raises_with_the_name(self):
        server_args = object()
        with (
            patch(
                "sglang.srt.arg_groups.model_override_base.resolving_view",
                lambda s: _cfg(attn_dp_size=2),
            ),
            patch(
                "sglang.srt.arg_groups.model_override_base.model_config_of",
                lambda s: SimpleNamespace(hf_config=_hf()),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_ENABLE_PP_SPEC.*attention DP"
            ):
                launch.validate_pp_dspark_launch(server_args)


class TestSingleStageFolds(CustomTestCase):
    """Invariant 32."""

    def test_the_list_is_the_designs(self):
        self.assertEqual(
            set(launch._SINGLE_STAGE_FOLDS),
            {
                "SGLANG_OPT_DSV41_TARGET_METADATA_EARLY",
                "SGLANG_OPT_DSV41_DSPARK_POST_FOLD",
                "SGLANG_DSPARK_DEFER_CONTEXT_KV",
                "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE",
                "SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH",
            },
        )

    def test_they_resolve_off_under_the_relay(self):
        fields = [getattr(envs, n) for n in launch._SINGLE_STAGE_FOLDS]
        scopes = [f.override(True) for f in fields]
        for s in scopes:
            s.__enter__()
        self.addCleanup(lambda: [s.__exit__(None, None, None) for s in scopes])
        changed = launch.resolve_single_stage_folds_off()
        self.assertEqual(changed, list(launch._SINGLE_STAGE_FOLDS))
        self.assertFalse(any(f.get() for f in fields))
        self.assertEqual(launch.resolve_single_stage_folds_off(), [])

    def test_they_are_off_by_default_and_nothing_changes(self):
        self.assertEqual(launch.resolve_single_stage_folds_off(), [])


class TestEveryOtherLaunchKeepsItsRefusal(CustomTestCase):
    """Invariant 31, on the hooks: without the flag or outside the plain launch the
    refusals are the upstream ones, by the same messages."""

    @staticmethod
    def _pp(**overrides):
        fields = dict(
            disable_overlap_schedule=True,
            speculative_algorithm="DSPARK",
            enable_multi_layer_eagle=False,
            disaggregation_mode="null",
            min_free_slots_delay=None,
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_flag_off_refuses_outside_pd_prefill(self):
        for mode in ("null", "decode"):
            with self.assertRaisesRegex(AssertionError, "DSPARK.*prefill"):
                check_pipeline_parallel_compat(self._pp(disaggregation_mode=mode))
        check_pipeline_parallel_compat(self._pp(disaggregation_mode="prefill"))

    def test_flag_on_refuses_pd(self):
        with patch.object(
            validation_hook.envs.SGLANG_ENABLE_PP_SPEC, "get", return_value=True
        ):
            check_pipeline_parallel_compat(self._pp())
            with self.assertRaisesRegex(AssertionError, "prefill"):
                check_pipeline_parallel_compat(self._pp(disaggregation_mode="decode"))
            with self.assertRaisesRegex(AssertionError, "SGLANG_ENABLE_PP_SPEC"):
                check_pipeline_parallel_compat(self._pp(disaggregation_mode="prefill"))

    def test_relay_enabled_is_only_this_launch(self):
        self.assertFalse(
            launch.pp_dspark_relay_enabled(pp_size=3, spec_algorithm="DSPARK")
        )
        with envs.SGLANG_ENABLE_PP_SPEC.override(True):
            self.assertTrue(
                launch.pp_dspark_relay_enabled(pp_size=3, spec_algorithm="dspark")
            )
            self.assertFalse(
                launch.pp_dspark_relay_enabled(pp_size=1, spec_algorithm="DSPARK")
            )
            self.assertFalse(
                launch.pp_dspark_relay_enabled(pp_size=3, spec_algorithm="EAGLE")
            )
            self.assertFalse(
                launch.pp_dspark_relay_enabled(pp_size=3, spec_algorithm=None)
            )

    def _handle(self, *, flag, mode):
        """_handle_dspark up to its PP block; past it a sentinel stops the stub's walk."""
        cfg = SimpleNamespace(
            device="cuda",
            attn_dp_size=1,
            pp_size=3,
            disaggregation_mode=mode,
            speculative_draft_model_path=None,
            # The draft lm_head's quantised dtype (upstream #49595): None, the default.
            speculative_dspark_lm_head_dtype=None,
        )
        args = object()
        rules = []

        class _PastTheBlock(Exception):
            pass

        def past(_):
            raise _PastTheBlock

        with (
            patch.object(speculative_hook, "resolving_view", return_value=cfg),
            patch.object(
                speculative_hook, "_target_checkpoint_bundles_dspark_draft", past
            ),
            patch.object(
                launch, "validate_pp_dspark_launch", lambda a: rules.append(a)
            ),
            patch.object(
                speculative_hook.envs.SGLANG_ENABLE_PP_SPEC, "get", return_value=flag
            ),
        ):
            try:
                speculative_hook._handle_dspark(args)
            except _PastTheBlock:
                return rules == [args], True
        return rules == [args], False

    def test_the_relayed_launch_goes_through_the_rules(self):
        self.assertEqual(self._handle(flag=True, mode="null"), (True, True))

    def test_every_other_launch_keeps_the_refusal_and_never_sees_the_rules(self):
        for flag, mode in ((False, "null"), (False, "decode"), (True, "decode")):
            with self.assertRaisesRegex(ValueError, "requires PD prefill"):
                self._handle(flag=flag, mode=mode)
        self.assertEqual(self._handle(flag=False, mode="prefill"), (False, True))


if __name__ == "__main__":
    unittest.main()
