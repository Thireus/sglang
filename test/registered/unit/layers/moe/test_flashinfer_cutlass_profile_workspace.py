"""The MoE profiler's workspace goes back to the device after a tuning pass.

FlashInfer's cached CUTLASS MoE runner keeps the workspace of the last bucket
its profiler prepared (0.49 GiB at 320 experts and 2,048 tokens). A pass that
profiled the MoE re-prepares the profiler at one token with a handful of
experts, on a stream of its own; a pass served from the cache has no workspace
and runs nothing.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as cutlass
from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase

GEMM1 = "trtllm::fused_moe::gemm1"
GEMM2 = "trtllm::fused_moe::gemm2"


def _fake_flashinfer(known_runner=True, fail=False, streams=None):
    class FakeRunner:
        runner_dict = {}
        prepared = []

        def __init__(
            self,
            x_dtype,
            weight_dtype,
            output_dtype,
            top_k,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            use_w4_group_scaling,
            use_mxfp8_act_scaling,
            enable_pdl,
            activation_type,
            use_fused_finalize,
        ):
            key = (x_dtype, weight_dtype, use_mxfp8_act_scaling, use_fused_finalize)
            type(self).runner_dict.setdefault(key, object())
            self.top_k = top_k

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            if fail:
                raise RuntimeError("no profiler")
            x, fc1, _, fc2, _ = inputs
            type(self).prepared.append(
                (tuple(x.shape), fc1.shape[0], fc2.shape[0], tactic, do_preparation)
                + tuple(sorted(kwargs.items()))
                + (tuple(streams) if streams is not None else ())
            )

    if known_runner:
        FakeRunner.runner_dict[(torch.float8_e4m3fn, torch.int64, True, False)] = (
            object()
        )
    core = SimpleNamespace(
        get_cutlass_fused_moe_module=lambda arch: SimpleNamespace(MoERunner=FakeRunner)
    )
    modules = {
        "flashinfer": SimpleNamespace(),
        "flashinfer.fused_moe": SimpleNamespace(core=core),
        "flashinfer.fused_moe.core": core,
        "flashinfer.utils": SimpleNamespace(device_support_pdl=lambda device: True),
    }
    return FakeRunner, modules


FLAGS = dict(
    tp_size=1,
    tp_rank=0,
    ep_size=1,
    ep_rank=0,
    use_w4_group_scaling=False,
    use_mxfp8_act_scaling=True,
    activation_type=3,
    use_fused_finalize=False,
)


class TestReleaseProfileWorkspace(CustomTestCase):
    def setUp(self):
        cutlass._profiled_cutlass_calls.clear()
        cutlass._profile_workspace_streams.clear()
        self.addCleanup(cutlass._profile_workspace_streams.clear)
        # The streams entered at the time of a call, innermost last.
        self.entered = []
        self.made = []

    def _note(self, experts=20, top_k=6):
        x = torch.zeros((54, 64), dtype=torch.float8_e4m3fn)
        fc1 = torch.zeros((experts, 8, 4), dtype=torch.int64)
        fc2 = torch.zeros((experts, 4, 2), dtype=torch.int64)
        cutlass._note_profiled_cutlass_call(x, fc1, fc2, top_k, FLAGS)

    def _release(self, modules, profiled=True):
        def make_stream(device=None):
            self.made.append(f"side stream {len(self.made)} of {device}")
            return self.made[-1]

        @contextmanager
        def enter(stream):
            self.entered.append(stream)
            try:
                yield
            finally:
                self.entered.pop()

        with (
            patch.dict(sys.modules, modules),
            patch.object(
                cutlass.torch.cuda, "get_device_capability", return_value=(12, 0)
            ),
            patch.object(cutlass.torch.cuda, "Stream", side_effect=make_stream),
            patch.object(cutlass.torch.cuda, "stream", side_effect=enter),
            patch.object(cutlass.torch.cuda, "synchronize") as sync,
            patch.object(cutlass.torch.cuda, "empty_cache") as empty,
        ):
            handled = cutlass.release_flashinfer_cutlass_profile_workspace(profiled)
        return handled, sync.call_count, empty.call_count

    def test_one_small_preparation_on_a_stream_of_its_own(self):
        runner, modules = _fake_flashinfer(streams=self.entered)
        self._note(experts=20)
        self._note(experts=20)  # the same runner again (the next layer): one entry
        handled, syncs, empties = self._release(modules)
        self.assertEqual(handled, 1)
        self.assertEqual(len(self.made), 1)
        self.assertEqual(
            runner.prepared,
            [((1, 64), 8, 8, -1, True, ("gemm_idx", 2), self.made[0])],
        )
        # Synchronized before and after; the cache is emptied once the old
        # workspace is free, and the side stream is left again.
        self.assertEqual((syncs, empties), (2, 1))
        self.assertEqual(self.entered, [])
        # FlashInfer's runner cache is as it was: same runner, same tactics.
        self.assertEqual(len(runner.runner_dict), 1)
        # The notes are spent.
        self.assertEqual(self._release(modules)[0], 0)
        self.assertEqual(len(runner.prepared), 1)

    def test_a_device_keeps_one_side_stream(self):
        runner, modules = _fake_flashinfer(streams=self.entered)
        for _ in range(2):  # the target's pass, then the draft's
            self._note()
            self.assertEqual(self._release(modules)[0], 1)
        self.assertEqual(len(self.made), 1)
        self.assertEqual([p[-1] for p in runner.prepared], [self.made[0]] * 2)

    def test_a_cache_hit_pass_runs_nothing(self):
        runner, modules = _fake_flashinfer()
        self._note()
        self.assertEqual(self._release(modules, profiled=False), (0, 0, 0))
        self.assertEqual(runner.prepared, [])
        self.assertEqual(self.made, [])
        self.assertEqual(cutlass._profiled_cutlass_calls, {})

    def test_a_pass_without_a_cutlass_moe_call_runs_nothing(self):
        runner, modules = _fake_flashinfer()
        self.assertEqual(self._release(modules), (0, 0, 0))
        self.assertEqual(runner.prepared, [])

    def test_fewer_experts_than_the_kept_count(self):
        runner, modules = _fake_flashinfer()
        self._note(experts=5, top_k=3)
        self._release(modules)
        self.assertEqual([p[1:3] for p in runner.prepared], [(5, 5)])

    def test_top_k_above_the_kept_count_keeps_top_k_experts(self):
        runner, modules = _fake_flashinfer()
        self._note(experts=64, top_k=12)
        self._release(modules)
        self.assertEqual([p[1:3] for p in runner.prepared], [(12, 12)])

    def test_the_last_noted_layer_serves_the_runner(self):
        # Layers with other expert counts share the runner: one preparation.
        runner, modules = _fake_flashinfer()
        for experts in (222, 296, 222):
            self._note(experts=experts)
        self.assertEqual(self._release(modules)[0], 1)
        self.assertEqual(len(runner.prepared), 1)

    def test_an_unknown_runner_is_not_created(self):
        runner, modules = _fake_flashinfer(known_runner=False)
        self._note()
        self.assertEqual(self._release(modules)[0], 0)
        self.assertEqual(runner.prepared, [])
        self.assertEqual(runner.runner_dict, {})

    def test_a_failing_profiler_is_reported_not_raised(self):
        runner, modules = _fake_flashinfer(fail=True)
        self._note()
        with self.assertLogs(cutlass.logger, level="WARNING"):
            self.assertEqual(self._release(modules)[0], 0)
        self.assertEqual(self.entered, [])

    def test_flashinfer_without_the_entry_point(self):
        self._note()
        modules = {
            "flashinfer": SimpleNamespace(),
            "flashinfer.fused_moe": SimpleNamespace(core=SimpleNamespace()),
            "flashinfer.fused_moe.core": SimpleNamespace(),
        }
        self.assertEqual(self._release(modules)[0], 0)


class TestMoeProfileCount(CustomTestCase):
    def test_counts_profiled_moe_buckets_only(self):
        tuner = SimpleNamespace(stats=SimpleNamespace(tuned_op_successful_configs={}))
        before = autotune._moe_profile_count(tuner)
        # a pass served from the cache changes nothing
        self.assertEqual(autotune._moe_profile_count(tuner), before)
        # another operation profiled: no MoE workspace was prepared
        tuner.stats.tuned_op_successful_configs["flashinfer::mm_mxfp8"] = 12
        self.assertEqual(autotune._moe_profile_count(tuner), before)
        tuner.stats.tuned_op_successful_configs[GEMM1] = 16
        tuner.stats.tuned_op_successful_configs[GEMM2] = 16
        self.assertEqual(autotune._moe_profile_count(tuner), before + 32)

    def test_a_tuner_without_statistics(self):
        self.assertEqual(autotune._moe_profile_count(SimpleNamespace()), 0)


class TestTuningContextReleasesTheWorkspace(CustomTestCase):
    """The tuning context asks for the release when its pass profiled the MoE."""

    def _run_pass(self, profile, loaded=True):
        stats = SimpleNamespace(tuned_op_successful_configs={GEMM1: 16, GEMM2: 16})
        tuner = SimpleNamespace(stats=stats, load_configs=Mock(), save_configs=Mock())
        modules = {
            "flashinfer": SimpleNamespace(autotune_v2=lambda **_: nullcontext()),
            "flashinfer.autotuner": SimpleNamespace(
                AutoTuner=SimpleNamespace(get=lambda: tuner),
                _collect_metadata=lambda: {"flashinfer_version": "0.7.3"},
                autotune=lambda *_, **__: nullcontext(),
                get_autotune_process_group=lambda: None,
                set_autotune_process_group=lambda _: None,
            ),
            "sglang.srt.layers.logits_processor": SimpleNamespace(
                autotune_dummy_run_mode=lambda **_: nullcontext()
            ),
        }
        runner = SimpleNamespace(
            device="cuda", forward_stream=Mock(), is_draft_worker=False
        )
        release = Mock(return_value=1)
        with (
            tempfile.TemporaryDirectory() as directory,
            get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, SGLANG_FLASHINFER_AUTOTUNE_CACHE="1"),
            patch.multiple(
                autotune,
                flashinfer_autotune_cache_path=lambda _: Path(directory) / "rank.json",
                get_flashinfer_autotune_skip_ops=lambda _: set(),
                get_exec=lambda: SimpleNamespace(
                    moe=SimpleNamespace(moe_runner_backend="flashinfer_mxfp4")
                ),
            ),
            patch.object(autotune.torch.cuda, "current_stream", return_value=Mock()),
            patch.object(
                autotune.torch,
                "get_device_module",
                return_value=SimpleNamespace(stream=lambda _: nullcontext()),
            ),
            patch.object(
                cutlass, "release_flashinfer_cutlass_profile_workspace", release
            ),
        ):
            if not loaded:
                # as in a process whose model never imported the CUTLASS MoE
                del sys.modules[cutlass.__name__]
            with autotune.flashinfer_autotune_context(runner, run_lm_head=False):
                release.assert_not_called()
                for op, buckets in profile.items():
                    stats.tuned_op_successful_configs[op] = (
                        stats.tuned_op_successful_configs.get(op, 0) + buckets
                    )
            # The tactics are saved before the workspace goes.
            tuner.save_configs.assert_called_once()
        self.assertIs(sys.modules[cutlass.__name__], cutlass)
        if not loaded:
            release.assert_not_called()
            return None
        release.assert_called_once()
        return release.call_args.kwargs["profiled"]

    def test_a_pass_that_profiled_the_moe(self):
        self.assertTrue(self._run_pass({GEMM1: 16, GEMM2: 16}))

    def test_a_pass_served_from_the_cache(self):
        self.assertFalse(self._run_pass({}))

    def test_a_pass_that_profiled_another_operation_only(self):
        self.assertFalse(self._run_pass({"flashinfer::mm_mxfp8": 12}))

    def test_without_the_cutlass_moe_module_nothing_is_imported_or_run(self):
        self.assertIsNone(self._run_pass({GEMM1: 16, GEMM2: 16}, loaded=False))


if __name__ == "__main__":
    unittest.main()
