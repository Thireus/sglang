"""One FlashInfer tactic cache file per environment.

FlashInfer ignores, and never overwrites, a cache file stamped by another
environment. Two builds with the same FlashInfer version used to share one
path, so the second one profiled again at every start; the path now carries a
digest of the stamp.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.test.test_utils import CustomTestCase

STAMP = {
    "flashinfer_version": "0.7.3",
    "cuda_version": "13.0",
    "cutlass_dsl_version": "torch==2.14.1:aaaa",
    "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
}


def _flashinfer(stamp):
    def collect():
        if isinstance(stamp, Exception):
            raise stamp
        return dict(stamp)

    tuner = SimpleNamespace(_collect_metadata=collect)
    return {
        "flashinfer": SimpleNamespace(__version__="0.7.3", autotuner=tuner),
        "flashinfer.autotuner": tuner,
    }


class TestFlashinferEnvironmentKey(CustomTestCase):
    def _key(self, stamp):
        with patch.dict(sys.modules, _flashinfer(stamp)):
            return autotune.flashinfer_environment_key()

    def test_same_stamp_same_key(self):
        self.assertEqual(self._key(STAMP), self._key(dict(reversed(STAMP.items()))))
        self.assertEqual(len(self._key(STAMP)), 12)

    def test_another_build_of_the_same_version_gets_its_own_key(self):
        other = dict(STAMP, cutlass_dsl_version="torch==2.14.1:bbbb")
        self.assertNotEqual(self._key(STAMP), self._key(other))

    def test_no_stamp_is_one_fixed_key(self):
        self.assertEqual(self._key(RuntimeError("no stamp")), "nostamp")

    def _path(self, stamp, cache_dir):
        runner = SimpleNamespace(
            device="cuda",
            dtype=torch.bfloat16,
            is_draft_worker=False,
            gpu_id=0,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), quantization=None
            ),
        )
        parallel = SimpleNamespace(
            tp_size=1,
            pp_size=3,
            attn_dp_size=1,
            moe_ep_size=1,
            tp_rank=0,
            pp_rank=1,
            dp_rank=None,
        )
        with (
            patch.dict(sys.modules, _flashinfer(stamp)),
            patch.dict(os.environ, SGLANG_CACHE_DIR=cache_dir),
            patch.multiple(
                autotune,
                get_model=lambda: SimpleNamespace(model_path="/m/x", quantization=None),
                get_exec=lambda: SimpleNamespace(
                    moe=SimpleNamespace(moe_runner_backend="flashinfer_mxfp4")
                ),
                get_parallel=lambda: parallel,
                get_flashinfer_autotune_skip_ops=lambda _: set(),
            ),
            patch.object(
                autotune.torch.cuda, "get_device_capability", return_value=(12, 0)
            ),
        ):
            return autotune.flashinfer_autotune_cache_path(runner)

    def test_cache_path_is_per_environment(self):
        other = dict(STAMP, cutlass_dsl_version="torch==2.14.1:bbbb")
        with tempfile.TemporaryDirectory() as directory:
            first = self._path(STAMP, directory)
            again = self._path(STAMP, directory)
            second = self._path(other, directory)
        self.assertEqual(first, again)
        self.assertNotEqual(first.parent, second.parent)
        # Same version, architecture and rank file: only the key directory moves.
        self.assertEqual(first.parent.parent, second.parent.parent)
        self.assertEqual(first.name, "rank_tp0_pp1_dp0.json")
        self.assertEqual(second.name, first.name)


if __name__ == "__main__":
    unittest.main()
