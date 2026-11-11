"""The eager input buffers are sized by the prefill ceiling, not by the KV pool.

input_ids, positions, out_cache_loc and mrope_positions hold one int64 row per
token (48 bytes). Sized by the pool they took 0.21 GiB on every rank at 4.6M
pool tokens; a chunked prefill batch never holds more than the scheduler's
ceiling plus the decode rows.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.runner import eager_runner
from sglang.srt.model_executor.runner.eager_runner import (
    EagerRunner,
    eager_token_capacity,
)
from sglang.test.test_utils import CustomTestCase


@dataclass
class _Batch:
    batch_size: int
    input_ids: Optional[torch.Tensor] = None
    input_embeds: Optional[torch.Tensor] = None


class TestEagerTokenCapacity(CustomTestCase):
    def test_chunked_prefill_uses_the_ceiling_and_the_decode_rows(self):
        # production: chunk 2,048, 27 requests x 6 verify tokens, 1M context
        for pool in (4_626_176, 8_000_000):
            self.assertEqual(
                eager_token_capacity(
                    max_total_num_tokens=pool,
                    prefill_buffer_tokens=2048,
                    decode_tokens=27 * 6,
                ),
                2048 + 162,
            )

    def test_without_chunked_prefill_the_pool_size_stays(self):
        self.assertEqual(
            eager_token_capacity(
                max_total_num_tokens=4_626_176,
                prefill_buffer_tokens=0,
                decode_tokens=162,
            ),
            4_626_176,
        )

    def test_never_larger_than_before(self):
        # a pool smaller than the ceiling keeps the old size
        self.assertEqual(
            eager_token_capacity(
                max_total_num_tokens=1000, prefill_buffer_tokens=2048, decode_tokens=64
            ),
            2048,
        )
        # a decode batch wider than the pool was the old size too
        self.assertEqual(
            eager_token_capacity(
                max_total_num_tokens=100, prefill_buffer_tokens=0, decode_tokens=512
            ),
            512,
        )

    def test_bytes_given_back(self):
        row = 6 * 8  # input_ids + positions + out_cache_loc + 3 x mrope_positions
        for pool, gib in ((4_626_176, 0.2067), (8_000_000, 0.3575)):
            new = eager_token_capacity(
                max_total_num_tokens=pool, prefill_buffer_tokens=2048, decode_tokens=162
            )
            self.assertAlmostEqual((pool - new) * row / 2**30, gib, places=3)


class TestEagerLoadBatch(CustomTestCase):
    def _runner(self, capacity):
        registry = Mock()
        registry.extract_buffer.return_value = "from the static buffers"
        return SimpleNamespace(_eager_registry=registry, _eager_max_num_token=capacity)

    def test_a_batch_that_fits_is_copied(self):
        runner = self._runner(2210)
        batch = _Batch(batch_size=1, input_ids=torch.zeros(2048, dtype=torch.int64))
        out = EagerRunner.load_batch(runner, batch)
        self.assertEqual(out, "from the static buffers")
        runner._eager_registry.fill_from.assert_called_once()
        self.assertEqual(
            runner._eager_registry.fill_from.call_args.kwargs["raw_num_tokens"], 2048
        )

    def test_a_batch_that_does_not_fit_runs_on_its_own_tensors(self):
        runner = self._runner(2210)
        ids = torch.arange(4096, dtype=torch.int64)
        batch = _Batch(batch_size=1, input_ids=ids)
        out = EagerRunner.load_batch(runner, batch)
        runner._eager_registry.fill_from.assert_not_called()
        self.assertIsNot(out, batch)
        self.assertIs(out.input_ids, ids)

    def test_embeddings_count_as_tokens(self):
        runner = self._runner(8)
        batch = _Batch(batch_size=1, input_embeds=torch.zeros(9, 4))
        out = EagerRunner.load_batch(runner, batch)
        runner._eager_registry.fill_from.assert_not_called()
        self.assertIs(out.input_embeds, batch.input_embeds)

    def test_module_keeps_the_no_copy_switch(self):
        self.assertTrue(hasattr(eager_runner.envs, "SGLANG_EAGER_INPUT_NO_COPY"))


if __name__ == "__main__":
    unittest.main()
