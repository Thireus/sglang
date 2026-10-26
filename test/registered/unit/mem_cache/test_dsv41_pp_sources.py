"""DeepSeek-V4.1 pipeline stages: which earlier producers a stage replays, what
it receives and sends, and the pool storage and layer mapping that follow."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.deepseek_v41_pp_sources import (
    pipeline_stages,
    replayed_sources_of,
    stage_sources,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    collect_sources_by_ratio,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# DeepSeek-V4.1-Flash(-1017) config.json: 40 decoder layers, 3 MTP entries.
RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
V41_FLASH = SimpleNamespace(
    compress_ratios=RATIOS,
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    num_hidden_layers=40,
)


def _stages(*sizes):
    out, start = [], 0
    for size in sizes:
        out.append(range(start, start + size))
        start += size
    return out


class TestStageSources(CustomTestCase):
    def test_single_stage_replays_nothing(self):
        self.assertEqual(replayed_sources_of(V41_FLASH, range(0, 40)), ())

    def test_pp3_13_14_13(self):
        stages = _stages(13, 14, 13)
        plans = [stage_sources(V41_FLASH, stages, r) for r in range(3)]
        self.assertEqual([p.replayed for p in plans], [(), (8,), (20, 24)])
        self.assertEqual([p.received for p in plans], [(), (8,), (20, 24)])
        self.assertEqual([p.sent for p in plans], [(8,), (20, 24), ()])
        self.assertEqual([p.captured for p in plans], [(8,), (20, 24), ()])

    def test_pp4_with_a_three_layer_first_stage(self):
        stages = _stages(3, 12, 13, 12)
        plans = [stage_sources(V41_FLASH, stages, r) for r in range(4)]
        self.assertEqual([p.replayed for p in plans], [(), (2,), (14,), (20,)])
        self.assertEqual([p.sent for p in plans], [(2,), (14,), (20,), ()])
        self.assertEqual([p.captured for p in plans], [(2,), (14,), (20,), ()])

    def test_stage_starting_at_an_index_source_needs_kv_and_candidates_only(self):
        # Layer 28 scores its own top-k, but reads layer 20's KV and candidates.
        self.assertEqual(replayed_sources_of(V41_FLASH, range(28, 40)), (20,))

    def test_a_stage_passes_on_what_a_later_stage_needs(self):
        stages = _stages(10, 3, 7, 20)
        plan = stage_sources(V41_FLASH, stages, 1)
        self.assertEqual(plan.replayed, (8,))
        self.assertEqual(plan.received, (8,))
        self.assertEqual(plan.sent, (8,))
        self.assertEqual(plan.captured, ())

    def test_sent_matches_next_received(self):
        for sizes in [(13, 14, 13), (3, 12, 13, 12), (10, 3, 7, 20), (5,) * 8]:
            stages = _stages(*sizes)
            for rank in range(len(stages) - 1):
                self.assertEqual(
                    stage_sources(V41_FLASH, stages, rank).sent,
                    stage_sources(V41_FLASH, stages, rank + 1).received,
                    sizes,
                )

    def test_model_without_low_ratio_sources(self):
        v4 = SimpleNamespace(compress_ratios=[4, 128] * 20, kv_source_layer_ids=[])
        self.assertEqual(replayed_sources_of(v4, range(20, 40)), ())

    def test_pipeline_stages_follow_the_partition_variable(self):
        with patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": "3,12,13,12"}):
            self.assertEqual(pipeline_stages(40, 4), tuple(_stages(3, 12, 13, 12)))


class TestReplayedPoolStorage(CustomTestCase):
    def test_replayed_kv_sources_own_storage(self):
        stage = range(27, 40)
        self.assertEqual(
            collect_sources_by_ratio(RATIOS, [2, 8, 14, 20], stage), {}
        )
        self.assertEqual(
            collect_sources_by_ratio(RATIOS, [2, 8, 14, 20], stage, (20, 24)),
            {1: [20]},
        )

    def test_layer_mapping_covers_replayed_layers(self):
        pool = SimpleNamespace(
            compression_ratios=RATIOS,
            _stage_start=27,
            _stage_end=40,
            replayed_layers=(20, 24),
            kv_pools={1: "c1-pool", 2: None},
        )
        pool.sources_by_ratio = collect_sources_by_ratio(
            RATIOS, [2, 8, 14, 20], range(27, 40), pool.replayed_layers
        )
        pool.source_layer_of = (
            lambda layer_id: DeepSeekV4TokenToKVPool.source_layer_of(pool, layer_id)
        )
        pool._mapped_layers = lambda: DeepSeekV4TokenToKVPool._mapped_layers(pool)
        DeepSeekV4TokenToKVPool._init_compressed_layer_mapping(pool)
        for layer_id in (20, 24, 27, 39):
            item = pool.layer_mapping[layer_id]
            self.assertEqual((item.compress_ratio, item.compress_layer_id), (1, 0))
            self.assertEqual(item.compress_kv_pool, "c1-pool")
        self.assertIsNone(pool.layer_mapping[19])
        self.assertIsNone(pool.layer_mapping[26])


if __name__ == "__main__":
    unittest.main()
