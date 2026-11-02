"""DSpark under pipeline parallelism at the HOPE launch: split 14/14/12, the draft as
shipped (E = 128), K = 312 or 304, the draft's embedding on the card.

The port's other tests pin the 14/13/13 launch with a draft pruned to E = 64; they stay,
since they describe that launch. This module carries their four facts over to this one:
  * test_pp_dspark_pool's planners: no inflation under the relay, the draft's SWA priced
    on the last stage (kv x (12 + 3)), the SWA cap 81,920, and the pool cap 4,626,176
    kept at production's fraction 0.95 (and from 0.934 at K = 312, 0.913 at K = 304),
    lost just below;
  * test_pp_dspark_boundary's plan: stage 1 replays nothing and sends 20 only, stage 2
    replays and receives 20 only (24 is an index source the stage at 28 does not need);
  * the draft's target layers 37-39 on the last stage; no Engram layer (1, 14) there;
  * the launch rule accepts 14/14/12 with the gpu route (it reads no expert count).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.mem_cache import deepseek_v41_pp_sources as pp
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dspark_components import dspark_pp_launch as launch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GIB = 1 << 30
RATIOS = [0, 0] + [2] * 18 + [1] * 20
SPLIT = ((0, 14), (14, 28), (28, 40))
POOL_CAP = 4626176
BEFORE_LOAD = 93.79
# Free GiB after weights on the binding card (GPU 0) by K, with the full draft and
# its embedding on GPU 2: the budget of the K = 312 and 304 rows.
AFTER_WEIGHTS_MIN = {312: 11.18, 304: 13.14}
V41 = SimpleNamespace(
    compress_ratios=RATIOS + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    num_hidden_layers=40,
)


class TestPoolAt141412(CustomTestCase):
    def _sized(self, rank, use, spec="DSPARK", width=6, relay=True):
        fields = dict(
            enable_encoder_swa_bounded_replay=False,
            speculative_algorithm=spec,
            speculative_num_draft_tokens=width,
            speculative_dspark_block_size=(width - 1) if width else None,
            page_size=256,
            max_running_requests=27,
            chunked_prefill_size=2048,
            disable_overlap_schedule=True,
            pp_size=3,
            enable_hisparse=False,
            dsv4_attn_backend="flashmla",
        )
        override = get_context().override_server_args(**fields)
        override.install()
        try:
            cfg = SimpleNamespace(
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                index_head_dim=128,
                context_len=1 << 20,
                compress_ratios=RATIOS,
                window_size=128,
                hf_config=SimpleNamespace(
                    compress_ratios=RATIOS,
                    kv_source_layer_ids=[2, 8, 14, 20],
                    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
                    candidate_source_layer_id=20,
                    num_hidden_layers=40,
                ),
            )
            start, end = SPLIT[rank]
            algorithm = SimpleNamespace(
                is_dspark=lambda: spec == "DSPARK",
                is_none=lambda: spec is None,
                is_eagle=lambda: spec == "EAGLE",
            )
            kvc = SimpleNamespace(
                kv_cache_dtype_str="fp8_e4m3",
                model_config=cfg,
                layer_info=SimpleNamespace(start_layer=start, end_layer=end),
                pp_size=3,
                pp_group=SimpleNamespace(rank_in_group=rank),
                is_draft_worker=False,
                attn_dp_size=1,
                sliding_window_size=128,
                page_size=256,
                spec_algorithm=algorithm,
                spec_aux_config=SimpleNamespace(
                    dflash_draft_num_layers=40, dspark_num_stages=3
                ),
            )
            with envs.SGLANG_ENABLE_PP_SPEC.override(relay):
                return use(DSV4PoolConfigurator(kvc))
        finally:
            override.restore()

    def test_swa_cap_and_no_inflation_and_the_draft_priced_on_the_last_stage(self):
        for rank in range(3):
            self.assertEqual(self._sized(rank, lambda p: p.swa_cap_tokens), 81920)
            self.assertEqual(self._sized(rank, lambda p: p._spec_infl), 1.0)
        kv = self._sized(0, lambda p: p._get_paged_kv_bytes_per_token())
        self.assertEqual(self._sized(0, lambda p: p.bytes_per_swa_token), kv * 14)
        self.assertEqual(self._sized(1, lambda p: p.bytes_per_swa_token), kv * 14)
        self.assertEqual(self._sized(2, lambda p: p.bytes_per_swa_token), kv * (12 + 3))

    def test_the_pool_cap_holds_at_production_fraction(self):
        for k, (fits, lost) in {312: (0.934, 0.933), 304: (0.913, 0.912)}.items():
            for fraction, want in ((0.95, True), (fits, True), (lost, False)):
                available = int(
                    (AFTER_WEIGHTS_MIN[k] - BEFORE_LOAD * (1 - fraction) - 0.10) * GIB
                )
                got = [
                    self._sized(
                        r,
                        lambda p: (
                            p.calculate_pool_sizes(
                                available, 256
                            ).full_max_total_num_tokens
                        ),
                    )
                    for r in range(3)
                ]
                self.assertEqual(min(got) >= POOL_CAP, want, (k, fraction, got))


class TestPlanAt141412(CustomTestCase):
    def test_plan_of_the_split(self):
        stages = [range(a, b) for a, b in SPLIT]
        plans = [pp.stage_sources(V41, stages, r) for r in range(3)]
        self.assertEqual([p.replayed for p in plans], [(), (), (20,)])
        self.assertEqual(plans[0].sent, ())
        self.assertEqual(plans[1].sent, (20,))
        self.assertEqual(plans[2].received, (20,))

    def test_draft_target_layers_on_the_last_stage_and_no_engram_layer_there(self):
        last = range(*SPLIT[2])
        self.assertTrue(all(i in last for i in (37, 38, 39)))
        self.assertFalse([i for i in (1, 14) if i in last])


class TestLaunchAt141412(CustomTestCase):
    def test_the_hook_accepts_the_split_and_the_gpu_route(self):
        cfg = SimpleNamespace(
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
        hf = SimpleNamespace(
            model_type="deepseek_v41",
            text_config=SimpleNamespace(
                dspark_target_layer_ids=[37, 38, 39],
                num_hidden_layers=40,
                dspark_n_routed_experts=128,
            ),
        )
        with (
            mock.patch.dict("os.environ", {"SGLANG_PP_LAYER_PARTITION": "14,14,12"}),
            envs.SGLANG_PP_DSPARK_EMBED.override("gpu"),
        ):
            self.assertIsNone(launch.pp_dspark_refusal(cfg, hf))


if __name__ == "__main__":
    unittest.main()
