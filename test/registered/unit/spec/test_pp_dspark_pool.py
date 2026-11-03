"""The KV budget of DSpark under pipeline parallelism."""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GIB = 1 << 30
RATIOS = [0, 0] + [2] * 18 + [1] * 20
SPLIT = ((0, 14), (14, 27), (27, 40))
POOL_CAP = 4626176


class TestPipelineBudget(CustomTestCase):
    def _planner(self, *, rank, spec="DSPARK", width=6, relay=True, stage=None):
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
        self.addCleanup(override.restore)
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
        start, end = stage or SPLIT[rank]
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
            return DSV4PoolConfigurator(kvc)

    def test_swa_cap_is_the_launchs(self):
        """QA fix 2: 64,512 is the plain launch; this one is 81,920."""
        self.assertEqual(self._planner(rank=2).swa_cap_tokens, 81920)
        plain = self._planner(rank=2, spec=None, width=None, relay=False)
        self.assertEqual(plain.swa_cap_tokens, 64512)

    def test_no_inflation_and_the_draft_is_paid_on_the_last_stage(self):
        """Invariant 33."""
        first = self._planner(rank=0)
        middle = self._planner(rank=1)
        last = self._planner(rank=2)
        for planner in (first, middle, last):
            self.assertEqual(planner._spec_infl, 1.0)
        kv = first._get_paged_kv_bytes_per_token()
        self.assertEqual(first.bytes_per_swa_token, kv * 14 + _c4(first))
        self.assertEqual(middle.bytes_per_swa_token, kv * 13 + _c4(middle))
        self.assertEqual(last.bytes_per_swa_token, kv * (13 + 3) + _c4(last))

    def test_flag_off_keeps_the_stock_inflation(self):
        planner = self._planner(rank=1, relay=False)
        self.assertEqual(planner._spec_infl, (13 + 1) / 13)

    def test_the_pool_cap_fits_at_fraction_955(self):
        """The pool rule's one budget for all stages, at the chosen row (GPU 0 binds, at
        14 layers): the real planners keep the cap at 0.955 and lose it just below."""
        # budget.py, MEASURED inputs: free after weights on the tightest card, GiB.
        after_weights_min = 9.216
        before_load = 93.79
        for fraction, fits in ((0.955, True), (0.945, False)):
            available = int(
                (after_weights_min - before_load * (1 - fraction) - 0.10) * GIB
            )
            got = []
            for rank in range(3):
                sizes = self._planner(rank=rank).calculate_pool_sizes(available, 256)
                got.append(sizes.full_max_total_num_tokens)
            self.assertEqual(min(got) >= POOL_CAP, fits, (fraction, got))


def _c4(planner) -> float:
    # V4.1 has no ratio-4 layer, so this is zero; kept in the sum to say so.
    return planner._get_bytes_per_swa_token() - (
        planner._get_paged_kv_bytes_per_token() * planner.num_layers_total
    )


if __name__ == "__main__":
    unittest.main()
