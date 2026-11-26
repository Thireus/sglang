import itertools
import unittest

from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    read_pp_min_free_slots,
    resolve_min_free_slots,
    resolve_min_free_slots_rule,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolveMinFreeSlots(unittest.TestCase):
    def test_unset_non_dflash_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 512, is_dflash_family=False))

    def test_unset_dflash_auto_enables(self):
        self.assertEqual(resolve_min_free_slots(None, 512, is_dflash_family=True), 4)
        self.assertEqual(resolve_min_free_slots(None, 8, is_dflash_family=True), 2)

    def test_unset_dflash_small_cluster_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 7, is_dflash_family=True))
        self.assertIsNone(resolve_min_free_slots(None, 0, is_dflash_family=True))

    def test_le_one_disables(self):
        # <= 1 can never batch, so it is a no-op.
        self.assertIsNone(resolve_min_free_slots(1, 512))
        self.assertIsNone(resolve_min_free_slots(0, 512))

    def test_explicit_value_survives_small_cluster(self):
        # The < 8 guard belongs to the DFlash auto-default, not explicit values.
        self.assertEqual(resolve_min_free_slots(4, 7), 4)
        self.assertEqual(resolve_min_free_slots(4, 7, is_dflash_family=True), 4)

    def test_non_dflash_uses_explicit_value(self):
        self.assertEqual(resolve_min_free_slots(2, 8), 2)
        self.assertEqual(resolve_min_free_slots(3, 512), 3)
        self.assertEqual(resolve_min_free_slots(8, 512), 8)
        self.assertEqual(resolve_min_free_slots(16, 512), 16)

    def test_explicit_value_is_capped_to_max_running_requests(self):
        self.assertEqual(resolve_min_free_slots(16, 8), 8)

    def test_user_value_overrides_dflash_default(self):
        self.assertEqual(resolve_min_free_slots(3, 512, is_dflash_family=True), 3)
        self.assertEqual(resolve_min_free_slots(16, 512, is_dflash_family=True), 16)

    def test_explicit_one_disables_dflash_default(self):
        self.assertIsNone(resolve_min_free_slots(1, 512, is_dflash_family=True))


class TestResolveMinFreeSlotsRule(unittest.TestCase):
    """SGLANG_PP_DSPARK_MIN_FREE_SLOTS under pipeline parallelism (pp_value)."""

    PP = dict(pp_size=3, pp_micro_batch_cap=9)

    def test_unset_is_the_stock_rule(self):
        for user_value, mrr, dflash, pp in itertools.product(
            (None, 0, 1, 2, 3, 16), (0, 7, 8, 27, 512), (False, True), (1, 3)
        ):
            threshold, why = resolve_min_free_slots_rule(
                user_value,
                mrr,
                is_dflash_family=dflash,
                pp_size=pp,
                pp_micro_batch_cap=max(mrr // pp, 1),
            )
            self.assertEqual(
                threshold,
                resolve_min_free_slots(user_value, mrr, is_dflash_family=dflash),
            )
            self.assertNotIn("SGLANG_PP_DSPARK_MIN_FREE_SLOTS", why)

    def test_unset_dspark_pipeline_holds_for_four(self):
        threshold, why = resolve_min_free_slots_rule(
            None, 27, is_dflash_family=True, **self.PP
        )
        self.assertEqual(threshold, 4)
        self.assertIn("DFlash-family default", why)

    def test_one_or_less_turns_the_hold_off(self):
        for value in (1, 0, -1):
            threshold, why = resolve_min_free_slots_rule(
                None, 27, is_dflash_family=True, pp_value=value, **self.PP
            )
            self.assertIsNone(threshold)
            self.assertIn(f"SGLANG_PP_DSPARK_MIN_FREE_SLOTS={value}", why)
            self.assertIn("stock: 4", why)

    def test_value_counts_against_the_micro_batch(self):
        for value, expected in ((2, 2), (3, 3), (9, 9), (12, 9)):
            threshold, _ = resolve_min_free_slots_rule(
                None, 27, is_dflash_family=True, pp_value=value, **self.PP
            )
            self.assertEqual(threshold, expected)

    def test_ignored_without_the_pipeline_or_dflash_family(self):
        for kwargs in (
            dict(is_dflash_family=True, pp_size=1, pp_micro_batch_cap=27),
            dict(is_dflash_family=False, **self.PP),
        ):
            threshold, why = resolve_min_free_slots_rule(None, 27, pp_value=1, **kwargs)
            self.assertEqual(
                threshold,
                resolve_min_free_slots(None, 27, kwargs["is_dflash_family"]),
            )
            self.assertIn("ignored", why)


class TestReadPpMinFreeSlots(unittest.TestCase):
    """The switch as the launch environment gives it."""

    NAME = "SGLANG_PP_DSPARK_MIN_FREE_SLOTS"

    def test_unset_is_none(self):
        self.assertIsNone(read_pp_min_free_slots({}))
        self.assertIsNone(read_pp_min_free_slots({"OTHER": "off"}))

    def test_whole_numbers_are_read(self):
        for raw, value in (
            ("-1", -1),
            ("0", 0),
            ("1", 1),
            ("2", 2),
            ("9", 9),
            ("10", 10),
            ("+2", 2),
            (" 2 ", 2),
        ):
            self.assertEqual(read_pp_min_free_slots({self.NAME: raw}), value, raw)

    def test_anything_else_stops_the_launch(self):
        for raw in ("off", "", " ", "2.5", "1e3", "two", "0x2", "1_0", "2 3", "None"):
            with self.assertRaises(ValueError, msg=repr(raw)) as ctx:
                read_pp_min_free_slots({self.NAME: raw})
            text = str(ctx.exception)
            self.assertIn(f"{self.NAME}={raw!r}", text)
            self.assertIn("not a whole number", text)
            self.assertIn("1 or less", text)
            self.assertIn("N >= 2", text)

    def test_boundaries_resolve_as_documented(self):
        pp = dict(is_dflash_family=True, pp_size=3, pp_micro_batch_cap=9)
        for raw, expected in (
            ("-1", None),
            ("0", None),
            ("1", None),
            ("2", 2),
            ("9", 9),
            ("10", 9),
        ):
            value = read_pp_min_free_slots({self.NAME: raw})
            threshold, why = resolve_min_free_slots_rule(None, 27, pp_value=value, **pp)
            self.assertEqual(threshold, expected, raw)
            self.assertEqual("capped" in why, value > 9, why)


class TestMinFreeSlotsDelayer(unittest.TestCase):
    def test_delays_below_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertTrue(delayer.should_delay(running_bs=100, num_allocatable_reqs=2))

    def test_no_delay_at_or_above_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertFalse(delayer.should_delay(running_bs=100, num_allocatable_reqs=4))
        self.assertFalse(delayer.should_delay(running_bs=100, num_allocatable_reqs=8))

    def test_no_delay_when_idle(self):
        # Nothing running: no decode batch to protect, prefill at once.
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertFalse(delayer.should_delay(running_bs=0, num_allocatable_reqs=0))


if __name__ == "__main__":
    unittest.main()
