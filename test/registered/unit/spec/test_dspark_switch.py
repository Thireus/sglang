"""The adaptive switch of DSpark under pipeline parallelism."""

import json
import unittest

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_switch import (
    DEFAULT_BREAKEVEN_EXCHANGE_FIRST,
    DEFAULT_BREAKEVEN_LAUNCH_FIRST,
    DRAFT_MARGIN,
    NEW_REQUEST_TOKENS_PER_VERIFY,
    PROBE_EVERY_ROUNDS,
    RESUME_MARGIN,
    DsparkPPSwitch,
    interpolate_breakeven,
    parse_breakeven,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TABLE = sorted(DEFAULT_BREAKEVEN_EXCHANGE_FIRST.items())


def _switch(**kw):
    return DsparkPPSwitch(mode=kw.pop("mode", "auto"), breakeven=TABLE, **kw)


def _ask(switch, *, rids=("a",), mb=0, c=1, **kw):
    return switch.should_draft(mb_id=mb, rids=list(rids), decoding_requests=c, **kw)


def _settle(switch, rid, tokens, rounds=60):
    for _ in range(rounds):
        switch.note_verify([rid], [tokens])


class TestSwitch(CustomTestCase):
    def test_threshold_at_one_stream_and_at_twenty_five(self):
        """Draft iff tokens per verify >= 1.1 x break-even at the load."""
        be1, be25 = dict(TABLE)[1], dict(TABLE)[25]
        s = _switch()
        _settle(s, "a", DRAFT_MARGIN * be1 + 0.05)
        self.assertTrue(_ask(s, c=1))
        s = _switch()
        _settle(s, "a", DRAFT_MARGIN * be1 + 0.05)
        self.assertFalse(_ask(s, c=25), f"{DRAFT_MARGIN * be1 + 0.05} under {be25}")
        s = _switch()
        _settle(s, "a", DRAFT_MARGIN * be25 + 0.05)
        self.assertTrue(_ask(s, c=25))

    def test_new_request_starts_at_the_prior(self):
        s = _switch()
        self.assertEqual(s.tokens_per_verify(["new"]), NEW_REQUEST_TOKENS_PER_VERIFY)
        # 3.0 tokens clears the margin at one stream but not at 25.
        self.assertTrue(_ask(s, rids=("new",), c=1))
        self.assertFalse(_ask(_switch(), rids=("new",), c=25))

    def test_ewma_moves_toward_the_rate(self):
        s = _switch()
        s.note_verify(["a"], [6])
        first = s.tokens_per_verify(["a"])
        self.assertGreater(first, NEW_REQUEST_TOKENS_PER_VERIFY)
        self.assertLess(first, 6)
        _settle(s, "a", 2)
        self.assertAlmostEqual(s.tokens_per_verify(["a"]), 2, places=2)

    def test_micro_batch_rate_is_the_mean_of_its_rows(self):
        s = _switch()
        _settle(s, "a", 6)
        _settle(s, "b", 2)
        self.assertAlmostEqual(s.tokens_per_verify(["a", "b"]), 4, places=2)

    def test_hysteresis(self):
        """Off below 1.1 x BE, back on only at 1.25 x BE."""
        be = dict(TABLE)[1]
        between = (DRAFT_MARGIN + RESUME_MARGIN) / 2 * be
        s = _switch()
        _settle(s, "a", 1.0)
        self.assertFalse(_ask(s))
        _settle(s, "a", between)
        self.assertFalse(_ask(s), "between the margins stays off")
        _settle(s, "a", RESUME_MARGIN * be + 0.05)
        self.assertTrue(_ask(s))
        _settle(s, "a", between)
        self.assertTrue(_ask(s), "between the margins stays on")

    def test_probe_refreshes_the_rate(self):
        """Off, every 32nd round still drafts, so a verify can bring the rate up."""
        s = _switch()
        _settle(s, "a", 1.0)
        answers = [_ask(s) for _ in range(PROBE_EVERY_ROUNDS * 3)]
        self.assertEqual(sum(answers), 3)
        self.assertEqual(s.num_probes, 3)
        self.assertTrue(answers[PROBE_EVERY_ROUNDS - 1])

    def test_micro_batches_keep_their_own_state(self):
        s = _switch()
        _settle(s, "a", 1.0)
        _settle(s, "b", 5.0)
        self.assertFalse(_ask(s, rids=("a",), mb=0))
        self.assertTrue(_ask(s, rids=("b",), mb=1))

    def test_forced_modes_and_what_they_cannot_change(self):
        on = _switch(mode="on")
        _settle(on, "a", 1.0)
        self.assertTrue(_ask(on, c=25))
        self.assertFalse(_ask(on, return_logprob=True))
        self.assertFalse(_ask(_switch(mode="off")))
        capped = _switch(mode="on", max_verify_bs=4)
        self.assertTrue(_ask(capped, rids=list("abcd")))
        self.assertFalse(_ask(capped, rids=list("abcde")))

    def test_forget_and_clear(self):
        s = _switch()
        _settle(s, "a", 1.0)
        s.forget("a")
        self.assertEqual(s.tokens_per_verify(["a"]), NEW_REQUEST_TOKENS_PER_VERIFY)
        _settle(s, "a", 1.0)
        _ask(s)
        s.clear()
        self.assertEqual(s._micro_batches, {})

    def test_breakeven_table(self):
        self.assertEqual(interpolate_breakeven(TABLE, 1), TABLE[0][1])
        self.assertEqual(interpolate_breakeven(TABLE, 25), dict(TABLE)[25])
        self.assertEqual(interpolate_breakeven(TABLE, 999), dict(TABLE)[25])
        between = interpolate_breakeven(TABLE, 17)
        self.assertAlmostEqual(between, (dict(TABLE)[9] + dict(TABLE)[25]) / 2)
        self.assertEqual(
            parse_breakeven(json.dumps({"3": 2.5, "1": 2.0}), {}),
            [(1, 2.0), (3, 2.5)],
        )
        with self.assertRaises(ValueError):
            parse_breakeven("{}", {})

    def test_from_env_picks_the_order_and_the_window_figures(self):
        with envs.SGLANG_PP_DSPARK_LAUNCH_FIRST.override(True):
            s = DsparkPPSwitch.from_env(envs)
        self.assertEqual(s.breakeven, sorted(DEFAULT_BREAKEVEN_LAUNCH_FIRST.items()))
        with envs.SGLANG_PP_DSPARK_BREAKEVEN.override('{"1": 9.0}'):
            s = DsparkPPSwitch.from_env(envs)
        self.assertEqual(s.breakeven, [(1, 9.0)])


if __name__ == "__main__":
    unittest.main()
