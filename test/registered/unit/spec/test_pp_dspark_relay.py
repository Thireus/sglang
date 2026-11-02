"""DSpark under pipeline parallelism: the relay, on three stages in one process.

The event loop, the relay and the accept are the real code (pp_dspark_toy_ring.py). The
measure of right is plain greedy: DSpark changes how fast tokens come, not which.
"""

import random
import unittest
from unittest import mock

import pp_dspark_toy_ring as toy_ring
import torch
from pp_dspark_toy_ring import (
    GAMMA,
    PP,
    VOCAB,
    WIDTH,
    ToyBatch,
    ToyReq,
    ToyRing,
    plain_output,
    toy_next,
)

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_exchange_outputs_before_forward,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components.dspark_pp_relay import DSparkPPRelayInput
from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _requests(num, *, seed=0, arrival_gap=0, max_new=40, prompt_len=8, **extra):
    rng = random.Random(seed)
    return [
        {
            "rid": f"r{i}",
            "prompt": [rng.randrange(3, VOCAB) for _ in range(prompt_len + i)],
            "max_new": max_new + 3 * i,
            "arrival": i * arrival_gap,
            **extra,
        }
        for i in range(num)
    ]


class _RingCase(CustomTestCase):
    def setUp(self):
        scope = envs.SGLANG_ENABLE_PP_SPEC.override(True)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def _assert_all_plain_greedy(self, ring, requests):
        for rank, stage in enumerate(ring.stages):
            for r in requests:
                self.assertEqual(
                    stage.finished_rids.get(r["rid"]),
                    plain_output(r["prompt"], r["max_new"], r.get("stop")),
                    f"stage {rank}, {r['rid']}",
                )


class TestRelayOnThreeStages(_RingCase):
    def test_block_size_three_needs_no_six(self):
        """The bring-up's row of block size 3: nothing on the path may assume 6 rows."""
        requests = _requests(5, arrival_gap=3)
        with (
            mock.patch.object(toy_ring, "WIDTH", 4),
            mock.patch.object(toy_ring, "GAMMA", 3),
        ):
            ring = ToyRing(requests, accuracy=0.8, seed=11)
            ring.run()
        self._assert_all_plain_greedy(ring, requests)
        logs = [stage.round_log for stage in ring.stages]
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(logs[0], logs[2])
        verify_rounds = [entry for entry in logs[0] if entry[0] == "verify"]
        self.assertGreater(len(verify_rounds), 10)

    def test_a_right_draft_commits_whole_blocks(self):
        """The relayed chain is what the verify reads: a draft that is always right commits
        all six rows of a round, so the round count says the chain arrived intact."""
        requests = _requests(1, max_new=120)
        ring = ToyRing(requests, accuracy=1.0, seed=4)
        ring.run()
        self._assert_all_plain_greedy(ring, requests)
        verify = [entry for entry in ring.stages[0].round_log if entry[0] == "verify"]
        self.assertLessEqual(len(verify), 120 // WIDTH + 2)
        self.assertGreaterEqual(len(verify), 120 // WIDTH - 1)

    def test_the_switch_hears_each_verify_and_stops_a_useless_draft(self):
        """A draft that is never right drives the request's tokens per verify under the
        break-even; the micro-batch runs plain, with a probe now and then."""
        requests = _requests(1, max_new=150)
        ring = ToyRing(requests, accuracy=0.0, mode="auto", seed=5)
        ring.run()
        self._assert_all_plain_greedy(ring, requests)
        kinds = [entry[0] for entry in ring.stages[0].round_log]
        self.assertGreater(ring.switch.num_skipped, 50)
        self.assertGreater(kinds.count("plain"), 3 * kinds.count("verify"))

    def test_stages_rebuild_identical_verify(self):
        """Invariant 1: all stages run the same rounds: rids, tokens, positions."""
        requests = _requests(5, arrival_gap=4)
        ring = ToyRing(requests, accuracy=0.8)
        ring.run()
        self._assert_all_plain_greedy(ring, requests)
        logs = [stage.round_log for stage in ring.stages]
        self.assertGreater(sum(kind == "verify" for kind, *_ in logs[0]), 10)
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(logs[0], logs[2])

    def test_rows_follow_rids(self):
        """Invariant 2: merge from prefill and finishes keep each row with its request.
        The toy forward asserts every chain's anchor and position against the request's
        own tokens; output equal to plain greedy says no row ever crossed."""
        requests = _requests(8, arrival_gap=2, max_new=20)
        for i, r in enumerate(requests):
            r["stop"] = plain_output(r["prompt"], r["max_new"])[5 + i]
        ring = ToyRing(requests, accuracy=0.7, seed=3, max_micro_batch=3)
        ring.run()
        self._assert_all_plain_greedy(ring, requests)
        self.assertTrue(all(s.round_log for s in ring.stages))

    def test_relay_rows_filter_merge_adopt_reindex(self):
        """Invariant 2, on the rows themselves: retract is a filter, as are finish and abort."""
        tokens = torch.arange(18, dtype=torch.int64).view(3, 6)
        relay = DSparkPPRelayInput(
            ["a", "b", "c"], tokens, has_chain=[True, False, True]
        )
        relay.filter_batch(torch.tensor([0, 2]), [0, 2])
        self.assertEqual(relay.rids, ["a", "c"])
        self.assertEqual(relay.has_chain, [True, True])
        self.assertEqual(relay.tokens[1, 0].item(), 12)
        other = DSparkPPRelayInput.degenerate(["d"], torch.tensor([7]), 6)
        relay.merge_batch(other)
        self.assertEqual(relay.rids, ["a", "c", "d"])
        self.assertEqual(relay.has_chain, [True, True, False])
        self.assertEqual(relay.tokens[2].tolist(), [7, 0, 0, 0, 0, 0])
        self.assertEqual(relay.tokens[1].tolist(), list(range(12, 18)))
        # "d" merged after the forward that relayed "c": it keeps its own row.
        relayed = DSparkPPRelayInput(
            ["c"], torch.full((1, 6), 5, dtype=torch.int64), has_chain=[True]
        )
        relay.adopt(relayed)
        self.assertEqual(relay.tokens[1].tolist(), [5] * 6)
        self.assertEqual(relay.tokens[2].tolist(), [7, 0, 0, 0, 0, 0])
        self.assertEqual(relay.has_chain, [True, True, False])
        back = relay.reindex(["d", "a"])
        self.assertEqual(back.rids, ["d", "a"])
        self.assertEqual(back.has_chain, [False, True])
        self.assertEqual(back.tokens[0, 0].item(), 7)
        empty = DSparkPPRelayInput([], torch.empty((0, 6), dtype=torch.int64))
        empty.merge_batch(other)
        self.assertEqual(empty.rids, ["d"])

    def test_relay_survives_next_replay(self):
        """Invariant 3: the sampler's static buffer is overwritten by the next draft
        while the ring still carries this one. The channels read a tensor when the peer
        takes it, so an aliased chain would corrupt the verify on the other stages."""
        requests = _requests(6, max_new=30)
        ring = ToyRing(requests, accuracy=0.9, seed=1)
        ring.run()
        self.assertGreater(ring.proposer.num_proposals, 10)
        self._assert_all_plain_greedy(ring, requests)
        logs = [stage.round_log for stage in ring.stages]
        self.assertEqual(logs[0], logs[2])

    def test_output_stride_per_round(self):
        """Invariant 4: plain rounds are one token wide, verify rounds six; the tokens
        of the output keys agree with the request on all stages in both."""
        requests = _requests(2, max_new=25)
        ring = ToyRing(requests, accuracy=0.8)
        ring.run()
        kinds = {kind for kind, *_ in ring.stages[0].round_log}
        self.assertEqual(kinds, {"plain", "verify"})
        self._assert_all_plain_greedy(ring, requests)

    def test_round_kind(self):
        """Invariant 5: no chain anywhere is a plain round on all stages, with the
        switch off every round is."""
        requests = _requests(3, max_new=20)
        ring = ToyRing(requests, mode="off")
        ring.run()
        for stage in ring.stages:
            self.assertEqual({kind for kind, *_ in stage.round_log}, {"plain"})
        self.assertEqual(ring.proposer.num_proposals, 0)
        self._assert_all_plain_greedy(ring, requests)

    def test_degenerate_row_commit(self):
        """Invariant 5: a request merged into a micro-batch that has chains carries bonus
        plus zeros and commits its bonus alone, whatever its zeros would have matched."""
        batch = ToyBatch(reqs=[ToyReq("a", [1], 9), ToyReq("b", [1], 9)])
        relay = DSparkPPRelayInput(
            ["a", "b"],
            torch.tensor([[5, 6, 7, 8, 9, 10], [11, 0, 0, 0, 0, 0]]),
            has_chain=[True, False],
        )
        self.assertFalse(relay.runs_plain(batch))
        self.assertEqual(relay.verify_cutoff().tolist(), [6, 1])
        from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy

        # Row b's target even agrees with the zero draft: the cutoff still commits 1.
        logits = torch.zeros(12, VOCAB)
        for row, tok in enumerate([6, 7, 8, 30, 31, 32, 0, 0, 0, 0, 0, 0]):
            logits[row, tok] = 5.0
        correct, bonus, _ = AcceptGreedy.execute(
            candidates=relay.tokens,
            target_logits=logits,
            verify_num_draft_tokens=6,
            cutoff_verify_lens=relay.verify_cutoff(),
        )
        self.assertEqual(correct.tolist(), [3, 0])
        self.assertEqual(bonus.tolist(), [30, 0])

    def test_chain_survives_chunked_prefill(self):
        """Invariant 6: a decoding row sits through a later request's prefill chunks; its
        chain and its slot do not change, and nothing it drafted is lost."""
        requests = _requests(3, arrival_gap=6, max_new=30, prompt_len=20)
        ring = ToyRing(requests, accuracy=0.9, chunk=4, seed=5, max_micro_batch=3)
        ring.run()
        self._assert_all_plain_greedy(ring, requests)
        log = ring.stages[0].round_log
        self.assertTrue(any(kind == "verify" for kind, *_ in log))

    def test_no_none_in_relay(self):
        """Invariant 7: the output dict sends no None and no topology for DSpark."""
        stage = SchedulerPPMixin()
        stage._pp_spec_relay = True
        stage._pp_dspark_relay = True
        batch = ToyBatch(reqs=[ToyReq("a", [1], 9)])
        result = GenerationBatchResult(
            next_token_ids=torch.zeros(6, dtype=torch.int64),
            accept_lens=torch.ones(1, dtype=torch.int32),
            new_seq_lens=torch.ones(1, dtype=torch.int64),
            next_draft_input=DSparkPPRelayInput.degenerate(
                ["a"], torch.tensor([3]), 6
            ),
            next_verify_chain=torch.zeros(6, dtype=torch.int64),
        )
        result.next_draft_input.bonus_tokens = torch.tensor([3])
        tensor_dict = stage._pp_prepare_tensor_dict(result, batch)
        self.assertNotIn(None, list(tensor_dict.values()))
        self.assertNotIn("spec_next_parents", tensor_dict)
        self.assertNotIn("spec_next_top_scores", tensor_dict)
        self.assertEqual(
            set(tensor_dict),
            {
                "next_token_ids",
                "spec_accept_lens",
                "spec_new_seq_lens",
                "spec_bonus_tokens",
                "spec_next_chain",
            },
        )

    def test_grammar_order(self):
        """Invariant 22: stage 2 reads the grammar FSM for a verify after the ring has
        processed the round before: every token this request has is applied by then."""
        requests = _requests(1, max_new=30)
        ring = ToyRing(requests, accuracy=0.9)
        seen = []
        real = ring.stages[-1].toy_forward

        def check(forward_batch, proxies):
            for row, req in enumerate(forward_batch.reqs):
                # Processed results reach the anchor of this round, no further back.
                anchor_pos = forward_batch.positions.view(len(forward_batch.reqs), -1)[
                    row, 0
                ]
                seen.append(len(req.tokens) - int(anchor_pos))
            return real(forward_batch, proxies)

        ring.stages[-1].toy_forward = check
        ring.run()
        self.assertEqual(set(seen), {1})

    def test_stop_inside_block(self):
        """Invariant 23: a stop token inside a block trims to the same output on all
        stages, and the rest of the block is dropped."""
        requests = _requests(4, max_new=50)
        for i, r in enumerate(requests):
            r["stop"] = plain_output(r["prompt"], r["max_new"])[11 + i]
        ring = ToyRing(requests, accuracy=1.0)
        ring.run()
        self._assert_all_plain_greedy(ring, requests)
        for r in requests:
            lengths = {len(s.finished_rids[r["rid"]]) for s in ring.stages}
            self.assertEqual(len(lengths), 1)

    def test_round_kind_is_one_fact_under_bursts(self):
        """Invariant 27: random arrivals and the auto switch; every stage ran the same
        kinds for the same rows in the same order, across many rounds."""
        rng = random.Random(9)
        requests = [
            {
                "rid": f"b{i}",
                "prompt": [rng.randrange(3, VOCAB) for _ in range(5 + i % 7)],
                "max_new": 10 + rng.randrange(60),
                "arrival": rng.randrange(400),
            }
            for i in range(40)
        ]
        ring = ToyRing(
            requests,
            accuracy=0.55,
            mode="auto",
            seed=4,
            chunk=3,
            breakeven=[(1, 2.0), (2, 2.65), (9, 2.76), (25, 3.59)],
        )
        ring.run(timeout=240)
        logs = [stage.round_log for stage in ring.stages]
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(logs[0], logs[2])
        kinds = [kind for kind, *_ in logs[0]]
        self.assertGreater(kinds.count("plain"), 0)
        self.assertGreater(kinds.count("verify"), 0)
        self._assert_all_plain_greedy(ring, requests)

    def test_health_never_reaches_a_verify(self):
        """Invariant 35: one token, it ends in prefill: no decode round, no draft."""
        requests = [{"rid": "health", "prompt": [5, 6], "max_new": 1, "arrival": 0}]
        ring = ToyRing(requests, mode="on")
        ring.run()
        self.assertEqual(ring.stages[0].round_log, [])
        self.assertEqual(ring.proposer.num_proposals, 0)
        self._assert_all_plain_greedy(ring, requests)

    def test_engram_history_on_all_stages_is_the_committed_tokens(self):
        """Invariant 10 at ring level: anchor plus accepted drafts, once each, in order,
        whether the round verified or ran plain."""
        requests = _requests(4, max_new=30, arrival_gap=3)
        ring = ToyRing(requests, accuracy=0.7, mode="auto", seed=2)
        ring.run()
        for stage in ring.stages:
            for i, r in enumerate(requests):
                output = stage.finished_rids[r["rid"]]
                committed = stage.engram.committed.get(i + 1, [])
                # Every token but the last has been the anchor of a commit, or the
                # run went past the request's end by at most one block.
                self.assertEqual(committed[: len(output) - 1], output[:-1][: len(committed)])
                self.assertGreaterEqual(len(committed), len(output) - 1 - WIDTH)

    def test_all_stages_insert_the_same_committed_tokens(self):
        """The tree's insert (checkpoint_into_tree) reads a request's fill ids up to its
        committed length; every stage holds the same ones when the request ends."""
        requests = _requests(5, max_new=25, arrival_gap=2)
        ring = ToyRing(requests, accuracy=0.75, mode="auto", seed=6)
        ring.run()
        commits = [stage.finished_commits for stage in ring.stages]
        self.assertEqual(commits[0], commits[1])
        self.assertEqual(commits[0], commits[2])
        self.assertEqual(set(commits[0]), {r["rid"] for r in requests})

    def test_draft_window_gets_every_committed_row(self):
        """Invariant 14 at ring level: each committed position, once, in order; plain
        rounds commit one row and verify rounds their accepted length."""
        requests = _requests(3, max_new=30)
        ring = ToyRing(requests, accuracy=0.6, mode="auto", seed=8)
        ring.run()
        for i, r in enumerate(requests):
            rows = ring.injector.rows[i + 1]
            start = len(r["prompt"])
            self.assertEqual(rows, list(range(start, start + len(rows))))


class TestExchangeOrder(CustomTestCase):
    def _decision(self, **kw):
        base = dict(
            cur_batch=ToyBatch(forward_mode=ForwardMode.DECODE),
            spec_relay=True,
            is_last_rank=False,
            async_batch_depth=0,
        )
        base.update(kw)
        return _pp_exchange_outputs_before_forward(**base)

    def test_stock_order_keeps_exchange_first_for_verify(self):
        self.assertTrue(self._decision())
        self.assertFalse(self._decision(launch_first=True))
        self.assertTrue(self._decision(async_batch_depth=1, launch_first=True))
        self.assertFalse(
            self._decision(cur_batch=ToyBatch(forward_mode=ForwardMode.EXTEND))
        )

    def test_plain_rounds_keep_the_plain_order_and_verify_rounds_the_flag(self):
        """Exchanging before a plain round would cap every stage at 2/3; a verify round
        takes launch-first only with SGLANG_PP_DSPARK_LAUNCH_FIRST."""
        reqs = [ToyReq("a", [1], 9)]
        plain = ToyBatch(
            reqs=reqs,
            spec_info=DSparkPPRelayInput.degenerate(["a"], torch.tensor([3]), WIDTH),
        )
        verify = ToyBatch(
            reqs=reqs,
            spec_info=DSparkPPRelayInput(
                ["a"], torch.zeros((1, WIDTH), dtype=torch.int64), has_chain=[True]
            ),
        )
        stage = SchedulerPPMixin()
        self.assertFalse(stage._pp_dspark_launches_first(plain))
        stage._pp_dspark_relay = True
        self.assertTrue(stage._pp_dspark_launches_first(plain))
        self.assertFalse(stage._pp_dspark_launches_first(verify))
        self.assertFalse(stage._pp_dspark_launches_first(None))
        stage._pp_dspark_launch_first = True
        self.assertTrue(stage._pp_dspark_launches_first(verify))

    def test_rings_do_not_deadlock_in_either_order(self):
        """Invariant 30: rendezvous sends, one to three micro-batches, prefill mixed in."""
        for launch_first in (False, True):
            for num, gap, chunk in ((1, 0, 1 << 30), (2, 1, 3), (3, 0, 2), (6, 2, 4)):
                with self.subTest(launch_first=launch_first, num=num):
                    with envs.SGLANG_ENABLE_PP_SPEC.override(True):
                        with envs.SGLANG_PP_DSPARK_LAUNCH_FIRST.override(launch_first):
                            requests = _requests(num, arrival_gap=gap, max_new=25)
                            ring = ToyRing(requests, chunk=chunk, seed=num)
                            ring.run(timeout=120)
                    for r in requests:
                        self.assertEqual(
                            ring.stages[-1].finished_rids[r["rid"]],
                            plain_output(r["prompt"], r["max_new"]),
                        )
                    self.assertEqual(
                        ring.stages[0]._pp_dspark_launch_first, launch_first
                    )


class TestEagleRelayUntouched(CustomTestCase):
    def test_eagle_rows_keep_their_class_and_topology(self):
        relay = PPSpecRelayInput.degenerate(["a"], torch.tensor([3]), 4)
        self.assertNotIsInstance(relay, DSparkPPRelayInput)
        self.assertIsNone(relay.parents)
        # GAMMA, WIDTH and PP name what the ring runs; a typo here would fail loudly.
        self.assertEqual((PP, WIDTH), (3, GAMMA + 1))
        self.assertEqual(toy_next([1, 2, 3]), toy_next([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
