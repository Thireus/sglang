"""DSpark under pipeline parallelism: a verify length per request (CPU).

``SGLANG_PP_DSPARK_VERIFY_SURVIVAL`` makes the last stage give each request a verify
length from the draft's confidence head; the length rides the ring after the chain's
rows, every stage masks the rows past it for the routed experts, and the accept is
the full accept cut at the length. What is checked here:

* the policy, the wire format and the row mask;
* the draft: both forms of the proposal give the lengths, from the trained head
  only, and the head is built in static mode for them;
* the relay: lengths carried through filter, merge, adopt and reindex, and the
  generalised cutoff;
* the accept with lengths equals "full verify, then cut", whatever the skipped rows
  hold;
* three stages in one process (pp_dspark_toy_ring.py): every request still gets the
  output plain greedy gives it, the stages agree on every mask, rows are skipped, and
  no skipped row is committed;
* off (unset or 0): the ring's messages, the forwards' inputs and the layers are the
  ones of a build without this file's subject;
* the graph slot, the route masks of both kinds of layer (a hot/cold layer in
  every form of its route split), the launch's refusals;
* the slot beside the per-request GLP alpha rows of --glp-alpha-per-request in one
  verify runner: the capture batch, a padded replay and a pre-planned replay fill
  each slot with its own rows, and each is the same with the other feature off.

The kernels' side (kept rows bit-equal in graph replay and in eager calls on dirty
memory) is the GPU test test_dsv41_moe_row_mask_gpu.py.
"""

import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pp_dspark_toy_ring as toy_ring
import torch
from pp_dspark_toy_ring import GAMMA, WIDTH, ToyRing, plain_output

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import dspark_binding
from sglang.srt.layers.moe import row_mask as row_mask_module
from sglang.srt.layers.moe.topk import StandardTopKOutput, StandardTopKOutputPacked
from sglang.srt.model_executor import forward_context as fc
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    FillContext,
    GraphSlot,
    PaddingPolicy,
    build_decode_registry,
    moe_verify_row_mask_slot,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as graph_runner
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.runtime_context import get_context, get_serving
from sglang.srt.speculative.dspark_components import dspark_config, dspark_draft
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
from sglang.srt.speculative.dspark_components import dspark_pp_launch as launch
from sglang.srt.speculative.dspark_components import dspark_pp_verify_schedule as vs
from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise
from sglang.srt.speculative.dspark_components.dspark_pp_relay import DSparkPPRelayInput
from sglang.srt.speculative.dspark_components.dspark_verify import build_accept_outs
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

VOCAB = 53


class _ScheduleCase(CustomTestCase):
    """The switch is read once per process; each test names its own value."""

    SURVIVAL = 0.0

    def setUp(self):
        for field, value in (
            (envs.SGLANG_ENABLE_PP_SPEC, True),
            (envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL, self.SURVIVAL),
        ):
            scope = field.override(value)
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)
        vs.forget_threshold()
        self.addCleanup(vs.forget_threshold)


def _toy_draft_model(confidence):
    """A draft for the folded proposal (after test_dspark_draft_block.py) whose
    confidence head answers ``confidence``."""
    return SimpleNamespace(
        markov_head=VanillaMarkov(vocab_size=VOCAB, markov_rank=8),
        sample_from_anchor=True,
        confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1))),
        lm_head=SimpleNamespace(org_vocab_size=VOCAB),
        compute_base_logits=lambda h: (torch.zeros(h.shape[0], VOCAB), h + 1),
        compute_confidence=lambda **kw: (confidence, confidence.clone()),
    )


def _folded_sampler(model, acceptance_estimator=None):
    """The draft graph's tail on CPU, as maybe_build_draft_sampler builds it."""
    with (
        mock.patch.object(
            dspark_binding, "megakernel_draft_sampler", return_value=None
        ),
        mock.patch.object(dspark_binding, "megakernel_agree", mock.Mock()),
        mock.patch.object(dspark_binding, "get_dp_tp_group", return_value="group"),
    ):
        return ds.maybe_build_draft_sampler(
            draft_model=model,
            gamma=GAMMA,
            max_bs=8,
            device="cpu",
            tp_rank=1,
            noise=DraftNoise(key=3, offset=torch.zeros(1, dtype=torch.int64)),
            greedy_draft=True,
            tp_sync="sync",
            acceptance_estimator=acceptance_estimator,
        )


def _requests(num, *, seed=0, arrival_gap=0, max_new=40, prompt_len=8, **extra):
    rng = random.Random(seed)
    return [
        {
            "rid": f"r{i}",
            "prompt": [rng.randrange(3, toy_ring.VOCAB) for _ in range(prompt_len + i)],
            "max_new": max_new + 3 * i,
            "arrival": i * arrival_gap,
            **extra,
        }
        for i in range(num)
    ]


class TestPolicyAndWire(_ScheduleCase):
    SURVIVAL = 0.08

    def test_the_switch_is_read_once_and_needs_the_relay(self):
        self.assertEqual(vs.verify_survival_threshold(), 0.08)
        self.assertTrue(vs.verify_schedule_enabled())
        with envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(0.5):
            self.assertEqual(vs.verify_survival_threshold(), 0.08)
            vs.forget_threshold()
            self.assertEqual(vs.verify_survival_threshold(), 0.5)
        vs.forget_threshold()
        with envs.SGLANG_ENABLE_PP_SPEC.override(False):
            self.assertEqual(vs.verify_survival_threshold(), 0.0)
        vs.forget_threshold()
        with envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(1.5):
            with self.assertRaises(ValueError):
                vs.verify_survival_threshold()

    def test_lengths_are_one_plus_the_drafts_that_survive(self):
        gen = torch.Generator().manual_seed(3)
        confidence = torch.rand(200, GAMMA, generator=gen)
        for threshold in (0.02, 0.06, 0.08, 0.14, 0.5, 1.0):
            lens = vs.scheduled_verify_lens(confidence, threshold)
            self.assertEqual(lens.dtype, torch.int64)
            want = []
            for row in confidence.tolist():
                survival, kept = torch.tensor(1.0), 0
                for c in row:
                    survival = survival * torch.tensor(c)
                    if float(survival) < threshold:
                        break
                    kept += 1
                want.append(1 + kept)
            self.assertEqual(lens.tolist(), want, threshold)
            self.assertTrue(1 <= int(lens.min()) and int(lens.max()) <= GAMMA + 1)
        # A stricter threshold never verifies more rows.
        loose = vs.scheduled_verify_lens(confidence, 0.06)
        strict = vs.scheduled_verify_lens(confidence, 0.3)
        self.assertTrue(bool((strict <= loose).all()))
        sure = vs.scheduled_verify_lens(torch.ones(3, GAMMA), 0.9)
        self.assertEqual(sure.tolist(), [GAMMA + 1] * 3)
        # A head that says nothing useful verifies the anchor row alone.
        nan = vs.scheduled_verify_lens(torch.full((2, GAMMA), float("nan")), 0.1)
        self.assertEqual(nan.tolist(), [1, 1])

    def test_the_chain_carries_the_lengths_after_its_rows(self):
        rows = torch.arange(3 * WIDTH, dtype=torch.int64).view(3, WIDTH)
        flat = vs.pack_chain(rows, None)
        self.assertTrue(torch.equal(flat, rows.reshape(-1)))
        tokens, lens = vs.unpack_chain(flat, 3, WIDTH)
        self.assertTrue(torch.equal(tokens, rows))
        self.assertIsNone(lens)
        static = torch.tensor([2, WIDTH, 1], dtype=torch.int64)
        packed = vs.pack_chain(rows, static)
        self.assertEqual(packed.numel(), 3 * (WIDTH + 1))
        # A copy: the draft graph's next replay overwrites its own buffer.
        static.fill_(0)
        tokens, lens = vs.unpack_chain(packed, 3, WIDTH)
        self.assertTrue(torch.equal(tokens, rows))
        self.assertEqual(lens.tolist(), [2, WIDTH, 1])
        self.assertTrue(tokens.is_contiguous() and lens.is_contiguous())
        with self.assertRaises(RuntimeError):
            vs.unpack_chain(packed[:-1], 3, WIDTH)

    def test_row_mask(self):
        lens = torch.tensor([1, 4, WIDTH], dtype=torch.int64)
        mask = vs.verify_row_mask(lens, 3, WIDTH, "cpu")
        self.assertEqual(mask.dtype, torch.int32)
        self.assertEqual(
            mask.view(3, WIDTH).tolist(),
            [[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
        )
        whole = vs.verify_row_mask(None, 2, WIDTH, "cpu")
        self.assertEqual(whole.tolist(), [1] * (2 * WIDTH))

    def test_stats_line(self):
        stats = vs.VerifyStats(every=4, width=WIDTH)
        with self.assertLogs(vs.logger, level="INFO") as logs:
            stats.note([3, 6], None)
            stats.note([2, 4], [2, 5])
        self.assertEqual(
            (stats.rounds, stats.request_rounds, stats.rows, stats.tokens),
            (2, 4, WIDTH + WIDTH + 2 + 5, 15),
        )
        # One request was cut with every verified draft right.
        self.assertEqual(stats.capped, 1)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("request_rounds=4 rows=19 tokens=15 capped=1", logs.output[0])
        self.assertIn("rows_hist=0,1,0,0,1,2", logs.output[0])
        # A request's end puts what is not yet logged on the log, once.
        with self.assertLogs(vs.logger, level="INFO") as logs:
            stats.note([1], [1])
            stats.flush()
            stats.flush()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("request_rounds=5 rows=20 tokens=16", logs.output[0])
        commit, rows = vs.read_round(
            torch.tensor([3, 1], dtype=torch.int32), torch.tensor([4, 1])
        )
        self.assertEqual((commit, rows), ([3, 1], [4, 1]))
        self.assertEqual(
            vs.read_round(torch.tensor([3, 1], dtype=torch.int32), None),
            ([3, 1], None),
        )


class TestRelayCarriesLengths(_ScheduleCase):
    SURVIVAL = 0.08

    def _relay(self, rids, lens=None, has_chain=None):
        tokens = torch.arange(len(rids) * WIDTH, dtype=torch.int64).view(-1, WIDTH)
        return DSparkPPRelayInput(
            list(rids),
            tokens,
            has_chain=[True] * len(rids) if has_chain is None else has_chain,
            verify_lens=None if lens is None else torch.tensor(lens, dtype=torch.int64),
        )

    def test_filter_reindex(self):
        relay = self._relay("abcd", [2, 3, 4, 5])
        relay.filter_batch(torch.tensor([0, 2, 3]), [0, 2, 3])
        self.assertEqual(relay.rids, ["a", "c", "d"])
        self.assertEqual(relay.verify_lens.tolist(), [2, 4, 5])
        swapped = relay.reindex(["d", "a", "c"])
        self.assertEqual(swapped.verify_lens.tolist(), [5, 2, 4])
        self.assertEqual(swapped.tokens[:, 0].tolist(), [18, 0, 12])
        self.assertIsNone(self._relay("ab").reindex(["b", "a"]).verify_lens)

    def test_merge(self):
        relay = self._relay("ab", [2, 3])
        relay.merge_batch(self._relay("c", has_chain=[False]))
        # A request merging in from prefill has no length: its whole (empty) block.
        self.assertEqual(relay.verify_lens.tolist(), [2, 3, WIDTH])
        self.assertEqual(relay.verify_cutoff().tolist(), [2, 3, 1])
        plain = self._relay("ab")
        plain.merge_batch(self._relay("c", [4]))
        self.assertEqual(plain.verify_lens.tolist(), [WIDTH, WIDTH, 4])
        neither = self._relay("ab")
        neither.merge_batch(self._relay("c"))
        self.assertIsNone(neither.verify_lens)
        empty = DSparkPPRelayInput([], torch.zeros((0, WIDTH), dtype=torch.int64))
        empty.merge_batch(self._relay("c", [4]))
        self.assertEqual(empty.verify_lens.tolist(), [4])

    def test_adopt(self):
        live = self._relay("abc", [2, 3, 4])
        # The relay covers b and c in another order; a keeps its own length.
        live.adopt(self._relay("cb", [6, 5]))
        self.assertEqual(live.verify_lens.tolist(), [2, 5, 6])
        # A relay without lengths (no schedule that round) clears the rows it covers.
        live.adopt(self._relay("abc"))
        self.assertIsNone(live.verify_lens)
        live = self._relay("abc", [2, 3, 4])
        live.adopt(self._relay("b"))
        self.assertEqual(live.verify_lens.tolist(), [2, WIDTH, 4])
        # A relay that mentions none of the rows changes nothing.
        live.adopt(self._relay("z", [1]))
        self.assertEqual(live.verify_lens.tolist(), [2, WIDTH, 4])
        fresh = self._relay("ab")
        fresh.adopt(self._relay("ab", [3, 1]))
        self.assertEqual(fresh.verify_lens.tolist(), [3, 1])

    def test_the_cutoff_is_one_mechanism(self):
        self.assertIsNone(self._relay("ab").verify_cutoff())
        self.assertEqual(self._relay("ab", [3, 6]).verify_cutoff().tolist(), [3, 6])
        mixed = self._relay("abc", [3, 6, 4], has_chain=[True, False, True])
        self.assertEqual(mixed.verify_cutoff().tolist(), [3, 1, 4])
        today = self._relay("abc", has_chain=[True, False, True])
        self.assertEqual(today.verify_cutoff().tolist(), [WIDTH, 1, WIDTH])
        mask = vs.relay_row_mask(mixed, 3, WIDTH, "cpu").view(3, WIDTH)
        self.assertEqual(mask.sum(dim=1).tolist(), [3, 1, 4])
        whole = vs.relay_row_mask(self._relay("ab"), 2, WIDTH, "cpu")
        self.assertEqual(whole.tolist(), [1] * (2 * WIDTH))


class TestAcceptWithLengths(CustomTestCase):
    """Exactly the full accept, cut at each request's length."""

    def _round(self, gen, bs):
        candidates = torch.randint(3, VOCAB, (bs, WIDTH), generator=gen)
        truth = torch.randint(3, VOCAB, (bs, WIDTH), generator=gen)
        # Each request's drafts are right up to a drawn point.
        right = torch.randint(0, WIDTH + 1, (bs,), generator=gen)
        for row in range(bs):
            for col in range(min(int(right[row]), WIDTH - 1)):
                truth[row, col] = candidates[row, col + 1]
        logits = torch.zeros(bs * WIDTH, VOCAB)
        logits[torch.arange(bs * WIDTH), truth.reshape(-1)] = 9.0
        return candidates, truth, logits

    def _outs(self, candidates, logits, lens):
        correct_len, bonus, trim = AcceptGreedy.torch(
            candidates=candidates,
            target_logits=logits,
            verify_num_draft_tokens=WIDTH,
            cutoff_verify_lens=lens,
        )
        return build_accept_outs(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=trim,
            prefix_lens=torch.full((candidates.shape[0],), 100, dtype=torch.int64),
            draft_tokens=candidates[:, 1:],
            gamma=GAMMA,
            verify_num_draft_tokens=WIDTH,
            tp_sync=SpecTpSync(SimpleNamespace(world_size=1)),
            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GREEDY,
        )

    def test_equal_to_full_verify_then_cut(self):
        gen = torch.Generator().manual_seed(11)
        rounds = seen_cut = seen_bonus_row = 0
        for _ in range(300):
            bs = int(torch.randint(1, 10, (1,), generator=gen))
            candidates, truth, logits = self._round(gen, bs)
            lens = torch.randint(1, WIDTH + 1, (bs,), generator=gen)
            full = self._outs(candidates, logits, None)
            # What a skipped row holds: the next draft called right, then noise.
            dirty = logits.clone().view(bs, WIDTH, VOCAB)
            for row in range(bs):
                for col in range(int(lens[row]), WIDTH):
                    dirty[row, col] = torch.randn(VOCAB, generator=gen)
                    if col + 1 < WIDTH:
                        dirty[row, col, candidates[row, col + 1]] = 50.0
            cut = self._outs(candidates, dirty.view(bs * WIDTH, VOCAB), lens)
            for row in range(bs):
                rows = int(lens[row])
                want = min(int(full.commit_lens[row]), rows)
                self.assertEqual(int(cut.commit_lens[row]), want)
                self.assertEqual(int(cut.new_seq_lens[row]), 100 + want)
                # The tokens a round commits: the accepted drafts, then the
                # target's choice at the last row it read.
                committed = cut.out_tokens.view(bs, -1)[row, :want].tolist()
                expect = candidates[row, 1:want].tolist() + [int(truth[row, want - 1])]
                self.assertEqual(committed, expect)
                self.assertEqual(int(cut.bonus[row]), int(truth[row, want - 1]))
                # The same tokens the full verify commits first.
                self.assertEqual(
                    committed[: want - 1],
                    full.out_tokens.view(bs, -1)[row, : want - 1].tolist(),
                )
                seen_cut += int(want < int(full.commit_lens[row]))
                seen_bonus_row += int(want == rows)
                rounds += 1
        self.assertGreater(rounds, 1000)
        self.assertGreater(seen_cut, 100)
        self.assertGreater(seen_bonus_row, 100)

    def test_a_round_of_right_drafts_commits_one_more(self):
        candidates = torch.tensor([[5, 6, 7, 8, 9, 10]])
        logits = torch.zeros(WIDTH, VOCAB)
        for col, tok in enumerate([6, 7, 8, 9, 10, 11]):
            logits[col, tok] = 9.0
        for rows in range(1, WIDTH + 1):
            cut = self._outs(candidates, logits, torch.tensor([rows]))
            self.assertEqual(int(cut.commit_lens[0]), rows)
            self.assertEqual(int(cut.bonus[0]), 5 + rows)


class TestScheduledRing(_ScheduleCase):
    SURVIVAL = 0.3

    def _run(self, requests, **kwargs):
        ring = ToyRing(requests, **kwargs)
        self.assertEqual(ring.survival, self.SURVIVAL)
        ring.run()
        return ring

    def _assert_all_plain_greedy(self, ring, requests):
        for rank, stage in enumerate(ring.stages):
            for r in requests:
                self.assertEqual(
                    stage.finished_rids.get(r["rid"]),
                    plain_output(r["prompt"], r["max_new"], r.get("stop")),
                    f"stage {rank}, {r['rid']}",
                )

    def test_any_length_gives_plain_greedy(self):
        """A committed token is the target's own choice at any verify length: with a
        confidence head that knows nothing, drafts right four times in five and
        skipped rows that call every draft right, the output is plain greedy's."""
        requests = _requests(5, arrival_gap=4)
        ring = self._run(requests, accuracy=0.8, seed=7)
        self._assert_all_plain_greedy(ring, requests)
        self.assertGreater(ring.masked_rows, 100)
        logs = [stage.round_log for stage in ring.stages]
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(logs[0], logs[2])
        # Every stage masked the same rows of the same rounds.
        masks = [stage.mask_log for stage in ring.stages]
        self.assertEqual(masks[0], masks[1])
        self.assertEqual(masks[0], masks[2])
        verify_masks = [m for m in masks[0] if m is not None]
        self.assertGreater(len(verify_masks), 10)
        self.assertTrue(any(0 in m for m in verify_masks))
        self.assertTrue(any(0 not in m for m in verify_masks))

    def test_a_right_draft_is_cut_only_by_the_length(self):
        """Drafts always right: a round commits exactly the rows it verified."""
        requests = _requests(1, max_new=150)
        with envs.SGLANG_PP_DSPARK_VERIFY_STATS.override(1 << 30):
            ring = self._run(requests, accuracy=1.0, seed=2)
        self._assert_all_plain_greedy(ring, requests)
        stats = ring.last_stage.verify_stats
        self.assertGreater(stats.request_rounds, 20)
        self.assertLess(stats.rows, WIDTH * stats.request_rounds)
        # The last round is cut by the request's end instead.
        self.assertGreaterEqual(stats.tokens, stats.rows - WIDTH)
        self.assertLessEqual(stats.tokens, stats.rows)
        self.assertEqual(
            sum(n * (i + 1) for i, n in enumerate(stats.rows_hist[1:])), stats.rows
        )

    def test_rows_follow_rids_with_lengths(self):
        """Merges from prefill, finishes and three micro-batches: a request's length
        stays with its rows (the toy forward checks each mask is a prefix per row,
        and the commit never passes the rows verified)."""
        requests = _requests(8, arrival_gap=2, max_new=20)
        for i, r in enumerate(requests):
            r["stop"] = plain_output(r["prompt"], r["max_new"])[5 + i]
        ring = self._run(requests, accuracy=0.7, seed=3, max_micro_batch=3)
        self._assert_all_plain_greedy(ring, requests)
        self.assertGreater(ring.masked_rows, 20)

    def test_the_lengths_ride_the_chain(self):
        requests = _requests(2, max_new=30)
        ring = self._run(requests, accuracy=0.8, seed=5)
        chains = [
            msg["spec_next_chain"] for _, msg in ring.sent if "spec_next_chain" in msg
        ]
        self.assertTrue(chains)
        for dtype, shape in chains:
            self.assertEqual(dtype, "torch.int64")
            self.assertEqual(shape[0] % (WIDTH + 1), 0)
        # No tensor was added to the ring: the same keys as with the switch off.
        with envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(0.0):
            vs.forget_threshold()
            off = ToyRing(requests, accuracy=0.8, seed=5)
            off.run()
        vs.forget_threshold()
        self.assertEqual(off.masked_rows, 0)
        keys = lambda r: set().union(*(msg.keys() for _, msg in r.sent))
        self.assertEqual(keys(ring), keys(off))
        self.assertIn("spec_next_chain", keys(off))

    def test_switch_auto_and_launch_first(self):
        requests = _requests(3, arrival_gap=2, max_new=60)
        ring = self._run(requests, accuracy=0.5, mode="auto", seed=9, launch_first=True)
        self._assert_all_plain_greedy(ring, requests)


class TestOffIsInert(_ScheduleCase):
    SURVIVAL = 0.0

    def test_the_ring_is_todays(self):
        requests = _requests(4, arrival_gap=3)
        ring = ToyRing(requests, accuracy=0.8, seed=7)
        ring.run()
        self.assertEqual(ring.masked_rows, 0)
        self.assertIsNone(ring.last_stage.verify_stats)
        for stage in ring.stages:
            for r in requests:
                self.assertEqual(
                    stage.finished_rids.get(r["rid"]),
                    plain_output(r["prompt"], r["max_new"]),
                )
        chains = [
            msg["spec_next_chain"] for _, msg in ring.sent if "spec_next_chain" in msg
        ]
        self.assertTrue(chains)
        for _, shape in chains:
            self.assertEqual(shape[0] % WIDTH, 0)
            self.assertNotEqual(shape[0] % (WIDTH + 1), 0)

    def test_nothing_is_built(self):
        self.assertFalse(vs.verify_schedule_enabled())
        relay = DSparkPPRelayInput(
            ["a", "b"],
            torch.zeros((2, WIDTH), dtype=torch.int64),
            has_chain=[True, True],
        )
        self.assertIsNone(relay.verify_lens)
        self.assertIsNone(relay.verify_cutoff())
        self.assertIsNone(vs.relay_row_mask(relay, 2, WIDTH, "cpu"))
        self.assertIsNone(vs.VerifyStats.from_env(WIDTH))
        self.assertIsNone(fc.current_moe_row_mask())
        from sglang.srt.models.deepseek_v4_dspark import (
            build_dspark_v4_confidence_head,
        )

        config = SimpleNamespace(hidden_size=16, enable_confidence_head=True)
        self.assertFalse(dspark_config.builds_confidence_head(config))
        self.assertIsNone(build_dspark_v4_confidence_head(config=config, markov_rank=4))
        sampler = _folded_sampler(_toy_draft_model(torch.zeros(4, GAMMA)))
        self.assertIsNone(sampler.verify_lens_out)
        # A draft without a head folds its proposal as before.
        headless = _toy_draft_model(torch.zeros(4, GAMMA))
        headless.confidence_head = None
        self.assertIsNone(_folded_sampler(headless).verify_lens_out)

    def test_the_layers_take_no_mask(self):
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        out = StandardTopKOutput(torch.ones(2, 2), ids, torch.zeros(2, 4))
        experts = SimpleNamespace(num_experts=4)
        self.assertIs(row_mask_module.mask_plain_routes(experts, out), out)


class TestHeadInStaticMode(_ScheduleCase):
    SURVIVAL = 0.08

    def test_the_head_is_built_for_the_schedule(self):
        from sglang.srt.models.deepseek_v4_dspark import (
            build_dspark_v4_confidence_head,
        )

        config = SimpleNamespace(hidden_size=16, enable_confidence_head=True)
        self.assertTrue(dspark_config.builds_confidence_head(config))
        head = build_dspark_v4_confidence_head(config=config, markov_rank=4)
        self.assertEqual(tuple(head.proj.weight.shape), (1, 20))

    def test_the_schedule_does_not_bring_back_a_head_that_is_left_out(self):
        config = SimpleNamespace(hidden_size=16, enable_confidence_head=False)
        self.assertFalse(dspark_config.builds_confidence_head(config))
        config.enable_confidence_head = True
        with envs.SGLANG_DSPARK_USE_ONLINE_ESTIMATOR.override(True):
            self.assertFalse(dspark_config.builds_confidence_head(config))

    def test_the_other_verify_modes_decide_as_before(self):
        config = SimpleNamespace(hidden_size=16, enable_confidence_head=True)
        for survival in (0.08, 0.0):
            with envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(survival):
                vs.forget_threshold()
                with mock.patch.object(
                    dspark_config,
                    "read_ragged_verify_mode",
                    return_value=RaggedVerifyMode.COMPACT,
                ):
                    self.assertTrue(dspark_config.builds_confidence_head(config))
                    with envs.SGLANG_DSPARK_USE_ONLINE_ESTIMATOR.override(True):
                        self.assertFalse(dspark_config.builds_confidence_head(config))
        vs.forget_threshold()


class TestDraftGivesLengths(_ScheduleCase):
    """The proposal's two forms: the draft graph's tail, and the eager block."""

    SURVIVAL = 0.3

    def _confidence(self):
        # Survival at 0.3 keeps 5, 1, 0 and 3 drafts of these four blocks.
        return torch.tensor(
            [[0.9] * GAMMA, [0.5] * GAMMA, [0.2] * GAMMA, [0.7] * GAMMA]
        )

    def test_the_graph_tail_writes_the_lengths_from_the_head(self):
        confidence = self._confidence()
        model = _toy_draft_model(confidence)
        sampler = _folded_sampler(model)
        self.assertEqual(sampler.verify_lens_out.dtype, torch.int64)
        self.assertEqual(tuple(sampler.verify_lens_out.shape), (8,))
        sampler.verify_lens_out.fill_(-1)
        tokens = torch.full((4, GAMMA), 7)
        with mock.patch.object(ds, "draft_markov_block", return_value=(tokens, None)):
            sampler(torch.ones(4 * GAMMA, 16), torch.arange(4 * GAMMA))
        self.assertEqual(sampler.verify_lens_out[:4].tolist(), [6, 2, 1, 4])
        # Only the rows of the batch are written.
        self.assertEqual(sampler.verify_lens_out[4:].tolist(), [-1] * 4)
        self.assertTrue(torch.equal(sampler.confidence_out[:4], confidence))

    def test_the_graph_tail_refuses_a_draft_without_the_head(self):
        headless = _toy_draft_model(self._confidence())
        headless.confidence_head = None
        with self.assertRaisesRegex(RuntimeError, "confidence head"):
            _folded_sampler(headless)
        # The online estimator's predictions are not the head's.
        with self.assertRaisesRegex(RuntimeError, "confidence head"):
            _folded_sampler(
                _toy_draft_model(self._confidence()),
                acceptance_estimator=mock.Mock(),
            )

    def _propose(self, *, sampler, model, estimator=None, can_run_graph=True):
        bs = 4
        proposer = SimpleNamespace(
            draft_model=model,
            sample_from_anchor=True,
            gamma=GAMMA,
            _draft_sampler=sampler,
            _acceptance_estimator=estimator,
            _greedy_draft=True,
            _sharded_accept=False,
            _noise=None,
            _tp_sync=None,
            _base_logits_context=mock.MagicMock(),
            _run_forward=lambda **kw: SimpleNamespace(
                draft_block_ids=torch.zeros((bs, GAMMA), dtype=torch.int64),
                raw_hidden=torch.ones(bs * GAMMA, 16),
                draft_hidden_3d=torch.ones(bs, GAMMA, 16),
                can_run_graph=can_run_graph,
            ),
        )
        block = SimpleNamespace(
            draft_tokens=torch.full((bs, GAMMA), 7),
            corrected_logits=None,
            greedy_mask=torch.ones(bs, dtype=torch.bool),
            temperatures=torch.ones(bs),
        )
        with (
            mock.patch.object(dspark_draft, "unwrap_lora_layer", lambda m: m),
            mock.patch.object(dspark_draft, "sample_draft_block", return_value=block),
        ):
            return dspark_draft.DraftBlockProposer.propose(
                proposer,
                batch=None,
                draft_input=None,
                verify_window=None,
                bs=bs,
                device="cpu",
                target_model=SimpleNamespace(get_input_embeddings=lambda: None),
                sampling_info=None,
            )

    def test_the_proposal_carries_the_lengths_in_both_forms(self):
        confidence = self._confidence()
        want = [6, 2, 1, 4]
        model = _toy_draft_model(confidence)
        sampler = _folded_sampler(model)
        sampler.verify_lens_out[:4] = torch.tensor(want)
        folded = self._propose(sampler=sampler, model=model)
        self.assertTrue(folded.folded)
        self.assertEqual(folded.verify_lens.tolist(), want)
        # The eager block (a batch size without a draft graph) asks the head itself.
        eager = self._propose(sampler=sampler, model=model, can_run_graph=False)
        self.assertFalse(eager.folded)
        self.assertEqual(eager.verify_lens.tolist(), want)
        self.assertTrue(torch.equal(eager.confidence, confidence))
        # An estimator's predictions schedule nothing: the whole block is verified.
        estimator = SimpleNamespace(predict=lambda **kw: ("features", confidence))
        estimated = self._propose(
            sampler=None, model=model, estimator=estimator, can_run_graph=False
        )
        self.assertIsNone(estimated.verify_lens)
        self.assertEqual(estimated.acceptance_features, "features")


class TestGraphSlotAndRoutes(_ScheduleCase):
    SURVIVAL = 0.08

    def _registry(self):
        registry = CudaGraphBufferRegistry(
            device=torch.device("cpu"), max_bs=4, max_num_tokens=4 * WIDTH
        )
        slot = registry.register_slot(moe_verify_row_mask_slot())
        return registry, slot

    def test_slot_copies_the_mask_and_masks_the_padding(self):
        registry, slot = self._registry()
        self.assertEqual(slot.buffer.dtype, torch.int32)
        mask = vs.verify_row_mask(torch.tensor([2, 5, 6]), 3, WIDTH, "cpu")
        fb = SimpleNamespace(moe_verify_row_mask=mask)
        slot.buffer.fill_(7)
        registry.fill_from(
            fb, raw_bs=3, padded_bs=4, raw_num_tokens=18, padded_num_tokens=24
        )
        self.assertEqual(slot.buffer[:18].tolist(), mask.tolist())
        self.assertEqual(slot.buffer[18:].tolist(), [0] * 6)

    def test_slot_without_a_mask_runs_every_row(self):
        registry, slot = self._registry()
        slot.buffer.zero_()
        registry.fill_from(
            SimpleNamespace(moe_verify_row_mask=None),
            raw_bs=2,
            padded_bs=2,
            raw_num_tokens=12,
            padded_num_tokens=12,
        )
        self.assertEqual(slot.buffer[:12].tolist(), [1] * 12)
        slot.post_fill(
            slot.buffer,
            SimpleNamespace(),
            FillContext(raw_bs=1, padded_bs=1, raw_num_tokens=6, padded_num_tokens=6),
        )
        self.assertEqual(slot.buffer[:6].tolist(), [1] * 6)

    def test_the_context_publishes_the_batch_mask(self):
        mask = torch.tensor([1, 0, 1], dtype=torch.int32)
        fb = SimpleNamespace(moe_num_token_non_padded=None, moe_verify_row_mask=mask)
        ctx = fc.ForwardContext(attn_backend=None)
        self.assertIsNone(fc.current_moe_row_mask())
        with fc.forward_context(ctx, forward_batch=fb):
            self.assertIs(fc.current_moe_row_mask(), mask)
            # A nested scope without its own batch keeps it.
            with fc.forward_context(fc.ForwardContext(attn_backend=None)):
                self.assertIs(fc.current_moe_row_mask(), mask)
            # A nested forward with its own batch (the draft) has its own.
            other = SimpleNamespace(moe_num_token_non_padded=None)
            with fc.forward_context(ctx, forward_batch=other):
                self.assertIsNone(fc.current_moe_row_mask())
        self.assertIsNone(fc.current_moe_row_mask())

    def test_plain_layer_routes(self):
        ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int32)
        weights = torch.rand(3, 2)
        logits = torch.zeros(3, 4)
        mask = torch.tensor([1, 0, 1], dtype=torch.int32)
        fb = SimpleNamespace(moe_num_token_non_padded=None, moe_verify_row_mask=mask)
        experts = SimpleNamespace(
            num_experts=4, _dsv4_mxfp4_backend="flashinfer_cutlass_sm120"
        )
        with fc.forward_context(fc.ForwardContext(attn_backend=None), forward_batch=fb):
            for out in (
                StandardTopKOutput(weights, ids, logits),
                StandardTopKOutputPacked(weights, ids, logits, torch.zeros_like(ids)),
            ):
                masked = row_mask_module.mask_plain_routes(experts, out)
                self.assertIsInstance(masked, StandardTopKOutput)
                self.assertEqual(masked.topk_ids.tolist(), [[0, 1], [4, 4], [1, 0]])
                self.assertEqual(masked.topk_ids.dtype, torch.int32)
                self.assertIs(masked.topk_weights, weights)
            # A layer that masks in its own route split is left alone.
            own = SimpleNamespace(masks_rows_in_route_split=True)
            out = StandardTopKOutput(weights, ids, logits)
            self.assertIs(row_mask_module.mask_plain_routes(own, out), out)
            for refused in (
                SimpleNamespace(num_experts=4, _mxfp4_backend="marlin"),
                SimpleNamespace(num_experts=4),
                SimpleNamespace(
                    num_experts=4,
                    _dsv4_mxfp4_backend="flashinfer_cutlass_sm120",
                    moe_ep_size=2,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    row_mask_module.mask_plain_routes(refused, out)
            short = SimpleNamespace(
                moe_num_token_non_padded=None, moe_verify_row_mask=mask[:2]
            )
            with fc.forward_context(
                fc.ForwardContext(attn_backend=None), forward_batch=short
            ):
                with self.assertRaises(RuntimeError):
                    row_mask_module.mask_plain_routes(experts, out)


def _verify_graph_runner(*, glp_alpha_per_request, max_bs=4):
    """A DSpark target verify DecodeCudaGraphRunner on CPU with the slots its
    __init__ adds for the two per-row inputs, in its order: the GLP alpha rows
    (its block for --glp-alpha-per-request, as written there), then the row mask
    (the runner's own _masks_verify_rows and moe_verify_row_mask_slot). The other
    slots are build_decode_registry's; the rest is what capture_prepare and
    load_batch read."""
    runner = graph_runner.DecodeCudaGraphRunner.__new__(
        graph_runner.DecodeCudaGraphRunner
    )
    runner.model_runner = SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
        attn_tp_sequence_sharded=lambda num_tokens: False,
        lora_manager=None,
        runner_ids=SimpleNamespace(bind_runner_slots=lambda fb: None),
        hisparse_coordinator=None,
    )
    runner.device = torch.device("cpu")
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner.captured_req_width = WIDTH
    runner.capture_bs = [1, 2, max_bs]
    runner.max_num_token = max_bs * WIDTH
    registry = build_decode_registry(
        device=runner.device,
        max_bs=max_bs,
        max_num_token=runner.max_num_token,
        seq_len_fill_value=1,
        cache_loc_dtype=torch.int64,
        share_pool=False,
    )
    runner.buffer_registry = registry
    with get_context().override_server_args(
        glp_alpha_per_request=glp_alpha_per_request
    ):
        if (
            get_serving().glp_alpha_per_request
            and not runner.model_runner.is_draft_worker
        ):
            runner._glp_alpha_slot = registry.register_slot(
                GraphSlot(
                    "glp_alpha_rows",
                    lambda _bs, tokens: (tokens,),
                    torch.float32,
                    axis="tokens",
                    padding_policy=PaddingPolicy.ZERO,
                ),
                bind=torch.zeros(runner.max_num_token, dtype=torch.float32),
            )
    if runner._masks_verify_rows():
        registry.register_slot(moe_verify_row_mask_slot())
    runner.buffers = SimpleNamespace(
        input_ids=registry.get_slot("input_ids").buffer,
        positions=registry.get_slot("positions").buffer,
        next_token_logits_buffer=torch.zeros(runner.max_num_token, 4),
        aux_hidden_states=None,
        rids_int=None,
        bootstrap_room_ids_int=None,
        num_token_non_padded=None,
        pp_proxy_tensors=None,
        global_num_tokens_gpu=registry.get_slot("global_num_tokens_gpu").buffer,
        global_num_tokens_for_logprob_gpu=registry.get_slot(
            "global_num_tokens_for_logprob_gpu"
        ).buffer,
        ngram_embedding_info=None,
    )
    runner.attn_backend = None
    runner.get_spec_info = lambda num_tokens, ragged_verify_layout=None: None
    runner.enable_prefill_cp = runner.ragged_verify_mode = False
    runner.require_mlp_tp_gather = runner.require_attn_tp_gather = False
    runner.pp_size = 1
    runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
    runner.dllm_uses_input_embeds = runner.enable_two_batch_overlap = False
    runner.dllm_input_preparation = runner.attention_graph_variants = None
    runner.attention_variant_sizes = runner._verify_logits_processor = None
    runner.record_nolora_graph = runner._restage_skip = False
    runner._skips_prefilled_inputs = False
    runner._early_verify_metadata = runner._prefilled_inputs = None
    runner._draft_metadata_glue = runner._target_metadata_glue = False
    runner._init_replay_attn_metadata = lambda forward_batch, **kwargs: None
    return runner


def _verify_batch(bs, alpha_rows, row_mask):
    """A target verify batch as load_batch reads it, not yet pre-planned."""
    tokens = bs * WIDTH
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        input_ids=torch.arange(100, 100 + tokens),
        positions=torch.arange(tokens),
        input_embeds=None,
        glp_alpha_rows=alpha_rows,
        moe_verify_row_mask=row_mask,
        needs_forward_metadata_init=lambda: True,
    )


class TestVerifySlotsBesideGlpAlpha(_ScheduleCase):
    """The row mask beside the per-request GLP alpha rows (--glp-alpha-per-request)
    in one target verify runner: the capture batch reads both slots, a padded replay
    and a pre-planned replay fill each with its own rows, and each slot is the same
    with the other feature on or off."""

    SURVIVAL = 0.08
    # Three requests of WIDTH rows each; the third verifies its anchor row only.
    ALPHA = [0.0] * WIDTH + [2.0] * WIDTH + [0.25] * WIDTH
    LENS = [2, WIDTH, 1]

    def _runner(self, *, glp=True, schedule=True):
        if schedule:
            return _verify_graph_runner(glp_alpha_per_request=glp)
        with envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(0.0):
            vs.forget_threshold()
            try:
                return _verify_graph_runner(glp_alpha_per_request=glp)
            finally:
                vs.forget_threshold()

    def _capture(self, runner, bs):
        with mock.patch.object(
            graph_runner, "enable_num_token_non_padded", return_value=False
        ):
            forward_batch, _, _ = runner.capture_prepare(bs)
        return forward_batch

    def _mask(self):
        return vs.verify_row_mask(torch.tensor(self.LENS), 3, WIDTH, "cpu")

    def test_the_capture_batch_reads_both_slots(self):
        runner = self._runner()
        alpha = runner.buffer_registry.get_slot("glp_alpha_rows").buffer
        mask = runner.buffer_registry.get_slot("moe_verify_row_mask").buffer
        # What earlier replays left behind; a capture (also mid-serving) resets it.
        alpha.fill_(9.0)
        mask.fill_(7)
        fb = self._capture(runner, 2)
        tokens = 2 * WIDTH
        self.assertEqual(fb.glp_alpha_rows.data_ptr(), alpha.data_ptr())
        self.assertEqual(tuple(fb.glp_alpha_rows.shape), (tokens,))
        self.assertEqual(alpha.tolist(), [0.0] * alpha.numel())
        self.assertTrue(fb.glp_any_steered)
        self.assertEqual(fb.moe_verify_row_mask.data_ptr(), mask.data_ptr())
        self.assertEqual(fb.moe_verify_row_mask.tolist(), [1] * tokens)
        # Two storages: neither slot's fill can reach the other.
        self.assertNotEqual(
            alpha.untyped_storage().data_ptr(), mask.untyped_storage().data_ptr()
        )

    def test_each_replay_fills_each_slot_with_its_own_rows(self):
        runner = self._runner()
        alpha = runner.buffer_registry.get_slot("glp_alpha_rows").buffer
        mask = runner.buffer_registry.get_slot("moe_verify_row_mask").buffer
        self._capture(runner, 4)
        rows, row_mask = torch.tensor(self.ALPHA), self._mask()
        pad_alpha, pad_mask = [0.0] * WIDTH, [0] * WIDTH
        fb = _verify_batch(3, rows, row_mask)
        alpha.fill_(9.0)
        mask.fill_(7)
        # The plan-time load: three requests replay the graph of four.
        runner.load_batch(fb)
        self.assertEqual((runner.raw_bs, runner.bs, runner.raw_num_token), (3, 4, 18))
        self.assertEqual(alpha[:24].tolist(), self.ALPHA + pad_alpha)
        self.assertEqual(mask[:24].tolist(), row_mask.tolist() + pad_mask)
        # Its pre-planned replay copies the alpha rows again; the mask keeps the
        # plan-time fill of the same batch.
        fb.needs_forward_metadata_init = lambda: False
        alpha[:18].fill_(9.0)
        runner.load_batch(fb)
        self.assertEqual(alpha[:24].tolist(), self.ALPHA + pad_alpha)
        self.assertEqual(mask[:24].tolist(), row_mask.tolist() + pad_mask)
        # A verify without a mask runs every row; the alpha rows are unaffected.
        runner.load_batch(_verify_batch(3, rows, None))
        self.assertEqual(mask[:24].tolist(), [1] * 18 + pad_mask)
        self.assertEqual(alpha[:24].tolist(), self.ALPHA + pad_alpha)

    def _run(self, *, glp, schedule):
        """The capture, a plan-time load of three requests and that batch's
        pre-planned replay, with each feature on or off: what each step leaves in
        the runner's buffers, and what the capture batch carries."""
        runner = self._runner(glp=glp, schedule=schedule)
        registry = runner.buffer_registry
        seen = {"slots": set(registry.slot_names())}

        def snapshot(step):
            for name in ("glp_alpha_rows", "moe_verify_row_mask", "input_ids"):
                if registry.has_slot(name):
                    seen[step, name] = registry.get_slot(name).buffer.tolist()
            seen[step, "positions"] = registry.get_slot("positions").buffer.tolist()

        capture = self._capture(runner, 4)
        seen["capture"] = (
            None if capture.glp_alpha_rows is None else capture.glp_alpha_rows.tolist(),
            capture.glp_any_steered,
            (
                None
                if capture.moe_verify_row_mask is None
                else capture.moe_verify_row_mask.tolist()
            ),
        )
        snapshot("capture")
        # A batch carries the rows of a feature that is on, as init_new and
        # prepare_for_verify build it.
        fb = _verify_batch(
            3,
            torch.tensor(self.ALPHA) if glp else None,
            self._mask() if schedule else None,
        )
        runner.load_batch(fb)
        snapshot("plan")
        fb.needs_forward_metadata_init = lambda: False
        runner.load_batch(fb)
        snapshot("replay")
        return seen

    def test_each_feature_is_the_same_with_the_other_off(self):
        both = self._run(glp=True, schedule=True)
        glp_only = self._run(glp=True, schedule=False)
        schedule_only = self._run(glp=False, schedule=True)
        neither = self._run(glp=False, schedule=False)
        base = neither["slots"]
        self.assertFalse({"glp_alpha_rows", "moe_verify_row_mask"} & base)
        self.assertEqual(
            both["slots"], base | {"glp_alpha_rows", "moe_verify_row_mask"}
        )
        self.assertEqual(glp_only["slots"], base | {"glp_alpha_rows"})
        self.assertEqual(schedule_only["slots"], base | {"moe_verify_row_mask"})
        tokens = 4 * WIDTH
        self.assertEqual(both["capture"], ([0.0] * tokens, True, [1] * tokens))
        self.assertEqual(glp_only["capture"], ([0.0] * tokens, True, None))
        self.assertEqual(schedule_only["capture"], (None, False, [1] * tokens))
        self.assertEqual(neither["capture"], (None, False, None))
        for step in ("capture", "plan", "replay"):
            self.assertEqual(
                glp_only[step, "glp_alpha_rows"], both[step, "glp_alpha_rows"], step
            )
            self.assertEqual(
                schedule_only[step, "moe_verify_row_mask"],
                both[step, "moe_verify_row_mask"],
                step,
            )
            for other in (glp_only, schedule_only, neither):
                for name in ("input_ids", "positions"):
                    self.assertEqual(other[step, name], both[step, name], (step, name))


class TestHotColdLayerMask(_ScheduleCase):
    """The two-tier layer of the toy checkpoint, with float stand-ins for the kernels."""

    SURVIVAL = 0.08

    def setUp(self):
        super().setUp()
        from sglang.srt.layers.quantization import hotcold as hc
        from sglang.srt.models import deepseek_v2
        from sglang.test import hotcold_fixture as fx

        self.fx = fx
        self.enterContext(fx.cpu_moe_context())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=fx.toy_hf_config(self.ckpt.path),
            layer_id=0,
            quant_config=fx.fp4_quant_config(),
            prefix="model.layers.0.mlp",
            is_deepseek_v4=True,
        )
        fx.load_toy_layer(self.mlp, self.ckpt, 0)
        self.experts = self.mlp.experts
        self.experts.hot.forward = fx.hot_reference_forward(self.experts.hot)
        self.hc = hc
        self.gen = torch.Generator().manual_seed(4)

    def _routes(self, tokens, top_k=3):
        ids = torch.stack(
            [torch.randperm(8, generator=self.gen)[:top_k] for _ in range(tokens)]
        ).to(torch.int32)
        return ids, torch.rand(tokens, top_k, generator=self.gen) + 0.1

    def _layer(self, x, ids, weights, mask):
        fb = SimpleNamespace(moe_num_token_non_padded=None, moe_verify_row_mask=mask)
        out = StandardTopKOutput(weights, ids, torch.zeros(x.shape[0], 8))
        with fc.forward_context(fc.ForwardContext(attn_backend=None), forward_batch=fb):
            with torch.no_grad():
                return self.experts(x, out)

    def test_a_masked_row_takes_the_padded_route_in_the_split(self):
        ids = torch.tensor([[0, 1, 7], [7, 6, 5], [2, 3, -1]], dtype=torch.int32)
        mask = torch.tensor([1, 0, 1], dtype=torch.int32)
        masked = self.experts.mask_rows(ids, mask)
        self.assertEqual(masked.tolist(), [[0, 1, 7], [-1, -1, -1], [2, 3, -1]])
        self.assertEqual(masked.dtype, torch.int32)
        self.assertTrue(masked.is_contiguous())
        hot, cold = self.experts.split_routes(masked)
        free_hot, free_cold = self.experts.split_routes(ids)
        self.assertEqual(hot[1].tolist(), [self.experts.num_hot] * 3)
        self.assertEqual(cold[1].tolist(), [-1] * 3)
        for row in (0, 2):
            self.assertEqual(hot[row].tolist(), free_hot[row].tolist())
            self.assertEqual(cold[row].tolist(), free_cold[row].tolist())
        self.assertEqual(hot.dtype, torch.int32)
        # Every dtype of ids the router may hand in.
        masked64 = self.experts.mask_rows(ids.long(), mask)
        self.assertEqual(masked64.dtype, torch.int64)
        self.assertEqual(masked64.tolist(), masked.tolist())
        ones = torch.ones(3, dtype=torch.int32)
        self.assertTrue(torch.equal(self.experts.mask_rows(ids, ones), ids))
        with self.assertRaises(RuntimeError):
            self.experts.mask_rows(ids, mask[:2])

    def test_every_form_of_the_split_reads_the_masked_ids(self):
        """On CUDA the split is one launch, and on a decode batch that launch also
        groups the cold routes (the kernels' seams of test_dsv41_hotcold_moe.py). A
        masked forward hands each form ids with the padded route on its masked rows,
        so a masked row has no hot slot, no cold slot and no cold group."""
        from sglang.srt.layers.moe import hotcold as hc_moe

        tables = (self.experts.remap_hot, self.experts.remap_cold)
        marks = tuple(torch.zeros(1, dtype=torch.int32) for _ in range(3))
        calls = []

        def gathers(topk_ids):
            index = (topk_ids + 1).reshape(-1).long()
            return tuple(t[index].view(topk_ids.shape) for t in tables)

        def split(topk_ids, remap_hot, remap_cold):
            calls.append(("split", topk_ids))
            return gathers(topk_ids)

        def split_align(topk_ids, remap_hot, remap_cold, num_cold, block):
            calls.append(("split align", topk_ids))
            return gathers(topk_ids) + marks

        builds = frozenset({22, 144, 145})
        lens = [3, WIDTH, 1]
        tokens = len(lens) * WIDTH
        x = torch.randn(tokens, 256, generator=self.gen)
        ids, weights = self._routes(tokens)
        mask = vs.verify_row_mask(torch.tensor(lens), len(lens), WIDTH, "cpu")
        keep = mask.bool()
        want_ids = torch.where(keep.view(-1, 1), ids, -1)
        free = self._layer(x, ids, weights, None)
        with (
            mock.patch.object(hc_moe, "_split_kernel_serves", lambda t: True),
            mock.patch.object(hc_moe, "_split_kernel", split),
            mock.patch.object(hc_moe, "_split_align_kernel", split_align),
            mock.patch.object(self.hc, "_mmq_types", lambda: builds),
            mock.patch.object(self.hc, "_split_align_serves", lambda n, block: True),
        ):
            for switches, form in (
                ({}, "split align"),
                ({"SGLANG_HOTCOLD_DECODE_ALIGN": "0"}, "split"),
                ({"SGLANG_HOTCOLD_SPLIT_KERNEL": "0"}, None),
            ):
                with mock.patch.dict(os.environ, switches):
                    calls.clear()
                    same = self._layer(x, ids, weights, None)
                    # No mask: the split reads the router's own tensor.
                    self.assertEqual([c[0] for c in calls], [form] if form else [])
                    for _, seen in calls:
                        self.assertIs(seen, ids)
                    self.assertTrue(torch.equal(same, free), form)
                    calls.clear()
                    dirty = x.clone()
                    dirty[~keep] = float("nan")
                    got = self._layer(dirty, ids, weights, mask)
                    self.assertEqual([c[0] for c in calls], [form] if form else [])
                    for _, seen in calls:
                        self.assertTrue(torch.equal(seen, want_ids), form)
                        self.assertEqual(seen.dtype, torch.int32)
                        self.assertTrue(seen.is_contiguous())
                    self.assertTrue(torch.equal(got[keep], free[keep]), form)
                    self.assertTrue(bool((got[~keep] == 0).all()), form)

    def test_kept_rows_keep_their_bits_and_masked_rows_give_zero(self):
        tokens = 2 * WIDTH
        x = torch.randn(tokens, 256, generator=self.gen)
        ids, weights = self._routes(tokens)
        free = self._layer(x, ids, weights, None)
        for lens in ([1, WIDTH], [3, 4], [WIDTH, WIDTH], [1, 1]):
            mask = vs.verify_row_mask(torch.tensor(lens), 2, WIDTH, "cpu")
            keep = mask.bool()
            # A masked row's input is nobody's: fill it with NaN.
            dirty = x.clone()
            dirty[~keep] = float("nan")
            got = self._layer(dirty, ids, weights, mask)
            self.assertTrue(torch.equal(got[keep], free[keep]), lens)
            self.assertTrue(bool((got[~keep] == 0).all()), lens)
            self.assertTrue(bool(torch.isfinite(got).all()), lens)

    def test_the_block_masks_a_plain_layer_through_the_non_local_id(self):
        """DeepseekV2MoE's own call sites: with a plain layer under it, the ids it
        hands on name no expert on a masked row."""
        seen = {}

        class Plain(torch.nn.Module):
            num_experts = 8
            _dsv4_mxfp4_backend = "flashinfer_cutlass_sm120"

            def forward(self, hidden_states, topk_output, pre_quant_input=None):
                seen["ids"] = topk_output.topk_ids.clone()
                return torch.zeros_like(hidden_states)

        ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
        out = StandardTopKOutput(torch.ones(2, 3), ids, torch.zeros(2, 8))
        mask = torch.tensor([0, 1], dtype=torch.int32)
        fb = SimpleNamespace(moe_num_token_non_padded=None, moe_verify_row_mask=mask)
        plain = Plain()
        with fc.forward_context(fc.ForwardContext(attn_backend=None), forward_batch=fb):
            masked = row_mask_module.mask_plain_routes(plain, out)
            plain(torch.zeros(2, 4), masked)
        self.assertEqual(seen["ids"].tolist(), [[8, 8, 8], [3, 4, 5]])


class TestBlockCallSites(_ScheduleCase):
    """DeepseekV2MoE.forward_normal: the mask reaches both kinds of layer under it."""

    SURVIVAL = 0.08

    def setUp(self):
        super().setUp()
        from sglang.test import hotcold_fixture as fx

        self.fx = fx
        self.enterContext(fx.cpu_moe_context())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))
        self.mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32)

    def _block(self, hot_cold):
        from sglang.srt.models import deepseek_v2

        return deepseek_v2.DeepseekV2MoE(
            config=self.fx.toy_hf_config(self.ckpt.path if hot_cold else None),
            layer_id=0,
            quant_config=self.fx.fp4_quant_config(),
            prefix="model.layers.0.mlp",
            is_deepseek_v4=True,
        )

    def _forward(self, block, mask):
        fb = SimpleNamespace(moe_num_token_non_padded=None, moe_verify_row_mask=mask)
        x = torch.randn(4, 256, generator=torch.Generator().manual_seed(8))
        with fc.forward_context(fc.ForwardContext(attn_backend=None), forward_batch=fb):
            with torch.no_grad():
                block.forward_normal(x, skip_shared_experts=True)

    def test_a_plain_layer_gets_the_non_local_id(self):
        block = self._block(hot_cold=False)
        self.assertFalse(getattr(block.experts, "masks_rows_in_route_split", False))
        block.experts._dsv4_mxfp4_backend = "flashinfer_cutlass_sm120"
        seen = []

        def experts(hidden_states, topk_output, pre_quant_input=None):
            seen.append(topk_output.topk_ids.clone())
            return torch.zeros_like(hidden_states)

        block.experts.forward = experts
        self._forward(block, None)
        self._forward(block, self.mask)
        free, masked = seen
        num = block.experts.num_experts
        self.assertTrue(bool((free < num).all()))
        for row, keep in enumerate(self.mask.tolist()):
            want = free[row].tolist() if keep else [num] * free.shape[1]
            self.assertEqual(masked[row].tolist(), want)

    def test_a_hot_cold_layer_masks_the_ids_its_route_split_reads(self):
        block = self._block(hot_cold=True)
        self.assertTrue(block.experts.masks_rows_in_route_split)
        seen, handed = [], []
        split = block.experts.split_routes
        experts = block.experts.forward

        def spy(topk_ids):
            seen.append(topk_ids.clone())
            return split(topk_ids)

        def layer(hidden_states, topk_output, pre_quant_input=None):
            handed.append(topk_output.topk_ids.clone())
            return experts(hidden_states, topk_output)

        block.experts.split_routes = spy
        block.experts.forward = layer
        block.experts.hot.forward = lambda h, t, pre_quant_input=None: torch.zeros_like(
            h
        )
        block.experts.cold.forward = lambda h, w, ids, **kw: torch.zeros_like(h)
        self._forward(block, None)
        self._forward(block, self.mask)
        free_ids, ids = seen
        # The block hands the layer the router's ids both times: the layer masks them.
        self.assertTrue(torch.equal(handed[0], handed[1]))
        self.assertTrue(torch.equal(free_ids, handed[0]))
        self.assertTrue(bool((free_ids >= 0).all()))
        for row, keep in enumerate(self.mask.tolist()):
            want = free_ids[row].tolist() if keep else [-1] * free_ids.shape[1]
            self.assertEqual(ids[row].tolist(), want)


class TestLaunch(CustomTestCase):
    def _refusal(self):
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
                dspark_target_layer_ids=[37, 38, 39], num_hidden_layers=40
            ),
        )
        with mock.patch.dict("os.environ", {"SGLANG_PP_LAYER_PARTITION": "14,13,13"}):
            return launch.pp_dspark_refusal(cfg, hf)

    def test_refusals(self):
        with (
            envs.SGLANG_ENABLE_PP_SPEC.override(True),
            envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(0.0),
        ):
            self.assertIsNone(self._refusal())
            with envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(0.08):
                self.assertIsNone(self._refusal())
                with envs.SGLANG_SIMULATE_ACC_LEN.override(3.0):
                    self.assertIn("SGLANG_SIMULATE_ACC_LEN", self._refusal())
                # The schedule reads the trained head, which that switch leaves out.
                with envs.SGLANG_DSPARK_USE_ONLINE_ESTIMATOR.override(True):
                    self.assertIn("SGLANG_DSPARK_USE_ONLINE_ESTIMATOR", self._refusal())
            with envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.override(1.2):
                self.assertIn("0 (off) to 1", self._refusal())
            with envs.SGLANG_SIMULATE_ACC_LEN.override(3.0):
                self.assertIsNone(self._refusal())
            with envs.SGLANG_DSPARK_USE_ONLINE_ESTIMATOR.override(True):
                self.assertIsNone(self._refusal())


if __name__ == "__main__":
    unittest.main()
