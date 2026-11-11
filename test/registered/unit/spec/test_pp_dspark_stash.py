"""Draft logits between the tail draft and the verify that reads them."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
from triton.runtime.jit import JITFunction

from sglang.kernels.ops.speculative.reject_sampling import (
    speculative_sampling_classic_kernel,
)
from sglang.srt.speculative.dspark_components import dspark_pp_round as round_module
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftProposal,
)
from sglang.srt.speculative.dspark_components.dspark_pp_round import (
    DSparkPPLastStage,
    _RoundOutcome,
)
from sglang.srt.speculative.dspark_components.dspark_pp_stash import DsparkPPStash
from sglang.srt.speculative.dspark_components.dspark_switch import (
    NEW_REQUEST_TOKENS_PER_VERIFY,
    DsparkPPSwitch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, GAMMA, VOCAB = 2, 3, 11


def _triton_runs() -> bool:
    """A GPU, or the interpreter from the process's start (TRITON_INTERPRET=1): a kernel
    is bound to one or the other when its module is imported."""
    if torch.cuda.device_count() > 0:
        return True
    return not isinstance(speculative_sampling_classic_kernel, JITFunction)


TRITON_OK = _triton_runs()


def _block(seed=0, bs=BS):
    return torch.randn(
        (bs, GAMMA, VOCAB), generator=torch.Generator().manual_seed(seed)
    )


def _tokens(bs=BS):
    return torch.arange(bs * GAMMA, dtype=torch.int64).view(bs, GAMMA) % VOCAB


class TestStash(CustomTestCase):
    def test_one_stream_copies_nothing(self):
        """The verify takes its rows before the next draft: no spill, the static buffer."""
        stash = DsparkPPStash(capacity=9)
        static = _block()
        stash.put(rids=["a", "b"], seq_lens=[5, 9], logits=static, borrowed=True)
        got = stash.take(
            rids=["a", "b"], seq_lens=[5, 9], draft_tokens=_tokens(), vocab=VOCAB
        )
        self.assertEqual(stash.num_spilled_rows, 0)
        self.assertEqual(got.data_ptr(), static.data_ptr())
        self.assertEqual(len(stash), 0)
        stash.spill_borrowed()
        self.assertEqual(stash.num_spilled_rows, 0)

    def test_another_drafts_overwrite_is_spilled_around(self):
        """Stash returns the logits of the chain they drew, by (rid, seq_len)."""
        stash = DsparkPPStash(capacity=9)
        static = _block(1)
        want = static.clone()
        stash.put(rids=["a", "b"], seq_lens=[5, 9], logits=static, borrowed=True)
        stash.spill_borrowed()
        self.assertEqual(stash.num_spilled_rows, 2)
        static.fill_(float("nan"))  # the next micro-batch's draft
        got = stash.take(
            rids=["b", "a"], seq_lens=[9, 5], draft_tokens=_tokens(), vocab=VOCAB
        )
        self.assertTrue(torch.equal(got[0], want[1]))
        self.assertTrue(torch.equal(got[1], want[0]))

    def test_composition_change_reads_by_rid(self):
        stash = DsparkPPStash(capacity=9)
        logits = _block(2, bs=3)
        stash.put(
            rids=["a", "b", "c"], seq_lens=[1, 2, 3], logits=logits, borrowed=False
        )
        got = stash.take(
            rids=["c", "a"], seq_lens=[3, 1], draft_tokens=_tokens(), vocab=VOCAB
        )
        self.assertTrue(torch.equal(got[0], logits[2]))
        self.assertTrue(torch.equal(got[1], logits[0]))

    def test_miss_is_a_one_hot_at_the_rows_tokens(self):
        """A degenerate row, or a chain whose seq_len moved: _draft_probs reads a one-hot,
        which the rejection accepts at the target's probability."""
        stash = DsparkPPStash(capacity=9)
        stash.put(rids=["a"], seq_lens=[5], logits=_block(bs=1), borrowed=False)
        tokens = _tokens()
        got = stash.take(
            rids=["a", "b"], seq_lens=[6, 7], draft_tokens=tokens, vocab=VOCAB
        )
        self.assertEqual(stash.num_missed_rows, 2)
        probs = dspark_verify._draft_probs(
            DraftBlockResult(
                draft_tokens=tokens,
                corrected_logits=got,
                greedy_mask=torch.zeros(BS, dtype=torch.bool),
                temperatures=torch.tensor([1.0, 1e-5]),
                greedy_draft=False,
            ),
            VOCAB,
            "cpu",
        )
        want = torch.nn.functional.one_hot(tokens, VOCAB).float()
        self.assertTrue(torch.equal(probs, want))

    def test_mixture_of_hit_and_miss(self):
        stash = DsparkPPStash(capacity=9)
        logits = _block(3, bs=1)
        stash.put(rids=["a"], seq_lens=[5], logits=logits, borrowed=False)
        got = stash.take(
            rids=["a", "b"], seq_lens=[5, 7], draft_tokens=_tokens(), vocab=VOCAB
        )
        self.assertTrue(torch.equal(got[0], logits[0]))
        self.assertEqual(stash.num_missed_rows, 1)
        self.assertEqual(got[1].argmax(dim=-1).tolist(), _tokens()[1].tolist())

    def test_abort_and_flush_clear_rows(self):
        """Invariant 34."""
        stash = DsparkPPStash(capacity=9)
        stash.put(rids=["a", "b"], seq_lens=[5, 9], logits=_block(), borrowed=True)
        stash.drop("a")
        self.assertEqual(len(stash), 1)
        stash.clear()
        self.assertEqual(len(stash), 0)
        stash.spill_borrowed()
        self.assertEqual(stash.num_spilled_rows, 0)

    def test_capacity_bounds_memory_and_oldest_goes_first(self):
        stash = DsparkPPStash(capacity=2)
        for i, rid in enumerate("abc"):
            stash.put(rids=[rid], seq_lens=[i], logits=_block(bs=1), borrowed=False)
        self.assertEqual(len(stash), 2)
        got = stash.take(rids=["a"], seq_lens=[0], draft_tokens=_tokens(1), vocab=VOCAB)
        self.assertEqual(stash.num_missed_rows, 1)
        self.assertEqual(got.shape, (1, GAMMA, VOCAB))

    def test_a_block_without_logits_drops_the_old_rows(self):
        stash = DsparkPPStash(capacity=9)
        stash.put(rids=["a"], seq_lens=[1], logits=_block(bs=1), borrowed=False)
        stash.put(rids=["a"], seq_lens=[2], logits=None, borrowed=False)
        self.assertEqual(len(stash), 0)


class _Proposer:
    """The sampler's graph output: one static buffer, overwritten by each draft."""

    def __init__(self):
        self.static = torch.zeros((BS, GAMMA, VOCAB))
        self.tokens = torch.zeros((BS, GAMMA), dtype=torch.int64)
        self.drafts = 0

    def propose(
        self,
        *,
        batch,
        draft_input,
        verify_window,
        bs,
        device,
        target_model,
        sampling_info,
    ):
        self.drafts += 1
        self.static.fill_(float(self.drafts))
        return DraftProposal(
            draft_block_ids=self.tokens,
            draft_block=DraftBlockResult(
                draft_tokens=self.tokens,
                corrected_logits=self.static,
                greedy_mask=torch.zeros(BS, dtype=torch.bool),
                temperatures=torch.ones(BS),
                greedy_draft=False,
            ),
            folded=True,
        )


class TestLastStageKeepsTheLogitsOfEachChain(CustomTestCase):
    """The round's own calls, in the order two micro-batches make them on one stage."""

    def _stage(self):
        worker = SimpleNamespace(
            verify_num_draft_tokens=GAMMA + 1,
            gamma=GAMMA,
            device="cpu",
            _draft_context=nullcontext,
            _proposer=_Proposer(),
            _greedy_draft=False,
            target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=None)),
        )
        stage = DSparkPPLastStage(
            worker,
            stash=DsparkPPStash(capacity=9),
            switch=DsparkPPSwitch(mode="on"),
        )
        stage._window = lambda batch, bs: None
        return stage

    def _tail(self, stage, rids, seq_lens):
        lens = torch.tensor(seq_lens, dtype=torch.int64)
        outcome = _RoundOutcome(
            logits_output=None,
            next_token_ids=torch.zeros(BS, dtype=torch.int64),
            commit_lens=torch.ones(BS, dtype=torch.int32),
            new_seq_lens=lens,
            bonus=torch.zeros(BS, dtype=torch.int64),
            can_run_graph=True,
            stride=1,
        )
        batch = SimpleNamespace(
            seq_lens=None,
            seq_lens_cpu=None,
            seq_lens_sum=None,
            input_ids=None,
            spec_info=None,
        )
        return stage._tail_draft(
            batch, outcome, rids, SimpleNamespace(is_all_greedy=False)
        )

    def _read(self, stage, rids, seq_lens):
        params = SimpleNamespace(greedy_mask=None, temperatures=None)
        with mock.patch.object(
            round_module, "resolve_draft_params", return_value=params
        ):
            return stage._draft_block(
                rids,
                torch.tensor(seq_lens, dtype=torch.int64),
                _tokens(),
                SimpleNamespace(is_all_greedy=False),
                torch.zeros(1, VOCAB),
            ).corrected_logits

    def test_a_second_micro_batch_draft_does_not_reach_the_first_ones_logits(self):
        """Invariant 20: the sampler reuses one buffer, the first chain's verify comes later."""
        stage = self._stage()
        self._tail(stage, ["a", "b"], [5, 9])
        self._tail(stage, ["c", "d"], [7, 3])
        self.assertEqual(stage.worker._proposer.drafts, 2)
        self.assertEqual(stage.stash.num_spilled_rows, 2)
        self.assertTrue(bool((self._read(stage, ["a", "b"], [5, 9]) == 1.0).all()))
        self.assertTrue(bool((self._read(stage, ["c", "d"], [7, 3]) == 2.0).all()))

    def test_one_micro_batch_copies_nothing(self):
        """One stream reads the static buffer before anything overwrites it."""
        stage = self._stage()
        self._tail(stage, ["a", "b"], [5, 9])
        got = self._read(stage, ["a", "b"], [5, 9])
        self.assertEqual(stage.stash.num_spilled_rows, 0)
        self.assertEqual(got.data_ptr(), stage.worker._proposer.static.data_ptr())
        self._tail(stage, ["a", "b"], [8, 11])
        self.assertEqual(stage.stash.num_spilled_rows, 0)

    def test_the_anchors_reach_the_embedding_before_the_draft_runs(self):
        """The host route's staged buffer holds this round's bonus when the draft reads it."""
        stage = self._stage()
        order = []
        stage.embedding = SimpleNamespace(
            stage_anchors=lambda ids: order.append(("stage", ids.tolist()))
        )
        propose = stage.worker._proposer.propose

        def spy(**kwargs):
            order.append(("draft", None))
            return propose(**kwargs)

        stage.worker._proposer.propose = spy
        self._tail(stage, ["a", "b"], [5, 9])
        self.assertEqual(order, [("stage", [0, 0]), ("draft", None)])

    def test_a_finished_request_leaves_nothing_behind(self):
        """Invariant 34 through the worker's call."""
        stage = self._stage()
        self._tail(stage, ["a", "b"], [5, 9])
        stage.switch.note_verify(["a"], [3])
        stage.note_request_finished("a")
        self.assertEqual(len(stage.stash), 1)
        self.assertEqual(
            stage.switch.tokens_per_verify(["a"]), NEW_REQUEST_TOKENS_PER_VERIFY
        )
        stage.note_request_finished("b")
        self.assertEqual(len(stage.stash), 0)
        self._tail(stage, ["a", "b"], [5, 9])
        stage.clear()
        self.assertEqual(len(stage.stash), 0)


@unittest.skipUnless(TRITON_OK, "needs CUDA or TRITON_INTERPRET=1")
class TestSampledAcceptLaw(CustomTestCase):
    """The accept after the ring: the first token that leaves a verify follows the
    target's law whether the row found its draft's logits or fell back to the one-hot."""

    N = 6000
    V = 7

    def _first_token_counts(self, draft_logits_of_row, drafts):
        n, gamma, v = self.N, GAMMA, self.V
        torch.manual_seed(1234)
        target_row = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, 1.5])
        target_logits = target_row.repeat(n * (gamma + 1), 1)
        candidates = torch.cat([torch.zeros(n, 1, dtype=torch.int64), drafts], dim=1)
        block = DraftBlockResult(
            draft_tokens=drafts,
            corrected_logits=draft_logits_of_row,
            greedy_mask=torch.zeros(n, dtype=torch.bool),
            temperatures=torch.ones(n),
            greedy_draft=False,
        )
        info = SimpleNamespace(
            temperatures=torch.ones(n),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            top_ks=torch.full((n,), 100),
        )
        correct, bonus, _ = dspark_verify.accept_draft_tokens(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=block,
            sampling_info=info,
            draft_input=SimpleNamespace(max_top_k=v, uniform_top_k_value=v),
            gamma=gamma,
            verify_num_draft_tokens=gamma + 1,
        )
        first = torch.where(correct > 0, drafts[:, 0], bonus)
        return torch.bincount(first, minlength=v).float(), target_row

    def _chi_square(self, counts, target_row):
        expected = torch.softmax(target_row, dim=0) * counts.sum()
        return float(((counts - expected) ** 2 / expected).sum())

    def test_sampled_accept_with_the_drafts_logits_keeps_the_target_law(self):
        n, gamma = self.N, GAMMA
        torch.manual_seed(7)
        draft_row = torch.tensor([0.0, 0.5, 1.0, -1.0, 2.0, 0.0, -0.5])
        logits = draft_row.repeat(n, gamma, 1)
        drafts = torch.multinomial(
            torch.softmax(draft_row, 0), n * gamma, replacement=True
        ).view(n, gamma)
        counts, target = self._first_token_counts(logits, drafts)
        # Chi-square, 6 degrees of freedom: 27.9 is p = 1e-4.
        self.assertLess(self._chi_square(counts, target), 27.9)

    def test_miss_keeps_the_target_law(self):
        n, gamma = self.N, GAMMA
        drafts = torch.full((n, gamma), 3, dtype=torch.int64)
        stash = DsparkPPStash(capacity=1)
        logits = stash.take(
            rids=[str(i) for i in range(n)],
            seq_lens=[0] * n,
            draft_tokens=drafts,
            vocab=self.V,
        )
        counts, target = self._first_token_counts(logits, drafts)
        self.assertLess(self._chi_square(counts, target), 27.9)


if __name__ == "__main__":
    unittest.main()
