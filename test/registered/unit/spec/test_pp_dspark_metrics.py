"""Counts under the relay: a round settles once per request on every stage."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid):
    return SimpleNamespace(
        rid=rid,
        is_retracted=False,
        grammar=None,
        finished=lambda: False,
        kv=SimpleNamespace(
            kv_committed_len=10, commit_decode=lambda n, s=None: None
        ),
        spec_verify_ct=0,
        spec_num_correct_drafts=0,
        hist=[],
        update_spec_correct_drafts_histogram=lambda n: None,
    )


def _settle(reqs, tokens, accept, stride):
    commits = {r.rid: [] for r in reqs}
    for r in reqs:
        r.kv = SimpleNamespace(commit_decode=commits[r.rid].append)
        r.update_spec_correct_drafts_histogram = r.hist.append
    processor = SimpleNamespace(
        metrics_reporter=SimpleNamespace(snapshot_spec_decode_metrics=lambda b, r: None),
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        advance_grammar_fsm=lambda r, b: None,
    )
    result = GenerationBatchResult(
        next_token_ids=torch.tensor(tokens, dtype=torch.int64),
        accept_lens=torch.tensor(accept, dtype=torch.int32),
        speculative_num_draft_tokens=6,
        speculative_output_stride=stride,
    )
    out = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
        processor, result, SimpleNamespace(reqs=reqs), [True] * len(reqs)
    )
    return out, commits, result


class TestRoundCounts(CustomTestCase):
    def test_verify_round(self):
        """Invariant 36: one verify per request, drafts accepted without the bonus."""
        reqs = [_req("a"), _req("b")]
        tokens = [1, 2, 3, 0, 0, 0, 7, 0, 0, 0, 0, 0]
        out, commits, result = _settle(reqs, tokens, [3, 1], 6)
        self.assertEqual(out, [[1, 2, 3], [7]])
        self.assertEqual(commits, {"a": [3], "b": [1]})
        self.assertEqual([r.spec_verify_ct for r in reqs], [1, 1])
        self.assertEqual([r.spec_num_correct_drafts for r in reqs], [2, 0])
        self.assertEqual(result.num_correct_drafts, 2)
        self.assertEqual(result.get_num_generated_tokens(2), 4)

    def test_plain_round_reports_accept_length_one_at_stride_one(self):
        """Invariant 4 and 36: the same keys, one token wide; no draft is counted."""
        reqs = [_req("a"), _req("b")]
        out, commits, result = _settle(reqs, [4, 5], [1, 1], 1)
        self.assertEqual(out, [[4], [5]])
        self.assertEqual(commits, {"a": [1], "b": [1]})
        self.assertEqual([r.spec_verify_ct for r in reqs], [1, 1])
        self.assertEqual(result.num_correct_drafts, 0)
        self.assertEqual(result.get_num_generated_tokens(2), 2)


if __name__ == "__main__":
    unittest.main()
