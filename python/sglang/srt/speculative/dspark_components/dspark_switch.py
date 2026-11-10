"""When DSpark under pipeline parallelism is worth its verify rounds.

A verify forward costs about twice a plain one and holds up every stage, so past a
number of concurrent requests drafting loses. The last stage decides at the tail draft,
per micro-batch: it drafts, or relays no chain and that micro-batch's next round is
plain on every stage. Every stage reads the same fact, so the stages cannot disagree.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Break-even tokens per verify by requests decoding, from the ring model of the pipeline
# (the plain and verify period at this box's measured stage times). A window's own
# figures replace them through SGLANG_PP_DSPARK_BREAKEVEN.
DEFAULT_BREAKEVEN_EXCHANGE_FIRST = {
    1: 1.97,
    2: 2.65,
    3: 2.63,
    6: 2.70,
    9: 2.76,
    25: 3.59,
}
DEFAULT_BREAKEVEN_LAUNCH_FIRST = {1: 1.97, 2: 2.06, 3: 2.14, 6: 2.21, 9: 2.27, 25: 2.81}

DRAFT_MARGIN = 1.1
RESUME_MARGIN = 1.25
PROBE_EVERY_ROUNDS = 32
NEW_REQUEST_TOKENS_PER_VERIFY = 3.0
EWMA_WEIGHT = 0.2


def parse_breakeven(
    text: Optional[str], default: Dict[int, float]
) -> List[Tuple[int, float]]:
    table = (
        default if not text else {int(k): float(v) for k, v in json.loads(text).items()}
    )
    if not table:
        raise ValueError("SGLANG_PP_DSPARK_BREAKEVEN is empty")
    return sorted(table.items())


def interpolate_breakeven(table: Sequence[Tuple[int, float]], requests: int) -> float:
    keys = [k for k, _ in table]
    i = bisect.bisect_left(keys, requests)
    if i == 0:
        return table[0][1]
    if i == len(table):
        return table[-1][1]
    (lo, lo_v), (hi, hi_v) = table[i - 1], table[i]
    return lo_v + (hi_v - lo_v) * (requests - lo) / (hi - lo)


@dataclass
class _MicroBatchState:
    drafting: bool = True
    rounds_off: int = 0


class DsparkPPSwitch:
    def __init__(
        self,
        *,
        mode: str = "auto",
        breakeven: Sequence[Tuple[int, float]] = (),
        max_verify_bs: int = 0,
    ) -> None:
        assert mode in ("auto", "on", "off"), mode
        self.mode = mode
        self.breakeven = list(breakeven) or sorted(
            DEFAULT_BREAKEVEN_EXCHANGE_FIRST.items()
        )
        self.max_verify_bs = int(max_verify_bs)
        self._tokens_per_verify: Dict[str, float] = {}
        self._micro_batches: Dict[int, _MicroBatchState] = {}
        self.num_drafted = 0
        self.num_skipped = 0
        self.num_probes = 0

    @classmethod
    def from_env(cls, envs) -> DsparkPPSwitch:
        default = (
            DEFAULT_BREAKEVEN_LAUNCH_FIRST
            if envs.SGLANG_PP_DSPARK_LAUNCH_FIRST.get()
            else DEFAULT_BREAKEVEN_EXCHANGE_FIRST
        )
        return cls(
            mode=envs.SGLANG_PP_DSPARK_SWITCH.get(),
            breakeven=parse_breakeven(envs.SGLANG_PP_DSPARK_BREAKEVEN.get(), default),
            max_verify_bs=envs.SGLANG_PP_DSPARK_VERIFY_MAX_BS.get(),
        )

    def note_verify(self, rids: Sequence[str], commit_lens: Sequence[int]) -> None:
        """A verify's committed tokens per request; a plain round says nothing."""
        for rid, tokens in zip(rids, commit_lens):
            old = self._tokens_per_verify.get(rid, NEW_REQUEST_TOKENS_PER_VERIFY)
            self._tokens_per_verify[rid] = old + EWMA_WEIGHT * (tokens - old)

    def forget(self, rid: str) -> None:
        self._tokens_per_verify.pop(rid, None)

    def clear(self) -> None:
        self._tokens_per_verify.clear()
        self._micro_batches.clear()

    def tokens_per_verify(self, rids: Sequence[str]) -> float:
        return sum(
            self._tokens_per_verify.get(rid, NEW_REQUEST_TOKENS_PER_VERIFY)
            for rid in rids
        ) / max(1, len(rids))

    def should_draft(
        self,
        *,
        mb_id: int,
        rids: Sequence[str],
        decoding_requests: int,
        return_logprob: bool = False,
    ) -> bool:
        """Whether to draft for this micro-batch's next round."""
        decision = self._decide(
            mb_id=mb_id,
            rids=rids,
            decoding_requests=decoding_requests,
            return_logprob=return_logprob,
        )
        if decision:
            self.num_drafted += 1
        else:
            self.num_skipped += 1
        return decision

    def _decide(self, *, mb_id, rids, decoding_requests, return_logprob) -> bool:
        # Logprob batches run plain, and a micro-batch above the verify graph list
        # would verify eagerly: neither is a choice the mode can change.
        if return_logprob or (0 < self.max_verify_bs < len(rids)):
            return False
        if self.mode != "auto":
            return self.mode == "on"
        state = self._micro_batches.setdefault(mb_id, _MicroBatchState())
        be = interpolate_breakeven(self.breakeven, max(1, decoding_requests))
        margin = DRAFT_MARGIN if state.drafting else RESUME_MARGIN
        state.drafting = self.tokens_per_verify(rids) >= margin * be
        if state.drafting:
            state.rounds_off = 0
            return True
        state.rounds_off += 1
        if state.rounds_off >= PROBE_EVERY_ROUNDS:
            # Off, the rate goes stale: one verify refreshes it.
            state.rounds_off = 0
            self.num_probes += 1
            return True
        return False
