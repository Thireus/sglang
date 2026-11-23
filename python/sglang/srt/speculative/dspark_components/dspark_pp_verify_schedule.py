"""A verify length per request under the DSpark relay, from the draft's confidence head.

Round r verifies, for each request, the anchor row and the first drafts of the block
the last stage drafted at the tail of round r-1. How many: the drafts whose survival,
the running product of the confidence head's outputs along the block, is at least a
threshold (``SGLANG_PP_DSPARK_VERIFY_SURVIVAL``). The length rides the ring in the
chain's own tensor, after its rows; every stage turns it into a row mask among its
verify graph's inputs, and the routed experts send a masked row's routes to the
padded route, so neither tier computes them. Shapes, graphs and the ring's messages
stay as they are.

The accept is the full accept cut at the length: it reads no masked row, and a round
whose verified drafts are all right commits one more token, the target's choice at
its last verified row. A committed token is therefore the target's own choice at any
length; the length only moves where a round ends.

A masked row still runs everything but the routed experts (attention, the dense
parts, its KV write, its logits). It is never committed: its slots are those a
rejected draft writes today, position-addressed and overwritten before anyone reads
them, and no verified row attends to a later one.

Unset or 0 nothing here runs: the draft builds no confidence head, the chain has no
extra entries, no mask exists and the verify graphs are the ones captured before.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_UNREAD = object()
_threshold = _UNREAD


def verify_survival_threshold() -> float:
    """The switch, read once: 0.0 unless the relayed flow and the threshold are set."""
    global _threshold
    if _threshold is _UNREAD:
        value = 0.0
        if envs.SGLANG_ENABLE_PP_SPEC.get():
            value = float(envs.SGLANG_PP_DSPARK_VERIFY_SURVIVAL.get() or 0.0)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "SGLANG_PP_DSPARK_VERIFY_SURVIVAL is a survival probability: "
                f"0 (off) to 1, got {value}."
            )
        _threshold = value
    return _threshold


def verify_schedule_enabled() -> bool:
    return verify_survival_threshold() > 0.0


def forget_threshold() -> None:
    """Tests only: the next read takes the environment again."""
    global _threshold
    _threshold = _UNREAD


def scheduled_verify_lens(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    """Rows to verify per request, int64 ``[bs]`` in ``1 .. gamma + 1``: the anchor
    row and the drafts whose survival is at least ``threshold``. ``confidence`` is
    the head's ``[bs, gamma]`` output; survival never rises along a block, so the
    kept drafts are a prefix of it."""
    survival = torch.cumprod(confidence.to(torch.float32), dim=1)
    return (survival >= threshold).sum(dim=1, dtype=torch.int64) + 1


def pack_chain(rows: torch.Tensor, verify_lens: Optional[torch.Tensor]) -> torch.Tensor:
    """What the ring carries for the next round, flat: the ``[bs, width]`` rows, then
    the ``[bs]`` verify lengths when there are any."""
    flat = rows.reshape(-1)
    if verify_lens is None:
        return flat
    return torch.cat([flat, verify_lens.to(flat.dtype).reshape(-1)])


def unpack_chain(
    chain: torch.Tensor, bs: int, width: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``pack_chain`` undone on a stage: the rows ``[bs, width]`` and the lengths."""
    chain = chain.to(torch.int64)
    if chain.numel() == bs * width:
        return chain.reshape(bs, width), None
    if chain.numel() != bs * (width + 1):
        raise RuntimeError(
            f"DSpark relayed chain of {chain.numel()} entries for {bs} requests of "
            f"width {width}: neither the rows alone nor the rows and their lengths."
        )
    flat = chain.reshape(-1)
    return flat[: bs * width].view(bs, width), flat[bs * width :]


def verify_row_mask(verify_lens: Optional[torch.Tensor], bs: int, width: int, device):
    """int32 ``[bs * width]``, 1 on the rows to verify: the first ``verify_lens`` of
    each request's ``width``, or every row without lengths."""
    if verify_lens is None:
        return torch.ones(bs * width, dtype=torch.int32, device=device)
    cols = torch.arange(width, dtype=torch.int64, device=device).view(1, width)
    mask = torch.empty((bs, width), dtype=torch.int32, device=device)
    torch.lt(cols, verify_lens.to(device).view(bs, 1), out=mask)
    return mask.view(-1)


def relay_row_mask(relay, bs: int, width: int, device) -> Optional[torch.Tensor]:
    """A verify round's row mask from the relayed rows, the same on every stage; None
    with the schedule off. A request without a chain verifies its anchor row alone."""
    if not verify_schedule_enabled():
        return None
    return verify_row_mask(relay.verify_cutoff(), bs, width, device)


class VerifyStats:
    """``SGLANG_PP_DSPARK_VERIFY_STATS``: the last stage's counters of what its verify
    rounds did since boot, logged every N request-rounds and whenever a request
    finishes. Host integers only."""

    def __init__(self, every: int, width: int) -> None:
        self.every = int(every)
        self.width = int(width)
        self.rounds = 0
        self.request_rounds = 0
        self.rows = 0
        self.tokens = 0
        self.rows_hist = [0] * (self.width + 1)
        self.capped = 0
        self._logged_at = 0

    @classmethod
    def from_env(cls, width: int) -> Optional[VerifyStats]:
        every = int(envs.SGLANG_PP_DSPARK_VERIFY_STATS.get() or 0)
        return cls(every, width) if every > 0 else None

    def note(self, commit_lens: Sequence[int], verify_lens: Optional[Sequence[int]]):
        """One verify round: tokens committed and rows verified, per request."""
        self.rounds += 1
        for i, tokens in enumerate(commit_lens):
            rows = self.width if verify_lens is None else int(verify_lens[i])
            self.request_rounds += 1
            self.rows += rows
            self.tokens += int(tokens)
            self.rows_hist[rows] += 1
            # Every verified draft was right: the length, not a wrong draft, ended it.
            self.capped += int(rows < self.width and int(tokens) == rows)
        if self.request_rounds - self._logged_at >= self.every:
            self.flush()

    def flush(self) -> None:
        """Log the counters now if they moved since the last line. The last stage
        calls it when a request finishes, so a load's last line holds all of it."""
        if self.request_rounds != self._logged_at:
            self._logged_at = self.request_rounds
            logger.info("%s", self.line())

    def line(self) -> str:
        return (
            f"DSpark PP verify stats: width={self.width} "
            f"threshold={verify_survival_threshold():g} rounds={self.rounds} "
            f"request_rounds={self.request_rounds} rows={self.rows} "
            f"tokens={self.tokens} capped={self.capped} "
            f"rows_hist={','.join(str(n) for n in self.rows_hist[1:])}"
        )


def read_round(
    commit_lens: torch.Tensor, verify_lens: Optional[torch.Tensor]
) -> Tuple[List[int], Optional[List[int]]]:
    """The round's commit lengths and verify lengths on the host, in one read."""
    if verify_lens is None:
        return commit_lens.tolist(), None
    both = torch.stack(
        [commit_lens.to(torch.int64), verify_lens.to(commit_lens.device)]
    ).tolist()
    return both[0], both[1]
