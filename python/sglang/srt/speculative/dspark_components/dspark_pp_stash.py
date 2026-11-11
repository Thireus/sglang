"""The draft's step logits between the tail draft and the verify that reads them.

Under pipeline parallelism a chain is drafted at the end of a round and accepted a
ring later, and the draft of another micro-batch runs in between. In a draft graph the
step logits live in the sampler's static buffer, so that next draft would overwrite
them. Entries are keyed by (rid, seq_len), and spilled to private memory only when
another draft is about to run: one stream copies nothing.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, NamedTuple, Optional, Sequence

import torch

# A row of this makes SoftmaxTemp one-hot at the row's token, at any temperature.
_ONE_HOT_OFF = -1.0e9


class _Entry(NamedTuple):
    seq_len: int
    logits: torch.Tensor  # [rows, gamma, vocab]: a block, or this row alone
    row: int
    borrowed: bool  # logits alias a buffer the next draft overwrites


class DsparkPPStash:
    def __init__(self, *, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self.num_spilled_rows = 0
        self.num_missed_rows = 0

    def __len__(self) -> int:
        return len(self._entries)

    def put(
        self,
        *,
        rids: Sequence[str],
        seq_lens: Sequence[int],
        logits: Optional[torch.Tensor],
        borrowed: bool,
    ) -> None:
        """Register a block's rows. ``logits`` None (a greedy batch or a greedy draft):
        nothing reads them, so the rows only drop what the requests held."""
        for row, (rid, seq_len) in enumerate(zip(rids, seq_lens)):
            self._entries.pop(rid, None)
            if logits is not None:
                self._entries[rid] = _Entry(int(seq_len), logits, row, borrowed)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def spill_borrowed(self) -> None:
        """Before another draft runs: give every row still reading a static buffer its
        own copy. Rows a verify already took are gone, so one stream copies nothing."""
        for rid, entry in self._entries.items():
            if entry.borrowed:
                own = entry.logits[entry.row : entry.row + 1].clone()
                self._entries[rid] = _Entry(entry.seq_len, own, 0, False)
                self.num_spilled_rows += 1

    def take(
        self,
        *,
        rids: Sequence[str],
        seq_lens: Sequence[int],
        draft_tokens: torch.Tensor,
        vocab: int,
    ) -> torch.Tensor:
        """[bs, gamma, vocab] logits of the chain each row verifies. A row with no
        entry for its (rid, seq_len) gets a one-hot at its own tokens, which the
        rejection accepts at the target's probability: the accept stays exact in law,
        only its acceptance drops. The result may alias the sampler's buffer: read it
        before the next draft."""
        hits: List[Optional[_Entry]] = []
        for rid, seq_len in zip(rids, seq_lens):
            entry = self._entries.pop(rid, None)
            hits.append(entry if entry and entry.seq_len == int(seq_len) else None)
        bs, gamma = draft_tokens.shape
        first = hits[0]
        if first is not None and all(
            h is not None and h.logits is first.logits and h.row == i
            for i, h in enumerate(hits)
        ):
            return first.logits[:bs]
        out = self._one_hot(draft_tokens, vocab, like=_first_tensor(hits))
        for i, hit in enumerate(hits):
            if hit is None:
                self.num_missed_rows += 1
            else:
                out[i].copy_(hit.logits[hit.row].to(out.dtype))
        return out

    def drop(self, rid: str) -> None:
        self._entries.pop(rid, None)

    def clear(self) -> None:
        self._entries.clear()

    @staticmethod
    def _one_hot(draft_tokens, vocab, like) -> torch.Tensor:
        bs, gamma = draft_tokens.shape
        if like is not None:
            vocab = like.shape[-1]
        out = torch.full(
            (bs, gamma, vocab),
            _ONE_HOT_OFF,
            dtype=torch.float32 if like is None else like.dtype,
            device=draft_tokens.device,
        )
        out.scatter_(-1, draft_tokens.unsqueeze(-1), 0.0)
        return out


def _first_tensor(hits) -> Optional[torch.Tensor]:
    for hit in hits:
        if hit is not None:
            return hit.logits
    return None
