"""What a request carries between rounds of DSpark under pipeline parallelism.

The last stage drafts at the tail of a round; the block rides the output ring and
every stage verifies it next round. Stage-agnostic: the round's kind and its inputs
are computed here from the relayed rows, so all stages agree without talking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_planner import (
    VerifyWindow,
    alloc_verify_window,
)
from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class DSparkPPRelayInput(PPSpecRelayInput):
    """Rows ``[bs, gamma + 1]``: column 0 is the bonus token, the rest the chain.

    ``has_chain`` says, per request, whether the last stage drafted for this row. It
    is host state decided by the relay's own keys, so reading it never waits on a GPU.
    The topology of the base class stays None: a DSpark block is a chain.
    """

    def __init__(
        self,
        rids: List[str],
        tokens: torch.Tensor,
        parents: Optional[torch.Tensor] = None,
        top_scores: Optional[torch.Tensor] = None,
        *,
        has_chain: Optional[List[bool]] = None,
        alloc_state: Optional[DFlashDraftInputV2] = None,
    ):
        assert parents is None and top_scores is None, "a DSpark block is a chain"
        super().__init__(rids=rids, tokens=tokens)
        self.has_chain = [False] * len(rids) if has_chain is None else list(has_chain)
        # spec_prepare_for_decode calls prepare_for_decode on the DFlash family's
        # spec_info; the allocation it does is this object's.
        self.alloc_state = (
            alloc_state
            if alloc_state is not None
            else DFlashDraftInputV2.create_idle_input(device=tokens.device)
        )

    def __repr__(self) -> str:
        return (
            f"DSparkPPRelayInput(bs={len(self.rids)}, chains={sum(self.has_chain)})"
        )

    def prepare_for_decode(self, batch: ScheduleBatch) -> None:
        self.alloc_state.prepare_for_decode(batch)

    def filter_batch(
        self, new_indices: torch.Tensor, new_indices_cpu: Optional[List[int]] = None
    ) -> None:
        keep = new_indices_cpu if new_indices_cpu is not None else new_indices.tolist()
        self.has_chain = [self.has_chain[i] for i in keep]
        super().filter_batch(new_indices, new_indices_cpu)
        # prepare_for_decode rebuilds it for the rows that run next.
        self.alloc_state.nxt_kv_lens_cpu = None
        self.alloc_state.nxt_kv_lens_sum = None

    def merge_batch(self, other: DSparkPPRelayInput) -> None:
        if not other.rids:
            return
        if not self.rids:
            self.rids, self.tokens = list(other.rids), other.tokens
            self.has_chain = list(other.has_chain)
            return
        self.rids = self.rids + list(other.rids)
        self.tokens = torch.cat([self.tokens, other.tokens])
        self.has_chain = self.has_chain + list(other.has_chain)
        self.alloc_state.nxt_kv_lens_cpu = None
        self.alloc_state.nxt_kv_lens_sum = None

    def adopt(self, relayed: DSparkPPRelayInput) -> None:
        """Take the relayed rows for the requests they cover; a request the relay
        does not mention (merged in after that forward) keeps its row."""
        by_rid = {rid: i for i, rid in enumerate(relayed.rids)}
        rows = [by_rid.get(rid) for rid in self.rids]
        if all(r is None for r in rows):
            return
        device = self.tokens.device
        take = torch.tensor(
            [0 if r is None else r for r in rows], dtype=torch.long, device=device
        )
        keep = torch.tensor([r is None for r in rows], dtype=torch.bool, device=device)
        self.tokens = torch.where(
            keep.unsqueeze(1), self.tokens, relayed.tokens.to(device)[take]
        )
        self.has_chain = [
            own if r is None else relayed.has_chain[r]
            for own, r in zip(self.has_chain, rows)
        ]

    def reindex(self, rids: List[str]) -> DSparkPPRelayInput:
        by_rid = {rid: i for i, rid in enumerate(self.rids)}
        rows = [by_rid[rid] for rid in rids]
        take = torch.tensor(rows, dtype=torch.long, device=self.tokens.device)
        return DSparkPPRelayInput(
            rids=list(rids),
            tokens=self.tokens[take],
            has_chain=[self.has_chain[r] for r in rows],
            alloc_state=self.alloc_state,
        )

    def runs_plain(self, batch: ScheduleBatch, *, verify_max_bs: int = 0) -> bool:
        """The one fact every stage reads to pick the round's forward: its rows, the
        batch's logprob flag and the batch size are the same on all of them."""
        if batch.return_logprob or not any(self.has_chain):
            return True
        return verify_max_bs > 0 and len(self.rids) > verify_max_bs

    def verify_cutoff(self) -> Optional[torch.Tensor]:
        """Per-row verify length when some row carries no chain: its zeros are not
        drafts, so that row commits the bonus alone. None when every row has one."""
        if all(self.has_chain):
            return None
        width = self.tokens.shape[1]
        return torch.tensor(
            [width if c else 1 for c in self.has_chain],
            dtype=torch.int64,
            device=self.tokens.device,
        )


def clamp_window_to_context(
    window: VerifyWindow, seq_lens_cpu, width: int, limit: int
) -> VerifyWindow:
    """Rows past the position table belong to a request about to end by length: its
    output is cut at the limit, so they only must not read outside the table."""
    if seq_lens_cpu is None or int(seq_lens_cpu.max()) + width <= limit:
        return window
    return VerifyWindow(
        positions_2d=window.positions_2d.clamp(max=limit - 1),
        verify_cache_loc=window.verify_cache_loc,
        verify_cache_loc_2d=window.verify_cache_loc_2d,
        iteration_ids=window.iteration_ids,
    )


def position_limit(model_config) -> int:
    limit = int(model_config.context_len)
    table = getattr(model_config.hf_text_config, "max_position_embeddings", None)
    return min(limit, int(table)) if table else limit


def relay_verify_window(
    batch: ScheduleBatch, *, bs, device, width, block_pos_offsets, model_runner
) -> VerifyWindow:
    """A round's window from ``batch.seq_lens``, the same on every stage."""
    window = alloc_verify_window(
        batch=batch,
        bs=bs,
        device=device,
        verify_num_draft_tokens=width,
        block_pos_offsets=block_pos_offsets,
        model_runner=model_runner,
        seq_lens_cpu=batch.seq_lens_cpu,
    )
    return clamp_window_to_context(
        window, batch.seq_lens_cpu, width, position_limit(model_runner.model_config)
    )


def prepare_plain_round(batch: ScheduleBatch, relay: DSparkPPRelayInput, runner_ids):
    """Turn a decode batch into the plain forward's shape: one row per request at the
    next position, the sequence length counting it. Runs inside forward isolation, so
    the scheduler's batch comes back as it was. Returns the iteration's slot plan."""
    prefix_lens = batch.seq_lens
    plan = runner_ids.plan_uniform_extend(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=prefix_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        width=1,
    )
    batch.input_ids = relay.tokens[:, 0].to(prefix_lens.device, dtype=torch.int64)
    batch.out_cache_loc = plan.write_virtual
    batch.forward_mode = ForwardMode.DECODE
    batch.seq_lens = prefix_lens + 1
    if batch.orig_seq_lens is not None:
        batch.orig_seq_lens = batch.seq_lens
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = batch.seq_lens_cpu + 1
    batch.seq_lens_sum = None
    batch.spec_info = None
    return plan


def relay_verify_input(batch: ScheduleBatch, verify_ids_2d, verify_window, width: int):
    """The verify input of a round, the same on every stage (the static verify of
    TargetVerifyExecutor.run_non_compact). Runs inside forward isolation."""
    batch.out_cache_loc = verify_window.verify_cache_loc
    return DFlashVerifyInput(
        draft_token=verify_ids_2d.reshape(-1),
        positions=verify_window.positions_2d.reshape(-1),
        draft_token_num=width,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        live_seq_lens_cpu=batch.seq_lens_cpu,
        iteration_ids=verify_window.iteration_ids,
    )
