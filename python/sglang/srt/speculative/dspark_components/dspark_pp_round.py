"""The last pipeline stage's half of a DSpark decode round.

Round r, for a micro-batch: verify the chain drafted at the tail of round r-1 (or run
the plain forward when no row has one), accept, commit, then draft for round r+1. The
next round's tokens go back round the ring with the accept; earlier stages must have
them before they start, which is why the draft moved to the tail.

The order on this stage is the single-stage order (draft, verify, accept, commit), one
draft shifted. Everything it writes is position-addressed and overwritten before anyone
reads it, so nothing rolls back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_exec, get_schedule
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    make_next_draft_input,
    resolve_draft_params,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_pp_embed import (
    install_pp_draft_embedding,
)
from sglang.srt.speculative.dspark_components.dspark_pp_relay import (
    DSparkPPRelayInput,
    prepare_plain_round,
    relay_verify_window,
)
from sglang.srt.speculative.dspark_components.dspark_pp_stash import DsparkPPStash
from sglang.srt.speculative.dspark_components.dspark_switch import DsparkPPSwitch
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyResult
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.srt.speculative.spec_utils import GrammarTree, build_grammar_vocab_mask

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2

logger = logging.getLogger(__name__)


class _RoundOutcome(NamedTuple):
    logits_output: object
    next_token_ids: torch.Tensor  # flat, stride rows per request
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    bonus: torch.Tensor
    can_run_graph: bool
    stride: int


class DSparkPPLastStage:
    def __init__(
        self,
        worker: DSparkWorkerV2,
        *,
        stash: Optional[DsparkPPStash] = None,
        switch: Optional[DsparkPPSwitch] = None,
        embedding=None,
    ) -> None:
        self.worker = worker
        self.width = int(worker.verify_num_draft_tokens)
        self.gamma = int(worker.gamma)
        self.warm_when_off = envs.SGLANG_PP_DSPARK_WARM_WHEN_OFF.get()
        self.stash = (
            stash
            if stash is not None
            else DsparkPPStash(capacity=get_schedule().max_running_requests or 48)
        )
        self.switch = switch if switch is not None else DsparkPPSwitch.from_env(envs)
        self.embedding = embedding

    @classmethod
    def build(cls, worker: DSparkWorkerV2, folder: str) -> DSparkPPLastStage:
        embedding = install_pp_draft_embedding(
            draft_model=worker.draft_model,
            route=envs.SGLANG_PP_DSPARK_EMBED.get(),
            folder=folder,
            # An eager draft can serve any micro-batch; a graph at most its list.
            max_bs=max(
                *get_exec().graph.cuda_graph_config.decode.bs,
                get_schedule().max_running_requests or 1,
            ),
            query_token_num=int(worker.query_token_num),
            mask_token_id=int(worker._mask_token_id),
            device=worker.device,
        )
        logger.info(
            "DSpark on the pipeline, last stage: draft experts=%s, switch=%s, "
            "launch_first=%s, embedding=%s.",
            getattr(worker.draft_model.config, "dspark_n_routed_experts", None),
            envs.SGLANG_PP_DSPARK_SWITCH.get(),
            envs.SGLANG_PP_DSPARK_LAUNCH_FIRST.get(),
            envs.SGLANG_PP_DSPARK_EMBED.get(),
        )
        return cls(worker, embedding=embedding)

    def forward_round(
        self,
        batch: ScheduleBatch,
        relay: DSparkPPRelayInput,
        *,
        plain: bool,
        pp_proxy_tensors,
        mb_id: int,
        decoding_requests: int,
    ) -> GenerationBatchResult:
        """Runs inside the scheduler's forward isolation: edits to ``batch`` stay in it."""
        rids = [req.rid for req in batch.reqs]
        sampling_info = batch.sampling_info
        return_logprob = batch.return_logprob
        outcome = (
            self._run_plain(batch, relay, pp_proxy_tensors)
            if plain
            else self._run_verify(batch, relay, pp_proxy_tensors, rids)
        )
        if not plain:
            self.switch.note_verify(rids, outcome.commit_lens.tolist())
        chain = None
        if self.switch.should_draft(
            mb_id=mb_id,
            rids=rids,
            decoding_requests=decoding_requests,
            return_logprob=return_logprob,
        ):
            chain = self._tail_draft(batch, outcome, rids, sampling_info)
        else:
            # This request's chain will not be verified; nor will its logits be read.
            for rid in rids:
                self.stash.drop(rid)
        return GenerationBatchResult(
            logits_output=outcome.logits_output,
            next_token_ids=outcome.next_token_ids,
            accept_lens=outcome.commit_lens,
            can_run_cuda_graph=outcome.can_run_graph,
            next_draft_input=make_next_draft_input(
                bonus_tokens=outcome.bonus,
                new_seq_lens=outcome.new_seq_lens,
            ),
            speculative_num_draft_tokens=self.width,
            speculative_output_stride=outcome.stride,
            new_seq_lens=outcome.new_seq_lens,
            next_verify_chain=chain,
        )

    # ---- this round ------------------------------------------------------------

    def _window(self, batch: ScheduleBatch, bs: int) -> VerifyWindow:
        worker = self.worker
        return relay_verify_window(
            batch,
            bs=bs,
            device=worker.device,
            width=self.width,
            block_pos_offsets=worker._block_pos_offsets,
            model_runner=worker.model_runner,
        )

    def _run_verify(
        self, batch, relay: DSparkPPRelayInput, pp_proxy_tensors, rids
    ) -> _RoundOutcome:
        worker = self.worker
        executor = worker._verify_executor
        bs = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        sampling_info = batch.sampling_info
        verify_ids_2d = relay.tokens.to(worker.device, dtype=torch.int64)
        draft_tokens = verify_ids_2d[:, 1:]
        seq_lens_cpu = batch.seq_lens_cpu
        verify_window = self._window(batch, bs)
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )
        executor.begin_simulated_step()
        target_verify: TargetVerifyResult = executor.run_non_compact(
            batch=batch,
            draft_input=relay.alloc_state,
            verify_ids_2d=verify_ids_2d,
            verify_window=verify_window,
            sampling_info=sampling_info,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        logits_output = target_verify.logits_output
        target_logits = logits_output.next_token_logits
        if batch.has_grammar:
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=target_logits.device,
                barrier=None,
            )
            if grammar_mask is not None:
                grammar_mask.apply(target_logits)
        accept = executor.accept_and_finalize(
            folded_accept=False,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=target_logits,
            draft_block=self._draft_block(
                rids, seq_lens_cpu, draft_tokens, sampling_info, target_logits
            ),
            sampling_info=sampling_info,
            draft_input=relay.alloc_state,
            layout=_cutoff_layout(relay),
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            simulate_bonus_sampling_info=_simulated_bonus_sampling(
                worker, sampling_info
            ),
        )
        # This stage holds a hasher too: no layer here reads it, but it runs in every
        # forward, and one code path commits on all three stages.
        worker.model_runner.ngram_embedding_manager.update_after_verify(
            verify_ids_2d=verify_ids_2d,
            req_pool_indices=batch.req_pool_indices,
            commit_lens=accept.commit_lens,
        )
        with worker._draft_context():
            executor.commit_hidden(
                batch=batch,
                layout=None,
                hidden_strided=None,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=False,
            )
        logits_output.hidden_states = None
        return _RoundOutcome(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            commit_lens=accept.commit_lens,
            new_seq_lens=accept.new_seq_lens,
            bonus=accept.bonus,
            can_run_graph=target_verify.can_run_cuda_graph,
            stride=self.width,
        )

    def _draft_block(
        self, rids, seq_lens_cpu, draft_tokens, sampling_info, target_logits
    ) -> DraftBlockResult:
        worker = self.worker
        bs = draft_tokens.shape[0]
        params = resolve_draft_params(
            sampling_info=sampling_info, bs=bs, device=worker.device
        )
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        corrected_logits = None
        if worker._greedy_draft or all_greedy:
            # A greedy accept reads no draft logits, and a greedy draft keeps none.
            for rid in rids:
                self.stash.drop(rid)
        else:
            corrected_logits = self.stash.take(
                rids=rids,
                seq_lens=seq_lens_cpu.tolist(),
                draft_tokens=draft_tokens,
                vocab=target_logits.shape[-1],
            )
        return DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=corrected_logits,
            greedy_mask=params.greedy_mask,
            temperatures=params.temperatures,
            greedy_draft=worker._greedy_draft,
        )

    def _run_plain(self, batch, relay: DSparkPPRelayInput, pp_proxy_tensors) -> _RoundOutcome:
        worker = self.worker
        bs = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        plan = prepare_plain_round(batch, relay, worker.model_runner.runner_ids)
        output = worker.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if self.warm_when_off else CaptureHiddenMode.NULL
            ),
        )
        logits_output = output.logits_output
        next_token_ids = worker._tp_sync.sync(
            SpecTpSyncSite.DSPARK_TARGET, output.next_token_ids
        )
        commit_lens = torch.ones_like(prefix_lens, dtype=torch.int32)
        if self.warm_when_off:
            self._commit_plain_row(batch, plan, logits_output, prefix_lens, commit_lens)
        logits_output.hidden_states = None
        if batch.return_logprob and logits_output.next_token_logprobs is not None:
            # The result processor reads a speculative row per request.
            logits_output.next_token_logprobs = logits_output.next_token_logprobs.view(
                bs, -1
            )
        return _RoundOutcome(
            logits_output=logits_output,
            next_token_ids=next_token_ids.reshape(-1),
            commit_lens=commit_lens,
            new_seq_lens=prefix_lens + 1,
            bonus=next_token_ids.reshape(-1).to(torch.int64),
            can_run_graph=output.can_run_cuda_graph,
            stride=1,
        )

    def _commit_plain_row(self, batch, plan, logits_output, prefix_lens, commit_lens):
        worker = self.worker
        injector = worker._kv_injector
        bs = prefix_lens.shape[0]
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSpark plain round expected target hidden, got None.")
        cache_loc = injector.ids_for(plan)
        state_slot = (
            batch.req_pool_indices[:bs].reshape(-1)
            if injector.swa_ring_size is not None
            else None
        )
        with worker._draft_context():
            injector.inject_target_hidden(
                target_hidden=hidden.reshape(bs, -1),
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc.view(bs, 1),
                positions=prefix_lens.to(torch.int64),
                commit_lens=commit_lens,
                state_slot=state_slot,
            )

    # ---- the next round's chain --------------------------------------------------

    def _tail_draft(
        self, batch, outcome: _RoundOutcome, rids, sampling_info
    ) -> torch.Tensor:
        """The state the proposer reads at the head of an iteration; the isolation
        around this round undoes it. Returns the rows [bs, gamma + 1] the ring carries:
        a new tensor, so a later replay of the draft graph cannot change it in flight."""
        worker = self.worker
        bs = len(rids)
        new_seq_lens = outcome.new_seq_lens
        new_seq_lens_cpu = new_seq_lens.to("cpu")
        self.stash.spill_borrowed()
        batch.seq_lens = new_seq_lens
        batch.seq_lens_cpu = new_seq_lens_cpu
        batch.seq_lens_sum = int(new_seq_lens_cpu.sum())
        batch.input_ids = None
        batch.spec_info = None
        draft_input = make_next_draft_input(
            bonus_tokens=outcome.bonus, new_seq_lens=new_seq_lens
        )
        if self.embedding is not None:
            self.embedding.stage_anchors(draft_input.bonus_tokens)
        with worker._draft_context():
            proposal = worker._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=self._window(batch, bs),
                bs=bs,
                device=worker.device,
                target_model=worker.target_worker.model_runner.model,
                sampling_info=sampling_info,
            )
        block = proposal.draft_block
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        self.stash.put(
            rids=rids,
            seq_lens=new_seq_lens_cpu.tolist(),
            logits=None if all_greedy else block.corrected_logits,
            borrowed=proposal.folded,
        )
        return torch.cat(
            [draft_input.bonus_tokens.view(bs, 1), block.draft_tokens.to(torch.int64)],
            dim=1,
        ).reshape(-1)

    # ---- housekeeping ------------------------------------------------------------

    def note_request_finished(self, rid: str) -> None:
        self.stash.drop(rid)
        self.switch.forget(rid)

    def clear(self) -> None:
        self.stash.clear()
        self.switch.clear()


class _CutoffLayout(NamedTuple):
    verify_lens: torch.Tensor


def _cutoff_layout(relay: DSparkPPRelayInput) -> Optional[_CutoffLayout]:
    cutoff = relay.verify_cutoff()
    return None if cutoff is None else _CutoffLayout(cutoff)


def _simulated_bonus_sampling(worker, sampling_info):
    """As the single-stage step: simulated acceptance draws its bonus at that row."""
    if (
        worker._simulated_correct_len is not None
        and sampling_info is not None
        and not sampling_info.is_all_greedy
    ):
        return sampling_info
    return None
