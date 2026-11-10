"""One decode round of DSpark under pipeline parallelism, on any stage.

Prefill runs the path it always ran. A decode round either verifies the chain the
last stage drafted at the tail of the previous round, or, when no row has one, runs
the plain decode forward. Non-last stages run their layers and hand the proxies on;
accept, commit and the next draft live on the last stage (DSparkPPLastStage).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.draft_worker_common import build_block_pos_offsets
from sglang.srt.speculative.dspark_components.dspark_pp_relay import (
    DSparkPPRelayInput,
    prepare_plain_round,
    relay_verify_input,
    relay_verify_window,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


def pp_dspark_round_runs_plain(batch: Optional[ScheduleBatch]) -> bool:
    relay = None if batch is None else batch.spec_info
    return isinstance(relay, DSparkPPRelayInput) and relay.runs_plain(
        batch, verify_max_bs=envs.SGLANG_PP_DSPARK_VERIFY_MAX_BS.get()
    )


def _aligned_relay(batch: ScheduleBatch) -> DSparkPPRelayInput:
    relay = batch.spec_info
    if not isinstance(relay, DSparkPPRelayInput):
        raise RuntimeError(
            f"DSpark under pipeline parallelism expected relayed rows, got {relay!r}."
        )
    # The rows track the batch through filter and merge; a recomposition that
    # bypasses those must not hand a request another's chain.
    live_rids = [req.rid for req in batch.reqs]
    if relay.rids != live_rids:
        relay = relay.reindex(live_rids)
        batch.spec_info = relay
    return relay


def run_pp_dspark_decode_round(
    self: Scheduler,
    batch: ScheduleBatch,
    pp_proxy_tensors: Optional[PPProxyTensors],
) -> GenerationBatchResult:
    relay = _aligned_relay(batch)
    plain = relay.runs_plain(
        batch, verify_max_bs=envs.SGLANG_PP_DSPARK_VERIFY_MAX_BS.get()
    )
    if self.pp_group.is_last_rank:
        return _run_last_stage(self, batch, relay, plain, pp_proxy_tensors)
    with self._forward_isolation(batch, overlap=False):
        if plain:
            prepare_plain_round(batch, relay, self.tp_worker.model_runner.runner_ids)
            result = self.tp_worker.forward_batch_generation(
                batch, pp_proxy_tensors=pp_proxy_tensors
            )
        else:
            result = _forward_verify_on_earlier_stage(
                self, batch, relay, pp_proxy_tensors
            )
    batch.input_ids = None
    return result


def _forward_verify_on_earlier_stage(
    self: Scheduler,
    batch: ScheduleBatch,
    relay: DSparkPPRelayInput,
    pp_proxy_tensors: Optional[PPProxyTensors],
) -> GenerationBatchResult:
    model_runner = self.tp_worker.model_runner
    width = get_spec().speculative_num_draft_tokens
    bs = len(batch.seq_lens)
    verify_ids_2d = relay.tokens.to(self.device, dtype=torch.int64)
    window = relay_verify_window(
        batch,
        bs=bs,
        device=self.device,
        width=width,
        block_pos_offsets=_block_pos_offsets(self, width),
        model_runner=model_runner,
    )
    verify_input = relay_verify_input(batch, verify_ids_2d, window, width)
    forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
        batch, self.tp_worker
    )
    result = self.tp_worker.forward_batch_generation(
        batch=None,
        forward_batch=forward_batch,
        pp_proxy_tensors=pp_proxy_tensors,
        is_verify=True,
        # prepare_for_verify planned the attention backend for these rows already.
        skip_attn_backend_init=True,
    )
    result.can_run_cuda_graph = can_run_cuda_graph
    # Commit once the accept comes round the ring; the batch forgets these with
    # the isolation's restore.
    result.spec_verify_ids = verify_ids_2d
    return result


def _block_pos_offsets(self: Scheduler, width: int) -> torch.Tensor:
    offsets = getattr(self, "_pp_dspark_block_pos_offsets", None)
    if offsets is None:
        offsets = build_block_pos_offsets(length=width, device=self.device)
        self._pp_dspark_block_pos_offsets = offsets
    return offsets


def _run_last_stage(
    self: Scheduler,
    batch: ScheduleBatch,
    relay: DSparkPPRelayInput,
    plain: bool,
    pp_proxy_tensors: Optional[PPProxyTensors],
) -> GenerationBatchResult:
    mb_id = next((i for i, mb in enumerate(self.mbs) if mb is batch), 0)
    decoding_requests = sum(mb.batch_size() for mb in self.running_mbs)
    with self._forward_isolation(batch, overlap=False):
        result = self.model_worker.pp_last_stage.forward_round(
            batch,
            relay,
            plain=plain,
            pp_proxy_tensors=pp_proxy_tensors,
            mb_id=mb_id,
            decoding_requests=decoding_requests,
        )
    batch.spec_info = relay
    batch.seq_lens = result.new_seq_lens
    if batch.seq_lens_cpu is not None:
        # A plain round's lengths are known here; reading the device tensor would
        # hold the scheduler until the forward ends.
        batch.seq_lens_cpu = (
            batch.seq_lens_cpu + 1 if plain else result.new_seq_lens.to("cpu")
        )
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    batch.input_ids = None
    self.update_cache_from_scheduler(batch, result)
    return result


def commit_relayed_engram(
    self: Scheduler,
    fwd_batch: ScheduleBatch,
    verify_ids_2d: Optional[torch.Tensor],
    pp_outputs: PPProxyTensors,
) -> None:
    """A verify's Engram history commits after the accept: the anchor and the accepted
    drafts, never the bonus. The last stage commits inside its round; a plain round
    committed in the forward and left no ids here."""
    if verify_ids_2d is None or self.pp_group.is_last_rank:
        return
    self.tp_worker.model_runner.ngram_embedding_manager.update_after_verify(
        verify_ids_2d=verify_ids_2d,
        req_pool_indices=fwd_batch.req_pool_indices.to(verify_ids_2d.device),
        commit_lens=pp_outputs["spec_accept_lens"].to(verify_ids_2d.device),
    )


def adopt_relayed_chain(
    self: Scheduler,
    batch: ScheduleBatch,
    fwd_rids: list[str],
    pp_outputs: PPProxyTensors,
) -> None:
    """Fold what the last stage drafted (or the bonus alone) into the live batch."""
    width = get_spec().speculative_num_draft_tokens
    chain = pp_outputs.tensors.get("spec_next_chain")
    if chain is None:
        relayed = DSparkPPRelayInput.degenerate(
            rids=fwd_rids,
            bonus_tokens=pp_outputs["spec_bonus_tokens"],
            num_draft_tokens=width,
        )
    else:
        relayed = DSparkPPRelayInput(
            rids=fwd_rids,
            tokens=chain.to(torch.int64).reshape(len(fwd_rids), width),
            has_chain=[True] * len(fwd_rids),
        )
    self._pp_spec_set_relay(batch, relayed)
