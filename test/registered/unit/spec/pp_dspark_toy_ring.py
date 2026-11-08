"""Three pipeline stages in one process, running the real PP event loop.

``SchedulerPPMixin.event_loop_pp`` drives each stage on its own thread with the real
relay code: launch order, output exchange, ``_pp_prep_batch_result``, the DSpark round
of ``scheduler_pp_dspark`` and the last stage's ``DSparkPPLastStage`` with the real
accept and finalize. Stubbed are the edges: the devices, NCCL (in-process channels
whose sends complete only when the peer takes them, reading the tensor at that moment,
so an aliased buffer shows), the forwards (a toy model whose next token is a function
of the history) and the draft (right with a chosen probability).

Being right here means: every request gets the output plain greedy decoding gives it,
through any accept lengths, merges, finishes and switch decisions.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest import mock

import torch

from sglang.srt.managers import scheduler_pp_dspark, scheduler_pp_mixin
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    _get_speculative_output_stride,
)
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftProposal,
)
from sglang.srt.speculative.dspark_components.dspark_pp_round import DSparkPPLastStage
from sglang.srt.speculative.dspark_components.dspark_pp_stash import DsparkPPStash
from sglang.srt.speculative.dspark_components.dspark_switch import DsparkPPSwitch
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0, suite="base-a-test-cpu", disabled="PP DSpark test helpers")

VOCAB = 97
WIDTH = 6
GAMMA = 5
PP = 3
HIDDEN = 8
RING_TIMEOUT_S = 120.0


def toy_next(history: List[int]) -> int:
    """The toy target's greedy next token; never 0, the degenerate chains' filler."""
    return (sum(history[-3:]) * 31 + len(history) * 7 + 11) % (VOCAB - 3) + 3


def plain_output(prompt: List[int], max_new: int, stop: Optional[int] = None) -> List[int]:
    tokens, out = list(prompt), []
    while len(out) < max_new:
        tok = toy_next(tokens)
        out.append(tok)
        tokens.append(tok)
        if tok == stop:
            break
    return out


class _Stop(Exception):
    pass


class ToyChannel:
    """One direction of a stage pair: a send completes when the peer takes it."""

    def __init__(self, stop: threading.Event):
        self._q: queue.Queue = queue.Queue()
        self._stop = stop

    def put(self, tensor_dict) -> threading.Event:
        done = threading.Event()
        self._q.put((tensor_dict, done))
        return done

    def get(self):
        while True:
            try:
                tensor_dict, done = self._q.get(timeout=0.05)
                break
            except queue.Empty:
                if self._stop.is_set():
                    raise _Stop() from None
        # NCCL reads the buffer about now, not when the send was posted.
        received = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in tensor_dict.items()
        }
        done.set()
        return received


class _Work:
    def __init__(self, done: threading.Event, stop: threading.Event):
        self._done, self._stop = done, stop
        self.work = self

    def wait(self):
        while not self._done.wait(timeout=0.05):
            if self._stop.is_set():
                raise _Stop()


@dataclass
class ToyReq:
    rid: str
    prompt: List[int]
    max_new: int
    stop_token: Optional[int] = None
    slot: int = 0
    arrival: int = 0
    is_health_check: bool = False
    output_ids: List[int] = field(default_factory=list)
    kv_committed_len: int = 0
    prompt_done: int = 0
    finished_flag: bool = False
    is_retracted: bool = False
    spec_verify_ct: int = 0
    spec_num_correct_drafts: int = 0
    return_logprob: bool = False

    def finished(self) -> bool:
        return self.finished_flag

    @property
    def tokens(self) -> List[int]:
        return self.prompt + self.output_ids


@dataclass
class ToyBatch:
    reqs: List[ToyReq] = field(default_factory=list)
    forward_mode: ForwardMode = ForwardMode.DECODE
    is_extend_in_batch: bool = False
    seq_lens: Optional[torch.Tensor] = None
    seq_lens_cpu: Optional[torch.Tensor] = None
    seq_lens_sum: Optional[int] = None
    orig_seq_lens: Optional[torch.Tensor] = None
    req_pool_indices: Optional[torch.Tensor] = None
    out_cache_loc: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    spec_info: object = None
    sampling_info: object = None
    spec_algorithm: SpeculativeAlgorithm = SpeculativeAlgorithm.DSPARK
    return_logprob: bool = False
    has_grammar: bool = False
    contains_last_prefill_chunk: bool = True
    chunked_req: Optional[ToyReq] = None
    extend_tokens: Optional[List[int]] = None
    device: str = "cpu"

    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return not self.reqs

    def copy(self) -> ToyBatch:
        return ToyBatch(
            reqs=self.reqs[:],
            forward_mode=self.forward_mode,
            is_extend_in_batch=self.is_extend_in_batch,
            seq_lens=self.seq_lens,
            seq_lens_cpu=self.seq_lens_cpu,
            req_pool_indices=self.req_pool_indices,
            out_cache_loc=self.out_cache_loc,
            spec_info=self.spec_info,
            spec_algorithm=self.spec_algorithm,
            return_logprob=self.return_logprob,
            contains_last_prefill_chunk=self.contains_last_prefill_chunk,
            chunked_req=self.chunked_req,
        )

    def filter_batch(self, keep: List[int]) -> None:
        index = torch.tensor(keep, dtype=torch.int64)
        self.reqs = [self.reqs[i] for i in keep]
        for name in ("seq_lens", "seq_lens_cpu", "orig_seq_lens", "req_pool_indices"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[index])
        self.seq_lens_sum = None
        self.return_logprob = any(r.return_logprob for r in self.reqs)
        if self.spec_info is not None and hasattr(self.spec_info, "filter_batch"):
            self.spec_info.filter_batch(new_indices=index, new_indices_cpu=keep)

    def merge_batch(self, other: ToyBatch) -> None:
        self.reqs = self.reqs + other.reqs
        for name in ("seq_lens", "orig_seq_lens", "req_pool_indices"):
            setattr(self, name, torch.cat([getattr(self, name), getattr(other, name)]))
        if self.seq_lens_cpu is not None and other.seq_lens_cpu is not None:
            self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.return_logprob = any(r.return_logprob for r in self.reqs)
        if self.spec_info is not None:
            self.spec_info.merge_batch(other.spec_info)


class ToyPlan:
    def __init__(self, write_virtual):
        self.write_virtual = write_virtual

    def write_ids(self, runner_ids, cols=None):
        return self.write_virtual


class ToyRunnerIds:
    """A virtual slot id tells which request and position wrote it."""

    def plan_uniform_extend(
        self, *, req_pool_indices, seq_lens, seq_lens_cpu, width, **_
    ):
        offsets = torch.arange(width, dtype=torch.int64)
        rows = seq_lens.to(torch.int64).unsqueeze(1) + offsets
        return ToyPlan((req_pool_indices.unsqueeze(1) * 100000 + rows).reshape(-1))


class ToyEngram:
    """What each stage's history saw committed: anchor plus accepted drafts."""

    def __init__(self):
        self.committed: Dict[int, List[int]] = {}

    def update_after_verify(self, *, verify_ids_2d, req_pool_indices, commit_lens):
        for row, slot in enumerate(req_pool_indices.tolist()):
            n = int(commit_lens[row])
            self.committed.setdefault(slot, []).extend(verify_ids_2d[row, :n].tolist())

    def commit_forward(self, ids, req_pool_indices):
        for row, slot in enumerate(req_pool_indices.tolist()):
            self.committed.setdefault(slot, []).append(int(ids[row]))


class ToyInjector:
    """The draft's window: which positions got a committed row."""

    swa_ring_size = None

    def __init__(self):
        self.rows: Dict[int, List[int]] = {}

    def ids_for(self, plan, cols=None):
        return plan.write_ids(None)

    def inject_target_hidden(
        self,
        *,
        target_hidden,
        cache_loc,
        cache_loc_2d,
        positions,
        commit_lens,
        state_slot,
        iteration_ids=None,  # passed by upstream's verify since #48001
    ):
        bs, width = cache_loc_2d.shape
        assert target_hidden.shape[0] == bs * width
        assert positions.numel() == bs * width
        pos = positions.view(bs, width)
        for row in range(bs):
            slot = int(cache_loc_2d[row, 0]) // 100000
            for col in range(int(commit_lens[row])):
                assert int(cache_loc_2d[row, col]) % 100000 == int(pos[row, col])
                self.rows.setdefault(slot, []).append(int(pos[row, col]))


class ToyProposer:
    """Drafts from the history the verify just settled; right with ``accuracy``."""

    def __init__(self, ring: ToyRing, accuracy: float, seed: int):
        self.ring = ring
        self.accuracy = accuracy
        self.rng = random.Random(seed)
        # A static buffer, as the sampler's graph output is.
        self.static_tokens = torch.zeros((32, GAMMA), dtype=torch.int64)
        self.num_proposals = 0

    def propose(
        self, *, batch, draft_input, verify_window, bs, device, target_model, sampling_info,
        anchor_prepared=False,
    ):
        self.num_proposals += 1
        new_seq_lens = draft_input.new_seq_lens.tolist()
        bonus = draft_input.bonus_tokens.tolist()
        assert verify_window.positions_2d[0, 0] == new_seq_lens[0]
        for row, req in enumerate(batch.reqs):
            base = self.ring.last_rows[req.rid][: new_seq_lens[row]] + [bonus[row]]
            chain = []
            for _ in range(GAMMA):
                tok = toy_next(base + chain)
                if self.rng.random() >= self.accuracy:
                    tok = (tok + 1 + self.rng.randrange(VOCAB - 2)) % VOCAB
                chain.append(tok)
            self.static_tokens[row] = torch.tensor(chain, dtype=torch.int64)
        tokens = self.static_tokens[:bs]
        block = DraftBlockResult(
            draft_tokens=tokens,
            corrected_logits=None,
            greedy_mask=torch.ones(bs, dtype=torch.bool),
            temperatures=torch.ones(bs),
            greedy_draft=False,
        )
        return DraftProposal(
            draft_block_ids=torch.zeros((bs, GAMMA), dtype=torch.int64),
            draft_block=block,
            folded=True,
        )


class ToyTargetWorker:
    """The stages' forwards."""

    def __init__(self, stage: ToyStage):
        self.stage = stage
        self.model_runner = SimpleNamespace(
            runner_ids=stage.ring.runner_ids,
            ngram_embedding_manager=stage.engram,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    model_type="deepseek_v41",
                    max_position_embeddings=stage.ring.max_position,
                ),
                context_len=stage.ring.max_position,
            ),
            attn_backend=SimpleNamespace(make_forward_metadata_from_raw_verify=None),
            sampling_observer=None,
            device="cpu",
            model=None,
        )

    def forward_batch_generation(
        self, batch=None, forward_batch=None, pp_proxy_tensors=None, is_verify=False,
        skip_attn_backend_init=None, capture_hidden_mode=None,
    ):
        if batch is not None:
            # A plain round, as ForwardBatch.init_new reads it.
            assert batch.forward_mode.is_decode(), batch.forward_mode
            forward_batch = SimpleNamespace(
                forward_mode=batch.forward_mode,
                input_ids=batch.input_ids,
                positions=(batch.seq_lens - 1).to(torch.int64),
                req_pool_indices=batch.req_pool_indices,
                reqs=batch.reqs,
                out_cache_loc=batch.out_cache_loc,
            )
        return self.stage.toy_forward(forward_batch, pp_proxy_tensors)


class ToyStage(SchedulerPPMixin):
    def __init__(self, ring: ToyRing, rank: int):
        self.ring = ring
        self.rank = rank
        self.device = "cpu"
        # The loop asks the pause state whether it may launch (upstream #49144).
        self.pause_state = PauseState()
        # The loops run the state checks through it (upstream #48997); a toy checks nothing.
        self.invariant_checker = SimpleNamespace(run=lambda *a, **k: None)
        self.spec_algorithm = SpeculativeAlgorithm.DSPARK
        self.pp_group = SimpleNamespace(
            is_first_rank=rank == 0,
            is_last_rank=rank == PP - 1,
            send_tensor_dict=self._send_tensor_dict,
            recv_tensor_dict=self._recv_tensor_dict,
        )
        self.attn_tp_group = None
        self.engram = ToyEngram()
        self.tp_worker = ToyTargetWorker(self)
        self.model_worker = (
            SimpleNamespace(pp_last_stage=None) if rank == PP - 1 else self.tp_worker
        )
        self.draft_worker = None
        self.update_cache_from_scheduler = lambda batch, result: None
        self.device_module = SimpleNamespace(
            Event=lambda: SimpleNamespace(
                record=lambda stream=None: None, synchronize=lambda: None
            ),
            current_stream=lambda: SimpleNamespace(wait_event=lambda event: None),
            stream=lambda s: nullcontext(),
        )
        self.schedule_stream = SimpleNamespace()
        self.forward_stream = SimpleNamespace(
            wait_stream=lambda s: None, wait_event=lambda e: None
        )
        self.copy_stream = SimpleNamespace(
            wait_stream=lambda s: None, wait_event=lambda e: None
        )
        self.forward_stream_ctx = nullcontext()
        self.copy_stream_ctx = nullcontext()
        self.waiting: List[ToyReq] = []
        self.iterations = 0
        self.verify_log: List[tuple] = []
        self.round_log: List[tuple] = []
        self.draft_rows: Dict[int, List[int]] = {}
        self.finished_rids: Dict[str, List[int]] = {}
        self.finished_commits: Dict[str, List[int]] = {}
        self.spec_verify_cts: Dict[str, int] = {}

    # ---- the event loop's edges ----------------------------------------------

    def init_pp_loop_state(self):
        super().init_pp_loop_state()
        self.running_mbs = [ToyBatch() for _ in range(self.pp_loop_size)]

    def increment_scheduler_iterations(self):
        self.iterations += 1
        self.ring.note_iteration()
        if self.ring.stop.is_set():
            raise _Stop()

    def ingest_requests(self, stop_at_pause=False):
        for req in self.ring.requests_for(self.rank, self.iterations):
            self.waiting.append(req)
        return []

    def _pp_send_pyobj_to_next_stage(self, data, async_send=False):
        return []

    def _send_tensor_dict(self, tensor_dict, all_gather_group=None, async_send=True):
        channel = self.ring.channels[(self.rank + 1) % PP]
        done = channel.put(dict(tensor_dict))
        return [_Work(done, self.ring.stop)]

    def _recv_tensor_dict(self, all_gather_group=None):
        return self.ring.channels[self.rank].get()

    def on_idle(self):
        time.sleep(0.0005)

    def get_next_batch_to_run(self, running_batch, last_batch):
        if last_batch is not None and last_batch.forward_mode.is_extend():
            chunked = last_batch.chunked_req
            keep = [i for i, r in enumerate(last_batch.reqs) if r is not chunked]
            if len(keep) != last_batch.batch_size():
                last_batch.filter_batch(keep)
            if not last_batch.is_empty() and last_batch is not running_batch:
                if running_batch.is_empty():
                    running_batch = last_batch
                else:
                    running_batch.merge_batch(last_batch)
        if not running_batch.is_empty():
            alive = [i for i, r in enumerate(running_batch.reqs) if not r.finished()]
            if len(alive) != running_batch.batch_size():
                running_batch.filter_batch(alive)
        prefill = self._next_prefill(running_batch)
        batch = prefill
        if batch is None and not running_batch.is_empty():
            batch = running_batch
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_info.prepare_for_decode(batch)
        return SimpleNamespace(running_batch=running_batch, batch_to_run=batch)

    def _next_prefill(self, running_batch) -> Optional[ToyBatch]:
        if not self.waiting or running_batch.batch_size() >= self.ring.max_micro_batch:
            return None
        req = self.waiting[0]
        if self.ring.chunk_in_flight.get((self.rank, req.rid)):
            return None
        start = req.prompt_done
        end = min(len(req.prompt), start + self.ring.chunk)
        req.prompt_done = end
        last = end == len(req.prompt)
        if last:
            self.waiting.pop(0)
        self.ring.chunk_in_flight[(self.rank, req.rid)] = True
        return ToyBatch(
            reqs=[req],
            forward_mode=ForwardMode.EXTEND,
            seq_lens=torch.tensor([end], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([end], dtype=torch.int64),
            orig_seq_lens=torch.tensor([end], dtype=torch.int64),
            req_pool_indices=torch.tensor([req.slot], dtype=torch.int64),
            contains_last_prefill_chunk=last,
            chunked_req=None if last else req,
            extend_tokens=req.prompt[start:end],
            return_logprob=req.return_logprob,
        )

    # ---- the forwards --------------------------------------------------------

    def run_batch(self, batch, pp_proxy_tensors):
        if batch.forward_mode.is_extend():
            return self._toy_prefill(batch, pp_proxy_tensors)
        return scheduler_pp_dspark.run_pp_dspark_decode_round(
            self, batch, pp_proxy_tensors
        )

    _forward_isolation = Scheduler._forward_isolation

    def _toy_prefill(self, batch, pp_proxy_tensors):
        rows = len(batch.extend_tokens)
        if self.rank > 0:
            assert pp_proxy_tensors["hidden_states"].shape[0] == rows
        if self.rank < PP - 1:
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=PPProxyTensors(
                    {"hidden_states": torch.zeros((rows, HIDDEN))}
                ),
            )
        req = batch.reqs[0]
        token = (
            toy_next(req.prompt)
            if batch.contains_last_prefill_chunk
            else 0  # a middle chunk's placeholder, which nobody may read
        )
        return GenerationBatchResult(
            next_token_ids=torch.tensor([token], dtype=torch.int64),
            new_seq_lens=batch.seq_lens,
        )

    def toy_forward(self, forward_batch, pp_proxy_tensors):
        verify = forward_batch.forward_mode.is_target_verify()
        ids = forward_batch.input_ids
        positions = forward_batch.positions
        reqs = forward_batch.reqs
        bs = len(reqs)
        width = ids.numel() // bs
        assert width == (WIDTH if verify else 1), (verify, width)
        rows_2d = ids.view(bs, width)
        pos_2d = positions.view(bs, width)
        self.round_log.append(
            (
                "verify" if verify else "plain",
                tuple(r.rid for r in reqs),
                tuple(rows_2d.reshape(-1).tolist()),
                tuple(pos_2d.reshape(-1).tolist()),
            )
        )
        if pp_proxy_tensors is not None:
            # The previous stage's hand-off is this round's rows, in this order.
            received = pp_proxy_tensors["hidden_states"]
            assert received.shape[0] == ids.numel()
            assert torch.equal(received[:, 0].to(torch.int64), ids.reshape(-1))
        if not verify:
            # DECODE commits the Engram history inside the forward.
            self.engram.commit_forward(ids, forward_batch.req_pool_indices)
        if self.rank < PP - 1:
            hidden = torch.zeros((ids.numel(), HIDDEN))
            hidden[:, 0] = ids.reshape(-1).float()
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=PPProxyTensors(
                    {"hidden_states": hidden}
                ),
            )
        logits = torch.zeros((ids.numel(), VOCAB))
        for row, req in enumerate(reqs):
            first = int(pos_2d[row, 0])
            history = req.tokens[:first]
            for col in range(width):
                history = history + [int(rows_2d[row, col])]
                logits[row * width + col, toy_next(history)] = 10.0
            self.ring.last_rows[req.rid] = history
        hidden = torch.ones((ids.numel(), HIDDEN))
        output = SimpleNamespace(
            next_token_logits=logits,
            hidden_states=hidden,
            next_token_logprobs=None,
            auxiliary_device_output=None,
            sampling_mask_output=None,
        )
        return GenerationBatchResult(
            logits_output=output,
            next_token_ids=logits.argmax(dim=-1) if not verify else None,
            can_run_cuda_graph=False,
        )

    # ---- what process_batch_result does to a request ---------------------------

    def process_batch_result(self, batch, result):
        if batch.forward_mode.is_extend():
            self.ring.chunk_in_flight[(self.rank, batch.reqs[0].rid)] = False
            if batch.contains_last_prefill_chunk:
                req = batch.reqs[0]
                req.output_ids.append(int(result.next_token_ids[0]))
                req.kv_committed_len = len(req.prompt)
                self._maybe_finish(req)
            return
        stride = _get_speculative_output_stride(result)
        tokens = result.next_token_ids.tolist()
        accept = result.accept_lens.tolist()
        for i, req in enumerate(batch.reqs):
            if req.finished():
                continue
            run = tokens[i * stride : i * stride + accept[i]]
            req.kv_committed_len += len(run)
            req.spec_verify_ct += 1
            for tok in run:
                req.output_ids.append(tok)
                if self._maybe_finish(req):
                    break

    def _maybe_finish(self, req: ToyReq) -> bool:
        if req.output_ids and req.output_ids[-1] == req.stop_token:
            req.finished_flag = True
        if len(req.output_ids) >= req.max_new:
            req.finished_flag = True
        if req.finished_flag and req.rid not in self.finished_rids:
            self.finished_rids[req.rid] = list(req.output_ids)
            self.finished_commits[req.rid] = req.tokens[: req.kv_committed_len]
            self.spec_verify_cts[req.rid] = req.spec_verify_ct
            last_stage = self.model_worker.pp_last_stage if self.rank == PP - 1 else None
            if last_stage is not None:
                last_stage.note_request_finished(req.rid)
            self.ring.note_finished(self.rank)
        return req.finished_flag


class ToyRing:
    def __init__(
        self,
        requests: List[dict],
        *,
        accuracy: float = 0.8,
        mode: str = "on",
        chunk: int = 1 << 30,
        max_micro_batch: int = 9,
        seed: int = 0,
        launch_first: bool = False,
        breakeven=(),
        max_position: int = 1 << 20,
    ):
        self.stop = threading.Event()
        self.channels = [ToyChannel(self.stop) for _ in range(PP)]
        self.chunk = chunk
        self.max_micro_batch = max_micro_batch
        self.max_position = max_position
        self.chunk_in_flight: Dict[tuple, bool] = {}
        self.runner_ids = ToyRunnerIds()
        self.last_rows: Dict[str, List[int]] = {}
        self.launch_first = launch_first
        self.requests = requests
        self._lock = threading.Lock()
        self._finished = [0] * PP
        self._last_progress = time.monotonic()
        self.stages = [ToyStage(self, rank) for rank in range(PP)]
        self.per_stage_reqs = [
            {
                (r["arrival"], r["rid"]): ToyReq(
                    rid=r["rid"],
                    prompt=list(r["prompt"]),
                    max_new=r["max_new"],
                    stop_token=r.get("stop"),
                    slot=r.get("slot", i + 1),
                    arrival=r["arrival"],
                    return_logprob=r.get("return_logprob", False),
                )
                for i, r in enumerate(requests)
            }
            for _ in range(PP)
        ]
        self.proposer = ToyProposer(self, accuracy, seed)
        self.injector = ToyInjector()
        self.switch = DsparkPPSwitch(mode=mode, breakeven=breakeven)
        last = self.stages[-1]
        tp_sync = SpecTpSync(SimpleNamespace(world_size=1))
        worker = SimpleNamespace(
            verify_num_draft_tokens=WIDTH,
            gamma=GAMMA,
            model_runner=last.tp_worker.model_runner,
            device="cpu",
            _block_pos_offsets=torch.arange(WIDTH, dtype=torch.int64),
            _greedy_draft=False,
            _simulated_correct_len=None,
            _tp_sync=tp_sync,
            _kv_injector=self.injector,
            _proposer=self.proposer,
            target_worker=last.tp_worker,
            _draft_context=nullcontext,
        )
        worker._verify_executor = TargetVerifyExecutor(
            target_worker=last.tp_worker,
            gamma=GAMMA,
            verify_num_draft_tokens=WIDTH,
            model_runner=last.tp_worker.model_runner,
            kv_injector=self.injector,
            tp_sync=tp_sync,
        )
        self.last_stage = DSparkPPLastStage(
            worker, stash=DsparkPPStash(capacity=27), switch=self.switch
        )
        last.model_worker.pp_last_stage = self.last_stage
        self.errors: List[BaseException] = []

    def requests_for(self, rank: int, iteration: int) -> List[ToyReq]:
        reqs = self.per_stage_reqs[rank]
        due = [k for k in reqs if k[0] <= iteration]
        return [reqs.pop(k) for k in due]

    def note_iteration(self):
        with self._lock:
            self._last_progress = time.monotonic()

    def note_finished(self, rank: int):
        with self._lock:
            self._finished[rank] += 1
            if all(n >= len(self.requests) for n in self._finished):
                self.stop.set()

    def run(self, timeout: float = RING_TIMEOUT_S) -> None:
        """Run every stage's real event loop until all requests finished on all stages."""
        ctx = threading.local()

        def fake_parallel():
            return SimpleNamespace(
                pp_size=PP,
                pp_rank=ctx.rank,
                pp_async_batch_depth=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
            )

        fake_spec = SimpleNamespace(speculative_num_draft_tokens=WIDTH)

        def stage_main(stage: ToyStage):
            ctx.rank = stage.rank
            try:
                stage.event_loop_pp()
            except _Stop:
                pass
            except BaseException as err:  # noqa: BLE001
                self.errors.append(err)
                self.stop.set()

        def prepare_for_verify(verify_input, batch, target_worker, **_):
            batch.input_ids = verify_input.draft_token
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            forward_batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                input_ids=verify_input.draft_token,
                positions=verify_input.positions,
                req_pool_indices=batch.req_pool_indices,
                reqs=batch.reqs,
                out_cache_loc=batch.out_cache_loc,
                capture_hidden_mode=verify_input.capture_hidden_mode,
            )
            assert verify_input.capture_hidden_mode == CaptureHiddenMode.FULL
            return forward_batch, False

        def prepare_for_decode(self_input, batch):
            bs = batch.batch_size()
            committed = [int(r.kv_committed_len) for r in batch.reqs]
            batch.seq_lens_cpu = torch.tensor(committed, dtype=torch.int64)
            batch.seq_lens_sum = sum(committed)
            self_input.nxt_kv_lens_cpu = batch.seq_lens_cpu + 2 * WIDTH
            assert bs == batch.seq_lens.numel()

        patches = [
            mock.patch.object(scheduler_pp_mixin, "get_parallel", fake_parallel),
            mock.patch.object(scheduler_pp_mixin, "get_spec", lambda: fake_spec),
            mock.patch.object(scheduler_pp_dspark, "get_spec", lambda: fake_spec),
            mock.patch.object(
                scheduler_pp_mixin, "set_time_batch", lambda *a, **k: None
            ),
            mock.patch.object(DFlashVerifyInput, "prepare_for_verify", prepare_for_verify),
            mock.patch.object(DFlashDraftInputV2, "prepare_for_decode", prepare_for_decode),
        ]
        for p in patches:
            p.start()
        threads = [
            threading.Thread(target=stage_main, args=(stage,), daemon=True)
            for stage in self.stages
        ]
        try:
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + timeout
            while any(t.is_alive() for t in threads):
                if time.monotonic() > deadline:
                    self.stop.set()
                    raise TimeoutError("the toy ring did not finish: a deadlock")
                time.sleep(0.01)
        finally:
            self.stop.set()
            for thread in threads:
                thread.join(timeout=5)
            for p in patches:
                p.stop()
        if self.errors:
            raise self.errors[0]
