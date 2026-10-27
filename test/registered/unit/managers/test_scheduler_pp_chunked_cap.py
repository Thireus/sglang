"""Under PP a chunked request finishes only in a micro-batch under the request cap.

Pipeline parallelism runs one running batch per micro-batch and caps each at
pp_max_micro_batch_size. A request split across prefill chunks is admitted by one
micro-batch, but it joins the running batch of the micro-batch whose pass runs its
last chunk, and that pass used to add the last chunk whatever the batch held: a
micro-batch at the cap went one over it, past the decode CUDA graphs captured for
the cap. The last chunk now waits for a micro-batch with room; earlier chunks run
anywhere, as before.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CAP = 9  # pp_max_micro_batch_size: --max-running-requests 27 over --pp-size 3
NUM_MICRO_BATCHES = 3
CHUNK = 2048
PAGE = 256


class _Req:
    """The fields the scheduler and the real PrefillAdder read from a request."""

    extend_len = Req.extend_len

    def __init__(self, rid, prompt_len=1028, prefix_len=0, holds_row=True):
        self.rid = rid
        self.full_untruncated_fill_ids = array("q", range(prompt_len))
        self.prefix_len = prefix_len
        self.extend_end = prefix_len
        self.output_ids = array("q")
        self.sampling_params = SimpleNamespace(max_new_tokens=512, ignore_eos=False)
        self.kv = SimpleNamespace(
            holds_kv=holds_row,
            holds_mamba=False,
            req_pool_idx=0 if holds_row else None,
        )
        self.retracted_stain = False
        self.inflight_middle_chunks = 0
        self.beam_group = None
        self.token_indices_to_pool = None
        self.next_round_inputs = 0

    def init_next_round_input(self, tree_cache=None):
        self.next_round_inputs += 1


def _tree_cache():
    cache = MagicMock()
    cache.evictable_size.return_value = 0
    cache.full_evictable_size.return_value = 0
    cache.swa_evictable_size.return_value = 0
    cache.supports_mamba.return_value = False
    cache.supports_prefix_sharing.return_value = False
    cache.disable = False
    cache.buffer_pipeline = None
    return cache


def _allocator():
    allocator = MagicMock()
    allocator.page_size = PAGE
    allocator.available_size.return_value = 1 << 22
    allocator.full_available_size.return_value = 1 << 22
    allocator.request_slot_reserve.return_value = 0
    allocator.create_prefill_budget.side_effect = lambda tree_cache, **kw: (
        PrefillBudget(allocator, tree_cache, **kw)
    )
    return allocator


def _adder(running_batch):
    return PrefillAdder(
        PAGE,
        _tree_cache(),
        _allocator(),
        running_batch,
        1.0,
        16384,
        CHUNK,
    )


def _running(n):
    return ScheduleBatch(reqs=[_Req(f"running-{i}", prompt_len=1) for i in range(n)])


def _chunked(remaining, done=768):
    # The state after the stash of the previous chunk: prefix_len == extend_end.
    return _Req("chunked", prompt_len=done + remaining, prefix_len=done)


class _PriorityReq(_Req):
    """A stub request with the fields priority preemption and admission read."""

    def __init__(self, rid, priority=0, **kwargs):
        super().__init__(rid, **kwargs)
        self.priority = priority
        self.time_stats = SimpleNamespace(wait_queue_entry_time=0.0)
        self.kv.mamba_cow_src_index = None
        self.kv.mamba_needs_clear = False
        self.kv.mamba_pool_idx = None
        self.last_node = None
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.auxiliary_swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        self.storage_hit_length = 0
        self.storage_hit_start = 0
        self.host_loaded_length = 0
        self.host_hit_is_storage = False
        self.cache_request_handle = None
        self.lora_id = None

    def finished(self):
        return False

    def needs_host_load_back(self):
        return False

    def materialized_host_hit_len(self):
        return 0

    def fulfilled_storage_hit_len(self, prefix_len):
        return 0


def _filter_reqs(batch, keep_indices=None, **kwargs):
    # Stub requests carry no tensors: keep the list operation only.
    if keep_indices is not None:
        batch.reqs = [batch.reqs[i] for i in keep_indices]


class _BatchBuilt(Exception):
    pass


class TestScheduler(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        override = get_context().override_server_args(pp_max_micro_batch_size=CAP)
        override.install()
        self.addCleanup(override.restore)

    def _scheduler(self, chunked_req, waiting=()):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = list(waiting)
        s.chunked_req = chunked_req
        s.min_free_slots_delayer = None
        s.policy = Mock()
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = CHUNK
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: PrefillAdder
        s.page_size = PAGE
        s.tree_cache = _tree_cache()
        s.token_to_kv_pool_allocator = _allocator()
        s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        s.max_prefill_tokens = 16384
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = CAP * NUM_MICRO_BATCHES
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: 4)
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        return s

    def _pass(self, s, running_batch):
        """One prefill pass of one micro-batch: the batch it builds, or None."""

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            try:
                ret = s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
            except _BatchBuilt as built:
                return built.args[0]
        self.assertIsNone(ret[0])
        self.assertIs(ret[1], running_batch)
        return None

    def test_last_chunk_waits_in_a_micro_batch_at_the_cap(self):
        req = _chunked(remaining=260)
        waiting = _Req("waiting", holds_row=False)
        s = self._scheduler(req, waiting=[waiting])
        running = _running(CAP)

        built = self._pass(s, running)

        self.assertIsNone(built, "the last chunk joined a micro-batch at the cap")
        self.assertIs(s.chunked_req, req)
        # Nothing ran for it, so the stash gate leaves its prefix alone next step.
        self.assertEqual(req.extend_end, req.prefix_len)
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertTrue(running.batch_is_full)
        self.assertEqual(s.waiting_queue, [waiting])
        self.assertEqual(waiting.next_round_inputs, 0)
        self.assertEqual(len(running.reqs), CAP)

    def test_last_chunk_finishes_in_a_micro_batch_with_room(self):
        req = _chunked(remaining=260)
        s = self._scheduler(req, waiting=[_Req("waiting", holds_row=False)])

        built = self._pass(s, _running(CAP - 1))

        self.assertEqual(built, [req])
        self.assertIsNone(s.chunked_req)
        self.assertEqual(req.extend_len, 260)

    def test_middle_chunk_still_runs_in_a_micro_batch_at_the_cap(self):
        req = _chunked(remaining=4000)
        s = self._scheduler(req)

        built = self._pass(s, _running(CAP))

        self.assertEqual(built, [req])
        self.assertIs(s.chunked_req, req)
        self.assertEqual(req.extend_len, CHUNK)
        self.assertEqual(req.inflight_middle_chunks, 1)

    def test_pipeline_puts_the_request_in_the_first_micro_batch_with_room(self):
        # Micro-batches 0 and 1 are at the cap and 2 has room, as when 25 streams
        # fill slots of 9, 9 and 7 while the 25th prompt is still being chunked.
        slots = [_running(CAP), _running(CAP), _running(CAP - 2)]
        req = _chunked(remaining=260)
        s = self._scheduler(req)
        placed = None
        for step in range(2 * NUM_MICRO_BATCHES):
            mb = step % NUM_MICRO_BATCHES
            built = self._pass(s, slots[mb])
            if built:
                # The next pass of this micro-batch merges every finished
                # prefill of its last batch into its running batch.
                slots[mb].reqs.extend(built)
                placed = mb
                break
        sizes = [len(b.reqs) for b in slots]
        self.assertLessEqual(max(sizes), CAP, f"a micro-batch over the cap: {sizes}")
        self.assertEqual(placed, 2)
        self.assertIsNone(s.chunked_req)
        self.assertEqual(sizes, [CAP, CAP, CAP - 1])

    def test_priority_preemption_waits_while_the_last_chunk_is_parked(self):
        # With priority preemption a full micro-batch could make room by
        # preempting, and the waiting request, longer than a chunk, would become
        # a second chunked request beside the parked one. The pass returns first.
        high = -10 if get_schedule().schedule_low_priority_values_first else 10
        req = _chunked(remaining=260)
        waiting = _PriorityReq(
            "waiting", priority=high, prompt_len=5000, holds_row=False
        )
        s = self._scheduler(req, waiting=[waiting])
        s.enable_priority_preemption = True
        s._add_request_to_queue = lambda r, is_retracted=False: s.waiting_queue.append(
            r
        )
        s.req_to_token_pool = SimpleNamespace(
            available_size=lambda: CAP * NUM_MICRO_BATCHES - CAP - 1
        )
        running = ScheduleBatch(
            reqs=[_PriorityReq(f"running-{i}", prompt_len=1) for i in range(CAP)]
        )
        members = list(running.reqs)

        with (
            patch.object(ScheduleBatch, "release_req"),
            patch.object(ScheduleBatch, "filter_batch", _filter_reqs),
        ):
            built = self._pass(s, running)

        self.assertIsNone(built, "the last chunk joined a micro-batch at the cap")
        self.assertIs(s.chunked_req, req)
        self.assertEqual(running.reqs, members, "a running request was preempted")
        self.assertEqual(s.waiting_queue, [waiting])
        self.assertTrue(running.batch_is_full)


class TestPrefillAdder(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_last_chunk_parks_when_it_cannot_finish(self):
        adder = _adder(_running(0))
        adder.chunked_req_can_finish = False
        req = _chunked(remaining=260)

        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.rem_chunk_tokens, CHUNK)
        self.assertEqual(req.extend_end, req.prefix_len)

    def test_middle_chunk_runs_when_the_request_cannot_finish(self):
        adder = _adder(_running(0))
        adder.chunked_req_can_finish = False
        req = _chunked(remaining=4000)

        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(req.extend_len, CHUNK)

    def test_last_chunk_finishes_by_default(self):
        adder = _adder(_running(0))
        req = _chunked(remaining=260)

        self.assertIsNone(adder.add_chunked_req(req))
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(req.extend_len, 260)


if __name__ == "__main__":
    unittest.main()
