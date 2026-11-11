"""Order of the pipeline hand-offs with SGLANG_PP_COMM_OVERLAP=1, on stub streams.

With the overlap, sends and receives run on their own stream. A forward that
replayed a CUDA graph hands the next stage views of the graph's static outputs,
so the next forward on this stage must wait until NCCL has read them; an eager
forward (a prefill chunk) hands on fresh tensors and need not wait. A DSpark
verify round on an earlier stage replays a graph, so it keeps that wait.

The streams and events below only log what is asked of them; the order of the
log is the order the device would see.
"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler_pp_dspark
from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_can_skip_output_comm,
    _pp_exchange_outputs_before_forward,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Event:
    def __init__(self, dev, name):
        self.dev, self.name, self.stream = dev, name, None

    def record(self, stream=None):
        self.stream = stream if stream is not None else self.dev.current
        self.dev.log.append(("record", self.stream.name, self.name))

    def __repr__(self):
        return self.name


class _Stream:
    def __init__(self, dev, name):
        self.dev, self.name = dev, name

    def wait_event(self, event):
        self.dev.log.append(("wait_event", self.name, event.name))

    def wait_stream(self, other):
        self.dev.log.append(("wait_stream", self.name, other.name))


class _Device:
    """A device module that logs: current stream, events by name."""

    def __init__(self):
        self.log = []
        self.n = 0
        self.schedule = _Stream(self, "schedule")
        self.forward = _Stream(self, "forward")
        self.comm = _Stream(self, "comm")
        self.current = self.schedule

    def Event(self):
        self.n += 1
        return _Event(self, f"ev{self.n}")

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, s):
        prev, self.current = self.current, s
        try:
            yield
        finally:
            self.current = prev


class _Work:
    def __init__(self, dev, name):
        self.dev, self.name = dev, name

    def wait(self):
        self.dev.log.append(("nccl_wait", self.dev.current.name, self.name))


class _Stage(SchedulerPPMixin):
    @property
    def pp_comm_stream_ctx(self):
        # init_pp_loop_state builds one context; a fresh one per use here.
        if self.pp_comm_stream is None:
            return nullcontext()
        return self.device_module.stream(self.pp_comm_stream)


def _stage(overlap=True, last=False):
    dev = _Device()
    st = object.__new__(_Stage)
    sends = []

    def send_tensor_dict(tensor_dict, all_gather_group=None, async_send=True):
        name = f"send{len(sends)}:{tensor_dict['__msg_type__']}"
        sends.append(name)
        dev.log.append(("nccl_send", dev.current.name, name))
        return [P2PWork(work=_Work(dev, name), payload=None)]

    def recv_tensor_dict(all_gather_group=None):
        dev.log.append(("nccl_recv", dev.current.name, "proxy"))
        return {"__msg_type__": "proxy", "hidden_states": torch.zeros(2)}

    st.__dict__.update(
        device_module=dev,
        schedule_stream=dev.schedule,
        forward_stream=dev.forward,
        pp_comm_stream=dev.comm if overlap else None,
        pp_group=SimpleNamespace(
            is_first_rank=False,
            is_last_rank=last,
            send_tensor_dict=send_tensor_dict,
            recv_tensor_dict=recv_tensor_dict,
        ),
        attn_tp_group=None,
        send_proxy_work=[],
        send_proxy_requires_forward_fence=False,
        pp_send_done_event=None,
        pp_proxy_recv_event=None,
        launch_event=None,
        _pp_tensor_dict_inbox={},
    )
    return st, dev


def _forward(st, dev, label):
    """What _pp_launch_batch does before run_batch: the forward stream's waits."""
    with dev.stream(dev.forward):
        st._pp_wait_forward_dependencies()
        dev.log.append(("forward", "forward", label))
        ev = dev.Event()
        ev.record(dev.current)
    return ev


def _result(graph):
    return SimpleNamespace(
        pp_hidden_states_proxy_tensors=SimpleNamespace(
            tensors={"hidden_states": torch.zeros(2)}
        ),
        can_run_cuda_graph=graph,
    )


def _iteration_pair(st, dev, first_graph):
    """Iteration i: forward i, send its proxy. Iteration i+1: commit that send
    (event_loop_pp line order: commit proxy send work, then launch), forward i+1."""
    st.launch_event = _forward(st, dev, "i")
    st._pp_send_proxy_to_next_stage(_result(first_graph))
    st._pp_commit_proxy_send_work()
    _forward(st, dev, "i+1")
    return dev.log


def _index(log, entry):
    return log.index(entry)


class TestOverlapFenceOrder(CustomTestCase):
    def test_graph_proxy_send_fences_the_next_forward(self):
        """Case 1: a replayed forward's proxy is read by NCCL before the next
        forward on this stage starts; the wait goes through the comm stream."""
        st, dev = _stage(overlap=True)
        log = _iteration_pair(st, dev, first_graph=True)
        launch = st.launch_event.name
        send = _index(log, ("nccl_send", "comm", "send0:proxy"))
        self.assertLess(_index(log, ("wait_event", "comm", launch)), send)
        wait = _index(log, ("nccl_wait", "comm", "send0:proxy"))
        done = next(e for e in log[wait:] if e[0] == "record" and e[1] == "comm")[2]
        fwd2 = _index(log, ("forward", "forward", "i+1"))
        self.assertLess(_index(log, ("wait_event", "forward", done)), fwd2)
        # The schedule stream takes no dependency on the forward or the send.
        self.assertFalse([e for e in log if e[1] == "schedule" and e[0] != "record"])

    def test_eager_proxy_send_does_not_fence(self):
        """Case 2: a prefill chunk's proxy (eager, fresh tensors): the next forward
        waits for nothing the comm stream records."""
        st, dev = _stage(overlap=True)
        log = _iteration_pair(st, dev, first_graph=False)
        fwd2 = _index(log, ("forward", "forward", "i+1"))
        comm_events = {e[2] for e in log if e[0] == "record" and e[1] == "comm"}
        waits = [e for e in log[:fwd2] if e[0] == "wait_event" and e[1] == "forward"]
        self.assertFalse([e for e in waits if e[2] in comm_events])
        self.assertIsNone(st.pp_send_done_event)

    def test_dspark_verify_on_an_earlier_stage_is_fenced(self):
        """Case 3: the earlier stage's verify takes can_run_cuda_graph from
        prepare_for_verify, so a replayed verify sets the fence for the next forward."""
        for graph in (True, False):
            with self.subTest(graph=graph):
                st, dev = _stage(overlap=True)
                st.device = "cpu"
                st._pp_dspark_block_pos_offsets = torch.arange(6)
                fb = SimpleNamespace()
                verify_input = mock.Mock()
                verify_input.prepare_for_verify.return_value = (fb, graph)
                out = _result(None)
                st.tp_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(),
                    forward_batch_generation=mock.Mock(return_value=out),
                )
                batch = SimpleNamespace(seq_lens=torch.zeros(2))
                relay = SimpleNamespace(tokens=torch.zeros(2, 6, dtype=torch.int64))
                spec = SimpleNamespace(speculative_num_draft_tokens=6)
                with (
                    mock.patch.object(scheduler_pp_dspark, "get_spec", lambda: spec),
                    mock.patch.object(scheduler_pp_dspark, "relay_verify_window"),
                    mock.patch.object(
                        scheduler_pp_dspark,
                        "relay_verify_input",
                        return_value=verify_input,
                    ),
                ):
                    res = scheduler_pp_dspark._forward_verify_on_earlier_stage(
                        st, batch, relay, None
                    )
                self.assertIs(res.can_run_cuda_graph, graph)
                kw = st.tp_worker.forward_batch_generation.call_args.kwargs
                self.assertTrue(kw["is_verify"])
                st.launch_event = dev.Event()
                st._pp_send_proxy_to_next_stage(res)
                self.assertIs(st.send_proxy_requires_forward_fence, graph)
                st._pp_commit_proxy_send_work()
                self.assertEqual(st.pp_send_done_event is not None, graph)
                # The next forward on this stage waits for the replayed verify's
                # send, and only then.
                done = st.pp_send_done_event
                _forward(st, dev, "next")
                fwd = _index(dev.log, ("forward", "forward", "next"))
                fence = [
                    e
                    for e in dev.log[:fwd]
                    if e[0] == "wait_event"
                    and e[1] == "forward"
                    and done is not None
                    and e[2] == done.name
                ]
                self.assertEqual(len(fence), 1 if graph else 0)

    def test_verify_after_an_eager_extend_waits_only_for_its_input(self):
        """Case 4: a verify replay right after a prefill chunk's proxy send waits for
        its own proxy receive (comm stream) and the schedule stream, not the send."""
        st, dev = _stage(overlap=True)
        st.launch_event = _forward(st, dev, "extend")
        st._pp_send_proxy_to_next_stage(_result(False))
        st._pp_recv_proxy_tensors()
        recv_ev = st.pp_proxy_recv_event.name
        self.assertEqual(st.pp_proxy_recv_event.stream.name, "comm")
        st._pp_commit_proxy_send_work()
        log = dev.log
        _forward(st, dev, "verify")
        fwd = _index(log, ("forward", "forward", "verify"))
        waits = [e for e in log[:fwd] if e[1] == "forward" and e[0].startswith("wait")]
        self.assertEqual(
            waits[-2:],
            [
                ("wait_stream", "forward", "schedule"),
                ("wait_event", "forward", recv_ev),
            ],
        )

    def test_without_overlap_every_send_fences_through_the_schedule_stream(self):
        """Case 5, today's launch: no comm stream; the proxy send makes the schedule
        stream wait for the forward and for NCCL, and the next forward waits for the
        schedule stream, graph or not."""
        for graph in (True, False):
            with self.subTest(graph=graph):
                st, dev = _stage(overlap=False)
                log = _iteration_pair(st, dev, first_graph=graph)
                launch = st.launch_event.name
                self.assertIn(("wait_event", "schedule", launch), log)
                wait = _index(log, ("nccl_wait", "schedule", "send0:proxy"))
                fwd2 = _index(log, ("forward", "forward", "i+1"))
                self.assertLess(wait, fwd2)
                self.assertIn(("wait_stream", "forward", "schedule"), log[wait:fwd2])
                self.assertIsNone(st.pp_send_done_event)


class TestSwitchPredicates(CustomTestCase):
    def test_launch_first_holds_for_every_round_of_our_launch(self):
        """With the relay, launch-first and no async depth, no batch exchanges its
        outputs before its forward: the proxy commit is the only fence before a
        launch."""
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            batch = SimpleNamespace(forward_mode=mode, is_extend_in_batch=False)
            for last in (False, True):
                self.assertFalse(
                    _pp_exchange_outputs_before_forward(
                        cur_batch=batch,
                        spec_relay=True,
                        is_last_rank=last,
                        async_batch_depth=0,
                        launch_first=True,
                    )
                )

    def test_skip_engages_only_on_a_lone_middle_chunk(self):
        def b(mode, n=1, last=False, logprob=False):
            return SimpleNamespace(
                forward_mode=mode,
                reqs=[object()] * n,
                contains_last_prefill_chunk=last,
                return_logprob=logprob,
            )

        with envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(True):
            self.assertTrue(_pp_can_skip_output_comm(b(ForwardMode.EXTEND)))
            for x in (
                b(ForwardMode.EXTEND, n=2),
                b(ForwardMode.EXTEND, last=True),
                b(ForwardMode.EXTEND, logprob=True),
                b(ForwardMode.DECODE),
                b(ForwardMode.TARGET_VERIFY),
                b(ForwardMode.MIXED),
                None,
            ):
                self.assertFalse(_pp_can_skip_output_comm(x), x)
        with envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(False):
            self.assertFalse(_pp_can_skip_output_comm(b(ForwardMode.EXTEND)))


if __name__ == "__main__":
    unittest.main()
