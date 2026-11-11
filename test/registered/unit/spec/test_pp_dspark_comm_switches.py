"""DSpark under pipeline parallelism with the two hand-off switches on, on three stages
in one process (pp_dspark_toy_ring.py: the real event loop, relay and accept).

SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1 takes a lone middle prefill chunk's output
off the ring; SGLANG_PP_COMM_OVERLAP=1 moves the sends and receives to their own
stream (here a logging stand-in, the toy has no device). Right means: every request
still gets plain greedy, the skip really happened, and on every stage each forward
that follows a replayed forward's proxy send waits for that send.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pp_dspark_toy_ring as toy_ring
from pp_dspark_toy_ring import ToyRing, plain_output
from test_pp_dspark_relay import _requests

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _Ev:
    def __init__(self, dev, n):
        self.dev, self.name = dev, f"ev{n}"

    def record(self, stream=None):
        s = stream if stream is not None else self.dev.current
        self.dev.log.append(("record", s.name, self.name))

    def synchronize(self):
        pass


class _St:
    def __init__(self, dev, name):
        self.dev, self.name = dev, name

    def wait_event(self, ev):
        self.dev.log.append(("wait_event", self.name, ev.name))

    def wait_stream(self, other):
        self.dev.log.append(("wait_stream", self.name, other.name))


class _Dev:
    def __init__(self):
        self.log, self.n = [], 0
        self.schedule, self.forward, self.comm = (
            _St(self, "schedule"),
            _St(self, "forward"),
            _St(self, "comm"),
        )
        self.current = self.schedule

    def Event(self):
        self.n += 1
        return _Ev(self, self.n)

    def Stream(self):
        return _St(self, "side")

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, s):
        prev, self.current = self.current, s
        try:
            yield
        finally:
            self.current = prev


class _Ctx:
    """A reusable context (init_pp_loop_state builds one and reuses it)."""

    def __init__(self, dev, stream):
        self.dev, self.stream, self.prev = dev, stream, []

    def __enter__(self):
        self.prev.append(self.dev.current)
        self.dev.current = self.stream

    def __exit__(self, *exc):
        self.dev.current = self.prev.pop()
        return False


def _run(ring, overlap, graph_rounds):
    """Run the ring with the overlap's stream stand-in on every stage. graph_rounds:
    decode and verify forwards report a CUDA graph replay, as the engine's do."""
    devs = {}
    real_init = toy_ring.ToyStage.init_pp_loop_state
    real_run_batch = toy_ring.ToyStage.run_batch
    real_send_proxy = SchedulerPPMixin._pp_send_proxy_to_next_stage
    real_commit = SchedulerPPMixin._pp_commit_proxy_send_work
    real_skip = SchedulerPPMixin._pp_make_skip_output_result
    skips = []

    def init(self):
        real_init(self)
        if overlap:
            dev = devs[self.rank] = _Dev()
            self.device_module = dev
            self.schedule_stream, self.forward_stream = dev.schedule, dev.forward
            self.forward_stream_ctx = _Ctx(dev, dev.forward)
            self.copy_stream = _St(dev, "copy")
            self.copy_stream_ctx = _Ctx(dev, self.copy_stream)
            self.pp_comm_stream = dev.comm
            self.pp_comm_stream_ctx = _Ctx(dev, dev.comm)
        else:
            # A skipped output's done event waits on a side stream of its own.
            self.device_module.Stream = lambda: SimpleNamespace(
                wait_event=lambda event: None
            )

    def log(self, *entry):
        if overlap:
            self.device_module.log.append(entry)

    def run_batch(self, batch, pp_proxy_tensors):
        # inside _pp_launch_batch, after the forward stream's waits
        log(self, "forward", batch.forward_mode.name)
        return real_run_batch(self, batch, pp_proxy_tensors)

    def send_proxy(self, result):
        batch = self.cur_batch_for_debug
        if graph_rounds and not batch.forward_mode.is_extend():
            result.can_run_cuda_graph = True
        log(self, "proxy", batch.forward_mode.name, bool(result.can_run_cuda_graph))
        return real_send_proxy(self, result)

    def commit(self):
        had = bool(self.send_proxy_work)
        real_commit(self)
        ev = self.pp_send_done_event
        if had:
            log(self, "commit", None if ev is None else ev.name)

    def skip(self, batch, meta):
        skips.append((self.rank, batch.reqs[0].rid))
        return real_skip(self, batch, meta)

    with (
        mock.patch.object(toy_ring.ToyStage, "init_pp_loop_state", init),
        mock.patch.object(toy_ring.ToyStage, "run_batch", run_batch),
        mock.patch.object(SchedulerPPMixin, "_pp_send_proxy_to_next_stage", send_proxy),
        mock.patch.object(SchedulerPPMixin, "_pp_commit_proxy_send_work", commit),
        mock.patch.object(SchedulerPPMixin, "_pp_make_skip_output_result", skip),
    ):
        ring.run()
    return devs, skips


def _fence_violations(log):
    """After a graph proxy send, its commit records an event on the comm stream and the
    next forward waits for it before it runs; after an eager proxy send, the commit
    records none."""
    bad, proxy, fence, waits = [], None, None, []
    for e in log:
        if e[0] == "proxy":
            proxy, fence = e[2], None
        elif e[0] == "commit":
            fence = e[1]
            if proxy is True and fence is None:
                bad.append(("no fence after a graph proxy", e))
            if proxy is False and fence is not None:
                bad.append(("a fence after an eager proxy", e))
            waits = []
        elif e[0] == "wait_event" and e[1] == "forward":
            waits.append(e[2])
        elif e[0] == "forward":
            if fence is not None and fence not in waits:
                bad.append(("forward ran before its fence", e[1]))
            proxy, fence, waits = None, None, []
    return bad


class TestCommSwitchesOnTheRing(CustomTestCase):
    def setUp(self):
        for env in (
            envs.SGLANG_ENABLE_PP_SPEC.override(True),
            envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(True),
            envs.SGLANG_PP_COMM_OVERLAP.override(True),
        ):
            env.__enter__()
            self.addCleanup(env.__exit__, None, None, None)

    def _check(self, ring, requests):
        for rank, stage in enumerate(ring.stages):
            for r in requests:
                self.assertEqual(
                    stage.finished_rids.get(r["rid"]),
                    plain_output(r["prompt"], r["max_new"], r.get("stop")),
                    f"stage {rank}, {r['rid']}",
                )

    def test_skip_and_overlap_keep_plain_greedy(self):
        """Chunked prompts, launch first, both switches: plain greedy on every stage, the
        skip taken on every stage for middle chunks, graph proxies fenced."""
        for launch_first in (False, True):
            with self.subTest(launch_first=launch_first):
                requests = _requests(4, arrival_gap=5, max_new=30, prompt_len=17)
                ring = ToyRing(
                    requests,
                    accuracy=0.85,
                    chunk=4,
                    seed=7,
                    max_micro_batch=3,
                    launch_first=launch_first,
                )
                with envs.SGLANG_PP_DSPARK_LAUNCH_FIRST.override(launch_first):
                    devs, skips = _run(ring, overlap=True, graph_rounds=True)
                self._check(ring, requests)
                self.assertEqual({rank for rank, _ in skips}, {0, 1, 2})
                for rank, dev in devs.items():
                    if rank == 2:
                        continue  # the last stage sends no proxy
                    self.assertEqual(_fence_violations(dev.log), [], f"stage {rank}")
                    kinds = {e[1:] for e in dev.log if e[0] == "proxy"}
                    self.assertIn(("EXTEND", False), kinds)
                    self.assertTrue(any(k[1] for k in kinds))

    def test_skip_alone_keeps_plain_greedy(self):
        """The skip without the overlap stand-in: the toy's own streams."""
        requests = _requests(3, arrival_gap=4, max_new=25, prompt_len=13)
        ring = ToyRing(requests, accuracy=0.9, chunk=3, seed=3, max_micro_batch=3)
        _, skips = _run(ring, overlap=False, graph_rounds=False)
        self._check(ring, requests)
        self.assertTrue(skips)


if __name__ == "__main__":
    unittest.main()
