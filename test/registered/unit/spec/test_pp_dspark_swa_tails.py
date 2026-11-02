"""031's cache behaviour with the speculative reserve, and one slot id per request row.

DSpark's decode prep holds a page past the committed length and a verify commits one to
six tokens. The flood and idle scenarios of 031 must come out to the token, on the real
cache, allocator and tree.
"""

import importlib.util
import random
import unittest
from pathlib import Path

import torch

from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

RESERVE = 12


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "mem_cache"
        / "test_swa_passed_tail_demotion.py"
    )
    spec = importlib.util.spec_from_file_location("dsv41_tail_engine_reserve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _module()


class _ReserveEngine(MODULE._Engine):
    """Decode as DSpark does: keep ``committed + 12`` rounded up to a page allocated, and
    commit a random one to six tokens a step."""

    def __init__(self, switches, *, swa_slots, seed=0):
        super().__init__(switches, swa_slots=swa_slots)
        self._rng = random.Random(seed)
        get_context().override(
            "test.dspark_reserve",
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=6,
        )

    def _extend(self, req, start, end):
        target = (end + RESERVE + MODULE.PAGE - 1) // MODULE.PAGE * MODULE.PAGE
        target = max(target, req.kv.kv_allocated_len)
        cur = req.kv.kv_allocated_len
        if target > cur and cur >= end:
            super()._extend(req, cur, target)
        elif target > start and cur < end:
            super()._extend(req, max(start, cur), target)
        req.kv.kv_committed_len = end


class TestReserveKeepsThePrefixCache(CustomTestCase):
    def _flood(self, switches, engine_cls):
        rng = random.Random(0)
        eng = engine_cls(switches, swa_slots=16384)
        trunk = MODULE._tokens(rng, 2600, 0)
        prompt_a = trunk + MODULE._tokens(rng, 12000 - 2600, 100_000)
        out_a = MODULE._tokens(rng, 300, 200_000)
        eng.run_turn(prompt_a, out_a)
        next_a = prompt_a + out_a + MODULE._tokens(rng, 200, 300_000)
        for i in range(6):
            eng.run_turn(
                trunk + MODULE._tokens(rng, 12000 - 2600, 400_000 + 100_000 * i),
                MODULE._tokens(rng, 300, 900_000),
            )
        hit = eng.run_turn(next_a, MODULE._tokens(rng, 300, 950_000))
        eng.check_conservation(self)
        return hit, len(prompt_a) + len(out_a)

    def test_flood_to_the_token(self):
        """Invariant 19: the next turn of A finds what it found without the reserve."""
        plain_hit, prev = self._flood(MODULE.SWITCHES_ON, MODULE._Engine)
        reserve_hit, _ = self._flood(MODULE.SWITCHES_ON, _ReserveEngine)
        self.assertEqual(reserve_hit, plain_hit)
        self.assertEqual(reserve_hit, MODULE._page_floor(prev))


class TestDraftSlotIdsAreTheTargets(CustomTestCase):
    def test_a_private_draft_pool_writes_the_targets_slot_ids(self):
        """Draft KV follows the target's window: the same ids off the same plan."""
        engine = MODULE._Engine({}, swa_slots=16384)
        kv = engine.alloc.get_kvcache()
        target = RunnerIds(
            req_to_token=engine.pool.req_to_token,
            token_to_kv_pool_allocator=engine.alloc,
            token_to_kv_pool=kv,
            page_size=MODULE.PAGE,
            device="cpu",
        )
        draft = RunnerIds(
            req_to_token=engine.pool.req_to_token,
            token_to_kv_pool_allocator=engine.alloc,
            token_to_kv_pool=kv,
            page_size=MODULE.PAGE,
            device="cpu",
            is_draft_worker=True,
        )
        window = torch.arange(600, 612, dtype=torch.int64)
        plan = target.plan(
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([10, 20]),
            seq_lens_cpu=torch.tensor([10, 20]),
            write_virtual=window,
            read_extent=6,
        )
        self.assertTrue(torch.equal(plan.write_ids(draft), window))
        self.assertTrue(torch.equal(plan.write_ids(target), window))


if __name__ == "__main__":
    unittest.main()
