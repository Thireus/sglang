"""Sliding-window tails a request has gone past move to the SWA LRU's eviction end.

A prefix is reusable only up to a node whose last window still holds live SWA. A
chunked prefill publishes such a tail at every chunk boundary, at MRU, so on a
small paged SWA pool (DeepSeek-V4.1: 64,512 slots per rank) one long prefill
evicts the turn-end tails of every other conversation. Their next turn then
matches only up to the deepest tail still live, the branch point of a shared
system prompt (2,560 tokens here, as in production), and prefills the rest again.

The tests drive the real UnifiedRadixCache (FULL + SWA, Python tree core) and a
real SWATokenToKVPoolAllocator on CPU through the scheduler's own calls: the
admission match and lock, paged extend allocation with eviction,
checkpoint_kv_cache after every chunk and after the last one, decode-window
eviction, release_kv_cache. Geometry as on DeepSeek-V4.1: page 256, window 128,
chunk 2,048. The switches are read when the cache is built.
"""

import os
import random
import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.allocation import (
    alloc_paged_token_slots_extend,
    retire_window,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE, WINDOW, CHUNK = 256, 128, 2048
SWITCHES_ON = {
    "SGLANG_SWA_DEMOTE_CHUNK_TAILS": "1",
    "SGLANG_SWA_DEMOTE_SUPERSEDED_TAILS": "1",
}
CHUNK_TAILS_ONLY = {
    "SGLANG_SWA_DEMOTE_CHUNK_TAILS": "1",
    "SGLANG_SWA_DEMOTE_SUPERSEDED_TAILS": "0",
}
SWITCHES_OFF = {
    "SGLANG_SWA_DEMOTE_CHUNK_TAILS": "0",
    "SGLANG_SWA_DEMOTE_SUPERSEDED_TAILS": "0",
}


def _page_floor(n):
    return n // PAGE * PAGE


class _Engine:
    """One request at a time, through the calls the scheduler makes."""

    def __init__(self, switches, *, swa_slots, full_slots=1 << 19):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE)
        )
        self.pool = ReqToTokenPool(
            size=4, max_context_len=1 << 17, device="cpu", enable_memory_saver=False
        )
        kv = SWAKVPool(
            size=full_slots,
            size_swa=swa_slots,
            page_size=PAGE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=full_slots,
            size_swa=swa_slots,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=self.pool,
            token_to_kv_pool_allocator=self.alloc,
            page_size=PAGE,
            disable=False,
            sliding_window_size=WINDOW,
            tree_slots=tree_slots_of(
                (ComponentType.FULL, ComponentType.SWA), window=WINDOW
            ),
            eviction_policy="lru",
            chunked_prefill_size=CHUNK,
        )
        with (
            mock.patch.dict(os.environ, switches),
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
        ):
            self.cache = UnifiedRadixCache(params=params)
        self.full_total = self.alloc.full_available_size()
        self.swa_total = self.alloc.swa_available_size()
        self._rid = 0

    def peek(self, tokens):
        result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        return result.device_prefix_len, result.full_kv_hit_length

    def _extend(self, req, start, end):
        row = self.pool.req_to_token[req.kv.req_pool_idx]
        prefix = row[:start].to(torch.int64)
        last_loc = prefix[-1:] if start else torch.full((1,), -1, dtype=torch.int64)
        prefix_lens = torch.tensor([start], dtype=torch.int64)
        seq_lens = torch.tensor([end], dtype=torch.int64)
        out = alloc_paged_token_slots_extend(
            self.cache,
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            end - start,
        )
        self.pool.write((req.kv.req_pool_idx, slice(start, end)), out.to(torch.int32))
        req.kv.kv_allocated_len = req.kv.kv_committed_len = end

    def run_turn(self, prompt, output, on_checkpoint=None):
        """Admission, chunked prefill, decode, finish. Returns the matched length."""
        self._rid += 1
        req = Req(
            rid=f"r{self._rid}",
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=len(output)),
        )
        req.output_ids = array("q")
        req.extra_key = None
        req.lock = None
        req.init_next_round_input(self.cache)
        hit = req.prefix_len
        req.extend_end = min(len(prompt), req.prefix_len + CHUNK)
        req.lock = self.cache.lock(req.last_node)
        prefix = (
            self.cache.prefix_device_indices(req)
            if req.prefix_len
            else torch.empty(0, dtype=torch.int64)
        )
        self.pool.alloc([req])
        if req.prefix_len:
            self.pool.write(
                (req.kv.req_pool_idx, slice(0, req.prefix_len)), prefix.to(torch.int32)
            )
        while True:
            self._extend(req, req.prefix_len, req.extend_end)
            if req.extend_end == len(prompt):
                break
            checkpoint_kv_cache(req, self.cache)  # Scheduler.stash_chunked_request
            if on_checkpoint:
                on_checkpoint(req, "middle")
            req.extend_end = min(len(prompt), req.prefix_len + CHUNK)
        req.output_ids.append(output[0])
        checkpoint_kv_cache(req, self.cache)  # after the last chunk's forward
        if on_checkpoint:
            on_checkpoint(req, "prompt_end")
        done = 1
        while done < len(output):  # decode a page at a time, maybe_evict_swa's gate
            step = min(PAGE, len(output) - done)
            start = len(prompt) + done - 1
            req.output_ids.extend(output[done : done + step])
            req.refresh_fill_ids()
            self._extend(req, start, start + step)
            done += step
            seqlen = len(prompt) + done
            evicted = req.kv.get_evicted_seqlen(ComponentType.SWA)
            if seqlen - 1 - WINDOW >= evicted + envs.SGLANG_SWA_EVICTION_INTERVAL.get():
                retire_window(req, seqlen - 1, self.cache)
        if req.kv.kv_committed_len < len(prompt) + len(output):
            self._extend(req, req.kv.kv_committed_len, len(prompt) + len(output))
        req.finished_reason = FINISH_LENGTH(length=len(output))
        release_kv_cache(req, self.cache, checkpoint=True)
        return hit

    def check_conservation(self, test):
        test.assertEqual(
            self.alloc.full_available_size() + self.cache.full_evictable_size(),
            self.full_total,
        )
        test.assertEqual(
            self.alloc.swa_available_size() + self.cache.swa_evictable_size(),
            self.swa_total,
        )
        self.cache.sanity_check()

    def swa_lru_node_ids(self):
        """SWA LRU from MRU to the eviction end."""
        lru = self.cache.tree_core.lru_lists[ComponentType.SWA]
        pt, out, cur = lru._pt, [], lru.head.lru_next[lru._pt]
        while cur is not lru.tail:
            out.append(cur)
            cur = cur.lru_next[pt]
        return out


def _tokens(rng, n, base):
    return [base + rng.randrange(1, 50_000) for _ in range(n)]


class TestSWAPassedTailDemotion(unittest.TestCase):
    def _flood(self, switches):
        """Conversation A, then six others sharing its 2,600-token system prompt,
        then A's next turn. Returns (A's next-turn match, A's cached Full length,
        A's previous sequence length)."""
        rng = random.Random(0)
        eng = _Engine(switches, swa_slots=16384)
        trunk = _tokens(rng, 2600, 0)
        prompt_a = trunk + _tokens(rng, 12000 - 2600, 100_000)
        out_a = _tokens(rng, 300, 200_000)
        self.assertEqual(eng.run_turn(prompt_a, out_a), 0)
        next_a = prompt_a + out_a + _tokens(rng, 200, 300_000)
        others = [
            eng.run_turn(
                trunk + _tokens(rng, 12000 - 2600, 400_000 + 100_000 * i),
                _tokens(rng, 300, 900_000),
            )
            for i in range(6)
        ]
        self.assertEqual(others, [2048] + [2560] * 5)
        _, full_hit = eng.peek(next_a)
        hit = eng.run_turn(next_a, _tokens(rng, 300, 950_000))
        eng.check_conservation(self)
        return hit, full_hit, len(prompt_a) + len(out_a)

    def test_flood_keeps_the_turn_end_tail(self):
        hit, _, prev = self._flood(SWITCHES_ON)
        self.assertEqual(hit, _page_floor(prev))

    def test_flood_without_the_switches_caps_at_the_shared_prefix(self):
        # The stock order: A's Full KV is still cached, its SWA tails are not.
        hit, full_hit, prev = self._flood(SWITCHES_OFF)
        self.assertEqual(hit, 2560)
        self.assertGreaterEqual(full_hit, _page_floor(prev) - PAGE)

    def _idle(self, switches):
        """Conversation A goes idle while five others take eight turns each."""
        rng = random.Random(1)
        eng = _Engine(switches, swa_slots=16384)
        trunk = _tokens(rng, 2600, 0)
        prompt_a = trunk + _tokens(rng, 8000 - 2600, 100_000)
        out_a = _tokens(rng, 400, 200_000)
        eng.run_turn(prompt_a, out_a)
        history = {
            i: trunk + _tokens(rng, 6000 - 2600, 1_000_000 * (i + 1)) for i in range(5)
        }
        for turn in range(8):
            for i in range(5):
                base = 1_000_000 * (i + 1) + 10_000 * (turn + 1)
                prompt = history[i] + _tokens(rng, 600, base)
                out = _tokens(rng, 400, base + 5000)
                hit = eng.run_turn(prompt, out)
                if turn:
                    self.assertGreaterEqual(hit, _page_floor(len(history[i])) - PAGE)
                history[i] = prompt + out
        hit = eng.run_turn(prompt_a + out_a + _tokens(rng, 50, 300_000), [7] * 8)
        eng.check_conservation(self)
        return hit, len(prompt_a) + len(out_a)

    def test_idle_conversation_keeps_its_tail(self):
        hit, prev = self._idle(SWITCHES_ON)
        self.assertEqual(hit, _page_floor(prev))

    def test_idle_conversation_needs_superseded_tails_moved(self):
        # Chunk tails alone: every finished turn of the active conversations keeps
        # its turn-end tails at MRU and pushes out A's (and, with no new
        # conversation to touch it, the shared prefix's too).
        hit, prev = self._idle(CHUNK_TAILS_ONLY)
        self.assertLessEqual(hit, 2560)
        self.assertGreater(prev, 8000)

    def test_chunk_tails_stay_matchable_until_evicted(self):
        for switches in (SWITCHES_ON, SWITCHES_OFF):
            with self.subTest(switches=switches):
                rng = random.Random(2)
                eng = _Engine(switches, swa_slots=65536)
                prompt = _tokens(rng, 40000, 0)
                eng.run_turn(prompt, _tokens(rng, 100, 100_000))
                fork = prompt[:20000] + _tokens(rng, 3000, 200_000)
                # The last chunk boundary at or before the fork, 9 chunks in.
                self.assertEqual(eng.run_turn(fork, [5] * 8), 9 * CHUNK)
                eng.check_conservation(self)

    def test_lru_order_and_locks(self):
        rng = random.Random(3)
        eng = _Engine(SWITCHES_ON, swa_slots=65536)
        trunk = _tokens(rng, 2600, 0)
        eng.run_turn(trunk + _tokens(rng, 3000, 100_000), _tokens(rng, 50, 150_000))
        prompt = trunk + _tokens(rng, 11000, 200_000)
        seen = []
        tree = eng.cache.tree_core

        def tail_ends(nodes):
            ends = []
            for node in nodes:
                end, cur = 0, node
                while cur is not tree.root_node:
                    end += len(cur.key)
                    cur = cur.parent
                ends.append(end)
            return ends

        def on_checkpoint(req, kind):
            lru = eng.swa_lru_node_ids()
            locked = [n for n in lru if n.component_data[ComponentType.SWA].lock_ref]
            free = [n for n in lru if not n.component_data[ComponentType.SWA].lock_ref]
            seen.append(
                (kind, req.kv.cache_protected_len, tail_ends(locked), tail_ends(free))
            )

        hit = eng.run_turn(prompt, _tokens(rng, 300, 300_000), on_checkpoint)
        # The first conversation's chunk tail; the Full KV reaches 2,560, so the
        # first checkpoint publishes a branch tail there.
        self.assertEqual(hit, 2048)
        self.assertEqual(
            [(kind, protected) for kind, protected, _, _ in seen],
            [("middle", p) for p in (2560, 6144, 8192, 10240, 12288)]
            + [("prompt_end", _page_floor(len(prompt)))],
        )
        for i, (kind, protected, locked_ends, free_ends) in enumerate(seen):
            # The request's own window stays locked where it ends.
            self.assertIn(protected, locked_ends)
            if i == 1:
                # A branch tail is shared: it keeps its place.
                self.assertNotIn(2560, free_ends[-4:])
            elif i > 1:
                # The previous chunk's tail is next out (after the superseded
                # tails at the prompt end).
                self.assertIn(seen[i - 1][1], free_ends[-5:])
            if kind == "middle":
                # Nothing at or below the admission match moves for a chunk.
                self.assertNotIn(2048, free_ends[-4:])
        # Once the prompt is in, the tails behind the admission match on the single
        # path from the root go too; the branch tail at 2,560 still stays.
        free_ends = seen[-1][3]
        self.assertIn(2048, free_ends[-4:])
        self.assertNotIn(2560, free_ends[-8:])
        eng.check_conservation(self)

    def test_switches_off_move_nothing(self):
        for switches, expect_calls in ((SWITCHES_OFF, False), (SWITCHES_ON, True)):
            with self.subTest(switches=switches):
                rng = random.Random(4)
                eng = _Engine(switches, swa_slots=65536)
                tree = eng.cache.tree_core
                with (
                    mock.patch.object(
                        tree, "demote_window_tails", wraps=tree.demote_window_tails
                    ) as window,
                    mock.patch.object(
                        tree,
                        "demote_superseded_tails",
                        wraps=tree.demote_superseded_tails,
                    ) as superseded,
                ):
                    prompt = _tokens(rng, 9000, 0)
                    out = _tokens(rng, 200, 100_000)
                    eng.run_turn(prompt, out)
                    eng.run_turn(prompt + out + _tokens(rng, 3000, 200_000), [3] * 8)
                self.assertEqual(window.called, expect_calls)
                self.assertEqual(superseded.called, expect_calls)
                eng.check_conservation(self)


if __name__ == "__main__":
    unittest.main()
