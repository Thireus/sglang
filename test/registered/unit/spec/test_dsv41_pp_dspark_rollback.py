"""No rollback: everything a verify writes is position-addressed and overwritten before
anyone can read it. A torch reference of V4.1's addressing, per stage.

The reference writes what the kernels write and where (deepseek_v4/c2.cuh for ratio 2:
even rows store (kv, score) in the request's ring at ``pos % ring``; odd rows pair with
the block's previous row, or the ring for a block's first row, and store at
``raw_out_loc // 2``; ratio 1 stores at ``raw_out_loc``; index keys follow the KV slot,
-1 and 0 suppressed). The GPU kernels themselves are for the window's 5090 run.
"""

import random
import unittest

from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WIDTH = 6
RING = get_compress_state_ring_size(2, WIDTH)


def row_value(pos, token):
    return (pos, token)


class Stage:
    """One stage's rows: SWA per layer, compressed KV and index keys per source."""

    def __init__(self, sources):
        self.swa = {}
        self.sources = {layer: dict(ratio=ratio) for layer, ratio in sources.items()}
        for s in self.sources.values():
            s.update(kv={}, index={}, ring={})

    def write_row(self, pos, token, partner_in_block):
        value = row_value(pos, token)
        self.swa[pos] = value
        for s in self.sources.values():
            if s["ratio"] == 1:
                s["kv"][pos] = value
                s["index"][pos] = value
                continue
            if pos % 2 == 0:
                s["ring"][pos % RING] = value
                continue
            if partner_in_block is not None:
                partner = partner_in_block
            else:
                partner = s["ring"].get((pos - 1) % RING)
            latent = (partner, value)
            s["kv"][pos // 2] = latent
            s["index"][pos // 2] = latent

    def verify(self, pos, chain):
        for col, token in enumerate(chain):
            partner = row_value(pos + col - 1, chain[col - 1]) if col else None
            self.write_row(pos + col, token, partner)

    def plain(self, pos, token):
        self.write_row(pos, token, None)

    def assert_committed(self, tokens, end, *, reference):
        """Every position a reader can reach holds what one-by-one decode stored."""
        for pos in range(end):
            assert self.swa[pos] == row_value(pos, tokens[pos]), ("swa", pos)
        for layer, s in self.sources.items():
            ref = reference.sources[layer]
            for slot, value in ref["kv"].items():
                assert s["kv"][slot] == value, (layer, "kv", slot)
                assert s["index"][slot] == value, (layer, "index", slot)
            if s["ratio"] == 2 and end % 2 == 1:
                # The next block's first row, at an odd position, reads this.
                assert s["ring"][(end - 1) % RING] == ref["ring"][(end - 1) % RING], (
                    layer,
                    "pair ring",
                    end,
                )


# The KV sources each stage writes or replays at 14/13/13: ratio-2 on stage 0 and 1,
# the ratio-1 replay of 20 on stage 2. Layer 24 is an index source with no KV of its own.
STAGES = {
    "stage 0": {2: 2, 8: 2},
    "stage 1": {14: 2},
    "stage 2": {20: 1},
}


class TestNothingRollsBack(CustomTestCase):
    def test_ring_is_wider_than_the_block(self):
        self.assertEqual(RING, 8)
        self.assertGreater(RING, WIDTH + 1)

    def _run(self, seed, *, plain_share):
        rng = random.Random(seed)
        for name, sources in STAGES.items():
            stage, reference = Stage(sources), Stage(sources)
            tokens = [rng.randrange(1, 99) for _ in range(600)]
            pos = 0
            for i in range(0, 41):
                stage.plain(i, tokens[i])
                reference.plain(i, tokens[i])
            pos = 41
            while pos < 500:
                if rng.random() < plain_share:
                    stage.plain(pos, tokens[pos])
                    reference.plain(pos, tokens[pos])
                    pos += 1
                else:
                    k = rng.randrange(1, WIDTH + 1)
                    chain = tokens[pos : pos + k] + [1000 + i for i in range(WIDTH - k)]
                    stage.verify(pos, chain)
                    for j in range(k):
                        reference.plain(pos + j, tokens[pos + j])
                    pos += k
                stage.assert_committed(tokens, pos, reference=reference)
            self.assertGreater(pos, 400, name)

    def test_random_accepts_match_one_by_one_decode(self):
        """Invariant 13: random accept sequences, verify only."""
        for seed in range(5):
            self._run(seed, plain_share=0.0)

    def test_verify_and_plain_mixes_match_one_by_one_decode(self):
        """Invariant 13: draft_len 1 and 6 agree at ring 8 under any mix."""
        for seed in range(5):
            self._run(100 + seed, plain_share=0.4)

    def test_every_reader_looks_behind_its_row(self):
        """A verify row reads the window behind it and its own block's earlier rows. A
        rejected row of an earlier block is overwritten before a reader gets there."""
        rng = random.Random(3)
        stage = Stage({})
        tokens = [rng.randrange(1, 99) for _ in range(400)]
        for i in range(10):
            stage.plain(i, tokens[i])
        pos = 10
        while pos < 300:
            k = rng.randrange(1, WIDTH + 1)
            chain = tokens[pos : pos + k] + [1000 + i for i in range(WIDTH - k)]
            for col, token in enumerate(chain):
                for back in range(max(0, pos + col - 128), pos + col):
                    want = chain[back - pos] if back >= pos else tokens[back]
                    self.assertEqual(stage.swa[back], row_value(back, want))
                stage.write_row(pos + col, token, None)
            pos += k


if __name__ == "__main__":
    unittest.main()
