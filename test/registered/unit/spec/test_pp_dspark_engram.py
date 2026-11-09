"""Engram history under the DSpark relay. CPU branch of the real hasher.

A verify hashes history plus chain and commits after the accept; a plain round hashes
and commits in the forward. Whatever mix a request sees, its history must equal the
last n - 1 committed tokens, and every hash must equal what one-by-one decode gives.
"""

import random
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 64
WIDTH = 6
NGRAM = 3
SLOTS = 4


def _hasher():
    hasher = EngramHasher.__new__(EngramHasher)
    nn.Module.__init__(hasher)
    hasher.max_ngram_size = NGRAM
    hasher.pad_id = 0
    hasher.image_token_id = None
    hasher.token_map = torch.arange(VOCAB)
    hasher.multipliers = torch.tensor([[3, 5, 7]])
    hasher.primes = torch.tensor([[[1009, 1013], [1019, 1021]]])
    hasher.offsets = torch.zeros(1, 4, dtype=torch.int64)
    hasher.init_history(num_req_slots=SLOTS, device="cpu")
    return hasher


def _decode(hasher, slot, token, pos):
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([slot]),
        positions=torch.tensor([pos]),
        out_cache_loc=torch.tensor([pos + 1]),
    )
    return hasher(torch.tensor([token]), batch)


def _verify(hasher, slot, chain, pos):
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        ragged_verify_layout=None,  # one equal block per request (upstream #42718)
        req_pool_indices=torch.tensor([slot]),
        positions=torch.arange(pos, pos + len(chain)),
        spec_info=DFlashVerifyInput(
            draft_token=torch.tensor(chain),
            positions=torch.arange(pos, pos + len(chain)),
            draft_token_num=len(chain),
        ),
        out_cache_loc=torch.arange(1, len(chain) + 1),
    )
    return hasher(torch.tensor(chain), batch)


class TestEngramCommitOnTheRelay(CustomTestCase):
    def test_history_after_relay_equals_single_stage_commit(self):
        """Invariant 10: random accepts and verify/plain mixes."""
        rng = random.Random(4)
        for trial in range(20):
            relayed, one_by_one = _hasher(), _hasher()
            tokens = [rng.randrange(1, VOCAB) for _ in range(200)]
            pos = 3
            while pos < 150:
                if rng.random() < 0.3:
                    # A plain round commits inside the forward.
                    _decode(relayed, 1, tokens[pos], pos)
                    _decode(one_by_one, 1, tokens[pos], pos)
                    pos += 1
                else:
                    k = rng.randrange(1, WIDTH + 1)
                    chain = tokens[pos : pos + k] + [
                        rng.randrange(1, VOCAB) for _ in range(WIDTH - k)
                    ]
                    _verify(relayed, 1, chain, pos)
                    relayed.commit_after_verify(
                        torch.tensor([chain]), torch.tensor([1]), torch.tensor([k])
                    )
                    for i in range(k):
                        _decode(one_by_one, 1, tokens[pos + i], pos + i)
                    pos += k
                self.assertEqual(
                    relayed.history[1].tolist(),
                    one_by_one.history[1].tolist(),
                    f"trial {trial} at {pos}",
                )
            self.assertEqual(relayed.history[1].tolist(), tokens[pos - NGRAM + 1 : pos])

    def test_verify_hash_equals_decode_hash(self):
        """Invariant 11: the chain's hash rows are what decoding those tokens gives."""
        rng = random.Random(6)
        for _ in range(20):
            verify, decode = _hasher(), _hasher()
            start = rng.randrange(1, VOCAB)
            history = [rng.randrange(1, VOCAB) for _ in range(NGRAM - 1)]
            for h in (verify, decode):
                h.history[2] = torch.tensor(history, dtype=h.history.dtype)
            chain = [start] + [rng.randrange(1, VOCAB) for _ in range(WIDTH - 1)]
            got = _verify(verify, 2, chain, 40)
            for i, tok in enumerate(chain):
                want = _decode(decode, 2, tok, 40 + i)
                self.assertTrue(torch.equal(got[i], want[0]), f"row {i}")

    def test_verify_does_not_commit_in_the_forward(self):
        """A rejected chain must leave no trace: only the accept commits."""
        hasher = _hasher()
        hasher.history[1] = torch.tensor([7, 8], dtype=hasher.history.dtype)
        _verify(hasher, 1, [1, 2, 3, 4, 5, 6], 10)
        self.assertEqual(hasher.history[1].tolist(), [7, 8])
        hasher.commit_after_verify(
            torch.tensor([[1, 2, 3, 4, 5, 6]]), torch.tensor([1]), torch.tensor([1])
        )
        self.assertEqual(hasher.history[1].tolist(), [8, 1])


class TestStageTwoHistory(CustomTestCase):
    """Invariant 12, as QA corrected it: the last stage has a hasher and a history, written
    in every forward and never read there, and the earlier stages' are their own."""

    # text_config.engram_layer_ids of DeepSeek-V4.1-Flash-1017.
    ENGRAM_LAYERS = (1, 14)

    def test_no_layer_on_the_last_stage_reads_hash_ids(self):
        ids = self.ENGRAM_LAYERS
        for split in ((14, 13, 13), (13, 13, 14), (13, 14, 13)):
            stages = _ranges(split)
            self.assertFalse(
                [i for i in ids if i in stages[2]],
                f"split {split} puts an Engram layer on the last stage",
            )

    def test_earlier_histories_do_not_depend_on_the_last_stage(self):
        rng = random.Random(9)
        stage0, stage2 = _hasher(), _hasher()
        chain = [rng.randrange(1, VOCAB) for _ in range(WIDTH)]
        stage0.history[1] = torch.tensor([5, 6], dtype=stage0.history.dtype)
        before = stage0.history[1].tolist()
        _verify(stage2, 1, chain, 8)
        stage2.commit_after_verify(
            torch.tensor([chain]), torch.tensor([1]), torch.tensor([3])
        )
        self.assertEqual(stage0.history[1].tolist(), before)
        stage0.commit_after_verify(
            torch.tensor([chain]), torch.tensor([1]), torch.tensor([3])
        )
        self.assertEqual(stage0.history[1].tolist(), stage2.history[1].tolist()[-2:])


def _ranges(split):
    out, start = [], 0
    for size in split:
        out.append(range(start, start + size))
        start += size
    return out


if __name__ == "__main__":
    unittest.main()
