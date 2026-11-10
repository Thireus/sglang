"""The draft's window takes only committed rows. Real TargetHiddenKvInjector, both pools."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _injector(ring=None):
    runner_ids = SimpleNamespace(ids_in=lambda kind, cache_loc: cache_loc + 1000)
    return TargetHiddenKvInjector(
        draft_model=None,
        draft_model_runner=SimpleNamespace(
            runner_ids=runner_ids,
            token_to_kv_pool=SimpleNamespace(
                swa_req_ring_size=ring, sliding_window=128
            ),
        ),
        model_runner=None,
        device="cpu",
        verify_num_draft_tokens=6,
        block_pos_offsets=torch.arange(6),
    )


class TestCommitGate(CustomTestCase):
    def _loc(self, injector, commit_lens, width, ring=None):
        bs = len(commit_lens)
        cache_loc = torch.arange(bs * width, dtype=torch.int64)
        positions = torch.arange(bs * width, dtype=torch.int64) + 500
        pool = SimpleNamespace(
            swa_req_ring_size=ring,
            sliding_window=128,
            full_to_swa_index_mapping=None,
        )
        return injector.committed_swa_loc(
            pool=pool,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc.view(bs, width),
            commit_lens=torch.tensor(commit_lens, dtype=torch.int32),
            state_slot=(
                torch.arange(bs).repeat_interleave(width) if ring is not None else None
            ),
        ).view(bs, width)

    def test_rows_past_the_commit_are_suppressed(self):
        """Invariant 14: paged pool, a block's rows at any commit length."""
        loc = self._loc(_injector(), [1, 3, 6], 6)
        for row, commit in enumerate((1, 3, 6)):
            self.assertTrue((loc[row, :commit] >= 1000).all(), row)
            self.assertTrue((loc[row, commit:] == -1).all(), row)

    def test_one_row_of_a_plain_round_commits(self):
        """Invariant 28: cache_loc_2d of shape (bs, 1), one row each, nothing suppressed."""
        loc = self._loc(_injector(), [1, 1], 1)
        self.assertEqual(loc.shape, (2, 1))
        self.assertTrue((loc >= 1000).all())

    def test_ring_pool_gates_too(self):
        loc = self._loc(_injector(ring=134), [2, 6], 6, ring=134)
        self.assertTrue((loc[0, 2:] == -1).all())
        self.assertTrue((loc[0, :2] >= 0).all())
        self.assertTrue((loc[1] >= 0).all())


if __name__ == "__main__":
    unittest.main()
