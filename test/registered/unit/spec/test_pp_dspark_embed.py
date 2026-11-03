"""The draft's input embedding on the last pipeline stage."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sglang.srt.speculative.dspark_components.dspark_pp_embed import (
    HostEmbeddingTable,
    PPDraftHostEmbedding,
    install_pp_draft_embedding,
    locate_embedding,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, HIDDEN = 32, 8


def _checkpoint(root, table):
    shard = "model-00002-of-00002.safetensors"
    save_file(
        {"embed.weight": table, "other.weight": torch.zeros(3, 3)},
        os.path.join(root, shard),
    )
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump(
            {
                "weight_map": {
                    "embed.weight": shard,
                    "other.weight": shard,
                    "mtp.0.embed.weight": shard,
                }
            },
            fh,
        )


class TestHostEmbedding(CustomTestCase):
    def setUp(self):
        self.table = torch.randn(
            VOCAB, HIDDEN, generator=torch.Generator().manual_seed(0)
        ).bfloat16()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        _checkpoint(self.dir.name, self.table)

    def test_rows_equal_the_table_bit_for_bit(self):
        """Invariant 21 on a toy table; the module below does it on the checkpoint."""
        shard, key = locate_embedding(self.dir.name)
        self.assertEqual(key, "embed.weight")
        table = HostEmbeddingTable(shard, key)
        rows = table.rows(list(range(VOCAB)))
        self.assertEqual(rows.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(rows.view(torch.uint16), self.table.view(torch.uint16)))

    def test_staged_anchors_then_mask_rows(self):
        module = PPDraftHostEmbedding(
            HostEmbeddingTable(*locate_embedding(self.dir.name)),
            max_rows=4 * 5,
            rows_per_request=5,
            mask_token_id=7,
            device="cpu",
        )
        module.stage_anchors(torch.tensor([3, 9]))
        out = module(torch.zeros(10, dtype=torch.int64))
        self.assertEqual(out.shape, (10, HIDDEN))
        self.assertTrue(torch.equal(out[0], self.table[3]))
        self.assertTrue(torch.equal(out[5], self.table[9]))
        for row in (1, 2, 3, 4, 6, 7, 8, 9):
            self.assertTrue(torch.equal(out[row], self.table[7]), row)
        # A smaller batch next: the rows it reads are its own.
        module.stage_anchors(torch.tensor([4]))
        out = module(torch.zeros(5, dtype=torch.int64))
        self.assertEqual(out.shape[0], 5)
        self.assertTrue(torch.equal(out[0], self.table[4]))

    def test_install_routes(self):
        draft = SimpleNamespace(embed_tokens=None)
        args = dict(
            draft_model=draft,
            folder=self.dir.name,
            max_bs=2,
            query_token_num=5,
            mask_token_id=1,
            device="cpu",
        )
        module = install_pp_draft_embedding(route="host", **args)
        self.assertIs(draft.embed_tokens, module)
        self.assertIsNone(install_pp_draft_embedding(route="gpu", **args))
        ids = torch.tensor([0, 5, 31])
        self.assertTrue(torch.equal(draft.embed_tokens(ids), self.table[ids]))
        self.assertFalse(draft.embed_tokens.weight.requires_grad)

    def test_only_a_two_dimensional_bf16_table_is_read(self):
        path = os.path.join(self.dir.name, "bad.safetensors")
        save_file({"embed.weight": torch.zeros(4, 4)}, path)
        with self.assertRaisesRegex(ValueError, "BF16"):
            HostEmbeddingTable(path, "embed.weight")


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_DSV41_CHECKPOINT"), "needs SGLANG_TEST_DSV41_CHECKPOINT"
)
class TestHostEmbeddingOnTheCheckpoint(CustomTestCase):
    def test_every_row_of_the_checkpoint(self):
        """Invariant 21: all 129,280 rows, against the table safetensors reads."""
        import safetensors

        shard, key = locate_embedding(os.environ["SGLANG_TEST_DSV41_CHECKPOINT"])
        with safetensors.safe_open(shard, framework="pt", device="cpu") as fh:
            want = fh.get_tensor(key)
        self.assertEqual(want.shape, (129280, 5120))
        table = HostEmbeddingTable(shard, key)
        self.assertEqual((table.num_rows, table.width), (129280, 5120))
        for start in range(0, table.num_rows, 16384):
            ids = list(range(start, min(start + 16384, table.num_rows)))
            self.assertTrue(
                torch.equal(table.rows(ids).view(torch.uint16), want[ids].view(torch.uint16)),
                start,
            )


if __name__ == "__main__":
    unittest.main()
