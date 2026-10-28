"""DeepSeek-V4.1's sparse-attention selection comes out in a fixed order.

The ratio-1/2 index layers keep the index_topk (512) best compressed positions of
each query row. A row that sees at most k positions keeps all of them, and the
top-k v2 kernel writes those in order. A longer row gets the exact top-k set, but
each pick lands in whatever output slot its thread claims first, which changes
from launch to launch. The sparse attention reads the list in order, 64 entries a
tile, so two runs of the same ids attended in a different order past position 512
and gave different outputs. The selection is now sorted: a row over k positions
comes out ascending with the -1 padding last, the set is unchanged, and a row
with at most k positions is never touched, so nothing below position 512 changes.

CPU only. The kernels are stood in for by torch references that return the exact
top-k set in a shuffled order, as the kernel can; DeepSelect is asked for its
sorted_index output instead.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.deep_select as deep_select
import sglang.kernels.ops.attention.dsv4 as dsv4_ops
from sglang.srt.layers.attention.dsv4.v41_indexer import (
    dense_blocks,
    full_topk,
    scoring,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
    DenseBlocksBackend,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DeepGEMMPrefillData
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BLOCK = 8


def exact_topk(scores: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The k best of scores[:n] by value, ties to the lower position: the set the
    kernel selects."""
    return torch.sort(-scores[:n].float(), stable=True).indices[:k]


class RaggedTopK:
    """Stands in for topk_transform_ragged_v2. A row of at most k positions gets
    0, 1, ... then -1 (topk_v2.cuh, the seq_len <= topk path); a longer row gets
    its exact top-k set in a shuffled order, as the atomic slot claims leave it."""

    def __init__(self, seed: int = 0):
        self.gen = torch.Generator().manual_seed(seed)
        self.calls = 0

    def __call__(self, scores, seq_lens, *, out_offsets, out_indices, row_starts=None):
        assert row_starts is None
        self.calls += 1
        k = out_indices.shape[1]
        for b in range(out_indices.shape[0]):
            n, offset = int(seq_lens[b]), int(out_offsets[b])
            if n <= k:
                row = torch.full((k,), -1, dtype=torch.int64)
                row[:n] = torch.arange(n) + offset
            else:
                picks = exact_topk(scores[b], n, k)
                row = picks[torch.randperm(k, generator=self.gen)] + offset
            out_indices[b] = row.to(out_indices.dtype)


def flat_selection_slots(raw, starts, slots, pages):
    """Torch reference of the Triton kernel of the same name."""
    rows, k = raw.shape
    columns = (raw.to(torch.int64) + starts[:, None].to(torch.int64)).clamp_min(0)
    pages[:rows, :k] = torch.where(raw >= 0, slots[columns], -1).to(pages.dtype)


def make_batch(seq_lens, extend_lens, *, k, ratio=1, seed=0):
    """One prefill chunk: request r holds the query rows of positions
    [seq_lens[r] - extend_lens[r], seq_lens[r]), its flattened-K columns follow the
    earlier requests', and its index-K pool slots are scattered."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.cat([torch.arange(s - e, s) for s, e in zip(seq_lens, extend_lens)])
    rows = pos.numel()
    lens_per_request = [s // ratio for s in seq_lens]
    starts = [0]
    for n in lens_per_request[:-1]:
        starts.append(starts[-1] + n)
    columns = sum(lens_per_request)
    width = -(-max(lens_per_request) // 64) * 64
    k_slots = torch.randperm(4 * columns, generator=g)[:columns]
    data = DeepGEMMPrefillData(
        k_slots=k_slots,
        request_starts=torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32), torch.tensor(extend_lens)
        ),
        lens_per_request=lens_per_request,
        rows_per_request=list(extend_lens),
        compress_lens=((pos + 1) // ratio).to(torch.int32),
        q_fp4=torch.zeros(rows, 1, 64, dtype=torch.int8),
        q_sf=torch.zeros(rows, 1, dtype=torch.int32),
        weights=torch.zeros(rows, 1),
    )
    inputs = PrefillInputs(
        indexer=SimpleNamespace(index_topk=k),
        layer_id=20,
        compress_ratio=ratio,
        freqs_cis=None,
        x=None,
        q_lora=None,
        positions=pos,
        req_rows=None,
        req_pool_indices=None,
        k_slots=k_slots,
        index_plans=None,
        kv_page_table=None,
        seq_lens_cpu=list(seq_lens),
        rows_per_request=list(extend_lens),
        rows_per_request_device=None,
        out_raw_indices=torch.full((rows, k), -1, dtype=torch.int32),
        out_page_indices=torch.full((rows, k), -1, dtype=torch.int32),
    )
    logits = torch.randn(rows, width, generator=g)
    return data, inputs, logits


def tiles(logits, cuts):
    """score_tiles stand-in: rows [cuts[i], cuts[i + 1]) are one tile with its own
    copy of the scores, as the generator frees and rescores each tile."""

    def score_tiles(data, kv):
        for a, b in zip(cuts[:-1], cuts[1:]):
            yield slice(a, b), logits[a:b].clone()

    return score_tiles


POOL = SimpleNamespace(get_low_ratio_index_k_fp4=lambda layer_id, slots: (None, None))


class SelectionChecks:
    def assert_selection(self, selection, lens, k, sets, written):
        """Every row keeps its set; a row over k positions is ascending with the
        padding last; a row with at most k is exactly as the top-k wrote it."""
        for b in range(selection.shape[0]):
            row = selection[b].tolist()
            picked = [v for v in row if v >= 0]
            self.assertEqual(
                len(picked), len(sets[b]), f"row {b}: picks lost or doubled"
            )
            self.assertEqual(set(picked), sets[b], f"row {b}: the selected set changed")
            if int(lens[b]) > k:
                expected = sorted(picked) + [-1] * (k - len(picked))
                self.assertEqual(
                    row,
                    expected,
                    f"row {b} ({int(lens[b])} positions) is not ascending",
                )
            else:
                self.assertEqual(
                    row, written[b], f"row {b} ({int(lens[b])} positions) was touched"
                )

    def assert_slots(self, inputs, data):
        """The pool slots follow the positions entry by entry."""
        raw = inputs.out_raw_indices.to(torch.int64)
        columns = (raw + data.request_starts[:, None].to(torch.int64)).clamp_min(0)
        expected = torch.where(raw >= 0, data.k_slots[columns], -1).to(torch.int32)
        torch.testing.assert_close(inputs.out_page_indices, expected, rtol=0, atol=0)


def kernel_order(lens, k):
    """What the kernel writes for a row of at most k positions: 0, 1, ... then -1."""
    return [
        list(range(int(n))) + [-1] * (k - int(n)) if int(n) <= k else None for n in lens
    ]


class TestSortSelection(CustomTestCase):
    def test_long_rows_ascending_with_the_padding_last(self):
        sel = torch.tensor(
            [
                [5, 30, 3, 19, 41, 7, 2, 11],
                [5, -1, 3, 19, -1, 7, 2, 11],  # underfilled: the padding moves last
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        before = [set(r.tolist()) - {-1} for r in sel]
        scoring.sort_selection_(sel, torch.tensor([50, 20, 9], dtype=torch.int32))
        self.assertEqual(sel[0].tolist(), [2, 3, 5, 7, 11, 19, 30, 41])
        self.assertEqual(sel[1].tolist(), [2, 3, 5, 7, 11, 19, -1, -1])
        self.assertEqual(sel[2].tolist(), [-1] * 8)
        self.assertEqual([set(r.tolist()) - {-1} for r in sel], before)

    def test_rows_with_at_most_k_positions_are_untouched(self):
        # Rows in any order, including k positions exactly (a query at position
        # k - 1 on a ratio-1 layer): not one entry moves.
        sel = torch.tensor(
            [
                [3, 1, 2, -1, 0, -1, -1, -1],
                [7, 6, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        before = sel.clone()
        scoring.sort_selection_(sel, torch.tensor([4, 8, 6, 0], dtype=torch.int32))
        torch.testing.assert_close(sel, before, rtol=0, atol=0)

    def test_int64_positions_and_empty_selections(self):
        sel = torch.tensor([[9, 4, 7], [2, 0, 1]], dtype=torch.int64)
        scoring.sort_selection_(sel, torch.tensor([10, 3], dtype=torch.int32))
        self.assertEqual(sel.tolist(), [[4, 7, 9], [2, 0, 1]])
        for shape in ((0, 512), (3, 0)):
            empty = torch.empty(shape, dtype=torch.int32)
            scoring.sort_selection_(empty, torch.zeros(shape[0], dtype=torch.int32))


class TestFullTopKPrefill(SelectionChecks, CustomTestCase):
    """Layers 2, 8 and 14 on SM12x: DeepGEMM scores, top-k v2 (fp32 logits)."""

    def _run(self, seq_lens, extend_lens, k, cuts, seed=0):
        data, inputs, logits = make_batch(seq_lens, extend_lens, k=k, seed=seed)
        kernel = RaggedTopK(seed)
        indexer = FullTopKIndexer(
            token_to_kv_pool=POOL,
            req_to_token=None,
            use_deep_gemm_prefill=True,
            use_deep_gemm_decode=False,
        )
        with (
            patch.object(full_topk, "get_deep_gemm_prefill_data", return_value=data),
            patch.object(scoring, "score_tiles", tiles(logits, cuts)),
            patch.object(dsv4_ops, "topk_transform_ragged_v2", kernel),
            patch.object(scoring, "flat_selection_slots", flat_selection_slots),
        ):
            indexer.topk_prefill(inputs)
        self.assertEqual(kernel.calls, len(cuts) - 1)
        lens = data.compress_lens
        sets = [
            set(range(int(n)))
            if int(n) <= k
            else set(exact_topk(logits[b], int(n), k).tolist())
            for b, n in enumerate(lens)
        ]
        return data, inputs, lens, sets

    def test_positions_and_slots_come_out_ascending(self):
        k = 16
        # Request 0: positions 0-39 (rows 0-15 see at most 16 positions); request 1:
        # positions 20-29 of a 30-token sequence. Two score tiles, the cut inside request 0.
        data, inputs, lens, sets = self._run([40, 30], [40, 10], k, [0, 23, 50])
        self.assert_selection(
            inputs.out_raw_indices, lens, k, sets, kernel_order(lens, k)
        )
        self.assert_slots(inputs, data)

    def test_v41_boundary_at_position_512(self):
        # index_topk 512, ratio 1: the rows at positions 505-511 see 506-512
        # positions and are left as the kernel wrote them; 512-520 are sorted.
        k = 512
        data, inputs, lens, sets = self._run([521], [16], k, [0, 16], seed=3)
        self.assertEqual(lens.tolist(), list(range(506, 522)))
        self.assert_selection(
            inputs.out_raw_indices, lens, k, sets, kernel_order(lens, k)
        )
        self.assert_slots(inputs, data)
        for b in range(7):
            self.assertEqual(
                inputs.out_raw_indices[b, : int(lens[b])].tolist(),
                list(range(int(lens[b]))),
            )


class TestDenseBlocksPrefill(SelectionChecks, CustomTestCase):
    """Layer 20 (the candidate source: its own top-k plus the blocks it publishes)
    and layers 24-36 (the consumers: a top-k among the published blocks)."""

    K = 16
    TOPK_BLOCKS = 2048

    def _backend(self):
        return DenseBlocksBackend(
            token_to_kv_pool=POOL,
            req_to_token=None,
            candidate_topk_blocks=self.TOPK_BLOCKS,
            candidate_block_size=BLOCK,
            use_deep_gemm_prefill=True,
        )

    def _publish(self, backend, data, inputs, logits, cuts, seed=0):
        kernel = RaggedTopK(seed)
        with (
            patch.object(dense_blocks, "get_deep_gemm_prefill_data", return_value=data),
            patch.object(dense_blocks, "score_tiles", tiles(logits, cuts)),
            patch.object(dsv4_ops, "topk_transform_ragged_v2", kernel),
        ):
            published = backend.publish_prefill(inputs)
        self.assertEqual(kernel.calls, len(cuts) - 1)
        return published

    def test_source_selection_comes_out_ascending(self):
        k = self.K
        data, inputs, logits = make_batch([40, 30], [40, 10], k=k, seed=1)
        self._publish(self._backend(), data, inputs, logits, [0, 23, 50], seed=1)
        lens = data.compress_lens
        sets = [
            set(range(int(n)))
            if int(n) <= k
            else set(exact_topk(logits[b], int(n), k).tolist())
            for b, n in enumerate(lens)
        ]
        self.assert_selection(
            inputs.out_raw_indices, lens, k, sets, kernel_order(lens, k)
        )
        self.assert_slots(inputs, data)

    def test_consumer_long_rows_ascending_short_rows_keep_their_order(self):
        k = self.K
        backend = self._backend()
        data, inputs, logits = make_batch([40, 30], [40, 10], k=k, seed=2)
        published = self._publish(backend, data, inputs, logits, [0, 50], seed=2)
        real = dense_blocks.topk_among_blocks
        gen = torch.Generator().manual_seed(2)
        written = []

        def topk_among_blocks(scores, lens, blocks, k_, block_size):
            # The real top-k among the published blocks; rows over k positions
            # shuffled, as an unordered top-k may leave them.
            picks = real(scores, lens, blocks, k_, block_size)
            for r in range(picks.shape[0]):
                if int(lens[r]) > k_:
                    picks[r] = picks[r][torch.randperm(k_, generator=gen)]
            written.append(picks.clone())
            return picks

        inputs.out_raw_indices.fill_(-7)
        with (
            patch.object(dense_blocks, "get_deep_gemm_prefill_data", return_value=data),
            patch.object(dense_blocks, "score_tiles", tiles(logits, [0, 23, 50])),
            patch.object(dense_blocks, "topk_among_blocks", topk_among_blocks),
        ):
            backend.consume_prefill(inputs, published)
        order = torch.cat(written).tolist()
        lens = data.compress_lens
        sets = [set(v for v in row if v >= 0) for row in order]
        self.assert_selection(inputs.out_raw_indices, lens, k, sets, order)
        self.assert_slots(inputs, data)
        # Every visible position of a short row is a candidate (2,048 blocks of 8 cover it).
        for b, n in enumerate(lens.tolist()):
            if n <= k:
                self.assertEqual(sets[b], set(range(n)))


class TestDeepSelectSortedIndex(CustomTestCase):
    """Where the DeepSelect top-k runs (bf16 prefill logits on SM100, the paged
    decode on SM90 and SM100) the kernel sorts: sorted_index=True."""

    def test_bf16_prefill_asks_the_kernel_for_sorted_indices(self):
        calls = []
        logits = torch.zeros(3, 64, dtype=torch.bfloat16)
        lens = torch.tensor([10, 40, 64], dtype=torch.int32)
        out = torch.full((3, 16), -1, dtype=torch.int32)
        offsets = torch.tensor([0, 0, 100], dtype=torch.int32)
        with (
            patch.object(deep_select, "topk", lambda x, k, **kw: calls.append((k, kw))),
            patch.object(dsv4_ops, "topk_transform_ragged_v2") as v2,
        ):
            scoring.topk_ragged(logits, lens, out=out, offsets=offsets)
        v2.assert_not_called()
        self.assertEqual(len(calls), 1)
        k, kw = calls[0]
        self.assertEqual(k, 16)
        self.assertIs(kw.get("sorted_index"), True)
        self.assertIs(kw["output_idx"], out)
        self.assertIs(kw["output_idx_offset"], offsets)
        self.assertIs(kw["end"], lens)
        self.assertEqual(kw["idx_oob_fill_value"], -1)

    def _decode(self, deep_select_supported):
        bs = 2
        d = SimpleNamespace(
            scores=torch.zeros(bs, 1024), bs=bs, lens=torch.tensor([700, 300])
        )
        inputs = SimpleNamespace(
            indexer=SimpleNamespace(index_topk=512),
            paged_metadata=SimpleNamespace(
                page_table=torch.zeros(4, 16, dtype=torch.int32),
                compressed_page_size=64,
            ),
            out_page_indices=torch.full((4, 512), -1, dtype=torch.int32),
        )
        with patch.object(
            full_topk, "is_deep_select_supported", return_value=deep_select_supported
        ):
            indexer = FullTopKIndexer(
                token_to_kv_pool=None,
                req_to_token=None,
                use_deep_gemm_prefill=True,
                use_deep_gemm_decode=False,
            )
        calls = []
        with (
            patch.object(full_topk, "decode_scores", return_value=d) as scores,
            patch.object(
                full_topk,
                "topk_page_transform",
                lambda x, k, **kw: calls.append((k, kw)),
            ),
            patch.object(full_topk, "select_decode") as select,
        ):
            indexer.topk_decode(inputs)
        return scores, select, calls

    def test_paged_decode_asks_the_kernel_for_sorted_indices(self):
        _, select, calls = self._decode(deep_select_supported=True)
        select.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 512)
        self.assertIs(calls[0][1]["sorted_index"], True)

    def test_sm12x_decode_takes_the_finish_that_sorts(self):
        # Unchanged by the fix: DeepSelect has no SM12x build, so the SM12x decode
        # selects through select_decode, whose finish_paged_indexer_topk already
        # sorts the positions before it maps them to pool slots.
        self.assertNotIn((12, 0), deep_select.SUPPORTED_CUDA_ARCHS)
        self.assertNotIn((12, 1), deep_select.SUPPORTED_CUDA_ARCHS)
        scores, select, calls = self._decode(deep_select_supported=False)
        self.assertEqual(calls, [])
        select.assert_called_once()
        self.assertIs(scores.call_args.kwargs["compute_topk_plan"], True)


if __name__ == "__main__":
    unittest.main()
