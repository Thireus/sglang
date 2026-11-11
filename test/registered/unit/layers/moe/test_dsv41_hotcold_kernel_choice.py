"""The cold tier's kernel choice and the grouped routing of the hot/cold MoE, on CPU.

* ``select_cold_kernel`` / ``select_mmq_wide`` / ``select_mmq_staged`` /
  ``select_prefill_compact``: the rule by token count and ggml type, every switch,
  and the "legacy" mode that restores the earlier kernels.
* The routing the grouped kernels read (``moe_align_block_size`` blocks, the lanes of
  the multi-route vector kernel, the column groups and passes of the wide-block MMQ
  kernel), through the CPU model in sglang/test/kernels/gguf_moe_grouped_reference.py:
  every route with a non-negative id is visited exactly once, nothing else is, on
  random routes and on the edge cases (no cold route, every route cold, one expert on
  every route, padding ids, a single token).
* ``cold_moe``'s CUDA branch on the toy checkpoint with the grouped kernels replaced
  by float stand-ins that walk that routing: the block sizes it asks the alignment
  for, one alignment for both projections, and the float reference's numbers.
* The compact layout of a staged MMQ call (one row per cold route in buffers of the
  dense sizes, the cold route count kept on the device): the row map, the sorted rows
  handed to the kernels, that nothing of the call depends on a host read of the count,
  and the same numbers as the dense layout, on routes with a top-k of 6 (the toy
  checkpoint's own top-k of 3 keeps the dense one).

* The decode pack (the SwiGLU with the down projection's Q8_1 quantisation as one
  launch, the route weights, the sum and the add into the hot output as one launch,
  the lanes by batch size): the rules, the calls cold_moe makes with each switch on
  and at 0 (the earlier launches), and the same numbers either way.

The kernels' own numerics and their bit equality with the kernels they replace are
GPU tests (test/registered/kernels/ops/moe/test_gguf_moe_grouped_iq2s.py).
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import hotcold as hc
from sglang.test import hotcold_fixture as fx
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels import gguf_moe_grouped_reference as gr
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

BUILT = frozenset({22, 144, 145})
LIMIT = 10.0
SWITCHES = (
    "SGLANG_HOTCOLD_COLD_KERNEL",
    "SGLANG_HOTCOLD_MMQ_MIN_TOKENS",
    "SGLANG_HOTCOLD_MMQ_WIDE",
    "SGLANG_HOTCOLD_MMQ_WIDE_MIN_TOKENS",
    "SGLANG_HOTCOLD_MMQ_STAGED",
    "SGLANG_HOTCOLD_PREFILL_COMPACT",
    "SGLANG_HOTCOLD_PREFILL_FLOAT",
    "SGLANG_HOTCOLD_VEC_MULTI",
    "SGLANG_HOTCOLD_VEC_MULTI_MIN_TOKENS",
    "SGLANG_HOTCOLD_VECM_ROUTES",
    "SGLANG_HOTCOLD_VECM_LANES",
    "SGLANG_HOTCOLD_VECM_LANE_TABLE",
    "SGLANG_HOTCOLD_SPLIT_KERNEL",
    "SGLANG_HOTCOLD_DECODE_ALIGN",
    "SGLANG_HOTCOLD_DECODE_SWIGLU_Q8",
    "SGLANG_HOTCOLD_DECODE_SUM_FOLD",
)
# The decode pack's switches at 0: the launches before it.
PACK_OFF = dict(
    SGLANG_HOTCOLD_DECODE_SWIGLU_Q8=0,
    SGLANG_HOTCOLD_DECODE_SUM_FOLD=0,
    SGLANG_HOTCOLD_VECM_LANE_TABLE=0,
)


def _env(**kv):
    """The switches as given, every other one at its default."""
    clean = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    clean.update({k: str(v) for k, v in kv.items()})
    return patch.dict(os.environ, clean, clear=True)


class TestColdKernelChoice(CustomTestCase):
    def _sel(self, tokens, qtype=22, built=BUILT, **kv):
        with _env(**kv), patch.object(hc, "_mmq_types", lambda: built):
            return hc.select_cold_kernel(tokens, qtype, qtype)

    def test_default_rule_for_iq2_s(self):
        self.assertEqual([self._sel(n) for n in (1, 2)], ["vec", "vec"])
        self.assertEqual({self._sel(n) for n in (3, 6, 12, 24, 48, 54, 64)}, {"vecm"})
        self.assertEqual({self._sel(n) for n in (65, 512, 2048, 8192)}, {"mmq"})

    def test_switches(self):
        off = dict(SGLANG_HOTCOLD_VEC_MULTI="0")
        self.assertEqual({self._sel(n, **off) for n in (1, 6, 54, 64)}, {"vec"})
        late = dict(SGLANG_HOTCOLD_VEC_MULTI_MIN_TOKENS=48)
        self.assertEqual(
            [self._sel(n, **late) for n in (24, 47, 48, 64, 65)],
            ["vec", "vec", "vecm", "vecm", "mmq"],
        )
        early = dict(SGLANG_HOTCOLD_MMQ_MIN_TOKENS=24)
        self.assertEqual(
            [self._sel(n, **early) for n in (2, 24, 25, 54)],
            ["vec", "vecm", "mmq", "mmq"],
        )
        for n in (1, 54, 4096):
            self.assertEqual(self._sel(n, SGLANG_HOTCOLD_COLD_KERNEL="vec"), "vec")
            self.assertEqual(self._sel(n, SGLANG_HOTCOLD_COLD_KERNEL="vecm"), "vecm")
            self.assertEqual(self._sel(n, SGLANG_HOTCOLD_COLD_KERNEL="mmq"), "mmq")

    def test_legacy_restores_the_earlier_kernels(self):
        legacy = dict(SGLANG_HOTCOLD_COLD_KERNEL="legacy")
        self.assertEqual(
            [self._sel(n, **legacy) for n in (1, 6, 54, 64, 65, 2048)],
            ["vec"] * 4 + ["mmq"] * 2,
        )
        with _env(**legacy):
            self.assertEqual(hc._vec_rows_per_block(), 1)  # ggml's one row per CTA
            self.assertEqual(hc.select_mmq_wide(2048, 22, 22), 0)
            with patch.object(hc, "_mmq_block", lambda t: 16):
                self.assertEqual(hc.select_mmq_block(2048, 22), 16)
        with _env(
            SGLANG_HOTCOLD_COLD_KERNEL="legacy", SGLANG_HOTCOLD_MMQ_WIDE_MIN_TOKENS=1
        ):
            with patch.object(hc, "_mmq_block", lambda t: 16):
                self.assertEqual(hc.select_mmq_block(2048, 22), 16)
        with _env():
            self.assertIsNone(hc._vec_rows_per_block())  # the op's own default

    def test_other_types_stay_on_the_vector_kernel(self):
        for qtype in (144, 145):
            self.assertEqual({self._sel(n, qtype) for n in (1, 6, 54, 64)}, {"vec"})
            self.assertEqual(self._sel(65, qtype), "mmq")
            with self.assertRaises(RuntimeError):
                self._sel(6, qtype, SGLANG_HOTCOLD_COLD_KERNEL="vecm")
        self.assertEqual(
            self._sel(4096, 22, built=frozenset()), "vec"
        )  # no MMQ build: the chunked vector kernel
        with self.assertRaises(RuntimeError):
            self._sel(4096, 22, built=frozenset(), SGLANG_HOTCOLD_COLD_KERNEL="mmq")

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self._sel(1, SGLANG_HOTCOLD_COLD_KERNEL="fast")

    def test_wide_block_choice(self):
        with _env():
            self.assertEqual(hc.select_mmq_wide(2048, 22, 22), 64)
            self.assertEqual(
                hc.select_mmq_wide(65, 22, 22), 64
            )  # one answer whatever the size
            self.assertEqual(hc.select_mmq_wide(2048, 144, 144), 0)
            self.assertEqual(hc.select_mmq_wide(2048, 22, 145), 0)
        with _env(SGLANG_HOTCOLD_MMQ_WIDE=0):
            self.assertEqual(hc.select_mmq_wide(2048, 22, 22), 0)
        with _env(SGLANG_HOTCOLD_MMQ_WIDE=32):
            self.assertEqual(hc.select_mmq_wide(2048, 22, 22), 32)
        with _env(SGLANG_HOTCOLD_MMQ_WIDE=48):
            with self.assertRaises(ValueError):
                hc.select_mmq_wide(2048, 22, 22)

    def test_staged_choice(self):
        with _env():
            self.assertEqual(hc.select_mmq_staged(2048, 22, 22), 64)
            self.assertEqual(
                hc.select_mmq_staged(65, 22, 22), 64
            )  # one answer whatever the size
            self.assertEqual(hc.select_mmq_staged(2048, 144, 144), 0)
            self.assertEqual(hc.select_mmq_staged(2048, 22, 145), 0)
        with _env(SGLANG_HOTCOLD_MMQ_STAGED=0):
            self.assertEqual(hc.select_mmq_staged(2048, 22, 22), 0)
        with _env(SGLANG_HOTCOLD_MMQ_STAGED=32):
            self.assertEqual(hc.select_mmq_staged(2048, 22, 22), 32)
        with _env(SGLANG_HOTCOLD_MMQ_STAGED=16):
            with self.assertRaises(ValueError):
                hc.select_mmq_staged(2048, 22, 22)
        with _env(SGLANG_HOTCOLD_COLD_KERNEL="legacy"):
            self.assertEqual(hc.select_mmq_staged(2048, 22, 22), 0)

    def test_float_choice(self):
        with _env():
            self.assertFalse(
                hc.select_prefill_float(2048, 22, 22)
            )  # off by default: other output bits
        with _env(SGLANG_HOTCOLD_PREFILL_FLOAT=1):
            self.assertTrue(hc.select_prefill_float(2048, 22, 22))
            self.assertTrue(hc.select_prefill_float(65, 22, 22))
            self.assertFalse(hc.select_prefill_float(2048, 144, 144))  # IQ2_S only
            # a decode batch never reaches it: the kernel choice stays a vector kernel up to the threshold
            with patch.object(hc, "_mmq_types", lambda: BUILT):
                self.assertEqual(hc.select_cold_kernel(54, 22, 22), "vecm")
                self.assertEqual(hc.select_cold_kernel(64, 22, 22), "vecm")
                self.assertEqual(hc.select_cold_kernel(65, 22, 22), "mmq")
        with _env(SGLANG_HOTCOLD_PREFILL_FLOAT=1, SGLANG_HOTCOLD_MMQ_STAGED=0):
            self.assertFalse(
                hc.select_prefill_float(2048, 22, 22)
            )  # it is the staged kernel's arithmetic
        with _env(SGLANG_HOTCOLD_PREFILL_FLOAT=1, SGLANG_HOTCOLD_COLD_KERNEL="legacy"):
            self.assertFalse(hc.select_prefill_float(2048, 22, 22))

    def test_compact_choice(self):
        x = torch.zeros(4, 8, dtype=torch.bfloat16)
        with patch.object(hc, "_compact_sum_serves", lambda dtype, top_k: top_k > 4):
            with _env():
                self.assertTrue(hc.select_prefill_compact(x, 6, None))
                self.assertFalse(
                    hc.select_prefill_compact(x, 3, None)
                )  # moe_sum's own kernels for 2..4
                self.assertFalse(
                    hc.select_prefill_compact(x, 6, object())
                )  # the imatrix collector reads the dense layout
            with _env(SGLANG_HOTCOLD_PREFILL_COMPACT=0):
                self.assertFalse(hc.select_prefill_compact(x, 6, None))
            with _env(SGLANG_HOTCOLD_COLD_KERNEL="legacy"):
                self.assertFalse(hc.select_prefill_compact(x, 6, None))
            with _env():
                # the counted SwiGLU is the clamped one: a layer without a limit keeps the dense layout
                self.assertFalse(hc.select_prefill_compact(x, 6, None, None))
                self.assertTrue(hc.select_prefill_compact(x, 6, None, 10.0))

    def test_compact_choice_does_not_ask_whether_a_graph_is_captured(self):
        # the compact call reads nothing on the host, so it is the same call eager and captured
        import inspect

        self.assertNotIn(
            "is_current_stream_capturing", inspect.getsource(hc.select_prefill_compact)
        )
        self.assertNotIn("int(count)", inspect.getsource(hc.cold_moe))
        self.assertNotIn(".item()", inspect.getsource(hc.cold_moe))

    def test_lane_table(self):
        with _env(SGLANG_HOTCOLD_VECM_LANE_TABLE="6:2,24:8"):
            self.assertEqual(
                [hc.select_vecm_lanes(n) for n in (1, 6, 7, 24, 25, 54, 64)],
                [2, 2, 8, 8, 16, 16, 16],  # VECM_LANES above the last entry
            )
        with _env(SGLANG_HOTCOLD_VECM_LANE_TABLE="6:2", SGLANG_HOTCOLD_VECM_LANES=32):
            self.assertEqual([hc.select_vecm_lanes(n) for n in (6, 7)], [2, 32])
        for off in ("", "0"):
            with _env(SGLANG_HOTCOLD_VECM_LANE_TABLE=off):
                self.assertEqual({hc.select_vecm_lanes(n) for n in (1, 6, 54)}, {16})
        with _env(
            SGLANG_HOTCOLD_VECM_LANE_TABLE="6:2", SGLANG_HOTCOLD_COLD_KERNEL="legacy"
        ):
            self.assertEqual(hc.select_vecm_lanes(6), 16)
        for bad in ("6", "6:2,3:4", "6:2,6:4", "a:b", "6:0", "0:4", "6:70000"):
            with _env(SGLANG_HOTCOLD_VECM_LANE_TABLE=bad):
                with self.assertRaises(ValueError, msg=bad):
                    hc.select_vecm_lanes(6)

    def test_the_default_lane_table_is_well_formed(self):
        from sglang.srt.environ import envs

        with _env():
            table = envs.SGLANG_HOTCOLD_VECM_LANE_TABLE.get()
            entries = hc._parse_lane_table(table)
            self.assertIsInstance(entries, tuple)
            for tokens in (1, 6, 12, 24, 48, 54, 63, 64):
                self.assertTrue(1 <= hc.select_vecm_lanes(tokens) <= 65535)

    def test_decode_align_choice(self):
        serves = lambda num_cold, block: num_cold <= 1024
        with (
            patch.object(hc, "_mmq_types", lambda: BUILT),
            patch.object(hc, "_split_align_serves", serves),
        ):
            with _env():
                # the multi-route kernel's batches only, in its blocks of routes
                self.assertEqual(
                    [
                        hc.select_decode_align(n, 22, 22, 189)
                        for n in (1, 2, 3, 6, 54, 64)
                    ],
                    [0, 0, 4, 4, 4, 4],
                )
                self.assertEqual(
                    hc.select_decode_align(65, 22, 22, 189), 0
                )  # a prefill
                self.assertEqual(
                    hc.select_decode_align(6, 144, 144, 189), 0
                )  # the vector kernel
                self.assertEqual(hc.select_decode_align(6, 22, 22, 2000), 0)
            with _env(SGLANG_HOTCOLD_VECM_ROUTES=8):
                self.assertEqual(hc.select_decode_align(6, 22, 22, 189), 8)
            for off in (
                dict(SGLANG_HOTCOLD_DECODE_ALIGN=0),
                dict(SGLANG_HOTCOLD_COLD_KERNEL="legacy"),
                dict(SGLANG_HOTCOLD_VEC_MULTI=0),
            ):
                with _env(**off):
                    self.assertEqual(hc.select_decode_align(6, 22, 22, 189), 0, off)

    def test_decode_swiglu_q8_choice(self):
        h = torch.zeros(4, 512, dtype=torch.bfloat16)
        with patch.object(hc, "_swiglu_q8_serves", lambda dtype, dim: dim % 32 == 0):
            with _env():
                self.assertTrue(hc.select_decode_swiglu_q8(h, 10.0, None))
                # the kernel is the clamped SwiGLU's; the collector reads the earlier launches' activations
                self.assertFalse(hc.select_decode_swiglu_q8(h, None, None))
                self.assertFalse(hc.select_decode_swiglu_q8(h, 10.0, object()))
                self.assertFalse(
                    hc.select_decode_swiglu_q8(h[:, :80], 10.0, None)
                )  # 40 columns: not whole Q8_1 blocks
            with _env(SGLANG_HOTCOLD_DECODE_SWIGLU_Q8=0):
                self.assertFalse(hc.select_decode_swiglu_q8(h, 10.0, None))
            with _env(SGLANG_HOTCOLD_COLD_KERNEL="legacy"):
                self.assertFalse(hc.select_decode_swiglu_q8(h, 10.0, None))

    def test_decode_sum_fold_choice(self):
        y = torch.zeros(24, 8, dtype=torch.bfloat16)
        hot = torch.zeros(4, 8, dtype=torch.bfloat16)
        with patch.object(hc, "_compact_sum_serves", lambda dtype, top_k: top_k > 4):
            with _env():
                self.assertTrue(hc.select_decode_sum_fold(y, 6, hot))
                self.assertFalse(
                    hc.select_decode_sum_fold(y, 6, None)
                )  # a caller that wants the cold tier's own output
                self.assertFalse(
                    hc.select_decode_sum_fold(y, 3, torch.zeros(8, 8, dtype=y.dtype))
                )  # moe_sum's own kernels for 2..4
                self.assertFalse(hc.select_decode_sum_fold(y, 6, hot.float()))
                self.assertFalse(hc.select_decode_sum_fold(y, 6, hot[:3]))
                wide = torch.zeros(4, 16, dtype=torch.bfloat16)
                self.assertFalse(
                    hc.select_decode_sum_fold(y, 6, wide[:, ::2])
                )  # not contiguous
            with _env(SGLANG_HOTCOLD_DECODE_SUM_FOLD=0):
                self.assertFalse(hc.select_decode_sum_fold(y, 6, hot))
            with _env(SGLANG_HOTCOLD_COLD_KERNEL="legacy"):
                self.assertFalse(hc.select_decode_sum_fold(y, 6, hot))

    def test_the_pack_does_not_ask_whether_a_graph_is_captured(self):
        # one answer per shape and switch set: a captured graph and its replay run the same launches
        import inspect

        for fn in (
            hc.select_vecm_lanes,
            hc.select_decode_align,
            hc.select_decode_swiglu_q8,
            hc.select_decode_sum_fold,
        ):
            self.assertNotIn("is_current_stream_capturing", inspect.getsource(fn))

    def test_compact_row_map(self):
        ids = torch.tensor([[1, -1, 3], [-1, -1, -1], [0, 2, -1]], dtype=torch.int32)
        # sorted columns: an active route takes its compact row, padding (9 = numel) stays 9, an inactive route -1
        sorted_ids = torch.tensor(
            [1, 4, 5, 8, 6, 0, 9, 9, 2, 7, 9, 9], dtype=torch.int32
        )
        rows, sorted_rows, count = hc._compact_row_map(
            ids, sorted_ids, torch.tensor([12], dtype=torch.int32)
        )
        self.assertEqual(
            (rows.tolist(), count.tolist()),
            ([[0, -1, 1], [-1, -1, -1], [2, 3, -1]], [4]),
        )
        self.assertEqual(
            (rows.dtype, count.dtype, tuple(count.shape)),
            (torch.int32, torch.int32, (1,)),
        )  # the count stays a tensor
        self.assertEqual(sorted_rows.tolist(), [-1, -1, -1, -1, 2, 0, 9, 9, 1, 3, 9, 9])
        none, _, zero = hc._compact_row_map(
            torch.full((5, 6), -1, dtype=torch.int32),
            torch.arange(30, dtype=torch.int32),
            torch.tensor([30], dtype=torch.int32),
        )
        self.assertEqual((zero.tolist(), int(none.max())), ([0], -1))


class _DeviceOnly:
    """The cold route count as cold_moe may use it: handed from kernel to kernel. Reading it
    on the host (int(), bool(), an index, a comparison) raises; a stand-in kernel peeks."""

    def __init__(self, count):
        self._count = count

    def peek(self):
        return int(self._count)

    def _host_read(self, *a, **k):
        raise AssertionError("cold_moe read the cold route count on the host")

    __int__ = __index__ = __bool__ = __eq__ = __lt__ = __gt__ = __le__ = __ge__ = (
        item
    ) = tolist = _host_read
    __hash__ = None


def _edge_routes(tokens, top_k, experts, gen):
    """name -> flat ids (-1 = not a cold route) for the cases the brief of the kernels lists."""
    rand = lambda: torch.stack(
        [torch.randperm(experts, generator=gen)[:top_k] for _ in range(tokens)]
    )
    mixed = rand()
    mixed[torch.rand(tokens, top_k, generator=gen) < 0.7] = -1
    third = rand()
    third[torch.rand(tokens, top_k, generator=gen) < 1 / 3] = -1
    rows = rand()
    rows[tokens // 2 :] = -1
    return {
        "no cold route": torch.full((tokens, top_k), -1),
        "every route cold": rand(),
        "one expert on every route": torch.full((tokens, top_k), experts - 1),
        "first expert on every route": torch.zeros((tokens, top_k), dtype=torch.long),
        "mostly hot": mixed,
        "padding on a third": third,
        "padded rows": rows,
    }


class TestGroupedRoutingModel(CustomTestCase):
    EXPERTS, TOP_K = 11, 6

    def _cases(self):
        gen = torch.Generator().manual_seed(7)
        for tokens in (1, 2, 6, 54, 65, 300):
            for name, ids in _edge_routes(
                tokens, self.TOP_K, self.EXPERTS, gen
            ).items():
                yield f"{name}, {tokens} tokens", ids.reshape(-1).tolist()

    def test_alignment_blocks(self):
        for label, flat in self._cases():
            numel = len(flat)
            for block in (2, 4, 8, 16, 64, 128):
                with self.subTest(label=label, block=block):
                    sorted_ids, expert_ids, post = gr.align_reference(
                        flat, block, self.EXPERTS
                    )
                    self.assertEqual(post, len(sorted_ids))
                    self.assertEqual(post, block * len(expert_ids))
                    self.assertLessEqual(
                        len(expert_ids), gr.max_blocks(numel, block, self.EXPERTS)
                    )
                    self.assertLessEqual(
                        len(expert_ids), max(numel, 1)
                    )  # a block in use holds a route
                    self.assertEqual(
                        sorted(s for s in sorted_ids if s < numel), list(range(numel))
                    )
                    for b, expert in enumerate(expert_ids):
                        cols = sorted_ids[b * block : (b + 1) * block]
                        valid = [s for s in cols if s < numel]
                        self.assertTrue(valid, "no block without a route")
                        self.assertEqual(
                            cols, valid + [numel] * (block - len(valid))
                        )  # padding comes last
                        self.assertEqual({flat[s] for s in valid}, {expert})

    def test_lanes_visit_every_cold_route_once(self):
        for label, flat in self._cases():
            numel = len(flat)
            want = sorted(s for s, e in enumerate(flat) if e >= 0)
            for routes in (2, 4, 8):
                sorted_ids, expert_ids, post = gr.align_reference(
                    flat, routes, self.EXPERTS
                )
                capacity = gr.max_blocks(numel, routes, self.EXPERTS)
                for lanes in (1, 3, 16, 65535):
                    with self.subTest(label=label, routes=routes, lanes=lanes):
                        visits = gr.lane_walk(
                            sorted_ids,
                            expert_ids,
                            post,
                            routes,
                            lanes,
                            numel,
                            self.EXPERTS,
                            capacity,
                        )
                        self.assertEqual(
                            sorted(s for _, _, _, valid in visits for s in valid), want
                        )
                        self.assertEqual(
                            len({b for _, b, _, _ in visits}), len(visits)
                        )  # one CTA per block
                        for lane, block, expert, valid in visits:
                            self.assertEqual(block % min(lanes, capacity), lane)
                            self.assertEqual({flat[s] for s in valid}, {expert})
                        # the blocks in use are consecutive: the lanes share them within one block
                        load = [
                            sum(1 for v in visits if v[0] == lane)
                            for lane in range(min(lanes, capacity))
                        ]
                        self.assertLessEqual(max(load) - min(load), 1)

    def test_wide_blocks_multiply_only_the_groups_in_use(self):
        for label, flat in self._cases():
            numel = len(flat)
            for wide in (16, 32, 64, 128):
                with self.subTest(label=label, wide=wide):
                    sorted_ids, expert_ids, post = gr.align_reference(
                        flat, wide, self.EXPERTS
                    )
                    covered = 0
                    for b, expert in enumerate(expert_ids):
                        valid = sum(
                            1
                            for s in sorted_ids[b * wide : (b + 1) * wide]
                            if s < numel
                        )
                        groups, passes = gr.wide_block_plan(sorted_ids, b, wide, numel)
                        self.assertEqual(groups, -(-valid // gr.WIDE_GROUP))
                        self.assertEqual(sum(passes), groups)
                        self.assertTrue(
                            all(0 < p <= gr.WIDE_PASS // gr.WIDE_GROUP for p in passes)
                        )
                        if expert >= 0:
                            covered += valid
                    self.assertEqual(covered, sum(1 for e in flat if e >= 0))


class TestColdMoeGroupedPaths(CustomTestCase):
    """cold_moe's CUDA branch with the grouped kernels swapped for float stand-ins."""

    def setUp(self):
        self.enterContext(fx.cpu_moe_context())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = fx.build_toy_checkpoint(os.path.join(self.tmp.name, "ckpt"))
        from sglang.srt.models import deepseek_v2

        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=fx.toy_hf_config(self.ckpt.path),
            layer_id=0,
            quant_config=fx.fp4_quant_config(),
            prefix="model.layers.0.mlp",
            is_deepseek_v4=True,
        )
        fx.load_toy_layer(self.mlp, self.ckpt, 0)
        self.experts = self.mlp.experts
        self.cold = self.experts.cold
        self.gen = torch.Generator().manual_seed(3)

    def _seams(self, calls, lanes=3):
        cold = self.cold
        w13_f, w2_f = cold.reference_weights()
        self.lanes_seen = []  # the lanes cold_moe hands to the multi-route kernel (None = the op's own)
        walk_lanes = lanes  # the stand-ins' own lane count when cold_moe hands none

        def align(ids, block, num_experts):
            calls.append(("align", block, num_experts))
            sorted_ids, expert_ids, post = gr.align_reference(
                ids.reshape(-1).tolist(), block, num_experts
            )
            return (
                torch.tensor(sorted_ids),
                torch.tensor(expert_ids),
                torch.tensor([post]),
            )

        def grouped(kind):
            def project(
                x,
                weight,
                sorted_ids,
                expert_ids,
                post_padded,
                qtype,
                rows,
                top_k,
                tokens,
                block,
                sorted_rows=None,
                out_rows=None,
                lanes=None,
            ):
                if kind == "vecm":
                    self.lanes_seen.append(lanes)
                dense = w13_f if weight is cold.w13_qweight else w2_f
                numel = tokens * top_k
                if sorted_rows is None:
                    calls.append((kind, tokens, top_k, rows, block))
                    out = torch.zeros(numel, rows, dtype=torch.float32)
                    place = lambda route: route
                else:
                    # a compact output: NaN where the kernel would write nothing, so a missed row shows
                    calls.append((kind, tokens, top_k, rows, block, "rows", out_rows))
                    out = torch.full(
                        (out_rows, rows), float("nan"), dtype=torch.float32
                    )
                    row_of = {
                        s: r
                        for s, r in zip(sorted_ids.tolist(), sorted_rows.tolist())
                        if 0 <= s < numel
                    }
                    place = lambda route: row_of[route]
                seen = []
                for _, _, expert, valid in gr.lane_walk(
                    sorted_ids.tolist(),
                    expert_ids.tolist(),
                    int(post_padded[0]),
                    block,
                    walk_lanes if lanes is None else lanes,
                    numel,
                    weight.shape[0],
                    len(expert_ids),
                ):
                    for route in valid:
                        out[place(route)] = dense[expert] @ x[route // top_k].float()
                        seen.append(route)
                assert len(seen) == len(set(seen)), "a route was written twice"
                return out.to(x.dtype)

            return project

        def swiglu_q8(h, ids, limit):
            # NaN in the rows of the other routes: the kernel leaves them as they are
            active = ids.reshape(-1) >= 0
            calls.append(("swiglu q8", h.shape[0], int(active.sum())))
            assert not bool(h[active].isnan().any()), (
                "a cold route's row of the gate/up output was not written"
            )
            a = torch.full((h.shape[0], h.shape[1] // 2), float("nan"), dtype=h.dtype)
            a[active] = hc._clamped_swiglu_reference(h[active], limit)
            # the stand-in for the Q8_1 blocks: the same rows, as a tensor of its own
            return a, a.clone()

        def vecm_q8(
            a,
            quantized,
            weight,
            sorted_ids,
            expert_ids,
            post_padded,
            qtype,
            rows,
            tokens,
            routes,
            lanes=None,
        ):
            calls.append(("vecm q8", tokens, rows, routes))
            self.lanes_seen.append(lanes)
            assert weight is cold.w2_qweight and quantized is not a
            assert tuple(quantized.shape) == tuple(a.shape)
            out = torch.full((tokens, rows), float("nan"), dtype=torch.float32)
            seen = []
            for _, _, expert, valid in gr.lane_walk(
                sorted_ids.tolist(),
                expert_ids.tolist(),
                int(post_padded[0]),
                routes,
                walk_lanes if lanes is None else lanes,
                tokens,
                weight.shape[0],
                len(expert_ids),
            ):
                for route in valid:
                    # the projection reads the quantised rows only
                    row = quantized[route].float()
                    assert not bool(row.isnan().any()), (
                        "the quantised row of a cold route was not written"
                    )
                    out[route] = w2_f[expert] @ row
                    seen.append(route)
            assert len(seen) == len(set(seen)), "a route was written twice"
            return out.to(a.dtype)

        def dense_sum_fold(y, topk_weights, ids, fold_into):
            active = ids >= 0
            calls.append(("sum fold", y.shape[0], int(active.sum())))
            tokens, top_k = ids.shape
            picked = y.float().view(tokens, top_k, -1)
            assert not bool(picked[active].isnan().any()), (
                "a cold route's row of the down projection was not written"
            )
            terms = (picked * topk_weights.float().unsqueeze(-1)).to(y.dtype)
            total = (
                terms.float().masked_fill(~active.unsqueeze(-1), 0).sum(1).to(y.dtype)
            )
            return fold_into.add_(total)

        def row_map(ids, sorted_ids, post_padded):
            rows, sorted_rows, count = hc._compact_row_map_reference(
                ids, sorted_ids, post_padded
            )
            calls.append(("row map", int(count)))
            return rows, sorted_rows, _DeviceOnly(count)

        def counted_swiglu(h, count, limit):
            # NaN in the rows past the count: the kernel leaves them as they are
            n = count.peek()
            calls.append(("swiglu", h.shape[0], n))
            assert not bool(h[:n].isnan().any()), (
                "a compact row of the gate/up output was not written"
            )
            a = torch.full((h.shape[0], h.shape[1] // 2), float("nan"), dtype=h.dtype)
            a[:n] = hc._clamped_swiglu_reference(h[:n], limit)
            return a

        def staged_counted(
            a,
            weight,
            sorted_rows,
            expert_ids,
            post_padded,
            qtype,
            rows,
            bound,
            block,
            count,
        ):
            n = count.peek()
            calls.append(("staged counted", bound, rows, block, n))
            assert a.shape[0] == bound, (
                "the down projection's input keeps the dense row count"
            )
            out = torch.full((bound, rows), float("nan"), dtype=torch.float32)
            seen = []
            for _, _, expert, valid in gr.lane_walk(
                sorted_rows.tolist(),
                expert_ids.tolist(),
                int(post_padded[0]),
                block,
                walk_lanes,
                bound,
                weight.shape[0],
                len(expert_ids),
            ):
                for r in valid:
                    assert r < n, "a sorted id points past the quantised rows"
                    out[r] = w2_f[expert] @ a[r].float()
                    seen.append(r)
            assert sorted(seen) == list(range(n)), (
                "every compact row is projected exactly once"
            )
            return out.to(a.dtype)

        def compact_weight_and_sum(y, topk_weights, rows):
            active = rows >= 0
            calls.append(("compact sum", y.shape[0], int(active.sum())))
            picked = y.float()[rows.clamp_min(0).long()]
            assert not bool(picked[active].isnan().any()), (
                "a compact row was not written"
            )
            terms = (picked * topk_weights.float().unsqueeze(-1)).to(y.dtype)
            return (
                terms.float().masked_fill(~active.unsqueeze(-1), 0).sum(1).to(y.dtype)
            )

        def vec(x, weight, ids, top_k, qtype, rows, tokens):
            calls.append(("vec", tokens, top_k, rows, hc._vec_rows_per_block()))
            return hc._reference_project(
                x, w13_f if weight is cold.w13_qweight else w2_f, ids, top_k, rows
            )

        def weight_and_sum(y, topk_weights, ids, tokens, top_k):
            valid = (ids >= 0).reshape(-1, 1)
            wts = topk_weights.reshape(-1, 1).float().masked_fill(~valid, 0)
            # apply_gguf_moe_weights writes 0 for a route that is not cold without reading its row
            y = (y.float() * wts).to(y.dtype).masked_fill(~valid, 0)
            return y.view(tokens, top_k, -1).float().sum(1).to(y.dtype)

        return [
            patch.object(hc, "_mmq_types", lambda: BUILT),
            patch.object(hc, "_mmq_block", lambda t: 16),
            patch.object(hc, "_mmq_align", align),
            patch.object(hc, "_vec_project", vec),
            patch.object(hc, "_vecm_project", grouped("vecm")),
            patch.object(hc, "_mmq_project", grouped("mmq")),
            patch.object(hc, "_mmq_wide_project", grouped("wide")),
            patch.object(hc, "_mmq_staged_project", grouped("staged")),
            patch.object(hc, "_f16_staged_project", grouped("half-float")),
            patch.object(hc, "_compact_sum_serves", lambda dtype, top_k: top_k > 4),
            patch.object(hc, "_compact_row_map", row_map),
            patch.object(hc, "_counted_swiglu", counted_swiglu),
            patch.object(hc, "_mmq_staged_project_counted", staged_counted),
            patch.object(hc, "_compact_weight_and_sum", compact_weight_and_sum),
            patch.object(hc, "_clamped_swiglu", hc._clamped_swiglu_reference),
            patch.object(hc, "_weight_and_sum", weight_and_sum),
            patch.object(hc, "_swiglu_q8_serves", lambda dtype, dim: dim % 32 == 0),
            patch.object(hc, "_swiglu_q8", swiglu_q8),
            patch.object(hc, "_vecm_project_q8", vecm_q8),
            patch.object(hc, "_dense_sum_fold", dense_sum_fold),
        ]

    def _routes(self, tokens, top_k=3):
        ids = torch.stack(
            [torch.randperm(8, generator=self.gen)[:top_k] for _ in range(tokens)]
        ).to(torch.int32)
        return ids, torch.rand(tokens, top_k, generator=self.gen) + 0.1

    def _run(self, tokens, ids=None, **kv):
        x = torch.randn(tokens, 256, generator=self.gen)
        rids, w = self._routes(tokens)
        ids = rids if ids is None else ids
        _, cold_ids = self.experts.split_routes(ids)
        args = (
            x,
            self.cold.w13_qweight,
            self.cold.w2_qweight,
            w,
            cold_ids,
            22,
            22,
            LIMIT,
        )
        expect = hc.cold_moe(
            *args, reference_weights=self.cold.reference_weights(), use_kernels=False
        )
        calls = []
        with _env(**kv):
            ctx = self._seams(calls)
            for c in ctx:
                c.start()
            try:
                got = hc.cold_moe(*args, use_kernels=True)
            finally:
                for c in ctx:
                    c.stop()
        torch.testing.assert_close(
            got, expect, rtol=0, atol=1e-5 * max(expect.abs().max().item(), 1e-6)
        )
        return calls

    def test_decode_batch_takes_the_multi_route_kernel(self):
        calls = self._run(7)
        # one alignment in blocks of 4 routes for both projections: w13 over the tokens, then the SwiGLU
        # with the Q8_1 blocks of the cold routes' rows, and w2 over the routes from those blocks
        self.assertEqual(calls[:2], [("align", 4, 4), ("vecm", 7, 3, 512, 4)])
        self.assertEqual(
            [c[:2] for c in calls[2:]], [("swiglu q8", 21), ("vecm q8", 21)]
        )
        self.assertEqual(calls[3], ("vecm q8", 21, 256, 4))
        # the earlier launches: the SwiGLU of every row, then w2 with its own quantisation
        self.assertEqual(
            self._run(7, **PACK_OFF),
            [("align", 4, 4), ("vecm", 7, 3, 512, 4), ("vecm", 21, 1, 256, 4)],
        )
        self.assertEqual(self._run(7, SGLANG_HOTCOLD_VECM_ROUTES=8)[0], ("align", 8, 4))

    def test_small_batches_and_the_switch_stay_on_the_vector_kernel(self):
        self.assertEqual([c[0] for c in self._run(2)], ["vec", "vec"])
        self.assertEqual(
            self._run(7, SGLANG_HOTCOLD_VEC_MULTI=0),
            [("vec", 7, 3, 512, None), ("vec", 21, 1, 256, None)],
        )
        self.assertEqual(
            self._run(7, SGLANG_HOTCOLD_COLD_KERNEL="legacy"),
            [("vec", 7, 3, 512, 1), ("vec", 21, 1, 256, 1)],
        )

    def test_prefill_takes_the_staged_kernel(self):
        # the toy checkpoint's top-k is 3: the dense layout (moe_sum has a kernel of its own for it)
        calls = self._run(70)
        self.assertEqual(
            calls,
            [("align", 64, 4), ("staged", 70, 3, 512, 64), ("staged", 210, 1, 256, 64)],
        )
        calls = self._run(70, SGLANG_HOTCOLD_MMQ_STAGED=32)
        self.assertEqual(
            calls,
            [("align", 32, 4), ("staged", 70, 3, 512, 32), ("staged", 210, 1, 256, 32)],
        )
        calls = self._run(70, SGLANG_HOTCOLD_MMQ_STAGED=0)
        self.assertEqual(
            calls,
            [("align", 64, 4), ("wide", 70, 3, 512, 64), ("wide", 210, 1, 256, 64)],
        )
        calls = self._run(70, SGLANG_HOTCOLD_MMQ_STAGED=0, SGLANG_HOTCOLD_MMQ_WIDE=0)
        self.assertEqual(
            calls, [("align", 16, 4), ("mmq", 70, 3, 512, 16), ("mmq", 210, 1, 256, 16)]
        )
        calls = self._run(70, SGLANG_HOTCOLD_COLD_KERNEL="legacy")
        self.assertEqual(
            calls, [("align", 16, 4), ("mmq", 70, 3, 512, 16), ("mmq", 210, 1, 256, 16)]
        )

    def _run_top6(self, tokens, cold_ids=None, **kv):
        """cold_moe on routes with a top-k of 6 over the toy layer's four cold experts."""
        top_k = 6
        x = torch.randn(tokens, 256, generator=self.gen)
        if cold_ids is None:
            cold_ids = torch.randint(
                0, 4, (tokens, top_k), generator=self.gen, dtype=torch.int32
            )
            cold_ids[torch.rand(tokens, top_k, generator=self.gen) < 0.6] = -1
        w = torch.rand(tokens, top_k, generator=self.gen) + 0.1
        args = (
            x,
            self.cold.w13_qweight,
            self.cold.w2_qweight,
            w,
            cold_ids,
            22,
            22,
            LIMIT,
        )
        expect = hc.cold_moe(
            *args, reference_weights=self.cold.reference_weights(), use_kernels=False
        )
        calls = []
        with _env(**kv):
            ctx = self._seams(calls)
            for c in ctx:
                c.start()
            try:
                got = hc.cold_moe(*args, use_kernels=True)
            finally:
                for c in ctx:
                    c.stop()
        torch.testing.assert_close(
            got, expect, rtol=0, atol=1e-5 * max(expect.abs().max().item(), 1e-6)
        )
        return calls, int((cold_ids >= 0).sum())

    def test_prefill_keeps_the_cold_rows_only(self):
        calls, count = self._run_top6(70)
        self.assertGreater(count, 0)
        # one alignment and one row map; the gate/up kernel writes the first count rows of a [70 * 6, 512]
        # buffer through the row map; the SwiGLU and the down projection take buffers of the same 420 rows
        # and the count as a device tensor (the stand-ins see it, cold_moe never reads it: _DeviceOnly)
        self.assertEqual(
            calls,
            [
                ("align", 64, 4),
                ("row map", count),
                ("staged", 70, 6, 512, 64, "rows", 420),
                ("swiglu", 420, count),
                ("staged counted", 420, 256, 64, count),
                ("compact sum", 420, count),
            ],
        )
        calls, _ = self._run_top6(70, SGLANG_HOTCOLD_MMQ_STAGED=32)
        self.assertEqual(
            [c[:2] for c in calls][:1] + [c[0] for c in calls][1:],
            [
                ("align", 32),
                "row map",
                "staged",
                "swiglu",
                "staged counted",
                "compact sum",
            ],
        )
        # the wide-block kernel has no counted activation quantisation: the dense layout
        calls, _ = self._run_top6(70, SGLANG_HOTCOLD_MMQ_STAGED=0)
        self.assertEqual(
            calls,
            [("align", 64, 4), ("wide", 70, 6, 512, 64), ("wide", 420, 1, 256, 64)],
        )
        calls, _ = self._run_top6(70, SGLANG_HOTCOLD_PREFILL_COMPACT=0)
        self.assertEqual(
            calls,
            [("align", 64, 4), ("staged", 70, 6, 512, 64), ("staged", 420, 1, 256, 64)],
        )
        calls, _ = self._run_top6(
            70, SGLANG_HOTCOLD_MMQ_STAGED=0, SGLANG_HOTCOLD_MMQ_WIDE=0
        )
        self.assertEqual(
            calls, [("align", 16, 4), ("mmq", 70, 6, 512, 16), ("mmq", 420, 1, 256, 16)]
        )  # the 16-route kernel: dense
        calls, _ = self._run_top6(
            7
        )  # a decode batch: the multi-route vector kernel, dense
        self.assertEqual(
            [c[0] for c in calls], ["align", "vecm", "swiglu q8", "vecm q8"]
        )
        calls, _ = self._run_top6(7, **PACK_OFF)
        self.assertEqual([c[0] for c in calls], ["align", "vecm", "vecm"])

    def test_the_float_switch_takes_the_half_float_kernel_above_the_threshold_only(
        self,
    ):
        # the half-float kernel keeps the dense layout (its activation step has no counted variant)
        calls, count = self._run_top6(70, SGLANG_HOTCOLD_PREFILL_FLOAT=1)
        self.assertEqual(
            calls,
            [
                ("align", 64, 4),
                ("half-float", 70, 6, 512, 64),
                ("half-float", 420, 1, 256, 64),
            ],
        )
        calls = self._run(
            70, SGLANG_HOTCOLD_PREFILL_FLOAT=1
        )  # top-k 3: the dense layout
        self.assertEqual(
            calls,
            [
                ("align", 64, 4),
                ("half-float", 70, 3, 512, 64),
                ("half-float", 210, 1, 256, 64),
            ],
        )
        calls, _ = self._run_top6(7, SGLANG_HOTCOLD_PREFILL_FLOAT=1, **PACK_OFF)
        self.assertEqual([c[0] for c in calls], ["align", "vecm", "vecm"])
        calls, _ = self._run_top6(64, SGLANG_HOTCOLD_PREFILL_FLOAT=1, **PACK_OFF)
        self.assertEqual([c[0] for c in calls], ["align", "vecm", "vecm"])
        calls, _ = self._run_top6(64, SGLANG_HOTCOLD_PREFILL_FLOAT=1)
        self.assertEqual(
            [c[0] for c in calls], ["align", "vecm", "swiglu q8", "vecm q8"]
        )
        calls = self._run(
            70, SGLANG_HOTCOLD_PREFILL_FLOAT=1, SGLANG_HOTCOLD_MMQ_STAGED=0
        )
        self.assertEqual([c[0] for c in calls], ["align", "wide", "wide"])

    def test_compact_edge_routes(self):
        for tokens in (65, 130):
            every = torch.randint(
                0, 4, (tokens, 6), generator=self.gen, dtype=torch.int32
            )
            rows = every.clone()
            rows[::2] = -1
            one = torch.full((tokens, 6), -1, dtype=torch.int32)
            one[tokens // 2, 3] = 2
            cases = {
                "every route cold": every,
                "one cold expert on every route": torch.full(
                    (tokens, 6), 3, dtype=torch.int32
                ),
                "padded rows": rows,
                "a single cold route": one,
            }
            for label, ids in cases.items():
                with self.subTest(label=label, tokens=tokens):
                    calls, count = self._run_top6(tokens, cold_ids=ids)
                    self.assertEqual(
                        [c[0] for c in calls],
                        [
                            "align",
                            "row map",
                            "staged",
                            "swiglu",
                            "staged counted",
                            "compact sum",
                        ],
                    )
                    self.assertEqual(calls[-1], ("compact sum", tokens * 6, count))
            with self.subTest(label="no cold route", tokens=tokens):
                calls, count = self._run_top6(
                    tokens, cold_ids=torch.full((tokens, 6), -1, dtype=torch.int32)
                )
                # no branch on the count: the same launches run (every block is skipped, no row is computed)
                # and the sum writes +0 for every token
                self.assertEqual(count, 0)
                self.assertEqual(
                    [c[0] for c in calls],
                    [
                        "align",
                        "row map",
                        "staged",
                        "swiglu",
                        "staged counted",
                        "compact sum",
                    ],
                )
                self.assertEqual(calls[-1], ("compact sum", tokens * 6, 0))

    def test_a_layer_without_a_swiglu_limit_keeps_the_dense_layout(self):
        tokens, top_k = 70, 6
        x = torch.randn(tokens, 256, generator=self.gen)
        cold_ids = torch.randint(
            0, 4, (tokens, top_k), generator=self.gen, dtype=torch.int32
        )
        w = torch.rand(tokens, top_k, generator=self.gen) + 0.1
        args = (
            x,
            self.cold.w13_qweight,
            self.cold.w2_qweight,
            w,
            cold_ids,
            22,
            22,
            None,
        )
        expect = hc.cold_moe(
            *args, reference_weights=self.cold.reference_weights(), use_kernels=False
        )
        calls = []
        with _env():
            ctx = self._seams(calls)
            for c in ctx:
                c.start()
            try:
                got = hc.cold_moe(*args, use_kernels=True)
            finally:
                for c in ctx:
                    c.stop()
        torch.testing.assert_close(
            got, expect, rtol=0, atol=1e-5 * max(expect.abs().max().item(), 1e-6)
        )
        self.assertEqual(
            calls,
            [("align", 64, 4), ("staged", 70, 6, 512, 64), ("staged", 420, 1, 256, 64)],
        )

    def test_edge_routes_through_the_grouped_paths(self):
        hot = sorted(
            self.experts.split_routes(torch.arange(8, dtype=torch.int32).view(-1, 1))[1]
            .view(-1)
            .lt(0)
            .nonzero()
            .view(-1)
            .tolist()
        )
        cold = [e for e in range(8) if e not in hot]
        for tokens in (3, 9, 70):
            pick = lambda pool: torch.tensor(pool, dtype=torch.int32)[
                torch.stack(
                    [
                        torch.randperm(len(pool), generator=self.gen)[:3]
                        for _ in range(tokens)
                    ]
                )
            ]
            padded = self._routes(tokens)[0]
            padded[::2] = -1
            cases = {
                "no cold route": pick(hot),
                "every route cold": pick(cold),
                "one cold expert on every route": torch.full(
                    (tokens, 3), cold[-1], dtype=torch.int32
                ),
                "padded rows": padded,
                "all padding": torch.full((tokens, 3), -1, dtype=torch.int32),
            }
            for label, ids in cases.items():
                with self.subTest(label=label, tokens=tokens):
                    kinds = {c[0] for c in self._run(tokens, ids=ids)}
                    self.assertEqual(
                        kinds,
                        (
                            {"align", "staged"}
                            if tokens > 64
                            else {"align", "vecm", "swiglu q8", "vecm q8"}
                        ),
                    )
                    kinds = {c[0] for c in self._run(tokens, ids=ids, **PACK_OFF)}
                    self.assertEqual(
                        kinds, {"align", "staged" if tokens > 64 else "vecm"}
                    )

    def _run_pack(self, tokens, cold_ids=None, aligned_block=0, **kv):
        """cold_moe with the hot tier's output handed in (top-k 6), as the layer calls it:
        (calls, cold route count, the lanes handed to the multi-route kernel). With
        aligned_block the cold routes come grouped in blocks of that many, as the layer's
        route split hands them."""
        top_k = 6
        x = torch.randn(tokens, 256, generator=self.gen)
        if cold_ids is None:
            cold_ids = torch.randint(
                0, 4, (tokens, top_k), generator=self.gen, dtype=torch.int32
            )
            cold_ids[torch.rand(tokens, top_k, generator=self.gen) < 0.6] = -1
        w = torch.rand(tokens, top_k, generator=self.gen) + 0.1
        hot = torch.randn(tokens, 256, generator=self.gen)
        args = (
            x,
            self.cold.w13_qweight,
            self.cold.w2_qweight,
            w,
            cold_ids,
            22,
            22,
            LIMIT,
        )
        expect = hot + hc.cold_moe(
            *args, reference_weights=self.cold.reference_weights(), use_kernels=False
        )
        calls = []
        target = hot.clone()
        aligned = None
        if aligned_block:
            # the split's grouping: the cold routes only (no block for the others)
            flat = [e if e >= 0 else 4 for e in cold_ids.reshape(-1).tolist()]
            sorted_ids, expert_ids, post = gr.align_reference(flat, aligned_block, 5)
            keep = [b for b, e in enumerate(expert_ids) if e < 4]
            sorted_ids = [
                s
                for b in keep
                for s in sorted_ids[b * aligned_block : (b + 1) * aligned_block]
            ]
            aligned = (
                aligned_block,
                torch.tensor(sorted_ids + [len(flat)] * 4, dtype=torch.int32),
                torch.tensor([expert_ids[b] for b in keep] + [-1], dtype=torch.int32),
                torch.tensor([len(sorted_ids)], dtype=torch.int32),
            )
        with _env(**kv):
            ctx = self._seams(calls)
            for c in ctx:
                c.start()
            try:
                got = hc.cold_moe(
                    *args, use_kernels=True, fold_into=target, aligned=aligned
                )
            finally:
                for c in ctx:
                    c.stop()
        self.assertIs(
            got, target
        )  # the sum lands in the hot tier's output, whatever the path
        torch.testing.assert_close(
            got, expect, rtol=0, atol=1e-5 * max(expect.abs().max().item(), 1e-6)
        )
        return calls, int((cold_ids >= 0).sum()), list(self.lanes_seen)

    def test_decode_pack_launches(self):
        calls, count, lanes = self._run_pack(7, SGLANG_HOTCOLD_VECM_LANE_TABLE="8:2")
        self.assertGreater(count, 0)
        # one alignment; gate/up; the SwiGLU with the Q8_1 blocks over the 42 route rows (computed
        # for the cold ones); the down projection from those blocks; one kernel for the route
        # weights, the sum and the add into the hot output
        self.assertEqual(
            calls,
            [
                ("align", 4, 4),
                ("vecm", 7, 6, 512, 4),
                ("swiglu q8", 42, count),
                ("vecm q8", 42, 256, 4),
                ("sum fold", 42, count),
            ],
        )
        self.assertEqual(lanes, [2, 2])  # both projections take the table's lanes
        _, _, lanes = self._run_pack(9, SGLANG_HOTCOLD_VECM_LANE_TABLE="8:2")
        self.assertEqual(
            lanes, [None, None]
        )  # above the table: the op's own VECM_LANES

    def test_the_split_alignment_replaces_the_cold_tiers_own(self):
        rest = ["vecm", "swiglu q8", "vecm q8", "sum fold"]
        for tokens in (3, 7, 54):
            calls, count, _ = self._run_pack(tokens, aligned_block=4)
            self.assertGreater(count, 0)
            # no alignment launch of the cold tier's own: both projections walk the split's blocks
            self.assertEqual([c[0] for c in calls], rest)
        calls, _, _ = self._run_pack(7, aligned_block=4, **PACK_OFF)
        self.assertEqual([c[0] for c in calls], ["vecm", "vecm"])
        # blocks of another size than the kernel's: not its alignment
        calls, _, _ = self._run_pack(7, aligned_block=8)
        self.assertEqual([c[0] for c in calls], ["align"] + rest)
        calls, _, _ = self._run_pack(7, aligned_block=8, SGLANG_HOTCOLD_VECM_ROUTES=8)
        self.assertEqual([c[0] for c in calls], rest)
        # a batch the multi-route kernel does not serve never reads it
        calls, _, _ = self._run_pack(2, aligned_block=4)
        self.assertEqual([c[0] for c in calls], ["vec", "vec", "sum fold"])
        calls, _, _ = self._run_pack(70, aligned_block=4)
        self.assertEqual(calls[0], ("align", 64, 4))
        none = torch.full((6, 6), -1, dtype=torch.int32)
        calls, count, _ = self._run_pack(6, cold_ids=none, aligned_block=4)
        self.assertEqual(
            (count, [c[0] for c in calls]), (0, rest)
        )  # no cold route: no block, the same launches

    def test_each_pack_switch_at_zero_gives_the_earlier_launches(self):
        old = [
            "align",
            "vecm",
            "vecm",
        ]  # then apply_gguf_moe_weights, moe_sum and add_ (not recorded)
        calls, _, lanes = self._run_pack(7, **PACK_OFF)
        self.assertEqual([c[0] for c in calls], old)
        self.assertEqual(lanes, [None, None])
        calls, _, _ = self._run_pack(7, SGLANG_HOTCOLD_DECODE_SWIGLU_Q8=0)
        self.assertEqual([c[0] for c in calls], old + ["sum fold"])
        calls, _, _ = self._run_pack(7, SGLANG_HOTCOLD_DECODE_SUM_FOLD=0)
        self.assertEqual(
            [c[0] for c in calls], ["align", "vecm", "swiglu q8", "vecm q8"]
        )
        calls, _, lanes = self._run_pack(7, SGLANG_HOTCOLD_VECM_LANE_TABLE=0)
        self.assertEqual(lanes, [None, None])
        self.assertEqual(calls[-1][0], "sum fold")
        # "legacy": the earlier kernels of the cold tier, whatever the pack's switches say
        calls, _, _ = self._run_pack(7, SGLANG_HOTCOLD_COLD_KERNEL="legacy")
        self.assertEqual([c[0] for c in calls], ["vec", "vec"])

    def test_the_fold_serves_the_vector_kernel_too_and_never_a_prefill(self):
        calls, _, _ = self._run_pack(
            2
        )  # below the multi-route kernel's first batch size
        self.assertEqual([c[0] for c in calls], ["vec", "vec", "sum fold"])
        calls, _, _ = self._run_pack(2, SGLANG_HOTCOLD_DECODE_SUM_FOLD=0)
        self.assertEqual([c[0] for c in calls], ["vec", "vec"])
        calls, count, _ = self._run_pack(
            70
        )  # a prefill: its own compact sum, then add_
        self.assertEqual(
            [c[0] for c in calls],
            [
                "align",
                "row map",
                "staged",
                "swiglu",
                "staged counted",
                "compact sum",
            ],
        )
        calls, _, _ = self._run_pack(70, SGLANG_HOTCOLD_PREFILL_COMPACT=0)
        self.assertEqual([c[0] for c in calls], ["align", "staged", "staged"])

    def test_decode_pack_edge_routes(self):
        for tokens in (3, 6, 54):
            every = torch.randint(
                0, 4, (tokens, 6), generator=self.gen, dtype=torch.int32
            )
            rows = every.clone()
            rows[::2] = -1
            one = torch.full((tokens, 6), -1, dtype=torch.int32)
            one[tokens // 2, 3] = 2
            cases = {
                "every route cold": every,
                "one cold expert on every route": torch.full(
                    (tokens, 6), 3, dtype=torch.int32
                ),
                "padded rows": rows,
                "a single cold route": one,
                "no cold route": torch.full((tokens, 6), -1, dtype=torch.int32),
            }
            for label, ids in cases.items():
                with self.subTest(label=label, tokens=tokens):
                    calls, count, _ = self._run_pack(tokens, cold_ids=ids)
                    # no branch on the routes: the same launches whatever is cold
                    self.assertEqual(
                        [c[0] for c in calls],
                        ["align", "vecm", "swiglu q8", "vecm q8", "sum fold"],
                    )
                    self.assertEqual(calls[2], ("swiglu q8", tokens * 6, count))
                    self.assertEqual(calls[-1], ("sum fold", tokens * 6, count))


if __name__ == "__main__":
    unittest.main()
