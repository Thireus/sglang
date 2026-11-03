"""Random accepts through the ratio-2 compressor against decoding the same tokens one by one.

Under DSpark every stage writes a whole block, rows past the accept included, and nothing rolls
back: those rows are overwritten before anyone reads them. Driving one committed stream through
verify blocks (and plain rounds among them) must leave the same main-KV bytes and the same
latents as ``draft_len = 1`` does, with padded rows beside them.

On a GPU this runs the real kernel. Without one it runs the arithmetic of
``deepseek_v4/c2.cuh`` written in torch, which proves the test's own bookkeeping and the rule.
"""

import random
import unittest
from typing import Dict, List, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
    c2_decode_norm_rope_store,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 64
NUM_PAGES = 16
EPS = 1e-6
WIDTH = 6
# Request r's full slots start here, even so that a (2k, 2k+1) pair shares a compressed slot;
# never 0, which marks a padded row.
SLOT_BASE = 64
Row = Tuple[int, int, int]  # request, position, token


def _gpu() -> bool:
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10


DEVICE = "cuda" if _gpu() else "cpu"


def _torch_c2(
    kv_input,
    kv_state,
    norm_weight,
    positions,
    req,
    raw_out_loc,
    eps,
    freqs_cis,
    k_cache,
    *,
    page_size,
    ring_size,
    draft_len=1,
    layout=None,
):
    """c2.cuh's addressing: an even row parks at ``req * ring + pos % ring``; an odd row pairs
    with the previous row of its block, or the ring for a block's first row, and stores its
    latent at ``raw_out_loc // 2``. The value is a stand-in for the pooling and the norm."""
    out = torch.zeros(kv_input.shape[0], HEAD_DIM, dtype=torch.float32)
    for i in range(kv_input.shape[0]):
        pos, rid, loc = int(positions[i]), int(req[i]), int(raw_out_loc[i])
        if pos % 2 == 0:
            if loc != 0:
                kv_state[rid * ring_size + pos % ring_size] = kv_input[i]
            continue
        if draft_len > 1 and i % draft_len != 0:
            partner = kv_input[i - 1]
        else:
            partner = kv_state[rid * ring_size + (pos - 1) % ring_size]
        if loc != 0:
            out[i] = (partner[:HEAD_DIM] + 2 * kv_input[i][:HEAD_DIM]) * norm_weight
            k_cache[loc // 2] = out[i]
    return out


COMPRESS = c2_decode_norm_rope_store if _gpu() else _torch_c2


def _past_accept_token(position: int) -> int:
    """A rejected draft's token; the same one at a position whenever it is drafted, so that
    the rows past a stream's end compare as well."""
    return 10_000 + position


class _Run:
    def __init__(self, layout: KVLayout, ring_size: int, requests: int, seed: int):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        self.weight = torch.randn(HEAD_DIM, generator=gen, device=DEVICE, dtype=torch.float32)
        if _gpu():
            self.weight = self.weight.to(torch.bfloat16)
            self.cache = torch.zeros(
                NUM_PAGES, layout.page_bytes(PAGE_SIZE), dtype=torch.uint8, device=DEVICE
            )
        else:
            self.cache = torch.zeros(NUM_PAGES * PAGE_SIZE, HEAD_DIM, dtype=torch.float32)
        angles = torch.randn(4096, ROPE_DIM // 2, generator=gen, device=DEVICE)
        self.freqs = torch.view_as_real(
            torch.polar(torch.ones_like(angles), angles)
        ).flatten(-2)
        self.state = torch.zeros(
            (requests + 1) * ring_size, 2 * HEAD_DIM, dtype=torch.float32, device=DEVICE
        )
        self.layout = layout
        self.ring_size = ring_size
        self.latents: Dict[Row, torch.Tensor] = {}
        self.last_position: Dict[int, int] = {}
        # One input row per (request, position, token): another token is another row. The
        # generator is shared by the two runs of a case through this table.
        self.inputs: Dict[Row, torch.Tensor] = {}
        self.input_gen = None

    def row(self, key: Row) -> torch.Tensor:
        if key not in self.inputs:
            self.inputs[key] = torch.randn(
                2 * HEAD_DIM, generator=self.input_gen, device=DEVICE, dtype=torch.float32
            )
        return self.inputs[key]

    def forward(self, blocks: List[List[Row]], draft_len: int, pad: int = 0) -> None:
        """Each block: ``draft_len`` rows. ``pad`` padded blocks after them, addressing
        request 0 and position 0, with ``raw_out_loc == 0``."""
        rows = [r for block in blocks for r in block]
        count = pad * draft_len
        kv = torch.stack(
            [self.row(r) for r in rows] + [torch.zeros(2 * HEAD_DIM, device=DEVICE)] * count
        )
        to = lambda values, dtype: torch.tensor(values, device=DEVICE, dtype=dtype)
        loc = [SLOT_BASE * (1 + r[0]) + r[1] for r in rows]
        out = COMPRESS(
            kv,
            self.state,
            self.weight,
            to([r[1] for r in rows] + [0] * count, torch.int32),
            to([r[0] for r in rows] + [0] * count, torch.int64),
            to(loc + [0] * count, torch.int32),
            EPS,
            self.freqs,
            self.cache,
            page_size=PAGE_SIZE,
            ring_size=self.ring_size,
            draft_len=draft_len,
            layout=self.layout,
        )
        for i, r in enumerate(rows):
            self.last_position[r[0]] = max(self.last_position.get(r[0], -1), r[1])
            if r[1] % 2 == 1:
                self.latents[r] = out[i].clone()


def _in_blocks(run: _Run, streams, starts, rng: random.Random, pad: int) -> int:
    """Rounds of a verify of WIDTH rows with a random accept per request, or a plain round.
    Returns how many rows the rejected drafts wrote."""
    done = [0] * len(streams)
    rejected = 0
    while True:
        live = [r for r in range(len(streams)) if done[r] < len(streams[r])]
        if not live:
            return rejected
        if rng.random() < 0.25:
            blocks = []
            for r in live:
                blocks.append([(r, starts[r] + done[r], streams[r][done[r]])])
                done[r] += 1
            run.forward(blocks, 1, pad)
            continue
        blocks, accepts = [], []
        for r in live:
            position = starts[r] + done[r]
            # The first row is the sure anchor; the rest agree up to the accept.
            accept = min(rng.randint(1, WIDTH), len(streams[r]) - done[r])
            block = []
            for col in range(WIDTH):
                token = (
                    streams[r][done[r] + col]
                    if col < accept
                    else _past_accept_token(position + col)
                )
                block.append((r, position + col, token))
            rejected += WIDTH - accept
            blocks.append(block)
            accepts.append((r, accept))
        run.forward(blocks, WIDTH, pad)
        for r, accept in accepts:
            done[r] += accept


class TestC2VerifyAgainstDecode(CustomTestCase):
    def _check(self, layout: KVLayout, seed: int, pad: int) -> None:
        ring_size = get_compress_state_ring_size(2, WIDTH)
        self.assertEqual(ring_size, 8)
        rng = random.Random(seed)
        # Request 0 starts even, request 1 odd after a parked row, request 2 even.
        starts = [0, 7, 2]
        streams = [[rng.randrange(1000) for _ in range(n)] for n in (41, 37, 29)]
        one = _Run(layout, ring_size, len(streams), seed)
        blocks = _Run(layout, ring_size, len(streams), seed)
        one.input_gen = torch.Generator(device=DEVICE).manual_seed(seed + 1000)
        blocks.input_gen = one.input_gen
        blocks.inputs = one.inputs
        for run in (one, blocks):
            run.forward([[(1, 6, 5)]], 1)
        self.assertGreater(_in_blocks(blocks, streams, starts, rng, pad), 20)
        for rid, stream in enumerate(streams):
            for offset, token in enumerate(stream):
                one.forward([[(rid, starts[rid] + offset, token)]], 1)
            # What the blocks wrote past the stream's end: the request's next round would
            # overwrite it before anyone reads it. Both runs write it now, from their rings.
            end = starts[rid] + len(stream)
            for position in range(end, blocks.last_position[rid] + 1):
                for run in (one, blocks):
                    run.forward([[(rid, position, _past_accept_token(position))]], 1)
        committed = 0
        for rid, stream in enumerate(streams):
            for offset, token in enumerate(stream):
                key = (rid, starts[rid] + offset, token)
                if key[1] % 2 == 1:
                    self.assertTrue(torch.equal(one.latents[key], blocks.latents[key]), key)
                    committed += 1
        self.assertGreater(committed, 40)
        differs = (one.cache != blocks.cache).sum().item()
        self.assertEqual(differs, 0, f"main-KV differs in {differs} of {one.cache.numel()}")

    def test_committed_pairs_equal_decoding_one_by_one(self):
        layouts = (KVLayout.V4, KVLayout.V41) if _gpu() else (KVLayout.V4,)
        for layout in layouts:
            for seed in range(6):
                for pad in (0, 2):
                    with self.subTest(layout=layout.name, seed=seed, pad=pad):
                        self._check(layout, seed, pad)

    def test_a_padded_block_writes_nothing(self):
        ring_size = get_compress_state_ring_size(2, WIDTH)
        run = _Run(KVLayout.V4, ring_size, 1, 3)
        state, cache = run.state.clone(), run.cache.clone()
        zeros = lambda dtype: torch.zeros(WIDTH, device=DEVICE, dtype=dtype)
        COMPRESS(
            torch.ones(WIDTH, 2 * HEAD_DIM, device=DEVICE),
            run.state,
            run.weight,
            zeros(torch.int32),
            zeros(torch.int64),
            zeros(torch.int32),
            EPS,
            run.freqs,
            run.cache,
            page_size=PAGE_SIZE,
            ring_size=ring_size,
            draft_len=WIDTH,
            layout=KVLayout.V4,
        )
        self.assertTrue(torch.equal(run.cache, cache))
        self.assertTrue(torch.equal(run.state, state))


if __name__ == "__main__":
    unittest.main()
