"""The MXFP4 (packed) variants of the DeepSeek-V4.1 Engram gather kernels on a GPU:
both Triton kernels against the torch lookup, from a device table, a pinned host
table (both layouts), peer shards and scheduler staging, plus a timing line
against the fp8 kernel on the same rows. One GPU, a few thousand rows.

Real rows are used when SGLANG_TEST_DSV41_FP8_DIR (the original
DeepSeek-V4.1-Flash-1017 snapshot) and SGLANG_TEST_DSV41_MXFP4_DIR (a checkpoint
with MXFP4 Engram tables) are set; otherwise random rows.
"""

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.embeddings.engram_gather import (
    engram_gather,
    engram_gather_peer,
)
from sglang.srt.arg_groups.engram_config import (
    EngramConfig,
    EngramLookup,
    EngramPlacement,
    EngramReplication,
)
from sglang.srt.layers import engram as engram_mod
from sglang.srt.layers.engram import EngramEmbedding, _PeerShards, unpack_mxfp4_codes
from sglang.srt.layers.engram_staging import EngramRowStager
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DIM, BLK = 256, 32
TABLE_ROWS = {1: 384006168, 14: 384016682}
FP8_DIR = os.environ.get("SGLANG_TEST_DSV41_FP8_DIR")
MXFP4_DIR = os.environ.get("SGLANG_TEST_DSV41_MXFP4_DIR")


def _read(snapshot, layer, lo, hi):
    from safetensors import safe_open

    w_name = f"layers.{layer}.engram.embed.weight"
    s_name = f"layers.{layer}.engram.embed.scale"
    with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][w_name]
    with safe_open(os.path.join(snapshot, shard), "pt") as f:
        return f.get_slice(w_name)[lo:hi], f.get_slice(s_name)[lo:hi]


def load_rows():
    """(packed uint8 [n, 128], e8m0 [n, 8], fp8 [n, 256] or None, e8m0 [n, 8] or
    None, where): 5,120 rows, real when the checkpoints are given."""
    if FP8_DIR and MXFP4_DIR:
        mx, f8 = [], []
        for layer, windows in ((1, 8), (14, 2)):
            n = TABLE_ROWS[layer]
            starts = [n * k // windows for k in range(windows - 1)] + [n - 512]
            for lo in starts:
                mx.append(_read(MXFP4_DIR, layer, lo, lo + 512))
                f8.append(_read(FP8_DIR, layer, lo, lo + 512))
        cat = lambda parts, i: torch.cat([p[i] for p in parts])  # noqa: E731
        return cat(mx, 0), cat(mx, 1), cat(f8, 0), cat(f8, 1), "real checkpoint rows"
    g = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (5120, DIM // 2), generator=g, dtype=torch.uint8)
    scale = torch.randint(100, 140, (5120, DIM // BLK), generator=g, dtype=torch.uint8)
    return packed, scale.view(torch.float8_e8m0fnu), None, None, "random rows"


def reference(packed, scale):
    """bf16 lookup of every row by the torch path (EngramEmbedding._owned_rows)."""
    vals = unpack_mxfp4_codes(packed).view(-1, DIM // BLK, BLK)
    return (vals * scale.float().unsqueeze(-1)).flatten(1).to(torch.bfloat16)


def transcode_to_fp8(packed):
    """Every E2M1 value is an e4m3 value: the fp8 table that dequantizes the same."""
    return unpack_mxfp4_codes(packed).to(torch.float8_e4m3fn)


def bits(t):
    return t.contiguous().view(torch.int16).cpu()


def parallel():
    return SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        attn_dp_size=1,
        attn_cp_size=1,
        tp_group=SimpleNamespace(
            rank_in_group=0,
            barrier=lambda: None,
            broadcast_object=lambda obj, src=0: obj,
        ),
    )


def engram_config(host_layout=None):
    """The --engram-config a layer resolves (resolve_engram_config): device rows
    with the all-reduce lookup, or host rows one shard per rank ("per_rank") or one
    copy per node ("shared"), what the deprecated SGLANG_*_ENGRAM_* switches select."""
    if host_layout is None:
        return EngramConfig(lookup=EngramLookup.ALL_REDUCE)
    return EngramConfig(
        placement=EngramPlacement.CPU,
        replication=(
            EngramReplication.PER_NODE
            if host_layout == "shared"
            else EngramReplication.NONE
        ),
        lookup=EngramLookup.ALL_REDUCE,
    )


def make_embed(rows, packed, host_layout=None):
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(engram_mod, "get_parallel", return_value=parallel())
        )
        stack.enter_context(
            patch.object(
                engram_mod,
                "resolve_engram_config",
                return_value=engram_config(host_layout),
            )
        )
        if host_layout is not None:
            return EngramEmbedding(rows, DIM, 1, (0, rows), packed=packed)
        with torch.device("cuda"):
            return EngramEmbedding(rows, DIM, 1, (0, rows), packed=packed)


def time_us(fn, iters=500):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TestEngramGatherMxfp4(CustomTestCase):
    # A host table stays mapped and registered for the life of the process (as in
    # serving); keep every one alive so no address range is registered twice.
    keep = []

    @classmethod
    def setUpClass(cls):
        cls.packed, cls.scale, cls.fp8, cls.fp8_scale, cls.source = load_rows()
        cls.n = cls.packed.shape[0]
        cls.ref = reference(cls.packed, cls.scale)
        g = torch.Generator().manual_seed(1)
        # 48 lookups per token (3 n-gram sizes x 8 heads x 2 layers), 256 tokens,
        # plus ids outside the table, which must come out as zero rows.
        ids = torch.randint(0, cls.n, (256 * 48,), generator=g)
        ids[::97] = cls.n + 5
        cls.ids_cpu = ids
        cls.ids = ids.cuda()
        print(f"\n[engram mxfp4] {cls.source}: {cls.n} rows, {ids.numel()} lookups")

    def expected(self, ids, lo=0, hi=None):
        hi = self.n if hi is None else hi
        ok = (ids >= lo) & (ids < hi)
        out = self.ref[ids.clamp(0, self.n - 1)]
        return out.masked_fill(~ok.unsqueeze(-1), 0)

    def test_device_table_kernel(self):
        w, s = self.packed.cuda(), self.scale.cuda()
        for lo, hi, sort in (
            (0, self.n, False),
            (1000, 3000, False),
            (0, self.n, True),
        ):
            out = torch.empty(
                self.ids.numel(), DIM, dtype=torch.bfloat16, device="cuda"
            )
            # A shard holding global rows [lo, hi) starts at row lo of the table.
            engram_gather(
                w[lo:].data_ptr(),
                s[lo:].data_ptr(),
                self.ids,
                out,
                DIM,
                BLK,
                row_lo=lo,
                row_hi=hi,
                sort_ids=sort,
                packed=True,
            )
            self.assertTrue(
                torch.equal(bits(out), bits(self.expected(self.ids_cpu, lo, hi))),
                (lo, hi, sort),
            )

    def test_packed_kernel_equals_fp8_kernel_on_transcoded_rows(self):
        out_mx = torch.empty(self.ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
        out_f8 = torch.empty_like(out_mx)
        w, s, w8 = (
            self.packed.cuda(),
            self.scale.cuda(),
            transcode_to_fp8(self.packed).cuda(),
        )
        engram_gather(
            w.data_ptr(),
            s.data_ptr(),
            self.ids,
            out_mx,
            DIM,
            BLK,
            row_hi=self.n,
            packed=True,
        )
        engram_gather(
            w8.data_ptr(), s.data_ptr(), self.ids, out_f8, DIM, BLK, row_hi=self.n
        )
        self.assertTrue(torch.equal(bits(out_mx), bits(out_f8)))
        if self.fp8 is not None:
            # Against the original fp8 rows the MXFP4 rows were made from: the
            # publisher measured cosine 0.9934 per row.
            w_orig, s_orig = self.fp8.cuda(), self.fp8_scale.cuda()
            out_orig = torch.empty_like(out_mx)
            engram_gather(
                w_orig.data_ptr(),
                s_orig.data_ptr(),
                self.ids,
                out_orig,
                DIM,
                BLK,
                row_hi=self.n,
            )
            inside = self.ids < self.n
            a, b = out_mx[inside].float(), out_orig[inside].float()
            cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))
            print(
                "[engram mxfp4] packed kernel vs fp8 kernel on the original rows: "
                f"cosine mean {cos.mean():.6f} min {cos.min():.6f} "
                f"over {cos.numel()} lookups"
            )
            self.assertGreater(cos.mean().item(), 0.99)

    def test_embedding_on_host_tables_and_device(self):
        for layout in (None, "per_rank", "shared"):
            embed = make_embed(self.n, packed=True, host_layout=layout)
            self.keep.append(embed)
            embed.weight.weight_loader(embed.weight, self.packed)
            embed.scale.weight_loader(embed.scale, self.scale)
            embed.finish_load(label="test")
            # The shared layout reads every id it is given: real hash ids are
            # always inside the table, so keep them inside here too.
            ids = self.ids.clamp(max=self.n - 1).view(256, 48)
            got = embed(ids)
            self.assertEqual(
                (got.dtype, tuple(got.shape)), (torch.bfloat16, (256, 48, DIM))
            )
            want = self.ref[ids.cpu()]
            self.assertTrue(torch.equal(bits(got), bits(want)), layout)
            where = layout or "device"
            print(f"[engram mxfp4] EngramEmbedding lookup, table {where}: equal")

    def test_scheduler_staging(self):
        embed = make_embed(self.n, packed=True, host_layout="shared")
        self.keep.append(embed)
        embed.weight.weight_loader(embed.weight, self.packed)
        embed.scale.weight_loader(embed.scale, self.scale)
        stager = EngramRowStager.__new__(EngramRowStager)
        stager.tables = [
            (embed.weight.data.view(torch.uint8), embed.scale.data.view(torch.uint8))
        ]
        stager.pool = ThreadPoolExecutor(4)
        ids = self.ids_cpu.clamp(max=self.n - 1)
        weight, scale = stager.tables[0]
        rows = stager._gather(weight, ids)
        self.assertEqual(tuple(rows.shape), (ids.numel(), 128))
        scales = stager._gather(scale, ids)
        copied = torch.cuda.Event()
        rows, scales = rows.cuda(), scales.cuda()
        copied.record()
        got = embed.dequant_staged(rows, scales, copied, torch.Size([256, 48]))
        want = self.ref[ids].view(256, 48, DIM)
        self.assertTrue(torch.equal(bits(got), bits(want)))
        stager.pool.shutdown()

    def test_peer_kernel_two_shards(self):
        """Two shards in separate allocations, biased the way _PeerShards.connect
        biases them, read by the packed peer kernel."""
        split = 2000
        shards, bias_w, bias_s = [], [], []
        for start, stop in ((0, split), (split, self.n)):
            rows = stop - start
            buf = torch.cat(
                [
                    self.packed[start:stop].flatten(),
                    self.scale[start:stop].view(torch.uint8).flatten(),
                ]
            ).cuda()
            shards.append(buf)
            row_bytes, n_scales = DIM // 2, DIM // BLK
            bias_w.append(buf.data_ptr() - start * row_bytes)
            bias_s.append(buf.data_ptr() + rows * row_bytes - start * n_scales)
        starts = torch.tensor([0, split], dtype=torch.int64, device="cuda")
        ids = self.ids.clamp(max=self.n - 1).view(256, 48)
        out = torch.empty(256, 48, DIM, dtype=torch.bfloat16, device="cuda")
        engram_gather_peer(
            ids,
            out,
            torch.tensor(bias_w, device="cuda"),
            torch.tensor(bias_s, device="cuda"),
            starts,
            DIM,
            BLK,
            packed=True,
        )
        want = self.ref[ids.cpu()]
        want = torch.where(
            want == 0, torch.zeros_like(want), want
        )  # the peer kernel writes +0
        self.assertTrue(torch.equal(bits(out), bits(want)))

    def test_peer_shards_object_one_rank(self):
        """_PeerShards' own views, bias arithmetic and lookup, on one rank."""
        group = SimpleNamespace(cpu_group=None, rank_in_group=0, world_size=1)
        peer = _PeerShards(self.n, DIM, group, packed=True)
        self.keep.append(peer)
        weight, scale = peer.views()
        self.assertEqual(
            (weight.dtype, tuple(weight.shape)), (torch.uint8, (self.n, 128))
        )
        weight.copy_(self.packed)
        scale.copy_(self.scale)

        def gather_object(out, obj, group=None):
            out[0] = obj

        with (
            patch(
                "sglang.srt.utils.cuda_vmm_utils.exchange_shareable_handles",
                return_value=(False, {}),
            ),
            patch("torch.distributed.all_gather_object", side_effect=gather_object),
            patch("torch.distributed.barrier"),
        ):
            peer.connect(0, "test")
        ids = self.ids.clamp(max=self.n - 1).view(256, 48)
        got = peer.lookup(ids)
        want = self.ref[ids.cpu()]
        want = torch.where(want == 0, torch.zeros_like(want), want)
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_timing_against_fp8(self):
        """Same rows, same ids: the fp8 kernel reads 264 bytes a row, the packed
        one 136. Printed, not asserted."""
        fp8 = self.fp8 if self.fp8 is not None else transcode_to_fp8(self.packed)
        fp8_scale = self.fp8_scale if self.fp8_scale is not None else self.scale
        ids = self.ids.clamp(max=self.n - 1)
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
        lines = []
        for where in ("device", "pinned host"):
            if where == "device":
                tables = {
                    "fp8": (fp8.cuda(), fp8_scale.cuda()),
                    "mxfp4": (self.packed.cuda(), self.scale.cuda()),
                }
            else:
                tables = {
                    "fp8": (fp8.pin_memory(), fp8_scale.pin_memory()),
                    "mxfp4": (self.packed.pin_memory(), self.scale.pin_memory()),
                }
            us = {}
            for name, (w, s) in tables.items():
                packed = name == "mxfp4"
                us[name] = time_us(
                    lambda w=w, s=s, packed=packed: engram_gather(
                        w.data_ptr(),
                        s.data_ptr(),
                        ids,
                        out,
                        DIM,
                        BLK,
                        row_hi=self.n,
                        packed=packed,
                    )
                )
            n = ids.numel()
            lines.append(
                f"[engram mxfp4] timing, {where} table, {n} lookups: "
                f"fp8 kernel {us['fp8']:.1f} us, packed kernel {us['mxfp4']:.1f} us "
                f"(ratio {us['mxfp4'] / us['fp8']:.2f}; "
                f"table bytes read {n * 264} vs {n * 136})"
            )
        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
