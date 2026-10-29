"""DeepSeek-V4.1 Engram tables stored as MXFP4: config detection, the 136-byte
row table (device and host layouts), the raw-byte loader, the torch lookup, and
the format checked against real checkpoint rows. CPU only.

The real-row tests read a few thousand rows of the Engram shards of two
checkpoints: set SGLANG_TEST_DSV41_FP8_DIR to the original
DeepSeek-V4.1-Flash-1017 snapshot (FP8 tables) and SGLANG_TEST_DSV41_MXFP4_DIR to
a checkpoint whose Engram tables are MXFP4 (DeepSeek-V4.1-Flash-1017-NVFP4).
Without them those tests are skipped.
"""

import json
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.arg_groups.engram_config import (
    EngramConfig,
    EngramLookup,
    EngramPlacement,
    EngramReplication,
)
from sglang.srt.layers import engram as engram_mod
from sglang.srt.layers.engram import (
    Engram,
    EngramEmbedding,
    engram_table_is_mxfp4,
    unpack_mxfp4_codes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DIM = 256
MXFP4_KEYS = {
    "quant_method": "fp8",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "engram_embed_quant": "mxfp4",
    "engram_embed_block_size": 32,
    "engram_embed_scale_dtype": "ue8m0",
    "engram_embed_layers": [1, 14],
}
FP8_KEYS = {k: v for k, v in MXFP4_KEYS.items() if not k.startswith("engram")}
TABLE_ROWS = {1: 384006168, 14: 384016682}


def e2m1_reference(code: int) -> float:
    """E2M1 from its definition: sign bit 3, exponent bits 2-1, mantissa bit 0."""
    sign = -1.0 if code & 8 else 1.0
    exp, man = (code >> 1) & 3, code & 1
    mag = 0.5 * man if exp == 0 else 2.0 ** (exp - 1) * (1 + 0.5 * man)
    return sign * mag


E2M1 = torch.tensor([e2m1_reference(c) for c in range(16)])


def decode_reference(packed, scale, low_first=True):
    """bf16 rows of MXFP4 bytes [n, 128] and e8m0 scales [n, 8], written from
    the format's definition, independent of the code under test."""
    lo, hi = (packed & 0x0F).long(), (packed >> 4).long()
    first, second = (lo, hi) if low_first else (hi, lo)
    vals = torch.stack([E2M1[first], E2M1[second]], -1).flatten(-2)
    return (vals.view(-1, DIM // 32, 32) * scale.float().unsqueeze(-1)).flatten(1)


def fp8_reference(weight, scale):
    """The lookup arithmetic of EngramEmbedding._owned_rows before this patch."""
    rows = weight.float().unflatten(-1, (-1, 32))
    return (rows * scale.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def parallel(tp_size=1, tp_rank=0):
    return SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_dp_size=1,
        attn_cp_size=1,
        tp_group=SimpleNamespace(rank_in_group=tp_rank),
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


class FakeHostTable:
    """_HostTable's interface over plain CPU memory (no pinning, no CUDA)."""

    def __init__(self, replication, nbytes, name, group):
        self.replication, self.nbytes, self.dirty = replication, nbytes, False
        self.bytes = torch.zeros(nbytes, dtype=torch.uint8)
        self.device_ptr = self.bytes.data_ptr()

    def finish_load(self, label):
        self.dirty = False


def make_embed(rows, packed, *, host_layout=None, tp_size=1, tp_rank=0):
    """An EngramEmbedding of ``rows`` rows split in two hash columns."""
    bounds = (0, rows // 2, rows)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                engram_mod, "get_parallel", return_value=parallel(tp_size, tp_rank)
            )
        )
        stack.enter_context(
            patch.object(
                engram_mod,
                "resolve_engram_config",
                return_value=engram_config(host_layout),
            )
        )
        if host_layout is not None:
            stack.enter_context(patch.object(engram_mod, "_HostTable", FakeHostTable))
            stack.enter_context(
                patch.object(engram_mod, "_sorted_host_reads", return_value=False)
            )
            stack.enter_context(
                patch.object(torch.cuda, "current_device", return_value=0)
            )
        return EngramEmbedding(
            rows, DIM, layer_id=1, column_bounds=bounds, packed=packed
        )


def random_mxfp4(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (rows, DIM // 2), generator=g, dtype=torch.uint8)
    scale = torch.randint(100, 140, (rows, DIM // 32), generator=g, dtype=torch.uint8)
    return packed, scale.view(torch.float8_e8m0fnu)


def random_fp8(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (rows, DIM), generator=g, dtype=torch.uint8)
    w = torch.where((w & 0x7F) == 0x7F, w - 1, w)  # 0x7f / 0xff are e4m3 NaN
    scale = torch.randint(100, 140, (rows, DIM // 32), generator=g, dtype=torch.uint8)
    return w.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)


def bits(t):
    return t.contiguous().view(torch.int16)


class TestConfigDetection(CustomTestCase):
    def test_no_key_means_fp8(self):
        for qc in (None, {}, dict(FP8_KEYS)):
            config = SimpleNamespace(quantization_config=qc)
            self.assertFalse(engram_table_is_mxfp4(config, 1))
        self.assertFalse(engram_table_is_mxfp4(SimpleNamespace(), 1))

    def test_mxfp4_keys(self):
        config = SimpleNamespace(quantization_config=dict(MXFP4_KEYS))
        self.assertTrue(engram_table_is_mxfp4(config, 1))
        self.assertTrue(engram_table_is_mxfp4(config, 14))
        # engram_embed_layers names the layers; an Engram layer not listed stays fp8.
        only_14 = dict(MXFP4_KEYS, engram_embed_layers=[14])
        self.assertFalse(
            engram_table_is_mxfp4(SimpleNamespace(quantization_config=only_14), 1)
        )
        # Block size 32 and ue8m0 are what MXFP4 means; leaving them out is allowed.
        bare = dict(FP8_KEYS, engram_embed_quant="mxfp4")
        self.assertTrue(
            engram_table_is_mxfp4(SimpleNamespace(quantization_config=bare), 1)
        )

    def test_other_formats_are_refused_by_key_and_value(self):
        for key, value in (
            ("engram_embed_quant", "nvfp4"),
            ("engram_embed_quant", "fp8"),
            ("engram_embed_block_size", 16),
            ("engram_embed_scale_dtype", "e4m3"),
        ):
            config = SimpleNamespace(
                quantization_config=dict(MXFP4_KEYS, **{key: value})
            )
            with self.assertRaises(ValueError) as ctx:
                engram_table_is_mxfp4(config, 1)
            self.assertIn(f"quantization_config.{key}={value!r}", str(ctx.exception))

    def test_engram_module_passes_the_format_to_its_table(self):
        layout = SimpleNamespace(
            layer_ids=(1, 14),
            num_embeddings=(64, 64),
            head_dim=DIM,
            max_ngram_size=4,
            n_heads=1,
            column_bounds=lambda i: (0, 32, 64),
        )
        for qc, expect in (
            (dict(MXFP4_KEYS), True),
            (dict(FP8_KEYS), False),
            (None, False),
        ):
            config = SimpleNamespace(
                quantization_config=qc, rms_norm_eps=1e-6, hidden_size=8, hc_mult=1
            )
            with (
                patch.object(engram_mod, "get_parallel", return_value=parallel()),
                patch.object(
                    engram_mod, "ReplicatedLinear", return_value=nn.Identity()
                ),
                patch.object(
                    engram_mod, "resolve_engram_config", return_value=engram_config()
                ),
            ):
                module = Engram(config, 14, layout)
            self.assertEqual(module.embed.packed, expect)
            self.assertEqual(
                module.embed.weight.dtype,
                torch.uint8 if expect else torch.float8_e4m3fn,
            )
            self.assertEqual(module.embed.weight.shape[1], DIM // 2 if expect else DIM)
        bad = dict(MXFP4_KEYS, engram_embed_quant="int4")
        config = SimpleNamespace(
            quantization_config=bad, rms_norm_eps=1e-6, hidden_size=8, hc_mult=1
        )
        with (
            patch.object(engram_mod, "get_parallel", return_value=parallel()),
            patch.object(engram_mod, "ReplicatedLinear", return_value=nn.Identity()),
            self.assertRaises(ValueError),
        ):
            Engram(config, 14, layout)


class TestPackedTableAndLoader(CustomTestCase):
    def test_device_table_is_136_bytes_per_row(self):
        embed = make_embed(40, packed=True)
        self.assertEqual(
            (embed.weight.dtype, tuple(embed.weight.shape)), (torch.uint8, (40, 128))
        )
        self.assertEqual(
            (embed.scale.dtype, tuple(embed.scale.shape)),
            (torch.float8_e8m0fnu, (40, 8)),
        )
        per_row = (
            embed.weight.element_size() * embed.weight.shape[1] + embed.scale.shape[1]
        )
        self.assertEqual(per_row, 136)

    def test_host_table_layouts(self):
        rows = 40
        for layout, tp_rank in (("shared", 0), ("per_rank", 1)):
            embed = make_embed(
                rows, packed=True, host_layout=layout, tp_size=2, tp_rank=tp_rank
            )
            n = rows if layout == "shared" else embed.rows
            table = embed.host_table
            self.assertEqual(table.nbytes, n * 136, layout)
            self.assertEqual(tuple(embed.weight.shape), (n, 128))
            self.assertEqual(embed.weight.dtype, torch.uint8)
            self.assertEqual(tuple(embed.scale.shape), (n, 8))
            base = table.bytes.data_ptr()
            self.assertEqual(embed.weight.data_ptr(), base)
            self.assertEqual(embed.scale.data_ptr(), base + n * 128)
            self.assertEqual(embed._table_ptrs(), (base, base + n * 128))

    def test_loader_copies_raw_bytes_for_every_layout(self):
        rows = 40
        packed, scale = random_mxfp4(rows, seed=1)
        cases = [(None, 1, 0), ("shared", 2, 1), ("per_rank", 2, 1)]
        for layout, tp_size, tp_rank in cases:
            embed = make_embed(
                rows, packed=True, host_layout=layout, tp_size=tp_size, tp_rank=tp_rank
            )
            for dtype in (
                torch.uint8,
                torch.int8,
            ):  # U8 in the checkpoint; I8 is the same bytes
                embed.weight.weight_loader(embed.weight, packed.view(dtype))
                embed.scale.weight_loader(embed.scale, scale)
                own = slice(embed.row_start, embed.row_start + embed.rows)
                local = (
                    slice(own.start, own.stop)
                    if layout == "shared"
                    else slice(0, embed.rows)
                )
                self.assertTrue(
                    torch.equal(embed.weight.data[local], packed[own]), layout
                )
                self.assertTrue(
                    torch.equal(
                        embed.scale.data[local].view(torch.uint8),
                        scale[own].view(torch.uint8),
                    )
                )
            if layout == "shared":
                # A rank writes only its own rows of the shared table.
                self.assertEqual(int(embed.weight.data[: embed.row_start].sum()), 0)
            if layout is not None:
                self.assertTrue(embed.host_table.dirty)

    def test_loader_refuses_a_table_that_is_not_mxfp4(self):
        embed = make_embed(16, packed=True)
        fp8, _ = random_fp8(16)
        for wrong in (
            fp8,
            torch.zeros(16, 256, dtype=torch.uint8),
            torch.zeros(16, 64, dtype=torch.uint8),
        ):
            with self.assertRaises(ValueError) as ctx:
                embed.weight.weight_loader(embed.weight, wrong)
            self.assertIn("engram_embed_quant='mxfp4'", str(ctx.exception))
            self.assertIn(str(wrong.dtype), str(ctx.exception))


class TestPackedTorchLookup(CustomTestCase):
    def test_every_byte_and_scale_edges(self):
        # Rows cover all 256 byte values twice; scales include 2**-127 and 2**127.
        rows = 4
        packed = (
            torch.arange(256, dtype=torch.uint8).repeat(rows, 1)[:, :128].contiguous()
        )
        packed[1] = torch.arange(128, 256, dtype=torch.uint8)
        packed[2:] = packed[:2].flip(-1)
        scale = torch.tensor(
            [[0, 1, 64, 126, 127, 128, 200, 254]] * rows, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        embed = make_embed(rows, packed=True)
        embed.weight.weight_loader(embed.weight, packed)
        embed.scale.weight_loader(embed.scale, scale)
        ids = torch.tensor([[0, 1], [2, 3], [3, 0]])
        got = embed._owned_rows(ids)
        want = decode_reference(packed, scale).to(torch.bfloat16)[ids]
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(tuple(got.shape), (3, 2, DIM))
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_unpack_order_and_values(self):
        # Byte 0x21: code 1 (0.5) in the low nibble, code 2 (1.0) in the high one.
        self.assertEqual(
            unpack_mxfp4_codes(torch.tensor([0x21], dtype=torch.uint8)).tolist(),
            [0.5, 1.0],
        )
        codes = torch.arange(16, dtype=torch.uint8)
        values = unpack_mxfp4_codes(codes)[0::2]
        self.assertTrue(torch.equal(values, E2M1))
        self.assertTrue(torch.equal(torch.signbit(values), torch.arange(16) >= 8))

    def test_rows_outside_the_shard_are_zero(self):
        rows = 40
        packed, scale = random_mxfp4(rows, seed=2)
        embed = make_embed(rows, packed=True, tp_size=2, tp_rank=1)
        embed.weight.weight_loader(embed.weight, packed)
        embed.scale.weight_loader(embed.scale, scale)
        ids = torch.arange(rows).view(-1, 1)
        got = embed._owned_rows(ids)[:, 0]
        want = decode_reference(packed, scale).to(torch.bfloat16)
        own = slice(embed.row_start, embed.row_start + embed.rows)
        self.assertTrue(torch.equal(bits(got[own]), bits(want[own])))
        self.assertEqual(int(got[: embed.row_start].abs().sum()), 0)


class TestFp8PathUnchanged(CustomTestCase):
    """Without engram_embed_quant everything is the fp8 table of before."""

    def test_tables_and_views(self):
        for layout in (None, "shared", "per_rank"):
            embed = make_embed(
                40, packed=False, host_layout=layout, tp_size=2, tp_rank=1
            )
            n = 40 if layout == "shared" else embed.rows
            self.assertFalse(embed.packed)
            self.assertEqual(
                (embed.weight.dtype, tuple(embed.weight.shape)),
                (torch.float8_e4m3fn, (n, DIM)),
            )
            self.assertEqual(
                (embed.scale.dtype, tuple(embed.scale.shape)),
                (torch.float8_e8m0fnu, (n, 8)),
            )
            if layout is not None:
                base = embed.host_table.bytes.data_ptr()
                self.assertEqual(embed.host_table.nbytes, n * 264)
                self.assertEqual(embed._table_ptrs(), (base, base + n * DIM))

    def test_loader_and_lookup(self):
        rows = 40
        weight, scale = random_fp8(rows, seed=3)
        embed = make_embed(rows, packed=False)
        embed.weight.weight_loader(embed.weight, weight)
        embed.scale.weight_loader(embed.scale, scale)
        self.assertTrue(
            torch.equal(embed.weight.data.view(torch.uint8), weight.view(torch.uint8))
        )
        ids = torch.randint(
            0, rows, (16, 3), generator=torch.Generator().manual_seed(4)
        )
        got = embed._owned_rows(ids)
        want = fp8_reference(weight[ids], scale[ids])
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_fp8_table_still_rejects_packed_rows_as_before(self):
        embed = make_embed(16, packed=False)
        with self.assertRaises(RuntimeError):
            embed.weight.weight_loader(
                embed.weight, torch.zeros(16, 128, dtype=torch.uint8)
            )

    def test_gather_entry_points_default_to_fp8(self):
        import inspect

        from sglang.kernels.ops.embeddings import engram_gather as g

        for fn in (g.engram_gather, g.engram_gather_peer):
            self.assertIs(inspect.signature(fn).parameters["packed"].default, False)


# ---- real checkpoint rows ----


def _shard_of(snapshot, name):
    with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
        return os.path.join(snapshot, json.load(f)["weight_map"][name])


def read_rows(snapshot, layer, lo, hi):
    from safetensors import safe_open

    w_name, s_name = (
        f"layers.{layer}.engram.embed.weight",
        f"layers.{layer}.engram.embed.scale",
    )
    with safe_open(_shard_of(snapshot, w_name), "pt") as f:
        return f.get_slice(w_name)[lo:hi], f.get_slice(s_name)[lo:hi]


def real_windows(layer, windows, size=512):
    n = TABLE_ROWS[layer]
    return [n * k // windows for k in range(windows - 1)] + [n - size]


FP8_DIR = os.environ.get("SGLANG_TEST_DSV41_FP8_DIR")
MXFP4_DIR = os.environ.get("SGLANG_TEST_DSV41_MXFP4_DIR")
SAMPLES = {1: real_windows(1, 8), 14: real_windows(14, 2)}  # 4,096 and 1,024 rows


def cosine(a, b):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))


@unittest.skipUnless(
    FP8_DIR and MXFP4_DIR,
    "set SGLANG_TEST_DSV41_FP8_DIR and SGLANG_TEST_DSV41_MXFP4_DIR",
)
class TestRealCheckpointRows(CustomTestCase):
    def _pairs(self, layer):
        for lo in SAMPLES[layer]:
            yield (
                read_rows(FP8_DIR, layer, lo, lo + 512),
                read_rows(MXFP4_DIR, layer, lo, lo + 512),
            )

    def test_nibble_order_on_real_rows(self):
        """The MXFP4 rows decoded low nibble first match the FP8 rows they were
        made from (cosine about 0.9934, the publisher's figure); high nibble first
        they do not. Requantizing the FP8 rows with the publisher's rule (scale
        = smallest power of two >= amax / 6, nearest E2M1 value, ties toward
        zero) and packing low nibble first gives back the checkpoint's bytes."""
        for layer in (1, 14):
            cos = {True: [], False: []}
            same_bytes = same_scales = total = 0
            for (w8, s8), (wm, sm) in self._pairs(layer):
                self.assertEqual(
                    (w8.dtype, wm.dtype, sm.dtype),
                    (torch.float8_e4m3fn, torch.uint8, torch.float8_e8m0fnu),
                )
                ref = (w8.float().view(-1, 8, 32) * s8.float().unsqueeze(-1)).flatten(1)
                for low_first in (True, False):
                    cos[low_first].append(
                        cosine(ref, decode_reference(wm, sm, low_first))
                    )
                v = ref.view(-1, 8, 32)
                exp = torch.ceil(
                    torch.log2((v.abs().amax(-1, keepdim=True) / 6).clamp(min=1e-38))
                )
                x = v / torch.pow(2.0, exp.clamp(-127, 127))
                code = torch.bucketize(x.abs().clamp(max=6), (E2M1[1:8] + E2M1[:7]) / 2)
                code = (code | ((x < 0).long() << 3)).view(-1, 128, 2)
                same_bytes += int(
                    ((code[..., 0] | (code[..., 1] << 4)) == wm.long()).sum()
                )
                same_scales += int(
                    (
                        torch.pow(2.0, exp.clamp(-127, 127))
                        .squeeze(-1)
                        .to(torch.float8_e8m0fnu)
                        .view(torch.uint8)
                        == sm.view(torch.uint8)
                    ).sum()
                )
                total += wm.numel()
            low, high = torch.cat(cos[True]), torch.cat(cos[False])
            print(
                f"layer {layer}: {low.numel()} rows, cosine low nibble first "
                f"mean {low.mean():.6f} min {low.min():.6f}; high nibble first "
                f"mean {high.mean():.6f}; requantized bytes equal "
                f"{same_bytes}/{total}, scales equal {same_scales}/{total // 16}"
            )
            self.assertGreater(low.mean().item(), 0.990)
            self.assertGreater(low.min().item(), 0.95)
            self.assertLess(high.mean().abs().item(), 0.1)
            self.assertEqual(same_bytes, total)
            self.assertEqual(same_scales, total // 16)

    def test_torch_lookup_of_real_rows(self):
        """The packed table's torch lookup gives the reference dequantization of
        the real rows, bit for bit, in the shape and dtype of the fp8 lookup."""
        for layer in (1, 14):
            mx = [read_rows(MXFP4_DIR, layer, lo, lo + 512) for lo in SAMPLES[layer]]
            packed, scale = torch.cat([m[0] for m in mx]), torch.cat([m[1] for m in mx])
            embed = make_embed(packed.shape[0], packed=True)
            embed.weight.weight_loader(embed.weight, packed)
            embed.scale.weight_loader(embed.scale, scale)
            ids = torch.randperm(
                packed.shape[0], generator=torch.Generator().manual_seed(layer)
            ).view(-1, 8)
            got = embed._owned_rows(ids)
            want = decode_reference(packed, scale).to(torch.bfloat16)[ids]
            self.assertEqual(
                (got.dtype, tuple(got.shape)), (torch.bfloat16, (*ids.shape, DIM))
            )
            self.assertTrue(torch.equal(bits(got), bits(want)))
            n = ids.numel()
            print(
                f"layer {layer}: torch lookup of {n} real rows equal to the reference"
            )


if __name__ == "__main__":
    unittest.main()
