"""DeepSeek-V4.1 HiCache under pipeline parallelism: a stage keeps copies of the
ratio-1/2 producers of earlier stages (deepseek_v41_pp_sources). Their host copies
must come back on a host hit, so they load at the stage's first transfer step, and
only with SGLANG_DSV41_PP_HICACHE; without it the stage refuses the host tier.

Real DeepSeekV4TokenToKVPool stages of DeepSeek-V4.1-Flash at toy token counts
(FP8 V4 layout, head 448 + 64, indexer 128, page 256), the real host stack and the
real per-layer transfer loop; only the copy kernels are CPU reference copies."""

import itertools
import unittest
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.environ import envs
from sglang.srt.mem_cache import l2_transfer
from sglang.srt.mem_cache import memory_pool_host as host_module
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    collect_sources_by_ratio,
)
from sglang.srt.mem_cache.deepseek_v41_pp_sources import (
    replayed_sources,
    replayed_sources_of,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE = 256
# DeepSeek-V4.1-Flash(-1017) config.json: 40 decoder layers, 3 MTP entries.
RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
KV_SOURCES = [2, 8, 14, 20]
INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]
CANDIDATE = 20
V41_FLASH = SimpleNamespace(
    compress_ratios=RATIOS,
    kv_source_layer_ids=KV_SOURCES,
    index_source_layer_ids=INDEX_SOURCES,
    candidate_source_layer_id=CANDIDATE,
    num_hidden_layers=40,
)
LOW_RATIO_REGIONS = (
    PoolName.DEEPSEEK_V4_C1,
    PoolName.DEEPSEEK_V4_C1_INDEXER,
    PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE,
    PoolName.DEEPSEEK_V4_C2,
    PoolName.DEEPSEEK_V4_C2_INDEXER,
    PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE,
)


def _stages(*sizes):
    out, start = [], 0
    for size in sizes:
        out.append(range(start, start + size))
        start += size
    return out


def _make_pool(stage, full_pages=4, swa_pages=2):
    replayed = replayed_sources_of(V41_FLASH, stage)
    return DeepSeekV4TokenToKVPool(
        max_num_reqs=4,
        swa_size=swa_pages * PAGE,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=PAGE,
        swa_page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=40,
        start_layer=stage.start,
        end_layer=stage.stop,
        device="cpu",
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        kv_source_layers=KV_SOURCES,
        full_size=full_pages * PAGE,
        kv_layout=KVLayout.V4,
        compressed_kv_layout=None,
        replayed_layers=replayed,
    )


def _build_stack(pool, num_host_pages=3, swa_host_pages=2):
    params = SimpleNamespace(
        page_size=PAGE,
        token_to_kv_pool_allocator=SimpleNamespace(
            swa_attn_allocator=SimpleNamespace(
                alloc=lambda n: None, free=lambda indices: None
            )
        ),
        mtp_draft_device_pools=(),
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )
    memory = SimpleNamespace(
        hicache_mem_layout="layer_first",
        hicache_write_policy="write_through_selective",
        hicache_io_backend="kernel",
        hicache_host_memory_mode="cache",
    )
    with (
        patch.object(assembler, "get_memory", return_value=memory),
        patch.object(
            assembler,
            "_deepseek_v4_num_host_pages",
            return_value=(num_host_pages, swa_host_pages, num_host_pages),
        ),
        patch.object(assembler, "_get_allocator_type", return_value="default"),
        patch.object(
            assembler,
            "DeepSeekV4PagedHostPool",
            partial(DeepSeekV4PagedHostPool, pin_memory=False),
        ),
        patch.object(assembler, "HybridCacheController"),
    ):
        group, _ = assembler.build_deepseek_v4_hicache_stack(
            params=params, kvcache=pool, load_cache_event=None, storage_backend=None
        )
    return group


def _stock_low_ratio_entries(pool, num_host_pages=3):
    """The low-ratio entries as the stack builds them without this change."""
    infos = assembler._dsv4_pool_infos(pool)
    with (
        patch.object(
            assembler,
            "get_memory",
            return_value=SimpleNamespace(hicache_mem_layout="layer_first"),
        ),
        patch.object(assembler, "_get_allocator_type", return_value="default"),
        patch.object(
            assembler,
            "DeepSeekV4PagedHostPool",
            partial(DeepSeekV4PagedHostPool, pin_memory=False),
        ),
    ):
        return assembler._dsv4_low_ratio_entries(
            pool, infos, PAGE, num_host_pages, len(pool.stage_layer_ids)
        )


def _reached(entry, steps):
    return sorted({entry.layer_mapper(step) for step in range(steps)} - {None})


# -- the copy kernels, as CPU reference copies of whole page rows ----------------
_BUFFERS: dict[int, torch.Tensor] = {}


def _rows(t):
    return t.view(torch.uint8).reshape(t.shape[0], -1)


def _ref_all_layer(
    src_layers, dst_layers, src_indices, dst_indices, item_size, num_layers
):
    for layer in range(num_layers):
        src = _rows(_BUFFERS[int(src_layers[layer])])
        dst = _rows(_BUFFERS[int(dst_layers[layer])])
        assert src.shape[1] == dst.shape[1] == item_size
        dst[dst_indices] = src[src_indices]


def _ref_per_layer(src, dst, src_indices, dst_indices, item_size):
    src, dst = _rows(src), _rows(dst)
    assert src.shape[1] == dst.shape[1] == item_size
    dst[dst_indices] = src[src_indices]


class _Event:
    def __init__(self, enable_timing=False):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True


class _Stream:
    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


class _CpuDeviceModule:
    Event = _Event
    Stream = _Stream

    @staticmethod
    def current_stream():
        return _Stream()

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


@contextmanager
def _cpu_transfers(entries):
    _BUFFERS.clear()
    for entry in entries:
        host = entry.host_pool
        for tensor in list(getattr(host, "device_buffers", ())) + list(
            getattr(host, "data_refs", ())
        ):
            _BUFFERS[tensor.data_ptr()] = tensor
    with (
        patch.object(l2_transfer, "device_module", _CpuDeviceModule),
        # create: a CPU-only process may not import the CUDA kernels at all.
        patch.object(
            host_module, "transfer_kv_all_layer_mla", _ref_all_layer, create=True
        ),
        patch.object(
            host_module, "transfer_kv_per_layer_mla", _ref_per_layer, create=True
        ),
    ):
        yield L2TransferEngine("kernel")


def _token_indices(pages):
    return torch.cat(
        [torch.arange(p * PAGE, (p + 1) * PAGE, dtype=torch.int64) for p in pages]
    )


def _round_trip(pool, entries):
    """Fill the device rows, back them up, clear them, load them back; per host
    entry and layer, whether the device rows came back."""
    torch.manual_seed(0)
    device_pages, host_pages = [1, 3], [0, 2]
    swa_device_pages, swa_host_pages = [1], [1]
    snapshots, transfers = {}, []
    for entry in entries:
        host = entry.host_pool
        if not getattr(host, "device_buffers", None):
            continue  # the logical KV anchor holds no data
        swa = entry.name == PoolName.SWA
        dev_idx = _token_indices(swa_device_pages if swa else device_pages)
        host_idx = _token_indices(swa_host_pages if swa else host_pages)
        rows = torch.tensor(swa_device_pages if swa else device_pages)
        for layer, buffer in enumerate(host.device_buffers):
            data = _rows(buffer)
            data.copy_(torch.randint(0, 256, data.shape, dtype=torch.uint8))
            snapshots[(entry.name, layer)] = (rows, data[rows].clone())
        transfers.append(
            L2Transfer(
                host_pool=host,
                device_pool=entry.device_pool,
                host_indices=host_idx,
                device_indices=dev_idx,
                layer_mapper=entry.layer_mapper,
            )
        )
    with _cpu_transfers(entries) as engine:
        engine.submit_device_to_host(transfers)
        for entry in entries:
            for buffer in getattr(entry.host_pool, "device_buffers", ()):
                _rows(buffer).zero_()
        engine.submit_host_to_device(
            transfers, transfer_layer_id_max=len(pool.stage_layer_ids)
        )
    restored = {}
    for (name, layer), (rows, expected) in snapshots.items():
        entry = next(e for e in entries if e.name == name)
        restored[(name, layer)] = torch.equal(
            _rows(entry.host_pool.device_buffers[layer])[rows], expected
        )
    return restored


class _V41Stages(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(page_size=PAGE)
        override.install()
        self.addCleanup(override.restore)


class TestProductionSplit(_V41Stages):
    def test_stage_plans_and_owned_producers(self):
        stages = _stages(13, 13, 14)
        pools = [_make_pool(stage) for stage in stages]
        # Stage 2 replays 24 too, an index source that owns no storage.
        self.assertEqual([p.replayed_layers for p in pools], [(), (8,), (20, 24)])
        layer_ids = [
            {
                info.pool_name: info.layer_ids
                for info in p.get_device_pool_infos()
                if info.pool_name in LOW_RATIO_REGIONS
            }
            for p in pools
        ]
        self.assertEqual(layer_ids[0][PoolName.DEEPSEEK_V4_C2], (2, 8))
        self.assertEqual(layer_ids[1][PoolName.DEEPSEEK_V4_C2], (8, 14))
        self.assertEqual(layer_ids[1][PoolName.DEEPSEEK_V4_C1], (20,))
        self.assertEqual(layer_ids[2][PoolName.DEEPSEEK_V4_C1], (20,))
        self.assertNotIn(PoolName.DEEPSEEK_V4_C2, layer_ids[2])

    def test_stock_mapping_never_loads_earlier_producers(self):
        """The defect: their steps are negative, the load loop never visits them."""
        for stage, region, lost in (
            (range(13, 26), PoolName.DEEPSEEK_V4_C2, 0),
            (range(26, 40), PoolName.DEEPSEEK_V4_C1, 0),
        ):
            pool = _make_pool(stage)
            entries = {e.name: e for e in _stock_low_ratio_entries(pool)}
            steps = len(pool.stage_layer_ids)
            for name in (region, PoolName(f"{region.value}_indexer")):
                self.assertNotIn(lost, _reached(entries[name], steps), name)

    def test_refused_without_the_switch(self):
        with envs.SGLANG_DSV41_PP_HICACHE.override(False):
            group = _build_stack(_make_pool(range(0, 13)))
            self.assertIn(PoolName.DEEPSEEK_V4_C2, group.entry_map)
            for stage in (range(13, 26), range(26, 40)):
                with self.assertRaisesRegex(ValueError, "SGLANG_DSV41_PP_HICACHE=1"):
                    _build_stack(_make_pool(stage))

    def test_switch_loads_every_host_layer_once(self):
        with envs.SGLANG_DSV41_PP_HICACHE.override(True):
            for stage in _stages(13, 13, 14):
                pool = _make_pool(stage)
                group = _build_stack(pool)
                steps = len(pool.stage_layer_ids)
                for entry in group.entries:
                    host = entry.host_pool
                    if not getattr(host, "device_buffers", None):
                        continue
                    visits = [entry.layer_mapper(s) for s in range(steps)]
                    hits = [v for v in visits if v is not None]
                    self.assertEqual(
                        sorted(hits), list(range(host.layer_num)), entry.name
                    )
                if pool.replayed_layers:
                    first = 8 if stage.start == 13 else 20
                    region = (
                        PoolName.DEEPSEEK_V4_C2
                        if stage.start == 13
                        else PoolName.DEEPSEEK_V4_C1
                    )
                    info = {i.pool_name: i for i in pool.get_device_pool_infos()}[
                        region
                    ]
                    self.assertEqual(
                        group.entry_map[region].layer_mapper(0),
                        info.layer_ids.index(first),
                    )

    def test_host_bytes_per_token_match_the_device_pool(self):
        """653 / 1,306 / 653 B a token on the three stages, as the boot logs."""
        with envs.SGLANG_DSV41_PP_HICACHE.override(True):
            per_stage = []
            for stage in _stages(13, 13, 14):
                group = _build_stack(_make_pool(stage))
                full = sum(
                    e.host_pool.item_bytes * e.host_pool.layer_num
                    for e in group.entries
                    if e.name in LOW_RATIO_REGIONS
                )
                swa = group.entry_map[PoolName.SWA].host_pool
                per_stage.append((full / PAGE, swa.item_bytes * swa.layer_num / PAGE))
        self.assertEqual([f for f, _ in per_stage], [653.0, 1306.0, 653.0])
        self.assertEqual([s for _, s in per_stage], [7605.0, 7605.0, 8190.0])


class TestFirstStageHeavySplit(_V41Stages):
    """14/13/13: only the last stage keeps earlier producers (20, and 24 with no
    storage of its own)."""

    def test_only_the_last_stage_replays(self):
        pools = [_make_pool(stage) for stage in _stages(14, 13, 13)]
        self.assertEqual([p.replayed_layers for p in pools], [(), (), (20, 24)])
        layer_ids = [
            {
                info.pool_name: info.layer_ids
                for info in p.get_device_pool_infos()
                if info.pool_name in LOW_RATIO_REGIONS
            }
            for p in pools
        ]
        self.assertEqual(layer_ids[0][PoolName.DEEPSEEK_V4_C2], (2, 8))
        self.assertEqual(layer_ids[1][PoolName.DEEPSEEK_V4_C2], (14,))
        self.assertEqual(layer_ids[1][PoolName.DEEPSEEK_V4_C1], (20,))
        self.assertEqual(layer_ids[2][PoolName.DEEPSEEK_V4_C1], (20,))
        self.assertNotIn(PoolName.DEEPSEEK_V4_C2, layer_ids[2])

    def test_refused_without_the_switch_on_the_last_stage_only(self):
        with envs.SGLANG_DSV41_PP_HICACHE.override(False):
            for stage in _stages(14, 13)[:2]:
                _build_stack(_make_pool(stage))
            with self.assertRaisesRegex(ValueError, "SGLANG_DSV41_PP_HICACHE=1"):
                _build_stack(_make_pool(range(27, 40)))

    def test_host_hit_restores_every_layer_with_the_switch(self):
        with envs.SGLANG_DSV41_PP_HICACHE.override(True):
            per_stage = []
            for stage in _stages(14, 13, 13):
                pool = _make_pool(stage)
                group = _build_stack(pool)
                restored = _round_trip(pool, group.entries)
                self.assertTrue(restored)
                self.assertTrue(all(restored.values()), (stage, restored))
                full = sum(
                    e.host_pool.item_bytes * e.host_pool.layer_num
                    for e in group.entries
                    if e.name in LOW_RATIO_REGIONS
                )
                swa = group.entry_map[PoolName.SWA].host_pool
                per_stage.append((full / PAGE, swa.item_bytes * swa.layer_num / PAGE))
            region = group.entry_map[PoolName.DEEPSEEK_V4_C1]
            self.assertEqual(region.layer_mapper(0), 0)
        # One ratio-1 layer stores a token per token, a ratio-2 layer one per two.
        self.assertEqual([f for f, _ in per_stage], [653.0, 979.5, 653.0])
        self.assertEqual([s for _, s in per_stage], [8190.0, 7605.0, 7605.0])

    def test_direct_linker_refuses_a_replaying_stage(self):
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _build_deepseek_v4_device_pool_group,
        )

        with self.assertRaisesRegex(ValueError, r"producers \(20, 24\)"):
            _build_deepseek_v4_device_pool_group(_make_pool(range(27, 40)), PAGE)


class TestRoundTrip(_V41Stages):
    def test_host_hit_restores_every_layer_with_the_switch(self):
        with envs.SGLANG_DSV41_PP_HICACHE.override(True):
            for stage in _stages(13, 13, 14):
                pool = _make_pool(stage)
                group = _build_stack(pool)
                restored = _round_trip(pool, group.entries)
                self.assertTrue(restored)
                self.assertTrue(all(restored.values()), (stage, restored))

    def test_stock_mapping_leaves_earlier_producers_stale(self):
        for stage, region in (
            (range(13, 26), PoolName.DEEPSEEK_V4_C2),
            (range(26, 40), PoolName.DEEPSEEK_V4_C1),
        ):
            pool = _make_pool(stage)
            entries = _stock_low_ratio_entries(pool)
            restored = _round_trip(pool, entries)
            replayed_index = 0  # the earlier producer is the region's first layer
            self.assertFalse(restored[(region, replayed_index)], stage)
            own = [ok for (name, layer), ok in restored.items() if name == region]
            self.assertEqual(own.count(False), 1, (stage, restored))


class TestWaits(_V41Stages):
    def test_earlier_producer_waits_for_the_first_step(self):
        pool = _make_pool(range(13, 26))
        waited = []
        pool.register_layer_transfer_counter(
            SimpleNamespace(wait_until=lambda step: waited.append(step))
        )
        pool.get_index_k_with_scale_buffer(8)
        pool.get_index_k_with_scale_buffer(14)
        pool.get_extra_key_buffer(13)
        pool.get_extra_key_buffer(20)
        self.assertEqual(waited, [0, 1, 0, 7])

    def test_without_host_tier_nothing_waits(self):
        pool = _make_pool(range(26, 40))
        pool.wait_layer_transfer(20)  # no counter registered: a no-op


class TestEverySplit(CustomTestCase):
    @staticmethod
    def _owned(stage):
        replayed = replayed_sources(RATIOS, KV_SOURCES, INDEX_SOURCES, CANDIDATE, stage)
        return collect_sources_by_ratio(RATIOS, KV_SOURCES, stage, replayed)

    def test_every_pp2_to_pp4_split_has_a_free_first_step(self):
        checked = 0
        for pp in (2, 3, 4):
            for cuts in itertools.combinations(range(1, 40), pp - 1):
                bounds = (0, *cuts, 40)
                for start, stop in zip(bounds, bounds[1:]):
                    stage = range(start, stop)
                    for ratio, layers in self._owned(stage).items():
                        if ratio not in (1, 2):
                            continue
                        mapping = assembler._dsv41_replayed_layer_mapping(
                            tuple(layers), start
                        )
                        self.assertEqual(
                            sorted(mapping.values()), list(range(len(layers)))
                        )
                        self.assertTrue(all(0 <= k < stop - start for k in mapping))
                        checked += 1
        self.assertGreater(checked, 10_000)

    def test_two_producers_or_a_taken_first_step_refuse(self):
        # A stage starting at its own source with an earlier one kept beside it.
        with self.assertRaisesRegex(ValueError, "only one earlier-stage producer"):
            assembler._dsv41_replayed_layer_mapping((2, 8), 8)
        with self.assertRaisesRegex(ValueError, "only one earlier-stage producer"):
            assembler._dsv41_replayed_layer_mapping((2, 8), 9)
        self.assertEqual(
            assembler._dsv41_replayed_layer_mapping((8, 14), 13), {0: 0, 1: 1}
        )


if __name__ == "__main__":
    unittest.main()
