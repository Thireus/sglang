"""DeepSeek-V4.1 images under pipeline parallelism: the first stage merges the
image embeddings, every stage finds the image positions in its own prompt IDs and
routes them with gate.bias_vl, and text batches keep the text path."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as dsv4
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_utils import pad_mm_tokens
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v41_vit import Aligner, ViT
from sglang.srt.multimodal.deepseek_v41_image_processing import image_token_types
from sglang.srt.multimodal.dsv41 import vl_routing
from sglang.srt.runtime_context import (
    SpawnRanks,
    get_context,
    get_parallel,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

IMAGE_ID = 60
HIDDEN = 16
VOCAB = 64
RATIO = 3
VISION = SimpleNamespace(
    vision_n_layers=2,
    vision_dim=32,
    vision_n_heads=4,
    vision_inter_dim=48,
    vision_patch_size=2,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=RATIO,
    dim=HIDDEN,
)


class _Batch:
    """The ForwardBatch fields the multimodal prefill path reads."""

    def __init__(self, mode, mm_inputs, prefix_lens, seq_lens):
        self.forward_mode = mode
        self.mm_inputs = mm_inputs
        self.extend_prefix_lens_cpu = prefix_lens
        self.extend_seq_lens_cpu = seq_lens
        self.mm_input_embeds = None

    def contains_image_inputs(self):
        return any(m is not None and m.contains_image_inputs() for m in self.mm_inputs)

    def contains_mm_inputs(self):
        return self.contains_image_inputs()


class _Stage(nn.Module):
    """One pipeline stage of DeepseekV4ForCausalLM, reduced to its multimodal entry."""

    get_image_feature = dsv4.DeepseekV4ForCausalLM.get_image_feature
    _prepare_mm_embeddings = dsv4.DeepseekV4ForCausalLM._prepare_mm_embeddings
    prepare_language_model_inputs = (
        dsv4.DeepseekV4ForCausalLM.prepare_language_model_inputs
    )

    def __init__(self, config, embed, tower=None, vision_routing=True):
        super().__init__()
        self.config = config
        self.embed = embed
        self.vision_routing = vision_routing
        self.vision = None
        if tower is not None:
            (
                self.vision,
                self.aligner,
                self.image_start,
                self.image_end,
                self.image_newline,
            ) = tower

    def get_input_embeddings(self):
        return self.embed


def _tower():
    torch.manual_seed(0)
    vision, aligner = ViT(VISION), Aligner(VISION)
    for parameter in list(vision.parameters()) + list(aligner.parameters()):
        nn.init.normal_(parameter, std=0.2)
    ends = [nn.Parameter(torch.randn(HIDDEN)) for _ in range(3)]
    return (vision, aligner, *ends)


class _Image:
    """One image request: text, the expanded image span, text."""

    def __init__(self, seed, n_vit_h=6, n_vit_w=7, before=3, after=4):
        torch.manual_seed(seed)
        self.n_vit_h, self.n_vit_w = n_vit_h, n_vit_w
        self.types = image_token_types(-(-n_vit_h // RATIO), -(-n_vit_w // RATIO))
        self.start, count = before, len(self.types)
        self.patches = torch.randn(n_vit_h * n_vit_w, 3, 2, 2)
        self.ids = (
            list(range(1, 1 + before))
            + [IMAGE_ID] * count
            + list(range(20, 20 + after))
        )
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=self.patches,
            offsets=[(before, before + count - 1)],
            model_specific_data={"n_vit_h": n_vit_h, "n_vit_w": n_vit_w},
        )
        item.set_hash(seed)
        self.mm_inputs = MultimodalInputs(mm_items=[item], im_token_id=IMAGE_ID)
        self.padded = list(pad_mm_tokens(array("q", self.ids), self.mm_inputs))
        self.span = slice(before, before + count)


class TestDeepseekV41PipelineVision(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", mm_attention_backend="sdpa"),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        mm_schedule.init_mm_embedding_cache(64 << 20)
        self.config = SimpleNamespace(
            image_token_id=IMAGE_ID, vision_downsample_ratio=RATIO, hidden_size=HIDDEN
        )
        torch.manual_seed(1)
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.tower = _tower()
        self.stages = [
            _Stage(self.config, self.embed, self.tower),
            _Stage(self.config, self.embed),
            _Stage(self.config, self.embed),
        ]

    def _expected(self, ids, image):
        """Text rows from the embedding; the span from the tower and the delimiters."""
        vision, aligner, start, end, newline = self.tower
        with torch.no_grad():
            rows = self.embed(torch.tensor(ids)).clone()
            features = aligner(
                vision(image.patches, image.n_vit_h, image.n_vit_w),
                image.n_vit_h,
                image.n_vit_w,
            )
            span = rows[image.span]
            span[image.types == 0] = start
            span[image.types == 1] = features
            span[image.types == 2] = newline
            span[image.types == 3] = end
            rows[image.span] = span
        return rows

    def _run(self, stage, ids, batch):
        scheduler_ids = torch.tensor(ids, dtype=torch.int64)
        before = scheduler_ids.clone()
        with torch.no_grad():
            mapped, embeds = stage.prepare_language_model_inputs(scheduler_ids, batch)
        # The scheduler's hash IDs key the radix and embedding caches.
        self.assertTrue(torch.equal(scheduler_ids, before))
        return mapped, embeds

    def test_first_stage_merges_and_every_stage_maps_image_positions(self):
        text = list(range(5, 13))
        image = _Image(seed=7)
        ids = text + image.padded
        mapped_by_stage = []
        for rank, stage in enumerate(self.stages):
            batch = _Batch(
                ForwardMode.EXTEND,
                [None, image.mm_inputs],
                [0, 0],
                [len(text), len(image.padded)],
            )
            mapped, embeds = self._run(stage, ids, batch)
            mapped_by_stage.append(mapped)
            if rank == 0:
                expected = torch.cat(
                    [self.embed(torch.tensor(text)), self._expected(image.ids, image)]
                )
                self.assertTrue(torch.equal(embeds, expected.detach()))
            else:
                self.assertIsNone(embeds)
        want = torch.tensor(text + image.ids)
        for mapped in mapped_by_stage:
            self.assertTrue(torch.equal(mapped, want))
        self.assertTrue(
            all(value >= MM_PAD_SHIFT_VALUE for value in image.padded[image.span])
        )

    def test_chunked_prefill_splits_an_image_across_chunks(self):
        image = _Image(seed=11)
        whole_batch = _Batch(
            ForwardMode.EXTEND, [image.mm_inputs], [0], [len(image.padded)]
        )
        _, whole = self._run(self.stages[0], image.padded, whole_batch)
        cut = image.start + 5
        for clear_cache in (False, True):
            with self.subTest(clear_cache=clear_cache):
                if clear_cache:
                    mm_schedule.init_mm_embedding_cache(64 << 20)
                pieces = []
                for begin, end in ((0, cut), (cut, len(image.padded))):
                    batch = _Batch(
                        ForwardMode.EXTEND, [image.mm_inputs], [begin], [end - begin]
                    )
                    for rank, stage in enumerate(self.stages):
                        mapped, embeds = self._run(
                            stage, image.padded[begin:end], batch
                        )
                        self.assertTrue(
                            torch.equal(mapped, torch.tensor(image.ids[begin:end]))
                        )
                        if rank == 0:
                            pieces.append(embeds)
                        else:
                            self.assertIsNone(embeds)
                self.assertTrue(torch.equal(torch.cat(pieces), whole))

    def test_text_batches_keep_the_text_path(self):
        text_a, text_b = list(range(5, 13)), list(range(30, 33))
        ids = torch.tensor(text_a + text_b)
        for rank, stage in enumerate(self.stages):
            batch = _Batch(
                ForwardMode.EXTEND, [None, None], [0, 0], [len(text_a), len(text_b)]
            )
            mapped, embeds = self._run(stage, ids.tolist(), batch)
            self.assertIsNone(embeds)
            self.assertTrue(torch.equal(mapped, ids))
            decode = _Batch(ForwardMode.DECODE, [None, None], [0, 0], [1, 1])
            step = torch.tensor([7, 9])
            same, embeds = stage.prepare_language_model_inputs(step, decode)
            self.assertIs(same, step)
            self.assertIsNone(embeds)

    def test_a_text_only_launch_leaves_scheduler_ids_alone(self):
        stage = _Stage(self.config, self.embed, vision_routing=False)
        image = _Image(seed=3)
        batch = _Batch(ForwardMode.EXTEND, [image.mm_inputs], [0], [len(image.padded)])
        mapped, embeds = self._run(stage, image.padded, batch)
        self.assertIsNone(embeds)
        self.assertTrue(torch.equal(mapped, torch.tensor(image.padded)))


class TestDeepseekV41StageCheckpoint(CustomTestCase):
    def test_each_stage_loads_its_share_of_the_vision_tensors(self):
        remap = dsv4.DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        names = {
            "tower": remap("vision.blocks.0.attn.wqkv.weight"),
            "aligner": remap("aligner.w1.weight"),
            "delimiter": remap("image_newline"),
            "bias_vl": remap("layers.13.ffn.gate.bias_vl"),
            "bias": remap("layers.13.ffn.gate.bias"),
        }
        self.assertEqual(names["tower"], "vision.blocks.0.attn.qkv_proj.weight")
        self.assertEqual(
            names["bias_vl"], "model.layers.13.mlp.gate.e_score_correction_bias_vl"
        )
        first = dsv4._VISION_ON_FIRST_STAGE
        cases = {
            # (holds_tower, routes_images): tower, aligner, delimiter, bias_vl, bias
            (False, False): ("vision", "vision", "vision", "gate.bias_vl", None),
            (True, True): (None, None, None, None, None),
            (False, True): (first, first, first, None, None),
        }
        for (holds, routes), expected in cases.items():
            with self.subTest(holds_tower=holds, routes_images=routes):
                got = tuple(
                    dsv4._v41_checkpoint_skip(
                        names[key], holds_tower=holds, routes_images=routes
                    )
                    for key in ("tower", "aligner", "delimiter", "bias_vl", "bias")
                )
                self.assertEqual(got, expected)

    def test_routes_images_falls_back_to_the_tower(self):
        self.assertTrue(dsv4._routes_images(SimpleNamespace(vision=object())))
        self.assertFalse(dsv4._routes_images(SimpleNamespace(vision=None)))
        self.assertTrue(
            dsv4._routes_images(SimpleNamespace(vision=None, vision_routing=True))
        )


class TestDeepseekV41VisionRouting(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = get_context().override_server_args()
        config.install()
        self.addCleanup(config.restore)
        initialize_moe_config()
        topology = get_parallel().override(
            attn_dp_size=1, attn_cp_size=1, attn_tp_size=1, attn_tp_rank=0
        )
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        for name in ("_is_cuda", "_is_hip", "_use_aiter"):
            scope = patch.object(topk_module, name, False)
            scope.start()
            self.addCleanup(scope.stop)

    @staticmethod
    def _batch(n):
        empty = torch.empty(0, dtype=torch.int32)
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.arange(n),
            req_pool_indices=empty,
            seq_lens=empty,
            out_cache_loc=empty,
            seq_lens_sum=0,
        )
        batch.num_token_non_padded = torch.tensor(n, dtype=torch.int32)
        batch.global_num_token_non_padded = batch.num_token_non_padded
        batch.dp_padding_mode = DpPaddingMode.SUM_LEN
        batch.global_dp_buffer_len = batch.tbo_padded_len = n
        return batch

    def test_image_rows_route_with_bias_vl_on_every_stage(self):
        experts, top_k = 8, 2
        torch.manual_seed(5)
        image = _Image(seed=19)
        ids = list(range(5, 9)) + image.padded
        stages = [
            _Stage(SimpleNamespace(image_token_id=IMAGE_ID), None, vision_routing=True)
            for _ in range(3)
        ]
        batch = _Batch(ForwardMode.EXTEND, [None, image.mm_inputs], [0, 0], [4, 1])
        for rank, stage in enumerate(stages):
            with self.subTest(stage=rank):
                # Only the first stage holds the tower; the IDs alone mark the rows.
                stage.vision = None
                mapped, _ = stage.prepare_language_model_inputs(
                    torch.tensor(ids), batch
                )
                bias = torch.randn(experts)
                bias_vl = torch.randn(experts)
                bias[rank] += 100.0
                bias_vl[experts - 1 - rank] += 100.0
                router = TopK(
                    top_k=top_k,
                    layer_id=rank,
                    renormalize=True,
                    scoring_func="sqrtsoftplus",
                    correction_bias=bias,
                    routed_scaling_factor=1.5,
                )
                moe = SimpleNamespace(
                    topk=router,
                    layer_id=rank,
                    config=SimpleNamespace(image_token_id=IMAGE_ID),
                    gate=SimpleNamespace(
                        e_score_correction_bias=bias,
                        e_score_correction_bias_vl=bias_vl,
                    ),
                )
                logits = torch.randn(len(ids), experts)
                with (
                    patch.object(vl_routing, "is_cuda", return_value=False),
                    forward_context(
                        ForwardContext(attn_backend=object()),
                        forward_batch=self._batch(len(ids)),
                    ),
                ):
                    out = vl_routing.vision_topk(moe, logits, mapped)
                scores = torch.nn.functional.softplus(logits).sqrt()
                is_image = mapped == IMAGE_ID
                self.assertEqual(int(is_image.sum()), len(image.types))
                rows = torch.where(is_image[:, None], bias_vl, bias)
                want = (scores + rows).topk(top_k, dim=-1).indices
                self.assertTrue(
                    torch.equal(
                        out.topk_ids.sort(-1).values, want.int().sort(-1).values
                    )
                )
                self.assertTrue(
                    bool((out.topk_ids[is_image] == experts - 1 - rank).any(-1).all())
                )
                self.assertTrue(bool((out.topk_ids[~is_image] == rank).any(-1).all()))

    def test_moe_takes_the_vision_top_k_only_for_batches_with_images(self):
        image = _Image(seed=23)
        moe = SimpleNamespace(
            gate=SimpleNamespace(e_score_correction_bias_vl=torch.zeros(4)),
            _enable_a2a_moe=False,
            _can_dual_stream_graph=lambda hidden: False,
            alt_stream=None,
            num_fused_shared_experts=0,
            forward_normal=lambda *args, **kwargs: kwargs["use_vision_topk"],
        )
        hidden = torch.zeros(3, 4)
        cases = (
            ("prefill with an image", ForwardMode.EXTEND, [image.mm_inputs], True),
            ("text prefill", ForwardMode.EXTEND, [None, None], False),
            ("decode", ForwardMode.DECODE, [None], False),
        )
        with (
            patch(
                "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
                return_value=False,
            ),
            patch.object(deepseek_v2, "_is_hip", False),
        ):
            for label, mode, mm_inputs, expected in cases:
                with self.subTest(case=label):
                    batch = _Batch(mode, mm_inputs, [0], [3])
                    self.assertIs(
                        deepseek_v2.DeepseekV2MoE.forward(moe, hidden, batch),
                        expected,
                    )
                with self.subTest(case=label, graph="breakable prefill"):
                    # A prefill graph replays batches its capture did not see.
                    with patch.object(
                        deepseek_v2, "is_in_breakable_cuda_graph", return_value=True
                    ):
                        self.assertTrue(
                            deepseek_v2.DeepseekV2MoE.forward(moe, hidden, batch)
                        )
            self.assertTrue(deepseek_v2.DeepseekV2MoE.forward(moe, hidden, None))
            text_only = SimpleNamespace(
                **{
                    **vars(moe),
                    "gate": SimpleNamespace(e_score_correction_bias_vl=None),
                }
            )
            batch = _Batch(ForwardMode.EXTEND, [image.mm_inputs], [0], [3])
            self.assertFalse(
                deepseek_v2.DeepseekV2MoE.forward(text_only, hidden, batch)
            )


if __name__ == "__main__":
    unittest.main()
