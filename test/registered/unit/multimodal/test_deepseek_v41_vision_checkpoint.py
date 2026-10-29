"""DeepSeek-V4.1 images on CPU with the checkpoint's own vision weights: an
Anthropic base64 image block through the chat encoding and the image processor
to the first stage's image span, checked against an independent transcription
of the published reference preprocessing and tower (inference/image_processor.py,
inference/vision.py), plus the gate.bias_vl tensors every pipeline stage loads.

Set SGLANG_TEST_DSV41_CHECKPOINT to a V4.1 checkpoint directory to run it; only
the vision.*, aligner.*, image_* and a few gate tensors are read."""

import asyncio
import base64
import io
import json
import math
import os
import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors import safe_open
from torch import nn

from sglang.srt.configs.deepseek_v41 import (
    DeepseekV41Config,
    normalize_deepseek_v41_config,
)
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.chat_encoding_adapters import ChatOutputHooks
from sglang.srt.entrypoints.openai.serving_deepseek_recipe import (
    DeepSeekRecipeChatEncodingAdapter,
)
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v41_vit import Aligner, ViT
from sglang.srt.multimodal.processors.deepseek_v41 import DeepseekV41ImageProcessor
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(
    est_time=60, suite="base-a-test-cpu", disabled="needs a local checkpoint"
)

CHECKPOINT = os.environ.get("SGLANG_TEST_DSV41_CHECKPOINT", "")
# One gate per stage of the 13/13/14 split: layers 0-12, 13-25 and 26-39.
STAGE_LAYERS = (0, 13, 39)


# ---- reference, transcribed from the published inference/vision.py ----------------
def _ref_cos_sin(n_h, n_w, dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _ref_rotary(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class _RefNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        return (self.weight * h).to(x.dtype)


class _RefBlock(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.heads, self.head_dim = a.vision_n_heads, a.vision_dim // a.vision_n_heads
        self.norm1, self.norm2 = _RefNorm(a.vision_dim), _RefNorm(a.vision_dim)
        self.attn = nn.Module()
        self.attn.wqkv = nn.Linear(a.vision_dim, 3 * a.vision_dim)
        self.attn.wo = nn.Linear(a.vision_dim, a.vision_dim)
        self.mlp = nn.Module()
        self.mlp.w1 = nn.Linear(a.vision_dim, 2 * a.vision_inter_dim, bias=False)
        self.mlp.w2 = nn.Linear(a.vision_inter_dim, a.vision_dim, bias=False)

    def forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (
            t.view(n, self.heads, self.head_dim)
            for t in self.attn.wqkv(self.norm1(x)).chunk(3, dim=-1)
        )
        q, k = _ref_rotary(q, cos, sin), _ref_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        x = x + self.attn.wo(o.transpose(0, 1).reshape(n, -1))
        gate, up = self.mlp.w1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.mlp.w2(F.silu(gate) * up)


class _RefTower(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.vision = nn.Module()
        self.vision.patch_embed = nn.Module()
        self.vision.patch_embed.proj = nn.Linear(
            3 * a.vision_patch_size**2, a.vision_dim
        )
        self.vision.blocks = nn.ModuleList(
            _RefBlock(a) for _ in range(a.vision_n_layers)
        )
        self.vision.norm = _RefNorm(a.vision_dim)
        r = a.vision_downsample_ratio
        self.aligner = nn.Module()
        self.aligner.w1 = nn.Linear(a.vision_dim * r * r, a.dim)
        self.aligner.w2 = nn.Linear(a.dim, a.dim)
        for name in ("image_start", "image_end", "image_newline"):
            setattr(self, name, nn.Parameter(torch.empty(a.dim)))

    def encode(self, patches, n_h, n_w):
        a, r = self.a, self.a.vision_downsample_ratio
        x = self.vision.patch_embed.proj(patches.flatten(1))
        cos, sin = _ref_cos_sin(
            n_h, n_w, a.vision_dim // a.vision_n_heads // 2, a.vision_rope_theta
        )
        for block in self.vision.blocks:
            x = block(x, cos, sin)
        x = self.vision.norm(x)
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.aligner.w2(F.gelu(self.aligner.w1(x)))

    def span(self, patches, n_h, n_w, n_llm_h, n_llm_w):
        """[START] + ([IMAGE] * n_llm_w + [NEW_LINE]) * n_llm_h + [END]"""
        features = self.encode(patches, n_h, n_w)
        rows, k = [self.image_start], 0
        for _ in range(n_llm_h):
            rows += list(features[k : k + n_llm_w]) + [self.image_newline]
            k += n_llm_w
        rows.append(self.image_end)
        return torch.stack([row.to(features.dtype) for row in rows])


# ---- reference, transcribed from the published inference/image_processor.py ---------
def _ref_grid(width, height, a):
    p, r, cap = a.vision_patch_size, a.vision_downsample_ratio, a.vision_max_n_token
    if a.vision_max_wh_ratio is not None and width > height * a.vision_max_wh_ratio:
        width = height * a.vision_max_wh_ratio
    if 0 < width * height < a.vision_min_pixels:
        ratio = (a.vision_min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p

    def llm(h, w):
        return math.ceil((h // p) / r), math.ceil((w // p) / r)

    lh, lw = llm(best_h, best_w)
    if lh * (lw + 1) + 2 > cap:
        ratio_hw = height / width
        max_w = math.sqrt((cap - 2) / ratio_hw + 0.25) - 0.5
        max_h = max_w * ratio_hw
        cell = p * r
        if max_w < 1.0:
            best_h, best_w = (cap - 2) // 2 * cell, cell
        elif max_h < 1.0:
            best_h, best_w = cell, (cap - 3) * cell
        else:
            beta = min(
                math.floor(max_w) * cell / width, math.floor(max_h) * cell / height
            )
            best_h = math.floor(height * beta / p) * p
            best_w = math.floor(width * beta / p) * p
        lh, lw = llm(best_h, best_w)
    return lh, lw, best_h, best_w


def _ref_patches(data, a):
    p = a.vision_patch_size
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    lh, lw, best_h, best_w = _ref_grid(image.width, image.height, a)
    n_h, n_w = best_h // p, best_w // p
    if (
        a.vision_max_wh_ratio is not None
        and image.width >= a.vision_max_wh_ratio * image.height
    ):
        image = image.resize((best_w, best_h))
    else:
        image = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4).reshape(n_h * n_w, 3, p, p)
    )
    return patches, n_h, n_w, lh, lw


# ---- the first pipeline stage, reduced to what embeds an image ----------------------
class _FirstStage(nn.Module):
    get_image_feature = DeepseekV4ForCausalLM.get_image_feature
    _prepare_mm_embeddings = DeepseekV4ForCausalLM._prepare_mm_embeddings
    prepare_language_model_inputs = DeepseekV4ForCausalLM.prepare_language_model_inputs

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_routing = True
        self.embed = _SmallEmbedding(config.vocab_size, config.hidden_size)

    def get_input_embeddings(self):
        return self.embed


class _SmallEmbedding(nn.Module):
    """A vocabulary-sized lookup backed by a small table (rows repeat every 997 ids)."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.num_embeddings = vocab_size
        generator = torch.Generator().manual_seed(0)
        self.table = torch.randn(997, dim, generator=generator).to(torch.bfloat16)

    def forward(self, ids):
        return self.table[ids % 997]


def _png(width, height, fmt="PNG"):
    rng = np.random.default_rng(width * 1000 + height)
    pixels = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    pixels[height // 4 : height // 2, width // 4 : width // 2] = (220, 20, 20)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=fmt, quality=95)
    return buffer.getvalue()


@unittest.skipUnless(
    os.path.isfile(os.path.join(CHECKPOINT, "model.safetensors.index.json")),
    "set SGLANG_TEST_DSV41_CHECKPOINT to a DeepSeek-V4.1 checkpoint directory",
)
class TestDeepseekV41VisionCheckpoint(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.set_num_threads(min(16, os.cpu_count() or 1))
        reset_context()
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                mm_attention_backend="sdpa",
                mm_process_config={},
            ),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        mm_schedule.init_mm_embedding_cache(256 << 20)
        with open(os.path.join(CHECKPOINT, "config.json")) as f:
            raw_config = json.load(f)
        cls.hf_config = DeepseekV41Config(**raw_config)
        values = normalize_deepseek_v41_config(raw_config)
        cls.args = SimpleNamespace(**values, dim=values["hidden_size"])
        cls.tokenizer = get_tokenizer(CHECKPOINT)
        with open(os.path.join(CHECKPOINT, "model.safetensors.index.json")) as f:
            cls.weight_map = json.load(f)["weight_map"]
        wanted = [
            k for k in cls.weight_map if k.startswith(("vision.", "aligner.", "image_"))
        ]
        cls.raw = cls._read(wanted)

        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            cls.stage = _FirstStage(cls.hf_config)
            cls.stage.vision = ViT(cls.args)
            cls.stage.aligner = Aligner(cls.args)
            for name in ("image_start", "image_end", "image_newline"):
                setattr(cls.stage, name, nn.Parameter(torch.empty(cls.args.dim)))
            cls.reference = _RefTower(cls.args)
        finally:
            torch.set_default_dtype(previous)
        params = dict(cls.stage.named_parameters())
        cls.unloaded = set(params)
        for name, tensor in cls.raw.items():
            target = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(name)
            param = params[target]
            getattr(param, "weight_loader", default_weight_loader)(param, tensor)
            cls.unloaded.discard(target)
        missing, unexpected = cls.reference.load_state_dict(cls.raw, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        hooks = ChatOutputHooks(
            thinking_requested=lambda request: False,
            tool_call_parsing_active=lambda request: False,
            history_tool_calls_count=lambda request: 0,
            tool_call_id=lambda call, count: "call",
        )
        cls.encoder = DeepSeekRecipeChatEncodingAdapter(
            chat_encoding_spec="dsv41",
            tokenizer=cls.tokenizer,
            reasoning_parser="deepseek-v41",
            tool_call_parser="deepseekv41",
            chat_hooks=hooks,
            default_chat_template_kwargs=None,
        )
        cls.processor = DeepseekV41ImageProcessor(
            cls.hf_config,
            ServerArgs(model_path="dummy", device="cpu"),
            cls.tokenizer,
            "default",
        )

    @classmethod
    def tearDownClass(cls):
        reset_context()
        super().tearDownClass()

    @classmethod
    def _read(cls, names):
        out = {}
        for shard in sorted({cls.weight_map[k] for k in names}):
            with safe_open(os.path.join(CHECKPOINT, shard), framework="pt") as f:
                for name in names:
                    if cls.weight_map[name] == shard:
                        out[name] = f.get_tensor(name)
        return out

    def _request(self, data, media_type):
        body = {
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the picture."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(data).decode(),
                            },
                        },
                    ],
                }
            ],
        }
        serving = SimpleNamespace(
            openai_serving_chat=SimpleNamespace(
                supports_native_reasoning_history=lambda: False,
                wrap_reasoning_history=lambda *args, **kwargs: None,
                apply_reasoning_enabled=lambda request, enabled: None,
            ),
            _merge_inline_system=False,
        )
        chat = AnthropicServing._convert_to_chat_completion_request(
            serving, AnthropicMessagesRequest(**body)
        )
        image_data = []
        prompt = self.encoder.encode_messages(
            messages=[m.model_dump() for m in chat.messages],
            request=chat,
            tools=None,
            thinking=False,
            image_data=image_data,
        )
        return prompt, image_data

    def test_the_checkpoint_fills_every_tower_parameter(self):
        self.assertEqual(len(self.raw), 266)
        self.assertEqual(self.unloaded, set())
        self.assertEqual(
            self.tokenizer.convert_ids_to_tokens(self.hf_config.image_token_id),
            "<｜deepseek_image｜>",
        )

    def test_an_anthropic_image_block_reaches_the_first_stage_span(self):
        image_id = self.hf_config.image_token_id
        for fmt, media_type, size in (
            ("PNG", "image/png", (64, 48)),
            ("JPEG", "image/jpeg", (300, 120)),
        ):
            with self.subTest(media_type=media_type, size=size):
                data = _png(*size, fmt=fmt)
                prompt, image_data = self._request(data, media_type)
                self.assertEqual(prompt.count(image_id), 1)
                self.assertEqual(len(image_data), 1)
                self.assertTrue(
                    image_data[0].url.startswith(f"data:{media_type};base64,")
                )
                out = asyncio.run(
                    self.processor.process_mm_data_async(image_data, prompt, None)
                )
                (item,) = out.mm_items
                patches, n_h, n_w, lh, lw = _ref_patches(data, self.args)
                count = lh * (lw + 1) + 2
                start = prompt.index(image_id)
                self.assertEqual(out.input_ids.count(image_id), count)
                self.assertEqual(len(out.input_ids), len(prompt) - 1 + count)
                self.assertEqual(item.offsets, [(start, start + count - 1)])
                self.assertEqual((item.n_vit_h, item.n_vit_w), (n_h, n_w))
                self.assertTrue(torch.equal(item.feature, patches))
                with torch.no_grad():
                    (span,) = self.stage.get_image_feature([item])
                    want = self.reference.span(patches, n_h, n_w, lh, lw)
                self.assertEqual(tuple(span.shape), (count, self.args.dim))
                cosine = F.cosine_similarity(span.float(), want.float(), dim=-1)
                rel = (span.float() - want.float()).norm() / want.float().norm()
                print(
                    f"{media_type} {size}: {n_h}x{n_w} patches, {count} tokens, "
                    f"span vs reference: min cosine {float(cosine.min()):.6f}, "
                    f"relative error {float(rel):.5f}"
                )
                self.assertGreater(float(cosine.min()), 0.999, msg=float(cosine.min()))
                self.assertLess(float(rel), 0.02, msg=float(rel))
                # Delimiter rows are the checkpoint's own vectors, exactly.
                self.assertTrue(torch.equal(span[0], self.raw["image_start"]))
                self.assertTrue(torch.equal(span[-1], self.raw["image_end"]))
                self.assertTrue(torch.equal(span[lw + 1], self.raw["image_newline"]))

    def test_the_merge_takes_the_span_into_the_first_stage_embeddings(self):
        image_id = self.hf_config.image_token_id
        data = _png(64, 48)
        prompt, image_data = self._request(data, "image/png")
        out = asyncio.run(
            self.processor.process_mm_data_async(image_data, prompt, None)
        )
        (item,) = out.mm_items
        item.set_hash(77)
        mm_inputs = MultimodalInputs(mm_items=[item], im_token_id=image_id)
        scheduler_ids = DeepseekV4ForCausalLM.pad_input_ids(
            None, array("q", out.input_ids), mm_inputs
        )
        ids = torch.tensor(list(scheduler_ids))
        stage = self.stage
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mm_inputs=[mm_inputs],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[len(ids)],
            mm_input_embeds=None,
        )
        with torch.no_grad():
            mapped, embeds = stage.prepare_language_model_inputs(ids.clone(), batch)
            (span,) = stage.get_image_feature([item])
        offset = item.offsets[0]
        rows = slice(offset[0], offset[1] + 1)
        self.assertTrue(torch.equal(embeds[rows], span))
        text = torch.ones(len(ids), dtype=torch.bool)
        text[rows] = False
        self.assertTrue(torch.equal(embeds[text], stage.embed(mapped[text])))
        self.assertTrue(bool((mapped[rows] == image_id).all()))
        self.assertFalse(bool((mapped[text] == image_id).any()))

    def test_every_stage_finds_its_bias_vl(self):
        names = []
        for layer in STAGE_LAYERS:
            names += [
                f"layers.{layer}.ffn.gate.bias",
                f"layers.{layer}.ffn.gate.bias_vl",
            ]
        tensors = self._read(names)
        experts = self.args.n_routed_experts
        torch.manual_seed(0)
        logits = torch.randn(4096, experts)
        scores = F.softplus(logits).sqrt()
        for layer in STAGE_LAYERS:
            with self.subTest(layer=layer):
                bias = tensors[f"layers.{layer}.ffn.gate.bias"]
                bias_vl = tensors[f"layers.{layer}.ffn.gate.bias_vl"]
                self.assertEqual(bias_vl.dtype, torch.float32)
                self.assertEqual(tuple(bias_vl.shape), (experts,))
                self.assertTrue(bool(torch.isfinite(bias_vl).all()))
                self.assertEqual(
                    DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
                        f"layers.{layer}.ffn.gate.bias_vl"
                    ),
                    f"model.layers.{layer}.mlp.gate.e_score_correction_bias_vl",
                )
                k = self.args.num_experts_per_tok
                text = (scores + bias).topk(k, dim=-1).indices.sort(-1).values
                image = (scores + bias_vl).topk(k, dim=-1).indices.sort(-1).values
                changed = float((text != image).any(-1).float().mean())
                self.assertGreater(changed, 0.0)
                print(
                    f"layer {layer}: bias_vl changes the top-{k} of {changed:.1%} of rows"
                )


if __name__ == "__main__":
    unittest.main()
