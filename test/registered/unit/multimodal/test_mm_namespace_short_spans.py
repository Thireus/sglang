"""SGLANG_MM_CACHE_NAMESPACE_SHORT_SPANS_ONLY: keep the request's own cache_salt
when every media span in the prompt ids is its item's full sentinel sequence.

The decision tests and the radix tests run on CPU with synthetic images laid out
by the DeepSeek-V4.1 image processing code. Set SGLANG_TEST_DSV41_CHECKPOINT to a
V4.1 checkpoint directory (config.json and the tokenizer are read, no weights) to
also run multi-turn Anthropic conversations through the real chat encoding and
image processor."""

import asyncio
import base64
import io
import json
import os
import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import pad_mm_tokens
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.multimodal.cache.digest import structured_digest
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    image_token_types,
    patchify_image,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import slot_of, tree_slots_of

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CHECKPOINT = os.environ.get("SGLANG_TEST_DSV41_CHECKPOINT", "")
IMAGE_TOKEN_ID = 129264
FINGERPRINT = "sha256:" + "22" * 32


class _Vision:
    """The V4.1 checkpoint's vision geometry."""

    vision_patch_size = 14
    vision_downsample_ratio = 3
    vision_max_n_token = 1024
    vision_min_pixels = 295936
    vision_max_wh_ratio = None


def _noise(seed, width, height):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


def _image_item(seed, start, width=320, height=240):
    patches, n_h, n_w, lh, lw = patchify_image(_noise(seed, width, height), _Vision)
    count = len(image_token_types(lh, lw))
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=patches,
        offsets=[(start, start + count - 1)],
        model_specific_data={"n_vit_h": n_h, "n_vit_w": n_w},
    )
    item.bind_processor_fingerprint(FINGERPRINT)
    item.set_pad_value()
    return item, count


def _id_item(tag, offsets, modality=Modality.IMAGE):
    item = MultimodalDataItem(modality=modality, feature=None, offsets=offsets)
    item.set_identity(structured_digest(("test-item", tag)))
    return item


def _text(seed, n):
    return [int(x) for x in np.random.default_rng(seed).integers(1, 120_000, n)]


def _padded(ids, items):
    mm = MultimodalInputs(mm_items=items, im_token_id=IMAGE_TOKEN_ID)
    return list(pad_mm_tokens(array("q", ids), mm))


def _req(ids, items, cache_salt=None, rid="r"):
    req = Req(
        rid=rid,
        origin_input_text=None,
        origin_input_ids=array("q", ids),
        sampling_params=SamplingParams(),
        cache_salt=cache_salt,
    )
    if items:
        req.extend_image_inputs(MultimodalInputs(mm_items=items))
    return req


def _upstream_salt(user_salt, items):
    identities = tuple(item.cache_key for item in items)
    return "sglang-mm-v1:" + structured_digest((user_salt, identities)).hex()


def _tree(page_size):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=page_size,
        tree_slots=tree_slots_of((ComponentType.FULL,)),
    )
    cache = SimpleNamespace(enable_session_radix_cache=False)
    component = FullComponent(cache, params, slot_of(params, ComponentType.FULL))
    return UnifiedTreeCore(params, {ComponentType.FULL: component})


def _cache(core, req, page_size):
    key = RadixKey(array("q", req.origin_input_ids), cache_salt=req.cache_salt)
    cached = len(key) // page_size * page_size
    step = core.begin_insert(InsertParams(key=key, value=torch.arange(cached)))
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    return cached


def _match(core, req):
    key = RadixKey(array("q", req.origin_input_ids), cache_salt=req.cache_salt)
    return core.match_prefix(MatchPrefixParams(key=key)).device_prefix_len


def _lcp(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


class _Switch:
    def _on(self):
        return envs.SGLANG_MM_CACHE_NAMESPACE_SHORT_SPANS_ONLY.override(True)

    def _off(self):
        return envs.SGLANG_MM_CACHE_NAMESPACE_SHORT_SPANS_ONLY.override(False)


class TestNamespaceDecision(_Switch, CustomTestCase):
    def _salt(self, ids, items, cache_salt=None):
        return _req(ids, items, cache_salt=cache_salt).cache_salt

    def test_switch_off_keeps_upstream_namespace(self):
        item, count = _image_item(1, 10)
        ids = _padded(_text(1, 10) + [IMAGE_TOKEN_ID] * count + _text(2, 5), [item])
        with self._off():
            for user_salt in (None, "tenant"):
                self.assertEqual(
                    self._salt(ids, [item], user_salt),
                    _upstream_salt(user_salt, [item]),
                )

    def test_long_spans_keep_the_user_salt(self):
        a, count_a = _image_item(1, 10)
        b, count_b = _image_item(2, 10 + count_a + 7, 1280, 800)
        raw = (
            _text(1, 10)
            + [IMAGE_TOKEN_ID] * count_a
            + _text(2, 7)
            + [IMAGE_TOKEN_ID] * count_b
            + _text(3, 5)
        )
        ids = _padded(raw, [a, b])
        with self._on():
            self.assertIsNone(self._salt(ids, [a, b]))
            self.assertEqual(self._salt(ids, [a, b], "tenant"), "tenant")

    def test_spans_shorter_than_the_sentinels_keep_the_namespace(self):
        for width in (1, 2, 3):
            with self.subTest(width=width):
                item = _id_item("short", [(4, 4 + width - 1)])
                ids = [1] * 4 + item.padding_sequence(width) + [2] * 4
                with self._on():
                    self.assertEqual(
                        self._salt(ids, [item]), _upstream_salt(None, [item])
                    )
        item = _id_item("four", [(4, 7)])
        ids = [1] * 4 + item.padding_sequence(4) + [2] * 4
        with self._on():
            self.assertIsNone(self._salt(ids, [item]))

    def test_spans_cut_by_the_prompt_end(self):
        # The offsets still describe the whole span; only the prompt is shorter.
        for left, namespaced in ((2, True), (3, True), (4, False), (40, False)):
            with self.subTest(left=left):
                item = _id_item("cut", [(8, 8 + 99)])
                ids = _padded([1] * 8 + [IMAGE_TOKEN_ID] * left, [item])
                with self._on():
                    salt = self._salt(ids, [item], "tenant")
                expected = _upstream_salt("tenant", [item]) if namespaced else "tenant"
                self.assertEqual(salt, expected)

    def test_ids_that_do_not_carry_the_identity_keep_the_namespace(self):
        item = _id_item("raw", [(4, 23)])
        cases = {
            "raw placeholder ids": [1] * 4 + [IMAGE_TOKEN_ID] * 20,
            "scalar padding": [1] * 4 + [item.pad_value] * 20,
            "shifted sentinels": [1] * 4 + item.padding_sequence(21)[1:],
            "stale offsets": [1] * 6 + item.padding_sequence(18),
        }
        for name, ids in cases.items():
            with self.subTest(case=name), self._on():
                self.assertEqual(self._salt(ids, [item]), _upstream_salt(None, [item]))

    def test_items_without_offsets_keep_the_namespace(self):
        item = _id_item("no-offsets", None)
        with self._on():
            self.assertEqual(
                self._salt([1, 2, 3], [item]), _upstream_salt(None, [item])
            )

    def test_mixed_media(self):
        image = _id_item("image", [(2, 11)])
        audio = _id_item("audio", [(14, 29)], Modality.AUDIO)
        ids = (
            [1] * 2
            + image.padding_sequence(10)
            + [2] * 2
            + audio.padding_sequence(16)
            + [3]
        )
        with self._on():
            self.assertEqual(self._salt(ids, [image, audio], "tenant"), "tenant")
            raw_audio = ids[:14] + [IMAGE_TOKEN_ID] * 16 + [3]
            self.assertEqual(
                self._salt(raw_audio, [image, audio], "tenant"),
                _upstream_salt("tenant", [image, audio]),
            )

    def test_text_only_requests_keep_the_client_salt(self):
        for switch in (self._on, self._off):
            with self.subTest(switch=switch.__name__), switch():
                self.assertEqual(self._salt([1, 2, 3], [], "tenant"), "tenant")
                self.assertIsNone(self._salt([1, 2, 3], []))

    def test_rebinding_follows_the_current_items(self):
        # The session path binds again with inherited items.
        first = _id_item("first", [(2, 9)])
        ids = [1] * 2 + first.padding_sequence(8) + [2] * 4
        with self._on():
            req = _req(ids, [first], "tenant")
            self.assertEqual(req.cache_salt, "tenant")
            short = _id_item("short", [(12, 13)])
            req.origin_input_ids = array("q", ids + short.padding_sequence(2))
            req.extend_image_inputs(MultimodalInputs(mm_items=[short]))
            self.assertEqual(req.cache_salt, _upstream_salt("tenant", [first, short]))


class TestRadixMatch(_Switch, CustomTestCase):
    """Real tree core, V4.1 span layout, page sizes 1 and 256."""

    def _conversation(self):
        text_a, text_b, text_c = _text(11, 300), _text(12, 120), _text(13, 40)
        a, count_a = _image_item(21, len(text_a))
        first_raw = text_a + [IMAGE_TOKEN_ID] * count_a + text_b
        answer = _text(14, 80)
        start_b = len(first_raw) + len(answer) + len(text_c)
        b, count_b = _image_item(22, start_b, 1280, 800)
        second_raw = first_raw + answer + text_c + [IMAGE_TOKEN_ID] * count_b + [7]
        first = _padded(first_raw, [a])
        second = _padded(second_raw, [a, b])
        return first, a, second, b, start_b

    def test_a_request_that_adds_an_image_keeps_the_previous_prefix(self):
        first, a, second, b, _ = self._conversation()
        for page_size in (1, 256):
            for on, expected in ((False, 0), (True, len(first))):
                with self.subTest(page=page_size, on=on):
                    with self._on() if on else self._off():
                        core = _tree(page_size)
                        cached = _cache(core, _req(first, [a], rid="1"), page_size)
                        hit = _match(core, _req(second, [a, b], rid="2"))
                    self.assertEqual(
                        hit, min(expected, cached) // page_size * page_size
                    )

    def test_a_different_image_stops_at_its_span_start(self):
        text, tail = _text(31, 1000), _text(32, 60)
        a, count = _image_item(41, len(text))
        b, count_b = _image_item(42, len(text))
        self.assertEqual(count, count_b)
        raw = text + [IMAGE_TOKEN_ID] * count + tail
        first, other = _padded(raw, [a]), _padded(raw, [b])
        self.assertEqual(_lcp(first, other), len(text))
        for page_size in (1, 256):
            with self.subTest(page=page_size), self._on():
                core = _tree(page_size)
                _cache(core, _req(first, [a], rid="1"), page_size)
                hit = _match(core, _req(other, [b], rid="2"))
                self.assertLessEqual(hit, len(text))
                self.assertEqual(hit, len(text) // page_size * page_size)

    def test_the_same_image_matches_through_its_span(self):
        text, tail, more = _text(51, 1000), _text(52, 60), _text(53, 300)
        a, count = _image_item(61, len(text))
        raw = text + [IMAGE_TOKEN_ID] * count + tail
        first = _padded(raw, [a])
        again = _padded(raw + more, [a])
        for page_size in (1, 256):
            for switch in (self._on, self._off):
                with self.subTest(page=page_size, switch=switch.__name__), switch():
                    core = _tree(page_size)
                    cached = _cache(core, _req(first, [a], rid="1"), page_size)
                    hit = _match(core, _req(again, [a], rid="2"))
                    self.assertEqual(hit, cached)
                    self.assertGreater(hit, len(text) + count - page_size)


def _png(seed, width, height):
    buffer = io.BytesIO()
    _noise(seed, width, height).save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(CHECKPOINT, "SGLANG_TEST_DSV41_CHECKPOINT is not set")
class TestDeepseekV41Conversations(_Switch, CustomTestCase):
    """Anthropic multi-turn image chats through the V4.1 encoding and processor."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
        from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
            ChatOutputHooks,
        )
        from sglang.srt.entrypoints.openai.serving_deepseek_recipe import (
            DeepSeekRecipeChatEncodingAdapter,
        )
        from sglang.srt.hf_transformers.tokenizer import get_tokenizer
        from sglang.srt.multimodal.processors.deepseek_v41 import (
            DeepseekV41ImageProcessor,
        )
        from sglang.srt.runtime_context import SpawnRanks, reset_context
        from sglang.srt.server_args import ServerArgs
        from sglang.test.parallel_groups import publish

        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu", mm_process_config={}),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        with open(os.path.join(CHECKPOINT, "config.json")) as f:
            cls.hf_config = DeepseekV41Config(**json.load(f))
        cls.tokenizer = get_tokenizer(CHECKPOINT)
        hooks = ChatOutputHooks(
            thinking_requested=lambda request: False,
            tool_call_parsing_active=lambda request: False,
            history_tool_calls_count=lambda request: 0,
            tool_call_id=lambda call, count: "call",
            effective_tools=lambda request: [],
            split_assistant_prefix=lambda messages, request: (messages, None),
            append_assistant_prefix=lambda ids, prefix: ids,
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
        from sglang.srt.runtime_context import reset_context

        reset_context()
        super().tearDownClass()

    def _prompt(self, messages):
        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
        from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        body = {"model": "m", "max_tokens": 16, "messages": messages}
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
        out = asyncio.run(
            self.processor.process_mm_data_async(image_data, prompt, None)
        )
        self.processor.finalize_cache_identities(out)
        mm = MultimodalInputs.from_processor_output(out)
        ids = DeepseekV4ForCausalLM.pad_input_ids(None, array("q", out.input_ids), mm)
        return list(ids), list(mm.mm_items)

    @staticmethod
    def _image(seed, width, height):
        data = base64.b64encode(_png(seed, width, height)).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        }

    def _turns(self, second_image_seed):
        text = "Here is the build log. " * 200
        turn1 = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    self._image(1, 320, 240),
                    {"type": "text", "text": "What does the first screenshot show?"},
                ],
            }
        ]
        turn2 = turn1 + [
            {"role": "assistant", "content": "A grey test pattern. " * 20},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And this one?"},
                    self._image(second_image_seed, 640, 480),
                ],
            },
        ]
        return self._prompt(turn1), self._prompt(turn2)

    def test_adding_an_image_matches_through_the_previous_prompt(self):
        (first, items1), (second, items2) = self._turns(2)
        self.assertEqual(len(items1), 1)
        self.assertEqual(len(items2), 2)
        lcp = _lcp(first, second)
        self.assertGreater(lcp, len(first) - 8)
        for page_size in (1, 256):
            for on in (False, True):
                with self.subTest(page=page_size, on=on):
                    with self._on() if on else self._off():
                        core = _tree(page_size)
                        cached = _cache(core, _req(first, items1, rid="1"), page_size)
                        hit = _match(core, _req(second, items2, rid="2"))
                    expected = min(lcp, cached) // page_size * page_size if on else 0
                    self.assertEqual(hit, expected)

    def test_a_different_image_at_the_same_position(self):
        (_, _), (second, items2) = self._turns(2)
        (_, _), (other, items3) = self._turns(3)
        span_start = items2[1].offsets[0][0]
        self.assertEqual(items3[1].offsets[0][0], span_start)
        self.assertEqual(_lcp(second, other), span_start)
        for page_size in (1, 256):
            with self.subTest(page=page_size), self._on():
                core = _tree(page_size)
                _cache(core, _req(second, items2, rid="1"), page_size)
                hit = _match(core, _req(other, items3, rid="2"))
                self.assertEqual(hit, span_start // page_size * page_size)


if __name__ == "__main__":
    unittest.main()
