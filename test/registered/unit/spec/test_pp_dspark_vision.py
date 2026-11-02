"""Images and verify rounds: invariant 26. The model's real multimodal entry and MoE gate."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as dsv4
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE_ID, VOCAB, HIDDEN = 60, 64, 8


class _ToyTower(nn.Module):
    def forward(self, *a, **k):
        raise AssertionError("a verify round must not reach the vision tower")


class _Stage(nn.Module):
    prepare_language_model_inputs = dsv4.DeepseekV4ForCausalLM.prepare_language_model_inputs

    def __init__(self, tower):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_ID)
        self.vision_routing = True
        self.vision = tower
        self.embed = nn.Embedding(VOCAB, HIDDEN)

    def get_input_embeddings(self):
        return self.embed


class _Batch:
    def __init__(self, mode, mm_inputs):
        self.forward_mode = mode
        self.mm_inputs = mm_inputs
        self.mm_input_embeds = None

    def contains_image_inputs(self):
        return any(m is not None for m in self.mm_inputs)

    def contains_mm_inputs(self):
        return self.contains_image_inputs()

    def moe_num_token_non_padded(self):
        return None


class TestVerifyOnImageRequests(CustomTestCase):
    def test_no_remap_and_no_tower_on_any_stage(self):
        """A request with an image still carries its features while it decodes; a verify's
        ids are vocabulary ids, so they pass as they are."""
        ids = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int64)
        batch = _Batch(ForwardMode.TARGET_VERIFY, [object()])
        for stage in (_Stage(_ToyTower()), _Stage(None), _Stage(None)):
            got, embeds = stage.prepare_language_model_inputs(ids, batch)
            self.assertIs(got, ids)
            self.assertIsNone(embeds)

    def test_prefill_still_remaps(self):
        """The contrast: an extend's pad ids map to the image token on every stage."""
        ids = torch.tensor([5, MM_PAD_SHIFT_VALUE + 3], dtype=torch.int64)
        batch = _Batch(ForwardMode.EXTEND, [None])
        got, _ = _Stage(None).prepare_language_model_inputs(ids, batch)
        self.assertEqual(got.tolist(), [5, IMAGE_ID])

    def test_graph_verify_and_plain_rows_keep_the_fused_top_k(self):
        """The verify graph is captured on a batch with no image, so its replay keeps the
        fused top-k whatever the request carries; a plain round reads its own mode. The
        eager fallback of a verify would ask for the vision top-k (ForwardMode.is_extend
        counts TARGET_VERIFY), whose text rows take the text bias all the same."""
        moe = SimpleNamespace(
            gate=SimpleNamespace(e_score_correction_bias_vl=torch.zeros(4)),
            _enable_a2a_moe=False,
            _can_dual_stream_graph=lambda hidden: False,
            alt_stream=None,
            num_fused_shared_experts=0,
            forward_normal=lambda *args, **kwargs: kwargs["use_vision_topk"],
        )
        hidden = torch.zeros(6, 4)
        with (
            patch(
                "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
                return_value=False,
            ),
            patch.object(deepseek_v2, "_is_hip", False),
        ):
            for mode, mm_inputs in (
                (ForwardMode.TARGET_VERIFY, [None]),
                (ForwardMode.DECODE, [object()]),
            ):
                batch = _Batch(mode, mm_inputs)
                self.assertIs(
                    deepseek_v2.DeepseekV2MoE.forward(moe, hidden, batch), False, mode
                )


if __name__ == "__main__":
    unittest.main()
