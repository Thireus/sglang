# SPDX-License-Identifier: Apache-2.0
"""Server-argument checks for DeepSeek-V4 hot/cold checkpoints.

Active only when the checkpoint's config.json carries ``"hotcold"``. The two
tiers are sized per layer from the manifest with one static shape each, so the
expert placement must stay trivial (expert_pack's rule): no TP/DP/attention-DP/EP,
no all-to-all backend, no redundant experts or EPLB, no BF16 dequant of the FP4
experts, and the hot tier on the FlashInfer MXFP4 runner. Pipeline parallelism
is allowed (moe_tp_size stays 1). Every error is reported together.
"""

from __future__ import annotations

import json
import os
from typing import Any

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.environ import envs


def _checkpoint_hotcold(model_path: Any):
    if not isinstance(model_path, str) or not os.path.isdir(model_path):
        return None
    path = os.path.join(model_path, "config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("hotcold")


def handle_hotcold(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    if _checkpoint_hotcold(cfg.model_path) is None:
        return
    errors = []
    for label, option, size in (
        ("tensor", "--tp-size", cfg.tp_size),
        ("data", "--dp-size", cfg.dp_size),
        ("attention data", "--attn-dp-size", cfg.attn_dp_size),
        ("expert", "--ep-size", cfg.ep_size),
    ):
        if size != 1:
            errors.append(f"{label} parallelism ({option}) must be 1, got {size}")
    if (cfg.moe_a2a_backend or "none") != "none":
        errors.append(f"--moe-a2a-backend must be none, got {cfg.moe_a2a_backend!r}")
    if (cfg.ep_num_redundant_experts or 0) > 0:
        errors.append(
            f"--ep-num-redundant-experts must be 0, got {cfg.ep_num_redundant_experts}"
        )
    if cfg.enable_eplb:
        errors.append("--enable-eplb is incompatible with a hot/cold checkpoint")
    if envs.SGLANG_DSV4_FP4_DEQUANT.get():
        errors.append(
            "SGLANG_DSV4_FP4_DEQUANT=1 is incompatible with a hot/cold checkpoint"
        )
    if cfg.moe_runner_backend != "flashinfer_mxfp4":
        errors.append(
            "the hot tier needs --moe-runner-backend flashinfer_mxfp4, got "
            f"{cfg.moe_runner_backend!r}"
        )
    if errors:
        details = "\n".join(f"- {e}" for e in errors)
        raise ValueError(f"Invalid hot/cold configuration:\n{details}")
