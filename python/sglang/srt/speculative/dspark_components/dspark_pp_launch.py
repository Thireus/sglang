"""The launch DSpark supports under pipeline parallelism, and nothing else.

Under ``SGLANG_ENABLE_PP_SPEC`` the last stage drafts for the next round and the
chain rides the output ring to every stage (scheduler_pp_dspark.py). That flow is
written for DeepSeek-V4.1, whose draft reads layers that sit on the last stage,
with static verify on the non-overlap schedule. Everything else keeps its refusal,
and this module names the one that stopped a launch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence

from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Graph folds written for the single-stage step: each moves work into a graph the
# relayed verify does not have (the head of the draft graph, the verify epilogue).
_SINGLE_STAGE_FOLDS = (
    "SGLANG_OPT_DSV41_TARGET_METADATA_EARLY",
    "SGLANG_OPT_DSV41_DSPARK_POST_FOLD",
    "SGLANG_DSPARK_DEFER_CONTEXT_KV",
    "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE",
    "SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH",
)


def pp_dspark_relay_enabled(*, pp_size: int, spec_algorithm: Optional[str]) -> bool:
    """The relayed DSpark flow, as a launch sees it; False on every other launch."""
    return (
        envs.SGLANG_ENABLE_PP_SPEC.get()
        and pp_size > 1
        and spec_algorithm is not None
        and str(spec_algorithm).upper() == "DSPARK"
    )


def target_layers_off_last_stage(
    target_layer_ids: Sequence[int], num_layers: int, pp_size: int
) -> list[int]:
    """Target layers the draft reads that the last stage does not hold."""
    start, end = get_pp_indices(num_layers, pp_size - 1, pp_size)
    return [i for i in target_layer_ids if not start <= i < end]


def pp_dspark_refusal(cfg: Any, hf_config: Any) -> Optional[str]:
    """Why this launch cannot run the relayed DSpark flow, or None."""
    if getattr(hf_config, "model_type", None) != "deepseek_v41":
        return f"it serves DeepSeek-V4.1 only, not {hf_config.model_type!r}"
    if cfg.attn_dp_size > 1 or cfg.attn_cp_size > 1:
        return "every stage must see the same batch, so no attention DP or CP"
    if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
        return "it needs SGLANG_RAGGED_VERIFY_MODE=static"
    if not cfg.disable_overlap_schedule:
        return "it needs --disable-overlap-schedule"
    if cfg.speculative_adaptive:
        return "--speculative-adaptive changes the block size at runtime"
    if cfg.enable_mixed_chunk:
        return "--enable-mixed-chunk mixes decode rows into a prefill forward"
    if cfg.enable_watermark:
        return "--enable-watermark is not wired through the relay"
    if cfg.enable_lora or (cfg.enable_lora is None and cfg.lora_paths):
        return "LoRA adds graph variants the plain and verify families do not carry"
    text_config = getattr(hf_config, "text_config", hf_config)
    target_layer_ids = getattr(text_config, "dspark_target_layer_ids", None)
    if not target_layer_ids:
        return "the checkpoint names no dspark_target_layer_ids"
    off_stage = target_layers_off_last_stage(
        target_layer_ids, int(text_config.num_hidden_layers), int(cfg.pp_size)
    )
    if off_stage:
        return (
            f"the draft reads target layers {off_stage}, which are not on the last "
            "stage; move the split so that stage starts at or before them"
        )
    if not envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get():
        # The target's table is not on the last stage; the draft embeds on its own.
        return "it needs SGLANG_DSPARK_EMBED_IN_GRAPH on"
    embed = envs.SGLANG_PP_DSPARK_EMBED.get()
    if embed not in ("host", "gpu"):
        return f"SGLANG_PP_DSPARK_EMBED must be host or gpu, got {embed!r}"
    switch = envs.SGLANG_PP_DSPARK_SWITCH.get()
    if switch not in ("auto", "on", "off"):
        return f"SGLANG_PP_DSPARK_SWITCH must be auto, on or off, got {switch!r}"
    return None


def resolve_single_stage_folds_off() -> list[str]:
    """Turn the single-stage folds off for this process; returns those it changed."""
    changed = []
    for name in _SINGLE_STAGE_FOLDS:
        field = getattr(envs, name)
        if field.get():
            field.set(False)
            changed.append(name)
    return changed


def validate_pp_dspark_launch(server_args: ServerArgs) -> None:
    """Refuse an unsupported relayed launch by name, and resolve what it cannot use."""
    from sglang.srt.arg_groups.model_override_base import (
        model_config_of,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    refusal = pp_dspark_refusal(cfg, model_config_of(server_args).hf_config)
    if refusal is not None:
        raise ValueError(f"DSpark under SGLANG_ENABLE_PP_SPEC: {refusal}.")
    changed = resolve_single_stage_folds_off()
    if changed:
        logger.warning(
            "DSpark under SGLANG_ENABLE_PP_SPEC runs its verify and accept outside "
            "the single-stage folds; turned off: %s",
            ", ".join(changed),
        )
