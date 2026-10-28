"""DeepSeek-V4.1 under pipeline parallelism: the earlier layers a stage replays.

A ratio-1/2 layer reads three things another layer produced in the same forward:
the compressed KV and index keys of the nearest earlier kv_source layer of its
ratio, the top-k of the nearest earlier index source of its ratio (unless it is an
index source itself), and, past the candidate source, the candidate blocks that
layer publishes. When such a producer sits on an earlier pipeline stage, this
stage keeps the producer's compressor and indexer, receives the producer's
attention input and q_lora with the hidden states, and replays it before its own
layers. The stage then holds its own copy of the producer's compressed KV and
index keys, and rebuilds the per-step top-k and candidates locally.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence, Tuple


class StageSources(NamedTuple):
    """What one pipeline stage replays, receives and sends, by layer id."""

    # Producers on earlier stages that this stage replays, ascending.
    replayed: Tuple[int, ...]
    # Producers whose attention input arrives from the previous stage.
    received: Tuple[int, ...]
    # Producers whose attention input goes on to the next stage.
    sent: Tuple[int, ...]
    # The sent producers that are this stage's own layers (captured here).
    captured: Tuple[int, ...]


def replayed_sources(
    compress_ratios: Sequence[int],
    kv_sources: Sequence[int],
    index_sources: Sequence[int],
    candidate_source: int,
    stage: range,
) -> Tuple[int, ...]:
    """Producers before ``stage`` whose outputs the stage's ratio-1/2 layers read."""
    kv_sources = set(kv_sources)
    index_sources = set(index_sources)
    replayed = set()
    for ratio in (1, 2):
        layers = [layer for layer in stage if compress_ratios[layer] == ratio]
        if not layers:
            continue
        first = layers[0]
        kv = [l for l in kv_sources if compress_ratios[l] == ratio and l <= first]
        if kv and max(kv) < stage.start:
            replayed.add(max(kv))
        if first not in index_sources:
            index = [
                l for l in index_sources if compress_ratios[l] == ratio and l < first
            ]
            if index and max(index) < stage.start:
                replayed.add(max(index))
    if 0 <= candidate_source < stage.start:
        readers = [
            layer
            for layer in (*stage, *replayed)
            if layer in index_sources and layer > candidate_source
        ]
        if readers:
            replayed.add(candidate_source)
    return tuple(sorted(replayed))


def _config_sources(config: Any) -> Tuple[Sequence[int], Sequence[int], Sequence[int], int]:
    return (
        list(getattr(config, "compress_ratios", None) or ()),
        list(getattr(config, "kv_source_layer_ids", None) or ()),
        list(getattr(config, "index_source_layer_ids", None) or ()),
        int(getattr(config, "candidate_source_layer_id", -1)),
    )


def replayed_sources_of(config: Any, stage: range) -> Tuple[int, ...]:
    """``replayed_sources`` read from a model config; () for a model without
    ratio-1/2 kv_source layers or for a stage that holds every layer it reads."""
    ratios, kv_sources, index_sources, candidate = _config_sources(config)
    if not kv_sources or not ratios:
        return ()
    return replayed_sources(ratios, kv_sources, index_sources, candidate, stage)


def stage_sources(config: Any, stages: Sequence[range], rank: int) -> StageSources:
    """The replay plan of stage ``rank`` given every stage's layer range."""
    replays = [replayed_sources_of(config, stage) for stage in stages]
    start, end = stages[rank].start, stages[rank].stop
    received = sorted(
        {p for later in replays[rank:] for p in later if p < start}
    )
    sent = sorted(
        {p for later in replays[rank + 1 :] for p in later if p < end}
    )
    captured = [p for p in sent if start <= p < end]
    return StageSources(
        tuple(replays[rank]), tuple(received), tuple(sent), tuple(captured)
    )


def pipeline_stages(num_layers: int, pp_size: int) -> Tuple[range, ...]:
    """Every stage's layer range, as get_pp_indices splits them."""
    from sglang.srt.distributed.utils import get_pp_indices

    return tuple(
        range(*get_pp_indices(num_layers, rank, pp_size)) for rank in range(pp_size)
    )


# ---------------------------------------------------------------------------
# What crosses a stage boundary besides the residual streams
# ---------------------------------------------------------------------------

PRE_KEY = "dsv41_pp_pre"


def source_input_key(layer_id: int) -> str:
    return f"dsv41_pp_x_{layer_id}"


def source_q_lora_key(layer_id: int) -> str:
    return f"dsv41_pp_q_{layer_id}"


class SourceTap:
    """The attention input and q_lora a producer saw in this forward, kept for the
    next pipeline stage. The attention backend records them; the model takes them
    once the stage's layers have run, and clears them when a forward starts."""

    __slots__ = ("x", "q_lora")

    def __init__(self) -> None:
        self.x = None
        self.q_lora = None

    def record(self, x, q_lora) -> None:
        # Copies, so no later layer can write into what the next stage replays; the
        # decode path hands the compressor and the indexer their inputs in two
        # calls, x alone first, and the first copy of each is kept.
        if x is not None and self.x is None:
            self.x = x.clone()
        if q_lora is not None and self.q_lora is None:
            self.q_lora = q_lora.clone()

    def take(self):
        x, q_lora = self.x, self.q_lora
        self.x = self.q_lora = None
        return x, q_lora


def _rows(t, num_rows: int, width: int, *, dtype, device):
    """``t`` as exactly ``num_rows`` contiguous rows; zeros where it is missing or
    short (an idle forward, or a padded batch whose producer saw only live rows)."""
    import torch

    if t is None:
        return torch.zeros((num_rows, width), dtype=dtype, device=device)
    if t.shape[0] == num_rows:
        return t.contiguous()
    out = torch.zeros((num_rows, width), dtype=t.dtype, device=t.device)
    n = min(num_rows, t.shape[0])
    out[:n].copy_(t[:n])
    return out


def pack_stage_outputs(
    tensors: dict,
    *,
    pre,
    sources: dict,
    num_rows: int,
    hidden_size: int,
    q_lora_rank: int,
    dtype,
    device,
) -> dict:
    """Add the pre-mix and each sent producer's (x, q_lora) to a stage's PP output."""
    if pre is not None:
        tensors[PRE_KEY] = pre.contiguous()
    for layer_id, (x, q_lora) in sorted(sources.items()):
        tensors[source_input_key(layer_id)] = _rows(
            x, num_rows, hidden_size, dtype=dtype, device=device
        )
        tensors[source_q_lora_key(layer_id)] = _rows(
            q_lora, num_rows, q_lora_rank, dtype=dtype, device=device
        )
    return tensors


def unpack_stage_inputs(tensors: dict, received: Sequence[int]):
    """(pre-mix or None, {producer: (x, q_lora)}) from a stage's PP input. A
    producer missing from the input maps to (None, None): the dummy forwards some
    runners make carry the residual alone."""
    pre = tensors.get(PRE_KEY)
    sources = {
        layer_id: (
            tensors.get(source_input_key(layer_id)),
            tensors.get(source_q_lora_key(layer_id)),
        )
        for layer_id in received
    }
    return pre, sources


def stage_input_specs(
    *,
    is_first_stage: bool,
    pre_width: int,
    received: Sequence[int],
    hidden_size: int,
    q_lora_rank: int,
) -> dict:
    """{key: (width, dtype or None for the model dtype)} of the extra tensors a
    non-first stage receives; the CUDA-graph runners allocate them beside
    hidden_states."""
    import torch

    if is_first_stage:
        return {}
    specs = {}
    if pre_width:
        specs[PRE_KEY] = (pre_width, torch.float32)
    for layer_id in received:
        specs[source_input_key(layer_id)] = (hidden_size, None)
        specs[source_q_lora_key(layer_id)] = (q_lora_rank, None)
    return specs
