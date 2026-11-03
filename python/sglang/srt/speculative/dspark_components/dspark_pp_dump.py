"""Measurement only, never proposed: the hidden states a draft reads, written to disk.

The draft's acceptance by keep set and block size is measured offline, on any machine, from
the hidden states of target layers 37-39 for every prefill row of a frozen set. No build for
serving sets SGLANG_DSV41_DRAFT_DUMP_DIR, and a launch without it never imports this.
"""

from __future__ import annotations

import logging
import os
from typing import List

import torch

logger = logging.getLogger(__name__)

DUMP_MAX_BYTES = 40 << 30


class DraftHiddenDump:
    def __init__(self, folder: str, layer_ids: List[int], max_bytes: int = DUMP_MAX_BYTES):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.layer_ids = list(layer_ids)
        self.max_bytes = max_bytes
        self.written = 0
        self.seq = 0
        self.rows = 0
        self.full = False

    def record(self, forward_batch, input_ids, positions, aux_hidden) -> None:
        if self.full:
            return
        size = aux_hidden.numel() * 2
        if self.written + size > self.max_bytes:
            self.full = True
            logger.warning(
                "Draft hidden dump stops at %d rows: %d bytes is its cap.",
                self.rows,
                self.max_bytes,
            )
            return
        path = os.path.join(self.folder, f"forward-{self.seq:07d}.pt")
        torch.save(
            {
                "rids": list(forward_batch.rids or []),
                "extend_seq_lens": list(forward_batch.extend_seq_lens_cpu or []),
                "input_ids": input_ids.detach().to("cpu", dtype=torch.int32),
                "positions": positions.detach().to("cpu", dtype=torch.int32),
                "layer_ids": self.layer_ids,
                "hidden": aux_hidden.detach().to("cpu", dtype=torch.bfloat16),
            },
            path + ".partial",
        )
        os.replace(path + ".partial", path)
        self.written += size
        self.rows += aux_hidden.shape[0]
        self.seq += 1


def install_draft_dump(causal_lm, folder: str, layer_ids: List[int]) -> DraftHiddenDump:
    """The last stage captures what a draft would read and writes each prefill forward."""
    causal_lm.set_dspark_layers_to_capture(layer_ids)
    dump = DraftHiddenDump(folder, layer_ids)
    inner = causal_lm.model
    forward = inner.forward

    def forward_and_dump(input_ids, positions, forward_batch, *args, **kwargs):
        out = forward(input_ids, positions, forward_batch, *args, **kwargs)
        if forward_batch.forward_mode.is_extend_without_speculative():
            dump.record(forward_batch, input_ids, positions, out[1])
        return out

    inner.forward = forward_and_dump
    logger.info(
        "Draft hidden dump on: layers %s of each prefill row, under %s.", layer_ids, folder
    )
    return dump
