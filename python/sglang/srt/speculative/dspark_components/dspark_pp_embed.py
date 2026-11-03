"""The draft's input embedding on the last pipeline stage.

The target's table lives on the first stage, and the draft embeds the anchor and the
mask token on the last. Two routes (SGLANG_PP_DSPARK_EMBED):

  * ``gpu``: the checkpoint's table on that stage, as the EAGLE drafts do. 1.23 GiB at
    V4.1's 129,280 x 5,120 BF16.
  * ``host``: the table stays in the mapped shard. A static device buffer holds the
    draft's input rows, mask rows filled once; the host copies the anchors in before
    each draft, and the draft reads the buffer in place of a gather.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.speculative.pp_draft_embedding import (
    SAFETENSORS_INDEX_NAME,
    _pick_embedding_key,
)

logger = logging.getLogger(__name__)


def locate_embedding(folder: str) -> Tuple[str, str]:
    """(shard path, tensor key) of the checkpoint's input embedding."""
    if not os.path.isdir(folder):
        raise ValueError(
            "DSpark under pipeline parallelism reads the draft's input embedding from "
            f"a local checkpoint directory; {folder!r} is not one."
        )
    with open(os.path.join(folder, SAFETENSORS_INDEX_NAME)) as fh:
        weight_map = json.load(fh)["weight_map"]
    key = _pick_embedding_key(weight_map.keys())
    if key is None:
        raise ValueError(f"No input embedding in the checkpoint index under {folder}.")
    return os.path.join(folder, weight_map[key]), key


class HostEmbeddingTable:
    """A BF16 table read from its safetensors shard through a read-only mapping."""

    def __init__(self, shard: str, key: str) -> None:
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        entry = header[key]
        if entry["dtype"] != "BF16" or len(entry["shape"]) != 2:
            raise ValueError(
                f"The host embedding reads a 2-D BF16 table, {key} is "
                f"{entry['dtype']} {entry['shape']}."
            )
        self.num_rows, self.width = int(entry["shape"][0]), int(entry["shape"][1])
        start = 8 + header_len + int(entry["data_offsets"][0])
        self._rows = np.memmap(
            shard,
            dtype=np.uint16,
            mode="r",
            offset=start,
            shape=(self.num_rows, self.width),
        )

    def rows(self, token_ids: List[int]) -> torch.Tensor:
        # fancy indexing copies out of the mapping, so the tensor owns writable memory
        return torch.from_numpy(self._rows[token_ids]).view(torch.bfloat16)


class PPDraftHostEmbedding(nn.Module):
    """Returns the staged rows: the graph records a read of the buffer, no gather."""

    def __init__(
        self,
        table: HostEmbeddingTable,
        *,
        max_rows: int,
        rows_per_request: int,
        mask_token_id: int,
        device,
    ) -> None:
        super().__init__()
        self.table = table
        self.rows_per_request = int(rows_per_request)
        self.embedding_dim = table.width
        self.device_ = device
        mask_row = table.rows([int(mask_token_id)]).to(device)
        self.staged = mask_row.repeat(int(max_rows), 1)
        self.num_stagings = 0

    def stage_anchors(self, anchor_ids: torch.Tensor) -> None:
        """Before the draft launches. The read of the anchors is a host sync, which
        this stage is already blocked on: the verify's accept has just been read."""
        ids = anchor_ids.view(-1).tolist()
        rows = self.table.rows(ids).to(self.device_, non_blocking=True)
        self.staged[:: self.rows_per_request][: len(ids)].copy_(rows)
        self.num_stagings += 1

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.staged[: input_ids.numel()]


def install_pp_draft_embedding(
    *,
    draft_model,
    route: str,
    folder: str,
    max_bs: int,
    query_token_num: int,
    mask_token_id: int,
    device,
) -> Optional[PPDraftHostEmbedding]:
    """Give the draft its embedding; returns the host module when one must be staged."""
    shard, key = locate_embedding(folder)
    if route == "gpu":
        import safetensors

        with safetensors.safe_open(shard, framework="pt", device="cpu") as fh:
            table = fh.get_tensor(key)
        draft_model.embed_tokens = nn.Embedding.from_pretrained(
            table.to(device), freeze=True
        )
        logger.info(
            "DSpark draft embedding on this stage: %s %s.", key, tuple(table.shape)
        )
        return None
    module = PPDraftHostEmbedding(
        HostEmbeddingTable(shard, key),
        max_rows=max_bs * query_token_num,
        rows_per_request=query_token_num,
        mask_token_id=mask_token_id,
        device=device,
    )
    draft_model.embed_tokens = module
    logger.info(
        "DSpark draft embedding in host memory: %s from %s, %d rows staged on device.",
        key,
        os.path.basename(shard),
        max_bs * query_token_num,
    )
    return module
