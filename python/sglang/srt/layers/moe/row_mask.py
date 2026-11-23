# SPDX-License-Identifier: Apache-2.0
"""Routed experts skipped per row.

A forward may carry a row mask (``ForwardBatch.moe_verify_row_mask``, int32 per
token row, published as ``ForwardContext.moe_row_mask``): the verify rows a
scheduled DSpark round does not need. The routed experts send every route of a
0 row to an id no expert owns, so no expert computes it and its routed output
is 0; the rows marked 1 keep their bits. Shapes do not change, so the same CUDA
graph serves any mask.

* A hot/cold layer masks inside its own route split (``HotColdFusedMoE``).
* A plain layer on FlashInfer's CUTLASS MXFP4 kernel gets the id
  ``num_experts``: the kernel treats an id outside its experts as non-local (no
  GEMM row, skipped at finalize), which is what a hot/cold layer's hot tier
  already receives for every cold route.

Any other routed-experts runner refuses a masked forward by name.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.model_executor.forward_context import current_moe_row_mask

# Runners whose kernel skips an id outside [0, num_experts) and writes 0 for a row
# that has no local route.
_NON_LOCAL_ID_BACKENDS = frozenset({"flashinfer_cutlass_sm120"})


def mask_plain_routes(experts, topk_output):
    """``topk_output`` with the masked rows' routes sent to the non-local id, for a
    layer that does not mask in its own route split. Unchanged without a mask."""
    row_mask = current_moe_row_mask()
    if row_mask is None or getattr(experts, "masks_rows_in_route_split", False):
        return topk_output
    backend = getattr(experts, "_dsv4_mxfp4_backend", None) or getattr(
        experts, "_mxfp4_backend", None
    )
    if backend not in _NON_LOCAL_ID_BACKENDS:
        raise RuntimeError(
            "A routed-experts row mask (SGLANG_PP_DSPARK_VERIFY_SURVIVAL) needs a "
            "hot/cold layer or the FlashInfer CUTLASS MXFP4 kernel on SM120, whose "
            f"finalize skips a non-local id; this layer runs {backend!r}."
        )
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise TypeError(
            f"a routed-experts row mask needs standard top-k output, got "
            f"{topk_output.format!r}"
        )
    if getattr(experts, "moe_ep_size", 1) > 1:
        raise RuntimeError(
            "a routed-experts row mask does not serve expert parallelism: the "
            "non-local id of one rank is another rank's expert"
        )
    topk_ids = topk_output.topk_ids
    if row_mask.shape[0] != topk_ids.shape[0]:
        raise RuntimeError(
            f"a row mask of {row_mask.shape[0]} rows for {topk_ids.shape[0]} "
            "routed rows"
        )
    masked_ids = torch.where(
        row_mask.view(-1, 1) != 0, topk_ids, int(experts.num_experts)
    )
    # The packed-id field (FlashInfer routed MoE) would still name the experts.
    return StandardTopKOutput(
        topk_output.topk_weights, masked_ids, topk_output.router_logits
    )
