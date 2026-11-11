# SPDX-License-Identifier: Apache-2.0
"""Importance matrix of every routed expert of a DeepSeek-V4 model, plain MXFP4 path.

Switched on by ``SGLANG_DSV41_EXPERT_IMATRIX_DIR`` (environ.py); with it unset
nothing is imported, installed or computed. ``install`` (called from
``DeepseekV4ForCausalLM.post_load_weights``) puts a forward hook on the top-k
module of every routed MoE layer of the stage: the hook sees the experts' input
``x`` [T, hidden] and the routes (``topk_ids`` / ``topk_weights`` [T, top_k]) of
the forward, the same tensors the experts module receives. Per layer and expert
it accumulates on the GPU, in float64:

* ``count``: the routes to the expert (a padded id, < 0 or >= the expert count,
  is never counted; a token routed twice to one expert counts twice);
* ``wsum``: the sum of the router weights of those routes;
* ``sum_x2`` [E, hidden]: the sum of x^2 over the routes (WEIGHTING "count", the
  default: each route once, as llama-imatrix / ik_llama.cpp's imatrix do for
  ``ffn_*_exps``: they square the expert's input row and add 1 to the count,
  the router weight is applied after the down projection and never seen) and/or
  ``wsum_x2``: the same sum with each route times its router weight
  (WEIGHTING "weight"; "both" keeps both);
* ``sum_a2`` / ``wsum_a2`` [E, intermediate]: the same for a, the input of the
  down projection w2 = the clamped SwiGLU of the expert's gate and up
  projections. The fused FlashInfer kernel does not expose a, so it is
  recomputed eagerly for the routed tokens from the expert's own MXFP4 weights
  (dequantised exactly in fp32, chunks of experts of about ``CHUNK_MIB``, fp32
  matmuls with TF32 off, SwiGLU in fp32: gate.clamp(max=limit),
  up.clamp(-limit, limit), silu(gate) * up). ``X_ONLY`` skips it. The
  dequant of the layer's 384 experts dominates its cost, so with
  ``A2_FLUSH_TOKENS`` > 0 the routed inputs are kept (a bf16 copy) and the
  recompute runs once per that many tokens, and before every dump but a
  SIGUSR1 one (which records ``a2_pending_tokens`` instead).

Only real extend forwards are counted (the model passes the result of
``dsv41_forward_skip_reason``: no capture, no warm-up, no decode, at least
``MIN_TOKENS`` tokens). With ``WINDOWS`` (comma-separated corpus window files,
one JSON object with ``ids`` per line) a forward counts only when the token ids
of every request in it are one of those windows, each window once: a smoke
load, a boot probe or a client's repeated first request is skipped (counted in
the meta block's ``window_filter``), so the ``EXPECT_FORWARDS`` counts stay the
window counts whatever else the boot is sent. A call inside a CUDA-graph capture while a forward is
active raises (eager only). At the end of each counted forward every layer of
the stage must have been seen exactly once, else the miss is counted and logged
(a replayed prefill graph or a path that skips the top-k module would otherwise
lose data silently).

Self-checks, first counted forward of each layer: the unswizzled E8M0 scales are
swizzled again with FlashInfer's ``block_scale_interleave`` and compared with the
layer's bytes (a mismatch raises); ``CHECK_TOKENS`` tokens are recomputed whole
(sum over the routes of weight * w2(a)) and compared with the fused kernel's
output (relative L2, written into the meta block; the kernel runs MXFP8
activations, so a few percent is expected and a layout error gives ~1).

Output: DIR/expert-imatrix-pp<P>-tp<T>.npz (keys ``L<layer>.<field>`` and a
``meta`` JSON string: forwards, tokens, requests and a sha256 of each request's
token ids, the corpus sha256 from ``CORPUS_SHA256``, the checks, the seconds
spent), written atomically every ``SAVE_EVERY`` forwards, at each count listed
in ``EXPECT_FORWARDS`` (then also kept as ...-f<count>.npz, so that a boot sent
the calibration half and then the held-out half yields both: the sums are
additive and the converter subtracts one snapshot from a later one), on SIGUSR1
(pid file beside it) and at exit.
SGLang stops its schedulers with SIGKILL, so the exit dump is a bonus only.
A converter outside the package turns the files into the per-expert means
``L<L>.w13.<E>`` / ``L<L>.w2.<E>`` a quantiser takes as its importance matrix.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import signal
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

TAG = "dsv41 expert imatrix"
FORMAT = "dsv41-expert-imatrix/1"
WEIGHTINGS = ("count", "weight", "both")

# E2M1 code -> value; bit 3 is the sign. Low nibble = even column.
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
MXFP4_BLOCK = 32


class ExpertImatrixError(RuntimeError):
    pass


# --------------------------------------------------------------- MXFP4 numerics


def swizzle_128x4_reference(sf: torch.Tensor) -> torch.Tensor:
    """[n, M, K] -> [n, M, K]: the SWIZZLED_128x4 layout FlashInfer's
    block_scale_interleave writes per expert (quantization_utils.cuh
    get_sf_out_offset_128x4), as a plain index computation; M % 128 == 0 and
    K % 4 == 0 (no padding). Tests and documentation only."""
    n, m_rows, k_cols = sf.shape
    if m_rows % 128 or k_cols % 4:
        raise ValueError(
            f"swizzle needs M % 128 == 0 and K % 4 == 0, got {m_rows} x {k_cols}"
        )
    m = torch.arange(m_rows).view(-1, 1)
    k = torch.arange(k_cols).view(1, -1)
    k_tiles = k_cols // 4
    offset = (
        (m // 128) * (k_tiles * 512)
        + (k // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (k % 4)
    )
    out = torch.empty(n, m_rows * k_cols, dtype=sf.dtype, device=sf.device)
    out[:, offset.reshape(-1).to(sf.device)] = sf.reshape(n, -1)
    return out.view(n, m_rows, k_cols)


def unswizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """Inverse of the per-expert SWIZZLED_128x4 layout: [n, M, K] -> [n, M, K]."""
    n, m_rows, k_cols = sf.shape
    if m_rows % 128 or k_cols % 4:
        raise ValueError(
            f"unswizzle needs M % 128 == 0 and K % 4 == 0, got {m_rows} x {k_cols}"
        )
    # stored order [mTile, kTile, outerM(32), innerM(4), innerK(4)]; m = mTile*128 + innerM*32 + outerM
    v = sf.reshape(n, m_rows // 128, k_cols // 4, 32, 4, 4)
    return v.permute(0, 1, 4, 3, 2, 5).reshape(n, m_rows, k_cols)


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes uint8 [n, R, C/2] (two E2M1 codes a byte, low nibble = even column),
    scales uint8 [n, R, C/32] (E8M0, unswizzled) -> fp32 [n, R, C], exact."""
    n, rows, half = codes.shape
    cols = half * 2
    if scales.shape != (n, rows, cols // MXFP4_BLOCK):
        raise ValueError(
            f"scales {tuple(scales.shape)} do not match codes {tuple(codes.shape)}"
        )
    # one gather from a 256-entry table of (low, high) value pairs writes the
    # interleaved columns directly
    out = _pair_lut(codes.device)[codes.to(torch.int32)].view(n, rows, cols)
    factor = torch.exp2(scales.to(torch.float32) - 127.0)
    out.view(n, rows, cols // MXFP4_BLOCK, MXFP4_BLOCK).mul_(factor.unsqueeze(-1))
    return out


_PAIR_LUTS: Dict[torch.device, torch.Tensor] = {}


def _pair_lut(device) -> torch.Tensor:
    lut = _PAIR_LUTS.get(device)
    if lut is None:
        v = torch.tensor(E2M1_VALUES, dtype=torch.float32)
        b = torch.arange(256)
        lut = torch.stack([v[b & 15], v[b >> 4]], dim=1).to(
            device
        )  # [256, 2]: low nibble = even column
        _PAIR_LUTS[device] = lut
    return lut


def clamped_swiglu_fp32(
    gate: torch.Tensor, up: torch.Tensor, limit: Optional[float]
) -> torch.Tensor:
    gate = gate.float()
    up = up.float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return gate * torch.sigmoid(gate) * up


@contextlib.contextmanager
def _no_tf32():
    if not torch.cuda.is_available():
        yield
        return
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m
        torch.backends.cudnn.allow_tf32 = c


class Mxfp4ExpertWeights:
    """Dequantised views of one layer's routed experts, per chunk of experts.

    w13 codes [E, 2I, H/2] and scales [E, 2I, H/32]; w2 codes [E, H, I/2] and
    scales [E, H, I/32]; ``up_first``: rows [0, I) of w13 are the up projection
    (w3) and [I, 2I) the gate (w1), the order the FlashInfer CUTLASS method loads
    (``load_up_proj_weight_first``); ``swizzled``: the scales are in FlashInfer's
    SWIZZLED_128x4 layout per expert (SM120)."""

    def __init__(
        self,
        w13_codes,
        w13_scales,
        w2_codes,
        w2_scales,
        *,
        up_first: bool,
        swizzled: bool,
    ):
        self.w13_codes = w13_codes
        self.w13_scales = w13_scales
        self.w2_codes = w2_codes
        self.w2_scales = w2_scales
        self.up_first = bool(up_first)
        self.swizzled = bool(swizzled)
        self.num_experts = int(w13_codes.shape[0])
        self.inter = int(w13_codes.shape[1]) // 2
        self.hidden = int(w13_codes.shape[2]) * 2

    @classmethod
    def from_experts_module(cls, experts) -> Mxfp4ExpertWeights:
        backend = getattr(experts, "_dsv4_mxfp4_backend", None)
        if backend != "flashinfer_cutlass_sm120":
            raise ExpertImatrixError(
                f"{TAG}: the a^2 recompute reads the FlashInfer CUTLASS SM120 layout "
                f"(raw E2M1 bytes, swizzled E8M0 scales); this layer's backend is {backend!r}: "
                "set SGLANG_DSV41_EXPERT_IMATRIX_X_ONLY=1 or serve on the SM120 path"
            )
        qm = getattr(experts, "quant_method", None)
        up_first = bool(getattr(qm, "load_up_proj_weight_first", False))
        u8 = lambda t: t.data.view(torch.uint8)  # noqa: E731
        return cls(
            u8(experts.w13_weight),
            u8(experts.w13_weight_scale_inv),
            u8(experts.w2_weight),
            u8(experts.w2_weight_scale_inv),
            up_first=up_first,
            swizzled=True,
        )

    def _scales(self, scales: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        s = scales.index_select(0, ids)
        return unswizzle_128x4(s) if self.swizzled else s

    def gate_up(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """fp32 gate [n, I, H] and up [n, I, H] of the experts ``ids``."""
        w = dequant_mxfp4(
            self.w13_codes.index_select(0, ids), self._scales(self.w13_scales, ids)
        )
        first, second = w[:, : self.inter], w[:, self.inter :]
        return (second, first) if self.up_first else (first, second)

    def down(self, ids: torch.Tensor) -> torch.Tensor:
        """fp32 w2 [n, H, I] of the experts ``ids``."""
        return dequant_mxfp4(
            self.w2_codes.index_select(0, ids), self._scales(self.w2_scales, ids)
        )

    def w13_transient_bytes(self) -> int:
        # the fp32 matrix plus one temporary of the same size inside dequant_mxfp4
        return 2 * (2 * self.inter) * self.hidden * 4

    def check_swizzle(self) -> str:
        """'exact', or 'unavailable: ...'; raises on a mismatch."""
        if not self.swizzled:
            return "not swizzled"
        try:
            from flashinfer import block_scale_interleave
        except Exception as e:  # noqa: BLE001
            return f"unavailable: {e!r}"
        probe = torch.tensor([0, self.num_experts - 1], device=self.w13_scales.device)
        for name, stored in (("w13", self.w13_scales), ("w2", self.w2_scales)):
            raw = unswizzle_128x4(stored.index_select(0, probe))
            try:
                again = torch.stack(
                    [
                        block_scale_interleave(raw[i].contiguous()).reshape(-1)
                        for i in range(raw.shape[0])
                    ]
                )
            except Exception as e:  # noqa: BLE001 - e.g. a CPU tensor; the output self-check still runs
                return f"unavailable: {e!r}"
            if not torch.equal(
                again, stored.index_select(0, probe).reshape(probe.numel(), -1)
            ):
                raise ExpertImatrixError(
                    f"{TAG}: the unswizzled {name} scales do not swizzle back to the layer's bytes; "
                    "the a^2 recompute would read wrong scales"
                )
        return "exact"


# --------------------------------------------------------------- accumulation


def flat_routes(ids: torch.Tensor, weights: torch.Tensor, num_experts: int):
    """(expert, token, weight float64, invalid count) of the valid routes, in
    row-major route order."""
    t, k = ids.shape
    flat = ids.reshape(-1).long()
    valid = (flat >= 0) & (flat < num_experts)
    tok = torch.arange(t, device=ids.device).repeat_interleave(k)
    w = weights.reshape(-1).to(torch.float64)
    return flat[valid], tok[valid], w[valid], (~valid).sum()


def accumulate_x2(
    x: torch.Tensor,
    experts: torch.Tensor,
    tokens: torch.Tensor,
    weights: torch.Tensor,
    count: torch.Tensor,
    wsum: torch.Tensor,
    sum_x2: Optional[torch.Tensor],
    wsum_x2: Optional[torch.Tensor],
    route_chunk: int = 8192,
) -> None:
    count.index_add_(0, experts, torch.ones_like(experts, dtype=count.dtype))
    wsum.index_add_(0, experts, weights)
    for s in range(0, experts.numel(), route_chunk):
        e = experts[s : s + route_chunk]
        rows = (
            x.index_select(0, tokens[s : s + route_chunk]).to(torch.float64).square_()
        )
        if sum_x2 is not None:
            sum_x2.index_add_(0, e, rows)
        if wsum_x2 is not None:
            wsum_x2.index_add_(
                0, e, rows.mul_(weights[s : s + route_chunk].unsqueeze(1))
            )


def _group_by_expert(experts, tokens, weights):
    order = torch.argsort(experts, stable=True)
    e_sorted = experts[order]
    uniq, counts = torch.unique_consecutive(e_sorted, return_counts=True)
    return uniq.tolist(), counts.tolist(), tokens[order], weights[order]


def accumulate_a2(
    x: torch.Tensor,
    experts: torch.Tensor,
    tokens: torch.Tensor,
    weights: torch.Tensor,
    weights_src: Mxfp4ExpertWeights,
    limit: Optional[float],
    sum_a2: Optional[torch.Tensor],
    wsum_a2: Optional[torch.Tensor],
    chunk_bytes: int,
) -> int:
    """Adds a^2 of every valid route; returns the number of expert chunks."""
    if experts.numel() == 0:
        return 0
    uniq, counts, tok_s, w_s = _group_by_expert(experts, tokens, weights)
    per_chunk = max(1, int(chunk_bytes) // weights_src.w13_transient_bytes())
    starts = [0]
    for c in counts:
        starts.append(starts[-1] + c)
    chunks = 0
    with _no_tf32():
        for c0 in range(0, len(uniq), per_chunk):
            ids = uniq[c0 : c0 + per_chunk]
            gate, up = weights_src.gate_up(torch.tensor(ids, device=x.device))
            chunks += 1
            for j, eid in enumerate(ids):
                a0, a1 = starts[c0 + j], starts[c0 + j + 1]
                xs = x.index_select(0, tok_s[a0:a1]).float()
                a2 = (
                    clamped_swiglu_fp32(xs @ gate[j].T, xs @ up[j].T, limit)
                    .to(torch.float64)
                    .square_()
                )
                if sum_a2 is not None:
                    sum_a2[eid].add_(a2.sum(0))
                if wsum_a2 is not None:
                    wsum_a2[eid].add_((a2 * w_s[a0:a1].unsqueeze(1)).sum(0))
            del gate, up
    return chunks


def recompute_routed_output(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    weights_src: Mxfp4ExpertWeights,
    limit: Optional[float],
) -> torch.Tensor:
    """fp32 [T, hidden]: sum over the valid routes of weight * w2 @ a(x)."""
    t = x.shape[0]
    out = torch.zeros(t, weights_src.hidden, dtype=torch.float32, device=x.device)
    e, tok, w, _ = flat_routes(ids, weights, weights_src.num_experts)
    if e.numel() == 0:
        return out
    uniq, counts, tok_s, w_s = _group_by_expert(e, tok, w)
    start = 0
    with _no_tf32():
        for eid, c in zip(uniq, counts):
            sel = torch.tensor([eid], device=x.device)
            gate, up = weights_src.gate_up(sel)
            down = weights_src.down(sel)[0]
            rows = tok_s[start : start + c]
            xs = x.index_select(0, rows).float()
            a = clamped_swiglu_fp32(xs @ gate[0].T, xs @ up[0].T, limit)
            out.index_add_(
                0, rows, (a @ down.T) * w_s[start : start + c].float().unsqueeze(1)
            )
            start += c
    return out


# --------------------------------------------------------------- the collector


class _LayerSums:
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        hidden: int,
        inter: int,
        weighting: str,
        x_only: bool,
        device,
    ):
        f64 = dict(dtype=torch.float64, device=device)
        self.layer_id = layer_id
        self.count = torch.zeros(num_experts, dtype=torch.int64, device=device)
        self.wsum = torch.zeros(num_experts, **f64)
        self.invalid_routes = torch.zeros((), dtype=torch.int64, device=device)
        cnt = weighting in ("count", "both")
        wgt = weighting in ("weight", "both")
        self.sum_x2 = torch.zeros(num_experts, hidden, **f64) if cnt else None
        self.wsum_x2 = torch.zeros(num_experts, hidden, **f64) if wgt else None
        self.sum_a2 = (
            torch.zeros(num_experts, inter, **f64) if cnt and not x_only else None
        )
        self.wsum_a2 = (
            torch.zeros(num_experts, inter, **f64) if wgt and not x_only else None
        )
        self.calls = 0
        self.last_forward = -1
        self.pending: list = []
        self.pending_tokens = 0
        self.checks: Dict[str, object] = {}

    def arrays(self) -> Dict[str, torch.Tensor]:
        out = {
            "count": self.count,
            "wsum": self.wsum,
            "invalid_routes": self.invalid_routes,
        }
        for name in ("sum_x2", "wsum_x2", "sum_a2", "wsum_a2"):
            t = getattr(self, name)
            if t is not None:
                out[name] = t
        return out


class ExpertImatrixCollector:
    def __init__(
        self,
        directory: str,
        *,
        weighting: str = "count",
        x_only: bool = False,
        chunk_bytes: int = 2048 << 20,
        min_tokens: int = 64,
        save_every: int = 50,
        expect_forwards: Sequence[int] = (),
        check_tokens: int = 64,
        corpus_sha256: str = "",
        pp_rank: int = 0,
        tp_rank: int = 0,
        sync_timing: bool = True,
        a2_flush_tokens: int = 0,
        windows: Sequence[str] = (),
    ):
        if weighting not in WEIGHTINGS:
            raise ExpertImatrixError(
                f"{TAG}: WEIGHTING must be one of {WEIGHTINGS}, got {weighting!r}"
            )
        self.directory = directory
        self.weighting = weighting
        self.x_only = bool(x_only)
        self.chunk_bytes = int(chunk_bytes)
        self.min_tokens = int(min_tokens)
        self.save_every = int(save_every)
        self.expect_forwards = sorted({int(n) for n in expect_forwards})
        self.check_tokens = int(check_tokens)
        self.corpus_sha256 = corpus_sha256
        self.pp_rank = int(pp_rank)
        self.tp_rank = int(tp_rank)
        self.sync_timing = bool(sync_timing)
        self.a2_flush_tokens = int(a2_flush_tokens)
        self.a2_flushes = 0
        self.layers: Dict[int, _LayerSums] = {}
        self.experts_modules: Dict[int, object] = {}
        self.weights_src: Dict[int, Mxfp4ExpertWeights] = {}
        self.limits: Dict[int, Optional[float]] = {}
        self.facts: Dict[str, object] = {}
        self.active = False
        self.forwards = 0
        self.tokens = 0
        self.requests = 0
        self.forward_log: List[dict] = []
        self.layer_misses: Dict[int, int] = {}
        self.unsupported_topk = 0
        self.seconds_x2 = 0.0
        self.seconds_a2 = 0.0
        self.a2_chunks = 0
        self.dumps = 0
        self.started = time.time()
        self._pending_check: Dict[int, tuple] = {}
        self._warned = set()
        self.window_files = [os.path.abspath(p) for p in windows]
        self.window_shas: Optional[Dict[str, str]] = (
            load_window_shas(self.window_files) if self.window_files else None
        )
        self.windows_seen: Dict[str, int] = {}
        self.window_skips: Dict[str, int] = {}
        self.path = os.path.join(
            directory, f"expert-imatrix-pp{self.pp_rank}-tp{self.tp_rank}.npz"
        )
        self.pid_file = os.path.join(
            directory, f"expert-imatrix-pp{self.pp_rank}-tp{self.tp_rank}.pid"
        )

    # -- registration ---------------------------------------------------------

    def add_layer(
        self,
        layer_id: int,
        experts,
        num_experts: int,
        hidden: int,
        inter: int,
        device,
        weights_src: Optional[Mxfp4ExpertWeights] = None,
        limit: Optional[float] = None,
    ) -> None:
        if layer_id in self.layers:
            raise ExpertImatrixError(f"{TAG}: layer {layer_id} registered twice")
        self.layers[layer_id] = _LayerSums(
            layer_id, num_experts, hidden, inter, self.weighting, self.x_only, device
        )
        self.experts_modules[layer_id] = experts
        if weights_src is not None:
            self.weights_src[layer_id] = weights_src
        self.limits[layer_id] = limit

    def _resolve(self, layer_id: int) -> None:
        """At a layer's first counted forward (after process_weights_after_loading):
        the weights view, the clamp, the swizzle check."""
        if layer_id in self.weights_src or self.x_only:
            return
        experts = self.experts_modules[layer_id]
        self.weights_src[layer_id] = Mxfp4ExpertWeights.from_experts_module(experts)
        self.limits[layer_id] = _swiglu_limit_of(experts)
        self.layers[layer_id].checks["swizzle"] = self.weights_src[
            layer_id
        ].check_swizzle()

    # -- per forward ----------------------------------------------------------

    def begin(
        self, skip_reason: Optional[str], input_ids=None, forward_batch=None
    ) -> None:
        self.active = skip_reason is None
        if not self.active:
            return
        lens = (
            getattr(forward_batch, "extend_seq_lens_cpu", None)
            if forward_batch is not None
            else None
        )
        entry = {}
        shas = None
        if input_ids is not None:
            ids = input_ids.detach().to("cpu", torch.int32)
            entry["tokens"] = int(ids.numel())
            lens = (
                [int(n) for n in lens]
                if lens is not None and len(lens) > 0
                else [int(ids.numel())]
            )
            if sum(lens) == ids.numel():
                shas, off = [], 0
                for n in lens:
                    shas.append(
                        hashlib.sha256(
                            ids[off : off + n].numpy().astype("<i4").tobytes()
                        ).hexdigest()
                    )
                    off += n
                entry["ids_sha256"] = shas
            entry["requests"] = len(lens)
        if self.window_shas is not None:
            skip = self._window_skip(shas)
            if skip is not None:
                self.active = False
                self.window_skips[skip] = self.window_skips.get(skip, 0) + 1
                if skip not in self._warned:
                    self._warned.add(skip)
                    logger.info(
                        "%s: a forward of %s tokens skipped by the window filter (%s); further ones are "
                        "counted in the meta block",
                        TAG,
                        entry.get("tokens", "?"),
                        skip,
                    )
                return
            for s in shas:
                self.windows_seen[s] = self.forwards + 1
        self.forwards += 1
        entry = {"i": self.forwards, **entry}
        self.tokens += entry.get("tokens", 0)
        self.requests += entry.get("requests", 0)
        self.forward_log.append(entry)

    def _window_skip(self, shas: Optional[List[str]]) -> Optional[str]:
        """None when every request of the forward is a window not seen before, else the reason to skip it."""
        if not shas:
            return "no request ids"
        if any(s not in self.window_shas for s in shas):
            return "not a window"
        if len(set(shas)) != len(shas) or any(s in self.windows_seen for s in shas):
            return "window seen before"
        return None

    def end(self) -> None:
        if not self.active:
            return
        self.active = False
        for layer_id, st in self.layers.items():
            if st.last_forward != self.forwards:
                self.layer_misses[layer_id] = self.layer_misses.get(layer_id, 0) + 1
                self._warn_once(
                    f"miss{layer_id}",
                    f"layer {layer_id} was not seen in counted forward {self.forwards} "
                    "(a replayed prefill graph or a path without the top-k module): its sums miss it",
                )
        if self.forwards in self.expect_forwards:
            self.safe_dump("expected", snapshot=True)
        elif self.save_every > 0 and self.forwards % self.save_every == 0:
            self.safe_dump("periodic")

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.error("%s: %s", TAG, msg)

    def observe(self, layer_id: int, x: torch.Tensor, topk_output) -> None:
        if not self.active:
            return
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            raise ExpertImatrixError(
                f"{TAG}: called inside a CUDA-graph capture; the collector is eager only"
            )
        st = self.layers.get(layer_id)
        if st is None:
            return
        ids, weights = _routes_of(topk_output)
        if ids is None:
            self.unsupported_topk += 1
            self._warn_once(
                "topk",
                f"top-k output {type(topk_output).__name__} has no explicit ids/weights; skipped",
            )
            return
        if st.last_forward == self.forwards:
            raise ExpertImatrixError(
                f"{TAG}: layer {layer_id} seen twice in forward {self.forwards}"
            )
        st.last_forward = self.forwards
        st.calls += 1
        if x.shape[0] != ids.shape[0]:
            raise ExpertImatrixError(
                f"{TAG}: layer {layer_id}: {x.shape[0]} tokens but {ids.shape[0]} route rows"
            )
        t0 = time.perf_counter()
        e, tok, w, invalid = flat_routes(ids, weights, st.count.shape[0])
        st.invalid_routes.add_(invalid)
        accumulate_x2(x, e, tok, w, st.count, st.wsum, st.sum_x2, st.wsum_x2)
        if self.sync_timing and x.is_cuda:
            torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        self.seconds_x2 += t1 - t0
        if not self.x_only:
            self._resolve(layer_id)
            if self.a2_flush_tokens > 0:
                # the dequant of the layer's experts dominates the recompute: keep the
                # routed inputs and run it once per a2_flush_tokens tokens
                st.pending.append((x.detach().clone(), e, tok, w))
                st.pending_tokens += x.shape[0]
                if st.pending_tokens >= self.a2_flush_tokens:
                    self._flush(layer_id)
            else:
                self.a2_chunks += accumulate_a2(
                    x,
                    e,
                    tok,
                    w,
                    self.weights_src[layer_id],
                    self.limits[layer_id],
                    st.sum_a2,
                    st.wsum_a2,
                    self.chunk_bytes,
                )
            if self.sync_timing and x.is_cuda:
                torch.cuda.current_stream().synchronize()
            self.seconds_a2 += time.perf_counter() - t1
            if (
                self.check_tokens > 0
                and "output" not in st.checks
                and layer_id not in self._pending_check
            ):
                n = min(self.check_tokens, x.shape[0])
                self._pending_check[layer_id] = (
                    x[:n].clone(),
                    ids[:n].clone(),
                    weights[:n].clone(),
                )

    def _flush(self, layer_id: int) -> None:
        st = self.layers[layer_id]
        if not st.pending:
            return
        xs, es, toks, ws, off = [], [], [], [], 0
        for x, e, tok, w in st.pending:
            xs.append(x)
            es.append(e)
            toks.append(tok + off)
            ws.append(w)
            off += x.shape[0]
        st.pending = []
        st.pending_tokens = 0
        self.a2_chunks += accumulate_a2(
            torch.cat(xs),
            torch.cat(es),
            torch.cat(toks),
            torch.cat(ws),
            self.weights_src[layer_id],
            self.limits[layer_id],
            st.sum_a2,
            st.wsum_a2,
            self.chunk_bytes,
        )
        self.a2_flushes += 1

    def flush_all(self) -> None:
        t0 = time.perf_counter()
        for layer_id in self.layers:
            self._flush(layer_id)
        self.seconds_a2 += time.perf_counter() - t0

    def check_output(self, layer_id: int, output) -> None:
        """The experts module's forward hook: the self-check against the fused output."""
        pending = self._pending_check.pop(layer_id, None)
        if pending is None:
            return
        st = self.layers[layer_id]
        if not isinstance(output, torch.Tensor):
            st.checks["output"] = f"unavailable: output is {type(output).__name__}"
            return
        x, ids, w = pending
        ref = recompute_routed_output(
            x, ids, w, self.weights_src[layer_id], self.limits[layer_id]
        )
        got = output[: x.shape[0]].float()
        rel = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
        st.checks["output"] = {
            "tokens": int(x.shape[0]),
            "rel_l2": rel,
            "pass": rel < 0.1,
        }
        (logger.info if rel < 0.1 else logger.error)(
            "%s: layer %d self-check on %d tokens: fused output vs the fp32 recompute rel L2 %.4f (%s)",
            TAG,
            layer_id,
            x.shape[0],
            rel,
            "pass"
            if rel < 0.1
            else "FAIL: the a^2 recompute does not model this kernel",
        )

    # -- output ---------------------------------------------------------------

    def meta(self) -> dict:
        return {
            "format": FORMAT,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "pid": os.getpid(),
            "layers": sorted(self.layers),
            "weighting": self.weighting,
            "x_only": self.x_only,
            "count_rule": "each valid route once (padded ids never); weighted sums multiply by the router weight",
            "a_rule": "a = silu(gate.clamp(max=limit)) * up.clamp(-limit, limit), fp32 from the exact MXFP4 dequant, TF32 off",
            "chunk_bytes": self.chunk_bytes,
            "min_tokens": self.min_tokens,
            "expect_forwards": self.expect_forwards,
            "corpus_sha256": self.corpus_sha256,
            "forwards": self.forwards,
            "tokens": self.tokens,
            "requests": self.requests,
            "forward_log": self.forward_log,
            "window_filter": None
            if self.window_shas is None
            else {
                "files": self.window_files,
                "windows": len(self.window_shas),
                "seen": len(self.windows_seen),
                "skipped_forwards": dict(self.window_skips),
            },
            "layer_calls": {str(k): v.calls for k, v in self.layers.items()},
            "layer_misses": {str(k): v for k, v in self.layer_misses.items()},
            "unsupported_topk": self.unsupported_topk,
            "checks": {str(k): v.checks for k, v in self.layers.items()},
            "swiglu_limit": {str(k): v for k, v in self.limits.items()},
            "seconds_x2": self.seconds_x2,
            "seconds_a2": self.seconds_a2,
            "a2_chunks": self.a2_chunks,
            "a2_flush_tokens": self.a2_flush_tokens,
            "a2_flushes": self.a2_flushes,
            "a2_pending_tokens": {
                str(k): v.pending_tokens
                for k, v in self.layers.items()
                if v.pending_tokens
            },
            "facts": self.facts,
            "started": self.started,
            "written": time.time(),
            "dumps": self.dumps,
        }

    def dump(self, reason: str = "manual", snapshot: bool = False) -> Optional[str]:
        import numpy as np

        if not self.layers:
            return None
        if reason != "sigusr1":
            # a signal can land inside a forward: its dump leaves the pending a^2 rows
            # pending (meta a2_pending_tokens) instead of running the recompute there
            self.flush_all()
        os.makedirs(self.directory, exist_ok=True)
        arrays = {}
        for layer_id, st in sorted(self.layers.items()):
            for name, t in st.arrays().items():
                arrays[f"L{layer_id}.{name}"] = t.detach().cpu().numpy()
        self.dumps += 1
        meta = self.meta()
        meta["reason"] = reason
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        tmp = self.path + ".partial.npz"
        np.savez(tmp, **arrays)
        if snapshot:
            snap = self.path[: -len(".npz")] + f".f{self.forwards}.npz"
            if os.path.lexists(snap):
                os.unlink(snap)
            os.link(tmp, snap)
        os.replace(tmp, self.path)
        logger.info(
            "%s: wrote %s (%s, %d forwards, %d tokens, %d layers)",
            TAG,
            self.path,
            reason,
            self.forwards,
            self.tokens,
            len(self.layers),
        )
        return self.path

    def safe_dump(self, reason: str, snapshot: bool = False) -> Optional[str]:
        try:
            return self.dump(reason, snapshot)
        except Exception as e:  # noqa: BLE001 - a failed dump must not stop serving
            logger.error("%s: dump (%s) failed: %r", TAG, reason, e)
            return None


# --------------------------------------------------------------- engine glue


def _routes_of(topk_output):
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    if TopKOutputChecker.format_is_standard(topk_output):
        return topk_output.topk_ids, topk_output.topk_weights
    return None, None


def _swiglu_limit_of(experts) -> Optional[float]:
    cfg = getattr(experts, "moe_runner_config", None)
    if cfg is not None:
        if (
            getattr(cfg, "gemm1_alpha", None) is not None
            or getattr(cfg, "gemm1_clamp_limit", None) is not None
        ):
            raise ExpertImatrixError(
                f"{TAG}: this layer's SwiGLU has alpha / a post-SiLU clamp; the recompute models the plain clamp only"
            )
        limit = getattr(cfg, "swiglu_limit", None)
        if limit is not None:
            return float(limit)
    t = getattr(getattr(experts, "quant_method", None), "_swiglu_limit_tensor", None)
    return (
        float(t.reshape(-1)[0].item())
        if isinstance(t, torch.Tensor) and t.numel()
        else None
    )


def routed_moe_blocks(model) -> List[torch.nn.Module]:
    found = {}
    for module in model.modules():
        if (
            not hasattr(module, "experts")
            or not hasattr(module, "topk")
            or not hasattr(module, "gate")
        ):
            continue
        if not getattr(module, "is_deepseek_v4", False) or getattr(
            module, "is_nextn", False
        ):
            continue
        layer_id = int(module.layer_id)
        if layer_id in found:
            raise ExpertImatrixError(f"{TAG}: two MoE blocks claim layer {layer_id}")
        found[layer_id] = module
    return [found[k] for k in sorted(found)]


def _proc_start_ticks() -> str:
    try:
        with open("/proc/self/stat") as f:
            stat = f.read()
        return stat[stat.rindex(")") + 2 :].split()[19]
    except (OSError, ValueError, IndexError):
        return "?"


def load_window_shas(paths: Sequence[str]) -> Dict[str, str]:
    """sha256 of each window's ids (little-endian int32, the converter's formula) -> window id."""
    import numpy as np

    shas: Dict[str, str] = {}
    for path in paths:
        n = 0
        with open(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                shas[
                    hashlib.sha256(
                        np.asarray(d["ids"], dtype="<i4").tobytes()
                    ).hexdigest()
                ] = str(d.get("id", n))
                n += 1
        if n == 0:
            raise ExpertImatrixError(f"{TAG}: window file {path} holds no window")
    return shas


def collector_from_env(pp_rank: int, tp_rank: int) -> Optional[ExpertImatrixCollector]:
    from sglang.srt.environ import envs

    directory = envs.SGLANG_DSV41_EXPERT_IMATRIX_DIR.get()
    if not directory:
        return None
    return ExpertImatrixCollector(
        directory,
        weighting=envs.SGLANG_DSV41_EXPERT_IMATRIX_WEIGHTING.get(),
        x_only=envs.SGLANG_DSV41_EXPERT_IMATRIX_X_ONLY.get(),
        chunk_bytes=int(envs.SGLANG_DSV41_EXPERT_IMATRIX_CHUNK_MIB.get()) << 20,
        min_tokens=envs.SGLANG_DSV41_EXPERT_IMATRIX_MIN_TOKENS.get(),
        save_every=envs.SGLANG_DSV41_EXPERT_IMATRIX_SAVE_EVERY.get(),
        expect_forwards=[
            int(v)
            for v in (
                envs.SGLANG_DSV41_EXPERT_IMATRIX_EXPECT_FORWARDS.get() or ""
            ).split(",")
            if v.strip()
        ],
        check_tokens=envs.SGLANG_DSV41_EXPERT_IMATRIX_CHECK_TOKENS.get(),
        a2_flush_tokens=envs.SGLANG_DSV41_EXPERT_IMATRIX_A2_FLUSH_TOKENS.get(),
        corpus_sha256=envs.SGLANG_DSV41_EXPERT_IMATRIX_CORPUS_SHA256.get(),
        windows=[
            p.strip()
            for p in (envs.SGLANG_DSV41_EXPERT_IMATRIX_WINDOWS.get() or "").split(",")
            if p.strip()
        ],
        pp_rank=pp_rank,
        tp_rank=tp_rank,
    )


def install(model, collector: ExpertImatrixCollector) -> ExpertImatrixCollector:
    """Hooks every routed MoE block of ``model`` (this stage) into ``collector``."""
    blocks = routed_moe_blocks(model)
    if not blocks:
        logger.info("%s: no routed MoE layer on this stage", TAG)
        return collector
    for moe in blocks:
        experts = moe.experts
        layer_id = int(moe.layer_id)
        if type(experts).__name__ == "HotColdFusedMoE":
            raise ExpertImatrixError(
                f"{TAG}: layer {layer_id} is a hot/cold layer; this collector runs on the plain MXFP4 "
                "path (all experts in the fused kernel): serve the unpruned source checkpoint"
            )
        for name in ("moe_tp_size", "moe_ep_size"):
            if int(getattr(experts, name, 1)) != 1:
                raise ExpertImatrixError(
                    f"{TAG}: {name} {getattr(experts, name)} != 1; use a pipeline-parallel layout"
                )
        if int(getattr(moe, "num_fused_shared_experts", 0) or 0) or int(
            getattr(experts, "num_fused_shared_experts", 0) or 0
        ):
            raise ExpertImatrixError(
                f"{TAG}: fused shared-expert slots are not supported"
            )
        num_experts = int(moe.config.n_routed_experts)
        local = int(getattr(experts, "num_local_experts", num_experts))
        if (
            local != num_experts
            or int(getattr(experts, "num_experts", num_experts)) != num_experts
        ):
            raise ExpertImatrixError(
                f"{TAG}: layer {layer_id}: {local} local experts for {num_experts} routed ones"
            )
        hidden = int(experts.hidden_size)
        inter = int(experts.intermediate_size_per_partition)
        topk = moe.topk
        if not isinstance(topk, torch.nn.Module):
            raise ExpertImatrixError(
                f"{TAG}: layer {layer_id}: the top-k is not a module; no hook point"
            )
        device = experts.w13_weight.device
        collector.add_layer(layer_id, experts, num_experts, hidden, inter, device)

        def topk_hook(module, args, kwargs, output, _layer=layer_id):
            x = args[0] if args else kwargs.get("hidden_states")
            collector.observe(_layer, x, output)

        def experts_hook(module, args, kwargs, output, _layer=layer_id):
            collector.check_output(_layer, output)

        topk.register_forward_hook(topk_hook, with_kwargs=True)
        if not collector.x_only and collector.check_tokens > 0:
            experts.register_forward_hook(experts_hook, with_kwargs=True)
    collector.facts = {
        "experts_class": type(blocks[0].experts).__name__,
        "quant_method": type(getattr(blocks[0].experts, "quant_method", None)).__name__,
        "num_experts": int(blocks[0].config.n_routed_experts),
        "top_k": int(blocks[0].config.num_experts_per_tok),
        "hidden": int(blocks[0].experts.hidden_size),
        "intermediate": int(blocks[0].experts.intermediate_size_per_partition),
        "weights_include_routed_scale": bool(
            getattr(
                blocks[0].experts, "should_fuse_routed_scaling_factor_in_topk", False
            )
        ),
    }
    os.makedirs(collector.directory, exist_ok=True)
    with open(collector.pid_file + ".partial", "w") as f:
        f.write(f"{os.getpid()} {_proc_start_ticks()}\n")
    os.replace(collector.pid_file + ".partial", collector.pid_file)
    atexit.register(collector.safe_dump, "exit")
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGUSR1)

        def handler(signum, frame):
            collector.safe_dump("sigusr1")
            if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                previous(signum, frame)

        signal.signal(signal.SIGUSR1, handler)
    else:
        logger.warning(
            "%s: not the main thread; no SIGUSR1 dump (periodic and expected dumps stay)",
            TAG,
        )
    logger.info(
        "%s: armed on pid %d (pp %d, tp %d) for MoE layers %d-%d (%d layers), weighting %s, x_only %s, "
        "chunk %d MiB, min tokens %d, save every %d, snapshots at forwards %s, check %d tokens, window filter %s; writes %s",
        TAG,
        os.getpid(),
        collector.pp_rank,
        collector.tp_rank,
        int(blocks[0].layer_id),
        int(blocks[-1].layer_id),
        len(blocks),
        collector.weighting,
        collector.x_only,
        collector.chunk_bytes >> 20,
        collector.min_tokens,
        collector.save_every,
        collector.expect_forwards,
        collector.check_tokens,
        "off"
        if collector.window_shas is None
        else f"{len(collector.window_shas)} windows",
        collector.path,
    )
    return collector
