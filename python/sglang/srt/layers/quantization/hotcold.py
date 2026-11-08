# SPDX-License-Identifier: Apache-2.0
"""The cold tier of a DeepSeek-V4 hot/cold MoE layer: ggml-quantised experts.

``ColdGGUFExperts`` holds the layer's cold experts as packed ggml rows,
preallocated from the manifest (no lazy GGUF parameter, nothing to materialise):
``w13_qweight`` uint8 [C, 2·I, row_bytes(hidden)], gate rows then up rows, and
``w2_qweight`` uint8 [C, hidden, row_bytes(I)].

``cold_moe`` runs the routes whose cold id is >= 0 (every other route, the
padded ones included, gives exactly 0): gate/up projection, the clamped SwiGLU
(``swiglu_limit``, the same clamp the hot tier applies inside FlashInfer), down
projection, route weights, sum over top-k. On CUDA the projections are the
GGUF vector kernel or, above ``SGLANG_HOTCOLD_MMQ_MIN_TOKENS`` tokens, the MoE
MMQ kernel when it is built for the cold types (``SGLANG_HOTCOLD_COLD_KERNEL``
switches). Off CUDA the projections are a float reference that dequantises the
rows (IQ2_S through gguf-py); it is what the CPU tests run and it does not
reproduce the kernels' q8_1 activation rounding.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.model_loader.hotcold_manifest import cold_row_bytes

logger = logging.getLogger(__name__)

# The vector kernel's grid limit: tokens * top_k <= 65535 (gguf_moe_vec.cuh).
VEC_MAX_ROWS = 65535


def _mmq_types():
    try:
        from sglang.kernels.ops.moe.gguf_moe_mmq import GGUF_MOE_MMQ_TYPES
    except ImportError:
        return frozenset()
    return frozenset(GGUF_MOE_MMQ_TYPES)


def select_cold_kernel(tokens: int, w13_type: int, w2_type: int) -> str:
    """'mmq' or 'vec' for a batch of ``tokens``; the same answer for a given
    (tokens, types), so a captured graph and its replay agree."""
    mode = envs.SGLANG_HOTCOLD_COLD_KERNEL.get()
    if mode not in ("auto", "vec", "mmq"):
        raise ValueError(
            f"SGLANG_HOTCOLD_COLD_KERNEL must be auto, vec or mmq, got {mode!r}"
        )
    if mode == "vec":
        return "vec"
    supported = {int(w13_type), int(w2_type)} <= _mmq_types()
    if mode == "mmq":
        if not supported:
            raise RuntimeError(
                f"SGLANG_HOTCOLD_COLD_KERNEL=mmq but gguf_moe_mmq does not serve types "
                f"{w13_type}/{w2_type}"
            )
        return "mmq"
    if supported and tokens > envs.SGLANG_HOTCOLD_MMQ_MIN_TOKENS.get():
        return "mmq"
    return "vec"


def _clamped_swiglu_reference(h: torch.Tensor, limit: Optional[float]) -> torch.Tensor:
    """The SiluAndMulClamp kernel's arithmetic: clamp in bf16 (gate above, up both
    sides), silu * up in fp32, rounded to the input dtype."""
    half = h.shape[-1] // 2
    gate, up = h[..., :half], h[..., half:]
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    gate = gate.float()
    return (gate * torch.sigmoid(gate) * up.float()).to(h.dtype)


def dequantize_cold_rows(packed: torch.Tensor, ggml_type: int) -> torch.Tensor:
    """float32 [..., K] from packed rows [..., row_bytes]; CPU, gguf-py types only."""
    import gguf
    import numpy as np

    try:
        qtype = gguf.GGMLQuantizationType(int(ggml_type))
        data = packed.detach().to("cpu").contiguous().numpy()
        out = gguf.quants.dequantize(data, qtype)
    except (ValueError, NotImplementedError) as e:
        raise NotImplementedError(
            f"gguf-py cannot dequantise ggml type {ggml_type} ({e}); the CPU "
            "reference serves the types gguf-py knows, IQ2_S (22) among them"
        ) from e
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))


class HotColdImatrixCollector:
    """Per (layer, cold slot): Σx² over the gate/up input channels and Σa² over the
    clamped SwiGLU output channels of the routes the cold tier ran, and the route
    count. Routes with a negative cold id (hot or padded) are never counted.
    Eager only; dumped as one ``.npz`` per rank at exit and on SIGUSR1."""

    _instance: Optional[HotColdImatrixCollector] = None
    _lock = threading.Lock()

    def __init__(self, directory: str):
        self.directory = directory
        self.sum_x2: Dict[int, torch.Tensor] = {}
        self.sum_a2: Dict[int, torch.Tensor] = {}
        self.count: Dict[int, torch.Tensor] = {}

    @classmethod
    def get(cls) -> Optional[HotColdImatrixCollector]:
        directory = envs.SGLANG_DSV41_HOTCOLD_IMATRIX_DIR.get()
        if not directory:
            return None
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(directory)
                atexit.register(cls._instance.dump)
                try:
                    signal.signal(signal.SIGUSR1, lambda *_: cls._instance.dump())
                except ValueError:  # not the main thread
                    logger.warning("hot/cold imatrix: SIGUSR1 dump not installed")
            return cls._instance

    def accumulate(
        self, layer_id: int, x: torch.Tensor, a: torch.Tensor, cold_ids: torch.Tensor
    ) -> None:
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the hot/cold imatrix collector is eager only; run with CUDA graphs disabled"
            )
        if layer_id not in self.count:
            raise RuntimeError(f"hot/cold imatrix: layer {layer_id} has no slot table")
        top_k = cold_ids.shape[1]
        flat = cold_ids.reshape(-1).long()
        valid = flat >= 0
        slots = flat[valid]
        x2 = x.float().pow(2).repeat_interleave(top_k, dim=0)[valid]
        a2 = a.float().pow(2)[valid]
        self.sum_x2[layer_id].index_add_(0, slots, x2.double())
        self.sum_a2[layer_id].index_add_(0, slots, a2.double())
        self.count[layer_id].index_add_(0, slots, torch.ones_like(slots))

    def set_num_slots(
        self, layer_id: int, num_slots: int, x_width: int, a_width: int, device
    ) -> None:
        if layer_id not in self.sum_x2:
            self.sum_x2[layer_id] = torch.zeros(
                num_slots, x_width, dtype=torch.float64, device=device
            )
            self.sum_a2[layer_id] = torch.zeros(
                num_slots, a_width, dtype=torch.float64, device=device
            )
            self.count[layer_id] = torch.zeros(
                num_slots, dtype=torch.int64, device=device
            )

    def dump(self) -> Optional[str]:
        import numpy as np

        if not self.count:
            return None
        rank = (
            os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or str(os.getpid())
        )
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"hotcold-imatrix-rank{rank}.npz")
        arrays = {}
        for layer_id in sorted(self.count):
            arrays[f"l{layer_id}.sum_x2"] = self.sum_x2[layer_id].cpu().numpy()
            arrays[f"l{layer_id}.sum_a2"] = self.sum_a2[layer_id].cpu().numpy()
            arrays[f"l{layer_id}.count"] = self.count[layer_id].cpu().numpy()
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        logger.info("hot/cold imatrix: wrote %s (%d layers)", path, len(self.count))
        return path


# Kernel seams: the CPU tests replace these to drive cold_moe's CUDA branch.
def _vec_project(x, weight, ids, top_k, qtype, rows, tokens):
    from sglang.kernels.ops.moe.gguf_moe_vec import gguf_moe_a8_vec

    return gguf_moe_a8_vec(x, weight, ids, top_k, qtype, rows, tokens)


def _mmq_block(qtype):
    from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq_block_size

    return gguf_moe_mmq_block_size(qtype)


def _mmq_align(ids, block, num_experts):
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(ids, block, num_experts, pad_expert_ids=True)


def _mmq_project(
    x, weight, sorted_ids, expert_ids, post_padded, qtype, rows, top_k, tokens, block
):
    from sglang.kernels.ops.moe.gguf_moe_mmq import gguf_moe_mmq

    return gguf_moe_mmq(
        x,
        weight,
        sorted_ids,
        expert_ids,
        post_padded,
        qtype,
        rows,
        top_k,
        tokens,
        block,
    )


def _clamped_swiglu(h, limit):
    if limit is None:
        from sglang.kernels.ops.activation.activation import silu_and_mul

        return silu_and_mul(h)
    from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp

    a = torch.empty((h.shape[0], h.shape[1] // 2), dtype=h.dtype, device=h.device)
    silu_and_mul_clamp(h, a, float(limit))
    return a


def _weight_and_sum(y, topk_weights, ids, tokens, top_k):
    from sgl_kernel import moe_sum

    from sglang.kernels.ops.moe.gguf_moe_weights import apply_gguf_moe_weights

    apply_gguf_moe_weights(y, topk_weights, ids)
    out = torch.empty((tokens, y.shape[1]), dtype=y.dtype, device=y.device)
    moe_sum(y.view(tokens, top_k, -1), out)
    return out


def _reference_project(x, dequantized, ids, top_k, rows):
    """[tokens * top_k, rows] in x's dtype; zero for negative ids."""
    flat = ids.reshape(-1).long()
    src = x.repeat_interleave(top_k, dim=0) if top_k > 1 else x
    out = torch.zeros(flat.numel(), rows, dtype=torch.float32, device=x.device)
    valid = flat >= 0
    if valid.any():
        w = dequantized[flat[valid]]  # [n, rows, K]
        out[valid] = torch.bmm(w, src[valid].float().unsqueeze(-1)).squeeze(-1)
    return out.to(x.dtype)


def cold_moe(
    x: torch.Tensor,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    topk_weights: torch.Tensor,
    cold_ids: torch.Tensor,
    w13_type: int,
    w2_type: int,
    swiglu_limit: Optional[float],
    *,
    layer_id: int = -1,
    reference_weights: Optional[tuple] = None,
    collector: Optional[HotColdImatrixCollector] = None,
    use_kernels: Optional[bool] = None,
) -> torch.Tensor:
    """Sum over the cold routes of ``topk_weights * expert(x)``; [tokens, hidden]."""
    tokens, hidden = x.shape
    top_k = cold_ids.shape[1]
    inter2 = w13_qweight.shape[1]
    if tokens == 0:
        return x.new_zeros((0, w2_qweight.shape[1]))
    if use_kernels is None:
        use_kernels = x.is_cuda
    if not use_kernels:
        if reference_weights is None:
            raise RuntimeError("the CPU cold path needs dequantised reference weights")
        w13_f, w2_f = reference_weights
        h = _reference_project(x, w13_f, cold_ids, top_k, inter2)
        a = _clamped_swiglu_reference(h, swiglu_limit)
        if collector is not None:
            collector.accumulate(layer_id, x, a, cold_ids)
        y = _reference_project(a, w2_f, cold_ids, 1, w2_qweight.shape[1])
        valid = (cold_ids >= 0).reshape(-1, 1)
        weights = topk_weights.reshape(-1, 1).float().masked_fill(~valid, 0.0)
        y = (y.float() * weights).to(x.dtype).masked_fill(~valid, 0)
        return y.view(tokens, top_k, -1).float().sum(dim=1).to(x.dtype)

    ids = (
        cold_ids
        if cold_ids.dtype == torch.int32
        else cold_ids.clamp_min(-1).to(torch.int32)
    )
    ids = ids.contiguous()
    rows2 = w2_qweight.shape[1]
    kernel = select_cold_kernel(tokens, w13_type, w2_type)
    if kernel == "mmq":
        block = _mmq_block(w13_type)
        if _mmq_block(w2_type) != block:
            raise RuntimeError("the cold w13 and w2 types need one MMQ block size")
        sorted_ids, expert_ids, post_padded = _mmq_align(
            ids, block, w13_qweight.shape[0]
        )
        h = _mmq_project(
            x,
            w13_qweight,
            sorted_ids,
            expert_ids,
            post_padded,
            w13_type,
            inter2,
            top_k,
            tokens,
            block,
        )
        a = _clamped_swiglu(h, swiglu_limit)
        if collector is not None:
            collector.accumulate(layer_id, x, a, cold_ids)
        y = _mmq_project(
            a,
            w2_qweight,
            sorted_ids,
            expert_ids,
            post_padded,
            w2_type,
            rows2,
            1,
            tokens * top_k,
            block,
        )
    else:
        # The vector kernel's grid holds tokens * top_k <= 65535 rows: an eager
        # prefill above that runs in chunks (no captured batch comes near it).
        step = VEC_MAX_ROWS // top_k
        parts = []
        for start in range(0, tokens, step):
            end = min(tokens, start + step)
            xs = x if (start, end) == (0, tokens) else x[start:end].contiguous()
            part_ids = (
                ids if (start, end) == (0, tokens) else ids[start:end].contiguous()
            )
            h = _vec_project(
                xs, w13_qweight, part_ids, top_k, w13_type, inter2, end - start
            )
            a = _clamped_swiglu(h, swiglu_limit)
            if collector is not None:
                collector.accumulate(layer_id, xs, a, cold_ids[start:end])
            parts.append(
                _vec_project(
                    a, w2_qweight, part_ids, 1, w2_type, rows2, (end - start) * top_k
                )
            )
        y = parts[0] if len(parts) == 1 else torch.cat(parts)
    return _weight_and_sum(y, topk_weights, ids, tokens, top_k)


class ColdGGUFExperts(nn.Module):
    """The cold experts of one layer, packed ggml rows by slot."""

    def __init__(
        self,
        *,
        num_cold: int,
        hidden_size: int,
        intermediate_size: int,
        w13_type: int,
        w2_type: int,
        swiglu_limit: Optional[float],
        layer_id: int,
    ):
        super().__init__()
        self.num_cold = num_cold
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w13_type = int(w13_type)
        self.w2_type = int(w2_type)
        self.swiglu_limit = swiglu_limit
        self.layer_id = layer_id
        self.w13_qweight = Parameter(
            torch.empty(
                num_cold,
                2 * intermediate_size,
                cold_row_bytes(w13_type, hidden_size),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w2_qweight = Parameter(
            torch.empty(
                num_cold,
                hidden_size,
                cold_row_bytes(w2_type, intermediate_size),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        for p in (self.w13_qweight, self.w2_qweight):
            p.weight_loader = self.weight_loader
        self._reference = None
        self._loaded = {"w13": set(), "w2": False}

    def weight_loader(
        self, param: Parameter, loaded: torch.Tensor, shard_id: Optional[int]
    ) -> None:
        if loaded.dtype != torch.uint8:
            raise TypeError(
                f"cold tensor dtype {loaded.dtype}, expected uint8 (layer {self.layer_id})"
            )
        inter = self.intermediate_size
        if param is self.w13_qweight:
            if shard_id not in (0, 1):
                raise ValueError(
                    f"w13 shard id must be 0 (gate) or 1 (up), got {shard_id}"
                )
            target = param.data[:, shard_id * inter : (shard_id + 1) * inter]
            self._loaded["w13"].add(shard_id)
        elif param is self.w2_qweight:
            target = param.data
            self._loaded["w2"] = True
        else:
            raise ValueError("not a cold parameter of this layer")
        if tuple(loaded.shape) != tuple(target.shape):
            raise ValueError(
                f"cold tensor of layer {self.layer_id} has shape {tuple(loaded.shape)}, "
                f"expected {tuple(target.shape)} (ggml types {self.w13_type}/{self.w2_type})"
            )
        target.copy_(loaded)
        self._reference = None

    def check_kernels(self) -> None:
        """At load on CUDA: build the kernels this layer will run (the JIT compiles
        here instead of at the first request) and refuse a type they lack."""
        if not self.w13_qweight.is_cuda:
            return
        from sglang.kernels.ops.moe.gguf_moe_vec import _jit_gguf_moe_vec_module

        for t in {self.w13_type, self.w2_type}:
            try:
                _jit_gguf_moe_vec_module(torch.bfloat16, t)
            except ValueError as e:
                raise RuntimeError(
                    f"hot/cold layer {self.layer_id}: the GGUF vector kernel does not "
                    f"serve cold type {t} (IQ4_KS = 144 and IQ2_KS = 145 need the GGUF MoE"
                    f" kernels that add those types): {e}"
                ) from e

    def reference_weights(self):
        if self._reference is None:
            self._reference = (
                dequantize_cold_rows(self.w13_qweight.data, self.w13_type).to(
                    self.w13_qweight.device
                ),
                dequantize_cold_rows(self.w2_qweight.data, self.w2_type).to(
                    self.w2_qweight.device
                ),
            )
        return self._reference

    def forward(
        self, x: torch.Tensor, topk_weights: torch.Tensor, cold_ids: torch.Tensor
    ) -> torch.Tensor:
        collector = HotColdImatrixCollector.get()
        if collector is not None:
            collector.set_num_slots(
                self.layer_id,
                self.num_cold,
                self.hidden_size,
                self.intermediate_size,
                x.device,
            )
        return cold_moe(
            x,
            self.w13_qweight,
            self.w2_qweight,
            topk_weights,
            cold_ids,
            self.w13_type,
            self.w2_type,
            self.swiglu_limit,
            layer_id=self.layer_id,
            reference_weights=None if x.is_cuda else self.reference_weights(),
            collector=collector,
        )
