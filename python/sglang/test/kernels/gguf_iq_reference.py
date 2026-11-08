"""Torch references for the GGUF types IQ2_S (22), IQ4_KS (144) and IQ2_KS (145).

``dequantize`` turns packed rows ``uint8 [..., rows, row_bytes]`` into float64
values on the rows' device, following ggml's block layouts (ik_llama.cpp
``dequantize_row_iq2_s`` / ``_iq4_ks`` / ``_iq2_ks``; IQ4_KS and IQ2_KS carry a
4-byte float / 2-byte half row header). ``random_rows`` builds valid packed
rows: random bytes everywhere, with finite positive scales written into the
fp16/fp32 scale fields, so every code, sign, grid index and sub-scale occurs.
"""

from __future__ import annotations

import torch

# ggml type id -> (row header bytes, block bytes per 256 values)
ROW_LAYOUT = {22: (0, 82), 144: (4, 136), 145: (2, 70)}

IQ4K_VALUES = (
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117,
)  # fmt: skip
IQ2NL_VALUES = (-31, -13, 1, 17, -26, -8, 6, 22)

_IQ2S_GRID = None


def row_bytes(quant_type: int, k: int) -> int:
    meta, block = ROW_LAYOUT[quant_type]
    assert k % 256 == 0
    return meta + k // 256 * block


def _iq2s_grid(device) -> torch.Tensor:
    global _IQ2S_GRID
    if _IQ2S_GRID is None:
        from gguf.quants import IQ2_S

        IQ2_S.init_grid()
        _IQ2S_GRID = torch.tensor(IQ2_S.grid.reshape(1024, 8), dtype=torch.float64)
    return _IQ2S_GRID.to(device)


def _f16(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float16).squeeze(-1).to(torch.float64)


def _f32(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float32).squeeze(-1).to(torch.float64)


def _dequant_iq2_s(blocks: torch.Tensor) -> torch.Tensor:
    # blocks [R, nb, 82]: d f16 | qs[32] grid low bits | signs[32] | qh[8] | scales[8]
    d = _f16(blocks[..., 0:2])[..., None, None, None]
    qs = blocks[..., 2:34].long().view(*blocks.shape[:-1], 8, 4)
    signs = blocks[..., 34:66].long().view(*blocks.shape[:-1], 8, 4)
    qh = blocks[..., 66:74].long()[..., None]
    sc = blocks[..., 74:82].long()
    shifts = torch.tensor([8, 6, 4, 2], device=blocks.device)
    idx = qs | ((qh << shifts) & 0x300)
    grid = _iq2s_grid(blocks.device)[idx]  # [R, nb, 8, 4, 8]
    bits = (signs[..., None] >> torch.arange(8, device=blocks.device)) & 1
    ls = torch.stack([sc & 0xF, sc >> 4], dim=-1)  # [R, nb, 8, 2]
    ls = ls.repeat_interleave(2, dim=-1).to(torch.float64)  # l = 0,1 low; 2,3 high
    db = d.squeeze(-1) * (0.5 + ls) * 0.25  # [R, nb, 8, 4]
    vals = grid * (1 - 2 * bits).to(torch.float64) * db[..., None]
    return vals.reshape(blocks.shape[0], -1)


def _dequant_iq4_ks(rows: torch.Tensor, k: int) -> torch.Tensor:
    d = _f32(rows[:, 0:4])[:, None, None, None]
    blocks = rows[:, 4:].view(rows.shape[0], k // 256, 136)
    sc = blocks[..., 0:8].long()  # [R, nb, 8]
    qs = blocks[..., 8:].long().view(rows.shape[0], k // 256, 8, 16)
    q = torch.cat([qs & 0xF, qs >> 4], dim=-1)  # [R, nb, 8, 32]
    table = torch.tensor(IQ4K_VALUES, dtype=torch.float64, device=rows.device)
    vals = table[q + ((sc & 1) << 4)[..., None]]
    ls = ((sc & 254) - 127).to(torch.float64)[..., None]
    return (d * ls * vals).reshape(rows.shape[0], k)


def _dequant_iq2_ks(rows: torch.Tensor, k: int) -> torch.Tensor:
    d = _f16(rows[:, 0:2])[:, None, None, None]
    blocks = rows[:, 2:].view(rows.shape[0], k // 256, 70)
    extra = (
        blocks[..., 0:2].contiguous().view(torch.int16).long().squeeze(-1) & 0xFFFF
    )  # [R, nb]
    sc = blocks[..., 2:6].long()  # [R, nb, 4]
    qs = blocks[..., 6:].long().view(rows.shape[0], k // 256, 2, 32)
    sb = torch.arange(8, device=rows.device)
    half = qs[..., sb // 4, :]  # [R, nb, 8, 32]
    q = (half >> (2 * (sb % 4))[:, None]) & 3
    hi = (extra[..., None] >> sb) & 1  # [R, nb, 8]
    table = torch.tensor(IQ2NL_VALUES, dtype=torch.float64, device=rows.device)
    vals = table[q + 4 * hi[..., None]]
    ls = ((sc[..., sb // 2] >> (4 * (sb % 2))) & 0xF) | (
        ((extra[..., None] >> (8 + sb)) & 1) << 4
    )
    scale = (ls - 16).to(torch.float64)[..., None]
    return (d * scale * vals).reshape(rows.shape[0], k)


def dequantize(packed: torch.Tensor, quant_type: int, k: int) -> torch.Tensor:
    """packed uint8 [..., rows, row_bytes] -> float64 [..., rows, k]."""
    lead = packed.shape[:-1]
    rows = packed.reshape(-1, packed.shape[-1])
    assert rows.shape[1] == row_bytes(quant_type, k)
    if quant_type == 22:
        out = _dequant_iq2_s(rows.view(rows.shape[0], k // 256, 82))
    elif quant_type == 144:
        out = _dequant_iq4_ks(rows, k)
    elif quant_type == 145:
        out = _dequant_iq2_ks(rows, k)
    else:
        raise ValueError(quant_type)
    return out.reshape(*lead, k)


def random_rows(
    quant_type: int, shape: tuple, k: int, generator: torch.Generator, device="cuda"
) -> torch.Tensor:
    """Valid packed rows uint8 [*shape, row_bytes] with finite positive scales."""
    rb = row_bytes(quant_type, k)
    packed = torch.randint(
        0, 256, (*shape, rb), dtype=torch.uint8, device=device, generator=generator
    )
    flat = packed.view(-1, rb)
    n = flat.shape[0]

    def scales(count, lo, hi):
        u = torch.rand(count, generator=generator, device=device, dtype=torch.float32)
        return lo + (hi - lo) * u

    if quant_type == 22:
        nb = k // 256
        d = (
            scales(n * nb, 2e-3, 6e-3)
            .to(torch.float16)
            .view(torch.uint8)
            .view(n, nb, 2)
        )
        flat.view(n, nb, 82)[..., 0:2] = d
    elif quant_type == 144:
        flat[:, 0:4] = scales(n, 2e-5, 6e-5).view(torch.uint8).view(n, 4)
    elif quant_type == 145:
        flat[:, 0:2] = (
            scales(n, 1e-3, 3e-3).to(torch.float16).view(torch.uint8).view(n, 2)
        )
    else:
        raise ValueError(quant_type)
    return packed


def nmse(y: torch.Tensor, ref: torch.Tensor) -> float:
    y = y.to(torch.float64)
    ref = ref.to(torch.float64)
    return float(((y - ref) ** 2).sum() / (ref**2).sum().clamp_min(1e-300))
