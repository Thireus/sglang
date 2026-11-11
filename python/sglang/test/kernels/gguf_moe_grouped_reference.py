"""CPU model of the routing the grouped GGUF MoE kernels read: the multi-route vector
kernel (``gguf_moe_vec_multi``: lanes of CTAs walking blocks of routes) and the
wide-block MMQ kernel (``gguf_moe_mmq_wide``: column groups and passes of one block).

``align_reference`` is ``moe_align_block_size(ids, block, E, pad_expert_ids=True)`` in
plain Python (bucket 0 collects the negative ids and comes first with expert -1; an
expert's routes come before its padding; padding is ``numel``). ``lane_walk`` and
``wide_block_plan`` repeat the index arithmetic of the device code, so a CPU test can
state what the kernels must visit: every route with a non-negative id exactly once,
nothing else.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

WIDE_GROUP = 8  # columns per group of the wide-block MMQ kernel
WIDE_PASS = 32  # activation columns gathered per pass


def align_reference(
    ids: Sequence[int], block: int, num_experts: int
) -> Tuple[List[int], List[int], int]:
    """(sorted_ids, expert_ids, num_tokens_post_padded) for flat route ids."""
    numel = len(ids)
    buckets: List[List[int]] = [[] for _ in range(num_experts + 1)]
    for route, expert in enumerate(ids):
        if expert >= num_experts:
            raise ValueError(
                f"route {route}: expert {expert} outside [0, {num_experts})"
            )
        buckets[expert + 1 if expert >= 0 else 0].append(route)
    sorted_ids: List[int] = []
    expert_ids: List[int] = []
    for bucket, routes in enumerate(buckets):
        for start in range(0, len(routes), block):
            chunk = routes[start : start + block]
            sorted_ids += chunk + [numel] * (block - len(chunk))
            expert_ids.append(bucket - 1)
    return sorted_ids, expert_ids, len(sorted_ids)


def max_blocks(numel: int, block: int, num_experts: int) -> int:
    """The static size of ``expert_ids`` (moe_align_block_size's buffer rule)."""
    if numel < num_experts + 1:
        padded = numel * block
    else:
        padded = numel + (num_experts + 1) * (block - 1)
    padded = (padded + 3) // 4 * 4
    return -(-padded // block)


def lane_walk(
    sorted_ids: Sequence[int],
    expert_ids: Sequence[int],
    post_padded: int,
    routes: int,
    lanes: int,
    numel: int,
    num_experts: int,
    capacity: int,
) -> List[Tuple[int, int, int, List[int]]]:
    """(lane, block, expert, routes of the block) for every block the multi-route vector
    kernel works on. The grid has min(lanes, capacity) lanes; lane y walks the blocks
    y, y + lanes, ... below min(post_padded / routes, capacity)."""
    grid_y = min(lanes, capacity)
    num_blocks = min(post_padded // routes, capacity)
    visits = []
    for lane in range(grid_y):
        for block in range(lane, num_blocks, grid_y):
            expert = expert_ids[block]
            if expert < 0 or expert >= num_experts:
                continue
            valid = [
                s
                for s in sorted_ids[block * routes : (block + 1) * routes]
                if 0 <= s < numel
            ]
            visits.append((lane, block, expert, valid))
    return visits


def wide_block_plan(
    sorted_ids: Sequence[int], block: int, wide: int, numel: int
) -> Tuple[int, List[int]]:
    """(groups in use, groups multiplied in each pass) of one block of the wide-block MMQ
    kernel: the groups run up to the last column that holds a route; a pass covers 32
    columns (4 groups), or the whole block when it is narrower."""
    cols = sorted_ids[block * wide : (block + 1) * wide]
    groups = 0
    for g in range(wide // WIDE_GROUP):
        if any(0 <= s < numel for s in cols[g * WIDE_GROUP : (g + 1) * WIDE_GROUP]):
            groups = g + 1
    pass_groups = min(wide, WIDE_PASS) // WIDE_GROUP
    passes = []
    for p in range(wide // min(wide, WIDE_PASS)):
        left = groups - p * pass_groups
        if left > 0:
            passes.append(min(left, pass_groups))
    return groups, passes
