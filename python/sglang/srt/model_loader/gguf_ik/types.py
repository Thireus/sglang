# SPDX-License-Identifier: Apache-2.0
"""The ggml quant types of ik_llama.cpp: ids, block sizes, row headers, repacks.

gguf-py (the ``gguf`` package) knows the mainline ggml types only: no id above
41 and no row header. ik_llama.cpp adds about seventy types, and several of
them start every row with a few bytes of scales (the "row header", ggml's
``row_meta_size``) before the blocks. A GGUF file of such a type cannot be
sized, let alone read, with ``GGML_QUANT_SIZES``.

This table is the one SGLang uses for those files. ``_ROWS`` is copied from the
ik_llama.cpp source named in ``IK_SOURCE`` (``ggml.h`` enum ggml_type,
``ggml.c`` type_traits, ``ggml-common.h`` block sizes); the unit test
``test_gguf_ik_types.py`` parses that source again when it is available and
checks every id, block and size, and checks ``row_size`` against
``ggml_row_size`` of ik's libggml.

``REPACKS`` lists the row-interleaved types (``_R4`` / ``_R8``) that hold the
bytes of a base type regrouped by 4 or 8 rows. ik makes them by quantising
with the base type and then repacking, so the base layout can be restored
exactly (``repack.py``). IQ1_S_R4 and IQ1_M_R4 are quantisers of their own and
are types, not repacks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

IK_SOURCE = "ik_llama.cpp 7c1deaf3"

QK_K = 256

# (id, name, values a block, bytes a block, row header bytes)
_ROWS: Tuple[Tuple[int, str, int, int, int], ...] = (
    (0, "F32", 1, 4, 0),
    (1, "F16", 1, 2, 0),
    (2, "Q4_0", 32, 18, 0),
    (3, "Q4_1", 32, 20, 0),
    (6, "Q5_0", 32, 22, 0),
    (7, "Q5_1", 32, 24, 0),
    (8, "Q8_0", 32, 34, 0),
    (9, "Q8_1", 32, 36, 0),
    (10, "Q2_K", 256, 84, 0),
    (11, "Q3_K", 256, 110, 0),
    (12, "Q4_K", 256, 144, 0),
    (13, "Q5_K", 256, 176, 0),
    (14, "Q6_K", 256, 210, 0),
    (15, "Q8_K", 256, 296, 0),
    (16, "IQ2_XXS", 256, 66, 0),
    (17, "IQ2_XS", 256, 74, 0),
    (18, "IQ3_XXS", 256, 98, 0),
    (19, "IQ1_S", 256, 50, 0),
    (20, "IQ4_NL", 32, 18, 0),
    (21, "IQ3_S", 256, 110, 0),
    (22, "IQ2_S", 256, 82, 0),
    (23, "IQ4_XS", 256, 136, 0),
    (24, "I8", 1, 1, 0),
    (25, "I16", 1, 2, 0),
    (26, "I32", 1, 4, 0),
    (27, "I64", 1, 8, 0),
    (28, "F64", 1, 8, 0),
    (29, "IQ1_M", 256, 56, 0),
    (30, "BF16", 1, 2, 0),
    (31, "Q4_0_4_4", 32, 18, 0),
    (32, "Q4_0_4_8", 32, 18, 0),
    (33, "Q4_0_8_8", 32, 18, 0),
    (36, "I2_S", 1, 1, 0),
    (39, "MXFP4", 32, 17, 0),
    (41, "Q1_0_G128", 128, 18, 0),
    (42, "Q2_0", 64, 18, 0),
    (97, "Q8_0_X4", 32, 34, 0),
    (98, "Q8_1_X4", 32, 36, 0),
    (99, "Q8_2_X4", 32, 36, 0),
    (133, "Q6_0", 32, 26, 0),
    (134, "IQ1_BN", 64, 13, 2),
    (135, "IQ2_BN", 64, 16, 4),
    (136, "Q8_K64", 64, 68, 0),
    (137, "IQ2_K", 256, 76, 0),
    (138, "IQ3_K", 256, 110, 0),
    (139, "IQ4_K", 256, 144, 0),
    (140, "IQ5_K", 256, 176, 0),
    (141, "IQ6_K", 256, 212, 0),
    (142, "PQ2_0", 128, 34, 0),
    (143, "PTQ1_0", 128, 28, 0),
    (144, "IQ4_KS", 256, 136, 4),
    (145, "IQ2_KS", 256, 70, 2),
    (146, "IQ4_KSS", 256, 128, 4),
    (147, "Q8_K16", 64, 64, 20),
    (148, "Q8_K32", 256, 296, 0),
    (149, "Q8_KR8", 256, 296, 0),
    (150, "Q8_K128", 128, 144, 0),
    (151, "Q8_KV", 32, 32, 8),
    (152, "IQ5_KS", 256, 168, 4),
    (153, "IQ2_KT", 256, 68, 4),
    (154, "IQ3_KT", 256, 100, 4),
    (155, "IQ4_KT", 256, 128, 4),
    (156, "IQ3_KS", 256, 102, 2),
    (157, "IQ2_KL", 256, 86, 2),
    (158, "IQ1_KT", 256, 56, 4),
    (159, "Q1_0_G128_R8", 128, 18, 0),
    (160, "PQ2_0_R8", 128, 34, 0),
    (161, "PTQ1_0_R8", 128, 28, 0),
    (162, "TQK6", 128, 26, 0),
    (163, "TQK7", 128, 30, 0),
    (164, "TQ2_T", 128, 34, 0),
    (165, "TQP5", 128, 42, 0),
    (166, "TQP6", 128, 50, 0),
    (167, "TQP7", 128, 58, 0),
    (202, "Q4_0_R8", 32, 18, 0),
    (206, "Q5_0_R4", 32, 22, 0),
    (208, "Q8_0_R8", 32, 34, 0),
    (210, "Q2_K_R4", 256, 84, 0),
    (211, "Q3_K_R4", 256, 110, 0),
    (212, "Q4_K_R4", 256, 144, 0),
    (213, "Q5_K_R4", 256, 176, 0),
    (214, "Q6_K_R4", 256, 210, 0),
    (216, "IQ2_XXS_R4", 256, 66, 0),
    (217, "IQ2_XS_R4", 256, 74, 0),
    (218, "IQ3_XXS_R4", 256, 98, 0),
    (219, "IQ1_S_R4", 32, 6, 2),
    (220, "IQ4_NL_R4", 32, 18, 0),
    (221, "IQ3_S_R4", 256, 110, 0),
    (222, "IQ2_S_R4", 256, 82, 0),
    (223, "IQ4_XS_R8", 256, 136, 0),
    (229, "IQ1_M_R4", 32, 7, 2),
    (230, "BF16_R16", 1, 2, 0),
    (233, "Q6_0_R4", 32, 26, 0),
    (335, "IQ2_BN_R4", 64, 16, 4),
    (337, "IQ2_K_R4", 256, 76, 0),
    (338, "IQ3_K_R4", 256, 110, 0),
    (339, "IQ4_K_R4", 256, 144, 0),
    (340, "IQ5_K_R4", 256, 176, 0),
    (344, "IQ4_KS_R4", 256, 136, 4),
    (345, "IQ4_KS_R16", 32, 17, 4),
    (346, "IQ3_KS_R16", 32, 13, 4),
    (347, "IQ5_KS_R16", 32, 21, 4),
    (352, "IQ5_KS_R4", 256, 168, 4),
    (353, "MXFP4_R8", 32, 17, 0),
    (354, "Q2_0_R8", 64, 18, 0),
    (396, "Q8_0_R16", 32, 36, 0),
    (397, "Q8_K_R16", 256, 260, 0),
    (398, "Q8_KV_R8", 32, 32, 4),
    (399, "Q8_K_R8", 256, 258, 0),
)

# Row-interleaved types that are a base type's bytes regrouped by ROWS rows:
# name -> (base name, rows). The pairs and row counts are ik's own run-time
# repack table (iqk_quantize.cpp get_repack_info); repack.py restores the base.
REPACKS: Dict[str, Tuple[str, int]] = {
    "Q4_0_R8": ("Q4_0", 8),
    "Q5_0_R4": ("Q5_0", 4),
    "Q6_0_R4": ("Q6_0", 4),
    "Q8_0_R8": ("Q8_0", 8),
    "Q2_K_R4": ("Q2_K", 4),
    "Q3_K_R4": ("Q3_K", 4),
    "Q4_K_R4": ("Q4_K", 4),
    "Q5_K_R4": ("Q5_K", 4),
    "Q6_K_R4": ("Q6_K", 4),
    "IQ2_XXS_R4": ("IQ2_XXS", 4),
    "IQ2_XS_R4": ("IQ2_XS", 4),
    "IQ2_S_R4": ("IQ2_S", 4),
    "IQ3_XXS_R4": ("IQ3_XXS", 4),
    "IQ3_S_R4": ("IQ3_S", 4),
    "IQ4_NL_R4": ("IQ4_NL", 4),
    "IQ4_XS_R8": ("IQ4_XS", 8),
    "IQ2_K_R4": ("IQ2_K", 4),
    "IQ3_K_R4": ("IQ3_K", 4),
    "IQ4_K_R4": ("IQ4_K", 4),
    "IQ5_K_R4": ("IQ5_K", 4),
    "IQ4_KS_R4": ("IQ4_KS", 4),
    "IQ5_KS_R4": ("IQ5_KS", 4),
    "IQ2_BN_R4": ("IQ2_BN", 4),
}

# Types that look like repacks by name but are not a byte-exact regrouping of
# a base type, with the reason a loader gives when it meets one.
NOT_REPACKS: Dict[str, str] = {
    "IQ1_S_R4": "is a quantiser of its own (block of 32, an fp16 scale a row), "
    "not IQ1_S regrouped by rows, so it needs kernels of its own",
    "IQ1_M_R4": "is a quantiser of its own (block of 32, an fp16 scale a row), "
    "not IQ1_M regrouped by rows, so it needs kernels of its own",
    "Q8_K_R8": "stores each block scale as fp16 where Q8_K has fp32 and drops "
    "the block sums, so no Q8_K bytes can be restored from it; it needs a "
    "dequantiser of its own",
}

# Ids a reader cannot size without knowing which ik_llama.cpp a file came from.
AMBIGUOUS_IDS: Dict[int, str] = {
    142: "PQ2_0 in ik_llama.cpp 7c1deaf3 but IQ2_TN in older trees (a0f1047e)",
    143: "PTQ1_0 in ik_llama.cpp 7c1deaf3 but IQ1_TN in older trees (a0f1047e)",
    36: "I2_S keeps one scale for the whole tensor after its data, which a "
    "per-row reader cannot place",
}

# A GGUF metadata key whose presence means ids 144-149 are the trellis TQ*
# types, renumbered by ik at load (ggml.c gguf_init); SGLang refuses the file.
TRELLIS_REMAP_KEY = "agention.trellis.state_bits"

# Types the GGUF model path serves today (layers/quantization/gguf.py:
# UNQUANTIZED_TYPES | STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES).
GGUF_SERVED_IDS = frozenset(
    {0, 1, 30, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    | {16, 17, 18, 19, 20, 21, 22, 23, 29}
)

# Types the hot/cold cold tier serves (hotcold_manifest.COLD_GGML_TYPES).
HOTCOLD_COLD_IDS = frozenset({22, 144, 145})

_KT_TAIL = {"IQ1_KT", "IQ2_KT", "IQ3_KT", "IQ4_KT"}


class GGMLType(int):
    """One ggml type: an int (its id) that also carries the table's facts.

    It stands in for a ``gguf.GGMLQuantizationType`` member where gguf-py has
    none: ``int(t)``, ``t.name`` and ``torch.tensor(t)`` work the same way.
    """

    name: str
    blck_size: int
    type_size: int
    row_meta: int

    def __new__(cls, tid: int, name: str, blck: int, size: int, meta: int):
        obj = super().__new__(cls, tid)
        obj.name = name
        obj.blck_size = blck
        obj.type_size = size
        obj.row_meta = meta
        return obj

    @property
    def value(self) -> int:
        return int(self)

    @property
    def bits_per_weight(self) -> float:
        return self.type_size * 8 / self.blck_size

    @property
    def repack_of(self) -> Optional[Tuple[GGMLType, int]]:
        """(base type, rows) when this type is a row-interleaved base type."""
        r = REPACKS.get(self.name)
        return None if r is None else (BY_NAME[r[0]], r[1])

    def __repr__(self) -> str:
        return f"<GGMLType {self.name}: {int(self)}>"

    def __reduce__(self):
        return (by_id, (int(self),))


BY_ID: Dict[int, GGMLType] = {r[0]: GGMLType(*r) for r in _ROWS}
BY_NAME: Dict[str, GGMLType] = {t.name: t for t in BY_ID.values()}


class GGMLTypeError(ValueError):
    """A ggml type id SGLang cannot read or serve; the message says why."""


def by_id(tid: int) -> GGMLType:
    tid = int(tid)
    if tid in AMBIGUOUS_IDS:
        raise GGMLTypeError(
            f"ggml type id {tid} is {AMBIGUOUS_IDS[tid]}; refusing rather than "
            f"guessing the layout"
        )
    t = BY_ID.get(tid)
    if t is None:
        raise GGMLTypeError(
            f"ggml type id {tid} is unknown to gguf-py and to the type table "
            f"SGLang carries ({IK_SOURCE})"
        )
    return t


def by_name(name: str) -> GGMLType:
    t = BY_NAME.get(name.upper())
    if t is None:
        raise GGMLTypeError(
            f"ggml type {name!r} is not in the type table SGLang carries ({IK_SOURCE})"
        )
    return t


def _pad(x: int, n: int) -> int:
    return (x + n - 1) // n * n


def row_blck_size(t: GGMLType) -> int:
    """The unit a row length must be a multiple of (ik's ggml_row_blck_size)."""
    if t.name in _KT_TAIL or t.name in ("Q8_K", "Q8_K_R16"):
        return 32
    return t.blck_size


def row_size(t, ne0: int) -> int:
    """Bytes of one row of ``ne0`` values, row header included (ggml_row_size)."""
    if not isinstance(t, GGMLType):
        t = by_id(int(t))
    if ne0 % row_blck_size(t):
        raise GGMLTypeError(
            f"{t.name}: a row of {ne0} values is not a multiple of the block "
            f"of {row_blck_size(t)}"
        )
    if t.name in ("Q8_K", "Q8_K_R16"):
        nb256, ntail = ne0 // QK_K, (ne0 % QK_K) // 32
        if t.name == "Q8_K":
            return nb256 * t.type_size + ntail * (2 * 4 + 2 * 2 + 32)
        return nb256 * t.type_size + (0 if ntail == 0 else 4 + 32 * ntail)
    if t.name == "IQ3_KS_R16":
        return _pad(64 + (ne0 // 32) * 204, 16) // 16
    if t.name in _KT_TAIL:
        nt = (ne0 % QK_K) // 32
        if t.name == "IQ1_KT":
            tail = 8 + 4 * nt + min(4 * nt, 16) if nt > 0 else 0
        elif t.name == "IQ2_KT":
            tail = 4 + 8 * nt if nt > 0 else 0
        elif t.name == "IQ3_KT":
            tail = (nt + 1) // 2 + 12 * nt
        else:
            tail = 16 * nt
        return _pad(t.row_meta + t.type_size * (ne0 // QK_K) + tail, 4)
    return t.row_meta + t.type_size * ne0 // t.blck_size


def served_today(t) -> bool:
    """Whether SGLang's GGUF model path has kernels for this (base) type."""
    return int(t) in GGUF_SERVED_IDS


def why_not_served(t) -> str:
    """The reason the GGUF model path cannot serve type ``t`` yet."""
    if not isinstance(t, GGMLType):
        t = by_id(int(t))
    if t.name in NOT_REPACKS:
        return f"{t.name} {NOT_REPACKS[t.name]}"
    if t.name in REPACKS:
        base = BY_NAME[REPACKS[t.name][0]]
        return f"{t.name} is {base.name} regrouped by rows, and " + why_not_served(base)
    if int(t) in HOTCOLD_COLD_IDS:
        return (
            f"{t.name} has no dense GGUF kernel yet: SGLang serves it only as "
            f"the cold tier of a hot/cold checkpoint"
        )
    return (
        f"{t.name} can be read, but SGLang has no dequantise or matmul kernel "
        f"for it in the GGUF model path yet"
    )
