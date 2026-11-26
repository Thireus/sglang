# SPDX-License-Identifier: Apache-2.0
"""GGUF files with ik_llama.cpp quant types, split sets and row-interleaved repacks.

* ``types``  - the ggml / ik type table (ids, blocks, row headers, repacks).
* ``reader`` - ``IkGGUFReader``: gguf-py's reader plus the ik types.
* ``split``  - ``open_gguf``: one file, or a ``-NNNNN-of-MMMMM`` split set.
* ``repack`` - ``undo_repack``: ``_R4`` / ``_R8`` rows back to the base type.
* ``rowgroups`` - ``split_row_groups``: the three types stored as interleaved
  row groups that are not repacks, regrouped into self-contained rows.

``served_type`` / ``served_tensor`` are what the GGUF model loader calls: a
classic tensor passes through untouched, a repack is restored to its base
type, IQ1_S_R4 / IQ1_M_R4 / Q8_K_R8 rows are regrouped, and a type the GGUF
model path cannot serve is refused with a message naming the tensor, the type
and the reason.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from sglang.srt.model_loader.gguf_ik.reader import (
    IkGGUFReader,
    TensorInfo,
    is_ik_type,
    resolve_type,
)
from sglang.srt.model_loader.gguf_ik.repack import undo_repack
from sglang.srt.model_loader.gguf_ik.rowgroups import split_row_groups
from sglang.srt.model_loader.gguf_ik.split import (
    GGUFFile,
    GGUFSplit,
    GGUFSplitError,
    open_gguf,
)
from sglang.srt.model_loader.gguf_ik.types import (
    IK_SOURCE,
    QK_K,
    REPACKS,
    ROW_GROUPS,
    GGMLType,
    GGMLTypeError,
    served_today,
    why_not_served,
)

__all__ = [
    "GGMLType",
    "GGMLTypeError",
    "GGUFFile",
    "GGUFSplit",
    "GGUFSplitError",
    "IkGGUFReader",
    "TensorInfo",
    "open_gguf",
    "served_tensor",
    "served_type",
    "split_row_groups",
    "undo_repack",
]

_TRELLIS = ("IQ1_KT", "IQ2_KT", "IQ3_KT", "IQ4_KT")


def _refuse(name: str, t: GGMLType) -> GGMLTypeError:
    return GGMLTypeError(
        f"GGUF tensor {name}: type {t.name} (ggml id {int(t)}, {IK_SOURCE}): "
        f"{why_not_served(t)}."
    )


def served_type(tensor):
    """The type the GGUF model path will see for ``tensor`` (a ReaderTensor or TensorInfo).

    Classic types are returned as they are. A repack whose base type is served
    is returned as the base type (gguf-py's member, or the ``GGMLType`` of an
    ik base type). An ik type that is served is returned as it is. Anything
    else raises.
    """
    t = tensor.tensor_type
    if not is_ik_type(t):
        return t
    st = resolve_type(int(t.repack_of[0])) if t.name in REPACKS else t
    if not served_today(st):
        raise _refuse(tensor.name, t)
    ne0 = int(tensor.shape[0])
    if is_ik_type(st) and st.name in _TRELLIS and ne0 % QK_K:
        raise GGMLTypeError(
            f"GGUF tensor {tensor.name}: type {t.name} (ggml id {int(t)}, "
            f"{IK_SOURCE}): a row of {ne0} values ends in the 32-value tail "
            f"blocks of the trellis types, which SGLang's dequantiser does not "
            f"read (it reads whole blocks of {QK_K})."
        )
    return st


def served_tensor(tensor) -> Tuple[object, np.ndarray]:
    """(type, data) of a ReaderTensor as the GGUF model path loads it."""
    t = tensor.tensor_type
    if not is_ik_type(t):
        return t, tensor.data
    st = served_type(tensor)
    if t.name in REPACKS:
        _base, data = undo_repack(t, tensor.data)
        return st, data
    if t.name in ROW_GROUPS:
        return st, split_row_groups(t, tensor.data)
    return st, tensor.data
