# SPDX-License-Identifier: Apache-2.0
"""GGUF files with ik_llama.cpp quant types, split sets and row-interleaved repacks.

* ``types``  - the ggml / ik type table (ids, blocks, row headers, repacks).
* ``reader`` - ``IkGGUFReader``: gguf-py's reader plus the ik types.
* ``split``  - ``open_gguf``: one file, or a ``-NNNNN-of-MMMMM`` split set.
* ``repack`` - ``undo_repack``: ``_R4`` / ``_R8`` rows back to the base type.

``served_type`` / ``served_tensor`` are what the GGUF model loader calls: a
classic tensor passes through untouched, a repack is restored to its base
type, and a type the GGUF model path has no kernels for is refused with a
message naming the tensor, the type and what is missing.
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
from sglang.srt.model_loader.gguf_ik.split import (
    GGUFFile,
    GGUFSplit,
    GGUFSplitError,
    open_gguf,
)
from sglang.srt.model_loader.gguf_ik.types import (
    IK_SOURCE,
    REPACKS,
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
    "undo_repack",
]


def _refuse(name: str, t: GGMLType) -> GGMLTypeError:
    return GGMLTypeError(
        f"GGUF tensor {name}: type {t.name} (ggml id {int(t)}, {IK_SOURCE}): "
        f"{why_not_served(t)}."
    )


def served_type(tensor):
    """The type the GGUF model path will see for ``tensor`` (a ReaderTensor or TensorInfo).

    Classic types are returned as they are. A repack whose base type is served
    is returned as the base type (gguf-py's member). Anything else raises.
    """
    t = tensor.tensor_type
    if not is_ik_type(t):
        return t
    if t.name in REPACKS:
        base = resolve_type(int(t.repack_of[0]))
        if not is_ik_type(base) and served_today(base):
            return base
    raise _refuse(tensor.name, t)


def served_tensor(tensor) -> Tuple[object, np.ndarray]:
    """(type, data) of a ReaderTensor as the GGUF model path loads it."""
    t = tensor.tensor_type
    if not is_ik_type(t):
        return t, tensor.data
    st = served_type(tensor)
    _base, data = undo_repack(t, tensor.data)
    return st, data
