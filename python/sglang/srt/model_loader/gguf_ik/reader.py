# SPDX-License-Identifier: Apache-2.0
"""A GGUF reader that also knows the ik_llama.cpp quant types.

``IkGGUFReader`` subclasses ``gguf.GGUFReader`` and replaces only the step that
turns tensor infos into tensors. A tensor of a type gguf-py knows is built
exactly as gguf-py builds it (same ``GGMLQuantizationType`` member, same byte
count, same array), so a file of classic types reads as before. A tensor of an
ik type gets a ``GGMLType`` (types.py) as its ``tensor_type`` and is sized with
ik's ``ggml_row_size``, row header included; its ``data`` is uint8 shaped
``[..., rows, row_bytes]``.

The installed ``gguf`` package is not modified.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional

import gguf
import numpy as np
from gguf.constants import GGML_QUANT_SIZES
from gguf.gguf_reader import ReaderTensor
from gguf.quants import quant_shape_to_byte_shape

from sglang.srt.model_loader.gguf_ik.types import (
    TRELLIS_REMAP_KEY,
    GGMLType,
    GGMLTypeError,
    by_id,
    row_size,
)

# Ids gguf-py knows keep gguf-py's meaning, whatever ik's table says; every
# other id is read through the ik table (types.py) or refused by it.
_GGUF_IDS = frozenset(int(m) for m in gguf.GGMLQuantizationType)


class TensorInfo(NamedTuple):
    """What a loader needs to plan a tensor before reading its bytes."""

    name: str
    tensor_type: Any  # gguf.GGMLQuantizationType, or GGMLType for an ik type
    shape: tuple  # ggml order (ne0 first), like ReaderTensor.shape
    n_bytes: int


def resolve_type(raw_id: int):
    """The type object a reader reports for a raw type id.

    gguf-py's own member where gguf-py knows the id, so nothing changes for a
    classic file; otherwise the ik table's ``GGMLType``; otherwise a
    ``GGMLTypeError`` that names the id.
    """
    if int(raw_id) in _GGUF_IDS:
        return gguf.GGMLQuantizationType(int(raw_id))
    return by_id(int(raw_id))


def is_ik_type(t) -> bool:
    return isinstance(t, GGMLType)


class IkGGUFReader(gguf.GGUFReader):
    """``gguf.GGUFReader`` plus the ik types (see the module docstring)."""

    def _build_tensors(self, start_offs: int, fields) -> None:
        if self.fields.get(TRELLIS_REMAP_KEY) is not None:
            raise GGMLTypeError(
                f"this GGUF carries {TRELLIS_REMAP_KEY}: its type ids 144-149 "
                f"are trellis TQ* types renumbered at load by the tool that "
                f"wrote it, not IQ4_KS / IQ2_KS / IQ4_KSS; SGLang does not "
                f"read such files"
            )
        if all(int(f.parts[4][0]) in _GGUF_IDS for f in fields):
            return super()._build_tensors(start_offs, fields)
        tensors: List[ReaderTensor] = []
        names = set()
        for field in fields:
            _nl, name_data, _nd, dims, raw_dtype, offset_tensor = field.parts
            name = str(bytes(name_data), encoding="utf-8")
            if name in names:
                raise ValueError(f"Found duplicated tensor with name {name}")
            names.add(name)
            raw = int(raw_dtype[0])
            data_offs = int(start_offs + offset_tensor[0])
            if raw in _GGUF_IDS:
                tensors.append(self._classic_tensor(field, name, dims, raw, data_offs))
                continue
            try:
                t = by_id(raw)
            except GGMLTypeError as e:
                raise GGMLTypeError(f"{name}: {e}") from None
            tensors.append(self._ik_tensor(field, name, dims, t, data_offs))
        self.tensors = tensors

    def _classic_tensor(self, field, name, dims, raw, data_offs):
        # The same steps as gguf.GGUFReader._build_tensors, for one tensor.
        WT = gguf.GGMLQuantizationType
        ggml_type = WT(raw)
        n_elems = int(np.prod(dims))
        np_dims = tuple(reversed(dims.tolist()))
        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        n_bytes = n_elems * type_size // block_size
        plain = {
            WT.F16: np.float16,
            WT.F32: np.float32,
            WT.F64: np.float64,
            WT.I8: np.int8,
            WT.I16: np.int16,
            WT.I32: np.int32,
            WT.I64: np.int64,
        }
        if ggml_type in plain:
            item_count, item_type = n_elems, plain[ggml_type]
        else:
            item_count, item_type = n_bytes, np.uint8
            np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)
        return ReaderTensor(
            name=name,
            tensor_type=ggml_type,
            shape=dims,
            n_elements=n_elems,
            n_bytes=n_bytes,
            data_offset=data_offs,
            data=self._get(data_offs, item_type, item_count).reshape(np_dims),
            field=field,
        )

    def _ik_tensor(self, field, name, dims, t: GGMLType, data_offs):
        ne = [int(d) for d in dims.tolist()]
        n_elems = int(np.prod(ne))
        try:
            rb = row_size(t, ne[0])
        except GGMLTypeError as e:
            raise GGMLTypeError(f"{name}: {e}") from None
        n_bytes = n_elems // ne[0] * rb
        if data_offs + n_bytes > len(self.data):
            raise GGMLTypeError(
                f"{name}: {t.name} {ne} needs {n_bytes} bytes at offset "
                f"{data_offs}, past the end of the file ({len(self.data)} bytes)"
            )
        np_dims = tuple(reversed(ne[1:])) + (rb,)
        return ReaderTensor(
            name=name,
            tensor_type=t,
            shape=dims,
            n_elements=n_elems,
            n_bytes=n_bytes,
            data_offset=data_offs,
            data=self._get(data_offs, np.uint8, n_bytes).reshape(np_dims),
            field=field,
        )

    def tensor_infos(self) -> List[TensorInfo]:
        return [
            TensorInfo(t.name, t.tensor_type, tuple(int(d) for d in t.shape), t.n_bytes)
            for t in self.tensors
        ]


def field_value(reader, key: str, default: Optional[Any] = None) -> Any:
    f = reader.fields.get(key)
    return default if f is None else f.contents()
