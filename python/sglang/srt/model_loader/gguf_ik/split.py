# SPDX-License-Identifier: Apache-2.0
"""GGUF split sets: one model in many ``<name>-NNNNN-of-MMMMM.gguf`` shards.

llama.cpp's ``gguf-split`` writes the keys ``split.no``, ``split.count`` and
``split.tensors.count`` into every shard; the first shard also carries the
model's metadata. The SPECIAL_SPLIT sets of the GGUF Tool Suite go further:
one tensor a shard, hundreds of shards, and a ``tensors.map`` beside them with
one line a tensor::

    <shard file>:<sha256>:<tensor name>:shape=...:dtype=...:elements=...:bytes=...

``GGUFSplit`` opens such a set from its first shard or its directory, finds
every tensor across the shards (from ``tensors.map`` when it is there, else
from the shard headers), and refuses an incomplete set with the names of the
missing shards and, when the map tells, of the tensors they hold. Shards are
opened one at a time while the tensors are read; nothing is merged.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

from sglang.srt.model_loader.gguf_ik.reader import IkGGUFReader, TensorInfo
from sglang.srt.model_loader.gguf_ik.types import BY_NAME, GGMLTypeError

_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")
MAP_NAMES = ("tensors.map",)


class GGUFSplitError(ValueError):
    """A split set that cannot be read as it is on disk; the message says what is wrong."""


class MapRow(NamedTuple):
    shard: str
    sha256: str
    name: str
    shape: Tuple[int, ...]
    dtype: str


def read_tensors_map(path: str) -> List[MapRow]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            parts = line.rstrip("\n").split(":")
            if len(parts) < 4:
                continue
            kv = dict(p.split("=", 1) for p in parts[3:] if "=" in p)
            if "shape" not in kv or "dtype" not in kv:
                raise GGUFSplitError(f"{path}:{n}: a row without shape= or dtype=")
            shape = tuple(int(x) for x in re.findall(r"-?\d+", kv["shape"]))
            rows.append(MapRow(parts[0], parts[1], parts[2], shape, kv["dtype"]))
    return rows


def type_of_map_dtype(dtype: str):
    """A tensors.map dtype (ggml's lower-case type name) -> the reader's type object."""
    import gguf

    up = dtype.upper()
    if up in gguf.GGMLQuantizationType.__members__:
        return gguf.GGMLQuantizationType[up]
    t = BY_NAME.get(up)
    if t is None:
        raise GGMLTypeError(
            f"tensors.map dtype {dtype!r} is not a ggml type SGLang knows"
        )
    return t


def shard_name(prefix: str, no: int, count: int) -> str:
    return f"{prefix}-{no:05d}-of-{count:05d}.gguf"


def find_first_shard(path: str) -> Optional[str]:
    """The first shard of a split, given that shard, any shard, or the directory; None if not a split."""
    if os.path.isdir(path):
        firsts = sorted(
            f
            for f in os.listdir(path)
            if (m := _SHARD_RE.match(f)) and m["no"] == "00001"
        )
        if not firsts:
            return None
        if len(firsts) > 1:
            raise GGUFSplitError(
                f"{path}: holds the first shard of {len(firsts)} split sets "
                f"({', '.join(firsts[:3])}); name the one to load"
            )
        return os.path.join(path, firsts[0])
    m = _SHARD_RE.match(os.path.basename(path))
    if m is None:
        return None
    return os.path.join(
        os.path.dirname(path), shard_name(m["prefix"], 1, int(m["count"]))
    )


class GGUFSplit:
    """A split GGUF set: metadata from the first shard, tensors across all shards."""

    def __init__(self, path: str, use_map: bool = True):
        first = find_first_shard(path)
        if first is None:
            raise GGUFSplitError(
                f"{path} is not a GGUF split shard nor a directory of one"
            )
        self.first_path = first
        self.dir = os.path.dirname(first)
        m = _SHARD_RE.match(os.path.basename(first))
        self.prefix, self.count = m["prefix"], int(m["count"])
        if not os.path.isfile(first):
            raise GGUFSplitError(
                f"{first}: the first shard of this split is missing; it holds the "
                f"model's metadata, so nothing can be read without it"
            )
        self.first = IkGGUFReader(first)
        self.fields = self.first.fields
        declared = self._field(self.first, "split.count")
        if declared is not None and int(declared) != self.count:
            raise GGUFSplitError(
                f"{first}: its name says {self.count} shards but split.count is {int(declared)}"
            )
        tc = self._field(self.first, "split.tensors.count")
        self.tensor_count = None if tc is None else int(tc)
        self.shards = [
            os.path.join(self.dir, shard_name(self.prefix, i, self.count))
            for i in range(1, self.count + 1)
        ]
        self.map_path = None
        self._map: Optional[Dict[str, MapRow]] = None
        if use_map:
            for n in MAP_NAMES:
                p = os.path.join(self.dir, n)
                if os.path.isfile(p):
                    self.map_path = p
                    break
        if self.map_path:
            self._map = self._load_map()

    @staticmethod
    def _field(reader, key):
        f = reader.fields.get(key)
        return None if f is None else f.contents()

    def _load_map(self) -> Dict[str, MapRow]:
        rows = read_tensors_map(self.map_path)
        mine = {os.path.basename(s) for s in self.shards}
        out = {}
        for r in rows:
            if r.shard not in mine:
                # A map of another set (or of a renamed one) next to these shards.
                raise GGUFSplitError(
                    f"{self.map_path} lists {r.shard}, which is not a shard name of "
                    f"{shard_name(self.prefix, 1, self.count)}'s set; the map belongs to another set"
                )
            if r.name in out:
                raise GGUFSplitError(
                    f"{self.map_path}: tensor {r.name} is listed twice"
                )
            out[r.name] = r
        if self.tensor_count is not None and len(out) != self.tensor_count:
            raise GGUFSplitError(
                f"{self.map_path}: lists {len(out)} tensors but the first shard "
                f"declares split.tensors.count = {self.tensor_count}"
            )
        return out

    # ------------------------------------------------------------ completeness
    def missing_shards(self) -> List[str]:
        return [s for s in self.shards if not os.path.isfile(s)]

    def check_complete(self) -> None:
        missing = self.missing_shards()
        if not missing:
            return
        names = [os.path.basename(s) for s in missing]
        msg = (
            f"{self.dir}: {len(missing)} of the {self.count} shards of this split "
            f"are missing: {', '.join(names[:5])}{' ...' if len(names) > 5 else ''}"
        )
        if self._map is not None:
            lost = sorted(r.name for r in self._map.values() if r.shard in set(names))
            msg += (
                f"; they hold {len(lost)} tensor(s), e.g. {', '.join(lost[:5])}"
                if lost
                else "; tensors.map puts no tensor in them"
            )
        raise GGUFSplitError(msg + ". Finish the download before loading this set.")

    # ------------------------------------------------------------- the tensors
    def tensor_infos(self) -> List[TensorInfo]:
        """Every tensor of the set (name, type, ggml shape, bytes), in shard order."""
        self.check_complete()
        if self._map is not None:
            from sglang.srt.model_loader.gguf_ik.types import row_size

            out = []
            for r in sorted(
                self._map.values(),
                key=lambda r: self.shards.index(os.path.join(self.dir, r.shard)),
            ):
                t = type_of_map_dtype(r.dtype)
                n = 1
                for d in r.shape:
                    n *= d
                n_bytes = (
                    n // r.shape[0] * row_size(int(t), r.shape[0]) if r.shape else 0
                )
                out.append(TensorInfo(r.name, t, tuple(r.shape), n_bytes))
            return out
        out, seen = [], {}
        for s in self.shards:
            rd = self.first if s == self.first_path else IkGGUFReader(s)
            for ti in rd.tensor_infos():
                if ti.name in seen:
                    raise GGUFSplitError(
                        f"tensor {ti.name} is in both {seen[ti.name]} and {os.path.basename(s)}"
                    )
                seen[ti.name] = os.path.basename(s)
                out.append(ti)
        self._check_count(len(out))
        return out

    def _check_count(self, n: int) -> None:
        if self.tensor_count is not None and n != self.tensor_count:
            raise GGUFSplitError(
                f"{self.dir}: the shards hold {n} tensors but the first shard "
                f"declares split.tensors.count = {self.tensor_count}"
            )

    def tensors(self) -> Iterator:
        """The ReaderTensors of every shard, one shard open at a time."""
        self.check_complete()
        n = 0
        for i, s in enumerate(self.shards, 1):
            rd = self.first if s == self.first_path else IkGGUFReader(s)
            no = self._field(rd, "split.no")
            if no is not None and int(no) != i - 1:
                raise GGUFSplitError(f"{s}: split.no is {int(no)}, expected {i - 1}")
            for t in rd.tensors:
                if self._map is not None:
                    self._check_against_map(s, t)
                n += 1
                yield t
            del rd
        self._check_count(n)

    def _check_against_map(self, shard: str, t) -> None:
        r = self._map.get(t.name)
        base = os.path.basename(shard)
        if r is None or r.shard != base:
            raise GGUFSplitError(
                f"{base}: holds tensor {t.name}, which tensors.map "
                f"{'does not list' if r is None else 'puts in ' + r.shard}; rebuild the map"
            )
        if (
            tuple(int(d) for d in t.shape) != tuple(r.shape)
            or t.tensor_type.name.lower() != r.dtype.lower()
        ):
            raise GGUFSplitError(
                f"{base}: {t.name} is {t.tensor_type.name} {list(int(d) for d in t.shape)} in the shard "
                f"but {r.dtype} {list(r.shape)} in tensors.map; rebuild the map"
            )

    def read(self, name: str):
        """One tensor by name (opens only its shard when tensors.map is present)."""
        if self._map is not None:
            r = self._map.get(name)
            if r is None:
                raise KeyError(name)
            path = os.path.join(self.dir, r.shard)
            if not os.path.isfile(path):
                raise GGUFSplitError(f"{path}: missing; it holds tensor {name}")
            rd = IkGGUFReader(path)
            for t in rd.tensors:
                if t.name == name:
                    self._check_against_map(path, t)
                    return t
            raise GGUFSplitError(
                f"{path}: tensors.map puts {name} here but the shard does not hold it"
            )
        for t in self.tensors():
            if t.name == name:
                return t
        raise KeyError(name)


class GGUFFile:
    """A single GGUF file behind the same interface as ``GGUFSplit``."""

    def __init__(self, path: str):
        self.first_path = path
        self.first = IkGGUFReader(path)
        self.fields = self.first.fields
        self.shards = [path]

    def tensor_infos(self) -> List[TensorInfo]:
        return self.first.tensor_infos()

    def tensors(self) -> Iterator:
        return iter(self.first.tensors)

    def read(self, name: str):
        for t in self.first.tensors:
            if t.name == name:
                return t
        raise KeyError(name)


def open_gguf(path: str):
    """A ``GGUFSplit`` for a shard (or directory) of a split set, else a ``GGUFFile``."""
    first = find_first_shard(path)
    if first is not None:
        if os.path.isfile(first):
            sp = GGUFSplit(first)
            if sp.count > 1:
                return sp
            return GGUFFile(first)
        return GGUFSplit(first)  # raises: the first shard is missing
    return GGUFFile(path)
