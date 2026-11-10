#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""zero_sign_repair.py - give every weight whose source is an exact zero a sign that keeps the mean.

  zero_sign_repair.py hotcold --ckpt DIR (--out NEWDIR | --in-place) [--source DIR] [--layers all|22|0-3]
                              [--rule auto|source|alternate] [--link hard|copy|none] [--dry-run]
  zero_sign_repair.py gguf    --quant FILE [FILE ...] --source FILE [FILE ...] (--out NEWDIR | --in-place)
                              [--tensors REGEX] [--rule ...] [--dry-run]

What it changes: only sign bits, and only at the weights whose source value is exactly 0. Grid indices,
scales and every other byte stay as they are, so the NMSE of every matrix is unchanged.

Rules (--rule):
  source     the zero's own sign in the source: E2M1 code 8 (-0) of MXFP4 / NVFP4, or IEEE -0.0.
  alternate  for sources with +0 only: along each row the zeros take turns +, -, +, - ..., and the first
             sign alternates from row to row. Deterministic.
  auto       (default) source when 25-75 % of the source's zeros are -0 (native MXFP4 / NVFP4
             checkpoints and floats dequantised from them), alternate otherwise.

Types: IQ2_S and IQ3_S (one sign bit per weight). Refused, with the reason: IQ2_XXS, IQ2_XS, IQ3_XXS
(parity-signed), IQ2_KT, IQ3_KT, IQ4_KT, IQ1_KT (trellis), anything else (no known sign field).

Safety: a file is rewritten as <file>.zsrepair.partial and renamed over the old name only after a
verification pass has re-read it (|offset| <= --max-offset for every matrix, NMSE unchanged, no byte outside
the sign fields changed). A hard link elsewhere to the old file keeps the old bytes. Every matrix goes into
the journal (--journal, default <destination>/zero-sign-repair.jsonl). Run it again after an interruption:
finished files are already repaired and the repair is idempotent.
Exit code: 0 done and verified, 1 something left unrepaired or failed verification, 2 refused / bad input.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import zero_sign as zs  # noqa: E402
from zero_sign_check import (  # noqa: E402
    chunk_rows,
    gguf_check_tensor,
    gguf_pairs,
    parse_layers,
)

PARTIAL = ".zsrepair.partial"


class Journal:
    def __init__(self, path, dry):
        self.path, self.dry = path, dry
        self.f = None if dry else open(path, "a")

    def write(self, rec):
        if self.f:
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self):
        if self.f:
            self.f.close()


def stage_copy(src_file, dst_file):
    """Copy src_file to dst_file + PARTIAL and return that path."""
    tmp = dst_file + PARTIAL
    if os.path.lexists(tmp):
        os.unlink(tmp)
    shutil.copyfile(src_file, tmp)
    return tmp


def commit(tmp, dst_file):
    fd = os.open(tmp, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    os.replace(tmp, dst_file)


def link_tree(src_root, dst_root, skip, how):
    """Every file of src_root not in skip, into dst_root: hard links, copies, or nothing."""
    n = 0
    if how == "none":
        return 0
    for d, dirs, files in os.walk(src_root):
        rel = os.path.relpath(d, src_root)
        os.makedirs(os.path.join(dst_root, rel), exist_ok=True)
        for fn in files:
            r = os.path.normpath(os.path.join(rel, fn))
            if r in skip or fn.endswith(PARTIAL) or fn.endswith(".partial"):
                continue
            s, t = os.path.join(d, fn), os.path.join(dst_root, r)
            if os.path.lexists(t):
                continue
            if os.path.islink(s):
                os.symlink(os.readlink(s), t)
            elif how == "hard":
                try:
                    os.link(s, t)
                except OSError as e:
                    raise zs.ZeroSignError(
                        f"cannot hard-link {s} into {dst_root} ({e}); use --link copy or none"
                    )
            else:
                shutil.copy2(s, t)
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ hotcold
_W = {}


def _init(lib, source):
    _W["g"] = zs.GGML(lib)
    _W["src"] = zs.SafetensorsDir(source)
    _W["fds"] = {}


def _fd(path):
    if path not in _W["fds"]:
        _W["fds"][path] = os.open(path, os.O_WRONLY)
    return _W["fds"][path]


def _repair(args):
    u, rule, max_offset, dst = args
    return zs.repair_unit(
        u, _W["src"], _W["g"], rule, None if dst is None else _fd(dst), max_offset
    )


def _check(args):
    u, max_offset = args
    return zs.check_unit(u, _W["src"], _W["g"], max_offset)


def pool_map(fn, items, lib, source, workers):
    if workers <= 1:
        _init(lib, source)
        try:
            return [fn(i) for i in items]
        finally:
            for fd in _W["fds"].values():
                os.close(fd)
            _W["fds"].clear()
    with mp.get_context("fork").Pool(
        workers, initializer=_init, initargs=(lib, source), maxtasksperchild=100
    ) as pool:
        return list(pool.imap(fn, items, chunksize=1))


def cmd_hotcold(a):
    ck = zs.HotColdCheckpoint(a.ckpt)
    source = a.source or ck.manifest.get("source")
    if not source or not os.path.isdir(source):
        raise zs.ZeroSignError("the source checkpoint is not found: pass --source DIR")
    source = os.path.abspath(source)
    lib = zs.find_libggml(a.lib)
    layers = parse_layers(a.layers, ck.layers)
    for w, t in ck.types.items():
        ok, why = zs.repair_support(t)
        if not ok:
            raise zs.ZeroSignError(f"refused: cold {w} is {why}")
    dest = ck.root if a.in_place else os.path.abspath(a.out)
    if not a.in_place and dest == ck.root:
        raise zs.ZeroSignError("--out must differ from --ckpt (or use --in-place)")
    if (
        not a.in_place
        and os.path.exists(os.path.join(dest, zs.HOTCOLD_MANIFEST))
        and not a.dry_run
    ):
        raise zs.ZeroSignError(
            f"{dest} already holds a {zs.HOTCOLD_MANIFEST}; pick an empty directory"
        )
    plan = []
    for L in layers:
        fn = ck.cold_file(L)
        units = ck.units(L, source_pattern=a.source_pattern)
        plan.append((L, fn, units))
    n_units = sum(len(p[2]) for p in plan)
    n_bytes = sum(os.path.getsize(os.path.join(ck.root, p[1])) for p in plan)
    print(
        f"checkpoint  {ck.root}\nsource      {source}\nlibggml     {lib}\ndestination {dest}"
        f"{' (in place)' if a.in_place else ''}\nrule        {a.rule}\n"
        f"{len(plan)} cold files, {n_units} matrices, {n_bytes:,} bytes to rewrite; types "
        f"{sorted({zs.type_name(t) for t in ck.types.values()})}",
        flush=True,
    )
    if a.dry_run:
        for L, fn, units in plan:
            p = os.path.join(ck.root, fn)
            links = os.stat(p).st_nlink
            print(
                f"  layer {L}: {fn} ({len(units)} matrices){'  note: ' + str(links) + ' hard links, the others keep the old bytes' if links > 1 else ''}"
            )
        print("dry run: nothing written")
        return 0
    os.makedirs(dest, exist_ok=True)
    journal = Journal(a.journal or os.path.join(dest, "zero-sign-repair.jsonl"), False)
    man = json.loads(json.dumps(ck.manifest))
    t0, failed, rules = time.time(), 0, set()
    for L, fn, units in plan:
        src_file, dst_file = os.path.join(ck.root, fn), os.path.join(dest, fn)
        tmp = stage_copy(src_file, dst_file)
        recs = pool_map(
            _repair,
            [(u, a.rule, a.max_offset, tmp) for u in units],
            lib,
            source,
            a.workers,
        )
        # verification pass: re-read the staged file
        vunits = ck.units(L, path=tmp, source_pattern=a.source_pattern)
        checks = {
            r["key"]: r
            for r in pool_map(
                _check, [(u, a.max_offset) for u in vunits], lib, source, a.workers
            )
        }
        bad = []
        for r in recs:
            c = checks[r["key"]]
            r["verified_offset"] = c["offset"]
            r["verified"] = c["verdict"] == "PASS" and c["nmse"] == r["nmse"]
            rules.add(r["rule"])
            journal.write(dict(r, file=fn))
            if not r["verified"]:
                bad.append(r["key"])
        if bad:
            failed += len(bad)
            os.unlink(tmp)
            print(
                f"layer {L}: verification FAILED on {len(bad)} matrices (first {bad[0]}); {fn} left unchanged",
                flush=True,
            )
            continue
        tens = ck.tensors(L, path=tmp)
        shas = {
            ck.tensor_name(L, w): zs.sha256_range(tmp, off, ln)
            for w, (off, ln, _) in tens.items()
        }
        file_sha = zs.sha256_file(tmp)
        commit(tmp, dst_file)
        man["cold_tensor_sha256"].update(shas)
        man["cold_files"][str(L)].update(
            sha256=file_sha, bytes=os.path.getsize(dst_file)
        )
        offs = [r["offset_before"] for r in recs], [r["offset_after"] for r in recs]
        journal.write(
            {
                "layer": L,
                "file": fn,
                "sha256": file_sha,
                "tensor_sha256": shas,
                "matrices": len(recs),
                "bits_set": sum(r["bits_set"] for r in recs),
                "bits_cleared": sum(r["bits_cleared"] for r in recs),
            }
        )
        print(
            f"layer {L}: {fn} repaired and verified; {len(recs)} matrices, offset {min(offs[0]):+.5f}..{max(offs[0]):+.5f} "
            f"-> {min(offs[1]):+.5f}..{max(offs[1]):+.5f}; sign bits set {sum(r['bits_set'] for r in recs):,}, "
            f"cleared {sum(r['bits_cleared'] for r in recs):,}; sha256 {file_sha[:16]} ({time.time() - t0:.0f} s)",
            flush=True,
        )
    journal.close()
    if failed:
        print(
            f"RESULT: FAIL ({failed} matrices did not verify); hotcold.json not updated"
        )
        return 1
    man["cold_quant"]["zero_sign_repair"] = {
        "tool": "zero_sign_repair.py",
        "rule": sorted(rules),
        "layers": [p[0] for p in plan],
        "max_offset": a.max_offset,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not a.in_place:
        n = link_tree(
            ck.root,
            dest,
            {zs.HOTCOLD_MANIFEST, os.path.basename(journal.path)}
            | {p[1] for p in plan},
            a.link,
        )
        print(
            f"{n} other files {'hard-linked' if a.link == 'hard' else 'copied' if a.link == 'copy' else 'left out'} "
            f"from {ck.root}"
        )
    zs.write_new_file(
        os.path.join(dest, zs.HOTCOLD_MANIFEST), json.dumps(man, indent=1).encode()
    )
    print(f"hotcold.json written with the new sha256 of {3 * len(plan)} cold tensors")
    print("RESULT: PASS (run zero_sign_check.py hotcold on the destination to see it)")
    return 0


# ------------------------------------------------------------------------------------------------ gguf
def gguf_rule(st, rule, g):
    """The rule for a whole tensor: auto counts all of the source's +0 and -0 once."""
    if rule != "auto":
        return rule
    p = n = 0
    for r0, r1 in chunk_rows(st):
        _, pos0, neg0, _ = zs.read_gguf_source_rows(st, r0, r1, g)
        p += int(pos0.sum())
        n += int(neg0.sum())
    return zs.choose_rule("auto", np.array([p]), np.array([n]))


def gguf_repair_tensor(qt, st, g, rule, fd, max_offset):
    rb = g.row_size(qt.ggml_type, qt.k)
    hdr = g.row_header(qt.ggml_type)
    before, after = zs.Stats(), zs.Stats()
    n_set = n_clr = 0
    for r0, r1 in chunk_rows(qt):
        n = r1 - r0
        x, pos0, neg0, _ = zs.read_gguf_source_rows(st, r0, r1, g)
        raw = np.frombuffer(zs.pread(qt.path, qt.offset + r0 * rb, n * rb), np.uint8)
        blob = raw.copy()
        before.add(x, g.dequant(qt.ggml_type, raw, n, qt.k), pos0, neg0)
        s, c = zs.apply_zero_signs(
            blob,
            qt.ggml_type,
            n,
            qt.k,
            pos0 | neg0,
            zs.target_negative(rule, pos0, neg0, r0),
            rb,
            hdr,
        )
        outside = ~zs.sign_field_mask(qt.ggml_type, n, rb, hdr)
        if not np.array_equal(blob[outside], raw[outside]):
            raise zs.ZeroSignError(f"{qt.name}: a byte outside the sign field changed")
        after.add(x, g.dequant(qt.ggml_type, blob, n, qt.k), pos0, neg0)
        n_set, n_clr = n_set + s, n_clr + c
        if fd is not None:
            zs.pwrite_all(fd, blob, qt.offset + r0 * rb)
    b, f = before.summary(max_offset), after.summary(max_offset)
    return dict(
        name=qt.name,
        type=zs.type_name(qt.ggml_type),
        source_type=zs.type_name(st.ggml_type),
        rule=rule,
        zeros=f["zeros"],
        bits_set=n_set,
        bits_cleared=n_clr,
        offset_before=b["offset"],
        offset_after=f["offset"],
        rows_pos_before=b["rows_pos"],
        rows_pos_after=f["rows_pos"],
        nmse_before=b["nmse"],
        nmse=f["nmse"],
        verdict=f["verdict"],
    )


def cmd_gguf(a):
    g = zs.GGML(a.lib)
    pairs, skipped = gguf_pairs(a.quant, a.source, a.tensors, g)
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    todo, refused = {}, []
    for qt, st in pairs:
        ok, why = zs.repair_support(qt.ggml_type)
        if ok:
            todo.setdefault(qt.path, []).append((qt, st))
        else:
            refused.append((qt, st, why))
    print(
        f"libggml {g.path}\n{sum(len(v) for v in todo.values())} tensors to repair in {len(todo)} files; "
        f"{len(refused)} quantised tensors of other types (checked, never changed)",
        flush=True,
    )
    if a.dry_run:
        for p, ts in todo.items():
            print(
                f"  {os.path.basename(p)}: "
                + ", ".join(f"{qt.name} ({zs.type_name(qt.ggml_type)})" for qt, _ in ts)
            )
        print("dry run: nothing written")
        return 0
    if not a.in_place:
        os.makedirs(a.out, exist_ok=True)
    dest_dir = None if a.in_place else os.path.abspath(a.out)
    jpath = a.journal or os.path.join(
        dest_dir or os.path.dirname(os.path.abspath(a.quant[0])),
        "zero-sign-repair.jsonl",
    )
    journal = Journal(jpath, False)
    failed = 0
    for path, ts in sorted(todo.items()):
        dst = path if a.in_place else os.path.join(dest_dir, os.path.basename(path))
        if not a.in_place and os.path.abspath(dst) == os.path.abspath(path):
            raise zs.ZeroSignError(
                "--out must differ from the folder of the quantised files (or use --in-place)"
            )
        tmp = stage_copy(path, dst)
        fd = os.open(tmp, os.O_WRONLY)
        recs = []
        try:
            for qt, st in ts:
                recs.append(
                    gguf_repair_tensor(
                        qt, st, g, gguf_rule(st, a.rule, g), fd, a.max_offset
                    )
                )
        finally:
            os.close(fd)
        # verification pass on the staged file
        staged = zs.read_gguf(tmp, g)[1]
        bad = []
        for r, (qt, st) in zip(recs, ts):
            c = gguf_check_tensor(staged[qt.name], st, g, a.max_offset)
            r["verified_offset"] = c["offset"]
            r["verified"] = c["verdict"] == "PASS" and abs(
                c["nmse"] - r["nmse_before"]
            ) <= 1e-12 * max(r["nmse_before"], 1e-30)
            journal.write(dict(r, file=os.path.basename(path)))
            print(
                f"  {qt.name} ({r['type']}, rule {r['rule']}): offset {r['offset_before']:+.5f} -> {c['offset']:+.5f}, "
                f"rows_pos {r['rows_pos_before']:.3f} -> {r['rows_pos_after']:.3f}, NMSE {r['nmse']:.6f} (unchanged), "
                f"sign bits set {r['bits_set']:,} cleared {r['bits_cleared']:,} -> {'verified' if r['verified'] else 'FAILED'}",
                flush=True,
            )
            if not r["verified"]:
                bad.append(qt.name)
        if bad:
            failed += len(bad)
            os.unlink(tmp)
            print(
                f"{os.path.basename(path)}: verification FAILED on {len(bad)} tensors; file left unchanged"
            )
            continue
        sha = zs.sha256_file(tmp)
        commit(tmp, dst)
        journal.write(
            {
                "file": os.path.basename(path),
                "written": dst,
                "sha256": sha,
                "tensors": len(recs),
            }
        )
        print(
            f"{os.path.basename(path)}: repaired and verified -> {dst} (sha256 {sha[:16]})",
            flush=True,
        )
    still_bad = []
    for qt, st, why in refused:
        c = gguf_check_tensor(qt, st, g, a.max_offset)
        if c["verdict"] != "PASS":
            still_bad.append(qt.name)
            print(f"  NOT REPAIRED {qt.name}: offset {c['offset']:+.5f}; {why}")
    journal.close()
    if not a.in_place and todo:
        print(
            f"note: only the repaired files were written to {dest_dir}; the other files of the set are unchanged"
        )
    if failed or still_bad:
        print(
            f"RESULT: FAIL ({failed} tensors did not verify, {len(still_bad)} affected tensors of a type this tool cannot repair)"
        )
        return 1
    print("RESULT: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument(
            "--lib", help=f"libggml.so of ik_llama.cpp (default: ${zs.LIB_ENV})"
        )
        p.add_argument("--rule", choices=zs.RULES, default="auto")
        p.add_argument("--max-offset", type=float, default=zs.MAX_OFFSET)
        p.add_argument(
            "--journal",
            help="JSON lines file (default: zero-sign-repair.jsonl in the destination)",
        )
        p.add_argument("--dry-run", action="store_true")
        m = p.add_mutually_exclusive_group(required=True)
        m.add_argument("--out", help="write the repaired files into this new directory")
        m.add_argument(
            "--in-place", action="store_true", help="replace the files where they are"
        )

    p = sub.add_parser("hotcold")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--source",
        help="the full-precision source checkpoint (default: hotcold.json's 'source')",
    )
    p.add_argument("--source-pattern", default=zs.SOURCE_PATTERN)
    p.add_argument("--layers", default="all")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument(
        "--link",
        choices=("hard", "copy", "none"),
        default="hard",
        help="with --out: how the files that do not change get into the new directory",
    )
    p = sub.add_parser("gguf")
    common(p)
    p.add_argument("--quant", nargs="+", required=True)
    p.add_argument("--source", nargs="+", required=True)
    p.add_argument(
        "--tensors", help="only tensors whose name matches this regular expression"
    )
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        return {"hotcold": cmd_hotcold, "gguf": cmd_gguf}[a.cmd](a)
    except zs.ZeroSignError as e:
        print(f"zero_sign_repair.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
