#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""zero_sign_check.py - measure the zero-sign offset of low-bit quants against their source. Read only.

  zero_sign_check.py demo    [--type IQ2_S]
      A synthetic matrix with 12 % exact zeros, quantised by libggml, before and after the repair.
      The quickest way to see that libggml loads and what the problem looks like.

  zero_sign_check.py hotcold --ckpt DIR [--source DIR] [--layers all|22|0-3,7] [--sample N] [--verify-sha]
                             [--json OUT]
      Every cold matrix of a hot/cold checkpoint (hotcold.json) against its source expert
      (MXFP4 weight + scale, or a BF16 / F16 / F32 weight). One table row per layer and projection.

  zero_sign_check.py gguf --quant FILE [FILE ...] --source FILE [FILE ...] [--tensors REGEX] [--json OUT]
      Every quantised tensor of a set of GGUF files against the tensor of the same name in the source
      GGUF files (F32 / F16 / BF16 / MXFP4). One table row per tensor.

Common: --lib libggml.so (or $IK_LIBGGML), --max-offset 0.005, --workers N (hotcold).

Offset = mean(quant - source) / mean|source|. A healthy matrix has |offset| <= 0.005 and about half of its
rows with a positive summed error. Exit code: 0 every matrix passes, 1 at least one has an offset,
2 the inputs could not be read.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import zero_sign as zs  # noqa: E402


def parse_layers(s, available):
    if s == "all":
        return list(available)
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    missing = sorted(out - set(available))
    if missing:
        raise zs.ZeroSignError(f"layers {missing} are not in the checkpoint")
    return sorted(out)


def fmt_row(cols, widths):
    return "  ".join(
        str(c).rjust(w) if i else str(c).ljust(w)
        for i, (c, w) in enumerate(zip(cols, widths))
    )


# ------------------------------------------------------------------------------------------------ demo
def cmd_demo(a):
    g = zs.GGML(a.lib)
    t = zs.TYPE_IDS[a.type.upper()]
    rng = np.random.default_rng(1234)
    n, k = 256, 1024
    x = rng.standard_normal((n, k)).astype(np.float32)
    z = rng.random((n, k)) < 0.12
    neg = rng.random((n, k)) < 0.5
    x[z] = np.where(neg[z], np.float32(-0.0), np.float32(0.0))
    pos0, neg0 = zs.zero_masks_from_float(x)
    q = g.quantize(
        t, x, np.ones(k, np.float32)
    )  # unit importance: some types (IQ2_XXS, IQ1_S) require one
    rb = g.row_size(t, k)
    print(f"libggml: {g.path}")
    print(
        f"synthetic matrix {n} x {k}, {100 * z.mean():.1f} % exact zeros (half +0, half -0), quantised to {zs.type_name(t)}"
    )
    before = zs.matrix_stats(x, g.dequant(t, q, n, k), pos0, neg0)
    print(
        f"as quantised: offset {before['offset']:+.5f}, rows with a positive summed error {before['rows_pos']:.3f}, "
        f"zeros decoded positive {before['zeros_decoded_positive']:.3f}, NMSE {before['nmse']:.6f} -> {before['verdict']}"
    )
    ok, why = zs.repair_support(t)
    if not ok:
        print(f"no repair: {why}")
        return 0 if before["verdict"] == "PASS" else 1
    zs.apply_zero_signs(
        q, t, n, k, pos0 | neg0, zs.target_negative("source", pos0, neg0), rb
    )
    after = zs.matrix_stats(x, g.dequant(t, q, n, k), pos0, neg0)
    print(
        f"repaired:     offset {after['offset']:+.5f}, rows with a positive summed error {after['rows_pos']:.3f}, "
        f"zeros decoded positive {after['zeros_decoded_positive']:.3f}, NMSE {after['nmse']:.6f} -> {after['verdict']}"
    )
    return 0 if after["verdict"] == "PASS" else 1


# ------------------------------------------------------------------------------------------------ hotcold
_W = {}


def _init(lib, source):
    _W["g"] = zs.GGML(lib)
    _W["src"] = zs.SafetensorsDir(source)


def _check(args):
    u, max_offset = args
    return zs.check_unit(u, _W["src"], _W["g"], max_offset)


def run_checks(units, lib, source, workers, max_offset):
    if workers <= 1:
        _init(lib, source)
        return [_check((u, max_offset)) for u in units]
    with mp.get_context("fork").Pool(
        workers, initializer=_init, initargs=(lib, source)
    ) as pool:
        return list(pool.imap(_check, [(u, max_offset) for u in units], chunksize=1))


def table_hotcold(recs, max_offset):
    w = (6, 4, 7, 6, 7, 21, 9, 9, 9, 7)
    print(
        fmt_row(
            (
                "layer",
                "proj",
                "type",
                "units",
                "zeros",
                "offset min..max",
                "rows_pos",
                "zero_neg",
                "NMSE",
                "verdict",
            ),
            w,
        )
    )
    groups = {}
    for r in recs:
        groups.setdefault((r["L"], r["w"]), []).append(r)
    for (L, wn), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], zs.COLD_PROJECTIONS.index(kv[0][1]))
    ):
        offs = [r["offset"] for r in rs]
        bad = sum(r["verdict"] != "PASS" for r in rs)
        print(
            fmt_row(
                (
                    L,
                    wn,
                    rs[0]["type"],
                    len(rs),
                    f"{100 * np.mean([r['zero_share'] for r in rs]):.2f}%",
                    f"{min(offs):+.5f}..{max(offs):+.5f}",
                    f"{np.mean([r['rows_pos'] for r in rs]):.3f}",
                    f"{np.mean([r['zeros_decoded_negative'] for r in rs]):.3f}",
                    f"{np.mean([r['nmse'] for r in rs]):.5f}",
                    "PASS" if not bad else f"OFFSET:{bad}",
                ),
                w,
            )
        )


def cmd_hotcold(a):
    ck = zs.HotColdCheckpoint(a.ckpt)
    source = a.source or ck.manifest.get("source")
    if not source or not os.path.isdir(source):
        raise zs.ZeroSignError("the source checkpoint is not found: pass --source DIR")
    lib = zs.find_libggml(a.lib)
    layers = parse_layers(a.layers, ck.layers)
    units = []
    for L in layers:
        units += ck.units(
            L,
            slots=zs.pick_slots(len(ck.cold_ids(L)), a.sample),
            source_pattern=a.source_pattern,
        )
    t0 = time.time()
    print(
        f"checkpoint {ck.root}\nsource     {os.path.abspath(source)}\nlibggml    {lib}\n"
        f"{len(units)} cold matrices in {len(layers)} layers, {a.workers} workers, max offset {a.max_offset}",
        flush=True,
    )
    recs = run_checks(units, lib, source, a.workers, a.max_offset)
    table_hotcold(recs, a.max_offset)
    bad = [r for r in recs if r["verdict"] != "PASS"]
    sha_bad = []
    if a.verify_sha:
        for L in layers:
            p = os.path.join(ck.root, ck.cold_file(L))
            want = ck.manifest["cold_files"][str(L)].get("sha256")
            got = zs.sha256_file(p)
            ok = got == want
            for w, (off, ln, _) in ck.tensors(L).items():
                name = ck.tensor_name(L, w)
                ok = ok and zs.sha256_range(p, off, ln) == ck.manifest.get(
                    "cold_tensor_sha256", {}
                ).get(name)
            print(
                f"layer {L}: {ck.cold_file(L)} sha256 {got[:16]} {'matches' if ok else 'DOES NOT MATCH'} hotcold.json "
                f"(file and its 3 cold tensors)"
            )
            if not ok:
                sha_bad.append(L)
    worst = max(recs, key=lambda r: abs(r["offset"]))
    print(
        f"{len(recs)} matrices, {len(bad)} with an offset beyond {a.max_offset}; largest |offset| {worst['offset']:+.5f} "
        f"({worst['key']}); {time.time() - t0:.0f} s"
    )
    if a.json:
        zs.write_new_file(
            a.json,
            json.dumps(
                {
                    "checkpoint": ck.root,
                    "max_offset": a.max_offset,
                    "units": recs,
                    "sha_mismatch_layers": sha_bad,
                    "pass": not bad and not sha_bad,
                },
                indent=1,
            ).encode(),
        )
    if sha_bad:
        print(f"RESULT: FAIL (layers {sha_bad}: the files do not match hotcold.json)")
        return 1
    print(
        "RESULT: " + ("PASS" if not bad else "FAIL (run zero_sign_repair.py hotcold)")
    )
    return 0 if not bad else 1


# ------------------------------------------------------------------------------------------------ gguf
def gguf_pairs(quant, source, pattern, g):
    qm = zs.gguf_tensor_map(quant, g)
    sm = zs.gguf_tensor_map(source, g)
    rx = re.compile(pattern) if pattern else None
    pairs, skipped = [], []
    for name, qt in sorted(qm.items()):
        if rx and not rx.search(name):
            continue
        if qt.ggml_type in (0, 1, 30):
            continue  # float tensors are not quantised
        st = sm.get(name)
        if st is None:
            skipped.append((name, "no tensor of that name in the source"))
            continue
        if st.ne != qt.ne:
            skipped.append((name, f"shape {qt.ne} differs from the source's {st.ne}"))
            continue
        if st.ggml_type == qt.ggml_type:
            skipped.append(
                (name, f"same type as the source ({zs.type_name(qt.ggml_type)})")
            )
            continue
        if st.ggml_type not in (0, 1, 30, 39):
            skipped.append(
                (
                    name,
                    f"source type {zs.type_name(st.ggml_type)} is not F32 / F16 / BF16 / MXFP4",
                )
            )
            continue
        if qt.k % zs.QK_K and qt.ggml_type in zs.SIGN_FIELD:
            skipped.append((name, f"row of {qt.k} values is not a multiple of 256"))
            continue
        pairs.append((qt, st))
    return pairs, skipped


def chunk_rows(t, budget=1 << 24):
    step = max(1, budget // max(t.k, 1))
    return [(r, min(r + step, t.nrows)) for r in range(0, t.nrows, step)]


def gguf_check_tensor(qt, st, g, max_offset):
    acc = zs.Stats()
    rb = g.row_size(qt.ggml_type, qt.k)
    fmt = None
    for r0, r1 in chunk_rows(qt):
        x, pos0, neg0, fmt = zs.read_gguf_source_rows(st, r0, r1, g)
        blob = np.frombuffer(
            zs.pread(qt.path, qt.offset + r0 * rb, (r1 - r0) * rb), np.uint8
        )
        acc.add(x, g.dequant(qt.ggml_type, blob, r1 - r0, qt.k), pos0, neg0)
    s = acc.summary(max_offset)
    return dict(
        name=qt.name,
        type=zs.type_name(qt.ggml_type),
        source_type=zs.type_name(st.ggml_type),
        source=fmt,
        ne=list(qt.ne),
        file=os.path.basename(qt.path),
        **s,
    )


def cmd_gguf(a):
    g = zs.GGML(a.lib)
    pairs, skipped = gguf_pairs(a.quant, a.source, a.tensors, g)
    print(
        f"libggml {g.path}\n{len(pairs)} quantised tensors paired with a source tensor, {len(skipped)} skipped",
        flush=True,
    )
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    w = (44, 8, 6, 7, 10, 9, 9, 9, 7)
    print(
        fmt_row(
            (
                "tensor",
                "type",
                "src",
                "zeros",
                "offset",
                "rows_pos",
                "zero_neg",
                "NMSE",
                "verdict",
            ),
            w,
        )
    )
    recs = []
    for qt, st in pairs:
        r = gguf_check_tensor(qt, st, g, a.max_offset)
        recs.append(r)
        print(
            fmt_row(
                (
                    r["name"][-44:],
                    r["type"],
                    r["source_type"],
                    f"{100 * r['zero_share']:.2f}%",
                    f"{r['offset']:+.5f}",
                    f"{r['rows_pos']:.3f}",
                    f"{r['zeros_decoded_negative']:.3f}",
                    f"{r['nmse']:.5f}",
                    r["verdict"],
                ),
                w,
            ),
            flush=True,
        )
    bad = [r for r in recs if r["verdict"] != "PASS"]
    if a.json:
        zs.write_new_file(
            a.json,
            json.dumps(
                {
                    "max_offset": a.max_offset,
                    "tensors": recs,
                    "skipped": skipped,
                    "pass": not bad,
                },
                indent=1,
            ).encode(),
        )
    if not recs:
        print("RESULT: nothing to check")
        return 2
    print(f"{len(recs)} tensors, {len(bad)} with an offset beyond {a.max_offset}")
    print("RESULT: " + ("PASS" if not bad else "FAIL (run zero_sign_repair.py gguf)"))
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument(
            "--lib", help=f"libggml.so of ik_llama.cpp (default: ${zs.LIB_ENV})"
        )
        p.add_argument("--max-offset", type=float, default=zs.MAX_OFFSET)

    p = sub.add_parser("demo")
    common(p)
    p.add_argument("--type", default="IQ2_S")
    p = sub.add_parser("hotcold")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--source",
        help="the full-precision source checkpoint (default: hotcold.json's 'source')",
    )
    p.add_argument("--source-pattern", default=zs.SOURCE_PATTERN)
    p.add_argument("--layers", default="all")
    p.add_argument(
        "--sample",
        type=int,
        help="check N cold experts per layer (spread evenly) instead of all",
    )
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument(
        "--verify-sha",
        action="store_true",
        help="also check the sha256 of each cold file and tensor against hotcold.json",
    )
    p.add_argument("--json")
    p = sub.add_parser("gguf")
    common(p)
    p.add_argument("--quant", nargs="+", required=True)
    p.add_argument("--source", nargs="+", required=True)
    p.add_argument(
        "--tensors", help="only tensors whose name matches this regular expression"
    )
    p.add_argument("--json")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        return {"demo": cmd_demo, "hotcold": cmd_hotcold, "gguf": cmd_gguf}[a.cmd](a)
    except zs.ZeroSignError as e:
        print(f"zero_sign_check.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
