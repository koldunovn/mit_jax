#!/usr/bin/env python3
"""Compare two jaxdump sets and report the FIRST (iteration, stage, field) that differs beyond tolerance.

    diffdump.py REF_DIR TEST_DIR [--rtol 1e-13] [--tol FIELD=RTOL ...] [--halos] [--all]

Comparison per (iteration, stage, field), in the reference's call order, over wet points only: C/W/S point masks
from the reference's own `hFacC/hFacW/hFacS` at the first stage that dumped them (dry points are skipped, never
zero-filled). Metric: max|test - ref| / max|ref| over wet points (relative to the field's scale, so small values
inside a field do not inflate it). Interior points by default; --halos also compares halo cells.
Flags (always reported, never silently passed):
  ZERO    field is identically zero on wet points in BOTH sets (allocated but never computed? -> check)
  MISSING key present in only one set
  NAN     non-finite values on wet points
Exit status 1 if anything exceeds tolerance or is MISSING/NAN; ZERO alone is a warning.
Tolerance classes (docs/plan): map/gather ~1e-15, scatter/reduction ~1e-12, cg2d fields at solver tolerance;
set per field with --tol.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitgcm_jax.io.dump import DumpSet  # noqa: E402


def masks_from(ds):
    """{kind: {tile: bool (nz, ny+2oly, nx+2olx)}} from the first dumped hFacC/W/S."""
    out = {}
    for kind, name in (("C", "hFacC"), ("W", "hFacW"), ("S", "hFacS")):
        for key in ds.keys():
            if key[2] == name:
                out[kind] = {t: r.data > 0 for t, r in ds.tiles(*key).items()}
                break
    return out


def _select(r, arr, halos):
    return arr if halos else arr[:, r.oly:arr.shape[1] - r.oly, r.olx:arr.shape[2] - r.olx]


def compare(ref, test, rtol=1e-13, tols=None, halos=False):
    tols = tols or {}
    masks = masks_from(ref)
    rows = []
    for key in ref.keys():
        it, stage, field = key
        if key not in test.index:
            rows.append((key, "MISSING", np.inf, 0))
            continue
        rt, tt = ref.tiles(*key), test.tiles(*key)
        if set(rt) != set(tt):
            rows.append((key, "MISSING", np.inf, 0))
            continue
        kind = next(iter(rt.values())).kind
        dmax, scale, n, nonfinite = 0.0, 0.0, 0, False
        for tile, r in rt.items():
            a, b = r.data, tt[tile].data
            m = masks.get(kind, {}).get(tile)
            if m is None or m.shape[0] != a.shape[0]:
                m = np.ones_like(a, dtype=bool) if m is None else np.broadcast_to(m[:1], a.shape)
            a, b, m = _select(r, a, halos), _select(r, b, halos), _select(r, m, halos)
            av, bv = a[m], b[m]
            if not (np.isfinite(av).all() and np.isfinite(bv).all()):
                nonfinite = True
            if av.size:
                dmax = max(dmax, float(np.nanmax(np.abs(bv - av))))
                scale = max(scale, float(np.nanmax(np.abs(av))), float(np.nanmax(np.abs(bv))))
                n += av.size
        if nonfinite:
            rows.append((key, "NAN", np.inf, n))
        elif scale == 0.0:
            rows.append((key, "ZERO", 0.0, n))
        else:
            rel = dmax / scale
            tol = tols.get(field, rtol)
            rows.append((key, "FAIL" if rel > tol else "ok", rel, n))
    for key in test.keys():
        if key not in ref.index:
            rows.append((key, "MISSING", np.inf, 0))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--rtol", type=float, default=1e-13)
    ap.add_argument("--tol", nargs="*", default=[], help="FIELD=RTOL overrides")
    ap.add_argument("--halos", action="store_true")
    ap.add_argument("--all", action="store_true", help="print every comparison, not just the first failure")
    a = ap.parse_args(argv)
    tols = {k: float(v) for k, v in (t.split("=") for t in a.tol)}
    rows = compare(DumpSet(a.ref), DumpSet(a.test), a.rtol, tols, a.halos)
    bad = [r for r in rows if r[1] in ("FAIL", "MISSING", "NAN")]
    zero = [r for r in rows if r[1] == "ZERO"]
    for (it, stage, field), status, rel, n in (rows if a.all else bad[:1]):
        print(f"{status:7s} iter {it} {stage:28s} {field:18s} rel {rel:.3e}  n={n}")
    for (it, stage, field), *_ in zero:
        print(f"ZERO    iter {it} {stage:28s} {field} (zero on wet points in both)")
    print(f"{len(rows)} compared, {len(bad)} bad, {len(zero)} zero-on-both")
    if bad and not a.all:
        (it, stage, field), status, rel, n = bad[0]
        print(f"FIRST DIFFERENCE: iter {it}, stage {stage}, field {field} ({status}, rel {rel:.3e})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
