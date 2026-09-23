#!/usr/bin/env python3
"""Compare a Fortran reference run's state dumps with the PO.DAAC V4r4 native-grid snapshot at the same time.

    compare_to_podaac.py RUNDIR ITER [--snapdir DIR] [--json OUT]

Reads `<FIELD>.<ITER %010d>` MDS dumps written by the run (`dumpInitAndLast=.TRUE.` writes T, S, Eta, ... at the
last step; pkg/seaice writes AREA, HEFF, HSNOW, UICE, VICE), maps them to the 13-tile layout, and compares with the
snapshot variables on wet points (product mask: hFacC>0 from the geometry product). Reports per field: max |diff|,
RMS diff, max |diff| / (max - min of the product field), and the count of points compared. The products are float32,
so ~1e-7 relative is the storage floor; beyond that the difference measures compiler/platform effects (production:
ifort -O2 -fp-model precise on 96 ranks; ours: gfortran).
"""

import argparse
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitgcm_jax.io.llc import compact_to_tiles  # noqa: E402
from mitgcm_jax.io.mds import read_mds  # noqa: E402

DATA = Path("/work/ab0995/a270088/MIT/data/eccov4r4")
SNAP = DATA / "products_snap_19920102"
GEOM = DATA / "products_fixed/ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"

# model dump prefix -> (product file stem, variable, 3-D?, point kind)
PAIRS = {
    "T": ("OCEAN_TEMPERATURE_SALINITY", "THETA", True, "C"),
    "S": ("OCEAN_TEMPERATURE_SALINITY", "SALT", True, "C"),
    "Eta": ("SEA_SURFACE_HEIGHT", "ETAN", False, "C"),
    "AREA": ("SEA_ICE_CONC_THICKNESS", "SIarea", False, "C"),
    "HEFF": ("SEA_ICE_CONC_THICKNESS", "SIheff", False, "C"),
    "HSNOW": ("SEA_ICE_CONC_THICKNESS", "SIhsnow", False, "C"),
    "UICE": ("SEA_ICE_VELOCITY", "SIuice", False, "W"),
    "VICE": ("SEA_ICE_VELOCITY", "SIvice", False, "S"),
}


def product(stem, var, snapdir):
    hits = sorted(Path(snapdir).glob(f"*/{stem}_snap_*.nc"))
    if len(hits) != 1:
        raise SystemExit(f"expected one {stem} snapshot in {snapdir}, found {len(hits)}")
    with netCDF4.Dataset(hits[0]) as g:
        return np.asarray(g[var][0], dtype=np.float64)   # drop time


def compare(rundir, it, snapdir=SNAP):
    with netCDF4.Dataset(GEOM) as g:
        masks = {"C": np.asarray(g["maskC"][:]) > 0, "W": np.asarray(g["maskW"][:]) > 0,
                 "S": np.asarray(g["maskS"][:]) > 0}
    rows = {}
    for prefix, (stem, var, is3d, kind) in PAIRS.items():
        f = Path(rundir) / f"{prefix}.{it:010d}"
        if not f.with_suffix(".data").exists() and not Path(str(f) + ".data").exists():
            continue
        arr, _ = read_mds(f)
        model = compact_to_tiles(arr[0])                       # (nz, 13, 90, 90) or (13, 90, 90)
        prod = product(stem, var, snapdir)
        mask = masks[kind] if is3d else masks[kind][0]
        d = (model - prod)[mask]
        span = float(prod[mask].max() - prod[mask].min()) or 1.0
        rows[prefix] = {"product": var, "n": int(mask.sum()), "max_abs": float(np.abs(d).max()),
                        "rms": float(np.sqrt((d ** 2).mean())), "max_rel_span": float(np.abs(d).max() / span),
                        "dry_nonzero_model": int((model[~mask] != 0).sum())}
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir")
    ap.add_argument("iter", type=int)
    ap.add_argument("--snapdir", default=str(SNAP))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = compare(a.rundir, a.iter, a.snapdir)
    if not rows:
        raise SystemExit(f"no dumps at iteration {a.iter} in {a.rundir}")
    print(f"{'field':6s} {'product':9s} {'n':>9s} {'max|d|':>11s} {'rms':>11s} {'max|d|/span':>12s} {'dry!=0':>7s}")
    for k, r in rows.items():
        print(f"{k:6s} {r['product']:9s} {r['n']:9d} {r['max_abs']:11.3e} {r['rms']:11.3e} "
              f"{r['max_rel_span']:12.3e} {r['dry_nonzero_model']:7d}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
