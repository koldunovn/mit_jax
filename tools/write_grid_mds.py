#!/usr/bin/env python3
"""Write the LLC90 grid files the plotting tools read (XC, YC, RAC, Depth, hFacC, RC, DRF as MDS .meta/.data pairs),
computed by the JAX grid initialisation from a run directory: the mesh for tools/animate_globe.py and
tools/plot_diff_maps.py --mesh without a Fortran run.

    python tools/write_grid_mds.py RUNDIR OUTDIR [--grid-dir DIR]

Runs in the model env. RUNDIR: a run directory (reference/make_rundir.py, e.g. with --no-binary); the tileNNN.mitgrid
files come from --grid-dir (default $MITJAX_GRID_DIR, mitgcm_jax/paths.py). OUTDIR must not exist. Format as the
Fortran's WRITE_GRID (write_grid.F:84-130, writeBinaryPrec = 32): big-endian float32, compact 1170x90 layout;
Depth = Ro_surf - R_low (write_grid.F:66); hFacC = the initial h0FacC (ini_masks_etc.F).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mitgcm_jax  # noqa: E402,F401
from mitgcm_jax import paths  # noqa: E402
from mitgcm_jax.grid.load import grid_from_files  # noqa: E402
from mitgcm_jax.io.llc import tiles_to_compact  # noqa: E402
from mitgcm_jax.layout import Layout  # noqa: E402


def write_mds(prefix, a):
    """One-record float32 MDS file: a is (ny, nx) or (nz, ny, nx) global, or (nz,) for a vertical vector."""
    a = np.asarray(a, dtype=">f4")
    dims = [(n, 1, n) for n in reversed(a.shape)] if a.ndim > 1 else [(1, 1, 1), (1, 1, 1), (a.size, 1, a.size)]
    dl = ",\n".join(" " + ",".join(f"{v:6d}" for v in d) for d in dims)
    Path(str(prefix) + ".meta").write_text(
        f" nDims = [ {len(dims):3d} ];\n dimList = [\n{dl}\n ];\n dataprec = [ 'float32' ];\n nrecords = [      1 ];\n")
    a.tofile(str(prefix) + ".data")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir")
    ap.add_argument("outdir")
    ap.add_argument("--grid-dir", default=str(paths.GRID_DIR))
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    if out.exists():
        raise SystemExit(f"{out} exists (nothing is overwritten)")
    L = Layout()
    g = grid_from_files(a.rundir, a.grid_dir, layout=L)

    def compact(x):           # [tile, (k,) j, i] with halos -> compact interior ((k,) 1170, 90)
        x = np.asarray(x)[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx]
        return tiles_to_compact(np.moveaxis(x, 0, -3))
    f = g.f
    fields = {"XC": compact(f["xC"]), "YC": compact(f["yC"]), "RAC": compact(f["rA"]),
              "Depth": compact(np.asarray(f["Ro_surf"]) - np.asarray(f["R_low"])), "hFacC": compact(f["h0FacC"]),
              "RC": np.asarray(f["rC"]), "DRF": np.asarray(f["drF"])}
    out.mkdir(parents=True)
    for k, v in fields.items():
        write_mds(out / k, v)
    print(f"wrote {', '.join(fields)} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
