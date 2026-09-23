#!/usr/bin/env python3
"""Compare a run_jax.py state_final.npz with the Fortran MDS output of the same iteration (float32 files).

    compare_final.py STATE_FINAL.npz FORTRAN_RUNDIR

Fields: T (theta), S (salt), U (uVel), V (vVel), W (wVel), Eta (etaN) at the iteration stored in the npz. Prints the
number of float32 values that differ (bitwise) and the max abs / relative difference over wet points."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitgcm_jax.io.llc import tiles_to_compact  # noqa: E402
from mitgcm_jax.io.mds import read_mds  # noqa: E402

PAIRS = [("T", "theta", "hFacC"), ("S", "salt", "hFacC"), ("U", "uVel", "hFacW"), ("V", "vVel", "hFacS"),
         ("W", "wVel", "hFacC"), ("Eta", "etaN", None)]


def main(argv=None):
    argv = argv or sys.argv[1:]
    z = np.load(argv[0])
    run = Path(argv[1])
    it = int(z["it"])
    OL, N = 4, 90
    for fname, key, hkey in PAIRS:
        p = run / f"{fname}.{it:010d}"
        if not Path(str(p) + ".meta").exists():
            print(f"{fname}: no Fortran file at it={it}")
            continue
        ref, _ = read_mds(p)
        ref = ref[0].astype(np.float32)
        a = np.asarray(z[key])[..., OL:OL + N, OL:OL + N]
        a = tiles_to_compact(a).astype(np.float32).reshape(ref.shape)
        wet = np.ones(ref.shape, bool)
        if hkey is not None:
            h = tiles_to_compact(np.asarray(z[hkey])[..., OL:OL + N, OL:OL + N]).reshape(ref.shape)
            wet = h > 0
        d = np.abs(a.astype(np.float64) - ref.astype(np.float64))
        ndiff = int(np.count_nonzero(a != ref))
        scale = np.abs(ref[wet]).max()
        print(f"{fname:4s} it={it}: {ndiff} of {a.size} float32 values differ; wet max abs {d[wet].max():.3e} "
              f"(rel to max {d[wet].max() / scale:.3e})")


if __name__ == "__main__":
    sys.exit(main())
