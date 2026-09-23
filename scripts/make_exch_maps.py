#!/usr/bin/env python3
"""Generate mitgcm_jax/data/exch_maps_<T>x<sNx>x<sNy>.npz from a jaxdump exchange probe (plan Task 7).

    make_exch_maps.py RUNDIR [--iter N]

RUNDIR must hold a jaxdump run with the G00_geometry stage (group X). Prints, per exchange kind, how many halo points
the Fortran exchange writes, how many take the other vector component, how many change sign, and how many it leaves
untouched; checks that EXCH_3D equals EXCH_XY and EXCH_UV_3D equals EXCH_UV_XY (same routines underneath).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mitgcm_jax.io.dump import DumpSet  # noqa: E402
from mitgcm_jax.parallel.exchange import MAP_DIR, build_maps  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir")
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ds = DumpSet(Path(a.rundir) / "jaxdump")
    m = build_maps(ds, a.iter)
    L = m.layout
    halo = np.ones(L.shape2d, bool)
    halo[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = False
    halo = halo.reshape(-1)
    print(f"{'map':10s} {'written':>8s} {'other':>6s} {'neg':>6s} {'untouched':>9s}   (halo points: {halo.sum()})")
    for k, (s, c, g) in sorted(m.maps.items()):
        own = {"_u": 1, "_v": 2}.get(k[-2:], 1)
        other = 3 - own if k[-2:] in ("_u", "_v") else 0
        print(f"{k:10s} {np.sum((c > 0) & halo):8d} {np.sum(c == other) if other else 0:6d} {np.sum(g < 0):6d} "
              f"{np.sum((c == 0) & halo):9d}")
    for a1, a2 in (("T", "3D"), ("UVs_u", "UV3s_u"), ("UVs_v", "UV3s_v")):
        same = all(np.array_equal(x, y) for x, y in zip(m.maps[a1], m.maps[a2]))
        print(f"{a1} == {a2}: {same}")
        if not same:
            return 1
    out = Path(a.out) if a.out else MAP_DIR / f"exch_maps_{L.nTiles}x{L.sNx}x{L.sNy}.npz"
    m.save(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
