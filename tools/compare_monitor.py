#!/usr/bin/env python3
"""Compare %MON dynstat statistics of a JAX run (monitor.txt from scripts/run_jax.py) with a Fortran STDOUT.0000.

    compare_monitor.py JAX_MONITOR FORTRAN_STDOUT [--every 24]

Prints, per compared iteration, the worst relative difference over dynstat_{eta,uvel,vvel,wvel,theta,salt}_{max,min,
mean,sd,del2} (Fortran prints 14 significant digits: identical states give <= ~1e-13), and the overall worst."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitgcm_jax.io.monitor import read_monitor  # noqa: E402

KEYS = [f"dynstat_{v}_{s}" for v in ("eta", "uvel", "vvel", "wvel", "theta", "salt")
        for s in ("max", "min", "mean", "sd", "del2")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jax")
    ap.add_argument("fortran")
    ap.add_argument("--every", type=int, default=24)
    a = ap.parse_args(argv)
    mj, mf = read_monitor(a.jax), read_monitor(a.fortran)
    common = sorted(set(mj) & set(mf))
    worst_all = (0.0, None, None)
    for it in common:
        w = (0.0, None)
        for k in KEYS:
            if k in mj[it] and k in mf[it]:
                ref = mf[it][k]
                d = abs(mj[it][k] - ref) / max(abs(ref), 1e-30)
                if "wvel" in k and "mean" in k:
                    d = abs(mj[it][k] - ref) / 1e-9
                if d > w[0]:
                    w = (d, k)
        if w[0] >= worst_all[0]:
            worst_all = (w[0], w[1], it)
        if it % a.every == 0 or it == common[-1]:
            print(f"it {it:6d}  worst rel diff {w[0]:.2e}  ({w[1]})")
    print(f"compared {len(common)} iterations ({common[0]}..{common[-1]}); overall worst {worst_all[0]:.2e} "
          f"({worst_all[1]} at it {worst_all[2]})")


if __name__ == "__main__":
    sys.exit(main())
