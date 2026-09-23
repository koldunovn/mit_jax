#!/usr/bin/env python3
"""Compare %MON statistics of a JAX run (monitor.txt from scripts/run_jax.py) with a Fortran STDOUT.0000.

    compare_monitor.py JAX_MONITOR FORTRAN_STDOUT [--every 24] [--groups dynstat,seaice,exf] [--per-stat]

Prints, per compared iteration, the worst relative difference over the statistics of the chosen groups
(dynstat_{eta,uvel,vvel,wvel,theta,salt}_{max,min,mean,sd,del2}; full tree also seaice_{uice,vice,area,heff,hsnow}_*
(SEAICE_MONITOR) and exf_<field>_* (EXF_MONITOR)), and the overall worst. Fortran prints 14 significant digits:
identical states give <= ~1e-13. --per-stat: the max over all compared iterations for every statistic (absolute and
relative difference). The near-zero dynstat_wvel_mean is judged relative to 1e-9 (as before)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitgcm_jax.io.monitor import read_monitor  # noqa: E402

STATS = ("max", "min", "mean", "sd", "del2")
GROUPS = {
    "dynstat": [f"dynstat_{v}_{s}" for v in ("eta", "uvel", "vvel", "wvel", "theta", "salt") for s in STATS],
    "seaice": [f"seaice_{v}_{s}" for v in ("uice", "vice", "area", "heff", "hsnow") for s in STATS],
    "exf": [f"exf_{v}_{s}" for v in ("ustress", "vstress", "hflux", "sflux", "wspeed", "atemp", "aqh", "lwflux", "evap",
                                     "precip", "swflux", "swdown", "lwdown", "apressure", "runoff") for s in STATS],
}
KEYS = GROUPS["dynstat"]


def rel_diff(k, x, ref):
    if "wvel" in k and "mean" in k:
        return abs(x - ref) / 1e-9
    return abs(x - ref) / max(abs(ref), 1e-30)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jax")
    ap.add_argument("fortran")
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("--groups", default="dynstat", help="comma-separated: dynstat, seaice, exf (or 'all')")
    ap.add_argument("--per-stat", action="store_true", help="max difference per statistic over all iterations")
    a = ap.parse_args(argv)
    groups = list(GROUPS) if a.groups == "all" else a.groups.split(",")
    keys = [k for gname in groups for k in GROUPS[gname]]
    mj, mf = read_monitor(a.jax), read_monitor(a.fortran)
    common = sorted(set(mj) & set(mf))
    worst_all = (0.0, None, None)
    per = {}
    for it in common:
        w = (0.0, None)
        for k in keys:
            if k in mj[it] and k in mf[it]:
                ref = mf[it][k]
                d = rel_diff(k, mj[it][k], ref)
                a_, r_, n_ = per.get(k, (0.0, 0.0, 0))
                per[k] = (max(a_, abs(mj[it][k] - ref)), max(r_, d), n_ + 1)
                if d > w[0]:
                    w = (d, k)
        if w[0] >= worst_all[0]:
            worst_all = (w[0], w[1], it)
        if it % a.every == 0 or it == common[-1]:
            print(f"it {it:6d}  worst rel diff {w[0]:.2e}  ({w[1]})")
    print(f"compared {len(common)} iterations ({common[0]}..{common[-1]}); overall worst {worst_all[0]:.2e} "
          f"({worst_all[1]} at it {worst_all[2]})")
    missing = [k for k in keys if k not in per]
    if missing:
        print(f"statistics in neither or only one file: {missing}")
    if a.per_stat:
        print(f"{'statistic':32s} {'n':>4s} {'max |diff|':>12s} {'max rel':>10s}")
        for k in keys:
            if k in per:
                ad, rd, n = per[k]
                print(f"{k:32s} {n:4d} {ad:12.3e} {rd:10.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
