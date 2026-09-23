#!/usr/bin/env python3
"""Compare two MITgcm (or later JAX) run directories: state dumps and %MON.

    compare_runs.py RUN_A RUN_B [--iters N ...] [--json OUT]

For every MDS state dump present in both runs (T, S, Eta, U, V, W, PH at each dumped iteration, or --iters):
bitwise equality, and on wet points (hFacC > 0 of RUN_A's own grid output) max|B-A|, max|B-A| / max|A|, RMS.
For %MON: the worst relative difference per statistic over all common steps, and the statistics present in only one
run (SST/SSS stats exist only with one tile per process, pkg/monitor/monitor.F:125-128).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitgcm_jax.io.mds import read_mds  # noqa: E402
from mitgcm_jax.io.monitor import compare_monitors, read_monitor  # noqa: E402

FIELDS = ("T", "S", "Eta", "U", "V", "W", "PH")


def dumped_iters(run):
    its = set()
    for p in Path(run).glob("T.*.data"):
        m = re.fullmatch(r"T\.(\d{10})\.data", p.name)
        if m:
            its.add(int(m.group(1)))
    return sorted(its)


def wet_mask(run):
    h, _ = read_mds(Path(run) / "hFacC")
    return h[0] > 0   # (Nr, 1170, 90)


def compare_states(a, b, iters=None):
    iters = iters or sorted(set(dumped_iters(a)) & set(dumped_iters(b)))
    mask = wet_mask(a)
    rows = []
    for it in iters:
        for f in FIELDS:
            pa, pb = Path(a) / f"{f}.{it:010d}", Path(b) / f"{f}.{it:010d}"
            if not (Path(str(pa) + ".data").exists() and Path(str(pb) + ".data").exists()):
                continue
            same = Path(str(pa) + ".data").read_bytes() == Path(str(pb) + ".data").read_bytes()
            xa, _ = read_mds(pa)
            xb, _ = read_mds(pb)
            xa, xb = xa[0], xb[0]
            m = mask if xa.ndim == 3 else mask[0]
            d = (xb - xa)[m]
            scale = float(np.abs(xa[m]).max()) or 1.0
            rows.append({"iter": it, "field": f, "bitwise": same, "max_abs": float(np.abs(d).max()),
                         "max_rel": float(np.abs(d).max()) / scale, "rms": float(np.sqrt((d ** 2).mean()))})
    return rows


def compare_mon(a, b):
    ma, mb = read_monitor(Path(a) / "STDOUT.0000"), read_monitor(Path(b) / "STDOUT.0000")
    diffs, only = compare_monitors(ma, mb)
    worst = {}
    for (it, name), r in diffs.items():
        if r > worst.get(name, (-1, 0))[0]:
            worst[name] = (r, it)
    return worst, only, len(set(ma) & set(mb))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--iters", nargs="*", type=int)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = compare_states(a.run_a, a.run_b, a.iters)
    worst, only, nsteps = compare_mon(a.run_a, a.run_b)
    print(f"{'iter':>6s} {'field':5s} {'bitwise':7s} {'max|d|':>10s} {'max|d|/max|A|':>13s} {'rms':>10s}")
    for r in rows:
        print(f"{r['iter']:6d} {r['field']:5s} {str(r['bitwise']):7s} {r['max_abs']:10.3e} {r['max_rel']:13.3e} "
              f"{r['rms']:10.3e}")
    top = sorted(worst.items(), key=lambda kv: -kv[1][0])[:8]
    print(f"%MON: {nsteps} common steps; worst relative differences:")
    for name, (r, it) in top:
        print(f"   {name:28s} {r:.3e} (iter {it})")
    if only:
        print("   only in one run:", ", ".join(only))
    if a.json:
        Path(a.json).write_text(json.dumps({"states": rows, "mon_worst": worst, "mon_only": only}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
