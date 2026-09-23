#!/usr/bin/env python3
"""Time series of global %MON statistics for several runs, and their differences from a reference run.

Runs in the nereus env (matplotlib; no jax needed):
    python tools/plot_monitor_ts.py OUT.png --ref "Fortran 13 tiles=RUN/STDOUT.0000" \
        --run "JAX (1 A100)=runs_jax/.../monitor_all.txt" --run "Fortran 96 ranks=RUN/STDOUT.0000" \
        [--stats dynstat_theta_mean,dynstat_eta_sd,...] [--every 24] [--dt 3600] [--start 1992-01-01] [--title T]

Each statistic gets two panels: the values of every run (top; they lie on top of each other when the runs agree),
and the signed difference of every run minus the reference (bottom, linear axis). Only iterations present in all
runs (and multiples of --every) are used. Iteration n is at time start + (n - 1) * dt (nIter0 = 1 in V4r4).
"""

import argparse
import datetime as dt_
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _read_monitor():
    spec = importlib.util.spec_from_file_location("monitor_io", REPO / "mitgcm_jax" / "io" / "monitor.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.read_monitor


DEFAULT_STATS = ("dynstat_theta_mean", "dynstat_salt_mean", "dynstat_theta_sd", "dynstat_eta_sd", "dynstat_eta_min",
                 "dynstat_uvel_max")
LABELS = {"dynstat_theta_mean": "global mean theta (degC)", "dynstat_salt_mean": "global mean salinity (psu)",
          "dynstat_theta_sd": "theta std (degC)", "dynstat_eta_sd": "SSH std (m)", "dynstat_eta_min": "SSH min (m)",
          "dynstat_eta_max": "SSH max (m)", "dynstat_uvel_max": "max u (m/s)", "dynstat_vvel_max": "max v (m/s)",
          "dynstat_theta_max": "max theta (degC)", "dynstat_theta_min": "min theta (degC)",
          "seaice_area_mean": "mean ice concentration", "seaice_heff_mean": "mean ice thickness (m)",
          "seaice_area_max": "max ice concentration", "seaice_heff_max": "max ice thickness (m)",
          "seaice_hsnow_mean": "mean snow thickness (m)", "seaice_uice_max": "max ice u (m/s)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--ref", required=True, help="LABEL=PATH of the reference run")
    ap.add_argument("--run", action="append", default=[], help="LABEL=PATH (repeat)")
    ap.add_argument("--stats", default=",".join(DEFAULT_STATS))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--dt", type=float, default=3600.0)
    ap.add_argument("--start", default="1992-01-01")
    ap.add_argument("--title", default="")
    ap.add_argument("--ncol", type=int, default=3)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    read_monitor = _read_monitor()
    specs = [a.ref] + a.run
    runs = [(s.split("=", 1)[0], read_monitor(s.split("=", 1)[1])) for s in specs]
    stats = a.stats.split(",")
    its = sorted(set.intersection(*[set(m) for _, m in runs]))
    its = [i for i in its if i % a.every == 0 and all(all(k in m[i] for k in stats) for _, m in runs)]
    if not its:
        raise SystemExit("no common iterations with all statistics")
    t0 = dt_.datetime.fromisoformat(a.start)
    t = [t0 + dt_.timedelta(seconds=(i - 1) * a.dt) for i in its]
    ncol = min(a.ncol, len(stats))
    nrow = -(-len(stats) // ncol)
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(5.6 * ncol, 5.2 * nrow), dpi=110, sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1] * nrow})
    axes = np.atleast_2d(axes)
    colors = ["black", "tab:red", "tab:blue", "tab:green", "tab:orange"]
    styles = ["-", "--", ":", "-.", "-"]
    ref_label, ref = runs[0]
    for n, k in enumerate(stats):
        r, c = divmod(n, ncol)
        axv, axd = axes[2 * r, c], axes[2 * r + 1, c]
        for j, (lab, m) in enumerate(runs):
            axv.plot(t, [m[i][k] for i in its], styles[j % 5], color=colors[j % 5], lw=1.4 if j == 0 else 1.1,
                     label=lab)
        for j, (lab, m) in enumerate(runs[1:], start=1):
            d = np.array([m[i][k] - ref[i][k] for i in its])
            axd.plot(t, d, color=colors[j % 5], lw=1.0, label=f"{lab} - {ref_label}")
        axd.axhline(0, color="0.5", lw=0.6)
        axv.set_title(LABELS.get(k, k), fontsize=10)
        axd.set_ylabel("difference", fontsize=8)
        axd.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axv.ticklabel_format(axis="y", useOffset=False)
        if n == 0:
            axv.legend(fontsize=7, loc="best")
            axd.legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b" if (t[-1] - t[0]).days > 60 else "%d %b"))
    for n in range(len(stats), nrow * ncol):
        r, c = divmod(n, ncol)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    fig.suptitle(a.title or f"%MON time series ({len(its)} common outputs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out, len(its), "iterations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
