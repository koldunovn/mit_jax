#!/usr/bin/env python3
"""Plan Task 22 (M2 adjoint acceptance) figures: sensitivity maps of the full-V4r4 gradient (nereus; Robinson, the
western tropical Pacific and the Arctic), screen traces and FD h-sweeps, from what scripts/adjoint/fullgrad.py wrote
(grid.npz, grad_<mode>_<days>d_r0.npz, results.jsonl). Runs in the nereus env (never earthkit), no JAX:

    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/adjoint/plot_fullgrad.py \
        --grads ecco=DIR1/grad_ecco_7d_r0.npz,exact_full=DIR2/grad_exact_full_7d_r0.npz --tag 7d \
        --runs DIR1,DIR2,... --figdir FIGDIR
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PACIFIC = (100, 179.9, -20, 35)
UNITS = {"atemp": "m per K" , "aqh": "m per kg/kg", "tauu": "per N/m^2", "tauv": "per N/m^2",
         "swdown": "per W/m^2", "lwdown": "per W/m^2", "precip": "per m/s", "heff": "m per m"}


def rows(dirs):
    out = []
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    out.append(json.loads(line))
    return out


def lim_of(a, q=99.5):
    v = np.abs(a[np.isfinite(a)])
    v = v[v > 0]
    return float(np.percentile(v, q)) if v.size else 1.0


def plot_map(nereus, ccrs, data, grid, title, label, fname, region="global", lim=None, sat=1.0):
    lon, lat = grid["xC"].ravel(), grid["yC"].ravel()
    v = data.ravel()
    lim = (lim_of(v) if lim is None else lim) * sat
    kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim, method="nearest", resolution=0.5)
    if region == "global":
        fig, ax, _ = nereus.plot(v, lon, lat, projection="rob", colorbar_label=label, title=title,
                                 influence_radius=150e3, figsize=(10, 5.5), **kw)
    elif region == "pacific":
        fig, ax, _ = nereus.plot(v, lon, lat, projection="pc", extent=PACIFIC, colorbar_label=label, title=title,
                                 influence_radius=150e3, figsize=(9, 5), **kw)
    else:   # arctic
        fig = plt.figure(figsize=(6.4, 7.0), dpi=120)
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        ax = fig.add_axes([0.03, 0.12, 0.94, 0.78], projection=proj)
        fig, ax, _ = nereus.plot(v, lon, lat, projection=proj, ax=ax, colorbar=False, coastlines=True,
                                 influence_radius=80e3, **kw)
        ax.set_extent([-180, 180, 62, 90], crs=ccrs.PlateCarree())
        cax = fig.add_axes([0.2, 0.07, 0.6, 0.025])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-lim, lim))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(label)
        fig.text(0.5, 0.93, title, ha="center", fontsize=10)
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def maps(a, fig_dir):
    import cartopy.crs as ccrs
    import nereus
    grads = {}
    for item in a.grads.split(","):
        name, path = item.split("=", 1)
        grads[name] = dict(np.load(path))
        grid_path = Path(a.grid) if a.grid else Path(path).parent / "grid.npz"
    grid = dict(np.load(grid_path))
    wet = grid["maskC"] > 0
    rC = grid["rC"]
    G = {"xC": grid["xC"], "yC": grid["yC"]}
    tag = a.tag
    for mode, g in grads.items():
        t = f"{mode}, {tag}"
        for k in (16, 0):
            d = np.where(wet[:, k], g["theta"][:, k], np.nan)
            plot_map(nereus, ccrs, d, G, f"dJ/dtheta0, level {k + 1} ({-rC[k]:.0f} m), {t}", "K per K (per cell)",
                     fig_dir / f"dJdtheta0_k{k + 1:02d}_{mode}_{tag}_pacific.png", region="pacific", sat=0.1)
        for c in ("atemp", "tauu", "tauv", "aqh", "swdown", "lwdown", "precip"):
            if c in g:
                d = np.where(wet[:, 0], g[c], np.nan)
                plot_map(nereus, ccrs, d, G, f"dJ/d{c} (time-constant), {t}", UNITS[c],
                         fig_dir / f"dJd{c}_{mode}_{tag}_pacific.png", region="pacific")
        # sea-ice cost part: the Arctic
        lim_atemp = None
        for c in ("atemp", "heff", "tauu", "tauv", "swdown", "lwdown"):
            if c in g:
                d = np.where(wet[:, 0], g[c], np.nan)
                if c == "heff":   # the identity part dJ/dHEFF0 = w_ice inside the region: show d ln(HEFF0) scaling
                    d = np.where(wet[:, 0], g[c] * grid["HEFF0"], np.nan)
                    label = "m (dJ / d ln HEFF0, per cell)"
                else:
                    label = UNITS[c]
                plot_map(nereus, ccrs, d, G, f"dJ/d{c}{' ln HEFF0' if c == 'heff' else ''}, {t}", label,
                         fig_dir / f"dJd{c}_{mode}_{tag}_arctic.png", region="arctic",
                         lim=lim_atemp if c == "atemp" else None)
    if a.diff:
        m1, m2 = a.diff.split("-")
        g1, g2 = grads[m1], grads[m2]
        for c in ("atemp", "heff", "tauu"):
            if c in g1 and c in g2:
                d1 = np.where(wet[:, 0], g1[c], np.nan)
                d = np.where(wet[:, 0], g1[c] - g2[c], np.nan)
                plot_map(nereus, ccrs, d, G, f"dJ/d{c}: {m1} - {m2}, {tag}", UNITS[c],
                         fig_dir / f"dJd{c}_diff_{m1}_minus_{m2}_{tag}_arctic.png", region="arctic",
                         lim=lim_of(d1))
        d = np.where(wet[:, 16], g1["theta"][:, 16] - g2["theta"][:, 16], np.nan)
        plot_map(nereus, ccrs, d, G, f"dJ/dtheta0 level 17: {m1} - {m2}, {tag}", "K per K (per cell)",
                 fig_dir / f"dJdtheta0_k17_diff_{m1}_minus_{m2}_{tag}_pacific.png", region="pacific",
                 lim=lim_of(np.where(wet[:, 16], g1["theta"][:, 16], np.nan)) * 0.1)


def traces(a, fig_dir):
    rs = rows(a.runs.split(","))
    sc = [r for r in rs if r["action"] == "screen"]
    if sc:
        groups = ("dynamic", "theta", "prognostic", "seaice")
        fig, axs = plt.subplots(1, 4, figsize=(19, 4.3))
        for r in sorted(sc, key=lambda r: (r["days"], r["mode"])):
            ft = r.get("field_trace") or []
            cst = set(r.get("carried_constants") or ())
            sel = {"dynamic": lambda k: k not in cst, "theta": lambda k: k == "theta",
                   "prognostic": lambda k: k in ("theta", "salt", "uVel", "vVel", "etaN"),
                   "seaice": lambda k: k in ("AREA", "HEFF", "HSNOW", "TICES", "UICE", "VICE")}
            for ax, gname in zip(axs, groups):
                tr = np.array([np.sqrt(sum(v * v for k, v in f.items() if sel[gname](k))) for f in ft])
                x = -np.arange(len(tr)) * r["chunk"] / 24.0 + r["days"]
                ls = {"ecco": "--", "exact_nodyn": "-", "exact_full": ":"}.get(r["mode"], "-.")
                ax.semilogy(x, tr, ls, marker=".", label=f"{r['mode']} {r['days']:g} d")
                ax.set_title(gname, fontsize=10)
                ax.set_xlabel("model day (reverse sweep runs right to left)")
        axs[0].set_ylabel("State-cotangent norm at the chunk boundary")
        axs[-1].legend(fontsize=7)
        fig.suptitle("Full V4r4: reverse propagation of the end-of-window seed (box-mean theta + Arctic ice)",
                     fontsize=11)
        fig.savefig(fig_dir / "amplification_traces_full.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fig_dir / "amplification_traces_full.png")
    fd = [r for r in rs if r["action"] == "fd"]
    grads = {(r["days"], r["mode"]): r for r in rs if r["action"] == "grad" and r.get("repeat", 0) == 0
             and r.get("nproc", 1) == 1}
    if fd:
        dirs = sorted({r["direction"] for r in fd})
        fig, axs = plt.subplots(1, len(dirs), figsize=(3.2 * len(dirs), 3.6), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, n in zip(axs, dirs):
            for days in sorted({r["days"] for r in fd if r["direction"] == n}):
                sel = sorted([r for r in fd if r["direction"] == n and r["days"] == days], key=lambda r: r["h"])
                for mode, ls in (("exact_full", "-"), ("exact_nodyn", "-."), ("ecco", "--")):
                    gref = grads.get((days, mode))
                    if gref is None:
                        continue
                    ad = gref["dirderiv"][n]
                    err = [max(abs(r["fd"] - ad) / max(abs(ad), 1e-300), 1e-16) for r in sel]
                    ax.loglog([r["h"] for r in sel], err, ls, marker="o", label=f"{mode} {days:g} d")
            ax.axhline(1e-3, color="0.6", lw=0.8)
            ax.set_title(n, fontsize=9)
            ax.set_xlabel("h")
        axs[0].set_ylabel("|FD - AD| / |AD|")
        axs[-1].legend(fontsize=6)
        fig.savefig(fig_dir / "fd_sweeps_full.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fig_dir / "fd_sweeps_full.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grads", default="", help="comma list mode=path of gradient npz files (interior fields)")
    ap.add_argument("--grid", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--diff", default="", help="MODE1-MODE2: also map the difference of two --grads entries")
    ap.add_argument("--runs", default="", help="comma list of run directories (results.jsonl): traces, FD sweeps")
    ap.add_argument("--figdir", required=True)
    a = ap.parse_args()
    fig_dir = Path(a.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if a.grads:
        maps(a, fig_dir)
    if a.runs:
        traces(a, fig_dir)


if __name__ == "__main__":
    main()
