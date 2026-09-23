#!/usr/bin/env python3
"""Plan Task 21 figures: sensitivity maps (nereus, Robinson), amplification traces and FD h-sweeps.

Runs in the nereus env (never the earthkit env), reading what scripts/adjoint/multiweek_grad.py wrote (grid.npz,
grad_*.npz / screen_*.npz, results.jsonl); needs no JAX:

    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/adjoint/plot_sensitivity.py \
        --runs DIR1,DIR2,... --grad DIR/grad_ecco_28d_r0.npz --tag ecco_28d --figdir FIGDIR
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def rows(dirs):
    out = []
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                r["_dir"] = str(d)
                out.append(r)
    return out


def sym_limit(a, q=99.5):
    v = np.abs(a[np.isfinite(a)])
    v = v[v > 0]
    return float(np.percentile(v, q)) if v.size else 1.0


def map_plot(nereus, data, grid, title, label, fname, extent=None, sat=1.0):
    """sat < 1 saturates the colour scale at sat * the 99.5th percentile of |data| (shows the weak far field)."""
    lon, lat = grid["xC"].ravel(), grid["yC"].ravel()
    v = data.ravel()
    lim = sym_limit(v) * sat
    kw = dict(projection="rob", cmap="RdBu_r", vmin=-lim, vmax=lim, colorbar_label=label, title=title,
              method="nearest", influence_radius=150e3, resolution=0.5, figsize=(10, 5.5))
    if extent is not None:
        kw.update(projection="pc", extent=extent, figsize=(9, 5))
    fig, ax, _ = nereus.plot(v, lon, lat, **kw)
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma list of run directories (results.jsonl) for traces/FD plots")
    ap.add_argument("--grad", default="", help="gradient npz for the maps (interior fields)")
    ap.add_argument("--grid", default="", help="grid.npz (default: next to --grad)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--levels", default="0,16", help="0-based levels for the theta0 / kapGM maps")
    ap.add_argument("--minus", default="", help="second gradient npz: also map (--grad) - (--minus) of dJ/dtheta0")
    a = ap.parse_args()
    fig_dir = Path(a.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if a.grad:
        import nereus
        gpath = Path(a.grad)
        grid = dict(np.load(Path(a.grid) if a.grid else gpath.parent / "grid.npz"))
        g = dict(np.load(gpath))
        wet = grid["maskC"] > 0
        rC = grid["rC"]
        for k in [int(x) for x in a.levels.split(",")]:
            d = np.where(wet[:, k], g["theta"][:, k], np.nan)
            map_plot(nereus, d, {"xC": grid["xC"], "yC": grid["yC"]},
                     f"dJ/dtheta0, level {k + 1} ({-rC[k]:.0f} m), {a.tag}", "K per K (per cell)",
                     fig_dir / f"dJdtheta0_k{k + 1:02d}_{a.tag}.png")
            map_plot(nereus, d, {"xC": grid["xC"], "yC": grid["yC"]},
                     f"dJ/dtheta0, level {k + 1} ({-rC[k]:.0f} m), {a.tag} (colour scale saturated at 10 %)",
                     "K per K (per cell)", fig_dir / f"dJdtheta0_k{k + 1:02d}_{a.tag}_pacific.png",
                     extent=(100, 179.9, -20, 35), sat=0.1)
            if "kapGM" in g:
                d = np.where(wet[:, k], g["kapGM"][:, k] * grid["kapGM"][:, k], np.nan)
                map_plot(nereus, d, {"xC": grid["xC"], "yC": grid["yC"]},
                         f"dJ/dln(kapGM), level {k + 1} ({-rC[k]:.0f} m), {a.tag}", "K (per cell)",
                         fig_dir / f"dJdlnkapGM_k{k + 1:02d}_{a.tag}_pacific.png", extent=(100, 179.9, -20, 35))
        if a.minus:
            g2 = dict(np.load(a.minus))
            for k in [int(x) for x in a.levels.split(",")]:
                d = np.where(wet[:, k], g["theta"][:, k] - g2["theta"][:, k], np.nan)
                map_plot(nereus, d, {"xC": grid["xC"], "yC": grid["yC"]},
                         f"dJ/dtheta0 difference, level {k + 1} ({-rC[k]:.0f} m), {a.tag}", "K per K (per cell)",
                         fig_dir / f"dJdtheta0_diff_{a.tag}_k{k + 1:02d}_pacific.png", extent=(100, 179.9, -20, 35))
        for c, unit in (("tflux", "K per W/m^2"), ("taux", "K per N/m^2"), ("tauy", "K per N/m^2")):
            if c in g:
                d = np.where(wet[:, 0], g[c], np.nan)
                map_plot(nereus, d, {"xC": grid["xC"], "yC": grid["yC"]}, f"dJ/d{c} (time-constant), {a.tag}", unit,
                         fig_dir / f"dJd{c}_{a.tag}.png")
                map_plot(nereus, d, {"xC": grid["xC"], "yC": grid["yC"]}, f"dJ/d{c} (time-constant), {a.tag}", unit,
                         fig_dir / f"dJd{c}_{a.tag}_pacific.png", extent=(100, 179.9, -20, 35))
    if a.runs:
        rs = rows(a.runs.split(","))
        sc = [r for r in rs if r["action"] == "screen"]
        if sc:
            const = ("gsNm_1", "gsNm_2", "gtNm_1", "gtNm_2", "hFac_surfC", "hFac_surfS", "hFac_surfW", "hMixLayer",
                     "runoff", "sIceLoad")
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
            for r in sorted(sc, key=lambda r: (r["days"], r["mode"])):
                ft = r.get("field_trace") or []
                cst = set(r.get("carried_constants") or const)
                trs = {"all State fields (incl. carried constants)": np.asarray(r["trace"], float),
                       "dynamic fields (no carried constants)":
                           np.array([np.sqrt(sum(v * v for k, v in f.items() if k not in cst)) for f in ft]),
                       "theta only": np.array([f.get("theta", 0.0) for f in ft])}
                for ax, (name, tr) in zip(axs, trs.items()):
                    x = -np.arange(len(tr)) * r["chunk"] / 24.0 + r["days"]
                    ax.semilogy(x, tr, "-" if r["mode"] == "exact" else "--", marker=".",
                                label=f"{r['mode']} {r['days']:g} d")
                    ax.set_title(name, fontsize=10)
                    ax.set_xlabel("model day (the reverse sweep runs right to left)")
            axs[0].set_ylabel("cotangent norm at the chunk boundary")
            axs[-1].legend(fontsize=8)
            fig.suptitle("Reverse propagation of the end-of-window box-mean seed", fontsize=11)
            fig.savefig(fig_dir / "amplification_traces.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            print("wrote", fig_dir / "amplification_traces.png")
        fd = [r for r in rs if r["action"] == "fd"]
        grads = {(r["days"], r["mode"], r.get("repeat", 0)): r for r in rs if r["action"] == "grad"}
        if fd:
            dirs = sorted({r["direction"] for r in fd})
            fig, axs = plt.subplots(1, len(dirs), figsize=(3.2 * len(dirs), 3.6), sharey=True)
            axs = np.atleast_1d(axs)
            for ax, n in zip(axs, dirs):
                for days in sorted({r["days"] for r in fd if r["direction"] == n}):
                    sel = sorted([r for r in fd if r["direction"] == n and r["days"] == days], key=lambda r: r["h"])
                    for mode, ls in (("exact", "-"), ("ecco", "--")):
                        gref = grads.get((days, mode, 0))
                        if gref is None:
                            continue
                        ad = gref["dirderiv"][n]
                        err = [abs(r["fd"] - ad) / max(abs(ad), 1e-300) for r in sel]
                        ax.loglog([r["h"] for r in sel], err, ls, marker="o", label=f"{mode} {days:g} d")
                ax.set_title(n, fontsize=9)
                ax.set_xlabel("h")
            axs[0].set_ylabel("|FD - AD| / |AD|")
            axs[-1].legend(fontsize=7)
            fig.savefig(fig_dir / "fd_sweeps.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            print("wrote", fig_dir / "fd_sweeps.png")


if __name__ == "__main__":
    main()
