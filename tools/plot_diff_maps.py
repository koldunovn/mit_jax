#!/usr/bin/env python3
"""2-D maps of differences between runs (JAX vs Fortran, Fortran vs Fortran), plotted with nereus.

Runs in the nereus env (not the model env):
    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python tools/plot_diff_maps.py ff_year OUT.png
    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python tools/plot_diff_maps.py full_month OUT.png [--jax STATE]

Signed differences on wet points only (surface hFacC > 0), linear symmetric colour scales (Nikolay 2026-09-23: no
log scale), float64 restarts (pickup*.ckptA) on both sides. Figures:
  ff_year     flux-forced production run, end of 1992 (it 8761): SST, theta at 300 m, SSS, SSH for JAX - Fortran
              96 ranks, Fortran 13 ranks - Fortran 96 ranks (the Fortran's own spread between tilings), JAX - Fortran
              13 tiles (runs_jax/ff_prod_1992_gpu_v2: one A100). Time evolution: tools/plot_monitor_ts.py.
  full_month  full V4r4 (EXF bulk + sea ice), one month (it 745), float64 restarts: Fortran 96 ranks minus Fortran 13
              tiles: SST, SSH, sea-ice concentration and thickness (Arctic, Antarctic) — the yardstick for the JAX full model
              (the LSR sea-ice solver is tile-local, so the two tilings converge to different iterates). With
              --jax STATE (a JAX full-model state .npz at it 745) a second row shows JAX minus Fortran 13 ranks.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/work/ab0995/a270088/MIT/reference/runs")
JAXRUNS = Path("/work/ab0995/a270088/MIT/runs_jax")
MESH = RUNS / "smoke_ff_v3_forced_jd_3steps"          # any run dir with the LLC90 grid files (XC, YC, hFacC ...)
OL = 4                                                  # JAX halo width (mitgcm_jax/layout.py)


def _llc():
    """mitgcm_jax/io/llc.py loaded by path (numpy only; importing the package would need jax)."""
    spec = importlib.util.spec_from_file_location("llc", REPO / "mitgcm_jax" / "io" / "llc.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LLC = _llc()


def jax_state(path, var, k=None):
    """JAX State checkpoint (run_jax --checkpoint-every): tiles with halos [13, (Nr), 98, 98] -> compact (1170, 90)."""
    a = np.load(path)[var]
    a = a[..., OL:-OL, OL:-OL]
    if k is not None:
        a = a[:, k]
    return LLC.tiles_to_compact(a).astype(np.float32)


def fortran(run, prefix, it, k=None):
    from nereus.models.mitgcm.io import read_mds
    _, a = read_mds(RUNS / run / prefix, it)
    a = np.asarray(a)
    return (a[k] if k is not None else a).astype(np.float32)


def fortran_pickup(run, name, suffix="ckptA", pickup="pickup_seaice"):
    from nereus.models.mitgcm.io import read_mds
    meta, a = read_mds(RUNS / run / f"{pickup}.{suffix}")
    flds = [f.strip() for f in meta["fldList"]]
    return np.asarray(a)[flds.index(name)]                 # float64 (pickups are float64)


def mesh():
    import nereus as nr
    m = nr.mitgcm.load_mesh(MESH, mask_land=True)
    return m["lon"].values, m["lat"].values, m["land_mask"].values


def panel(ax, nr, ccrs, v, lon, lat, land, cmap, vmin, vmax, title, interp, extent=None):
    v = np.where(land, np.nan, np.asarray(v, dtype=np.float64).ravel())
    proj = ax.projection
    fig, ax, interp = nr.plot(v, lon, lat, projection=proj, ax=ax, interpolator=interp, method="nearest",
                              resolution=0.5 if extent is None else 0.25, cmap=cmap, vmin=vmin, vmax=vmax,
                              coastlines=True, colorbar=False, land=False)
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("#bdbdbd")
    ax.set_title(title, fontsize=8.5)
    return interp


def pickup_rec(run, rec, suffix="ckptA"):
    """One record of the float64 ocean pickup (fldList Uvel,Vvel,Theta,Salt,GuNm1,GuNm2,GvNm1,GvNm2 x Nr, then EtaN,
    dEtaHdt, EtaH): SST = record 2*Nr = 100, EtaN = 8*Nr = 400 (Nr = 50)."""
    from nereus.models.mitgcm.io import read_mds
    _, a = read_mds(RUNS / run / f"pickup.{suffix}")
    return np.asarray(a)[rec]


def _jax64(path, var, k=None):
    a = np.load(path)[var][..., OL:-OL, OL:-OL]
    if k is not None:
        a = a[:, k]
    return LLC.tiles_to_compact(a)


def _vmax(ds, wet, q=99.9):
    """Symmetric colour limit: the q-th percentile of |d| over all panels of a row, rounded to one significant digit."""
    v = max(float(np.percentile(np.abs(d.ravel()[wet]), q)) for d in ds)
    if v == 0:
        return 1e-30
    e = np.floor(np.log10(v))
    return float(np.ceil(v / 10 ** e) * 10 ** e)


def _signed_row(fig, nr, ccrs, plt, nrow, ncol, r, diffs, labels, name, units, lon, lat, land, interp, proj,
                extent=None):
    wet = ~land
    vmax = _vmax(diffs, wet)
    cmap = plt.get_cmap("RdBu_r")
    for c, (d, lab) in enumerate(zip(diffs, labels)):
        ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection=proj)
        dd = d.ravel()
        rms = float(np.sqrt(np.mean(dd[wet] ** 2)))
        interp = panel(ax, nr, ccrs, d, lon, lat, land, cmap, -vmax, vmax,
                       f"{name}: {lab}\nmax|d| {np.abs(dd[wet]).max():.1e} {units}, rms {rms:.1e}", interp, extent)
    return interp, vmax


def _row_cbar(fig, plt, axes_row_bottom, vmax, label):
    cax = fig.add_axes([0.35, axes_row_bottom, 0.3, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap("RdBu_r"), norm=plt.Normalize(-vmax, vmax)), cax=cax,
                      orientation="horizontal", extend="both")
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()


def fig_ff_year(out):
    """End of 1992 (it 8761), float64 restarts, signed differences on linear symmetric scales (one colour limit per
    row: the 99.9th percentile of |d| over the row's three panels; hotspots saturate). Rows: SST, theta at 300 m,
    SSS, SSH. Columns: JAX - Fortran 96 ranks, Fortran 13 ranks - Fortran 96 ranks, JAX - Fortran 13 ranks."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nereus as nr

    lon, lat, land = mesh()
    jf = JAXRUNS / "ff_prod_1992_gpu_v2" / "part2" / "state_final.npz"
    f13y, f96y = "ref_ff_serial13_1year", "ref_ff_mpi96_1year"
    # pickup records: Theta 100..149, Salt 150..199 (k = 0..49), EtaN 400; RC(19) = -299.9 m
    rows = [("SST", "theta", 100, 0, "degC"), ("theta at 300 m", "theta", 119, 19, "degC"),
            ("SSS", "salt", 150, 0, "psu"), ("SSH", "etaN", 400, None, "m")]
    labels = ["JAX (1 A100) - Fortran 96 ranks", "Fortran 13 ranks - Fortran 96 ranks", "JAX (1 A100) - Fortran 13 ranks"]
    nrow, ncol = len(rows), 3
    fig = plt.figure(figsize=(17, 3.35 * nrow + 0.8), dpi=110)
    top, bottom = 0.93, 0.02
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom + 0.03, wspace=0.03, hspace=0.55)
    interp = None
    for r, (name, jv, rec, k, units) in enumerate(rows):
        J = _jax64(jf, jv, k)
        F96, F13 = pickup_rec(f96y, rec), pickup_rec(f13y, rec)
        interp, vmax = _signed_row(fig, nr, ccrs, plt, nrow, ncol, r, [J - F96, F13 - F96, J - F13], labels, name,
                                   units, lon, lat, land, interp, ccrs.Robinson(-150))
        axb = fig.axes[-1].get_position().y0
        _row_cbar(fig, plt, axb - 0.022, vmax, f"{name} difference ({units})")
    fig.suptitle("MITgcm ECCO v4r4 flux-forced, production configuration: state differences at the end of 1992 "
                 "(8760 steps, float64 restarts)\nJAX vs Fortran next to Fortran vs Fortran (two tilings): the same "
                 "hotspots with flipped signs = round-off growing in the unstable parts of the flow", fontsize=12,
                 y=0.995)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_full_month(out, jax_path=None, f13="ref_full_serial13_1month", f96="ref_full_mpi96_1month",
                   when="after one month (1992-02-01, it 745)", jax_label="JAX"):
    """Full V4r4 after one month (it 745), float64 restarts, signed differences on linear scales: Fortran 96 ranks -
    Fortran 13 ranks (and JAX - Fortran 13 ranks with --jax, same colour limits per panel)."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nereus as nr

    lon, lat, land = mesh()
    wet = ~land
    jv = {100: ("theta", 0), 400: ("etaN", None), "siAREA": ("AREA", None), "siHEFF": ("HEFF", None)}

    def fortran_get(run, what):
        return pickup_rec(run, what) if isinstance(what, int) else fortran_pickup(run, what)

    rows = [("Fortran 96 ranks - Fortran 13 ranks", lambda what: fortran_get(f96, what))]
    if jax_path:
        rows.append((f"{jax_label} - Fortran 13 ranks", lambda what: _jax64(jax_path, *jv[what])))
    panels = [("SST", 100, "degC", None, ccrs.Robinson(-150)),
              ("SSH", 400, "m", None, ccrs.Robinson(-150)),
              ("ice concentration, Arctic", "siAREA", "", [-180, 180, 55, 90], ccrs.NorthPolarStereo()),
              ("ice thickness, Arctic", "siHEFF", "m", [-180, 180, 55, 90], ccrs.NorthPolarStereo()),
              ("ice concentration, Antarctic", "siAREA", "", [-180, 180, -90, -55], ccrs.SouthPolarStereo()),
              ("ice thickness, Antarctic", "siHEFF", "m", [-180, 180, -90, -55], ccrs.SouthPolarStereo())]
    refs = {p[1]: fortran_get(f13, p[1]) for p in panels}
    diffs = {(r, p[1]): get(p[1]) - refs[p[1]] for r, (_, get) in enumerate(rows) for p in panels}
    ncol = len(panels)
    nrow = len(rows)
    fig = plt.figure(figsize=(3.4 * ncol, 4.2 * nrow + 1.2), dpi=110)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.14, wspace=0.06, hspace=0.35)
    cmap = plt.get_cmap("RdBu_r")
    for c, (name, what, units, extent, proj) in enumerate(panels):
        vmax = _vmax([diffs[(r, what)] for r in range(nrow)], wet)
        for r, (lab, _) in enumerate(rows):
            d = diffs[(r, what)].ravel()
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection=proj)
            u = f" {units}" if units else ""
            panel(ax, nr, ccrs, d, lon, lat, land, cmap, -vmax, vmax,
                  f"{name}\n{lab}\nmax|d| {np.abs(d[wet]).max():.1e}{u}", None, extent)
        pos = fig.axes[-1].get_position()
        cax = fig.add_axes([pos.x0 + 0.1 * pos.width, 0.08, 0.8 * pos.width, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax)), cax=cax,
                          orientation="horizontal", extend="both")
        cb.set_label(f"{name.split(',')[0]} diff{' (' + units + ')' if units else ''}", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"MITgcm ECCO v4r4 full model (EXF bulk formulae + sea ice) {when}, "
                 "float64 restarts: signed differences (linear scale)", fontsize=12, y=0.97)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figure", choices=("ff_year", "full_month", "full_year"))
    ap.add_argument("out")
    ap.add_argument("--jax", default=None, help="full_month: a JAX full-model state .npz at it 745")
    a = ap.parse_args(argv)
    if a.figure == "ff_year":
        fig_ff_year(a.out)
    elif a.figure == "full_month":
        fig_full_month(a.out, a.jax)
    else:
        fig_full_month(a.out, a.jax, "ref_full_mpi13_1year", "ref_full_mpi96_1year", "at the end of 1992 (it 8761)",
                       "JAX (1 GH200)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
