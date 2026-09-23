#!/usr/bin/env python3
"""2-D maps of differences between runs (JAX vs Fortran, Fortran vs Fortran), plotted with nereus.

Runs in the nereus env (not the model env):
    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python tools/plot_diff_maps.py ff_year OUT.png
    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python tools/plot_diff_maps.py full_month OUT.png [--jax STATE]

Differences on wet points only (surface hFacC > 0), shown as log10|difference|. Monthly Fortran dumps are float32
(JAX float64 states are rounded to float32 for them, "float32-equal" = identical); restarts (pickup*.ckptA) are
float64 and compared in float64. Figures:
  ff_year     flux-forced production run, 1992: log10|diff| of SST and SSH at the Fortran monthly dump iterations,
              JAX (runs_jax/ff_prod_1992_gpu_v2, one A100, states saved every 720 steps) minus the Fortran twin with
              the same 13 tiles (serial13), next to the Fortran 96-rank run minus the Fortran 13-tile run (the spread
              of the Fortran itself between tilings / global-sum orders). "identical" = float32-equal.
  full_month  full V4r4 (EXF bulk + sea ice), one month (it 745), float64 restarts: Fortran 96 ranks minus Fortran 13
              tiles: SST, SSH, sea-ice concentration and thickness (Arctic, Antarctic) — the yardstick for the JAX full model
              (the LSR sea-ice solver is tile-local, so the two tilings converge to different iterates). With
              --jax STATE (a JAX full-model state .npz at it 745) a second row shows JAX minus Fortran 13 tiles.
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


def logdiff(a, b, land):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).ravel()
    wet = ~land
    nz = d[wet]
    frac_eq = float(np.mean(nz == 0))
    mx = float(nz.max())
    with np.errstate(divide="ignore"):
        ld = np.where(d > 0, np.log10(np.where(d > 0, d, 1.0)), -np.inf)
    return ld, mx, frac_eq


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


def _logpanel(fig, nr, ccrs, plt, pos, proj, a, b, lon, lat, land, rng, title, interp, extent=None, f32=False):
    ld, mx, eq = logdiff(a, b, land)
    lo, hi = rng
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_under("white")
    ax = fig.add_subplot(*pos, projection=proj)
    t = f"{title}\nmax|d| {mx:.1e}" + (f", float32-equal {100 * eq:.1f}%" if f32 else "")
    return panel(ax, nr, ccrs, np.where(np.isfinite(ld), ld, lo - 1), lon, lat, land, cmap, lo, hi, t, interp, extent)


def _cbar(fig, plt, rect, rng, label):
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_under("white")
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*rng))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(label, fontsize=9)


def fig_ff_year(out):
    """Rows: monthly float32 dumps (JAX - Fortran 13 tiles | Fortran 96 - Fortran 13) for SST and SSH; last row: the
    float64 end-of-year state (pickup.ckptA, it 8761): JAX - Fortran 96 ranks, and Fortran 13 - Fortran 96 when the
    13-tile year has finished."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nereus as nr

    lon, lat, land = mesh()
    J = JAXRUNS / "ff_prod_1992_gpu_v2"
    its = [720, 2880, 5040, 7920]
    part = {720: 1, 2880: 1, 5040: 2, 7920: 2}
    dates = {720: "1992-01-31", 2880: "1992-04-30", 5040: "1992-07-29", 7920: "1992-11-27"}
    f13y, f96y = "ref_ff_serial13_1year", "ref_ff_mpi96_1year"
    have13 = (RUNS / f13y / "pickup.ckptA.data").exists()
    nrow = len(its) + 1
    fig = plt.figure(figsize=(17, 2.75 * nrow + 1.0), dpi=110)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08, wspace=0.03, hspace=0.42)
    interp = None
    r32 = {"SST": (-8, -4), "SSH": (-9, -5)}
    for r, it in enumerate(its):
        jst = J / f"part{part[it]}" / f"state_{it:010d}.npz"
        for c, (name, jv, fp, k, units) in enumerate((("SST", "theta", "T", 0, "K"), ("SSH", "etaN", "Eta", None, "m"))):
            f13, f96, jx = fortran(f13y, fp, it, k), fortran(f96y, fp, it, k), jax_state(jst, jv, k)
            for jj, (a, b, lab) in enumerate(((jx, f13, "JAX (1 A100) - Fortran 13 tiles"),
                                              (f96, f13, "Fortran 96 ranks - Fortran 13 tiles"))):
                interp = _logpanel(fig, nr, ccrs, plt, (nrow, 4, 4 * r + 2 * c + jj + 1), ccrs.Robinson(-150), a, b,
                                   lon, lat, land, r32[name], f"{name} {dates[it]} (it {it}), float32 dumps [{units}]\n{lab}",
                                   interp, f32=True)
    # float64 end of year
    jfin = J / "part2" / "state_final.npz"
    r64 = {"SST": (-12, -5), "SSH": (-13, -6)}
    for c, (name, jv, rec, k, units) in enumerate((("SST", "theta", 100, 0, "K"), ("SSH", "etaN", 400, None, "m"))):
        f96 = pickup_rec(f96y, rec)
        pairs = [(_jax64(jfin, jv, k), f96, "JAX (1 A100) - Fortran 96 ranks")]
        if have13:
            pairs.append((pickup_rec(f13y, rec), f96, "Fortran 13 tiles - Fortran 96 ranks"))
        for jj, (a, b, lab) in enumerate(pairs):
            interp = _logpanel(fig, nr, ccrs, plt, (nrow, 4, 4 * len(its) + 2 * c + jj + 1), ccrs.Robinson(-150), a, b,
                               lon, lat, land, r64[name], f"{name} 1992-12-31 (it 8761), float64 restart [{units}]\n{lab}",
                               interp)
    _cbar(fig, plt, [0.03, 0.045, 0.2, 0.01], r32["SST"], "log10|d| SST (K), float32 rows; white = identical")
    _cbar(fig, plt, [0.28, 0.045, 0.2, 0.01], r32["SSH"], "log10|d| SSH (m), float32 rows")
    _cbar(fig, plt, [0.53, 0.045, 0.2, 0.01], r64["SST"], "log10|d| SST (K), last row (float64)")
    _cbar(fig, plt, [0.78, 0.045, 0.2, 0.01], r64["SSH"], "log10|d| SSH (m), last row (float64)")
    fig.suptitle("MITgcm ECCO v4r4 flux-forced 1992, production configuration: JAX vs Fortran, next to Fortran vs "
                 "Fortran (two tilings)", fontsize=13, y=0.985)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_full_month(out, jax_path=None):
    """Full V4r4 after one month (it 745), float64 restarts: Fortran 96 ranks - Fortran 13 tiles (and JAX - Fortran 13
    tiles with --jax): log10|d| of SST, SSH, ice concentration and thickness (Arctic, Antarctic)."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nereus as nr

    lon, lat, land = mesh()
    f13, f96 = "ref_full_serial13_1month", "ref_full_mpi96_1month"
    rows = [("Fortran 96 ranks - Fortran 13 tiles", lambda what: (pickup_rec(f96, what) if isinstance(what, int)
                                                                  else fortran_pickup(f96, what)))]
    if jax_path:
        jv = {100: ("theta", 0), 400: ("etaN", None), "siAREA": ("AREA", None), "siHEFF": ("HEFF", None)}
        rows.append(("JAX - Fortran 13 tiles", lambda what: _jax64(jax_path, *jv[what])))
    panels = [("SST", 100, "K", (-7, -1), None, ccrs.Robinson(-150)),
              ("SSH", 400, "m", (-8, -1), None, ccrs.Robinson(-150)),
              ("ice concentration, Arctic", "siAREA", "", (-7, -1), [-180, 180, 55, 90], ccrs.NorthPolarStereo()),
              ("ice thickness, Arctic", "siHEFF", "m", (-7, -1), [-180, 180, 55, 90], ccrs.NorthPolarStereo()),
              ("ice concentration, Antarctic", "siAREA", "", (-7, -1), [-180, 180, -90, -55], ccrs.SouthPolarStereo()),
              ("ice thickness, Antarctic", "siHEFF", "m", (-7, -1), [-180, 180, -90, -55], ccrs.SouthPolarStereo())]
    ncol = 3
    nrow = 2 * len(rows)
    fig = plt.figure(figsize=(16, 5.2 * nrow + 1.0), dpi=110)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.07, wspace=0.05, hspace=0.3)
    for r, (lab, get) in enumerate(rows):
        interp = {}
        for c, (name, what, units, rng, extent, proj) in enumerate(panels):
            ref = pickup_rec(f13, what) if isinstance(what, int) else fortran_pickup(f13, what)
            key = (type(proj).__name__, str(extent))
            u = f" [{units}]" if units else ""
            interp[key] = _logpanel(fig, nr, ccrs, plt, (nrow, ncol, r * len(panels) + c + 1), proj, get(what), ref,
                                    lon, lat, land, rng, f"{name}{u}\n{lab}", interp.get(key), extent)
    _cbar(fig, plt, [0.2, 0.035, 0.6, 0.01], (-7, -1),
          "log10 |difference| (SST in K, SSH in m [colour range -8..-1], ice concentration, ice thickness in m)")
    fig.suptitle("MITgcm ECCO v4r4 full model (EXF bulk formulae + sea ice), after one month (1992-02-01, it 745), "
                 "float64 restarts", fontsize=13, y=0.975)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figure", choices=("ff_year", "full_month"))
    ap.add_argument("out")
    ap.add_argument("--jax", default=None, help="full_month: a JAX full-model state .npz at it 745")
    a = ap.parse_args(argv)
    if a.figure == "ff_year":
        fig_ff_year(a.out)
    else:
        fig_full_month(a.out, a.jax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
