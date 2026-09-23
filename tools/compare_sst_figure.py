#!/usr/bin/env python3
"""Figure: JAX SST vs Fortran SST at one iteration, and their difference (nereus env).

    compare_sst_figure.py JAX_FRAME.npz FORTRAN_RUNDIR OUT.png
The JAX frame (run_jax.py frames/frame_*.npz) holds its iteration; the Fortran T.<iter>.data must exist."""

import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    argv = argv or sys.argv[1:]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nereus as nr
    from nereus.models.mitgcm.io import read_mds

    z = np.load(argv[0])
    it = int(z["iter"])
    run = Path(argv[1])
    _, T = read_mds(run / f"T.{it:010d}")
    fsst = np.asarray(T)[0].astype(np.float64).ravel()
    jsst = np.asarray(z["sst"], np.float64).ravel()
    mesh = nr.mitgcm.load_mesh(run, mask_land=True)
    lon, lat, land = mesh["lon"].values, mesh["lat"].values, mesh["land_mask"].values
    fsst = np.where(land, np.nan, fsst)
    jsst = np.where(land, np.nan, jsst)
    d = jsst - fsst
    fig = plt.figure(figsize=(16, 3.6))
    interp = None
    panels = [(jsst, "JAX MITgcm (A100 GPU)", "RdYlBu_r", -2, 30, "°C"),
              (fsst, "Fortran MITgcm c66g (gfortran, 1 CPU)", "RdYlBu_r", -2, 30, "°C")]
    dm = np.nanmax(np.abs(d))
    panels.append((d, f"JAX − Fortran (max |Δ| = {dm:.1e} °C)", "RdBu_r", -max(dm, 1e-12), max(dm, 1e-12), "°C"))
    for n, (v, title, cmap, lo, hi, unit) in enumerate(panels):
        ax = fig.add_subplot(1, 3, n + 1, projection=__import__("cartopy.crs", fromlist=["crs"]).Robinson())
        fig, ax, interp = nr.plot(v, lon, lat, projection="rob", ax=ax, interpolator=interp, cmap=cmap, vmin=lo,
                                  vmax=hi, colorbar=True, colorbar_label=unit, resolution=0.5)
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"Sea surface temperature, {z['date']} (iteration {it}), ECCO v4r4 flux-forced LLC90", fontsize=13, y=1.02)
    fig.savefig(argv[2], dpi=110, bbox_inches="tight")
    wet = ~np.isnan(d)
    print(f"it {it}: max |JAX-Fortran| SST = {dm:.3e} degC, rms {np.sqrt(np.nanmean(d[wet] ** 2)):.3e}; "
          f"float32 values equal: {np.sum(np.float32(jsst[wet]) == np.float32(fsst[wet]))} of {wet.sum()}")


if __name__ == "__main__":
    sys.exit(main())
