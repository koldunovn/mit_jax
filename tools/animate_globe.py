#!/usr/bin/env python3
"""Render model fields on a rotating globe with nereus and stitch the frames into a movie.

Runs in the nereus env (NOT the model env):
    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python tools/animate_globe.py FRAMES_DIR MESH_RUNDIR OUT \
        [--var sst] [--vmin -2 --vmax 30] [--lat0 20] [--spin 360] [--fps 12] [--label "JAX MITgcm"]

FRAMES_DIR holds frame_<n>.npz files, each with a compact LLC90 field (1170, 90) under the name --var and scalars
`iter` and `date` (written by the model driver, mitgcm_jax/diagnostics/frames.py, or by frames_from_fortran below).
MESH_RUNDIR is a Fortran run directory with the model's grid files (XC, YC, RAC, hFacC ...), read by
nereus.mitgcm.load_mesh, whose point order is the same compact layout. Land (hFacC == 0 at the surface) is masked.
The globe turns by --spin degrees of longitude over the whole movie. Writes OUT/png/*.png, OUT.mp4 and OUT.gif.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def frames_from_fortran(rundir, out, prefix="T", level=0):
    """Frames from a Fortran run's T.<iter>.data snapshots (for testing the pipeline on reference output)."""
    import nereus as nr  # noqa: F401  (env check)
    from nereus.models.mitgcm.io import read_mds

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i, meta in enumerate(sorted(Path(rundir).glob(f"{prefix}.*.meta"))):
        _, a = read_mds(meta.with_suffix(""))
        it = int(meta.name.split(".")[1])
        np.savez(out / f"frame_{i:05d}.npz", sst=np.asarray(a)[level], iter=it, date=f"iter {it}")


def _render(a, files, idx, png):
    """Render frames files[i] for i in idx to png/<i>.png (one process; the regrid interpolator is reused)."""
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nereus as nr

    mesh = nr.mitgcm.load_mesh(a.mesh, mask_land=True)
    lon, lat = mesh["lon"].values, mesh["lat"].values
    land = mesh["land_mask"].values
    interp = None
    n = len(files)
    for i in idx:
        z = np.load(files[i])
        v = np.asarray(z[a.var], dtype=np.float64).ravel()
        v = np.where(land, np.nan, v)
        proj = ccrs.Orthographic(central_longitude=a.lon0 + a.spin * i / max(n - 1, 1), central_latitude=a.lat0)
        fig = plt.figure(figsize=(7.2, 8.0), dpi=110, facecolor="#0b1020")
        ax = fig.add_axes([0.03, 0.12, 0.94, 0.80], projection=proj)
        ax.set_facecolor("#0b1020")
        fig, ax, interp = nr.plot(v, lon, lat, projection=proj, ax=ax, interpolator=interp, method="nearest",
                                  resolution=a.resolution, cmap=a.cmap, vmin=a.vmin, vmax=a.vmax, coastlines=True,
                                  colorbar=False, land=False)
        ax.set_global()
        cax = fig.add_axes([0.2, 0.07, 0.6, 0.022])
        sm = plt.cm.ScalarMappable(cmap=a.cmap, norm=plt.Normalize(a.vmin, a.vmax))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(f"sea surface temperature ({a.units})", color="w")
        cb.ax.tick_params(colors="w")
        fig.text(0.5, 0.955, a.label, ha="center", color="w", fontsize=13)
        fig.text(0.5, 0.925, str(z["date"]), ha="center", color="#c8d0e0", fontsize=11)
        fig.savefig(png / f"{i:05d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        if i % 50 == 0:
            print(f"frame {i + 1}/{n} {z['date']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frames")
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--var", default="sst")
    ap.add_argument("--vmin", type=float, default=-2.0)
    ap.add_argument("--vmax", type=float, default=30.0)
    ap.add_argument("--cmap", default="RdYlBu_r")
    ap.add_argument("--lat0", type=float, default=20.0)
    ap.add_argument("--lon0", type=float, default=-30.0)
    ap.add_argument("--spin", type=float, default=360.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--label", default="MITgcm (JAX port), ECCO v4r4 LLC90")
    ap.add_argument("--units", default="°C")
    ap.add_argument("--jobs", type=int, default=1, help="render frames in N processes")
    a = ap.parse_args(argv)

    files = sorted(Path(a.frames).glob("frame_*.npz"))
    if not files:
        raise SystemExit(f"no frames in {a.frames}")
    png = Path(a.out) / "png"
    png.mkdir(parents=True, exist_ok=True)
    n = len(files)
    chunks = [list(range(k, n, a.jobs)) for k in range(a.jobs)]
    if a.jobs == 1:
        _render(a, files, chunks[0], png)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            pool.starmap(_render, [(a, files, c, png) for c in chunks])
    out = Path(a.out)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(png / "%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(out) + ".mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(png / "%05d.png"),
                    "-vf", "scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    str(out) + ".gif"], check=True)
    print(f"wrote {out}.mp4 and {out}.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
