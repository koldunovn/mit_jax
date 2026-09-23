#!/bin/bash
# Figures and movies of a one-year JAX run (docs/RUN_ONE_YEAR.md, "Plots and movies"). Plain bash; run it inside a
# CPU job (Levante: sbatch -p shared -A ACCOUNT -c 32 --mem=64G --time=00:30:00 --wrap "bash scripts/runs/plot_one_year.sh ...").
#   bash scripts/runs/plot_one_year.sh TREE RUNDIR FIGDIR RUNOUT [RUNOUT_PART2 ...]
# TREE ff | full; RUNDIR the run directory of the run (for the grid); FIGDIR a new directory for the results;
# RUNOUT the scripts/run_jax.py output directory, or several in order for a run made in parts.
# FIGDIR gets: grid_mds/ (tools/write_grid_mds.py), frames/ (links to every frame_*.npz), monitor_all.txt,
# monitor_ts.png (daily global statistics), sst_robinson.{mp4,gif} and, full tree, ice_arctic / ice_antarctic.{mp4,gif}.
# Interpreters: $MITJAX_PYTHON (model env: grid files) and $MITJAX_NEREUS_PYTHON (nereus + cartopy + matplotlib;
# ffmpeg on PATH). JOBS (default 16) processes render movie frames; STRIDE (default 4) uses every 4th 6-hourly frame.
set -eu
SELF=$0
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
NPY=${MITJAX_NEREUS_PYTHON:-/work/ab0995/a270088/mambaforge/envs/nereus/bin/python}
JOBS=${JOBS:-16}; STRIDE=${STRIDE:-4}
TREE=${1:?TREE ff|full}; RUNDIR=${2:?RUNDIR}; FIG=${3:?FIGDIR}; shift 3
[ $# -ge 1 ] || { echo "give at least one run_jax output directory"; exit 2; }
case $TREE in
  ff)   STATS=dynstat_theta_mean,dynstat_salt_mean,dynstat_theta_sd,dynstat_eta_sd,dynstat_eta_min,dynstat_uvel_max
        NAME="flux-forced ocean" ;;
  full) STATS=dynstat_theta_mean,dynstat_eta_sd,dynstat_uvel_max,seaice_area_mean,seaice_heff_mean,seaice_hsnow_mean
        NAME="full model (bulk forcing + sea ice)" ;;
  *)    echo "TREE must be ff or full"; exit 2 ;;
esac
cd "$REPO"
mkdir "$FIG" "$FIG/frames"                      # new directory: nothing is overwritten
for d in "$@"; do
  cat "$d/monitor.txt" >> "$FIG/monitor_all.txt"
  ln -s "$(cd "$d/frames" && pwd)"/frame_*.npz "$FIG/frames/"
done
echo "$(ls "$FIG/frames" | wc -l) frames, $(grep -c dynstat_theta_mean "$FIG/monitor_all.txt") monitor outputs"
JAX_PLATFORMS=cpu "$PY" tools/write_grid_mds.py "$RUNDIR" "$FIG/grid_mds"
"$NPY" tools/plot_monitor_ts.py "$FIG/monitor_ts.png" --every 24 --ref "JAX=$FIG/monitor_all.txt" --stats "$STATS" \
  --title "MITgcm in JAX, ECCO v4r4 $NAME: daily global statistics"
L="MITgcm in JAX · ECCO v4r4 LLC90 $NAME"
A="--stride $STRIDE --fps 40 --jobs $JOBS"
"$NPY" tools/animate_globe.py "$FIG/frames" "$FIG/grid_mds" "$FIG/sst_robinson" --projection robinson --lon0 -150 $A \
  --label "$L · SST"
if [ "$TREE" = full ]; then
  for P in arctic antarctic; do
    [ $P = arctic ] && LON0=-45 || LON0=0
    "$NPY" tools/animate_globe.py "$FIG/frames" "$FIG/grid_mds" "$FIG/ice_$P" --var area --projection $P --lon0 $LON0 \
      $A --vmin 0 --vmax 1 --cmap Blues_r --units fraction --cbar-label "sea-ice concentration" --label "$L"
  done
fi
ls -la "$FIG"
