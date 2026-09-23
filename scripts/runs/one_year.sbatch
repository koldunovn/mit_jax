#!/bin/bash
#SBATCH --job-name=mitjax_year
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/one_year/%j.out
#
# One year of ECCO v4r4 (1992) with the JAX model on one GPU, with the output settings of the validated runs
# (docs/RUN_ONE_YEAR.md; scripts/cabeus/one_year_*.pbs are the PBS versions of this script):
#   sbatch [-A ACCOUNT -p PARTITION --output=LOG] scripts/runs/one_year.sbatch TREE OUTDIR RUNDIR [NSTEPS [RESTART FRAME_OFFSET]]
# TREE          ff (flux-forced ocean) | full (bulk-formula EXF + sea ice)
# OUTDIR        new directory for the output of scripts/run_jax.py (run.log, monitor.txt, frames/, snap_*, state_*)
# RUNDIR        run directory of that tree: reference/make_rundir.py TREE serial13 NAME --nsteps 8760 --no-binary
# NSTEPS        default 8760 = 1992-01-01 12:00 to 1992-12-31 12:00 (deltaT = 3600 s)
# RESTART FRAME_OFFSET   continue from state_<iter>.npz / state_final.npz of an earlier part; FRAME_OFFSET = number of
#               frame files the earlier parts wrote (ls OUTDIR_part1/frames | wc -l), so frame numbers continue
# DKRZ Levante: A100-80   sbatch -A ab0995_gpu -p gpu --constraint=a100_80 scripts/runs/one_year.sbatch ...
#               GH200     MITJAX_ENV=dolpung sbatch -A mh1571 -p dolpung scripts/runs/one_year.sbatch ...
# Measured (1 GPU, this output): full 0.29 s/step on GH200 (8760 steps in 48 min); ff 0.17 s/step on an A100-40.
set -u
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
if [ "${MITJAX_ENV:-}" = dolpung ]; then
  source "$REPO/scripts/dolpung/env_dolpung.sh"                    # sets PY (aarch64 venv)
else
  PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
fi
cd "$REPO"

TREE=${1:?TREE: ff or full}; OUT=${2:?OUTDIR}; RUNDIR=${3:?RUNDIR}; NSTEPS=${4:-8760}; RESTART=${5:-}; FOFF=${6:-0}
case $TREE in
  ff)   SNAP=240 ;;          # as the validated flux-forced year (10-day snapshots)
  full) SNAP=720 ;;          # as the validated full year (30-day snapshots)
  *)    echo "TREE must be ff or full, not $TREE"; exit 2 ;;
esac
if [ -n "$RESTART" ]; then START="--restart $RESTART --frame-offset $FOFF"; else START="--init-oracle pickup"; fi
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(date -Is); repo $REPO $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi -L
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
"$PY" scripts/run_jax.py "$OUT" --rundir "$RUNDIR" --tree "$TREE" $START --nsteps "$NSTEPS" --cg2d-unroll 5 \
  --monitor-every 24 --frame-every 6 --snapshot-every $SNAP --checkpoint-every 720
