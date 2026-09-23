#!/bin/bash
# Flux-forced reference runs, stage 1 (plan Tasks 4-5): short runs that need only 1992 forcing.
#   1-day mpi96 twice (bitwise reproducibility), 1-day serial13 (tile-layout spread), 3-step jaxdump oracle,
#   1-day gcov run (branch coverage -> docs/BRANCHES.md). Every run dir is new; make_rundir refuses missing inputs.
set -euo pipefail
# sbatch from inside a job inherits SLURM_MEM_PER_*; srun then refuses ("mutually exclusive"), 2026-09-23
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
eval "$("$PY" "$REPO/mitgcm_jax/paths.py" --sh)"
cd "$REPO"
R=$MITJAX_REFERENCE_RUNS
mk() { $PY reference/make_rundir.py "$@" | grep -E "RUNDIR|REFUSED"; }
mk ff mpi96    ref_ff_mpi96_1day_a    --nsteps 24 --monitor 3600
mk ff mpi96    ref_ff_mpi96_1day_b    --nsteps 24 --monitor 3600
mk ff serial13 ref_ff_serial13_1day   --nsteps 24 --monitor 3600
mk ff serial13 ref_ff_serial13_jaxdump_3steps --nsteps 3 --monitor 3600 --variant _jaxdump
mk ff serial13 ref_ff_serial13_gcov_1day --nsteps 24 --monitor 3600 --variant _gcov
mkdir "$R/ref_ff_serial13_jaxdump_3steps/jaxdump"
sbatch -p compute --ntasks=96 --time=00:30:00 reference/jobs/run.sbatch $R/ref_ff_mpi96_1day_a
sbatch -p compute --ntasks=96 --time=00:30:00 reference/jobs/run.sbatch $R/ref_ff_mpi96_1day_b
sbatch -p shared --ntasks=1 --mem=32G --time=02:00:00 reference/jobs/run.sbatch $R/ref_ff_serial13_1day
sbatch -p shared --ntasks=1 --mem=32G --time=01:00:00 \
  --export=ALL,JAXDUMP_DIR=$R/ref_ff_serial13_jaxdump_3steps/jaxdump,JAXDUMP_STEPS=1:2:3 \
  reference/jobs/run.sbatch $R/ref_ff_serial13_jaxdump_3steps
sbatch -p shared --ntasks=1 --mem=32G --time=04:00:00 reference/jobs/run.sbatch $R/ref_ff_serial13_gcov_1day
