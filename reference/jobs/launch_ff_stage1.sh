#!/bin/bash
# Flux-forced reference runs, stage 1 (plan Tasks 4-5): short runs that need only 1992 forcing.
#   1-day mpi96 twice (bitwise reproducibility), 1-day serial13 (tile-layout spread), 3-step jaxdump oracle,
#   1-day gcov run (branch coverage -> docs/BRANCHES.md). Every run dir is new; make_rundir refuses missing inputs.
set -euo pipefail
# sbatch from inside a job inherits SLURM_MEM_PER_*; srun then refuses ("mutually exclusive"), 2026-09-23
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
cd /home/a/a270088/MIT
PY=/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python
R=/work/ab0995/a270088/MIT/reference/runs
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
