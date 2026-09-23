#!/bin/bash
#SBATCH --job-name=mitjax_gh200
#SBATCH --account=mh1571
#SBATCH --partition=dolpung
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/dolpung/%j_bench.out
# GH200 vs A100 on the same code: (1) SEAICE_DYNSOLVER cost (scripts/runs/gpu_lsr_bench.py; A100-40: 183 ms/sweep,
# job 27647012); (2) production ff config, 240 steps from the pickup with cg2d unroll 5 (A100-80: ~0.17 s/step).
# MITJAX_TREE = source tree (a clean worktree of a commit). sbatch scripts/dolpung/bench.sbatch
set -u
source /home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh
cd ${MITJAX_TREE:-/home/a/a270088/MIT}
git log --oneline -1 2>/dev/null
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L; uname -m
$PY scripts/runs/gpu_lsr_bench.py --repeat 3
$PY scripts/run_jax.py /work/ab0995/a270088/MIT/runs/dolpung/prod240_${SLURM_JOB_ID} \
  --rundir /work/ab0995/a270088/MIT/reference/runs/ref_ff_serial13_1day --init-oracle pickup --nsteps 240 \
  --monitor-every 24 --cg2d-unroll 5
