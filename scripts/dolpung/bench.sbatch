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
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
source "$REPO/scripts/dolpung/env_dolpung.sh"                      # sets PY (aarch64 venv)
eval "$("$PY" "$REPO/mitgcm_jax/paths.py" --sh)"
cd "$REPO"
git log --oneline -1 2>/dev/null
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L; uname -m
$PY scripts/runs/gpu_lsr_bench.py --repeat 3
$PY scripts/run_jax.py $MITJAX_RUNS/dolpung/prod240_${SLURM_JOB_ID} \
  --rundir $MITJAX_REFERENCE_RUNS/ref_ff_serial13_1day --init-oracle pickup --nsteps 240 \
  --monitor-every 24 --cg2d-unroll 5
