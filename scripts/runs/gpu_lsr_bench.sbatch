#!/bin/bash
#SBATCH --job-name=mitjax_lsrb
#SBATCH --account=ab0995_gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/gpu_lsr_bench/%j.out
# M2.4 cost check (SEAICE_DYNSOLVER on one GPU). sbatch -p gpu-devel scripts/runs/gpu_lsr_bench.sbatch
set -u
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
cd "$REPO"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L
"$PY" scripts/runs/gpu_lsr_bench.py --repeat 3
