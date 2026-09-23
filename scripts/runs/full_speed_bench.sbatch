#!/bin/bash
#SBATCH --job-name=mitjax_speed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/speed/%j.out
# Full V4r4 forward speed (one year by default) on 1 or 4 GPUs:
#   A100-80:  sbatch -A ab0995_gpu -p gpu --constraint=a100_80 --gpus=N scripts/runs/full_speed_bench.sbatch NSTEPS N
#   GH200:    MITJAX_ENV=dolpung sbatch -A mh1571 -p dolpung --gpus=N scripts/runs/full_speed_bench.sbatch NSTEPS N
set -u
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
if [ "${MITJAX_ENV:-}" = dolpung ]; then
  source "$REPO/scripts/dolpung/env_dolpung.sh"
else
  PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
fi
cd "$REPO"
git log --oneline -1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L
$PY scripts/runs/full_speed_bench.py --nsteps ${1:-8760} --nproc ${2:-1}
