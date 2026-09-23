#!/bin/bash
#SBATCH --job-name=mitjax_adj21
#SBATCH --account=ab0995_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ab0995/a270088/MIT/dev/adjoint21/logs/adj21-%j.out
# Plan Task 21: multi-week gradient on LLC90 (one A100-80, default XLA flags, sum_unroll=5).
# Every argument is passed to scripts/adjoint/multiweek_grad.py, e.g.
#   sbatch scripts/adjoint/multiweek_grad.sbatch --out /work/ab0995/a270088/MIT/runs/adjoint/w7_exact \
#          --days 7 --mode exact --actions grad,screen,tl
# The initial-state cache is built once on CPU with the gate flags (bitwise the Fortran start-of-run state):
#   XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp" JAX_PLATFORMS=cpu \
#     python scripts/adjoint/multiweek_grad.py --out <dir> --actions cache
set -u
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
eval "$("$PY" "$REPO/mitgcm_jax/paths.py" --sh)"
cd "$REPO"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONPYCACHEPREFIX=$MITJAX_WORK/dev/adjoint21/pycache_${SLURM_JOB_ID:-local}
nvidia-smi -L
if [ "${1:-}" = "--multi" ]; then
  # one process per GPU: sbatch --gpus=4 --cpus-per-task=64 --mem=480G multiweek_grad.sbatch --multi "<args 1>" ...
  # (one Slurm job for several configurations: the account's running-job limit counts jobs, not GPUs)
  shift
  i=0
  for cfg in "$@"; do
    echo "=== GPU $i: $cfg"
    LOG=$MITJAX_WORK/dev/adjoint21/logs/adj21-${SLURM_JOB_ID}-gpu$i.out
    if [ "${cfg%% *}" = "PYTEST" ]; then   # "PYTEST <pytest args>": a test on this GPU (tier 2)
      CUDA_VISIBLE_DEVICES=$i $PY -m pytest -q -s -p no:cacheprovider ${cfg#PYTEST } > $LOG 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$i $PY scripts/adjoint/multiweek_grad.py $cfg > $LOG 2>&1 &
    fi
    i=$((i+1))
  done
  wait
  echo "all done"
else
  $PY scripts/adjoint/multiweek_grad.py "$@"
  echo "exit $?"
fi
