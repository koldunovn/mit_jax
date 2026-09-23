#!/bin/bash
#SBATCH --job-name=mitjax_lsrp
#SBATCH --account=mh1571
#SBATCH --partition=dolpung
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/lsrperf/%j_gh200.out
# LSOR sweep variants on one GH200 (scripts/runs/lsr_perf_bench.py). MITJAX_TREE = source tree; bench arguments follow the script.
#   sbatch --export=ALL,MITJAX_TREE=... scripts/dolpung/lsr_perf_bench.sbatch --variants pallas,xla
set -u
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
source "$REPO/scripts/dolpung/env_dolpung.sh"                      # sets PY (aarch64 venv)
cd "$REPO"
git log --oneline -1 2>/dev/null
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L; uname -m
$PY scripts/runs/lsr_perf_bench.py "$@"
