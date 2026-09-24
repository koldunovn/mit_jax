#!/bin/bash
#SBATCH --job-name=mitjax_m2acc
#SBATCH --account=mh1571
#SBATCH --partition=dolpung
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/adjoint_m2/logs/m2-%j.out
# M2 adjoint acceptance (plan Task 22): scripts/adjoint/fullgrad.py on the dolpung GH200 partition, from the checkout
# holding this script (or $MITJAX_TREE). #SBATCH --output is fixed at submission: pass --output=... to redirect it.
#   sbatch scripts/adjoint/fullgrad_dolpung.sbatch --out DIR --days 7 --mode ecco --actions fwdcheck,grad --repeats 3
#   sbatch --gpus=4 --cpus-per-task=288 --mem=800G scripts/adjoint/fullgrad_dolpung.sbatch --multi "<args 1>" ...
# (--multi: one process per GPU; fullgrad.py binds each to its GH200's Grace socket when all 288 CPUs are allocated;
#  one Slurm job counts once against MaxJobs;
#  "<args a> +++ <args b>" runs a then b on the same GPU, one process each: one heavy stage per process)
#   sbatch --gpus=4 --cpus-per-task=288 --mem=800G scripts/adjoint/fullgrad_dolpung.sbatch --out DIR --nproc 4 ...
# (one process on 4 GPUs: the sharded model)
set -u
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: the dolpung venv (env_dolpung.sh).
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
source "$REPO/scripts/dolpung/env_dolpung.sh"
eval "$("$PY" "$REPO/mitgcm_jax/paths.py" --sh)"
cd "$REPO"
cat .git 2>/dev/null
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONPYCACHEPREFIX=$MITJAX_WORK/dev/m2acc/pyc_${SLURM_JOB_ID:-local}
LOGDIR=$MITJAX_RUNS/adjoint_m2/logs
mkdir -p "$LOGDIR"
nvidia-smi -L; nvidia-smi --query-gpu=index,memory.total --format=csv; uname -m
if [ "${1:-}" = "--multi" ]; then
  shift
  i=0
  for cfg in "$@"; do
    echo "=== GPU $i: $cfg"
    LOG=$LOGDIR/m2-${SLURM_JOB_ID}-gpu$i.out
    # no numactl binding: with --cpus-per-task < the whole node the job's cpuset need not cover NUMA node $i
    # (numactl --cpunodebind=$i then fails with "Invalid argument" and the process never starts)
    BIND=""
    (
      IFS=$'\n'
      for sub in $(echo "$cfg" | sed 's/ +++ /\n/g'); do
        echo "--- $(date +%T) GPU $i: $sub"
        unset IFS
        CUDA_VISIBLE_DEVICES=$i $BIND $PY scripts/adjoint/fullgrad.py $sub
        echo "--- $(date +%T) exit $?"
        IFS=$'\n'
      done
    ) > $LOG 2>&1 &
    i=$((i+1))
  done
  wait
  echo "all done"
else
  $PY scripts/adjoint/fullgrad.py "$@"
  echo "exit $?"
fi
