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
# M2 adjoint acceptance (plan Task 22): scripts/adjoint/fullgrad.py on the dolpung GH200 partition, from THIS checkout
# (MITJAX_TREE, default the wt_m2acc worktree), never from the main tree.
#   sbatch scripts/adjoint/fullgrad_dolpung.sbatch --out DIR --days 7 --mode ecco --actions fwdcheck,grad --repeats 3
#   sbatch --gpus=4 --cpus-per-task=64 --mem=800G scripts/adjoint/fullgrad_dolpung.sbatch --multi "<args 1>" ...
# (--multi: one process per GPU, each bound to its GH200's NUMA node; one Slurm job counts once against MaxJobs;
#  "<args a> +++ <args b>" runs a then b on the same GPU, one process each: one heavy stage per process)
#   sbatch --gpus=4 --cpus-per-task=64 --mem=800G scripts/adjoint/fullgrad_dolpung.sbatch --out DIR --nproc 4 ...
# (one process on 4 GPUs: the sharded model)
set -u
TREE=${MITJAX_TREE:-/work/ab0995/a270088/MIT/dev/wt_m2acc}
source /home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh
cd $TREE
cat .git
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONPYCACHEPREFIX=/work/ab0995/a270088/MIT/dev/m2acc/pyc_${SLURM_JOB_ID:-local}
LOGDIR=/work/ab0995/a270088/MIT/runs/adjoint_m2/logs
nvidia-smi -L; nvidia-smi --query-gpu=index,memory.total --format=csv; uname -m
if [ "${1:-}" = "--multi" ]; then
  shift
  i=0
  for cfg in "$@"; do
    echo "=== GPU $i: $cfg"
    LOG=$LOGDIR/m2-${SLURM_JOB_ID}-gpu$i.out
    BIND=""
    command -v numactl >/dev/null && BIND="numactl --cpunodebind=$i --preferred=$i"
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
