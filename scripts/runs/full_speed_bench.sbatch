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
if [ "${MITJAX_ENV:-}" = dolpung ]; then
  source /home/a/a270088/MIT/scripts/dolpung/env_dolpung.sh
else
  PY=/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python
fi
cd ${MITJAX_TREE:-/home/a/a270088/MIT}
git log --oneline -1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L
$PY scripts/runs/full_speed_bench.py --nsteps ${1:-8760} --nproc ${2:-1}
