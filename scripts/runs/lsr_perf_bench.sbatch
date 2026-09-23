#!/bin/bash
#SBATCH --job-name=mitjax_lsrp
#SBATCH --account=ab0995_gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/lsrperf/%j_a100.out
# LSOR sweep variants on one A100 (scripts/runs/lsr_perf_bench.py). MITJAX_TREE = source tree; bench arguments follow the script.
#   sbatch -p gpu-devel --export=ALL,MITJAX_TREE=... scripts/runs/lsr_perf_bench.sbatch --variants pallas,xla
set -u
cd ${MITJAX_TREE:-/home/a/a270088/MIT}
git log --oneline -1 2>/dev/null
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L
/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python scripts/runs/lsr_perf_bench.py "$@"
