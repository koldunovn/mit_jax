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
cd ${MITJAX_TREE:-/home/a/a270088/MIT}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L
/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python scripts/runs/gpu_lsr_bench.py --repeat 3
