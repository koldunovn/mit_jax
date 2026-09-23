#!/bin/bash
#SBATCH --job-name=mitjax_shard4
#SBATCH --account=ab0995_gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/gpu_sharding/%j.out
# Plan Task 20 (tier 2): 1 vs P A100 on one node. sbatch -p gpu --constraint=a100_80 scripts/runs/gpu_sharding.sbatch [NSTEPS [P]]
# (gpu-devel nodes have 2 GPUs: sbatch -p gpu-devel --gpus=2 ... 24 2)
set -u
cd /home/a/a270088/MIT
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
nvidia-smi -L
/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python scripts/runs/gpu_sharding.py --nsteps ${1:-24} --nproc ${2:-4}
