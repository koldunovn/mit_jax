#!/bin/bash
#SBATCH --job-name=mitjax_whl
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/dolpung/%j_fetch_wheels.out
# Download the aarch64 (dolpung GH200) wheels of the pinned env set (constraints.txt) on an x86 node with internet;
# (pip --platform does not expand manylinux_2_X to older X: every tag 2_17..2_34 is listed; RHEL 9 = glibc 2.34)
# scripts/dolpung/make_env.sbatch installs them offline on a dolpung node. sbatch scripts/dolpung/fetch_wheels.sbatch
set -eu
WH=/work/ab0995/a270088/MIT/envs/wheels_aarch64_cp312
mkdir -p $WH
/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/pip download --no-cache-dir -d $WH \
    --only-binary=:all: --implementation cp --python-version 3.12 --abi cp312 --abi abi3 --abi none \
    $(for v in $(seq 17 34); do echo --platform manylinux_2_${v}_aarch64; done) --platform manylinux2014_aarch64 \
    -c /home/a/a270088/MIT/constraints.txt \
    "jax[cuda12]==0.10.1" numpy scipy netCDF4 pytest
ls $WH | wc -l; du -sh $WH
