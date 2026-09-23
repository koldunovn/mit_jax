#!/bin/bash
#SBATCH --job-name=eccov4r4_extract
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/ab0995/a270088/MIT/data/eccov4r4/logs/extract-%j.out
#
# One streaming pass over a downloaded archive: index every member, unpack non-yearly files and the requested
# years only (the archive itself is kept; later years are unpacked from it when needed). Never overwrites.
#   sbatch [--dependency=afterok:<fetch job>] scripts/extract_job.sbatch ARCHIVE [--years 1992 ...] [--exclude GLOB ...]
set -u
cd /home/a/a270088/MIT
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is): extract $*"
exec /work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python -u scripts/fetch_eccov4r4.py extract "$@"
