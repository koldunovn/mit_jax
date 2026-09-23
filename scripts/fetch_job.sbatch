#!/bin/bash
#SBATCH --job-name=eccov4r4_fetch
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/ab0995/a270088/MIT/data/eccov4r4/logs/fetch-%j.out
#
# Download ECCO v4r4 groups from PO.DAAC as a batch job (survives logouts; shared nodes have internet).
#   sbatch scripts/fetch_job.sbatch GROUP [GROUP ...]      e.g.  sbatch scripts/fetch_job.sbatch flux_forcing
# Resumes an interrupted download from its .part file; verifies the published checksum; never deletes.
# Run at most ONE job per group at a time: two jobs on the same group would append to the same .part file.
set -u
cd /home/a/a270088/MIT
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is), groups: $*"
exec /work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python -u scripts/fetch_eccov4r4.py fetch "$@"
