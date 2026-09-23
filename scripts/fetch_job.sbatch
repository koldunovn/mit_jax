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
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
cd "$REPO"
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is), groups: $*"
exec "$PY" -u scripts/fetch_eccov4r4.py fetch "$@"
