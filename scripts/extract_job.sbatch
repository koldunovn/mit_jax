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
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
cd "$REPO"
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is): extract $*"
exec "$PY" -u scripts/fetch_eccov4r4.py extract "$@"
