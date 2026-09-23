#!/bin/bash
#SBATCH --job-name=iceadj
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/dev/iceadj/logs/iceadj-%j.out
# Sea-ice-only adjoint window (scripts/adjoint/seaice_window.py): one CPU process per argument string, each with
# $SLURM_CPUS_PER_TASK cores, all on one node, e.g.
#   sbatch [--time=01:00:00] scripts/adjoint/seaice_window.sbatch \
#     "--out $R/w06/fd_tight_HEFF --steps 6 --actions fd --tol tight --dirs HEFF" \
#     "--out $R/w06/fd_tight_AREA --steps 6 --actions fd --tol tight --dirs AREA" ...
# The LSR forward (a sequential line sweep) and the GMRES of its implicit derivative use ~1-4 cores; for many such
# processes: sbatch --ntasks=32 --cpus-per-task=4 ... (measured: tight forward 28 vs 21 s/step with 4 vs 16 cores).
# Every process appends to its --out/results.jsonl (one JSON line per result). Nothing is deleted.
set -u
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
eval "$("$PY" "$REPO/mitgcm_jax/paths.py" --sh)"
cd "$REPO"
export JAX_PLATFORMS=cpu
export PYTHONPYCACHEPREFIX=$MITJAX_WORK/dev/iceadj/pyc_${SLURM_JOB_ID:-local}
NC=${SLURM_CPUS_PER_TASK:-16}
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(date -Is), commit $(git rev-parse --short HEAD), $# processes x $NC cores"
i=0
for cfg in "$@"; do
  echo "=== process $i: $cfg"
  srun --exclusive -n1 -c$NC --mem-per-cpu=1700M $PY scripts/adjoint/seaice_window.py $cfg \
    > $MITJAX_WORK/dev/iceadj/logs/iceadj-${SLURM_JOB_ID:-local}-p$i.out 2>&1 &
  i=$((i+1))
done
wait
echo "all done $(date -Is)"
