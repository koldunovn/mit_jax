#!/bin/bash
#SBATCH --job-name=mitjax_tier1
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00          # short limit: backfills sooner; tier-1 budget is 10 min
#SBATCH --output=/work/ab0995/a270088/MIT/runs/tier1/slurm-%j.out
#
# Tier 1: every smoke + tier1 test on one CPU compute node (budget: < 10 min, < 100 tests).
#   sbatch scripts/run_tier1.sbatch            (the slurm log dir must exist; elsewhere: sbatch -A ... -p ... --output=...)
# Pass/fail is decided by scripts/check_pytest_report.py on the JUnit report AND pytest's exit code.
# Each job writes into its own new OUT_DIR and never deletes anything.
set -u

# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
eval "$("$PY" "$REPO/mitgcm_jax/paths.py" --sh)"
OUT_DIR=$MITJAX_RUNS/tier1/${SLURM_JOB_ID:?run this through sbatch}
if [ -e "$OUT_DIR" ]; then echo "FAIL: OUT_DIR $OUT_DIR already exists"; exit 1; fi
mkdir -p "$OUT_DIR"

cd "$REPO"
{
  echo "job      $SLURM_JOB_ID on $(hostname), $(date -Is)"
  echo "commit   $(git rev-parse HEAD)"
  echo "dirty    $(git status --porcelain | wc -l) paths"
  echo "python   $PY"
} | tee "$OUT_DIR/provenance.txt"
git status --porcelain > "$OUT_DIR/git_status.txt"
git diff > "$OUT_DIR/git_diff.patch"

# XLA_FLAGS (4 fake devices) is set by conftest.py. No persistent compilation cache (user decision).
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

T0=$(date +%s)
"$PY" -m pytest -m "smoke or tier1" -q -rfEs \
    --basetemp="$OUT_DIR/pytest_tmp" --junitxml="$OUT_DIR/report.xml" --durations=15 \
    2>&1 | tee "$OUT_DIR/pytest.log"
PYTEST_RC=${PIPESTATUS[0]}
WALL=$(( $(date +%s) - T0 ))

echo "pytest exit code $PYTEST_RC, wall ${WALL} s" | tee -a "$OUT_DIR/provenance.txt"
"$PY" scripts/check_pytest_report.py "$OUT_DIR/report.xml" --max-tests 100 | tee "$OUT_DIR/verdict.txt"
CHECK_RC=${PIPESTATUS[0]}
if [ "$WALL" -gt 600 ]; then
  echo "WARNING: tier 1 took ${WALL} s, over the 600 s budget" | tee -a "$OUT_DIR/verdict.txt"
fi

if [ "$PYTEST_RC" -ne 0 ] || [ "$CHECK_RC" -ne 0 ]; then
  echo "TIER1 FAIL (pytest rc $PYTEST_RC, report check rc $CHECK_RC) -> $OUT_DIR"
  exit 1
fi
echo "TIER1 PASS -> $OUT_DIR"
