#!/bin/bash
#SBATCH --job-name=mitjax_armenv
#SBATCH --account=mh1571
#SBATCH --partition=dolpung
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ab0995/a270088/MIT/runs/dolpung/%j_make_env.out
# Build the aarch64 venv offline from scripts/dolpung/fetch_wheels.sbatch's wheels, then gate it (J0: GPU visible,
# a jitted matmul, x64). sbatch --dependency=afterok:<fetch job> scripts/dolpung/make_env.sbatch
set -eu
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
source "$REPO/scripts/dolpung/env_dolpung.sh"                      # sets MITJAX_ARM_VENV, PY
WORK=${MITJAX_WORK:-/work/ab0995/a270088/MIT}
WH=${MITJAX_WHEELS:-$WORK/envs/wheels_aarch64_cp312}
uname -m; nvidia-smi -L
/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2/python-3.12.13-hzcoloj/bin/python3.12 -m venv $MITJAX_ARM_VENV
$PY -m pip install --no-cache-dir --no-index --find-links $WH -c "$REPO/constraints.txt" \
    "jax[cuda12]==0.10.1" numpy scipy netCDF4 pytest
$PY -m pip freeze > $WORK/runs/dolpung/pip_freeze_arm.txt
$PY - <<'PY'
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print(jax.__version__, jax.devices())
x = jnp.ones((2000, 2000))
print("matmul", float(jax.jit(lambda a: (a @ a).sum())(x)), x.dtype)
PY
