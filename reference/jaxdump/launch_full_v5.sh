#!/bin/bash
#SBATCH --job-name=mitgcm_launch
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/reference/logs/launch-%j.out
# M2.0 full-tree dump oracle (jaxdump v5: EXF bulk + sea-ice stages, reference/jaxdump/SUBSTEPS.md).
#   sbatch --dependency=afterok:<full jaxdump build job> reference/jaxdump/launch_full_v5.sh
# Makes (new run dirs; make_rundir refuses existing names) and submits:
#   smoke_full_v5_plain_2steps   plain full serial13 binary (the one of ref_full_serial13_1day), 2 steps
#   smoke_full_v5_jd_off_2steps  newest full serial13 jaxdump binary, dumps off   } invisibility pair: output must be
#   smoke_full_v5_jd_on_2steps   same binary, JAXDUMP_STEPS=1:2                    } byte-identical to the plain run
#   ref_full_serial13_jaxdump_v5_3steps  same binary, JAXDUMP_STEPS=1:2:3 -> runs.json "full_jaxdump_v5"
set -euo pipefail
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU   # sbatch from inside a job (see launch_ff_stage1.sh)
# repository: $MITJAX_TREE, else the checkout holding this script (under sbatch $0 is a spool copy: scontrol knows
# the original). Paths: mitgcm_jax/paths.py (MITJAX_WORK, ...); python: $MITJAX_PYTHON.
SELF=$0; [ -f "$(dirname "$SELF")/../../mitgcm_jax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p')
REPO=${MITJAX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitgcm_jax/paths.py" ] || { echo "FAIL: no mitgcm_jax repository at $REPO (set MITJAX_TREE)"; exit 1; }
PY=${MITJAX_PYTHON:-/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python}
eval "$("$PY" "$REPO/mitgcm_jax/paths.py" --sh)"
cd "$REPO"
R=$MITJAX_REFERENCE_RUNS
PLAIN=$MITJAX_REFERENCE/bin/mitgcmuv_full_serial13_f24b2b6eca38
mk() { $PY reference/make_rundir.py "$@" | grep -E "RUNDIR|REFUSED"; }
mk full serial13 smoke_full_v5_plain_2steps  --nsteps 2 --monitor 3600 --binary $PLAIN
mk full serial13 smoke_full_v5_jd_off_2steps --nsteps 2 --monitor 3600 --variant _jaxdump
mk full serial13 smoke_full_v5_jd_on_2steps  --nsteps 2 --monitor 3600 --variant _jaxdump
mk full serial13 ref_full_serial13_jaxdump_v5_3steps --nsteps 3 --monitor 3600 --variant _jaxdump
mkdir "$R/smoke_full_v5_jd_on_2steps/jaxdump" "$R/ref_full_serial13_jaxdump_v5_3steps/jaxdump"
S="sbatch -p shared --ntasks=1 --mem=32G"
$S --time=00:40:00 reference/jobs/run.sbatch $R/smoke_full_v5_plain_2steps
$S --time=00:40:00 reference/jobs/run.sbatch $R/smoke_full_v5_jd_off_2steps
$S --time=00:40:00 --export=ALL,JAXDUMP_DIR=$R/smoke_full_v5_jd_on_2steps/jaxdump,JAXDUMP_STEPS=1:2 \
  reference/jobs/run.sbatch $R/smoke_full_v5_jd_on_2steps
$S --time=01:00:00 --export=ALL,JAXDUMP_DIR=$R/ref_full_serial13_jaxdump_v5_3steps/jaxdump,JAXDUMP_STEPS=1:2:3 \
  reference/jobs/run.sbatch $R/ref_full_serial13_jaxdump_v5_3steps
