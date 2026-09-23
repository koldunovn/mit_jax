#!/bin/bash
#SBATCH --job-name=mitgcm_ref
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/ab0995/a270088/MIT/reference/logs/run-%j.out
#
# Run a reference run directory made by reference/make_rundir.py.
#   serial13: sbatch -p shared --ntasks=1 --cpus-per-task=1 --mem=32G reference/jobs/run.sbatch RUNDIR
#   mpi96:    sbatch -p compute --ntasks=96                              reference/jobs/run.sbatch RUNDIR
# Success = exit 0 AND "Execution ended Normally" in STDOUT.0000 (both checked). Never deletes anything.
set -u
RUN=${1:?run dir}
cd "$RUN" || exit 1
[ -e STDOUT.0000 ] && { echo "$RUN already ran (STDOUT.0000 exists); make a new run dir"; exit 1; }
module purge
module load gcc/11.2.0-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0 netcdf-fortran/4.5.3-openmpi-4.1.2-gcc-11.2.0
ulimit -s unlimited
LAYOUT=$(awk '/^tree/ {print $4}' OVERRIDES.txt)
echo "job $SLURM_JOB_ID $(hostname) $(date -Is) layout $LAYOUT binary $(readlink -f mitgcmuv)" | tee run_provenance.txt
T0=$(date +%s)
if [ "$LAYOUT" = mpi96 ]; then
  srun -n 96 ./mitgcmuv; RC=$?
else
  ./mitgcmuv > STDOUT.0000 2> STDERR.0000; RC=$?
fi
echo "exit $RC wall $(( $(date +%s) - T0 )) s" | tee -a run_provenance.txt
grep -q "Execution ended Normally" STDOUT.0000 || { echo "RUN FAILED (no normal end)"; tail -20 STDOUT.0000 STDERR.0000 2>/dev/null; exit 1; }
[ "$RC" -eq 0 ] || { echo "RUN FAILED rc=$RC"; exit 1; }
echo "RUN OK $RUN"
