#!/bin/bash
#SBATCH --job-name=mitgcm_build
#SBATCH --partition=shared
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ab0995/a270088/MIT/reference/logs/build-%j.out
# sbatch reference/jobs/build.sbatch TREE LAYOUT   (see reference/build.sh)
exec /home/a/a270088/MIT/reference/build.sh "$@"
