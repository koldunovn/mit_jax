# env_cabeus.sh -- site settings for NASA NAS Cabeus, sourced by scripts/cabeus/*.pbs (docs/RUN_ONE_YEAR.md, section
# "NASA NAS Cabeus"). Edit the defaults below or export the variables before qsub (qsub -v VAR=...).
# GH200 nodes are aarch64 and A100 nodes x86_64: each architecture needs its own Python environment.
export MITJAX_WORK=${MITJAX_WORK:-/nobackup/$USER/mitjax}                      # data, run directories, output
case $(uname -m) in
  aarch64) export MITJAX_PYTHON=${MITJAX_PYTHON_GH200:-/nobackup/$USER/envs/mitgcm-jax-gh200/bin/python} ;;
  x86_64)  export MITJAX_PYTHON=${MITJAX_PYTHON_A100:-/nobackup/$USER/envs/mitgcm-jax-a100/bin/python} ;;
  *)       echo "env_cabeus.sh: unknown architecture $(uname -m)"; exit 1 ;;
esac
export MITJAX_NEREUS_PYTHON=${MITJAX_NEREUS_PYTHON:-/nobackup/$USER/envs/nereus/bin/python}   # plots only
[ -x "$MITJAX_PYTHON" ] || { echo "env_cabeus.sh: no python at $MITJAX_PYTHON (install: docs/RUN_ONE_YEAR.md)"; exit 1; }
