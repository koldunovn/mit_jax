#!/bin/bash
# Build the MITgcm c66g Fortran reference for one ECCO v4r4 tree and tile layout (plan Task 4).
#
#   reference/build.sh TREE LAYOUT        TREE = full | ff (flux-forced);  LAYOUT = mpi96 | serial13
#   variants (env): JAXDUMP=1  instrumented with the per-substep dump shim (reference/jaxdump/instrument.py)
#                   GCOV=1     -O0 --coverage, for the branch-coverage run (docs/BRANCHES.md)
#   sbatch reference/jobs/build.sbatch TREE LAYOUT     (preferred: compiles on a shared node)
#
# Code dir = the tree's override code (+ SIZE.h for serial13), full packages.conf (autodiff/ctrl/ecco compiled:
# ALLOW_AUTODIFF changes forward branches, docs/OVERRIDES.md). Each build goes to a NEW directory
# /work/.../MIT/reference/build/<tree>_<layout>_<timestamp>; the executable is frozen as
# /work/.../MIT/reference/bin/mitgcmuv_<tree>_<layout>_<sha256[:12]> with a .txt provenance file.
# Nothing is deleted or overwritten.
set -euo pipefail

TREE=${1:?tree: full | ff}; LAYOUT=${2:?layout: mpi96 | serial13}
REPO=/home/a/a270088/MIT
ROOT=$REPO/MITgcm_c66g
V4=$REPO/"ECCO-v4-Configurations/ECCOv4 Release 4"
WORK=/work/ab0995/a270088/MIT/reference
case $TREE in full) CODE="$V4/code" ;; ff) CODE="$V4/flux-forced/code" ;; *) echo "bad tree $TREE"; exit 2 ;; esac
case $LAYOUT in mpi96) MPIFLAG=-mpi ;; serial13) MPIFLAG= ;; *) echo "bad layout $LAYOUT"; exit 2 ;; esac

GIT=$(command -v git)   # before module purge (git comes from a module)
GITINFO="MITgcm $($GIT -C "$ROOT" describe --tags --always)  V4r4-configs $($GIT -C "$REPO/ECCO-v4-Configurations" rev-parse --short HEAD 2>/dev/null || echo '?')  mitgcm-jax $($GIT -C "$REPO" rev-parse --short HEAD) dirty=$($GIT -C "$REPO" status --porcelain | wc -l)"
module purge
module load gcc/11.2.0-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0 netcdf-fortran/4.5.3-openmpi-4.1.2-gcc-11.2.0
unset NETCDF_ROOT NETCDF_HOME NETCDF_INC NETCDF_LIB NETCDF_INCDIR NETCDF_LIBDIR
export LEVANTE_NF_PREFIX=$(nf-config --prefix)
export LEVANTE_NC_LIBDIR=$(dirname "$(ldd "$LEVANTE_NF_PREFIX/lib/libnetcdff.so" | awk '/libnetcdf\.so/ {print $3}')")
export MPI_INC_DIR=$(dirname "$(which mpif90)")/../include

VARIANT=""
[ "${JAXDUMP:-0}" = 1 ] && VARIANT="${VARIANT}_jaxdump"
[ "${GCOV:-0}" = 1 ] && VARIANT="${VARIANT}_gcov"
BUILD=$WORK/build/${TREE}_${LAYOUT}${VARIANT}_$(date +%Y%m%d_%H%M%S)
[ -e "$BUILD" ] && { echo "exists: $BUILD"; exit 1; }
mkdir -p "$BUILD/code" "$BUILD/bld" "$WORK/bin"
cp "$CODE"/*.F "$CODE"/*.h "$CODE"/packages.conf "$BUILD/code/"
[ "$LAYOUT" = serial13 ] && cp "$REPO/reference/SIZE.h_13x90x90_serial" "$BUILD/code/SIZE.h"
DEVIATION=""
if [ "$TREE" = ff ]; then
  # DEVIATION from the published flux-forced tree (approved by Nikolay 2026-09-23, docs/OVERRIDES.md): its
  # mdsio_write_meta.F writes nrecords as I6 but it compiles c66g's mdsio_read_meta.F (I5), so it cannot read its
  # own pickups (403 -> 40). Use the full tree's reader (I6). I/O only; forward physics unchanged.
  cp "$V4/code/mdsio_read_meta.F" "$BUILD/code/"
  DEVIATION="ff: mdsio_read_meta.F from the full V4r4 tree (I6 nrecords reader)"
fi
cp "$REPO/reference/optfile_levante_gfortran" "$BUILD/"
if [ "${JAXDUMP:-0}" = 1 ]; then
  /work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python "$REPO/reference/jaxdump/instrument.py" "$TREE" "$BUILD/code" > "$BUILD/instrument.log"
fi
if [ "${GCOV:-0}" = 1 ]; then
  cat >> "$BUILD/optfile_levante_gfortran" <<'EOG'
#- GCOV variant (branch coverage): no optimisation, instrumented
FOPTIM='-O0'
FFLAGS="$FFLAGS --coverage"
CFLAGS="$CFLAGS --coverage"
LIBS="$LIBS --coverage"
EOG
fi

cd "$BUILD/bld"
{
  echo "tree $TREE  layout $LAYOUT  variant ${VARIANT:-none}  host $(hostname)  $(date -Is)"
  echo "$GITINFO"
  echo "netcdf: fortran $LEVANTE_NF_PREFIX  c $LEVANTE_NC_LIBDIR"
  [ -n "$DEVIATION" ] && echo "DEVIATION: $DEVIATION"
  gfortran --version | head -1; mpif90 --version | head -1; nf-config --version
} | tee ../provenance.txt

"$ROOT/tools/genmake2" -rootdir="$ROOT" -mods=../code -optfile=../optfile_levante_gfortran $MPIFLAG > ../genmake2.log 2>&1 \
  || { tail -30 ../genmake2.log; exit 1; }
grep -E "NetCDF-enabled" ../genmake2.log
# genmake2 silently DISABLES pkg/profiles when the NetCDF test fails: the reference must have the production set
if grep -q 'package is now DISABLED' ../genmake2.log; then grep -B2 -A2 DISABLED ../genmake2.log; exit 1; fi
make depend > ../make_depend.log 2>&1 || { tail -30 ../make_depend.log; exit 1; }
make -j "${SLURM_CPUS_PER_TASK:-8}" > ../make.log 2>&1 || { grep -iE "error" ../make.log | head -30; exit 1; }

# the compiled package list and CPP defines are the build's truth: keep them next to the binary
cp PACKAGES_CONFIG.h AD_CONFIG.h ../ 2>/dev/null || true
SHA=$(sha256sum mitgcmuv | cut -c1-64)
BIN=$WORK/bin/mitgcmuv_${TREE}_${LAYOUT}${VARIANT}_${SHA:0:12}
[ -e "$BIN" ] || cp mitgcmuv "$BIN"
chmod a-w "$BIN"
{ cat ../provenance.txt; echo "sha256 $SHA"; echo "build $BUILD"; grep -E "^FFLAGS|^FOPTIM|^NOOPTFILES|^LIBS" Makefile
  echo "packages: $(grep -E '^#define ALLOW_' PACKAGES_CONFIG.h | awk '{print $2}' | sort | tr '\n' ' ')"; } > "$BIN.txt"
echo "BUILT $BIN"
