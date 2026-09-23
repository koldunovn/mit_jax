# Fortran reference: builds and runs (plan Task 4)

Everything heavy lives in `/work/ab0995/a270088/MIT/reference/` (`bin/`, `build/`, `runs/`, `logs/`). Nothing there is
deleted or overwritten; every build and run gets a new directory.

## Builds (2026-09-23)

gfortran 11.2.0 (Spack), OpenMPI 4.1.2, netCDF-Fortran 4.5.3 (netCDF-C 4.8.1), optfile `reference/optfile_levante_gfortran`
(ECCO's Docker gfortran optfile + `-ffp-contract=off`; `-O3 -funroll-loops`; `ini_masks_etc.F`, `seaice_growth.F` at
`-O0`). Full V4r4 `packages.conf` in both trees (`ALLOW_AUTODIFF`, `ALLOW_PROFILES` etc. compiled; forward build:
`ALLOW_ADJOINT_RUN` undefined). Built by `sbatch reference/jobs/build.sbatch TREE LAYOUT` at mitgcm-jax `efe0333`.

| binary (`reference/bin/`) | tree | layout | use |
|---|---|---|---|
| `mitgcmuv_ff_serial13_8668b6b46487` | flux-forced | 1 process, 13 tiles of 90×90 | per-substep oracle (Task 5) |
| `mitgcmuv_ff_mpi96_6e7d5bac461a` | flux-forced | 96 ranks × 30×30 (production) | 1-month / 1-year references |
| `mitgcmuv_full_serial13_f24b2b6eca38` | full V4r4 | 1 process, 13 tiles of 90×90 | per-substep oracle (M2) |
| `mitgcmuv_full_mpi96_128048fa1823` | full V4r4 | 96 ranks × 30×30 (production) | 11-step PO.DAAC check, 1-month / 1-year |

Each binary has a `.txt` with provenance, sha256, compiler flags and the compiled package list. Earlier binaries in
`bin/` from 07:35–07:38 are superseded: the first set was built without NetCDF (`pkg/profiles` silently disabled by
genmake2 — `build.sh` now fails on that), the second from an uncommitted tree.

Build notes: gfortran ≥ 10 needs `-fallow-argument-mismatch` for c66g; Levante's netcdf-fortran prefix does not contain
`libnetcdf`, and the `nc-config` on `PATH` belongs to a different netCDF-C than the one `libnetcdff` links — the
library directory is taken from `ldd libnetcdff.so` and set as rpath.

## Run directories (`reference/make_rundir.py`)

`make_rundir.py TREE LAYOUT NAME --nsteps N [--monitor S]` copies the tree's production namelists, applies the
standing overrides (printed, read back, and written to `OVERRIDES.txt`), links every input derived by
`scripts/audit_run_inputs.py`, records `INPUTS.sha256`, and refuses (creating nothing) if a required input is missing.
Standing overrides: `useECCO=F`, `useProfiles=F`, `useCAL=T` (cost-function packages; their observation inputs are
not staged) — assumed forward-neutral, to be confirmed by one same-binary run with them on; `nTimeSteps`,
`monitorFreq`. serial13 also swaps in `reference/data.exch2_13x90x90` (no `blankList`: every 90×90 tile is wet).

Run: `sbatch -p shared --ntasks=1 --mem=32G reference/jobs/run.sbatch RUNDIR` (serial13) or
`sbatch -p compute --ntasks=96 reference/jobs/run.sbatch RUNDIR` (mpi96). Success requires exit 0 and
"Execution ended Normally" in `STDOUT.0000`.

Timing: the model clock is 1992-01-01 13:00 at iteration 1 (`startDate` 12:00, `nIter0=1`), so `--nsteps 11` ends at
1992-01-02 00:00 — the first PO.DAAC snapshot.

## Runs

(none yet — waiting for the forcing archives, docs/DATA.md)
