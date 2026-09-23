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
| `mitgcmuv_ff_serial13_ae16b9e60261` | flux-forced (+ I6 meta reader) | 1 process, 13 tiles of 90×90 | spread runs, restarts |
| `mitgcmuv_ff_serial13_jaxdump_325cc5e9d0ef` | same + dump shim | 1 process, 13 tiles | per-substep oracle (Task 5) |
| `mitgcmuv_ff_serial13_gcov_304441fbebdb` | same, -O0 --coverage | 1 process, 13 tiles | branch coverage |
| `mitgcmuv_ff_mpi96_233c705dd0fe` | flux-forced (+ I6 meta reader) | 96 ranks × 30×30 (production) | 1-month / 1-year references |
| `mitgcmuv_full_serial13_f24b2b6eca38` | full V4r4 | 1 process, 13 tiles of 90×90 | per-substep oracle (M2) |
| `mitgcmuv_full_mpi96_128048fa1823` | full V4r4 | 96 ranks × 30×30 (production) | 11-step PO.DAAC check, 1-month / 1-year |
| `mitgcmuv_ff_mpi13_36a17954ad91` | flux-forced (+ I6 meta reader) | 13 ranks × one 90×90 tile | bitwise = serial13, ~5x faster (section mpi13) |
| `mitgcmuv_full_mpi13_edbe8101af15` | full V4r4 | 13 ranks × one 90×90 tile | bitwise = serial13: long JAX twins (section mpi13) |

Flux-forced binaries include the approved I6 `.meta`-reader deviation (docs/OVERRIDES.md); the earlier
`ff_*_8668b6b46487`/`6e7d5bac461a` builds (without it) cannot restart and are superseded.

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
`monitorFreq`. serial13 and mpi13 also swap in `reference/data.exch2_13x90x90` (no `blankList`: every 90×90 tile is
wet).

Run: `sbatch -p shared --ntasks=1 --mem=32G reference/jobs/run.sbatch RUNDIR` (serial13) or
`sbatch -p compute --ntasks=96 reference/jobs/run.sbatch RUNDIR` (mpi96) or
`sbatch -p shared --ntasks=13 --mem=32G reference/jobs/run.sbatch RUNDIR` (mpi13). Success requires exit 0 and
"Execution ended Normally" in `STDOUT.0000`.

Timing: the model clock is 1992-01-01 13:00 at iteration 1 (`startDate` 12:00, `nIter0=1`), so `--nsteps 11` ends at
1992-01-02 00:00 — the first PO.DAAC snapshot.

## Runs

All in `/work/ab0995/a270088/MIT/reference/runs/`. "smoke" runs have `useEXF=F, useCTRL=F, useSMOOTH=F,
geothermalFile=' '` (no forcing staged yet): machinery checks, not references.

| run | binary | steps | result |
|---|---|---|---|
| `smoke_noforcing_ff_serial13` | `ff_serial13_8668b6b46487` | 2 | ends normally; grid, pickup, 13-tile exch2 OK |
| `smoke_noforcing_ff_mpi96` | `ff_mpi96_6e7d5bac461a` | 2 | ends normally; vs serial13: max rel. diff of %MON 8e-10 after 2 steps (cg2d residual 6e-10) — tile-order sums |
| `smoke_noforcing_ff_s13_jd_off2`, `_off3` | `ff_serial13_jaxdump_f6b94ea956f9`, `_390b5db49021` (27 stages), dumps off | 2 | T,S,Eta,U,V,W,PH and every %MON line byte-identical to the plain build |
| `smoke_noforcing_ff_s13_jd_on2`, `_on3` | same, `JAXDUMP_STEPS=1:2` | 2 | byte-identical output too; 20 (on2) / 27 (on3) stages x 2 iterations, 14 / 19 GB; dumped theta = model output (float32 rounding); end of step 1 == start of step 2 bitwise (theta, salt, u, v, etaN); tile halos = neighbour interior |

`%MON` SST/SSS statistics are printed only with one tile per process (`pkg/monitor/monitor.F:125-128`), so serial13
runs lack them — expected, not a difference.

## Dump shim (Task 5)

`JAXDUMP=1 sbatch reference/jobs/build.sbatch TREE serial13` builds with `reference/jaxdump/` (stages:
`reference/jaxdump/SUBSTEPS.md`). Coverage builds: `GCOV=1` (`-O0 --coverage`):
`mitgcmuv_ff_serial13_gcov_a0b40f12642b`, `mitgcmuv_full_serial13_gcov_162aa519f1e5`.

## Session 3 runs (2026-09-23)

Dump binaries (jaxdump, 41 stages incl. G00 geometry + exchange probe, per-level phi_hyd/mom_vecinv, temp/salt
integrate substeps): `mitgcmuv_ff_serial13_jaxdump_4d0f097d4e65`, `_5b71e689d00d` (probe with coded halos).

| run (reference/runs.json name) | config | steps | result |
|---|---|---|---|
| `smoke_ff_jaxdump_v3` | ff, useEXF/CTRL/SMOOTH=F, no geothermal | 2 (dumps 1,2) | kernel oracle without forcing |
| `forced_ff_jaxdump_v3` | ff, useEXF=T, useCTRL=F, no geothermal | 3 (dumps 1-3) | main kernel/step oracle; cg2d 164/161/158 iterations |
| `probe_exch_ff_v4` | as smoke, probe with coded halos | 1 | source of `mitgcm_jax/data/exch_maps_13x90x90.npz` |
| `ref_ff_jaxdump_v4` | ff production (useCTRL=T, geothermal) | 3 (dumps 1-3) | oracle for Task 8b (ctrl adjustments) |
| `ref_full_jaxdump_v4` | full V4r4 | 3 | M2 oracle |
| `ref_ff_mpi96_1day_a/_b` | ff production, 96 ranks | 24 | bitwise identical twin |
| `ref_ff_serial13_1day` | ff production, 13 tiles | 24 | spread vs 96 ranks after 1 day: U 7e-9, V 1.1e-8, W 6e-8, PH 1.4e-9 rel |
| `ref_full_mpi96_1day_a/_b` | full V4r4, 96 ranks | 24 | bitwise twin |
| `ref_full_mpi96_11steps` | full V4r4, 96 ranks | 11 | vs PO.DAAC 1992-01-02T00: T max 4.0e-4 °C rms 3.8e-7, S 1.9e-4 / 1.3e-7, Eta 7.8e-5 m / 3.2e-7 |
| `ref_ff_mpi96_1month` | ff production, 96 ranks | 744 | 174 s wall (0.2 s/step on 96 cores) |
| `twin_ff_nogeo_serial13_1month` | ff, useCTRL=F, no geothermal, daily dumps | 744 | Fortran twin of the JAX January run |
| `ref_ff_mpi96_1year`, `ref_ff_serial13_1year` | ff production | 8760 | queued (stage 2) |

Wall times: 13-tile serial ~3.2 s/step (1 core); 96 ranks ~0.2 s/step; JAX: one A100 0.35 s/step, 64 CPU cores ~3.5 s/step.
A launcher job that calls sbatch passes its SLURM_MEM_* on; srun then fails ("mutually exclusive") — launchers unset them.

## mpi13: 13 ranks × one 90×90 tile, bitwise twin of serial13 (2026-09-23)

serial13 (one process, the JAX port's 13 tiles) costs ~3.7 s/step on the full tree (~10 h per model year). mpi13 keeps
the same 13 tiles (so the tile-local sea-ice LSR solver and every tile-sum sees the same data) and runs them on 13
ranks: `reference/SIZE.h_13x90x90_mpi13` (`sNx=sNy=90, nSx=nSy=1, nPx=13, nPy=1`), the same
`reference/data.exch2_13x90x90`, `genmake2 -mpi`. Build: `sbatch reference/jobs/build.sbatch TREE mpi13`; run dir:
`make_rundir.py TREE mpi13 NAME ...`; run: `sbatch -p shared --ntasks=13 --mem=32G reference/jobs/run.sbatch RUNDIR`.

Why the two are bitwise identical:
- c66g `eesupp/inc/CPP_EEOPTIONS.h:132` defines `GLOBAL_SUM_ORDER_TILES` (V4r4 does not override CPP_EEOPTIONS.h).
  `GLOBAL_SUM_TILE_RL` (`eesupp/src/global_sum_tile.F`) with MPI: each rank writes its tile sums into a zeroed array
  at `(myXGlobalLo-1)/sNx+bi`, `MPI_Allreduce(MPI_SUM)` (x + 0 is exact), then sums the array from 0 in global tile
  order — the same sequence as the serial loop over `bi = 1..13`.
- `W2_MAP_PROCS` (`pkg/exch2/w2_map_procs.F`) with `nSx=nSy=nPy=1` puts exch2 tile J+1 on rank J, and `INI_PROCS`
  (MPI_CART, 13×1) gives rank J `myXGlobalLo = 1+90 J`: global tile index = exch2 tile number, as in serial13.
- Process-level `GLOBAL_SUM_R8` (order set by MPI) appears in the compiled forward code only for printed diagnostics
  (SEAICE_LSR residuals, SURF_ADJUSTMENT volume), integer point counts (MON_KE) and inactive packages
  (cost/ecco/profiles, diagnostics statistics, DIAG_CALC_PSIVEL). None feeds the state.
- The build is otherwise the serial13 one: code dir identical except SIZE.h, same FFLAGS/FOPTIM/NOOPTFILES and
  package list; `-DALLOW_USE_MPI` (usingMPI=T).

Builds (jobs 27649108/27649109, from mitgcm-jax 53f2830 + the uncommitted mpi13 layout, provenance `dirty=11`):
`mitgcmuv_ff_mpi13_36a17954ad91`, `mitgcmuv_full_mpi13_edbe8101af15`.

Gate: 1 day (24 steps, hourly %MON), overrides identical to the serial13 1-day runs (namelists, INPUTS.sha256 and input
links identical). Compared: every %MON value as printed, and every regular file in the run dir byte for byte (state
at it 1 and 25, PH/PHL, grid, pickups incl. pickup_seaice/pickup_ggl90, SBO, xx_*.effective, smooth operators,
diagnostics); the AD tapes (`tapes/tapes{2,3}.<rank>.data`, one file per process) after reassembly per tile (each 2-D
slice of the serial file = 13 tile blocks incl. halos). Controls of the comparison: 96-rank twin a vs b -> bitwise;
96 ranks vs serial13 -> 1034 %MON values and 33 files differ.

| run (runs.json) | binary | vs | %MON values | files | tapes | wall |
|---|---|---|---|---|---|---|
| `ref_ff_mpi13_1day` | `ff_mpi13_36a17954ad91` | `ref_ff_serial13_1day` | 2908 identical | 178/178 identical | identical | 69 s (serial13 373 s) |
| `ref_full_mpi13_1day` | `full_mpi13_edbe8101af15` | `ref_full_serial13_1day` | 4423 identical | 347/347 identical | identical | 162 s (serial13 359 s) |

Test: `scripts/tests/test_reference.py::test_mpi13_twin_is_bitwise_serial13` (state, pickups, %MON; fails when pointed
at the 96-rank run).

Expected differences, not model differences: mpi13 prints the `dynstat_sst_*`/`dynstat_sss_*` %MON lines (one tile per
process, `pkg/monitor/monitor.F:125`); STDOUT.0000 holds only rank 0's per-tile prints (tile topology, grid listing,
`ctrl-wet nvarlength` is per process); other ranks write STDOUT.0001-0012.

Timing (MITgcm timers, rank 0): 1-day wall time is dominated by the one-time INITIALISE_VARIA, ff 291 s -> 52 s,
full 257 s -> 124 s. Time stepping: ff 3.2 -> 0.48 s/step, full 4.0 -> 1.42 s/step. mpi13 speed depends on where
Slurm places the 13 ranks on a shared node (one physical core each; EPYC 7763 in NPS4 mode = 8 NUMA domains of 16
cores with 2 memory channels each): `ref_full_mpi13_1year` got ranks in 3 NUMA domains (5+5+3) and runs at
0.59 s/step; `ref_full_mpi13_1month` got 12 of 13 ranks in one domain (cores 16-27, node load ~80) and runs at
1.32 s/step. The step is memory-bandwidth bound: rank timers show a uniform ~3x slowdown in every section, not waiting.
Not yet tried: spreading the ranks over NUMA domains explicitly (e.g. `srun --cpu-bind=map_cpu:...`, placement does
not change the numbers).

Long full-tree twins (same overrides as `ref_full_serial13_1month`/`_1year`, which keep running as the cross-check):

| run | steps | monitor | dumpFreq | job | result |
|---|---|---|---|---|---|
| `ref_full_mpi13_1month` | 744 | 3600 | 2592000 | 27649302 (shared, 40 min) | ended normally, 1047 s wall (serial13 3046 s); vs `ref_full_serial13_1month`: 134023 %MON values over 745 blocks and 547/547 files (358 diagnostics, monthly dumps, all pickups) identical, tapes identical per tile |
| `ref_full_mpi13_1year` | 8760 | 86400 | 2592000 | 27649303 (shared, 5 h) | running (0.59 s/step, ~1.5 h) |
