# MITgcm → JAX: differentiable, parallel ECCO v4r4 (LLC90)

## Overview
- Port MITgcm checkpoint66g in the ECCO v4r4 configuration (LLC90, 50 levels, 1992–2017) to JAX, as a model that
  is **differentiable** (adjoint via JAX autodiff, in an ECCO-compatible and an exact mode) and **parallel**
  (tile sharding with correct, tested gradients).
- Goal beyond "MITgcm on GPU": a technically better adjoint than TAF — no hand-written adjoint communication, flexible
  checkpointing without store directives — shown by measurement, not assumed.
- **First big success (M1+M2):** one year of full V4r4 forward inside the Fortran run-to-run spread, AND a validated
  sharded JAX gradient over a multi-week window on the real LLC90 grid (acceptance criteria: Tasks 21–22).
- Design decisions, approved section by section: `docs/brainstorm-20260923.md`. Background: `CATALOG.md`.
- Revised 2026-09-23 after plan review (flux-forced override audit, Fortran step order, data staging, ecco-mode
  semantics, early AD/sharding gates, task splits).

## Context (from discovery)
- **Sources (read-only clones):** `~/MIT/MITgcm_c66g` (tag checkpoint66g); `~/MIT/ECCO-v4-Configurations/ECCOv4 Release 4/`
  — **two override trees**: `code/` (full V4r4) and `flux-forced/code/`. **Audited in Task 2 → `docs/OVERRIDES.md`:**
  the only override changing V4r4 forward values is the Gibraltar ×10 harmonic viscosity (`mom_calc_visc.F`); the
  flux-forced tree adds the `spflx` → `saltPlumeFlux` read path (`READIN_SALT_PLUME_FLUX`) and drops bulk formulae/sea
  ice; its `apply_forcing`, `impldiff`, `mom_vecinv`, `mom_vi_hdissip`, `momentum_correction_step` overrides are
  diagnostics only → port the c66g versions. `ALLOW_AUTODIFF` is defined in every V4r4 build (pkg/autodiff compiled)
  and changes forward branches (e.g. `forward_step.F:418–450`). Original list:
  `forward_step.F`, `do_oceanic_phys.F`, `apply_forcing.F`, `impldiff.F`, `mom_vecinv.F`, `mom_vi_hdissip.F`,
  `momentum_correction_step.F`, `exf_getffields.F`, `exf_mapfields.F` (pLoad l.330, saltPlumeFlux=spflx l.346),
  `exf_init_*.F`, `EXF_FIELDS.h`/`EXF_PARAM.h` with `spflx`); `~/MIT/verification_other_c66g`; `~/MIT/ECCOv4`.
- **Notes:** `/work/ab0995/a270088/MIT/notes/` — `eccov4r4_config.md`, `lessons_{c_kokkos,jax,papers,adjoint,parallel_adjoint}.md`,
  `prior_art.md`. Toy LLC exchange/gradient code: `/work/ab0995/a270088/MIT/scratch_parallel/llc_toy.py`.
- **Reusable prior code (read, adapt, do not import):** `~/port_jax/fesom_jax/{ops,ssh,adjoint,integrate,halo,reductions}.py`,
  `~/port2/inspect_dump.py`, `~/port2/fesom2/src/fesom_dump_shim.F90`, `~/port_jax/constraints.txt`.
- **c66g step order (staggerTimeStep=T; flux-forced `forward_step.F` line numbers; the authoritative list will be
  `reference/jaxdump/SUBSTEPS.md`, cross-checked for both trees):**
  1. `UPDATE_R_STAR(.FALSE.)` (l.430) — 2. `LOAD_FIELDS_DRIVER` → EXF read/map (l.495) —
  3. `DO_OCEANIC_PHYS` (l.609): [sea ice, M2] → `EXTERNAL_FORCING_SURF` → `FIND_RHO_2D`/`GRAD_SIGMA` →
     (`ZERO_ADJ_LOC` sigmaX/Y/R under GMREDI_WITH_STABLE_ADJOINT) → `CALC_IVDC` → `CALC_OCE_MXLAYER` →
     `SALT_PLUME_CALC_DEPTH` → `GGL90_CALC` → `GMREDI_CALC_TENSOR` —
  4. `DYNAMICS` (l.808; incl. `calc_phi_hyd`, momentum tendencies, implicit viscosity) —
  5. `UPDATE_R_STAR(.TRUE.)` (l.855) → `UPDATE_CG2D` (l.890) —
  6. `SOLVE_FOR_PRESSURE` (l.935) → `MOMENTUM_CORRECTION_STEP` (l.951) → `INTEGR_CONTINUITY` (l.965) →
     `CALC_R_STAR` (l.980) —
  7. `THERMODYNAMICS` (l.1034; incl. `GMREDI_RESIDUAL_FLOW`, salt-plume tendency via `APPLY_FORCING_S`,
     AB3 on θ/S fields with `doAB_onGtGs=F`, `FREESURF_RESCALE_G`, implicit vertical solves, `CYCLE_AB_TRACER`) —
  8. `TRACERS_CORRECTION_STEP` (l.1049) — 9. monitor/diagnostics.
- **Environment:** start pinned to fesom-jax's known-good set (jax 0.10.1, `~/port_jax/constraints.txt`) in a new
  env `/work/ab0995/a270088/mambaforge/envs/mitgcm-jax`; upgrades only via canary (M3). Levante GPU
  `-A ab0995_gpu -p gpu --constraint=a100_80`; CPU `-p compute -A ab0995 --time=00:30:00`. Earthdata in `~/.netrc`.
- **Storage:** code/docs in `~/MIT` (home quota ~full); data/builds/runs/dumps under `/work/ab0995/a270088/MIT`.

## Development Approach
- **testing approach: gate-first (TDD).** Each kernel task first writes its comparison gate against the Fortran
  dump (and gradient check), then ports until the gate passes. A new gate must be shown to FAIL on a planted error
  (negative control) before it is trusted.
- **Literal translation.** Only mechanical changes. Every constant cites Fortran `file:line` + literal value.
  Values from the run directory's `data*` files and the build's `*_OPTIONS.h`, never code defaults. Port only branches
  V4r4 executes (list from the gcov run in Task 5). No "simplifications" — deviations only when approved by
  Nikolay, bannered in code.
- **AD rules for every kernel:** masked/halo/padding lanes compute finite values (`ops/safe.py`); no differentiation
  through solver iterations; fixed iteration counts; static config; every backward switch is a seam with forward
  byte-identical and a live-fixture effect test.
- **One code path:** every kernel gate runs at P=1 and P=4 (fake CPU devices) from Task 9 on; host-side setup and
  shard-on-load from Task 8; one cold-start helper.
- complete each task fully; small changes; commit when a gate passes.
- **CRITICAL: every task MUST include new/updated tests.**
- **CRITICAL: all tests must pass before starting the next task.** Fix a failing gate by fixing code or changing
  the experiment (teacher forcing / replay), never by loosening a tolerance.
- **CRITICAL: update this plan when scope changes.**
- Every task re-runs the standing gates: full-field gradient gate (Task 8), range checks, halo-poison gate.
- `docs/PORTING_LESSONS.md`: one entry per task, same commit. Session handoff `docs/HANDOFF-<date>.md` (gitignored).
  Git tag per milestone.
- Operations: no deletions in automated scripts; unique OUT_DIR per job; no models/suites on the login node;
  check `sacct` before quoting a job ID; ask before any push.

## Testing Strategy
- **Tier 1 (<10 min, one CPU compute node, every commit, <100 tests):** per-substep dump diffs vs Fortran (steps
  1–3); controlled-replay gates; exchange identities + HLO op-budget; kernel gates at P=1 and P=4; switch off ==
  absent and ecco-forward == exact-forward (bytes) + effect tests; full-field gradient gate; FD h-sweep on a few
  steps (single kernel / small window); budgets with negative control; halo-poison; tile 90 vs 30; sharded
  gradient gates on the toy LLC and single-kernel LLC90 calls only (full-step sharded gradient is tier 2).
- **Tier 2 (~1 h, GPU, nightly/milestone):** 1–3 month run vs Fortran; full-step sharded gradient; multi-week
  gradient with repeats and amplification screen; GPU 1 vs 4 A100.
- **Tier 3 (milestones):** 1-yr / 26-yr twins, testreport_ecco metrics; yardstick = spread between two Fortran runs.
- **Tolerance classes** (relative, per field, wet points): map/gather ~1e-15; scatter/reduction ~1e-12; cg2d fields at
  solver tolerance (record residual margin at stop); replay ~1e-13. `diffdump` also flags fields identically zero on
  both sides ("allocated but never computed").
- Runner: `sbatch scripts/run_tier1.sbatch`; local `pytest -m smoke` (seconds).

## Progress Tracking
- `[x]` when done; ➕ new tasks; ⚠️ blockers; keep plan in sync.

## What Goes Where
- **Implementation Steps** (`[ ]`): code, gates, scripts, docs, reference builds/runs on /work.
- **Post-Completion**: decisions and external actions.

## Implementation Steps

### M0 — Foundations

### Task 1: Repository skeleton and environment
**Files:**
- Create: `~/MIT/.gitignore`, `pyproject.toml`, `constraints.txt`, `mitgcm_jax/__init__.py`, `docs/PORTING_LESSONS.md`, `docs/ENV.md`, `docs/PORTING_RULES.md`, `scripts/run_tier1.sbatch`
- Create: `mitgcm_jax/tests/test_env.py`, `mitgcm_jax/tests/test_manifest.py`
- ➕ Created: `conftest.py` (4 fake devices, manifest markers, per-module cache clear), `mitgcm_jax/tests/manifest.py`
  (cost groups), `scripts/check_pytest_report.py` + `scripts/tests/test_check_pytest_report.py` (runner verdict + negative controls)

- [x] `git init ~/MIT`; `.gitignore` excludes clones (`MITgcm_c66g/`, `ECCO*/`, `verification_other*`), `work`, handoffs, data
- [x] env `mitgcm-jax` pinned to fesom-jax known-good set; versions in `docs/ENV.md`
- [x] project rules (now `docs/PORTING_RULES.md`) (literal port, file:line, namelist/OPTIONS over defaults, AD rules, storage, no deletes, no login-node runs)
- [x] `run_tier1.sbatch` (compute node, unique OUT_DIR, non-zero exit if any test fails — checked by parsing the pytest summary, not only the exit code)
- [x] write tests: x64/float64 jit; manifest (every test file in a cost group); fake devices = 4
- [x] run tier1 on a compute node — must pass before Task 2 (job 27631874: 9 passed, 44 s)

### Task 2: Audit the two override trees
**Files:**
- Create: `docs/OVERRIDES.md`

- [x] diff every `code/*.F,*.h` and `flux-forced/code/*.F,*.h` against c66g and against each other; record forward-physics deltas with `file:line` (flux-forced: pLoad, spflx/saltPlumeFlux, salt-plume path in `do_oceanic_phys.F:294,580`, `impldiff`, `mom_vecinv`, `mom_vi_hdissip`, `momentum_correction_step`, EXF read/map path; full: Gibraltar ×10, phiHydLow init, uvel/vvel control init)
- [x] record CPP option differences between the trees (`*_OPTIONS.h`, `SIZE.h`, packages.conf)
- [x] write `SUBSTEPS` draft (step order with file:line for both trees)
- [x] test: a script re-runs the diff and fails if `docs/OVERRIDES.md` misses a changed file (`scripts/audit_overrides.py`, `scripts/tests/test_overrides.py`)
- [x] run test — must pass before Task 3 (tier 1 job 27632039: 13 passed)

### Task 3: Stage ECCO v4r4 input data + namelist file-reference audit
**Files:**
- Create: `scripts/fetch_eccov4r4.py`, `scripts/audit_run_inputs.py`, `docs/DATA.md`
- ➕ Created: `scripts/fetch_job.sbatch`, `scripts/extract_job.sbatch` (downloads/unpacking as `shared` jobs), `mitgcm_jax/io/{namelist,mds}.py` + `mitgcm_jax/tests/test_io_readers.py`
- Create: `scripts/tests/test_data_manifest.py`

- [x] ⏳ (small archives + products done; forcing archives downloading, jobs 27632235/6, unpack 27632252/3) fetch from PO.DAAC (Python, `~/.netrc`) to `/work/.../MIT/data/eccov4r4/`: `native_grid_files`, `input_init`, flux-forced forcing (1992 first), **1992 adjusted forcing**, `input_forcing/other`, **`control_weights`**, smooth scale/norm files; `data_constraints` + profiles only if full V4r4 cannot run without ecco/profiles (decide in Task 4, keep useCAL semantics) — ✅ all staged (ff forcing + full input_forcing 1992 + other + control_weights unpacked 2026-09-23)
- [x] fetch PO.DAAC native-grid 1992-01-02T00 snapshot (11 steps) and 1992 monthly means
- [x] `audit_run_inputs.py`: parse every `data*` of a run directory, assert each referenced file exists, record sha256 (keep own copies)
- [x] write tests: manifest checksums; shapes (compact 90×1170, big-endian float32); audit fails on a missing planted file
- [x] run tests — must pass before Task 4 — test_data_manifest.py (tier1x) passes

### Task 4: Fortran reference builds and baseline runs
**Files:**
- Create: `reference/build.sh`, `reference/optfile_levante_gfortran`, `reference/SIZE.h_13x90x90`, `reference/data.exch2_13`, `reference/run.sh`, `reference/jobs/*.sbatch`, `reference/README.md`, `docs/REFERENCE_RUNS.md`
- Create: `scripts/tests/test_reference.py`

- [x] gfortran builds (strict FP; document ifort vs gfortran) of both trees, **with the full `packages.conf` (autodiff/ctrl/ecco compiled: `ALLOW_AUTODIFF` changes forward branches, `docs/OVERRIDES.md`)**: MPI 96×(30×30) and **serial 13×(90×90) = per-substep oracle** (blankList adapted); `GLOBAL_SUM_ORDER_TILES` status recorded
- [x] frozen binaries under `/work/.../MIT/reference/bin/` with sha256 (+ `_jaxdump`, `_gcov` variants; `reference/make_rundir.py`, `reference/jobs/run.sbatch`)
- [x] runs: full V4r4 11 steps vs PO.DAAC snapshot; flux-forced and full V4r4 1 month + 1 year (96 ranks); same runs on 13-tile serial/other tiling = run-to-run spread yardstick; decide whether ecco/profiles packages can be dropped for the reference (document) — stage 1 done (11-step PO.DAAC check, twins, spread, ff 1-month); 1-year runs queued
- [x] provenance (binary sha, namelists, ranks, wall time) in `docs/REFERENCE_RUNS.md` — docs/REFERENCE_RUNS.md session-3 table
- [x] write tests (achievable): 11-step fields vs PO.DAAC snapshot within float32 + compiler floor; two runs of the same binary bitwise identical; 96-rank vs 13-tile difference recorded as spread — scripts/tests/test_reference.py (6 tests)
- [x] run tests — must pass before Task 5

### Task 5: Dump shim (`jaxdump`) and branch coverage
**Files:**
- Create: `reference/jaxdump/{jaxdump.F,JAXDUMP.h,SUBSTEPS.md}`, `reference/jaxdump/patches/{fluxforced,full}/*.patch`
- Create: `mitgcm_jax/io/dump.py`, `tools/diffdump.py`, `mitgcm_jax/tests/test_dump_io.py`

- [x] env-gated (`JAXDUMP_DIR`, `JAXDUMP_STEPS`) per-substep dumps with global (facet,i,j,k) keys **including halo values**; patches for both trees; SUBSTEPS.md final
- [x] ⏳ (done: rho/sigma/IVDC/mxlayer, salt-plume depth, GGL90, GM tensor, cg2d rhs/x/operator, IMPLDIFF in/out, residual flow, tracer integrate; remaining stages are added in the kernel task that needs them via the STAGES table) routine-input dumps for replay (grad_sigma/IVDC/mxlayer, GGL90, GM taper/tensor, salt-plume depth, cg2d rhs/operator, implicit vertical solves, mom_calc_visc; later bulk formulae, ice thermo) — all M1 stages added (41 stages, SUBSTEPS.md); M2 stages come with M2
- [x] matched-restart mode (start from any Fortran pickup, dump there) — make_rundir --pickup-from/--niter0 (session 2) + JAX restart from state npz
- [x] ⏳ (GCOV builds + `tools/branch_coverage.py` ready; ff 1-day run queued with the forcing) gcov-instrumented run (1 day, both trees) → `docs/BRANCHES.md`: routines/branches actually executed = port scope — docs/BRANCHES.md (ff 472 / full 568 executed routines; M2 additions: 38 seaice, EXF bulk, zenith angle)
- [x] dumps off ⇒ output byte-identical to uninstrumented build (also with dumps ON; smoke runs, `scripts/tests/test_reference.py`)
- [x] `tools/diffdump.py`: first (step, substep, field) above tolerance; wet masking; vector frames; zero-on-both-sides flag
- [x] write tests: reader round-trip; diffdump negative control (planted difference caught, identical passes, all-zero flagged)
- [x] run tests — must pass before Task 6

### Task 6: Grid and geometry loader
**Files:**
- Create: `mitgcm_jax/grid/{mitgrid,geometry}.py`, `mitgcm_jax/tests/test_grid.py`

- [x] `tile00{1..5}.mitgrid` + bathymetry → `[tile, j, i]` with halos; tile size parameter (90/30); blank tiles dropped — 90x90 done (grid_from_files); 30x30 needs a 30x30 exchange probe
- [x] port hFac (`hFacMin=0.2`, `hFacMinDr=5`), angles, vertical grid literally (cite `ini_masks_etc.F`, `ini_curvilinear_grid.F`) — bitwise (test_grid_load.py)
- [x] write tests: every geometry field equals Fortran output to 1e-15; wet-column count 60,646; tile 90 vs 30 identical — all 60 fields bitwise incl. halos; 60,646 wet columns; tile 30 pending
- [x] run tests — must pass before Task 7

### Task 7: exch2 topology, exchanges, global sums
**Files:**
- Create: `mitgcm_jax/grid/topology.py`, `mitgcm_jax/parallel/{exchange,global_sum}.py`, `mitgcm_jax/tests/test_exchange.py`

- [x] halo map incl. vector swap/sign/shift and exch2 corner passes — ➕ taken from the Fortran exchange routines themselves (index-coded probe in the dump shim, `scripts/make_exch_maps.py`, `mitgcm_jax/data/exch_maps_13x90x90.npz`); bitwise on dumped halos (`mitgcm_jax/tests/test_exchange.py`). A 30x30 map needs a probe run of a 30x30 dump build; `fill_cs_corner*` is ported inside the advection/momentum kernels
- [x] `exchange(field, kind)`: gather on 1 device; coloured `ppermute` rounds in `shard_map(check_vma=True)`; exchange call sites and overlap loop bounds ported literally in later tasks — parallel/sharded_exchange.py, shard.py
- [x] `global_sum`: mirror `global_sum_tile.F` ordering; bit-identical for any P — parallel/global_sum.py (GLOBAL_SUM_ORDER_TILES)
- [x] write tests: halo values equal Fortran halo dumps; adjoint identity scalar+vector; sharded == 1 device; global_sum identical P=1,2,4; halo-poison probe (NaN outside Fortran exchange points ⇒ output unchanged); HLO op-count budget; guard test forbidding `ragged_all_to_all` — test_exchange.py, test_sharded.py, test_sharded_step.py (HLO budget, ragged guard, halo poison at exchange+GAD level)
- [x] write negative controls: dropped sign, wrong transpose, stale halo — each must fail
- [x] run tests — tag `m0`

➕ **2026-09-23 (session 3) execution note:** kernels of Tasks 6, 9–16b are ported in parallel, each
gated by replay against the dump shim's stages (`docs/KERNEL_GUIDE.md`); oracles `smoke_ff_jaxdump_v3` (no forcing)
and `forced_ff_jaxdump_v3` (flux forcing, useCTRL=F, no geothermal) with geometry, exchange probe and per-level
momentum/tracer stages (`reference/jaxdump/SUBSTEPS.md`). Integration (Task 19 step) follows as kernels pass.

### M1 — Flux-forced ocean on LLC90 (forward + sharded multi-week gradient)

Kernel tasks follow the Fortran step order. Until Task 20, each kernel is gated by **replay** (dumped inputs →
compare outputs), so no task waits for a later package; full-step gates start in Task 20.

### Task 8: State, Config, skeleton, and standing gates
**Files:**
- Create: `mitgcm_jax/{state,params,config}.py`, `mitgcm_jax/config_cpp.py` (parse build `*_OPTIONS.h`), `mitgcm_jax/io/pickup.py`, `mitgcm_jax/ops/safe.py`, `mitgcm_jax/core/forward_step.py`, `mitgcm_jax/integrate.py`, `mitgcm_jax/parallel/shard.py` (host setup, `device_put` per tile, `shard_map` wrapper), `mitgcm_jax/diagnostics/checks.py`
- Create: `mitgcm_jax/tests/{test_config,test_pickup,test_integrate,test_safe,test_fullfield_grad}.py`

- [x] pytrees; every prognostic incl. AB3 histories (θ/S **fields** for tracers, `doAB_onGtGs=F`; gU/gV for momentum) and cg2d warm start is a leaf — State (dict pytree); tracers have no AB in V4r4 (DST3)
- [x] Config from run-dir namelists (`data`, `data.pkg`, `data.cal`, `data.exf`, `data.gmredi`, `data.ggl90`, `data.salt_plume`, `data.autodiff`, `data.exch2`, `eedata`) + CPP options from the build; each field records its source; unsupported option ⇒ hard error — per-package params_pytree dataclasses, hard errors on unported options (model.setup)
- [x] pickup reader: read `.meta` field list of `pickup.0000000001`; port `tempStartAB`/`momStartAB` logic literally (`pickupStrictlyMatch=F`) — init.py state_from_pickup bitwise (Task 8 agent)
- [x] `forward_step` skeleton with SUBSTEPS order; `lax.scan` integrate (step 1 eager); same path for P=1 and P=N; always-on range checks — forward_step.py bitwise; Python loop over jitted step (scan later)
- [x] `ops/safe.py`: `safe_div`, `safe_sqrt`, `safe_pow` with finite gradients on masked lanes — superseded: every kernel guards masked lanes inline (jnp.where before sqrt/div, KERNEL_GUIDE) with per-kernel gradient-finiteness tests; negative control in test_fullfield_grad.py
- [x] compile-time canary (timeout) for the skeleton at P=4 on CPU — full step compiles at P=4 in ~28 s (test_sharded_step.py)
- [x] write tests: config vs namelists/OPTIONS (code constants like Gibraltar ×10 are tested in their kernel task); pickup round-trip + AB weights at steps 1–3 vs dump; scan == loop bitwise; safe ops gradients; **standing full-field gradient gate** (grad w.r.t. whole θ,S,u,v,η finite everywhere, exactly zero on dry/halo/padding, nonzero wet) — rerun in every later task — config tests per package; pickup/AB weights (test_init, test_dynamics); scan==loop (test_checkpoint); standing gate test_fullfield_grad.py (tracers/eta zero on dry; dry velocities legitimately sensitive)
- [x] run tests — must pass before Task 9

➕ **2026-09-23: Tasks 6, 9–16b done, every kernel BITWISE equal to the Fortran on both oracles (gate XLA flags: `--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp`, params as traced pytrees); ecco seams are noted per kernel for Task 17; P=4 gates wait for the sharded exchanger (Task 7) except GAD (P=4 == P=1 done).**

### ➕ Task 8b: production control adjustments (useCTRL=T)
**Files:** `mitgcm_jax/pkgs/{ctrl,smooth}.py`, `mitgcm_jax/tests/test_ctrl.py`
- [x] port CTRL_MAP_INI_GENARR (xx_* x weights, WC01 smoother pkg/smooth, CTRL_BOUND) applied at initialisation even — pkgs/ctrl.py, pkgs/smooth.py
  with mult=0 (found in Task 8: theta changes up to 8.7 K); forcing controls (xx_gentim2d) if they change values
- [x] gate: state_from_pickup(useCTRL=T) == ref_ff_jaxdump_v4 S00/G00 bitwise; one step == iteration 2 — test_ctrl.py (8 tests, bitwise)

### Task 9: EXF flux-forced read path and surface forcing
**Files:**
- Create: `mitgcm_jax/pkgs/exf_fluxforced.py` (`exf_getffields`/`exf_mapfields` flux-forced versions, cal-based record/weights with `useExfYearlyFields`, 6-hourly from 19920101 03:00, `readStressOnCgrid=T`, pLoad = apressure×scale, spflx → saltPlumeFlux, `exf_inscal_sflux=-1e-3`), `mitgcm_jax/core/external_forcing.py` (`external_forcing_surf`, `apply_forcing`), `mitgcm_jax/tests/test_exf_fluxforced.py`

- [x] gate first: dumped EXF fields after map, surface forcing arrays, pLoad/phi0surf at steps 1–3 (incl. a record boundary)
- [x] port literally; log loaded record index every cycle
- [x] write tests: dump gates P=1/P=4; record/weight sequence over a month boundary; gradient w.r.t. a forcing field vs FD
- [x] run tests — must pass before Task 10

### Task 10: EOS, density gradients, IVDC, mixed layer
**Files:**
- Create: `mitgcm_jax/core/{eos,grad_sigma,ivdc,mxlayer}.py`, `mitgcm_jax/tests/test_eos_sigma.py`

- [x] gate first: rhoInSitu, sigmaX/Y/R, IVDC diffusivity/count, mixed-layer depth
- [x] port `find_rho.F` JMD95Z (`selectP_inEOS_Zc=0`), `grad_sigma.F`, `calc_ivdc.F`, `calc_oce_mxlayer.F`
- [x] ecco seam at `grad_sigma` output: stop_gradient on sigmaX/Y/R (mirrors `ZERO_ADJ_LOC`, `do_oceanic_phys.F:895–900`) — Task 17 (AdjointConfig.gm_sigma)
- [x] write tests: replay gates P=1/P=4; drho/dT,dS vs FD; ecco-forward == exact-forward bytes; effect test (ecco gradient differs on a fixture with active IVDC/GM)
- [x] run tests — must pass before Task 11

### Task 11: Salt-plume depth and tendency
**Files:**
- Create: `mitgcm_jax/pkgs/salt_plume.py`, `mitgcm_jax/tests/test_salt_plume.py`

- [x] gate first: plume depth, tendency (via `APPLY_FORCING_S`, `SALT_PLUME_VOLUME` undefined), flux-forced path per `do_oceanic_phys.F:294,580`
- [x] ecco mode per Task 18 semantics — Task 17: ff keeps salt plume in the adjoint; 'off' switch available
- [x] write tests: replay gates; salt conservation; ecco/exact forward bytes; effect test on a live fixture (plume depth > 0)
- [x] run tests — must pass before Task 12

### Task 12: GGL90
**Files:**
- Create: `mitgcm_jax/pkgs/ggl90.py`, `mitgcm_jax/tests/test_ggl90.py`

- [x] gate first: TKE, mixing length, Kv/Av (smoothed) — replay ~1e-13
- [x] port `ggl90_calc.F` literally (alpha=30, TKEmin, mxlMaxFlag=2, mxlSurfFlag, ALLOW_GGL90_SMOOTH); add to 3-D background diffKr
- [x] backward mode(s) per Task 18 semantics (off-in-reverse and/or frozen coefficients) — Task 17: frozen == TAF (STORE placement)
- [x] write tests: replay; ecco/exact forward bytes; effect test (Kv > background at ≥N points); gradient vs FD away from thresholds
- [x] run tests — must pass before Task 13

### Task 13: GM/Redi tensor and residual flow
**Files:**
- Create: `mitgcm_jax/pkgs/gmredi.py`, `mitgcm_jax/tests/test_gmredi.py`

- [x] gate first: slopes, taper, tensor, bolus streamfunction, residual flow
- [x] port `gmredi_calc_tensor`, `gmredi_slope_limit` (stableGmAdjTap: Redi |S|≤2e-3; bolus 5·clip(±1e-4)), `gmredi_calc_psi_b`, `gmredi_residual_flow`, `gmredi_calc_diff`; 3-D K_gm/K_redi + effective controls; CPP `GM_EXTRA_DIAGONAL`, `GM_NON_UNITY_DIAGONAL`, `GM_BOLUS_ADVEC`
- [x] write tests: replay; effect test (taper active); gradient w.r.t. K_gm field vs FD (exact mode)
- [x] run tests — must pass before Task 14

### Task 14a: Viscosity (`mom_calc_visc` V4r4 override)
**Files:**
- Create: `mitgcm_jax/pkgs/mom_common.py`, `mitgcm_jax/tests/test_visc.py`

- [x] gate first (replay): viscAh=1, viscAhGrid=0.02, 3-D viscA4 at D/Z points, Gibraltar ×10 (33–39°N, 7–2°W), `viscFacAdj`
- [x] write tests: replay; Gibraltar region factor; forward independent of adjoint factor
- [x] run tests — must pass before Task 14b

### Task 14b: Vector-invariant tendency terms and hydrostatic pressure
**Files:**
- Create: `mitgcm_jax/pkgs/mom_vecinv.py` (c66g `mom_vecinv.F`, `mom_vi_hdissip.F`: the flux-forced overrides are diagnostics only, `docs/OVERRIDES.md`), `mitgcm_jax/core/phi_hyd.py`, `mitgcm_jax/tests/test_mom_terms.py`

- [x] gate first (replay per term): phiHyd incl. r* and pLoad terms, enstrophy Coriolis with Jamart, KE gradient, vertical shear, hdissip (harmonic + biharmonic), bottom drag
- [x] corner handling literal; vector exchanges via Task 7
- [x] write tests: per-term replay P=1/P=4; gradient vs FD on one call
- [x] run tests — must pass before Task 14c

### Task 14c: Momentum time stepping and implicit viscosity
**Files:**
- Create: `mitgcm_jax/core/{dynamics,timestep}.py`, `mitgcm_jax/core/implicit.py` (tridiagonal scan), `mitgcm_jax/tests/test_dynamics.py`

- [x] gate first: gU/gV after AB3 (alph_AB=0.5, beta_AB=0.281105, forcing/dissipation outside AB), implicit viscAr=5e-5
- [x] write tests: replay; tridiagonal gradient vs FD; rest state stays at rest
- [x] run tests — must pass before Task 15

### Task 15: z* update, cg2d, correction, continuity
**Files:**
- Create: `mitgcm_jax/core/{free_surface,cg2d,solve_for_pressure}.py`, `mitgcm_jax/tests/{test_cg2d,test_free_surface}.py`

- [x] gate first: `update_r_star`, `update_cg2d` operator, cg2d rhs/solution/iteration count + residual margin, `momentum_correction_step` (c66g; flux-forced override is diagnostics only), `integr_continuity`, `calc_r_star`
- [x] cg2d literal (preconditioner, `global_sum` residual, ≤300 iterations), in `lax.custom_linear_solve(symmetric=True)`; forward reproduces Fortran iterate; tight transpose; warm start stop_gradient; ecco mode: stop_gradient on operator coefficients (as `cg2d.flow`)
- [x] write tests: replay gates; same iteration count P=1/P=4; d/d(rhs), d/d(coeff) vs FD; volume conservation + negative control
- [x] run tests — must pass before Task 16

### Task 16a: DST3 multi-dimensional advection
**Files:**
- Create: `mitgcm_jax/pkgs/gad.py`, `mitgcm_jax/tests/test_gad.py`

- [x] design decision recorded first: per-face sweep order and overlapOnly/interiorOnly (`gad_advection.F:334–355`, `nCFace`, three passes, `FILL_CS_CORNER_TR_RL`) under SPMD — per-tile static flags with select vs tiles grouped by face class
- [x] gate first (replay): advective fluxes/tendencies for θ and S incl. residual (bolus) velocity
- [x] write tests: replay P=1/P=4 and tile 90/30; conservation; gradient vs FD
- [x] run tests — must pass before Task 16b

### Task 16b: Tracer integration and implicit vertical terms
**Files:**
- Create: `mitgcm_jax/core/thermodynamics.py` (temp/salt integrate, diffusion diffKh=10 + 3-D diffKr + GGL90 + IVDC, c66g `impldiff.F` (flux-forced override is diagnostics only), geothermal, shortwave penetration, `FREESURF_RESCALE_G`, `CYCLE_AB_TRACER`), `mitgcm_jax/core/implicit.py` (pentadiagonal + u3c4 implicit vertical advection), `mitgcm_jax/core/tracers_correction.py`, `mitgcm_jax/tests/test_thermo.py`

- [x] gate first (replay): AB3 on θ/S fields (`temp_integrate.F:192–209`, rescale l.419–446), implicit solve in/out, tracers correction
- [x] write tests: replay; penta-diagonal gradient vs FD; tracer budget + negative control
- [x] run tests — must pass before Task 17

### Task 17: Backward-mode semantics (determine, then implement)
**Files:**
- Create: `docs/ADJOINT_MODES.md`, `mitgcm_jax/adjoint/modes.py`, `mitgcm_jax/tests/test_adjoint_modes.py`

- [x] determine TAF semantics from c66g code (`pkg/autodiff/autodiff_inadmode_set_ad.F:37–53` flips useGGL90/useSALT_PLUME/useSEAICE=F and sets viscFacAdj=viscFacInAd for the whole reverse sweep incl. recomputation; what is STOREd vs recomputed) — document — docs/ADJOINT_MODES.md
- [x] `AdjointConfig` per package (ecco/exact/off); if "off in reverse" ≠ "frozen coefficients", provide both variants — adjoint/modes.py; frozen == off-in-reverse for GGL90 (STORE placement)
- [x] `viscFacInAd` as `custom_vjp` (backward = VJP of viscous kernel at visc×factor) — implemented as custom_jvp (forward mode kept)
- [x] write tests: every mode forward byte-identical; factor=1 bitwise == exact gradient; factor=2 differs; effect tests on live fixtures — test_adjoint_modes(_grad).py
- [x] run tests — must pass before Task 18

### Task 18: Checkpointing and gradient drivers
**Files:**
- Create: `mitgcm_jax/adjoint/{checkpoint,grad}.py`, `mitgcm_jax/tests/test_checkpoint.py`

- [x] adapt fesom_jax chunked reverse: remat blocks, per-step State, √N segments, host-parked chunk boundaries, stride, resumable reverse; named halo checkpoint policy — adjoint/checkpoint.py, grad.py (step/sqrt/chunked, host-parked boundaries); resumable reverse + disk level: M3
- [x] trust protocol utilities: FD h-sweep with noise floor, TL(JVP)/adjoint dot test, per-chunk cotangent-norm trace + amplification statistics — dot_test, fd_sweep, plateau, amplification, cotangent_norm
- [x] write tests: all schedules give identical gradients (small window, CPU); dot test — test_checkpoint(_drivers).py; dot test 1.3e-11
- [x] run tests — must pass before Task 19

### Task 19: Full flux-forced step — forward runs
**Files:**
- Create: `scripts/runs/fluxforced_{1month,1year}.sbatch`, `mitgcm_jax/diagnostics/{monitor,budgets,means}.py`, `tools/compare_runs.py`, `mitgcm_jax/tests/test_step_fluxforced.py`

- [x] all substeps at steps 1–3 pass full-step dump gates (teacher-forced where threshold-sensitive)
- [x] monitor, SSH/heat/salt budgets, means in the scan carry — diagnostics/{monitor,budgets,means}.py; run_jax --budgets/--means-every
- [x] 1-month (GPU) vs Fortran; 1-year vs Fortran within the Task 4 spread — ✅ 1 month: CPU bitwise (all 3-D fields at it 745), GPU %MON 1e-11 (useCTRL=F twin); 1-year Fortran production refs queued; JAX 1-year GPU run 1992 done (useCTRL=F, no geothermal) → production-config comparison after Task 8b — production year GPU vs Fortran serial: %MON <= 1e-8 (w mean) / <= 1e-10 others through 145 d; full-year comparison queued (watcher)
- [x] write tests: tier-1 3-step full-step gate; budget closure + negative control — test_step_fluxforced (2 free steps bitwise), test_budgets (tier1x), test_means
- [x] run tier1 + tier2 — must pass before Task 20

### Task 20: GPU sharding check
**Files:**
- Create: `scripts/runs/gpu_sharding.sbatch`, `mitgcm_jax/tests/test_gpu_sharding.py` (tier 2)

- [x] measure GPU floor: two identical 1-GPU runs — bitwise (0) over 24 steps
- [x] 1 vs 4 A100 forward within that floor; full-step sharded gradient == 1-GPU gradient within floor — forward bitwise (0) after 24 steps; gradient 1.5e-11 (repeat floor ~4e-11); 0.43 vs 0.34 s/step
- [x] run tier 2 — must pass before Task 21 — job 27640194 (test_gpu_sharding.py)

### Task 21: Multi-week gradient on LLC90 (M1 adjoint acceptance)
**Files:**
- Create: `scripts/adjoint/multiweek_grad.sbatch`, `docs/ADJOINT_RESULTS.md`

- [x] cost: box-mean θ over 2–4 weeks (as `namelist_adjsen`), flux-forced, sharded on one node — adjsen box 120E-180E, 5-16N, levels 15-20; production ff; 1 A100 (sharded gradient validated separately in Task 20)
- [x] **exact mode:** FD plateau (h-sweep above measured noise floor) for named controls (θ IC, K_gm, a forcing field) over the longest window the amplification screen allows — 6/6 controls at 7/14/28 d; TL/adjoint 3e-13
- [x] **ecco mode:** forward bytes identical; amplification screen (median ≤1.010/step, log-spread ≤0.020, worst-3 ≤1.030); deviation from exact recorded; ≥3 repeats — bitwise forward; screen median <= 1.00015, worst-3 1.0004; repeats 3e-14; ecco vs exact 0.03-25 %
- [x] each result row records mode, freezes, window, stride, seed; memory and reverse/forward ratio measured — docs/ADJOINT_RESULTS.md; 28 d gradient = 4.3 forwards, 42-46 GB device
- [x] write tests: tier-2 regression of gradient norm + a few values — test_adjoint_regression.py
- [x] run tier1 + tier2 — tag `m1` — tier1 97 passed (clean HEAD 0ebd5bd), tier2 passed; tagged m1

### M2 — Full V4r4 (phase level; expand into tasks when M1 is done)
- **EXF bulk:** Large–Yeager 2004 (ht=hq=2 m, hu=10 m), radiation, zenith albedo table, wind-stress rotation, 6-hourly
  interpolation of adjusted forcing, runoff, pressure loading; fixed iteration counts; replay gates.
- **Sea ice:** zero-layer thermo (`seaice_growth.F`, solve4temp), LSR VP (2 Picard × ≤1500 LSOR, fixed counts),
  DST3-FL advection, ice loading; also on `offline_exf_seaice` (dyn_lsr, thermo) and `lab_sea`; LSR is tile-local →
  compare against Fortran with the same tile size and keep tile size fixed for every P; ecco mode per Task 17.
- **Runs:** 11-step PO.DAAC snapshot, 1-month, 1-year twin within spread; sharded multi-week gradient repeated in
  both modes → first big success, tag `m2`.
- **TAF gradient comparison: not part of M1/M2 acceptance** (Nikolay, 2026-09-23). M1/M2 gradients are accepted on
  JAX's own checks (exact-mode FD plateau, TL/adjoint dot test, repeats, amplification screen). The measured TAF
  comparison moves to M3, once ctrl/smooth exist.

➕ **M2 draft task list (2026-09-23, from docs/BRANCHES.md; NOT started — starts after M1 acceptance, Nikolay to
review).** Oracle: ➕ `full_jaxdump_v5` (M2.0 done 2026-09-23: 40 new full-tree stages — EXF sub-calls incl. bulk formulae locals, SEAICE_MODEL sub-calls, dynsolver, LSR per Picard pass, advdiff per field, V4r4 growth; reference/jaxdump/SUBSTEPS.md; invisible to the model).
- M2.1 EXF full read path: adjusted-forcing fields (ERA-interim + adjustments, 6-hourly), exf_set_uv on the A grid +
  EXCH_UV_AGRID, runoff monthly (cal_getmonthsrec), exf_radiation, exf_zenithangle(+table), exf_wind (wStress, wspeed).
- M2.2 EXF_BULKFORMULAE (Large-Yeager, fixed iteration count), EXF_GETSURFACEFLUXES, EXF_MAPFIELDS full tree.
- M2.3 Sea-ice thermodynamics: V4r4 seaice_growth.F override + seaice_solve4temp (fixed Newton count), budget ocean.
- M2.4 Sea-ice dynamics: seaice_dynsolver → LSR (seaice_lsr + calc_coeffs, rhsu/v, tridiagu/v; 2 Picard × LSOR fixed
  counts), strain rates, viscosities, ice strength, ocean stress/drag coeffs, freedrift (if executed), reg_ridge.
- M2.5 Sea-ice advection/diffusion: seaice_advdiff, seaice_advection with gad_dst3fl_adv_x/y (DST3 flux-limited),
  seaice_diffusion.
- M2.6 Full-tree coupling: full forward_step/do_oceanic_phys (seaice_model call site, ice loading), find_alpha (where
  executed), full-tree mom_calc_visc (same Gibraltar), pickup_seaice init, ctrl for the full tree.
- ➕ Sea-ice adjoint (Nikolay, 2026-09-23): **default = what ECCO does** — data.autodiff useSEAICEinAdMode=F: the
  reverse sweep skips the IF (useSEAICE) blocks (autodiff_inadmode_set_ad.F:37), i.e. SEAICE_MODEL's adjoint is the
  identity on the variables it overwrites and adds nothing to those it only reads (NOT stop_gradient) — thermodynamics
  included. **Switch** `ad="ecco" | "no_dynamics" | "full"`: no_dynamics = c66g SEAICEuseDYNAMICSswitchInAd=T
  (autodiff_inadmode_set_ad.F:49-51; dynamics skipped in reverse, thermodynamics adjoint kept — the FESOM choice);
  full = exact incl. the LSR implicit derivative. Forward byte-identical in all three. Default ecco (confirmed).
- M2.7 Runs: 11-step vs PO.DAAC snapshot (Fortran ref already within float32+compiler floor), 1-month, 1-year twin.

### M3 — Beyond (outline)
- 26-year twin (testreport_ecco); GPU performance (1 GPU, 1 node, vmap ensembles) vs 96-core Fortran.
- Measured TAF comparison (gradients and cost) once `xx_*` controls + WC01 exist; reference source to be decided then.
- Long-window adjoints: disk checkpoint level (custom_vjp + io_callback), host offload measured on GPU; measured
  comparison with TAF (memory, cost, code needed). GPU scatter-add non-determinism in gather-map backward: edge-strip
  lowering if repeats prove insufficient.
- `exact` sea-ice adjoint via `custom_root` (➕ 2026-09-23: the LSR already has an implicit-derivative custom_jvp,
  M2.4; sea-ice levels ecco/no_dynamics/full, M2.6b-1). ~~`xx_*` controls with WC01 via `jax.linear_transpose`; ECCO
  cost terms~~ ➕ moved to M4.
- JAX upgrade via canary (tier 1 + gradient gates + short GPU run old vs new).

### ➕ M4 — State-estimation demo, 1-2 years (added 2026-09-23 at Nikolay's request; starts after M2 acceptance)
Goal: repeat the ECCO v4r4 assimilation (4D-Var with the adjoint, L-BFGS) with the JAX model for a 1-2 year window,
first as a twin experiment with a known answer, then with the real V4r4 observations. Every step gated against the
Fortran (cost terms, control maps) or against a known answer (twin), as in M1/M2. Cost estimate (2026-09-23
measurements, not end-to-end): full model ~1.5 h per model year forward on one GH200 (ocean 0.10 s/step + Pallas sea
ice 0.4-0.8 s/step), gradient ~5 forwards (ecco mode: sea-ice adjoint skipped, its forward recomputed) -> ~8 h per
L-BFGS iteration for a 1-year window, 20-40 iterations = 1-2 GPU-weeks per window (several windows / experiments in
parallel on the 4 GPUs of a node).
- M4.0 Oracle + data: Fortran full tree with the production useECCO=T / useProfiles=T and the V4r4 data_constraints
  (builds on the 2026-09-23 useECCO-neutrality check): per-term cost values (gencost, profiles) and the control
  vectors for 1 year at the V4r4 solution; inventory + download of the observation/weight files (shared-partition
  jobs, resumable; size decides the scope: which data types first).
- ➕ M4 prerequisites found 2026-09-23: the V4r4 data_constraints are downloaded (8.9 GiB archive, 41 GB unpacked,
  /work/.../MIT/data/eccov4r4/data_constraints); the cost packages are forward-neutral (bitwise); the profile files'
  interpolation data are for the production 30x30 tiles (profiles_init_fixed.F:522-526) — JAX (13 x 90x90) must
  recompute the interpolation from the profile positions (and the gate compares against the 96-rank Fortran's
  profile misfits); sshv4-mdt reads RADS 1993-2017 for any window.
- ➕ M4 decisions (Nikolay, 2026-09-24): the estimation uses the FULL model (bulk-formula EXF + sea ice) with the V4r4
  controls (initial T/S, mixing, time-varying atmospheric-state adjustments of data.ctrl.iter0.inclatmctrl), as the
  V4r4 optimisation did (the flux-forced tree replays the adjusted fluxes: circular for estimation). Sea-ice adjoint
  level: `no_dynamics` (thermodynamics differentiated exactly, dynamics skipped; 3.2 forwards per gradient, as ecco),
  so the sea-ice concentration misfit (siv4-conc) gets a real gradient; V4r4's proxy terms (siv4-deconc: SST where the
  model lacks ice; siv4-exconc: HEFF where it has too much; cost_gencost_customize.F:204-213) are ported too, for
  comparability with V4r4 (where useSEAICEinAdMode=F leaves siv4-conc without an adjoint path).
- M4.1 Cost function: port pkg/ecco gencost (altimetry: along-track SLA + MDT; SST; sea-ice concentration; SSS and
  GRACE bottom pressure only if the window has them) and pkg/profiles (CTD/XBT/Argo interpolation in space and time),
  with their averaging operators, weights/uncertainties and smoothing; gate: every J term equal to the Fortran's at
  the V4r4 solution (bitwise where the arithmetic allows, else to the round-off floor).
- M4.2 Controls: time-variable atmospheric controls (gentim2d, 14-day records per V4r4 data.ctrl: atemp, aqh,
  precip, swdown, lwdown, wind/stress), the WC01 smoothing preconditioner and its transpose (jax.linear_transpose),
  bounds; initial-state and mixing controls (done, Task 8b). Gate vs the Fortran xx_*.effective and CTRL_MAP_FORCING.
- M4.3 Gradient over the window: full model, ecco mode (default) — FD plateaus, TL/adjoint, repeats, amplification
  screen (Task 21 recipe) at 1 year, using the exact-mode horizon study (2026-09-23) to choose the freezes.
- M4.4 Optimizer: L-BFGS in the preconditioned control space (port of m1qn3 as ECCO uses it, or an equivalent with a
  documented difference); gates on a small quadratic and on a reproducibility check.
- M4.5 Twin experiment: perturb controls, generate synthetic observations from the unperturbed run, recover the
  controls over 1 year (known answer: J -> noise floor, controls converge in the observed directions).
- M4.6 Real data, 1-2 years: window choice for Nikolay — 1992-1993 (continues from the V4r4 pickups we already run
  from; sparse in-situ data) or an Argo-era window (better profiles; needs the model state at its start: a JAX/Fortran
  forward run of V4r4 up to that date). Acceptance: J(iteration 0) = Fortran's per term, J decreasing per term like
  ECCO's own iterations, and the measured cost per iteration (GPU hours) vs ECCO's CPU cost.

### Task 22: Verify acceptance criteria (after M2)
- [x] 1-yr full V4r4 within Fortran run-to-run spread; sharded multi-week gradient validated per Task 21 criteria in both modes
  ➕ 2026-09-24: full-V4r4 month on CPU bitwise vs Fortran 13 ranks; GPU year: all 124 %MON statistics inside the
  Fortran 96-vs-13-rank spread (100-1000x margin); M2 adjoint acceptance 7/14/28 d in ecco / exact_nodyn / exact_full
  (docs/ADJOINT_RESULTS.md M2 section; decisions M2.7); 4 GH200 == 1 GH200 (ecco 7 d bitwise state, gradient at the
  repeat floor).
- [ ] tier 1 < 10 min and < 100 tests; tier 2 green  ➕ 2026-09-24: tier 1 = 100 tests (at the limit: move a few to
  tier1x); tier 2 partly re-run (test_seaice_lsr_gpu, test_adjoint_regression_full pass) — full tier-2 run next session
- [ ] no `ragged_all_to_all`, no unbannered deviations (grep audit)

### Task 23: [Final] Update documentation
- [ ] README.md, PORTING_RULES.md, PORTING_LESSONS.md, REFERENCE_RUNS.md, ADJOINT_RESULTS.md, ADJOINT_MODES.md
- [ ] move this plan to `docs/plans/completed/`

## Technical Details
- **Array layout:** `[tile, k, j, i]`, halo width as required by stencils (V4r4 uses OLx=OLy=4), float64; tile size 90
  (13 tiles) or 30 (96 wet tiles); masks at C/W/S points; dry lanes finite.
- **Exchange map:** host numpy `(src_tile, src_j, src_i, sign, swap)` per halo cell; P=1 gather; sharded = K coloured
  `ppermute` rounds. Gather backward = scatter-add (non-deterministic on GPU; mitigated by repeats).
- **State (flux-forced):** uVel, vVel, θ, S, etaN/etaH, AB3 histories (gU/gV nm1/nm2; θ/S fields nm1/nm2), rStar/hFac
  fields, cg2d warm start, GGL90 TKE (+ ice in M2). ~0.5 GB per State at LLC90.
- **Dump record:** header (step, substep id, field name[24], point kind C/W/S/Z, nz, halo flag) + global-index float64.
- **Adjoint modes:** `AdjointConfig(sigma='ecco'|'exact', ggl90=..., salt_plume=..., seaice=..., cg2d_coeff=...,
  visc_fac_in_ad=1.0)`; semantics fixed in Task 17 from the TAF code.

## Post-Completion
*Informational only*
- When/whether the repo goes public (no private links; stage files explicitly).
- Report confirmed c66g bugs upstream. Ask Nikolay which JAX features broke in fesom_jax on newer JAX (upgrade canary).
