# Porting lessons

One entry per task, written in the same commit as the task. Cite `file:line`; state what was measured.

## Task 1 — repository skeleton and environment (2026-09-23)

- Env `mitgcm-jax` reproduces the fesom-jax set exactly: `constraints.txt` is the full `pip freeze` of env
  `fesom-jax` (minus its editable package and conda-owned build tools); after install every package matched it.
  `test_env.py::test_pinned_versions_match_constraints` turns silent drift into a failure.
- Home is at 56/60 GB: pip must run with `--no-cache-dir` (its default cache is `~/.cache/pip`).
- The fake-device count must reach XLA before the first backend initialises. Setting `XLA_FLAGS` in the root
  `conftest.py` works because importing jax does not initialise a backend. Negative control: with
  `XLA_FLAGS=--xla_force_host_platform_device_count=2` preset, `test_four_fake_cpu_devices_and_shard_map_psum` fails.
- Cost groups live in one file (`mitgcm_jax/tests/manifest.py`); `conftest.py` applies them as markers and aborts
  collection on an unlisted test file (checked by removing an entry: "test files missing from manifest.py").
- The tier runner judges a run from the JUnit report (tests > 0, no failures/errors, testcases consistent with
  totals, ≤ 100 tests) plus pytest's exit code; each failure mode has a negative control in
  `scripts/tests/test_check_pytest_report.py`.

## Task 2 — audit of the two override trees (2026-09-23)

- Read the diff before trusting a file list. The plan named five flux-forced overrides (`apply_forcing`, `impldiff`,
  `mom_vecinv`, `mom_vi_hdissip`, `momentum_correction_step`) as physics; all five only add budget diagnostics, and
  the flux-forced run does not even switch diagnostics on. Porting them "because they are overrides" would have added
  code with no reference to test it against. Likewise CATALOG.md's "3 forward changes" were 1: `phiHydLow` at init
  feeds only the ECCO cost, and the u/v control fix changes nothing when `data.ctrl` lists both controls.
- The forward differences between c66g and V4r4 live in CPP options and namelists, not in overridden source. The
  audit therefore generates a table of every macro whose state differs (`scripts/audit_overrides.py`); Task 8's
  `config_cpp.py` must read the build's headers, never assume c66g defaults.
- A compiled package changes forward code even when it only serves the adjoint: `pkg/autodiff` in `packages.conf`
  defines `ALLOW_AUTODIFF`, and c66g then resets the r* variables every step (`forward_step.F:418–450`), zeroes
  `saltPlumeFlux` each step (`do_oceanic_phys.F:286–298`), etc. The oracle must be built with the full package list.
- The forward `AUTODIFF_INADMODE_SET/UNSET` are empty; only their TAF adjoints flip `viscFacAdj` and the package
  switches. V4r4's `mom_calc_visc.F` scales all four 3-D viscosity fields by `viscFacAdj` (c66g: two) — the JAX
  `viscFacInAd` seam must match the override, not c66g.
- The audit is a test: the inventory and CPP tables in `docs/OVERRIDES.md` are generated (`--update`) and compared
  byte-for-byte (`--check`); every changed file needs a hand-written classification row. Negative controls: planted
  new file, missing row, extra row, edited count, missing markers.

## Task 3 — ECCO v4r4 data staging and input audit (2026-09-23, in progress: forcing archives downloading)

- PO.DAAC publishes the V4r4 forcing only as two whole archives (192 GiB full, 92 GiB flux-forced); "fetch only 1992"
  is impossible there. Download whole, keep as own copies, unpack selectively (`extract`: one streaming pass that
  also writes a member index). ECCO Drive has per-file access but needs a separate WebDAV password.
- Downloads belong in batch jobs, not on the login node: `shared` nodes have internet, a job survives logout, and a
  `.part` file + HTTP Range makes a restart lose nothing (the flux-forced archive resumed at 9.9 GiB).
  Rate is capped at ~25 MiB/s per connection; two archives in two jobs run at ~25 MiB/s each.
- Python on Levante: the mambaforge base CA bundle is broken (same failure as curl); use `/etc/ssl/certs/ca-bundle.crt`.
- A namelist-derived input list needs code knowledge, not just key names: the T/S atlases in PARM05 are read only
  when `nIter0=0` (`ini_fields.F:30`) and are not in `input_init`; EXF yearly files carry `_YYYY`; ctrl files carry
  the optimcycle; smooth builds its file names in code. The audit cites the code for each rule and prints any
  file-like key without a rule as UNRESOLVED instead of dropping it.
- V4r4 namelists end groups three ways (`/`, `&`, `&end`); the first parser version silently returned empty groups
  for `&`-terminated files. Cross-check: parsed key count equals the count of `key =` lines in every file.
- Namelist diff between the trees found a forward difference the code audit could not: ff `data` sets
  `temp_EvPrRn = 0.` (added to docs/OVERRIDES.md), and ff `data.autodiff` keeps salt plume in the adjoint.
- A stream that ends is not a finished download. PO.DAAC's signed redirect URL expired mid-transfer: the 92 GiB
  archive stopped at 36.6 GiB with no error, the script treated EOF as completion, and only the sha512 check caught
  it. The downloader now trusts only the server's Content-Range/Length, resumes every short read with a fresh
  authenticated request, and treats HTTP 416 on resume as "already complete"; the checksum still decides.

## Task 5 — dump shim, coverage, matched restart (2026-09-23)

- An observer must be proven invisible: the instrumented build, with dumps off AND on, writes byte-identical state
  files and %MON to the plain build (gfortran -O3 without fast-math/FMA keeps inserted calls from changing results).
- `forward_step.F:823` advances `myIter` right after `DYNAMICS`; everything after it (incl. inside
  `SOLVE_FOR_PRESSURE`, `THERMODYNAMICS`) sees the next iteration. Dump points there pass `myIter-1`, and
  `instrument.py` asserts the counter update sits where it assumes. Found because stages S06-S14 went missing.
- `sbatch --export=VAR=1,2` splits at commas: `JAXDUMP_STEPS` accepts `:`.
- Coverage counters (.gcda) accumulate in the build dir across runs; `GCOV_PREFIX` puts them under each run.
- The published flux-forced tree cannot read its own pickups (I6 writer override, I5 c66g reader: 403 -> 40). The
  audit had flagged the asymmetry as a "reader note"; running a restart turned it into a hard failure. Restart
  itself is bitwise exact on the full tree.

## Task 7 (part) — exch2 exchanges from the Fortran exchange itself (2026-09-23)

- The halo maps are not re-derived from `data.exch2`: the dump shim runs every exch2 exchange routine the model uses
  (EXCH_XY/3D, EXCH_UV_XY with and without signs, EXCH_Z, A-, B-, D-grid vector exchanges, EXCH_SM_3D) on fields whose
  interior holds an exact integer code (component, tile, j, i) and zero halos, and dumps the result. Decoding each halo
  value gives source point, source component and sign — the map IS the Fortran behaviour, including the two-pass corner
  update of `exch2_3d_rx.template` (first pass without corners, second with). One gather per exchange in JAX.
- Result for 13 tiles of 90x90: 17,984 of 19,552 C-point halo points are written; the 1,568 untouched points are the
  four open Antarctic facet edges (4 x 360) plus eight facet-corner blocks (8 x 16). EXCH_3D == EXCH_XY and
  EXCH_UV_3D == EXCH_UV_XY bitwise (same routines underneath). EXCH_S3D works on halo-1 arrays and was not probed.
- Gate: zero the halo points a map writes in a dumped field (theta, salt, etaN, uVel/vVel at S00_begin), exchange in
  JAX, compare with the dump: bitwise equal. Negative controls (all u-map signs flipped; sources shifted by one point)
  fail. First version kept the u array as "own value" for the v output of a vector exchange — the real-field gate
  caught it on the first run; the probe-only round trip could not have.
- The dump reader must be lazy: one dumped iteration of the forced oracle is ~11 GB (41 stages).

## Session 3 kernels — cross-cutting: three XLA rewrites that break bitwise agreement (2026-09-23)

The oracle is gfortran with `-ffp-contract=off`. Three XLA:CPU behaviours each caused 1-ulp differences that
differencing stencils amplified to ~1e-13 relative (e.g. dPhiHydX); each was isolated by comparing against a plain
numpy loop in Fortran order (bitwise) and then against jit:
1. FMA contraction at the default AVX2 ISA (EOS: 5896 points up to 4.5e-13) → `--xla_cpu_max_isa=AVX`.
2. algsimp folds products of compile-time constants `(x*c1)*c2 -> x*(c1*c2)` (42 % of points), simplifies
   `c+(y-c) -> y`, rewrites `x/broadcast(d) -> x*(1/d)` even for traced d, and `A/SQRT(B) -> A*RSQRT(B)`
   → `--xla_disable_hlo_passes=algsimp` for gates, and float parameters as traced pytree leaves
   (`params_io.params_pytree`) — no optimization barriers in kernels.
With both flags (set in conftest.py) EXF, grid, GGL90, MOM_VECINV/MOM_CALC_VISC, CALC_PHI_HYD, TIMESTEP (AB3),
IMPLDIFF and the DYNAMICS driver are bitwise equal to the Fortran at every point, halos included, on both oracles.
Production runs may use XLA defaults (ulp-level differences only).

## Task 6 — grid and geometry loader (2026-09-23, sub-agent)
- `grid_from_files` is bitwise on all 60 geometry fields incl. halos. Python's `math.sin/cos` call the same glibc as
  gfortran; numpy's SIMD trig does not. gcc -O3 fuses SIN/COS of one argument into `sincos()` (ini_cori.F:95/99), whose
  cos differs from `cos()` by up to 2 ulp at 70 points — call what the binary calls.
- MDS_FACEF_READ fills the i=sNx+1 / j=sNy+1 halo row before the exchanges; on open edges those values survive or are
  shifted into other points (EXCH_Z east-edge shift). The first exchange probe (zero halos) could not see copies from
  halo points; poisoning the unwritten halo points in repeated runs gave an exact dependency mask, and the probe was
  then fixed to code halo points too (strict xfail turned XPASS → full bitwise gate).

## Task 9 — EXF flux-forced forcing (2026-09-23, sub-agent)
- A missing stage is information: S03 (CTRL_MAP_FORCING) sits inside `IF (useCTRL)`; the test asserts its absence.
- Keep Fortran REAL*4 literals in calendar code; record logic tested by hand computation across leap day and year
  ends, dates against datetime. At the first step `changed=F` and the reads come from `first`.
- Halos a loop never writes keep old values (fu's i=1-OLx column, EXF halos at fldConst): pass prior arrays, gate
  every point. Signed exchanges produce -0 where Fortran has +0 (harmless; compare values, not bits).
- FD checks: subtract outputs pointwise before weighting (a large scalar J loses ~1e-7 to cancellation).

## Task 12 — GGL90 (2026-09-23, sub-agent)
- V4r4 GGL90_OPTIONS.h enables ALLOW_GGL90_SMOOTH (c66g default off): confirm branches in the preprocessed
  `reference/build/*/bld/*.f`. Under ALLOW_AUTODIFF the caller zeroes viscArU/V, diffKr before the call.
- SOLVE_DIAGONAL_KINNER solves every column (ignores iMin..iMax). SQRTTWO=1.41421356237310D0 is a literal, not sqrt(2).
- Measure before choosing negative controls: the planned corner-mask control was a no-op (all 20x16 facet-corner halo
  points dry at every level); one blind gradient point was dry.

## Tasks 14a–c — momentum (2026-09-23, sub-agents)
- implicitViscosity=T removes the fVer k-to-k carry: MOM_VECINV is exact vectorised over k; kappaRU enters only the
  bottom drag (at k+1), and bottom drag lands in guDissip. ALLOW_AUTODIFF selects IMPLDIFF over MOM_U_IMPLICIT_R.
- FILL_CS_CORNER_TR_RL fills the caller's hDiv in place (side effect ported). All 8 cube corners are dry in V4r4, so
  corner code has no oracle signal: gated against a line-by-line transcription on random data instead.
- c66g AB3 start-up quirk: mom_StartAB = nIter0 is compared with the counts 0/1 (adams_bashforth3.F:87-92): step 1
  after the V4r4 pickup uses AB2 weights, AB3 afterwards; both gated, forcing AB3 at step 1 fails.
- GGL90_CALC_VISC masks the V increment but not the U increment (ggl90_calc_visc.F:49 vs 56) — literal c66g.

## Tasks 10–11 — EOS, density gradients, IVDC, mixed layer, salt plume (2026-09-23, sub-agent)
- Bitwise on both oracles. FMA (default XLA) gave 1.6e-14 in rhoInSitu and 2.4–3.8e-13 in sigmaX/Y (differencing
  amplifies ulps): compare against a numpy transcription before doubting a literal port.
- A corner halo fill can be invisible in the output (masks 0 at corner halos): negative control first, then gate the
  corner code against a transcription test.
- The do_oceanic_phys k-loop has no recurrence (vectorised); SALT_PLUME_CALC_DEPTH is one (vectorised FIND_RHO,
  `lax.scan` for the carry). saltPlumeFlux = spflx (READIN_SALT_PLUME_FLUX): not zeroed in DO_OCEANIC_PHYS, only
  saltPlumeDepth is. calcMixLayerDepth=F in V4r4: hMixLayer stays 0.
- ecco-mode seam (Task 17): ZERO_ADJ_LOC ≙ `lax.stop_gradient` on sigmaX/Y/R, rhoInSitu keeps its gradient.
- Jitting the function returned by `jax.vjp` captures residuals as constants (3 GB): jit a function that calls vjp.
- Right after an edit on the login node, compute nodes may still see stale files (home FS cache): checksum inside srun
  before trusting a surprising failure.

## Task 13 — GM/Redi (2026-09-23, sub-agent)
- Bitwise on both oracles (P05 tensor, P06 exchange, T01 residual flow, T13 kappaRk via GMREDI_CALC_DIFF).
- stableGmAdjTap hard-codes its limits: tensor slope <= 2e-3 (gmredi_slope_limit.F:593), bolus 5*min(|S|,1e-4)
  (gmredi_slope_psi.F:377): data.gmredi GM_maxSlope/GM_Scrit/GM_Sd/GM_slopeSqCutoff have no effect, taper factors 1.
  With GM_skewflx=0 kapGM reaches only GM_PsiX/Y. CTRL's ALLOW_KAPGM/KAPREDI_CONTROL takes GM_isopycK/GM_background_K
  out of the tensor (3-D kapGM/kapRedi instead).
- "Eager bitwise, jit not" = an XLA rewrite: dump the compiled HLO and grep rsqrt/divide. A/sqrt(B) -> A*rsqrt(B)
  happened only where the sqrt had one user (same code exact in one loop, 1 ulp off in another).
- Inputs missing from a stage can be rebuilt bitwise when they are pointwise maps of dumped fields
  (recip_hFacW = where(maskW, 1/hFacW, 0), update_r_star.F:76-79).
