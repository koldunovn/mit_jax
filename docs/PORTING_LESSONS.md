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

## Task 16b — tracer integration (2026-09-23, sub-agent)
- No Adams-Bashforth on T/S in V4r4: with DST3 (scheme 30) all AB flags are F (gad_init_fixed.F:146-165; STDOUT
  confirms), TEMP_INTEGRATE ends with CYCLE_TRACER; gtNm/gsNm stay 0. Check derived runtime flags in STDOUT
  (AdamsBashforth_T/Gt, tracForcingOutAB, diffKrNrS) before porting a branch the plan assumed.
- TRACERS_CORRECTION_STEP does nothing in V4r4 (cAdjFreq=0, no filters).
- With ALLOW_AUTODIFF, CALC_ADV_FLOW recomputes rTransKp: the tracer k loop vectorises exactly; fVer(kDown) is the kUp
  array shifted one level.
- Replaying intermediate dumps localises errors: a 6e-14 T13 error was 1-ulp input differences (algsimp division
  rewrite) amplified by the implicit solve; the solver fed dumped inputs was bitwise.

## Task 16a — DST3 multi-dimensional advection (2026-09-23, sub-agent)
- Design decision (plan Task 16a): per-tile static tables + select. Every pass runs the X and Y blocks on all tiles;
  tables carry each tile's update region (interiorOnly/overlapOnly bounds, edge flags) and the corner-fill gathers.
  Per-facet order: f1 X,Y; f2 X, X-overlap, Y; f3 Y-overlap, X, Y; f4 Y,X; f5 Y, Y-overlap, X. Costs 6 DST3 flux
  evaluations instead of 3, one code path, shards unchanged (P=4 == P=1 bitwise).
- The LLC90 facet corners are land: dropping FILL_CS_CORNER_TR_RL changes no dumped value. Gated on an all-wet
  synthetic case with constancy/conservation (with fills 7e-16 / 1e-18, without 3e-3 / 1e-8).
- Under z* GAD_ADVECTION reads UPDATE_R_STAR's hFacW/S, recip_hFacC (not recip_hFacNew). Every shared face flux was
  checked bitwise equal on both sides of every tile/facet edge (signed vector exchange of fluxes).

## Task 15 — free surface, r*, cg2d (2026-09-23, sub-agent)
- CG reproduces the Fortran bitwise, iteration count included (SMOKE 179/172, FORCED 164/161/158), with: sequential
  per-tile partial sums, fixed tile order, no FMA, traced parameters. cg2dNorm as a constant let XLA regroup
  `(b*cg2dNorm)*rhsNorm` (iterate drift 1e-9, same count). A numpy CG with sequential sums located the rewrite.
- Residual margin at stop is only 0.8 % in the tightest step (9.92e-8 vs 1e-7): iteration counts are fragile.
- Sum order is literal (`sum_order="fortran"`, 0.146 s/solve CPU) — a tree sum ("tile", 0.110 s) gives the same counts
  and x within 4e-9 but is a floating-point-order deviation: Nikolay's call for production/GPU.
- CALC_R_STAR's STOP (r* < hFacInf) cannot fire inside jit: counters returned, the driver must check them.
- Forward-mode AD of cg2d reuses the forward solve (Fortran tolerance, warm start); reverse mode is exact.
- A pre-exchange (C02) gate pins the solver's halo semantics; keeping the unknown interior-only makes the adjoint exact.

## Task 19 (first) — integrated step (2026-09-23)
- The composed FORWARD_STEP (forward_step.py, model.py: parameters from namelists, grid from files) is BITWISE equal
  to the Fortran at every dumped stage of step 1 and at the start of iterations 2 and 3 (free run), cg2d iteration
  counts included, under the gate flags. ~4 s per step on 32 CPU cores after compilation (compile ~12 s).
- tools/step_vs_dump.py = first-divergence harness (stage by stage vs dumps).

## Task 8 — initialisation from the pickup (2026-09-23, sub-agent)
- state_from_pickup == Fortran start-of-run state bitwise on all 88 fields (both oracles); one step from it == iteration 2.
- "mult_*=0" does NOT switch controls off: ctrl_map_ini_genarr adds the smoothed, weighted xx_* adjustments to theta,
  salt, u, v, etaN, kapGM, kapRedi, diffKr whatever mult is (mult only weights the cost). Production V4r4 (useCTRL=T)
  vs pickup: theta up to 8.7 K, salt 7.0, v 0.79 m/s at single points. The oracles so far use useCTRL=F → port
  CTRL_MAP_INI_GENARR + pkg/smooth (WC01) + CTRL_BOUND next (gate: ref_ff_jaxdump_v4, useCTRL=T).
- INI_CG2D writes pW/pS/pC halos that UPDATE_CG2D never rewrites: only a halo-inclusive gate finds such init state.
- MDSIO reads = fill interior, then exchange: pickup-field halos are 0 (TKE: GGL90TKEmin*maskC) where exch2 does not
  write. mom_StartAB = nIter0 (=1) with the V4r4 pickup: AB2 weights at step 1.

## Task 17 — backward-mode semantics (2026-09-23, sub-agent)
- "Package off in the adjoint" is decided by where the TAF STOREs sit, not by the flag: kappaRk / kappaRU/RV are
  stored after the GGL90 terms, so TAF's ecco adjoint uses the FORWARD Kv/Av in the implicit solves = frozen
  coefficients = stop_gradient on the GGL90 outputs (not "recomputed without GGL90"). ZERO_ADJ_LOC(sigma) also cuts
  GGL90's N^2 derivative (ggl90_calc.F:218-219) — the source of a 97.6 K/K single-point spike in the exact gradient.
- Backward-only changes: custom_jvp "value at args, tangent at alt" (viscFacInAd) keeps forward and reverse mode and is
  bitwise plain AD when alt == args. Exact mode is proven identical by jaxpr-text equality with HEAD (2 s per trace).
- Choose the cost function per seam: a one-step theta cost is blind to salt plume and cg2d; test each seam on the
  routine that feeds it and look for exact structural zeros.
- Open for Nikolay: add a "recomputed without GGL90" variant?; TAF-like cg2d adjoint tolerance (~1e-7 vs 1e-13) for the
  M3 comparison?; salt-plume adjoint semantics differ between the ff and full trees.

## Task 19 — one month free run vs Fortran (2026-09-23)
- JAX (CPU, gate flags: no FMA, no algsimp) run for 744 steps (January 1992, flux-forced, useCTRL=F, no geothermal)
  from the Fortran iteration-1 state: the final T, S, U, V, W (5.3M values each) and Eta are BITWISE equal to the
  Fortran twin's float32 output at iteration 745, and every hourly %MON dynstat agrees to print precision.
- JAX on one A100 with default XLA flags (FMA, algsimp): after the same month global %MON stats agree to 1e-14..1e-11
  relative, SST max |diff| 1.2e-7 degC (60,640 of 60,646 surface float32 values identical) — far inside the
  Fortran 13-tile vs 96-rank spread (1e-8 after ONE day). 0.35 s/step (96-core Fortran: 0.2 s/step).
- A per-step host-side monitor (copying the 3-D state) cost more than the GPU step: monitor on device.

## Process — parallel agents in one working tree (2026-09-23)
- Agents develop against each other's UNCOMMITTED edits: init.py (committed) called Exchanger.global_max, which
  existed only in the sharding agent's uncommitted exchange.py — HEAD failed 3 tier-1 tests. Check every commit in a
  clean worktree (dev/wt_check, clones symlinked) before trusting it; stage only whole-agent files (blob staging for
  shared files: manifest.py, exchange.py).
- Tier 1 on HEAD 4455541: 83 passed in 5.7 min (clean worktree).

## Task 7 (sharded) — shard_map over tiles, P=4 == P=1 bitwise (2026-09-23, sub-agent)
- Tiles in P contiguous blocks, padded with bitwise replicas of tile 1 (finite, never read, dropped). Exchanges from the
  probed maps: local gathers + greedy-coloured ppermute rounds (K = 0/1/3 at P = 1/2/4); a P=4 step has 69
  collective-permutes, 9 all-reduces, no all-gather/all-to-all. Full step P=2 and P=4 bitwise == P=1 on 90 fields;
  cg2d 164 iterations at every P; d/dtheta0 at P=4 within 1.6e-16 of P=1.
- Global sums = GLOBAL_SUM_ORDER_TILES (global_sum_tile.F:164-194): zero-padded [Tpad] buffer, one psum, ordered sum
  over tiles 1..13 — exact, P-independent, differentiable (lax.pmax has no JVP: use the same buffer for max).
- custom_linear_solve in shard_map(check_vma=True): invariants the solves close over must be pcast to varying; aux
  outputs vary like b; padding must be zeroed in the transpose solve (the padded operator is not symmetric).
- Hidden cross-tile dependencies were global reductions (calc_r_star counters, cg2d max), per-tile tables keyed by the
  global tile number (use Grid.tile_index), and shapes from L.nTiles.
- "Not written by the exchange" != "not read": exch2's corner pass reads open-edge halos (halo-poison sets must exclude them).
- NFS: a fresh PYTHONPYCACHEPREFIX per run avoids importing a stale .pyc right after an edit.

## Task 8b — production control adjustments (2026-09-23, sub-agent)
- useCTRL=T initial state and step 1 bitwise vs the production oracle. WC01 = sqrt(recip_rA*recip_drF) * 150
  pseudo-steps of SMOOTH_DIFF3D (explicit RHS + AB2 + implicit vertical) * norm; divide by sqrt(weight), add, bound,
  exchange. Bitwise needed the real*4 write/read round trip of the smoothing operators (smoothprec=32) and the model's
  nIter0 in the smoother's AB2 start (abFac=0 at pseudo-step 2).
- The forcing controls (xx_qnet ... xx_spflx) are all zero, but CTRL_MAP_FORCING's EXCH_XY_RS(saltFlux) rewrites 1538
  halo values: an all-zero stage is not automatically an identity.
- Cost: setup with useCTRL ~160 s + state_from_pickup ~150 s (7 controls x 150 pseudo-steps) — the whole-array gather
  exchange dominates; a halo-only exchange is the main speed-up.

## Task 18 — checkpointing, gradient drivers, reverse cost (2026-09-23, sub-agent)
- The "44x reverse/forward" was compile time: closed-over P, g, EXF inputs became constants XLA folded (275 s compile;
  a fresh jit per "second call" recompiled). Warm: CPU 4.1x, A100 1.54x (one step). Time warm calls on one jitted
  object; pass model data (incl. the Exchanger, now a pytree) as jit arguments (step-VJP compile 64 s -> 26 s).
- GPU: 82 % of the forward was the literal cg2d's Fortran-order tile sums (scan over 90 rows, ~44k tiny kernels per
  solve). Unrolling the scan (Cg2dParams.sum_unroll=5) keeps the same additions in the same order (bitwise) and cuts
  the forward 0.312 -> 0.141 s/step. GPU drivers should set it.
- custom_linear_solve's JVP runs `solve` on the tangent: with a literal (Fortran-tolerance, warm-started) primal
  solve the tangent-linear model disagreed with the adjoint (2-step dot test 1.9e-9). cg2d_solve is now a custom_jvp
  with an explicit implicit-derivative rule dx = A^-1(db - dA x) (tight CG): dot test 1.3e-11 (round-off floor), and
  the primal x can be named for the remat policy (the literal loop no longer re-runs in the reverse).
- A100-80, 24-step window, exact mode: step schedule 10.3 s (3.0x fwd, 38.7 GB), sqrt 13.4 s (29 GB), chunked
  (6-step chunks) 21 s, device flat ~33 GB + 1.9 GB host per boundary. ~0.69 GB per stored step: step schedule fits
  ~3 days, sqrt ~60 days (extrapolated), chunked any length.
- GPU gradient repeats differ by up to 4e-11 relative (J bitwise); source not yet found (M3 item).

## Task 20 — GPU sharding (2026-09-23)
- 4x A100-80 (one node): two 1-GPU runs bitwise identical over 24 steps (floor 0); the 4-GPU tile-sharded run is
  bitwise identical to the 1-GPU run on every field after 24 steps; the sharded gradient agrees to 1.5e-11 (GPU
  gradient repeat floor ~4e-11). 4 GPUs are slower (0.43 vs 0.34 s/step): LLC90 (13 tiles) is too small to scale;
  the sequential cg2d sums dominate (sum_unroll not set in this run).

## Task 19 — production configuration, Fortran-free (2026-09-23)
- JAX from the pickup (init.state_from_pickup with the useCTRL=T control adjustments, geothermal flux) on one A100
  (default XLA flags) vs the Fortran production run on 96 ranks (ref_ff_mpi96_1month): after 744 steps every global
  %MON statistic agrees to <= 4e-13 relative (theta/salt mean and sd identical to all printed digits; worst over the
  month 3.5e-9 in the near-zero wvel mean). No Fortran dump is used anywhere in this JAX run.
- The theta minimum of -4.8 degC in the useCTRL=F runs is gone with the control adjustments (-2.08 degC): the optimised
  IC adjustments matter physically, not just for the cost function.

## Task 19 — budgets and means (2026-09-23, sub-agent)
- Under r* the State's hFacC is one step behind its theta: content = rA*drF*h0FacC*rStarFacC*theta (wrong pairing:
  4e-2 relative error in heat).
- With temp_EvPrRn = salt_EvPrRn = 0 the EmPmR*theta_surf terms of continuity/advection cancel exactly against
  PmEpR*(EvPrRn - tracer) (external_forcing_surf.F:122-132, 257-277) — they are 10 % of Qnet: a budget that includes
  them looks almost right and is wrong. Shortwave telescopes to swfrac(0)=1 (nothing leaves the floor); the salt
  plume is a pure redistribution.
- Closure measured in round-off floors (eps x quadrature sum of cell contents): volume <= 5.4, SSH 0.8, heat 37,
  salt 7.7 over 48 LLC90 steps; the Fortran's own steps (from dumps, no JAX) <= 3.5. Negative controls fail by >= 1e4
  floors, incl. model-side ones (geothermal x(1+1e-5); EvPrRn unset).
- My run_jax snapshots had the tile axis in the wrong place for 3-D fields (tiles_to_compact wants tiles at -3) — the
  budgets agent found it; fixed with the compact() helper.

## M2.0 — full-V4r4 dump oracle (2026-09-23, sub-agent)
- 40 new tree-scoped stages (EXF sub-calls incl. bulk-formula locals before/after the stability iterations; SEAICE_MODEL
  sub-calls; dynsolver; LSR per Picard pass `_p1/_p2`; advdiff per field; V4r4 seaice_growth); new kinds U: (interior
  locals) and N: (scalars). 12.5 GB per dumped iteration. Dumps on/off byte-identical (T,S,Eta,U,V,W,PH,PHL, all
  pickups, %MON).
- Generated fixed-form Fortran must fit 72 columns (instrument.py re-wraps; a test checks); continuation scanning must
  skip #ifdef lines inside a CALL (SEAICE_SOLVE4TEMP); a stage inside a loop needs a pass suffix (the dump index keeps
  the last record of a repeated key — the test rejects duplicates).
- Serial header indexing of a full dump directory: ~110 s on cold Lustre (3 ms/record); parallel reading ~3 s.
- Process: the agent ran `rm -rf` once on a non-existent scratch path (nothing deleted) — against the no-deletion rule;
  reported to Nikolay.

## Task 21 — M1 adjoint acceptance (2026-09-23, sub-agent)
- 28-day gradients of box-mean theta (adjsen box) on LLC90, production ff, one A100-80: exact and ecco modes pass
  every bar (FD plateau for 6 controls, TL/adjoint 3e-13, amplification screen, forward bitwise, 3 repeats 3e-14).
  The exact adjoint needs no ECCO freezes at 4 weeks here (unlike fesom_jax); the freezes change directional
  derivatives by 0.03-25 %.
- Pass-through State fields (runoff, sIceLoad, ...) accumulate cotangent and fake a 1.0105/step "growth" on a
  full-State norm: identify them from the jaxpr and screen per field with an end-of-window seed.
- The GPU forward is noise-free (J bitwise), so FD precision is limited by switches (TFLUX direction plateaus ~1e-3).
- Closing over the model inside jax.jvp constant-folded the grid until the CUBIN did not fit the GPU: pass model and
  state as jit arguments, one heavy stage per process. GPU account limit: 5 running jobs (not GPUs).
- Open for Nikolay: adjsen box edge (script tests YC<=151 which is always true -> box to 180E); J scaling (literal
  adjsen divides by box volume twice); default science mode (exact is stable here); units-weighted screen norm.

## M2.5 — sea-ice advection/diffusion + reg_ridge (2026-09-23, sub-agent)
- SEAICE_ADVDIFF (DST3-FL, SEAICEadvScheme=33, flux form, HEFF/AREA/HSNOW) and SEAICE_REG_RIDGE bitwise vs
  full_jaxdump_v5 at it 1-3, halos included (A01-A06, I02, I03 incl. all 7 TICES levels); P=4 == P=1.
- SEAICE_ADVECTION is an older variant of GAD_ADVECTION (interiorOnly only in pass 1, different fill conditions): the
  pass tables differ only in each facet's last pass, so reusing GAD's table leaves the STATE unchanged — only
  halo-inclusive gates on the A-stage diagnostics catch it. Facet-edge logic can also leave the oracle state unchanged
  (LLC90 facet corners are land): an all-wet synthetic conservation test (9e-18 vs 3e-7 without the corner fills)
  proves the logic matters.
- JAX's division JVP forms b**-2, which underflows for |b| < 1e-154 -> 0*inf = NaN backward. Real sea-ice states
  hold values like HSNOW = -1.2e-240: the limiter ratio uses a quotient-rule custom_jvp (forward = the same IEEE
  division). Stacked fields share the uTrans cotangent, so one NaN lane poisons every field's gradient.
- At limiter kinks JAX splits the derivative 0.5/0.5 — a convention, not TAF's value (relevant for M3).

## M2.3 — sea-ice thermodynamics (2026-09-23, sub-agent)
- V4r4 SEAICE_GROWTH override + SEAICE_SOLVE4TEMP + SEAICE_BUDGET_OCEAN bitwise vs full_jaxdump_v5 at every dumped
  stage (H01-H06, I04; it 1-3), replayed per stage and composed; SEAICE_multDim=1 so only category 1 is computed.
- The oracle's transcendentals can block bitwise: gfortran calls glibc 2.28's ifunc-selected FMA exp (not correctly
  rounded), XLA's exp is 1-2 ulp off at ~14 % of arguments. `glibc_exp` transcribes glibc's algorithm with FMA
  emulated exactly (0 mismatches in 1.4e7 values); `jnp.exp` stays available for production runs. algsimp folds
  `(c + t) - c` to `t`, so shift-trick results are read from the integer bit pattern.
- A fixed Newton count (IMAX_TICE=10) is visible only to a bitwise gate (9 vs 10 steps: 9e-15).
- `cpp -traditional` with line markers and the build's flags gives the active branches with .F line numbers directly.
- ecco mode (open for Nikolay): with useSEAICEinAdMode=F TAF skips the IF(useSEAICE) blocks in the reverse sweep, so
  adjoints of what sea ice overwrote (Qnet, EmPmR, saltFlux) pass through as if the package were the identity — not a
  stop_gradient. The kernel gives the exact derivative for now.

## M2.1-2 — full-tree EXF: reads, radiation, zenith angle, wind, bulk formulae, mapfields (2026-09-23, sub-agent)
- EXF_GETFORCING (EXF_GETFFIELDS ... EXF_MAPFIELDS) bitwise vs full_jaxdump_v5 at X01-X08 incl. the bulk-formula
  locals X05a/X05b, it 1-3, halos included; only the unread diagnostic zen_fsol_daily differs (2e-16, XLA arccos).
- Run each gate twice, once with glibc injected through a test-only pure_callback: that separates "the port is
  literal" from "the libm differs". XLA's log/atan/sin/cos match glibc 2.28 bit for bit; exp (~14 % of arguments) and
  arccos (~7 %) do not. `exp_glibc` transcribes glibc's FMA-variant exp from the libm machine code (RHEL 8 glibc 2.28
  = the old IBM fast path without its correctly-rounded slow path). objdump of the oracle's .o files shows which libm
  routine each call really is (sincos fusion, pow, tan).
- Dump stages inside a routine can sit after partial updates (lwflux is already set at X02, inside EXF_RADIATION).
- Parallel indexing of the full oracle: ~5 s instead of ~110 s.

## M2.4 — sea-ice dynamics: SEAICE_DYNSOLVER with LSR (2026-09-23, sub-agent)
- Whole dynsolver bitwise vs full_jaxdump_v5 it 1-3 (I00 -> Y01..Y06 -> L01-L04 per Picard pass -> I01), LSOR sweep
  counts 178/118, 112/82, 84/58 and S1/S2/WFAU/WFAV equal (stopping margin small: S2 = 1.986e-4 vs LSR_ERROR 2e-4);
  only the unread uice_fd/vice_fd differ by 1 ulp at 27-45 points (gcc fuses SIN/COS into sincos). P=4 == P=1.
- Forward = the literal loop in lax.while_loop (line Gauss-Seidel in Fortran order, u rows + transposed v columns as
  26 lanes, Thomas scans). Derivative = custom_jvp implicit derivative of the converged system via
  custom_linear_solve (GMRES(40), fixed 8 cycles, one-sweep line-SOR preconditioner, transpose from
  linear_transpose; tile-order inner products); nothing is differentiated through iterations. Because LSR_ERROR=2e-4
  leaves the Fortran iterate far from the solution, the FD checks run the forward to 1e-12. Masked rows of A are
  decoupled: judge the transpose residual on wet points only.
- gfortran 11 vectorises EXP in seaice_calc_ice_strength into libmvec `_ZGVbN2v_exp`; neither scalar glibc exp nor
  XLA's reproduces PRESS0. `exp_libmvec` transliterates the kernel (8M arguments bitwise). Check `nm *.o | grep _ZGV`
  before trusting any libm call.
- The Fortran re-initialises locals only where ALLOW_AUTODIFF_TAMC does: unwritten halo points keep earlier values
  (seaiceMass starts at 1000), so all-point gates need the entry values carried in the state.
- Cost: ~1.2 s/step on 16 CPU cores (~3.5 ms per sweep); ~16k sequential scan steps per sweep — GPU cost unmeasured.

## M2.6a — ocean kernels and initial state on the full tree (2026-09-23, sub-agent)
- Every M1 ocean kernel replays bitwise on full_jaxdump_v5 (it 1-3, halos) — only the parameter readers refused the
  full namelists. The full-tree start-of-step-1 state (grid, ctrl mixing, S00/G00, 6 sea-ice fields with TICES x7,
  sIceLoad) is bitwise from setup + state_from_pickup (+ pickup_seaice, pkgs/seaice_init.py).
- A line-level gcov diff of both trees' 1-day coverage runs listed the full-tree ocean branches quickly; apart from
  diagnostics fills there were four: MXLDEPTH in data.diagnostics switches on CALC_OCE_MXLAYER method 1 + FIND_ALPHA
  (a diagnostics request changes a dumped state field, hMixLayer); saltPlumeFlux zeroed before SEAICE_MODEL (visible
  only in halos: growth writes the interior, SEAICE_MODEL does not exchange it); temp_EvPrRn unset; SEAICE_INIT_VARIA.
- CTRL_MAP_FORCING is value-identical in the full tree (c66g EXF_MAPFIELDS already exchanged the fields): a stage that
  changes nothing needs its negative control planted in the operation itself (unsigned exchange).
- Reading dump headers in parallel threads: full-oracle index 110 s -> 3 s (DumpSet).
- Open for Nikolay: keep the diagnostic-only hMixLayer/FIND_ALPHA in the production step (literal) or skip it
  (deviation); tree detection via spflxfile in data.exf (stand-in for READIN_SALT_PLUME_FLUX).

## gm_sigma="gm_only" adjoint switch (2026-09-23)
- Nikolay asked for the fesom_jax `freeze_gm_slope` analogue "for completeness": stop_gradient on sigmaX/Y/R only
  where they enter GMREDI_CALC_TENSOR; GGL90_CALC keeps its N^2 derivative. Not a TAF mode (TAF's ZERO_ADJ_LOC in
  GMREDI_WITH_STABLE_ADJOINT cuts sigma for every reader = "stable"); ecco() never selects it. Effect test: GM path
  exactly 0, GGL90 path bitwise the exact one, "stable" differs there; forward bitwise.

## M2.6b-1 — SEAICE_MODEL driver, fixed sea-ice fields, shared libm, sea-ice adjoint levels (2026-09-23, sub-agent)
- The whole SEAICE_MODEL (wind exchange, DYNSOLVER incl. clipping, ADVDIFF, REG_RIDGE, GROWTH, post-growth
  exchanges), chained 1->2->3 on its own state with grid + fixed fields from the files, is bitwise vs full_jaxdump_v5
  at I00-I04 and P00, halos included (only the unread uice_fd/vice_fd: 1 ulp).
- seaiceMaskU/V are static in V4r4: the per-step recomputation sits inside `#ifndef ALLOW_AUTODIFF_TAMC` — check CPP
  guards before believing a "recomputed every step" comment. The partly-written DYNSOLVER arrays carry no information
  across steps (rebuilding them each step is bitwise-identical; tested).
- One ops/libm.py (full-range glibc exp, libmvec exp, Libm bundles); the EXF copy used to fall back to jnp.exp for
  small |x|. Every gate stays bitwise.
- TAF-skipped blocks = identity on overwritten variables, zero on read-only ones (not stop_gradient): a linear
  custom_jvp (ops/ad_skip.py) gives exact identity/zero VJPs with a byte-identical forward. Levels: ecco (default,
  all of SEAICE_MODEL), no_dynamics (the SEAICEuseDYNAMICS blocks: FREEDRIFT+LSR, clipping), full.
- jax caches a trace per function object: re-jitting the same functools.partial after a monkeypatch replays the old
  trace — negative controls must jit a fresh lambda.
- The thermodynamics has exact-zero branches (HSNOW > 0): ulp noise under production XLA flags makes 1e-24 m snow
  and O(10 W/m2) Qnet jumps within one carried step — only gate flags give a bitwise chain; GPU/production twins with
  sea ice must be compared statistically.
- Cost on 16 CPU cores: 0.7-1.3 s/step (LSR ~3.8 ms/sweep); gradient ecco/no_dynamics ~1.8 s, full 25 s.

## Fortran mpi13 twin: 13 ranks x one 90x90 tile == serial13 bitwise (2026-09-23, sub-agent)
- With GLOBAL_SUM_ORDER_TILES (c66g default, CPP_EEOPTIONS.h:132) GLOBAL_SUM_TILE_RL is decomposition-independent
  (zeroed per-tile array, MPI_Allreduce, sum in fixed tile order) and W2_MAP_PROCS puts tile J+1 on rank J: the JAX
  tiling runs on 13 ranks bitwise = serial13 (1 day ff + full, 1 month full: every output file, the AD tapes
  reassembled per tile, every %MON value). Process-level GLOBAL_SUM_R8 left in the build feeds prints only.
- Compare the whole STDOUT, not just %MON: the only other differences are rank 0's per-tile prints and the SST/SSS
  %MON stats (printed only with one tile per process).
- The forward ALLOW_AUTODIFF build writes per-process sparse AD tapes (165 GB apparent, 0.75 GB data): compare data
  regions only, reassembled per tile; never cmp/hash the apparent size.
- mpi13 speed depends on rank placement (memory bandwidth): 0.59 s/step over 3 NUMA domains, 1.32 s/step with 12
  ranks in one; serial13 3.7-4.0 s/step. The comparison script was controlled first (96-rank twin bitwise; 96 vs 13
  caught; planted 1-bit tape flip caught).
- Amplification screens (Nikolay: "whatever ECCO does"): ECCO/MITgcm monitors adjoint variables per field
  (mon_AdVarExch, AUTODIFF_PARAMS.h) and checks gradients point-wise (grdchk) — no cross-field norm; our per-field
  screen + FD sweeps match that.
