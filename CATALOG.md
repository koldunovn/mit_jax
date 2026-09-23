# MITgcm → JAX (ECCO v4r4): catalog of prior experience and sources

Compiled 2026-09-23 from six read-only surveys of this account plus web prior art. Detailed notes live in
`~/MIT/work/notes/` (`= /work/ab0995/a270088/MIT/notes/`); this file is the map and the short version.

| Note | Content |
|---|---|
| `lessons_c_kokkos.md` | FESOM2 → C (port2, succeeded; FESOM_port, failed) → Kokkos: methodology, validation ladder, incidents |
| `lessons_jax.md` | FESOM2 → JAX (fesom_jax): architecture, tolerance classes, incidents, "do differently" |
| `lessons_papers.md` | Synthesis from the papers/talks: pitfall catalogue, workflow, key numbers |
| `lessons_adjoint.md` | fesom_jax adjoint: AD design, stability horizon, checkpointing, calibration/NN |
| `lessons_parallel_adjoint.md` | Parallelism that works with the adjoint; JAX vs TAF; toy experiments |
| `eccov4r4_config.md` | Exact V4r4 configuration, port sizing, LLC90 topology, data, references, staging ladder, adjoint settings |
| `prior_art.md` | Veros/Veris, Oceananigans/DJ4Earth, TAF/Tapenade adjoint, MITgcm GPU history, readers |

## 1. Workspace

| Path | What |
|---|---|
| `~/MIT/` | small things only: this catalog, docs, plans, source clones (home quota 56.5/60 GB) |
| `~/MIT/work` → `/work/ab0995/a270088/MIT` | everything heavy: data, builds, runs, dumps, notes |
| `~/MIT/MITgcm_c66g` | MITgcm at tag checkpoint66g (shallow) — the version V4r4 uses (verified 3 ways) |
| `~/MIT/ECCO-v4-Configurations/ECCOv4 Release 4/` | V4r4 `code/`, `namelist/`, `namelist_adjsen/`, `flux-forced/`, `Docker/`, `doc/` |
| `~/MIT/ECCOv4` | gaelforget/ECCOv4 (docs, `test/testreport_ecco.jl` long-run metrics) |
| `~/MIT/verification_other` → `/work/.../MIT/verification_other` | MITgcm/verification_other master (4786e08): `global_oce_llc90`, `global_oce_cs32`; results made with c69j–c69p |
| `~/MIT/verification_other_c66g` → `/work/.../MIT/verification_other_c66g` | same repo at tag checkpoint66g (git worktree) — **use these reference outputs for a c66g port** (made with c66f/c66d) |

Credentials: `~/.netrc` has an Earthdata entry (needed for PO.DAAC V4r4 data, ~200 GB compressed).
The `global_oce_llc90` README's ftp server no longer resolves — its ~595 MB test inputs need another source.

## 2. Map of earlier work on this account

| Where | What | Read first |
|---|---|---|
| `~/FESOM_port` | FAILED first C port (Mar–Apr 2026): code-first, no dumps, simplifications | `CLAUDE.md`, memory = blow-up hunt log |
| `~/port2` | SUCCESSFUL literal C port + instrumented Fortran reference | `FRESH_START.md`, `fesom2_port/docs/PORT_EXPERIENCE_REPORT.md`, `.../PORTING_LESSONS.md`, `inspect_dump.py`, `fesom2/src/fesom_dump_shim.F90` |
| `~/port_kokkos` (+ `_base _ice _int _mp _part _pre _sp _ssh _wh _xmach` worktrees) | C++/Kokkos GPU port, bit-identical Serial ladder, GPU fidelity | `docs/KOKKOS_PORTING_LESSONS.md` (D1–D22, L1–L132), `GPU_FIDELITY.md`, `SCATTER_STRATEGY.md`, `REFERENCE_RUNS.md` |
| `~/port_jax` (github koldunovn/fesom_jax; worktrees `_b296 _main_ab _review`) | FESOM2 in JAX: sharded, differentiable, 62-yr hindcast | `docs/PORTING_LESSONS.md` (6.9k lines), `ADJOINT_HORIZON.md`, `ADJOINT_CHECKPOINTING.md`, `LIMITER_GRADIENTS.md`, `PARALLELISM.md`, `JAX_RAGGED_A2A_BUG.md`, `REFERENCE_RUNS.md` |
| `~/ocean_calib`, `~/nn_eddy` | calibration / NN closures trained through fesom_jax | memory dirs |
| `~/paper_jax` | GMD paper on fesom_jax | `sections/02_model.tex` |
| `~/port_paper/paper_v2` | JAMES paper "An Ocean Model Ported by a Language Model" | `sections/methods.tex`, `discussion.tex`, `tab_pitfalls.tex`, `tab_ladder.tex` |
| `~/paper_speed` | three speed strategies (repartitioning, wide ice halo, barotropic solver) | `sections/discussion.tex` |
| `~/presentations/fesom_conversion`, `~/presentations/ccs_jpl`, `~/port_presentation` | talks; dense speaker notes, "what failed", risks | `index.html` |

Environment and machines (from fesom_jax): env `/work/ab0995/a270088/mambaforge/envs/fesom-jax` (py3.12,
jax 0.10.1, CUDA 12.9 pip wheels). Levante GPU `-A ab0995_gpu -p gpu --constraint=a100_80` (always pin 80 GB),
`gpu-devel` for quick checks; CPU tests `-p compute -A ab0995 --time=00:30:00`. Also JUPITER GH200, Albedo.
Never run models or the test suite on the login node. No persistent XLA cache (user decision).

## 3. The target in numbers (details: `eccov4r4_config.md`)

- MITgcm checkpoint66g + 26 `.F` overrides (only 3 change forward physics: Gibraltar ×10 viscosity,
  phiHydLow init, uvel/vvel control init). LLC90: 5 facets, 13 tiles of 90×90, 105,300 columns, 50 levels
  (~5.3 M cells, ~2.4 M wet). dt = 3600 s, 227,903 steps (1992–2017), 96 ranks × 30×30 tiles.
- Physics: z* nonlinear free surface (nonlinFreeSurf=4, select_rStar=2), cg2d, JMD95Z EOS, vector-invariant
  momentum, AB3, DST3 (scheme 30) tracers with implicit 3rd-order vertical advection (pentadiagonal), GM/Redi
  (bolus form, stableGmAdjTap), GGL90, salt plume, LSR viscous-plastic sea ice (zero-layer thermo), EXF with
  Large–Yeager bulk formulae on pre-interpolated 6-hourly adjusted forcing.
- Size: physics ~123k lines (~84k code) in c66g packages; relevant subset maybe ~45k. Largest: sea ice, core,
  exch2 topology, advection, GM/Redi, EXF.
- V4r4 is a free forward run: controls enter only via initial conditions and mixing coefficients
  (`xx_*.effective.*` can be read instead of porting ctrl/smooth for forward work).
- References: PO.DAAC native-grid products (first snapshot after 11 steps), testreport_ecco.jl long-run metrics,
  verification_other_c66g `global_oce_llc90/input.ecco_v4` (8 steps, %MON stats to 14 digits, made with c66f;
  master's c69m version differs by 1e-10 (θ) to 1e-6 (sea ice) — fine for ~6 digits, not for bit-level),
  MITgcm verification adjoint references `results/output_adm*.txt` (TAF gradient vs FD) — no TAF licence needed.
- ECCO adjoint is approximate by design: `data.autodiff` switches GGL90, salt plume and sea ice off in the
  backward sweep; GM slopes' density dependence dropped; cg2d operator constant in adjoint. TAF grdchk on
  llc90 ecco_v4: adjoint values stable to 6–7 digits across 66g→69j, but FD changes sign between versions
  (RMS 1−FD/AD = 12.6 vs 34.2) — compare JAX approximate-mode gradient to the TAF adjoint values (seaice/GGL90/
  salt plume under stop_gradient, viscFacInAd=2), and check JAX exactness separately (JVP + FD).

## 4. What worked (cross-port synthesis)

1. **Reference dump harness first, before any port code.** Env-gated per-substep dumps from the instrumented
   Fortran, keyed by global index; a script reports the first stage that differs. (port2: ~2 days, paid back at
   the first divergence. FESOM_port had none and died hunting symptoms.)
2. **Literal translation.** Only mechanical changes; any difference = bug. Cite `file:line` + literal value for
   every constant. Port only the branches the target run dispatches, with values from the *run's* namelists.
3. **Verification techniques ladder:** per-substep dump diff at steps 1–3 → identical-input operator diff →
   controlled replay (inject reference inputs into one routine) → matched-state gate (start from the same
   reference restart) → climate twin.
4. **Tolerance classes, not bit identity** (JAX): maps/gathers ~1e-15, scatters/reductions ~1e-12, solver
   fields at solver tolerance; relative per-field; teacher-forced multi-step (free-running decorrelates in 2 steps
   through limiters/thresholds). Insist on ~1e-13 on identical inputs — "1e-4 relative" hid a 5.4× bug.
5. **Staged physics:** minimal dynamics (rest stays at rest, gravity wave at √(gH)) → full numerics small grid →
   real grid 1 device → parallel → sea ice → GM/Redi → mixing → the rest.
6. **Static configuration**: inactive scheme absent from the compiled program; each option's off state
   byte-identical to the scheme-absent code (one test each).
7. **JAX architecture**: frozen-dataclass pytrees; static sizes; dense arrays + masks; `lax.scan` time loop with
   State as carry; float64; truncated reference constants; host-side setup in numpy then `device_put` sharded;
   compile once, reuse executable; accumulate means in the scan carry.
8. **Physical testing as acceptance:** production dt, real namelists, 1–2 model years minimum; global-mean
   SSH and conservation budgets first when drifting; depth-band T/S drift, ice area/volume, SST/SSS RMS vs a
   reference-run internal-variability yardstick.
9. **Process:** brainstorm → plan → plan review → exec; short sessions, one narrow task each; lessons log from
   day one (one entry per task, cite file:line); handoff doc per session; git tag every milestone; commit when a
   gate passes; "tests decide, not the assistant's confidence"; "always measure, do not guess".

## 5. Mistakes to avoid (each one happened)

- **Simplifying** ("improving") instead of translating — killed FESOM_port.
- **Code default instead of namelist value** — 3 incidents incl. a 5.4× ice diffusion bug found after 60 years.
- **Latent constant / small-dt validation** — ≥6 bugs passed at dt=500 and failed at production dt=1800.
- **Config desync between components** (ice_dt 500 vs ocean dt 1800 → ice 3.6× too slow; every gate hid it).
- **Halo loop bounds** (owned vs owned+halo) — dominant multi-rank bug in every port.
- **Allocated but never computed** — zero on both sides, comparison blind; need always-on range checks.
- **Wrong stride** (loop range vs allocated shape) → ×1000 gradient.
- **Symptom ≠ cause** — sessions spent proving a damping operator correct.
- **Comparator traps** — rotated vs geographic vectors; NaN vs 0 over masks; fill values; 00-UTC snapshot aliasing.
- **Masked-lane NaN in gradients** — forward `where` does not stop backward 0·inf (hit ≥4×).
- **Drivers duplicating setup** — keep ONE cold-start path.
- **Partition-dependent initial conditions** masquerading as solver bugs.
- **Measurement** — compile time in timings; re-jit per chunk; closing over big arrays; timing NaN runs; min
  over repeats; single-sample claims.
- **Checks that pass without checking** — exit 0 on "not found"; COMPLETED with failed tests; wrong checkout
  imported; knobs that never fired; tautological conservation tests; invented job IDs.
- **Operations** — `rm -rf` in unattended chains (lost 27 years of output); `git add docs/` pushed private files;
  upstream data changed silently (keep own copies); unique OUT_DIR per job.
- **JAX/XLA** — `ragged_all_to_all` has a wrong transpose; `XLA_PYTHON_CLIENT_MEM_FRACTION` ≤ 0.85; compile
  cliffs; multi-GPU atomics non-deterministic; FMA inside jit.

## 6. Testing philosophy for this port

Fast suite, physics emphasis (user requirement). Earlier suites: fesom_jax ~920 tests, 2.5–3.5 h — too slow.
Unit-testing single routines is impractical for a GCM; the useful checks are whole sub-steps against the
reference, whole runs against the reference climate, conservation, and gradients.

## 7. Differentiability and parallelism (hard requirements)

Details: `lessons_adjoint.md` (fesom_jax AD experience mapped onto MITgcm packages) and
`lessons_parallel_adjoint.md` (+ toy experiments in `~/MIT/work/scratch_parallel/`).

**Adjoint (fesom_jax):**
- Design every kernel for AD: masked lanes must compute finite values (forward `where` does not stop backward
  0·inf; hit 4× on one device, 8× more with sharding padding). Gate: gradient w.r.t. a whole IC field incl. padding.
- Never differentiate solver iterations: CG in `custom_linear_solve` (loose forward, tight transpose). Short
  contracting fixed-count loops (bulk formula 5, Newton 5) are fine to unroll; a relaxation far from its fixed
  point is not (mEVP 15 % off after 120 subcycles → adjoint overflow).
- Horizon is set by state-dependent parameterisation feedbacks, not chaos: unfrozen 1.21–1.55×/step (30-day
  gradient 1.6e154); freezing Kv/Av → 1.007×/step; + GM/Redi slope density dependence → 60–315-day windows clean.
  = ECCO practice. Freezing convection, K33, TKE state or PGF is 1.85–26× WORSE. A freeze is wrong for the
  gradient of the frozen closure's own parameters.
- Sea ice needs its own backward mode ("prescribed" ≈ ECCO; "frozen" silently zeroes dJ/d(atm); "exact" NaN).
- Checkpointing that fits one 80 GB A100 at any window: L1 remat blocks inside the step, L2 State-only per step,
  L3 √N segments, chunked reverse with day-boundary States (1.93 GB) on host RAM, boundary stride, resumable
  reverse across jobs. Device memory flat at 43 GB from 1 to 365 days; reverse/forward 4.6–6.5×; exact.
  For 26 yr LLC90: stride 1 ≈ 18 TB host → need a disk level (custom_vjp + io_callback, analogue of nchklev_3).
- Trust a gradient only after: repeats (GPU not bit-reproducible), per-chunk cotangent-norm screen, FD h-sweep
  above the forward-noise floor, a loss that actually depends on the parameter. TL via JVP of
  `custom_linear_solve` uses the loose forward solve — persist the primal trajectory and drive TL/adjoint/FD from it.
- Slow targets (equilibrium stratification, 10-yr means) are beyond one window: ensemble short bursts or EKI.

**Parallel + adjoint (toy LLC experiments, 8 CPU devices, jax 0.10.1):**
- 13-tile toy LLC with ppermute exchange incl. vector swaps/sign flips: sharded == dense to ~1e-16 relative
  (P=2,4,8); FD agrees to 1e-11; adjoint identity ⟨Ex,y⟩=⟨x,Eᵀy⟩ to 1e-15. A dropped sign in the vector exchange
  is invisible to conservation checks but caught by the grad-of-scalar vs vector-exchange identity; a wrong
  hand-written adjoint is caught by FD and the adjoint identity.
- `jax.checkpoint` inside shard_map recomputes collectives (36 vs 24 permutes) unless a `save_only_these_names`
  / offload-to-pinned_host policy saves the halo; gradients identical either way.
- GSPMD (jnp.roll / gather exchange, no shard_map) also gives correct gradients.
- `psum` of device-local sums differs across P at 1e-14; per-tile partials + fixed-order sum is P-independent
  and bit-reproducible → use for cg2d dot products and global diagnostics.
- CG via `custom_linear_solve` inside shard_map: same iteration count as dense, d/dα FD rel 1.8e-12.
- scan-inside-shard_map and shard_map-inside-scan both correct.
- Avoid `ragged_all_to_all` (wrong transpose). Gate in the fast suite: sharded grad == dense grad == FD.
- Checkpoint schedules on the toy (1024 steps): per-step 8.8 MB / 1.4 s; √N nested scans 0.98 MB / 1.8 s;
  binary (log N) 0.62 MB / 4.4 s; host-parked chunk boundaries give the identical gradient.
- `check_vma=True` keeps the backward to 1 all-reduce (vs 2); constant scan carries need `lax.pcast(..., to='varying')`.
- GSPMD turns the LLC gather exchange into whole-field all-gathers → use explicit shard_map + ppermute.
- **Unproven:** no sharded adjoint longer than 48 steps has ever been run (fesom_jax long windows were 1 GPU);
  pinned-host offload not applied on CPU — measure on GPU. `ragged_all_to_all` still defective in JAX 0.11.1.
- JAX vs TAF: automatic exchange/global-sum transposes replace hand-written `exch2_ad_*`/`global_adsum`;
  sharding divides checkpoint memory; TAF still leads on routine 26-yr disk-checkpointed adjoints and on per-step
  cost (store vs JAX recompute ≈4.7×).
