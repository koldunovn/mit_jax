# Porting rules

The rules every part of the MITgcm → JAX port follows (MITgcm checkpoint66g, ECCO v4r4 LLC90). The working recipe for
a kernel is `docs/KERNEL_GUIDE.md`; the plan is `docs/plans/20260923-mitgcm-jax-port.md`.

## Literal translation
- Only mechanical changes; any difference from the Fortran is a bug. No simplifications or "improvements" — a
  deviation needs the project lead's approval and a `# DEVIATION:` banner in the code.
- Every constant cites the Fortran `file:line` and its literal value. Sources: MITgcm c66g and the V4r4 override trees
  (`ECCOv4 Release 4/{code, flux-forced/code}`; overrides win over c66g).
- Values come from the run's `data*` namelists and the build's `*_OPTIONS.h`, never from code defaults.
- Port only branches V4r4 executes (`docs/BRANCHES.md`). An unsupported option is a hard error.
- Oracle = an instrumented gfortran c66g build with per-substep dumps (`reference/jaxdump/`). Gate first, then port;
  a new gate must fail on a planted error (negative control) before it is trusted. A failing gate is fixed by fixing
  the code or the experiment (teacher forcing / replay), never by loosening a tolerance.

## Automatic differentiation and parallelism
- Masked, halo and padding lanes compute finite values; a forward `where` does not stop a backward `0 * inf`.
- Never differentiate through solver iterations (cg2d and the sea-ice LSR use implicit derivatives:
  `custom_jvp` / `custom_linear_solve`); fixed iteration counts; static configuration.
- Every backward-only switch (the ECCO adjoint semantics, `mitgcm_jax/adjoint/modes.py`) keeps the forward
  byte-identical and has an effect test on a live fixture.
- One code path for 1 and N devices: arrays `[tile, k, j, i]`, halos filled from one exch2 map; exchanges via
  `ppermute` in `shard_map(check_vma=True)`. `ragged_all_to_all` is banned (its JAX transpose is wrong). Global sums
  in fixed tile order.
- float64 everywhere (`mitgcm_jax/__init__.py` enables x64).

## Tests
- Every test file is listed with its cost group in `mitgcm_jax/tests/manifest.py` (an unlisted file is a collection
  error): smoke, tier1 (every commit, < 10 min, < 100 tests), tier1x (kernel gates), nightly (long full-model gates),
  tier2 (GPU), tier3 (climate twins).
- Tests decide, not confidence. Measure, do not guess.
- Bitwise gates run on x86 CPUs with `XLA_FLAGS=--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp` (no FMA, no
  algebraic rewrites; the oracle is built with `-ffp-contract=off`) and pass float parameters as traced leaves.

## Environment and runs
- Environment pinned to `constraints.txt` (jax 0.10.1); upgrades only through a canary run of the forward and
  gradient gates (`docs/ENV.md`).
- Models and test suites run as batch jobs, not on login nodes; every job writes into its own new output directory;
  nothing is deleted by scripts.
- No persistent XLA compilation cache.

## Process
- One narrow task at a time; a `docs/PORTING_LESSONS.md` entry per task in the same commit; commit when a gate
  passes; a git tag per milestone; keep the plan in sync.
