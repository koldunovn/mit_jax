# Kernel porting guide (M1 kernels, plan Tasks 9–16b)

Read `CLAUDE.md` (project rules) first. This file is the working recipe every kernel port follows.

## Rules that decide everything
- **Literal translation** of the Fortran the V4r4 flux-forced build executes (c66g `MITgcm_c66g/`, overridden by
  `ECCO-v4-Configurations/ECCOv4 Release 4/flux-forced/code/` when a file exists there). Same operations, same order
  of floating-point operations where it matters (sums in the Fortran order; `a*b*c` stays `(a*b)*c`), same branches.
  No simplification, no "improvement". A needed deviation is **not** implemented: stop and report it (Nikolay decides).
- **Every constant and every branch cites Fortran `file:line`** in a comment (`# gad_dst3_adv_x.F:112`).
- **Parameters come from the run's namelists** (`mitgcm_jax/params_io.RunNamelists`) or, when unset there, from the
  Fortran default, passed explicitly with its citation (`set_defaults.F:123`, `ggl90_readparms.F:88`). CPP options
  come from the build's `*_OPTIONS.h` (flux-forced `code/` overrides). Port only the branches the V4r4 options select;
  an option/parameter value you did not port ⇒ `raise NotImplementedError` at setup (hard error), never a silent path.
- float64 everywhere (`import mitgcm_jax` enables x64). Masked / dry / halo lanes must compute **finite** values:
  guard divisions and sqrt with `jnp.where(mask, x, 1.0)` *before* the operation (a forward `where` after it does not
  stop a backward 0·inf).

## Layout (`mitgcm_jax/layout.py`)
- Arrays are `[tile, j, i]` or `[tile, k, j, i]` with halos (OLx=OLy=4), exactly the Fortran tile arrays with the tile
  axis first; 13 tiles of 90×90 (W2 order). Fortran `i` sits at Python index `i-1+OLx`, `k` at `k-1`.
- A Fortran loop `DO j=jMin,jMax / DO i=iMin,iMax: out(i,j) = a(i,j) - a(i-1,j)` is written for all tiles at once:
  ```python
  L = g.layout
  J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
  Im1 = L.is_(iMin - 1, iMax - 1)
  out = jnp.zeros_like(a).at[..., J, I].set(a[..., J, I] - a[..., J, Im1])
  ```
  Points the Fortran loop does not write keep what the Fortran array held before (usually 0 from an explicit
  initialisation loop — port that loop too). Use Fortran loop bounds verbatim (`1-OLx+1`, `sNx+OLx-1`, ...).
- k loops: vectorise over k when iterations are independent; genuine recurrences (vertical integrals, tridiagonal
  sweeps) use `jax.lax.scan` or `jnp.cumsum` only when the summation order is identical to Fortran's.
- `Grid` (`mitgcm_jax/grid/geometry.py`): Fortran names as attributes (`g.dxC`, `g.recip_rA`, `g.maskC`,
  `g.hFacC` is NOT static under z* — take hFac/rStar fields from the dumped state), 1-D vertical arrays `g.drF`,
  `g.rC`, `g.rF`, `g.recip_drF`, ... In tests: `g = grid_from_dump(ds, it)`.
- Exchanges (`mitgcm_jax/parallel/exchange.py`): `ex = default_exchanger()`; `ex.exch_xy(a)` (EXCH_XY/XYZ/3D_RL),
  `ex.exch_uv_xy(u, v, with_signs)` (EXCH_UV_XY/XYZ/3D_RL), `ex.exch_z`, `ex.exch_uv_agrid`, `ex.exch_uv_bgrid`,
  `ex.scalar(a, "SMs")`. Call them exactly where the Fortran calls its exchange (they are bitwise equal to exch2).
  Corner filling for advection (`FILL_CS_CORNER_TR_RL` etc.) is separate code in the tracer/momentum kernels.

## Kernel shape
- One module per package/area under `mitgcm_jax/core/` or `mitgcm_jax/pkgs/`; pure functions
  `kernel(params, g, <input arrays>) -> outputs`, jit-able, no Python loops over tiles, no host callbacks.
- A frozen dataclass per package for its parameters (`GGL90Params`), with `from_namelists(nml)` citing each default,
  decorated with `mitgcm_jax.params_io.params_pytree`: fields annotated `float` are pytree leaves, everything else
  (int/bool/str/tuple flags that select branches) is static. Pass the params object as a jit ARGUMENT, never close
  over it: XLA folds `(x*c1)*c2` -> `x*(c1*c2)` for compile-time constants (1-ulp changes at ~40% of points).
  No `lax.optimization_barrier` in kernels.
- Global sums (cg2d): fixed tile order, same as `global_sum_tile.F` (per-tile partial sums, then tiles in order).

## Gates (tests) — write the gate first, then port until it passes
- Oracle: `from mitgcm_jax.tests import oracle`; `ds = oracle.dumpset(oracle.SMOKE)` (2-step no-forcing run, dumps at
  iterations 1 and 2; a forced 3-step oracle replaces it later — keep the oracle name a module constant);
  `oracle.field(ds, it, stage, name)` → `[T,(k),j,i]` float64 with halos (per-level `K:` dumps are stacked over k).
  Stages and what they hold: `reference/jaxdump/SUBSTEPS.md`. Namelists: `oracle.run_dir(oracle.SMOKE)`.
- **Replay gate**: feed the kernel the dumped inputs of one stage, compare its output with the dumped output of the
  next, at BOTH dumped iterations. Compare exactly the points the Fortran computes (its loop ranges, halos included
  where it computes them). Tolerance classes (relative to the field's max abs over the compared region): pointwise
  maps ~1e-15; stencils/reductions ~1e-13 (report the achieved max rel. error in the test docstring); never loosen a
  tolerance to pass — find the difference (dump more, compare intermediate terms).
- **Negative control** for every new gate: a test that plants an error (perturb one constant by 1e-6 relative, drop
  one term, or shift one index) and asserts the same comparison FAILS.
- Gradient test for the kernel: `jax.grad` of a scalar of the output w.r.t. an input field is finite everywhere and
  matches a central finite difference at a few wet points (h-sweep, report the plateau); gradient on dry/halo
  lanes finite (no NaN).
- Register each new test file in `mitgcm_jax/tests/manifest.py` (group `tier1`; keep each file < ~60 s).

## Running
- Never run LLC90-sized tests on the login node. Run them inside the development allocation:
  `srun --jobid=$(squeue -u a270088 -n mitjax_dev -t R -h -o %i | head -1) --overlap -n1 -c16 --mem=64G \
     env JAX_PLATFORMS=cpu /work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python -m pytest -q <file>`
  (from `/home/a/a270088/MIT`). The login node may only run `pytest -m smoke` (seconds).
- Python: `/work/ab0995/a270088/mambaforge/envs/mitgcm-jax/bin/python` (jax 0.10.1, pinned).
- **No FMA on CPU for gates:** `conftest.py` sets `XLA_FLAGS=--xla_cpu_max_isa=AVX` (XLA:CPU's default AVX2 fuses
  a*b+c into FMA; the oracle is built with -ffp-contract=off). Standalone scripts must set it themselves:
  `env XLA_FLAGS=--xla_cpu_max_isa=AVX JAX_PLATFORMS=cpu python ...`. With it, pointwise kernels can match bitwise.
- Dev allocation now RUNNING: interactive partition job 27635402 — use `srun --jobid=27635402 --overlap -n1 -c12 --mem=24G ...`.
- Never delete anything (no `rm`, no `git clean`); never `scancel`. Scratch files go under
  `/work/ab0995/a270088/MIT/dev/<your area>/`.
