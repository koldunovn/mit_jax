# MITgcm ECCO v4r4 in JAX

A port of the MIT general circulation model (MITgcm, tag `checkpoint66g`) in the configuration of the ECCO version 4
release 4 state estimate (V4r4) to [JAX](https://github.com/jax-ml/jax). The configuration is the global LLC90 grid
(13 tiles of 90 × 90 points, 50 levels) with a one-hour time step. Two V4r4 configurations ("trees") are ported:

- **flux-forced ocean** (`ff`): surface fluxes of heat, freshwater and momentum read from files;
- **full model** (`full`): bulk-formula atmospheric forcing (`pkg/exf`) and the dynamic-thermodynamic sea-ice model
  (`pkg/seaice`).

The port translates, line by line, the Fortran branches that V4r4 executes, and every part is checked against an
instrumented gfortran build of the same code. The JAX model runs on CPUs and NVIDIA GPUs, computes gradients with
JAX's reverse mode (an `ecco` mode with the semantics of the ECCO adjoint, and an `exact` mode), and can distribute
its tiles over several devices of one node.

## Status

From [docs/VALIDATION.md](docs/VALIDATION.md). "Bitwise" means identical float64 values; the CPU comparisons use XLA
flags that switch off fused multiply-add and algebraic rewrites, as the Fortran reference is compiled.

| check | result |
|---|---|
| every ported kernel and the whole time step, both trees, CPU | bitwise identical to the Fortran |
| full model, one month (744 steps), CPU | final ocean and sea-ice state bitwise identical to the Fortran on the same 13 tiles |
| flux-forced ocean, 1992 (8760 steps), one A100 | daily global statistics within 1.6e-10 (temperature, salinity), 5e-11 (sea surface height) and 4.7e-9 (velocity) relative of the Fortran on 96 ranks |
| full model, 1992, one GH200 | each of the 124 daily global statistics closer to the Fortran 13-rank run than the Fortran 96-rank run is (typically 100-1000 times) |
| gradients, flux-forced ocean, 7-28 days | finite-difference plateau for all tested controls, tangent-linear and adjoint agree to 3e-13 |

On GPUs the compiler reorders floating-point operations, so results differ from the Fortran at round-off level. With
sea ice these differences grow in the same way as the differences between two Fortran runs on different tile layouts;
staying inside that spread is the acceptance criterion for the GPU runs.

Speed (LLC90, 50 levels, seconds per one-hour step): the full model runs at 0.29 on one GH200 (a model year in
46 minutes) and 0.43 on one A100-80; the Fortran takes 0.20 on 96 CPU cores (one DKRZ Levante node). The flux-forced
ocean runs at 0.10 on a GH200 and 0.17 on an A100. LLC90 is too small to run faster on several GPUs.

Not ported yet: the ECCO cost function and observation operators (`pkg/ecco`, `pkg/profiles`) and the time-variable
forcing controls. The forward runs use the V4r4 namelists with these packages switched off, which leaves the forward
model unchanged (verified bitwise, [docs/REFERENCE_RUNS.md](docs/REFERENCE_RUNS.md)). Multi-node runs are not
implemented.

## Running the model

[docs/RUN_ONE_YEAR.md](docs/RUN_ONE_YEAR.md) is a step-by-step guide: installation, the ECCO v4r4 input data, run
directories, one-year runs of both configurations (1992) on one GPU, checks against known results, plots and movies,
optional Fortran reference runs, and the test suite. It covers any Linux system with an NVIDIA GPU and has sections
for DKRZ Levante and NASA NAS Cabeus.

Machine-specific locations (input data, run directories, output) are set with environment variables such as
`MITJAX_WORK`; they are defined in one place, [mitgcm_jax/paths.py](mitgcm_jax/paths.py).

## Repository layout

```
mitgcm_jax/            the model
  core/                time step: dynamics, free surface and cg2d solver, tracer integration, equation of state
  pkgs/                packages: exf (flux-forced and bulk), gmredi, ggl90, gad, mom_vecinv, salt_plume, ctrl, seaice
  grid/ io/ parallel/  grid set-up, MITgcm file formats and LLC layout, exch2 exchanges and tile sharding
  adjoint/             adjoint modes, checkpointing, gradient drivers
  diagnostics/         %MON statistics, budgets, time means
  paths.py             locations of data and output (environment overrides)
  tests/               test suite; tiers in tests/manifest.py
scripts/               run_jax.py (forward runs), fetch_eccov4r4.py (input data), check_env.py (installation check),
                       batch scripts: runs/ (Slurm), cabeus/ (PBS, NASA NAS), dolpung/ (DKRZ GH200 nodes)
reference/             Fortran reference: build script, run directories (make_rundir.py), dump instrumentation (jaxdump/)
tools/                 comparisons, plots and movies
docs/                  documentation
```

## Documentation

- [docs/RUN_ONE_YEAR.md](docs/RUN_ONE_YEAR.md): installation and one-year runs
- [docs/VALIDATION.md](docs/VALIDATION.md): validation against the Fortran, speed
- [docs/ENV.md](docs/ENV.md): the pinned software environment
- [docs/DATA.md](docs/DATA.md): ECCO v4r4 input data and their integrity checks
- [docs/REFERENCE_RUNS.md](docs/REFERENCE_RUNS.md): Fortran builds and reference runs
- [docs/PORTING_RULES.md](docs/PORTING_RULES.md), [docs/KERNEL_GUIDE.md](docs/KERNEL_GUIDE.md): how the port is done
- [docs/BRANCHES.md](docs/BRANCHES.md), [docs/OVERRIDES.md](docs/OVERRIDES.md): which Fortran branches and V4r4 code
  overrides are ported
- [docs/ADJOINT_MODES.md](docs/ADJOINT_MODES.md), [docs/ADJOINT_RESULTS.md](docs/ADJOINT_RESULTS.md): gradients
- [docs/ECCO_ISSUES.md](docs/ECCO_ISSUES.md): possible issues in ECCO v4r4 / checkpoint66g found during the port
- [docs/PORTING_LESSONS.md](docs/PORTING_LESSONS.md): lessons per task
- [docs/plans/20260923-mitgcm-jax-port.md](docs/plans/20260923-mitgcm-jax-port.md): plan and status

## License

TBD. MITgcm is distributed under the MIT license (`LICENSE.txt` in the MITgcm repository); the ECCO v4r4 configuration
and data have their own terms.

## How to cite

A citation will be added here.
