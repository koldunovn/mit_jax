# Validation status of the JAX port (V4r4 flux-forced, LLC90) — 2026-09-23

Oracle: gfortran c66g builds of the V4r4 trees (`docs/REFERENCE_RUNS.md`), per-substep dumps (`reference/jaxdump/`).
"Bitwise" = identical float64 bits (dumps) or identical float32 output files. Gate runs use
`XLA_FLAGS=--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp` (no FMA, no algebraic rewrites; the oracle is
built with `-ffp-contract=off`) and pass float parameters as traced pytree leaves.

## Kernels (replay gates against dumped inputs/outputs, both oracles, every dumped iteration, halos included)
| area | result | tests |
|---|---|---|
| grid/geometry from mitgrid + bathymetry (Task 6) | bitwise, 60 fields | test_grid_load |
| exch2 exchanges (probe-derived maps), single device | bitwise on dumped halos | test_exchange |
| EXF flux-forced read/interp/map + EXTERNAL_FORCING_SURF (9) | 0 error | test_exf_fluxforced |
| EOS JMD95Z, GRAD_SIGMA, IVDC, mixed layer, salt plume (10-11) | bitwise | test_eos_sigma, test_salt_plume |
| GGL90 (12) | bitwise | test_ggl90 |
| GM/Redi tensor, psi, residual flow (13) | bitwise | test_gmredi |
| MOM_VECINV + MOM_CALC_VISC (V4r4 Gibraltar override) (14a-b) | bitwise | test_mom_vecinv, test_visc |
| CALC_PHI_HYD, TIMESTEP (AB3), IMPLDIFF, DYNAMICS (14b-c) | bitwise | test_phi_hyd, test_dynamics |
| r*, UPDATE_CG2D, CG2D (iteration counts), momentum correction, continuity (15) | bitwise | test_cg2d, test_free_surface |
| DST3 multi-dim advection on the LLC (16a) | bitwise; P=4 == P=1 | test_gad |
| THERMODYNAMICS / temp+salt integrate (16b) | bitwise | test_thermo |
| initialisation from the pickup (8), production ctrl adjustments incl. WC01 smoother (8b) | bitwise, 88 fields | test_init, test_ctrl |

## Integrated model
| check | result |
|---|---|
| one FORWARD_STEP (forced oracle): every dumped stage + end state | bitwise, cg2d 164 iterations (test_step_fluxforced) |
| two free steps | bitwise at iteration 3 (cg2d 161) |
| production config (useCTRL=T, geothermal) from the pickup, step 1 | bitwise (test_ctrl) |
| **one month free run** (744 steps, useCTRL=F twin), CPU gate flags | **T, S, U, V, W, Eta bitwise** vs Fortran at it 745; every hourly %MON to print precision |
| one month on one A100 (default XLA flags) vs Fortran twin | %MON <= 1e-11 rel; SST max 1.2e-7 degC (60,640/60,646 float32 equal) |
| production config on one A100 from the pickup vs Fortran 96 ranks, one month | every global %MON stat <= 4e-13 rel |
| production config on one A100 (FMA, cg2d unroll 5) vs Fortran 13-tile serial, 1992 (8760 steps) | daily global %MON: <= 1.1e-10 rel (most 1e-13); **full year (365 days): theta <= 1.2e-10, salt <= 6.5e-12, eta <= 6.2e-11, u/v <= 3.3e-9 rel** (compare_vs_fortran_serial_year.txt) |
| **production config on one A100, full year 1992 (8760 steps) vs Fortran 96 ranks** | daily global %MON over 365 days: theta/salt <= 1.6e-10 rel (means 3e-14), eta <= 5e-11, u/v <= 4.7e-9, w <= 2.2e-8 (w mean 4e-16 abs); runs_jax/ff_prod_1992_gpu_v2/compare_vs_fortran_mpi96_year.txt; maps: runs_jax/diff_maps/ff_year_jax_vs_fortran.png (end of year, float64: SST <= 1e-6 K, SSH <= 4e-9 m) |
| discrete volume / SSH / heat / salt budgets per step (48 steps) | close to round-off: <= 5.4 / 0.8 / 37 / 7.7 floors; Fortran's own steps <= 3.5; negative controls >= 1e4 floors (test_budgets) |
| Fortran yardsticks | 96-rank twin bitwise; 13-tile vs 96-rank spread 1e-8 after 1 day; 11 steps vs PO.DAAC product T 4e-4 max |
| **full V4r4 (EXF bulk + sea ice), one month on one GH200** (production XLA flags, Pallas LSR; 0.27 s/step) vs Fortran 13 tiles | every one of 124 hourly %MON statistics (ocean, sea ice, EXF) closer to the Fortran 13-tile run than the Fortran 96-rank run is (typically 10-1000x): e.g. mean theta 2e-10 vs 1.7e-8, SSH sd 2e-8 vs 2e-5, ice area mean 8.7e-7 vs 4.7e-6 |
| full V4r4, step 1 / free run to it 4 / 24 steps from the pickup (CPU gate flags) | bitwise (820 stage fields; every State field vs the Fortran pickups) (test_step_full) |
| sharded: shard_map P=2, P=4 (CPU fake devices) | bitwise == P=1 full step (test_sharded, test_sharded_step) |
| 4 A100 vs 1 A100 (Task 20) | forward bitwise after 24 steps; GPU floor 0; gradient 1.5e-11 |

## Differentiability
| check | result |
|---|---|
| per-kernel gradients vs central FD (h-sweeps) | plateaus 1e-9 ... 1e-15 (kernel tests) |
| full step: gradient w.r.t. whole theta, S, u, v, eta | finite everywhere; zero on dry tracer/eta points (test_fullfield_grad) |
| TL (jvp) vs adjoint dot test, 2 steps | 1.3e-11 (round-off floor; cg2d custom_jvp implicit derivative) |
| gradient schedules step / sqrt / chunked | identical to <= 1e-13 (test_checkpoint) |
| ecco mode (TAF semantics: GGL90 frozen, sigma cut, cg2d operator passive) | forward byte-identical; effect tests (test_adjoint_modes*) |
| cost on A100-80, 24-step window | gradient 10.3 s = 3.0x forward (step schedule); chunked: device memory flat ~33 GB |
| **multi-week gradient acceptance (Task 21)**, box-mean theta, production ff, 7/14/28 d, 1 A100 | exact: FD plateau 6/6 controls, TL/adjoint <= 3e-13; ecco: forward bitwise, amplification median <= 1.00015/step, repeats 3e-14; 28 d gradient = 4.3 forwards, 46 GB device (docs/ADJOINT_RESULTS.md) |

## Performance (LLC90, 50 levels)
Fortran: 13-tile serial 1 core ~3.2 s/step; 96 ranks ~0.2 s/step. JAX: one A100 0.34 s/step (0.14 s/step with
cg2d sum unroll 5, bitwise); 64 CPU cores ~3.5 s/step (gate flags); 4 A100 0.43 s/step (LLC90 too small to scale).
One GH200 (dolpung, aarch64): 0.10 s/step (production ff, cg2d unroll 5; job 27647358).
Sea-ice SEAICE_DYNSOLVER (literal LSR sweep order, M2.4): bitwise with the Fortran on CPU, A100 and GH200 (same
sweep counts), but launch-bound on GPUs: 183 ms/sweep (A100-40), 92 ms/sweep (GH200) = 13-54 s per step, vs
~3.5 ms/sweep on 16 CPU cores (~1.2 s/step). scripts/runs/gpu_lsr_bench.py.
