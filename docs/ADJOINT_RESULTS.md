# Multi-week gradients on LLC90: M1 adjoint acceptance (plan Task 21)

2026-09-23. The JAX port of the V4r4 **flux-forced** configuration, production set-up, one NVIDIA A100-80.
Drivers: `mitgcm_jax/adjoint/{checkpoint,grad,modes}.py`. Experiment: `scripts/adjoint/multiweek_grad.py` (+ `.sbatch`).
Tables: `scripts/adjoint/summarize_results.py`. Figures: `scripts/adjoint/plot_sensitivity.py` (nereus env).
Every result is one JSON line (job id, git HEAD, device, XLA flags, mode and freezes, window, chunk, stride, schedule,
h or seed, memory, times) in `/work/ab0995/a270088/MIT/runs/adjoint/<run>/results.jsonl`; the gradient fields are the
`grad_<mode>_<days>d_r<repeat>.npz` next to it (interior points, float64).

## 1. Set-up

**Model.** Run directory `reference/runs/ref_ff_serial13_1day` (production: `useCTRL=T` with the WC01-smoothed
`xx_theta/salt/uvel/vvel/etan/kapgm/kapredi/diffkr` adjustments, geothermal flux, 6-hourly flux forcing), start
1992-01-01 00:00 (`nIter0=1`), dt = 3600 s. Initial state: `init.state_from_pickup`, computed once on CPU with the gate
XLA flags (bitwise the Fortran start-of-run state, Task 8b) and cached in
`/work/ab0995/a270088/MIT/runs/adjoint/init_ref_ff_serial13_1day` (1.9 GB; `setup` 101 s + `state_from_pickup` 107 s on
16 CPU cores). The GPU runs use XLA's default flags (FMA, algebraic simplifier on) and `Cg2dParams.sum_unroll=5`.
Windows: 7, 14 and 28 days = 168, 336 and 672 steps from 1992-01-01.

**Cost.** The ECCO adjoint-sensitivity experiment (`namelist_adjsen/data.ecco`: gencost `boxmean` of theta, 3-D mask
`objmask`, `scripts/prepare_run_adjsen.py`):

    J = (1/N) sum_{n=1..N} [ sum_box theta_n hFacC_n drF rA ] / [ sum_box h0FacC drF rA ]

i.e. the volume-weighted box-mean theta at the end of every step (ECCO_PHYS, ff `forward_step.F:1189`;
`ecco_phys.F:314-348` weights with the current r* `hFacC` and divides by the static box volume `eccoVol_0`,
`ecco_check.F:67`), averaged over the N steps of the window. The box is the adjsen mask as the script writes it:
`maskC * (XC >= 120) * (YC <= 151) * (YC >= 5) * (YC <= 16)`, levels 15-20: 120.5E-179.5E, 5.8N-15.3N (cell centres),
146-323 m, 624 columns, 3743 wet cells, 1.269e15 m^3, on tiles 5, 6, 8, 9. Two deliberate readings, both documented
in the script: (a) the script's comment says 120E-151E but its test is `YC <= 151` (always true), so the experiment's
box reaches 180E; it is used as written. (b) The script divides `objmask` by the box volume and ecco_phys divides
again, so the literal adjsen J is this J / 1.269e15; J here is in K (gradients differ by that constant only).
J = 12.6265 K (7 d), 12.6195 K (14 d), 12.6267 K (28 d).

**Controls** (one pytree; zero = the production forward, value-identical):

| control | shape | how it enters |
|---|---|---|
| `theta` | 3-D | added to theta at the start of iteration 1 (interior), then EXCH_XYZ_RL: the `xx_theta` path of CTRL_MAP_INI_GENARR (nothing after it in INITIALISE_VARIA reads theta) |
| `kapGM` | 3-D | added to the GM diffusivity (interior), then EXCH_XYZ_RL (the `xx_kapgm` path; `Model.g` rebuilt in `params_fn`) |
| `tflux` | 2-D | time-constant shift of the TFLUX record buffers (W/m^2, downward; `exf_inscal_hflux=-1`), added to both EXF_SET_FLD buffers of every step |
| `taux`, `tauy` | 2-D | the same for oceTAUX / oceTAUY (N/m^2, model-grid components) |

**Named directions** for FD and TL (Fortran indices; depth = cell centre):

| name | definition |
|---|---|
| theta_A_centre | theta at tile 8, i=90, j=8, level 17 (149.5E, 10.5N, 195 m): inside the box |
| theta_B_above | same column, level 14 (140 m): just above the box top (outside) |
| theta_C_south | tile 9, i=8, j=9, level 17 (150.5E, 3.6N, 195 m): 2 deg south of the box (outside) |
| kapGM_scale | direction = kapGM itself (wet interior): dJ/d ln(kapGM), a global relative change |
| tflux_box | +1 W/m^2 (downward) TFLUX over the 624 box columns |
| taux_box | +1 N/m^2 oceTAUX (model-grid x) over the 624 box columns |

**Modes.** exact = `AdjointConfig()` (no seam). ecco = `AdjointConfig.ecco(nml)` of this run directory:
`ggl90="frozen"`, `gm_sigma="stable"`, `salt_plume="exact"`, `cg2d="passive"`, `visc_fac_in_ad=1.0`
(docs/ADJOINT_MODES.md).

**Drivers.** Gradient: `grad.chunked_value_and_grad`, chunks of 24 steps (1 day), boundary stride 1 (every day
boundary on the host), in-chunk schedule `step` (per-step remat, the cg2d solution saved). FD and TL: one jitted
`lax.scan` over the whole window (schedule `none`), same device type and XLA flags; TL = `jax.jvp` of J. FD: central
differences at four h per direction, both evaluations in the same process; the forward noise floor is the spread of
repeated J evaluations (0 in every window: the GPU forward is deterministic).

**Screen.** A separate chunked gradient of the end-of-window box mean alone (terminal seed, no running cost): with
the running cost every chunk injects new cotangent and the norm grows by accumulation. The per-chunk norm of the State
cotangent is recorded per field (`ChunkedGrad.field_trace`, new). Left out of the norm: the *carried constants*,
State fields FORWARD_STEP passes through unchanged (found from the step's jaxpr: runoff, sIceLoad, hMixLayer,
hFac_surf*, gt/gsNm*): their cotangent accumulates the gradient with respect to a constant field (runoff: linear growth
to 1e3 in 7 days, which read as a median 1.0105/step "growth" on the full norm). Left out of the statistics: the seed chunk
(it maps the theta/hFacC seed onto the whole State, a change between field sets) and, for the norm over all dynamic
fields, the chunk ending at iteration 1 (AB2 start: gu/gvNm_2 are not read there). Bars (fesom_jax): median <=
1.010/step, log-spread <= 0.020, worst 3 consecutive chunks <= 1.030/step. Three norms are reported: all dynamic fields
(dominated by the momentum AB histories gu/gvNm_1,2, whose cotangent units are large), theta alone, and the
prognostic set (theta, salt, uVel, vVel, etaN).

## 2. Acceptance summary

One A100-80 per run; every number below is from the tables in section 3 (job ids there).

| criterion | bar | 7 d (168 steps) | 14 d (336 steps) | 28 d (672 steps) |
|---|---|---|---|---|
| **exact**: FD plateau, 6 named controls | >= 2 of 4 h with rel. error <= 1e-3 | 6/6; best rel. error per control 1.0e-9 (theta_A) ... 3.4e-4 (TFLUX: 3/4 h <= 1e-3, all <= 1.4e-3) | 6/6; 1.7e-8 (theta_A) ... 8.3e-4 (TFLUX: 2/4 h <= 1e-3, all <= 1.8e-3) | 6/6; 2.0e-7 (theta_C) ... 4.4e-4 (TFLUX: 2/4 h <= 1e-3, all <= 1.6e-3) |
| **exact**: TL (jax.jvp) vs adjoint, combined direction, amplitude 1 and 1e-6 | round-off | 3.3e-14, 3.3e-14 | 1.3e-13, 1.3e-13 | 3.2e-13, 3.2e-13 |
| **exact**: TL vs adjoint, each named direction (max) | round-off | 8.6e-13 | 1.0e-12 | 2.9e-12 |
| **exact**: screen, theta / prognostic / dynamic: median, worst-3 per step | <= 1.010, <= 1.030 (log-spread <= 0.020) | 1.00004/1.00038; 0.99905/1.00089; 0.99768/0.99773: pass | 1.00002/1.00038; 1.00009/1.00132; 0.99764/0.99972: pass | 1.00010/1.00040; 1.00034/1.00123; 0.99892/1.00060: pass |
| **ecco**: forward == exact forward (bytes, 90 State fields + J, whole window) | bitwise | bitwise | bitwise | bitwise |
| **ecco**: screen (same three norms) | as above | 0.99997/1.00038; 0.99896/1.00090; 0.99769/0.99774: pass | 0.99995/1.00036; 0.99990/1.00139; 0.99761/0.99972: pass | 1.00015/1.00040; 1.00036/1.00131; 0.99891/1.00060: pass |
| **ecco**: 3 repeats | within the GPU floor (4e-11, Task 18) | J bitwise; gradient 1.9e-14 | J bitwise; 1.8e-14 | J bitwise; 3.1e-14 |
| **ecco** vs exact, directional derivatives | recorded | theta_A 1.3e-3, kapGM 5.4e-3, TFLUX 3.1e-2, oceTAUX 2.2e-2, theta_B 6.4e-2, theta_C 6.5e-2 | 3.4e-4, 2.2e-3, 1.9e-2, 2.8e-2, 9.9e-2, 1.4e-1 | 4.4e-3, 6.9e-3, 2.1e-2, 3.6e-2, 1.1e-1, 2.5e-1 |
| cost: warm gradient / one plain forward; device peak; host | - | 4.3x (103 s vs 24 s); 41.7 GB (ecco) / 45.6 GB (exact); 15 GB | 4.4x (208 s); 41.7 / 45.6 GB; 29 GB | 4.3x (414 s vs 95.5 s); 41.8 / 45.6 GB; 55 GB |

Window reached: **28 days in both modes**, the longest asked for; no statistic of the screen, the FD sweeps or the TL
check degrades from 7 to 28 days, so longer windows were not ruled out (not tried).

## 3. Tables

Generated by `python3 scripts/adjoint/summarize_results.py /work/ab0995/a270088/MIT/runs/adjoint/w{07,14,28}_*`.
Runs: `w07_exact` (27641126), `w07_ecco` (27641127), `w07_fd` (27641128); `w14_{exact,ecco,fd}` and `w28_fd_theta`
(27641755, 4 GPUs, one process per GPU); `w28_exact` (27641132), `w28_ecco` (27641133), `w28_fd_other` (27641135).
Every run: chunk 24 steps, boundary stride 1, in-chunk schedule `step`, sum_unroll 5, XLA default flags, seed 0 (the
random weights of the combined TL direction). FD directions and h are in the result rows.

### Exact mode: FD h-sweep vs adjoint (and TL) per named control

| window | direction | adjoint (exact) | TL (exact) | FD rel. error at h = ... | plateau (rel <= 0.001) | ecco adjoint rel. to best FD | job (grad / FD) |
|---|---|---|---|---|---|---|---|
| 7 d | kapGM_scale | 1.539701e-03 | 1.539701e-03 | 0.1: 6.0e-05; 0.01: 1.2e-05; 0.001: 4.6e-05; 0.0001: 7.5e-05 | 4/4 h (0.0001..0.1) | 5.3e-03 | 27641126 / 27641128 |
| 7 d | taux_box | -0.06755661 | -0.06755661 | 0.01: 5.0e-05; 0.001: 9.3e-06; 0.0001: 1.1e-06; 1e-05: 3.8e-05 | 4/4 h (1e-05..0.01) | 2.2e-02 | 27641126 / 27641128 |
| 7 d | tflux_box | -3.397525e-06 | -3.397525e-06 | 10: 5.3e-04; 1: 9.2e-04; 0.1: 1.4e-03; 0.01: 3.4e-04 | 3/4 h (0.01..10) | 3.1e-02 | 27641126 / 27641128 |
| 7 d | theta_A_centre | 2.335519e-04 | 2.335519e-04 | 0.1: 6.5e-06; 0.01: 8.1e-06; 0.001: 1.0e-09; 0.0001: 1.7e-07 | 4/4 h (0.0001..0.1) | 1.3e-03 | 27641126 / 27641128 |
| 7 d | theta_B_above | 3.501376e-05 | 3.501376e-05 | 0.1: 7.3e-05; 0.01: 7.7e-05; 0.001: 7.2e-08; 0.0001: 3.0e-07 | 4/4 h (0.0001..0.1) | 6.4e-02 | 27641126 / 27641128 |
| 7 d | theta_C_south | 4.428277e-06 | 4.428277e-06 | 0.1: 9.0e-05; 0.01: 3.1e-06; 0.001: 2.0e-06; 0.0001: 3.4e-06 | 4/4 h (0.0001..0.1) | 6.5e-02 | 27641126 / 27641128 |
| 14 d | kapGM_scale | 3.134998e-03 | 3.134998e-03 | 0.1: 3.7e-05; 0.01: 4.4e-05; 0.001: 2.8e-05; 0.0001: 7.9e-05 | 4/4 h (0.0001..0.1) | 2.2e-03 | 27641755 / 27641755 |
| 14 d | taux_box | -0.1412442 | -0.1412442 | 0.01: 1.2e-04; 0.001: 2.4e-05; 0.0001: 2.4e-05; 1e-05: 5.2e-05 | 4/4 h (1e-05..0.01) | 2.8e-02 | 27641755 / 27641755 |
| 14 d | tflux_box | -7.584061e-06 | -7.584061e-06 | 10: 1.8e-03; 1: 8.3e-04; 0.1: 1.3e-03; 0.01: 8.4e-04 | 2/4 h (0.01..1) | 1.8e-02 | 27641755 / 27641755 |
| 14 d | theta_A_centre | 2.351843e-04 | 2.351843e-04 | 0.1: 1.9e-05; 0.01: 8.0e-06; 0.001: 1.7e-08; 0.0001: 1.1e-07 | 4/4 h (0.0001..0.1) | 3.4e-04 | 27641755 / 27641755 |
| 14 d | theta_B_above | 3.870123e-05 | 3.870123e-05 | 0.1: 8.0e-05; 0.01: 7.0e-05; 0.001: 2.7e-07; 0.0001: 3.3e-06 | 4/4 h (0.0001..0.1) | 9.9e-02 | 27641755 / 27641755 |
| 14 d | theta_C_south | 4.576290e-06 | 4.576290e-06 | 0.1: 9.3e-05; 0.01: 6.9e-06; 0.001: 3.4e-06; 0.0001: 1.4e-05 | 4/4 h (0.0001..0.1) | 1.4e-01 | 27641755 / 27641755 |
| 28 d | kapGM_scale | 6.245992e-03 | 6.245992e-03 | 0.1: 1.2e-04; 0.01: 4.6e-06; 0.001: 1.8e-04; 0.0001: 1.8e-04 | 4/4 h (0.0001..0.1) | 6.9e-03 | 27641132 / 27641135 |
| 28 d | taux_box | -0.3323112 | -0.3323112 | 0.01: 2.2e-04; 0.001: 2.1e-05; 0.0001: 2.3e-05; 1e-05: 1.1e-05 | 4/4 h (1e-05..0.01) | 3.6e-02 | 27641132 / 27641135 |
| 28 d | tflux_box | -1.803175e-05 | -1.803175e-05 | 10: 1.6e-03; 1: 4.4e-04; 0.1: 4.7e-04; 0.01: 1.2e-03 | 2/4 h (0.1..1) | 2.1e-02 | 27641132 / 27641135 |
| 28 d | theta_A_centre | 2.381192e-04 | 2.381192e-04 | 0.1: 6.1e-05; 0.01: 6.7e-06; 0.001: 4.7e-07; 0.0001: 2.7e-07 | 4/4 h (0.0001..0.1) | 4.4e-03 | 27641132 / 27641755 |
| 28 d | theta_B_above | 4.786285e-05 | 4.786285e-05 | 0.1: 7.6e-05; 0.01: 4.4e-05; 0.001: 3.4e-05; 0.0001: 1.5e-06 | 4/4 h (0.0001..0.1) | 1.1e-01 | 27641132 / 27641755 |
| 28 d | theta_C_south | 5.743817e-06 | 5.743817e-06 | 0.1: 9.5e-05; 0.01: 1.2e-05; 0.001: 2.0e-07; 0.0001: 7.2e-06 | 4/4 h (0.0001..0.1) | 2.5e-01 | 27641132 / 27641755 |

Forward noise floor (spread of repeated J evaluations at the base point, same process): 7 d: 0.0e+00 (J = 12.626517387977033, 24.0 s/forward); 14 d: 0.0e+00 (J = 12.619516747988408, 47.6 s/forward); 28 d: 0.0e+00 (J = 12.626699320687701, 95.5 s/forward)

### TL (jax.jvp) vs adjoint (chunked jax.vjp): combined direction

| window | mode | amplitude | TL | adjoint <g, v> | rel. difference | job |
|---|---|---|---|---|---|---|
| 7 d | exact | 1 | -9.514260072388146e-02 | -9.514260072388464e-02 | 3.3e-14 | 27641126 |
| 7 d | exact | 1e-06 | -9.514260072388153e-02 | -9.514260072388464e-02 | 3.3e-14 | 27641126 |
| 14 d | exact | 1 | -1.992375679715089e-01 | -1.992375679715349e-01 | 1.3e-13 | 27641755 |
| 14 d | exact | 1e-06 | -1.992375679715091e-01 | -1.992375679715349e-01 | 1.3e-13 | 27641755 |
| 28 d | exact | 1 | -4.691552685655884e-01 | -4.691552685657399e-01 | 3.2e-13 | 27641132 |
| 28 d | exact | 1e-06 | -4.691552685655882e-01 | -4.691552685657399e-01 | 3.2e-13 | 27641132 |

Per named direction, TL vs adjoint (max rel. difference over directions): 7 d: 8.6e-13; 14 d: 1.0e-12; 28 d: 2.9e-12

### Amplification screen (terminal seed: end-of-window box mean; per-chunk State-cotangent norm)

Statistics over the chunks after the seed chunk (the first chunk maps the theta/hFacC seed onto the whole State: a change of norm between field sets, not growth); carried constants left out; the dynamic norm also without the chunk ending at the window start (AB2 start: gu/gvNm_2 unread).

| window | mode | freezes | chunk | norm over | median /step | log-spread | worst-3 /step | passes | norm at window end -> start | job |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | dynamic | 0.99769 | 0.0040 | 0.99774 | yes | 6.895e+01 -> 1.261e+01 | 27641127 |
| 7 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | theta | 0.99997 | 0.0015 | 1.00038 | yes | 1.882e-02 -> 1.863e-02 | 27641127 |
| 7 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | prognostic | 0.99896 | 0.0042 | 1.00090 | yes | 5.897e-02 -> 5.592e-02 | 27641127 |
| 7 d | exact | none | 24 | dynamic | 0.99768 | 0.0041 | 0.99773 | yes | 6.879e+01 -> 1.253e+01 | 27641126 |
| 7 d | exact | none | 24 | theta | 1.00004 | 0.0014 | 1.00038 | yes | 1.876e-02 -> 1.856e-02 | 27641126 |
| 7 d | exact | none | 24 | prognostic | 0.99905 | 0.0041 | 1.00089 | yes | 5.831e-02 -> 5.478e-02 | 27641126 |
| 14 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | dynamic | 0.99761 | 0.0028 | 0.99972 | yes | 6.860e+01 -> 9.275e+00 | 27641755 |
| 14 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | theta | 0.99995 | 0.0011 | 1.00036 | yes | 1.882e-02 -> 1.890e-02 | 27641755 |
| 14 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | prognostic | 0.99990 | 0.0033 | 1.00139 | yes | 5.868e-02 -> 5.927e-02 | 27641755 |
| 14 d | exact | none | 24 | dynamic | 0.99764 | 0.0029 | 0.99972 | yes | 6.846e+01 -> 9.219e+00 | 27641755 |
| 14 d | exact | none | 24 | theta | 1.00002 | 0.0011 | 1.00038 | yes | 1.877e-02 -> 1.883e-02 | 27641755 |
| 14 d | exact | none | 24 | prognostic | 1.00009 | 0.0032 | 1.00132 | yes | 5.809e-02 -> 5.788e-02 | 27641755 |
| 28 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | dynamic | 0.99891 | 0.0021 | 1.00060 | yes | 7.044e+01 -> 8.125e+00 | 27641133 |
| 28 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | theta | 1.00015 | 0.0008 | 1.00040 | yes | 1.902e-02 -> 2.012e-02 | 27641133 |
| 28 d | ecco | ggl90=frozen, gm_sigma=stable, cg2d=passive, visc_fac_in_ad=1.0 | 24 | prognostic | 1.00036 | 0.0022 | 1.00131 | yes | 6.089e-02 -> 7.139e-02 | 27641133 |
| 28 d | exact | none | 24 | dynamic | 0.99892 | 0.0022 | 1.00060 | yes | 7.025e+01 -> 8.072e+00 | 27641132 |
| 28 d | exact | none | 24 | theta | 1.00010 | 0.0008 | 1.00040 | yes | 1.895e-02 -> 2.004e-02 | 27641132 |
| 28 d | exact | none | 24 | prognostic | 1.00034 | 0.0021 | 1.00123 | yes | 6.014e-02 -> 6.953e-02 | 27641132 |

### Gradient runs: repeats, ecco vs exact, cost

| window | mode | repeat | J | norm g_theta0 | norm g_kapGM | norm g_tflux | norm g_taux | max rel. diff to r0 (theta0, kapGM, tflux) | forward s | reverse s | rev/fwd | device peak GB | host GB | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 d | ecco | 0 | 12.626517387977033 | 0.01733 | 3.400e-07 | 3.108e-07 | 0.0261 | - | 75 | 138 | 1.85 | 41.47 | 15.21 | 27641127 |
| 7 d | ecco | 1 | 12.626517387977033 | 0.01733 | 3.400e-07 | 3.108e-07 | 0.0261 | 1.9e-14, 2.6e-15, 6.0e-15 (J bitwise) | 38 | 65 | 1.70 | 41.73 | 15.21 | 27641127 |
| 7 d | ecco | 2 | 12.626517387977033 | 0.01733 | 3.400e-07 | 3.108e-07 | 0.0261 | 1.4e-14, 1.9e-15, 5.7e-15 (J bitwise) | 37 | 65 | 1.76 | 41.73 | 15.21 | 27641127 |
| 7 d | exact | 0 | 12.626517387977033 | 0.01733 | 3.445e-07 | 3.216e-07 | 0.02613 | - | 74 | 149 | 2.02 | 45.64 | 15.21 | 27641126 |
| 14 d | ecco | 0 | 12.619516747988406 | 0.01757 | 6.464e-07 | 7.470e-07 | 0.0479 | - | 118 | 216 | 1.82 | 41.58 | 28.51 | 27641755 |
| 14 d | ecco | 1 | 12.619516747988406 | 0.01757 | 6.464e-07 | 7.470e-07 | 0.0479 | 1.8e-14, 1.8e-15, 1.9e-15 (J bitwise) | 77 | 131 | 1.72 | 41.74 | 28.51 | 27641755 |
| 14 d | ecco | 2 | 12.619516747988406 | 0.01757 | 6.464e-07 | 7.470e-07 | 0.0479 | 1.3e-14, 1.5e-15, 1.6e-15 (J bitwise) | 77 | 132 | 1.71 | 41.75 | 28.51 | 27641755 |
| 14 d | exact | 0 | 12.619516747988406 | 0.01758 | 6.610e-07 | 7.556e-07 | 0.04802 | - | 111 | 232 | 2.10 | 45.59 | 28.51 | 27641755 |
| 28 d | ecco | 0 | 12.626699320687701 | 0.0179 | 1.207e-06 | 1.817e-06 | 0.08784 | - | 191 | 337 | 1.76 | 41.59 | 55.12 | 27641133 |
| 28 d | ecco | 1 | 12.626699320687701 | 0.0179 | 1.207e-06 | 1.817e-06 | 0.08784 | 2.7e-14, 2.0e-15, 1.1e-15 (J bitwise) | 153 | 261 | 1.70 | 41.77 | 55.12 | 27641133 |
| 28 d | ecco | 2 | 12.626699320687701 | 0.0179 | 1.207e-06 | 1.817e-06 | 0.08784 | 3.1e-14, 1.7e-15, 1.2e-15 (J bitwise) | 154 | 261 | 1.70 | 41.77 | 55.12 | 27641133 |
| 28 d | exact | 0 | 12.626699320687701 | 0.01793 | 1.252e-06 | 1.837e-06 | 0.0882 | - | 181 | 374 | 2.07 | 45.62 | 55.12 | 27641132 |

Warm gradient (chunked forward + reverse, ecco repeats) against one plain jitted forward of the same window: 7 d: 103 s = 4.3 x one plain forward (24.0 s, 0.143 s/step); 14 d: 208 s = 4.4 x one plain forward (47.6 s, 0.142 s/step); 28 d: 414 s = 4.3 x one plain forward (95.5 s, 0.142 s/step)

ECCO mode vs exact mode, relative difference of the directional derivatives:

| window | per direction |
|---|---|
| 7 d | theta_A_centre 1.3e-03; theta_B_above 6.4e-02; theta_C_south 6.5e-02; kapGM_scale 5.4e-03; tflux_box 3.1e-02; taux_box 2.2e-02 |
| 14 d | theta_A_centre 3.4e-04; theta_B_above 9.9e-02; theta_C_south 1.4e-01; kapGM_scale 2.2e-03; tflux_box 1.9e-02; taux_box 2.8e-02 |
| 28 d | theta_A_centre 4.4e-03; theta_B_above 1.1e-01; theta_C_south 2.5e-01; kapGM_scale 6.9e-03; tflux_box 2.1e-02; taux_box 3.6e-02 |

ECCO-mode forward vs exact-mode forward (whole window, all State fields, bytes): 7 d: bitwise (90 fields, J 12.626517387977033, job 27641127); 14 d: bitwise (90 fields, J 12.619516747988406, job 27641755); 28 d: bitwise (90 fields, J 12.626699320687701, job 27641133)

## 4. Figures

In `/work/ab0995/a270088/MIT/runs/adjoint/figures/` (not in the repository: home quota):

| file | content |
|---|---|
| `amplification_traces.png` | per-chunk cotangent norm of the screen runs: all State fields (dominated by the runoff accumulation), dynamic fields, theta alone; exact and ecco, 7/14/28 d |
| `fd_sweeps.png` | relative FD error (FD vs adjoint) against h for every named direction, exact (solid) and ecco (dashed) adjoint, all windows |
| `dJdtheta0_k01_{exact,ecco}_28d.png` | dJ/dtheta0 at the surface (level 1, 5 m), Robinson |
| `dJdtheta0_k17_{exact,ecco}_28d.png`, `..._pacific.png` | dJ/dtheta0 at 195 m (level 17, inside the box), global and western Pacific (scale saturated at 10 % to show the far field) |
| `dJdlnkapGM_k{01,17}_{exact,ecco}_28d_pacific.png` | dJ/d ln kapGM at 5 m and 195 m |
| `dJd{tflux,taux,tauy}_{exact,ecco}_28d{,_pacific}.png` | sensitivity to time-constant TFLUX / oceTAUX / oceTAUY shifts |
| `dJdtheta0_diff_ecco_28d_k{01,17}_pacific.png` | ecco minus exact dJ/dtheta0 at 5 m and 195 m |

What the maps show (28 d): inside the box dJ/dtheta0 ~ 2e-4 per cell (the cell's share of the box volume);
the far field is upstream: east of the dateline along the North Equatorial Current band and around the
Philippines, up to ~1e-5 per cell at 195 m. At the surface the sensitivity is 1e-6 per cell, in zonal stripes at the
box latitudes. The oceTAUX map jumps sign at the face-2/face-4 edge near 143E because the control is the model-grid
x component (face 4 is rotated); `taux_box` is therefore a model-grid direction, not an eastward stress.

## 5. Findings

1. **The exact adjoint of the flux-forced configuration is usable over the whole 28-day window.** Every named control
   has an FD plateau against the exact gradient in every window (7, 14, 28 d): interior theta points to 1e-9..1e-6 at
   h = 1e-3..1e-4 K, kapGM and oceTAUX to 1e-6..2e-4, the TFLUX shift to 3e-4..2e-3. The TL model (`jax.jvp`)
   agrees with the adjoint to 3e-13 on the combined direction at amplitudes 1 and 1e-6 (linear) and to <= 3e-12 on
   each named direction. The reverse sweep does not amplify: theta-only and prognostic norms change by 0.9990-1.0004
   per step (median; worst 3 chunks <= 1.0014) in both modes, far inside the fesom_jax bars. Unlike fesom_jax (1.2-1.5/step unfrozen), no
   freeze is needed for stability at 4 weeks: here GGL90's state dependence and the GM slopes do not blow up in
   January 1992 over this box. Windows beyond 28 d were not tried (the task asked for 2-4 weeks); nothing in the
   traces limits them yet.
2. **ECCO mode = exact forward, different gradient.** The ecco-mode forward is bitwise the exact-mode forward over the
   whole window in every window (all 90 State fields, J). The ecco gradient is an approximation of the true one: it
   differs from the exact gradient (and from FD) by 0.03-0.4 % inside the box (theta_A), 0.2-0.7 % for kapGM,
   2-4 % for the TFLUX/oceTAUX shifts, and 6-25 % for theta points outside the box (theta_B above the box top: 6.4 %
   -> 11 %; theta_C south of it: 6.5 % -> 25 %, growing with the window): the sensitivity that reaches the box through
   vertical mixing is what freezing GGL90 (and cutting the sigma derivatives) removes. The maps show the differences
   concentrated at the box's northern and southern edges and its western end.
3. **Repeats.** Three ecco-mode gradients per window: J bitwise, gradients within 1.9e-14 (7 d), 1.8e-14 (14 d),
   3.1e-14 (28 d) relative (max over theta0; kapGM and the forcing shifts ~2e-15), i.e. far below Task 18's
   4e-11 GPU floor. The exact mode's one-day repeat: 1.3e-12 (job 27640768). The forward is deterministic
   (FD noise floor: repeated J evaluations have zero spread in every window).
4. **FD precision is set by switches, not by noise.** Interior theta perturbations give plateaus down to 1e-9 at
   h = 1e-3; the surface-forcing direction (TFLUX over 624 columns) stays at 3e-4-2e-3 for h from 1e-2 to 10 W/m^2. An error
   that does not shrink with h over three decades is the signature of many small discontinuities (IVDC convective
   switching, clipping) whose number grows with h, not of truncation or round-off. The TL agrees with the adjoint to
   1e-12 in the same direction, so the adjoint is the exact derivative of the piecewise-smooth forward.
5. **Cost.** One A100-80, chunks of 24 steps, stride 1, per-step remat: device peak 41.6 GB (ecco) / 45.6 GB (exact),
   flat from 7 to 28 days; host 15 / 29 / 55 GB (one 1.9 GB State per day boundary). Warm gradient (forward with the
   boundary transfers + reverse) = 4.3-4.4 x one plain forward of the window (28 d: 414 s vs 95.5 s; 0.142 s per
   forward step); reverse / chunked forward = 1.7 (ecco) - 2.1 (exact). First call (compilation included): +2 min.
   The TL costs ~1.9-2.0 plain forwards per direction.

## 6. Lessons

- **A field the step carries unchanged fakes growth in the screen.** `runoff` (and `sIceLoad`, hMixLayer, ...) sit in
  the State but FORWARD_STEP never rewrites them; their cotangent is the gradient with respect to a constant field and
  grows linearly over the window (runoff: 1e3 after 7 days). On the full-State norm it read as a steady 1.0105/step,
  failing the median bar with nothing amplifying. The pass-through fields are found exactly from the step's jaxpr
  (`multiweek_grad.carried_constants`) and left out.
- **Screen with a terminal seed, and skip the chunks that change field sets.** A running (time-mean) cost injects
  cotangent in every chunk, so its trace grows by accumulation. With a terminal seed, the first chunk still maps the
  theta/hFacC seed onto the whole State (x1e2-1e3 "growth" in units, not dynamics), and the chunk ending at iteration 1
  loses the gu/gvNm_2 cotangent (AB2 start). The per-field trace (`ChunkedGrad.field_trace`) is what made both visible.
- **FD on this GPU forward is noise-free but switch-limited.** J repeats bitwise, so the classic noise floor is 0;
  interior theta directions plateau at 1e-9-1e-6, while the TFLUX direction stays at 3e-4-2e-3 and the global kapGM scale at
  1e-5-2e-4 for every h over three to four decades: many small discontinuities, not
  an adjoint error (the TL agrees with the adjoint to 1e-12 along the same directions).
- **The exact adjoint does not need the ECCO freezes at 4 weeks here**, and the freezes are not free: the ecco gradient
  is 2-4 % off for the forcing shifts and 6-25 % off for theta just outside the box (growing with the window). The
  freezes are a modelling choice to be justified by a longer window or the full V4r4 (sea ice, bulk formulae), not by
  this one.
- **Pass the model as jit arguments in every new driver, including the TL.** The first TL (`jax.jvp` in a lambda that
  closed over the model) constant-folded the grid for minutes and its executable failed to load next to the cached
  gradient executables (CUDA_ERROR_OUT_OF_MEMORY on the CUBIN): model and state as arguments, one heavy stage per
  process, `_CHUNK_CACHE` cleared between stages.
- **Scheduling:** the account's GPU limit is 5 running jobs (not GPUs); one 4-GPU job with one process per GPU uses one
  slot (`multiweek_grad.sbatch --multi`), but a whole 4-GPU node can wait hours (estimated 17:31 at 14:40) while single
  GPUs start within a minute.

## 7. Reproduce

    # once, CPU, gate flags (bitwise the Fortran start-of-run state); writes .../runs/adjoint/init_ref_ff_serial13_1day
    XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp" JAX_PLATFORMS=cpu \
      python scripts/adjoint/multiweek_grad.py --out <dir> --actions cache
    # per window W (days), one A100-80 each
    sbatch scripts/adjoint/multiweek_grad.sbatch --out $R/wW_exact --days W --mode exact --actions grad,screen,tl
    sbatch scripts/adjoint/multiweek_grad.sbatch --out $R/wW_ecco  --days W --mode ecco --actions fwdcheck,grad,screen --repeats 3
    sbatch scripts/adjoint/multiweek_grad.sbatch --out $R/wW_fd    --days W --mode exact --actions fd [--dirs ...]
    python3 scripts/adjoint/summarize_results.py $R/w*_*
    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/adjoint/plot_sensitivity.py --runs ... --figdir ...
    # tier 2: mitgcm_jax/tests/test_adjoint_regression.py (one day, both modes, recorded values): passed, 338 s,
    #   job 27644579 (sbatch /work/ab0995/a270088/MIT/dev/adjoint21/tier2_regression.sbatch)

Wall clock on one A100-80 (compilation included): 28-day exact gradient + screen + 8 TL directions 45 min; ecco
forward check + 3 gradients + screen 37 min; 12 FD evaluations of the 28-day J ~20 min.

## 8. Open questions (for Nikolay)

1. Box: the adjsen script's condition `YC <= 151` makes the experiment's box 120E-180E, not the 120E-151E its comment
   says. Used as written; switch to 151E?
2. Scale: the literal adjsen J is divided twice by the box volume (objmask/totvol, then ecco_phys's areavolGlob);
   J here is in K. Keep K, or reproduce the literal scaling for the M3 TAF comparison?
3. The plan says "sharded on one node". These runs are on one GPU: Task 20 measured 4 GPUs slower than 1 on LLC90
   (0.43 vs 0.34 s/step), and the chunked driver is not wired to `ShardedModel` (its step takes padded, sharded
   dicts). Is the 1-GPU multi-week gradient plus the Task 20 sharded-gradient check enough for M1, or should the
   sharded step be put behind `checkpoint.integrate` first?
4. Which mode for science: the exact adjoint is stable and FD-exact to 28 days in the flux-forced model; the ECCO
   freezes change the gradient by up to 25 % for points outside the box. Keep ecco as the default for ECCO
   compatibility, or exact where it is stable (re-screen every window)?
5. The dynamic-field norm is dominated by the momentum AB-history cotangents (units m/s^2); a scaled norm (per-field
   RMS weights) would make the screen independent of units. Worth adopting?

## M2: sea-ice-only adjoint window, before ocean coupling (2026-09-23)
SEAICE_MODEL stepped on its own carried state (18 fields) with the ocean and EXF inputs of oracle.FULL iteration 1
prescribed (harness gate: cycling iterations 1-3 reproduces P00 bitwise); controls initial HEFF, AREA, UICE, VICE and
time-constant shifts of atemp, fu, fv; costs Arctic (>70N) ice volume J1 and area J3, Southern Ocean (<60S) volume J2,
after 6, 24, 48 one-hour steps from 1992-01-01; levels ecco, no_dynamics, full; FD with the LSR converged to 1e-12
(same code path). scripts/adjoint/seaice_window.py (--summarize), rows in runs/seaice_adjoint/w{06,24,48}.
1. **full is exact**: TL/adjoint 1e-14 to 8e-14; tight-FD plateaus 1e-10 to 2e-4 for J1, J3 along every control at 48
   steps; repeats bitwise; no amplification (HEFF cotangent 0.9995/step; velocity cotangents accumulate linearly).
2. **ecco is an exact identity** over the window (no sea-ice sensitivity at all). no_dynamics matches full to 1e-4 to
   5e-3 for thickness and air-temperature directions (J1) but is zero or wrong-signed for stress and velocity.
3. **The production LSR tolerance** (LSR_ERROR = 2e-4) breaks FD along dynamics directions (1-25 % for J1, up to 330 %
   for J3) and shifts the full (implicit, converged-system) gradient by up to 12 %.
4. **Southern Ocean summer**: cells with ice and exactly zero snow get 1e-26 m of snow under a 1e-8 perturbation and
   take the snow branch (HEFF jump ~1e-5 m): FD error ~1/h for J2 except along HEFF.
5. Cost (16 CPU cores): forward 0.65 s/step (production tolerance), 21-51 s/step tight; reverse ecco 0.04 s, no_dynamics
   0.6-1.0 s, full 40-46 s per step and cost function (GMRES(40) x 8 per Picard pass, Arnoldi-dominated).
