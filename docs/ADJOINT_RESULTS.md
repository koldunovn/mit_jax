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

## Adjoint horizon beyond 28 days, flux-forced (2026-09-24, GH200)
Task 21 recipe (J = running box-mean theta in K from 1992-01-01; controls theta/kapGM/tflux/taux/tauy) on one GH200;
windows 56-365 d in exact and ecco mode, 130/150/240/300-d exact screens, 365-d single-switch screens. Code: branch
`adjoint-horizon` (not yet merged); data /work/.../MIT/runs/adjoint_horizon/ (figures/).
- **Exact passes every bar to 112 d** (worst-3 screens <= 1.0019/step; FD 7/7 directions, best errors 4.9e-5..2.7e-4;
  TL/adjoint <= 4e-10; repeats <= 5.9e-11, J bitwise).
- **First failure: dynamic-norm screen between 130 and 150 d**; prognostic fails at 182-240 d, theta at 240-300 d.
- **At 365 d the far-field gradient is wrong**: dJ/d ln kapGM adjoint -0.115 (TL agrees to 6e-4) vs FD +0.0657
  (converged); eastern-Pacific theta sensitivity ~1e3 too large. Near-box directions stay <= 3e-3 from FD.
- **Cause: one event**, reverse days ~127 -> 104 (mid-April to early May 1992): the linearised GGL90 closure is unstable
  in the eastern equatorial Pacific surface layer (5-25 m, 84-100W, 0-2N); damage grows with the sensitivity that has
  reached there (dynamic-norm growth 22x at 182 d, 2500x at 365 d).
- **ggl90="frozen" alone removes it for a year** (screens equal to full ECCO); gm_sigma "stable" or "gm_only" alone fail
  and amplify more (cutting a damping N^2 derivative while keeping the unstable shear loop). Opposite to fesom_jax.
- ECCO vs FD grows with the window: 56 d up to 35 %, 182 d 1.4-52 %, 365 d 3-17 % near the box; GGL90-only is closer
  to FD than ECCO on most directions.
- Medians never see the event (365 d: 1.00005/step), worst-3 does, the per-field screen sees it one window earlier (TKE
  at 130 d). Repeats (5e-12) and the TL dot test (1e-7) cannot see a deterministic linear instability: put an FD probe
  where the screen localises the burst.
- GH200 cost: forward 0.081 s/step, warm gradient 0.36-0.40 s/step (4.5-5 forwards), device peak 43 GB flat, host RSS
  <= 107 GB at 365 d (GH200 host memory ~122 GB per GPU: EXF buffers stored once, boundary stride ~sqrt(chunks)).

# M2 adjoint acceptance: multi-week gradients of the full V4r4 model (plan Task 22)

2026-09-24. The JAX port of the **full** V4r4 configuration (EXF bulk formulae + sea ice), production set-up, NVIDIA
GH200 (dolpung), one GPU per run; the sharded check on 4 GH200 of one node. Driver: `scripts/adjoint/fullgrad.py`
(imports the M1 driver `multiweek_grad.py` unchanged; `fullgrad_dolpung.sbatch`), tables
`scripts/adjoint/fullgrad_summary.py`, figures `scripts/adjoint/plot_fullgrad.py` (nereus env). Rows (job id, git
HEAD, device, XLA flags, mode and freezes, window, chunk, stride, ice weight, h or seed, memory, times) in
`/work/ab0995/a270088/MIT/runs/adjoint_m2/w{07,14,28}/<run>/results.jsonl`, gradient fields next to them
(`grad_<mode>_<days>d_r<repeat>.npz`, interior points, float64). Branch `m2-acceptance` (worktree
`/work/ab0995/a270088/MIT/dev/wt_m2acc`); model code = master plus the `seaice_lsr._precond` shard_map fix
(finding 9, docs/PORTING_LESSONS.md).

## M2.1 Set-up

**Model.** Run directory `reference/runs/ref_full_serial13_1day` (production full tree: `useCTRL=T` with the WC01
initial-state and mixing adjustments, bulk formulae, SEAICE_MODEL, 6-hourly adjusted ERA-interim), start 1992-01-01
(`nIter0=1`), dt = 3600 s. Initial state from `init.state_from_pickup` (incl. `pickup_seaice`), computed once on CPU
with the gate XLA flags (bitwise the Fortran start-of-run state, M2.6a) and cached in
`/work/ab0995/a270088/MIT/runs/adjoint_m2/init_ref_full_serial13_1day` (setup 89 s + init 107 s, 32 cores; 125 State
fields). GPU runs: XLA default flags, `sum_unroll=5`, LSR forward sweep = Pallas kernel (`lsr_impl` auto). Windows 7,
14, 28 days = 168, 336, 672 steps.

**Cost.** `J = J_theta + J_ice`:
- `J_theta` = the Task 21 adjsen box-mean theta, running mean over the window, in K (the adjsen box as written:
  120E-180E, 5N-16N, levels 15-20, 3743 wet cells; `multiweek_grad --j-scaling kelvin` semantics);
- `J_ice` = Arctic ice volume at the end of the window divided by the fixed ocean area north of 70N (6092 surface
  cells, 1.124e13 m^2): the Arctic-mean effective ice thickness in m (same region as J1 of the sea-ice-only study).
J = 13.751 (7 d; J_theta 12.626 K, J_ice 1.125 m), 13.817 (14 d), 13.940 (28 d; 12.626 K, 1.314 m). The two parts live
in different regions; each named direction tests one or the other (per-part FD: FD rows record both parts, the J_theta
adjoint comes from a gradient with `--ice-weight 0`, the J_ice adjoint as the difference).

**Controls** (one pytree; zero = the production forward, value-identical):

| control | shape | how it enters |
|---|---|---|
| `theta`, `kapGM` | 3-D | as Task 21 (xx_theta / xx_kapgm paths: interior add + EXCH_XYZ_RL) |
| `heff` | 2-D | initial HEFF (interior add + EXCH_XY_RL); not a V4r4 control: the sea-ice sensitivity map |
| `atemp`, `aqh`, `tauu`, `tauv`, `swdown`, `lwdown`, `precip` | 2-D | time-constant adjustments of the EXF atmospheric state (the V4r4 gentim2d set, `data.ctrl.iter0.inclatmctrl`) in the units and sign of the EXF field after EXF_SET_FLD: added to both record buffers of every step divided by `exf_inscal_<field>` (-1 for ustress, vstress, swdown, lwdown), so the interpolated field moves by the adjustment; they enter before EXF_RADIATION / EXF_WIND / EXF_BULKFORMULAE like ctrl_map_gentim2d's xx_atemp, ... . `tauu/tauv` are eastward/northward stress before the A-grid rotation; unlike xx_tauu (added in EXF_GETSURFACEFLUXES, after EXF_WIND) they also enter EXF_WIND (wStress, cw/sw -> uwind/vwind of the sea-ice air drag). |

**Named directions** (Fortran indices; FD and TL):

| name | definition |
|---|---|
| theta_A_centre, theta_B_above, theta_C_south, kapGM_scale | as Task 21 (box centre 149.5E 10.5N 195 m; 140 m above; 3.6N south; dJ/d ln kapGM) |
| atemp_box | +1 K atemp over the 624 box columns |
| tauu_box | +1 N/m^2 eastward stress over the 624 box columns |
| atemp_arctic | +1 K atemp over the 6092 ocean columns north of 70N |
| heff_arctic | direction = HEFF0 north of 70N (dJ/d ln HEFF0) |
| atemp_pt | +1 K atemp at the surface column of theta_A (one column) |
| atemp_arctic_pt, heff_pt | +1 K atemp / +1 m HEFF0 at one Canada Basin column (150.5W, 77.8N; HEFF0 = 1.61 m) |

**Modes** (forward values identical, checked): `ecco` = `AdjointConfig.ecco(nml)` of the full run directory (seaice
"ecco" = SEAICE_MODEL skipped in reverse, ggl90 frozen, salt_plume off, gm_sigma stable, cg2d passive, viscFacInAd 1);
`exact_nodyn` = `AdjointConfig(seaice="no_dynamics")` (exact ocean, sea-ice thermodynamics/advection adjoint, LSR
skipped in reverse); `exact_full` = `AdjointConfig(seaice="full")` (exact; LSR by its implicit derivative). The FD
reference is exact_full where it was run (7, 14 d) and exact_nodyn at 28 d. Note (docs/ADJOINT_MODES.md, Nikolay
2026-09-23): the full derivative is that of the converged LSR system; the production forward stops at LSR_ERROR = 2e-4.

**Drivers, screen.** As Task 21: chunks of 24 steps, stride 1, per-step remat, cg2d solution saved; FD and TL on one
jitted whole-window scan. Screen: terminal seed (end-of-window box mean + J_ice), per-field chunk-boundary norms; groups
dynamic (all but the carried constants gt/gsNm_1,2, hFac_surfC/W/S, saltflx, snowprecip), theta, prognostic
(theta, salt, uVel, vVel, etaN), seaice (AREA, HEFF, HSNOW, TICES, UICE, VICE) and seaice_thermo (without UICE,
VICE, whose cotangent accumulates in exact_nodyn: the skipped LSR is the identity).

## M2.2 Acceptance summary

One GH200 per run; every number is from the tables in M2.3 (job ids there). FD reference: exact_full at 7 and 14 d,
exact_nodyn at 28 d (exact_full not run there). "part" = the FD of the cost part the direction acts on (J_theta: box;
J_ice: Arctic) against that part's adjoint.

| criterion | bar | 7 d (168 steps) | 14 d (336 steps) | 28 d (672 steps) |
|---|---|---|---|---|
| **exact**: FD plateau, ocean state + stress: theta_A/B/C, tauu_box, J_theta part of kapGM_scale | >= 2 of 4 h with rel. error <= 1e-3 | 5/5 (all 4/4; 1e-6 .. 5e-4) | 5/5 (4/4) | 5/5 (4/4) |
| **exact**: FD plateau, sea-ice thickness: heff_arctic, heff_pt | same | 2/2 (4/4; 8e-5 .. 7e-4) | heff_arctic 4/4; heff_pt 1/4 (2.6e-3 .. 2.8e-3 at 3 of 4 h, constant in h) | 0/2 vs exact_nodyn (constant 5.5e-3: the dynamics derivative nodyn leaves out) |
| **exact**: FD with a converged LSR forward (LSR_ERROR 1e-8), heff_pt | same | - | 2/2 h (2.5e-4, 3.0e-4; production forward: 2.7e-3, 2.8e-3 at the same h) | - |
| **exact**: FD plateau, air temperature, one column: atemp_pt (J_theta), atemp_arctic_pt (J_ice) | same | 2/2 (2/4, 3/4) | 0/2 (1/4; 0/4 at a constant 2.9e-3 .. 3.6e-3) | 0/2 (0/4, 0/4) |
| **exact**: FD, footprint / global directions: atemp_box, atemp_arctic, J_ice part of kapGM_scale | recorded (switch-limited) | 0/4 (2e-3 .. 9e-3); 1/4; 0/4 (4e-3 .. 3e-2) | 1/4; 1/4; 0/4 | 2/4; 0/4; 0/4 |
| **exact**: TL (jax.jvp) vs adjoint, combined direction, amplitude 1 and 1e-6 | round-off | full 3.0e-13, 3.0e-13; nodyn 3.9e-13, 4.0e-13 | full 1.1e-11, 1.1e-11; nodyn 9.1e-12, 9.1e-12 | nodyn 3.4e-12, 3.4e-12 |
| **exact**: TL vs adjoint, each named direction (max) | round-off | nodyn 2.1e-11 | nodyn 4.7e-10 | nodyn 2.6e-10 |
| **exact**: screen, theta / prognostic / sea-ice thermo: median, worst-3 per step | <= 1.010, <= 1.030 (log-spread <= 0.020) | full 1.00007/1.00041; 0.99905/1.00089; 1.00020/1.00035: pass (nodyn the same) | full 1.00004/1.00040; 1.00011/1.00132; 1.00068/1.00089: pass (nodyn the same) | nodyn 1.00010/1.00041; 1.00034/1.00123; 1.00030/1.00090: pass |
| **ecco**: forward == exact forwards (bytes, 126 State fields + both cost parts, whole window) | bitwise | bitwise (vs nodyn and full) | bitwise | bitwise |
| **ecco**: screen (same norms) | as above | 0.99997/1.00038; 0.99895/1.00090; 1/1: pass | 0.99995/1.00036; 0.99990/1.00139; 1/1: pass | 1.00015/1.00040; 1.00036/1.00131; 1/1: pass |
| **ecco**: 3 repeats | within the GPU floor (4e-11) | J bitwise; 1.3e-14 | J bitwise; 1.2e-14 | J bitwise; 1.8e-14 |
| exact repeats (2) | recorded | J bitwise; full 5.0e-11, nodyn 4.6e-11 (theta0; others <= 1e-12) | nodyn 9.1e-11 | nodyn 1.2e-11 |
| **ecco** TL vs adjoint | round-off | 0 / 1.3e-16 (named 4e-11 on a 1e-9 value) | 5.6e-16 / 2.8e-16 | 4.0e-16 / 8.1e-16 |
| **sharded** (4 GH200, shard_map) vs 1 GPU | within the repeat floor | ecco 7 d: final State bitwise, gradient 1.1e-14 (floor 1.3e-14); exact_full 2 d: J bitwise, gradient 1.6e-11 (exact floor 5e-11) | - | - |
| cost: warm gradient / plain forward; device peak; host | - | ecco 3.2 (138 s vs 43.8 s), nodyn 3.2, full 26.5 (1161 s); 44 / 47 / 52 GB; 15.5 GB | ecco 3.1 (274 s vs 87 s); 44 / 47 / 51 GB; 29 GB | ecco 3.2 (582 s vs 184 s), nodyn 3.2; 44 / 47 GB; 56 GB |

**Verdict against the M1 bars.** Passed in every window and mode: forward identity of the modes, the TL/adjoint dot test
(linear; beyond 1e-12 only where the two programs linearise slightly different sea-ice trajectories, finding 6), the
amplification screen, the ecco repeats, and the sharded comparison. The FD plateau bar is passed by the ocean state and
the wind stress in every window, by the Arctic ice-thickness footprint against exact_full (7, 14 d), and by the
single-column air-temperature and ice-thickness directions at 7 days; it is **not** passed by air-temperature footprints
(switch-limited: the error does not shrink with h) and, from 14 days on, by single sea-ice columns, whose FD differs
from the implicit-LSR adjoint by a constant ~3e-3 at every h (28 d: ~6e-3 against exact_nodyn, which leaves out the
dynamics derivative; exact_full was not run at 28 d). That offset is the production LSR tolerance: with the LSR
converged to 1e-8 (same code path) the FD of heff_pt at 14 d agrees with the exact_full adjoint to 2.5e-4 / 3.0e-4
(production forward 2.7e-3 / 2.8e-3), as docs/ADJOINT_MODES.md prescribes for FD checks of dynamics directions.

## M2.3 Tables

Generated by `python3 scripts/adjoint/fullgrad_summary.py <run dirs>` in `/work/ab0995/a270088/MIT/runs/adjoint_m2`
(7 d: w07/{ecco,nodyn,full,fd,fd_pt,fd_parts,nodyn_ice0,full_ice0,p4_ecco}, 14 d: w14/..., 28 d: w28/...;
--ref-mode exact_nodyn at 28 d). Every run: chunk 24 steps, stride 1, schedule step, sum_unroll 5, XLA default
flags, GH200. The 14-day rows of job 27655089 (ecco r0, nodyn r0/r1, full r0) ran with the host memory of four
processes on one NUMA node (docs/PORTING_LESSONS.md): their times are 4-12x too long; use the warm repeats.

### Window 7 days

#### FD h-sweep vs the exact_full adjoint, per named direction

Relative error |FD - AD| / |AD| at each h, AD = the exact_full adjoint (r0, J = J_theta + J_ice); plateau = number of h with rel. error <= 0.001 (bar: >= 2 of 4); J_theta / J_ice parts: the same against the ice-weight-0 adjoint (J_theta) and the difference (J_ice), where both parts were recorded and the part carries >= 0.1 % of the derivative; the other modes' adjoints relative to the best FD value.

| window | direction | AD exact_full | TL exact_full | FD rel. error at h = ... | plateau | J_theta part: errors, plateau | J_ice part: errors, plateau | other modes rel. to best FD | FD job |
|---|---|---|---|---|---|---|---|---|---|
| 7 d | atemp_arctic | -3.091089e-03 | - | 0.001: 1.5e-02; 0.0001: 3.4e-03; 1e-05: 3.5e-02; 1e-06: 1.3e-04 | 1/4 | not tested (3e-06 of AD) | 1.5e-02; 3.4e-03; 3.5e-02; 9.5e-05 (1/4) | exact_nodyn 1.8e-03; ecco 1.0e+00 | 27656226 |
| 7 d | atemp_arctic_pt | -2.372898e-07 | - | 0.1: 4.1e-04; 0.01: 5.9e-04; 0.001: 3.5e-04; 0.0001: 4.7e-03 | 3/4 | not tested (2e-06 of AD) | 4.1e-04; 3.9e-04; 2.4e-04; 1.7e-03 (3/4) | exact_nodyn 3.4e-04; ecco 1.0e+00 | 27656226 |
| 7 d | atemp_box | -4.449485e-05 | - | 0.1: 2.2e-03; 0.01: 2.0e-03; 0.001: 4.7e-03; 0.0001: 8.9e-03 | 0/4 | 2.2e-03; 2.0e-03; 4.7e-03; 8.9e-03 (0/4) | not tested (3e-05 of AD) | exact_nodyn 2.1e-03; ecco 4.9e-02 | 27656226 |
| 7 d | atemp_pt | 5.265918e-08 | - | 1: 6.5e-03; 0.1: 6.0e-03; 0.01: 4.2e-04; 0.001: 1.5e-03 | 1/4 | 6.5e-03; 6.0e-03; 4.7e-04; 8.4e-04 (2/4) | not tested (-2e-05 of AD) | exact_nodyn 4.4e-04; ecco 8.7e-02 | 27656226 |
| 7 d | heff_arctic | 1.002619e+00 | - | 0.01: 3.2e-04; 0.001: 1.7e-04; 0.0001: 2.4e-04; 1e-05: 7.8e-05 | 4/4 | not tested (2e-04 of AD) | 3.2e-04; 1.7e-04; 2.4e-04; 7.8e-05 (4/4) | exact_nodyn 2.3e-03; ecco 3.4e-02 | 27656226 |
| 7 d | heff_pt | 1.644130e-04 | - | 0.1: 7.4e-04; 0.01: 6.3e-04; 0.001: 6.3e-04; 0.0001: 6.3e-04 | 4/4 | not tested (2e-04 of AD) | 7.4e-04; 6.3e-04; 6.3e-04; 6.3e-04 (4/4) | exact_nodyn 1.8e-03; ecco 1.3e-02 | 27656226 |
| 7 d | kapGM_scale | 4.511588e-03 | - | 0.1: 6.5e-03; 0.01: 2.1e-02; 0.001: 2.6e-03; 0.0001: 2.8e-03 | 0/4 | 1.7e-05; 9.4e-06; 1.5e-04; 2.5e-04 (4/4) | 9.9e-03; 3.2e-02; 4.0e-03; 4.3e-03 (0/4) | exact_nodyn 2.2e-02; ecco 6.6e-01 | 27656226 |
| 7 d | tauu_box | -2.952547e-01 | - | 0.01: 3.6e-05; 0.001: 1.7e-06; 0.0001: 2.1e-06; 1e-05: 9.9e-06 | 4/4 | - | - | exact_nodyn 1.9e-05; ecco 4.7e-03 | 27655028 |
| 7 d | theta_A_centre | 2.335670e-04 | - | 0.1: 2.1e-05; 0.01: 2.0e-05; 0.001: 1.0e-06; 0.0001: 3.5e-06 | 4/4 | - | - | exact_nodyn 1.7e-06; ecco 1.4e-03 | 27655028 |
| 7 d | theta_B_above | 3.511726e-05 | - | 0.1: 1.1e-04; 0.01: 7.9e-05; 0.001: 3.6e-06; 0.0001: 3.7e-06 | 4/4 | - | - | exact_nodyn 6.4e-06; ecco 6.3e-02 | 27655028 |
| 7 d | theta_C_south | 4.431647e-06 | - | 0.1: 1.7e-04; 0.01: 1.8e-05; 0.001: 8.5e-05; 0.0001: 4.8e-04 | 4/4 | - | - | exact_nodyn 2.3e-06; ecco 6.5e-02 | 27655028 |

Forward noise floor 7 d: spread 0.0e+00 (J = 13.75120067706965, 43.7 s per plain forward, job 27656226)

#### FD h-sweep vs the exact_nodyn adjoint, per named direction

Relative error |FD - AD| / |AD| at each h, AD = the exact_nodyn adjoint (r0, J = J_theta + J_ice); plateau = number of h with rel. error <= 0.001 (bar: >= 2 of 4); J_theta / J_ice parts: the same against the ice-weight-0 adjoint (J_theta) and the difference (J_ice), where both parts were recorded and the part carries >= 0.1 % of the derivative; the other modes' adjoints relative to the best FD value.

| window | direction | AD exact_nodyn | TL exact_nodyn | FD rel. error at h = ... | plateau | J_theta part: errors, plateau | J_ice part: errors, plateau | other modes rel. to best FD | FD job |
|---|---|---|---|---|---|---|---|---|---|
| 7 d | atemp_arctic | -3.085197e-03 | -3.085197e-03 | 0.001: 1.3e-02; 0.0001: 1.5e-03; 1e-05: 3.3e-02; 1e-06: 1.8e-03 | 0/4 | not tested (4e-06 of AD) | 1.3e-02; 1.5e-03; 3.3e-02; 2.0e-03 (0/4) | exact_full 3.4e-03; ecco 1.0e+00 | 27656226 |
| 7 d | atemp_arctic_pt | -2.374526e-07 | - | 0.1: 1.1e-03; 0.01: 1.3e-03; 0.001: 3.4e-04; 0.0001: 4.0e-03 | 1/4 | not tested (2e-06 of AD) | 1.1e-03; 1.1e-03; 9.2e-04; 2.3e-03 (1/4) | exact_full 3.5e-04; ecco 1.0e+00 | 27656226 |
| 7 d | atemp_box | -4.449362e-05 | -4.449362e-05 | 0.1: 2.2e-03; 0.01: 2.1e-03; 0.001: 4.7e-03; 0.0001: 8.9e-03 | 0/4 | 2.2e-03; 2.0e-03; 4.7e-03; 8.9e-03 (0/4) | not tested (-7e-07 of AD) | exact_full 2.0e-03; ecco 4.9e-02 | 27656226 |
| 7 d | atemp_pt | 5.266042e-08 | - | 1: 6.5e-03; 0.1: 6.0e-03; 0.01: 4.4e-04; 0.001: 1.5e-03 | 1/4 | 6.5e-03; 6.0e-03; 4.7e-04; 8.4e-04 (2/4) | not tested (6e-07 of AD) | exact_full 4.2e-04; ecco 8.7e-02 | 27656226 |
| 7 d | heff_arctic | 1.004801e+00 | 1.004801e+00 | 0.01: 2.5e-03; 0.001: 2.3e-03; 0.0001: 2.4e-03; 1e-05: 2.2e-03 | 0/4 | not tested (2e-04 of AD) | 2.5e-03; 2.3e-03; 2.4e-03; 2.2e-03 (0/4) | exact_full 7.8e-05; ecco 3.4e-02 | 27656226 |
| 7 d | heff_pt | 1.646113e-04 | - | 0.1: 1.9e-03; 0.01: 1.8e-03; 0.001: 1.8e-03; 0.0001: 1.8e-03 | 0/4 | not tested (2e-04 of AD) | 1.9e-03; 1.8e-03; 1.8e-03; 1.8e-03 (0/4) | exact_full 6.3e-04; ecco 1.3e-02 | 27656226 |
| 7 d | kapGM_scale | 4.425845e-03 | 4.425845e-03 | 0.1: 1.3e-02; 0.01: 1.9e-03; 0.001: 2.2e-02; 0.0001: 2.2e-02 | 0/4 | 9.5e-06; 3.6e-05; 1.8e-04; 2.8e-04 (4/4) | 1.9e-02; 2.8e-03; 3.4e-02; 3.4e-02 (0/4) | exact_full 2.1e-02; ecco 6.5e-01 | 27656226 |
| 7 d | tauu_box | -2.952598e-01 | -2.952598e-01 | 0.01: 1.9e-05; 0.001: 1.9e-05; 0.0001: 1.9e-05; 1e-05: 7.4e-06 | 4/4 | - | - | exact_full 9.9e-06; ecco 4.7e-03 | 27655028 |
| 7 d | theta_A_centre | 2.335671e-04 | 2.335671e-04 | 0.1: 2.2e-05; 0.01: 1.9e-05; 0.001: 1.7e-06; 0.0001: 2.8e-06 | 4/4 | - | - | exact_full 1.0e-06; ecco 1.4e-03 | 27655028 |
| 7 d | theta_B_above | 3.511736e-05 | 3.511736e-05 | 0.1: 1.1e-04; 0.01: 7.7e-05; 0.001: 6.4e-06; 0.0001: 8.8e-07 | 4/4 | - | - | exact_full 3.7e-06; ecco 6.3e-02 | 27655028 |
| 7 d | theta_C_south | 4.431716e-06 | 4.431716e-06 | 0.1: 1.5e-04; 0.01: 2.3e-06; 0.001: 1.0e-04; 0.0001: 4.6e-04 | 4/4 | - | - | exact_full 1.8e-05; ecco 6.5e-02 | 27655028 |

Forward noise floor 7 d: spread 0.0e+00 (J = 13.75120067706965, 43.7 s per plain forward, job 27656226)

#### TL (jax.jvp) vs adjoint (chunked jax.vjp)

| window | mode | combined amp 1 | combined amp 1e-6 | named directions: max rel. difference | job |
|---|---|---|---|---|---|
| 7 d | exact_full | 3.0e-13 | 3.0e-13 | - (0 dirs) | 27655028 |
| 7 d | exact_nodyn | 3.9e-13 | 4.0e-13 | 2.1e-11 (8 dirs) | 27655028 |
| 7 d | ecco | 0.0e+00 | 1.3e-16 | 4.2e-11 (8 dirs) | 27655028 |

#### Amplification screen (terminal seed; per-chunk State-cotangent norm per field group)

Groups: dynamic = every State field except the carried constants; prognostic = theta, salt, uVel, vVel, etaN; seaice = AREA, HEFF, HSNOW, TICES, UICE, VICE; seaice_thermo = without UICE, VICE. Statistics after the seed chunk (dynamic: also without the chunk ending at iteration 1).

| window | mode | nproc | group | median /step | log-spread | worst-3 /step | passes | norm end -> start | job |
|---|---|---|---|---|---|---|---|---|---|
| 7 d | exact_full | 1 | dynamic | 0.99768 | 0.0041 | 0.99773 | yes | 6.880e+01 -> 1.289e+01 | 27655028 |
| 7 d | exact_full | 1 | theta | 1.00007 | 0.0014 | 1.00041 | yes | 1.879e-02 -> 1.866e-02 | 27655028 |
| 7 d | exact_full | 1 | prognostic | 0.99905 | 0.0041 | 1.00089 | yes | 5.834e-02 -> 5.484e-02 | 27655028 |
| 7 d | exact_full | 1 | seaice | 1.00096 | 0.0032 | 1.00272 | yes | 1.256e-02 -> 1.623e-02 | 27655028 |
| 7 d | exact_full | 1 | seaice_thermo | 1.00020 | 0.0005 | 1.00035 | yes | 1.253e-02 -> 1.248e-02 | 27655028 |
| 7 d | exact_nodyn | 1 | dynamic | 0.99768 | 0.0041 | 0.99773 | yes | 6.880e+01 -> 1.289e+01 | 27655028 |
| 7 d | exact_nodyn | 1 | theta | 1.00007 | 0.0014 | 1.00041 | yes | 1.879e-02 -> 1.866e-02 | 27655028 |
| 7 d | exact_nodyn | 1 | prognostic | 0.99905 | 0.0041 | 1.00089 | yes | 5.834e-02 -> 5.484e-02 | 27655028 |
| 7 d | exact_nodyn | 1 | seaice | 1.00632 | 0.0010 | 1.00697 | yes | 1.347e-02 -> 3.208e-02 | 27655028 |
| 7 d | exact_nodyn | 1 | seaice_thermo | 1.00021 | 0.0006 | 1.00037 | yes | 1.253e-02 -> 1.247e-02 | 27655028 |
| 7 d | ecco | 1 | dynamic | 0.99770 | 0.0040 | 0.99774 | yes | 6.896e+01 -> 1.262e+01 | 27655028 |
| 7 d | ecco | 1 | theta | 0.99997 | 0.0015 | 1.00038 | yes | 1.882e-02 -> 1.862e-02 | 27655028 |
| 7 d | ecco | 1 | prognostic | 0.99895 | 0.0042 | 1.00090 | yes | 5.899e-02 -> 5.592e-02 | 27655028 |
| 7 d | ecco | 1 | seaice | 1.00000 | 0.0000 | 1.00000 | yes | 1.284e-02 -> 1.284e-02 | 27655028 |
| 7 d | ecco | 1 | seaice_thermo | 1.00000 | 0.0000 | 1.00000 | yes | 1.284e-02 -> 1.284e-02 | 27655028 |

#### Gradient runs: repeats, cost, memory

| window | mode | nproc | repeat | J | max rel. diff to r0 (theta, kapGM, heff, atemp, tauu) | forward s | reverse s | rev/fwd | wall s | wall / plain forward | device peak GB | host GB | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 d | exact_full | 1 | 0 | 13.75120067706965 | - | 111 | 1283 | 11.52 | 1395 | 32.0 | 51.5 | 15.5 | 27655028 |
| 7 d | exact_full | 1 | 1 | 13.75120067706965 | 5.0e-11, 1.1e-12, 8.8e-13, 8.7e-13, 6.7e-14 (J bitwise) | 47 | 1114 | 23.75 | 1161 | 26.6 | 51.7 | 15.5 | 27655028 |
| 7 d | exact_full (ice weight 0) | 1 | 0 | 12.62649211901107 | - | 119 | 1269 | 10.65 | 1389 | 31.8 | 51.5 | 15.5 | 27656226 |
| 7 d | exact_nodyn | 1 | 0 | 13.75120067706965 | - | 112 | 234 | 2.10 | 346 | 7.9 | 46.6 | 15.5 | 27655028 |
| 7 d | exact_nodyn | 1 | 1 | 13.75120067706965 | 4.6e-11, 8.9e-13, 7.4e-13, 8.5e-13, 6.9e-15 (J bitwise) | 46 | 96 | 2.07 | 142 | 3.2 | 46.6 | 15.5 | 27655028 |
| 7 d | exact_nodyn (ice weight 0) | 1 | 0 | 12.62649211901107 | - | 122 | 234 | 1.92 | 356 | 8.2 | 46.4 | 15.5 | 27656226 |
| 7 d | ecco | 1 | 0 | 13.75120067706965 | - | 114 | 220 | 1.94 | 334 | 7.7 | 43.7 | 15.5 | 27655028 |
| 7 d | ecco | 1 | 1 | 13.75120067706965 | 1.3e-14, 1.7e-15, 0.0e+00, 2.9e-15, 5.7e-16 (J bitwise) | 46 | 92 | 1.98 | 138 | 3.2 | 43.9 | 15.5 | 27655028 |
| 7 d | ecco | 1 | 2 | 13.75120067706965 | 8.9e-15, 2.1e-15, 0.0e+00, 3.2e-15, 5.7e-16 (J bitwise) | 48 | 92 | 1.91 | 140 | 3.2 | 44.0 | 15.5 | 27655028 |
| 7 d | ecco | 4 | 0 | 13.75120067706965 | - | 223 | 875 | 3.92 | 1100 | 25.2 | 20.8 | 19.1 | 27655183 |
| 7 d | ecco | 4 | 1 | 13.75120067706965 | 1.2e-14, 1.9e-15, 0.0e+00, 2.9e-15, 5.7e-16 (J bitwise) | 52 | 100 | 1.92 | 152 | 3.5 | 21.0 | 19.1 | 27655183 |

Directional derivatives per mode (r0) and relative difference to the first listed mode:

| 7 d direction | exact_full | exact_nodyn | ecco |
|---|---|---|---|
| theta_A_centre | 2.335670e-04 | 2.335671e-04 (6.9e-07) | 2.338823e-04 (1.3e-03) |
| theta_B_above | 3.511726e-05 | 3.511736e-05 (2.9e-06) | 3.734211e-05 (6.3e-02) |
| theta_C_south | 4.431647e-06 | 4.431716e-06 (1.5e-05) | 4.719323e-06 (6.5e-02) |
| kapGM_scale | 4.511588e-03 | 4.425845e-03 (1.9e-02) | 1.530434e-03 (6.6e-01) |
| atemp_box | -4.449485e-05 | -4.449362e-05 (2.7e-05) | -4.241219e-05 (4.7e-02) |
| tauu_box | -2.952547e-01 | -2.952598e-01 (1.7e-05) | -2.966379e-01 (4.7e-03) |
| atemp_arctic | -3.091089e-03 | -3.085197e-03 (1.9e-03) | 1.497881e-09 (1.0e+00) |
| heff_arctic | 1.002619e+00 | 1.004801e+00 (2.2e-03) | 1.036965e+00 (3.4e-02) |

#### Forward values per mode (chunked forward, whole window, every State field + both cost parts, bytes)

| window | nproc | mode vs ref | bitwise | fields differing | J_theta | J_ice | job |
|---|---|---|---|---|---|---|---|
| 7 d | 1 | exact_full vs ecco | yes | - (126 fields) | 12.62649211901107 | 1.124708558058582 | 27655028 |
| 7 d | 1 | exact_nodyn vs ecco | yes | - (126 fields) | 12.62649211901107 | 1.124708558058582 | 27655028 |

#### Sharded (P GPUs, shard_map) vs 1 GPU

| sharded run | 1-GPU run | file | J sharded | J 1 GPU | rel. diff per control (max |a-b| / max |b|) | 1-GPU repeat floor (r1 vs r0) |
|---|---|---|---|---|---|---|
| w07/p4_ecco | w07/ecco | grad_ecco_7d_r0.npz | 13.75120067706965 | 13.75120067706965 | aqh 3.0e-15, atemp 3.4e-15, heff 0.0e+00, kapGM 1.9e-15, lwdown 3.5e-15, precip 1.5e-15, swdown 3.1e-15, tauu 5.7e-16, tauv 3.9e-16, theta 1.1e-14 | aqh 2.7e-15, atemp 2.9e-15, heff 0.0e+00, kapGM 1.7e-15, lwdown 2.4e-15, precip 1.4e-15, swdown 2.3e-15, tauu 5.7e-16, tauv 4.5e-16, theta 1.3e-14 |
| smoke_gh200_2d/p4_full | smoke_gh200_2d/full | grad_exact_full_2d_r0.npz | 13.69685064504429 | 13.69685064504429 | aqh 6.7e-13, atemp 4.5e-13, heff 2.7e-13, kapGM 8.6e-13, lwdown 1.1e-12, precip 6.1e-13, swdown 4.2e-13, tauu 1.2e-14, tauv 8.0e-15, theta 1.6e-11 | - |

Forward (final State of the chunked forward, whole window): sharded vs 1 GPU

| sharded run | 1-GPU run | file | J_theta, J_ice sharded | 1 GPU | fields differing (points) | max rel. diff |
|---|---|---|---|---|---|---|
| w07/p4_ecco | w07/ecco | fwd_final_ecco_7d.npz | (12.626492119011068, 1.1247085580585814) | (12.62649211901107, 1.1247085580585816) | none (bitwise) | 0.0e+00 |


### Window 14 days

#### FD h-sweep vs the exact_full adjoint, per named direction

Relative error |FD - AD| / |AD| at each h, AD = the exact_full adjoint (r0, J = J_theta + J_ice); plateau = number of h with rel. error <= 0.001 (bar: >= 2 of 4); J_theta / J_ice parts: the same against the ice-weight-0 adjoint (J_theta) and the difference (J_ice), where both parts were recorded and the part carries >= 0.1 % of the derivative; the other modes' adjoints relative to the best FD value.

| window | direction | AD exact_full | TL exact_full | FD rel. error at h = ... | plateau | J_theta part: errors, plateau | J_ice part: errors, plateau | other modes rel. to best FD | FD job |
|---|---|---|---|---|---|---|---|---|---|
| 14 d | atemp_arctic | -4.679761e-03 | - | 0.001: 2.2e-02; 0.0001: 6.1e-03; 1e-05: 5.9e-02; 1e-06: 3.8e-04 | 1/4 | not tested (1e-05 of AD) | 2.2e-02; 6.1e-03; 5.9e-02; 1.5e-04 (1/4) | exact_nodyn 1.2e-03; ecco 1.0e+00 | 27656226 |
| 14 d | atemp_arctic_pt | -5.562216e-07 | - | 0.1: 2.9e-03; 0.01: 2.8e-03; 0.001: 3.7e-03; 0.0001: 4.2e-03 | 0/4 | not tested (5e-06 of AD) | 2.9e-03; 3.1e-03; 3.2e-03; 3.6e-03 (0/4) | exact_nodyn 1.5e-03; ecco 1.0e+00 | 27656226 |
| 14 d | atemp_box | -1.082606e-04 | - | 0.1: 1.3e-03; 0.01: 1.7e-03; 0.001: 3.0e-04; 0.0001: 3.7e-03 | 1/4 | 1.3e-03; 1.7e-03; 3.0e-04; 3.7e-03 (1/4) | not tested (9e-05 of AD) | exact_nodyn 3.9e-04; ecco 7.2e-03 | 27656226 |
| 14 d | atemp_pt | 1.046371e-07 | - | 1: 4.4e-03; 0.1: 3.1e-03; 0.01: 4.9e-04; 0.001: 5.8e-03 | 1/4 | 4.4e-03; 3.1e-03; 6.2e-04; 5.9e-03 (1/4) | not tested (-2e-04 of AD) | exact_nodyn 6.6e-04; ecco 8.2e-02 | 27656226 |
| 14 d | heff_arctic | 9.744604e-01 | - | 0.01: 3.1e-04; 0.001: 1.4e-04; 0.0001: 2.9e-04; 1e-05: 1.2e-04 | 4/4 | not tested (3e-04 of AD) | 3.1e-04; 1.4e-04; 2.9e-04; 1.2e-04 (4/4) | exact_nodyn 3.3e-03; ecco 6.4e-02 | 27656226 |
| 14 d | heff_pt | 1.625654e-04 | - | 0.1: 5.5e-04; 0.01: 2.6e-03; 0.001: 2.7e-03; 0.0001: 2.8e-03 | 1/4 | not tested (3e-04 of AD) | 5.5e-04; 2.6e-03; 2.7e-03; 2.7e-03 (1/4) | exact_nodyn 3.0e-03; ecco 2.4e-02 | 27656226 |
| 14 d | kapGM_scale | 1.019112e-02 | - | 0.1: 2.1e-02; 0.01: 3.4e-02; 0.001: 4.4e-03; 0.0001: 6.2e-02 | 0/4 | 3.7e-05; 4.0e-05; 5.8e-05; 3.4e-04 (4/4) | 3.1e-02; 4.9e-02; 6.3e-03; 8.9e-02 (0/4) | exact_nodyn 1.4e-02; ecco 6.9e-01 | 27656226 |
| 14 d | tauu_box | -6.169303e-01 | - | 0.01: 4.6e-06; 0.001: 4.2e-06; 0.0001: 4.2e-06; 1e-05: 4.0e-06 | 4/4 | - | - | exact_nodyn 6.3e-06; ecco 9.7e-03 | 27655089 |
| 14 d | theta_A_centre | 2.352730e-04 | - | 0.1: 1.1e-06; 0.01: 2.1e-05; 0.001: 1.2e-07; 0.0001: 1.9e-05 | 4/4 | - | - | exact_nodyn 1.0e-06; ecco 4.7e-04 | 27655089 |
| 14 d | theta_B_above | 3.884180e-05 | - | 0.1: 1.1e-04; 0.01: 7.6e-05; 0.001: 3.7e-06; 0.0001: 6.9e-05 | 4/4 | - | - | exact_nodyn 7.0e-06; ecco 9.8e-02 | 27655089 |
| 14 d | theta_C_south | 4.578031e-06 | - | 0.1: 2.2e-04; 0.01: 3.4e-05; 0.001: 2.0e-04; 0.0001: 8.5e-04 | 4/4 | - | - | exact_nodyn 1.0e-05; ecco 1.4e-01 | 27655089 |

Forward noise floor 14 d: spread 0.0e+00 (J = 13.81665089194191, 87.2 s per plain forward, job 27656226)

#### FD from paired single evaluations (fdeval; LSR settings per row) vs the exact_full adjoint

| window | direction | LSR_ERROR / max iter | h | FD | FD J_ice part | AD | rel. error | J_ice part rel. error | jobs |
|---|---|---|---|---|---|---|---|---|---|
| 14 d | heff_pt | 1e-08 / 20000 | 0.001 | 1.626146e-04 | 1.625709e-04 | 1.625654e-04 | 3.0e-04 | 3.0e-04 | 27659200,27659200 |
| 14 d | heff_pt | 1e-08 / 20000 | 0.01 | 1.626067e-04 | 1.625633e-04 | 1.625654e-04 | 2.5e-04 | 2.5e-04 | 27659200,27659200 |

#### FD h-sweep vs the exact_nodyn adjoint, per named direction

Relative error |FD - AD| / |AD| at each h, AD = the exact_nodyn adjoint (r0, J = J_theta + J_ice); plateau = number of h with rel. error <= 0.001 (bar: >= 2 of 4); J_theta / J_ice parts: the same against the ice-weight-0 adjoint (J_theta) and the difference (J_ice), where both parts were recorded and the part carries >= 0.1 % of the derivative; the other modes' adjoints relative to the best FD value.

| window | direction | AD exact_nodyn | TL exact_nodyn | FD rel. error at h = ... | plateau | J_theta part: errors, plateau | J_ice part: errors, plateau | other modes rel. to best FD | FD job |
|---|---|---|---|---|---|---|---|---|---|
| 14 d | atemp_arctic | -4.672233e-03 | -4.672233e-03 | 0.001: 2.1e-02; 0.0001: 4.5e-03; 1e-05: 5.7e-02; 1e-06: 1.2e-03 | 0/4 | not tested (1e-05 of AD) | 2.1e-02; 4.5e-03; 5.7e-02; 1.5e-03 (0/4) | exact_full 3.8e-04; ecco 1.0e+00 | 27656226 |
| 14 d | atemp_arctic_pt | -5.569837e-07 | -5.569837e-07 | 0.1: 1.6e-03; 0.01: 1.5e-03; 0.001: 2.3e-03; 0.0001: 5.5e-03 | 0/4 | not tested (5e-06 of AD) | 1.6e-03; 1.8e-03; 1.8e-03; 2.2e-03 (0/4) | exact_full 2.8e-03; ecco 1.0e+00 | 27656226 |
| 14 d | atemp_box | -1.082502e-04 | -1.082502e-04 | 0.1: 1.4e-03; 0.01: 1.8e-03; 0.001: 3.9e-04; 0.0001: 3.8e-03 | 1/4 | 1.3e-03; 1.7e-03; 3.0e-04; 3.7e-03 (1/4) | not tested (-5e-06 of AD) | exact_full 3.0e-04; ecco 7.2e-03 | 27656226 |
| 14 d | atemp_pt | 1.046550e-07 | 1.046550e-07 | 1: 4.2e-03; 0.1: 3.2e-03; 0.01: 6.6e-04; 0.001: 5.7e-03 | 1/4 | 4.4e-03; 3.1e-03; 6.2e-04; 5.9e-03 (1/4) | not tested (8e-06 of AD) | exact_full 4.9e-04; ecco 8.2e-02 | 27656226 |
| 14 d | heff_arctic | 9.776011e-01 | 9.776011e-01 | 0.01: 3.5e-03; 0.001: 3.4e-03; 0.0001: 3.5e-03; 1e-05: 3.3e-03 | 0/4 | not tested (3e-04 of AD) | 3.5e-03; 3.3e-03; 3.5e-03; 3.3e-03 (0/4) | exact_full 1.2e-04; ecco 6.4e-02 | 27656226 |
| 14 d | heff_pt | 1.631405e-04 | 1.631405e-04 | 0.1: 3.0e-03; 0.01: 9.1e-04; 0.001: 8.0e-04; 0.0001: 7.7e-04 | 3/4 | not tested (3e-04 of AD) | 3.0e-03; 9.1e-04; 8.1e-04; 8.2e-04 (3/4) | exact_full 2.8e-03; ecco 2.1e-02 | 27656226 |
| 14 d | kapGM_scale | 1.000157e-02 | 1.000157e-02 | 0.1: 2.8e-03; 0.01: 1.5e-02; 0.001: 1.4e-02; 0.0001: 4.4e-02 | 0/4 | 4.3e-05; 4.6e-05; 6.3e-05; 3.4e-04 (4/4) | 4.0e-03; 2.3e-02; 2.1e-02; 6.4e-02 (0/4) | exact_full 2.2e-02; ecco 6.9e-01 | 27656226 |
| 14 d | tauu_box | -6.169367e-01 | -6.169367e-01 | 0.01: 5.8e-06; 0.001: 1.5e-05; 0.0001: 1.5e-05; 1e-05: 6.3e-06 | 4/4 | - | - | exact_full 4.6e-06; ecco 9.7e-03 | 27655089 |
| 14 d | theta_A_centre | 2.352732e-04 | 2.352732e-04 | 0.1: 2.0e-06; 0.01: 2.0e-05; 0.001: 1.0e-06; 0.0001: 2.0e-05 | 4/4 | - | - | exact_full 1.2e-07; ecco 4.7e-04 | 27655089 |
| 14 d | theta_B_above | 3.884193e-05 | 3.884193e-05 | 0.1: 1.1e-04; 0.01: 7.2e-05; 0.001: 7.0e-06; 0.0001: 7.2e-05 | 4/4 | - | - | exact_full 3.7e-06; ecco 9.8e-02 | 27655089 |
| 14 d | theta_C_south | 4.578141e-06 | 4.578141e-06 | 0.1: 2.0e-04; 0.01: 1.0e-05; 0.001: 2.2e-04; 0.0001: 8.2e-04 | 4/4 | - | - | exact_full 3.4e-05; ecco 1.4e-01 | 27655089 |

Forward noise floor 14 d: spread 0.0e+00 (J = 13.81665089194191, 87.2 s per plain forward, job 27656226)

#### TL (jax.jvp) vs adjoint (chunked jax.vjp)

| window | mode | combined amp 1 | combined amp 1e-6 | named directions: max rel. difference | job |
|---|---|---|---|---|---|
| 14 d | exact_full | 1.1e-11 | 1.1e-11 | - (0 dirs) | 27655089 |
| 14 d | exact_nodyn | 9.1e-12 | 9.1e-12 | 4.7e-10 (11 dirs) | 27655089 |
| 14 d | ecco | 5.6e-16 | 2.8e-16 | 4.5e-11 (11 dirs) | 27655089 |

#### Amplification screen (terminal seed; per-chunk State-cotangent norm per field group)

Groups: dynamic = every State field except the carried constants; prognostic = theta, salt, uVel, vVel, etaN; seaice = AREA, HEFF, HSNOW, TICES, UICE, VICE; seaice_thermo = without UICE, VICE. Statistics after the seed chunk (dynamic: also without the chunk ending at iteration 1).

| window | mode | nproc | group | median /step | log-spread | worst-3 /step | passes | norm end -> start | job |
|---|---|---|---|---|---|---|---|---|---|
| 14 d | exact_full | 1 | dynamic | 0.99763 | 0.0029 | 0.99973 | yes | 6.846e+01 -> 9.907e+00 | 27655089 |
| 14 d | exact_full | 1 | theta | 1.00004 | 0.0011 | 1.00040 | yes | 1.879e-02 -> 1.894e-02 | 27655089 |
| 14 d | exact_full | 1 | prognostic | 1.00011 | 0.0032 | 1.00132 | yes | 5.809e-02 -> 5.794e-02 | 27655089 |
| 14 d | exact_full | 1 | seaice | 1.00045 | 0.0039 | 1.00360 | yes | 1.299e-02 -> 1.890e-02 | 27655089 |
| 14 d | exact_full | 1 | seaice_thermo | 1.00068 | 0.0006 | 1.00089 | yes | 1.241e-02 -> 1.385e-02 | 27655089 |
| 14 d | exact_nodyn | 1 | dynamic | 0.99763 | 0.0029 | 0.99973 | yes | 6.846e+01 -> 9.912e+00 | 27655089 |
| 14 d | exact_nodyn | 1 | theta | 1.00004 | 0.0011 | 1.00040 | yes | 1.879e-02 -> 1.894e-02 | 27655089 |
| 14 d | exact_nodyn | 1 | prognostic | 1.00011 | 0.0032 | 1.00132 | yes | 5.809e-02 -> 5.794e-02 | 27655089 |
| 14 d | exact_nodyn | 1 | seaice | 1.00425 | 0.0021 | 1.00800 | yes | 1.356e-02 -> 6.098e-02 | 27655089 |
| 14 d | exact_nodyn | 1 | seaice_thermo | 1.00068 | 0.0007 | 1.00090 | yes | 1.241e-02 -> 1.383e-02 | 27655089 |
| 14 d | ecco | 1 | dynamic | 0.99760 | 0.0028 | 0.99972 | yes | 6.860e+01 -> 9.273e+00 | 27655089 |
| 14 d | ecco | 1 | theta | 0.99995 | 0.0011 | 1.00036 | yes | 1.882e-02 -> 1.889e-02 | 27655089 |
| 14 d | ecco | 1 | prognostic | 0.99990 | 0.0033 | 1.00139 | yes | 5.867e-02 -> 5.926e-02 | 27655089 |
| 14 d | ecco | 1 | seaice | 1.00000 | 0.0000 | 1.00000 | yes | 1.284e-02 -> 1.284e-02 | 27655089 |
| 14 d | ecco | 1 | seaice_thermo | 1.00000 | 0.0000 | 1.00000 | yes | 1.284e-02 -> 1.284e-02 | 27655089 |

#### Gradient runs: repeats, cost, memory

| window | mode | nproc | repeat | J | max rel. diff to r0 (theta, kapGM, heff, atemp, tauu) | forward s | reverse s | rev/fwd | wall s | wall / plain forward | device peak GB | host GB | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 d | exact_full | 1 | 0 | 13.81665089194191 | - | 155 | 5559 | 35.76 | 5715 | 65.6 | 51.3 | 29.1 | 27655089 |
| 14 d | exact_full (ice weight 0) | 1 | 0 | 12.61951490289781 | - | 149 | 2379 | 15.97 | 2529 | 29.0 | 51.5 | 29.1 | 27656226 |
| 14 d | exact_nodyn | 1 | 0 | 13.81665089194191 | - | 156 | 1603 | 10.27 | 1760 | 20.2 | 46.4 | 29.1 | 27655089 |
| 14 d | exact_nodyn | 1 | 1 | 13.81665089194191 | 9.1e-11, 7.6e-13, 2.6e-12, 2.6e-12, 4.7e-15 (J bitwise) | 96 | 481 | 5.02 | 577 | 6.6 | 46.6 | 29.1 | 27655089 |
| 14 d | exact_nodyn (ice weight 0) | 1 | 0 | 12.61951490289781 | - | 152 | 324 | 2.13 | 477 | 5.5 | 46.5 | 29.1 | 27656226 |
| 14 d | ecco | 1 | 0 | 13.81665089194191 | - | 580 | 3573 | 6.16 | 4154 | 47.7 | 43.8 | 29.1 | 27655089 |
| 14 d | ecco | 1 | 1 | 13.81665089194191 | 1.2e-14, 1.9e-15, 0.0e+00, 2.2e-15, 6.2e-16 (J bitwise) | 91 | 183 | 2.00 | 274 | 3.1 | 43.9 | 29.1 | 27655089 |
| 14 d | ecco | 1 | 2 | 13.81665089194191 | 1.1e-14, 2.2e-15, 0.0e+00, 2.2e-15, 6.2e-16 (J bitwise) | 94 | 183 | 1.96 | 277 | 3.2 | 44.0 | 29.1 | 27655089 |

Directional derivatives per mode (r0) and relative difference to the first listed mode:

| 14 d direction | exact_full | exact_nodyn | ecco |
|---|---|---|---|
| theta_A_centre | 2.352730e-04 | 2.352732e-04 (8.7e-07) | 2.351627e-04 (4.7e-04) |
| theta_B_above | 3.884180e-05 | 3.884193e-05 (3.3e-06) | 4.264457e-05 (9.8e-02) |
| theta_C_south | 4.578031e-06 | 4.578141e-06 (2.4e-05) | 5.234808e-06 (1.4e-01) |
| kapGM_scale | 1.019112e-02 | 1.000157e-02 (1.9e-02) | 3.138162e-03 (6.9e-01) |
| atemp_box | -1.082606e-04 | -1.082502e-04 (9.7e-05) | -1.090766e-04 (7.5e-03) |
| tauu_box | -6.169303e-01 | -6.169367e-01 (1.0e-05) | -6.229115e-01 (9.7e-03) |
| atemp_arctic | -4.679761e-03 | -4.672233e-03 (1.6e-03) | -4.515098e-10 (1.0e+00) |
| heff_arctic | 9.744604e-01 | 9.776011e-01 (3.2e-03) | 1.036965e+00 (6.4e-02) |

#### Forward values per mode (chunked forward, whole window, every State field + both cost parts, bytes)

| window | nproc | mode vs ref | bitwise | fields differing | J_theta | J_ice | job |
|---|---|---|---|---|---|---|---|
| 14 d | 1 | exact_full vs ecco | yes | - (126 fields) | 12.61951490289781 | 1.197135989044098 | 27655089 |
| 14 d | 1 | exact_nodyn vs ecco | yes | - (126 fields) | 12.61951490289781 | 1.197135989044098 | 27655089 |


### Window 28 days

#### FD h-sweep vs the exact_nodyn adjoint, per named direction

Relative error |FD - AD| / |AD| at each h, AD = the exact_nodyn adjoint (r0, J = J_theta + J_ice); plateau = number of h with rel. error <= 0.001 (bar: >= 2 of 4); J_theta / J_ice parts: the same against the ice-weight-0 adjoint (J_theta) and the difference (J_ice), where both parts were recorded and the part carries >= 0.1 % of the derivative; the other modes' adjoints relative to the best FD value.

| window | direction | AD exact_nodyn | TL exact_nodyn | FD rel. error at h = ... | plateau | J_theta part: errors, plateau | J_ice part: errors, plateau | other modes rel. to best FD | FD job |
|---|---|---|---|---|---|---|---|---|---|
| 28 d | atemp_arctic | -7.978181e-03 | -7.978181e-03 | 0.001: 6.8e-03; 0.0001: 2.4e-02; 1e-05: 3.0e-02; 1e-06: 1.5e-03 | 0/4 | not tested (2e-05 of AD) | 6.8e-03; 2.4e-02; 3.0e-02; 1.8e-03 (0/4) | ecco 1.0e+00 | 27655182 |
| 28 d | atemp_arctic_pt | -1.121796e-06 | -1.121796e-06 | 0.1: 1.4e-03; 0.01: 1.2e-03; 0.001: 3.6e-04; 0.0001: 3.7e-03 | 1/4 | not tested (1e-05 of AD) | 1.4e-03; 1.1e-03; 1.1e-03; 3.1e-03 (0/4) | ecco 1.0e+00 | 27658145 |
| 28 d | atemp_box | -2.484157e-04 | -2.484157e-04 | 0.1: 5.4e-04; 0.01: 1.9e-04; 0.001: 3.1e-03; 0.0001: 1.7e-03 | 2/4 | 6.6e-04; 7.0e-05; 3.2e-03; 1.8e-03 (2/4) | not tested (-3e-05 of AD) | ecco 1.1e-02 | 27655182 |
| 28 d | atemp_pt | 1.757180e-07 | 1.757180e-07 | 1: 2.4e-02; 0.1: 3.8e-03; 0.01: 4.5e-03; 0.001: 4.9e-02 | 0/4 | 2.4e-02; 3.6e-03; 4.5e-03; 4.9e-02 (0/4) | not tested (6e-05 of AD) | ecco 9.1e-02 | 27658145 |
| 28 d | heff_arctic | 9.400911e-01 | 9.400911e-01 | 0.01: 5.6e-03; 0.001: 5.5e-03; 0.0001: 5.8e-03; 1e-05: 5.5e-03 | 0/4 | not tested (3e-04 of AD) | 5.6e-03; 5.5e-03; 5.8e-03; 5.4e-03 (0/4) | ecco 1.1e-01 | 27655182 |
| 28 d | heff_pt | 1.603494e-04 | 1.603494e-04 | 0.1: 6.9e-03; 0.01: 5.7e-03; 0.001: 5.6e-03; 0.0001: 5.6e-03 | 0/4 | not tested (3e-04 of AD) | 6.9e-03; 5.7e-03; 5.6e-03; 5.7e-03 (0/4) | ecco 4.4e-02 | 27658145 |
| 28 d | kapGM_scale | 2.175598e-02 | 2.175598e-02 | 0.1: 4.3e-03; 0.01: 2.1e-02; 0.001: 1.8e-02; 0.0001: 1.6e-02 | 0/4 | 8.9e-05; 2.5e-05; 1.1e-04; 4.9e-05 (4/4) | 6.0e-03; 2.9e-02; 2.5e-02; 2.2e-02 (0/4) | ecco 7.2e-01 | 27655182 |
| 28 d | tauu_box | -1.269547e+00 | -1.269547e+00 | 0.01: 8.5e-05; 0.001: 1.3e-06; 0.0001: 1.7e-05; 1e-05: 1.7e-06 | 4/4 | 8.9e-05; 2.6e-06; 1.4e-05; 2.2e-06 (4/4) | not tested (4e-07 of AD) | ecco 1.7e-02 | 27655182 |
| 28 d | theta_A_centre | 2.383939e-04 | 2.383939e-04 | 0.1: 4.4e-05; 0.01: 2.2e-05; 0.001: 3.9e-06; 0.0001: 1.8e-05 | 4/4 | 4.5e-05; 2.3e-05; 3.6e-06; 2.5e-05 (4/4) | not tested (3e-07 of AD) | ecco 4.9e-03 | 27655182 |
| 28 d | theta_B_above | 4.790035e-05 | 4.790035e-05 | 0.1: 9.6e-05; 0.01: 6.1e-05; 0.001: 1.0e-05; 0.0001: 7.7e-05 | 4/4 | 9.9e-05; 6.4e-05; 7.8e-06; 1.2e-04 (4/4) | not tested (1e-06 of AD) | ecco 1.1e-01 | 27655182 |
| 28 d | theta_C_south | 5.749487e-06 | 5.749487e-06 | 0.1: 2.1e-04; 0.01: 3.4e-04; 0.001: 3.4e-04; 0.0001: 5.2e-04 | 4/4 | 2.3e-04; 3.3e-04; 3.3e-04; 5.7e-04 (4/4) | not tested (3e-06 of AD) | ecco 2.5e-01 | 27655182 |

Forward noise floor 28 d: spread 0.0e+00 (J = 13.93963468820163, 183.4 s per plain forward, job 27658145)

#### TL (jax.jvp) vs adjoint (chunked jax.vjp)

| window | mode | combined amp 1 | combined amp 1e-6 | named directions: max rel. difference | job |
|---|---|---|---|---|---|
| 28 d | exact_nodyn | 3.4e-12 | 3.4e-12 | 2.6e-10 (11 dirs) | 27655182 |
| 28 d | ecco | 4.0e-16 | 8.1e-16 | 9.6e-12 (11 dirs) | 27655182 |

#### Amplification screen (terminal seed; per-chunk State-cotangent norm per field group)

Groups: dynamic = every State field except the carried constants; prognostic = theta, salt, uVel, vVel, etaN; seaice = AREA, HEFF, HSNOW, TICES, UICE, VICE; seaice_thermo = without UICE, VICE. Statistics after the seed chunk (dynamic: also without the chunk ending at iteration 1).

| window | mode | nproc | group | median /step | log-spread | worst-3 /step | passes | norm end -> start | job |
|---|---|---|---|---|---|---|---|---|---|
| 28 d | exact_nodyn | 1 | dynamic | 0.99892 | 0.0022 | 1.00061 | yes | 7.024e+01 -> 8.775e+00 | 27655182 |
| 28 d | exact_nodyn | 1 | theta | 1.00010 | 0.0007 | 1.00041 | yes | 1.897e-02 -> 2.016e-02 | 27655182 |
| 28 d | exact_nodyn | 1 | prognostic | 1.00034 | 0.0021 | 1.00123 | yes | 6.016e-02 -> 6.960e-02 | 27655182 |
| 28 d | exact_nodyn | 1 | seaice | 1.00218 | 0.0027 | 1.00953 | yes | 1.408e-02 -> 1.230e-01 | 27655182 |
| 28 d | exact_nodyn | 1 | seaice_thermo | 1.00030 | 0.0004 | 1.00090 | yes | 1.246e-02 -> 1.508e-02 | 27655182 |
| 28 d | ecco | 1 | dynamic | 0.99890 | 0.0021 | 1.00060 | yes | 7.042e+01 -> 8.125e+00 | 27655182 |
| 28 d | ecco | 1 | theta | 1.00015 | 0.0008 | 1.00040 | yes | 1.902e-02 -> 2.011e-02 | 27655182 |
| 28 d | ecco | 1 | prognostic | 1.00036 | 0.0022 | 1.00131 | yes | 6.090e-02 -> 7.139e-02 | 27655182 |
| 28 d | ecco | 1 | seaice | 1.00000 | 0.0000 | 1.00000 | yes | 1.284e-02 -> 1.284e-02 | 27655182 |
| 28 d | ecco | 1 | seaice_thermo | 1.00000 | 0.0000 | 1.00000 | yes | 1.284e-02 -> 1.284e-02 | 27655182 |

#### Gradient runs: repeats, cost, memory

| window | mode | nproc | repeat | J | max rel. diff to r0 (theta, kapGM, heff, atemp, tauu) | forward s | reverse s | rev/fwd | wall s | wall / plain forward | device peak GB | host GB | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 28 d | exact_nodyn | 1 | 0 | 13.93963468820163 | - | 268 | 530 | 1.98 | 799 | 4.4 | 46.5 | 56.3 | 27655182 |
| 28 d | exact_nodyn | 1 | 1 | 13.93963468820163 | 1.2e-11, 3.8e-13, 2.8e-12, 1.9e-12, 4.1e-15 (J bitwise) | 191 | 400 | 2.09 | 592 | 3.2 | 46.7 | 56.3 | 27655182 |
| 28 d | exact_nodyn (ice weight 0) | 1 | 0 | 12.62591698667399 | - | 266 | 530 | 1.99 | 796 | 4.3 | 46.6 | 56.3 | 27658145 |
| 28 d | ecco | 1 | 0 | 13.93963468820163 | - | 262 | 514 | 1.96 | 778 | 4.2 | 43.7 | 56.3 | 27655182 |
| 28 d | ecco | 1 | 1 | 13.93963468820163 | 1.8e-14, 2.1e-15, 0.0e+00, 1.6e-15, 5.9e-16 (J bitwise) | 194 | 388 | 2.00 | 582 | 3.2 | 43.9 | 56.3 | 27655182 |
| 28 d | ecco | 1 | 2 | 13.93963468820163 | 1.7e-14, 2.5e-15, 0.0e+00, 1.2e-15, 5.9e-16 (J bitwise) | 194 | 388 | 2.00 | 582 | 3.2 | 43.9 | 56.3 | 27655182 |

Directional derivatives per mode (r0) and relative difference to the first listed mode:

| 28 d direction | exact_nodyn | ecco |
|---|---|---|
| theta_A_centre | 2.383939e-04 | 2.372204e-04 (4.9e-03) |
| theta_B_above | 4.790035e-05 | 5.311789e-05 (1.1e-01) |
| theta_C_south | 5.749487e-06 | 7.206581e-06 (2.5e-01) |
| kapGM_scale | 2.175598e-02 | 6.191542e-03 (7.2e-01) |
| atemp_box | -2.484157e-04 | -2.511035e-04 (1.1e-02) |
| tauu_box | -1.269547e+00 | -1.290846e+00 (1.7e-02) |
| atemp_arctic | -7.978181e-03 | -4.315871e-08 (1.0e+00) |
| heff_arctic | 9.400911e-01 | 1.036965e+00 (1.0e-01) |
| atemp_pt | 1.757180e-07 | 1.909359e-07 (8.7e-02) |
| atemp_arctic_pt | -1.121796e-06 | -6.701891e-12 (1.0e+00) |
| heff_pt | 1.603494e-04 | 1.664900e-04 (3.8e-02) |

#### Forward values per mode (chunked forward, whole window, every State field + both cost parts, bytes)

| window | nproc | mode vs ref | bitwise | fields differing | J_theta | J_ice | job |
|---|---|---|---|---|---|---|---|
| 28 d | 1 | exact_full vs ecco | yes | - (126 fields) | 12.62591698667399 | 1.313717701527636 | 27655182 |
| 28 d | 1 | exact_nodyn vs ecco | yes | - (126 fields) | 12.62591698667399 | 1.313717701527636 | 27655182 |



## M2.4 Figures

In `/work/ab0995/a270088/MIT/runs/adjoint_m2/figures/` (not in the repository), `<mode>` = ecco | exact_nodyn |
exact_full, `<w>` = 7d | 14d (all modes), 28d (ecco, exact_nodyn); J is in "K or m" (J_theta + J_ice):

| file | content |
|---|---|
| `dJdatemp_<mode>_<w>_arctic.png` | **sea-ice cost sensitivity**: dJ/d atemp (time-constant) north of 62N; ecco: 0 (no sea-ice adjoint) |
| `dJdheff_<mode>_<w>_arctic.png` | dJ/d ln HEFF0 (per cell): the Arctic ice-thickness sensitivity |
| `dJd{tauu,tauv,swdown,lwdown}_<mode>_<w>_arctic.png` | the other atmospheric controls over the Arctic |
| `dJd{atemp,aqh,tauu,tauv,swdown,lwdown,precip}_<mode>_<w>_pacific.png` | the box cost's atmospheric sensitivities, western tropical Pacific |
| `dJdtheta0_k{01,17}_<mode>_<w>_pacific.png` | dJ/dtheta0 at 5 m and 195 m (colour scale saturated at 10 %) |
| `dJd{atemp,heff,tauu}_diff_exact_full_minus_exact_nodyn_<w>_arctic.png`, `dJdtheta0_k17_diff_..._pacific.png` | the LSR (ice-dynamics) part of the sea-ice sensitivity (7, 14 d) |
| `dJd{atemp,heff,tauu}_diff_exact_nodyn_minus_ecco_28d_arctic.png`, `dJdtheta0_k17_diff_exact_nodyn_minus_ecco_28d_pacific.png` | what the ECCO semantics leaves out (28 d) |
| `amplification_traces_full.png`, `fd_sweeps_full.png` | screen traces per field group; FD error against h |

What the maps show: dJ/d atemp over the Arctic is negative everywhere (warmer air, less ice), largest (-2e-6 J/K per
cell at 14 d) in the Barents and Kara Seas, where the ice is thin and growing in January; dJ/d ln HEFF0 is positive,
largest over the thick ice north of Greenland and the Canadian Archipelago (5e-4 per cell); the LSR (dynamics) part
(exact_full - exact_nodyn) is 3-4 % of the ice-thickness sensitivity and 4-6 % of the Arctic air-temperature one in the
2-norm (7, 14 d; up to 20-57 % at single points near the ice edge and the coasts), 19-22 % of the meridional-stress one.
Over the box, dJ/d atemp shows zonal stripes of both signs at the box latitudes (1e-6 per cell at 14 d), as the TFLUX
sensitivity of Task 21.


## M2.5 Findings

1. **The adjoint of the full V4r4 model works in all three sea-ice levels over 7, 14 and 28 days** (exact_full at 7
   and 14 d; ecco and exact_nodyn at 7, 14, 28 d). The forward is the same bytes in every mode (126 State fields incl.
   sea ice and both cost parts, every window), J repeats bitwise, the TL (`jax.jvp`) equals the adjoint on the
   combined direction at amplitudes 1 and 1e-6 (linear), and the reverse sweep does not amplify: theta, prognostic and
   sea-ice-thermodynamic cotangent norms change by 0.9989-1.0007 per step (median; worst 3 chunks <= 1.0014) in every
   mode and window, as in the flux-forced model. Only UICE/VICE grow in exact_nodyn (worst-3 1.0095/step at 28 d):
   accumulation through the skipped LSR (identity), not amplification (sea-ice-only study).
2. **FD plateaus.** The ocean state (theta A, B, C), the tropical wind-stress footprint and the J_theta part of the
   global kapGM scale have plateaus against the exact adjoint in every window (1e-7 .. 5e-4; kapGM 9e-6 .. 3e-4, as in
   Task 21); the Arctic ice-thickness footprint against exact_full at 7 and 14 d (8e-5 .. 3e-4), and at 7 d also the
   single columns (atemp_pt in J_theta, atemp_arctic_pt in J_ice, heff_pt: 2e-4 .. 8e-4). What does not reach 1e-3: (a)
   footprint perturbations of the air temperature (624 box columns: 2e-3 .. 9e-3 at 7 d, 7e-5 .. 4e-3 at 14 and 28 d;
   6092 Arctic columns: 1e-4 .. 6e-2) and the J_ice part of the global kapGM scale (4e-3 .. 9e-2): the error does not
   shrink with h over 3-4 decades, the signature of many small discontinuities (bulk-formula Stanton-number switch at
   the sign of the air-sea temperature difference and of the Obukhov length; SEAICE_GROWTH exact-zero branches) whose
   number grows with h, while TL = adjoint along the same directions to <= 5e-10; the single air-temperature column in
   the box joins them at 14 and 28 d (6e-4 .. 5e-2). (b) From 14 d on, single sea-ice columns (heff_pt, atemp_arctic_pt)
   differ from the exact_full adjoint by a CONSTANT 2.6e-3 .. 3.6e-3 at every h (28 d: 1.1e-3 .. 5.7e-3 against
   exact_nodyn): not switch noise but a bias, which grows with the window as the ice dynamics' share grows. It is the
   production LSR tolerance: the exact_full derivative is that of the converged LSR system, and a forward with LSR_ERROR
   = 1e-8 (SEAICElinearIterMax 20000, same code path; 37 min per 14-day forward on a GH200 instead of 1.5 min) gives
   FD(heff_pt, 14 d) = 1.62607e-4 / 1.62615e-4 at h = 1e-2 / 1e-3 against the adjoint 1.62565e-4 (2.5e-4 / 3.0e-4),
   where the production forward gives 2.7e-3 / 2.8e-3. The J_ice part of a combined J also picks up switch noise from
   perturbations far from the Arctic (round-off reaching the ice through the global cg2d flips exact-zero branches):
   split FD by cost part.
3. **exact_nodyn vs exact_full** (the LSR derivative): 1e-7..2e-5 for the ocean-box directions, 1.6e-3 .. 3.2e-3 for the
   Arctic ice directions, 1.9e-2 for the global kapGM scale (its J_ice part); FD sides with exact_full (heff_arctic 7 d:
   full 4/4 plateau at 8e-5..3e-4, nodyn 2.2e-3 off).
4. **ecco vs exact.** In ecco mode the sea-ice cost has no adjoint path except the identity to HEFF0 (dJ/d atemp_arctic
   = 1e-9 instead of -3e-3; dJ/d ln HEFF0 = J_ice(0) exactly, 3-10 % off). For the ocean-box cost the differences are
   those of Task 21 (theta_B/C 6-25 %, growing with the window; atemp_box 1-5 %; tauu_box 0.5-1.7 %); the global
   kapGM scale is 66-72 % off because it also drives J_ice.
5. **Repeats.** ecco: J bitwise, gradients 1.8e-14 (max over all controls). Exact modes: J bitwise, 1.2e-11..9e-11
   (theta0, relative to max |g|; the maximum sits at the largest gradient value, a deep cell west of the box).
6. **TL vs adjoint beyond round-off in the exact modes (<= 5e-10)** comes from linearising two slightly different
   trajectories: the jvp program's forward J differs from the chunked forward by 3.6e-14 (7 d) - 1e-13 (14 d) relative
   (different XLA fusion), and sea-ice exact-zero branches turn that into a different Jacobian. In ecco mode (no
   sea-ice derivative) the TL J is bitwise the chunked J and TL = adjoint to 1e-15.
7. **Sharded (4 GH200, shard_map, ecco, 7 d).** Final State bitwise = 1 GPU (every prognostic and sea-ice field),
   gradient within the repeat floor (1.1e-14 vs 1.3e-14), J_theta/J_ice differ by 1-2 ulp (cost sums over padded
   tiles). 152 s per warm gradient vs 138 s on one GPU, 21 vs 44 GB device memory per GPU.
8. **Cost (GH200).** Plain forward 0.26-0.27 s/step (7 d 43.8 s, 28 d 184 s). Warm gradient (chunked forward + reverse)
   = 3.2 plain forwards in ecco and exact_nodyn at every window (28 d: 582 / 592 s); exact_full 26.5 (7 d: 1161 s: the
   implicit LSR derivative, GMRES 40 x 8 per Picard pass, ~6 s/step). TL: 1.15 plain forwards (ecco, nodyn), ~24
   (full). Device peak 44 (ecco) / 47 (nodyn) / 52 GB (full), flat from 7 to 28 d; host 1.94 GB per kept day boundary
   (15.5 / 29 / 56 GB).
9. **Sharded sea-ice derivative.** The implicit-LSR derivative did not trace inside shard_map(check_vma=True) (the
   preconditioner scan's zero first iterate was typed invariant); with the first iterate typed varying
   (`seaice_lsr._precond`, nothing changes on one device) the exact_full gradient on 4 GH200 equals 1 GPU within the
   exact-mode repeat floor (2 d: J bitwise, theta0 1.6e-11, other controls <= 1.1e-12). CPU gates unchanged (36 tests
   of test_seaice_dyn/model/lsr_pallas, gate flags); new test_lsr_sharded_p4_derivative fails without the fix.
10. **Tier 2.** `test_adjoint_regression_full.py` (one day, ecco + exact_full; J, gradient norms, directional
   derivatives recorded on a GH200, rel. 1e-9) passed on a second GH200 run (job 27659220, 11 min).

## M2.6 Reproduce

    # once, CPU, gate flags: the full-tree initial-state cache (bitwise the Fortran start-of-run state)
    XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp" JAX_PLATFORMS=cpu \
      python scripts/adjoint/multiweek_grad.py --out <dir> --append --actions cache \
        --rundir /work/ab0995/a270088/MIT/reference/runs/ref_full_serial13_1day \
        --cache /work/ab0995/a270088/MIT/runs/adjoint_m2/init_ref_full_serial13_1day
    # per window W: one 4-GH200 job, one process per GPU (use all 288 CPUs: each process binds to its Grace socket)
    sbatch --gpus=4 --cpus-per-task=288 --mem=800G scripts/adjoint/fullgrad_dolpung.sbatch --multi \
      "--out $R/ecco --days W --mode ecco --actions fwdcheck,grad,screen --repeats 3 +++ --out $R/ecco --append --days W --mode ecco --actions tl" \
      "--out $R/nodyn --days W --mode exact_nodyn --actions grad,screen --repeats 2 +++ ... --actions tl" \
      "--out $R/full --days W --mode exact_full --actions grad +++ ... --actions tl --tl-dirs combined" \
      "--out $R/fd --days W --mode exact_full --actions fd"
    # per-part FD: J_theta adjoint with --ice-weight 0; single columns: --dirs atemp_pt,atemp_arctic_pt,heff_pt
    # converged-LSR FD: --actions fdeval --dirs heff_pt --hs H --fd-sign +1|-1 --lsr-error 1e-8 --lsr-maxiter 20000
    # sharded: sbatch --gpus=4 ... fullgrad_dolpung.sbatch --out $R/p4 --days 7 --mode ecco --actions fwdcheck,grad --nproc 4
    python3 scripts/adjoint/fullgrad_summary.py <run dirs> [--ref-mode exact_nodyn] [--shard P4DIR:1GPUDIR]
    /work/ab0995/a270088/mambaforge/envs/nereus/bin/python scripts/adjoint/plot_fullgrad.py --grads mode=path,... \
      --tag 7d --diff exact_full-exact_nodyn --runs ... --figdir ...
    # tier 2: mitgcm_jax/tests/test_adjoint_regression_full.py (one day, ecco + exact_full, GH200 record)

Wall clock (GH200, compilation included): 7 d all modes + FD + TL in one 4-GPU job 1 h 42 min (the exact_full
gradients, screen and TL dominate); 28 d ecco + exact_nodyn + FD (8 directions) 1 h 52 min.

## M2.7 Open questions (for Nikolay)

1. **FD bar for sea-ice directions beyond 7 days.** Single sea-ice columns keep a constant ~3e-3 FD-vs-adjoint offset
   at 14 d against the production forward; against a converged LSR (1e-8) it drops to 2.5e-4 / 3.0e-4 (heff_pt).
   Accept the sea-ice FD bar on the converged-forward check (the rule of docs/ADJOINT_MODES.md), with the other
   single columns (atemp_arctic_pt at 14 d, both at 28 d against exact_full) still to be run that way (37 min per
   14-day forward, ~75 min at 28 d)?
2. **Air-temperature footprints are switch-limited** (bulk-formula stability switch; sea-ice exact-zero branches):
   2e-3 .. 6e-2 at every h, TL = adjoint. Accept them as recorded (as TFLUX in Task 21) with single-column directions
   carrying the FD check, or is a smoothed switch wanted for gradient work (a deviation)?
3. **Controls tauu/tauv enter before EXF_WIND** (buffer adjustments, as asked), while V4r4's xx_tauu/xx_tauv enter after
   it (EXF_GETSURFACEFLUXES), so ours also change the wind direction the sea-ice drag sees. For M4 (gentim2d controls)
   the literal place is EXF_GETSURFACEFLUXES: agreed?
4. **ECCO semantics and sea-ice costs.** With seaice="ecco" a cost on the sea-ice state has no adjoint path to the
   atmosphere or the ocean (only the identity to its own initial state): that is what TAF computes for V4r4. How V4r4
   still uses ice data (its sea-ice cost terms / SEAICE_COST_SENSI proxies acting on the ocean) is to be read in M4.1.
   If the M4 cost has a direct sea-ice term, keep ecco (as ECCO) or use exact_nodyn (3.2 forwards per gradient, like
   ecco; exact_full costs 26)?
5. **Exact-mode repeat spread** grows to 9e-11 (theta0, 14 d) vs 1e-14 in ecco mode: above Task 18's 4e-11 floor at
   one point (the largest gradient value). Worth an investigation (scatter-add order in exchange transposes under the
   sea-ice adjoint), or record as the floor for exact modes?
6. **Fake-CPU-device sharded gradient** (NaN / deadlock with XLA:CPU in-process collectives; GPU fine): drop CPU as a
   stand-in for sharded gradients of the full tree, or investigate further (XLA:CPU concurrent collectives)?
