# Possible issues in the ECCO v4r4 set-up / MITgcm c66g found while porting

Collected for discussion with the ECCO / MITgcm developers (Nikolay, 2026-09-23: "note those potential problems in
some separate file, so we can present them to ecco developers"). The JAX port follows the Fortran literally in every
case below unless a line says "deviation"; each entry gives the evidence and what the port does. Sources:
`~/MIT/MITgcm_c66g` and the V4r4 trees `ECCO-v4-Configurations/ECCOv4 Release 4/{code, flux-forced/code}`.

## Likely bugs

1. **Flux-forced tree cannot read its own pickups.** The ff tree overrides the `.meta` writer (I6 record count) but
   keeps c66g's I5 reader, so a restart from an ff pickup fails ("403 -> 40" records). Port/reference builds: approved
   DEVIATION in `reference/build.sh:50-52` (the full tree's `mdsio_read_meta.F`); I/O only, physics unchanged; restarts
   then bitwise exact (docs/PORTING_LESSONS.md, Task 8).
2. **Adjoint-sensitivity box edge.** ECCO's `prepare_run_adjsen.py`: the comment says the box is 120E-151E, but the
   condition tests `YC <= 151` (always true; presumably `XC <= 151` was meant), so the box written is 120E-180E,
   5N-16N, levels 15-20. Port: used as written, a NOTE is printed (`scripts/adjoint/multiweek_grad.py`).
3. **Adjoint-sensitivity cost scaled by the box volume twice.** `prepare_run_adjsen.py` divides `objmask` by the box
   volume, and `ecco_phys.F` (boxmean gencost) divides by the box volume `eccoVol_0` again, so J is box-mean theta
   / volume (K/m^3) instead of K. Gradients differ by the constant factor 1/volume only. Port: literal by default,
   with a printed WARNING; `--j-scaling kelvin` gives J in K.

## Surprising behaviour (probably intended, worth confirming)

4. **stableGmAdjTap ignores the data.gmredi slope parameters.** The tensor slope is clipped at a hard-coded 2e-3
   (`gmredi_slope_limit.F:593`) and the bolus slope at `5*min(|S|, 1e-4)` (`gmredi_slope_psi.F:377`); `GM_maxSlope`,
   `GM_Scrit`, `GM_Sd`, `GM_slopeSqCutoff` in data.gmredi have no effect, taper factors are 1. Port: literal.
5. **`ALLOW_AUTODIFF` changes the forward model.** Every V4r4 build compiles it, and c66g then resets the r* variables
   every step (`forward_step.F:418-450`), zeroes saltPlumeFlux before SEAICE_MODEL (`do_oceanic_phys.F:286-297`),
   takes the implicit-viscosity path, runs autodiff store/restore, and keeps the sea-ice masks static (item 6): an
   AD-enabled forward differs from a plain forward of the same configuration (docs/OVERRIDES.md).
6. **Sea-ice masks are static in AD builds.** The per-step recomputation of seaiceMaskU/V in `seaice_dynsolver.F:141-171`
   sits inside `#ifndef ALLOW_AUTODIFF_TAMC`, so in V4r4 they keep their initial values (`seaice_init_varia.F:378-391`).
7. **A diagnostics request changes the computed state.** MXLDEPTH in the full tree's data.diagnostics switches on
   CALC_OCE_MXLAYER method 1 + FIND_ALPHA (`calc_oce_mxlayer.F:74-131`), which changes hMixLayer (read by no physics in
   V4r4). Port: literal (Nikolay: keep it).
8. **The two trees' adjoints treat the salt plume differently.** ff `data.autodiff` keeps `useSALT_PLUMEinAdMode = .TRUE.`
   (default), the full tree sets `.FALSE.` (docs/ADJOINT_MODES.md, docs/OVERRIDES.md).
9. **Whole sea-ice package skipped in the V4r4 adjoint.** `useSEAICEinAdMode = .FALSE.` (full data.autodiff) skips
   thermodynamics too (`autodiff_inadmode_set_ad.F:37`), not only the dynamics
   (`SEAICEuseDYNAMICSswitchInAd` exists for that, :49-51). Port: default = this ECCO semantics; switch to
   "no_dynamics" / "full".

13. **Profile observation files are bound to the production tiling.** The pkg/profiles input files carry
    interpolation indices for 30x30 tiles (`prof_interp_i` <= 30); a profile is used only if the stored tile-corner
    coordinates match (`profiles_init_fixed.F:522-526`), so with any other tiling (e.g. 13 x 90x90) no profile is
    sampled, silently. (Verified 2026-09-23: 0 profiles on 90x90 tiles, 69 on 96 ranks for the same day.)
14. **Silent STOP on a missing RADS year.** `sshv4-mdt` reads RADS for every year of its MDT period (1993-2017) even for
    a 1-day 1992 run; a missing year ends the run with a bare `STOP` and no message (`cost_sla_read_yd.F:105-109`).
15. **SSH/OBP diagnostics depend on useECCO.** With useECCO=F the SSH, SSHIBC, SSHNOIBC, OBP, OBPGMAP diagnostics are
    silently zero (filled from pkg/ecco arrays computed only in ECCO_PHYS: `diagnostics_fill_state.F:75-79`,
    `dynamics.F:697-700`, `forward_step.F:1189`).
16. **Unpublished input.** The flux-forced boxmean cost mask `mask_BeaufortSea{C,W,S,K}` is not in the published data,
    so that cost term switches itself off (`ecco_check.F:460-474`).

## Reproducibility / sensitivity (not bugs; they limit twin comparisons)

10. **Loose solver stopping criteria with small margins.** LSR stops at LSR_ERROR = 2e-4 with the last residual at
    1.986e-4 (it 1) — a 0.7 % margin; cg2d's margin at it 2 is 0.4 %. Iteration counts, and hence results, depend on
    round-off and on the tiling: the LSR is tile-local, so 96 x 30x30 and 13 x 90x90 layouts converge to different
    iterates (differences of the size of the tolerance, not round-off).
11. **Exact-zero branches in the sea-ice thermodynamics.** E.g. `HSNOW > 0` in SEAICE_SOLVE4TEMP: ulp-level noise
    (compiler flags, FMA) creates 1e-24 m of snow where the reference has 0, flipping the branch (Qnet -26 vs -40 W/m^2
    at a point). Runs with sea ice are therefore reproducible only with identical arithmetic; across compilers or
    platforms they agree only statistically.
12. **libm dependence.** gfortran calls glibc's `exp` (not correctly rounded) and, vectorised, libmvec's exp
    (`seaice_calc_ice_strength`); results depend on the libm version. The port reproduces both bit for bit
    (`mitgcm_jax/ops/libm.py`).
