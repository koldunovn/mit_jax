# Backward-mode semantics: what the TAF adjoint of ECCO v4r4 computes, and the JAX equivalent

Plan Task 17, 2026-09-23. Scope: the V4r4 **flux-forced** build (ff `code/` + c66g) with its `namelist/`, and where it
differs, the full V4r4 tree. Paths: c66g = `~/MIT/MITgcm_c66g`; ff = `ECCOv4 Release 4/flux-forced/code/`; full =
`ECCOv4 Release 4/code/`. Line numbers refer to the override file when one exists. The JAX side is
`mitgcm_jax/adjoint/modes.py` (`AdjointConfig`) and the seams marked `# ADJOINT SEAM` in
`mitgcm_jax/core/forward_step.py`.

**Caveat.** No TAF-generated code exists on this machine (the `*_ad.F` files in the build trees are the hand-written
ones from `pkg/autodiff`). Everything below is read from the source, the `CADJ` directives and the `.flow` files. Section 5
lists the one reading that a TAF build could confirm.

## 1. Summary

| package / switch | setting in the ff build | what the TAF adjoint computes | JAX (`AdjointConfig` field) | ECCO value |
|---|---|---|---|---|
| GGL90 | `useGGL90inAdMode = .FALSE.` (ff `namelist/data.autodiff:11`) | GGL90 is skipped in the reverse sweep, recomputations included. The total vertical diffusivity and viscosity reach the adjoint from the tape, stored **after** the GGL90 terms were added. The implicit solves are therefore linearised with the forward (GGL90-inclusive) coefficients, and no derivative flows through the TKE or the coefficients' dependence on the state. | `ggl90="frozen"`: `stop_gradient` on the four GGL90_CALC outputs (TKE, viscArU, viscArV, diffKr) | `frozen` |
| GM/Redi slopes | `GMREDI_WITH_STABLE_ADJOINT` (ff `GMREDI_OPTIONS.h:21`) | `ZERO_ADJ_LOC` zeroes the adjoint of sigmaX/Y/R. No reader of sigma passes a derivative back (GMREDI tensor, GGL90 N², CALC_IVDC). rhoInSitu keeps its derivative. | `gm_sigma="stable"`: `stop_gradient` on sigmaX/Y/R after GRAD_SIGMA | `stable` |
| GM/Redi package | `useGMRediInAdMode` not set (default `.TRUE.`) | GM/Redi differentiated, apart from the sigma cut above | none (unported alternative: `NotImplementedError`) | kept |
| salt plume | ff: `useSALT_PLUMEinAdMode = .TRUE.` (`data.autodiff:12`); full: `.FALSE.` | ff: exact. Full tree: every `IF (useSALT_PLUME)` block is skipped in the reverse sweep, so no derivative flows through saltPlumeFlux or saltPlumeDepth. | `salt_plume="exact"` or `"off"` (`stop_gradient` on saltPlumeFlux at DO_OCEANIC_PHYS entry -- full tree: on SEAICE_MODEL's output, section 3.4 -- and on saltPlumeDepth) | `exact` (ff), `off` (full) |
| sea ice | ff: `useSEAICE` unset (F). Full: `useSEAICEinAdMode = .FALSE.`, `SEAICEapproxLevInAd = 0`, so no SEAICE_FAKE. | ff: nothing to do. Full: SEAICE_MODEL skipped in the reverse sweep: identity on what it overwrites, nothing to what it only reads. | `seaice="ecco"` (default everywhere) \| `"no_dynamics"` \| `"full"` (M2.6b; section 3.4) | `ecco` (full) |
| KPP | not used | — | refused if used | — |
| viscFacInAd | not set, so 1.0 (`autodiff_readparms.F:73`; STDOUT prints `1.0E+00`) | Adjoint of MOM_VECINV with the viscosities recomputed at `viscFacAdj = viscFacInAd` (the 3-D file fields are scaled before clipping). With 1.0 this equals the exact adjoint. | `visc_fac_in_ad`: `differentiate_at` around MOM_VECINV | `1.0` |
| cg2d | `pkg/autodiff/cg2d.flow:7-12` | Hand-written adjoint: CG2D applied to the adjoint right-hand side. The operator aW2d/aS2d/aC2d is passive. | `cg2d="passive"`: `Cg2dParams.stop_coeff_grad` | `passive` |
| inAdExact | not set (default `.TRUE.`) | `inAdMode` stays `.FALSE.` in the reverse sweep. It would only change DST3 flux-limited advection, which V4r4 does not use. | refused if `.FALSE.` | — |

`AdjointConfig.ecco(RunNamelists(rundir))` builds the last column from the run's `data.autodiff` and `data.pkg`. For the
full V4r4 tree (M2.6b-2) it returns `AdjointConfig(ggl90='frozen', gm_sigma='stable', salt_plume='off', cg2d='passive',
visc_fac_in_ad=1.0, seaice='ecco')` (the full oracle's STDOUT.0000 prints useSEAICEinAdMode = useGGL90inAdMode =
useSALT_PLUMEinAdMode = F; code/GMREDI_OPTIONS.h:21 defines GMREDI_WITH_STABLE_ADJOINT as in the ff tree). For the
FORCED oracle it returns `AdjointConfig(ggl90='frozen', gm_sigma='stable', salt_plume='exact', cg2d='passive',
visc_fac_in_ad=1.0)` (seaice 'ecco', unused without sea ice), and its STDOUT.0000 prints the matching values (`useGGL90inAdMode = F`,
`useSALT_PLUMEinAdMode = T`, `useGMRediInAdMode = T`, `useKPPinAdMode = F`, `useSEAICEinAdMode = F`, `inAdExact = T`,
`viscFacInAd = 1.0E+00`). `AdjointConfig()` (the default) is the exact mode and inserts no ocean seam (its sea-ice level is `"ecco"`, which acts only in the full tree). Its traced step is
character-for-character the jaxpr of the pre-Task-17 `forward_step` (checked against `git show HEAD:` of that file).

## 2. The mechanism: which values the reverse sweep uses

1. **Switches.** `autodiff_readparms.F:66-73` sets the defaults (all `*InAdMode = .TRUE.`, `inAdExact = .TRUE.`,
   `viscFacInAd = 1`). `:109-120` saves the forward values (`use*InFwdMode = use*`) and ANDs the adjoint values with
   them. `:98-104`: with `inAdExact`, both `inAdTrue` and `inAdFalse` are `.FALSE.`.
2. **Forward.** ff `forward_step.F:392` calls `AUTODIFF_INADMODE_UNSET` first in every step, and `:1228` calls
   `AUTODIFF_INADMODE_SET` last. Both are empty routines (`autodiff_inadmode_set.F:6-22`), so the forward never
   changes.
3. **Reverse.** TAF uses the hand-written adjoints (`autodiff_inadmode.flow:7-30`). `ADAUTODIFF_INADMODE_SET`
   (`autodiff_inadmode_set_ad.F:33-53`: `useKPP/useGMRedi/useSEAICE/useGGL90/useSALT_PLUME = *InAdMode`,
   `viscFacAdj = viscFacInAd`) is the adjoint of the step's last statement, so it runs **first** in the reverse sweep
   of each step. `ADAUTODIFF_INADMODE_UNSET` (`autodiff_inadmode_unset_ad.F:33-53`) runs last and restores the forward
   values, including `viscFacAdj = 1`.
4. **Tapes.** `the_main_loop.F:416-418` declares `comlev1`, `comlev1_bibj` and `comlev1_bibj_k` as memory tapes sized
   by `nchklev_1` (ff `tamc.h:64`: 25), with keys that include `ikey_dynamics`. The innermost checkpoint loop is
   therefore split: a taping forward sweep over its `nchklev_1` steps executes every `CADJ STORE`, then the reverse
   steps run.
   - The taping sweep runs with the **forward** switch values. Before the first block nothing has flipped them, and
     after that every reverse step ends with `ADAUTODIFF_INADMODE_UNSET`.
   - Inside a reverse step, every adjoint routine recomputes what was not stored, and it does so with the
     **adjoint** switch values. A skipped `IF (useGGL90)` block is skipped both in the recomputation and in the
     adjoint.
5. **Rule.** A value downstream of a package that is off in the adjoint enters the linearisation with its forward
   value if it was STOREd after the package contributed to it. Otherwise it enters with the value recomputed without
   the package. The trajectory itself (checkpointed states) is always the true forward.
6. **TLM.** `g_autodiff_inadmode_set.F` only sets `inAdmode = .FALSE.`, so every package stays on in the
   tangent-linear model. `ZERO_ADJ_LOC` has no `g_` version in c66g.

## 3. Per package

### 3.1 GGL90 (`useGGL90inAdMode = .FALSE.`)
Forward data flow in one step:
- ff `do_oceanic_phys.F:669-673` zeroes GGL90viscArU/V and GGL90diffKr (ALLOW_AUTODIFF).
- `:1054-1055` STOREs GGL90TKE; `:1058-1066` runs `IF (useGGL90) CALL GGL90_CALC(bi, bj, sigmaR, ...)`.
- `calc_3d_diffusivity.F:230-235` adds `GGL90_CALC_DIFF` to kappaRk; the caller is `temp_integrate.F:245` (and
  `salt_integrate.F:237`).
- `calc_viscosity.F:97-104` adds `GGL90_CALC_VISC` to kappaRU/kappaRV; the caller is `dynamics.F:383`.

What is stored after the GGL90 terms:
- `temp_integrate.F:488` and `:505` STORE kappaRk right before `GAD_IMPLICIT_R` / `IMPLDIFF`; so do
  `salt_integrate.F:480/497`.
- `dynamics.F:399-402` STOREs kappaRU and kappaRV right after `CALC_VISCOSITY`. Their readers come later:
  `MOM_VECINV`'s bottom drag (`:536`) and `IMPLDIFF` (`:596-604`).
- The step-start STOREs of `checkpoint_lev1_directives.h:109` → `ggl90_ad_check_lev1_dir.h:4-7` hold the previous
  step's GGL90 fields. They are overwritten by the zeroing at `:669-673`, so they play no role.

In the reverse sweep, `GGL90_CALC`, `GGL90_CALC_DIFF` and `GGL90_CALC_VISC` are skipped. The recomputed kappaRk and
kappaRU/RV (background plus IVDC plus GM only) are replaced by the taped, GGL90-inclusive values at the STORE points,
before any reader. Nothing else reads the GGL90 outputs within a step. **TAF therefore computes the adjoint with frozen
coefficients**: the tracer and momentum implicit operators use the forward Kv/Av (the matrices and the solutions of
`GAD_IMPLICIT_R` / `IMPLDIFF` are those of the forward). The derivative of Kv/Av with respect to the state (TKE, shear,
N², surface stress, hFac) is zero, and dJ/dTKE is zero.

The variants the plan asked about, compared:
- *stop_gradient on the GGL90 outputs* and *frozen forward coefficients* are the same thing in JAX. A VJP is always
  evaluated at the forward values, so cutting the outputs keeps exactly the forward Kv/Av in the operators. This is
  TAF's semantics, implemented as `ggl90="frozen"`.
- *Recomputed without GGL90* would linearise the implicit diffusion and viscosity with background-only coefficients.
  In the mixed layer, where Kv is far above background, this differs strongly from the variant above. It would be
  TAF's result only if kappaRk/kappaRU were not stored, and they are. **Not implemented** (question 2 in section 5).

Effects measured on the FORCED oracle, step 1. J = Σ θ(end of step) over tile 3, levels 1-10, 40×40 points.
- Exact mode: dJ/dTKE is non-zero at 305 616 points. In the frozen mode it is exactly 0.
- `ggl90="frozen"` alone changes dJ/dθ by up to 99.8 % of max|dJ/dθ_exact|. The exact gradient has a spike of 97.6
  K/K at one point, the level just below the box (tile 3, k=11). The spike goes through GGL90's Kv(N²). It is 0.198
  with GGL90 frozen and 0.197 with the sigma cut, because the sigma cut also removes GGL90's N² derivative.
- dJ/du changes by 100 %: the shear path of Kv.
- At the DO_OCEANIC_PHYS level, the derivatives of the GGL90 outputs are exactly 0. In the exact mode they are
  non-zero at 2.9e6 (TKE), 2.9e6 (uVel) and 2.8e6 (θ) points.

### 3.2 GM/Redi stable adjoint (`GMREDI_WITH_STABLE_ADJOINT`)
ff `do_oceanic_phys.F:900-907` calls `ZERO_ADJ_LOC(Nr, sigmaX/Y/R)` after every `GRAD_SIGMA`. The forward routine is
empty (`zero_adj.F:45`). Its adjoint `ADZERO_ADJ_LOC` (`adzero_adj.F:51-84`) zeroes the whole 3-D adjoint arrays, and
the flow file `zero_adj.flow:16-21` declares only the array argument as active.

In the reverse k loop the first zeroing happens before any `ADGRAD_SIGMA`, so every contribution collected from
downstream readers is discarded. Those readers are:
- `GMREDI_CALC_TENSOR` (`:1100`) and, through it, the GM bolus streamfunction;
- `GGL90_CALC`, which reads sigmaR (`ggl90_calc.F:218-219`);
- `CALC_IVDC` (a step function anyway);
- `CALC_OCE_MXLAYER` (computes nothing in V4r4).

`SALT_PLUME_CALC_DEPTH` uses the density criterion (CriterionType = 1, `salt_plume_readparms.F:63` default, `salt_plume_calc_depth.F:90-138`), so it reads
rhoInSitu and FIND_RHO, not sigma. rhoInSitu and the pressure path keep their derivatives. The derivatives with respect
to kapGM/kapRedi (the controls) are kept.

JAX: `gm_sigma="stable"` puts `stop_gradient` on the three arrays that `rho_sigma_ivdc_mxlayer` returns, before GGL90
and GM read them. IVDC is computed inside that function from the uncut sigmaR, but its `where` carries no derivative.
Measured:
- At the DO_OCEANIC_PHYS level, d(GM tensor, PsiX/Y)/dθ,S is exactly 0 (exact mode: non-zero at 2.9e6 points each).
- d(rhoInSitu)/dθ,S is bitwise unchanged.
- Step level: dJ/dθ changes by up to 99.8 % of max|dJ/dθ| (max 97.6 → 1.29), and dJ/dS by 100 % (max 378 → 0.014).

### 3.3 Salt plume
Everything the package does runs under `IF (useSALT_PLUME)`:
- `SALT_PLUME_DO_EXCH` (ff `do_oceanic_phys.F:579-582`);
- `SALT_PLUME_FORCING_SURF`, which subtracts the flux from surfaceForcingS (`external_forcing_surf.F:235-239`);
- `SALT_PLUME_CALC_DEPTH` (`:948-951`);
- `SALT_PLUME_TENDENCY_APPLY_S` (`apply_forcing.F:931-936`); the `_T` version is empty without `SALT_PLUME_VOLUME`.

In the ff tree `useSALT_PLUMEinAdMode = .TRUE.`, so the package is differentiated (exact, and the ECCO mode inserts no
seam). In the full tree (`.FALSE.`) all four are skipped in the reverse sweep. The adjoint is then that of a model
whose surface salt flux does not subtract the plume flux and which has no plume redistribution: d/d(saltPlumeFlux) = 0
by both routes, and the depth's dependence on ρ(θ,S) is cut. Stopping only the tendency would still leave the
−saltPlumeFlux path through surfaceForcingS alive; the JAX seam cuts both.

The forward values are unchanged. The `saltPlumeDepth` STOREd at ff `do_oceanic_phys.F:757-760` is the zeroed value
from before `CALC_DEPTH`.

JAX: `salt_plume="off"`. Measured: at the DO_OCEANIC_PHYS level, d(surfaceForcingS)/d(saltPlumeFlux) is exactly 0 (exact
mode: non-zero at 106 740 surface points), and d(surfaceForcingS)/dS is bitwise unchanged. At the step level with a θ
cost there is no effect, because within one step θ does not depend on the plume.

### 3.4 Sea ice (full V4r4 only)
`useSEAICEinAdMode = .FALSE.` switches the package off in the reverse sweep. `SEAICEapproxLevInAd` becomes
`MIN(0, 0) = 0` (`autodiff_readparms.F:124-125`), and `ADAUTODIFF_INADMODE_SET:51` sets `SEAICEadjMODE = 0`. So
`SEAICE_FAKE`, which needs `SEAICEadjMODE = -1` (ff `do_oceanic_phys.F:383`), does not run either: the full-V4r4
adjoint has no sea-ice sensitivity at all. The ff run has `useSEAICE = F`.

JAX (M2.6b): `AdjointConfig.seaice`, default `"ecco"` in every configuration (Nikolay, 2026-09-23), including the
otherwise exact `AdjointConfig()`: SEAICE_MODEL wrapped in `ops/ad_skip.skipped_in_reverse` (a linear custom_jvp:
identity VJP on the variables it overwrites, zero on those it only reads; forward byte-identical). `"no_dynamics"`
(SEAICEuseDYNAMICSswitchInAd semantics) and `"full"` (exact, LSR implicit derivative) are explicit choices
(pkgs/seaice_model.py). The seam sits at the SEAICE_MODEL call in `forward_step.do_oceanic_phys`.
Salt plume in the full tree: saltPlumeFlux is zeroed at c66g do_oceanic_phys.F:293 and set by SEAICE_GROWTH (V4r4
seaice_growth.F:2032, under `#ifdef ALLOW_SALT_PLUME` only), and every reader is an `IF (useSALT_PLUME)` block, so the
`salt_plume="off"` stop_gradient sits on SEAICE_MODEL's saltPlumeFlux output (before SALT_PLUME_DO_EXCH) rather than
on the entry value. With `seaice="ecco"` that flux seam is redundant (measured: d surfaceForcingS / d salt bitwise the
same with and without it); only the saltPlumeDepth seam changes derivatives there (tests/test_adjoint_modes_full.py).

### 3.5 viscFacInAd
The V4r4 `mom_calc_visc.F:406,425,516,535` add `viscFacAdj*visc{Ah,A4}{D,Z}fld` to the linear viscosity before the
Min/Max clipping and the Gibraltar ×10 (c66g scales only two fields). In the reverse sweep `viscFacAdj = viscFacInAd`
(`autodiff_inadmode_set_ad.F:53`). `mom_vecinv.F` STOREs no viscosity (its STOREs are uFld, vFld, KE, vort3, hFacZ,
...), so `ADMOM_VECINV` recomputes `MOM_CALC_VISC` with the adjoint factor. The trajectory values that MOM_VECINV's
adjoint needs, and everything downstream (gU, gV are STOREd at ff `forward_step.F:817`), come from the forward. The
TAF adjoint is therefore the VJP of MOM_VECINV evaluated at the recomputed viscosities.

This is literally "the viscosity recomputed with viscFacAdj = viscFacInAd", not "visc × factor": only the 3-D file
fields are scaled, before clipping. In V4r4 these are the Fenty biharmonic `viscA4D/Zfile`; `viscAhD/Zfld` are zero.

JAX: `differentiate_at(fn, args, alt)` in `adjoint/modes.py` is a `custom_jvp`. It returns `fn(*args)`, and its
tangent (and so, by transposition, its cotangent) is `fn`'s tangent at `alt`. `forward_step.mom_vecinv_adj` applies it
to the whole MOM_VECINV with `alt` = the viscosities of `mom_calc_visc(viscFacAdj=viscFacInAd)`. It is a `custom_jvp`
rather than a `custom_vjp` so that forward mode still works.

V4r4 does not set viscFacInAd, so the ECCO value 1.0 is a no-op. The seam is present, and its gradient is bitwise the
exact one (tested). Measured with factor 2:
- the MOM_VECINV-level d/d(u,v) changes by up to 52 %;
- it equals plain autodiff of MOM_VECINV at the scaled viscosities to 0 (bitwise);
- the step-level dJ/du changes by 5.0e-3 and dJ/dv by 2.9e-3 (relative to the max);
- dJ/dθ is unchanged bitwise.

### 3.6 cg2d
`pkg/autodiff/cg2d.flow:7-12` declares `ADNAME = cg2d`, `INPUT = 1,2,6,7,8`, `OUTPUT = 2..7` and `ACTIVE = 1,2`, i.e.
only cg2d_b and cg2d_x are active. aW2d, aS2d and aC2d are common-block arrays that are not arguments, so TAF sees no
dependence of the solution on the operator. The r*-dependent operator (from hFacW/S) is **passive**, and the adjoint is
CG2D itself applied to the adjoint right-hand side (the operator is symmetric).

JAX: `cg2d="passive"` sets `Cg2dParams.stop_coeff_grad` (`core/cg2d.py:283`, which predates Task 17 and has its own
kernel effect test in `test_cg2d.py`). Measured at the step level: only dJ/drStarFacW and dJ/drStarFacS change (up to
79 % and 50 % relative); every other field is bitwise unchanged.

Adjoint-solve accuracy is not part of the mode. TAF's adjoint solve runs to `cg2dTargetResidual` (1e-7 normalised) from
whatever first guess the ADNAME call passes; that detail cannot be checked without generated code. The JAX transpose
solve runs to 1e-13 from zero (`Cg2dParams.adj_tolerance`).

### 3.7 Not switched
- `inAdExact` is `.TRUE.`: `inAdMode` is read only by DST3 flux-limited advection (`gad_calc_rhs.F:245,385,516,559`),
  which V4r4 does not use.
- `useKPPinAdMode` and `useSmoothCorrel2DinAdMode` (`ECCO_CTRL_DEPRECATED`) are irrelevant here.
- `dumpAdVarExch`, `mon_AdVarExch` and `dumpAdByRec` control output only.
- In the full tree, `SEAICEuseFREEDRIFTswitchInAd` and `SEAICEuseDYNAMICSswitchInAd` are `.FALSE.` by default.
- The `ALLOW_AUTODIFF` forward branches (`docs/OVERRIDES.md`) are forward code, already ported.

## 4. JAX implementation

```python
from mitgcm_jax.adjoint.modes import AdjointConfig
adj = AdjointConfig.ecco(RunNamelists(rundir))        # or AdjointConfig() (exact ocean; sea ice "ecco")
st1, aux = forward_step(P, g, ex, kLowC, st0, exf_in, adj=adj)   # adj is static: close over it
```

**Default gradient semantics = ECCO (Nikolay, 2026-09-23); exact available.** The gradient drivers default to the
run's `AdjointConfig.ecco(nml)` (flux-forced or full tree; sea ice `"ecco"`): `adjoint/checkpoint.make_step(nml=...)`
(no config and no namelists is an error, so the choice is never silent) and `scripts/adjoint/multiweek_grad.py`
(`--mode ecco` by default). The exact adjoint is `make_step(AdjointConfig())` / `--mode exact`; any other
`AdjointConfig` can be passed. `AdjointConfig()` itself stays the explicit exact configuration (the tests of the exact
gradient, dot tests and FD checks use it: test_checkpoint.py, test_budgets.py pass it explicitly). The sea-ice level is
`"ecco"` in every configuration unless set (`seaice="no_dynamics"` or `"full"` are explicit).

Seams, all in `core/forward_step.py` (none changes a forward value):
- `do_oceanic_phys`:
  - salt plume flux (`:81`);
  - sigma (`:91`);
  - salt-plume depth (`:93`);
  - GGL90 outputs (`:101`).
- `mom_vecinv_adj` (`:120-130`): viscFacInAd.
- `forward_step` (`:181`): cg2d.

Tests:
- `tests/test_adjoint_modes.py` (tier 1, 3 tests, 67 s standalone, most of it loading the dump index, which a suite
  run shares with `test_step_fluxforced`):
  - the ECCO config from the namelists;
  - a seam census: in the traced step, each switch adds exactly its own `stop_gradient` / `custom_jvp_call` equations
    (sigma 3, GGL90 4, cg2d 3, salt 2, viscFacInAd 1 custom_jvp), and the exact mode adds none;
  - the step with every switch on (viscFacInAd = 2) is **bitwise equal to the Fortran** at the dumped stages of step 1
    and at the start of iteration 2 (FORCED oracle, cg2d 164 iterations).
- `tests/test_adjoint_modes_grad.py` (tier1x, 3 tests, 123 s):
  - DO_OCEANIC_PHYS effect tests: structural zeros, the rhoInSitu path bitwise unchanged, forward outputs bitwise;
  - viscFacInAd: factor 1 bitwise equal to exact, factor 2 equal to autodiff at the scaled viscosity;
  - one full-step gradient in the exact and the ECCO mode: finite everywhere for every State field, dJ/dTKE 0 in the
    ECCO mode, dJ/dθ differs.

## 5. Open questions (Nikolay)
1. **Confirm the GGL90 reading on a TAF build.** It rests on the STOREs of kappaRk (`temp_integrate.F:488/505`,
   `salt_integrate.F:480/497`) and kappaRU/RV (`dynamics.F:399-402`) being taped in the forward-flag sweep and restored
   before `ADGAD_IMPLICIT_R` / `ADIMPLDIFF`. In `adtemp_integrate`, kappaRk should come from `comlev1_bibj`, not from
   a recomputed `CALC_3D_DIFFUSIVITY`.
2. **Variant "recomputed without GGL90"** (background-only Kv/Av in the linearised implicit operators). It is not
   TAF's, and it is not implemented. Is it wanted as a separate experiment?
3. **Tangent-linear mode.** TAF's TLM keeps every package. A non-exact `AdjointConfig` also cuts tangents, because
   `stop_gradient` is a zero tangent. Use `AdjointConfig()` for TL / dot-test work unless the ECCO approximation is
   wanted there.
4. **cg2d adjoint accuracy.** Keep the tight 1e-13 transpose solve, or add a TAF-precision option (1e-7) for
   side-by-side comparison with TAF gradients (M3)?
5. **Salt plume differs between the trees.** The ff ECCO mode keeps the plume and the full ECCO mode drops it.
   `AdjointConfig.ecco` follows the run's `data.autodiff`.

## Sea-ice derivative at the production LSR tolerance (Nikolay, 2026-09-23: keep current)
The "full" / "no_dynamics" sea-ice derivatives are the implicit derivative of the converged LSR system (M2.4). At the
production LSR_ERROR = 2e-4 the Fortran-like forward stops early, so FD of the production model differs from this
derivative by 1-12 % along ice-dynamics directions (sea-ice window study, docs/ADJOINT_RESULTS.md). Decision: keep the
implicit derivative (no differentiation through iterations, no tighter production tolerance); FD checks of dynamics
directions use a converged forward (same code path, tighter LSR_ERROR), stated where used.
