"""FORWARD_STEP of the V4r4 configuration, composed from the ported kernels (plan Task 19: flux-forced tree; plan
M2.6b-2: the full tree with bulk-formula EXF and sea ice). Every call is at the place of the Fortran call, with its
file:line. One function for both trees; the tree is a static property of the parameters (P.exfb is not None: full).

    st1, aux = forward_step(P, g, ex, kLowC, st0, exf_in, adj=EXACT, record=False)

`P` (ModelParams, a pytree: pass it as a jit argument), `g` (Grid), `ex` (Exchanger), `st0` (State at the start of
iteration st0.it), `exf_in` = the host-side EXF inputs of this step:
  flux-forced: dict(bufs, facs, myTime) from exf_fluxforced.ExfRecordLoader.load;
  full tree:   dict(bufs, facs, myTime, zt) from exf_full.ExfFullRecordLoader.load and exf_full.zenith_time (the
               date-only zenith-angle scalars; the grid-only ones are Grid fields zs_*, model.setup).
Returns the State at the start of the next iteration, plus `aux` (per-step intermediates keyed by dump stage name, for
the first-divergence harness, gates and diagnostics). record=True (static) adds the stage records that cost extra
outputs: the EXF sub-stages X01-X08, every SEAICE_MODEL record (I00-I04, P00 and the DYNSOLVER Y*/L* records incl. the
LSOR sweep counts), D00b (MOM_VECINV tendencies), S06-S11, P05/P06, T10-T13/T20-T23; it changes no computed value.

Order (forward_step.F; full code/forward_step.F differs from flux-forced/code/forward_step.F only at :1189, the
ECCO_PHYS call of the cost):
  :418-450  RESET_NLFS_VARS, UPDATE_R_STAR(.FALSE.)                 (every step: ALLOW_AUTODIFF)
  :495      LOAD_FIELDS_DRIVER -> EXF_GETFORCING (load_fields_driver.F:139-147; CTRL_MAP_GENTIM2D :119-121 has no
            gentim2d control in V4r4; EXTERNAL_FIELDS_LOAD :163-169 does nothing, periodicExternalForcing = F)
              flux-forced: pkgs/exf_fluxforced.exf_getforcing (read fluxes)
              full:        pkgs/exf_full.exf_getforcing (EXF_GETFFIELDS, RADIATION + ZENITHANGLE, WIND, BULKFORMULAE,
                           GETSURFACEFLUXES, MAPFIELDS)
  :528      CTRL_MAP_FORCING (useCTRL=T: zero forcing controls + FFIELDS exchanges; value-identical in the full tree)
  :609      DO_OCEANIC_PHYS  flux-forced: EXTERNAL_FORCING_SURF, ...; full (c66g model/src/do_oceanic_phys.F):
                               :286-298 saltPlumeDepth = saltPlumeFlux = 0, :397-481 SEAICE_MODEL,
                               :573-576 SALT_PLUME_DO_EXCH, :606-608 EXTERNAL_FORCING_SURF; then (both trees) the tile
                               loop: rho/sigma/IVDC/mxlayer, salt plume depth, GGL90, GM/Redi (full tree:
                               CALC_OCE_MXLAYER method 1 + FIND_ALPHA because data.diagnostics requests MXLDEPTH;
                               kept literally in production, Nikolay 2026-09-23: core/mxlayer.py)
  :808      DYNAMICS
  :823      myIter <- myIter+1
  :855-1015 UPDATE_R_STAR(.TRUE.), UPDATE_CG2D, SOLVE_FOR_PRESSURE, MOMENTUM_CORRECTION_STEP, INTEGR_CONTINUITY,
            CALC_R_STAR, DO_STAGGER_FIELDS_EXCHANGES                (solve_for_pressure.step_after_dynamics)
  :1034     THERMODYNAMICS (GMREDI_RESIDUAL_FLOW, GAD_ADVECTION, TEMP/SALT_INTEGRATE)
  :1049     TRACERS_CORRECTION_STEP (nothing in V4r4)
  :1080     DO_FIELDS_BLOCKING_EXCHANGES (theta, salt; staggerTimeStep=T, storePhiHyd4Phys=F, no GGL90 horizdiff)

Full-tree State (init.state_from_pickup): the flux-forced fields plus the 28 EXF_FIELDS arrays (pkgs/exf_full.
EXF_ARRAYS; no spflx), the sea-ice state ICE_STATE (AREA, HEFF, HSNOW, TICES [T, 7, ny, nx], UICE, VICE) and DYN_CARRY
(pkgs/seaice_model: the SEAICE_DYNSOLVER arrays it writes only partly -- seaiceMassC/U/V, FORCEX0/Y0, e11, e22, e12,
DWATN, FORCEX/Y). DYN_CARRY is carried, the literal choice (the Fortran keeps them in common blocks; Nikolay
2026-09-23). Re-initialising them every step (seaice_model.dyn_carry_init) gives bitwise the same step (M2.6b-1,
tests/test_seaice_model.py::test_dyn_carry_reinit_equivalent) but was deliberately not chosen.

Backward-mode seams (plan Task 17, mitgcm_jax/adjoint/modes.py, docs/ADJOINT_MODES.md): `adj` (a static
AdjointConfig; default: exact ocean, "ecco" sea ice) selects which derivatives the reverse pass keeps. Every seam is
marked `# ADJOINT SEAM` below; none changes a forward value (stop_gradient / differentiate_at / the sea-ice
skipped_in_reverse custom_jvp are the identity forward).
"""

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp

from mitgcm_jax.adjoint.modes import EXACT, differentiate_at, stop_gradient_if, visc_params_in_ad
from mitgcm_jax.core import dynamics as dyn_mod
from mitgcm_jax.core import external_forcing as ef
from mitgcm_jax.core import free_surface as fs
from mitgcm_jax.core import grad_sigma as rs_mod
from mitgcm_jax.core import solve_for_pressure as sfp
from mitgcm_jax.core import thermodynamics as th_mod
from mitgcm_jax.core import tracers_correction as tc
from mitgcm_jax.core.cg2d import Cg2dParams
from mitgcm_jax.pkgs import ctrl as ctrl_mod
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.pkgs import exf_full as exfb_mod
from mitgcm_jax.pkgs import gad as gad_mod
from mitgcm_jax.pkgs import ggl90 as ggl_mod
from mitgcm_jax.pkgs import gmredi as gm_mod
from mitgcm_jax.pkgs import mom_common as mc
from mitgcm_jax.pkgs import mom_vecinv as mv_mod
from mitgcm_jax.pkgs import salt_plume as sp_mod
from mitgcm_jax.pkgs import seaice_init as si_mod
from mitgcm_jax.pkgs import seaice_model as sm_mod
from mitgcm_jax.state import State

# EXF_FIELDS arrays carried from step to step (their halos keep fldConst / exchanged values)
EXF_STATE = ("ustress", "vstress", "hflux", "sflux", "swflux", "apressure", "saltflx", "spflx", "runoff")
FF_FIELDS = ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "saltPlumeFlux", "pLoad", "sIceLoad")
# full tree: the EXF_FIELDS arrays carried from step to step (pkgs/exf_full.EXF_ARRAYS, 28) and the sea-ice fields
EXFB_STATE = exfb_mod.EXF_ARRAYS
SEAICE_STATE = sm_mod.SEAICE_CARRIED            # ICE_STATE + DYN_CARRY
# the grid-only factors of EXF_ZENITHANGLE (pkgs/exf_full.zenith_static), carried in the Grid as zs_<key> (model.setup)
ZS_KEYS = ("table", "iLat1", "iLat2", "wLat1", "wLat2", "SJ", "CJ", "tanLat", "xCrad")


def zenith_static_of(g):
    """The EXF_ZENITHANGLE grid factors (pkgs/exf_full.zenith_static) from the Grid's zs_* fields."""
    return {k: g.f["zs_" + k] for k in ZS_KEYS}


class ModelParams(NamedTuple):
    exf: exf_mod.ExfParams
    sf: ef.SurfForcingParams
    rs: rs_mod.RhoSigmaParams
    sp: sp_mod.SaltPlumeParams
    ggl: ggl_mod.GGL90Params
    gm: gm_mod.GMRediParams
    dyn: dyn_mod.DynamicsParams
    mv: mv_mod.MomVecinvParams
    fs: fs.FreeSurfParams
    cg: Cg2dParams
    gadT: gad_mod.GADParams
    gadS: gad_mod.GADParams
    th: th_mod.ThermoParams
    tc: object
    ctrl: object = None       # pkgs/ctrl.CtrlConfig when useCTRL=T (static pytree), else None
    exfb: object = None       # full tree: pkgs/exf_full.ExfFullParams (bulk-formula EXF); None: flux-forced (P.exf)
    seaice: object = None     # full tree: pkgs/seaice_model.SeaiceParams (useSEAICE); None: no sea ice


def do_oceanic_phys(P, g, ex, f, kLowC, adj=EXACT, record=False):
    """DO_OCEANIC_PHYS (flux-forced ff/do_oceanic_phys.F; full tree c66g model/src/do_oceanic_phys.F, cited `c66g :n`):
    returns the dict of fields it writes (FFIELDS surface forcing, rhoInSitu, sigma*, IVDConvCount, hMixLayer,
    saltPlumeDepth, GGL90*, GM/Redi tensor + Psi; full tree also the sea-ice state, DYN_CARRY and uwind/vwind).
    f: fields at DO_OCEANIC_PHYS entry. adj: backward-mode semantics (adjoint/modes.py); forward values are the same in
    every mode. Keys starting with "_" are records for forward_step (popped there)."""
    out = {}
    ff = {k: f[k] for k in FF_FIELDS}
    stages = {}
    if P.sf.useSEAICE:
        # --- full tree -----------------------------------------------------------------------------------------------
        # c66g :286-298 (ALLOW_AUTODIFF, ALLOW_SALT_PLUME): saltPlumeDepth = 0, saltPlumeFlux = 0 (full range)
        ff, spd0 = ef.oceanic_phys_pre_seaice(P.sf, ff, f["saltPlumeDepth"])
        if record:   # not a dump stage: what the zeroing wrote (visible in P00/P01 dumps)
            stages["_pre_seaice"] = dict(saltPlumeDepth=spd0, saltPlumeFlux=ff["saltPlumeFlux"])
        # c66g :397-481 IF (useSEAICE) CALL SEAICE_MODEL (pkgs/seaice_model.py): inputs as SEAICE_MODEL finds them --
        # the carried sea-ice state and DYN_CARRY, the FFIELDS after EXF_MAPFIELDS / CTRL_MAP_FORCING and the zeroing
        # above (sIceLoad of the previous step), the EXF_FIELDS after EXF_GETFORCING, the ocean surface level of the
        # start-of-step state (uVel, vVel, theta, salt at k = 1: nothing writes them before this point in the step)
        ins = {k: f[k] for k in SEAICE_STATE + sm_mod.EXF_INOUT + sm_mod.EXF_READ}
        ins.update({k: ff[k] for k in sm_mod.FF_INOUT})
        ins.update(uVel_s=f["uVel"][:, 0], vVel_s=f["vVel"][:, 0], theta_s=f["theta"][:, 0], salt_s=f["salt"][:, 0])
        sg = {k: g.f[k] for k in si_mod.ICE_FIXED}          # fixed sea-ice fields (Grid, model.setup)
        # ADJOINT SEAM seaice (adj.seaice): "ecco" (useSEAICEinAdMode = .FALSE., autodiff_inadmode_set_ad.F:37) =
        # SEAICE_MODEL skipped in the reverse sweep: identity VJP on what it overwrites, nothing to what it only reads;
        # "no_dynamics" (SEAICEuseDYNAMICSswitchInAd, :49-51); "full" = exact (pkgs/seaice_model.py docstring)
        so, rec = sm_mod.seaice_model(P.seaice, sm_mod.seaice_grid(g), sg, ex, ins, ad=adj.seaice, record=record)
        out.update({k: so[k] for k in SEAICE_STATE + sm_mod.EXF_INOUT})
        ff.update({k: so[k] for k in sm_mod.FF_INOUT})
        out["_P00_seaice_model"] = so
        if record:
            out["_seaice_rec"] = rec
        # ADJOINT SEAM salt_plume="off" (full tree, useSALT_PLUMEinAdMode=F): the saltPlumeFlux that SEAICE_GROWTH set
        # (V4r4 seaice_growth.F:2032, #ifdef ALLOW_SALT_PLUME only) is read only by IF (useSALT_PLUME) blocks, all
        # skipped in the reverse sweep: SALT_PLUME_DO_EXCH (c66g :573-576), SALT_PLUME_FORCING_SURF
        # (external_forcing_surf.F:235-239), SALT_PLUME_TENDENCY_APPLY_S (apply_forcing.F:931-936); so no derivative
        # reaches SEAICE_MODEL's saltPlumeFlux output (the entry value was zeroed at c66g :293)
        ff["saltPlumeFlux"] = stop_gradient_if(adj.salt_plume == "off", ff["saltPlumeFlux"])
        # c66g :573-576 SALT_PLUME_DO_EXCH, :606-608 EXTERNAL_FORCING_SURF
        ff, sfo = ef.oceanic_phys_post_seaice(P.sf, g, ex, ff, f["theta"], f["salt"])
    else:
        # --- flux-forced tree ----------------------------------------------------------------------------------------
        # ADJOINT SEAM salt_plume="off" (useSALT_PLUMEinAdMode=F): the reverse sweep skips SALT_PLUME_DO_EXCH
        # (:579-582), SALT_PLUME_FORCING_SURF (external_forcing_surf.F:235-239) and SALT_PLUME_TENDENCY_APPLY_S
        # (apply_forcing.F:931)
        ff["saltPlumeFlux"] = stop_gradient_if(adj.salt_plume == "off", ff["saltPlumeFlux"])
        ff, sfo, spd = ef.oceanic_phys_forcing(P.sf, g, ex, ff, f["saltPlumeDepth"], f["theta"], f["salt"])  # :288-614
    out.update(ff)
    out.update(sfo)                        # surfaceForcingU/V/T/S, PmEpR, phi0surf
    if record:
        stages["P01_external_forcing_surf"] = dict(ff, **sfo)
    # tile loop (:616-1110): zeroing, FIND_RHO_2D, GRAD_SIGMA, CALC_IVDC, CALC_OCE_MXLAYER (:640-945)
    r = rs_mod.rho_sigma_ivdc_mxlayer(P.rs, g, f["theta"], f["salt"], f["hMixLayer"], kLowC)
    out.update(rhoInSitu=r["rhoInSitu"], IVDConvCount=r["IVDConvCount"], hMixLayer=r["hMixLayer"])
    # ADJOINT SEAM gm_sigma="stable": :900-907 ZERO_ADJ_LOC(sigmaX/Y/R) (GMREDI_WITH_STABLE_ADJOINT) cuts the adjoint
    # of the density gradients for every reader (GGL90_CALC, GMREDI_CALC_TENSOR; CALC_IVDC's flag and
    # CALC_OCE_MXLAYER carry no derivative); rhoInSitu keeps its derivative
    sigX, sigY, sigR = stop_gradient_if(adj.gm_sigma in ("stable", "gm_only"), r["sigmaX"], r["sigmaY"],
                                        r["sigmaR"])
    # ADJOINT SEAM gm_sigma="gm_only" (not a TAF mode, adjoint/modes.py): only the GM/Redi slopes are cut; GGL90_CALC
    # keeps the derivative of its N^2 (sigmaR)
    sigR_ggl = r["sigmaR"] if adj.gm_sigma == "gm_only" else sigR
    # :949 SALT_PLUME_CALC_DEPTH (saltPlumeDepth was zeroed at :292; c66g :941-944)
    out["saltPlumeDepth"] = stop_gradient_if(  # ADJOINT SEAM salt_plume="off" (:948-951 skipped in the reverse)
        adj.salt_plume == "off",
        sp_mod.salt_plume_calc_depth(P.sp, P.rs.eos, g, r["rhoInSitu"][:, 0], f["theta"], f["salt"], kLowC))
    # :1063 GGL90_CALC (viscArU/V, diffKr zeroed at :661-667; the kernel returns zeros outside its loops)
    tke, vU, vV, dK = ggl_mod.ggl90_calc(P.ggl, g, f["GGL90TKE"], f["uVel"], f["vVel"], sigR_ggl,
                                         sfo["surfaceForcingU"], sfo["surfaceForcingV"], f["recip_hFacC"])
    # ADJOINT SEAM ggl90="frozen" (useGGL90inAdMode=F): no derivative through GGL90_CALC; the implicit solves keep
    # the forward kappaRk / kappaRU / kappaRV (TAF STOREs them after the GGL90 terms: docs/ADJOINT_MODES.md)
    tke, vU, vV, dK = stop_gradient_if(adj.ggl90 == "frozen", tke, vU, vV, dK)
    out.update(GGL90TKE=tke, GGL90viscArU=vU, GGL90viscArV=vV, GGL90diffKr=dK)
    # :1100 GMREDI_CALC_TENSOR, :1163 GMREDI_DO_EXCH
    t = gm_mod.gmredi_calc_tensor(P.gm, g, sigX, sigY, sigR, g.kapGM, g.kapRedi)
    if record:
        stages["P05_gmredi_tensor"] = dict(t)
    psx, psy = gm_mod.gmredi_do_exch(P.gm, ex, t["GM_PsiX"], t["GM_PsiY"])
    t = dict(t, GM_PsiX=psx, GM_PsiY=psy)
    out.update(t)
    out["_sigma"] = (r["sigmaX"], r["sigmaY"], r["sigmaR"])
    if record:
        out["_stages"] = stages
    return out


def _mom_vecinv_visc(p, g, uVel, vVel, wVel, hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW, recip_hFacS, kU, kV,
                     visc):
    """MOM_VECINV (gU, gV, guDissip, gvDissip) with the viscosities passed in (visc=None: computed inside)."""
    o = mv_mod.mom_vecinv(p, g, uVel, vVel, wVel, hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW, recip_hFacS, kU, kV,
                          visc=visc)
    return o["gU"], o["gV"], o["guDissip"], o["gvDissip"]


def mom_vecinv_adj(P, g, adj, uVel, vVel, wVel, hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW, recip_hFacS, kU, kV):
    """MOM_VECINV as DYNAMICS calls it (dynamics.F:536), with the viscFacInAd seam of `adj`."""
    args = (P.mv, g, uVel, vVel, wVel, hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW, recip_hFacS, kU, kV)
    if adj.visc_fac_in_ad is None:
        return _mom_vecinv_visc(*args, None)
    # ADJOINT SEAM visc_fac_in_ad: the TAF reverse sweep recomputes MOM_CALC_VISC with viscFacAdj = viscFacInAd
    # (autodiff_inadmode_set_ad.F:53; V4r4 mom_calc_visc.F:406,425,516,535; the viscosities are not STOREd) and
    # differentiates MOM_VECINV there
    visc = mc.mom_calc_visc(P.mv.visc, g)  # mom_vecinv.F:359-373, as MOM_VECINV computes it itself
    visc_ad = mc.mom_calc_visc(visc_params_in_ad(P.mv.visc, adj.visc_fac_in_ad), g)
    return differentiate_at(_mom_vecinv_visc, args + (visc,), args + (visc_ad,))


def forward_step(P, g, ex, kLowC, st: State, exf_in, adj=EXACT, record=False):
    """One FORWARD_STEP (module docstring). adj: static AdjointConfig (backward-mode semantics; forward values
    identical in every mode). record: static; True adds the costly stage records to aux (no value changes)."""
    full = P.exfb is not None
    if full:
        if not P.sf.useSEAICE or P.seaice is None:
            raise NotImplementedError("bulk-formula EXF without SEAICE_MODEL (useSEAICE = F): not a V4r4 configuration")
        if P.exf is not None:
            raise ValueError("ModelParams with both the flux-forced (exf) and the full-tree (exfb) EXF")
    elif P.exf is None or P.sf.useSEAICE:
        raise NotImplementedError("useSEAICE with the flux-forced EXF, or no EXF parameters: not a V4r4 configuration "
                                  "(model.setup builds P.exf for the flux-forced tree, P.exfb + P.seaice for the full)")
    f = dict(st.f)
    aux = {}
    myIter = st.it
    # forward_step.F:418-450  RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) with rStarFacNm1
    f["pStarFacK"] = fs.reset_nlfs_vars(f["rStarFacC"])
    (f["hFacC"], f["hFacW"], f["hFacS"], f["recip_hFacC"], f["recip_hFacW"], f["recip_hFacS"]) = fs.update_r_star(
        g, f["rStarFacNm1C"], f["rStarFacNm1W"], f["rStarFacNm1S"], f["recip_hFacC"], f["recip_hFacW"],
        f["recip_hFacS"])
    aux["S01_update_rstar_F"] = {k: f[k] for k in ("hFacC", "hFacW", "hFacS", "recip_hFacC")}
    # :495 LOAD_FIELDS_DRIVER -> EXF_GETFORCING (load_fields_driver.F:139-147)
    ff = {k: f[k] for k in FF_FIELDS}
    if full:
        # pkgs/exf_full.exf_getforcing (exf_getforcing.F:149-299): theta at the start of the step (SST of the bulk
        # formulae and the long-wave emission), the EXF_FIELDS arrays as the previous step left them
        exf = {k: f[k] for k in EXFB_STATE}
        exf, ff, xst = exfb_mod.exf_getforcing(P.exfb, g, ex, exf, ff, exf_in["bufs"], exf_in["facs"], f["theta"],
                                               exf_in["myTime"], exf_in["zt"], zenith_static_of(g),
                                               return_stages=True)
        # exf_getforcing.F:296 EXF_MONITOR sees the arrays after :280-288 (hflux + swflux) and before EXF_MAPFIELDS
        # (:299), which caps ustress/vstress at windstressmax: the X07 stresses
        aux["exf_monitor"] = dict(exf, ustress=xst["X07"]["ustress"], vstress=xst["X07"]["vstress"])
        if record:
            for key, stage in (("X01", "X01_exf_getffields"), ("X03", "X03_exf_radiation"), ("X04", "X04_exf_wind"),
                               ("X05", "X05_exf_bulkformulae"), ("X06", "X06_exf_hflux_sflux"),
                               ("X07", "X07_exf_getsurfacefluxes")):
                aux[stage] = xst[key]
            aux["X08_exf_mapfields"] = dict(exf, **ff)
    else:
        exf = {k: f[k] for k in EXF_STATE}
        exf, ff = exf_mod.exf_getforcing(P.exf, g, ex, exf, ff, exf_in["bufs"], exf_in["facs"], exf_in["myTime"])
    aux["S02_load_fields"] = dict(exf, **ff)
    # :524-530 CTRL_MAP_FORCING (useCTRL=T): zero xx_gentim2d adds + the FFIELDS exchanges (pkgs/ctrl.py)
    if P.ctrl is not None:
        ff = ctrl_mod.ctrl_map_forcing(P.ctrl, g, ex, ff)
    f.update(exf)
    f.update({k: ff[k] for k in FF_FIELDS if k in ff})
    if full:
        aux["S03_ctrl_map_forcing"] = dict(ff)
    # :609 DO_OCEANIC_PHYS
    op = do_oceanic_phys(P, g, ex, f, kLowC, adj, record)
    sig = op.pop("_sigma")
    aux["P02"] = {"sigmaX": sig[0], "sigmaY": sig[1], "sigmaR": sig[2]}
    if full:
        aux["P00_seaice_model"] = op.pop("_P00_seaice_model")
        if record:
            aux["seaice"] = op.pop("_seaice_rec")
    if record:
        aux.update(op.pop("_stages"))
    f.update({k: v for k, v in op.items() if k in f or k in ("PmEpR",)})
    aux["S04_oceanic_phys"] = op
    # :808 DYNAMICS
    uVel, vVel, wVel = f["uVel"], f["vVel"], f["wVel"]
    d00b = {}

    def mom_vecinv(kU, kV):
        o = mom_vecinv_adj(P, g, adj, uVel, vVel, wVel, f["hFacC"], f["hFacW"], f["hFacS"], f["recip_hFacC"],
                           f["recip_hFacW"], f["recip_hFacS"], kU, kV)
        if record:
            d00b.update(zip(("gU", "gV", "guDissip", "gvDissip"), o))
        return o

    s = {"uVel": uVel, "vVel": vVel, "guNm": jnp.stack([f["guNm_1"], f["guNm_2"]]),
         "gvNm": jnp.stack([f["gvNm_1"], f["gvNm_2"]]), "etaH": f["etaH"], "rStarFacC": f["rStarFacC"],
         "recip_hFacW": f["recip_hFacW"], "recip_hFacS": f["recip_hFacS"], "rhoInSitu": f["rhoInSitu"],
         "totPhiHyd": f["totPhiHyd"], "phi0surf": op["phi0surf"], "surfaceForcingU": op["surfaceForcingU"],
         "surfaceForcingV": op["surfaceForcingV"], "GGL90viscArU": f["GGL90viscArU"],
         "GGL90viscArV": f["GGL90viscArV"]}
    d = dyn_mod.dynamics(P.dyn, g, kLowC, s, mom_vecinv, myIter)
    f.update(gU=d["gU"], gV=d["gV"], guNm_1=d["guNm"][0], guNm_2=d["guNm"][1], gvNm_1=d["gvNm"][0],
             gvNm_2=d["gvNm"][1], totPhiHyd=d["totPhiHyd"], phiHydLow=d["phiHydLow"])
    aux["S05_dynamics"] = d
    if record:
        aux["D00b_mom_vecinv"] = d00b
    # :823 myIter = nIter0 + iLoop ; :846-1016 free surface chain
    s2 = {k: f[k] for k in ("gU", "gV", "uVel", "vVel", "wVel", "etaN", "etaH", "dEtaHdt", "EmPmR", "rStarFacC",
                            "rStarFacW", "rStarFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS", "pW", "pS",
                            "pC")}
    s2["Bo_surf"], s2["recip_Bo"] = g.Bo_surf, g.recip_Bo
    cg = P.cg
    if adj.cg2d == "passive":  # ADJOINT SEAM: cg2d.flow:7-12, operator aW2d/aS2d/aC2d passive in the adjoint
        cg = dataclasses.replace(P.cg, stop_coeff_grad=True)
    a = sfp.step_after_dynamics(P.fs, cg, g, ex, s2)
    aux["cg2d"] = a.pop("cg2d")
    aux["rstar_checks"] = a.pop("rstar_checks")
    stages = a.pop("stages")
    if record:
        aux.update(stages)
    for k, v in a.items():
        f[k] = v
    aux["S12_stagger_exchanges"] = a
    # :1034 THERMODYNAMICS (thermodynamics.F:267 GMREDI_RESIDUAL_FLOW, temp/salt_integrate.F:285 GAD_ADVECTION)
    uFld, vFld, wFld = gm_mod.gmredi_residual_flow(P.gm, g, f["uVel"], f["vVel"], f["wVel"], f["GM_PsiX"],
                                                   f["GM_PsiY"], f["recip_hFacW"], f["recip_hFacS"])
    gT_adv = gad_mod.gad_advection(P.gadT, g, uFld, vFld, wFld, f["theta"], f["hFacW"], f["hFacS"],
                                   f["recip_hFacC"])
    gS_adv = gad_mod.gad_advection(P.gadS, g, uFld, vFld, wFld, f["salt"], f["hFacW"], f["hFacS"],
                                   f["recip_hFacC"])
    fields = {k: f[k] for k in ("theta", "salt", "recip_hFacC", "hFacW", "hFacS", "rStarExpC", "surfaceForcingT",
                                "surfaceForcingS", "Qsw", "saltPlumeDepth", "saltPlumeFlux", "IVDConvCount",
                                "GGL90diffKr", "Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz")}
    fields["geothermalFlux"] = g.geothermalFlux
    fields["diffKr"] = g.diffKr
    fields["kLowC"] = kLowC
    t = th_mod.thermodynamics(P.th, g, fields, uFld, vFld, wFld, gT_adv, gS_adv)
    aux["T01_residual_flow"] = {"uFld": uFld, "vFld": vFld, "wFld": wFld}
    aux["T10_T20_adv"] = {"gT": gT_adv, "gS": gS_adv}
    aux["S13_thermodynamics"] = t
    if record:
        aux.update(T10_temp_adv={"gT_loc": gT_adv}, T11_temp_gT={"gT_loc": t["T_gExplicit"]},
                   T12_temp_step={"gT_loc": t["T_gStep"]},
                   T13_temp_impl={"gT_loc": t["T_gImpl"], "kappaRk": t["T_kappaRk"],
                                  "recip_hFac": t["recip_hFacNew"]},
                   T02_temp_integrate={"theta": t["theta"]},
                   T20_salt_adv={"gS_loc": gS_adv}, T21_salt_gS={"gS_loc": t["S_gExplicit"]},
                   T22_salt_step={"gS_loc": t["S_gStep"]},
                   T23_salt_impl={"gS_loc": t["S_gImpl"], "kappaRk": t["S_kappaRk"],
                                  "recip_hFac": t["recip_hFacNew"]},
                   T03_salt_integrate={"salt": t["salt"]})
    theta, salt = t["theta"], t["salt"]
    # :1049 TRACERS_CORRECTION_STEP
    theta, salt = tc.tracers_correction_step(P.tc, theta, salt)
    if record:
        aux["S14_tracers_correction"] = {"theta": theta, "salt": salt}
    # :1080 DO_FIELDS_BLOCKING_EXCHANGES
    theta = ex.exch_xy(theta)
    salt = ex.exch_xy(salt)
    f["theta"], f["salt"] = theta, salt
    return State(f, myIter + 1), aux
