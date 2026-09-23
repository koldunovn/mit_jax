"""FORWARD_STEP of the V4r4 flux-forced configuration (flux-forced code/forward_step.F), composed from the ported
kernels (plan Task 19). Every call is at the place of the Fortran call, with its file:line.

    st1 = forward_step(P, g, ex, st0, exf_in)

`P` (ModelParams, a pytree: pass it as a jit argument), `g` (Grid), `ex` (Exchanger), `st0` (State at the start of
iteration st0.it), `exf_in` = dict(bufs, facs, myTime) from the host-side EXF record loader for this step
(exf_fluxforced.ExfRecordLoader.load). Returns the State at the start of the next iteration, plus `aux` (per-step
intermediates keyed by dump stage, for the first-divergence harness and diagnostics).

Order (forward_step.F, flux-forced tree):
  :418-450  RESET_NLFS_VARS, UPDATE_R_STAR(.FALSE.)                 (every step: ALLOW_AUTODIFF)
  :495      LOAD_FIELDS_DRIVER -> EXF_GETFORCING
  :528      CTRL_MAP_FORCING (useCTRL=T: zero forcing controls + FFIELDS exchanges)
  :609      DO_OCEANIC_PHYS  (EXTERNAL_FORCING_SURF, rho/sigma/IVDC/mxlayer, salt plume depth, GGL90, GM/Redi)
  :808      DYNAMICS
  :823      myIter <- myIter+1
  :855-1015 UPDATE_R_STAR(.TRUE.), UPDATE_CG2D, SOLVE_FOR_PRESSURE, MOMENTUM_CORRECTION_STEP, INTEGR_CONTINUITY,
            CALC_R_STAR, DO_STAGGER_FIELDS_EXCHANGES                (solve_for_pressure.step_after_dynamics)
  :1034     THERMODYNAMICS (GMREDI_RESIDUAL_FLOW, GAD_ADVECTION, TEMP/SALT_INTEGRATE)
  :1049     TRACERS_CORRECTION_STEP (nothing in V4r4)
  :1080     DO_FIELDS_BLOCKING_EXCHANGES (theta, salt; staggerTimeStep=T, storePhiHyd4Phys=F, no GGL90 horizdiff)

Backward-mode seams (plan Task 17, mitgcm_jax/adjoint/modes.py, docs/ADJOINT_MODES.md): `adj` (a static
AdjointConfig, default exact = no seam) selects which derivatives the reverse pass keeps. Every seam is marked
`# ADJOINT SEAM` below; none changes a forward value (stop_gradient / differentiate_at are the identity forward).
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
from mitgcm_jax.pkgs import gad as gad_mod
from mitgcm_jax.pkgs import ggl90 as ggl_mod
from mitgcm_jax.pkgs import gmredi as gm_mod
from mitgcm_jax.pkgs import mom_common as mc
from mitgcm_jax.pkgs import mom_vecinv as mv_mod
from mitgcm_jax.pkgs import salt_plume as sp_mod
from mitgcm_jax.state import State

# EXF_FIELDS arrays carried from step to step (their halos keep fldConst / exchanged values)
EXF_STATE = ("ustress", "vstress", "hflux", "sflux", "swflux", "apressure", "saltflx", "spflx", "runoff")
FF_FIELDS = ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "saltPlumeFlux", "pLoad", "sIceLoad")


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


def do_oceanic_phys(P, g, ex, f, kLowC, adj=EXACT):
    """ff/do_oceanic_phys.F: returns the dict of fields it writes (FFIELDS surface forcing, rhoInSitu, sigma*,
    IVDConvCount, hMixLayer, saltPlumeDepth, GGL90*, GM/Redi tensor + Psi). f: fields at DO_OCEANIC_PHYS entry.
    adj: backward-mode semantics (adjoint/modes.py); forward values are the same in every mode."""
    out = {}
    ff = {k: f[k] for k in FF_FIELDS}
    # ADJOINT SEAM salt_plume="off" (useSALT_PLUMEinAdMode=F): the reverse sweep skips SALT_PLUME_DO_EXCH (:579-582),
    # SALT_PLUME_FORCING_SURF (external_forcing_surf.F:235-239) and SALT_PLUME_TENDENCY_APPLY_S (apply_forcing.F:931)
    ff["saltPlumeFlux"] = stop_gradient_if(adj.salt_plume == "off", ff["saltPlumeFlux"])
    ff, sfo, spd = ef.oceanic_phys_forcing(P.sf, g, ex, ff, f["saltPlumeDepth"], f["theta"], f["salt"])  # :288-614
    out.update(ff)
    out.update(sfo)                        # surfaceForcingU/V/T/S, PmEpR, phi0surf
    # tile loop (:616-1110): zeroing, FIND_RHO_2D, GRAD_SIGMA, CALC_IVDC, CALC_OCE_MXLAYER (:640-945)
    r = rs_mod.rho_sigma_ivdc_mxlayer(P.rs, g, f["theta"], f["salt"], f["hMixLayer"])
    out.update(rhoInSitu=r["rhoInSitu"], IVDConvCount=r["IVDConvCount"], hMixLayer=r["hMixLayer"])
    # ADJOINT SEAM gm_sigma="stable": :900-907 ZERO_ADJ_LOC(sigmaX/Y/R) (GMREDI_WITH_STABLE_ADJOINT) cuts the adjoint
    # of the density gradients for every reader (GGL90_CALC, GMREDI_CALC_TENSOR; CALC_IVDC's flag and
    # CALC_OCE_MXLAYER carry no derivative); rhoInSitu keeps its derivative
    sigX, sigY, sigR = stop_gradient_if(adj.gm_sigma == "stable", r["sigmaX"], r["sigmaY"], r["sigmaR"])
    # :949 SALT_PLUME_CALC_DEPTH (saltPlumeDepth was zeroed at :292)
    out["saltPlumeDepth"] = stop_gradient_if(  # ADJOINT SEAM salt_plume="off" (:948-951 skipped in the reverse)
        adj.salt_plume == "off",
        sp_mod.salt_plume_calc_depth(P.sp, P.rs.eos, g, r["rhoInSitu"][:, 0], f["theta"], f["salt"], kLowC))
    # :1063 GGL90_CALC (viscArU/V, diffKr zeroed at :661-667; the kernel returns zeros outside its loops)
    tke, vU, vV, dK = ggl_mod.ggl90_calc(P.ggl, g, f["GGL90TKE"], f["uVel"], f["vVel"], sigR,
                                         sfo["surfaceForcingU"], sfo["surfaceForcingV"], f["recip_hFacC"])
    # ADJOINT SEAM ggl90="frozen" (useGGL90inAdMode=F): no derivative through GGL90_CALC; the implicit solves keep
    # the forward kappaRk / kappaRU / kappaRV (TAF STOREs them after the GGL90 terms: docs/ADJOINT_MODES.md)
    tke, vU, vV, dK = stop_gradient_if(adj.ggl90 == "frozen", tke, vU, vV, dK)
    out.update(GGL90TKE=tke, GGL90viscArU=vU, GGL90viscArV=vV, GGL90diffKr=dK)
    # :1100 GMREDI_CALC_TENSOR, :1163 GMREDI_DO_EXCH
    t = gm_mod.gmredi_calc_tensor(P.gm, g, sigX, sigY, sigR, g.kapGM, g.kapRedi)
    psx, psy = gm_mod.gmredi_do_exch(P.gm, ex, t["GM_PsiX"], t["GM_PsiY"])
    t = dict(t, GM_PsiX=psx, GM_PsiY=psy)
    out.update(t)
    out["_sigma"] = (r["sigmaX"], r["sigmaY"], r["sigmaR"])
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


def forward_step(P, g, ex, kLowC, st: State, exf_in, adj=EXACT):
    """One FORWARD_STEP. adj: static AdjointConfig (backward-mode semantics; forward values identical in every
    mode)."""
    f = dict(st.f)
    aux = {}
    myIter = st.it
    # forward_step.F:418-450  RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) with rStarFacNm1
    f["pStarFacK"] = fs.reset_nlfs_vars(f["rStarFacC"])
    (f["hFacC"], f["hFacW"], f["hFacS"], f["recip_hFacC"], f["recip_hFacW"], f["recip_hFacS"]) = fs.update_r_star(
        g, f["rStarFacNm1C"], f["rStarFacNm1W"], f["rStarFacNm1S"], f["recip_hFacC"], f["recip_hFacW"],
        f["recip_hFacS"])
    aux["S01_update_rstar_F"] = {k: f[k] for k in ("hFacC", "hFacW", "hFacS", "recip_hFacC")}
    # :495 LOAD_FIELDS_DRIVER -> EXF_GETFORCING
    exf = {k: f[k] for k in EXF_STATE}
    ff = {k: f[k] for k in FF_FIELDS}
    exf, ff = exf_mod.exf_getforcing(P.exf, g, ex, exf, ff, exf_in["bufs"], exf_in["facs"], exf_in["myTime"])
    # :524-530 CTRL_MAP_FORCING (useCTRL=T): zero xx_gentim2d adds + the FFIELDS exchanges (pkgs/ctrl.py)
    if P.ctrl is not None:
        ff = ctrl_mod.ctrl_map_forcing(P.ctrl, g, ex, ff)
    f.update(exf)
    f.update({k: ff[k] for k in FF_FIELDS if k in ff})
    aux["S02_load_fields"] = dict(exf, **ff)
    # :609 DO_OCEANIC_PHYS
    op = do_oceanic_phys(P, g, ex, f, kLowC, adj)
    sig = op.pop("_sigma")
    aux["P02"] = {"sigmaX": sig[0], "sigmaY": sig[1], "sigmaR": sig[2]}
    f.update({k: v for k, v in op.items() if k in f or k in ("PmEpR",)})
    aux["S04_oceanic_phys"] = op
    # :808 DYNAMICS
    uVel, vVel, wVel = f["uVel"], f["vVel"], f["wVel"]

    def mom_vecinv(kU, kV):
        return mom_vecinv_adj(P, g, adj, uVel, vVel, wVel, f["hFacC"], f["hFacW"], f["hFacS"], f["recip_hFacC"],
                              f["recip_hFacW"], f["recip_hFacS"], kU, kV)

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
    theta, salt = t["theta"], t["salt"]
    # :1049 TRACERS_CORRECTION_STEP
    theta, salt = tc.tracers_correction_step(P.tc, theta, salt)
    # :1080 DO_FIELDS_BLOCKING_EXCHANGES
    theta = ex.exch_xy(theta)
    salt = ex.exch_xy(salt)
    f["theta"], f["salt"] = theta, salt
    return State(f, myIter + 1), aux
