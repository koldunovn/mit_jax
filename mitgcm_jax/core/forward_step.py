"""FORWARD_STEP of the V4r4 flux-forced configuration (flux-forced code/forward_step.F), composed from the ported
kernels (plan Task 19). Every call is at the place of the Fortran call, with its file:line.

    st1 = forward_step(P, g, ex, st0, exf_in)

`P` (ModelParams, a pytree: pass it as a jit argument), `g` (Grid), `ex` (Exchanger), `st0` (State at the start of
iteration st0.it), `exf_in` = dict(bufs, facs, myTime) from the host-side EXF record loader for this step
(exf_fluxforced.ExfRecordLoader.load). Returns the State at the start of the next iteration, plus `aux` (per-step
intermediates keyed by dump stage, for the first-divergence harness and diagnostics).

Order (forward_step.F, flux-forced tree):
  :418-450  RESET_NLFS_VARS, UPDATE_R_STAR(.FALSE.)                 (every step: ALLOW_AUTODIFF)
  :495      LOAD_FIELDS_DRIVER -> EXF_GETFORCING                     (CTRL_MAP_FORCING: useCTRL=F in the oracles)
  :609      DO_OCEANIC_PHYS  (EXTERNAL_FORCING_SURF, rho/sigma/IVDC/mxlayer, salt plume depth, GGL90, GM/Redi)
  :808      DYNAMICS
  :823      myIter <- myIter+1
  :855-1015 UPDATE_R_STAR(.TRUE.), UPDATE_CG2D, SOLVE_FOR_PRESSURE, MOMENTUM_CORRECTION_STEP, INTEGR_CONTINUITY,
            CALC_R_STAR, DO_STAGGER_FIELDS_EXCHANGES                (solve_for_pressure.step_after_dynamics)
  :1034     THERMODYNAMICS (GMREDI_RESIDUAL_FLOW, GAD_ADVECTION, TEMP/SALT_INTEGRATE)
  :1049     TRACERS_CORRECTION_STEP (nothing in V4r4)
  :1080     DO_FIELDS_BLOCKING_EXCHANGES (theta, salt; staggerTimeStep=T, storePhiHyd4Phys=F, no GGL90 horizdiff)
"""

from typing import NamedTuple

import jax.numpy as jnp

from mitgcm_jax.core import dynamics as dyn_mod
from mitgcm_jax.core import external_forcing as ef
from mitgcm_jax.core import free_surface as fs
from mitgcm_jax.core import grad_sigma as rs_mod
from mitgcm_jax.core import solve_for_pressure as sfp
from mitgcm_jax.core import thermodynamics as th_mod
from mitgcm_jax.core import tracers_correction as tc
from mitgcm_jax.core.cg2d import Cg2dParams
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.pkgs import gad as gad_mod
from mitgcm_jax.pkgs import ggl90 as ggl_mod
from mitgcm_jax.pkgs import gmredi as gm_mod
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


def do_oceanic_phys(P, g, ex, f, kLowC):
    """ff/do_oceanic_phys.F: returns the dict of fields it writes (FFIELDS surface forcing, rhoInSitu, sigma*,
    IVDConvCount, hMixLayer, saltPlumeDepth, GGL90*, GM/Redi tensor + Psi). f: fields at DO_OCEANIC_PHYS entry."""
    out = {}
    ff = {k: f[k] for k in FF_FIELDS}
    ff, sfo, spd = ef.oceanic_phys_forcing(P.sf, g, ex, ff, f["saltPlumeDepth"], f["theta"], f["salt"])  # :288-614
    out.update(ff)
    out.update(sfo)                        # surfaceForcingU/V/T/S, PmEpR, phi0surf
    # tile loop (:616-1110): zeroing, FIND_RHO_2D, GRAD_SIGMA, CALC_IVDC, CALC_OCE_MXLAYER (:640-945)
    r = rs_mod.rho_sigma_ivdc_mxlayer(P.rs, g, f["theta"], f["salt"], f["hMixLayer"])
    out.update(rhoInSitu=r["rhoInSitu"], IVDConvCount=r["IVDConvCount"], hMixLayer=r["hMixLayer"])
    # :949 SALT_PLUME_CALC_DEPTH (saltPlumeDepth was zeroed at :292)
    out["saltPlumeDepth"] = sp_mod.salt_plume_calc_depth(P.sp, P.rs.eos, g, r["rhoInSitu"][:, 0], f["theta"],
                                                         f["salt"], kLowC)
    # :1063 GGL90_CALC (viscArU/V, diffKr zeroed at :661-667; the kernel returns zeros outside its loops)
    tke, vU, vV, dK = ggl_mod.ggl90_calc(P.ggl, g, f["GGL90TKE"], f["uVel"], f["vVel"], r["sigmaR"],
                                         sfo["surfaceForcingU"], sfo["surfaceForcingV"], f["recip_hFacC"])
    out.update(GGL90TKE=tke, GGL90viscArU=vU, GGL90viscArV=vV, GGL90diffKr=dK)
    # :1100 GMREDI_CALC_TENSOR, :1163 GMREDI_DO_EXCH
    t = gm_mod.gmredi_calc_tensor(P.gm, g, r["sigmaX"], r["sigmaY"], r["sigmaR"], g.kapGM, g.kapRedi)
    psx, psy = gm_mod.gmredi_do_exch(P.gm, ex, t["GM_PsiX"], t["GM_PsiY"])
    t = dict(t, GM_PsiX=psx, GM_PsiY=psy)
    out.update(t)
    out["_sigma"] = (r["sigmaX"], r["sigmaY"], r["sigmaR"])
    return out


def forward_step(P, g, ex, kLowC, st: State, exf_in):
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
    f.update(exf)
    f.update({k: ff[k] for k in FF_FIELDS if k in ff})
    aux["S02_load_fields"] = dict(exf, **ff)
    # :609 DO_OCEANIC_PHYS
    op = do_oceanic_phys(P, g, ex, f, kLowC)
    sig = op.pop("_sigma")
    aux["P02"] = {"sigmaX": sig[0], "sigmaY": sig[1], "sigmaR": sig[2]}
    f.update({k: v for k, v in op.items() if k in f or k in ("PmEpR",)})
    aux["S04_oceanic_phys"] = op
    # :808 DYNAMICS
    uVel, vVel, wVel = f["uVel"], f["vVel"], f["wVel"]

    def mom_vecinv(kU, kV):
        o = mv_mod.mom_vecinv(P.mv, g, uVel, vVel, wVel, f["hFacC"], f["hFacW"], f["hFacS"], f["recip_hFacC"],
                              f["recip_hFacW"], f["recip_hFacS"], kU, kV)
        return o["gU"], o["gV"], o["guDissip"], o["gvDissip"]

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
    a = sfp.step_after_dynamics(P.fs, P.cg, g, ex, s2)
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
