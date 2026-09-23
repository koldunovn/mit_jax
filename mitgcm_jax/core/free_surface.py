"""Non-linear free surface / r* (z*) kernels of FORWARD_STEP (plan Task 15), V4r4 flux-forced branches.

Literal ports (c66g model/src; none is overridden in the flux-forced code/ tree):
  reset_nlfs_vars.F   pStarFacK = 1 (ocean)                                       -> reset_nlfs_vars
  update_r_star.F     hFac = h0Fac * rStarFac (useLatest) or * rStarFacNm1         -> update_r_star
  calc_r_star.F       rStarFac from etaH, exchanges, rStarExp, rStarDh*Dt          -> calc_r_star
  integr_continuity.F dEtaHdt, etaN (exactConserv), wVel (integrate_for_w.F r* branch), update_etah.F
                                                                                   -> integr_continuity
  do_stagger_fields_exchanges.F  EXCH_UV_3D_RL(u,v), EXCH_3D_RL(w)                 -> do_stagger_fields_exchanges
Order inside FORWARD_STEP (flux-forced forward_step.F): RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) at :418-450 (every
step: ALLOW_AUTODIFF), UPDATE_R_STAR(.TRUE.) :855, UPDATE_CG2D :890, SOLVE_FOR_PRESSURE :935,
MOMENTUM_CORRECTION_STEP :951, INTEGR_CONTINUITY :965 (called after myIter was advanced, :823, so the
`myIter.EQ.nIter0` branches never run inside a step), CALC_R_STAR :980, DO_STAGGER_FIELDS_EXCHANGES :1015.
The pressure solve and the momentum correction live in solve_for_pressure.py and cg2d.py.

Vertical factors deepFacC, deepFac2F, recip_deepFac2F (set_grid_factors.F:50-62, deepAtmosphere=F) and rhoFacC,
rhoFacF, recip_rhoFacF (set_ref_state.F:74-81, rhoRefFile=' ') are 1; they are kept as named factors so every
expression keeps the Fortran operand order. kSurfC/W/S (ini_masks_etc.F:191, 204-209, 437-446: first k with
hFac != 0, else Nr+1) are computed from the masks (maskC = 1 <=> hFacC != 0, ini_masks_etc.F:484-503).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax

from mitgcm_jax.params_io import params_pytree


@params_pytree
@dataclass(frozen=True)
class FreeSurfParams:
    """Floats are traced pytree leaves (params_io.params_pytree): pass the object as a jit argument. As a
    compile-time constant, `x/deltaTFreeSurf` and `(x*c1)*c2` are rewritten by XLA (1-ulp differences, measured on
    rStarDhCDt and cg2d_b); traced, every kernel here is bitwise equal to the oracle."""
    deltaTMom: float
    deltaTFreeSurf: float
    freeSurfFac: float          # ini_parms.F:442 implicitFreeSurface -> 1
    implicSurfPress: float      # set_defaults.F:247 (1.)
    implicDiv2DFlow: float      # set_defaults.F:248 (1.)
    mass2rUnit: float           # ini_parms.F:1439 recip_rhoConst (usingZCoords)
    rUnit2mass: float           # ini_parms.F:1440 rhoConst
    facEmP: float               # integr_continuity.F:87-88 (fluidIsWater .AND. useRealFreshWaterFlux -> mass2rUnit)
    hFacInf: float              # set_defaults.F:253 (0.2)
    hFacSup: float              # set_defaults.F:254 (2.0)
    rStarAreaWeight: bool       # calc_r_star.F:67-71
    pfFacMom: float             # set_parms.F:186-190 momPressureForcing -> 1
    select_rStar: int
    nonlinFreeSurf: int
    useRealFreshWaterFlux: bool
    Nr: int = 50

    @classmethod
    def from_namelists(cls, nml, Nr=50):
        p1 = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
        p3 = lambda k, d: nml.get("data", "parm03", k, default=d)  # noqa: E731
        p4 = lambda k, d: nml.get("data", "parm04", k, default=d)  # noqa: E731
        pkg = lambda k: bool(nml.get("data.pkg", "packages", k, default=False))  # noqa: E731
        # --- options whose other branches are not ported (V4r4 value in brackets) --------------------------
        checks = [
            (str(p1("buoyancyRelation", "OCEANIC")).strip().upper() == "OCEANIC",   # set_defaults.F:175
             "buoyancyRelation ['OCEANIC'] (fluidIsWater, usingZCoords: ini_parms.F:418-421)"),
            (int(p1("select_rStar", 0)) > 0, "select_rStar [2] (> 0: r* code)"),        # set_defaults.F:255
            (int(p1("nonlinFreeSurf", 0)) > 2, "nonlinFreeSurf [4] (> 2: UPDATE_CG2D every step)"),  # :252
            (bool(p1("implicitFreeSurface", False)), "implicitFreeSurface [T]"),      # set_defaults.F:245
            (not bool(p1("rigidLid", False)), "rigidLid [F]"),                         # set_defaults.F:246
            (bool(p1("exactConserv", False)), "exactConserv [T]"),                     # set_defaults.F:249
            (float(p1("implicSurfPress", 1.0)) == 1.0, "implicSurfPress [1]"),         # set_defaults.F:247
            (float(p1("implicDiv2DFlow", 1.0)) == 1.0,                                 # set_defaults.F:248
             "implicDiv2DFlow [1] (calc_div_ghat.F:79, integr_continuity.F:206, update_etah.F:62 branches)"),
            (int(p1("selectAddFluid", 0)) == 0, "selectAddFluid [0]"),                 # set_defaults.F:257
            (not bool(p1("nonHydrostatic", False)), "nonHydrostatic [F]"),             # set_defaults.F:210
            (not bool(p1("implicitIntGravWave", False)), "implicitIntGravWave [F]"),   # set_defaults.F:180
            (bool(p1("staggerTimeStep", False)), "staggerTimeStep [T]"),               # set_defaults.F:181
            (not bool(p3("applyExchUV_early", False)), "applyExchUV_early [F]"),       # set_defaults.F:182
            (bool(p1("momStepping", True)), "momStepping [T]"),                        # set_defaults.F:189
            (bool(p1("momPressureForcing", True)), "momPressureForcing [T]"),          # set_defaults.F:188
            (int(p1("selectImplicitDrag", 0)) != 2, "selectImplicitDrag [0] (ALLOW_SOLVE4_PS_AND_DRAG)"),  # :206
            (not bool(p1("usePickupBeforeC54", False)), "usePickupBeforeC54 [F]"),     # set_defaults.F:220
            (str(p1("rhoRefFile", " ")).strip() == "", "rhoRefFile [' '] (rhoFac = 1)"),  # set_defaults.F:61
            (not bool(p4("deepAtmosphere", False)), "deepAtmosphere [F]"),             # set_defaults.F:73
            (int(p4("selectSigmaCoord", 0)) == 0, "selectSigmaCoord [0]"),             # set_defaults.F:51
            (not pkg("useOBCS"), "useOBCS [F]"),                                       # packages_boot.F
            (not pkg("useOffLine"), "useOffLine [F]"),                                 # packages_boot.F:142
        ]
        bad = [msg for ok, msg in checks if not ok]
        if bad:
            raise NotImplementedError("free-surface branch not ported: " + "; ".join(bad))
        # --- time steps: ini_parms.F:884-909 -------------------------------------------------------------------
        dT, dTclock, dTtr = float(p3("deltaT", 0.0)), float(p3("deltaTClock", 0.0)), float(p3("deltaTtracer", 0.0))
        dTmom, dTfs = float(p3("deltaTMom", 0.0)), float(p3("deltaTFreeSurf", 0.0))  # set_defaults.F:293-295
        for alt in (dTclock, dTtr, dTmom, dTfs):                                    # ini_parms.F:884-887
            if dT == 0.0:
                dT = alt
        if dT == 0.0:
            raise ValueError("no time step in data PARM03")
        if dTmom == 0.0:
            dTmom = dT                                                              # ini_parms.F:898
        if dTfs == 0.0:
            dTfs = dTmom                                                            # ini_parms.F:909
        rhoNil = float(p1("rhoNil", 999.8))                                         # set_defaults.F:106
        rhoConst = float(p1("rhoConst", rhoNil))                                    # ini_parms.F:445
        recip_rhoConst = 1.0 / rhoConst                                             # ini_parms.F:640
        useRFWF = bool(p1("useRealFreshWaterFlux", False))                          # set_defaults.F:258
        mass2rUnit = recip_rhoConst                                                 # ini_parms.F:1439
        vecinv = bool(p1("vectorInvariantMomentum", False))                         # set_defaults.F:190
        kesch = int(p1("selectKEscheme", 0))                                        # set_defaults.F:234
        return cls(deltaTMom=dTmom, deltaTFreeSurf=dTfs,
                   freeSurfFac=1.0,                                                 # ini_parms.F:442
                   implicSurfPress=1.0, implicDiv2DFlow=1.0,
                   mass2rUnit=mass2rUnit, rUnit2mass=rhoConst,                       # ini_parms.F:1439-1440
                   facEmP=mass2rUnit if useRFWF else 0.0,                           # integr_continuity.F:87-88
                   hFacInf=float(p1("hFacInf", 0.2)), hFacSup=float(p1("hFacSup", 2.0)),  # set_defaults.F:253-254
                   rStarAreaWeight=not (vecinv and kesch in (1, 3)),                # calc_r_star.F:67-71
                   pfFacMom=1.0,                                                    # set_parms.F:187
                   select_rStar=int(p1("select_rStar", 0)), nonlinFreeSurf=int(p1("nonlinFreeSurf", 0)),
                   useRealFreshWaterFlux=useRFWF, Nr=Nr)


def vertical_factors(Nr):
    """Named vertical factors (all 1 in V4r4): set_grid_factors.F:50-62, set_ref_state.F:74-81."""
    one_c, one_f = np.ones(Nr), np.ones(Nr + 1)
    return dict(deepFacC=one_c, recip_deepFacC=one_c, deepFac2F=one_f, recip_deepFac2F=one_f,
                rhoFacC=one_c, recip_rhoFacC=one_c, rhoFacF=one_f, recip_rhoFacF=one_f)


def k_surf(mask):
    """kSurf (1-based first wet level, Nr+1 for a dry column) from a [T, Nr, ny, nx] mask (ini_masks_etc.F:191,
    204-209 for C; 437-446 for W, S)."""
    Nr = mask.shape[1]
    wet = mask != 0
    k = jnp.argmax(wet, axis=1) + 1
    return jnp.where(jnp.any(wet, axis=1), k, Nr + 1).astype(jnp.int32)


# -------------------------------------------------------------------------------------------------------------------


def reset_nlfs_vars(rStarFacC):
    """RESET_NLFS_VARS (reset_nlfs_vars.F:47-68): pStarFacK = 1 on the full array (fluidIsAir = F)."""
    return jnp.ones_like(rStarFacC)


def update_r_star(g, rStarFacC, rStarFacW, rStarFacS, recip_hFacC, recip_hFacW, recip_hFacS):
    """UPDATE_R_STAR (update_r_star.F:51-120): the same loop for useLatest=.TRUE. (pass rStarFacC/W/S) and .FALSE.
    (pass rStarFacNm1C/W/S). Full index range; recip_hFac is only written where the mask is non-zero (elsewhere it
    keeps its value: pass the current recip_hFac*). Returns hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW,
    recip_hFacS."""
    hFacC = g.h0FacC * rStarFacC[:, None]                         # update_r_star.F:58-59 / 93-94
    hFacW = g.h0FacW * rStarFacW[:, None]                         # :60-61 / 95-96
    hFacS = g.h0FacS * rStarFacS[:, None]                         # :62-63 / 97-98
    out = []
    for h, m, old in ((hFacC, g.maskC, recip_hFacC), (hFacW, g.maskW, recip_hFacW), (hFacS, g.maskS, recip_hFacS)):
        wet = m != 0.0                                            # :74-79 / 109-114
        out.append(jnp.where(wet, 1.0 / jnp.where(wet, h, 1.0), old))
    return (hFacC, hFacW, hFacS, *out)


def calc_r_star(p: FreeSurfParams, g, ex, etaFld, rStarFacC, rStarFacW, rStarFacS):
    """CALC_R_STAR(etaH) (calc_r_star.F:65-336).

    Inputs: etaFld (= etaH after INTEGR_CONTINUITY), the current rStarFacC/W/S. Returns a dict with rStarFacNm1C/W/S,
    rStarFacC/W/S (new, exchanged), rStarExpC/W/S, rStarDhCDt/WDt/SDt, pStarFacK, and the counters of the
    hFacInf/hFacSup checks (calc_r_star.F:182-202: icntc1, icntw, icnts, icntc2 summed over tiles, maxhFacC). The
    Fortran STOPs when icntc1+icntw+icnts > 0 (:204-246); a jitted kernel cannot stop, so the caller must check.
    """
    L = g.layout
    rStarFacC, rStarFacW, rStarFacS = jnp.asarray(rStarFacC), jnp.asarray(rStarFacW), jnp.asarray(rStarFacS)
    kSurfC, kSurfW, kSurfS = k_surf(g.maskC), k_surf(g.maskW), k_surf(g.maskS)
    Nr = L.Nr
    out = dict(rStarFacNm1C=rStarFacC, rStarFacNm1S=rStarFacS, rStarFacNm1W=rStarFacW)   # :83-89
    expC, expW, expS = rStarFacC, rStarFacW, rStarFacS                                    # :94-100
    # :103-113 new column thickness on j=0..sNy+1, i=0..sNx+1
    J, I = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
    newC = jnp.where(kSurfC[:, J, I] <= Nr,
                     (etaFld[:, J, I] + g.Ro_surf[:, J, I] - g.R_low[:, J, I]) * g.recip_Rcol[:, J, I], 1.0)
    rC = rStarFacC.at[:, J, I].set(newC)
    if not p.rStarAreaWeight:
        raise NotImplementedError("calc_r_star.F:144-170 simple average (selectKEscheme 1 or 3) is not ported")
    # :116-129 area-weighted rStarFacW on j=1..sNy, i=1..sNx+1
    J, I, Im = L.js(1, L.sNy), L.is_(1, L.sNx + 1), L.is_(0, L.sNx)
    wet = kSurfW[:, J, I] <= Nr
    tmpfldW = g.rSurfW[:, J, I] - g.rLowW[:, J, I]
    tmpfldW = jnp.where(wet, tmpfldW, 1.0)
    newW = (0.5 * (etaFld[:, J, Im] * g.rA[:, J, Im] + etaFld[:, J, I] * g.rA[:, J, I])
            * g.recip_rAw[:, J, I] + tmpfldW) / tmpfldW
    rW = rStarFacW.at[:, J, I].set(jnp.where(wet, newW, 1.0))
    # :130-143 rStarFacS on j=1..sNy+1, i=1..sNx
    J, I, Jm = L.js(1, L.sNy + 1), L.is_(1, L.sNx), L.js(0, L.sNy)
    wet = kSurfS[:, J, I] <= Nr
    tmpfldS = jnp.where(wet, g.rSurfS[:, J, I] - g.rLowS[:, J, I], 1.0)
    newS = (0.5 * (etaFld[:, Jm, I] * g.rA[:, Jm, I] + etaFld[:, J, I] * g.rA[:, J, I])
            * g.recip_rAs[:, J, I] + tmpfldS) / tmpfldS
    rS = rStarFacS.at[:, J, I].set(jnp.where(wet, newS, 1.0))
    # :182-202 checks on j=1..sNy+1, i=1..sNx+1 (before the exchange)
    J, I = L.js(1, L.sNy + 1), L.is_(1, L.sNx + 1)
    lowC, lowW, lowS = rC[:, J, I] < p.hFacInf, rW[:, J, I] < p.hFacInf, rS[:, J, I] < p.hFacInf
    highC = rC[:, J, I] > p.hFacSup
    out.update(icntc1=jnp.sum(lowC), icntw=jnp.sum(lowW), icnts=jnp.sum(lowS), icntc2=jnp.sum(highC),
               maxhFacC=jnp.max(jnp.where(highC, rC[:, J, I], 0.0)))
    rC = ex.exch_xy(rC)                                           # :262 _EXCH_XY_RL
    rW, rS = ex.exch_uv_xy(rW, rS, False)                         # :263 EXCH_UV_XY_RL(.FALSE.)
    # :301-316 full range (W2_FILL_NULL_REGIONS undefined)
    out.update(rStarFacC=rC, rStarFacW=rW, rStarFacS=rS,
               rStarDhCDt=(rC - expC) / p.deltaTFreeSurf,
               rStarDhWDt=(rW - expW) / p.deltaTFreeSurf,
               rStarDhSDt=(rS - expS) / p.deltaTFreeSurf,
               # old rStarFac is never 0 (1 at init, ini_nlfs_vars.F:87; written only from etaH): guard for AD
               rStarExpC=rC / jnp.where(expC != 0.0, expC, 1.0),
               rStarExpW=rW / jnp.where(expW != 0.0, expW, 1.0),
               rStarExpS=rS / jnp.where(expS != 0.0, expS, 1.0),
               pStarFacK=jnp.ones_like(rC))                       # :324-331 (ALLOW_AUTODIFF, fluidIsAir=F)
    return out


def integr_continuity(p: FreeSurfParams, g, ex, uFld, vFld, hFacW, hFacS, EmPmR, etaN, etaH, dEtaHdt, wVel):
    """INTEGR_CONTINUITY(uVel, vVel) inside FORWARD_STEP (myIter != nIter0), integr_continuity.F:79-316.

    Inputs: corrected uFld, vFld; hFacW/S of UPDATE_R_STAR(.TRUE.); EmPmR; etaN (from SOLVE_FOR_PRESSURE), etaH,
    dEtaHdt, wVel (previous values: only interior points are rewritten). Returns a dict with dEtaHdt, PmEpR, etaN,
    wVel, etaHnm1, etaH.
    """
    L = g.layout
    Nr = L.Nr
    etaN, etaH, dEtaHdt, wVel, uFld, vFld = (jnp.asarray(a) for a in (etaN, etaH, dEtaHdt, wVel, uFld, vFld))
    vf = vertical_factors(Nr)
    kSurfC = k_surf(g.maskC)
    J1, I1 = L.js(1, L.sNy + 1), L.is_(1, L.sNx + 1)
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)

    dfc, rfc = jnp.asarray(vf["deepFacC"])[None, :, None, None], jnp.asarray(vf["rhoFacC"])[None, :, None, None]
    drF = jnp.asarray(g.drF)
    # integr_continuity.F:107-116 = integrate_for_w.F:64-73 (same expression), j,i = 1..sNy+1, 1..sNx+1, all k at
    # once (no k recurrence): ((((u*dyG)*deepFacC)*rhoFacC)*drF)*hFacW
    uT = uFld[:, :, J1, I1] * g.dyG[:, None, J1, I1] * dfc * rfc * drF[None, :, None, None] * hFacW[:, :, J1, I1]
    vT = vFld[:, :, J1, I1] * g.dxG[:, None, J1, I1] * dfc * rfc * drF[None, :, None, None] * hFacS[:, :, J1, I1]
    # uTrans(i+1,j)-uTrans(i,j)+vTrans(i,j+1)-vTrans(i,j) on j=1..sNy, i=1..sNx (window index = Fortran index - 1)
    div = uT[:, :, :-1, 1:] - uT[:, :, :-1, :-1] + vT[:, :, 1:, :-1] - vT[:, :, :-1, :-1]

    # :95-131 hDivFlow = 0; DO k=1,Nr: hDivFlow = hDivFlow + maskC*div  (sequential in k)
    def acc(h, t):
        return h + t, None
    hDivFlow, _ = lax.scan(acc, jnp.zeros((L.nTiles, L.sNy, L.sNx)),
                           jnp.moveaxis(g.maskC[:, :, J, I] * div, 1, 0))

    # :167-182 (myIter != nIter0): PmEpR = -EmPmR on the full range; dEtaHdt on the interior
    PmEpR = -EmPmR
    ks = kSurfC[:, J, I]
    dEtaHdt_i = (-hDivFlow * g.recip_rA[:, J, I] * jnp.asarray(vf["recip_deepFac2F"])[ks - 1]
                 - p.facEmP * EmPmR[:, J, I])
    dEtaHdt = dEtaHdt.at[:, J, I].set(dEtaHdt_i)
    # :202-219 exactConserv, myIter != nIter0, implicDiv2Dflow != 0
    etaN = etaN.at[:, J, I].set(etaH[:, J, I] + p.implicDiv2DFlow * dEtaHdt_i * p.deltaTFreeSurf)
    # :235-248 rStarDhDt
    rStarDhDt = (dEtaHdt_i * jnp.asarray(vf["deepFac2F"])[ks - 1] * jnp.asarray(vf["rhoFacF"])[ks - 1]
                 * g.recip_Rcol[:, J, I])
    # :255-293 INTEGRATE_FOR_W, k = Nr..1 (integrate_for_w.F:121-147, r* branch)
    conv2d = -div                                                 # integrate_for_w.F:76-77
    rA_i = g.recip_rA[:, J, I]
    rdf, rrf = jnp.asarray(vf["recip_deepFac2F"]), jnp.asarray(vf["recip_rhoFacF"])
    df2, rfF = jnp.asarray(vf["deepFac2F"]), jnp.asarray(vf["rhoFacF"])
    h0 = g.h0FacC[:, :, J, I]
    mC = g.maskC[:, :, J, I]
    k = Nr - 1                                                    # :126-135 k = Nr
    wNr = ((conv2d[:, k] * rA_i - rStarDhDt * drF[k] * h0[:, k]) * mC[:, k] * rdf[k] * rrf[k])

    def up(wk1, x):                                               # :136-146 k < Nr
        cv, h0k, mk, drFk, rdfk, rrfk, df2k1, rfFk1 = x
        w = (wk1 * df2k1 * rfFk1 + cv * rA_i - rStarDhDt * drFk * h0k) * mk * rdfk * rrfk
        return w, w

    xs = (jnp.moveaxis(conv2d[:, :-1], 1, 0), jnp.moveaxis(h0[:, :-1], 1, 0), jnp.moveaxis(mC[:, :-1], 1, 0),
          drF[:Nr - 1], rdf[:Nr - 1], rrf[:Nr - 1], df2[1:Nr], rfF[1:Nr])
    _, w_up = lax.scan(up, wNr, xs, reverse=True)                  # k = Nr-1 .. 1, stacked in k order
    w_new = jnp.concatenate([jnp.moveaxis(w_up, 0, 1), wNr[:, None]], axis=1)
    wVel = wVel.at[:, :, J, I].set(w_new)
    # :301-303 exactConserv, myIter != nIter0, implicDiv2Dflow != 0: EXCH_XY_RL(etaN)
    etaN = ex.exch_xy(etaN)
    # :304-305 implicitIntGravWave=F, myIter != nIter0: no wVel exchange
    # :309-316 UPDATE_ETAH (update_etah.F:53-77, implicDiv2Dflow = 1: full range copy; no exchange :92-94)
    etaHnm1 = etaH
    etaH = etaN
    return dict(dEtaHdt=dEtaHdt, PmEpR=PmEpR, etaN=etaN, wVel=wVel, etaHnm1=etaHnm1, etaH=etaH)


def do_stagger_fields_exchanges(ex, uVel, vVel, wVel):
    """DO_STAGGER_FIELDS_EXCHANGES (do_stagger_fields_exchanges.F:35-58; staggerTimeStep=T, applyExchUV_early=F,
    implicitIntGravWave=F, useOffLine=F)."""
    uVel, vVel = ex.vector(uVel, vVel, "UV3s")                   # :41-42 EXCH_UV_3D_RL(uVel, vVel, .TRUE., Nr)
    wVel = ex.scalar(wVel, "3D")                                  # :44-45 EXCH_3D_RL(wVel, Nr)
    return uVel, vVel, wVel
