"""Surface-pressure part of FORWARD_STEP (plan Task 15): UPDATE_CG2D, SOLVE_FOR_PRESSURE, MOMENTUM_CORRECTION_STEP,
and the whole sequence between DYNAMICS and THERMODYNAMICS (`step_after_dynamics`).

Literal ports (c66g model/src):
  update_cg2d.F            operator aW2d/aS2d/aC2d from the r* hFacW/S, preconditioner pW/pS/pC  -> update_cg2d
  solve_for_pressure.F     cg2d_b (EmPmR with useRealFreshWaterFlux, CALC_DIV_GHAT, exactConserv etaH term), first
                           guess cg2d_x = Bo_surf*etaN, CG2D, EXCH_XY_RL, etaN = recip_Bo*cg2d_x   -> cg2d_rhs,
                                                                                                 solve_for_pressure
  calc_div_ghat.F          (implicDiv2Dflow = 1 branch)                                           -> cg2d_rhs
  momentum_correction_step.F + calc_grad_phi_surf.F + correction_step.F          -> momentum_correction_step
The flux-forced override of momentum_correction_step.F only adds the Um_dPsdx/Vm_dPsdy diagnostics (diff checked:
lines 59-61 and 101-116 of the override); the c66g version is ported.
"""

import jax.numpy as jnp
from jax import lax

from mitgcm_jax.core import cg2d as _cg2d
from mitgcm_jax.core.free_surface import (FreeSurfParams, calc_r_star, do_stagger_fields_exchanges,
                                          integr_continuity, k_surf, update_r_star, vertical_factors)


def _ksum(init, terms):
    """init + terms[:, 0] + terms[:, 1] + ... (Fortran `DO k=1,Nr: a = a + t(k)`, sequential)."""
    out, _ = lax.scan(lambda a, t: (a + t, None), init, jnp.moveaxis(terms, 1, 0))
    return out


def update_cg2d(p: FreeSurfParams, cp: _cg2d.Cg2dParams, g, ex, hFacW, hFacS, recip_Bo, pW, pS, pC):
    """UPDATE_CG2D (update_cg2d.F:59-195). hFacW/S: after UPDATE_R_STAR(.TRUE.). pW, pS, pC: current preconditioner
    (only i,j = 1..sNx+1, 1..sNy+1 are rewritten). updatePreCond is .TRUE. every step (cg2dPreCondFreq = 1,
    update_cg2d.F:60-64; other frequencies are rejected by Cg2dParams.from_namelists). Returns aW2d, aS2d, aC2d, pW,
    pS, pC."""
    L = g.layout
    pW, pS, pC = jnp.asarray(pW), jnp.asarray(pS), jnp.asarray(pC)
    cg2dNorm, cg2dpcOffDFac = cp.cg2dNorm, cp.cg2dpcOffDFac
    J1, I1 = L.js(1, L.sNy + 1), L.is_(1, L.sNx + 1)
    z2 = jnp.zeros_like(hFacW[:, 0])                             # :72-80 (full range zeroed; typed like hFacW)
    drF = jnp.asarray(g.drF)[None, :, None, None]
    # :101-115 aW2d = aW2d + (dyG*drF(k)*hFacW)*recip_dxC, k = 1..Nr in order (terms for all k, then the k sum)
    tW = g.dyG[:, None, J1, I1] * drF * hFacW[:, :, J1, I1] * g.recip_dxC[:, None, J1, I1]
    tS = g.dxG[:, None, J1, I1] * drF * hFacS[:, :, J1, I1] * g.recip_dyC[:, None, J1, I1]
    aW, aS = _ksum(z2[:, J1, I1], tW), _ksum(z2[:, J1, I1], tS)
    aW = aW * cg2dNorm * p.implicSurfPress * p.implicDiv2DFlow                 # :119-132
    aS = aS * cg2dNorm * p.implicSurfPress * p.implicDiv2DFlow
    aW2d = z2.at[:, J1, I1].set(aW)
    aS2d = z2.at[:, J1, I1].set(aS)
    # :147-156 main diagonal (deepAtmosphere = F) on the interior
    J, I, Jp, Ip = L.js(1, L.sNy), L.is_(1, L.sNx), L.js(2, L.sNy + 1), L.is_(2, L.sNx + 1)
    aC = -(aW2d[:, J, I] + aW2d[:, J, Ip] + aS2d[:, J, I] + aS2d[:, Jp, I]
           + p.freeSurfFac * cg2dNorm * recip_Bo[:, J, I] * g.rA[:, J, I] / p.deltaTMom / p.deltaTFreeSurf)
    aC2d = z2.at[:, J, I].set(aC)
    aC2d = ex.exch_xy(aC2d)                                                    # :164 EXCH_XY_RS
    # :169-191 preconditioner on j=1..sNy+1, i=1..sNx+1
    Im, Jm = L.is_(0, L.sNx), L.js(0, L.sNy)
    a = aC2d[:, J1, I1]
    pC_n = jnp.where(a == 0.0, 1.0, 1.0 / jnp.where(a == 0.0, 1.0, a))        # :171-175
    pW_tmp = a + aC2d[:, J1, Im]                                               # :176
    z = pW_tmp == 0.0
    pW_n = jnp.where(z, 0.0, -aW2d[:, J1, I1] / (cg2dpcOffDFac * jnp.where(z, 1.0, pW_tmp)) ** 2)  # :177-182
    pS_tmp = a + aC2d[:, Jm, I1]                                               # :183
    z = pS_tmp == 0.0
    pS_n = jnp.where(z, 0.0, -aS2d[:, J1, I1] / (cg2dpcOffDFac * jnp.where(z, 1.0, pS_tmp)) ** 2)  # :184-189
    return (aW2d, aS2d, aC2d, pW.at[:, J1, I1].set(pW_n), pS.at[:, J1, I1].set(pS_n),
            pC.at[:, J1, I1].set(pC_n))


def cg2d_rhs(p: FreeSurfParams, g, gU, gV, hFacW, hFacS, EmPmR, etaN, etaH, Bo_surf):
    """cg2d_b and first guess cg2d_x of SOLVE_FOR_PRESSURE (solve_for_pressure.F:124-255; putPmEinXvector=F,
    exactConserv=T, NONHYDROSTATIC/OBCS off). gU, gV: after DYNAMICS (implicit viscosity); hFacW/S: after
    UPDATE_R_STAR(.TRUE.); etaN, etaH: current. Returns cg2d_b, cg2d_x (full arrays, halos as in Fortran)."""
    L = g.layout
    vf = vertical_factors(L.Nr)
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    etaN, etaH, Bo_surf, gU, gV = (jnp.asarray(a) for a in (etaN, etaH, Bo_surf, gU, gV))
    cg2d_x = Bo_surf * etaN                                                    # :132 (full range)
    cg2d_b = jnp.zeros_like(etaN)                                              # :133
    if p.useRealFreshWaterFlux:                                                # :136-145 (fluidIsWater)
        tmpFac = p.freeSurfFac * p.mass2rUnit * p.implicDiv2DFlow              # :137
        cg2d_b = cg2d_b.at[:, J, I].set(tmpFac * g.rA[:, J, I] * EmPmR[:, J, I] / p.deltaTMom * g.maskInC[:, J, I])
    # :177-182 CALC_DIV_GHAT for k = Nr..1 (calc_div_ghat.F:65-126, 138-169; implicDiv2Dflow = 1 branch); xA, yA and
    # pf for all k (no recurrence), the cg2d_b updates k = Nr..1 in order
    b = cg2d_b[:, J, I]
    J1, I1 = L.js(1, L.sNy + 1), L.is_(1, L.sNx + 1)
    Ju, Iu = L.js(1, L.sNy), L.is_(1, L.sNx + 1)          # pf for u: j=1..sNy, i=1..sNx+1
    Jv, Iv = L.js(1, L.sNy + 1), L.is_(1, L.sNx)          # pf for v: j=1..sNy+1, i=1..sNx
    dfc = jnp.asarray(vf["deepFacC"])[None, :, None, None]
    rfc = jnp.asarray(vf["rhoFacC"])[None, :, None, None]
    drF = jnp.asarray(g.drF)[None, :, None, None]
    xA = g.dyG[:, None, J1, I1] * dfc * drF * hFacW[:, :, J1, I1] * rfc              # :67-68
    yA = g.dxG[:, None, J1, I1] * dfc * drF * hFacS[:, :, J1, I1] * rfc              # :69-70
    pfu = xA[:, :, :-1, :] * gU[:, :, Ju, Iu] / p.deltaTMom                         # :81-85 [sNy, sNx+1]
    pfv = yA[:, :, :, :-1] * gV[:, :, Jv, Iv] / p.deltaTMom                         # :140-144 [sNy+1, sNx]

    def level(b, x):
        pu, pv = x
        b = b + pu[:, :, 1:] - pu[:, :, :-1]                                        # :121-126
        b = b + pv[:, 1:, :] - pv[:, :-1, :]                                        # :164-169
        return b, None

    b, _ = lax.scan(level, b, (jnp.moveaxis(pfu, 1, 0), jnp.moveaxis(pfv, 1, 0)), reverse=True)
    # :211-221 exactConserv: - freeSurfFac*rA*deepFac2F(ks)/deltaTMom/deltaTFreeSurf*etaH
    ks = k_surf(g.maskC)[:, J, I]
    b = b - (p.freeSurfFac * g.rA[:, J, I] * jnp.asarray(vf["deepFac2F"])[ks - 1]
             / p.deltaTMom / p.deltaTFreeSurf * etaH[:, J, I])
    return cg2d_b.at[:, J, I].set(b), cg2d_x


def solve_for_pressure(p: FreeSurfParams, cp: _cg2d.Cg2dParams, g, ex, ops, gU, gV, hFacW, hFacS, EmPmR, etaN,
                       etaH, Bo_surf, recip_Bo):
    """SOLVE_FOR_PRESSURE (solve_for_pressure.F:103-375). ops = (aW2d, aS2d, aC2d, pW, pS, pC) of UPDATE_CG2D.
    Returns (etaN, info): info holds cg2d_b, cg2d_x (first guess), the C02 cg2d_x as CG2D leaves it (x_fortran)
    and the CG2D diagnostics."""
    cg2d_b, cg2d_x = cg2d_rhs(p, g, gU, gV, hFacW, hFacS, EmPmR, etaN, etaH, Bo_surf)
    x, diag = _cg2d.cg2d_solve(cp, ex, ops, cg2d_b, cg2d_x)
    inner = jnp.asarray(_cg2d.interior_mask(g.layout))
    # after CG2D the interior holds the solution and every halo point the pre-solve value: exchanged (normalised)
    # first guess where cg2d.F:147 wrote it, the first guess itself where no exchange writes; EXCH_XY_RL at :312
    # then rewrites the exchanged points, so only the never-written points keep cg2d_x = Bo_surf*etaN.
    x = ex.exch_xy(jnp.where(inner, x, cg2d_x))                                 # :312 _EXCH_XY_RL(cg2d_x)
    etaN_new = recip_Bo * x                                                     # :367-375 (full range)
    info = dict(diag, cg2d_b=cg2d_b, cg2d_x0=cg2d_x)
    return etaN_new, info


def calc_grad_phi_surf(g, Bo_surf, etaFld, iMin, iMax, jMin, jMax):
    """CALC_GRAD_PHI_SURF (calc_grad_phi_surf.F:48-64) on j=jMin..jMax, i=iMin..iMax (0 elsewhere:
    momentum_correction_step.F:69-74)."""
    L = g.layout
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    Im, Jm = L.is_(iMin - 1, iMax - 1), L.js(jMin - 1, jMax - 1)
    z = jnp.zeros_like(etaFld)
    phiSurfX = z.at[:, J, I].set(g.recip_dxC[:, J, I] * (Bo_surf[:, J, I] * etaFld[:, J, I]
                                                         - Bo_surf[:, J, Im] * etaFld[:, J, Im]))
    phiSurfY = z.at[:, J, I].set(g.recip_dyC[:, J, I] * (Bo_surf[:, J, I] * etaFld[:, J, I]
                                                         - Bo_surf[:, Jm, I] * etaFld[:, Jm, I]))
    return phiSurfX, phiSurfY


def momentum_correction_step(p: FreeSurfParams, g, etaN, gU, gV, uVel, vVel, Bo_surf):
    """MOMENTUM_CORRECTION_STEP (momentum_correction_step.F:65-135, c66g; momStepping=T, no filters, OBCS off,
    applyExchUV_early=F) with CORRECTION_STEP (correction_step.F:93-133, use3Dsolver=F). etaN: after
    SOLVE_FOR_PRESSURE; uVel, vVel: current (points outside the loop range keep them). Returns uVel, vVel."""
    L = g.layout
    uVel, vVel = jnp.asarray(uVel), jnp.asarray(vVel)
    vf = vertical_factors(L.Nr)
    iMin, iMax, jMin, jMax = 1 - L.OLx + 1, L.sNx + L.OLx, 1 - L.OLy + 1, L.sNy + L.OLy  # :78-81
    phiSurfX, phiSurfY = calc_grad_phi_surf(g, Bo_surf, etaN, iMin, iMax, jMin, jMax)     # :84-88
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    psFac = (p.pfFacMom * p.implicSurfPress * jnp.asarray(vf["recip_deepFacC"])
             * jnp.asarray(vf["recip_rhoFacC"]))[None, :, None, None]           # correction_step.F:94-95
    u = (gU[:, :, J, I] - p.deltaTMom * psFac * phiSurfX[:, None, J, I]) * g.maskW[:, :, J, I]  # :106-112
    v = (gV[:, :, J, I] - p.deltaTMom * psFac * phiSurfY[:, None, J, I]) * g.maskS[:, :, J, I]  # :122-128
    return uVel.at[:, :, J, I].set(u), vVel.at[:, :, J, I].set(v)


def step_after_dynamics(p: FreeSurfParams, cp: _cg2d.Cg2dParams, g, ex, s):
    """forward_step.F:846-1016 between DYNAMICS and THERMODYNAMICS, as one function.

    s: dict with gU, gV (after DYNAMICS), uVel, vVel, wVel, etaN, etaH, dEtaHdt, EmPmR, rStarFacC/W/S,
    recip_hFacC/W/S, pW, pS, pC, Bo_surf, recip_Bo. Returns a dict with every field these routines write (the r*
    fields of CALC_R_STAR, hFac*, recip_hFac*, aW2d..pC, etaN, etaH, etaHnm1, dEtaHdt, PmEpR, uVel, vVel, wVel) and
    `cg2d` (solver diagnostics) and `rstar_checks` (calc_r_star.F:182-202 counters).
    """
    out = {}
    hFacC, hFacW, hFacS, rhC, rhW, rhS = update_r_star(                        # forward_step.F:855
        g, s["rStarFacC"], s["rStarFacW"], s["rStarFacS"], s["recip_hFacC"], s["recip_hFacW"], s["recip_hFacS"])
    out.update(hFacC=hFacC, hFacW=hFacW, hFacS=hFacS, recip_hFacC=rhC, recip_hFacW=rhW, recip_hFacS=rhS)
    ops = update_cg2d(p, cp, g, ex, hFacW, hFacS, s["recip_Bo"], s["pW"], s["pS"], s["pC"])  # :890
    out.update(zip(("aW2d", "aS2d", "aC2d", "pW", "pS", "pC"), ops))
    etaN, info = solve_for_pressure(p, cp, g, ex, ops, s["gU"], s["gV"], hFacW, hFacS, s["EmPmR"], s["etaN"],
                                    s["etaH"], s["Bo_surf"], s["recip_Bo"])     # :935
    out["cg2d"] = info
    uVel, vVel = momentum_correction_step(p, g, etaN, s["gU"], s["gV"], s["uVel"], s["vVel"], s["Bo_surf"])  # :951
    ic = integr_continuity(p, g, ex, uVel, vVel, hFacW, hFacS, s["EmPmR"], etaN, s["etaH"], s["dEtaHdt"],
                           s["wVel"])                                           # :965
    rs = calc_r_star(p, g, ex, ic["etaH"], s["rStarFacC"], s["rStarFacW"], s["rStarFacS"])  # :980
    out["rstar_checks"] = {k: rs.pop(k) for k in ("icntc1", "icntw", "icnts", "icntc2", "maxhFacC")}
    out.update(rs)
    uVel, vVel, wVel = do_stagger_fields_exchanges(ex, uVel, vVel, ic["wVel"])  # :1015
    out.update(dEtaHdt=ic["dEtaHdt"], PmEpR=ic["PmEpR"], etaN=ic["etaN"], etaH=ic["etaH"], etaHnm1=ic["etaHnm1"],
               uVel=uVel, vVel=vVel, wVel=wVel)
    return out
