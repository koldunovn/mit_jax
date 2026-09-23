"""Hydrostatic pressure (plan Task 14b-phi_hyd): CALC_PHI_HYD for all tiles and levels, V4r4 flux-forced branches.

Literal port of the code DYNAMICS runs for every level k (dynamics.F:481 `CALL CALC_PHI_HYD`):
  calc_phi_hyd.F       ocean (buoyancyRelation='OCEANIC'), density from `rhoInSitu` (DO_OCEANIC_PHYS, :174),
                       finite-difference form (integr_GeoPot=2) with uniformFreeSurfLev (:267-282),
  calc_grad_phi_hyd.F  r* (select_rStar=2, nonlinFreeSurf=4): gradient of phiHydC*rStarFacC + phi0surf (:83-88,
                       :143-160) plus the z* slope term (:164-189), masked (:255-260),
  diags_phi_rlow.F     phiHydLow (bottom pressure) at kLowC (:67-108) and its r* rescaling at k=Nr (:162-171),
  diags_phi_hyd.F      totPhiHyd (:101-111).
The k recursion (phiHydF carried from interface k to k+1) is a `lax.scan` in the Fortran order; everything else is
independent across k and vectorised. Loop range of every computation: DYNAMICS' iMin=0, iMax=sNx+1, jMin=0,
jMax=sNy+1 (dynamics.F:190-191); points outside keep what the Fortran arrays held (0 for the DYNAMICS locals
phiHydC/phiHydF/dPhiHydX/Y and for phiHydLow, which DYNAMICS zeroes per tile under ALLOW_AUTODIFF, dynamics.F:328;
the previous values for the global totPhiHyd).

Density: V4r4 runs with implicitIntGravWave=F and myIter>=0, so CALC_PHI_HYD takes rhoInSitu (calc_phi_hyd.F:171-176,
computed by FIND_RHO_2D in DO_OCEANIC_PHYS) and never calls the EOS itself; no EOS function is needed here. The branch
calc_phi_hyd.F:156-170 (implicitIntGravWave or myIter<0: FIND_RHO_2D on theta/salt) raises NotImplementedError; it
would take `eos.find_rho_2d` in place of the rhoInSitu argument.

Vertical factors gravFacC/F, recip_deepFacC, recip_rhoFacC are 1 in V4r4 (load_ref_files.F:161-166 with
gravityFile=' ', set_grid_factors.F:52-55 with deepAtmosphere=F, set_ref_state.F:74-80 with rhoRefFile=' '); they are
kept as named factors so every expression has the Fortran operand order.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.core.implicit import vertical_factors
from mitgcm_jax.params_io import params_pytree


@params_pytree
@dataclass(frozen=True)
class PhiHydParams:
    gravity: float          # data PARM01 gravity (default set_defaults.F:104 gravity = 9.81)
    recip_rhoConst: float   # ini_parms.F:640 recip_rhoConst = 1/rhoConst (rhoConst: data, default rhoNil, :445)
    select_rStar: int       # data PARM01 select_rStar
    nonlinFreeSurf: int     # data PARM01 nonlinFreeSurf
    integr_GeoPot: int = 2  # set_defaults.F:276
    uniformFreeSurfLev: bool = True  # set_parms.F:130 usingZCoords .AND. .NOT.useShelfIce

    @classmethod
    def from_namelists(cls, nml):
        g = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
        _check_supported(nml)
        rhoNil = float(g("rhoNil", 999.8))                   # set_defaults.F:106 rhoNil = 999.8
        rhoConst = float(g("rhoConst", rhoNil))              # ini_parms.F:445 IF (rhoConst.EQ.UNSET_RL) rhoConst=rhoNil
        integr = int(nml.get("data", "parm01", "integr_GeoPot", default=2))  # set_defaults.F:276
        if integr != 2:
            raise NotImplementedError(f"integr_GeoPot={integr}: only the finite-difference form 2 is ported")
        sel, nlfs = int(g("select_rStar", 0)), int(g("nonlinFreeSurf", 0))  # set_defaults.F: 0, 0
        if not (sel >= 2 and nlfs >= 4):
            raise NotImplementedError(f"select_rStar={sel}, nonlinFreeSurf={nlfs}: only the r* branches "
                                      "select_rStar>=2, nonlinFreeSurf>=4 (calc_grad_phi_hyd.F, diags_phi_*.F)")
        return cls(gravity=float(g("gravity", 9.81)), recip_rhoConst=1.0 / rhoConst,  # ini_parms.F:640
                   select_rStar=sel, nonlinFreeSurf=nlfs, integr_GeoPot=integr, uniformFreeSurfLev=True)


def _check_supported(nml):
    """Hard errors for the options whose branches are not ported (V4r4 values in brackets)."""
    g = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
    # defaults: set_defaults.F:175 buoyancyRelation='OCEANIC', :180 implicitIntGravWave=F, :211 quasiHydrostatic=F,
    # :210 nonHydrostatic=F, :51 selectSigmaCoord=0, :73 deepAtmosphere=F, :62 gravityFile=' ', :61 rhoRefFile=' ',
    # :188 momPressureForcing=T; useShelfIce default F (packages_boot.F)
    checks = [
        (str(g("buoyancyRelation", "OCEANIC")).strip().upper() == "OCEANIC", "buoyancyRelation ['OCEANIC']"),
        (not g("implicitIntGravWave", False), "implicitIntGravWave [F] (calc_phi_hyd.F:156 FIND_RHO_2D branch)"),
        (not g("quasiHydrostatic", False), "quasiHydrostatic [F] (calc_phi_hyd.F:202)"),
        (not g("nonHydrostatic", False), "nonHydrostatic [F]"),
        (int(nml.get("data", "parm04", "selectSigmaCoord", default=0)) == 0, "selectSigmaCoord [0]"),
        (not nml.get("data", "parm04", "deepAtmosphere", default=False), "deepAtmosphere [F]"),
        (str(g("gravityFile", " ")).strip() == "", "gravityFile [' '] (gravFacC/F = 1)"),
        (str(g("rhoRefFile", " ")).strip() == "", "rhoRefFile [' '] (rhoFac = 1)"),
        (bool(g("momPressureForcing", True)), "momPressureForcing [T]"),
        (not nml.get("data.pkg", "packages", "useShelfIce", default=False), "useShelfIce [F] (uniformFreeSurfLev)"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise NotImplementedError("CALC_PHI_HYD branch not ported: " + "; ".join(bad))


def kLowC_from_hFac(h0FacC):
    """ini_masks_etc.F:192,200: kLowC = deepest k with hFacC(k) .NE. 0 (0 for a dry column), from the initial hFacC
    (= h0FacC under r*). h0FacC: [T, Nr, ny, nx]. Returns int32 [T, ny, nx]."""
    h = np.asarray(h0FacC)
    k = np.arange(1, h.shape[1] + 1)[None, :, None, None]
    return np.max(np.where(h != 0.0, k, 0), axis=1).astype(np.int32)


def _find_rho_2d_not_ported(*args, **kw):
    raise NotImplementedError("FIND_RHO_2D inside CALC_PHI_HYD (calc_phi_hyd.F:165) is not executed by V4r4")


def calc_phi_hyd(p, g, kLowC, rhoInSitu, rStarFacC, etaH, phi0surf, totPhiHyd, *, myIter=0):
    """CALC_PHI_HYD for k=1..Nr on all tiles (dynamics.F:421-486 k loop, the phi_hyd part).

    Inputs (Fortran names, [T,(Nr,)ny,nx] with halos): rhoInSitu [T,Nr] (DO_OCEANIC_PHYS), rStarFacC, etaH, phi0surf
    [T] (SURFACE.h / FFIELDS.h at DYNAMICS time), totPhiHyd [T,Nr] (previous values: kept outside the loop range),
    kLowC [T] (`kLowC_from_hFac(g.h0FacC)`); grid: drC, rC, rF, recip_dxC, recip_dyC, recip_Rcol, Ro_surf, R_low,
    maskW, maskS.
    Returns dict of [T,Nr,ny,nx]: phiHydC, phiHydF (the value after level k, i.e. at interface k+1, as dumped),
    dPhiHydX, dPhiHydY, totPhiHyd; and phiHydLow [T,ny,nx].
    """
    if isinstance(myIter, int) and myIter < 0:  # calc_phi_hyd.F:156: (implicitIntGravWave .OR. myIter.LT.0)
        _find_rho_2d_not_ported()
    L = g.layout
    Nr = L.Nr
    vf = vertical_factors(Nr)
    iMin, iMax, jMin, jMax = 0, L.sNx + 1, 0, L.sNy + 1  # dynamics.F:190-191
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    gravity, recip_rhoConst = p.gravity, p.recip_rhoConst
    halfRL, zeroRL = 0.5, 0.0  # EEPARAMS.h halfRL = 0.5D0, zeroRL = 0.0D0
    drC, rC, rF = jnp.asarray(g.drC), jnp.asarray(g.rC), jnp.asarray(g.rF)
    gravFacF = jnp.asarray(vf["gravFacF"])

    # calc_phi_hyd.F:267-273 (per level; Fortran k -> index k-1)
    dRlocM = halfRL * drC[:Nr] * gravFacF[:Nr]                                   # :267
    dRlocM = dRlocM.at[0].set((rF[0] - rC[0]) * gravFacF[0])                     # :268 k=1
    dRlocP = halfRL * drC[1:Nr + 1] * gravFacF[1:Nr + 1]                         # :272
    dRlocP = dRlocP.at[Nr - 1].set((rC[Nr - 1] - rF[Nr]) * gravFacF[Nr])         # :270 k=Nr
    # diags_phi_rlow.F:91-96 (integr_GeoPot=2)
    ratioRm = jnp.ones(Nr).at[1:].set(halfRL * drC[1:Nr] / (rF[1:Nr] - rC[1:Nr]))          # :91, :93 k>1
    ratioRp = jnp.ones(Nr).at[:Nr - 1].set(halfRL * drC[1:Nr] / (rC[:Nr - 1] - rF[1:Nr]))  # :92, :94 k<Nr
    ratioRm = ratioRm * gravFacF[:Nr]                                            # :95
    ratioRp = ratioRp * gravFacF[1:Nr + 1]                                       # :96

    alpha = jnp.moveaxis(jnp.asarray(rhoInSitu)[:, :, J, I], 1, 0)  # calc_phi_hyd.F:174 alphaRho = rhoInSitu
    kLow = jnp.asarray(kLowC)[:, J, I]
    rlow_dd = rC[:, None, None, None] - jnp.asarray(g.R_low)[None, :, J, I]  # diags_phi_rlow.F:101 rC(k)-R_low

    def level(carry, x):
        phiHydF, phiHydLow = carry
        k, a, dM, dP, rM, rP, dd = x
        # calc_phi_hyd.F:277-280 (uniformFreeSurfLev)
        phiHydC = phiHydF + dM * gravity * a * recip_rhoConst
        phiHydF = phiHydC + dP * gravity * a * recip_rhoConst
        # diags_phi_rlow.F:100-105
        low = phiHydC + (jnp.minimum(zeroRL, dd) * rM + jnp.maximum(zeroRL, dd) * rP) * gravity * a * recip_rhoConst
        phiHydLow = jnp.where(k == kLow, low, phiHydLow)
        return (phiHydF, phiHydLow), (phiHydC, phiHydF)

    z = jnp.zeros_like(alpha[0])
    ks = jnp.arange(1, Nr + 1, dtype=jnp.int32)
    # phiHydF = 0 at k=1 (calc_phi_hyd.F:139-145); phiHydLow = 0 at k=1 (diags_phi_rlow.F:67-73)
    (_, low), (pC, pF) = jax.lax.scan(level, (z, z), (ks, alpha, dRlocM, dRlocP, ratioRm, ratioRp, rlow_dd))

    shape3 = (L.nTiles, Nr, L.ny, L.nx)
    phiHydC = jnp.zeros(shape3).at[:, :, J, I].set(jnp.moveaxis(pC, 0, 1))
    phiHydF = jnp.zeros(shape3).at[:, :, J, I].set(jnp.moveaxis(pF, 0, 1))
    alphRho = jnp.zeros(shape3).at[:, :, J, I].set(jnp.moveaxis(alpha, 0, 1))

    rSF = jnp.asarray(rStarFacC)[:, None]
    p0s = jnp.asarray(phi0surf)[:, None]
    etaH3 = jnp.asarray(etaH)[:, None]
    rdfc = jnp.asarray(vf["recip_deepFacC"])[None, :, None, None]
    rrfc = jnp.asarray(vf["recip_rhoFacC"])[None, :, None, None]
    rCk = rC[None, :, None, None]

    # --- calc_grad_phi_hyd.F (select_rStar>=2, nonlinFreeSurf>=4, ocean) ---
    varLoc = jnp.zeros(shape3).at[:, :, J, I].set(phiHydC[:, :, J, I] * rSF[..., J, I] + p0s[..., J, I])  # :85-86
    Jx, Ix, Ixm = J, L.is_(iMin + 1, iMax), L.is_(iMin, iMax - 1)              # :149-150
    Jy, Iy, Jym = L.js(jMin + 1, jMax), I, L.js(jMin, jMax - 1)                # :155-156
    rdxC = jnp.asarray(g.recip_dxC)[:, None]
    rdyC = jnp.asarray(g.recip_dyC)[:, None]
    dPhiHydX = jnp.zeros(shape3).at[..., Jx, Ix].set(                          # :145, :151-152
        rdxC[..., Jx, Ix] * rdfc * (varLoc[..., Jx, Ix] - varLoc[..., Jx, Ixm]) * rrfc)
    dPhiHydY = jnp.zeros(shape3).at[..., Jy, Iy].set(                          # :146, :157-158
        rdyC[..., Jy, Iy] * rdfc * (varLoc[..., Jy, Iy] - varLoc[..., Jym, Iy]) * rrfc)
    # z* slope term (:164-189): fluidIsWater .AND. usingZCoords
    factorZ = gravity * recip_rhoConst * rrfc * 0.5                            # :167
    rRcol = jnp.asarray(g.recip_Rcol)[:, None]
    varLoc = jnp.zeros(shape3).at[:, :, J, I].set(etaH3[..., J, I] * (1.0 + rCk * rRcol[..., J, I]))  # :170-171
    dPhiHydX = dPhiHydX.at[..., Jx, Ix].set(                                   # :176-179
        dPhiHydX[..., Jx, Ix] + factorZ * (alphRho[..., Jx, Ixm] + alphRho[..., Jx, Ix])
        * (varLoc[..., Jx, Ix] - varLoc[..., Jx, Ixm]) * rdxC[..., Jx, Ix] * rdfc)
    dPhiHydY = dPhiHydY.at[..., Jy, Iy].set(                                   # :184-187
        dPhiHydY[..., Jy, Iy] + factorZ * (alphRho[..., Jym, Iy] + alphRho[..., Jy, Iy])
        * (varLoc[..., Jy, Iy] - varLoc[..., Jym, Iy]) * rdyC[..., Jy, Iy] * rdfc)
    dPhiHydX = dPhiHydX * jnp.asarray(g.maskW)                                 # :257
    dPhiHydY = dPhiHydY * jnp.asarray(g.maskS)                                 # :258

    # --- diags_phi_rlow.F:162-171 (k=Nr, r* ocean): rescale phiHydLow ---
    rS2 = jnp.asarray(rStarFacC)[:, J, I]
    dPhiRef = (jnp.asarray(g.Ro_surf)[:, J, I] - jnp.asarray(g.R_low)[:, J, I]) * gravity           # :164-165
    low = low * rS2 + dPhiRef * (rS2 - 1.0) + jnp.asarray(phi0surf)[:, J, I]                       # :166-169
    phiHydLow = jnp.zeros((L.nTiles, L.ny, L.nx)).at[:, J, I].set(low)

    # --- diags_phi_hyd.F:101-111 (r* ocean). The first assignment (:56-58, phiHydC + Bo_surf*etaN + phi0surf) is
    # overwritten at every point of the same range (select_rStar>=1 .AND. nonlinFreeSurf>=4, :67), so it is omitted.
    dPhiRefk = (jnp.asarray(g.Ro_surf)[:, None, J, I] - rCk) * gravity                                # :103
    tot = (phiHydC[..., J, I] * rSF[..., J, I] + jnp.maximum(dPhiRefk, 0.0) * (rSF[..., J, I] - 1.0)
           + p0s[..., J, I])                                                                         # :104-108
    totPhiHyd = jnp.asarray(totPhiHyd).at[..., J, I].set(tot)

    return dict(phiHydC=phiHydC, phiHydF=phiHydF, dPhiHydX=dPhiHydX, dPhiHydY=dPhiHydY, totPhiHyd=totPhiHyd,
                phiHydLow=phiHydLow)
