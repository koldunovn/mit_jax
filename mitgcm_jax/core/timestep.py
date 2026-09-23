"""Momentum time step (plan Task 14c): TIMESTEP with APPLY_FORCING_U/V and ADAMS_BASHFORTH3, all tiles and levels.

Literal port of MITgcm_c66g/model/src/timestep.F in the branches the V4r4 flux-forced build executes:
ALLOW_ADAMSBASHFORTH_3 (adams_bashforth3.F, tendency form kArg>0), staggerTimeStep=T (grad phi_hyd added after AB,
timestep.F:116-128 skipped), momDissip_In_AB=F and momForcingOutAB=1 (forcing and dissipation outside AB,
:210-228), vectorInvariantMomentum=T (no r* rescale, :276-320), no CD scheme, no non-hydrostatic code. APPLY_FORCING_U/V
is the flux-forced override `code/apply_forcing.F` (identical to c66g apart from the Um_Wind/Vm_Wind diagnostics:
surface stress at k=kSurface=1, :139-152 / :341-354). TIMESTEP is called per level k from DYNAMICS
(dynamics.F:567); levels are independent, so all k are done at once.

AB3 start-up (adams_bashforth3.F:83-100): weights depend on myIter, nIter0 and mom_StartAB. mom_StartAB is nIter0
(ini_model_io.F:122, check_pickup.F:66) unless the pickup lacks GuNm1/GvNm1 (0, check_pickup.F:177) or
GuNm2/GvNm2 (MIN(.,1), :180). Note the c66g quirk: the test is `startAB.EQ.1`, so a run with nIter0=1 and a
complete pickup (mom_StartAB=1) takes AB2 weights at its first step, AB3 afterwards; nIter0=0 takes forward Euler,
then AB2, then AB3.
History slots: gTrNm(:,:,:,:,:,m) is stored as gTrNm[m-1] with a leading axis of 2 (m1 = 1+MOD(myIter+1,2),
m2 = 1+MOD(myIter,2): the slot of the current tendency alternates between steps).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mitgcm_jax.core.implicit import check_unit_vertical_factors, vertical_factors
from mitgcm_jax.parallel.tiles import n_tiles
from mitgcm_jax.params_io import params_pytree


@params_pytree
@dataclass(frozen=True)
class TimestepParams:
    deltaTMom: float        # data PARM03 deltaTMom (ini_parms.F:898: 0 -> deltaT)
    alph_AB: float          # data PARM03 alph_AB (set_defaults.F:312 0.5)
    beta_AB: float          # data PARM03 beta_AB (set_defaults.F:313 5/12)
    nIter0: int             # data PARM03 nIter0
    mom_StartAB: int        # RESTART.h (see module docstring)
    foFacMom: float = 1.0   # set_parms.F:175 (momForcing=T)
    pfFacMom: float = 1.0   # set_parms.F:187 (momPressureForcing=T)
    implicSurfPress: float = 1.0  # set_defaults.F:247
    momForcing: bool = True       # set_defaults.F:186
    momViscosity: bool = True     # set_defaults.F:184
    momDissip_In_AB: bool = False
    momForcingOutAB: int = 1

    @classmethod
    def from_namelists(cls, nml, mom_StartAB=None):
        """mom_StartAB: None -> nIter0 (complete pickup, or nIter0=0 without pickup); pass 0 / 1 for a pickup
        without GuNm1 / GuNm2 (check_pickup.F:175-180)."""
        p1 = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
        p3 = lambda k, d: nml.get("data", "parm03", k, default=d)  # noqa: E731
        check_unit_vertical_factors(nml)
        # ini_parms.F:884-898 deltaTMom (defaults 0: set_defaults.F:293-297)
        deltaT = float(p3("deltaT", 0.0))
        for k in ("deltaTClock", "deltaTtracer", "deltaTMom", "deltaTFreeSurf"):
            if deltaT == 0.0:
                deltaT = float(p3(k, 0.0))
        deltaTMom = float(p3("deltaTMom", 0.0)) or deltaT
        # ini_parms.F:827 forcing_In_AB = .TRUE. ; :936-938 momForcingOutAB = 1, 0 if forcing_In_AB
        momForcingOutAB = int(p3("momForcingOutAB", 0 if p3("forcing_In_AB", True) else 1))
        momDissip_In_AB = bool(p3("momDissip_In_AB", True))  # set_defaults.F:308
        nIter0 = int(p3("nIter0", 0))
        if p3("startFromPickupAB2", False):  # set_defaults.F:314 (ALLOW_ADAMSBASHFORTH_3)
            raise NotImplementedError("startFromPickupAB2 (ini_model_io.F:124)")
        bad = []
        if not p1("staggerTimeStep", False):          # set_defaults.F:181
            bad.append("staggerTimeStep=F (timestep.F:116 synchronous grad phi_hyd)")
        if p1("implicitIntGravWave", False):
            bad.append("implicitIntGravWave")
        if not p1("vectorInvariantMomentum", False):  # set_defaults.F:190
            bad.append("vectorInvariantMomentum=F (timestep.F:277 r* rescale, MOM_FLUXFORM)")
        if p1("useCDscheme", False):                  # set_defaults.F:225
            bad.append("useCDscheme")
        if p1("nonHydrostatic", False):
            bad.append("nonHydrostatic")
        if float(p1("implicSurfPress", 1.0)) != 1.0:  # set_defaults.F:247
            bad.append("implicSurfPress != 1 (dynamics.F:349 CALC_GRAD_PHI_SURF)")
        if momDissip_In_AB:
            bad.append("momDissip_In_AB=T (timestep.F:131)")
        if momForcingOutAB != 1:
            bad.append("momForcingOutAB=0 (timestep.F:141)")
        if bad:
            raise NotImplementedError("TIMESTEP branch not ported: " + "; ".join(bad))
        momForcing = bool(p1("momForcing", True))
        return cls(deltaTMom=deltaTMom, alph_AB=float(p3("alph_AB", 0.5)), beta_AB=float(p3("beta_AB", 5.0 / 12.0)),
                   nIter0=nIter0, mom_StartAB=nIter0 if mom_StartAB is None else int(mom_StartAB),
                   foFacMom=1.0 if momForcing else 0.0,                                   # set_parms.F:174-178
                   pfFacMom=1.0 if p1("momPressureForcing", True) else 0.0,               # set_parms.F:186-190
                   implicSurfPress=float(p1("implicSurfPress", 1.0)), momForcing=momForcing,
                   momViscosity=bool(p1("momViscosity", True)), momDissip_In_AB=momDissip_In_AB,
                   momForcingOutAB=momForcingOutAB)


def ab3_weights(myIter, nIter0, startAB, alph_AB, beta_AB):
    """adams_bashforth3.F:87-100 (ab0, ab1, ab2). myIter may be a Python int or a traced integer."""
    if isinstance(myIter, (int, np.integer)):
        if myIter == nIter0 and startAB == 0:
            return 0.0, 0.0, 0.0                                         # :88-90
        if (myIter == nIter0 and startAB == 1) or (myIter == 1 + nIter0 and startAB == 0):
            return alph_AB, -alph_AB, 0.0                                # :93-95
        return alph_AB + beta_AB, -alph_AB - 2. * beta_AB, beta_AB       # :97-99
    c0 = (myIter == nIter0) & (startAB == 0)
    c1 = ((myIter == nIter0) & (startAB == 1)) | ((myIter == 1 + nIter0) & (startAB == 0))
    pick = lambda w0, w1, w2: jnp.where(c0, w0, jnp.where(c1, w1, w2))  # noqa: E731
    return (pick(0.0, alph_AB, alph_AB + beta_AB), pick(0.0, -alph_AB, -alph_AB - 2. * beta_AB),
            pick(0.0, 0.0, beta_AB))


def adams_bashforth3(gTracer, gTrNm, myIter, nIter0, startAB, alph_AB, beta_AB, kArg=1):
    """ADAMS_BASHFORTH3 (adams_bashforth3.F) on whole arrays (all points, all levels).

    gTracer [..]; gTrNm [2, ..] (slot m at index m-1). kArg>0 (tendency, :117-127): returns
    (gTracer + AB_gTr, gTrNm with slot m2 := gTracer, AB_gTr). kArg=0 (state, :104-115): returns
    (gTracer, gTrNm with slot m2 := gTracer + AB_gTr, AB_gTr).
    """
    m1 = (myIter + 1) % 2   # :83 m1 = 1 + MOD(myIter+1,2)  (0-based slot)
    m2 = myIter % 2         # :84 m2 = 1 + MOD(myIter,2)
    ab0, ab1, ab2 = ab3_weights(myIter, nIter0, startAB, alph_AB, beta_AB)
    gTrNm = jnp.asarray(gTrNm)
    AB_gTr = ab0 * gTracer + ab1 * gTrNm[m1] + ab2 * gTrNm[m2]   # :109-111 / :121-123
    if kArg == 0:
        return gTracer, gTrNm.at[m2].set(gTracer + AB_gTr), AB_gTr  # :112
    return gTracer + AB_gTr, gTrNm.at[m2].set(gTracer), AB_gTr      # :124-125


def apply_forcing_uv(p, g, surfaceForcingU, surfaceForcingV, recip_hFacW, recip_hFacS):
    """APPLY_FORCING_U/V (flux-forced code/apply_forcing.F) into zeroed guExt/gvExt (timestep.F:91-114), all k.
    Ocean z-coordinates: kSurface = 1 (:108-114); surface stress on j=0..sNy+1, i=1..sNx+1 (U, :142-151) and
    j=1..sNy+1, i=0..sNx+1 (V, :342-353). No AIM/ATM_PHYS/FIZHI/EDDYPSI/RBCS/OBCS/MYPACKAGE in V4r4 ff."""
    L = g.layout
    shape3 = (n_tiles(g), L.Nr, L.ny, L.nx)
    guExt = jnp.zeros(shape3)
    gvExt = jnp.zeros(shape3)
    if not p.momForcing:
        return guExt, gvExt
    k = 1
    rdrF = jnp.asarray(g.recip_drF)[k - 1]
    Ju, Iu = L.js(0, L.sNy + 1), L.is_(1, L.sNx + 1)
    Jv, Iv = L.js(1, L.sNy + 1), L.is_(0, L.sNx + 1)
    sFU, sFV = jnp.asarray(surfaceForcingU), jnp.asarray(surfaceForcingV)
    rhW, rhS = jnp.asarray(recip_hFacW), jnp.asarray(recip_hFacS)
    guExt = guExt.at[:, k - 1, Ju, Iu].set(
        guExt[:, k - 1, Ju, Iu] + p.foFacMom * sFU[:, Ju, Iu] * rdrF * rhW[:, k - 1, Ju, Iu])   # :144-146
    gvExt = gvExt.at[:, k - 1, Jv, Iv].set(
        gvExt[:, k - 1, Jv, Iv] + p.foFacMom * sFV[:, Jv, Iv] * rdrF * rhS[:, k - 1, Jv, Iv])   # :346-348
    return guExt, gvExt


def timestep(p, g, myIter, gU, gV, guNm, gvNm, uVel, vVel, dPhiHydX, dPhiHydY, phiSurfX, phiSurfY,
             guDissip, gvDissip, guExt, gvExt):
    """TIMESTEP (timestep.F) for all levels: AB3 on the tendencies gU/gV (as left by MOM_VECINV), then
    gU := uVel + deltaTMom*(gU_AB + forcing + dissipation - psFac*phiSurfX - phxFac*dPhiHydX)*maskW on
    iMin..iMax=0..sNx+1, jMin..jMax=0..sNy+1 (dynamics.F:190-191); the halo points outside keep the AB-extrapolated
    tendency, as in Fortran. All fields [T,Nr,ny,nx] except guNm/gvNm [2,T,Nr,ny,nx] (slot m-1).
    Returns (gU, gV, guNm, gvNm)."""
    L = g.layout
    vf = vertical_factors(L.Nr)
    J, I = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
    col = lambda v: jnp.asarray(v)[None, :, None, None]  # noqa: E731
    # timestep.F:81-86
    psFac = p.pfFacMom * (1. - p.implicSurfPress) * col(vf["recip_deepFacC"]) * col(vf["recip_rhoFacC"])
    phxFac = p.pfFacMom
    phyFac = p.pfFacMom
    # timestep.F:165-174 ADAMS_BASHFORTH3 (momDissip_In_AB=F, momForcingOutAB=1: nothing added before AB)
    gU, guNm, _ = adams_bashforth3(jnp.asarray(gU), guNm, myIter, p.nIter0, p.mom_StartAB, p.alph_AB, p.beta_AB)
    gV, gvNm, _ = adams_bashforth3(jnp.asarray(gV), gvNm, myIter, p.nIter0, p.mom_StartAB, p.alph_AB, p.beta_AB)
    # timestep.F:200-205 gUtmp = gU on the loop range (0 elsewhere, :91-102)
    gUtmp = gU[..., J, I]
    gVtmp = gV[..., J, I]
    if p.momForcing and p.momForcingOutAB == 1:   # :211-218
        gUtmp = gUtmp + jnp.asarray(guExt)[..., J, I]
        gVtmp = gVtmp + jnp.asarray(gvExt)[..., J, I]
    if p.momViscosity and not p.momDissip_In_AB:  # :221-228
        gUtmp = gUtmp + jnp.asarray(guDissip)[..., J, I]
        gVtmp = gVtmp + jnp.asarray(gvDissip)[..., J, I]
    # :358-379
    gU = gU.at[..., J, I].set(jnp.asarray(uVel)[..., J, I] + p.deltaTMom * (
        gUtmp - psFac * jnp.asarray(phiSurfX)[:, None, J, I] - phxFac * jnp.asarray(dPhiHydX)[..., J, I]
    ) * jnp.asarray(g.maskW)[..., J, I])
    gV = gV.at[..., J, I].set(jnp.asarray(vVel)[..., J, I] + p.deltaTMom * (
        gVtmp - psFac * jnp.asarray(phiSurfY)[:, None, J, I] - phyFac * jnp.asarray(dPhiHydY)[..., J, I]
    ) * jnp.asarray(g.maskS)[..., J, I])
    return gU, gV, guNm, gvNm
