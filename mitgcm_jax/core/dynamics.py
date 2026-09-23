"""DYNAMICS driver (plan Tasks 14b/14c): model/src/dynamics.F (c66g; the flux-forced tree uses it unchanged) for all
tiles, V4r4 flux-forced branches, with MOM_VECINV supplied by the caller.

Sequence (dynamics.F, preprocessed build `bld/dynamics.f` checked for the active branches):
  :300-310  gU = gV = 0 on the whole tile (ALLOW_AUTODIFF re-initialisation)
  :311-338  phiHydF/phiHydC/phiSurfX/Y/guDissip/gvDissip = 0, phiHydLow = 0 (ALLOW_AUTODIFF)
  :349      implicSurfPress = 1: CALC_GRAD_PHI_SURF not called, phiSurfX/Y stay 0
  :369-376  kappaRU = kappaRV = 0 for k=1..Nr+1 (ALLOW_AUTODIFF: unconditional)
  :382-387  CALC_VISCOSITY (momViscosity): viscArNr + GGL90 (calc_viscosity.F, ggl90_calc_visc.F)
  :421-576  k loop: CALC_PHI_HYD (core/phi_hyd.py); MOM_VECINV (caller); TIMESTEP (core/timestep.py)
  :591-609  ALLOW_AUTODIFF: the `#if ... !(defined ALLOW_AUTODIFF)` guard (:579-580) removes the MOM_U/V_IMPLICIT_R
            branch, so implicitViscosity calls IMPLDIFF(-1, kappaRU, recip_hFacW, gU) and (-2, kappaRV, recip_hFacS, gV)
No exchange, OBCS, SMAG_3D, CD scheme, non-hydrostatic code or EP forcing is compiled or active in V4r4 ff.
The k loop is split into its independent parts (phi_hyd scan over k, then MOM_VECINV and TIMESTEP for all k):
levels only couple through phiHydF (inside calc_phi_hyd) and fVerU/V (inside MOM_VECINV), so the results are the
Fortran's.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mitgcm_jax.core.implicit import impldiff, impldiff_deltaTX
from mitgcm_jax.core.phi_hyd import PhiHydParams, calc_phi_hyd
from mitgcm_jax.core.timestep import TimestepParams, apply_forcing_uv, timestep
from mitgcm_jax.parallel.tiles import n_tiles


@dataclass(frozen=True)
class DynamicsParams:
    phi: PhiHydParams
    ts: TimestepParams
    viscArNr: tuple            # ini_parms.F:505-508 viscArNr(k) = viscAr (data PARM01 viscAr)
    useGGL90: bool             # data.pkg useGGL90
    implicitViscosity: bool    # data PARM01 (set_defaults.F:205 F)
    momViscosity: bool = True  # set_defaults.F:184
    momStepping: bool = True   # set_defaults.F:189

    @classmethod
    def from_namelists(cls, nml, Nr=50, mom_StartAB=None):
        p1 = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
        # ini_parms.F:484-508: viscAr from viscAr/viscAz/viscAp, else viscArDefault; viscArNr(k) = viscAr unless set
        if nml.has("data", "parm01", "viscArNr"):
            viscArNr = tuple(float(v) for v in nml.get("data", "parm01", "viscArNr", array=True))
            if len(viscArNr) != Nr:
                raise ValueError("partial viscArNr (ini_parms.F:490)")
        else:
            viscAr = p1("viscAr", None) if nml.has("data", "parm01", "viscAr") else \
                p1("viscAz", None) if nml.has("data", "parm01", "viscAz") else None
            if viscAr is None:
                raise NotImplementedError("viscAr unset: viscArDefault path (ini_parms.F:499) not ported")
            viscArNr = (float(viscAr),) * Nr
        bad = []
        for key, default, why in [("momImplVertAdv", False, "set_defaults.F:207"),
                                  ("selectImplicitDrag", 0, "set_defaults.F:206"),
                                  ("useSmag3D", False, "set_defaults.F:200")]:
            if p1(key, default):
                bad.append(f"{key} ({why})")
        for pkg in ("useKPP", "usePP81", "useKL10", "useMY82", "useOBCS"):
            if nml.get("data.pkg", "packages", pkg, default=False):
                bad.append(pkg + " (calc_viscosity.F / dynamics.F)")
        if bad:
            raise NotImplementedError("DYNAMICS branch not ported: " + "; ".join(bad))
        return cls(phi=PhiHydParams.from_namelists(nml),
                   ts=TimestepParams.from_namelists(nml, mom_StartAB=mom_StartAB),
                   viscArNr=viscArNr,
                   useGGL90=bool(nml.get("data.pkg", "packages", "useGGL90", default=False)),
                   implicitViscosity=bool(p1("implicitViscosity", False)),
                   momViscosity=bool(p1("momViscosity", True)),
                   momStepping=bool(p1("momStepping", True)))


# Parameters as a pytree (KERNEL_GUIDE: pass as a jit argument): the nested parameter sets and the viscArNr floats are
# leaves (traced), the flags static.
jax.tree_util.register_dataclass(DynamicsParams, data_fields=["phi", "ts", "viscArNr"],
                                 meta_fields=["useGGL90", "implicitViscosity", "momViscosity", "momStepping"])


def calc_viscosity(p, g, GGL90viscArU, GGL90viscArV):
    """kappaRU, kappaRV [T,Nr+1,ny,nx] as DYNAMICS leaves them before the k loop: zeroed (dynamics.F:369-376), then
    CALC_VISCOSITY (dynamics.F:382-387, calc_viscosity.F:51-118) with GGL90_CALC_VISC (ggl90_calc_visc.F:47-59) on
    iMin..iMax=0..sNx+1, jMin..jMax=0..sNy+1. Note the Fortran asymmetry: kappaRU gets the GGL90 increment unmasked,
    kappaRV masked by maskS (ggl90_calc_visc.F:49-50 vs :56-57)."""
    L = g.layout
    Nr = L.Nr
    shape = (n_tiles(g), Nr + 1, L.ny, L.nx)
    kappaRU = jnp.zeros(shape)
    kappaRV = jnp.zeros(shape)
    if not p.momViscosity:
        return kappaRU, kappaRV
    visc = jnp.stack([jnp.asarray(p.viscArNr[min(k, Nr) - 1]) for k in range(1, Nr + 2)])[None, :, None, None]
    kappaRU = jnp.broadcast_to(visc, shape)                          # calc_viscosity.F:56 (ki = MIN(k,Nr))
    kappaRV = jnp.broadcast_to(visc, shape)                          # calc_viscosity.F:57
    if p.useGGL90:
        J, I = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
        vk = jnp.stack([jnp.asarray(v) for v in p.viscArNr])[None, :, None, None]
        kappaRU = kappaRU.at[:, :Nr, J, I].set(
            kappaRU[:, :Nr, J, I] + (jnp.asarray(GGL90viscArU)[..., J, I] - vk))                       # :49-50
        kappaRV = kappaRV.at[:, :Nr, J, I].set(
            kappaRV[:, :Nr, J, I] + jnp.asarray(g.maskS)[..., J, I]
            * (jnp.asarray(GGL90viscArV)[..., J, I] - vk))                                               # :56-57
        # calc_viscosity.F:106-115 k = Nr+1 copies level Nr (usePP81/KL10/MY82/GGL90)
        kappaRU = kappaRU.at[:, Nr].set(kappaRU[:, Nr - 1])
        kappaRV = kappaRV.at[:, Nr].set(kappaRV[:, Nr - 1])
    return kappaRU, kappaRV


def dynamics(p, g, kLowC, s, mom_vecinv, myIter):
    """DYNAMICS on all tiles.

    s: dict of the fields DYNAMICS reads (Fortran names, [T,(Nr,)ny,nx] with halos, values at DYNAMICS entry):
      uVel, vVel                    DYNVARS.h (time n)
      guNm, gvNm                    [2,T,Nr,ny,nx] AB histories (slot m at index m-1)
      etaH, rStarFacC               SURFACE.h
      recip_hFacW, recip_hFacS      GRID.h as set by UPDATE_R_STAR at the start of the step
      rhoInSitu, totPhiHyd          DYNVARS.h (rhoInSitu from DO_OCEANIC_PHYS; totPhiHyd: previous values)
      phi0surf, surfaceForcingU/V   FFIELDS.h
      GGL90viscArU, GGL90viscArV    GGL90.h
    kLowC: phi_hyd.kLowC_from_hFac(g.h0FacC).
    mom_vecinv(kappaRU, kappaRV) -> (gU, gV, guDissip, gvDissip), each [T,Nr,ny,nx]: the MOM_VECINV calls of all
      levels (dynamics.F:536-542) with every other input bound by the caller; gU/gV are the whole arrays as MOM_VECINV
      leaves them when they enter as zeros (dynamics.F:300-310), guDissip/gvDissip likewise (zeroed per level,
      :501-506).
    myIter: the step's iteration number (Python int or traced integer).
    Returns dict: gU, gV (after implicit viscosity), guNm, gvNm, totPhiHyd, phiHydLow, kappaRU, kappaRV,
      gU_explicit, gV_explicit (after TIMESTEP, before IMPLDIFF), and the phi_hyd fields (phiHydC, phiHydF,
      dPhiHydX, dPhiHydY).
    """
    L = g.layout
    ts = p.ts
    shape2 = (n_tiles(g), L.ny, L.nx)
    phiSurfX = jnp.zeros(shape2)  # dynamics.F:323 (CALC_GRAD_PHI_SURF not called: implicSurfPress = 1, :349)
    phiSurfY = jnp.zeros(shape2)  # dynamics.F:324

    kappaRU, kappaRV = calc_viscosity(p, g, s["GGL90viscArU"], s["GGL90viscArV"])

    ph = calc_phi_hyd(p.phi, g, kLowC, s["rhoInSitu"], s["rStarFacC"], s["etaH"], s["phi0surf"], s["totPhiHyd"],
                      myIter=myIter)

    out = dict(ph, kappaRU=kappaRU, kappaRV=kappaRV, guNm=s["guNm"], gvNm=s["gvNm"])
    gU = jnp.zeros((n_tiles(g), L.Nr, L.ny, L.nx))  # dynamics.F:305-306
    gV = jnp.zeros_like(gU)
    if p.momStepping:  # dynamics.F:499
        gU, gV, guDissip, gvDissip = mom_vecinv(kappaRU, kappaRV)
        guExt, gvExt = apply_forcing_uv(ts, g, s["surfaceForcingU"], s["surfaceForcingV"],
                                        s["recip_hFacW"], s["recip_hFacS"])
        gU, gV, out["guNm"], out["gvNm"] = timestep(
            ts, g, myIter, gU, gV, s["guNm"], s["gvNm"], s["uVel"], s["vVel"], ph["dPhiHydX"], ph["dPhiHydY"],
            phiSurfX, phiSurfY, guDissip, gvDissip, guExt, gvExt)
    out["gU_explicit"], out["gV_explicit"] = gU, gV
    if p.implicitViscosity:  # dynamics.F:591-609
        dTX = impldiff_deltaTX(-1, ts.deltaTMom, Nr=L.Nr)
        rng = dict(iMin=0, iMax=L.sNx + 1, jMin=0, jMax=L.sNy + 1)  # dynamics.F:190-191
        gU = impldiff(g, -1, kappaRU, s["recip_hFacW"], gU, dTX, **rng)
        gV = impldiff(g, -2, kappaRV, s["recip_hFacS"], gV, dTX, **rng)
    out["gU"], out["gV"] = gU, gV
    return out
