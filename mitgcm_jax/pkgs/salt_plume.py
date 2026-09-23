"""pkg/salt_plume (plan Task 11): plume depth and the salt-plume tendency of salinity, V4r4 flux-forced options.

Build options (pkg/salt_plume/SALT_PLUME_OPTIONS.h, not overridden by the flux-forced tree): SALT_PLUME_IN_LEADS,
SALT_PLUME_SPLIT_BASIN and SALT_PLUME_VOLUME are all #undef. EXF_OPTIONS.h(ff):189 defines READIN_SALT_PLUME_FLUX.
data.salt_plume sets only SPsalFRAC = 0.5 (read, but used by no V4r4 routine: seaice / coupler only); everything else
is the default of salt_plume_readparms.F:62-78, 102-107: CriterionType = 1, PlumeMethod = 1, Npower = 0,
SPovershoot = 1, SaltPlumeCriterion = 0.4 (UNSET with CriterionType 1).

Where saltPlumeFlux comes from (flux-forced tree, not this module): EXF reads `spflx` and EXF_MAPFIELDS copies it at
every point (exf_mapfields.F(ff):343-349, range 1-OLx..sNx+OLx) and exchanges it (exf_mapfields.F:386-388); the
ALLOW_AUTODIFF zeroing at the top of DO_OCEANIC_PHYS resets only saltPlumeDepth, not saltPlumeFlux, because
READIN_SALT_PLUME_FLUX is visible there (do_oceanic_phys.F(ff):289-299, EXF_OPTIONS.h included under ALLOW_AUTODIFF);
SALT_PLUME_DO_EXCH exchanges it once more (do_oceanic_phys.F:578-583, salt_plume_do_exch.F).

Routines:
    salt_plume_do_exch              SALT_PLUME_DO_EXCH (EXCH_XY_RL of saltPlumeFlux)
    salt_plume_calc_depth           SALT_PLUME_CALC_DEPTH, CriterionType = 1 (do_oceanic_phys.F(ff):947-951)
    salt_plume_frac                 SALT_PLUME_FRAC, PlumeMethod = 1 (non-NEC branch)
    salt_plume_tendency_apply_s     SALT_PLUME_TENDENCY_APPLY_S (called from APPLY_FORCING_S, apply_forcing.F(ff))
SALT_PLUME_TENDENCY_APPLY_T is empty without SALT_PLUME_VOLUME (salt_plume_tendency_apply_t.F): nothing to port.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.core import eos as eos_mod
from mitgcm_jax.params_io import params_pytree


@params_pytree
@dataclass(frozen=True)
class SaltPlumeParams:
    """float fields are pytree leaves (pass the params as a jit argument); ints are static."""
    CriterionType: int
    PlumeMethod: int
    Npower: int
    SPovershoot: float
    SaltPlumeCriterion: float
    rhoConst: float
    mass2rUnit: float

    @classmethod
    def from_namelists(cls, nml):
        if not bool(nml.get("data.pkg", "packages", "useSALT_PLUME", default=False)):
            raise NotImplementedError("useSALT_PLUME=F: the model does not call pkg/salt_plume")
        grp = "salt_plume_parm01"
        ct = int(nml.get("data.salt_plume", grp, "CriterionType", default=1))      # salt_plume_readparms.F:63
        pm = int(nml.get("data.salt_plume", grp, "PlumeMethod", default=1))        # salt_plume_readparms.F:64
        npow = int(nml.get("data.salt_plume", grp, "Npower", default=0))           # salt_plume_readparms.F:74
        over = float(nml.get("data.salt_plume", grp, "SPovershoot", default=1.0))  # salt_plume_readparms.F:66
        # salt_plume_readparms.F:65 UNSET; :102-104 -> 0.4 for CriterionType 1 (0.005 for 2)
        crit = float(nml.get("data.salt_plume", grp, "SaltPlumeCriterion", default=0.4 if ct == 1 else 0.005))
        if ct != 1:
            raise NotImplementedError(f"CriterionType={ct}: only 1 (V4r4) is ported")
        if pm != 1:
            raise NotImplementedError(f"PlumeMethod={pm}: only 1 (V4r4) is ported")
        if nml.get("data", "parm01", "buoyancyRelation", default="OCEANIC") != "OCEANIC":
            raise NotImplementedError("mass2rUnit for p coordinates is not ported")
        # set_defaults.F:106-107, ini_parms.F:445 rhoConst; ini_parms.F:640 recip_rhoConst = 1/rhoConst;
        # ini_parms.F:1439 mass2rUnit = recip_rhoConst (z coordinates)
        rhoNil = nml.get("data", "parm01", "rhoNil", default=999.8)
        rhoConst = float(nml.get("data", "parm01", "rhoConst", default=rhoNil))
        mass2rUnit = float(np.float64(1.0) / np.float64(rhoConst))
        return cls(CriterionType=ct, PlumeMethod=pm, Npower=npow, SPovershoot=over, SaltPlumeCriterion=crit,
                   rhoConst=rhoConst, mass2rUnit=mass2rUnit)


def klowc(hFacC):
    """kLowC (ini_masks_etc.F:188-203): index of the deepest wet level (0 for a dry column), from the INITIAL hFacC
    (under r* the static h0FacC; kLowC is never recomputed in V4r4). hFacC [T, Nr, ny, nx] -> int32 [T, ny, nx]."""
    Nr = hFacC.shape[1]
    k = jnp.arange(1, Nr + 1, dtype=jnp.int32)[None, :, None, None]
    return jnp.max(jnp.where(hFacC != 0.0, k, 0), axis=1).astype(jnp.int32)


def salt_plume_do_exch(ex, saltPlumeFlux):
    """salt_plume_do_exch.F: IF (useSALT_PLUME) _EXCH_XY_RL(saltPlumeFlux) (no coupler)."""
    return ex.exch_xy(saltPlumeFlux)


def salt_plume_calc_depth(sp, eos, g, rhoSurf, theta, salt, kLowC):
    """SALT_PLUME_CALC_DEPTH, CriterionType = 1 (salt_plume_calc_depth.F:79-191), all tiles, full tile range.
    rhoSurf = rhoInSitu(:,:,1) [T, ny, nx]; theta, salt [T, Nr, ny, nx] (state at the start of the step);
    kLowC int [T, ny, nx]. Uses g.R_low, g.rF, g.rC, g.drC. Returns SaltPlumeDepth [T, ny, nx].
    The k loop 2..Nr is a genuine recurrence (rhoKm1, rhoMxL carried): lax.scan in the Fortran order; the
    FIND_RHO_2D calls it contains are independent of the carry and are evaluated for all k up front."""
    L = g.layout
    Nr = L.Nr
    rF = jnp.asarray(g.rF)
    rC = jnp.asarray(g.rC)
    drC = jnp.asarray(g.drC)
    R_low = g.R_low
    rF1 = rF[0]
    # salt_plume_calc_depth.F:80-87, 92-99
    rhoBigNb = sp.rhoConst * 1.0e10
    SPD = rF1 - R_low
    rhoKm1 = rhoSurf
    rhoMxL = rhoSurf + sp.SaltPlumeCriterion
    # salt_plume_calc_depth.F:114-118  potential density referenced to level 1: FIND_RHO_2D(kRef = 1), k = 2..Nr
    rhoLocAll = eos_mod.find_rho_levels(eos, theta[:, 1:], salt[:, 1:], np.ones(Nr - 1, np.int64))

    def body(carry, xs):
        spd, rKm1, rMxL = carry
        rhoLoc, k, rC_km1, drC_k = xs
        # salt_plume_calc_depth.F:122-123
        hit = (k <= kLowC) & (rhoLoc >= rMxL)
        # salt_plume_calc_depth.F:124-129 (denominator guarded where the branch is not taken)
        up = rhoLoc > rKm1
        den = jnp.where(up, rhoLoc - rKm1, 1.0)
        tmpFac = jnp.where(up, (rMxL - rKm1) / den, 0.0)
        # salt_plume_calc_depth.F:130-133
        spd = jnp.where(hit, rF1 - rC_km1 + tmpFac * drC_k, spd)
        rMxL = jnp.where(hit, rhoBigNb, rMxL)
        rKm1 = jnp.where(hit, rKm1, rhoLoc)
        return (spd, rKm1, rMxL), None

    ks = jnp.arange(2, Nr + 1, dtype=jnp.int32)
    xs = (jnp.moveaxis(rhoLocAll, 1, 0), ks, rC[ks - 2], drC[ks - 1])
    (SPD, _, _), _ = jax.lax.scan(body, (SPD, rhoKm1, rhoMxL), xs)
    # salt_plume_calc_depth.F:185-191
    return jnp.minimum(SPD, rF1 - R_low)


def salt_plume_frac(sp, fact, SPDepth, plumek):
    """SALT_PLUME_FRAC (salt_plume_frac.F:84-221), PlumeMethod = 1, not TARGET_NEC_SX, elementwise:
    fraction of the plume flux above depth |fact*plumek|."""
    facz = jnp.abs(fact * plumek)                                  # salt_plume_frac.F:85
    active = (SPDepth >= facz) & (SPDepth > 0.0)                   # salt_plume_frac.F:98
    dd20 = jnp.abs(jnp.where(active, SPDepth, 1.0))                # salt_plume_frac.F:102 (guarded for AD)
    S = 1.0                                                        # salt_plume_frac.F:113-118
    So = 1.0
    for _ in range(sp.Npower + 1):
        S = facz * S
        So = dd20 * So
    return jnp.where(active, jnp.maximum(0.0, S / So), 1.0)       # salt_plume_frac.F:119, 219


def salt_plume_tendency_s(sp, g, k, SaltPlumeDepth, saltPlumeFlux, maskC_k, recip_hFacC_k):
    """(inside, term): the term SALT_PLUME_TENDENCY_APPLY_S adds to gS_arr at level k (Fortran index, 1..Nr) where
    `inside` (salt_plume_tendency_apply_s.F:121-156, not NEC, no SALT_PLUME_VOLUME):
        IF ( SaltPlumeDepth > ABS(rF(k)) ):
            plumefrac = (FRAC(|rF(k+1)|) - FRAC(|rF(k)|))*maskC(k)
            plumetend = saltPlumeFlux*plumefrac
            gS_arr    = gS_arr + plumetend*recip_drF(k)*mass2rUnit*recip_hFacC(k)
        (ELSE gS_arr unchanged)
    maskC_k, recip_hFacC_k: [T, ny, nx] of level k (recip_hFacC is the r*-time-dependent one of the caller)."""
    rF = jnp.asarray(g.rF)
    recip_drF = jnp.asarray(g.recip_drF)
    absrFk = jnp.abs(rF[k - 1])
    absrFkp1 = jnp.abs(rF[k])
    minusone = -1.0                                                # salt_plume_tendency_apply_s.F:53
    inside = SaltPlumeDepth > absrFk                               # :124
    kb1 = salt_plume_frac(sp, minusone, SaltPlumeDepth, absrFk)    # :127, 129, 137-143
    kb2 = salt_plume_frac(sp, minusone, SaltPlumeDepth, absrFkp1)  # :128, 130
    plumefrac = (kb2 - kb1) * maskC_k                              # :144
    plumetend = saltPlumeFlux * plumefrac                          # :145
    term = plumetend * recip_drF[k - 1] * sp.mass2rUnit * recip_hFacC_k   # :149-150
    return inside, term


def salt_plume_tendency_apply_s(sp, g, gS_arr, k, SaltPlumeDepth, saltPlumeFlux, maskC_k, recip_hFacC_k,
                                iMin, iMax, jMin, jMax):
    """SALT_PLUME_TENDENCY_APPLY_S(gS_arr, iMin..iMax, jMin..jMax, k): gS_arr [T, ny, nx] of level k updated on the
    caller's range (salt_integrate.F:157-160 passes 0..sNx+1, 0..sNy+1):
        gS_arr(i,j) = gS_arr(i,j) + plumetend*recip_drF(k)*mass2rUnit*recip_hFacC(i,j,k)   (:149-150)"""
    L = g.layout
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    inside, term = salt_plume_tendency_s(sp, g, k, SaltPlumeDepth[..., J, I], saltPlumeFlux[..., J, I],
                                         maskC_k[..., J, I], recip_hFacC_k[..., J, I])
    old = gS_arr[..., J, I]
    return gS_arr.at[..., J, I].set(jnp.where(inside, old + term, old))
