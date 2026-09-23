"""Ocean mixed-layer depth hMixLayer (plan Task 10; full-tree branch M2.6a): model/src/calc_oce_mxlayer.F
(CALC_OCE_MXLAYER), as called from DO_OCEANIC_PHYS (do_oceanic_phys.F:941-945 (ff) / c66g :935-939) with
rhoSurf = rhoInSitu(k=1) for every tile.

calcMixLayerDepth (calc_oce_mxlayer.F:66-77):
    = GM_useSubMeso .OR. GM_taper_scheme.EQ.'fm07' .OR. GM_useK3D            [useGMRedi .AND. .NOT.useKPP]
      .OR. DIAGNOSTICS_IS_ON('MXLDEPTH')                                     [useDiagnostics]
  * flux-forced V4r4: data.gmredi sets GM_taper_scheme = 'stableGmAdjTap', GM_useSubMeso = GM_useK3D = .FALSE.
    (gmredi_readparms.F:136, 143) and useDiagnostics = F -> .FALSE.: hMixLayer keeps its INI_DYNVARS value 0.
  * full V4r4: useDiagnostics = T and data.diagnostics requests MXLDEPTH (list 45, monthly mean) -> .TRUE.; the
    default hMixCriteria = -0.8 (set_defaults.F:217; data does not set it) selects method 1 (calc_oce_mxlayer.F:81-83),
    hMixSmooth = 0 (set_defaults.F:219) skips the smoothing (:191). gcov (ref_full_serial13_gcov_1day) confirms
    method 1 with FIND_ALPHA (JMD95), no smoothing.
hMixLayer is read by the ocean physics only in GMREDI's fm07 taper / sub-mesoscale code (not V4r4): in V4r4 it is a
diagnostic that is part of the dumped state (S00_begin / P02 group m).

Method 1 (calc_oce_mxlayer.F:85-131), on the full tile 1-OLx..sNx+OLx, 1-OLy..sNy+OLy:
    rhoMxL    = FIND_ALPHA(k=1, kRef=1)                                   :96-98
    rhoKm1    = rhoSurf                                                   :102
    rhoMxL    = rhoSurf + MAX(rhoMxL*hMixCriteria, dRhoSmall)             :103-104
    hMixLayer = rF(1) - R_low                                             :105
    DO k = 2, Nr:  rhoLoc = FIND_RHO_2D(theta(k), salt(k), kRef = 1)      :108-114
      IF (k <= klowC .AND. rhoLoc >= rhoMxL): tmpFac = (rhoMxL-rhoKm1)/(rhoLoc-rhoKm1) if rhoLoc > rhoKm1 else 0,
         hMixLayer = rF(1) - rC(k-1) + tmpFac*drC(k), rhoMxL = rhoBigNb                        :118-127
      ELSE rhoKm1 = rhoLoc                                                                    :128-129
The k loop carries rhoKm1, rhoMxL and hMixLayer: a lax.scan over k = 2..Nr (FIND_RHO_2D of all levels is computed
first: it depends only on theta, salt).
AD: tmpFac's division is guarded (denominator 1 where rhoLoc <= rhoKm1) before the operation.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import lax
import numpy as np

from mitgcm_jax.core import eos as eos_mod
from mitgcm_jax.params_io import diagnostics_is_on, params_pytree


@params_pytree
@dataclass(frozen=True)
class MxLayerParams:
    calcMixLayerDepth: bool
    method: int = 0
    hMixCriteria: float = -0.8        # set_defaults.F:217 hMixCriteria = -.8 _d 0
    dRhoSmall: float = 1.0e-6         # set_defaults.F:218 dRhoSmall = 1. _d -6
    rhoBigNb: float = 999.8e10        # calc_oce_mxlayer.F:95 rhoConst*1. _d 10 (set in from_namelists)

    @classmethod
    def from_namelists(cls, nml):
        useGMRedi = bool(nml.get("data.pkg", "packages", "useGMRedi", default=False))  # packages_boot.F
        useKPP = bool(nml.get("data.pkg", "packages", "useKPP", default=False))        # packages_boot.F
        calc = False                                                                    # calc_oce_mxlayer.F:66
        if useGMRedi and not useKPP:                                                    # calc_oce_mxlayer.F:68
            grp = "gm_parm01"
            subMeso = bool(nml.get("data.gmredi", grp, "GM_useSubMeso", default=False))  # gmredi_readparms.F:136
            k3d = bool(nml.get("data.gmredi", grp, "GM_useK3D", default=False))          # gmredi_readparms.F:143
            taper = nml.get("data.gmredi", grp, "GM_taper_scheme", default=" ")          # gmredi_readparms.F:103
            calc = subMeso or taper == "fm07" or k3d                                    # calc_oce_mxlayer.F:69-70
            if calc:
                raise NotImplementedError("calcMixLayerDepth from GM_useSubMeso / fm07 / GM_useK3D: not a V4r4 "
                                          "branch (those GMREDI paths also read hMixLayer)")
        if not calc:                                                                    # calc_oce_mxlayer.F:74-76
            calc = diagnostics_is_on(nml, "MXLDEPTH")
        if not calc:
            return cls(calcMixLayerDepth=False)
        p1 = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
        hMixCriteria = float(p1("hMixCriteria", -0.8))              # set_defaults.F:217
        dRhoSmall = float(p1("dRhoSmall", 1.0e-6))                  # set_defaults.F:218
        hMixSmooth = float(p1("hMixSmooth", 0.0))                   # set_defaults.F:219
        method = 0                                                  # calc_oce_mxlayer.F:81
        if hMixCriteria < 0.0:
            method = 1                                              # :82
        if hMixCriteria > 1.0:
            method = 2                                              # :83
        if method != 1:
            raise NotImplementedError(f"CALC_OCE_MXLAYER method {method} (hMixCriteria={hMixCriteria}): only "
                                      "method 1 (V4r4 full tree) is ported")
        if hMixSmooth > 0.0:
            raise NotImplementedError("hMixSmooth > 0: calc_oce_mxlayer.F:191-214 smoothing not ported")
        rhoNil = p1("rhoNil", 999.8)                                # set_defaults.F:106
        rhoConst = float(p1("rhoConst", rhoNil))                    # ini_parms.F:445
        return cls(calcMixLayerDepth=True, method=method, hMixCriteria=hMixCriteria, dRhoSmall=dRhoSmall,
                   rhoBigNb=rhoConst * 1.0e10)                     # calc_oce_mxlayer.F:95


def calc_oce_mxlayer(p, hMixLayer, eos=None, g=None, kLowC=None, theta=None, salt=None, rhoSurf=None):
    """CALC_OCE_MXLAYER. calcMixLayerDepth = .FALSE. (flux-forced V4r4): hMixLayer unchanged (calc_oce_mxlayer.F:78,
    224). calcMixLayerDepth = .TRUE. (full V4r4): method 1 on the full tile; needs eos (EOSParams), g (R_low, rF, rC,
    drC), kLowC [T, ny, nx] (int, Fortran level of the deepest wet cell, 0 on land), theta/salt [T, Nr, ny, nx] and
    rhoSurf = rhoInSitu(k=1) [T, ny, nx]; every point of hMixLayer is rewritten (:105)."""
    if not p.calcMixLayerDepth:
        return hMixLayer
    Nr = theta.shape[1]
    rF, rC, drC = jnp.asarray(g.rF), jnp.asarray(g.rC), jnp.asarray(g.drC)
    # :96-98 FIND_ALPHA(bi, bj, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy, k = 1, kRef = 1, rhoMxL)
    rhoMxL = eos_mod.find_alpha(eos, theta[:, 0], salt[:, 0], 1)
    rhoKm1 = rhoSurf                                                              # :102
    rhoMxL = rhoSurf + jnp.maximum(rhoMxL * p.hMixCriteria, p.dRhoSmall)          # :103-104
    hMix = rF[0] - g.R_low                                                        # :105
    # :110-114 FIND_RHO_2D(1-OLx, sNx+OLx, 1-OLy, sNy+OLy, kRef = 1, theta(k), salt(k)), k = 2..Nr
    rhoLocs = eos_mod.find_rho_levels(eos, theta[:, 1:], salt[:, 1:], np.ones(Nr - 1, dtype=np.int64))
    ks = jnp.arange(2, Nr + 1, dtype=jnp.int32)

    def level(carry, x):
        rhoKm1, rhoMxL, hMix = carry
        k, rhoLoc, rCkm1, drCk = x
        found = (k <= kLowC) & (rhoLoc >= rhoMxL)                                 # :118-119
        up = rhoLoc > rhoKm1                                                      # :120
        tmpFac = jnp.where(up, (rhoMxL - rhoKm1) / jnp.where(up, rhoLoc - rhoKm1, 1.0), 0.0)   # :121-124
        hMix = jnp.where(found, rF[0] - rCkm1 + tmpFac * drCk, hMix)             # :126
        rhoMxL = jnp.where(found, p.rhoBigNb, rhoMxL)                            # :127
        rhoKm1 = jnp.where(found, rhoKm1, rhoLoc)                                # :129
        return (rhoKm1, rhoMxL, hMix), None

    (_, _, hMix), _ = lax.scan(level, (rhoKm1, rhoMxL, hMix),
                               (ks, jnp.moveaxis(rhoLocs, 1, 0), rC[:Nr - 1], drC[1:Nr]))
    return hMix
