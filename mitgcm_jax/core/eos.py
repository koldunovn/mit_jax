"""Equation of state, eosType='JMD95Z' (plan Task 10): literal port of the V4r4 path of

    model/src/find_rho.F        FIND_RHO_2D, FIND_RHOP0, FIND_BULKMOD (JMD95 branch)
    model/src/pressure_for_eos.F  the pressure fed to the EOS (usingZCoords, selectP_inEOS_Zc = 0)
    model/src/ini_eos.F         JMD95 coefficients
    model/src/set_ref_state.F   pRef4EOS (the horizontally uniform reference pressure)

Build facts (ff serial13 jaxdump build, preprocessed find_rho.f):
  * USE_FACTORIZED_EOS is defined only for TARGET_NEC_SX (find_rho.F:6-9), so the expanded (not Horner) polynomials
    are compiled: find_rho.F:363-369 (rfresh), 383-396 (rsalt), 516-521, 549-577 (bulk modulus).
  * CHECK_SALINITY_FOR_NEGATIVE_VALUES is undefined (CPP_OPTIONS.h), LOOK_FOR_NEG_SALINITY is not called.
  * eosType='JMD95Z' (data:PARM01) and selectP_inEOS_Zc unset -> 0 (set_parms.F:231-237), so the pressure of level
    kRef is pRef4EOS(kRef) at every point (pressure_for_eos.F:86-94), a static [Nr] vector (set_ref_state.F:94-98).
  * rhoInSitu = FIND_RHO_2D at every point of the tile, halos included, dry points included (no mask): every V4r4
    call site passes the full range 1-OLx..sNx+OLx, 1-OLy..sNy+OLy (do_oceanic_phys.F(ff):596-599 -> :804-810,
    salt_plume_calc_depth.F:114-118; ALLOW_AUTODIFF zeroes rhoLoc first, find_rho.F:77-85, so nothing is left over).

FIND_ALPHA (find_alpha.F:115-221, JMD95 branch; `find_alpha`) is called only by CALC_OCE_MXLAYER method 1
(calc_oce_mxlayer.F:94-97). The flux-forced tree never reaches it (calcMixLayerDepth = F); the full V4r4 tree does,
because its data.diagnostics requests MXLDEPTH (DIAGNOSTICS_IS_ON, calc_oce_mxlayer.F:73-77; gcov
ref_full_serial13_gcov_1day: find_alpha 38 %, the JMD95 branch). FIND_BETA and FIND_RHO_SCALAR (initialisation-only:
set_ref_state.F, ini_linear_phisurf.F) are not ported.

AD: s**1.5 = s*SQRT(s) only where s > 0 (find_rho.F:345-351, 498-504); dry points hold S = 0, so SQRT is taken of a
guarded argument (jnp.where before the sqrt) to keep the backward pass finite.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mitgcm_jax.params_io import params_pytree

# EOS.h:20  PARAMETER ( SItoBar = 1.D-05 )
SItoBar = 1.0e-05

# ini_eos.F:121-126  density of fresh water at p = 0
eosJMDCFw = (999.842594e+00, 6.793952e-02, -9.095290e-03, 1.001685e-04, -1.120083e-06, 6.536332e-09)
# ini_eos.F:128-136  density of sea water at p = 0
eosJMDCSw = (8.24493e-01, -4.0899e-03, 7.6438e-05, -8.2467e-07, 5.3875e-09, -5.72466e-03, 1.0227e-04,
             -1.6546e-06, 4.8314e-04)
# ini_eos.F:139-143  secant bulk modulus of fresh water at p = 0 (JMD95)
eosJMDCKFw = (1.965933e+04, 1.444304e+02, -1.706103e+00, 9.648704e-03, -4.190253e-05)
# ini_eos.F:145-151  secant bulk modulus of sea water at p = 0 (JMD95)
eosJMDCKSw = (5.284855e+01, -3.101089e-01, 6.283263e-03, -5.084188e-05, 3.886640e-01, 9.085835e-03, -4.619924e-04)
# ini_eos.F:153-166  secant bulk modulus of sea water at pressure p (JMD95)
eosJMDCKP = (3.186519e+00, 2.212276e-02, -2.984642e-04, 1.956415e-06, 6.704388e-03, -1.847318e-04, 2.059331e-07,
             1.480266e-04, 2.102898e-04, -1.202016e-05, 1.394680e-07, -2.040237e-06, 6.128773e-08, 6.207323e-10)


@params_pytree
@dataclass(frozen=True)
class EOSParams:
    """EOS set-up. rhoConst is a pytree leaf (traced when the params are a jit argument); `pRef4EOS` (the [Nr]
    reference pressure of set_ref_state.F:97, Pa) and the coefficient tuples are static."""
    eosType: str
    rhoConst: float
    selectP_inEOS_Zc: int
    pRef4EOS: tuple
    Fw: tuple = eosJMDCFw
    Sw: tuple = eosJMDCSw
    KFw: tuple = eosJMDCKFw
    KSw: tuple = eosJMDCKSw
    KP: tuple = eosJMDCKP

    @classmethod
    def from_namelists(cls, nml, rC, rF):
        """nml: RunNamelists of the run directory; rC [Nr], rF [Nr+1]: the model's vertical grid (g.rC, g.rF)."""
        eosType = nml.get("data", "parm01", "eosType", default="LINEAR")  # set_defaults.F:174 eosType = 'LINEAR'
        if eosType != "JMD95Z":
            raise NotImplementedError(f"eosType={eosType!r}: only 'JMD95Z' (V4r4) is ported")
        buoy = nml.get("data", "parm01", "buoyancyRelation", default="OCEANIC")  # set_defaults.F:175 'OCEANIC'
        if buoy != "OCEANIC":
            raise NotImplementedError(f"buoyancyRelation={buoy!r}: only 'OCEANIC' (usingZCoords) is ported")
        # set_parms.F:231-237: unset -> 0 for JMD95Z (2 only for JMD95P/UNESCO/MDJWF/TEOS10)
        sel = nml.get("data", "parm01", "selectP_inEOS_Zc", default=0)
        if sel != 0:
            raise NotImplementedError(f"selectP_inEOS_Zc={sel}: only 0 (pRef4EOS, V4r4) is ported")
        if nml.get("data", "parm01", "gravityFile", default=" ").strip():  # set_defaults.F:62 gravityFile = ' '
            raise NotImplementedError("gravityFile set: only the uniform-gravity phiRef/pRef4EOS is ported")
        # set_defaults.F:106 rhoNil = 999.8, :107 rhoConst = UNSET; ini_parms.F:445 UNSET -> rhoNil
        rhoNil = nml.get("data", "parm01", "rhoNil", default=999.8)
        rhoConst = float(nml.get("data", "parm01", "rhoConst", default=rhoNil))
        gravity = float(nml.get("data", "parm01", "gravity", default=9.81))  # set_defaults.F:104 gravity = 9.81
        top_Pres = float(nml.get("data", "parm04", "top_Pres", default=0.0))  # ini_parms.F:1216 UNSET -> 0.
        return cls(eosType=eosType, rhoConst=rhoConst, selectP_inEOS_Zc=0,
                   pRef4EOS=pref4eos(rhoConst, gravity, top_Pres, rC, rF))


def pref4eos(rhoConst, gravity, top_Pres, rC, rF):
    """set_ref_state.F:86-98 (OCEANIC, gravityFile=' '), as float64 in the Fortran operation order:
        pRefLocF(1) = top_Pres                                                       (set_ref_state.F:87)
        pRef4EOS(k) = pRefLocF(1) + rhoConst*(rC(k) - rF(1))*gravity*gravitySign      (set_ref_state.F:97-98)
    gravitySign = -1 for z coordinates (ini_vertical_grid.F:57)."""
    gravitySign = np.float64(-1.0)  # ini_vertical_grid.F:57
    pRefLocF1 = np.float64(top_Pres)
    rC = np.asarray(rC, np.float64)
    rF1 = np.float64(np.asarray(rF, np.float64)[0])
    out = [float(pRefLocF1 + np.float64(rhoConst) * (rC[k] - rF1) * np.float64(gravity) * gravitySign)
           for k in range(rC.shape[0])]
    return tuple(out)


def pressure_for_eos(eos, kRef):
    """pressure_for_eos.F:86-94: locPres = pRef4EOS(kRef) at every point (selectP_inEOS_Zc = 0).
    kRef: Fortran level index (int) or an integer array of them; returns a scalar / array to broadcast."""
    p = jnp.asarray(eos.pRef4EOS, jnp.float64)
    return p[jnp.asarray(kRef) - 1]


def _s_and_s3o2(sFld):
    """find_rho.F:345-351: s = S, s3o2 = S*SQRT(S) if S > 0, else both 0 (sqrt of a guarded argument for AD)."""
    pos = sFld > 0.0
    s_safe = jnp.where(pos, sFld, 1.0)
    s3o2 = jnp.where(pos, s_safe * jnp.sqrt(s_safe), 0.0)
    s = jnp.where(pos, sFld, 0.0)
    return s, s3o2


def find_rhop0(eos, tFld, sFld):
    """FIND_RHOP0 (find_rho.F:274-404), non-factorized polynomials: rho(S, T, p=0)."""
    Fw, Sw = eos.Fw, eos.Sw
    t = tFld
    t2 = t * t           # find_rho.F:337
    t3 = t2 * t          # find_rho.F:338
    t4 = t3 * t          # find_rho.F:339
    s, s3o2 = _s_and_s3o2(sFld)
    # find_rho.F:363-369
    rfresh = (Fw[0]
              + Fw[1] * t
              + Fw[2] * t2
              + Fw[3] * t3
              + Fw[4] * t4
              + Fw[5] * t4 * t)
    # find_rho.F:383-396
    rsalt = (s * (Sw[0]
                  + Sw[1] * t
                  + Sw[2] * t2
                  + Sw[3] * t3
                  + Sw[4] * t4)
             + s3o2 * (Sw[5]
                       + Sw[6] * t
                       + Sw[7] * t2)
             + Sw[8] * s * s)
    return rfresh + rsalt  # find_rho.F:399


def find_bulkmod(eos, locPres, tFld, sFld):
    """FIND_BULKMOD (find_rho.F:411-587), non-factorized polynomials: secant bulk modulus K(S, T, p)."""
    KFw, KSw, KP = eos.KFw, eos.KSw, eos.KP
    t = tFld
    t2 = t * t           # find_rho.F:490
    t3 = t2 * t          # find_rho.F:491
    t4 = t3 * t          # find_rho.F:492
    s, s3o2 = _s_and_s3o2(sFld)
    p = locPres * SItoBar  # find_rho.F:506
    p2 = p * p             # find_rho.F:507
    # find_rho.F:516-521
    bMfresh = (KFw[0]
               + KFw[1] * t
               + KFw[2] * t2
               + KFw[3] * t3
               + KFw[4] * t4)
    # find_rho.F:549-558
    bMsalt = (s * (KSw[0]
                   + KSw[1] * t
                   + KSw[2] * t2
                   + KSw[3] * t3)
              + s3o2 * (KSw[4]
                        + KSw[5] * t
                        + KSw[6] * t2))
    # find_rho.F:560-577
    bMpres = (p * (KP[0]
                   + KP[1] * t
                   + KP[2] * t2
                   + KP[3] * t3)
              + p * s * (KP[4]
                         + KP[5] * t
                         + KP[6] * t2)
              + p * s3o2 * KP[7]
              + p2 * (KP[8]
                      + KP[9] * t
                      + KP[10] * t2)
              + p2 * s * (KP[11]
                          + KP[12] * t
                          + KP[13] * t2))
    return bMfresh + bMsalt + bMpres  # find_rho.F:581


def find_rho(eos, tFld, sFld, locPres):
    """FIND_RHO_2D, JMD95 branch (find_rho.F:149-184), elementwise on arrays of any shape:
        rhoLoc = rhoP0/(1 - locPres*SItoBar/bulkMod) - rhoConst      (find_rho.F:178-181)
    locPres broadcasts against tFld (use pressure_for_eos)."""
    rhoP0 = find_rhop0(eos, tFld, sFld)                 # find_rho.F:158-162
    bulkMod = find_bulkmod(eos, locPres, tFld, sFld)    # find_rho.F:164-168
    return rhoP0 / (1.0 - locPres * SItoBar / bulkMod) - eos.rhoConst


def find_rho_2d(eos, tFld, sFld, kRef):
    """FIND_RHO_2D(iMin..iMax = full tile, kRef, tFld, sFld) for one level: tFld/sFld [T, ny, nx] (or any shape),
    kRef the Fortran pressure-reference level (pressure_for_eos.F:91)."""
    return find_rho(eos, tFld, sFld, pressure_for_eos(eos, kRef))


def find_alpha(eos, tFld, sFld, kRef):
    """FIND_ALPHA, JMD95 branch (find_alpha.F:115-221; equationOfState(1:5) = 'JMD95'), elementwise on the full
    tile: alphaLoc = d(rho)/d(theta) of level tFld/sFld at the reference pressure of level kRef.
    Fortran operation order kept: `n.*c*x` = (n*c)*x, `x**2` = x*x (gfortran integer power 2), sums left to right.
    s3o2 = SQRT(s1*s1*s1) here (find_alpha.F:150), not s1*SQRT(s1) as in FIND_RHOP0; the argument is guarded for AD
    (S = 0 on dry points)."""
    Fw, Sw, KFw, KSw, KP = eos.Fw, eos.Sw, eos.KFw, eos.KSw, eos.KP
    locPres = pressure_for_eos(eos, kRef)                     # find_alpha.F:119-122 PRESSURE_FOR_EOS(kRef)
    rhoP0 = find_rhop0(eos, tFld, sFld)                       # :124-128 FIND_RHOP0
    bulkMod = find_bulkmod(eos, locPres, tFld, sFld)          # :130-134 FIND_BULKMOD
    t1 = tFld                                                 # :140
    t2 = t1 * t1                                              # :141
    t3 = t2 * t1                                              # :142
    pos = sFld > 0.0                                          # :149
    s_safe = jnp.where(pos, sFld, 1.0)
    s3o2 = jnp.where(pos, jnp.sqrt(s_safe * s_safe * s_safe), 0.0)   # :150 / :153
    s1 = jnp.where(pos, sFld, 0.0)                            # :148 / :152
    p1 = locPres * SItoBar                                    # :156
    p2 = p1 * p1                                              # :157
    # :162-166 d(rho)/d(theta) of fresh water at p = 0
    drhoP0dthetaFresh = (Fw[1]
                         + 2. * Fw[2] * t1
                         + 3. * Fw[3] * t2
                         + 4. * Fw[4] * t3
                         + 5. * Fw[5] * t3 * t1)
    # :168-178 of salt water at p = 0
    drhoP0dthetaSalt = (s1 * (Sw[1]
                              + 2. * Sw[2] * t1
                              + 3. * Sw[3] * t2
                              + 4. * Sw[4] * t3)
                        + s3o2 * (+ Sw[6]
                                  + 2. * Sw[7] * t1))
    # :181-185 d(bulk modulus)/d(theta) of fresh water at p = 0
    dKdthetaFresh = (KFw[1]
                     + 2. * KFw[2] * t1
                     + 3. * KFw[3] * t2
                     + 4. * KFw[4] * t3)
    # :187-194 of sea water at p = 0
    dKdthetaSalt = (s1 * (KSw[1]
                          + 2. * KSw[2] * t1
                          + 3. * KSw[3] * t2)
                    + s3o2 * (KSw[5]
                              + 2. * KSw[6] * t1))
    # :196-209 of sea water at p
    dKdthetaPres = (p1 * (KP[1]
                          + 2. * KP[2] * t1
                          + 3. * KP[3] * t2)
                    + p1 * s1 * (KP[5]
                                 + 2. * KP[6] * t1)
                    + p2 * (KP[9]
                            + 2. * KP[10] * t1)
                    + p2 * s1 * (KP[12]
                                 + 2. * KP[13] * t1))
    drhoP0dtheta = drhoP0dthetaFresh + drhoP0dthetaSalt                       # :211-212
    dKdtheta = dKdthetaFresh + dKdthetaSalt + dKdthetaPres                    # :213-215
    # :216-220
    return ((bulkMod * bulkMod * drhoP0dtheta
             - bulkMod * p1 * drhoP0dtheta
             - rhoP0 * p1 * dKdtheta)
            / ((bulkMod - p1) * (bulkMod - p1)))


def find_rho_levels(eos, theta, salt, kRef):
    """FIND_RHO_2D for a stack of levels at once: theta/salt [T, n, ny, nx], kRef an integer sequence of length n
    (Fortran pressure-reference level of each slice). Each slice is an independent FIND_RHO_2D call."""
    p = pressure_for_eos(eos, np.asarray(kRef))
    return find_rho(eos, theta, salt, p[None, :, None, None])
