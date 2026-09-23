"""THERMODYNAMICS, TEMP_INTEGRATE, SALT_INTEGRATE (model/src) for the V4r4 flux-forced configuration (plan Task 16b).

Literal translation of c66g `model/src/thermodynamics.F`, `temp_integrate.F`, `salt_integrate.F` and the routines they
call on the branches the ff build executes (CPP: ff `code/CPP_OPTIONS.h`, `GMREDI_OPTIONS.h`; c66g `GAD_OPTIONS.h`;
namelists of the run): CALC_3D_DIFFUSIVITY (+ GMREDI_CALC_DIFF, GGL90_CALC_DIFF), CALC_ADV_FLOW, APPLY_FORCING_T/S
(ff `code/apply_forcing.F`: surface flux, geothermal flux, penetrating shortwave SWFRAC, salt-plume tendency),
GAD_CALC_RHS (GAD_DIFF_X/Y + GMREDI_X/Y/RTRANSPORT, divergence), FREESURF_RESCALE_G, TIMESTEP_TRACER,
GAD_IMPLICIT_R (`mitgcm_jax/core/tracer_implicit.py`) and CYCLE_TRACER.

Two pieces are other kernels and enter as inputs: the residual (Eulerian + GM bolus) flow uFld, vFld, wFld of
GMREDI_RESIDUAL_FLOW (thermodynamics.F:267) and the multi-dimensional DST3 advective tendency written into gT_loc /
gS_loc by GAD_ADVECTION (temp_integrate.F:285, salt_integrate.F:277).

Layout: every 3-D array is `[tile, k, j, i]` with halos, as the Fortran tile arrays. The Fortran k loop
(temp_integrate.F:298, DO k=Nr,1,-1) is vectorised over k: with ALLOW_AUTODIFF, CALC_ADV_FLOW recomputes rTransKp from
wFld (calc_adv_flow.F:92-97), so no quantity is carried from level k+1 to level k except the vertical flux fVer of the
interface below (slot kDown), which is the kUp flux GAD_CALC_RHS computed one iteration earlier: it is the same array
shifted by one level, zero below level Nr (temp_integrate.F:220-225 initialisation).

Branches the V4r4 run does not execute are not ported; `ThermoParams.from_namelists` raises NotImplementedError when
a namelist selects one. In particular, with tempAdvScheme = saltAdvScheme = 30 (DST3) GAD_INIT_FIXED sets
AdamsBashforthGt = AdamsBashforth_T = F (gad_init_fixed.F:146-165): no Adams-Bashforth on T/S, CYCLE_TRACER
(not CYCLE_AB_TRACER) ends the routine.
"""

import dataclasses
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.core.tracer_implicit import gad_implicit_r
from mitgcm_jax.pkgs import salt_plume as salt_plume_mod
from mitgcm_jax.pkgs.gmredi import gmredi_calc_diff
from mitgcm_jax.pkgs.gmredi_transport import (GMTransportParams, gmredi_rtransport, gmredi_xtransport,
                                              gmredi_ytransport)

# GAD.h tracer identifiers
GAD_TEMPERATURE = 1
GAD_SALINITY = 2
# GAD.h advection scheme numbers
ENUM_CENTERED_2ND = 2
ENUM_UPWIND_3RD = 3
ENUM_CENTERED_4TH = 4
ENUM_SOM_PRATHER = 80
ENUM_SOM_LIMITER = 81
# set_grid_factors.F:52-61 (deepAtmosphere = F): deepFacC = deepFac2C = recip_deepFacC = recip_deepFac2C = 1,
# deepFacF = deepFac2F = recip_deepFacF = recip_deepFac2F = 1
DEEPFAC = 1.0
# set_ref_state.F:75-80 (not anelastic): rhoFacC = recip_rhoFacC = rhoFacF = recip_rhoFacF = 1
RHOFAC = 1.0
# ini_grid.F:108 cosFacU = 1 (curvilinear grid)
COSFACU = 1.0
# PARAMS.h:20
PI = 3.14159265358979323844
# swfrac.F:77-79 (Jerlov water type IA, jwtype = 2, swfrac.F:95)
SW_RFAC = 0.62
SW_A1 = 0.6
SW_A2 = 20.0


def _swfrac(swdk, fact=-1.0):
    """swfrac.F:100-108 on one depth (host-side: depths are static grid constants; math.exp is the libm exp gfortran
    calls)."""
    facz = fact * swdk
    if facz < -200.0:
        return 0.0
    return SW_RFAC * math.exp(facz / SW_A1) + (1.0 - SW_RFAC) * math.exp(facz / SW_A2)


@dataclass(frozen=True)
class ThermoParams:
    """Parameters of THERMODYNAMICS. A pytree (registered below): the float scalars, the per-level float tuples and the
    nested SaltPlumeParams are leaves (traced when the params are passed as a jit argument, as the KERNEL_GUIDE
    requires); ints/bools and the GM switches are static."""
    Nr: int
    dTtracerLev: tuple          # ini_parms.F:902 (floats, one per level)
    diffKhT: float
    diffKhS: float
    ivdc_kappa: float
    KbryanLewis79: tuple        # calc_3d_diffusivity.F:80-81, one value per level
    diffKrNrS: tuple            # ini_parms.F:571-575
    rkSign: float               # ini_vertical_grid.F:56
    recip_Cp: float             # apply_forcing.F:508 (ff) recip_Cp = 1/HeatCapacity_Cp
    mass2rUnit: float           # ini_parms.F:1439 (z coordinates) = recip_rhoConst
    swfrac1: tuple              # apply_forcing.F:700-710 (ff): swfracb(1) per level (after SWFRAC)
    swfrac2: tuple              # swfracb(2) per level (0 at k = Nr, apply_forcing.F:707-710)
    swkp1: tuple                # kp1 per level (int)
    useGGL90: bool
    useSALT_PLUME: bool
    implicitDiffusion: bool
    tempImplVertAdv: bool
    saltImplVertAdv: bool
    tempVertAdvScheme: int
    saltVertAdvScheme: int
    tracForcingOutAB: int
    gm: GMTransportParams
    sp: object = None           # salt_plume.SaltPlumeParams when useSALT_PLUME
    tempForcing: bool = True
    saltForcing: bool = True

    @classmethod
    def from_namelists(cls, nml, g):
        """Parameters from the run's namelists (`RunNamelists`) and the static vertical grid of `g` (rF, rhoFacC)."""
        d = "data"
        p1, p3 = "parm01", "parm03"
        Nr = int(len(np.asarray(g.drF)))
        # --- packages (data.pkg; packages_readparms.F: every use* defaults to F)
        pkg = lambda n: bool(nml.get("data.pkg", "packages", n, default=False))
        useGMRedi, useGGL90, useSALT_PLUME = pkg("useGMRedi"), pkg("useGGL90"), pkg("useSALT_PlUME")
        for n in ("useKPP", "useShelfIce", "useOBCS", "useRBCS", "useDOWN_SLOPE", "useBBL", "useFRAZIL",
                  "useICEFRONT", "usePTRACERS", "useLayers", "useMATRIX", "useOPPS", "useSHAP_FILT", "useZONAL_FILT"):
            if pkg(n):
                raise NotImplementedError(f"{n}=T: not compiled in the V4r4 ff build / not ported")
        # --- coordinates (ini_parms.F:409-420): buoyancyRelation 'OCEANIC' => usingZCoords
        if nml.get(d, p1, "buoyancyRelation", default="OCEANIC").upper() != "OCEANIC":  # set_defaults.F:175
            raise NotImplementedError("only z coordinates (buoyancyRelation='OCEANIC') are ported")
        if nml.get(d, "parm04", "deepAtmosphere", default=False):  # set_defaults.F:73
            raise NotImplementedError("deepAtmosphere=T not ported (deepFac factors are 1)")
        if not np.all(np.asarray(g.rhoFacC) == 1.0) or not np.all(np.asarray(g.rhoFacF) == 1.0):
            raise NotImplementedError("anelastic rhoFac != 1 not ported")
        # --- free surface
        nonlinFreeSurf = int(nml.get(d, p1, "nonlinFreeSurf", default=0))  # set_defaults.F:252
        select_rStar = int(nml.get(d, p1, "select_rStar", default=0))  # set_defaults.F:255
        if not (nonlinFreeSurf > 0 and select_rStar > 0):
            raise NotImplementedError("only nonlinFreeSurf>0 with r* (select_rStar>0) is ported "
                                      "(thermodynamics.F:201-211, freesurf_rescale_g.F:42-50)")
        if nml.get(d, p1, "linFSConserveTr", default=False):  # set_defaults.F:250
            raise NotImplementedError("linFSConserveTr=T not ported (thermodynamics.F:139)")
        # --- stepping flags (set_defaults.F:191-196)
        for n in ("tempStepping", "saltStepping", "tempAdvection", "saltAdvection"):
            if not nml.get(d, p1, n, default=True):
                raise NotImplementedError(f"{n}=F not ported")
        tempForcing = bool(nml.get(d, p1, "tempForcing", default=True))  # set_defaults.F:193
        saltForcing = bool(nml.get(d, p1, "saltForcing", default=True))  # set_defaults.F:196
        # --- advection schemes and derived GAD flags (gad_init_fixed.F:121-165)
        multiDimAdvection = bool(nml.get(d, p1, "multiDimAdvection", default=True))  # set_defaults.F:223
        tAdv = int(nml.get(d, p1, "tempAdvScheme", default=2))  # set_defaults.F:221
        sAdv = int(nml.get(d, p1, "saltAdvScheme", default=2))  # set_defaults.F:222
        tVert = int(nml.get(d, p1, "tempVertAdvScheme", default=0)) or tAdv  # ini_parms.F:386,471
        sVert = int(nml.get(d, p1, "saltVertAdvScheme", default=0)) or sAdv  # ini_parms.F:387,472
        doAB_onGtGs = bool(nml.get(d, p3, "doAB_onGtGs", default=True))  # set_defaults.F:309
        for adv in (tAdv, sAdv):
            if ENUM_SOM_PRATHER <= adv <= ENUM_SOM_LIMITER:  # gad_init_fixed.F:121-126
                raise NotImplementedError("2nd-order-moment advection not compiled (GAD_ALLOW_TS_SOM_ADV undef)")
            multiDim = multiDimAdvection and adv not in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH)
            if not multiDim:  # gad_init_fixed.F:129-139 ; calcAdvection = .NOT.MultiDimAdvec (temp_integrate.F:297)
                raise NotImplementedError("advection inside GAD_CALC_RHS (calcAdvection=T) is not ported")
            abGt = adv in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH)  # gad_init_fixed.F:150-159
            if abGt:  # AdamsBashforthGt (doAB_onGtGs) or AdamsBashforth_T (.NOT.doAB_onGtGs), gad_init_fixed.F:160
                raise NotImplementedError(f"Adams-Bashforth on T/S (scheme {adv}, doAB_onGtGs={doAB_onGtGs}) "
                                          "is not ported: V4r4 uses DST3 (30), which sets all four AB flags F")
        forcing_In_AB = bool(nml.get(d, p3, "forcing_In_AB", default=True))  # ini_parms.F:827
        if nml.has(d, p3, "tracForcingOutAB"):
            tracForcingOutAB = int(nml.get(d, p3, "tracForcingOutAB"))
        else:  # set_defaults.F:307 UNSET_I -> ini_parms.F:939-942
            tracForcingOutAB = 0 if forcing_In_AB else 1
        # --- diffusion
        for n in ("diffK4T", "diffK4S"):
            if float(nml.get(d, p1, n, default=0.0)) != 0.0:  # set_defaults.F:154-155
                raise NotImplementedError(f"{n} != 0 (bi-harmonic tracer diffusion) not ported")
        for n in ("diffKr4T", "diffKr4S"):
            v = nml.get(d, p1, n, default=[0.0], array=True)  # set_defaults.F:170-171
            if any(float(x) != 0.0 for x in v):
                raise NotImplementedError(f"{n} != 0 not ported")
        implicitDiffusion = bool(nml.get(d, p1, "implicitDiffusion", default=False))  # set_defaults.F:204
        tImpl = bool(nml.get(d, p1, "tempImplVertAdv", default=False))  # set_defaults.F:208
        sImpl = bool(nml.get(d, p1, "saltImplVertAdv", default=False))  # set_defaults.F:209
        if not (implicitDiffusion and tImpl and sImpl):
            raise NotImplementedError("only implicitDiffusion=T with tempImplVertAdv=saltImplVertAdv=T is ported "
                                      "(GAD_IMPLICIT_R path, temp_integrate.F:484)")
        diffKhT = float(nml.get(d, p1, "diffKhT", default=0.0))  # set_defaults.F:152
        diffKhS = float(nml.get(d, p1, "diffKhS", default=0.0))  # set_defaults.F:153
        ivdc_kappa = float(nml.get(d, p1, "ivdc_kappa", default=0.0))  # set_defaults.F:216
        if float(nml.get(d, p3, "cAdjFreq", default=0.0)) != 0.0:  # set_defaults.F:320 (tracers_correction.py)
            raise NotImplementedError("cAdjFreq != 0 (CONVECTIVE_ADJUSTMENT) not ported")
        # Bryan & Lewis 1979 background profile (calc_3d_diffusivity.F:80-81), defaults set_defaults.F:158-161
        s = float(nml.get(d, p1, "diffKrBL79surf", default=0.0))
        dp = float(nml.get(d, p1, "diffKrBL79deep", default=0.0))
        scl = float(nml.get(d, p1, "diffKrBL79scl", default=200.0))
        Ho = float(nml.get(d, p1, "diffKrBL79Ho", default=-2000.0))
        # (ALLOW_BL79_LAT_VARY is undef in the ff CPP_OPTIONS.h:41: BL79LatVary is read but not used)
        rF = np.asarray(g.rF, dtype=np.float64)
        KBL = tuple(s + (dp - s) * (math.atan(-(float(rF[k]) - Ho) / scl) / PI + 0.5) for k in range(Nr))
        # diffKrNrS (ini_parms.F:548-583): used by GGL90_CALC_DIFF (ggl90_calc_diff.F:53-55)
        if nml.has(d, p1, "diffKrNrS"):
            diffKrNrS = tuple(float(x) for x in nml.get(d, p1, "diffKrNrS", array=True))
        else:
            diffKrS = None
            for n in ("diffKrS", "diffKzS", "diffKpS"):  # ini_parms.F:550-552
                if nml.has(d, p1, n):
                    diffKrS = float(nml.get(d, p1, n))
                    break
            if diffKrS is None:
                raise NotImplementedError("diffKrNrS from diffKrNrT defaults (ini_parms.F:577-582) not ported")
            diffKrNrS = (diffKrS,) * Nr  # ini_parms.F:571-575
        # --- time step (ini_parms.F:875-902)
        dTtr = float(nml.get(d, p3, "deltaTtracer", default=0.0))  # ini_parms.F:826
        lev = [float(x) for x in nml.get(d, p3, "dTtracerLev", default=[0.0] * Nr, array=True)]
        lev = (lev + [0.0] * Nr)[:Nr]
        if lev[0] != 0.0:
            dTtr = lev[0]  # ini_parms.F:882
        dTtracerLev = tuple(x if x != 0.0 else dTtr for x in lev)  # ini_parms.F:902
        # --- physical constants
        Cp = float(nml.get(d, p1, "HeatCapacity_Cp", default=3994.0))  # set_defaults.F:173
        rhoNil = float(nml.get(d, p1, "rhoNil", default=999.8))  # set_defaults.F:106
        rhoConst = float(nml.get(d, p1, "rhoConst", default=rhoNil))  # ini_parms.F:445
        recip_rhoConst = 1.0 / rhoConst  # ini_parms.F:640
        # --- penetrating shortwave: SWFRAC on |rF(k)|, |rF(k+1)| (apply_forcing.F:700-710, ff)
        sw = []
        for k in range(1, Nr + 1):
            b1 = _swfrac(abs(float(rF[k - 1])))  # apply_forcing.F:700-705
            b2 = _swfrac(abs(float(rF[k])))
            kp1 = k + 1  # apply_forcing.F:706-710
            if k == Nr:
                kp1 = k
                b2 = 0.0
            sw.append((b1, b2, kp1))
        return cls(Nr=Nr, dTtracerLev=dTtracerLev, diffKhT=diffKhT, diffKhS=diffKhS, ivdc_kappa=ivdc_kappa,
                   KbryanLewis79=KBL, diffKrNrS=diffKrNrS, rkSign=-1.0, recip_Cp=1.0 / Cp,
                   mass2rUnit=recip_rhoConst, swfrac1=tuple(x[0] for x in sw),
                   swfrac2=tuple(x[1] for x in sw), swkp1=tuple(x[2] for x in sw), useGGL90=useGGL90,
                   useSALT_PLUME=useSALT_PLUME,
                   implicitDiffusion=implicitDiffusion, tempImplVertAdv=tImpl, saltImplVertAdv=sImpl,
                   tempVertAdvScheme=tVert, saltVertAdvScheme=sVert, tracForcingOutAB=tracForcingOutAB,
                   gm=GMTransportParams.from_namelists(nml),
                   sp=salt_plume_mod.SaltPlumeParams.from_namelists(nml) if useSALT_PLUME else None,
                   tempForcing=tempForcing, saltForcing=saltForcing)


_THERMO_DATA = ("dTtracerLev", "diffKhT", "diffKhS", "ivdc_kappa", "KbryanLewis79", "diffKrNrS", "rkSign",
                "recip_Cp", "mass2rUnit", "swfrac1", "swfrac2", "sp")
# params_io.params_pytree makes only `float` fields leaves; the per-level tuples and the nested SaltPlumeParams (a
# pytree itself) must be leaves too, so the registration lists them explicitly
jax.tree_util.register_dataclass(
    ThermoParams, data_fields=list(_THERMO_DATA),
    meta_fields=[f.name for f in dataclasses.fields(ThermoParams) if f.name not in _THERMO_DATA])


def _col(v):
    """Per-level values -> [1, Nr, 1, 1] array."""
    return jnp.asarray(v, dtype=jnp.float64)[None, :, None, None]


def klowc_from_mask(maskC):
    """kLowC (ini_masks_etc.F:192,200): deepest level with hFacC .NE. 0 (maskC = 1 there, ini_masks_etc.F:485), 0 if
    none; returned 1-based as in Fortran, [T, j, i] int."""
    Nr = maskC.shape[1]
    k = jnp.arange(1, Nr + 1)[None, :, None, None]
    return jnp.max(jnp.where(maskC != 0.0, k, 0), axis=1)


# --------------------------------------------------------------------------------------------------------------------
# thermodynamics.F set-up
def recip_hfac_new(recip_hFacC, rStarExpC):
    """thermodynamics.F:189-211 (NONLIN_FRSURF, select_rStar>0): recip_hFacNew = recip_hFacC / rStarExpC, all points."""
    return recip_hFacC / rStarExpC[:, None]


def calc_3d_diffusivity(p, g, trIdentity, IVDConvCount, diffKr, GGL90diffKr, Kwz):
    """calc_3d_diffusivity.F (called with iMin=0,iMax=sNx+1,jMin=0,jMax=sNy+1, temp_integrate.F:245): net vertical
    diffusivity kappaRk of one tracer, [T, Nr, j, i]."""
    L = g.layout
    # calc_3d_diffusivity.F:78-109 (trUseKPP = F): IVDConvCount*ivdc_kappa + KbryanLewis79, all points
    kap = IVDConvCount * p.ivdc_kappa + _col(p.KbryanLewis79)
    # calc_3d_diffusivity.F:110-135 (ALLOW_3D_DIFFKR): + diffKr for temperature and salinity
    if trIdentity not in (GAD_TEMPERATURE, GAD_SALINITY):
        raise NotImplementedError("CALC_3D_DIFFUSIVITY: passive tracers not ported")
    kap = kap + diffKr
    J, I = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
    if p.gm.useGMRedi:
        # calc_3d_diffusivity.F:193-200 GMREDI_CALC_DIFF (pkg/gmredi kernel): + Kwz*maskInC on 0..sNx+1, 0..sNy+1
        kap = gmredi_calc_diff(g, kap, Kwz, 0, L.sNx + 1, 0, L.sNy + 1)
    if p.useGGL90:
        # ggl90_calc_diff.F:48-58: + (GGL90diffKr - diffKrNrS(k)) on iMin..iMax, jMin..jMax
        kap = kap.at[..., J, I].set(kap[..., J, I] + (GGL90diffKr[..., J, I] - _col(p.diffKrNrS)))
    return kap


# --------------------------------------------------------------------------------------------------------------------
def calc_adv_flow(g, uFld, vFld, wFld, hFacW, hFacS):
    """calc_adv_flow.F for every level (ALLOW_AUTODIFF: rTransKp recomputed from wFld, calc_adv_flow.F:92-97).
    Returns dict xA, yA, uTrans, vTrans, rTrans, rTransKp, maskUp, each [T, Nr, j, i]."""
    drF = _col(g.drF)
    maskC = g.maskC
    # calc_adv_flow.F:73-80
    xA = g.dyG[:, None] * DEEPFAC * drF * hFacW
    yA = g.dxG[:, None] * DEEPFAC * drF * hFacS
    rA = g.rA[:, None]
    # calc_adv_flow.F:83-104: rTransKp = wFld(k+1)*rA*maskC(k)*maskC(k+1)*deepFac2F(k+1)*rhoFacF(k+1); 0 at k=Nr
    rkp = wFld[:, 1:] * rA * maskC[:, :-1] * maskC[:, 1:] * DEEPFAC * RHOFAC
    rTransKp = jnp.concatenate([rkp, jnp.zeros_like(wFld[:, :1])], axis=1)
    # calc_adv_flow.F:108-113
    uTrans = uFld * xA * RHOFAC
    vTrans = vFld * yA * RHOFAC
    # calc_adv_flow.F:116-134: maskUp = maskC(k-1)*maskC(k), rTrans = wFld*rA*maskUp*deepFac2F*rhoFacF; 0 at k=1
    mUp = maskC[:, :-1] * maskC[:, 1:]
    maskUp = jnp.concatenate([jnp.zeros_like(maskC[:, :1]), mUp], axis=1)
    rt = wFld[:, 1:] * rA * mUp * DEEPFAC * RHOFAC
    rTrans = jnp.concatenate([jnp.zeros_like(wFld[:, :1]), rt], axis=1)
    return dict(xA=xA, yA=yA, uTrans=uTrans, vTrans=vTrans, rTrans=rTrans, rTransKp=rTransKp, maskUp=maskUp)


# --------------------------------------------------------------------------------------------------------------------
def apply_forcing_t(p, g, recip_hFacC, surfaceForcingT, Qsw, geothermalFlux, kLowC=None):
    """APPLY_FORCING_T (ff code/apply_forcing.F:421-791) for every level: gtForc [T, Nr, j, i] (zero outside
    0..sNx+1, 0..sNy+1; temp_integrate.F:329-333 zeroes gtForc before the call).

    Terms (ocean, z coordinates, kSurface = 1, apply_forcing.F:499-506): surface flux (:651-658), geothermal flux
    (:682-695, ALLOW_GEOTHERMAL_FLUX), penetrating shortwave (:697-721, SHORTWAVE_HEATING); SALT_PLUME_TENDENCY_APPLY_T
    (:746-752) is empty without SALT_PLUME_VOLUME (salt_plume_tendency_apply_t.F); ADDFLUID / FRICTION_HEATING are
    undef in the ff CPP_OPTIONS.h, AIM/ATM_PHYS/FIZHI/FRAZIL/SHELFICE/ICEFRONT/RBCS/OBCS/BBL not compiled."""
    L = g.layout
    Nr = p.Nr
    J, I = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
    rdrF = _col(g.recip_drF)
    rhC = recip_hFacC[..., J, I]
    gt = jnp.zeros_like(recip_hFacC)
    val = gt[..., J, I]
    # apply_forcing.F:651-658 (k .EQ. kSurface = 1)
    val = val.at[:, 0].set(val[:, 0] + surfaceForcingT[:, J, I] * rdrF[:, 0] * rhC[:, 0])
    # apply_forcing.F:682-695: IF ( k.EQ.kLowC ) += geothermalFlux*recip_Cp*mass2rUnit*recip_drF(k)*recip_hFacC
    if kLowC is None:
        kLowC = klowc_from_mask(g.maskC)
    kk = jnp.arange(1, Nr + 1)[None, :, None, None]
    geo = geothermalFlux[:, None, J, I] * p.recip_Cp * p.mass2rUnit * rdrF * rhC
    val = jnp.where(kk == kLowC[:, None, J, I], val + geo, val)
    # apply_forcing.F:700-719: penetrating shortwave
    sw1 = _col(p.swfrac1)
    sw2 = _col(p.swfrac2)
    kp1 = np.array(p.swkp1) - 1
    mC = g.maskC[..., J, I]
    val = val - Qsw[:, None, J, I] * (sw1 * mC - sw2 * mC[:, kp1]) * p.recip_Cp * p.mass2rUnit * rdrF * rhC
    return gt.at[..., J, I].set(val)


def apply_forcing_s(p, g, recip_hFacC, surfaceForcingS, saltPlumeDepth, saltPlumeFlux):
    """APPLY_FORCING_S (ff code/apply_forcing.F:797-1023) for every level: gsForc [T, Nr, j, i] (zero outside
    0..sNx+1, 0..sNy+1; salt_integrate.F:321-325 zeroes gsForc before the call). Terms: surface flux (:932-939,
    kSurface = 1) and SALT_PLUME_TENDENCY_APPLY_S (:978-984, useSALT_PLUME; pkg/salt_plume kernel
    `salt_plume.salt_plume_tendency_apply_s`, one call per level as in the Fortran k loop, with the caller's range
    iMin..iMax = 0..sNx+1, jMin..jMax = 0..sNy+1, salt_integrate.F:165-168)."""
    L = g.layout
    J, I = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
    gs = jnp.zeros_like(recip_hFacC)
    rdrF = _col(g.recip_drF)
    # apply_forcing.F:932-939 (k .EQ. kSurface = 1)
    gs = gs.at[:, 0, J, I].set(gs[:, 0, J, I] + surfaceForcingS[:, J, I] * rdrF[:, 0] * recip_hFacC[:, 0, J, I])
    if p.useSALT_PLUME:
        levels = []
        for k in range(1, p.Nr + 1):
            levels.append(salt_plume_mod.salt_plume_tendency_apply_s(
                p.sp, g, gs[:, k - 1], k, saltPlumeDepth, saltPlumeFlux, g.maskC[:, k - 1], recip_hFacC[:, k - 1],
                0, L.sNx + 1, 0, L.sNy + 1))
        gs = jnp.stack(levels, axis=1)
    return gs


# --------------------------------------------------------------------------------------------------------------------
def gad_calc_rhs(p, g, trIdentity, diffKh, flow, tracer, gTracer, recip_hFacC, K, implicitAdvection=True):
    """GAD_CALC_RHS (pkg/generic_advdiff/gad_calc_rhs.F) for every level k (called from temp_integrate.F:349 with
    iMin=0, iMax=sNx+1, jMin=0, jMax=sNy+1): adds the explicit diffusive + GM/Redi flux divergence to gTracer.

    V4r4 branch: calcAdvection = F (multi-dim advection done in GAD_ADVECTION), applyAB_onTracer = F, diffK4 = 0,
    implicitDiffusion = T (explicit vertical diffusion flux 0), trUseDiffKr4 = F, trUseKPP = F, trUseGMRedi = useGMRedi.
    flow: output of `calc_adv_flow`; K: dict of the GM/Redi tensor Kux, Kvy, Kuz, Kvz, Kwx, Kwy."""
    L = g.layout
    iMin, iMax, jMin, jMax = 0, L.sNx + 1, 0, L.sNy + 1
    xA, yA, maskUp = flow["xA"], flow["yA"], flow["maskUp"]
    uTrans, vTrans, rTrans, rTransKp = flow["uTrans"], flow["vTrans"], flow["rTrans"], flow["rTransKp"]
    # gad_calc_rhs.F:168-171
    advFac = 0.0  # calcAdvection = F
    rAdvFac = p.rkSign * advFac
    if implicitAdvection:
        rAdvFac = p.rkSign
    localT = tracer  # gad_calc_rhs.F:191-197 (applyAB_onTracer = F)
    zeros = jnp.zeros_like(tracer)
    # --- x: gad_calc_rhs.F:208-212 fZon = 0; :299-300 GAD_DIFF_X (gad_diff_x.F:51-59)
    dT = localT[..., :, 1:] - localT[..., :, :-1]
    dfx = zeros.at[..., :, 1:].set(-(diffKh * xA[..., :, 1:] * g.recip_dxC[:, None, :, 1:] * DEEPFAC * dT * COSFACU))
    # gad_calc_rhs.F:316-322 GMREDI_XTRANSPORT(iMin, iMax+1, jMin, jMax)
    dfx = gmredi_xtransport(p.gm, g, iMin, iMax + 1, jMin, jMax, xA, tracer, dfx, K["Kux"], K["Kuz"])
    fZon = zeros + dfx * RHOFAC  # gad_calc_rhs.F:325-329
    # --- y: gad_calc_rhs.F:348-352, :439-440 GAD_DIFF_Y (gad_diff_y.F:51-63; ISOTROPIC_COS_SCALING undef)
    dT = localT[..., 1:, :] - localT[..., :-1, :]
    dfy = zeros.at[..., 1:, :].set(-(diffKh * yA[..., 1:, :] * g.recip_dyC[:, None, 1:, :] * DEEPFAC * dT))
    # gad_calc_rhs.F:456-462 GMREDI_YTRANSPORT(iMin, iMax, jMin, jMax+1)
    dfy = gmredi_ytransport(p.gm, g, iMin, iMax, jMin, jMax + 1, yA, tracer, dfy, K["Kvy"], K["Kvz"])
    fMer = zeros + dfy * RHOFAC  # gad_calc_rhs.F:465-469
    # --- r: gad_calc_rhs.F:177 fVerT(kUp) = 0; :609-614 implicitDiffusion => df = 0;
    # :625-631 GMREDI_RTRANSPORT(iMin, iMax, jMin, jMax); :634-638 fVerT(kUp) += df*maskUp
    dfr = gmredi_rtransport(p.gm, g, iMin, iMax, jMin, jMax, tracer, zeros, K["Kwx"], K["Kwy"])
    fVer = zeros + dfr * maskUp
    # fVerT(kDown) = flux at interface k+1 (computed at level k+1); 0 below level Nr (temp_integrate.F:220-225)
    fVerDown = jnp.concatenate([fVer[:, 1:], jnp.zeros_like(fVer[:, :1])], axis=1)
    # --- divergence gad_calc_rhs.F:773-787 on j=1-OLy..sNy+OLy-1, i=1-OLx..sNx+OLx-1
    J, I = L.js(1 - L.OLy, L.sNy + L.OLy - 1), L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    Jp, Ip = L.js(2 - L.OLy, L.sNy + L.OLy), L.is_(2 - L.OLx, L.sNx + L.OLx)
    mIC = g.maskInC[:, None, J, I]
    fac = recip_hFacC[..., J, I] * _col(g.recip_drF) * g.recip_rA[:, None, J, I] * DEEPFAC * RHOFAC
    X = ((fZon[..., J, Ip] - fZon[..., J, I]) * mIC
         + (fMer[..., Jp, I] - fMer[..., J, I]) * mIC
         + (fVerDown[..., J, I] - fVer[..., J, I]) * p.rkSign
         - localT[..., J, I] * ((uTrans[..., J, Ip] - uTrans[..., J, I]) * advFac
                                + (vTrans[..., Jp, I] - vTrans[..., J, I]) * advFac
                                + (rTransKp[..., J, I] - rTrans[..., J, I]) * rAdvFac) * mIC)
    return gTracer.at[..., J, I].set(gTracer[..., J, I] - fac * X)


def freesurf_rescale_g(gTracer, rStarExpC):
    """freesurf_rescale_g.F:42-50 (nonlinFreeSurf>0, select_rStar>0): gTracer / rStarExpC, all points, every level."""
    return gTracer / rStarExpC[:, None]


def timestep_tracer(p, tracer, gTracer):
    """timestep_tracer.F:54-69: gTracer <= tracer + deltaTLev(k)*gTracer, all points."""
    return tracer + _col(p.dTtracerLev) * gTracer


# --------------------------------------------------------------------------------------------------------------------
def tracer_integrate(p, g, trIdentity, tracer, gTr_adv, uFld, vFld, wFld, recip_hFacNew, fields, forc):
    """TEMP_INTEGRATE / SALT_INTEGRATE (temp_integrate.F, salt_integrate.F: identical but for names and the forcing
    routine) after GAD_ADVECTION. `gTr_adv`: gT_loc after GAD_ADVECTION; `forc`: gtForc/gsForc of every level.

    Returns dict: kappaRk, gExplicit (before TIMESTEP_TRACER, dump T11/T21), gStep (after, T12/T22), gImpl (after
    GAD_IMPLICIT_R, T13/T23), tracer (after CYCLE_TRACER, T02/T03)."""
    temp = trIdentity == GAD_TEMPERATURE
    diffKh = p.diffKhT if temp else p.diffKhS
    implAdv = p.tempImplVertAdv if temp else p.saltImplVertAdv
    vertScheme = p.tempVertAdvScheme if temp else p.saltVertAdvScheme
    doForcing = p.tempForcing if temp else p.saltForcing
    f = fields
    # temp_integrate.F:227-233 (ALLOW_AUTODIFF) kappaRk = 0; :245-249 CALC_3D_DIFFUSIVITY
    kappaRk = calc_3d_diffusivity(p, g, trIdentity, f["IVDConvCount"], f["diffKr"], f["GGL90diffKr"], f["Kwz"])
    # temp_integrate.F:298-452, vectorised over k
    flow = calc_adv_flow(g, uFld, vFld, wFld, f["hFacW"], f["hFacS"])
    K = {n: f[n] for n in ("Kux", "Kvy", "Kuz", "Kvz", "Kwx", "Kwy")}
    gT = gad_calc_rhs(p, g, trIdentity, diffKh, flow, tracer, gTr_adv, f["recip_hFacC"], K, implicitAdvection=implAdv)
    # temp_integrate.F:381-387 forcing inside AB (tracForcingOutAB .NE. 1); AB not active (AdamsBashforthGt = F)
    if doForcing and p.tracForcingOutAB != 1:
        gT = gT + forc
    # temp_integrate.F:411-417 forcing outside AB
    if doForcing and p.tracForcingOutAB == 1:
        gT = gT + forc
    # temp_integrate.F:419-424 FREESURF_RESCALE_G (the gtNm rescaling at :425-447 needs AdamsBashforthGt = T)
    gT = freesurf_rescale_g(gT, f["rStarExpC"])
    gExplicit = gT
    # temp_integrate.F:475-479 TIMESTEP_TRACER
    gStep = timestep_tracer(p, tracer, gT)
    # temp_integrate.F:484-499 GAD_IMPLICIT_R
    gImpl = gad_implicit_r(p, g, implAdv, vertScheme, kappaRk, recip_hFacNew, wFld, gStep)
    # temp_integrate.F:541-547 CYCLE_TRACER (cycle_tracer.F:47-53): tracer = gTracer, all points
    return dict(kappaRk=kappaRk, gExplicit=gExplicit, gStep=gStep, gImpl=gImpl, tracer=gImpl)


def thermodynamics(p, g, fields, uFld, vFld, wFld, gT_adv, gS_adv):
    """THERMODYNAMICS (thermodynamics.F:167-382) on all tiles at once.

    fields (Fortran names, values at the time THERMODYNAMICS runs): theta, salt, recip_hFacC, hFacW, hFacS (after
    UPDATE_R_STAR / CALC_R_STAR), rStarExpC (CALC_R_STAR), surfaceForcingT, surfaceForcingS, Qsw, geothermalFlux,
    saltPlumeDepth, saltPlumeFlux, IVDConvCount, GGL90diffKr, Kwx, Kwy, Kwz, Kux, Kvy, Kuz, Kvz (DO_OCEANIC_PHYS),
    diffKr (3-D, static). uFld, vFld, wFld: GMREDI_RESIDUAL_FLOW output; gT_adv, gS_adv: GAD_ADVECTION outputs.

    Returns dict with theta, salt (new), recip_hFacNew, and the T*/S* intermediates of `tracer_integrate`."""
    f = fields
    # thermodynamics.F:189-211
    recip_hFacNew = recip_hfac_new(f["recip_hFacC"], f["rStarExpC"])
    gtForc = apply_forcing_t(p, g, f["recip_hFacC"], f["surfaceForcingT"], f["Qsw"], f["geothermalFlux"],
                             f.get("kLowC"))
    gsForc = apply_forcing_s(p, g, f["recip_hFacC"], f["surfaceForcingS"], f["saltPlumeDepth"], f["saltPlumeFlux"])
    # thermodynamics.F:315-324 TEMP_INTEGRATE, :326-335 SALT_INTEGRATE
    T = tracer_integrate(p, g, GAD_TEMPERATURE, f["theta"], gT_adv, uFld, vFld, wFld, recip_hFacNew, f, gtForc)
    S = tracer_integrate(p, g, GAD_SALINITY, f["salt"], gS_adv, uFld, vFld, wFld, recip_hFacNew, f, gsForc)
    out = dict(theta=T["tracer"], salt=S["tracer"], recip_hFacNew=recip_hFacNew, gtForc=gtForc, gsForc=gsForc)
    out.update({"T_" + k: v for k, v in T.items() if k != "tracer"})
    out.update({"S_" + k: v for k, v in S.items() if k != "tracer"})
    return out
