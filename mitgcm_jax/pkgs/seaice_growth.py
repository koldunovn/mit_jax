"""pkg/seaice thermodynamics (plan M2.3): SEAICE_GROWTH (V4r4 override) with SEAICE_BUDGET_OCEAN and SEAICE_SOLVE4TEMP.

Fortran (line numbers of these files are cited as `seaice_growth.F:NNN`, `seaice_solve4temp.F:NNN`, ...):
    ECCO-v4-Configurations/ECCOv4 Release 4/code/seaice_growth.F   (V4r4 override, replaces c66g's; compiled -O0)
    MITgcm_c66g/pkg/seaice/seaice_solve4temp.F, seaice_budget_ocean.F (c66g; -O3, -ffp-contract=off)
The active branches were read from the preprocessed build (reference/build/full_serial13_*/bld/seaice_growth.f) and
from `cpp -traditional` with line markers run on the three sources with the build's flags (active .F line numbers).

CPP options of the full V4r4 build (V4r4 code/SEAICE_OPTIONS.h, CPP_OPTIONS.h, EXF_OPTIONS.h, AUTODIFF_OPTIONS.h):
    SEAICE_EXTERNAL_FLUXES defined     -> SEAICE_BUDGET_OCEAN copies Qnet/Qsw (seaice_budget_ocean.F:108-110)
    SEAICE_ITD, SEAICE_VARIABLE_SALINITY, ALLOW_SITRACER, SEAICE_GREASE, SEAICE_CAP_SUBLIM, SEAICE_DISABLE_SUBLIM,
    SEAICE_DISABLE_HEATCONSFIX, SEAICE_MODIFY_GROWTH_ADJ, SEAICE_DEBUG, EXF_SEAICE_FRACTION   undefined
    SHORTWAVE_HEATING, ATMOSPHERIC_LOADING, ALLOW_BALANCE_FLUXES, ALLOW_RUNOFF, ALLOW_ATM_TEMP,
    ALLOW_DOWNWARD_RADIATION, ALLOW_SALT_PLUME, ALLOW_AUTODIFF_TAMC   defined
    SALT_PLUME_SPLIT_BASIN, SALT_PLUME_IN_LEADS   undefined (pkg/salt_plume/SALT_PLUME_OPTIONS.h)
    nITD = 7 (SEAICE_SIZE.h:27, no SEAICE_ITD); data.seaice SEAICE_multDim = 1: only category 1 of the 7 allocated
    TICES / *Mult levels is computed; levels 2..nITD are never touched by the V4r4 code path.
Runtime switches (data, data.seaice, data.exf, data.salt_plume) select one branch at every IF; the others raise
NotImplementedError in SeaiceGrowthParams.from_namelists (usePW79thermodynamics=T, growMeltByConv=F,
areaGainFormula=1, areaLossFormula=2, useFlooding=T, heatConsFix=T, mcPheeStepFunc=F, useMaykutSatVapPoly=F,
postSolvTempIter=2, snowPrecipFile=' ', useRelativeWind=F, temp_EvPrRn unset + useRealFreshWaterFlux +
nonlinFreeSurf>0, balanceEmPmR=balanceQnet=F, buoyancyRelation='OCEANIC'). SaltPlumeSouthernOcean (T in V4r4) is a
static switch with both branches (seaice_growth.F:2035-2038).

Not ported (output only): the DIAGNOSTICS_FILL calls, d_AREAbyATM/ICE/OCN (diagnostics only), SItflux / SIatmQnt /
SIatmFW (diagnostics and the balance sums, which do nothing with balanceEmPmR = balanceQnet = F: the global sums are
skipped, seaice_growth.F:2612-2694), and the TAF store directives (no forward effect).

Layout: every routine loops DO J=1,sNy / DO I=1,sNx (interior only). Inputs are full `[T, ny, nx]` arrays with halos;
outputs keep the input halo values (the Fortran never writes them). The stage functions below (growth_pre_budget,
budget_ocean, solve4temp, heat_stocks, thickness_updates, ocean_forcing) follow the dump stages H01..H06 / I04 of
reference/jaxdump/SUBSTEPS.md and work on interior arrays `[T, sNy, sNx]`; `seaice_growth` composes them.

Transcendental functions: SQRT is correctly rounded in both codes. EXP in SEAICE_SOLVE4TEMP (penetrating shortwave,
saturation humidity and its derivative: 22 calls per ice point) is glibc 2.28's exp (FMA ifunc variant on the AMD EPYC
nodes, not correctly rounded), which XLA's exp misses by 1-2 ulp at ~14 % of the arguments. `glibc_exp` below is a
bit-for-bit emulation (pure JAX, custom_jvp); with it every stage is bitwise (3 iterations). `solve4temp` /
`seaice_growth` take the exp as `expf` (default `glibc_exp`; `jnp.exp` gives <= 2.6e-14 relative in SOLVE4TEMP's
outputs and is ~20x cheaper per call). LOG(10) and EXP(aa2*LN10) are host constants (glibc == correctly rounded for
them, so it does not matter whether gfortran -O3 folded them).
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mitgcm_jax.params_io import params_pytree

UNSET_RL = 1.234567e5  # EEPARAMS.h:90


def _swfrac(swdk, fact=-1.0):
    """SWFRAC (swfrac.F:100-108) for one depth, Jerlov type IA (jwtype = 2, swfrac.F:95; rfac 0.62, a1 0.6, a2 20.0,
    swfrac.F:77-79). Host-side (a static grid constant); math.exp is the glibc exp gfortran calls."""
    facz = fact * swdk
    if facz < -200.0:
        return 0.0
    return 0.62 * math.exp(facz / 0.6) + (1.0 - 0.62) * math.exp(facz / 20.0)


@params_pytree
@dataclass(frozen=True)
class SeaiceGrowthParams:
    """Parameters of SEAICE_GROWTH / SEAICE_SOLVE4TEMP / SEAICE_BUDGET_OCEAN. float fields are pytree leaves (pass the
    params as a jit argument), everything else is static."""
    # --- seaice_growth.F
    deltaTtherm: float      # SEAICE_deltaTtherm
    rhoIce: float           # SEAICE_rhoIce
    rhoSnow: float          # SEAICE_rhoSnow
    lhFusion: float         # SEAICE_lhFusion
    lhEvap: float           # SEAICE_lhEvap
    area_reg: float         # SEAICE_area_reg
    hice_reg: float         # SEAICE_hice_reg
    area_max: float         # SEAICE_area_max
    EPS: float              # SEAICE_EPS
    tempFrz0: float         # SEAICE_tempFrz0
    dTempFrz_dS: float      # SEAICE_dTempFrz_dS
    mcPheePiston: float     # SEAICE_mcPheePiston
    mcPheeTaper: float      # SEAICE_mcPheeTaper
    frazilFrac: float       # SEAICE_frazilFrac
    HO: float
    HO_south: float
    salt0: float            # SEAICE_salt0
    facOpenGrow: float
    facOpenMelt: float
    SWFracB: float
    SPsalFRAC: float        # salt_plume
    celsius2K: float
    HeatCapacity_Cp: float
    rhoConst: float
    recip_rhoConst: float
    rhoConstFresh: float
    # --- seaice_solve4temp.F
    dalton: float           # SEAICE_dalton
    cpAir: float            # SEAICE_cpAir
    rhoAir: float           # SEAICE_rhoAir
    iceConduct: float       # SEAICE_iceConduct
    snowConduct: float      # SEAICE_snowConduct
    snowThick: float        # SEAICE_snowThick
    shortwave: float        # SEAICE_shortwave
    wetAlbTemp: float       # SEAICE_wetAlbTemp
    dryIceAlb: float
    wetIceAlb: float
    drySnowAlb: float
    wetSnowAlb: float
    dryIceAlb_south: float
    wetIceAlb_south: float
    drySnowAlb_south: float
    wetSnowAlb_south: float
    snow_emiss: float       # SEAICE_snow_emiss
    ice_emiss: float        # SEAICE_ice_emiss
    boltzmann: float        # SEAICE_boltzmann
    MIN_LWDOWN: float
    MIN_ATEMP: float
    aa1: float              # seaice_solve4temp.F:172-180 (constants of the saturation vapour pressure)
    aa2: float
    bb1: float
    Ppascals: float
    lnTEN: float
    cc1: float
    cc2: float
    # --- static
    multDim: int = 1        # SEAICE_multDim
    nITD: int = 7           # SEAICE_SIZE.h:27
    SEAICE_PDF: tuple = (1.0,)
    IMAX_TICE: int = 10
    postSolvTempIter: int = 2
    SaltPlumeSouthernOcean: bool = True
    useSALT_PLUME: bool = True

    @classmethod
    def from_namelists(cls, nml, g):
        """Values from data / data.seaice / data.exf / data.salt_plume, else the Fortran defaults (cited).
        g: Grid (static vertical grid: drF, rF for SWFracB)."""
        S = "data.seaice"
        P1 = "seaice_parm01"

        def s(key, default):
            return nml.get(S, P1, key, default=default)

        def d(group, key, default):
            return nml.get("data", group, key, default=default)

        pkgs = "data.pkg"
        if not bool(nml.get(pkgs, "packages", "useSEAICE", default=False)):
            raise NotImplementedError("useSEAICE=F: SEAICE_GROWTH is not called")
        if not bool(nml.get(pkgs, "packages", "useEXF", default=False)):
            raise NotImplementedError("useEXF=F: the SEAICE_rhoAir/cpAir/... defaults of the non-EXF branch "
                                      "(seaice_readparms.F:403-413) are not ported")
        useSP = bool(nml.get(pkgs, "packages", "useSALT_PLUME", default=False))
        if d("parm01", "buoyancyRelation", "OCEANIC") != "OCEANIC":  # set_defaults.F:175
            raise NotImplementedError("kSurface = Nr (OCEANICP) is not ported (seaice_growth.F:328-332)")
        if (bool(nml.get(pkgs, "packages", "useThSice", default=False))                 # seaice_readparms.F:912-916
                or not bool(s("usePW79thermodynamics", True))):                        # seaice_readparms.F:234
            raise NotImplementedError("usePW79thermodynamics=F: SEAICE_GROWTH is not called")
        # --- branch switches (one branch ported each)
        if bool(s("SEAICE_growMeltByConv", False)):                                    # seaice_readparms.F:228
            raise NotImplementedError("SEAICE_growMeltByConv=T (seaice_growth.F:1300, 1529) not ported")
        if int(s("SEAICE_areaGainFormula", 1)) != 1:                                   # seaice_readparms.F:452
            raise NotImplementedError("SEAICE_areaGainFormula != 1 not ported (seaice_growth.F:1815-1819)")
        if int(s("SEAICE_areaLossFormula", 1)) != 2:                                   # seaice_readparms.F:451
            raise NotImplementedError("SEAICE_areaLossFormula != 2 not ported (seaice_growth.F:1826-1842)")
        if not bool(s("SEAICEuseFlooding", True)):                                     # seaice_readparms.F:276
            raise NotImplementedError("SEAICEuseFlooding=F not ported (seaice_growth.F:1691)")
        if bool(s("SEAICE_mcPheeStepFunc", False)):                                    # seaice_readparms.F:440
            raise NotImplementedError("SEAICE_mcPheeStepFunc=T not ported (seaice_growth.F:1052-1054)")
        if bool(s("useMaykutSatVapPoly", False)):                                      # seaice_readparms.F:266
            raise NotImplementedError("useMaykutSatVapPoly=T not ported (seaice_solve4temp.F:376-379)")
        if bool(s("SEAICE_useMultDimSnow", False)):                                    # seaice_readparms.F:438
            raise NotImplementedError("SEAICE_useMultDimSnow=T not ported (seaice_growth.F:803)")
        if float(s("SEAICE_snowThick", 0.15)) <= 0.0:                                  # seaice_readparms.F:416
            raise NotImplementedError("SEAICE_snowThick <= 0: albedo branch seaice_solve4temp.F:299-300 not ported")
        postIt = int(s("postSolvTempIter", 2))                                         # seaice_readparms.F:467
        if postIt != 2:
            raise NotImplementedError(f"postSolvTempIter={postIt}: only 2 is ported (seaice_solve4temp.F:486)")
        heatConsFix = bool(s("SEAICEheatConsFix", False))                              # seaice_readparms.F:221
        rfwf = bool(d("parm01", "useRealFreshWaterFlux", False))                       # set_defaults.F
        nlfs = int(d("parm01", "nonlinFreeSurf", 0))                                   # set_defaults.F
        temp_EvPrRn = float(d("parm01", "temp_EvPrRn", UNSET_RL))                      # set_defaults.F:259
        if not (temp_EvPrRn == UNSET_RL and rfwf and nlfs > 0 and heatConsFix):
            raise NotImplementedError("only temp_EvPrRn=UNSET, useRealFreshWaterFlux=T, nonlinFreeSurf>0, "
                                      "SEAICEheatConsFix=T is ported (seaice_growth.F:2270-2291)")
        if bool(d("parm01", "balanceEmPmR", False)) or bool(d("parm01", "balanceQnet", False)):  # set_defaults:263-4
            raise NotImplementedError("balanceEmPmR/balanceQnet=T not ported (seaice_growth.F:2473-2694)")
        exf = "data.exf"
        if str(nml.get(exf, "exf_nml_02", "snowprecipfile", default=" ")).strip() != "":   # exf_readparms.F:606
            raise NotImplementedError("snowPrecipFile set: branch seaice_growth.F:1463-1474 not ported")
        if bool(nml.get(exf, "exf_nml_01", "useRelativeWind", default=False)):            # exf_readparms.F:325
            raise NotImplementedError("useRelativeWind=T not ported (seaice_growth.F:756-776)")
        multDim = int(s("SEAICE_multDim", 1))                                          # seaice_readparms.F:433
        if multDim != 1:
            raise NotImplementedError(f"SEAICE_multDim={multDim}: only 1 (V4r4) is ported")
        # seaice_readparms.F:432-436 SEAICE_PDF(l) = UNSET; :655-661 -> 1/multDim for l <= multDim
        pdf = nml.get(S, P1, "SEAICE_PDF", default=[UNSET_RL], array=True)
        pdf1 = float(pdf[0]) if pdf and pdf[0] != UNSET_RL else 1.0 / float(multDim)
        # --- time step: seaice_readparms.F:293 SEAICE_deltaTtherm = dTtracerLev(1); ini_parms.F:875-899
        dTlev = nml.get("data", "parm03", "dTtracerLev", default=[0.0], array=True)
        dTt = float(d("parm03", "deltaTtracer", 0.0))
        if dTlev and float(dTlev[0]) != 0.0:
            dT1 = float(dTlev[0])
        elif dTt != 0.0:
            dT1 = dTt
        else:
            raise NotImplementedError("deltaTtracer unset: deltaT fallback chain (ini_parms.F:884-899) not ported")
        deltaTtherm = float(s("SEAICE_deltaTtherm", dT1))
        if deltaTtherm != dT1:                                                          # seaice_readparms.F:679-690
            raise NotImplementedError("SEAICE_deltaTtherm != dTtracerLev(1) is rejected by SEAICE_READPARMS")
        # --- EXF-consistent constants (seaice_readparms.F:388-399, useEXF=T)
        rhoAir = float(nml.get(exf, "exf_nml_01", "atmrho", default=1.2))              # exf_readparms.F:336
        cpAir = float(nml.get(exf, "exf_nml_01", "atmcp", default=1005.0))             # exf_readparms.F:337
        lhEvap = float(nml.get(exf, "exf_nml_01", "flamb", default=2500000.0))         # exf_readparms.F:338
        lhFusion = float(nml.get(exf, "exf_nml_01", "flami", default=334000.0))        # exf_readparms.F:339
        boltzmann = 5.670e-8                                                            # EXF_CONSTANTS.h:48
        ice_emiss = float(nml.get(exf, "exf_nml_01", "ice_emissivity", default=0.95))   # exf_readparms.F:368
        snow_emiss = float(nml.get(exf, "exf_nml_01", "snow_emissivity", default=0.95))  # exf_readparms.F:369
        # --- albedos (seaice_readparms.F:355-365, 663-672: *_south default to the northern value)
        dIA = float(s("SEAICE_dryIceAlb", 0.75))
        wIA = float(s("SEAICE_wetIceAlb", 0.66))
        dSA = float(s("SEAICE_drySnowAlb", 0.84))
        wSA = float(s("SEAICE_wetSnowAlb", 0.7))
        HO = float(s("HO", 0.5))                                                        # seaice_readparms.F:359
        # --- mcPhee / frazil (seaice_readparms.F:441-448, 822-887; seaice_init_fixed.F:88-100)
        for k in ("SEAICE_availHeatTaper", "SEAICE_gamma_t", "SEAICE_gamma_t_frz", "SEAICE_availHeatFrac",
                  "SEAICE_availHeatFracFrz"):
            if nml.has(S, P1, k):
                raise NotImplementedError(f"{k} set: old-style mcPhee/frazil settings not ported")
        if not nml.has(S, P1, "SEAICE_mcPheePiston"):
            raise NotImplementedError("SEAICE_mcPheePiston unset: default from seaice_init_fixed.F:94-97 not ported")
        mcPheeTaper = float(s("SEAICE_mcPheeTaper", 0.0))                               # seaice_readparms.F:835-836
        frazilFrac = float(s("SEAICE_frazilFrac", 1.0))                                 # seaice_readparms.F:883-885
        # --- model constants
        rhoNil = float(d("parm01", "rhoNil", 999.8))                                    # set_defaults.F:106
        rhoConst = float(d("parm01", "rhoConst", rhoNil))                               # ini_parms.F:445
        rhoConstFresh = float(d("parm01", "rhoConstFresh", rhoConst))                   # ini_parms.F:446
        # --- SWFracB (seaice_init_fixed.F:73-83, SHORTWAVE_HEATING): SWFRAC(-1, |rF(2)|)
        SWFracB = _swfrac(abs(float(g.rF[1])))
        # --- solve4temp constants (seaice_solve4temp.F:171-180); host-side glibc log/exp: equal to the correctly
        #     rounded values (checked), so equal whether gfortran -O3 folded them or not
        lnTEN = math.log(10.0)
        aa1, aa2, bb1, Pp = 2663.5, 12.537, 0.622, 100000.0
        bb2 = 1.0 - bb1
        cc0 = math.exp(aa2 * lnTEN)
        cc1 = cc0 * aa1 * bb1 * Pp * lnTEN
        cc2 = cc0 * bb2
        return cls(
            deltaTtherm=deltaTtherm,
            rhoIce=float(s("SEAICE_rhoIce", 0.91e3)),                                   # seaice_readparms.F:349
            rhoSnow=float(s("SEAICE_rhoSnow", 330.0)),                                  # seaice_readparms.F:350
            lhFusion=float(s("SEAICE_lhFusion", lhFusion)), lhEvap=float(s("SEAICE_lhEvap", lhEvap)),
            area_reg=float(s("SEAICE_area_reg", 1.0e-5)),                  # seaice_readparms.F:487 siEps (PARAMS.h:567)
            hice_reg=float(s("SEAICE_hice_reg", 0.05)),                                  # seaice_readparms.F:488
            area_max=float(s("SEAICE_area_max", 1.0)),                                   # seaice_readparms.F:489
            EPS=float(s("SEAICE_EPS", 1.0e-10)),                                         # seaice_readparms.F:497
            tempFrz0=float(s("SEAICE_tempFrz0", 0.0901)),                                # seaice_readparms.F:453
            dTempFrz_dS=float(s("SEAICE_dTempFrz_dS", -0.0575)),                         # seaice_readparms.F:454
            mcPheePiston=float(s("SEAICE_mcPheePiston", UNSET_RL)), mcPheeTaper=mcPheeTaper,
            frazilFrac=frazilFrac,
            HO=HO, HO_south=float(s("HO_south", HO)),
            salt0=float(s("SEAICE_salt0", 0.0)),                                         # seaice_readparms.F:418
            # seaice_readparms.F:1241-1244 (doOpenWaterGrowth default T :449, doOpenWaterMelt default F :450)
            facOpenGrow=1.0 if bool(s("SEAICE_doOpenWaterGrowth", True)) else 0.0,
            facOpenMelt=1.0 if bool(s("SEAICE_doOpenWaterMelt", False)) else 0.0,
            SWFracB=SWFracB,
            SPsalFRAC=float(nml.get("data.salt_plume", "salt_plume_parm01", "SPsalFRAC",
                                    default=1.0)),                                       # salt_plume_readparms.F:75
            celsius2K=float(d("parm01", "celsius2K", 273.15)),                           # set_defaults.F:270
            HeatCapacity_Cp=float(d("parm01", "HeatCapacity_Cp", 3994.0)),              # set_defaults.F:173
            rhoConst=rhoConst, recip_rhoConst=1.0 / rhoConst,                            # ini_parms.F:640
            rhoConstFresh=rhoConstFresh,
            dalton=float(s("SEAICE_dalton", 1.75e-3)),                                   # seaice_readparms.F:387
            cpAir=float(s("SEAICE_cpAir", cpAir)), rhoAir=float(s("SEAICE_rhoAir", rhoAir)),
            iceConduct=float(s("SEAICE_iceConduct", 2.1656)),                            # seaice_readparms.F:414
            snowConduct=float(s("SEAICE_snowConduct", 0.31)),                            # seaice_readparms.F:415
            snowThick=float(s("SEAICE_snowThick", 0.15)),                                # seaice_readparms.F:416
            shortwave=float(s("SEAICE_shortwave", 0.30)),                                # seaice_readparms.F:417
            wetAlbTemp=float(s("SEAICE_wetAlbTemp", -1.0e-3)),                           # seaice_readparms.F:374
            dryIceAlb=dIA, wetIceAlb=wIA, drySnowAlb=dSA, wetSnowAlb=wSA,
            dryIceAlb_south=float(s("SEAICE_dryIceAlb_south", dIA)),
            wetIceAlb_south=float(s("SEAICE_wetIceAlb_south", wIA)),
            drySnowAlb_south=float(s("SEAICE_drySnowAlb_south", dSA)),
            wetSnowAlb_south=float(s("SEAICE_wetSnowAlb_south", wSA)),
            snow_emiss=float(s("SEAICE_snow_emiss", snow_emiss)), ice_emiss=float(s("SEAICE_ice_emiss", ice_emiss)),
            boltzmann=float(s("SEAICE_boltzmann", boltzmann)),
            MIN_LWDOWN=float(s("MIN_LWDOWN", 60.0)),                                     # seaice_readparms.F:494
            MIN_ATEMP=float(s("MIN_ATEMP", -50.0)),                                      # seaice_readparms.F:493
            aa1=aa1, aa2=aa2, bb1=bb1, Ppascals=Pp, lnTEN=lnTEN, cc1=cc1, cc2=cc2,
            multDim=multDim, SEAICE_PDF=(pdf1,),
            IMAX_TICE=int(s("IMAX_TICE", 10)),                                           # seaice_readparms.F:466
            postSolvTempIter=postIt,
            SaltPlumeSouthernOcean=bool(nml.get("data.salt_plume", "salt_plume_parm01", "SaltPlumeSouthernOcean",
                                                default=True)),                          # salt_plume_readparms.F:62
            useSALT_PLUME=useSP,
        )


# ------------------------------------------------------------------------------------------------ exp
#
# glibc_exp: bit-for-bit emulation of the exp() the gfortran oracle calls. RHEL8 glibc 2.28, /lib64/libm-2.28.so:
# `exp` -> `__exp_finite` (ifunc at 0x28380) -> on AMD EPYC 7763 (FMA + AVX2) the FMA-compiled IBM Accurate
# Mathematical Library exp at 0x771e0 (sysdeps/ieee754/dbl-64/e_exp.c after the 2.28 slow-path removal; NOT correctly
# rounded: 11 of 1e5 random results differ from the correctly rounded exp). XLA's exp differs from it by 1-2 ulp at
# ~14 % of arguments. Paths (hx = high word of x, n = hx & 0x7fffffff), transcribed from the disassembly, every
# vfmadd/vfnmadd as an (emulated, exact) fused multiply-add:
#     n <= 0x3e2fffff                 (|x| < 2^-28)       1 + x                                  (0x77253)
#     0x3e2fffff < n <= 0x3ecfffff    (|x| < ~3.8e-6)    fma(fma(x, 0.5, 1), x, 1)              (0x77490)
#     0x3ecfffff < n <= 0x3f862e41    (|x| <= ~0.01085)  1 + (x + poly(x))                      (0x775a0)
#     0x3f862e41 < n <= 0x3ff0a2b1    (|x| <= ~1.0397)   accurate-table path                    (0x772b0)
#     0x3ff0a2b1 < n <= 0x40862001    (|x| <= ~708.0039) coar/fine table path (IBM __exp)       (0x774b0)
#     otherwise (near overflow/underflow, inf, nan)      jnp.exp FALLBACK, not bitwise          (0x77368, specials)
# The rounding-mode save/restore (vstmxcsr) is a no-op in round-to-nearest. Tables (.rodata, VA == file offset): coar
# (0xfa4a0) = exp(k*2^-9), k = -178..177, fine (0xf84a0) = exp(m*2^-18), m = 0..511, as (hi, lo), hi = exp rounded to
# 27 significant bits, lo = RN(exp - hi): regenerated below (decimal, 80 digits), bit-identical to the binary; the
# accurate table (0xf7c40) holds 67 + 67 Gal-type nodes x_k ~ +-(k+1)/64 (listed: they cannot be derived) and
# RN(exp(x_k)) (regenerated). `glibc_exp_tables_vs_libm()` compares all of them and the 13 constants with the binary.
# FMA without hardware FMA: TwoProduct (Veltkamp split) a*b = uh + ul, TwoSum c + uh = th + tl, then
# fma = RN(th + RO(tl + ul)) (Boldo & Melquiond 2008; round-to-odd from TwoSum + one integer step of the bit pattern).
# Needs IEEE arithmetic as written (the gate flags: no FMA contraction, no algsimp); bexp and base are read from the
# integer words of y and yy (exact identities) because algsimp folds (c + t) - c -> t. Measured on a compute node with
# 2e6 uniform arguments per range vs Python math.exp (same glibc, same CPU model): 0 mismatches in [-15,-0.075],
# [+-0.0108,+-1.0397], [1,8], [20,30], log-uniform |x| in [0.0108,708] and [1e-12,0.0108] under the gate flags; default
# XLA flags: <= 7 per 2e6 for |x| >= 0.0108, 1.9 % below (all <= 1 ulp). Cost ~28 ms per 1e6 values (jnp.exp 1.2 ms).

_E_THREE51 = float.fromhex("0x1.8p+52")               # 0xd3320
_E_LOG2E = float.fromhex("0x1.71547652b82fep+0")      # 0xde4e0
_E_THREE33 = float.fromhex("0x1.8p+34")               # 0xfbae0
_E_LN_TWO1 = float.fromhex("0x1.62e42fefa3800p-1")    # 0xfbb00
_E_LN_TWO2 = float.fromhex("0x1.ef35793c76730p-45")   # 0xfbaf8
_E_P3 = float.fromhex("0x1.5555555555a0fp-3")         # 0xfbae8
_E_P2 = float.fromhex("0x1.00000000004dcp-1")         # 0xfbaf0
_E_C120 = float.fromhex("0x1.1111111111111p-7")       # 0xc55c8
_E_C24 = float.fromhex("0x1.5555555555555p-5")        # 0xc55d0
_E_C720 = float.fromhex("0x1.6c16c16c16c17p-10")      # 0xc55d8
_E_C6 = float.fromhex("0x1.5555555555555p-3")         # 0xc55e0
_E_HALF = 0.5                                         # 0xad0a8
_E_ONE = 1.0                                          # 0xac8b0
_E_CONST_ADDR = {0xd3320: _E_THREE51, 0xde4e0: _E_LOG2E, 0xfbae0: _E_THREE33, 0xfbb00: _E_LN_TWO1,
                 0xfbaf8: _E_LN_TWO2, 0xfbae8: _E_P3, 0xfbaf0: _E_P2, 0xc55c8: _E_C120, 0xc55d0: _E_C24,
                 0xc55d8: _E_C720, 0xc55e0: _E_C6, 0xad0a8: _E_HALF, 0xac8b0: _E_ONE}
_E_N_TINY, _E_N_SMALL, _E_N_POLY, _E_N_MED, _E_N_MAIN = 0x3e2fffff, 0x3ecfffff, 0x3f862e41, 0x3ff0a2b1, 0x40862001
_E_SPLIT = 134217729.0  # 2^27 + 1
# accurate-table nodes (0xf7c40, even entries): 67 positive, then 67 negative
_E_NODES_HEX = (
    "0x1.ffffffffffc82p-7", "0x1.fffffffffffdbp-6", "0x1.80000000000a0p-5", "0x1.fffffffffff79p-5",
    "0x1.3fffffffffffcp-4", "0x1.8000000000060p-4", "0x1.c000000000061p-4", "0x1.fffffffffffd6p-4",
    "0x1.1ffffffffff58p-3", "0x1.3ffffffffff75p-3", "0x1.5ffffffffff00p-3", "0x1.8000000000020p-3",
    "0x1.9ffffffffa629p-3", "0x1.c00000000000fp-3", "0x1.e00000000007fp-3", "0x1.0000000000072p-2",
    "0x1.0fffffffffecap-2", "0x1.1ffffffffff8fp-2", "0x1.300000000003bp-2", "0x1.4000000000034p-2",
    "0x1.4ffffffffff89p-2", "0x1.5ffffffffffe7p-2", "0x1.6ffffffffff78p-2", "0x1.7ffffffffff65p-2",
    "0x1.8ffffffffffd5p-2", "0x1.9ffffffffff6ep-2", "0x1.affffffffffc3p-2", "0x1.c000000000053p-2",
    "0x1.d00000000004dp-2", "0x1.e000000000096p-2", "0x1.efffffffffefap-2", "0x1.fffffffffffd0p-2",
    "0x1.0800000000002p-1", "0x1.100000000001fp-1", "0x1.17ffffffffff8p-1", "0x1.1fffffffffffap-1",
    "0x1.27fffffffffc4p-1", "0x1.2fffffffffffdp-1", "0x1.380000000001fp-1", "0x1.3ffffffffffd8p-1",
    "0x1.4800000000052p-1", "0x1.4ffffffffffc8p-1", "0x1.5800000000013p-1", "0x1.5ffffffffffbcp-1",
    "0x1.680000000002dp-1", "0x1.7000000000040p-1", "0x1.780000000004fp-1", "0x1.7ffffffffff6fp-1",
    "0x1.87fffffffffe5p-1", "0x1.9000000000035p-1", "0x1.97fffffffffb3p-1", "0x1.a000000000000p-1",
    "0x1.a80000000004ap-1", "0x1.affffffffffedp-1", "0x1.b7ffffffffffbp-1", "0x1.c00000000001dp-1",
    "0x1.c800000000079p-1", "0x1.cffffffffff51p-1", "0x1.d7fffffffff74p-1", "0x1.e000000000011p-1",
    "0x1.e80000000001ep-1", "0x1.effffffffff9ep-1", "0x1.f7fffffffffedp-1", "0x1.0000000000034p+0",
    "0x1.03fffffffffe2p+0", "0x1.07fffffffff4bp+0", "0x1.0bffffffffffdp+0", "-0x1.fffffffffffe4p-7",
    "-0x1.ffffffffffb0bp-6", "-0x1.7ffffffffffa7p-5", "-0x1.ffffffffffea8p-5", "-0x1.3ffffffffffb3p-4",
    "-0x1.7ffffffffffe3p-4", "-0x1.bffffffffff9ap-4", "-0x1.fffffffffff98p-4", "-0x1.1ffffffffffe9p-3",
    "-0x1.3ffffffffffe0p-3", "-0x1.5fffffffff553p-3", "-0x1.7ffffffffff8bp-3", "-0x1.9fffffffffe51p-3",
    "-0x1.bffffffffff6ep-3", "-0x1.dffffffffff7fp-3", "-0x1.fffffffffff7ap-3", "-0x1.0fffffffffffep-2",
    "-0x1.1ffffffffff41p-2", "-0x1.2ffffffffffbap-2", "-0x1.3fffffffffff8p-2", "-0x1.4ffffffffff90p-2",
    "-0x1.5ffffffffffdbp-2", "-0x1.6ffffffffff9ap-2", "-0x1.7ffffffffff9fp-2", "-0x1.8ffffffffffeep-2",
    "-0x1.9fffffffffc4ap-2", "-0x1.affffffffff30p-2", "-0x1.bfffffffffff0p-2", "-0x1.cfffffffffff3p-2",
    "-0x1.dfffffffffff3p-2", "-0x1.effffffffff80p-2", "-0x1.fffffffffffdfp-2", "-0x1.0800000000000p-1",
    "-0x1.0ffffffffffa4p-1", "-0x1.17fffffffff0ap-1", "-0x1.2000000000000p-1", "-0x1.27fffffffffbbp-1",
    "-0x1.2fffffffffe32p-1", "-0x1.37ffffffff042p-1", "-0x1.3ffffffffff77p-1", "-0x1.47fffffffff6bp-1",
    "-0x1.4fffffffffff1p-1", "-0x1.57ffffffffe02p-1", "-0x1.5ffffffffffe5p-1", "-0x1.67fffffffffb0p-1",
    "-0x1.6ffffffffffb2p-1", "-0x1.77fffffffff7fp-1", "-0x1.7ffffffffffe8p-1", "-0x1.87fffffffffc8p-1",
    "-0x1.8fffffffffb30p-1", "-0x1.97fffffffffefp-1", "-0x1.9ffffffffffa7p-1", "-0x1.a7fffffffffdcp-1",
    "-0x1.affffffffff95p-1", "-0x1.b7fffffffffcbp-1", "-0x1.bffffffffff32p-1", "-0x1.c7fffffffff6ap-1",
    "-0x1.cffffffffffb6p-1", "-0x1.d7fffffffffcap-1", "-0x1.dffffffffffcdp-1", "-0x1.e7ffffffffffbp-1",
    "-0x1.effffffffff88p-1", "-0x1.f7fffffffffbbp-1", "-0x1.fffffffffffdbp-1", "-0x1.03fffffffff00p+0",
    "-0x1.07ffffffffe6fp+0", "-0x1.0bfffffffffd6p+0",
)


def _glibc_exp_tables():
    """(coar[712], fine[1024], acc[268]) float64, regenerated from first principles (see the section comment)."""
    import decimal
    from fractions import Fraction

    import numpy as np

    def dec_exp(q):
        with decimal.localcontext() as ctx:
            ctx.prec = 80
            return Fraction((decimal.Decimal(q.numerator) / decimal.Decimal(q.denominator)).exp())

    def rn_bits(v, nb):
        e = v.numerator.bit_length() - v.denominator.bit_length()
        while Fraction(2) ** e > v:
            e -= 1
        while Fraction(2) ** (e + 1) <= v:
            e += 1
        scale = Fraction(2) ** (nb - 1 - e)
        return Fraction(round(v * scale)) / scale

    def hi_lo(q):
        ex = dec_exp(q)
        hi = rn_bits(ex, 27)
        return float(hi), float(ex - hi)

    coar = np.empty(712)
    for n, k in enumerate(range(-178, 178)):
        coar[2 * n], coar[2 * n + 1] = hi_lo(Fraction(k, 2 ** 9))
    fine = np.empty(1024)
    for m in range(512):
        fine[2 * m], fine[2 * m + 1] = hi_lo(Fraction(m, 2 ** 18))
    acc = np.empty(268)
    for n, h in enumerate(_E_NODES_HEX):
        x = float.fromhex(h)
        acc[2 * n], acc[2 * n + 1] = x, float(dec_exp(Fraction(x)))
    return coar, fine, acc


_E_TABLES = _glibc_exp_tables()


def glibc_exp_tables_vs_libm(path="/lib64/libm-2.28.so"):
    """List of differences between the regenerated tables/constants and the bytes of the libm binary ([] = none)."""
    import struct

    import numpy as np

    b = open(path, "rb").read()
    bad = [hex(a) for a, v in _E_CONST_ADDR.items() if struct.unpack("<d", b[a:a + 8])[0] != v]
    coar, fine, acc = _E_TABLES
    for name, a, t in (("acc", 0xf7c40, acc), ("fine", 0xf84a0, fine), ("coar", 0xfa4a0, coar)):
        nd = int(np.sum(np.frombuffer(b[a:a + 8 * t.size], "<u8") != t.view(np.uint64)))
        if nd:
            bad.append(f"{name}: {nd} entries differ")
    return bad


def _f2i(v):
    return jax.lax.bitcast_convert_type(v, jnp.int64)


def _i2f(v):
    return jax.lax.bitcast_convert_type(v, jnp.float64)


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_prod(a, b):
    def split(v):
        c = _E_SPLIT * v
        vh = c - (c - v)
        return vh, v - vh

    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


def fma_emulated(a, b, c):
    """RN(a*b + c) in float64 without a hardware FMA (Boldo-Melquiond; exact barring overflow/underflow)."""
    a, b, c = (jnp.asarray(v, jnp.float64) for v in (a, b, c))
    uh, ul = _two_prod(a, b)
    th, tl = _two_sum(c, uh)
    s, e = _two_sum(tl, ul)                                  # round-to-odd of tl + ul
    sb = _f2i(s)
    step = jnp.where(jnp.signbit(e) == jnp.signbit(s), jnp.int64(1), jnp.int64(-1))
    sb = jnp.where((e != 0) & ((sb & 1) == 0), sb + step, sb)
    return th + _i2f(sb)


def _exp_poly(d):
    """d + d^2*(1/2 + d/6) + d^4*(1/24 + d/120 + d^2/720) in the binary's FMA order (0x772b0 / 0x775a0)."""
    a = fma_emulated(d, _E_C120, _E_C24)
    b = fma_emulated(d, _E_C6, _E_HALF)
    d2 = d * d
    a = fma_emulated(d2, _E_C720, a)
    a = a * (d2 * d2)
    b = fma_emulated(b, d2, a)
    return b + d


def _signed_low(v):
    k = _f2i(v) & 0xffffffff
    return (k ^ 0x80000000) - 0x80000000


def _glibc_exp_impl(x):
    coar, fine, acc = (jnp.asarray(t, jnp.float64) for t in _E_TABLES)
    x = jnp.asarray(x, jnp.float64)
    hx = _f2i(x) >> 32
    n = hx & 0x7fffffff
    # main path (0x774b0)
    y = fma_emulated(x, _E_LOG2E, _E_THREE51)
    ky = _signed_low(y)
    bexp = ky.astype(jnp.float64)                            # == y - THREE51 exactly (0x774d4)
    t = fma_emulated(-bexp, _E_LN_TWO1, x)
    yy = _E_THREE33 + t
    kb = _signed_low(yy)
    base = kb.astype(jnp.float64) * 3.814697265625e-06       # == yy - THREE33 exactly (0x774fb), 2^-18
    i = jnp.clip(((kb >> 8) & ~1) + 356, 0, 710)
    j = jnp.clip((kb * 2) & 0x3fe, 0, 1022)
    dl = fma_emulated(-bexp, _E_LN_TWO2, t - base)
    pp = fma_emulated(dl, _E_P3, _E_P2)
    eps = fma_emulated(dl * dl, pp, dl)
    ch, cl = jnp.take(coar, i), jnp.take(coar, i + 1)
    fh, fl = jnp.take(fine, j), jnp.take(fine, j + 1)
    al = ch * fh
    bet = fma_emulated(ch, fl, fh * cl)
    bet = fma_emulated(cl, fl, bet)
    r = fma_emulated(bet, eps, bet)
    r = fma_emulated(eps, al, r)
    res_main = (r + al) * _i2f(((ky + 1023) & 0xfff) << 52)
    # accurate-table path (0x772b0)
    M = (hx & 0xfffff) | 0x100000
    m = M >> jnp.clip(1036 - (n >> 20), 0, 63)
    idx = (m - 1) & ~1
    idx = jnp.clip(jnp.where(hx < 0, idx + 134, idx), 0, 266)
    a0, a1 = jnp.take(acc, idx), jnp.take(acc, idx + 1)
    res_med = fma_emulated(a1, _exp_poly(x - a0), a1)
    # small |x| paths
    res_poly = _exp_poly(x) + _E_ONE
    res_quad = fma_emulated(fma_emulated(x, _E_HALF, _E_ONE), x, _E_ONE)
    res_tiny = _E_ONE + x
    return jnp.where(n <= _E_N_TINY, res_tiny,
                     jnp.where(n <= _E_N_SMALL, res_quad,
                               jnp.where(n <= _E_N_POLY, res_poly,
                                         jnp.where(n <= _E_N_MED, res_med,
                                                   jnp.where(n <= _E_N_MAIN, res_main, jnp.exp(x))))))


@jax.custom_jvp
def glibc_exp(x):
    """exp(x), float64, bit for bit the glibc 2.28 exp of the oracle for |x| <= ~708 (jnp.exp beyond). The derivative is
    glibc_exp(x) * x_dot (AD never enters the bit arithmetic)."""
    return _glibc_exp_impl(x)


@glibc_exp.defjvp
def _glibc_exp_jvp(primals, tangents):
    (x,), (xd,) = primals, tangents
    y = glibc_exp(x)
    return y, y * xd


def default_exp():
    """The exp SEAICE_SOLVE4TEMP uses: glibc_exp (bitwise with the oracle's libm). Pass expf=jnp.exp for XLA's exp
    (faster, 1-2 ulp from glibc at ~14 % of arguments)."""
    return glibc_exp


# ------------------------------------------------------------------------------------------------ helpers


def interior(L, a):
    """[..., ny, nx] -> [..., sNy, sNx] (Fortran 1..sNy, 1..sNx)."""
    return a[..., L.js(1, L.sNy), L.is_(1, L.sNx)]


def set_interior(L, full, x):
    """Write the interior of `full` (halos keep their values: the Fortran loops never write them)."""
    return jnp.asarray(full).at[..., L.js(1, L.sNy), L.is_(1, L.sNx)].set(x)


def zero_halo(L, x):
    """Interior array -> full array with zero halos (layout of the jaxdump U: dumps of routine locals)."""
    shape = x.shape[:-2] + (L.ny, L.nx)
    return jnp.zeros(shape, x.dtype).at[..., L.js(1, L.sNy), L.is_(1, L.sNx)].set(x)


# ------------------------------------------------------------------------------------------------ PART 1


def growth_pre_budget(p, AREA, HEFF, HSNOW, wspeed, theta_s):
    """Interior arrays in, stage H01 out. seaice_growth.F:511-526 (pre-thermodynamics copies), 658-679 (regularised
    thicknesses, not SEAICE_ITD), 731-737 (TmixLoc, UG)."""
    area_reg_sq = p.area_reg * p.area_reg                                   # seaice_growth.F:359
    hice_reg_sq = p.hice_reg * p.hice_reg                                   # seaice_growth.F:360
    HEFFpreTH, HSNWpreTH, AREApreTH = HEFF, HSNOW, AREA                     # :513-515
    pos = HEFFpreTH > 0.0                                                   # :660
    tmpscal1 = jnp.sqrt(AREApreTH * AREApreTH + area_reg_sq)                # :662
    tmpscal2 = HEFFpreTH / tmpscal1                                         # :664
    heffActual = jnp.where(pos, jnp.sqrt(tmpscal2 * tmpscal2 + hice_reg_sq), 0.0)   # :666, :674
    hsnowActual = jnp.where(pos, HSNWpreTH / tmpscal1, 0.0)                 # :668, :675
    recip_heffActual = jnp.where(
        pos, AREApreTH / jnp.sqrt(HEFFpreTH * HEFFpreTH + hice_reg_sq), 0.0)  # :670-671, :676
    TmixLoc = theta_s + p.celsius2K                                         # :734
    UG = jnp.maximum(p.EPS, wspeed)                                         # :736
    return dict(HEFFpreTH=HEFFpreTH, HSNWpreTH=HSNWpreTH, AREApreTH=AREApreTH, heffActual=heffActual,
                hsnowActual=hsnowActual, recip_heffActual=recip_heffActual, UG=UG, TmixLoc=TmixLoc)


def budget_ocean(Qnet, Qsw):
    """SEAICE_BUDGET_OCEAN with SEAICE_EXTERNAL_FLUXES (seaice_budget_ocean.F:106-110): the open-water heat budget is
    the ocean's own Qnet/Qsw (W/m2, interior). Returns (a_QbyATM_open, a_QSWbyATM_open) = stage H02."""
    return Qnet, Qsw


# ------------------------------------------------------------------------------------------------ SOLVE4TEMP


def solve4temp(p, UG, HICE_ACTUAL, HSNOW_ACTUAL, TSURFin, lwdown, atemp, swdown, aqh, salt_s, yC, expf=None):
    """SEAICE_SOLVE4TEMP (seaice_solve4temp.F), interior arrays. Returns (TSURFout, F_ia, IcePenetSW, FWsublim)
    = (ticeOutMult, a_QbyATMmult_cover, a_QSWbyATMmult_cover, a_FWbySublimMult) of one category (stage H04).
    IMAX_TICE Newton steps (fixed count, seaice_solve4temp.F:357), postSolvTempIter = 2 flux recomputation (:486-518).
    Non-ice points (HICE_ACTUAL <= 0) keep TSURFout = TSURFin and zero fluxes; their lanes are guarded (t1 = TMELT)
    so that every lane stays finite in the forward and backward pass."""
    ex = expf if expf is not None else default_exp()
    # seaice_solve4temp.F:189-215
    D1 = p.dalton * p.cpAir * p.rhoAir                                      # :189
    lhSublim = p.lhEvap + p.lhFusion                                        # :192
    D1I = p.dalton * lhSublim * p.rhoAir                                    # :193
    TMELT = p.celsius2K                                                     # :196
    XKI = p.iceConduct                                                      # :199
    XKS = p.snowConduct                                                     # :202
    HCUT = p.snowThick                                                      # :207
    recip_HCUT = 1.0 / HCUT                                                 # :208-209 (HCUT > 0 checked below)
    XIO = p.shortwave                                                       # :212
    SurfMeltTemp = TMELT + p.wetAlbTemp                                     # :215
    ice = HICE_ACTUAL > 0.0                                                 # :231 iceOrNot
    # :241-246
    lwdownLoc = jnp.maximum(p.MIN_LWDOWN, lwdown)
    atempLoc = jnp.maximum(p.celsius2K + p.MIN_ATEMP, atemp)
    tempFrz = p.dTempFrz_dS * salt_s + p.tempFrz0 + p.celsius2K
    # :250-268
    snow = HSNOW_ACTUAL > 0.0
    D3 = jnp.where(snow, p.snow_emiss * p.boltzmann, p.ice_emiss * p.boltzmann)
    lwdownLoc = jnp.where(snow, p.snow_emiss * lwdownLoc, p.ice_emiss * lwdownLoc)
    tsurfLoc = TSURFin                                                      # :239
    # :276-294 albedo (tsurfLoc = TSURFin here)
    south = yC < 0.0
    wet = tsurfLoc >= SurfMeltTemp
    ALB_ICE = jnp.where(south, jnp.where(wet, p.wetIceAlb_south, p.dryIceAlb_south),
                        jnp.where(wet, p.wetIceAlb, p.dryIceAlb))
    ALB_SNOW = jnp.where(south, jnp.where(wet, p.wetSnowAlb_south, p.drySnowAlb_south),
                         jnp.where(wet, p.wetSnowAlb, p.drySnowAlb))
    # :297-306 (HCUT > 0: the HCUT.LE.ZERO branch is not taken)
    ALB = jnp.where(HSNOW_ACTUAL > HCUT, ALB_SNOW,
                    jnp.minimum(ALB_ICE + HSNOW_ACTUAL * recip_HCUT * (ALB_SNOW - ALB_ICE), ALB_SNOW))
    # :311-321
    hiceG = jnp.where(ice, HICE_ACTUAL, 1.0)
    penetSWFrac = jnp.where(snow, 0.0, XIO * ex(-1.5 * hiceG))
    IcePenetSW = jnp.where(ice, -(1.0 - ALB) * penetSWFrac * swdown, 0.0)
    absorbedSW = jnp.where(ice, (1.0 - ALB) * (1.0 - penetSWFrac) * swdown, 0.0)
    # :328-329 effective conductivity (denominator guarded on non-ice lanes)
    den = jnp.where(ice, XKS * HICE_ACTUAL + XKI * HSNOW_ACTUAL, 1.0)
    effConduct = XKI * XKS / den

    def sat(t1):
        # :383-389 saturation specific humidity (not Maykut)
        mm_log10pi = -p.aa1 / t1 + p.aa2
        mm_pi = ex(mm_log10pi * p.lnTEN)
        return p.bb1 * mm_pi / (p.Ppascals - (1.0 - p.bb1) * mm_pi)

    tsurfLoc = jnp.where(ice, tsurfLoc, TMELT)                              # guard (non-ice lanes: finite, unused)
    F_ia = jnp.zeros_like(tsurfLoc)
    # :357-465 Newton iterations (fixed count)
    for _ in range(p.IMAX_TICE):
        t1 = tsurfLoc                                                       # :370-373
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2
        qhice = sat(t1)
        cc3t = ex(p.aa1 / t1 * p.lnTEN)                                     # :393
        dqh_dTs = p.cc1 * cc3t / ((p.cc2 - cc3t * p.Ppascals) ** 2 * t2)    # :395
        F_c = effConduct * (tempFrz - tsurfLoc)                             # :403
        F_lh = D1I * UG * (qhice - aqh)                                     # :404
        F_lwu = t4 * D3                                                     # :414
        F_sens = D1 * UG * (t1 - atempLoc)                                  # :415
        F_ia = -lwdownLoc - absorbedSW + F_lwu + F_sens + F_lh              # :416-417
        dFia_dTs = 4.0 * D3 * t3 + D1 * UG + D1I * UG * dqh_dTs             # :419-420
        tsurfLoc = tsurfLoc + (F_c - F_ia) / (effConduct + dFia_dTs)        # :434-435
        tsurfLoc = jnp.minimum(tsurfLoc, TMELT)                             # :450
    # :468-535 (postSolvTempIter = 2)
    t1 = tsurfLoc
    t2 = t1 * t1
    t4 = t2 * t2
    qhice = sat(t1)
    F_lh = D1I * UG * (qhice - aqh)                                         # :507
    F_lwu = t4 * D3                                                         # :513
    F_sens = D1 * UG * (t1 - atempLoc)                                      # :514
    F_ia = -lwdownLoc - absorbedSW + F_lwu + F_sens + F_lh                  # :516-517
    FWsublim = F_lh / lhSublim                                              # :535
    TSURFout = jnp.where(ice, tsurfLoc, TSURFin)                            # :224, :473
    return TSURFout, jnp.where(ice, F_ia, 0.0), IcePenetSW, jnp.where(ice, FWsublim, 0.0)


# ------------------------------------------------------------------------------------------------ PART 2


def category_inputs(p, heffActual, hsnowActual, TICES_in):
    """seaice_growth.F:789-814: per category IT (1..multDim) ticeInMult = TICES(IT), heffActualMult =
    heffActual*pFac, hsnowActualMult = hsnowActual*pFacSnow (useMultDimSnow = F). TICES_in: [T, nITD, sNy, sNx]
    interior. Returns lists over IT (stage H03)."""
    denominator = 0.0                                                       # :370-375
    for IT in range(1, p.multDim + 1):
        denominator = denominator + IT * p.SEAICE_PDF[IT - 1]
    denominator = (2.0 * denominator) - 1.0
    recip_denominator = 1.0 / denominator
    hM, sM, tin = [], [], []
    for IT in range(1, p.multDim + 1):
        pFac = (2.0 * IT - 1.0) * recip_denominator                         # :801
        pFacSnow = 1.0                                                      # :802
        tin.append(TICES_in[:, IT - 1])                                     # :792
        hM.append(heffActual * pFac)                                        # :806
        sM.append(hsnowActual * pFacSnow)                                   # :807
    return hM, sM, tin


def heat_stocks(p, pre, a_QbyATM_open, a_QSWbyATM_open, s4t, theta_s, salt_s, maskC1, drF1):
    """seaice_growth.F:861-900 (sum over categories), 962-985 (W/m2 -> effective ice metres), 1036-1067 (ocean heat).
    s4t: list over IT of (TSURFout, F_ia, IcePenetSW, FWsublim). Returns (stage H05 dict, [TICES out per IT])."""
    QI = p.rhoIce * p.lhFusion                                              # :350
    recip_QI = 1.0 / QI                                                     # :351
    recip_rhoIce = 1.0 / p.rhoIce                                           # :340
    convertQ2HI = p.deltaTtherm / QI                                        # :363
    AREApreTH = pre["AREApreTH"]
    z = jnp.zeros_like(AREApreTH)
    a_QbyATM_cover, a_QSWbyATM_cover, a_FWbySublim = z, z, z                # :426, :432, :453
    tout = []
    for IT in range(1, p.multDim + 1):
        TSURFout, F_ia, IcePenetSW, FWsublim = s4t[IT - 1]
        tout.append(TSURFout)                                               # :879
        a_QbyATM_cover = a_QbyATM_cover + F_ia * p.SEAICE_PDF[IT - 1]       # :891-892
        a_QSWbyATM_cover = a_QSWbyATM_cover + IcePenetSW * p.SEAICE_PDF[IT - 1]   # :893-894
        a_FWbySublim = a_FWbySublim + FWsublim * p.SEAICE_PDF[IT - 1]       # :895-896
    # :962-983
    a_QbyATM_cover = a_QbyATM_cover * convertQ2HI * AREApreTH
    a_QSWbyATM_cover = a_QSWbyATM_cover * convertQ2HI * AREApreTH
    a_QbyATM_open = a_QbyATM_open * convertQ2HI * (1.0 - AREApreTH)
    a_QSWbyATM_open = a_QSWbyATM_open * convertQ2HI * (1.0 - AREApreTH)
    r_QbyATM_cover = a_QbyATM_cover
    r_QbyATM_open = a_QbyATM_open
    a_FWbySublim = p.deltaTtherm * recip_rhoIce * a_FWbySublim * AREApreTH
    r_FWbySublim = a_FWbySublim
    # :1036-1067 ocean heat
    tempFrz = p.tempFrz0 + p.dTempFrz_dS * salt_s                           # :1039-1040
    tmpscal1 = jnp.where(theta_s >= tempFrz, p.mcPheePiston,
                         p.frazilFrac * drF1 / p.deltaTtherm)               # :1042-1046
    MixedLayerTurbulenceFactor = jnp.where(AREApreTH > 0.0, 1.0 - p.mcPheeTaper * AREApreTH, 1.0)  # :1048-1057
    tmpscal2 = (-(p.HeatCapacity_Cp * p.rhoConst * recip_QI) * (theta_s - tempFrz)
                * p.deltaTtherm * maskC1)                                   # :1059-1061
    a_QbyOCN = tmpscal1 * tmpscal2 * MixedLayerTurbulenceFactor             # :1063-1064
    r_QbyOCN = a_QbyOCN
    return dict(a_QbyATM_cover=a_QbyATM_cover, a_QSWbyATM_cover=a_QSWbyATM_cover, a_QbyATM_open=a_QbyATM_open,
                a_QSWbyATM_open=a_QSWbyATM_open, r_QbyATM_cover=r_QbyATM_cover, r_QbyATM_open=r_QbyATM_open,
                a_FWbySublim=a_FWbySublim, r_FWbySublim=r_FWbySublim, a_QbyOCN=a_QbyOCN, r_QbyOCN=r_QbyOCN), tout


# ------------------------------------------------------------------------------------------------ PARTS 3-7


def thickness_updates(p, pre, hs, AREA, HEFF, HSNOW, precip, snowPrecip, HEFFM, yC):
    """seaice_growth.F:1229-1870 (PART 3: sublimation, ocean melt, snow melt, cover growth/melt, snow fall, open-water
    growth, flooding, AREA). Interior arrays; AREA/HEFF/HSNOW are the values entering SEAICE_GROWTH. Returns the
    updated AREA, HEFF, HSNOW and the increments / residuals of stage H06."""
    ICE2SNOW = p.rhoIce / p.rhoSnow                                         # :346
    SNOW2ICE = 1.0 / ICE2SNOW                                               # :347
    convertPRECIP2HI = p.deltaTtherm * p.rhoConstFresh / p.rhoIce           # :366
    recip_multDim = 1.0 / float(p.multDim)                                  # :335-336
    denominator = 0.0                                                       # :370-379
    for IT in range(1, p.multDim + 1):
        denominator = denominator + IT * p.SEAICE_PDF[IT - 1]
    denominator = (2.0 * denominator) - 1.0
    areaPDFfac = denominator * recip_multDim
    AREApreTH = pre["AREApreTH"]
    a_QbyATM_cover = hs["a_QbyATM_cover"]
    r_QbyATM_cover = hs["r_QbyATM_cover"]
    r_QbyATM_open = hs["r_QbyATM_open"]
    r_FWbySublim = hs["r_FWbySublim"]
    r_QbyOCN = hs["r_QbyOCN"]
    z = jnp.zeros_like(AREApreTH)
    d_HEFFbyATMonOCN = z                                                    # :444
    # sublimation :1229-1286
    tmpscal2 = jnp.maximum(jnp.minimum(r_FWbySublim, HSNOW * SNOW2ICE), 0.0)   # :1232, 1242
    d_HSNWbySublim = -tmpscal2 * ICE2SNOW                                   # :1243
    HSNOW = HSNOW - tmpscal2 * ICE2SNOW                                     # :1244
    r_FWbySublim = r_FWbySublim - tmpscal2                                  # :1245
    tmpscal2 = jnp.maximum(jnp.minimum(r_FWbySublim, HEFF), 0.0)            # :1256, 1264
    d_HEFFbySublim = -tmpscal2                                              # :1265
    HEFF = HEFF - tmpscal2                                                  # :1266
    r_FWbySublim = r_FWbySublim - tmpscal2                                  # :1267
    a_QbyATM_cover = a_QbyATM_cover - r_FWbySublim                          # :1282
    r_QbyATM_cover = r_QbyATM_cover - r_FWbySublim                          # :1283
    # ice-ocean interaction :1331-1340 (SEAICE_growMeltByConv = F)
    d_HEFFbyOCNonICE = jnp.maximum(r_QbyOCN, -HEFF)                         # :1333
    r_QbyOCN = r_QbyOCN - d_HEFFbyOCNonICE                                  # :1334
    HEFF = HEFF + d_HEFFbyOCNonICE                                          # :1335
    # snow melt by the atmosphere :1375-1389
    tmpscal1 = jnp.maximum(r_QbyATM_cover, -HSNOW * SNOW2ICE)               # :1379
    tmpscal2 = jnp.minimum(tmpscal1, 0.0)                                   # :1380
    d_HSNWbyATMonSNW = tmpscal2 * ICE2SNOW                                  # :1385
    HSNOW = HSNOW + tmpscal2 * ICE2SNOW                                     # :1386
    r_QbyATM_cover = r_QbyATM_cover - tmpscal2                              # :1387
    # ice growth/melt under the cover :1437-1453
    tmpscal2 = jnp.maximum(-HEFF, r_QbyATM_cover + AREApreTH * r_QbyOCN)    # :1440-1442
    d_HEFFbyATMonOCN_cover = tmpscal2                                       # :1444
    d_HEFFbyATMonOCN = d_HEFFbyATMonOCN + tmpscal2                          # :1445
    r_QbyATM_cover = r_QbyATM_cover - tmpscal2                              # :1446
    HEFF = HEFF + tmpscal2                                                  # :1447
    # snow fall :1478-1510 (snowPrecipFile = ' ')
    grow = a_QbyATM_cover >= 0.0                                            # :1483
    d_HFRWbyRAIN = jnp.where(grow, 0.0, -convertPRECIP2HI * precip * AREApreTH)            # :1485, 1490-1491
    d_HSNWbyRAIN = jnp.where(grow, convertPRECIP2HI * ICE2SNOW * precip * AREApreTH, 0.0)  # :1486-1487, 1492
    HSNOW = HSNOW + d_HSNWbyRAIN                                            # :1508
    # snow melt by the ocean :1553-1566
    tmpscal1 = jnp.maximum(r_QbyOCN * ICE2SNOW, -HSNOW)                     # :1555
    tmpscal2 = jnp.minimum(tmpscal1, 0.0)                                   # :1556
    d_HSNWbyOCNonSNW = tmpscal2                                             # :1561
    r_QbyOCN = r_QbyOCN - d_HSNWbyOCNonSNW * SNOW2ICE                       # :1562-1563
    HSNOW = HSNOW + d_HSNWbyOCNonSNW                                        # :1564
    # open-water growth :1584-1667
    tmpscal4 = HEFF                                                         # :1595
    tmpscal1 = r_QbyATM_open + r_QbyOCN * (1.0 - AREApreTH)                 # :1599-1600
    tmpscal2 = p.SWFracB * hs["a_QSWbyATM_open"]                            # :1603
    tmpscal3 = p.facOpenGrow * jnp.maximum(tmpscal1 - tmpscal2, -tmpscal4 * p.facOpenMelt) * HEFFM   # :1606-1607
    d_HEFFbyATMonOCN_open = tmpscal3                                        # :1662
    d_HEFFbyATMonOCN = d_HEFFbyATMonOCN + tmpscal3                          # :1663
    r_QbyATM_open = r_QbyATM_open - tmpscal3                                # :1664
    HEFF = HEFF + tmpscal3                                                  # :1665
    # flooding :1691-1726
    tmpscal0 = (HSNOW * p.rhoSnow + HEFF * p.rhoIce) * p.recip_rhoConst     # :1716-1717
    tmpscal1 = jnp.maximum(0.0, tmpscal0 - HEFF)                            # :1718
    d_HEFFbyFLOODING = tmpscal1                                             # :1719
    HEFF = HEFF + d_HEFFbyFLOODING                                          # :1720
    HSNOW = HSNOW - d_HEFFbyFLOODING * ICE2SNOW                             # :1721-1722
    # AREA :1791-1853 (areaGainFormula = 1, areaLossFormula = 2)
    recip_HO = jnp.where(yC < 0.0, 1.0 / p.HO_south, 1.0 / p.HO)            # :1800-1804
    recip_HH = pre["recip_heffActual"]                                      # :1806
    tmpscal4 = jnp.maximum(0.0, d_HEFFbyATMonOCN_open)                      # :1816
    tmpscal3 = jnp.minimum(0.0, d_HEFFbyATMonOCN_cover + d_HEFFbyATMonOCN_open + d_HEFFbyOCNonICE)  # :1831-1832
    AREA = jnp.where((HEFF > 0.0) | (HSNOW > 0.0),
                     jnp.maximum(0.0, jnp.minimum(p.area_max, AREA + recip_HO * tmpscal4
                                                  + 0.5 * recip_HH * tmpscal3 * areaPDFfac)),
                     0.0)                                                   # :1845-1853
    return dict(AREA=AREA, HEFF=HEFF, HSNOW=HSNOW, a_QbyATM_cover=a_QbyATM_cover,
                d_HEFFbyOCNonICE=d_HEFFbyOCNonICE, d_HEFFbyATMonOCN=d_HEFFbyATMonOCN,
                d_HEFFbyFLOODING=d_HEFFbyFLOODING, d_HEFFbyATMonOCN_open=d_HEFFbyATMonOCN_open,
                d_HEFFbyATMonOCN_cover=d_HEFFbyATMonOCN_cover, d_HSNWbyATMonSNW=d_HSNWbyATMonSNW,
                d_HSNWbyOCNonSNW=d_HSNWbyOCNonSNW, d_HSNWbyRAIN=d_HSNWbyRAIN, d_HFRWbyRAIN=d_HFRWbyRAIN,
                d_HEFFbySublim=d_HEFFbySublim, d_HSNWbySublim=d_HSNWbySublim, r_QbyATM_cover=r_QbyATM_cover,
                r_QbyATM_open=r_QbyATM_open, r_FWbySublim=r_FWbySublim, r_QbyOCN=r_QbyOCN)


def ocean_forcing(p, pre, hs, th, d_HEFFbyNEG, d_HSNWbyNEG, theta_s, salt_s, evap, precip, snowPrecip, runoff,
                  HEFFM, maskC1, yC):
    """seaice_growth.F:1987-2041 (salt flux, salt-plume flux), 2200-2230 (Qnet, Qsw in W/m2), 2257-2293 (advective
    heat flux of the ice-ocean water exchange, SEAICEheatConsFix), 2360-2399 (EmPmR), 2454-2468 (sIceLoad).
    th: thickness_updates output. Interior arrays. Returns Qnet_H06 (before the heatConsFix term, for the H06 gate)
    and the ocean forcing dict."""
    ICE2SNOW = p.rhoIce / p.rhoSnow                                         # :346
    SNOW2ICE = 1.0 / ICE2SNOW                                               # :347
    QI = p.rhoIce * p.lhFusion                                              # :350
    convertQ2HI = p.deltaTtherm / QI                                        # :363
    convertHI2Q = 1.0 / convertQ2HI                                         # :364
    convertPRECIP2HI = p.deltaTtherm * p.rhoConstFresh / p.rhoIce           # :366
    convertHI2PRECIP = 1.0 / convertPRECIP2HI                               # :367
    recip_deltaTtherm = 1.0 / p.deltaTtherm                                 # :339
    AREApreTH = pre["AREApreTH"]
    # salt flux :1989-2000
    tmpscal1 = (d_HEFFbyNEG + th["d_HEFFbyOCNonICE"] + th["d_HEFFbyATMonOCN"] + th["d_HEFFbyFLOODING"]
                + th["d_HEFFbySublim"])
    tmpscal3 = jnp.maximum(0.0, jnp.minimum(p.salt0, salt_s))              # :1996-1997
    tmpscal2 = tmpscal1 * tmpscal3 * HEFFM * recip_deltaTtherm * p.rhoIce   # :1998-1999
    saltFlux = tmpscal2                                                     # :2000
    out = dict(saltFlux=saltFlux)
    if p.useSALT_PLUME:
        localSPfrac = p.SPsalFRAC                                           # :2013
        tmpscal3 = tmpscal1 * salt_s * HEFFM * recip_deltaTtherm * p.rhoIce  # :2030-2031
        spf = jnp.maximum(tmpscal3 - tmpscal2, 0.0) * localSPfrac           # :2032-2033
        if not p.SaltPlumeSouthernOcean:                                    # :2035-2038
            spf = jnp.where(yC < 0.0, 0.0, spf)
        out["saltPlumeFlux"] = spf
    # Qnet, Qsw :2200-2230
    Qnet = (th["r_QbyATM_cover"] + th["r_QbyATM_open"] + hs["a_QSWbyATM_cover"]
            - (th["d_HEFFbyOCNonICE"] + th["d_HSNWbyOCNonSNW"] * SNOW2ICE + d_HEFFbyNEG + d_HSNWbyNEG * SNOW2ICE
               - convertPRECIP2HI * snowPrecip * (1.0 - AREApreTH)) * maskC1)
    Qsw = hs["a_QSWbyATM_cover"] + hs["a_QSWbyATM_open"]
    Qnet = Qnet * convertHI2Q                                               # :2227
    Qsw = Qsw * convertHI2Q                                                 # :2228
    Qnet_H06 = Qnet
    # :2257-2291 (temp_EvPrRn = UNSET, useRealFreshWaterFlux, nonlinFreeSurf > 0, SEAICEheatConsFix)
    tmpscal3 = p.rhoConstFresh * maskC1 * (
        (th["d_HSNWbyATMonSNW"] * SNOW2ICE + th["d_HSNWbyOCNonSNW"] * SNOW2ICE + th["d_HEFFbyOCNonICE"]
         + th["d_HEFFbyATMonOCN"] + d_HEFFbyNEG + d_HSNWbyNEG * SNOW2ICE) * convertHI2PRECIP
        - snowPrecip * (1.0 - AREApreTH))                                   # :2262-2268
    tmpscal1 = -tmpscal3 * p.HeatCapacity_Cp * theta_s                      # :2276-2277
    Qnet = Qnet + tmpscal1                                                  # :2289-2291
    # EmPmR :2362-2381
    tmpscal1 = (th["d_HSNWbyATMonSNW"] * SNOW2ICE + th["d_HFRWbyRAIN"] + th["d_HSNWbyOCNonSNW"] * SNOW2ICE
                + th["d_HEFFbyOCNonICE"] + th["d_HEFFbyATMonOCN"] + d_HEFFbyNEG + d_HSNWbyNEG * SNOW2ICE
                + th["r_FWbySublim"])
    EmPmR = maskC1 * ((evap - precip) * (1.0 - AREApreTH) - runoff
                      + tmpscal1 * convertHI2PRECIP) * p.rhoConstFresh
    # sIceLoad :2454-2468 (useRealFreshWaterFlux)
    sIceLoad = th["HEFF"] * p.rhoIce + th["HSNOW"] * p.rhoSnow
    out.update(Qnet=Qnet, Qsw=Qsw, EmPmR=EmPmR, sIceLoad=sIceLoad)
    return Qnet_H06, out


# ------------------------------------------------------------------------------------------------ driver


def seaice_growth(p, g, HEFFM, ice, ocn, exf, flx, expf=None):
    """SEAICE_GROWTH (V4r4 override, seaice_growth.F:18-2717) for all tiles, as a pure function.

    p: SeaiceGrowthParams; g: Grid (maskC, yC, drF); HEFFM [T, ny, nx] (SEAICE_INIT_FIXED).
    ice: AREA, HEFF, HSNOW, d_HEFFbyNEG, d_HSNWbyNEG [T, ny, nx] after SEAICE_REG_RIDGE; TICES [T, nITD, ny, nx].
    ocn: theta_s, salt_s [T, ny, nx] = theta, salt at kSurface = 1 (start-of-step state).
    exf: wspeed, atemp, aqh, lwdown, swdown, evap, precip, snowPrecip, runoff [T, ny, nx] (EXF, after EXF_GETFORCING).
    flx: Qnet, Qsw (read by SEAICE_BUDGET_OCEAN, then overwritten), EmPmR, saltFlux, saltPlumeFlux, sIceLoad
         [T, ny, nx]: values before the call; only their interiors are overwritten. (Full tree: DO_OCEANIC_PHYS zeroes
         saltPlumeFlux everywhere before SEAICE_MODEL under ALLOW_AUTODIFF, do_oceanic_phys.F:286-297.)
    expf: exp implementation for SEAICE_SOLVE4TEMP (static; default `default_exp()`).
    Returns (ice_out, flx_out, diag): ice_out AREA, HEFF, HSNOW, TICES (full arrays, halos unchanged); flx_out Qnet,
    Qsw,
    EmPmR, saltFlux, saltPlumeFlux, sIceLoad (full arrays); diag: the routine locals of stages H01-H06 as interior
    arrays (tests; not needed by the model)."""
    L = g.layout
    I = lambda a: interior(L, jnp.asarray(a))  # noqa: E731
    maskC1 = I(g.maskC[:, 0])
    yC = I(g.yC)
    drF1 = g.drF[0]
    AREA, HEFF, HSNOW = I(ice["AREA"]), I(ice["HEFF"]), I(ice["HSNOW"])
    TICES = jnp.asarray(ice["TICES"])
    theta_s, salt_s = I(ocn["theta_s"]), I(ocn["salt_s"])
    e = {k: I(v) for k, v in exf.items()}
    HEFFMi = I(HEFFM)
    # PART 1
    pre = growth_pre_budget(p, AREA, HEFF, HSNOW, e["wspeed"], theta_s)
    # PART 2
    a_QbyATM_open, a_QSWbyATM_open = budget_ocean(I(flx["Qnet"]), I(flx["Qsw"]))
    hM, sM, tin = category_inputs(p, pre["heffActual"], pre["hsnowActual"], I(TICES))
    s4t = [solve4temp(p, pre["UG"], hM[k], sM[k], tin[k], e["lwdown"], e["atemp"], e["swdown"], e["aqh"], salt_s, yC,
                      expf=expf) for k in range(p.multDim)]
    hs, tout = heat_stocks(p, pre, a_QbyATM_open, a_QSWbyATM_open, s4t, theta_s, salt_s, maskC1, drF1)
    # PARTS 3-7
    th = thickness_updates(p, pre, hs, AREA, HEFF, HSNOW, e["precip"], e["snowPrecip"], HEFFMi, yC)
    dNEG, dSNEG = I(ice["d_HEFFbyNEG"]), I(ice["d_HSNWbyNEG"])
    Qnet_H06, of = ocean_forcing(p, pre, hs, th, dNEG, dSNEG, theta_s, salt_s, e["evap"], e["precip"],
                                 e["snowPrecip"], e["runoff"], HEFFMi, maskC1, yC)
    TICESi = I(TICES)
    for k in range(p.multDim):
        TICESi = TICESi.at[:, k].set(tout[k])                               # :794, :879
    ice_out = dict(AREA=set_interior(L, ice["AREA"], th["AREA"]), HEFF=set_interior(L, ice["HEFF"], th["HEFF"]),
                   HSNOW=set_interior(L, ice["HSNOW"], th["HSNOW"]), TICES=set_interior(L, TICES, TICESi))
    flx_out = {k: set_interior(L, flx[k], of[k]) if k in of else jnp.asarray(flx[k]) for k in flx}
    diag = dict(pre=pre, a_QbyATM_open_Wm2=a_QbyATM_open, a_QSWbyATM_open_Wm2=a_QSWbyATM_open,
                heffActualMult=hM, hsnowActualMult=sM, ticeInMult=tin, s4t=s4t, hs=hs, th=th, Qnet_H06=Qnet_H06)
    return ice_out, flx_out, diag
