"""pkg/seaice: SEAICE_DYNSOLVER as ECCO v4r4 runs it (plan M2.4), pure functions.

Literal port of c66g `pkg/seaice/seaice_dynsolver.F` (SEAICE_CGRID) with the routines it calls in V4r4:
`seaice_get_dynforcing.F` (useAtmWind = F: stress from the ocean surface stress fu/fv), `seaice_calc_ice_strength.F`
(Hibler 1979 strength, SEAICEpresPow0/1 = 1), `seaice_freedrift.F` (called because LSR_mixIniGuess = 0; its uice_fd,
vice_fd are not used by the LSR solve in V4r4 and it exchanges TAUX/TAUY), `seaice_lsr.F` (Picard-LSR solver; see
mitgcm_jax/pkgs/seaice_lsr.py for the solver and the AD design decision) and `seaice_ocean_stress.F`, followed by the
velocity clipping (SEAICE_ALLOW_CLIPVELS, SEAICE_clipVelocities = T). No V4r4 override of these files. CPP options:
`ECCOv4 Release 4/code/SEAICE_OPTIONS.h` (SEAICE_CGRID, SEAICE_ALLOW_DYNAMICS, SEAICE_ALLOW_FREEDRIFT,
SEAICE_ALLOW_EVP, SEAICE_ALLOW_CLIPVELS defined; SEAICE_ALLOW_JFNK/KRYLOV, SEAICE_ALLOW_BOTTOMDRAG undefined),
CPP_OPTIONS.h (ATMOSPHERIC_LOADING) and pkg/autodiff (ALLOW_AUTODIFF_TAMC defined: the TAMC re-initialisations are
part of the forward). Parameters: data.seaice (SEAICE_PARM01), data, eedata, data.exf of the run directory; defaults
cited from seaice_readparms.F. Values the V4r4 run does not use raise NotImplementedError in
`SeaiceDynParams.from_namelists`.

Not ported (no effect on any output in V4r4):
  - seaice_dynsolver.F:73-92 (TAMC re-initialisation of PRESS0, ZMAX, ZMIN, TAUX, TAUY, uice_fd, vice_fd on
    2-OL..sN+OL): every one of these points is overwritten before it is read (TAUX/TAUY zeroed at :176-191 and set by
    SEAICE_GET_DYNFORCING on all points, PRESS0/ZMAX/ZMIN by SEAICE_CALC_ICE_STRENGTH on all points, uice_fd/vice_fd
    zeroed on all points by SEAICE_FREEDRIFT).
  - phiSurf (:207-231): read only by the tilt term (SEAICEuseTILT = F).
  - SEAICE_RESIDUAL calls inside SEAICE_LSR (printing only).

Math library. gfortran 11 VECTORISES the EXP of SEAICE_CALC_ICE_STRENGTH: seaice_calc_ice_strength.o calls glibc
libmvec `_ZGVbN2v_exp` (glibc's math-vector-fortran.h declares EXP simd for gfortran, no -ffast-math needed), whose
SSE4.1 kernel (libmvec-2.28, SVML-derived, ~1-4 ulp) is not libm's exp. Measured on the oracle's PRESS0: glibc scalar
exp differs at 2978 points, XLA's exp at ~3100, libmvec's vector exp (called through a C shim) at 0. `exp_libmvec`
below is a transliteration of that kernel (disassembled from /usr/lib64/libmvec.so.1, _ZGVbN2v_exp -> SSE4 variant;
table = correctly rounded 2^(j/1024), verified equal to the library's; now in mitgcm_jax/ops/libm.py) and makes
PRESS0/ZMAX bitwise. ATAN2, SQRT agree bitwise with glibc. SIN/COS in SEAICE_FREEDRIFT are one glibc `sincos` call
(gcc fuses them), whose cos-part can differ from cos() by up to 2 ulp; uice_fd/vice_fd are diagnostics in V4r4. The
turning-angle SIN/COS (argument 0) are exact in every library.

Arrays: `[tile, j, i]` with halos (layout.py). Grid fields from `Grid`; the sea-ice fields set at initialisation
(HEFFM, seaiceMaskU/V, tensileStrFac: seaice_init_varia.F:67-74, 384-387, 312/708; k1AtC, k1AtZ, k2AtC, k2AtZ:
SEAICE_INIT_FIXED metric terms) in a dict `sg` (not ported here: M2.6 takes them from the dumps or ports the init).

Gates (mitgcm_jax/tests/test_seaice_dyn.py, oracle full_jaxdump_v5, iterations 1-3): the whole SEAICE_DYNSOLVER from
SEAICE_MODEL's inputs to stage I01 is bitwise at every point of every dumped stage (Y01-Y06, L01-L04 per Picard pass,
I01), LSOR sweep counts included, except uice_fd/vice_fd (1 ulp at 27-45 points, glibc sincos; read by nothing).
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp

from mitgcm_jax.ops import libm as _libm
from mitgcm_jax.ops.ad_skip import skipped_in_reverse
from mitgcm_jax.params_io import params_pytree
from mitgcm_jax.pkgs import seaice_lsr as lsr

HALF = 0.5  # SEAICE_PARAMS.h:565
ONE = 1.0  # SEAICE_PARAMS.h:563
ZERO = 0.0
UNSET_RL = 1.234567e5  # EEPARAMS.h UNSET_RL
UNSET_I = 123456789  # EEPARAMS.h UNSET_I
deg2rad = 2.0 * 3.14159265358979323844 / 360.0  # EEPARAMS.h: deg2rad = 2.d0*PI/360.d0


@params_pytree  # float fields are traced leaves: pass the params as a jit argument (KERNEL_GUIDE)
@dataclass(frozen=True)
class SeaiceDynParams:
    SEAICE_rhoIce: float  # seaice_readparms.F:349  0.91e3
    SEAICE_drag: float  # :352  0.002 (data.seaice 0.001)
    SEAICE_drag_south: float  # :360, :661-662  = SEAICE_drag
    OCEAN_drag: float  # :353  0.001
    SEAICE_waterDrag: float  # :354  5.5
    SEAICE_waterDrag_south: float  # :361, :663-664  = SEAICE_waterDrag
    SEAICE_strength: float  # data.seaice 2.25e4
    SEAICE_cStar: float  # :381  20.
    SEAICE_area_max: float  # :489  1.00 (data.seaice 0.97)
    SEAICEpresH0: float  # data.seaice 2.
    SEAICE_zetaMaxFac: float  # :324  2.5e8
    SEAICE_zetaMin: float  # :323  0.
    SEAICE_eccen: float  # :383  2.
    SEAICE_deltaMin: float  # :496, :649  = SEAICE_EPS = 1e-10
    SEAICEpressReplFac: float  # :382  1.
    SEAICE_deltaTdyn: float  # :294  dTtracerLev(1)
    SEAICE_LSRrelaxU: float  # :471  0.95
    SEAICE_LSRrelaxV: float  # :472  0.95
    LSR_ERROR: float  # :483  1e-4 (data.seaice 2e-4)
    SEAICEstressFactor: float  # :458  1.
    SINWAT: float  # SIN(SEAICE_waterTurnAngle*deg2rad), seaice_lsr.F:257 (glibc sincos at setup)
    COSWAT: float  # COS(...)
    SINWIN: float  # seaice_get_dynforcing.F (air turning angle)
    COSWIN: float
    SEAICEpresPow0: int  # data.seaice 1
    SEAICEpresPow1: int
    SEAICEnonLinIterMax: int  # :797 = 2 with LSR
    SEAICElinearIterMax: int  # :808 = 1500
    SOLV_NCHECK: int  # :473  2
    MPSEUDOTIMESTEPS: int  # SEAICE_SIZE.h:43  2
    SEAICE_no_slip: bool
    SEAICE_clipVelocities: bool
    useCubedSphereExchange: bool  # eedata
    lsr_ad_restart: int = 40  # JAX only: GMRES restart length of the implicit-derivative solves (seaice_lsr.py)
    lsr_ad_cycles: int = 8  # JAX only: fixed number of GMRES restart cycles
    # JAX only: implementation of the LSOR sweeps (seaice_lsr.py "Implementations"; the same operations in the same
    # order, bitwise equal): "auto" (CPU: "xla"; CUDA: "pallas" forward; other backends: "xla_unrolled"), or an
    # explicit "pallas" | "xla_unrolled" | "xla" (| "pallas_interpret": the Pallas kernel interpreted, CPU tests)
    lsr_impl: str = "auto"

    @classmethod
    def from_namelists(cls, nml, lsr_ad_restart=40, lsr_ad_cycles=8):
        f, grp = "data.seaice", "SEAICE_PARM01"

        def get(key, default):
            return nml.get(f, grp, key, default=default)

        def flag(key, default):
            return bool(get(key, default))

        # branches the port covers (NotImplementedError otherwise)
        must = dict(SEAICEuseDYNAMICS=(True, True), SEAICEuseFREEDRIFT=(False, False), SEAICEuseEVP=(False, False),
                    SEAICEuseJFNK=(False, False), SEAICEuseKrylov=(False, False), SEAICEuseTILT=(False, True),
                    SEAICEscaleSurfStress=(False, False), SEAICEaddSnowMass=(False, False),
                    SEAICE_maskRHS=(False, False), useHB87stressCoupling=(False, False),
                    SEAICEuseStrImpCpl=(False, False), SEAICEuseBDF2=(False, False),
                    SEAICEusePicardAsPrecon=(False, False), SEAICEuseTEM=(False, False),
                    SEAICE_no_slip=(True, False), SEAICE_clipVelocities=(True, False))
        # (value the port needs, Fortran default: seaice_readparms.F:261-500)
        for key, (need, default) in must.items():
            if flag(key, default) != need:
                raise NotImplementedError(f"data.seaice {key} = {not need}: branch not ported (V4r4: {need})")
        for key, need, default in (("SEAICEetaZmethod", 0, 0), ("LSR_mixIniGuess", 0, 0),
                                   ("SEAICE_OLx", 0, UNSET_I), ("SEAICE_OLy", 0, UNSET_I)):
            v = int(get(key, default))
            if key.startswith("SEAICE_OL") and v == UNSET_I:
                v = 0  # seaice_readparms.F:821-828 (LSR)
            if v != need:
                raise NotImplementedError(f"data.seaice {key} = {v}: not ported (V4r4: {need})")
        if float(get("SEAICE_tensilFac", 0.0)) != 0.0:  # :384; tensileStrFac enters as an input field anyway
            pass
        if nml.get("data.exf", "EXF_NML_01", "useAtmWind", default=False):  # exf_readparms.F: useAtmWind
            raise NotImplementedError("useAtmWind = T: SEAICE_GET_DYNFORCING wind branch not ported")
        cubed = bool(nml.get("eedata", "EEPARMS", "useCubedSphereExchange", default=False))  # eeset_parms.F:104
        nonLin = int(get("SEAICEnonLinIterMax", UNSET_I))
        if nonLin == UNSET_I:
            nonLin = 2  # seaice_readparms.F:797
        nonLin = max(nonLin, 2)  # :801-802
        if nonLin > 2:
            raise NotImplementedError("SEAICEnonLinIterMax > 2: not ported (V4r4: 2)")
        linMax = int(get("SEAICElinearIterMax", UNSET_I))
        if linMax == UNSET_I:
            linMax = 1500  # seaice_readparms.F:808
        p0, p1 = int(get("SEAICEpresPow0", 1)), int(get("SEAICEpresPow1", 1))
        if (p0, p1) != (1, 1):
            raise NotImplementedError("SEAICEpresPow0/1 != 1: not ported")
        dT = float(nml.get("data", "PARM03", "deltaTtracer",
                           default=nml.get("data", "PARM03", "deltaT", default=0.0)))  # dTtracerLev(1)
        dTdyn = float(get("SEAICE_deltaTdyn", dT))  # seaice_readparms.F:294
        dTtherm = float(get("SEAICE_deltaTtherm", dT))  # :293
        if dTdyn != dTtherm:
            raise NotImplementedError("SEAICE_deltaTdyn != SEAICE_deltaTtherm (DIFFERENT_MULTIPLE branch) not ported")
        drag = float(get("SEAICE_drag", 0.002))
        wdrag = float(get("SEAICE_waterDrag", 5.5))
        drag_s = float(get("SEAICE_drag_south", UNSET_RL))
        wdrag_s = float(get("SEAICE_waterDrag_south", UNSET_RL))
        eps = float(get("SEAICE_EPS", 1e-10))  # :497
        dmin = float(get("SEAICE_deltaMin", UNSET_RL))
        eccen = float(get("SEAICE_eccen", 2.0))
        if eccen == 0.0:
            raise NotImplementedError("SEAICE_eccen = 0 not ported")
        wta = float(get("SEAICE_waterTurnAngle", 0.0))  # :492
        ata = float(get("SEAICE_airTurnAngle", 0.0))  # :491
        if wta != 0.0 or ata != 0.0:
            # the Fortran calls glibc sincos (gcc fuses SIN/COS): grid/load.py _sincos would reproduce it
            raise NotImplementedError("non-zero turning angles not ported (V4r4: 0)")
        return cls(
            SEAICE_rhoIce=float(get("SEAICE_rhoIce", 0.91e3)),
            SEAICE_drag=drag, SEAICE_drag_south=drag if drag_s == UNSET_RL else drag_s,
            OCEAN_drag=float(get("OCEAN_drag", 0.001)),
            SEAICE_waterDrag=wdrag, SEAICE_waterDrag_south=wdrag if wdrag_s == UNSET_RL else wdrag_s,
            SEAICE_strength=float(get("SEAICE_strength", 2.75e4)),  # :380
            SEAICE_cStar=float(get("SEAICE_cStar", 20.0)),
            SEAICE_area_max=float(get("SEAICE_area_max", 1.0)),
            SEAICEpresH0=float(get("SEAICEpresH0", 1.0)),  # :325
            SEAICE_zetaMaxFac=float(get("SEAICE_zetaMaxFac", 2.5e8)),
            SEAICE_zetaMin=float(get("SEAICE_zetaMin", 0.0)),
            SEAICE_eccen=eccen,
            SEAICE_deltaMin=eps if dmin == UNSET_RL else dmin,
            SEAICEpressReplFac=float(get("SEAICEpressReplFac", 1.0)),
            SEAICE_deltaTdyn=dTdyn,
            SEAICE_LSRrelaxU=float(get("SEAICE_LSRrelaxU", 0.95)),
            SEAICE_LSRrelaxV=float(get("SEAICE_LSRrelaxV", 0.95)),
            LSR_ERROR=float(get("LSR_ERROR", 0.0001)),
            SEAICEstressFactor=float(get("SEAICEstressFactor", 1.0)),
            SINWAT=math.sin(wta * deg2rad), COSWAT=math.cos(wta * deg2rad),
            SINWIN=math.sin(ata * deg2rad), COSWIN=math.cos(ata * deg2rad),
            SEAICEpresPow0=p0, SEAICEpresPow1=p1,
            SEAICEnonLinIterMax=nonLin, SEAICElinearIterMax=linMax,
            SOLV_NCHECK=int(get("SOLV_NCHECK", 2)), MPSEUDOTIMESTEPS=2,
            SEAICE_no_slip=True, SEAICE_clipVelocities=True, useCubedSphereExchange=cubed,
            lsr_ad_restart=lsr_ad_restart, lsr_ad_cycles=lsr_ad_cycles)


# ---------------------------------------------------------------------------------------------------------------
# glibc libmvec _ZGVbN2v_exp (SSE4.1 kernel), the EXP gfortran calls in the vectorised SEAICE_CALC_ICE_STRENGTH loop:
# mitgcm_jax/ops/libm.py (module global here, so calc_ice_strength looks it up at call time).
exp_libmvec = _libm.exp_libmvec


# ---------------------------------------------------------------------------------------------------------------


def _full(L):
    return L.js(1 - L.OLy, L.sNy + L.OLy), L.is_(1 - L.OLx, L.sNx + L.OLx)


def ice_mass(p, g, HEFF, seaiceMassC, seaiceMassU, seaiceMassV):
    """seaice_dynsolver.F:112-122 on j,i = 2-OL..sN+OL (SEAICEaddSnowMass = F); the first halo row/column keeps the
    passed values."""
    L = g.layout
    seaiceMassC, seaiceMassU, seaiceMassV = (jnp.asarray(a) for a in (seaiceMassC, seaiceMassU, seaiceMassV))
    J, I = L.js(2 - L.OLy, L.sNy + L.OLy), L.is_(2 - L.OLx, L.sNx + L.OLx)
    Jm, Im = L.js(1 - L.OLy, L.sNy + L.OLy - 1), L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    mC = p.SEAICE_rhoIce * HEFF[:, J, I]  # :116
    mU = (p.SEAICE_rhoIce * HALF) * (HEFF[:, J, I] + HEFF[:, J, Im])  # :117-118
    mV = (p.SEAICE_rhoIce * HALF) * (HEFF[:, J, I] + HEFF[:, Jm, I])  # :119-120
    return (seaiceMassC.at[:, J, I].set(mC), seaiceMassU.at[:, J, I].set(mU), seaiceMassV.at[:, J, I].set(mV))


def get_dynforcing(p, g, fu, fv):
    """SEAICE_GET_DYNFORCING (seaice_get_dynforcing.F:244-262, useAtmWind = F): TAUX, TAUY on every point."""
    south = g.yC < ZERO  # :250
    CDAIR = jnp.where(south, p.SEAICE_drag_south / p.OCEAN_drag, p.SEAICE_drag / p.OCEAN_drag)  # :250-254
    TAUX = (CDAIR * fu) * g.maskW[:, 0]  # :255-256
    TAUY = (CDAIR * fv) * g.maskS[:, 0]  # :257-258
    return TAUX, TAUY


def calc_ice_strength(p, HEFF, AREA, HEFFM):
    """SEAICE_CALC_ICE_STRENGTH (seaice_calc_ice_strength.F:84-122, Hibler 1979, presPow0 = presPow1 = 1:
    tmpscal2 = HEFF) on every point: PRESS0, ZMAX, ZMIN."""
    tmpscal2 = HEFF  # :104-116 (both power branches need SEAICEpresPow* .NE. 1)
    # :117-118 (EXP vectorised by gfortran: libmvec _ZGVbN2v_exp, see module docstring)
    PRESS0 = (p.SEAICE_strength * tmpscal2) * exp_libmvec(-p.SEAICE_cStar * (p.SEAICE_area_max - AREA))
    ZMAX = p.SEAICE_zetaMaxFac * PRESS0  # :119
    ZMIN = jnp.zeros_like(PRESS0) + p.SEAICE_zetaMin  # :120
    PRESS0 = PRESS0 * HEFFM  # :121
    return PRESS0, ZMAX, ZMIN


def freedrift(p, g, ex, TAUX, TAUY, FORCEX0, FORCEY0, HEFF, uVel, vVel):
    """SEAICE_FREEDRIFT (seaice_freedrift.F:55-181): returns TAUX, TAUY (after EXCH_UV_XY_RL) and uice_fd, vice_fd."""
    L = g.layout
    HEFF = jnp.asarray(HEFF)
    TAUX, TAUY = ex.exch_uv_xy(TAUX, TAUY, True)  # :71 EXCH_UV_XY_RL( TAUX, TAUY, .TRUE. )
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    Jp, Ip = L.js(2, L.sNy + 1), L.is_(2, L.sNx + 1)
    taux_c = HALF * (FORCEX0[:, J, I] + FORCEX0[:, J, Ip])  # :84-85
    tauy_c = HALF * (FORCEY0[:, J, I] + FORCEY0[:, Jp, I])  # :86-87
    mIceCor = (p.SEAICE_rhoIce * HEFF[:, J, I]) * g.fCori[:, J, I]  # :89
    uvel_c = HALF * (uVel[:, J, I] + uVel[:, J, Ip])  # :91
    vvel_c = HALF * (vVel[:, J, I] + vVel[:, Jp, I])  # :92
    rhs_x = -taux_c - mIceCor * vvel_c  # :94
    rhs_y = -tauy_c + mIceCor * uvel_c  # :95
    t1 = rhs_x * rhs_x + rhs_y * rhs_y  # :98
    pos = t1 > ZERO
    rhs_n = jnp.where(pos, jnp.sqrt(jnp.where(pos, rhs_x * rhs_x + rhs_y * rhs_y, 1.0)), 0.0)  # :99-105
    rhs_a = jnp.where(pos, jnp.arctan2(jnp.where(pos, rhs_y, 0.0), jnp.where(pos, rhs_x, 1.0)), 0.0)
    south = g.yC[:, J, I] < ZERO  # :109
    Cd = jnp.where(south, p.SEAICE_waterDrag_south, p.SEAICE_waterDrag)
    tmp1 = 1.0 / Cd  # :110-113
    tmp2 = ((tmp1 * tmp1) * mIceCor) * mIceCor  # :115
    tmp3 = ((tmp1 * tmp1) * rhs_n) * rhs_n  # :116
    tmp4 = tmp2 * tmp2 + 4.0 * tmp3  # :118
    pos = tmp3 > ZERO
    inner = jnp.where(pos, HALF * (jnp.sqrt(jnp.where(pos, tmp4, 1.0)) - tmp2), 1.0)
    sol_n = jnp.where(pos, jnp.sqrt(inner), 0.0)  # :119-123
    tmp2 = (Cd * sol_n) * sol_n  # :127-133
    tmp3 = mIceCor * sol_n  # :134
    tmp4 = tmp2 * tmp2 + tmp3 * tmp3  # :136
    pos = tmp4 > ZERO
    sol_a = jnp.where(pos, rhs_a - jnp.arctan2(jnp.where(pos, tmp3, 0.0), jnp.where(pos, tmp2, 1.0)), 0.0)  # :137-141
    # :145-146 (gcc: one glibc sincos(sol_a); XLA cos/sin here, see module docstring)
    z = jnp.zeros_like(HEFF)
    uice_cntr = z.at[:, J, I].set(uvel_c - sol_n * jnp.cos(sol_a))
    vice_cntr = z.at[:, J, I].set(vvel_c - sol_n * jnp.sin(sol_a))
    uice_cntr, vice_cntr = ex.exch_uv_agrid(uice_cntr, vice_cntr, True)  # :157 EXCH_UV_AGRID_3D_RL(..,.TRUE.,1)
    Jm, Im = L.js(0, L.sNy - 1), L.is_(0, L.sNx - 1)
    uice_fd = z.at[:, J, I].set(HALF * (uice_cntr[:, J, Im] + uice_cntr[:, J, I]))  # :163-164
    vice_fd = z.at[:, J, I].set(HALF * (vice_cntr[:, Jm, I] + vice_cntr[:, J, I]))  # :165-166
    uice_fd, vice_fd = ex.exch_uv_xy(uice_fd, vice_fd, True)  # :172
    uice_fd = uice_fd * g.maskW[:, 0]  # :179
    vice_fd = vice_fd * g.maskS[:, 0]  # :180
    return TAUX, TAUY, uice_fd, vice_fd


def ocean_stress(p, g, ex, fu, fv, uIce, vIce, DWATN, AREA, uVel, vVel):
    """SEAICE_OCEAN_STRESS (seaice_ocean_stress.F:91-128, useHB87StressCoupling = F): fu, fv on the interior, then
    EXCH_UV_XY_RS(fu, fv, .TRUE.)."""
    L = g.layout
    fu, fv = jnp.asarray(fu), jnp.asarray(fv)
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    Jm, Jp, Im, Ip = L.js(0, L.sNy - 1), L.js(2, L.sNy + 1), L.is_(0, L.sNx - 1), L.is_(2, L.sNx + 1)
    D = DWATN
    sgn = jnp.copysign(p.SINWAT, g.fCori[:, J, I])  # SIGN(SINWAT, fCori)
    fuIce = ((HALF * (D[:, J, I] + D[:, J, Im])) * p.COSWAT) * (uIce[:, J, I] - uVel[:, J, I]) \
        - (sgn * 0.5) * ((D[:, J, I] * 0.5) * (((vIce[:, J, I] - vVel[:, J, I]) + vIce[:, Jp, I]) - vVel[:, Jp, I])
                         + (D[:, J, Im] * 0.5) * (((vIce[:, J, Im] - vVel[:, J, Im]) + vIce[:, Jp, Im])
                                                  - vVel[:, Jp, Im]))  # :95-105
    fvIce = ((HALF * (D[:, J, I] + D[:, Jm, I])) * p.COSWAT) * (vIce[:, J, I] - vVel[:, J, I]) \
        + (sgn * 0.5) * ((D[:, J, I] * 0.5) * (((uIce[:, J, I] - uVel[:, J, I]) + uIce[:, J, Ip]) - uVel[:, J, Ip])
                         + (D[:, Jm, I] * 0.5) * (((uIce[:, Jm, I] - uVel[:, Jm, I]) + uIce[:, Jm, Ip])
                                                  - uVel[:, Jm, Ip]))  # :106-116
    areaW = (0.5 * (AREA[:, J, I] + AREA[:, J, Im])) * p.SEAICEstressFactor  # :117-118
    areaS = (0.5 * (AREA[:, J, I] + AREA[:, Jm, I])) * p.SEAICEstressFactor  # :119-120
    fu = fu.at[:, J, I].set((ONE - areaW) * fu[:, J, I] + areaW * fuIce)  # :121
    fv = fv.at[:, J, I].set((ONE - areaS) * fv[:, J, I] + areaS * fvIce)  # :122
    return ex.exch_uv_xy(fu, fv, True)  # :128


# ---------------------------------------------------------------------------------------------------------------
# the IF (SEAICEuseDYNAMICS) parts of SEAICE_DYNSOLVER, as separate blocks for the adjoint level "no_dynamics"
# (SEAICEuseDYNAMICSswitchInAd = .TRUE.: TAF skips exactly these blocks in the reverse sweep, ops/ad_skip.py)

# variables the SEAICEuseDYNAMICS block (seaice_dynsolver.F:270-341) overwrites and that later code reads: uIce, vIce
# (SEAICE_LSR; read by SEAICE_OCEAN_STRESS, SEAICE_ADVDIFF, the next step), TAUX, TAUY (EXCH_UV_XY_RL in
# SEAICE_FREEDRIFT), e11, e22, e12, DWATN, FORCEX, FORCEY (SEAICE_LSR writes them only partly: the other points carry
# over to the next step; DWATN is read by SEAICE_OCEAN_STRESS). Its other results (uIceNm1, vIceNm1, deltaC, ETA,
# etaZ, ZETA, zetaZ, PRESS, uice_fd, vice_fd) are recomputed from scratch (zeroed first) every time and read by nothing
# outside the block.
DYN_BLOCK_INOUT = ("uIce", "vIce", "TAUX", "TAUY", "e11", "e22", "e12", "DWATN", "FORCEX", "FORCEY")
CLIP_INOUT = ("uIce", "vIce")
UICE_CLIP = 0.40  # seaice_dynsolver.F:378, :380  MAX(MIN(uIce, 0.40 _d +00), -0.40 _d +00)


def dynamics_block(record, max_iter, lsr_error, ins, rest):
    """seaice_dynsolver.F:270-341, the IF (SEAICEuseDYNAMICS) block: SEAICE_FREEDRIFT (:273-277, LSR_mixIniGuess = 0;
    SEAICEuseFREEDRIFT = F: :278-291 not executed) and SEAICE_LSR (:313-316); OBCS, EVP, Krylov, JFNK not used.
    ins: DYN_BLOCK_INOUT + the read-only FORCEX0, FORCEY0, HEFF, uVel, vVel, seaiceMassC/U/V, PRESS0, ZMAX, ZMIN;
    rest = (p, g, sg, ex). Returns (out, passes): out = seaice_lsr's results + TAUX, TAUY, uice_fd, vice_fd."""
    p, g, sg, ex = rest
    TAUX, TAUY, uice_fd, vice_fd = freedrift(p, g, ex, ins["TAUX"], ins["TAUY"], ins["FORCEX0"], ins["FORCEY0"],
                                             ins["HEFF"], ins["uVel"], ins["vVel"])
    ls = {k: ins[k] for k in ("uIce", "vIce", "uVel", "vVel", "seaiceMassC", "seaiceMassU", "seaiceMassV", "FORCEX0",
                              "FORCEY0", "PRESS0", "ZMAX", "ZMIN", "e11", "e22", "e12", "DWATN", "FORCEX", "FORCEY")}
    lo, passes = lsr.seaice_lsr(p, g, sg, ex, ls, record=record, max_iter=max_iter, lsr_error=lsr_error)
    return dict(lo, TAUX=TAUX, TAUY=TAUY, uice_fd=uice_fd, vice_fd=vice_fd), passes


def clip_block(ins, rest):
    """seaice_dynsolver.F:365-385 (SEAICE_ALLOW_CLIPVELS, IF (SEAICEuseDYNAMICS .AND. SEAICE_clipVelocities)): ice
    velocities capped to +-0.40 m/s on the full tile."""
    uIce = jnp.maximum(jnp.minimum(ins["uIce"], UICE_CLIP), -UICE_CLIP)
    vIce = jnp.maximum(jnp.minimum(ins["vIce"], UICE_CLIP), -UICE_CLIP)
    return dict(uIce=uIce, vIce=vIce), None


_dynamics_block_skipped = skipped_in_reverse(dynamics_block, DYN_BLOCK_INOUT, n_static=3)
_clip_block_skipped = skipped_in_reverse(clip_block, CLIP_INOUT)
AD_DYNAMICS = ("exact", "skip")


def dynsolver(p, g, sg, ex, st, record=False, max_iter=None, lsr_error=None, ad_dynamics="exact"):
    """SEAICE_DYNSOLVER (seaice_dynsolver.F:100-387) for one step.

    st (the values on entry, [T, ny, nx]): HEFF, AREA, uIce, vIce, uVel, vVel (surface level of DYNVARS uVel/vVel),
    fu, fv (FFIELDS, after EXF); and the arrays the routine writes only partly, as they are on entry (their other
    points keep these values): seaiceMassC/U/V, FORCEX0, FORCEY0, e11, e22, e12, DWATN, FORCEX, FORCEY.
    sg: HEFFM, k1AtC, k1AtZ, k2AtC, k2AtZ, seaiceMaskU, seaiceMaskV, tensileStrFac.
    ad_dynamics (static; derivatives only, the forward is byte-identical): "exact" = the derivative of everything
    (implicit LSR derivative, seaice_lsr.py); "skip" = what TAF computes with SEAICEuseDYNAMICSswitchInAd = .TRUE.
    (autodiff_inadmode_set_ad.F:49-51: SEAICEuseDYNAMICS = .FALSE. in the reverse sweep): the two IF
    (SEAICEuseDYNAMICS) blocks, `dynamics_block` (:270-341) and `clip_block` (:365-385), are identity maps on the
    variables they overwrite (DYN_BLOCK_INOUT, CLIP_INOUT) and pass nothing to the ones they only read
    (ops/ad_skip.py); ice masses, GET_DYNFORCING, FORCEX0/Y0, CALC_ICE_STRENGTH and OCEAN_STRESS keep their exact
    derivatives (outside the IF, :112-268 and :361).
    Returns (out, rec): out = the groups u, y, v and fu, fv at stage I01_dynsolver (UICE, VICE after clipping,
    seaiceMassC/U/V, TAUX, TAUY, FORCEX0/Y0, PRESS0, ZMAX, ZMIN, uice_fd, vice_fd, e11, e22, e12, deltaC, ETA,
    etaZ, ZETA, zetaZ, PRESS, DWATN, FORCEX, FORCEY, uIceNm1, vIceNm1, stressDivergenceX/Y, fu, fv);
    rec (record=True) = the stage values Y01..Y06 and the per-pass LSR records (seaice_lsr.seaice_lsr)."""
    if ad_dynamics not in AD_DYNAMICS:
        raise ValueError(f"ad_dynamics={ad_dynamics!r}: one of {AD_DYNAMICS}")
    skip = ad_dynamics == "skip"
    L = g.layout
    st = {k: jnp.asarray(v) for k, v in st.items()}
    rec = {}
    # :100-102 DIFFERENT_MULTIPLE(SEAICE_deltaTdyn, myTime, SEAICE_deltaTtherm): always true (dyn = therm step)
    mC, mU, mV = ice_mass(p, g, st["HEFF"], st["seaiceMassC"], st["seaiceMassU"], st["seaiceMassV"])
    zero = jnp.zeros_like(st["HEFF"])
    stressDivergenceX = zero  # :176-191 (ALLOW_AUTODIFF_TAMC, SEAICE_ALLOW_EVP)
    stressDivergenceY = zero
    TAUX, TAUY = get_dynforcing(p, g, st["fu"], st["fv"])  # :198-201
    if record:
        rec["Y01"] = dict(seaiceMassC=mC, seaiceMassU=mU, seaiceMassV=mV, TAUX=TAUX, TAUY=TAUY)
    J, I = L.js(2 - L.OLy, L.sNy + L.OLy), L.is_(2 - L.OLx, L.sNx + L.OLx)
    FORCEX0 = st["FORCEX0"].at[:, J, I].set(TAUX[:, J, I])  # :243-248 (SEAICEscaleSurfStress = F)
    FORCEY0 = st["FORCEY0"].at[:, J, I].set(TAUY[:, J, I])
    PRESS0, ZMAX, ZMIN = calc_ice_strength(p, st["HEFF"], st["AREA"], sg["HEFFM"])  # :265
    if record:
        rec["Y02"] = dict(FORCEX0=FORCEX0, FORCEY0=FORCEY0, PRESS0=PRESS0, ZMAX=ZMAX, ZMIN=ZMIN)
    # :270-341 IF (SEAICEuseDYNAMICS): SEAICE_FREEDRIFT (:273-277) + SEAICE_LSR (:313-316)
    blk_in = dict(uIce=st["uIce"], vIce=st["vIce"], TAUX=TAUX, TAUY=TAUY, e11=st["e11"], e22=st["e22"],
                  e12=st["e12"], DWATN=st["DWATN"], FORCEX=st["FORCEX"], FORCEY=st["FORCEY"], FORCEX0=FORCEX0,
                  FORCEY0=FORCEY0, HEFF=st["HEFF"], uVel=st["uVel"], vVel=st["vVel"], seaiceMassC=mC,
                  seaiceMassU=mU, seaiceMassV=mV, PRESS0=PRESS0, ZMAX=ZMAX, ZMIN=ZMIN)
    blk = _dynamics_block_skipped if skip else dynamics_block
    lo, passes = blk(record, max_iter, lsr_error, blk_in, (p, g, sg, ex))
    TAUX, TAUY, uice_fd, vice_fd = lo["TAUX"], lo["TAUY"], lo["uice_fd"], lo["vice_fd"]
    uIce, vIce = lo["uIce"], lo["vIce"]
    if record:
        rec["Y03"] = dict(TAUX=TAUX, TAUY=TAUY, uice_fd=uice_fd, vice_fd=vice_fd)
        rec["passes"] = passes
        rec["Y05"] = dict(UICE=uIce, VICE=vIce)
    fu, fv = ocean_stress(p, g, ex, st["fu"], st["fv"], uIce, vIce, lo["DWATN"], st["AREA"],
                          st["uVel"], st["vVel"])  # :361
    if record:
        rec["Y06"] = dict(fu=fu, fv=fv)
    if p.SEAICE_clipVelocities:  # :365-385
        clip = _clip_block_skipped if skip else clip_block
        cl, _ = clip(dict(uIce=uIce, vIce=vIce), None)
        uIce, vIce = cl["uIce"], cl["vIce"]
    out = dict(UICE=uIce, VICE=vIce, seaiceMassC=mC, seaiceMassU=mU, seaiceMassV=mV, TAUX=TAUX, TAUY=TAUY,
               FORCEX0=FORCEX0, FORCEY0=FORCEY0, PRESS0=PRESS0, ZMAX=ZMAX, ZMIN=ZMIN, uice_fd=uice_fd,
               vice_fd=vice_fd, e11=lo["e11"], e22=lo["e22"], e12=lo["e12"], deltaC=lo["deltaC"], ETA=lo["ETA"],
               etaZ=lo["etaZ"], ZETA=lo["ZETA"], zetaZ=lo["zetaZ"], PRESS=lo["PRESS"], DWATN=lo["DWATN"],
               FORCEX=lo["FORCEX"], FORCEY=lo["FORCEY"], uIceNm1=lo["uIceNm1"], vIceNm1=lo["vIceNm1"],
               stressDivergenceX=stressDivergenceX, stressDivergenceY=stressDivergenceY, fu=fu, fv=fv)
    return out, rec
