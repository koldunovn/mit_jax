"""pkg/gmredi as the V4r4 flux-forced build runs it (plan Task 13): GM/Redi tensor, bolus streamfunction, exchange,
residual (Eulerian + bolus) velocity, and the Kwz contribution to the vertical diffusivity.

Literal translation of c66g `pkg/gmredi` (no flux-forced override exists for any gmredi file) under the build's
GMREDI_OPTIONS.h (`ECCOv4 Release 4/flux-forced/code/GMREDI_OPTIONS.h`):
    GMREDI_WITH_STABLE_ADJOINT, GM_EXCLUDE_CLIPPING, GM_EXCLUDE_FM07_TAP, GM_EXCLUDE_AC02_TAP, GM_EXCLUDE_SUBMESO,
    GM_NON_UNITY_DIAGONAL, GM_EXTRA_DIAGONAL, GM_BOLUS_ADVEC; GM_VISBECK_VARIABLE_K, GM_K3D, GM_BOLUS_BVP undefined;
plus CTRL_OPTIONS.h ALLOW_KAPGM_CONTROL / ALLOW_KAPREDI_CONTROL (the _OLD variants undefined), so the 3-D fields
kapGM / kapRedi (CTRL_FIELDS.h) set every diffusivity, and ALLOW_AUTODIFF (pkg autodiff compiled), which adds the
zero-initialisation loops. The branch structure was checked against the preprocessed sources of the oracle build
(`reference/build/ff_serial13_jaxdump_*/bld/gmredi_*.f`).

Call sequence in the model (all tiles at once here):
    DO_OCEANIC_PHYS  do_oceanic_phys.F(ff):1100  GMREDI_CALC_TENSOR   -> gmredi_calc_tensor
                     do_oceanic_phys.F(ff):1163  GMREDI_DO_EXCH       -> gmredi_do_exch
    THERMODYNAMICS   thermodynamics.F:267       GMREDI_RESIDUAL_FLOW -> gmredi_residual_flow
    TEMP/SALT_INTEGRATE -> CALC_3D_DIFFUSIVITY calc_3d_diffusivity.F:195 GMREDI_CALC_DIFF -> gmredi_calc_diff
The flux divergences GMREDI_X/Y/RTRANSPORT belong to the tracer kernels.

Taper scheme 'stableGmAdjTap' makes every taper factor 1: the tensor slope is rescaled to |S| <= 2e-3
(gmredi_slope_limit.F:593-612) and the bolus slope to 5*min(|S|, 1e-4)*sign(S) (gmredi_slope_psi.F:377-403).

Values the Fortran computes but no output depends on are not reproduced: the edge rows/columns of the local work
arrays SlopeX/SlopeY/SlopeSqr/taperFct (outside 1-OLx+1..sNx+OLx-1 x 1-OLy+1..sNy+OLy-1) that the stableGmAdjTap loop
(gmredi_slope_limit.F:598-612) also touches, the ldd97/fm07 transition-layer arrays (hTransLay, baseSlope,
recipLambda, locMixLayer: inputs of taper schemes V4r4 does not use), half_K (gmredi_calc_psi_b.F:143, unused with
ALLOW_KAPGM_CONTROL), and the diagnostics / time-average paths.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mitgcm_jax.params_io import params_pytree

# GMREDI.h:11-12
OP5 = 0.5  # PARAMETER( op5  = 0.5 _d 0 )
OP25 = 0.25  # PARAMETER( op25 = 0.25 _d 0 )
# gmredi_slope_limit.F:124  GM_bigSlope = 1. _d +02
GM_BIGSLOPE = 1.0e2
# gmredi_slope_limit.F:593  slopeMax= 2. _d -3   (stableGmAdjTap; hard-coded, not GM_maxSlope)
SLOPE_MAX = 2.0e-3
# gmredi_slope_psi.F:377  slopeMaxSpec=1. _d -4
SLOPE_MAX_SPEC = 1.0e-4
# gmredi_slope_psi.F:386,388  `5.*SlopeX(i,j)...` (default-real literal 5., exact in float64)
FIVE = 5.0


@params_pytree
@dataclass(frozen=True)
class GMRediParams:
    """GM_PARM01 (data.gmredi) plus the derived constants of gmredi_readparms.F, the two model switches that decide
    GMREDI_DO_EXCH, and the Fortran literal constants of the ported routines. Build with `from_namelists`; unported
    values raise NotImplementedError. Pass the object as a jit ARGUMENT: its float fields (namelist values and the
    literal constants alike) are traced pytree leaves, so XLA cannot re-associate constant products
    (`5.*S*slopeMaxSpec` -> `S*(5*slopeMaxSpec)`, params_io.params_pytree)."""

    # static switches (pytree metadata)
    GM_AdvForm: bool
    GM_AdvSeparate: bool
    GM_InMomAsStress: bool
    GM_taper_scheme: str
    GM_UseBVP: bool
    GM_useSubMeso: bool
    GM_useK3D: bool
    GM_iso2dFile: str
    GM_iso1dFile: str
    GM_bol2dFile: str
    GM_bol1dFile: str
    GM_ExtraDiag: bool
    useCubedSphereExchange: bool
    useMultiDimAdvec: bool
    # namelist / derived reals (traced leaves)
    GM_isopycK: float
    GM_background_K: float
    GM_maxSlope: float
    GM_Kmin_horiz: float
    GM_Small_Number: float
    GM_slopeSqCutoff: float
    GM_Visbeck_alpha: float
    GM_skewflx: float
    GM_advect: float
    # Fortran literal constants (traced leaves; values and citations at the module constants above)
    op5: float = OP5
    op25: float = OP25
    GM_bigSlope: float = GM_BIGSLOPE
    slopeMax: float = SLOPE_MAX
    slopeMaxSpec: float = SLOPE_MAX_SPEC
    five: float = FIVE

    @classmethod
    def from_namelists(cls, nml):
        g = lambda key, default: nml.get("data.gmredi", "GM_PARM01", key, default=default)  # noqa: E731
        GM_AdvForm = bool(g("GM_AdvForm", False))  # gmredi_readparms.F:94
        GM_AdvSeparate = bool(g("GM_AdvSeparate", False))  # gmredi_readparms.F:95
        GM_InMomAsStress = bool(g("GM_InMomAsStress", False))  # gmredi_readparms.F:96
        GM_isopycK = float(g("GM_isopycK", -999.0))  # gmredi_readparms.F:97
        GM_background_K = float(g("GM_background_K", 0.0))  # gmredi_readparms.F:98
        GM_maxSlope = float(g("GM_maxSlope", 1.0e-2))  # gmredi_readparms.F:99
        GM_Kmin_horiz = float(g("GM_Kmin_horiz", 0.0))  # gmredi_readparms.F:100
        GM_Small_Number = float(g("GM_Small_Number", 1.0e-20))  # gmredi_readparms.F:101
        GM_slopeSqCutoff = float(g("GM_slopeSqCutoff", 1.0e48))  # gmredi_readparms.F:102
        GM_taper_scheme = str(g("GM_taper_scheme", " ")).rstrip() or " "  # gmredi_readparms.F:103
        GM_iso2dFile = str(g("GM_iso2dFile", " "))  # gmredi_readparms.F:110
        GM_iso1dFile = str(g("GM_iso1dFile", " "))  # gmredi_readparms.F:111
        GM_bol2dFile = str(g("GM_bol2dFile", " "))  # gmredi_readparms.F:112
        GM_bol1dFile = str(g("GM_bol1dFile", " "))  # gmredi_readparms.F:113
        GM_Visbeck_alpha = float(g("GM_Visbeck_alpha", 0.0))  # gmredi_readparms.F:122
        GM_UseBVP = bool(g("GM_UseBVP", False))  # gmredi_readparms.F:131
        GM_useSubMeso = bool(g("GM_useSubMeso", False))  # gmredi_readparms.F:136
        GM_useK3D = bool(g("GM_useK3D", False))  # gmredi_readparms.F:143
        # gmredi_readparms.F:185  IF (GM_isopycK.EQ.-999.) GM_isopycK = GM_background_K
        if GM_isopycK == -999.0:
            GM_isopycK = GM_background_K
        # gmredi_readparms.F:195-207
        if GM_AdvForm:
            GM_skewflx, GM_advect = 0.0, 1.0
            GM_ExtraDiag = GM_Visbeck_alpha != 0.0 or GM_isopycK != 0.0
        else:
            GM_skewflx, GM_advect = 1.0, 0.0
            GM_ExtraDiag = GM_isopycK != GM_background_K
        if GM_iso2dFile != GM_bol2dFile or GM_iso1dFile != GM_bol1dFile:
            GM_ExtraDiag = True
        # eeset_parms.F:104  useCubedSphereExchange = .FALSE.  (eedata EEPARMS)
        useCubedSphereExchange = bool(nml.get("eedata", "EEPARMS", "useCubedSphereExchange", default=False))
        # gad_init_fixed.F:129-144 (useMultiDimAdvec: set_defaults.F:224 .FALSE., then OR-ed per tracer)
        multiDimAdvection = bool(nml.get("data", "PARM01", "multiDimAdvection", default=True))  # set_defaults.F:223
        tempAdvection = bool(nml.get("data", "PARM01", "tempAdvection", default=True))  # set_defaults.F:192
        saltAdvection = bool(nml.get("data", "PARM01", "saltAdvection", default=True))  # set_defaults.F:195
        tempAdvScheme = int(nml.get("data", "PARM01", "tempAdvScheme", default=2))  # set_defaults.F:221
        saltAdvScheme = int(nml.get("data", "PARM01", "saltAdvScheme", default=2))  # set_defaults.F:222
        not_md = (2, 3, 4)  # GAD.h:28,32,36 ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH
        tempMD = multiDimAdvection and tempAdvection and tempAdvScheme not in not_md
        saltMD = multiDimAdvection and saltAdvection and saltAdvScheme not in not_md
        p = cls(GM_AdvForm=GM_AdvForm, GM_AdvSeparate=GM_AdvSeparate, GM_InMomAsStress=GM_InMomAsStress,
                GM_taper_scheme=GM_taper_scheme, GM_UseBVP=GM_UseBVP, GM_useSubMeso=GM_useSubMeso,
                GM_useK3D=GM_useK3D, GM_iso2dFile=GM_iso2dFile, GM_iso1dFile=GM_iso1dFile,
                GM_bol2dFile=GM_bol2dFile, GM_bol1dFile=GM_bol1dFile, GM_ExtraDiag=GM_ExtraDiag,
                useCubedSphereExchange=useCubedSphereExchange, useMultiDimAdvec=tempMD or saltMD,
                GM_isopycK=GM_isopycK, GM_background_K=GM_background_K, GM_maxSlope=GM_maxSlope,
                GM_Kmin_horiz=GM_Kmin_horiz, GM_Small_Number=GM_Small_Number, GM_slopeSqCutoff=GM_slopeSqCutoff,
                GM_Visbeck_alpha=GM_Visbeck_alpha, GM_skewflx=GM_skewflx, GM_advect=GM_advect)
        p.check()
        return p

    def check(self):
        """Hard error for every value whose branch is not ported (the V4r4 values are the only ported ones)."""
        bad = []
        if self.GM_taper_scheme != "stableGmAdjTap":
            bad.append(f"GM_taper_scheme={self.GM_taper_scheme!r} (only 'stableGmAdjTap' is ported)")
        if not self.GM_AdvForm:
            bad.append("GM_AdvForm=F (skew-flux form)")
        if self.GM_AdvSeparate:
            bad.append("GM_AdvSeparate=T")
        if self.GM_InMomAsStress:
            bad.append("GM_InMomAsStress=T")
        if self.GM_Visbeck_alpha != 0.0:
            bad.append("GM_Visbeck_alpha != 0 (GM_VISBECK_VARIABLE_K is undefined in the build)")
        if self.GM_UseBVP or self.GM_useSubMeso or self.GM_useK3D:
            bad.append("GM_UseBVP / GM_useSubMeso / GM_useK3D = T")
        if not self.GM_ExtraDiag:
            bad.append("GM_ExtraDiag=F (Kuz/Kvz branch gmredi_calc_tensor.F:759,977 not taken)")
        if not (self.useCubedSphereExchange and self.useMultiDimAdvec):
            bad.append("GMREDI_DO_EXCH without its exchange (gmredi_do_exch.F:49-52 condition false)")
        if bad:
            raise NotImplementedError("pkg/gmredi: unported configuration: " + "; ".join(bad))


def _kup(a):
    """a(k+1) with kp1 = MIN(Nr,k+1) (level axis 1)."""
    return jnp.concatenate([a[:, 1:], a[:, -1:]], axis=1)


def _kdown(a):
    """a(km1) with km1 = MAX(k-1,1) (level axis 1)."""
    return jnp.concatenate([a[:, :1], a[:, :-1]], axis=1)


def _maskp1(Nr):
    """maskp1 = 1, 0 at k=Nr (gmredi_calc_tensor.F:669-670, gmredi_residual_flow.F:62-63), shaped [1,Nr,1,1]."""
    m = np.ones(Nr)
    m[Nr - 1] = 0.0
    return jnp.asarray(m)[None, :, None, None]


def gmredi_slope_limit(p, dSigmaDx, dSigmaDy, dSigmaDr):
    """GMREDI_SLOPE_LIMIT with GM_taper_scheme='stableGmAdjTap' (gmredi_slope_limit.F:12). Pointwise: the arguments
    hold the points of the Fortran loops j=1-OLy+1..sNy+OLy-1, i=1-OLx+1..sNx+OLx-1 (any shape).
    Returns SlopeX, SlopeY, SlopeSqr, taperFct and the updated dSigmaDr."""
    eps = p.GM_Small_Number
    # gmredi_slope_limit.F:452-459  avoid reverse slope where Sigma_Z >= -GM_Small_Number (but not where it is 0)
    dSigmaDr = jnp.where((dSigmaDr != 0.0) & (dSigmaDr >= -eps), -eps, dSigmaDr)
    # gmredi_slope_limit.F:467-488
    zero = dSigmaDr == 0.0
    dRdSigmaLtd = 1.0 / jnp.where(zero, 1.0, dSigmaDr)  # :481 (guarded: where dSigmaDr = 0 the value is unused)
    big_x = jnp.where(dSigmaDx != 0.0, jnp.where(dSigmaDx > 0.0, p.GM_bigSlope, -p.GM_bigSlope), 0.0)  # :470-474 SIGN
    big_y = jnp.where(dSigmaDy != 0.0, jnp.where(dSigmaDy > 0.0, p.GM_bigSlope, -p.GM_bigSlope), 0.0)  # :475-479
    SlopeX = jnp.where(zero, big_x, -dSigmaDx * dRdSigmaLtd)  # :482
    SlopeY = jnp.where(zero, big_y, -dSigmaDy * dRdSigmaLtd)  # :483
    # gmredi_slope_limit.F:495-505  (SlopeSqr/taperFct are overwritten below; kept for the literal order)
    SlopeSqr = SlopeX * SlopeX + SlopeY * SlopeY
    cut = SlopeSqr > p.GM_slopeSqCutoff
    SlopeSqr = jnp.where(cut, p.GM_slopeSqCutoff, SlopeSqr)
    taperFct = jnp.where(cut, 0.0, 1.0)
    # gmredi_slope_limit.F:582-612  stableGmAdjTap: rescale |S| to slopeMax
    slopeSqTmp = SlopeX * SlopeX + SlopeY * SlopeY  # :600-601
    lim = slopeSqTmp > p.slopeMax * p.slopeMax  # :603  slopeMax**2 (gfortran: x*x)
    # :604 slopeTmp=sqrt(slopeSqTmp) (guarded). Note: with XLA's algsimp pass enabled, A/sqrt(B) below becomes
    # A*rsqrt(B) where the sqrt has a single user (the U/V loops): 1-ulp differences in Kuz/Kvz at 66201/66214
    # points (SMOKE it 1, measured). The gates run with --xla_disable_hlo_passes=algsimp (conftest.py).
    slopeTmp = jnp.sqrt(jnp.where(lim, slopeSqTmp, 1.0))
    SlopeX = jnp.where(lim, SlopeX * p.slopeMax / slopeTmp, SlopeX)  # :605
    SlopeY = jnp.where(lim, SlopeY * p.slopeMax / slopeTmp, SlopeY)  # :606
    SlopeSqr = SlopeX * SlopeX + SlopeY * SlopeY  # :608-609
    taperFct = jnp.ones_like(taperFct)  # :610
    return SlopeX, SlopeY, SlopeSqr, taperFct, dSigmaDr


def gmredi_slope_psi(p, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS):
    """GMREDI_SLOPE_PSI with GM_taper_scheme='stableGmAdjTap' (gmredi_slope_psi.F). Pointwise on the points of its
    X loops (j=1-OLy..sNy+OLy, i=1-OLx+1..sNx+OLx) and Y loops (j=1-OLy+1..sNy+OLy, i=1-OLx..sNx+OLx).
    Returns taperX, taperY, SlopeX, SlopeY."""
    eps = p.GM_Small_Number

    def one(S, dSigmaDr):
        dSigmaDr = jnp.where(dSigmaDr >= -eps, -eps, dSigmaDr)  # :195-200 / :229-234
        S = -S / dSigmaDr  # :206 / :240   (dSigmaDr <= -GM_Small_Number < 0: no guard needed)
        # (:213-222 / :247-256 slope cut-off skipped: GM_taper_scheme = 'stableGmAdjTap')
        # :366-403 stableGmAdjTap
        slopeTmpSpec = jnp.abs(S)  # :384 / :395
        gt = slopeTmpSpec > p.slopeMaxSpec  # :385 / :396
        S = jnp.where(gt, p.five * S * p.slopeMaxSpec / jnp.where(gt, slopeTmpSpec, 1.0),  # :386 / :397 (guarded)
                      p.five * S)  # :388 / :399
        return jnp.ones_like(S), S  # taper = 1. (:390 / :401)

    taperX, SlopeX = one(SlopeX, dSigmaDrW)
    taperY, SlopeY = one(SlopeY, dSigmaDrS)
    return taperX, taperY, SlopeX, SlopeY


def gmredi_calc_psi_b(p, g, sigmaX, sigmaY, sigmaR, kapGM):
    """GMREDI_CALC_PSI_B (gmredi_calc_psi_b.F), GM_AdvForm=T, ALLOW_KAPGM_CONTROL. Returns GM_PsiX, GM_PsiY with
    level k=1 and the points outside the loops = 0 (zeroed in gmredi_calc_tensor.F:219-228)."""
    L = g.layout
    maskW, maskS = g.maskW, g.maskS
    # levels k=2..Nr (gmredi_calc_psi_b.F:94-95): python [1:], km1 = k-1 -> [:-1]
    # X points: j=1-OLy..sNy+OLy, i=1-OLx+1..sNx+OLx (:109-110, :145-146)
    JX = L.js(1 - L.OLy, L.sNy + L.OLy)
    IX, IXm1 = L.is_(1 - L.OLx + 1, L.sNx + L.OLx), L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    # Y points: j=1-OLy+1..sNy+OLy, i=1-OLx..sNx+OLx (:117-118, :168-169)
    JY, JYm1 = L.js(1 - L.OLy + 1, L.sNy + L.OLy), L.js(1 - L.OLy, L.sNy + L.OLy - 1)
    IY = L.is_(1 - L.OLx, L.sNx + L.OLx)
    k, km1 = slice(1, None), slice(None, -1)

    mW = maskW[:, k, JX, IX]
    SlopeX = p.op5 * (sigmaX[:, km1, JX, IX] + sigmaX[:, k, JX, IX]) * mW  # :111-112
    dSigmaDrW = p.op5 * (sigmaR[:, k, JX, IXm1] + sigmaR[:, k, JX, IX]) * mW  # :113-114
    mS = maskS[:, k, JY, IY]
    SlopeY = p.op5 * (sigmaY[:, km1, JY, IY] + sigmaY[:, k, JY, IY]) * mS  # :119-120
    dSigmaDrS = p.op5 * (sigmaR[:, k, JYm1, IY] + sigmaR[:, k, JY, IY]) * mS  # :121-122

    taperX, taperY, SlopeX, SlopeY = gmredi_slope_psi(p, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS)  # :127-132

    # :147-162  GM_PsiX = SlopeX*taperX*( op25*( kapGM(i-1,km1)+kapGM(i,km1) + kapGM(i-1,k)+kapGM(i,k) ) )*maskW
    kX = p.op25 * (kapGM[:, km1, JX, IXm1] + kapGM[:, km1, JX, IX] + kapGM[:, k, JX, IXm1] + kapGM[:, k, JX, IX])
    PsiX = SlopeX * taperX * kX * mW
    # :170-185
    kY = p.op25 * (kapGM[:, km1, JYm1, IY] + kapGM[:, km1, JY, IY] + kapGM[:, k, JYm1, IY] + kapGM[:, k, JY, IY])
    PsiY = SlopeY * taperY * kY * mS

    GM_PsiX = jnp.zeros(sigmaX.shape).at[:, k, JX, IX].set(PsiX)
    GM_PsiY = jnp.zeros(sigmaY.shape).at[:, k, JY, IY].set(PsiY)
    return GM_PsiX, GM_PsiY


def gmredi_calc_tensor(p, g, sigmaX, sigmaY, sigmaR, kapGM, kapRedi):
    """GMREDI_CALC_TENSOR (gmredi_calc_tensor.F:18) for all tiles, iMin..jMax unused by the V4r4 branches.

    Inputs (Fortran names, [tile, k, j, i] with halos): sigmaX (W points), sigmaY (S points), sigmaR (C points,
    interface above level k) from GRAD_SIGMA; kapGM, kapRedi (CTRL_FIELDS.h 3-D fields); grid masks g.maskC/W/S.
    Returns dict Kwx, Kwy, Kwz, Kux, Kvy, Kuz, Kvz, GM_PsiX, GM_PsiY (the GMREDI.h arrays after the call; every point
    the Fortran does not write is 0 from the ALLOW_AUTODIFF initialisation, gmredi_calc_tensor.F:219-247)."""
    L = g.layout
    maskC, maskW, maskS = g.maskC, g.maskW, g.maskS
    zeros = jnp.zeros(maskC.shape)
    # interior loops j=1-OLy+1..sNy+OLy-1, i=1-OLx+1..sNx+OLx-1 (every tensor loop of this routine)
    J, I = L.js(1 - L.OLy + 1, L.sNy + L.OLy - 1), L.is_(1 - L.OLx + 1, L.sNx + L.OLx - 1)
    Jp1, Ip1 = L.js(1 - L.OLy + 2, L.sNy + L.OLy), L.is_(1 - L.OLx + 2, L.sNx + L.OLx)
    Jm1, Im1 = L.js(1 - L.OLy, L.sNy + L.OLy - 2), L.is_(1 - L.OLx, L.sNx + L.OLx - 2)

    # ---- 1rst loop on k (k=Nr..2, independent levels): tensor coefficients at W points (:295-495)
    k, km1 = slice(1, None), slice(None, -1)
    mC = maskC[:, k, J, I]
    dSigmaDx = p.op25 * (sigmaX[:, km1, J, Ip1] + sigmaX[:, km1, J, I]
                       + sigmaX[:, k, J, Ip1] + sigmaX[:, k, J, I]) * mC  # :314-316
    dSigmaDy = p.op25 * (sigmaY[:, km1, Jp1, I] + sigmaY[:, km1, J, I]
                       + sigmaY[:, k, Jp1, I] + sigmaY[:, k, J, I]) * mC  # :317-319
    dSigmaDr = sigmaR[:, k, J, I]  # :405-409
    SlopeX, SlopeY, SlopeSqr, taperFct, _ = gmredi_slope_limit(p, dSigmaDx, dSigmaDy, dSigmaDr)  # :422-430
    SlopeX = SlopeX * mC  # :435
    SlopeY = SlopeY * mC  # :436
    SlopeSqr = SlopeSqr * mC  # :437
    Kwx = zeros.at[:, k, J, I].set(SlopeX * taperFct)  # :452
    Kwy = zeros.at[:, k, J, I].set(SlopeY * taperFct)  # :453
    Kwz = zeros.at[:, k, J, I].set(SlopeSqr * taperFct)  # :454

    # ---- express the tensor in terms of diffusivity, k=1..Nr, km1 = MAX(k-1,1) (:519-582)
    kR, kRm1 = kapRedi[:, :, J, I], _kdown(kapRedi)[:, :, J, I]
    kG, kGm1 = kapGM[:, :, J, I], _kdown(kapGM)[:, :, J, I]
    Kgm_tmp = p.op5 * (kR + kRm1) + p.GM_skewflx * p.op5 * (kG + kGm1)  # :540, :549
    Kwx = Kwx.at[:, :, J, I].set(Kgm_tmp * Kwx[:, :, J, I])  # :561
    Kwy = Kwy.at[:, :, J, I].set(Kgm_tmp * Kwy[:, :, J, I])  # :562
    Kwz = Kwz.at[:, :, J, I].set((p.op5 * (kR + kRm1)) * Kwz[:, :, J, I])  # :567-568, :579

    # ---- stream functions of the advective form (:595-621)
    GM_PsiX, GM_PsiY = gmredi_calc_psi_b(p, g, sigmaX, sigmaY, sigmaR, kapGM)

    maskp1 = _maskp1(L.Nr)
    # ---- 2nd k loop (k=Nr..1, independent levels): tensor coefficients at U points (:667-847)
    mW = maskW[:, :, J, I]
    sR, sRp1 = sigmaR, _kup(sigmaR)
    dSigmaDx = sigmaX[:, :, J, I] * mW  # :678-679
    dSigmaDy = p.op25 * (sigmaY[:, :, Jp1, Im1] + sigmaY[:, :, Jp1, I]
                       + sigmaY[:, :, J, Im1] + sigmaY[:, :, J, I]) * mW  # :680-682
    dSigmaDr = p.op25 * (sR[:, :, J, Im1] + sR[:, :, J, I]
                       + (sRp1[:, :, J, Im1] + sRp1[:, :, J, I]) * maskp1) * mW  # :683-685
    SlopeX, SlopeY, SlopeSqr, taperFct, _ = gmredi_slope_limit(p, dSigmaDx, dSigmaDy, dSigmaDr)  # :701-709
    kRW = p.op5 * (kapRedi[:, :, J, I] + kapRedi[:, :, J, Im1])  # :725
    Kux = zeros.at[:, :, J, I].set(jnp.maximum(kRW * taperFct, p.GM_Kmin_horiz))  # :720-737, :747
    kGW = p.GM_skewflx * p.op5 * (kapGM[:, :, J, I] + kapGM[:, :, J, Im1])  # :777
    Kuz = zeros.at[:, :, J, I].set((kRW - kGW) * SlopeX * taperFct)  # :759-792 (GM_ExtraDiag=T)

    # ---- 3rd k loop (k=Nr..1, independent levels): tensor coefficients at V points (:885-1065)
    mS = maskS[:, :, J, I]
    dSigmaDx = p.op25 * (sigmaX[:, :, J, I] + sigmaX[:, :, J, Ip1]
                       + sigmaX[:, :, Jm1, I] + sigmaX[:, :, Jm1, Ip1]) * mS  # :895-897
    dSigmaDy = sigmaY[:, :, J, I] * mS  # :898-899
    dSigmaDr = p.op25 * (sR[:, :, Jm1, I] + sR[:, :, J, I]
                       + (sRp1[:, :, Jm1, I] + sRp1[:, :, J, I]) * maskp1) * mS  # :900-902
    SlopeX, SlopeY, SlopeSqr, taperFct, _ = gmredi_slope_limit(p, dSigmaDx, dSigmaDy, dSigmaDr)  # :916-924
    kRS = p.op5 * (kapRedi[:, :, J, I] + kapRedi[:, :, Jm1, I])  # :943
    Kvy = zeros.at[:, :, J, I].set(jnp.maximum(kRS * taperFct, p.GM_Kmin_horiz))  # :938-955, :965
    kGS = p.GM_skewflx * p.op5 * (kapGM[:, :, J, I] + kapGM[:, :, Jm1, I])  # :995
    Kvz = zeros.at[:, :, J, I].set((kRS - kGS) * SlopeY * taperFct)  # :977-1010 (GM_ExtraDiag=T)

    return dict(Kwx=Kwx, Kwy=Kwy, Kwz=Kwz, Kux=Kux, Kvy=Kvy, Kuz=Kuz, Kvz=Kvz, GM_PsiX=GM_PsiX, GM_PsiY=GM_PsiY)


def gmredi_do_exch(p, ex, GM_PsiX, GM_PsiY):
    """GMREDI_DO_EXCH (gmredi_do_exch.F:49-58): EXCH_UV_XYZ_RL(GM_PsiX, GM_PsiY, .TRUE.). The condition
    (useCubedSphereExchange .AND. GM_AdvForm .AND. .NOT.GM_AdvSeparate .AND. useMultiDimAdvec) is checked at setup
    (GMRediParams.check). Kwx..Kvz are not exchanged."""
    return ex.exch_uv_xy(GM_PsiX, GM_PsiY, True)


def gmredi_residual_flow(p, g, uFld, vFld, wFld, GM_PsiX, GM_PsiY, recip_hFacW, recip_hFacS):
    """GMREDI_RESIDUAL_FLOW (gmredi_residual_flow.F:57-92): add the bolus velocity to the Eulerian velocity.

    uFld, vFld, wFld: uVel, vVel, wVel as THERMODYNAMICS copies them (thermodynamics.F:255-263); GM_PsiX/Y after
    GMREDI_DO_EXCH; recip_hFacW/S: GRID.h fields at that time (UPDATE_R_STAR, update_r_star.F:76-79; the macro
    _recip_hFacW is recip_hFacW in this build). Returns the updated uFld, vFld, wFld."""
    L = g.layout
    uFld, vFld, wFld = jnp.asarray(uFld), jnp.asarray(vFld), jnp.asarray(wFld)
    maskp1 = _maskp1(L.Nr)  # :61-63
    rdrF = jnp.asarray(g.recip_drF)[None, :, None, None]
    # :65-72  (full array j=1-OLy..sNy+OLy, i=1-OLx..sNx+OLx)
    delPsi = _kup(GM_PsiX) * maskp1 - GM_PsiX
    uFld = uFld + delPsi * rdrF * recip_hFacW
    # :73-80
    delPsi = _kup(GM_PsiY) * maskp1 - GM_PsiY
    vFld = vFld + delPsi * rdrF * recip_hFacS
    # :81-90  j=1-OLy..sNy+OLy-1, i=1-OLx..sNx+OLx-1
    J, I = L.js(1 - L.OLy, L.sNy + L.OLy - 1), L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    Jp1, Ip1 = L.js(1 - L.OLy + 1, L.sNy + L.OLy), L.is_(1 - L.OLx + 1, L.sNx + L.OLx)
    dyG, dxG = g.dyG[:, None], g.dxG[:, None]
    delPsi = (dyG[..., J, Ip1] * GM_PsiX[..., J, Ip1]
              - dyG[..., J, I] * GM_PsiX[..., J, I]
              + dxG[..., Jp1, I] * GM_PsiY[..., Jp1, I]
              - dxG[..., J, I] * GM_PsiY[..., J, I]) * g.maskC[..., J, I]
    wFld = wFld.at[..., J, I].set(wFld[..., J, I] + delPsi * g.recip_rA[:, None, J, I])
    return uFld, vFld, wFld


def gmredi_calc_diff(g, KappaRx, Kwz, iMin, iMax, jMin, jMax):
    """GMREDI_CALC_DIFF with kArg=0, kSize=Nr (calc_3d_diffusivity.F:195-198; gmredi_calc_diff.F:52-70): add the
    GM/Redi vertical diffusivity Kwz*maskInC to KappaRx over i=iMin..iMax, j=jMin..jMax (both tracerIdentity branches
    are identical without ALLOW_LONGSTEP). TEMP/SALT_INTEGRATE use iMin=0, iMax=sNx+1 (temp_integrate.F:165-168)."""
    L = g.layout
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    KappaRx = jnp.asarray(KappaRx)
    return KappaRx.at[..., J, I].set(KappaRx[..., J, I] + Kwz[..., J, I] * g.maskInC[:, None, J, I])  # :58-59
