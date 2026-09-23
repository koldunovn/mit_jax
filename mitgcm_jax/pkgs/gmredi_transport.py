"""GM/Redi diffusive tracer fluxes (pkg/gmredi GMREDI_XTRANSPORT / _YTRANSPORT / _RTRANSPORT), plan Task 16b.

Literal translation of c66g `pkg/gmredi/gmredi_{x,y,r}transport.F` as compiled in the V4r4 flux-forced build
(`GMREDI_OPTIONS.h` of the ff `code/` tree: GM_NON_UNITY_DIAGONAL, GM_EXTRA_DIAGONAL, GM_BOLUS_ADVEC defined;
GM_VISBECK_VARIABLE_K undefined). The routines add, to the diffusive flux `df` of one level k (area-integrated,
tracer units * m^3/s), the Redi tensor fluxes built from the (exchanged) tensor Kux, Kvy, Kuz, Kvz, Kwx, Kwy.

All three functions work on every level at once: arrays are `[tile, k, j, i]` with halos (level k of the Fortran call
is index k-1 on axis 1); the Fortran k loop of GAD_CALC_RHS calls them once per level with only that level's `df`, and
every level's result depends on the tracer at k-1, k, k+1 only, so the vectorised form is the same arithmetic.
The loop ranges (iMin..iMax, jMin..jMax) are the caller's (GAD_CALC_RHS passes iMin..iMax+1 for x, jMin..jMax+1 for
y); points outside keep the input `df`.

The GM bolus branch (`GM_AdvForm .AND. GM_AdvSeparate .AND. .NOT.GM_InMomAsStress`, gmredi_xtransport.F:186) is not
ported: V4r4 has GM_AdvSeparate = F (bolus velocity added to the flow in GMREDI_RESIDUAL_FLOW); `GMTransportParams`
raises NotImplementedError otherwise.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# GMREDI.h: op5 = 0.5 (PARAMETER)
OP5 = 0.5


@dataclass(frozen=True)
class GMTransportParams:
    """Run-time switches of the GM/Redi transport routines (data.gmredi, GM_PARM01)."""
    useGMRedi: bool
    GM_ExtraDiag: bool

    @classmethod
    def from_namelists(cls, nml):
        useGMRedi = bool(nml.get("data.pkg", "packages", "useGMRedi", default=False))  # packages_readparms.F default F
        g = "gm_parm01"
        GM_AdvForm = bool(nml.get("data.gmredi", g, "GM_AdvForm", default=False))  # gmredi_readparms.F:94
        GM_AdvSeparate = bool(nml.get("data.gmredi", g, "GM_AdvSeparate", default=False))  # gmredi_readparms.F:95
        GM_InMomAsStress = bool(nml.get("data.gmredi", g, "GM_InMomAsStress", default=False))  # gmredi_readparms.F:96
        GM_background_K = float(nml.get("data.gmredi", g, "GM_background_K", default=0.0))  # gmredi_readparms.F:98
        GM_isopycK = float(nml.get("data.gmredi", g, "GM_isopycK", default=-999.0))  # gmredi_readparms.F:97
        if GM_isopycK == -999.0:  # gmredi_readparms.F:185
            GM_isopycK = GM_background_K
        GM_Visbeck_alpha = float(nml.get("data.gmredi", g, "GM_Visbeck_alpha", default=0.0))  # gmredi_readparms.F
        if GM_AdvForm:  # gmredi_readparms.F:195-203
            GM_ExtraDiag = GM_Visbeck_alpha != 0.0 or GM_isopycK != 0.0
        else:
            GM_ExtraDiag = GM_isopycK != GM_background_K
        files = {n: nml.get("data.gmredi", g, n, default=" ") for n in
                 ("GM_iso2dFile", "GM_bol2dFile", "GM_iso1dFile", "GM_bol1dFile")}
        if files["GM_iso2dFile"] != files["GM_bol2dFile"] or files["GM_iso1dFile"] != files["GM_bol1dFile"]:
            GM_ExtraDiag = True  # gmredi_readparms.F:204-207
        if useGMRedi and GM_AdvForm and GM_AdvSeparate and not GM_InMomAsStress:
            raise NotImplementedError("GM bolus advection in GMREDI_X/Y/RTRANSPORT (GM_AdvSeparate=T) is not ported "
                                      "(gmredi_xtransport.F:186)")
        return cls(useGMRedi=useGMRedi, GM_ExtraDiag=GM_ExtraDiag)


def _kidx(Nr):
    """km1 = MAX(k-1,1), kp1 = MIN(k+1,Nr) as 0-based index vectors over k=1..Nr (gmredi_xtransport.F:150-151)."""
    k = np.arange(1, Nr + 1)
    return np.maximum(k - 1, 1) - 1, np.minimum(k + 1, Nr) - 1


def gmredi_xtransport(pp, g, iMin, iMax, jMin, jMax, xA, Tracer, df, Kux, Kuz):
    """gmredi_xtransport.F: df += zonal Redi flux on every level. xA, Tracer, df, Kux, Kuz: [T, Nr, j, i]."""
    if not pp.useGMRedi:  # gmredi_xtransport.F:112
        return df
    L = g.layout
    Nr = Tracer.shape[1]
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    Im1 = L.is_(iMin - 1, iMax - 1)
    rdxC = g.recip_dxC[:, None]
    # gmredi_xtransport.F:126-146 (GM_NON_UNITY_DIAGONAL: Kux)
    t_i = Tracer[..., J, I]
    t_im1 = Tracer[..., J, Im1]
    val = df[..., J, I] - xA[..., J, I] * Kux[..., J, I] * rdxC[..., J, I] * (t_i - t_im1)
    if pp.GM_ExtraDiag:  # gmredi_xtransport.F:149
        km1, kp1 = _kidx(Nr)
        rdrC = jnp.asarray(g.recip_drC)
        rdrC_k = rdrC[:Nr][None, :, None, None]  # recip_drC(k)
        rdrC_kp1 = rdrC[kp1][None, :, None, None]  # recip_drC(kp1)
        mC = g.maskC
        # gmredi_xtransport.F:154-171: vertical gradient interpolated to U points
        dTdz = OP5 * (
            OP5 * rdrC_k * (mC[..., J, Im1] * (Tracer[:, km1][..., J, Im1] - Tracer[..., J, Im1])
                            + mC[..., J, I] * (Tracer[:, km1][..., J, I] - Tracer[..., J, I]))
            + OP5 * rdrC_kp1 * (mC[:, kp1][..., J, Im1] * (Tracer[..., J, Im1] - Tracer[:, kp1][..., J, Im1])
                                + mC[:, kp1][..., J, I] * (Tracer[..., J, I] - Tracer[:, kp1][..., J, I])))
        # gmredi_xtransport.F:177-181
        val = val - xA[..., J, I] * Kuz[..., J, I] * dTdz
    return df.at[..., J, I].set(val)


def gmredi_ytransport(pp, g, iMin, iMax, jMin, jMax, yA, Tracer, df, Kvy, Kvz):
    """gmredi_ytransport.F: df += meridional Redi flux on every level."""
    if not pp.useGMRedi:  # gmredi_ytransport.F:111
        return df
    L = g.layout
    Nr = Tracer.shape[1]
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    Jm1 = L.js(jMin - 1, jMax - 1)
    rdyC = g.recip_dyC[:, None]
    # gmredi_ytransport.F:125-145 (GM_NON_UNITY_DIAGONAL: Kvy)
    val = df[..., J, I] - yA[..., J, I] * Kvy[..., J, I] * rdyC[..., J, I] * (Tracer[..., J, I] - Tracer[..., Jm1, I])
    if pp.GM_ExtraDiag:  # gmredi_ytransport.F:148
        km1, kp1 = _kidx(Nr)
        rdrC = jnp.asarray(g.recip_drC)
        rdrC_k = rdrC[:Nr][None, :, None, None]
        rdrC_kp1 = rdrC[kp1][None, :, None, None]
        mC = g.maskC
        # gmredi_ytransport.F:153-169: vertical gradient interpolated to V points
        dTdz = OP5 * (
            OP5 * rdrC_k * (mC[..., Jm1, I] * (Tracer[:, km1][..., Jm1, I] - Tracer[..., Jm1, I])
                            + mC[..., J, I] * (Tracer[:, km1][..., J, I] - Tracer[..., J, I]))
            + OP5 * rdrC_kp1 * (mC[:, kp1][..., Jm1, I] * (Tracer[..., Jm1, I] - Tracer[:, kp1][..., Jm1, I])
                                + mC[:, kp1][..., J, I] * (Tracer[..., J, I] - Tracer[:, kp1][..., J, I])))
        # gmredi_ytransport.F:175-180
        val = val - yA[..., J, I] * Kvz[..., J, I] * dTdz
    return df.at[..., J, I].set(val)


def gmredi_rtransport(pp, g, iMin, iMax, jMin, jMax, Tracer, df, Kwx, Kwy):
    """gmredi_rtransport.F: df += vertical off-diagonal Redi flux at interface k (top of cell k), levels k >= 2.

    Level k=1 keeps the input df (gmredi_rtransport.F:97 `k.GT.1`)."""
    if not pp.useGMRedi:
        return df
    L = g.layout
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    Ip1, Im1 = L.is_(iMin + 1, iMax + 1), L.is_(iMin - 1, iMax - 1)
    Jp1, Jm1 = L.js(jMin + 1, jMax + 1), L.js(jMin - 1, jMax - 1)
    mW, mS = g.maskW, g.maskS
    rdxC = g.recip_dxC[:, None]
    rdyC = g.recip_dyC[:, None]
    T = Tracer[:, 1:]    # level k   (k = 2..Nr)
    Tu = Tracer[:, :-1]  # level k-1
    mWk, mWu = mW[:, 1:], mW[:, :-1]
    mSk, mSu = mS[:, 1:], mS[:, :-1]
    # gmredi_rtransport.F:102-115: horizontal gradients interpolated to W points
    dTdx = OP5 * (
        OP5 * (mWk[..., J, Ip1] * rdxC[..., J, Ip1] * (T[..., J, Ip1] - T[..., J, I])
               + mWk[..., J, I] * rdxC[..., J, I] * (T[..., J, I] - T[..., J, Im1]))
        + OP5 * (mWu[..., J, Ip1] * rdxC[..., J, Ip1] * (Tu[..., J, Ip1] - Tu[..., J, I])
                 + mWu[..., J, I] * rdxC[..., J, I] * (Tu[..., J, I] - Tu[..., J, Im1])))
    # gmredi_rtransport.F:117-130
    dTdy = OP5 * (
        OP5 * (mSk[..., J, I] * rdyC[..., J, I] * (T[..., J, I] - T[..., Jm1, I])
               + mSk[..., Jp1, I] * rdyC[..., Jp1, I] * (T[..., Jp1, I] - T[..., J, I]))
        + OP5 * (mSu[..., J, I] * rdyC[..., J, I] * (Tu[..., J, I] - Tu[..., Jm1, I])
                 + mSu[..., Jp1, I] * rdyC[..., Jp1, I] * (Tu[..., Jp1, I] - Tu[..., J, I])))
    # gmredi_rtransport.F:154-161
    rAm = (g.rA * g.maskInC)[:, None]
    val = df[:, 1:][..., J, I] - rAm[..., J, I] * (Kwx[:, 1:][..., J, I] * dTdx + Kwy[:, 1:][..., J, I] * dTdy)
    return df.at[:, 1:, J, I].set(val)
