"""pkg/mom_vecinv: MOM_VECINV, the vector-invariant momentum tendency of DYNAMICS (plan Tasks 14a/14b).

Literal port of c66g `MITgcm_c66g/pkg/mom_vecinv/mom_vecinv.F` and the routines it calls in the V4r4 flux-forced
build. The flux-forced overrides of mom_vecinv.F and mom_vi_hdissip.F (flux-forced/code/) only add DIAGNOSTICS_IS_ON
checks and DIAGNOSTICS_FILL calls (UBotDrag, UShIDrag, Um_dKEdx, Um_Diss2/4, ...): verified by `diff` against c66g
(2026-09-23), so the c66g arithmetic is what runs. MOM_CALC_VISC is the V4r4 override (mom_common.py).

Branches (data / eedata of the run, defaults cited in MomVecinvParams.from_namelists):
    momViscosity, momAdvection, useCoriolis = T; vectorInvariantMomentum = T
    useVariableVisc = useHarmonicVisc = useBiharmonicVisc = T, useStrainTensionVisc = F -> MOM_VI_HDISSIP
    implicitViscosity = T -> the explicit vertical viscous flux block (mom_vecinv.F:447-472, :520-543) is skipped:
        fVerUkp/fVerVkp are never written and fVerUkm/fVerVkm never read, so there is no k-to-k carry and all
        levels are computed at once
    no_slip_sides = T (sideDragFactor = 2) -> MOM_U/V_SIDEDRAG with h0FacZ (NONLIN_FRSURF, nonlinFreeSurf = 4)
    selectImplicitDrag = 0 with no_slip_bottom = T, bottomDragQuadratic = 1e-3 -> bottomDragTerms = T:
        MOM_U/V_BOTTOMDRAG add into guDissip (dissipation, kept out of Adams-Bashforth), not into gU
    useAbsVorticity = F, useCDscheme = F -> MOM_VI_CORIOLIS (useJamartWetPoints = T) + MOM_VI_U/V_CORIOLIS with
        relative vorticity, selectVortScheme = 1 (set_parms.F:147-155), upwind/highOrder vorticity = F
    momImplVertAdv = F -> MOM_VI_U/V_VERTSHEAR (upwindShear = F); use3dCoriolis = F, useNHMTerms = F
    useCubedSphereExchange = T (eedata) -> FILL_CS_CORNER_TR_RL on hDiv in MOM_VI_DEL2UV and the corner
        vorticity formulas of MOM_CALC_RELVORT3; useShelfIce = F (no shelfice package)
    not ported (hard error at set-up): every other value of these switches.

Inputs of MOM_VECINV at level k (Fortran names; all [tile, k, j, i] with halos, the values DYNAMICS sees):
    uVel, vVel, wVel      DYNVARS.h at the start of DYNAMICS (dump stage S01_update_rstar_F)
    hFacW, hFacS, hFacC   r*-updated (S04_oceanic_phys / S01); recip_hFacC likewise
    recip_hFacW/S         r*-updated (update_r_star.F:76-79: 1/hFacW where maskW != 0, previous value elsewhere)
    kappaRU, kappaRV      CALC_VISCOSITY output, [T, Nr+1, j, i] (read by the bottom drag only, at level k+1)
    geometry (Grid)       static fields incl. h0FacW/S, masks, viscA4Dfld/Zfld, viscAhDfld/Zfld, fCoriG
Outputs (the arrays DYNAMICS holds after the call, stage D00b_mom_vecinv):
    gU, gV                on i=iMin..iMax, j=jMin..jMax (= 0..sNx+1, 0..sNy+1, dynamics.F:190-191); 0 elsewhere
                          (dynamics.F zeroes gU/gV with ALLOW_AUTODIFF before the k loop)
    guDissip, gvDissip    harmonic + biharmonic dissipation (masked, 1-OLx..sNx+OLx-1), + side and bottom drag on
                          iMin..iMax x jMin..jMax; 0 elsewhere (mom_vecinv.F:198-199)
"""

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp

from mitgcm_jax.parallel.tiles import tile_index
from mitgcm_jax.pkgs import mom_common as mc
from mitgcm_jax.pkgs.mom_common import (BottomDragParams, MomViscParams, _sl, g2, vcol, recip_deepFacC,
                                        recip_deepFac2C, deepFac2F, cosFacU, cosFacV)

rkSign = -1.0  # ini_vertical_grid.F:56  rkSign = -1. _d 0
EPSIL = 1.0e-9  # mom_vi_coriolis.F:37, mom_vi_u_coriolis.F:55  epsil = 1. _d -9 (_RS = Real*8)
MOM_VI_ORIGINAL_VISCA4 = False  # MOM_VECINV_OPTIONS.h  #undef MOM_VI_ORIGINAL_VISCA4


@dataclass(frozen=True)
class MomVecinvParams:
    """Parameters of MOM_VECINV. A pytree: `visc` and `drag` (whose float fields are traced leaves) are children,
    the switches are static; pass it to jit as an argument (params_io.params_pytree)."""
    visc: MomViscParams
    drag: BottomDragParams
    no_slip_sides: bool
    useJamartWetPoints: bool
    selectVortScheme: int
    useCubedSphereExchange: bool
    bottomDragTerms: bool  # mom_vecinv.F:232-239
    iMin: int = 0  # dynamics.F:190  PARAMETER( iMin = 0 , iMax = sNx+1 )
    jMin: int = 0  # dynamics.F:191

    @classmethod
    def from_namelists(cls, nml):
        g = lambda key, default: nml.get("data", "parm01", key, default=default)  # noqa: E731
        momStepping = g("momStepping", True)  # set_defaults.F:189
        checks = {  # name: (value, required value, citation of the default)
            "vectorInvariantMomentum": (g("vectorInvariantMomentum", False), True, "set_defaults.F:190"),
            "momStepping": (momStepping, True, "set_defaults.F:189"),
            "momViscosity": (g("momViscosity", True), True, "set_defaults.F:184"),
            "momAdvection": (g("momAdvection", True), True, "set_defaults.F:185"),
            "useCoriolis": (g("useCoriolis", True), True, "set_defaults.F:187"),
            "implicitViscosity": (g("implicitViscosity", False), True, "set_defaults.F:205"),
            "selectImplicitDrag": (g("selectImplicitDrag", 0), 0, "set_defaults.F:206"),
            "momImplVertAdv": (g("momImplVertAdv", False), False, "set_defaults.F:207"),
            "useCDscheme": (g("useCDscheme", False), False, "set_defaults.F:225"),
            "useJamartMomAdv": (g("useJamartMomAdv", False), False, "set_defaults.F:228"),
            "upwindVorticity": (g("upwindVorticity", False), False, "set_defaults.F:230"),
            "highOrderVorticity": (g("highOrderVorticity", False), False, "set_defaults.F:231"),
            "useAbsVorticity": (g("useAbsVorticity", False), False, "set_defaults.F:232"),
            "upwindShear": (g("upwindShear", False), False, "set_defaults.F:233"),
            "selectKEscheme": (g("selectKEscheme", 0), 0, "set_defaults.F:234"),
            "useNHMTerms": (g("useNHMTerms", False), False, "set_defaults.F:199"),
            "useStrainTensionVisc": (g("useStrainTensionVisc", False), False, "set_defaults.F:203"),
            "nonHydrostatic": (g("nonHydrostatic", False), False, "set_defaults.F:210"),
            "quasiHydrostatic": (g("quasiHydrostatic", False), False, "set_defaults.F:211"),
            "no_slip_sides": (g("no_slip_sides", True), True, "set_defaults.F:132"),
            "useShelfIce": (nml.get("data.pkg", "packages", "useShelfIce", default=False), False, "packages"),
        }
        for name, (val, want, cite) in checks.items():
            if val != want:
                raise NotImplementedError(f"MOM_VECINV: {name}={val!r} (default {cite}) is not ported")
        # set_parms.F:147-155: selectVortScheme unset (set_defaults.F:229 UNSET_I) -> 1 for the vector-invariant
        # form when upwindVorticity = highOrderVorticity = F (both checked above)
        sel = int(g("selectVortScheme", 1))
        if sel != 1:
            raise NotImplementedError(f"selectVortScheme={sel}")
        # buoyancyRelation (set_defaults.F:175 'OCEANIC' -> usingZCoords, ini_parms.F:419-421)
        if g("buoyancyRelation", "OCEANIC").strip().upper() != "OCEANIC":
            raise NotImplementedError("only usingZCoords (buoyancyRelation='OCEANIC') is ported")
        # NONLIN_FRSURF h0FacZ branch of mom_vecinv.F:312 needs nonlinFreeSurf > 0 (set_defaults.F:252 default 0)
        if not g("nonlinFreeSurf", 0) > 0:
            raise NotImplementedError("nonlinFreeSurf <= 0 is not ported (h0FacZ, mom_vecinv.F:312)")
        visc = MomViscParams.from_namelists(nml)
        drag = BottomDragParams.from_namelists(nml)
        # mom_vecinv.F:232-239 (selectImplicitDrag = 0 checked above)
        bottomDragTerms = (drag.no_slip_bottom or drag.selectBotDragQuadr >= 0 or drag.bottomDragLinear != 0.0)
        return cls(visc=visc, drag=drag, no_slip_sides=True,
                   useJamartWetPoints=bool(g("useJamartWetPoints", False)),  # set_defaults.F:227
                   selectVortScheme=sel,
                   useCubedSphereExchange=bool(nml.get("eedata", "eeparms", "useCubedSphereExchange",
                                                       default=False)),  # eeset_parms.F:104
                   bottomDragTerms=bool(bottomDragTerms))


jax.tree_util.register_dataclass(
    MomVecinvParams, data_fields=["visc", "drag"],
    meta_fields=[f.name for f in fields(MomVecinvParams) if f.name not in ("visc", "drag")])


# --------------------------------------------------------------------------------------------------------------
# routines of pkg/mom_vecinv


def mom_vi_del2uv(p, g, hDiv, vort3, hFacZ, recip_hFacW, recip_hFacS):
    """MOM_VI_DEL2UV (mom_vi_del2uv.F:79-127). Returns del2u, del2v (on 2-OLx..sNx+OLx-1, 2-OLy..sNy+OLy-1, 0
    elsewhere as initialised in mom_vecinv.F:194-195) and hDiv with its corner halos filled: FILL_CS_CORNER_TR_RL
    modifies the caller's hDiv in place (first fill4dir=1, then fill4dir=2), and MOM_VECINV passes that hDiv on
    to MOM_VI_HDISSIP."""
    L = g.layout
    R = (2 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    hDiv = mc.fill_cs_corner_tr_rl(hDiv, 1, False, L, p.useCubedSphereExchange, tile_index(g))  # :82-86
    zv = hFacZ * vort3
    val = ((s(hDiv) - s(hDiv, 0, -1)) * s(g2(g, "recip_dxC"))
           - s(recip_hFacW) * (s(zv, 1, 0) - s(zv)) * s(g2(g, "recip_dyG"))) \
        * s(g.f["maskW"]) * recip_deepFacC  # :92-97
    del2u = jnp.zeros_like(hDiv).at[_sl(L, *R)].set(val)
    hDiv = mc.fill_cs_corner_tr_rl(hDiv, 2, False, L, p.useCubedSphereExchange, tile_index(g))  # :107-111
    val = ((s(hDiv) - s(hDiv, -1, 0)) * s(g2(g, "recip_dyC"))
           + s(recip_hFacS) * (s(zv, 0, 1) - s(zv)) * s(g2(g, "recip_dxG"))) \
        * s(g.f["maskS"]) * recip_deepFacC  # :117-122
    del2v = jnp.zeros_like(hDiv).at[_sl(L, *R)].set(val)
    return del2u, del2v, hDiv


def mom_vi_hdissip(p, g, hDiv, vort3, dStar, zStar, hFacZ, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                   recip_hFacW, recip_hFacS):
    """MOM_VI_HDISSIP (c66g mom_vi_hdissip.F; the V4r4 override only adds diagnostics), useVariableViscosity = T,
    harmonic and biharmonic, MOM_VI_ORIGINAL_VISCA4 undefined. uDissip/vDissip start as the caller's zeros
    (mom_vecinv.F:198-199)."""
    vp = p.visc
    if not (vp.useHarmonicVisc and vp.useBiharmonicVisc and vp.useVariableVisc) or MOM_VI_ORIGINAL_VISCA4:
        raise NotImplementedError("MOM_VI_HDISSIP: only variable harmonic + biharmonic viscosity is ported")
    L = g.layout
    R = (2 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    rdxC, rdyG, rdxG, rdyC = (g2(g, n) for n in ("recip_dxC", "recip_dyG", "recip_dxG", "recip_dyC"))
    # harmonic, useVariableViscosity (:52-79)
    Dij = s(hDiv) * s(viscAh_D)
    Dim = s(hDiv, -1, 0) * s(viscAh_D, -1, 0)
    Dmj = s(hDiv, 0, -1) * s(viscAh_D, 0, -1)
    zv = hFacZ * vort3
    Zij = s(zv) * s(viscAh_Z)
    Zip = s(zv, 1, 0) * s(viscAh_Z, 1, 0)
    Zpj = s(zv, 0, 1) * s(viscAh_Z, 0, 1)
    uD2 = cosFacU * (Dij - Dmj) * s(rdxC) - s(recip_hFacW) * (Zip - Zij) * s(rdyG)  # :62-64
    vD2 = s(recip_hFacS) * (Zpj - Zij) * s(rdxG) * cosFacV + (Dij - Dim) * s(rdyC)  # :68-71
    uDissip = uD2  # :76
    vDissip = vD2  # :77
    # biharmonic, useVariableViscosity, not MOM_VI_ORIGINAL_VISCA4 (:128-190)
    Dim = s(dStar, -1, 0)
    Dij = s(dStar)
    Dmj = s(dStar, 0, -1)
    zs = hFacZ * zStar
    Zip1 = s(zs, 1, 0)
    Zij1 = s(zs)
    Zpj1 = s(zs, 0, 1)
    Dij = Dij * s(viscA4_D)
    Dim = Dim * s(viscA4_D, -1, 0)
    Dmj = Dmj * s(viscA4_D, 0, -1)
    Zij = Zij1 * s(viscA4_Z)
    Zip = Zip1 * s(viscA4_Z, 1, 0)
    Zpj = Zpj1 * s(viscA4_Z, 0, 1)
    uD4 = cosFacU * (Dij - Dmj) * s(rdxC) - s(recip_hFacW) * (Zip - Zij) * s(rdyG)  # :171-173
    vD4 = s(recip_hFacS) * (Zpj - Zij) * s(rdxG) * cosFacV + (Dij - Dim) * s(rdyC)  # :177-180
    uDissip = uDissip - uD4  # :186
    vDissip = vDissip - vD4  # :187
    uD = jnp.zeros_like(hDiv).at[_sl(L, *R)].set(uDissip)
    vD = jnp.zeros_like(hDiv).at[_sl(L, *R)].set(vDissip)
    # :255-264 mask on 1-OLx..sNx+OLx-1, 1-OLy..sNy+OLy-1
    M = _sl(L, 1 - L.OLy, L.sNy + L.OLy - 1, 1 - L.OLx, L.sNx + L.OLx - 1)
    uD = uD.at[M].set(uD[M] * g.f["maskW"][M] * recip_deepFacC)
    vD = vD.at[M].set(vD[M] * g.f["maskS"][M] * recip_deepFacC)
    return uD, vD


def mom_vi_coriolis(p, g, uFld, vFld, hFacW, hFacS):
    """MOM_VI_CORIOLIS (mom_vi_coriolis.F), useJamartWetPoints = T: uCoriolisTerm on j=1-OLy..sNy+OLy-1,
    i=2-OLx..sNx+OLx; vCoriolisTerm on j=2-OLy..sNy+OLy, i=1-OLx..sNx+OLx-1 (0 elsewhere; only iMin..iMax x
    jMin..jMax is used)."""
    if not p.useJamartWetPoints:
        raise NotImplementedError("MOM_VI_CORIOLIS: useJamartWetPoints = F is not ported")
    L = g.layout
    dxG, dyG, fCoriG = g2(g, "dxG"), g2(g, "dyG"), g2(g, "fCoriG")
    R = (1 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    vdh = lambda dj, di: s(vFld, dj, di) * s(dxG, dj, di) * s(hFacS, dj, di)  # noqa: E731
    vBarXY = ((vdh(0, 0) + vdh(0, -1)) + (vdh(1, 0) + vdh(1, -1))) \
        / jnp.maximum(EPSIL, (s(hFacS) + s(hFacS, 0, -1)) + (s(hFacS, 1, 0) + s(hFacS, 1, -1)))  # :43-49
    val = 0.5 * (s(fCoriG) + s(fCoriG, 1, 0)) * vBarXY * s(g2(g, "recip_dxC")) * s(g.f["maskW"])  # :50-52
    uCf = jnp.zeros_like(uFld).at[_sl(L, *R)].set(val)
    R = (2 - L.OLy, L.sNy + L.OLy, 1 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    udh = lambda dj, di: s(uFld, dj, di) * s(dyG, dj, di) * s(hFacW, dj, di)  # noqa: E731
    uBarXY = ((udh(0, 0) + udh(-1, 0)) + (udh(0, 1) + udh(-1, 1))) \
        / jnp.maximum(EPSIL, (s(hFacW) + s(hFacW, -1, 0)) + (s(hFacW, 0, 1) + s(hFacW, -1, 1)))  # :76-82
    val = -(0.5 * (s(fCoriG) + s(fCoriG, 0, 1)) * uBarXY * s(g2(g, "recip_dyC")) * s(g.f["maskS"]))  # :83-85
    vCf = jnp.zeros_like(vFld).at[_sl(L, *R)].set(val)
    return uCf, vCf


def mom_vi_u_coriolis(p, g, vFld, omega3, hFacZ):
    """MOM_VI_U_CORIOLIS (mom_vi_u_coriolis.F:86-109), selectVortScheme = 1, upwindVort3 = F,
    useJamartMomAdv = F: j=1-OLy..sNy+OLy-1, i=2-OLx..sNx+OLx."""
    if p.selectVortScheme != 1:
        raise NotImplementedError(f"selectVortScheme={p.selectVortScheme}")
    L = g.layout
    dxG = g2(g, "dxG")
    R = (1 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    vBarXY = 0.5 * ((s(vFld) * s(dxG) * s(hFacZ) + s(vFld, 0, -1) * s(dxG, 0, -1) * s(hFacZ))
                    + (s(vFld, 1, 0) * s(dxG, 1, 0) * s(hFacZ, 1, 0)
                       + s(vFld, 1, -1) * s(dxG, 1, -1) * s(hFacZ, 1, 0))) \
        / jnp.maximum(EPSIL, s(hFacZ) + s(hFacZ, 1, 0))  # :91-96
    vort3u = 0.5 * (s(omega3) + s(omega3, 1, 0))  # :104
    val = vort3u * vBarXY * s(g2(g, "recip_dxC")) * s(g.f["maskW"])  # :106-107
    return jnp.zeros_like(vFld).at[_sl(L, *R)].set(val)


def mom_vi_v_coriolis(p, g, uFld, omega3, hFacZ):
    """MOM_VI_V_CORIOLIS (mom_vi_v_coriolis.F:86-109), selectVortScheme = 1: j=2-OLy..sNy+OLy,
    i=1-OLx..sNx+OLx-1."""
    if p.selectVortScheme != 1:
        raise NotImplementedError(f"selectVortScheme={p.selectVortScheme}")
    L = g.layout
    dyG = g2(g, "dyG")
    R = (2 - L.OLy, L.sNy + L.OLy, 1 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    uBarXY = 0.5 * ((s(uFld) * s(dyG) * s(hFacZ) + s(uFld, -1, 0) * s(dyG, -1, 0) * s(hFacZ))
                    + (s(uFld, 0, 1) * s(dyG, 0, 1) * s(hFacZ, 0, 1)
                       + s(uFld, -1, 1) * s(dyG, -1, 1) * s(hFacZ, 0, 1))) \
        / jnp.maximum(EPSIL, s(hFacZ) + s(hFacZ, 0, 1))  # :91-96
    vort3v = 0.5 * (s(omega3) + s(omega3, 0, 1))  # :104
    val = -(vort3v * uBarXY * s(g2(g, "recip_dyC")) * s(g.f["maskS"]))  # :106-107
    return jnp.zeros_like(uFld).at[_sl(L, *R)].set(val)


def _kshift(a):
    """(a(k-1) with k-1 -> MAX(k-1,1), a(k+1) with k+1 -> MIN(k+1,Nr)) along axis 1."""
    km1 = jnp.concatenate([a[:, :1], a[:, :-1]], axis=1)
    kp1 = jnp.concatenate([a[:, 1:], a[:, -1:]], axis=1)
    return km1, kp1


def _mask_k(Nr):
    """mask_Km1 (0 at k=1, else 1) and mask_Kp1 (0 at k=Nr, else 1) as [Nr,1,1] (mom_vi_u_vertshear.F:49-54)."""
    m1 = jnp.ones(Nr).at[0].set(0.0)
    p1 = jnp.ones(Nr).at[Nr - 1].set(0.0)
    return vcol(m1), vcol(p1)


def mom_vi_u_vertshear(p, g, uFld, wFld, recip_hFacW):
    """MOM_VI_U_VERTSHEAR (mom_vi_u_vertshear.F:44-136), rAdvAreaWeight = T (selectKEscheme = 0),
    upwindShear = F: j=1-OLy..sNy+OLy, i=2-OLx..sNx+OLx, all k."""
    L = g.layout
    Nr = uFld.shape[1]
    mask_Km1, mask_Kp1 = _mask_k(Nr)
    u_km1, u_kp1 = _kshift(uFld)
    w_kp1 = _kshift(wFld)[1]
    maskC_km1 = _kshift(jnp.asarray(g.f["maskC"]))[0]
    rhoFacF = jnp.asarray(g.f["rhoFacF"])  # [Nr+1]
    rhoFacF_k, rhoFacF_kp1 = vcol(rhoFacF[:Nr]), vcol(jnp.concatenate([rhoFacF[1:Nr], rhoFacF[Nr - 1:Nr]]))
    recip_rhoFacC = vcol(1.0 / jnp.asarray(g.f["rhoFacC"]))  # set_ref_state.F:76 (1.0)
    R = (1 - L.OLy, L.sNy + L.OLy, 2 - L.OLx, L.sNx + L.OLx)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    rA, recip_rAw = g2(g, "rA"), g2(g, "recip_rAw")
    wBarXm = 0.5 * (s(wFld) * s(rA) * s(maskC_km1) + s(wFld, 0, -1) * s(rA, 0, -1) * s(maskC_km1, 0, -1)) \
        * mask_Km1 * deepFac2F * rhoFacF_k * s(recip_rAw)  # :68-72
    wBarXp = 0.5 * (s(w_kp1) * s(rA) + s(w_kp1, 0, -1) * s(rA, 0, -1)) \
        * mask_Kp1 * deepFac2F * rhoFacF_kp1 * s(recip_rAw)  # :75-79
    uZm = (s(uFld) - mask_Km1 * s(u_km1)) * rkSign  # :96
    uZp = (mask_Kp1 * s(u_kp1) - s(uFld)) * rkSign  # :103
    val = -(0.5 * (wBarXp * uZp + wBarXm * uZm) * s(recip_hFacW) * vcol(g.f["recip_drF"])
            * recip_deepFac2C * recip_rhoFacC)  # :130-133
    return jnp.zeros_like(uFld).at[_sl(L, *R)].set(val)


def mom_vi_v_vertshear(p, g, vFld, wFld, recip_hFacS):
    """MOM_VI_V_VERTSHEAR (mom_vi_v_vertshear.F:44-136): j=2-OLy..sNy+OLy, i=1-OLx..sNx+OLx, all k."""
    L = g.layout
    Nr = vFld.shape[1]
    mask_Km1, mask_Kp1 = _mask_k(Nr)
    v_km1, v_kp1 = _kshift(vFld)
    w_kp1 = _kshift(wFld)[1]
    maskC_km1 = _kshift(jnp.asarray(g.f["maskC"]))[0]
    rhoFacF = jnp.asarray(g.f["rhoFacF"])
    rhoFacF_k, rhoFacF_kp1 = vcol(rhoFacF[:Nr]), vcol(jnp.concatenate([rhoFacF[1:Nr], rhoFacF[Nr - 1:Nr]]))
    recip_rhoFacC = vcol(1.0 / jnp.asarray(g.f["rhoFacC"]))
    R = (2 - L.OLy, L.sNy + L.OLy, 1 - L.OLx, L.sNx + L.OLx)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    rA, recip_rAs = g2(g, "rA"), g2(g, "recip_rAs")
    wBarYm = 0.5 * (s(wFld) * s(rA) * s(maskC_km1) + s(wFld, -1, 0) * s(rA, -1, 0) * s(maskC_km1, -1, 0)) \
        * mask_Km1 * deepFac2F * rhoFacF_k * s(recip_rAs)  # :68-72
    wBarYp = 0.5 * (s(w_kp1) * s(rA) + s(w_kp1, -1, 0) * s(rA, -1, 0)) \
        * mask_Kp1 * deepFac2F * rhoFacF_kp1 * s(recip_rAs)  # :75-79
    vZm = (s(vFld) - mask_Km1 * s(v_km1)) * rkSign  # :96
    vZp = (mask_Kp1 * s(v_kp1) - s(vFld)) * rkSign  # :103
    val = -(0.5 * (wBarYp * vZp + wBarYm * vZm) * s(recip_hFacS) * vcol(g.f["recip_drF"])
            * recip_deepFac2C * recip_rhoFacC)  # :130-133
    return jnp.zeros_like(vFld).at[_sl(L, *R)].set(val)


def mom_vi_u_grad_ke(g, KE):
    """MOM_VI_U_GRAD_KE (mom_vi_u_grad_ke.F:31-36): j=1-OLy..sNy+OLy, i=2-OLx..sNx+OLx."""
    L = g.layout
    R = (1 - L.OLy, L.sNy + L.OLy, 2 - L.OLx, L.sNx + L.OLx)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    val = -(s(g2(g, "recip_dxC")) * (s(KE) - s(KE, 0, -1)) * s(g.f["maskW"]) * recip_deepFacC)
    return jnp.zeros_like(KE).at[_sl(L, *R)].set(val)


def mom_vi_v_grad_ke(g, KE):
    """MOM_VI_V_GRAD_KE (mom_vi_v_grad_ke.F:31-36): j=2-OLy..sNy+OLy, i=1-OLx..sNx+OLx."""
    L = g.layout
    R = (2 - L.OLy, L.sNy + L.OLy, 1 - L.OLx, L.sNx + L.OLx)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    val = -(s(g2(g, "recip_dyC")) * (s(KE) - s(KE, -1, 0)) * s(g.f["maskS"]) * recip_deepFacC)
    return jnp.zeros_like(KE).at[_sl(L, *R)].set(val)


# --------------------------------------------------------------------------------------------------------------
# MOM_VECINV


def mom_vecinv(p, g, uVel, vVel, wVel, hFacC, hFacW, hFacS, recip_hFacC, recip_hFacW, recip_hFacS,
               kappaRU, kappaRV, terms=False, visc=None):
    """MOM_VECINV (mom_vecinv.F) for all levels k at once. Returns dict gU, gV, guDissip, gvDissip ([T, Nr, j, i]);
    with terms=True also the intermediate fields (KE, vort3, hDiv, viscosities, each tendency term).
    `visc` overrides the (viscAh_Z, viscAh_D, viscA4_Z, viscA4_D) tuple (tests; the viscFacInAd adjoint seam,
    core/forward_step.mom_vecinv_adj, passes the MOM_CALC_VISC result so it can differentiate at other viscosities).
    Pass `p` as a jit argument (not closed over), so its float leaves are traced: XLA re-associates products of
    compile-time constants, the Fortran multiplies by run-time namelist values."""
    L = g.layout
    J, I = _sl(L, p.jMin, L.sNy + 1, p.iMin, L.sNx + 1)[1:]  # jMin..jMax, iMin..iMax
    JI = (Ellipsis, J, I)
    maskW, maskS = g.f["maskW"], g.f["maskS"]
    out = {}
    # mom_vecinv.F:226-230 sideMaskFac only enters vort3BC / strainBC (Leith / Smagorinsky terms, diagnostics):
    # not used in V4r4
    # :242 open water fraction at vorticity points
    hFacZ, r_hFacZ = mc.mom_calc_hfacz(L, hFacW, hFacS)
    uFld, vFld = uVel, vVel  # :245-250
    KE = mc.mom_calc_ke(L, uFld, vFld)  # :271
    vort3 = mc.mom_calc_relvort3(g, uFld, vFld, p.useCubedSphereExchange)  # :273
    # :285-293 (vort3BC only feeds MOM_CALC_VISC's Leith term and diagnostics: not used in V4r4)
    vort3 = jnp.where(hFacZ == 0.0, 0.0, vort3)

    # ---- momViscosity block (:302-588)
    h0FacZ = hFacZ  # :306-310
    if mc.NONLIN_FRSURF and p.no_slip_sides:  # :312-320 (nonlinFreeSurf > 0 checked at set-up)
        R = _sl(L, 2 - L.OLy, L.sNy + L.OLy, 2 - L.OLx, L.sNx + L.OLx)
        s = lambda a, dj=0, di=0: a[_sl(L, 2 - L.OLy, L.sNy + L.OLy, 2 - L.OLx, L.sNx + L.OLx, dj, di)]  # noqa
        h0W, h0S = g.f["h0FacW"], g.f["h0FacS"]
        val = jnp.minimum(jnp.minimum(s(h0W), s(h0W, -1, 0)), jnp.minimum(s(h0S), s(h0S, 0, -1)))
        h0FacZ = h0FacZ.at[R].set(val)
    hDiv = mc.mom_calc_hdiv(g, uFld, vFld, hFacW, hFacS, recip_hFacC)  # :330
    # :332-345 MOM_CALC_TENSION / MOM_CALC_STRAIN: their outputs only feed the Smagorinsky term (viscC2smag =
    # viscC4smag = 0) and diagnostics, so they have no effect on any output and are not computed.
    if visc is None:
        visc = mc.mom_calc_visc(p.visc, g)  # :359-373
    viscAh_Z, viscAh_D, viscA4_Z, viscA4_D = visc
    # :396-409 biharmonic
    del2u, del2v, hDiv = mom_vi_del2uv(p, g, hDiv, vort3, hFacZ, recip_hFacW, recip_hFacS)
    dStar = mc.mom_calc_hdiv(g, del2u, del2v, hFacW, hFacS, recip_hFacC)  # :406
    zStar = mc.mom_calc_relvort3(g, del2u, del2v, p.useCubedSphereExchange)  # :407-408
    # :435-440
    guDiss, gvDiss = mom_vi_hdissip(p, g, hDiv, vort3, dStar, zStar, hFacZ, viscAh_Z, viscAh_D, viscA4_Z,
                                    viscA4_D, recip_hFacW, recip_hFacS)
    if terms:
        out.update(hDissU=guDiss, hDissV=gvDiss)
    # :447-472 vertical viscous flux: skipped (implicitViscosity = T, checked at set-up)
    if p.no_slip_sides:  # :475-488
        vF = mc.mom_u_sidedrag(p.visc, g, uFld, del2u, h0FacZ, viscAh_Z, viscA4_Z, recip_hFacW)
        guDiss = guDiss.at[JI].set(guDiss[JI] + vF[JI])
        if terms:
            out["sideDragU"] = vF
    if p.bottomDragTerms:  # :491-501
        vF = mc.mom_u_bottomdrag(p.drag, g, uFld, vFld, KE, kappaRU, recip_hFacW)
        guDiss = guDiss.at[JI].set(guDiss[JI] + vF[JI])
        if terms:
            out["botDragU"] = vF
    if p.no_slip_sides:  # :546-559
        vF = mc.mom_v_sidedrag(p.visc, g, vFld, del2v, h0FacZ, viscAh_Z, viscA4_Z, recip_hFacS)
        gvDiss = gvDiss.at[JI].set(gvDiss[JI] + vF[JI])
        if terms:
            out["sideDragV"] = vF
    if p.bottomDragTerms:  # :562-572
        vF = mc.mom_v_bottomdrag(p.drag, g, uFld, vFld, KE, kappaRV, recip_hFacS)
        gvDiss = gvDiss.at[JI].set(gvDiss[JI] + vF[JI])
        if terms:
            out["botDragV"] = vF

    # ---- advection and Coriolis (:597-792); useAbsVorticity = F: no omega3
    gU = jnp.zeros_like(uVel)  # dynamics.F (ALLOW_AUTODIFF): gU = gV = 0 before the k loop
    gV = jnp.zeros_like(vVel)
    uCf, vCf = mom_vi_coriolis(p, g, uFld, vFld, hFacW, hFacS)  # :618-619
    gU = gU.at[JI].set(uCf[JI])  # :621-626
    gV = gV.at[JI].set(vCf[JI])
    if terms:
        out.update(coriU=uCf, coriV=vCf)
    uCf = mom_vi_u_coriolis(p, g, vFld, vort3, hFacZ)  # :676-677
    gU = gU.at[JI].set(gU[JI] + uCf[JI])  # :679-683
    vCf = mom_vi_v_coriolis(p, g, uFld, vort3, hFacZ)  # :695-696
    gV = gV.at[JI].set(gV[JI] + vCf[JI])  # :698-702
    if terms:
        out.update(vortU=uCf, vortV=vCf)
    uCf = mom_vi_u_vertshear(p, g, uVel, wVel, recip_hFacW)  # :743
    gU = gU.at[JI].set(gU[JI] + uCf[JI])
    vCf = mom_vi_v_vertshear(p, g, vVel, wVel, recip_hFacS)  # :749
    gV = gV.at[JI].set(gV[JI] + vCf[JI])
    if terms:
        out.update(shearU=uCf, shearV=vCf)
    uCf = mom_vi_u_grad_ke(g, KE)  # :764
    gU = gU.at[JI].set(gU[JI] + uCf[JI])
    vCf = mom_vi_v_grad_ke(g, KE)  # :770
    gV = gV.at[JI].set(gV[JI] + vCf[JI])
    if terms:
        out.update(gradKEU=uCf, gradKEV=vCf)
    # :795-827 use3dCoriolis = F, useNHMTerms = F
    # :830-835 set du/dt, dv/dt on boundaries to zero
    gU = gU.at[JI].set(gU[JI] * maskW[JI])
    gV = gV.at[JI].set(gV[JI] * maskS[JI])
    out.update(gU=gU, gV=gV, guDissip=guDiss, gvDissip=gvDiss)
    if terms:
        out.update(KE=KE, vort3=vort3, hDiv=hDiv, hFacZ=hFacZ, h0FacZ=h0FacZ, del2u=del2u, del2v=del2v,
                   dStar=dStar, zStar=zStar, viscAh_Z=viscAh_Z, viscAh_D=viscAh_D, viscA4_Z=viscA4_Z,
                   viscA4_D=viscA4_D)
    return out
