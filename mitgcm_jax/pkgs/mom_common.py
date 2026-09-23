"""pkg/mom_common: the MOM_COMMON routines MOM_VECINV calls in the ECCO v4r4 flux-forced build (plan Task 14a/14b).

Literal ports of (c66g `MITgcm_c66g/pkg/mom_common/`, unless noted):
    MOM_INIT_FIXED      mom_init_fixed.F:77-124   length scales L2_D, L2_Z, L3_D, L3_Z, L4rdt_D, L4rdt_Z
    MOM_CALC_VISC       V4r4 override `flux-forced/code/mom_calc_visc.F` (Gibraltar x10 on viscAh_D / viscAh_Z,
                        viscFacAdj on all four 3-D fields); only the branches V4r4 runs: variable viscosity from
                        viscAhGrid + 3-D viscAhDfld/viscAhZfld/viscA4Dfld/viscA4Zfld, no Leith, no Smagorinsky,
                        no Reynolds-number limiter (each of those options raises NotImplementedError at set-up)
    MOM_CALC_HFACZ      mom_calc_hfacz.F:165-378 (not ALLOW_DEPTH_CONTROL; hZoption = 0, so no CS-corner block)
    MOM_CALC_KE         mom_calc_ke.F:73-89 (KEscheme = selectKEscheme = 0)
    MOM_CALC_RELVORT3   mom_calc_relvort3.F:45-307 incl. the cubed-sphere corner formulas
                        (CALC_CS_CORNER_EXTENDED undefined)
    MOM_CALC_HDIV       mom_calc_hdiv.F:50-66 (hDivScheme = 2)
    MOM_U/V_SIDEDRAG    mom_u_sidedrag.F:103-148, mom_v_sidedrag.F (sideDragFactor > 0: variable viscosity)
    MOM_U/V_BOTTOMDRAG  mom_u_bottomdrag.F, mom_v_bottomdrag.F (usingZCoords, no_slip_bottom, bottomVisc_pCell=F,
                        selectBotDragQuadr = 0, ALLOW_BOTTOMDRAG_CONTROL undefined)
    FILL_CS_CORNER_TR_RL eesupp/src/fill_cs_corner_tr_rl.F (fill4dir 1 and 2)

All fields are [tile, k, j, i] (or [tile, j, i]) with halos (mitgcm_jax/layout.py); every level k is computed at
once (the routines are called inside the k loop of DYNAMICS but only read level-k data, except the bottom drag,
which reads maskW/S(kDown) and kappaRU/RV(k+1): passed as shifted 3-D arrays). A Fortran loop over i, j writes
exactly its range; every other point keeps the value the Fortran array held (the explicit initialisation loops of
the caller are ported with it).

CPP options of the build (flux-forced/code/*_OPTIONS.h, reference/build/.../bld): NONLIN_FRSURF defined
(CPP_OPTIONS.h:91), ALLOW_3D_VISCAH / ALLOW_3D_VISCA4 defined (MOM_COMMON_OPTIONS.h), ALLOW_AUTODIFF(_TAMC)
defined (packages.conf: autodiff; AUTODIFF_OPTIONS.h:35), ISOTROPIC_COS_SCALING / ALLOW_OBCS / ALLOW_DEPTH_CONTROL /
ALLOW_BOTTOMDRAG_CONTROL / ALLOW_NONHYDROSTATIC / CALC_CS_CORNER_EXTENDED undefined. _RS is Real*8 (REAL4_IS_SLOW).
Grid factors that are identically 1 in this configuration (deepAtmosphere=F: set_grid_factors.F:51-62; cosFacU/V=1
for a curvilinear grid: ini_grid.F:108-109; recip_rhoFacC=1: set_ref_state.F:73-77) are kept as literal factors of
1.0 where the Fortran multiplies by them (multiplication by 1.0 is exact).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mitgcm_jax.io.llc import FACET_SHAPE
from mitgcm_jax.params_io import params_pytree

# EEPARAMS / grid factors that are exactly one in V4r4 (see module docstring)
recip_deepFacC = 1.0  # set_grid_factors.F:54  recip_deepFacC(k) = 1. _d 0   (deepAtmosphere = F, set_defaults.F:73)
recip_deepFac2C = 1.0  # set_grid_factors.F:55
deepFac2F = 1.0  # set_grid_factors.F:59
cosFacU = 1.0  # ini_grid.F:108  cosFacU(j,bi,bj) = 1.   (usingCurvilinearGrid: not overwritten)
cosFacV = 1.0  # ini_grid.F:109

# CPP switches of the build (see module docstring)
NONLIN_FRSURF = True  # flux-forced/code/CPP_OPTIONS.h:91  #define NONLIN_FRSURF
ALLOW_3D_VISCAH = True  # flux-forced/code/MOM_COMMON_OPTIONS.h  #define ALLOW_3D_VISCAH
ALLOW_3D_VISCA4 = True  # flux-forced/code/MOM_COMMON_OPTIONS.h  #define ALLOW_3D_VISCA4
HZOPTION = 0  # mom_calc_hfacz.F:78  PARAMETER ( hZoption = 0 )


def _get(nml, fname, group, key, default):
    return nml.get(fname, group, key, default=default)


# --------------------------------------------------------------------------------------------------------------
# parameters


@params_pytree
@dataclass(frozen=True)
class MomViscParams:
    """Horizontal viscosity parameters (PARAMS.h / MOM_VISC.h) as MOM_CALC_VISC and the side drag use them.
    float fields are pytree leaves (traced when the params are a jit argument, params_io.params_pytree)."""
    viscAhD: float
    viscAhZ: float
    viscA4D: float
    viscA4Z: float
    viscAhGrid: float
    viscA4Grid: float
    viscAhGridMin: float
    viscAhGridMax: float
    viscAhMax: float
    viscA4GridMin: float
    viscA4GridMax: float
    viscA4Max: float
    deltaTmom: float
    sideDragFactor: float
    viscFacAdj: float = 1.0  # set_defaults.F:131  viscFacAdj = 1. _d 0 (TAF adjoint flips it; forward stays 1)
    useHarmonicVisc: bool = True
    useBiharmonicVisc: bool = True
    useVariableVisc: bool = True
    useAreaViscLength: bool = False

    @classmethod
    def from_namelists(cls, nml):
        g = lambda key, default, grp="parm01": _get(nml, "data", grp, key, default)  # noqa: E731
        viscAh = g("viscAh", 0.0)  # set_defaults.F:120  viscAh = 0. _d 3
        viscA4 = g("viscA4", 0.0)  # set_defaults.F:139  viscA4 = 0. _d 11
        # ini_parms.F:474-477: viscAhD/Z, viscA4D/Z default (UNSET_RL) to viscAh / viscA4
        viscAhD = g("viscAhD", viscAh)
        viscAhZ = g("viscAhZ", viscAh)
        viscA4D = g("viscA4D", viscA4)
        viscA4Z = g("viscA4Z", viscA4)
        viscAhGrid = g("viscAhGrid", 0.0)  # set_defaults.F:122
        viscA4Grid = g("viscA4Grid", 0.0)  # set_defaults.F:140
        # not ported (MOM_CALC_VISC branches V4r4 does not run): Leith, Smagorinsky, Reynolds-number limiter
        for key, default, cite in (("viscC2leith", 0.0, "set_defaults.F:127"),
                                   ("viscC2leithD", 0.0, "set_defaults.F:128"),
                                   ("viscC2smag", 0.0, "set_defaults.F:129"),
                                   ("viscC4leith", 0.0, "set_defaults.F:145"),
                                   ("viscC4leithD", 0.0, "set_defaults.F:146"),
                                   ("viscC4smag", 0.0, "set_defaults.F:147"),
                                   ("viscAhReMax", 0.0, "set_defaults.F:126"),
                                   ("viscA4ReMax", 0.0, "set_defaults.F:144")):
            if g(key, default) != 0.0:
                raise NotImplementedError(f"MOM_CALC_VISC: {key} != 0 ({cite} default 0) is not ported")
        if g("useFullLeith", False) or g("useSmag3D", False) or g("useStrainTensionVisc", False):
            raise NotImplementedError("useFullLeith / useSmag3D / useStrainTensionVisc are not ported")
        if g("nonHydrostatic", False) or g("quasiHydrostatic", False):
            raise NotImplementedError("non-/quasi-hydrostatic momentum is not ported")
        if g("deepAtmosphere", False):
            raise NotImplementedError("deepAtmosphere (deepFac != 1) is not ported")
        momStepping = g("momStepping", True)  # set_defaults.F:189
        momViscosity = momStepping and g("momViscosity", True)  # set_defaults.F:184; set_parms.F:60
        files = [_get(nml, "data", "parm05", f, " ") for f in ("viscAhDfile", "viscAhZfile", "viscA4Dfile",
                                                                "viscA4Zfile")]
        has_file = [f.strip() != "" for f in files]
        # set_parms.F:99-121
        useVariableVisc = momViscosity and (viscAhGrid != 0.0 or viscA4Grid != 0.0 or any(has_file))
        useHarmonicVisc = momViscosity and (viscAh != 0.0 or viscAhD != 0.0 or viscAhZ != 0.0
                                            or viscAhGrid != 0.0 or has_file[0] or has_file[1])
        useBiharmonicVisc = momViscosity and (viscA4 != 0.0 or viscA4D != 0.0 or viscA4Z != 0.0
                                              or viscA4Grid != 0.0 or has_file[2] or has_file[3])
        if not (useVariableVisc and useHarmonicVisc and useBiharmonicVisc):
            raise NotImplementedError("only useVariableVisc = useHarmonicVisc = useBiharmonicVisc = T is ported")
        if not g("sideDragFactor", 2.0) > 0.0:  # mom_u_sidedrag.F:61 old version (sideDragFactor <= 0)
            raise NotImplementedError("sideDragFactor <= 0 (old side-drag version) is not ported")
        deltaTmom = _get(nml, "data", "parm03", "deltaTmom", 0.0)  # set_defaults.F:294 deltaTMom = 0.
        if deltaTmom == 0.0:  # ini_parms.F:898  IF ( deltaTMom .EQ. 0. ) deltaTMom = deltaT
            deltaTmom = _get(nml, "data", "parm03", "deltaT", 0.0)
        return cls(
            viscAhD=float(viscAhD), viscAhZ=float(viscAhZ), viscA4D=float(viscA4D), viscA4Z=float(viscA4Z),
            viscAhGrid=float(viscAhGrid), viscA4Grid=float(viscA4Grid),
            viscAhGridMin=float(g("viscAhGridMin", 0.0)),  # set_defaults.F:123
            viscAhGridMax=float(g("viscAhGridMax", 1.0e21)),  # set_defaults.F:124  1. _d 21
            viscAhMax=float(g("viscAhMax", 1.0e21)),  # set_defaults.F:125
            viscA4GridMin=float(g("viscA4GridMin", 0.0)),  # set_defaults.F:142
            viscA4GridMax=float(g("viscA4GridMax", 1.0e21)),  # set_defaults.F:141
            viscA4Max=float(g("viscA4Max", 1.0e21)),  # set_defaults.F:143
            deltaTmom=float(deltaTmom),
            sideDragFactor=float(g("sideDragFactor", 2.0)),  # set_defaults.F:135
            useAreaViscLength=bool(g("useAreaViscLength", False)),  # set_defaults.F:202
            useHarmonicVisc=useHarmonicVisc, useBiharmonicVisc=useBiharmonicVisc, useVariableVisc=useVariableVisc)


# --------------------------------------------------------------------------------------------------------------
# helpers


def _sl(L, jlo, jhi, ilo, ihi, dj=0, di=0):
    """Index tuple of the Fortran range j=jlo..jhi, i=ilo..ihi shifted by (dj, di), last two axes."""
    return (Ellipsis, L.js(jlo + dj, jhi + dj), L.is_(ilo + di, ihi + di))


def g2(g, name):
    """A 2-D grid field [T, j, i] as [T, 1, j, i] (broadcasts against [T, k, j, i])."""
    return g.f[name][:, None]


def vcol(v):
    """A per-level vector [Nr] as [Nr, 1, 1] (broadcasts against [T, k, j, i])."""
    return jnp.asarray(v)[:, None, None]


def cs_tile_topology(layout):
    """exch2 facet/corner flags of every tile: w2_set_tile2tiles.F:86-111 sets isN/S/E/Wedge = 1 when the tile
    edge is a facet edge; FILL_CS_CORNER_TR_RL / MOM_CALC_RELVORT3 use corner = isXedge .AND. isYedge
    (fill_cs_corner_tr_rl.F:77-85). Tiles are numbered facet by facet, i fastest within a facet (W2 default
    numbering, data.exch2 preDefTopol=0 with dimsFacets = FACET_SHAPE)."""
    L = layout
    face, sw, se, nw, ne = [], [], [], [], []
    for f in sorted(FACET_SHAPE):
        ny_f, nx_f = FACET_SHAPE[f]
        nbTx, nbTy = nx_f // L.sNx, ny_f // L.sNy
        for tj in range(nbTy):
            for ti in range(nbTx):
                isW, isE = ti == 0, ti == nbTx - 1
                isS, isN = tj == 0, tj == nbTy - 1
                face.append(f)
                sw.append(isW and isS)
                se.append(isE and isS)
                nw.append(isW and isN)
                ne.append(isE and isN)
    if len(face) != L.nTiles:
        raise ValueError(f"facet layout gives {len(face)} tiles, layout has {L.nTiles}")
    return {"face": np.array(face), "sw": np.array(sw), "se": np.array(se), "nw": np.array(nw),
            "ne": np.array(ne)}


def _tile_flag(flag, ndim):
    """[T] bool -> broadcastable against a point value of an array with `ndim` dims ([T] or [T, k])."""
    return jnp.asarray(flag).reshape((-1,) + (1,) * (ndim - 3))


def _set_point(a, L, j, i, flag, val):
    """a(i,j) = val on the tiles where flag is set (Fortran IF (corner) a(i,j) = val)."""
    cur = a[..., L.jj(j), L.ii(i)]
    return a.at[..., L.jj(j), L.ii(i)].set(jnp.where(_tile_flag(flag, a.ndim), val, cur))


def fill_cs_corner_tr_rl(fld, fill4dir, withSigns, L, useCubedSphereExchange=True):
    """FILL_CS_CORNER_TR_RL (eesupp/src/fill_cs_corner_tr_rl.F:71-270), fill4dir 1 (X) or 2 (Y).
    Source and destination points never overlap (the sources lie in the edge halos, not in the corner blocks),
    so the Fortran element loop is one gather + one scatter."""
    if not useCubedSphereExchange:  # fill_cs_corner_tr_rl.F:74
        return fld
    topo = cs_tile_topology(L)
    negOne = -1.0 if withSigns else 1.0  # fill_cs_corner_tr_rl.F:71-72
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    dst, src, cid = [], [], []
    for c in range(4):  # 0 SW, 1 SE, 2 NW, 3 NE (fortran (i, j) pairs below)
        for j in range(1, OLy + 1):
            for i in range(1, OLx + 1):
                if fill4dir == 1:
                    d, s = [((1 - i, 1 - j), (1 - j, i)),  # fill_cs_corner_tr_rl.F:168
                            ((sNx + i, 1 - j), (sNx + j, i)),  # :175
                            ((1 - i, sNy + j), (1 - j, sNy + 1 - i)),  # :182
                            ((sNx + i, sNy + j), (sNx + j, sNy + 1 - i))][c]  # :189
                elif fill4dir == 2:
                    d, s = [((1 - i, 1 - j), (j, 1 - i)),  # fill_cs_corner_tr_rl.F:238
                            ((sNx + i, 1 - j), (sNx + 1 - j, 1 - i)),  # :245
                            ((1 - i, sNy + j), (j, sNy + i)),  # :252
                            ((sNx + i, sNy + j), (sNx + 1 - j, sNy + i))][c]  # :259
                else:
                    raise NotImplementedError(f"FILL_CS_CORNER_TR_RL fill4dir={fill4dir} is not ported")
                dst.append((L.jj(d[1]), L.ii(d[0])))
                src.append((L.jj(s[1]), L.ii(s[0])))
                cid.append(c)
    dst, src, cid = np.array(dst), np.array(src), np.array(cid)
    assert not (set(map(tuple, dst)) & set(map(tuple, src)))
    flags = np.stack([topo["sw"], topo["se"], topo["nw"], topo["ne"]], axis=1)[:, cid]  # [T, n]
    flags = jnp.asarray(flags).reshape((flags.shape[0],) + (1,) * (fld.ndim - 3) + (flags.shape[1],))
    vals = fld[..., src[:, 0], src[:, 1]]
    cur = fld[..., dst[:, 0], dst[:, 1]]
    return fld.at[..., dst[:, 0], dst[:, 1]].set(jnp.where(flags, negOne * vals, cur))


# --------------------------------------------------------------------------------------------------------------
# viscosity (Task 14a)


def _recip_dt(p):
    """recip_dt = 1; IF ( deltaTmom.NE.0. ) recip_dt = 1/deltaTmom (mom_init_fixed.F:39-40,
    mom_calc_visc.F:167-168), with deltaTmom a traced leaf."""
    nz = p.deltaTmom != 0.0
    return jnp.where(nz, 1.0 / jnp.where(nz, p.deltaTmom, 1.0), 1.0)


def visc_length_scales(p, g):
    """MOM_INIT_FIXED length scales (mom_init_fixed.F:77-124), [T, j, i]."""
    recip_dt = _recip_dt(p)  # mom_init_fixed.F:39-40
    out = {}
    for pt, rA, rx, ry in (("D", "rA", "recip_dxF", "recip_dyF"), ("Z", "rAz", "recip_dxV", "recip_dyU")):
        L2 = jnp.asarray(g.f[rA])  # mom_init_fixed.F:80 / :104
        if not p.useAreaViscLength:
            a, b = jnp.asarray(g.f[rx]), jnp.asarray(g.f[ry])
            cond = (a != 0.0) | (b != 0.0)  # mom_init_fixed.F:86-87 / :110-111
            den = (a * a + b * b)  # **2 -> x*x (integer power)
            L2 = jnp.where(cond, 2.0 / jnp.where(cond, den, 1.0), L2)  # mom_init_fixed.F:88-89 / :112-113
        out["L2_" + pt] = L2
        out["L3_" + pt] = L2 ** 1.5  # mom_init_fixed.F:96 / :120
        out["L4rdt_" + pt] = (0.03125 * recip_dt) * (L2 * L2)  # mom_init_fixed.F:97-98 / :121-122
    return out


def gibraltar_mask(x, y):
    """V4r4 override mom_calc_visc.F:413-417 (D points, xC/yC) and :523-527 (Z points, xG/yG)."""
    return (y >= 33.0) & (y <= 39.0) & (x >= -7.0) & (x <= -2.0)


def mom_calc_visc(p, g, lengths=None, gibraltar=True):
    """Viscosity arrays as MOM_VECINV hands them to the dissipation: the constant initialisation of
    mom_vecinv.F:359-366 everywhere, overwritten by MOM_CALC_VISC (flux-forced/code/mom_calc_visc.F) on
    i,j = 2-OLx..sNx+OLx-1, 2-OLy..sNy+OLy-1. Returns viscAh_Z, viscAh_D, viscA4_Z, viscA4_D, all [T, Nr, j, i].
    `gibraltar=False` drops the V4r4 x10 region (negative controls only)."""
    L = g.layout
    lengths = visc_length_scales(p, g) if lengths is None else lengths
    Nr = g.f["maskC"].shape[1]
    shape = (L.nTiles, Nr, L.ny, L.nx)
    recip_dt = _recip_dt(p)  # mom_calc_visc.F:167-168
    # mom_calc_visc.F:170-217: viscAhRe_max = viscA4Re_max = 0, calcLeith = calcSmag = F (from_namelists checks)
    S = _sl(L, 2 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx - 1)
    zero = 0.0  # viscAh_DLth = viscAh_DSmg = ... = 0 (mom_calc_visc.F:227-243, :382-399 ELSE branches)
    Uscl = 0.0  # mom_calc_visc.F:335 / :449-454
    U4scl = 0.0  # mom_calc_visc.F:340 / :450-454
    out = {}
    for pt, xg, yg in (("D", "xC", "yC"), ("Z", "xG", "yG")):
        L2 = lengths["L2_" + pt][:, None][S]  # mom_calc_visc.F:324 / :434
        L2rdt = (0.25 * recip_dt) * L2  # :325 / :435
        L4rdt = lengths["L4rdt_" + pt][:, None][S]  # :327 / :437
        viscAhX = p.viscAhD if pt == "D" else p.viscAhZ
        viscA4X = p.viscA4D if pt == "D" else p.viscA4Z
        fAh = g.f["viscAh" + pt + "fld"][S]
        fA4 = g.f["viscA4" + pt + "fld"][S]
        # Harmonic (:403-411 / :513-521)
        Alin = viscAhX + p.viscAhGrid * L2rdt
        Alin = Alin + zero
        Alin = Alin + zero
        if ALLOW_3D_VISCAH:
            Alin = Alin + p.viscFacAdj * fAh
        ahMin = jnp.maximum(p.viscAhGridMin * L2rdt, Uscl)
        ah = jnp.maximum(ahMin, Alin)
        ahMax = jnp.minimum(p.viscAhGridMax * L2rdt, p.viscAhMax)
        ah = jnp.minimum(ahMax, ah)
        if gibraltar:  # V4r4 override :413-419 / :523-529
            box = gibraltar_mask(g.f[xg][:, None][S], g.f[yg][:, None][S])
            ah = jnp.where(box, 10.0 * ah, ah)
        # BiHarmonic (:422-430 / :532-540)
        Alin = viscA4X + p.viscA4Grid * L4rdt
        Alin = Alin + zero
        Alin = Alin + zero
        if ALLOW_3D_VISCA4:
            Alin = Alin + p.viscFacAdj * fA4
        a4Min = jnp.maximum(p.viscA4GridMin * L4rdt, U4scl)
        a4 = jnp.maximum(a4Min, Alin)
        a4Max = jnp.minimum(p.viscA4GridMax * L4rdt, p.viscA4Max)
        a4 = jnp.minimum(a4Max, a4)
        # mom_vecinv.F:359-366 initialisation, then the MOM_CALC_VISC loop range
        out["viscAh_" + pt] = jnp.full(shape, viscAhX).at[S].set(ah)
        out["viscA4_" + pt] = jnp.full(shape, viscA4X).at[S].set(a4)
    return out["viscAh_Z"], out["viscAh_D"], out["viscA4_Z"], out["viscA4_D"]


# --------------------------------------------------------------------------------------------------------------
# kinematics


def mom_calc_hfacz(L, hFacW, hFacS):
    """MOM_CALC_HFACZ (mom_calc_hfacz.F:165-378, hZoption = 0): hFacZ, r_hFacZ [T, k, j, i]."""
    hFacZ = jnp.zeros_like(hFacW)  # :177-182 first row & column = 0
    S = _sl(L, 2 - L.OLy, L.sNy + L.OLy, 2 - L.OLx, L.sNx + L.OLx)
    sh = lambda a, dj, di: a[_sl(L, 2 - L.OLy, L.sNy + L.OLy, 2 - L.OLx, L.sNx + L.OLx, dj, di)]  # noqa: E731
    if HZOPTION != 0:
        raise NotImplementedError("hZoption != 0")
    hFacZOpen = jnp.minimum(sh(hFacW, 0, 0), sh(hFacW, -1, 0))  # :225-226
    hFacZOpen = jnp.minimum(sh(hFacS, 0, 0), hFacZOpen)  # :227
    hFacZOpen = jnp.minimum(sh(hFacS, 0, -1), hFacZOpen)  # :228
    hFacZ = hFacZ.at[S].set(hFacZOpen)  # :229
    # :236 cubed-sphere block only for hZoption >= 1
    zero = hFacZ == 0.0  # :370-378
    r_hFacZ = jnp.where(zero, 0.0, 1.0 / jnp.where(zero, 1.0, hFacZ))
    return hFacZ, r_hFacZ


def mom_calc_ke(L, uFld, vFld, KEscheme=0):
    """MOM_CALC_KE (mom_calc_ke.F:56-89): KE on j=1-OLy..sNy+OLy-1, i=1-OLx..sNx+OLx-1, 0 elsewhere."""
    if KEscheme != 0:
        raise NotImplementedError(f"selectKEscheme={KEscheme}")
    KE = jnp.zeros_like(uFld)  # :57-61 (ALLOW_AUTODIFF)
    R = (1 - L.OLy, L.sNy + L.OLy - 1, 1 - L.OLx, L.sNx + L.OLx - 1)
    u0, u1 = uFld[_sl(L, *R)], uFld[_sl(L, *R, 0, 1)]
    v0, v1 = vFld[_sl(L, *R)], vFld[_sl(L, *R, 1, 0)]
    val = 0.25 * ((u0 * u0 + u1 * u1) + (v0 * v0 + v1 * v1))  # :82-87
    return KE.at[_sl(L, *R)].set(val)


def mom_calc_relvort3(g, uFld, vFld, useCubedSphereExchange=True):
    """MOM_CALC_RELVORT3 (mom_calc_relvort3.F:45-307) incl. the cubed-sphere corner formulas."""
    L = g.layout
    dxC, dyC, recip_rAz = g2(g, "dxC"), g2(g, "dyC"), g2(g, "recip_rAz")
    vort3 = jnp.zeros_like(uFld)  # :45-51 (ALLOW_AUTODIFF)
    R = (2 - L.OLy, L.sNy + L.OLy, 2 - L.OLx, L.sNx + L.OLx)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    val = (s(recip_rAz) * ((s(vFld) * s(dyC) - s(vFld, 0, -1) * s(dyC, 0, -1))
                           - (s(uFld) * s(dxC) - s(uFld, -1, 0) * s(dxC, -1, 0)))) * recip_deepFacC  # :57-63
    vort3 = vort3.at[_sl(L, *R)].set(val)
    if not useCubedSphereExchange:  # :83
        return vort3
    topo = cs_tile_topology(L)
    face = jnp.asarray(topo["face"]).reshape((-1,) + (1,) * (uFld.ndim - 3))

    def P(a, j, i):
        return a[..., L.jj(j), L.ii(i)]

    sNx, sNy = L.sNx, L.sNy
    # SW corner I=1, J=1 (:113-122)
    I, J = 1, 1
    val = P(recip_rAz, J, I) * ((P(vFld, J, I) * P(dyC, J, I) - P(uFld, J, I) * P(dxC, J, I))
                                + P(uFld, J - 1, I) * P(dxC, J - 1, I)) * recip_deepFacC
    vort3 = _set_point(vort3, L, J, I, topo["sw"], val)
    # SE corner I=sNx+1, J=1 (:154-179)
    I, J = sNx + 1, 1
    uIJ, uIJm, vImJ = P(uFld, J, I) * P(dxC, J, I), P(uFld, J - 1, I) * P(dxC, J - 1, I), \
        P(vFld, J, I - 1) * P(dyC, J, I - 1)
    v2 = P(recip_rAz, J, I) * ((-uIJ - vImJ) + uIJm) * recip_deepFacC  # myFace 2 (:158-164)
    v4 = P(recip_rAz, J, I) * ((-vImJ + uIJm) - uIJ) * recip_deepFacC  # myFace 4 (:165-171)
    vo = P(recip_rAz, J, I) * ((uIJm - uIJ) - vImJ) * recip_deepFacC  # else (:173-178)
    val = jnp.where(face == 2, v2, jnp.where(face == 4, v4, vo))
    vort3 = _set_point(vort3, L, J, I, topo["se"], val)
    # NW corner I=1, J=sNy+1 (:211-236)
    I, J = 1, sNy + 1
    uIJ, uIJm, vIJ = P(uFld, J, I) * P(dxC, J, I), P(uFld, J - 1, I) * P(dxC, J - 1, I), \
        P(vFld, J, I) * P(dyC, J, I)
    v1 = P(recip_rAz, J, I) * ((uIJm + vIJ) - uIJ) * recip_deepFacC  # myFace 1 (:215-221)
    v3 = P(recip_rAz, J, I) * ((-uIJ + uIJm) + vIJ) * recip_deepFacC  # myFace 3 (:222-228)
    vo = P(recip_rAz, J, I) * ((vIJ - uIJ) + uIJm) * recip_deepFacC  # else (:230-235)
    val = jnp.where(face == 1, v1, jnp.where(face == 3, v3, vo))
    vort3 = _set_point(vort3, L, J, I, topo["nw"], val)
    # NE corner I=sNx+1, J=sNy+1 (:268-286)
    I, J = sNx + 1, sNy + 1
    uIJ, uIJm, vImJ = P(uFld, J, I) * P(dxC, J, I), P(uFld, J - 1, I) * P(dxC, J - 1, I), \
        P(vFld, J, I - 1) * P(dyC, J, I - 1)
    vodd = P(recip_rAz, J, I) * ((-uIJ - vImJ) + uIJm) * recip_deepFacC  # MOD(myFace,2).EQ.1 (:272-278)
    veven = P(recip_rAz, J, I) * ((uIJm - uIJ) - vImJ) * recip_deepFacC  # (:280-285)
    val = jnp.where(face % 2 == 1, vodd, veven)
    vort3 = _set_point(vort3, L, J, I, topo["ne"], val)
    return vort3


def mom_calc_hdiv(g, uFld, vFld, hFacW, hFacS, recip_hFacC, hDivScheme=2):
    """MOM_CALC_HDIV (mom_calc_hdiv.F:50-66, hDivScheme = 2) on j=1-OLy..sNy+OLy-1, i=1-OLx..sNx+OLx-1; the
    caller's array keeps its other points (MOM_VECINV: 0, mom_vecinv.F:204)."""
    if hDivScheme != 2:
        raise NotImplementedError(f"hDivScheme={hDivScheme}")
    L = g.layout
    dyG, dxG, recip_rA = g2(g, "dyG"), g2(g, "dxG"), g2(g, "recip_rA")
    R = (1 - L.OLy, L.sNy + L.OLy - 1, 1 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    val = (((s(uFld, 0, 1) * s(dyG, 0, 1) * s(hFacW, 0, 1) - s(uFld) * s(dyG) * s(hFacW))
            + (s(vFld, 1, 0) * s(dxG, 1, 0) * s(hFacS, 1, 0) - s(vFld) * s(dxG) * s(hFacS)))
           * s(recip_rA) * recip_deepFacC * s(recip_hFacC))  # :55-61
    return jnp.zeros_like(uFld).at[_sl(L, *R)].set(val)


# --------------------------------------------------------------------------------------------------------------
# drag terms


def mom_u_sidedrag(p, g, uFld, del2u, hFacZ, viscAh_Z, viscA4_Z, recip_hFacW):
    """MOM_U_SIDEDRAG, sideDragFactor > 0 branch (mom_u_sidedrag.F:103-145; sideDragFactor <= 0 is refused in
    MomViscParams.from_namelists) with NONLIN_FRSURF (h0FacW); hFacZ is the caller's h0FacZ. Values on
    j=2-OLy..sNy+OLy-1, i=2-OLx..sNx+OLx-1 (0 elsewhere)."""
    L = g.layout
    R = (2 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    h0FacW = g.f["h0FacW"]
    hFacZClosedS = s(h0FacW) - s(hFacZ)  # :110
    hFacZClosedN = s(h0FacW) - s(hFacZ, 1, 0)  # :111
    dxV, recip_dyU, recip_rAw = g2(g, "dxV"), g2(g, "recip_dyU"), g2(g, "recip_rAw")
    recip_drF, drF = vcol(g.f["recip_drF"]), vcol(g.f["drF"])
    u, d2 = s(uFld), s(del2u)
    val = -(((s(recip_hFacW) * recip_drF * s(recip_rAw))
             * (hFacZClosedS * s(dxV) * s(recip_dyU) * (s(viscAh_Z) * u - s(viscA4_Z) * d2)
                + hFacZClosedN * s(dxV, 1, 0) * s(recip_dyU, 1, 0) * (s(viscAh_Z, 1, 0) * u - s(viscA4_Z, 1, 0) * d2)))
            * drF * p.sideDragFactor)  # :116-143
    return jnp.zeros_like(uFld).at[_sl(L, *R)].set(val)


def mom_v_sidedrag(p, g, vFld, del2v, hFacZ, viscAh_Z, viscA4_Z, recip_hFacS):
    """MOM_V_SIDEDRAG, sideDragFactor > 0 branch (mom_v_sidedrag.F:103-132)."""
    L = g.layout
    R = (2 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    h0FacS = g.f["h0FacS"]
    hFacZClosedW = s(h0FacS) - s(hFacZ)  # :107
    hFacZClosedE = s(h0FacS) - s(hFacZ, 0, 1)  # :108
    dyU, recip_dxV, recip_rAs = g2(g, "dyU"), g2(g, "recip_dxV"), g2(g, "recip_rAs")
    recip_drF, drF = vcol(g.f["recip_drF"]), vcol(g.f["drF"])
    v, d2 = s(vFld), s(del2v)
    val = -(((s(recip_hFacS) * recip_drF * s(recip_rAs))
             * (hFacZClosedW * s(dyU) * s(recip_dxV)
                * (s(viscAh_Z) * v * cosFacV - s(viscA4_Z) * d2 * cosFacV)
                + hFacZClosedE * s(dyU, 0, 1) * s(recip_dxV, 0, 1)
                * (s(viscAh_Z, 0, 1) * v * cosFacV - s(viscA4_Z, 0, 1) * d2 * cosFacV)))
            * drF * p.sideDragFactor)  # :113-130
    return jnp.zeros_like(vFld).at[_sl(L, *R)].set(val)


@params_pytree
@dataclass(frozen=True)
class BottomDragParams:
    no_slip_bottom: bool
    bottomDragLinear: float
    bottomDragQuadratic: float
    selectBotDragQuadr: int
    bottomVisc_pCell: bool

    @classmethod
    def from_namelists(cls, nml):
        g = lambda key, default: _get(nml, "data", "parm01", key, default)  # noqa: E731
        Cd = float(g("bottomDragQuadratic", 0.0))  # set_defaults.F:137
        sel = int(g("selectBotDragQuadr", -1))  # set_defaults.F:138
        if sel == -1 and Cd != 0.0:  # ini_parms.F:512-513
            sel = 0
        momViscosity = g("momStepping", True) and g("momViscosity", True)
        if Cd == 0.0 or not momViscosity:  # set_parms.F:122-123
            sel = -1
        p = cls(no_slip_bottom=bool(g("no_slip_bottom", True)),  # set_defaults.F:133
                bottomDragLinear=float(g("bottomDragLinear", 0.0)),  # set_defaults.F:136
                bottomDragQuadratic=Cd, selectBotDragQuadr=sel,
                bottomVisc_pCell=bool(g("bottomVisc_pCell", False)))  # set_defaults.F:134
        if p.selectBotDragQuadr not in (-1, 0):
            raise NotImplementedError(f"selectBotDragQuadr={p.selectBotDragQuadr} is not ported")
        if p.bottomVisc_pCell:
            raise NotImplementedError("bottomVisc_pCell is not ported")
        return p


def _bottomdrag_factors(g, mask, recip_hFac):
    """recDrF_bot and recDrC of MOM_U/V_BOTTOMDRAG (mom_u_bottomdrag.F:65-94, usingZCoords) for all k:
    kBottom = Nr, kDown = MIN(k+1,Nr), kLowF = k+1."""
    Nr = mask.shape[1]
    recip_drF = jnp.asarray(g.f["recip_drF"])
    recip_drC = jnp.asarray(g.f["recip_drC"])  # [Nr+1]
    # k < Nr: recDrC = recip_drC(kLowF) (:87); k = Nr: recip_drF(k) (:80)
    recDrC = jnp.concatenate([recip_drC[1:Nr], recip_drF[Nr - 1:Nr]])
    # k < Nr: ( 1 - maskW(kDown) ) (:90-91); k = Nr: no factor (:83), written as the exact factor 1.0
    fac = jnp.concatenate([1.0 - mask[:, 1:Nr], jnp.ones_like(mask[:, :1])], axis=1)
    recDrF_bot = recip_hFac * vcol(recip_drF) * fac
    return recDrF_bot, vcol(recDrC)


def mom_u_bottomdrag(p, g, uFld, vFld, KE, kappaRU, recip_hFacW):
    """MOM_U_BOTTOMDRAG (mom_u_bottomdrag.F:62-191) for all k; kappaRU is [T, Nr+1, j, i]. Values on
    j=1-OLy..sNy+OLy-1, i=2-OLx..sNx+OLx-1 (0 elsewhere)."""
    L = g.layout
    Nr = uFld.shape[1]
    viscFac = 2.0 if p.no_slip_bottom else 0.0  # :63-64
    dragFac = 1.0  # :71 (usingZCoords)
    recDrF_bot, recDrC = _bottomdrag_factors(g, g.f["maskW"], recip_hFacW)
    R = (1 - L.OLy, L.sNy + L.OLy - 1, 2 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    rb, u = s(recDrF_bot), s(uFld)
    val = -(rb * (p.bottomDragLinear * dragFac) * u)  # :99-106
    if p.no_slip_bottom:  # :122-131
        kap = s(kappaRU[:, 1:Nr + 1])  # kappaRU(i,j,kLowF), kLowF = k+1
        val = val - rb * (kap * recDrC * viscFac) * u
    if p.selectBotDragQuadr == 0:  # :135-146
        ke2 = s(KE) + s(KE, 0, -1)
        pos = ke2 > 0.0
        drag = rb * p.bottomDragQuadratic * jnp.sqrt(jnp.where(pos, ke2, 1.0)) * dragFac * u
        val = jnp.where(pos, val - drag, val)
    return jnp.zeros_like(uFld).at[_sl(L, *R)].set(val)


def mom_v_bottomdrag(p, g, uFld, vFld, KE, kappaRV, recip_hFacS):
    """MOM_V_BOTTOMDRAG (mom_v_bottomdrag.F:62-191): j=2-OLy..sNy+OLy-1, i=1-OLx..sNx+OLx-1 (0 elsewhere)."""
    L = g.layout
    Nr = vFld.shape[1]
    viscFac = 2.0 if p.no_slip_bottom else 0.0  # :63-64
    dragFac = 1.0  # :71
    recDrF_bot, recDrC = _bottomdrag_factors(g, g.f["maskS"], recip_hFacS)
    R = (2 - L.OLy, L.sNy + L.OLy - 1, 1 - L.OLx, L.sNx + L.OLx - 1)
    s = lambda a, dj=0, di=0: a[_sl(L, *R, dj, di)]  # noqa: E731
    rb, v = s(recDrF_bot), s(vFld)
    val = -(rb * (p.bottomDragLinear * dragFac) * v)  # :97-108
    if p.no_slip_bottom:  # :122-131
        kap = s(kappaRV[:, 1:Nr + 1])
        val = val - rb * (kap * recDrC * viscFac) * v
    if p.selectBotDragQuadr == 0:  # :135-146
        ke2 = s(KE) + s(KE, -1, 0)
        pos = ke2 > 0.0
        drag = rb * p.bottomDragQuadratic * jnp.sqrt(jnp.where(pos, ke2, 1.0)) * dragFac * v
        val = jnp.where(pos, val - drag, val)
    return jnp.zeros_like(vFld).at[_sl(L, *R)].set(val)
