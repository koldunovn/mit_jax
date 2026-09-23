"""Grid initialisation from the run's input files (plan Task 6): literal host-side (numpy float64) port of the
MITgcm c66g set-up of every field in `geometry.G2D/G3D/R3D/VROWS`, for the ECCO v4r4 flux-forced configuration.

Call order (model/src/initialise_fixed.F, then initialise_varia.F for INI_MIXING):
    INI_GRID              ini_grid.F:60-183
      INI_VERTICAL_GRID   ini_vertical_grid.F        drF, drC, rF, rC, recip_drF, recip_drC
      INI_CURVILINEAR_GRID ini_curvilinear_grid.F    MDS_FACEF_READ_RS of tileNNN.mitgrid, exchanges
        CALC_GRID_ANGLES  calc_grid_angles.F         angleCosC/SinC, u2zonDir, v2zonDir (+ exchange)
      recip_* loop        ini_grid.F:152-183
    LOAD_REF_FILES        load_ref_files.F           tRef, sRef, gravFacC/F
    SET_REF_STATE         set_ref_state.F            phiRef, rVel2wUnit, wUnit2rVel, rhoFacC/F (dBdrRef: NOT ported)
    INI_DEPTHS            ini_depths.F               R_low (bathyFile), Ro_surf
    INI_MASKS_ETC         ini_masks_etc.F            hFac, R_low/Ro_surf update, rLow/rSurf W/S, recip_Rcol,
      ADD_WALLS2MASKS     add_walls2masks.F          k indices, maskIn*, masks, h0Fac (NONLIN_FRSURF); walls at dxG/dyG=0
    PACKAGES_INIT_FIXED   MOM_INIT_FIXED (viscA4/viscAh fields), GMREDI_INIT_FIXED (GM_isoFac/bolFac)
    INI_CORI              ini_cori.F                 fCori, fCoriG, fCoriCos
    INI_MIXING            ini_mixing.F               diffKr, kapGM, kapRedi (initialise_varia.F:192)

Arrays are `[tile, (k,) j, i]` with halos (mitgcm_jax.layout). Fortran loops over `1-OLx..sNx+OLx` are whole-array
numpy expressions with the Fortran operation order kept (`a*b*c` is `(a*b)*c`); sums over k are sequential in k as in
Fortran. SIN/COS/SQRT: SQRT is correctly rounded in both codes; SIN/COS are evaluated with Python's `math.sin/cos`,
i.e. the same glibc libm the gfortran oracle calls (numpy's SIMD sin/cos are not guaranteed to agree bit for bit),
and with glibc `sincos` where gfortran fused a SIN/COS pair of one argument (INI_CORI, see `_sincos`).
Exchanges use `mitgcm_jax.parallel.exchange.Exchanger`, called where the Fortran calls them. Known gap (Task 7): the
probe-derived maps cannot express exch2 copies whose source is a halo point (e.g. exch2_z_3d_rx.template:108-116),
so values at halo points no probed exchange writes (open facet edges, facet corners) and what is computed from them
can differ from Fortran; mitgcm_jax/tests/test_grid_load.py measures and bounds that set.

Branches the V4r4 run does not execute raise NotImplementedError at parameter set-up (`GridParams.from_namelists`).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mitgcm_jax.grid.geometry import Grid
from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.io.mds import read_bin
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists

PI = 3.14159265358979323844          # PARAMS.h:20  PI = 3.14159265358979323844D0
DEG2RAD = 2.0 * PI / 360.0           # PARAMS.h:22  deg2rad = 2.D0*PI/360.D0
ZERO_RL, ONE_RL, HALF_RL = 0.0, 1.0, 0.5   # EEPARAMS.h:81-82
UNSET_RL = 1.234567e5                # EEPARAMS.h:90
PREC_FLOAT64 = 64                    # EEPARAMS.h:74 (ini_curvilinear_grid.F:251 fp = precFloat64)


# ---------------------------------------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------------------------------------
def _blank(s):
    return s is None or str(s).strip() == ""


@dataclass(frozen=True)
class GridParams:
    """Run-time parameters of the grid set-up, from the run's namelists or the cited Fortran default."""
    delR: tuple                 # data PARM04 delR (ini_parms.F:1302 delR(k) stays as read)
    rSphere: float              # ini_parms.F:1182-1188
    radius_fromHorizGrid: float  # ini_parms.F:1190-1192
    omega: float                # ini_parms.F:449-452
    selectCoriMap: int          # ini_parms.F:1250-1258
    seaLev_Z: float             # ini_parms.F:1217
    top_Pres: float             # ini_parms.F:1216
    hFacMin: float              # data PARM04 hFacMin (set_defaults.F:178 default 1.)
    hFacMinDr: float            # ini_parms.F:593-595
    gravity: float              # data PARM01 gravity (set_defaults.F:104 default 9.81)
    rhoConst: float             # ini_parms.F:445
    tRef: tuple                 # load_ref_files.F:48-56
    sRef: tuple                 # load_ref_files.F:80-87
    bathyFile: str
    readBinaryPrec: int         # data PARM01 readBinaryPrec (set_defaults.F:351 default precFloat32)
    diffKrFile: str
    diffKrNrS: tuple            # ini_parms.F:571-575
    GM_background_K: float      # data.gmredi (gmredi_readparms.F:98 default 0.)
    GM_isopycK: float           # data.gmredi (gmredi_readparms.F:97,185 default GM_background_K)
    GM_background_K3dFile: str  # data.gmredi (gmredi_readparms.F:114 default ' ')
    GM_isopycK3dFile: str       # data.gmredi (gmredi_readparms.F:115 default ' ')
    viscA4Dfile: str            # data PARM05 (set_defaults.F:370 default ' ')
    viscA4Zfile: str            # data PARM05 (set_defaults.F:371 default ' ')
    exch2_dimsFacets: tuple     # data.exch2 W2_EXCH2_PARM01 dimsFacets
    Nr: int

    @classmethod
    def from_namelists(cls, nml, layout=None):
        L = layout or Layout()
        Nr = L.Nr

        def get(f, g, k, default):
            return nml.get(f, g, k, default=default)

        def need(cond, what):
            if not cond:
                raise NotImplementedError(f"grid set-up: {what} is not ported (V4r4 flux-forced branch only)")

        # --- coordinate system (ini_parms.F:407-427, set_defaults.F:175) ---
        need(str(get("data", "parm01", "buoyancyRelation", "OCEANIC")).strip().upper() == "OCEANIC",
             "buoyancyRelation != 'OCEANIC'")
        # --- horizontal grid (set_defaults.F:85-88, 72) ---
        need(get("data", "parm04", "usingCurvilinearGrid", False) is True, "usingCurvilinearGrid=.FALSE.")
        for flag in ("usingCartesianGrid", "usingSphericalPolarGrid", "usingCylindricalGrid"):
            need(get("data", "parm04", flag, False) is False, flag)
        need(_blank(get("data", "parm04", "horizGridFile", " ")), "horizGridFile (angles read from file)")
        need(get("data", "parm04", "deepAtmosphere", False) is False, "deepAtmosphere")   # set_defaults.F:73
        need(get("data", "parm04", "selectSigmaCoord", 0) == 0, "selectSigmaCoord != 0")  # set_defaults.F:51
        need(_blank(get("data", "parm04", "delRFile", " ")) and _blank(get("data", "parm04", "delRcFile", " ")),
             "delRFile / delRcFile")
        need(not nml.has("data", "parm04", "delRc"), "delRc (setCenterDr)")               # ini_parms.F:1276-1294
        need(not nml.has("data", "parm04", "delZ") and not nml.has("data", "parm04", "delP"), "delZ/delP")
        delR = get("data", "parm04", "delR", None)
        need(delR is not None and len(delR) == Nr, "delR not given for all Nr levels")
        need(not nml.has("data", "parm04", "Ro_SeaLevel"), "Ro_SeaLevel")                  # ini_parms.F:1199-1215
        # --- planet radius (ini_parms.F:1182-1192) ---
        rSphere = get("data", "parm04", "rSphere", UNSET_RL)                               # set_defaults.F:86
        radius_fromHorizGrid = get("data", "parm04", "radius_fromHorizGrid", UNSET_RL)     # set_defaults.F:89
        if rSphere == UNSET_RL:
            rSphere = radius_fromHorizGrid if radius_fromHorizGrid != UNSET_RL else 6370.0e3  # ini_parms.F:1185-1187
        if radius_fromHorizGrid == UNSET_RL:
            radius_fromHorizGrid = rSphere                                                 # ini_parms.F:1191
        need(rSphere == radius_fromHorizGrid, "rSphere != radius_fromHorizGrid (ini_curvilinear_grid.F:387 scaling)")
        # --- rotation (ini_parms.F:449-457) ---
        rotationPeriod = get("data", "parm01", "rotationPeriod", 86164.0)                  # set_defaults.F:117
        omega = get("data", "parm01", "omega", UNSET_RL)                                   # set_defaults.F:118
        if omega == UNSET_RL:
            omega = 0.0
            if rotationPeriod != 0.0:
                omega = 2.0 * PI / rotationPeriod                                          # ini_parms.F:452
        selectCoriMap = get("data", "parm01", "selectCoriMap", -1)                         # set_defaults.F:94
        if selectCoriMap == -1:
            selectCoriMap = 2                                                              # ini_parms.F:1256
        need(selectCoriMap == 2, f"selectCoriMap={selectCoriMap}")
        # --- vertical axis origin (ini_parms.F:1216-1217) ---
        top_Pres = get("data", "parm04", "top_Pres", UNSET_RL)
        top_Pres = 0.0 if top_Pres == UNSET_RL else top_Pres
        seaLev_Z = get("data", "parm04", "seaLev_Z", UNSET_RL)
        seaLev_Z = 0.0 if seaLev_Z == UNSET_RL else seaLev_Z
        # --- hFac (ini_parms.F:590-595) ---
        hFacMin = get("data", "parm01", "hFacMin", 1.0)                                    # set_defaults.F:178
        hFacMinDr = get("data", "parm01", "hFacMinDr", UNSET_RL)                           # ini_parms.F:380
        hFacMinDz = get("data", "parm01", "hFacMinDz", UNSET_RL)                           # ini_parms.F:381
        hFacMinDp = get("data", "parm01", "hFacMinDp", UNSET_RL)                           # ini_parms.F:382
        if hFacMinDr == UNSET_RL:
            hFacMinDr = hFacMinDz
        if hFacMinDr == UNSET_RL:
            hFacMinDr = hFacMinDp
        if hFacMinDr == UNSET_RL:
            hFacMinDr = 0.0                                                                # set_defaults.F:179
        # --- reference state (ini_parms.F:445, load_ref_files.F, set_ref_state.F) ---
        gravity = get("data", "parm01", "gravity", 9.81)                                   # set_defaults.F:104
        rhoNil = get("data", "parm01", "rhoNil", 999.8)                                    # set_defaults.F:106
        rhoConst = get("data", "parm01", "rhoConst", UNSET_RL)                             # set_defaults.F:107
        if rhoConst == UNSET_RL:
            rhoConst = rhoNil                                                              # ini_parms.F:445
        need(_blank(get("data", "parm01", "gravityFile", " ")), "gravityFile")             # set_defaults.F:62
        need(_blank(get("data", "parm01", "rhoRefFile", " ")), "rhoRefFile")               # set_defaults.F:61
        need(_blank(get("data", "parm01", "tRefFile", " ")) and _blank(get("data", "parm01", "sRefFile", " ")),
             "tRefFile / sRefFile")
        thetaConst = get("data", "parm01", "thetaConst", UNSET_RL)                         # set_defaults.F:63
        tRef = _ref_profile(get("data", "parm01", "tRef", [UNSET_RL]), Nr,
                            20.0 if thetaConst == UNSET_RL else thetaConst)                # load_ref_files.F:49-56
        sRef = _ref_profile(get("data", "parm01", "sRef", [UNSET_RL]), Nr, 30.0)           # load_ref_files.F:82-87
        # --- depths (ini_depths.F) ---
        bathyFile = get("data", "parm05", "bathyFile", " ")                                # set_defaults.F:357
        need(not _blank(bathyFile), "bathyFile=' ' (flat bottom)")
        need(_blank(get("data", "parm05", "topoFile", " ")), "topoFile")                   # set_defaults.F:358
        need(_blank(get("data", "parm05", "addWwallFile", " ")) and
             _blank(get("data", "parm05", "addSwallFile", " ")), "addWwallFile / addSwallFile")
        readBinaryPrec = get("data", "parm01", "readBinaryPrec", 32)                       # set_defaults.F:351
        need(readBinaryPrec in (32, 64), f"readBinaryPrec={readBinaryPrec}")
        # --- ini_mixing.F: diffKr (ALLOW_3D_DIFFKR, CPP_OPTIONS.h) ---
        diffKrFile = get("data", "parm05", "diffKrFile", " ")                              # set_defaults.F:367
        need(not _blank(diffKrFile), "diffKrFile=' '")
        need(not any(nml.has("data", "parm01", k) for k in ("diffKrNrS", "diffKzS", "diffKpS")),
             "diffKrNrS / diffKzS / diffKpS")
        need(nml.has("data", "parm01", "diffKrS"), "diffKrS unset (diffKrNrS from diffKrNrT)")
        diffKrNrS = (get("data", "parm01", "diffKrS", None),) * Nr                         # ini_parms.F:571-575
        # --- GMRedi (gmredi_readparms.F, gmredi_init_fixed.F; ALLOW_KAPGM/KAPREDI_CONTROL + _3DFILE) ---
        need(get("data.pkg", "packages", "useGMRedi", False) is True, "useGMRedi=.FALSE.")  # packages_boot.F:123
        GM_background_K = get("data.gmredi", "gm_parm01", "GM_background_K", 0.0)          # gmredi_readparms.F:98
        GM_isopycK = get("data.gmredi", "gm_parm01", "GM_isopycK", -999.0)                 # gmredi_readparms.F:97
        if GM_isopycK == -999.0:
            GM_isopycK = GM_background_K                                                   # gmredi_readparms.F:185
        for f in ("GM_iso2dFile", "GM_iso1dFile", "GM_bol2dFile", "GM_bol1dFile"):         # gmredi_readparms.F:110-113
            need(_blank(get("data.gmredi", "gm_parm01", f, " ")), f)
        GM_background_K3dFile = get("data.gmredi", "gm_parm01", "GM_background_K3dFile", " ")
        GM_isopycK3dFile = get("data.gmredi", "gm_parm01", "GM_isopycK3dFile", " ")
        need(not _blank(GM_background_K3dFile) and not _blank(GM_isopycK3dFile), "GM 3-D K files unset")
        # --- mom_init_fixed.F (ALLOW_3D_VISCA4, ALLOW_3D_VISCAH) ---
        need(get("data", "parm01", "momStepping", True) is True, "momStepping=.FALSE.")   # set_defaults.F:189
        viscA4Dfile = get("data", "parm05", "viscA4Dfile", " ")
        viscA4Zfile = get("data", "parm05", "viscA4Zfile", " ")
        need(not _blank(viscA4Dfile) and not _blank(viscA4Zfile), "viscA4Dfile / viscA4Zfile unset")
        need(_blank(get("data", "parm05", "viscAhDfile", " ")) and _blank(get("data", "parm05", "viscAhZfile", " ")),
             "viscAhDfile / viscAhZfile")
        # --- exch2 topology (w2_readparms.F) ---
        g = nml.file("data.exch2").get("w2_exch2_parm01", {})
        need(g.get("predeftopol", [0]) == [0], "preDefTopol != 0")
        need(g.get("w2_mapio", [None]) == [1], "W2_mapIO != 1 (compact global files)")
        need(not any(k.startswith("blanklist") for k in g), "blankList (blank tiles)")
        dims = [v for k, v in g.items() if k.split("(")[0] == "dimsfacets"]
        need(len(dims) == 1, "dimsFacets")
        return cls(delR=tuple(float(x) for x in delR), rSphere=float(rSphere),
                   radius_fromHorizGrid=float(radius_fromHorizGrid), omega=float(omega),
                   selectCoriMap=int(selectCoriMap), seaLev_Z=float(seaLev_Z), top_Pres=float(top_Pres),
                   hFacMin=float(hFacMin), hFacMinDr=float(hFacMinDr), gravity=float(gravity),
                   rhoConst=float(rhoConst), tRef=tRef, sRef=sRef, bathyFile=str(bathyFile).strip(),
                   readBinaryPrec=int(readBinaryPrec), diffKrFile=str(diffKrFile).strip(),
                   diffKrNrS=tuple(float(x) for x in diffKrNrS), GM_background_K=float(GM_background_K),
                   GM_isopycK=float(GM_isopycK), GM_background_K3dFile=str(GM_background_K3dFile).strip(),
                   GM_isopycK3dFile=str(GM_isopycK3dFile).strip(), viscA4Dfile=str(viscA4Dfile).strip(),
                   viscA4Zfile=str(viscA4Zfile).strip(), exch2_dimsFacets=tuple(int(x) for x in dims[0]), Nr=Nr)


def _ref_profile(vals, Nr, tracerDefault):
    """load_ref_files.F:52-56: unset levels take the value of the level above (first: tracerDefault)."""
    vals = list(vals) + [UNSET_RL] * (Nr - len(vals))
    out = []
    for k in range(Nr):
        v = float(vals[k])
        if v == UNSET_RL:
            v = tracerDefault
        tracerDefault = v
        out.append(v)
    return tuple(out)


# ---------------------------------------------------------------------------------------------------------------------
# exch2 tile map (w2_set_map_tiles.F:157-190)
# ---------------------------------------------------------------------------------------------------------------------
def exch2_tiles(p, L):
    """[(facet, tBasex, tBasey, dNx, dNy)] for tile ids 1..nTiles, numbered facet by facet, x fastest."""
    d = p.exch2_dimsFacets
    tiles = []
    for j in range(len(d) // 2):                         # w2_set_map_tiles.F:161
        fNx, fNy = d[2 * j], d[2 * j + 1]                # :162-163
        if fNx % L.sNx or fNy % L.sNy:
            raise ValueError(f"facet {j + 1} ({fNx}x{fNy}) is not a multiple of the tile size")
        for ty in range(fNy // L.sNy):                   # :170
            for tx in range(fNx // L.sNx):               # :171
                tiles.append((j + 1, tx * L.sNx, ty * L.sNy, fNx, fNy))   # :183-184 tBasex=(tx-1)*tNx
    if len(tiles) != L.nTiles:
        raise ValueError(f"data.exch2 gives {len(tiles)} tiles, layout has {L.nTiles}")
    return tiles


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def _libm(fn, x):
    """Elementwise glibc libm function (the one gfortran's SIN/COS call), on a float64 array."""
    x = np.asarray(x, np.float64)
    return np.fromiter((fn(v) for v in x.ravel()), np.float64, x.size).reshape(x.shape)


_LIBM = None


def _sincos(x):
    """glibc sincos(x) elementwise -> (sin, cos). gfortran -O3 (GCC's cse_sincos pass) turns SIN(x) and COS(x) of the
    same argument in one loop body into one sincos() call; glibc 2.28's sincos cos-part differs from cos() by up to
    2 ulp (measured on fCoriCos: 70 points), its sin-part equals sin(). Used where the Fortran has that pattern."""
    global _LIBM
    import ctypes
    import ctypes.util
    if _LIBM is None:
        lib = ctypes.CDLL(ctypes.util.find_library("m"))
        lib.sincos.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        lib.sincos.restype = None
        _LIBM = lib
    x = np.asarray(x, np.float64)
    s_out = np.empty(x.size)
    c_out = np.empty(x.size)
    s, c = ctypes.c_double(), ctypes.c_double()
    ps, pc = ctypes.byref(s), ctypes.byref(c)
    for n, v in enumerate(x.ravel()):
        _LIBM.sincos(v, ps, pc)
        s_out[n] = s.value
        c_out[n] = c.value
    return s_out.reshape(x.shape), c_out.reshape(x.shape)


class _Ex:
    """Exchanger calls on host arrays (numpy in, numpy float64 out)."""

    def __init__(self, ex):
        self.ex = ex

    def s(self, a, kind):
        return np.array(self.ex.scalar(a, kind), np.float64)

    def v(self, u, w, kind):
        a, b = self.ex.vector(u, w, kind)
        return np.array(a, np.float64), np.array(b, np.float64)


def _read_global(path, L, nz, prec):
    """READ_REC_XY_RS / READ_FLD_XYZ_RL of a compact global file: interior of every tile (halos untouched)."""
    a = read_bin(path, nz=None if nz == 1 else nz, prec=prec)       # (1, [nz,] 1170, 90)
    t = compact_to_tiles(a[0])                                       # ([nz,] 13, 90, 90)
    return t if nz == 1 else np.moveaxis(t, 0, 1)                    # [T, (nz,) sNy, sNx]


def _set_interior(a, vals, L):
    a[..., L.js(1, L.sNy), L.is_(1, L.sNx)] = vals


# ---------------------------------------------------------------------------------------------------------------------
# INI_VERTICAL_GRID (ini_vertical_grid.F), LOAD_REF_FILES, SET_REF_STATE
# ---------------------------------------------------------------------------------------------------------------------
def ini_vertical_grid(p):
    Nr = p.Nr
    delR = np.array(p.delR)
    rkSign = -1.0                                        # ini_vertical_grid.F:56
    if np.any(delR <= 0.0):                              # :80-90
        raise ValueError("delR must be > 0")
    drF = delR.copy()                                    # :76-78 (setInterFDr)
    drC = np.zeros(Nr + 1)
    drC[0] = 0.5 * delR[0]                               # :126 (not setCenterDr)
    for k in range(2, Nr + 1):                           # :127-129
        drC[k - 1] = 0.5 * (delR[k - 2] + delR[k - 1])
    drC[Nr] = 0.5 * delR[Nr - 1]                         # :130
    rF = np.zeros(Nr + 1)
    rC = np.zeros(Nr)
    rF[0] = p.seaLev_Z                                   # :139-142 (rF(1)=UNSET_RS, usingZCoords, fluidIsWater)
    for k in range(1, Nr + 1):                           # :144-146
        rF[k] = rF[k - 1] + rkSign * drF[k - 1]
    rC[0] = rF[0] + rkSign * drC[0]                      # :147
    for k in range(2, Nr + 1):                           # :148-150
        rC[k - 1] = rC[k - 2] + rkSign * drC[k - 1]
    for k in range(Nr):                                  # :174-195 check
        tmp = (rF[k] - rC[k]) / (rC[k] - rF[k + 1]) if (rC[k] - rF[k + 1]) != 0.0 else 0.0
        if tmp < 1.0 / 100.0 or tmp > 100.0:
            raise ValueError(f"INI_VERTICAL_GRID: invalid relative position at k={k + 1}")
    recip_drC = 1.0 / drC                                # :198-200
    recip_drF = 1.0 / drF                                # :201-203
    return dict(drF=drF, drC=drC, rF=rF, rC=rC, recip_drF=recip_drF, recip_drC=recip_drC,
                rkSign=rkSign, gravitySign=-1.0)         # :57 gravitySign (not usingPCoords)


def set_ref_state(p, v):
    """load_ref_files.F (tRef, sRef, gravFac=1) and set_ref_state.F:53-134 for OCEANIC, gravityFile=' '."""
    Nr = p.Nr
    recip_rhoConst = 1.0 / p.rhoConst                    # ini_parms.F:640
    phiRef = np.zeros(2 * Nr + 1)                        # set_ref_state.F:54-56
    rF, rC = v["rF"], v["rC"]
    phiRef[0] = p.top_Pres * recip_rhoConst              # :86
    for k in range(1, Nr + 1):                           # :89-101
        phiRef[2 * k - 1] = phiRef[0] + ((rC[k - 1] - rF[0]) * p.gravity) * v["gravitySign"]
        phiRef[2 * k] = phiRef[0] + ((rF[k] - rF[0]) * p.gravity) * v["gravitySign"]
    # dBdrRef (:166-207) needs FIND_RHO_SCALAR (JMD95Z EOS): not ported here
    return dict(tRef=np.array(p.tRef), sRef=np.array(p.sRef),
                rVel2wUnit=np.ones(Nr + 1), wUnit2rVel=np.ones(Nr + 1),        # :68-71
                rhoFacC=np.ones(Nr), rhoFacF=np.ones(Nr + 1),                    # :74-81 (rhoRefFile=' ')
                phiRef=phiRef)


# ---------------------------------------------------------------------------------------------------------------------
# INI_GRID / INI_CURVILINEAR_GRID / CALC_GRID_ANGLES
# ---------------------------------------------------------------------------------------------------------------------
_MITGRID = ["xC", "yC", "dxF", "dyF", "rA", "xG", "yG", "dxV", "dyU", "rAz", "dxC", "dyC", "rAw", "rAs", "dxG", "dyG"]


def read_mitgrid(p, L, grid_dir):
    """ini_grid.F:69-114 initialisation + ini_curvilinear_grid.F:260-363 file reads (before any exchange)."""
    f = {}
    # ini_grid.F:69-114: every horizontal grid array = 0 (angleCosC = 1, u2zonDir = 1)
    for n in _MITGRID:
        f[n] = np.zeros(L.shape2d)
    f["angleCosC"] = np.ones(L.shape2d)
    f["angleSinC"] = np.zeros(L.shape2d)
    f["u2zonDir"] = np.ones(L.shape2d)
    f["v2zonDir"] = np.zeros(L.shape2d)
    f["tanPhiAtU"] = np.zeros(L.shape2d)                 # ini_grid.F:101-102, not set by the curvilinear grid
    f["tanPhiAtV"] = np.zeros(L.shape2d)
    # ini_curvilinear_grid.F:260-363: MDS_FACEF_READ_RS(fName, precFloat64, irec, fld, bi, bj), irec 1..16,
    # mdsio_facef_read.F:80-106 (EXCH2): record = row of dNx+1 values; tile rows 1..sNy+1 = facet rows
    # tBasey+1..tBasey+sNy+1, columns 1..sNx+1 = facet columns tBasex+1..tBasex+sNx+1
    cache = {}
    for t, (face, tbx, tby, dNx, dNy) in enumerate(exch2_tiles(p, L)):
        if face not in cache:
            fname = Path(grid_dir) / f"tile{face:03d}.mitgrid"          # ini_curvilinear_grid.F:277
            raw = np.fromfile(fname, dtype=">f8")
            nrec = raw.size // ((dNy + 1) * (dNx + 1))
            if nrec < len(_MITGRID) or raw.size != nrec * (dNy + 1) * (dNx + 1):
                raise ValueError(f"{fname}: {raw.size} values, expected >= 16 records of {dNy + 1}x{dNx + 1}")
            cache[face] = raw.reshape(nrec, dNy + 1, dNx + 1).astype(np.float64)
        F = cache[face]
        for irec, n in enumerate(_MITGRID):
            f[n][t, L.js(1, L.sNy + 1), L.is_(1, L.sNx + 1)] = F[irec, tby:tby + L.sNy + 1, tbx:tbx + L.sNx + 1]
    return f


def ini_curvilinear_grid(p, L, grid_dir, X):
    f = read_mitgrid(p, L, grid_dir)
    # ini_curvilinear_grid.F:371-381
    f["xC"] = X.s(f["xC"], "T")                                            # EXCH_XY_RS(xC)
    f["yC"] = X.s(f["yC"], "T")                                            # EXCH_XY_RS(yC)
    f["dxF"], f["dyF"] = X.v(f["dxF"], f["dyF"], "An")                     # EXCH_UV_AGRID_3D_RS(dxF,dyF,.FALSE.)
    f["rA"] = X.s(f["rA"], "T")                                            # EXCH_XY_RS(rA)
    f["xG"] = X.s(f["xG"], "Z")                                            # EXCH_Z_3D_RS(xG)
    f["yG"] = X.s(f["yG"], "Z")                                            # EXCH_Z_3D_RS(yG)
    f["dxV"], f["dyU"] = X.v(f["dxV"], f["dyU"], "Bn")                     # EXCH_UV_BGRID_3D_RS(dxV,dyU,.FALSE.)
    f["rAz"] = X.s(f["rAz"], "Z")                                          # EXCH_Z_3D_RS(rAz)
    f["dxC"], f["dyC"] = X.v(f["dxC"], f["dyC"], "UVn")                    # EXCH_UV_XY_RS(dxC,dyC,.FALSE.)
    f["rAw"], f["rAs"] = X.v(f["rAw"], f["rAs"], "UVn")                    # EXCH_UV_XY_RS(rAw,rAs,.FALSE.)
    f["dyG"], f["dxG"] = X.v(f["dyG"], f["dxG"], "UVn")                    # EXCH_UV_XY_RS(dyG,dxG,.FALSE.)
    # :387-410 rSphere scaling: rSphere == radius_fromHorizGrid (checked in GridParams) -> skipped
    # :414 CALC_GRID_ANGLES(anglesAreSet=.FALSE.): horizGridFile=' ' (:340-351)
    calc_grid_angles(p, L, f, skip_calc_angle_c=False)
    # :417 EXCH_UV_AGRID_3D_RS(angleSinC, angleCosC, .TRUE.)
    f["angleSinC"], f["angleCosC"] = X.v(f["angleSinC"], f["angleCosC"], "As")
    return f


def calc_grid_angles(p, L, f, skip_calc_angle_c):
    """calc_grid_angles.F:49-133 (all tiles at once)."""
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    yG, dxG, dyG = f["yG"], f["dxG"], f["dyG"]
    rSphere = p.rSphere
    uPseudo = np.full(L.shape2d, np.nan)                 # local, undefined outside its loop range
    vPseudo = np.full(L.shape2d, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        # :54-65
        J, Jp, I = L.js(1 - OLy, sNy + OLy - 1), L.js(2 - OLy, sNy + OLy), L.is_(1 - OLx, sNx + OLx)
        d = dyG[:, J, I]
        uPseudo[:, J, I] = np.where(d > 0.0, -(((yG[:, J, I] - yG[:, Jp, I]) * DEG2RAD) / d), 0.0)
        f["u2zonDir"][:, J, I] = rSphere * uPseudo[:, J, I]
        # :66-77
        J, I, Ip = L.js(1 - OLy, sNy + OLy), L.is_(1 - OLx, sNx + OLx - 1), L.is_(2 - OLx, sNx + OLx)
        d = dxG[:, J, I]
        vPseudo[:, J, I] = np.where(d > 0.0, +(((yG[:, J, I] - yG[:, J, Ip]) * DEG2RAD) / d), 0.0)
        f["v2zonDir"][:, J, I] = rSphere * vPseudo[:, J, I]
        # :78-89
        if not skip_calc_angle_c:
            J, Jp = L.js(1 - OLy, sNy + OLy - 1), L.js(2 - OLy, sNy + OLy)
            I, Ip = L.is_(1 - OLx, sNx + OLx - 1), L.is_(2 - OLx, sNx + OLx)
            uC = 0.5 * (uPseudo[:, J, I] + uPseudo[:, J, Ip])
            vC = 0.5 * (vPseudo[:, J, I] + vPseudo[:, Jp, I])
            uNorm = np.sqrt(uC * uC + vC * vC)
            uNorm = np.where(uNorm != 0.0, 1.0 / uNorm, uNorm)
            f["angleCosC"][:, J, I] = uC * uNorm
            f["angleSinC"][:, J, I] = -(vC * uNorm)
        # :94-111
        J, Jp, I = L.js(1 - OLy, sNy + OLy - 1), L.js(2 - OLy, sNy + OLy), L.is_(1 - OLx, sNx + OLx)
        tmpVal = f["rAw"][:, J, I] * _libm(math.cos, (DEG2RAD * (yG[:, J, I] + yG[:, Jp, I])) * HALF_RL)
        num = ((rSphere * (_libm(math.sin, yG[:, Jp, I] * DEG2RAD) - _libm(math.sin, yG[:, J, I] * DEG2RAD)))
               * f["dxC"][:, J, I])
        f["u2zonDir"][:, J, I] = np.where(tmpVal > 0.0, num / tmpVal, 1.0)
        # :112-129
        J, I, Ip = L.js(1 - OLy, sNy + OLy), L.is_(1 - OLx, sNx + OLx - 1), L.is_(2 - OLx, sNx + OLx)
        tmpVal = f["rAs"][:, J, I] * _libm(math.cos, (DEG2RAD * (yG[:, J, I] + yG[:, J, Ip])) * HALF_RL)
        num = (((-rSphere) * (_libm(math.sin, yG[:, J, Ip] * DEG2RAD) - _libm(math.sin, yG[:, J, I] * DEG2RAD)))
               * f["dyC"][:, J, I])
        f["v2zonDir"][:, J, I] = np.where(tmpVal > 0.0, num / tmpVal, 0.0)


def ini_grid_recip(f):
    """ini_grid.F:152-183 (recip_* initialised to 0 at :89-100; set where the length/area is non-zero)."""
    with np.errstate(divide="ignore"):
        for src, dst in (("dxG", "recip_dxG"), ("dyG", "recip_dyG"), ("dxC", "recip_dxC"), ("dyC", "recip_dyC"),
                         ("dxF", "recip_dxF"), ("dyF", "recip_dyF"), ("dxV", "recip_dxV"), ("dyU", "recip_dyU"),
                         ("rA", "recip_rA"), ("rAs", "recip_rAs"), ("rAw", "recip_rAw"), ("rAz", "recip_rAz")):
            a = f[src]
            f[dst] = np.where(a != 0.0, 1.0 / np.where(a != 0.0, a, 1.0), 0.0)


# ---------------------------------------------------------------------------------------------------------------------
# INI_DEPTHS, INI_MASKS_ETC, ADD_WALLS2MASKS
# ---------------------------------------------------------------------------------------------------------------------
def ini_depths(p, L, rundir, v, X):
    R_low = np.zeros(L.shape2d)                          # ini_depths.F:75-85
    Ro_surf = np.zeros(L.shape2d)
    # :122 READ_REC_XY_RS(bathyFile, R_low, 1, 0): interior, readBinaryPrec
    _set_interior(R_low, _read_global(Path(rundir) / p.bathyFile, L, 1, p.readBinaryPrec), L)
    R_low = X.s(R_low, "T")                              # :138 _EXCH_XY_RS(R_low)
    _set_interior(Ro_surf, v["rF"][0], L)                # :158-169 topoFile=' ': Ro_surf = rF(1)
    Ro_surf = X.s(Ro_surf, "T")                          # :220 _EXCH_XY_RS(Ro_surf)
    # :250-270 not usingSphericalPolarGrid: nothing; :297 EXCH2_CHECK_DEPTHS: check only
    return R_low, Ro_surf


def ini_masks_etc(p, L, v, f, R_low, Ro_surf, X):
    """ini_masks_etc.F:56-523 (selectSigmaCoord=0, useShelfIce=F, useMin4hFacEdges=T) + add_walls2masks.F."""
    Nr, sNx, sNy, OLx, OLy = L.Nr, L.sNx, L.sNy, L.OLx, L.OLy
    rF, drF, recip_drF = v["rF"], v["drF"], v["recip_drF"]
    T, ny, nx = L.shape2d
    R_low = R_low.copy()
    Ro_surf = Ro_surf.copy()
    rLowW, rSurfW, rLowS, rSurfS = (np.zeros(L.shape2d) for _ in range(4))
    # :77-110 first estimate of rLow/rSurf at W and S edges
    i0, j0 = L.ii(1 - OLx), L.jj(1 - OLy)
    rLowW[:, :, i0] = rF[0]
    rSurfW[:, :, i0] = rF[0]
    rLowS[:, j0, :] = rF[0]
    rSurfS[:, j0, :] = rF[0]

    def edges_w():
        I, Im = L.is_(2 - OLx, sNx + OLx), L.is_(1 - OLx, sNx + OLx - 1)
        rLowW[:, :, I] = np.maximum(R_low[:, :, Im], R_low[:, :, I])
        rSurfW[:, :, I] = np.minimum(Ro_surf[:, :, Im], Ro_surf[:, :, I])
        rSurfW[:, :, I] = np.maximum(rSurfW[:, :, I], rLowW[:, :, I])

    def edges_s():
        J, Jm = L.js(2 - OLy, sNy + OLy), L.js(1 - OLy, sNy + OLy - 1)
        rLowS[:, J, :] = np.maximum(R_low[:, Jm, :], R_low[:, J, :])
        rSurfS[:, J, :] = np.minimum(Ro_surf[:, Jm, :], Ro_surf[:, J, :])
        rSurfS[:, J, :] = np.maximum(rSurfS[:, J, :], rLowS[:, J, :])

    edges_w()                                            # :89-98
    edges_s()                                            # :99-108

    def hfac_min_rule(tmp, hFacMnSz):                    # :126-134 / :171-179
        return np.where(tmp < hFacMnSz, np.where(tmp < hFacMnSz * HALF_RL, 0.0, hFacMnSz), tmp)

    hFacC = np.zeros((T, Nr, ny, nx))
    # :117-137 over-estimate from the lower boundary
    for k in range(Nr):
        hFacMnSz = max(p.hFacMin, min(p.hFacMinDr * recip_drF[k], ONE_RL))
        tmp = (rF[k] - R_low) * recip_drF[k]
        tmp = np.minimum(np.maximum(tmp, ZERO_RL), ONE_RL)
        hFacC[:, k] = hfac_min_rule(tmp, hFacMnSz)
    # :140-156 R_low from hFacC (sequential sum over k)
    tmpVar1 = np.zeros(L.shape2d)
    for k in range(Nr):
        tmpVar1 = tmpVar1 + drF[k] * hFacC[:, k]
    R_low = rF[0] - tmpVar1
    # :160-182 remove the part above the reference surface
    for k in range(Nr):
        hFacMnSz = max(p.hFacMin, min(p.hFacMinDr * recip_drF[k], ONE_RL))
        tmp = (rF[k] - Ro_surf) * recip_drF[k]
        tmp = hFacC[:, k] - np.maximum(tmp, ZERO_RL)
        tmp = np.maximum(tmp, ZERO_RL)
        hFacC[:, k] = hfac_min_rule(tmp, hFacMnSz)
    # :187-217 Ro_surf, kSurfC, kLowC, maskInC
    tmpVar2 = np.zeros(L.shape2d)
    kSurfC = np.full(L.shape2d, Nr + 1, np.int32)
    kLowC = np.zeros(L.shape2d, np.int32)
    for k in range(Nr):
        tmpVar2 = tmpVar2 + drF[k] * hFacC[:, k]
        kLowC = np.where(hFacC[:, k] != 0.0, k + 1, kLowC)
    for k in range(Nr - 1, -1, -1):
        kSurfC = np.where(hFacC[:, k] != 0.0, k + 1, kSurfC)
    Ro_surf = R_low + tmpVar2
    maskInC = np.where(kSurfC <= Nr, 1.0, 0.0)
    # :238-250 recip_Rcol
    tmpVar1 = Ro_surf - R_low
    with np.errstate(divide="ignore"):
        recip_Rcol = np.where(tmpVar1 <= ZERO_RL, 0.0, 1.0 / np.where(tmpVar1 <= ZERO_RL, 1.0, tmpVar1))
    # :252-273 method 1: hFacW,S = min of adjacent hFacC
    hFacW = np.zeros_like(hFacC)
    hFacS = np.zeros_like(hFacC)
    I, Im = L.is_(2 - OLx, sNx + OLx), L.is_(1 - OLx, sNx + OLx - 1)
    hFacW[..., I] = np.minimum(hFacC[..., I], hFacC[..., Im])
    J, Jm = L.js(2 - OLy, sNy + OLy), L.js(1 - OLy, sNy + OLy - 1)
    hFacS[..., J, :] = np.minimum(hFacC[..., J, :], hFacC[..., Jm, :])
    # :340-359 update rLow/rSurf at W and S edges with the adjusted R_low, Ro_surf
    edges_w()
    edges_s()
    # :426-428 exchanges
    hFacW, hFacS = X.v(hFacW, hFacS, "UVn")              # EXCH_UV_XYZ_RS(hFacW, hFacS, .FALSE.)
    rSurfW, rSurfS = X.v(rSurfW, rSurfS, "UVn")          # EXCH_UV_XY_RS(rSurfW, rSurfS, .FALSE.)
    rLowW, rLowS = X.v(rLowW, rLowS, "UVn")              # EXCH_UV_XY_RS(rLowW, rLowS, .FALSE.)
    # :433 ADD_WALLS2MASKS (add_walls2masks.F:81-102; no wall files)
    wW = f["dyG"] == 0.0
    wS = f["dxG"] == 0.0
    hFacW = np.where(wW[:, None], 0.0, hFacW)
    rLowW = np.where(wW, rF[0], rLowW)
    rSurfW = np.where(wW, rF[0], rSurfW)
    hFacS = np.where(wS[:, None], 0.0, hFacS)
    rLowS = np.where(wS, rF[0], rLowS)
    rSurfS = np.where(wS, rF[0], rSurfS)
    # :436-453 kSurfW/S, maskInW/S
    kSurfW = np.full(L.shape2d, Nr + 1, np.int32)
    kSurfS = np.full(L.shape2d, Nr + 1, np.int32)
    for k in range(Nr - 1, -1, -1):
        kSurfW = np.where(hFacW[:, k] != 0.0, k + 1, kSurfW)
        kSurfS = np.where(hFacS[:, k] != 0.0, k + 1, kSurfS)
    maskInW = np.where(kSurfW <= Nr, 1.0, 0.0)
    maskInS = np.where(kSurfS <= Nr, 1.0, 0.0)
    # :478-506 masks (recip_hFac* are time-dependent under z*: not part of the static Grid)
    maskC = np.where(hFacC != 0.0, 1.0, 0.0)
    maskW = np.where(hFacW != 0.0, 1.0, 0.0)
    maskS = np.where(hFacS != 0.0, 1.0, 0.0)
    # :511-519 NONLIN_FRSURF: h0Fac = _hFac (plain hFac: no ALLOW_DEPTH_CONTROL, HFACC_MACROS.h:39-41)
    return dict(R_low=R_low, Ro_surf=Ro_surf, rLowW=rLowW, rLowS=rLowS, rSurfW=rSurfW, rSurfS=rSurfS,
                recip_Rcol=recip_Rcol, maskInC=maskInC, maskInW=maskInW, maskInS=maskInS,
                maskC=maskC, maskW=maskW, maskS=maskS, h0FacC=hFacC.copy(), h0FacW=hFacW.copy(),
                h0FacS=hFacS.copy(), kSurfC=kSurfC, kSurfW=kSurfW, kSurfS=kSurfS, kLowC=kLowC)


# ---------------------------------------------------------------------------------------------------------------------
# INI_CORI, MOM_INIT_FIXED, GMREDI_INIT_FIXED, INI_MIXING
# ---------------------------------------------------------------------------------------------------------------------
def ini_cori(p, f):
    """ini_cori.F:87-103 (selectCoriMap=2, spherical): no exchange (only selectCoriMap=3, :181-183).
    sin/cos(_yC*deg2rad) of :95 and :99 share one argument -> one sincos() call in the gfortran binary (_sincos)."""
    two_omega = 2.0 * p.omega                                          # :95 2. _d 0*omega*sin(...)
    sinC, cosC = _sincos(f["yC"] * DEG2RAD)
    return dict(fCori=two_omega * sinC,                                # :94-95
                fCoriG=two_omega * _libm(math.sin, f["yG"] * DEG2RAD),  # :96-97
                fCoriCos=two_omega * cosC)                             # :98-99


def mom_init_fixed(p, L, rundir, X):
    """mom_init_fixed.F:45-64 (zero), :141-160 (3-D files, readBinaryPrec, exchanges). L2_D etc. not needed."""
    out = {n: np.zeros(L.shape3d) for n in ("viscAhDfld", "viscAhZfld", "viscA4Dfld", "viscA4Zfld")}
    # :142-149 viscAhDfile = viscAhZfile = ' ' (checked in GridParams)
    _set_interior(out["viscA4Dfld"], _read_global(Path(rundir) / p.viscA4Dfile, L, L.Nr, p.readBinaryPrec), L)
    out["viscA4Dfld"] = X.s(out["viscA4Dfld"], "T")      # :154 EXCH_3D_RL(viscA4Dfld, Nr)
    _set_interior(out["viscA4Zfld"], _read_global(Path(rundir) / p.viscA4Zfile, L, L.Nr, p.readBinaryPrec), L)
    out["viscA4Zfld"] = X.s(out["viscA4Zfld"], "Z")      # :158 EXCH_Z_3D_RL(viscA4Zfld, Nr)
    return out


def ini_mixing(p, L, rundir, X):
    """ini_mixing.F:52-115 with ALLOW_3D_DIFFKR, ALLOW_CTRL+ALLOW_GMREDI, ALLOW_KAPGM/KAPREDI_CONTROL+_3DFILE.
    GM_bolFac2d = GM_isoFac2d = 1, GM_bolFac1d = GM_isoFac1d = 1 (gmredi_init_fixed.F:46-73, no factor files)."""
    one2d, one1d = 1.0, 1.0
    diffKr = np.broadcast_to(np.array(p.diffKrNrS)[None, :, None, None], L.shape3d).copy()   # :58
    kapGM = np.full(L.shape3d, (p.GM_background_K * one2d) * one1d)                          # :62-63
    kapRedi = np.full(L.shape3d, (p.GM_isopycK * one2d) * one1d)                             # :66-67
    _set_interior(diffKr, _read_global(Path(rundir) / p.diffKrFile, L, L.Nr, p.readBinaryPrec), L)   # :93
    diffKr = X.s(diffKr, "T")                                                                # :94 _EXCH_XYZ_RL
    _set_interior(kapGM, _read_global(Path(rundir) / p.GM_background_K3dFile, L, L.Nr, p.readBinaryPrec), L)
    kapGM = X.s(kapGM, "T")                                                                  # :104
    _set_interior(kapRedi, _read_global(Path(rundir) / p.GM_isopycK3dFile, L, L.Nr, p.readBinaryPrec), L)
    kapRedi = X.s(kapRedi, "T")                                                              # :113
    return dict(diffKr=diffKr, kapGM=kapGM, kapRedi=kapRedi)


# ---------------------------------------------------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------------------------------------------------
def grid_from_files(rundir, grid_dir=None, exchanger=None, layout=None, params=None):
    """Grid of the run in `rundir` (namelists + input files), tileNNN.mitgrid from `grid_dir` (default: rundir).

    Returns a `Grid` with every field of geometry.G2D, G3D, R3D and the vertical rows (VROWS + phiRef) except
    dBdrRef (EOS-dependent, not ported here), plus the integer level indices kSurfC/W/S, kLowC. All arrays numpy.
    `params` (a GridParams) overrides the namelists (used by negative controls)."""
    from mitgcm_jax.parallel.exchange import default_exchanger

    L = layout or Layout()
    rundir = Path(rundir)
    grid_dir = Path(grid_dir) if grid_dir is not None else rundir
    p = params or GridParams.from_namelists(RunNamelists(rundir), L)
    if p.Nr != L.Nr:
        raise ValueError("GridParams.Nr does not match the layout")
    X = _Ex(exchanger or default_exchanger(L))
    v = ini_vertical_grid(p)                                          # ini_grid.F:63
    f = ini_curvilinear_grid(p, L, grid_dir, X)                       # ini_grid.F:137-138
    ini_grid_recip(f)                                                 # ini_grid.F:152-183
    ref = set_ref_state(p, v)                                         # initialise_fixed.F:166-178
    R_low, Ro_surf = ini_depths(p, L, rundir, v, X)                   # initialise_fixed.F:193
    m = ini_masks_etc(p, L, v, f, R_low, Ro_surf, X)                  # initialise_fixed.F:204
    visc = mom_init_fixed(p, L, rundir, X)                            # packages_init_fixed.F:190
    cori = ini_cori(p, f)                                             # initialise_fixed.F:239
    mix = ini_mixing(p, L, rundir, X)                                 # initialise_varia.F:192
    out = {}
    out.update(f)
    out.update(m)
    out.update(visc)
    out.update(cori)
    out.update(mix)
    out.update({k: v[k] for k in ("drF", "drC", "rC", "rF", "recip_drF", "recip_drC")})
    out.update(ref)
    return Grid(out, L)
