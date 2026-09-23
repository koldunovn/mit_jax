"""pkg/ggl90: GGL90_CALC as the ECCO v4r4 flux-forced build runs it (plan Task 12).

Literal port of `MITgcm_c66g/pkg/ggl90/ggl90_calc.F` (no override in the V4r4 trees) with the build's
`flux-forced/code/GGL90_OPTIONS.h`: ALLOW_GGL90_SMOOTH defined, ALLOW_GGL90_HORIZDIFF and ALLOW_GGL90_IDEMIX
undefined; CPP_OPTIONS.h: SOLVE_DIAGONAL_KINNER defined, SOLVE_DIAGONAL_LOWMEMORY undefined (so
`model/src/solve_tridiagonal.F:147-209` is the tridiagonal solver). Parameters: data.ggl90 (GGL90_PARM01), data
(PARM01/PARM03), eedata, data.exch2 of the run directory; defaults cited from ggl90_readparms.F / set_defaults.F /
ini_parms.F.

Inputs of GGL90_CALC(bi, bj, sigmaR, ...) (all [tile, (k,) j, i] with halos, the values the Fortran reads):
    GGL90TKE          GGL90.h, TKE of the previous step (the value on entry; S00_begin in the dumps)
    uVel, vVel        DYNVARS.h, velocity at the start of the step (S00_begin / S01_update_rstar_F)
    sigmaR            argument, vertical density gradient from GRAD_SIGMA (do_oceanic_phys.F k loop; P02 T:sigmaR)
    surfaceForcingU/V FFIELDS.h, from EXTERNAL_FORCING_SURF (P01_external_forcing_surf)
    recip_hFacC       GRID.h, r*-updated (S01_update_rstar_F)
    geometry (Grid)   maskC, Ro_surf, R_low, drF, recip_drF, recip_drC (static); kLowC derived from maskC
                      (ini_masks_etc.F:192-200 / :481-488), mskCor from the exch2 facet layout (ggl90_init_fixed.F)
GGL90_CALC reads no equation of state: N^2 comes from the caller's sigmaR (ggl90_calc.F:218-219).
hFacC is read only to build recip_hFacI (ggl90_calc.F:151-166), which only IDEMIX / HORIZDIFF code uses (both
undefined in V4r4), so it has no effect on any output and is not computed here.

Outputs: GGL90TKE (k=1..Nr on i,j = 2-OLx..sNx+OLx-1; the outermost halo ring keeps its input value),
GGL90viscArU (k=2..Nr, i=1..sNx+1, j=1..sNy), GGL90viscArV (k=2..Nr, i=1..sNx, j=1..sNy+1), GGL90diffKr (k=2..Nr,
i=1..sNx, j=1..sNy). All other points of the three coefficient arrays are 0: with ALLOW_AUTODIFF (V4r4) the caller
zeroes them in the tile loop right before GGL90_CALC (flux-forced do_oceanic_phys.F:661-667).

How the outputs are used (callers, not ported here):
    calc_viscosity.F:97-104 -> GGL90_CALC_VISC (ggl90_calc_visc.F): KappaRU += GGL90viscArU - viscArNr(k)
                               (no mask); KappaRV += maskS * (GGL90viscArV - viscArNr(k)).
    calc_3d_diffusivity.F:229-235 -> GGL90_CALC_DIFF (ggl90_calc_diff.F): KappaRx += GGL90diffKr - diffKrNrS(k).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.parallel.tiles import tile_index, tile_rows
from mitgcm_jax.params_io import params_pytree

# GGL90.h:57-60 (PARAMETER)
SQRTTWO = 1.41421356237310  # GGL90.h:58  SQRTTWO = 1.41421356237310D0 (not the exact sqrt(2) double)
GGL90eps = 2.23e-16  # GGL90.h:60  GGL90eps = 2.23D-16
halfRL = 0.5  # EEPARAMS.h:82  halfRL = 0.5 _d 0
# ALLOW_GGL90_SMOOTH weights, set at the top of GGL90_CALC
p4 = 0.25  # ggl90_calc.F:135  p4  = 0.25   _d 0
p8 = 0.125  # ggl90_calc.F:136  p8  = 0.125  _d 0
p16 = 0.0625  # ggl90_calc.F:137  p16 = 0.0625 _d 0

# CPP options of the V4r4 flux-forced build (flux-forced/code/GGL90_OPTIONS.h, CPP_OPTIONS.h)
ALLOW_GGL90_SMOOTH = True  # GGL90_OPTIONS.h:24  #define ALLOW_GGL90_SMOOTH
ALLOW_GGL90_HORIZDIFF = False  # GGL90_OPTIONS.h:20  #undef ALLOW_GGL90_HORIZDIFF
ALLOW_GGL90_IDEMIX = False  # not in the V4r4 GGL90_OPTIONS.h -> undefined
SOLVE_DIAGONAL_KINNER = True  # flux-forced/code/CPP_OPTIONS.h:102  #define SOLVE_DIAGONAL_KINNER
SOLVE_DIAGONAL_LOWMEMORY = False  # flux-forced/code/CPP_OPTIONS.h:100  #undef SOLVE_DIAGONAL_LOWMEMORY


def _grp(nml, fname, group):
    return nml.file(fname).get(group.lower(), {})


def _array_key(nml, fname, group, name):
    """Value list of an array namelist entry written as `name`, `name(1:n)` or `name(1)` (n values from 1)."""
    g = _grp(nml, fname, group)
    for key, v in g.items():
        if key == name.lower():
            return list(v)
        if key.startswith(name.lower() + "("):
            idx = key[len(name) + 1:-1]
            if idx.split(":")[0] != "1":
                raise NotImplementedError(f"{fname}:{group}:{key}: only arrays given from index 1 are ported")
            return list(v)
    return None


@params_pytree  # float fields are traced leaves: pass the params as a jit argument (KERNEL_GUIDE)
@dataclass(frozen=True)
class GGL90Params:
    # data.ggl90 GGL90_PARM01; defaults ggl90_readparms.F:98-116, 203-205
    GGL90ck: float
    GGL90ceps: float
    GGL90alpha: float
    GGL90m2: float
    GGL90TKEmin: float
    GGL90TKEsurfMin: float
    GGL90TKEbottom: float
    GGL90viscMax: float
    GGL90diffMax: float
    GGL90diffTKEh: float
    GGL90mixingLengthMin: float
    mxlMaxFlag: int
    mxlSurfFlag: bool
    GGL90_dirichlet: bool
    calcMeanVertShear: bool
    useIDEMIX: bool
    # PARAMS.h (data PARM01 / PARM03)
    deltaTggl90: float  # ggl90_calc.F:141  deltaTggl90 = dTtracerLev(1)
    gravity: float
    gravitySign: float
    recip_rhoConst: float
    viscArNr: tuple  # [Nr]
    diffKrNrS: tuple  # [Nr]
    # eedata / data.exch2 (mskCor, ggl90_init_fixed.F)
    useCubedSphereExchange: bool
    dimsFacets: tuple  # (n1x, n1y, n2x, n2y, ...)

    @classmethod
    def from_namelists(cls, nml, Nr=50):
        get = nml.get
        f = "data.ggl90"
        grp = "GGL90_PARM01"
        TKEmin = float(get(f, grp, "GGL90TKEmin", default=1.0e-11))  # ggl90_readparms.F:103
        TKEbottom = get(f, grp, "GGL90TKEbottom", default="UNSET")  # ggl90_readparms.F:106 UNSET_RL
        if TKEbottom == "UNSET":
            TKEbottom = TKEmin  # ggl90_readparms.F:203-205
        p = dict(
            GGL90ck=float(get(f, grp, "GGL90ck", default=0.1)),  # ggl90_readparms.F:98
            GGL90ceps=float(get(f, grp, "GGL90ceps", default=0.7)),  # ggl90_readparms.F:99
            GGL90alpha=float(get(f, grp, "GGL90alpha", default=1.0)),  # ggl90_readparms.F:100
            GGL90m2=float(get(f, grp, "GGL90m2", default=3.75)),  # ggl90_readparms.F:102
            GGL90TKEmin=TKEmin,
            GGL90TKEsurfMin=float(get(f, grp, "GGL90TKEsurfMin", default=1.0e-4)),  # ggl90_readparms.F:105
            GGL90TKEbottom=float(TKEbottom),
            GGL90viscMax=float(get(f, grp, "GGL90viscMax", default=1.0e2)),  # ggl90_readparms.F:107
            GGL90diffMax=float(get(f, grp, "GGL90diffMax", default=1.0e2)),  # ggl90_readparms.F:108
            GGL90diffTKEh=float(get(f, grp, "GGL90diffTKEh", default=0.0)),  # ggl90_readparms.F:109
            GGL90mixingLengthMin=float(get(f, grp, "GGL90mixingLengthMin", default=1.0e-8)),  # ggl90_readparms.F:110
            mxlMaxFlag=int(get(f, grp, "mxlMaxFlag", default=0)),  # ggl90_readparms.F:111
            mxlSurfFlag=bool(get(f, grp, "mxlSurfFlag", default=False)),  # ggl90_readparms.F:112
            GGL90_dirichlet=bool(get(f, grp, "GGL90_dirichlet", default=True)),  # ggl90_readparms.F:114
            calcMeanVertShear=bool(get(f, grp, "calcMeanVertShear", default=False)),  # ggl90_readparms.F:115
            useIDEMIX=bool(get(f, grp, "useIDEMIX", default=False)),  # ggl90_readparms.F:116
        )
        # --- PARAMS.h
        d = "data"
        # dTtracerLev(1): ini_parms.F:875-902 (dTtracerLev defaults to 0, set_defaults.F:297; then = deltaTtracer)
        dTlev = _array_key(nml, d, "PARM03", "dTtracerLev")
        deltaTtracer = float(get(d, "PARM03", "deltaTtracer", default=0.0))  # set_defaults.F deltaTtracer = 0
        if dTlev is not None and float(dTlev[0]) != 0.0:
            if deltaTtracer != 0.0 and deltaTtracer != float(dTlev[0]):
                raise ValueError("ini_parms.F:875: deltaTtracer & dTtracerLev(1) not equal")
            dT1 = float(dTlev[0])
        elif deltaTtracer != 0.0:
            dT1 = deltaTtracer  # ini_parms.F:902 dTtracerLev(k) = deltaTtracer
        else:
            raise NotImplementedError("deltaTtracer from deltaT/deltaTClock fallback (ini_parms.F:884-898) not ported")
        p["deltaTggl90"] = dT1
        p["gravity"] = float(get(d, "PARM01", "gravity", default=9.81))  # set_defaults.F:104
        buoy = get(d, "PARM01", "buoyancyRelation", default="OCEANIC")  # set_defaults.F:175
        if buoy != "OCEANIC":
            raise NotImplementedError("GGL90 needs buoyancyRelation='OCEANIC' (ggl90_check.F:34)")
        p["gravitySign"] = -1.0  # ini_vertical_grid.F:57 (usingPCoords false for OCEANIC, ini_parms.F:419)
        rhoNil = float(get(d, "PARM01", "rhoNil", default=999.8))  # set_defaults.F:106
        rhoConst = float(get(d, "PARM01", "rhoConst", default=rhoNil))  # ini_parms.F:445 rhoConst=rhoNil if unset
        if rhoConst <= 0.0:
            raise NotImplementedError("rhoConst <= 0 (ini_parms.F:633-638)")
        p["recip_rhoConst"] = 1.0 / rhoConst  # ini_parms.F:640
        # viscArNr: ini_parms.F:484-509
        vNr = _array_key(nml, d, "PARM01", "viscArNr")
        viscAr = get(d, "PARM01", "viscAr", default="UNSET")
        if vNr is not None:
            if viscAr != "UNSET" or len(vNr) != Nr:
                raise ValueError("ini_parms.F:491-503: viscArNr partial or together with viscAr")
            p["viscArNr"] = tuple(float(x) for x in vNr)
        elif viscAr != "UNSET":
            p["viscArNr"] = (float(viscAr),) * Nr  # ini_parms.F:505-507
        else:
            raise NotImplementedError("viscAr from viscAz/viscAp/viscArDefault (ini_parms.F:485-498) not ported")
        # diffKrNrS: ini_parms.F:550-583
        sNr = _array_key(nml, d, "PARM01", "diffKrNrS")
        diffKrS = get(d, "PARM01", "diffKrS", default="UNSET")
        if sNr is not None:
            if diffKrS != "UNSET" or len(sNr) != Nr:
                raise ValueError("ini_parms.F:557-570: diffKrNrS partial or together with diffKrS")
            p["diffKrNrS"] = tuple(float(x) for x in sNr)
        elif diffKrS != "UNSET":
            p["diffKrNrS"] = (float(diffKrS),) * Nr  # ini_parms.F:571-575
        else:
            raise NotImplementedError("diffKrNrS from diffKrNrT (ini_parms.F:576-582) not ported")
        # eedata / data.exch2
        p["useCubedSphereExchange"] = bool(get("eedata", "EEPARMS", "useCubedSphereExchange", default=False))
        if int(get("data.exch2", "W2_EXCH2_PARM01", "preDefTopol", default=0)) != 0:
            raise NotImplementedError("mskCor needs the facet layout of data.exch2 preDefTopol=0")
        dims = _array_key(nml, "data.exch2", "W2_EXCH2_PARM01", "dimsFacets")
        if dims is None:
            raise NotImplementedError("data.exch2 without dimsFacets")
        p["dimsFacets"] = tuple(int(x) for x in dims)
        out = cls(**p)
        out.check()
        return out

    def check(self):
        """Hard errors for every option/parameter value whose branch is not ported (V4r4 values only)."""
        if not ALLOW_GGL90_SMOOTH or ALLOW_GGL90_HORIZDIFF or ALLOW_GGL90_IDEMIX:
            raise NotImplementedError("only the V4r4 GGL90_OPTIONS.h (SMOOTH on, HORIZDIFF/IDEMIX off) is ported")
        if not SOLVE_DIAGONAL_KINNER or SOLVE_DIAGONAL_LOWMEMORY:
            raise NotImplementedError("only SOLVE_DIAGONAL_KINNER (solve_tridiagonal.F:147-209) is ported")
        if self.mxlMaxFlag != 2:
            raise NotImplementedError(f"mxlMaxFlag={self.mxlMaxFlag}: only 2 (ggl90_calc.F:285-318) is ported")
        if not self.mxlSurfFlag:
            raise NotImplementedError("mxlSurfFlag=F not ported (V4r4: T, ggl90_calc.F:230-236)")
        if self.calcMeanVertShear:
            raise NotImplementedError("calcMeanVertShear=T not ported (V4r4: F, ggl90_calc.F:448-461)")
        if not self.GGL90_dirichlet:
            raise NotImplementedError("GGL90_dirichlet=F not ported (V4r4: T, ggl90_calc.F:648-658)")
        if self.useIDEMIX:
            raise NotImplementedError("useIDEMIX not ported")
        if self.GGL90diffTKEh != 0.0:
            raise ValueError("ggl90_check.F:59-65: GGL90diffTKEh /= 0 needs ALLOW_GGL90_HORIZDIFF")
        if not self.useCubedSphereExchange:
            raise NotImplementedError("mskCor without useCubedSphereExchange not ported (ggl90_init_fixed.F:61)")
        # ggl90_readparms.F:209-237
        if self.GGL90TKEmin <= 0.0 or self.GGL90TKEbottom < 0.0 or self.GGL90mixingLengthMin <= 0.0 \
                or self.GGL90viscMax <= 0.0 or self.GGL90diffMax <= 0.0:
            raise ValueError("GGL90 parameter out of range (ggl90_readparms.F:209-237)")


def tile_corners(dimsFacets, layout):
    """exch2 facet-corner flags per tile: [nTiles, 4] bool (SW, SE, NW, NE), tiles in W2 order.
    Tiles are numbered facet by facet, tx fastest (w2_set_map_tiles.F:170-184); a tile edge is a facet edge when
    it has no internal neighbour (w2_set_tile2tiles.F:88-111); corners: fill_cs_corner_tr_rl.F:77-84."""
    L = layout
    out = []
    for fct in range(len(dimsFacets) // 2):
        fNx, fNy = dimsFacets[2 * fct], dimsFacets[2 * fct + 1]
        nbTx, nbTy = fNx // L.sNx, fNy // L.sNy  # w2_set_map_tiles.F:164-165
        for ty in range(1, nbTy + 1):
            for tx in range(1, nbTx + 1):
                iLo, iHi = (tx - 1) * L.sNx + 1, tx * L.sNx
                jLo, jHi = (ty - 1) * L.sNy + 1, ty * L.sNy
                N, S, E, W = jHi >= fNy, jLo <= 1, iHi >= fNx, iLo <= 1  # w2_set_tile2tiles.F:91,97,103,109
                out.append((W and S, E and S, W and N, E and N))
    if len(out) != L.nTiles:
        raise ValueError(f"data.exch2 dimsFacets give {len(out)} tiles, layout has {L.nTiles}")
    return np.array(out, bool)


def mskcor(p, layout):
    """mskCor of ggl90_init_fixed.F:53-69: 1 everywhere, corner halo blocks of facet corners set to 0 by
    FILL_CS_CORNER_TR_RL(0, .FALSE., mskCor) (fill_cs_corner_tr_rl.F:86-117). numpy [nTiles, ny, nx]."""
    L = layout
    m = np.ones(L.shape2d)  # ggl90_init_fixed.F:57-61
    c = tile_corners(p.dimsFacets, L)
    js_lo, js_hi = L.js(1 - L.OLy, 0), L.js(L.sNy + 1, L.sNy + L.OLy)
    is_lo, is_hi = L.is_(1 - L.OLx, 0), L.is_(L.sNx + 1, L.sNx + L.OLx)
    for t in range(L.nTiles):
        sw, se, nw, ne = c[t]
        if sw:
            m[t, js_lo, is_lo] = 0.0  # fill_cs_corner_tr_rl.F:89-93  trFld(1-i,1-j)
        if se:
            m[t, js_lo, is_hi] = 0.0  # fill_cs_corner_tr_rl.F:96-100 trFld(sNx+i,1-j)
        if nw:
            m[t, js_hi, is_lo] = 0.0  # fill_cs_corner_tr_rl.F:103-107 trFld(1-i,sNy+j)
        if ne:
            m[t, js_hi, is_hi] = 0.0  # fill_cs_corner_tr_rl.F:110-114 trFld(sNx+i,sNy+j)
    return m


def klowc(maskC):
    """kLowC (ini_masks_etc.F:192-200): index of the lowest wet cell (hFacC /= 0 <=> maskC = 1,
    ini_masks_etc.F:481-488), 0 for a dry column. [T, ny, nx] int32, from maskC [T, Nr, ny, nx]."""
    Nr = maskC.shape[1]
    k = jnp.arange(1, Nr + 1, dtype=jnp.int32)[None, :, None, None]
    return jnp.max(jnp.where(maskC != 0.0, k, 0), axis=1)


def _sqrt(x):
    """SQRT with a finite derivative at x = 0 (TKE of dry cells): forward identical to SQRT for x >= 0; x < 0
    gives NaN as the Fortran does."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), jnp.where(x < 0.0, jnp.nan, 0.0))


def solve_tridiagonal_kinner(a3d, b3d, c3d, y3d):
    """solve_tridiagonal.F:145-209 (SOLVE_DIAGONAL_KINNER, not LOWMEMORY) on every column of [T, Nr, ny, nx]
    (the Fortran loops over the full 1-OLx..sNx+OLx, 1-OLy..sNy+OLy range, not over iMin..iMax)."""
    a, b, c, y = (jnp.moveaxis(v, 1, 0) for v in (a3d, b3d, c3d, y3d))  # [Nr, T, ny, nx]
    # k = 1: solve_tridiagonal.F:170-178
    b1ok = b[0] != 0.0
    b1 = jnp.where(b1ok, b[0], 1.0)
    cp1 = jnp.where(b1ok, c[0] / b1, 0.0)  # c3d_prime(1) = c3d_m1(i,j,1) / b3d(i,j,1)
    yp1 = jnp.where(b1ok, y[0] / b1, 0.0)  # y3d_prime(1) = y3d_m1(i,j,1) / b3d(i,j,1)

    def fwd(carry, x):  # solve_tridiagonal.F:179-190
        cpm, ypm = carry
        ak, bk, ck, yk = x
        tmpVar = bk - ak * cpm
        ok = tmpVar != 0.0
        recVar = 1.0 / jnp.where(ok, tmpVar, 1.0)
        cp = jnp.where(ok, ck * recVar, 0.0)
        yp = jnp.where(ok, (yk - ypm * ak) * recVar, 0.0)
        return (cp, yp), (cp, yp)

    _, (cps, yps) = jax.lax.scan(fwd, (cp1, yp1), (a[1:], b[1:], c[1:], y[1:]))
    cp = jnp.concatenate([cp1[None], cps])
    yp = jnp.concatenate([yp1[None], yps])

    def bwd(u_next, x):  # solve_tridiagonal.F:195-201
        cpk, ypk = x
        u = ypk - cpk * u_next
        return u, u

    _, us = jax.lax.scan(bwd, yp[-1], (cp[:-1], yp[:-1]), reverse=True)
    u = jnp.concatenate([us, yp[-1][None]])
    return jnp.moveaxis(u, 0, 1)


def ggl90_calc(p: GGL90Params, g, GGL90TKE, uVel, vVel, sigmaR, surfaceForcingU, surfaceForcingV, recip_hFacC):
    """GGL90_CALC (ggl90_calc.F) for all tiles. Returns (GGL90TKE, GGL90viscArU, GGL90viscArV, GGL90diffKr).
    See the module docstring for the inputs and the points each output covers."""
    L = g.layout
    Nr = L.Nr
    GGL90TKE, uVel, vVel, sigmaR, surfaceForcingU, surfaceForcingV, recip_hFacC = (
        jnp.asarray(a) for a in (GGL90TKE, uVel, vVel, sigmaR, surfaceForcingU, surfaceForcingV, recip_hFacC))
    maskC = jnp.asarray(g.maskC)
    iMin, iMax = 2 - L.OLx, L.sNx + L.OLx - 1  # ggl90_calc.F:132
    jMin, jMax = 2 - L.OLy, L.sNy + L.OLy - 1  # ggl90_calc.F:133
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    Ip1 = L.is_(iMin + 1, iMax + 1)
    Jp1 = L.js(jMin + 1, jMax + 1)
    dt = p.deltaTggl90  # ggl90_calc.F:141
    kSurf = 1  # ggl90_calc.F:143
    explDissFac = 0.0  # ggl90_calc.F:145
    implDissFac = 1.0 - explDissFac  # ggl90_calc.F:146
    drF = jnp.asarray(g.drF)
    recip_drF = jnp.asarray(g.recip_drF)
    recip_drC = jnp.asarray(g.recip_drC)
    viscArNr = jnp.asarray(p.viscArNr)
    diffKrNrS = jnp.asarray(p.diffKrNrS)
    kLowC = klowc(maskC)
    mskCor = jnp.asarray(tile_rows(mskcor(p, L), tile_index(g)))  # rows of g's tiles

    def col(v):  # [Nr] -> broadcast over [T, Nr, j, i]
        return v[None, :, None, None]

    tke_in = GGL90TKE
    mC = maskC[:, :, J, I]  # maskC on the inner range, all k
    # --- local initialisations: ggl90_calc.F:169-205 (full arrays; only what later code reads is kept)
    # SQRTTKE(:,:,1) over the full range (ggl90_calc.F:196), k >= 2 on the inner range (ggl90_calc.F:215)
    sq = _sqrt(tke_in[:, :, J, I])  # SQRTTKE on the inner range, all k (k=1 is the ggl90_calc.F:196 value)
    # Nsquare: ggl90_calc.F:218-219 (k >= 2); 0 at k = 1 (ggl90_calc.F:184)
    NsqFac = p.gravity * p.gravitySign * p.recip_rhoConst  # left-to-right product as in the Fortran expression
    Nsq = jnp.concatenate([jnp.zeros_like(sigmaR[:, :1, J, I]), NsqFac * sigmaR[:, 1:, J, I]], axis=1)
    # mixing length: ggl90_calc.F:223-224 (k >= 2); GGL90mixingLengthMin at k = 1 (ggl90_calc.F:177)
    # (XLA's algebraic simplifier would rewrite A/SQRT(B) as A*RSQRT(B), not correctly rounded: 1-ulp differences
    # in the mixing length. Bitwise needs XLA_FLAGS --xla_disable_hlo_passes=algsimp, as set by conftest.py.)
    ml_k = SQRTTWO * sq[:, 1:] / jnp.sqrt(jnp.maximum(Nsq[:, 1:], GGL90eps))
    ml1 = jnp.full_like(sq[:, :1], p.GGL90mixingLengthMin)
    # mxlSurfFlag: ggl90_calc.F:230-236  GGL90mixingLength(i,j,2) = drF(1)
    ml_k = ml_k.at[:, 0].set(jnp.broadcast_to(drF[0], ml_k[:, 0].shape))
    # mxlMaxFlag = 2: ggl90_calc.F:285-318
    ml_kT = jnp.moveaxis(ml_k, 1, 0)  # [Nr-1 (k=2..Nr), T, j, i]

    def down(prev, x):  # ggl90_calc.F:287-294  ml(k) = MIN(ml(k), ml(k-1)+drF(k-1))
        mlk, d = x
        new = jnp.minimum(mlk, prev + d)
        return new, new

    _, ml_kT = jax.lax.scan(down, ml1[:, 0], (ml_kT, drF[:-1]))
    # ggl90_calc.F:295-300  ml(Nr) = MIN(ml(Nr), GGL90mixingLengthMin+drF(Nr))
    mlNr = jnp.minimum(ml_kT[-1], p.GGL90mixingLengthMin + drF[Nr - 1])

    def up(nxt, x):  # ggl90_calc.F:301-308  ml(k) = MIN(ml(k), ml(k+1)+drF(k)), k = Nr-1..2
        mlk, d = x
        new = jnp.minimum(mlk, nxt + d)
        return new, new

    _, mid = jax.lax.scan(up, mlNr, (ml_kT[:-1], drF[1:Nr - 1]), reverse=True)
    ml_kT = jnp.concatenate([mid, mlNr[None]])
    ml_k = jnp.moveaxis(ml_kT, 0, 1)
    ml_k = jnp.maximum(ml_k, p.GGL90mixingLengthMin)  # ggl90_calc.F:313-314
    rml_k = 1.0 / ml_k  # ggl90_calc.F:315
    rml = jnp.concatenate([jnp.zeros_like(ml1), rml_k], axis=1)  # rMixingLength(:,:,1) = 0 (ggl90_calc.F:194)

    # --- "proper" k loop, k = 2..Nr: ggl90_calc.F:363-543 (levels independent -> vectorised over k)
    sqk = sq[:, 1:]
    mCk = mC[:, 1:]
    KappaM = p.GGL90ck * ml_k * sqk  # ggl90_calc.F:424
    visctmp_k = jnp.maximum(KappaM, col(diffKrNrS[1:])) * mCk  # ggl90_calc.F:425-426
    KappaM = jnp.maximum(KappaM, col(viscArNr[1:])) * mCk  # ggl90_calc.F:429
    # vertical shear, calcMeanVertShear = F: ggl90_calc.F:450-460
    rdrC = col(recip_drC[1:Nr])  # recip_drC(k), k = 2..Nr
    uk, ukm1 = uVel[:, 1:], uVel[:, :-1]
    vk, vkm1 = vVel[:, 1:], vVel[:, :-1]
    tempU = ((ukm1[..., J, I] + ukm1[..., J, Ip1]) - (uk[..., J, I] + uk[..., J, Ip1])) * halfRL * rdrC
    tempV = ((vkm1[..., J, I] + vkm1[..., Jp1, I]) - (vk[..., J, I] + vk[..., Jp1, I])) * halfRL * rdrC
    verticalShear = tempU * tempU + tempV * tempV  # ggl90_calc.F:458
    # Prandtl number: ggl90_calc.F:485-493
    Nsqk = Nsq[:, 1:]
    RiNumber = jnp.maximum(Nsqk, 0.0) / (verticalShear + GGL90eps)
    prTemp = jnp.where(RiNumber >= 0.2, 5.0 * RiNumber, 1.0)
    Pr_k = jnp.minimum(10.0, prTemp)
    # ggl90_calc.F:496-514
    KappaH = KappaM / Pr_k
    KappaE_k = p.GGL90alpha * KappaM * mCk  # ggl90_calc.F:500
    tke_k = tke_in[:, 1:, J, I]
    TKEdissipation = explDissFac * p.GGL90ceps * sqk * rml_k * tke_k  # ggl90_calc.F:503-505
    tke_k = tke_k + dt * ((KappaM * verticalShear - KappaH * Nsqk) - TKEdissipation)  # ggl90_calc.F:507-512
    KappaE = jnp.concatenate([jnp.zeros_like(KappaE_k[:, :1]), KappaE_k], axis=1)  # KappaE(:,:,1) = 0

    # --- implicit step, matrix: ggl90_calc.F:551-625
    rhC = recip_hFacC[:, :, J, I]
    # lower diagonal, k = 2..Nr, km1 = MAX(2,k-1): ggl90_calc.F:556-569
    KEkm1 = jnp.concatenate([KappaE[:, 1:2], KappaE[:, 1:Nr - 1]], axis=1)
    a_k = -(dt * col(recip_drF[:Nr - 1]) * rhC[:, :Nr - 1] * 0.5 * (KappaE[:, 1:] + KEkm1)
            * col(recip_drC[1:Nr]) * mC[:, 1:])
    a3d = jnp.concatenate([jnp.zeros_like(a_k[:, :1]), a_k], axis=1)  # a3d(:,:,1) = 0 (ggl90_calc.F:553)
    # upper diagonal, k = 2..Nr, kp1 = MAX(1,MIN(klowC,k+1)): ggl90_calc.F:576-589
    kL = kLowC[:, None, J, I]
    kk = jnp.arange(2, Nr + 1, dtype=jnp.int32)[None, :, None, None]
    kp1 = jnp.maximum(1, jnp.minimum(kL, kk + 1))
    KEkp1 = jnp.take_along_axis(KappaE, kp1 - 1, axis=1)
    c_k = -(dt * col(recip_drF[1:]) * rhC[:, 1:] * 0.5 * (KappaE[:, 1:] + KEkp1)
            * col(recip_drC[1:Nr]) * mC[:, :-1])
    c3d = jnp.concatenate([jnp.zeros_like(c_k[:, :1]), c_k], axis=1)  # c3d(:,:,1) = 0 (ggl90_calc.F:573)
    # (GGL90_dirichlet = T: the Neumann block ggl90_calc.F:604-612 is skipped)
    # center diagonal, k = 1..Nr, km1 = MAX(k-1,1): ggl90_calc.F:615-625
    mCkm1 = jnp.concatenate([mC[:, :1], mC[:, :-1]], axis=1)
    b3d = (1.0 - c3d - a3d) + implDissFac * dt * p.GGL90ceps * sq * rml * mC * mCkm1
    # right-hand side (TKE after the explicit k loop): level 1 untouched there
    y = jnp.concatenate([tke_in[:, :1, J, I], tke_k], axis=1)
    # surface boundary condition: ggl90_calc.F:629-646
    kp1s = min(Nr, kSurf + 1)  # ggl90_calc.F:629
    su = surfaceForcingU[:, J, I] + surfaceForcingU[:, J, Ip1]
    sv = surfaceForcingV[:, J, I] + surfaceForcingV[:, Jp1, I]
    su = 0.5 * su
    sv = 0.5 * sv
    uStarSquare = _sqrt(su * su + sv * sv)  # ggl90_calc.F:633-638 (x**2 -> x*x)
    tke1 = mC[:, kSurf - 1] * jnp.maximum(p.GGL90TKEsurfMin, p.GGL90m2 * uStarSquare)  # ggl90_calc.F:640-641
    y = y.at[:, kSurf - 1].set(tke1)
    y = y.at[:, kp1s - 1].set(y[:, kp1s - 1] - a3d[:, kp1s - 1] * tke1)  # ggl90_calc.F:642-643
    a3d = a3d.at[:, kp1s - 1].set(0.0)  # ggl90_calc.F:644
    # Dirichlet bottom boundary condition: ggl90_calc.F:648-658
    kBottom = jnp.maximum(kLowC[:, None, J, I], 1)
    atBot = jnp.arange(1, Nr + 1, dtype=jnp.int32)[None, :, None, None] == kBottom
    y = jnp.where(atBot, y - p.GGL90TKEbottom * c3d, y)  # ggl90_calc.F:653-654
    c3d = jnp.where(atBot, 0.0, c3d)  # ggl90_calc.F:655
    # tridiagonal solve over the full tile range (solve_tridiagonal.F:149-209): a3d, b3d, c3d keep their
    # initial 0, 1, 0 outside iMin..iMax, jMin..jMax (ggl90_calc.F:180-182), where GGL90TKE is untouched.
    A = jnp.zeros_like(tke_in).at[:, :, J, I].set(a3d)
    B = jnp.ones_like(tke_in).at[:, :, J, I].set(b3d)
    C = jnp.zeros_like(tke_in).at[:, :, J, I].set(c3d)
    Y = tke_in.at[:, :, J, I].set(y)
    tke = solve_tridiagonal_kinner(A, B, C, Y)
    # minimum TKE: ggl90_calc.F:668-676
    tke = tke.at[:, :, J, I].set(mC * jnp.maximum(tke[:, :, J, I], p.GGL90TKEmin))

    # --- smoothed viscosity / diffusivity (ALLOW_GGL90_SMOOTH): ggl90_calc.F:681-773
    visctmp = jnp.zeros_like(tke_in).at[:, 1:, J, I].set(visctmp_k)  # GGL90visctmp = 0 elsewhere (:174)
    Pr = jnp.ones_like(tke_in).at[:, 1:, J, I].set(Pr_k)  # TKEPrandtlNumber = 1 elsewhere (:176)
    vt = visctmp[:, 1:]  # k = 2..Nr
    mk = maskC[:, 1:]
    mc = mskCor[:, None]

    def s(a, jlo, jhi, ilo, ihi, dj, di):
        return a[..., L.js(jlo + dj, jhi + dj), L.is_(ilo + di, ihi + di)]

    def vm(jlo, jhi, ilo, ihi, dj, di):  # GGL90visctmp(i+di,j+dj,k)*mskCor(i+di,j+dj)
        return s(vt, jlo, jhi, ilo, ihi, dj, di) * s(mc, jlo, jhi, ilo, ihi, dj, di)

    def mm(jlo, jhi, ilo, ihi, dj, di):  # maskC(i+di,j+dj,k)*mskCor(i+di,j+dj)
        return s(mk, jlo, jhi, ilo, ihi, dj, di) * s(mc, jlo, jhi, ilo, ihi, dj, di)

    zero3 = jnp.zeros_like(tke_in)
    # GGL90diffKr: ggl90_calc.F:681-713, j = 1..sNy, i = 1..sNx
    r = (1, L.sNy, 1, L.sNx)
    num = (p4 * s(vt, *r, 0, 0) * s(mc, *r, 0, 0)
           + p8 * ((vm(*r, 0, -1) + vm(*r, 0, 1)) + (vm(*r, -1, 0) + vm(*r, 1, 0)))
           + p16 * ((vm(*r, 1, 1) + vm(*r, -1, -1)) + (vm(*r, -1, 1) + vm(*r, 1, -1))))
    den = (p4
           + p8 * ((mm(*r, 0, -1) + mm(*r, 0, 1)) + (mm(*r, -1, 0) + mm(*r, 1, 0)))
           + p16 * ((mm(*r, 1, 1) + mm(*r, -1, -1)) + (mm(*r, -1, 1) + mm(*r, 1, -1))))
    tmpVisc = num / den * s(mk, *r, 0, 0) * s(mc, *r, 0, 0)
    tmpVisc = jnp.minimum(tmpVisc / s(Pr[:, 1:], *r, 0, 0), p.GGL90diffMax)  # ggl90_calc.F:709
    diffKr = zero3.at[:, 1:, L.js(1, L.sNy), L.is_(1, L.sNx)].set(
        jnp.maximum(tmpVisc, col(diffKrNrS[1:])))  # ggl90_calc.F:710
    # GGL90viscArU: ggl90_calc.F:715-743, j = 1..sNy, i = 1..sNx+1
    r = (1, L.sNy, 1, L.sNx + 1)
    num = (p4 * (vm(*r, 0, -1) + vm(*r, 0, 0))
           + p8 * ((vm(*r, -1, -1) + vm(*r, -1, 0)) + (vm(*r, 1, -1) + vm(*r, 1, 0))))
    den = (p4 * 2.0
           + p8 * ((mm(*r, -1, -1) + mm(*r, -1, 0)) + (mm(*r, 1, -1) + mm(*r, 1, 0))))
    tmpVisc = (num / den * s(mk, *r, 0, -1) * s(mc, *r, 0, -1)
               * s(mk, *r, 0, 0) * s(mc, *r, 0, 0))
    tmpVisc = jnp.minimum(tmpVisc, p.GGL90viscMax)  # ggl90_calc.F:739
    viscArU = zero3.at[:, 1:, L.js(1, L.sNy), L.is_(1, L.sNx + 1)].set(
        jnp.maximum(tmpVisc, col(viscArNr[1:])))  # ggl90_calc.F:740
    # GGL90viscArV: ggl90_calc.F:745-773, j = 1..sNy+1, i = 1..sNx
    r = (1, L.sNy + 1, 1, L.sNx)
    num = (p4 * (vm(*r, -1, 0) + vm(*r, 0, 0))
           + p8 * ((vm(*r, -1, -1) + vm(*r, 0, -1)) + (vm(*r, -1, 1) + vm(*r, 0, 1))))
    den = (p4 * 2.0
           + p8 * ((mm(*r, -1, -1) + mm(*r, 0, -1)) + (mm(*r, -1, 1) + mm(*r, 0, 1))))
    tmpVisc = (num / den * s(mk, *r, -1, 0) * s(mc, *r, -1, 0)
               * s(mk, *r, 0, 0) * s(mc, *r, 0, 0))
    tmpVisc = jnp.minimum(tmpVisc, p.GGL90viscMax)  # ggl90_calc.F:769
    viscArV = zero3.at[:, 1:, L.js(1, L.sNy + 1), L.is_(1, L.sNx)].set(
        jnp.maximum(tmpVisc, col(viscArNr[1:])))  # ggl90_calc.F:770
    return tke, viscArU, viscArV, diffKr
