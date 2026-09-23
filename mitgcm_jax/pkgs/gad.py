"""pkg/generic_advdiff GAD_ADVECTION: multi-dimensional DST3 tracer advection on the LLC cubed-sphere (plan Task 16a).

Literal port of c66g `pkg/generic_advdiff/gad_advection.F` (no V4r4 override) as called from TEMP_INTEGRATE /
SALT_INTEGRATE (`model/src/temp_integrate.F:277-282`, `salt_integrate.F:277`), with GAD_DST3_ADV_X/Y
(`gad_dst3_adv_x.F`, `gad_dst3_adv_y.F`), FILL_CS_CORNER_TR_RL and FILL_CS_CORNER_UV_RS (`eesupp/src`).

Branches the V4r4 flux-forced build executes (and the only ones ported; anything else raises at setup):
  * `tempAdvScheme = saltAdvScheme = 30` = ENUM_DST3 (`GAD.h:48`): plain DST3, no flux limiter
    (`OLD_DST3_FORMULATION` is not defined anywhere in the build, so `gad_dst3_adv_x.F:113-117` is the formula).
  * `tempImplVertAdv = saltImplVertAdv = T` (implicitAdvection): only the horizontal passes run; the tendency is
    `gTracer = (localTij - tracer)/deltaTLev(k)` over the whole tile array, halos included (`gad_advection.F:779-789`),
    and the vertical block (`:835-1069`, GAD_DST3_ADV_R) is skipped entirely: vertical advection is done later by
    GAD_IMPLICIT_R (Task 16b). `wFld` is therefore an argument of the Fortran routine but not read here.
  * `GAD_MULTIDIM_COMPRESSIBLE` undefined (c66g `GAD_OPTIONS.h`): the non-compressible update
    (`gad_advection.F:492-498`, `:544-550`, `:701-707`, `:753-759`).
  * `ALLOW_OBCS` undefined (obcs not in packages.conf): `maskLocW/S = maskW/S` (`gad_advection.F:309-310`).
  * `ALLOW_AUTODIFF` defined (autodiff in packages.conf): `af` is reset to 0 before every X and Y block
    (`gad_advection.F:363-367`, `:572-576`), not only when fluxes are computed.
  * `useCubedSphereExchange = T` with exch2: npass = 3 and the per-facet pass table (`gad_advection.F:242-253`,
    `:333-351`); `deepAtmosphere = F` (deepFac* = 1, `set_grid_factors.F:52-55`); `rhoRefFile` unset
    (recip_rhoFacC = 1, `set_ref_state.F:76`). hFac macros are plain arrays (`HFACW_MACROS.h:40`,
    `RECIP_HFACC_MACROS.h:40`; ALLOW_DEPTH_CONTROL undefined). `_RS` is real*8 (`CPP_EEOPTIONS.h:61` REAL4_IS_SLOW).

Which geometry: `_hFacW`, `_hFacS`, `_recip_hFacC` are the model's current hFac arrays. Under z* (NONLIN_FRSURF,
select_rStar = 2) they were last set by UPDATE_R_STAR(.TRUE.) in FORWARD_STEP (`forward_step.F:855`, stage
S06_update_rstar_T); CALC_R_STAR (S11) does not change them. They are NOT `recip_hFacNew` (`thermodynamics.F:204`),
which TEMP_INTEGRATE uses later but GAD_ADVECTION never sees. The time step is dTtracerLev(k).

Design decision: per-facet sweep order and overlapOnly/interiorOnly under SPMD
------------------------------------------------------------------------------
Everything in GAD_ADVECTION is tile-local (no exchange inside; corner halos are filled by local copies), but the
control flow differs per tile: which direction each of the three passes sweeps (calc_fluxes_X/Y from nCFace =
exch2_myFace), whether the pass updates only the halo rows/columns across a facet edge (overlapOnly), everything but
those (interiorOnly), or all points, the facet-edge flags N/S/E_edge, W_edge that set the loop bounds, and which
FILL_CS_CORNER_TR_RL calls happen. On the 5-facet LLC grid the table (`gad_advection.F:337-351`) gives
    facet 1: X interior | Y interior        | -             facet 4: Y interior | X interior | -
    facet 2: X interior | X overlap         | Y interior    facet 5: Y interior | Y overlap  | X interior
    facet 3: Y overlap (fill 2 before, fill 1 after the flux) | X all points | Y interior
We map it to SPMD with **per-tile static tables and select**, not by grouping tiles by facet class: every pass
evaluates the X block and then the Y block on all tiles at once (vectorised over [tile, k, j, i]); per-tile tables
built in numpy from the exch2 topology decide where each block acts: update-region masks (the Fortran loop bounds
for that tile, empty when the tile does not sweep that direction in that pass) and corner-fill gathers (identity for
tiles that do not call the fill). A tile whose block does not run keeps localTij unchanged, which is exactly the
Fortran effect, so the order of operations per tile is the Fortran order. Cost: 6 DST3 flux evaluations per call
instead of 3; benefit: no data-dependent control flow and one code path whatever the tile distribution. The tables
carry the tile axis and are passed as arrays (`gad_tables`), so under shard_map they are sharded with the fields
(P=1 and P=N run the same code; `advect_tiles` is the per-shard kernel). Grouping by facet class would need gathers
and scatters over tile subsets that do not align with a sharding of the tile axis, and five code paths.
"""

import functools
import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.layout import Layout
from mitgcm_jax.parallel.tiles import tile_index, tile_rows
from mitgcm_jax.params_io import params_pytree

# GAD.h:48 ENUM_DST3 = 30 (3rd order direct space-time)
ENUM_DST3 = 30
# GAD.h:100 oneSixth = 1.D0/6.D0
ONE_SIXTH = 1.0 / 6.0
# GAD.h:121,124
GAD_TEMPERATURE = 1
GAD_SALINITY = 2


# ----------------------------------------------------------------------------------------------------------------
# exch2 topology of the tiles (only what GAD_ADVECTION and FILL_CS_CORNER_* read)
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TileTopology:
    """exch2_myFace and exch2_is[N,S,E,W]edge per tile (tile t = exch2 tile t+1)."""
    face: tuple
    N_edge: tuple
    S_edge: tuple
    E_edge: tuple
    W_edge: tuple

    @classmethod
    def from_facets(cls, dims_facets, layout):
        """w2_set_map_tiles.F:161-178 (tiles numbered facet by facet, tx fastest; no blank tiles) and
        w2_set_tile2tiles.F:88-110 (an edge flag is set when the tile edge is the facet edge)."""
        L = layout
        face, N, S, E, W = [], [], [], [], []
        nf = len(dims_facets) // 2
        for f in range(1, nf + 1):
            fNx, fNy = dims_facets[2 * f - 2], dims_facets[2 * f - 1]  # w2_set_map_tiles.F:161-162
            nbTx, nbTy = fNx // L.sNx, fNy // L.sNy                      # :163-164
            for ty in range(1, nbTy + 1):                                # :168
                for tx in range(1, nbTx + 1):                            # :169
                    tBasex, tBasey = (tx - 1) * L.sNx, (ty - 1) * L.sNy  # :181-182
                    iLo, iHi = tBasex + 1, tBasex + L.sNx                # w2_set_tile2tiles.F:76-77
                    jLo, jHi = tBasey + 1, tBasey + L.sNy                # :78-79
                    face.append(f)                                       # w2_set_map_tiles.F:178
                    N.append(not jHi < fNy)                              # w2_set_tile2tiles.F:91-92
                    S.append(not jLo > 1)                                # :97-98
                    E.append(not iHi < fNx)                              # :103-104
                    W.append(not iLo > 1)                                # :109-110
        if len(face) != L.nTiles:
            raise ValueError(f"facets give {len(face)} tiles, layout has {L.nTiles}")
        return cls(tuple(face), tuple(N), tuple(S), tuple(E), tuple(W))


def _nml_indexed(nml, fname, group, key):
    """A namelist array given as `key(lo:hi) = ...` (params_io keeps the index in the key): list or None."""
    g = nml.file(fname).get(group.lower(), {})
    if key.lower() in g:
        return list(g[key.lower()])
    for k, v in g.items():
        m = re.fullmatch(re.escape(key.lower()) + r"\((\d+):(\d+)\)", k)
        if m and int(m.group(1)) == 1:
            return list(v)
    return None


# ----------------------------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------------------------
@params_pytree
@dataclass(frozen=True)
class GADParams:
    """Parameters of one GAD_ADVECTION call (one tracer). Fields annotated `float` are traced pytree leaves (tuples of
    Nr floats for the per-level ones; params_io.params_pytree): pass the object as a jit argument, never close over
    it, so XLA cannot reassociate products of compile-time constants (e.g. `(x*oneSixth)*y`)."""
    trIdentity: int
    advectionScheme: int
    implicitAdvection: bool
    useCubedSphereExchange: bool
    topo: TileTopology
    layout: Layout
    dTtracerLev: float          # tuple: deltaTLev(k), k=1..Nr
    deepFacC: float             # tuple: set_grid_factors.F:52
    recip_deepFacC: float       # tuple: set_grid_factors.F:54
    recip_deepFac2C: float      # tuple: set_grid_factors.F:55
    recip_rhoFacC: float        # tuple: set_ref_state.F:76
    oneSixth: float             # GAD.h:100 oneSixth = 1.D0/6.D0

    @classmethod
    def from_namelists(cls, nml, tracer, layout=None):
        """tracer: 'temp' or 'salt'. Every value from the run's namelists, else the cited Fortran default."""
        L = layout or Layout()
        if tracer not in ("temp", "salt"):
            raise ValueError(tracer)
        trId = GAD_TEMPERATURE if tracer == "temp" else GAD_SALINITY
        # set_defaults.F:221-222 tempAdvScheme = saltAdvScheme = 2
        scheme = int(nml.get("data", "parm01", f"{tracer}AdvScheme", default=2))
        # set_defaults.F:208-209 tempImplVertAdv = saltImplVertAdv = .FALSE. (with T the vertical scheme,
        # {temp,salt}VertAdvScheme, is not used by GAD_ADVECTION)
        implicit = bool(nml.get("data", "parm01", f"{tracer}ImplVertAdv", default=False))
        # set_defaults.F:223 multiDimAdvection = .TRUE.; set_defaults.F:192,195 tempAdvection = saltAdvection = .TRUE.
        multi = bool(nml.get("data", "parm01", "multiDimAdvection", default=True))
        adv = bool(nml.get("data", "parm01", f"{tracer}Advection", default=True))
        # gad_init_fixed.F:129-139: tempMultiDimAdvec (GAD.h:28,32,36 ENUM_CENTERED_2ND/UPWIND_3RD/CENTERED_4TH=2,3,4)
        if not (multi and adv) or scheme in (2, 3, 4):
            raise NotImplementedError("GAD_ADVECTION is only called with tempMultiDimAdvec/saltMultiDimAdvec = T "
                                      "(gad_init_fixed.F:129-139)")
        if scheme != ENUM_DST3:
            raise NotImplementedError(f"{tracer}AdvScheme={scheme}: only ENUM_DST3=30 is ported (gad_advection.F:412)")
        if not implicit:
            raise NotImplementedError(f"{tracer}ImplVertAdv=F: the explicit vertical block "
                                      "(gad_advection.F:835-1069) is not ported")
        # eeset_parms.F:104 useCubedSphereExchange = .FALSE.
        cs = bool(nml.get("eedata", "eeparms", "useCubedSphereExchange", default=False))
        if not cs:
            raise NotImplementedError("useCubedSphereExchange=F (npass=2 branch, gad_advection.F:261-268) not ported")
        # set_defaults.F:73 deepAtmosphere = .FALSE.
        if bool(nml.get("data", "parm04", "deepAtmosphere", default=False)):
            raise NotImplementedError("deepAtmosphere=T (set_grid_factors.F:64-93) not ported")
        # set_defaults.F:61 rhoRefFile = ' '  (anelastic rhoFac, set_ref_state.F:343-359, not ported)
        if str(nml.get("data", "parm01", "rhoRefFile", default=" ")).strip():
            raise NotImplementedError("rhoRefFile set (anelastic rhoFacC, set_ref_state.F:343) not ported")
        # exch2: preDefTopol=0 with dimsFacets; other topologies / blank tiles not ported.
        # w2_readparms.F:68-69 default preDefTopol = 3 with useCubedSphereExchange; :74-75 blankList = 0
        if int(nml.get("data.exch2", "W2_EXCH2_PARM01", "preDefTopol", default=3)) != 0:
            raise NotImplementedError("exch2 preDefTopol != 0 not ported")
        if any(int(x) != 0 for x in (_nml_indexed(nml, "data.exch2", "W2_EXCH2_PARM01", "blankList") or [])):
            raise NotImplementedError("exch2 blankList not ported")
        dims = _nml_indexed(nml, "data.exch2", "W2_EXCH2_PARM01", "dimsFacets")
        topo = TileTopology.from_facets([int(x) for x in dims], L)
        dTl = _dTtracerLev(nml, L.Nr)
        one = (1.0,) * L.Nr
        return cls(trId, scheme, implicit, cs, topo, L, dTl, one, one, one, one, ONE_SIXTH)


def _dTtracerLev(nml, Nr):
    """ini_parms.F:875-903 with set_defaults.F:293,297 and ini_parms.F:826 defaults."""
    lev = nml.get("data", "parm03", "dTtracerLev", default=[0.0], array=True)  # set_defaults.F:297
    lev = [float(x) for x in lev] + [0.0] * (Nr - len(lev))
    deltaTtracer = float(nml.get("data", "parm03", "deltaTtracer", default=0.0))  # ini_parms.F:826
    if deltaTtracer != lev[0] and deltaTtracer != 0.0 and lev[0] != 0.0:        # ini_parms.F:875-880
        raise ValueError("deltaTtracer & dTtracerLev(1) not equal")
    elif lev[0] != 0.0:                                                         # ini_parms.F:881-882
        deltaTtracer = lev[0]
    deltaT = float(nml.get("data", "parm03", "deltaT", default=0.0))            # set_defaults.F:293
    if deltaT == 0.0:                                                           # ini_parms.F:884-887
        # deltaTClock: no default statement (COMMON, 0); deltaTMom, deltaTFreeSurf: set_defaults.F:294-295 = 0
        for name in ("deltaTClock", "deltaTtracer", "deltaTMom", "deltaTFreeSurf"):
            v = deltaTtracer if name == "deltaTtracer" else float(nml.get("data", "parm03", name, default=0.0))
            if v != 0.0:
                deltaT = v
                break
    if deltaT == 0.0:
        raise ValueError("no model time step in data PARM03")
    if deltaTtracer == 0.0:                                                     # ini_parms.F:899
        deltaTtracer = deltaT
    return tuple(deltaTtracer if x == 0.0 else x for x in lev)                  # ini_parms.F:901-903


# ----------------------------------------------------------------------------------------------------------------
# per-tile static tables (numpy): pass table, update-region masks, corner-fill gathers
# ----------------------------------------------------------------------------------------------------------------
def pass_flags(nCFace, ipass):
    """gad_advection.F:333-351 (useCubedSphereExchange): (overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y)."""
    overlapOnly = False                                            # :334
    interiorOnly = False                                           # :333
    if ipass == 1:
        overlapOnly = nCFace % 3 == 0                              # :338
        interiorOnly = nCFace % 3 != 0                             # :339
        cX = nCFace in (6, 1, 2)                                   # :340
        cY = nCFace in (3, 4, 5)                                   # :341
    elif ipass == 2:
        overlapOnly = nCFace % 3 == 2                              # :343
        interiorOnly = nCFace % 3 == 1                             # :344
        cX = nCFace in (2, 3, 4)                                   # :345
        cY = nCFace in (5, 6, 1)                                   # :346
    else:
        interiorOnly = True                                        # :348
        cX = nCFace in (5, 6)                                      # :349
        cY = nCFace in (2, 3)                                      # :350
    return overlapOnly, interiorOnly, cX, cY


NPASS = 3  # gad_advection.F:243 (useCubedSphereExchange)


def _fill_tr_pairs(L, fill4dir, sw, se, nw, ne):
    """FILL_CS_CORNER_TR_RL(fill4dir, .FALSE.) (fill_cs_corner_tr_rl.F:125-262) as (dest(i,j), src(i,j)) pairs, Fortran
    indices; withSigns=.FALSE. so negOne = 1 (:71-72)."""
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    out = []
    for j in range(1, OLy + 1):
        for i in range(1, OLx + 1):
            if fill4dir == 1:
                if sw: out.append(((1 - i, 1 - j), (1 - j, i)))                        # :168
                if se: out.append(((sNx + i, 1 - j), (sNx + j, i)))                    # :175
                if nw: out.append(((1 - i, sNy + j), (1 - j, sNy + 1 - i)))            # :182
                if ne: out.append(((sNx + i, sNy + j), (sNx + j, sNy + 1 - i)))        # :189
            elif fill4dir == 2:
                if sw: out.append(((1 - i, 1 - j), (j, 1 - i)))                        # :238
                if se: out.append(((sNx + i, 1 - j), (sNx + 1 - j, 1 - i)))            # :245
                if nw: out.append(((1 - i, sNy + j), (j, sNy + i)))                    # :252
                if ne: out.append(((sNx + i, sNy + j), (sNx + 1 - j, sNy + i)))        # :259
            else:
                raise NotImplementedError(f"fill4dir={fill4dir}")
    return out


def _fill_uv_pairs(L, sw, se, nw, ne):
    """FILL_CS_CORNER_UV_RS(.FALSE., uFld, vFld) (fill_cs_corner_uv_rs.F) as (dest comp, dest(i,j), src comp,
    src(i,j)) with comp 0 = u, 1 = v; withSigns=.FALSE. so negOne = 1."""
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    out = []
    rj, ri = range(1, OLy + 1), range(1, OLx + 1)
    if sw:
        out += [(0, (1 - i, 1 - j), 1, (1 - j, 1 + i)) for j in rj for i in ri]        # uFld(1-i,1-j)=vFld(1-j,1+i)
        out += [(1, (1 - i, 1 - j), 0, (1 + j, 1 - i)) for j in rj for i in ri]        # vFld(1-i,1-j)=uFld(1+j,1-i)
    if se:
        out += [(0, (sNx + i, 1 - j), 1, (sNx + j, i)) for j in rj for i in range(2, OLx + 1)]
        out += [(1, (sNx + i, 1 - j), 0, (sNx + 1 - j, 1 - i)) for j in rj for i in ri]
    if nw:
        out += [(0, (1 - i, sNy + j), 1, (1 - j, sNy + 1 - i)) for j in rj for i in ri]
        out += [(1, (1 - i, sNy + j), 0, (j, sNy + i)) for j in range(2, OLy + 1) for i in ri]
    if ne:
        out += [(0, (sNx + i, sNy + j), 1, (sNx + j, sNy + 2 - i)) for j in rj for i in range(2, OLx + 1)]
        out += [(1, (sNx + i, sNy + j), 0, (sNx + 2 - j, sNy + i)) for j in range(2, OLy + 1) for i in ri]
    return out


def _corners(topo, t):
    """fill_cs_corner_tr_rl.F:78-85 (same in fill_cs_corner_uv_rs.F)."""
    sw = topo.W_edge[t] and topo.S_edge[t]
    se = topo.E_edge[t] and topo.S_edge[t]
    ne = topo.E_edge[t] and topo.N_edge[t]
    nw = topo.W_edge[t] and topo.N_edge[t]
    return sw, se, nw, ne


def _flat(L, i, j):
    return L.jj(j) * L.nx + L.ii(i)


def _fill_tr_table(L, topo, fill4dir, apply):
    """[T, ny*nx] int32: per tile, the flat source index of every point (identity where not written / not applied)."""
    src = np.tile(np.arange(L.ny * L.nx, dtype=np.int32), (L.nTiles, 1))
    for t in range(L.nTiles):
        if not apply[t]:
            continue
        pairs = _fill_tr_pairs(L, fill4dir, *_corners(topo, t))
        dests = {_flat(L, *d) for d, _ in pairs}
        # every source lies outside the corner blocks, so one gather equals the Fortran sequence of assignments
        assert not dests & {_flat(L, *s) for _, s in pairs}
        for d, s in pairs:
            src[t, _flat(L, *d)] = _flat(L, *s)
    return src


def _fill_uv_table(L, topo):
    """([T, ny*nx], [T, ny*nx]) int32 sources for the u and v outputs, indices into concat(u_flat, v_flat)."""
    n = L.ny * L.nx
    su = np.tile(np.arange(n, dtype=np.int32), (L.nTiles, 1))
    sv = su + n
    for t in range(L.nTiles):
        pairs = _fill_uv_pairs(L, *_corners(topo, t))
        dests = {(c, _flat(L, *d)) for c, d, _, _ in pairs}
        assert not dests & {(c, _flat(L, *s)) for _, _, c, s in pairs}
        for c, d, cs, s in pairs:
            (su if c == 0 else sv)[t, _flat(L, *d)] = cs * n + _flat(L, *s)
    return su, sv


def _box(L, mask2d, ilo, ihi, jlo, jhi):
    mask2d[L.js(jlo, jhi), L.is_(ilo, ihi)] = True


@functools.lru_cache(maxsize=8)
def _tables_np(topo, layout):
    L = layout
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    T = L.nTiles
    tab = {}
    su, sv = _fill_uv_table(L, topo)
    tab["uvfill_u"], tab["uvfill_v"] = su, sv
    active = {}  # (name) -> any tile applies (static)
    for ipass in range(1, NPASS + 1):
        mX = np.zeros(L.shape2d, bool)
        mY = np.zeros(L.shape2d, bool)
        fXb, fXa, fYb, fYa = (np.zeros(T, bool) for _ in range(4))
        for t in range(T):
            N, S, E, W = topo.N_edge[t], topo.S_edge[t], topo.E_edge[t], topo.W_edge[t]
            overlapOnly, interiorOnly, cX, cY = pass_flags(topo.face[t], ipass)
            if cX:                                                                  # gad_advection.F:376
                doflux = (not overlapOnly) or N or S                                # :381
                fXb[t] = doflux and overlapOnly                                     # :384-387
                fXa[t] = doflux and overlapOnly and ipass == 1                      # :452-455
                if overlapOnly:                                                     # :472
                    iMinUpd, iMaxUpd = 1 - OLx + 1, sNx + OLx - 1                   # :473-474
                    if W: iMinUpd = 1                                               # :477
                    if E: iMaxUpd = sNx                                             # :478
                    if S: _box(L, mX[t], iMinUpd, iMaxUpd, 1 - OLy, 0)              # :480-502
                    if N: _box(L, mX[t], iMinUpd, iMaxUpd, sNy + 1, sNy + OLy)      # :503-525
                else:
                    jMinUpd, jMaxUpd = 1 - OLy, sNy + OLy                           # :529-530
                    if interiorOnly and S: jMinUpd = 1                              # :531
                    if interiorOnly and N: jMaxUpd = sNy                            # :532
                    _box(L, mX[t], 1 - OLx + 1, sNx + OLx - 1, jMinUpd, jMaxUpd)    # :533-534
            if cY:                                                                  # :585
                doflux = (not overlapOnly) or E or W                                # :590
                fYb[t] = doflux and overlapOnly                                     # :593-596
                fYa[t] = doflux and overlapOnly and ipass == 1                      # :661-664
                if overlapOnly:                                                     # :681
                    jMinUpd, jMaxUpd = 1 - OLy + 1, sNy + OLy - 1                   # :682-683
                    if S: jMinUpd = 1                                               # :686
                    if N: jMaxUpd = sNy                                             # :687
                    if W: _box(L, mY[t], 1 - OLx, 0, jMinUpd, jMaxUpd)              # :689-711
                    if E: _box(L, mY[t], sNx + 1, sNx + OLx, jMinUpd, jMaxUpd)      # :712-734
                else:
                    iMinUpd, iMaxUpd = 1 - OLx, sNx + OLx                           # :738-739
                    if interiorOnly and W: iMinUpd = 1                              # :740
                    if interiorOnly and E: iMaxUpd = sNx                            # :741
                    _box(L, mY[t], iMinUpd, iMaxUpd, 1 - OLy + 1, sNy + OLy - 1)    # :742-743
        tab[f"mX{ipass}"], tab[f"mY{ipass}"] = mX, mY
        active[f"mX{ipass}"], active[f"mY{ipass}"] = bool(mX.any()), bool(mY.any())
        for name, fl, d in ((f"fXb{ipass}", fXb, 1), (f"fXa{ipass}", fXa, 2),
                            (f"fYb{ipass}", fYb, 2), (f"fYa{ipass}", fYa, 1)):
            active[name] = bool(fl.any())
            if fl.any():
                tab[name] = _fill_tr_table(L, topo, d, fl)
    return tab, tuple(sorted(active.items()))


def gad_tables(params):
    """Per-tile tables (numpy, leading tile axis) for `advect_tiles`; shard them with the fields under shard_map."""
    tab, _ = _tables_np(params.topo, params.layout)
    return dict(tab)


def _active(params):
    return dict(_tables_np(params.topo, params.layout)[1])


# ----------------------------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------------------------
def _gather(a, src):
    """a: [T, K, ny, nx] (or a list of such, concatenated along the flat point axis); src: [T, ny*nx] -> [T,K,ny,nx]."""
    parts = a if isinstance(a, (list, tuple)) else [a]
    T, K, ny, nx = parts[0].shape
    flat = jnp.concatenate([p.reshape(T, K, ny * nx) for p in parts], axis=2) if len(parts) > 1 \
        else parts[0].reshape(T, K, ny * nx)
    out = jax.vmap(lambda f, s: f[:, s])(flat, jnp.asarray(src))
    return out.reshape(T, K, ny, nx)


def dst3_adv_x(L, oneSixth, deltaTloc, uTrans, uFld, maskLocW, tracer, recip_dxC, recip_deepFacC):
    """GAD_DST3_ADV_X(bi,bj,k, calcCFL=.TRUE., ...) (gad_dst3_adv_x.F:74-121), all tiles and levels at once.
    deltaTloc, recip_deepFacC: [1,K,1,1]; recip_dxC: [T,1,ny,nx]; others [T,K,ny,nx]."""
    sNx, OLx, OLy, sNy = L.sNx, L.OLx, L.OLy, L.sNy
    J = L.js(1 - OLy, sNy + OLy)                                    # :79
    I = L.is_(1 - OLx + 2, sNx + OLx - 1)                           # :80
    Ip1 = L.is_(1 - OLx + 3, sNx + OLx)
    Im1 = L.is_(1 - OLx + 1, sNx + OLx - 2)
    Im2 = L.is_(1 - OLx, sNx + OLx - 3)
    t = tracer
    m = maskLocW
    Rjp = (t[..., J, Ip1] - t[..., J, I]) * m[..., J, Ip1]         # :81
    Rj = (t[..., J, I] - t[..., J, Im1]) * m[..., J, I]            # :82
    Rjm = (t[..., J, Im1] - t[..., J, Im2]) * m[..., J, Im1]       # :83
    uCFL = jnp.abs(uFld[..., J, I] * deltaTloc * recip_dxC[..., J, I] * recip_deepFacC)  # :85-87 (calcCFL=T)
    d0 = (2. - uCFL) * (1. - uCFL) * oneSixth                      # :88
    d1 = (1. - uCFL * uCFL) * oneSixth                             # :89
    uTr = uTrans[..., J, I]
    uT = (0.5 * (uTr + jnp.abs(uTr)) * (t[..., J, Im1] + (d0 * Rj + d1 * Rjm))     # :113-117
          + 0.5 * (uTr - jnp.abs(uTr)) * (t[..., J, I] - (d0 * Rj + d1 * Rjp)))
    # uT(1-OLx,j) = uT(2-OLx,j) = uT(sNx+OLx,j) = 0 (:74-78): the zeros of the output array
    return jnp.zeros_like(uTrans).at[..., J, I].set(uT)


def dst3_adv_y(L, oneSixth, deltaTloc, vTrans, vFld, maskLocS, tracer, recip_dyC, recip_deepFacC):
    """GAD_DST3_ADV_Y (gad_dst3_adv_y.F:73-120)."""
    sNx, OLx, OLy, sNy = L.sNx, L.OLx, L.OLy, L.sNy
    J = L.js(1 - OLy + 2, sNy + OLy - 1)                            # :78
    Jp1 = L.js(1 - OLy + 3, sNy + OLy)
    Jm1 = L.js(1 - OLy + 1, sNy + OLy - 2)
    Jm2 = L.js(1 - OLy, sNy + OLy - 3)
    I = L.is_(1 - OLx, sNx + OLx)                                   # :79
    t = tracer
    m = maskLocS
    Rjp = (t[..., Jp1, I] - t[..., J, I]) * m[..., Jp1, I]         # :80
    Rj = (t[..., J, I] - t[..., Jm1, I]) * m[..., J, I]            # :81
    Rjm = (t[..., Jm1, I] - t[..., Jm2, I]) * m[..., Jm1, I]       # :82
    vCFL = jnp.abs(vFld[..., J, I] * deltaTloc * recip_dyC[..., J, I] * recip_deepFacC)  # :84-86
    d0 = (2. - vCFL) * (1. - vCFL) * oneSixth                      # :87
    d1 = (1. - vCFL * vCFL) * oneSixth                             # :88
    vTr = vTrans[..., J, I]
    vT = (0.5 * (vTr + jnp.abs(vTr)) * (t[..., Jm1, I] + (d0 * Rj + d1 * Rjm))     # :112-116
          + 0.5 * (vTr - jnp.abs(vTr)) * (t[..., J, I] - (d0 * Rj + d1 * Rjp)))
    # vT(i,1-OLy) = vT(i,2-OLy) = vT(i,sNy+OLy) = 0 (:73-77)
    return jnp.zeros_like(vTrans).at[..., J, I].set(vT)


def _vert(a, K):
    return jnp.asarray(a, dtype=jnp.float64).reshape(1, K, 1, 1)


def advect_tiles(params, tables, grid2d, vgrid, uFld, vFld, tracer, hFacW, hFacS, recip_hFacC):
    """GAD_ADVECTION (implicitAdvection=.TRUE.) on a set of tiles; returns gTracer [T, Nr, ny, nx].

    tables: `gad_tables(params)` (leading tile axis, same tiles as the fields). grid2d: dict of [T, ny, nx] fields
    dyG, dxG, recip_dxC, recip_dyC, recip_rA, maskInC and [T, Nr, ny, nx] maskW, maskS. vgrid: dict of [Nr] drF,
    recip_drF, rhoFacC. uFld, vFld (residual flow), tracer, hFacW, hFacS, recip_hFacC: [T, Nr, ny, nx]."""
    L = params.layout
    K = L.Nr
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    act = _active(params)
    dt = _vert(params.dTtracerLev, K)                   # deltaTLev(k)
    deepFacC = _vert(params.deepFacC, K)
    recip_deepFacC = _vert(params.recip_deepFacC, K)
    recip_deepFac2C = _vert(params.recip_deepFac2C, K)
    recip_rhoFacC = _vert(params.recip_rhoFacC, K)
    oneSixth = params.oneSixth
    drF = _vert(vgrid["drF"], K)
    recip_drF = _vert(vgrid["recip_drF"], K)
    rhoFacC = _vert(vgrid["rhoFacC"], K)
    g2 = {k: jnp.asarray(v)[:, None] if jnp.ndim(v) == 3 else jnp.asarray(v) for k, v in grid2d.items()}

    # gad_advection.F:279-286 (xA, yA) and :289-294 (uTrans, vTrans), full tile arrays
    xA = g2["dyG"] * deepFacC * drF * hFacW
    yA = g2["dxG"] * deepFacC * drF * hFacS
    uTrans = uFld * xA * rhoFacC
    vTrans = vFld * yA * rhoFacC
    # :297-313 localTij = tracer; maskLocW/S = maskW/S (no OBCS); :315-319 FILL_CS_CORNER_UV_RS(.FALSE.)
    localTij = tracer
    maskW, maskS = g2["maskW"], g2["maskS"]
    maskLocW = _gather([maskW, maskS], tables["uvfill_u"])
    maskLocS = _gather([maskW, maskS], tables["uvfill_v"])

    # update coefficient deltaTLev(k)*recip_rhoFacC(k)*_recip_hFacC*recip_drF(k)*recip_rA*recip_deepFac2C(k)
    # (:492-495), evaluated left to right as in Fortran
    coef = dt * recip_rhoFacC * recip_hFacC * recip_drF * g2["recip_rA"] * recip_deepFac2C
    maskInC = g2["maskInC"]

    Jall, Iall = L.js(1 - OLy, sNy + OLy), L.is_(1 - OLx, sNx + OLx)
    # d(af) and d(trans) at every point where the X update loop may run: i = 1-OLx .. sNx+OLx-1 (af(i+1)-af(i))
    IX, IXp1 = L.is_(1 - OLx, sNx + OLx - 1), L.is_(2 - OLx, sNx + OLx)
    JY, JYp1 = L.js(1 - OLy, sNy + OLy - 1), L.js(2 - OLy, sNy + OLy)

    def ddx(a):
        return jnp.zeros_like(a).at[..., Jall, IX].set(a[..., Jall, IXp1] - a[..., Jall, IX])

    def ddy(a):
        return jnp.zeros_like(a).at[..., JY, Iall].set(a[..., JYp1, Iall] - a[..., JY, Iall])

    dUx = ddx(uTrans)
    dVy = ddy(vTrans)

    def update(local, daf, dtrans, mask):
        # :492-498 / :544-550 (X), :701-707 / :753-759 (Y)
        new = local - coef * (daf - tracer * dtrans) * maskInC
        return jnp.where(jnp.asarray(mask)[:, None], new, local)

    for ipass in range(1, NPASS + 1):                                   # :323
        # --- X direction (:359-565). af = 0 (:363-367, ALLOW_AUTODIFF) for tiles that do not compute fluxes: those
        # tiles have an empty update region, so the flux array below is only read where Fortran computes it.
        if act[f"mX{ipass}"] or act[f"fXb{ipass}"] or act[f"fXa{ipass}"]:
            if act[f"fXb{ipass}"]:
                localTij = _gather(localTij, tables[f"fXb{ipass}"])       # :384-387 FILL_CS_CORNER_TR_RL(1)
            af = dst3_adv_x(L, oneSixth, dt, uTrans, uFld, maskLocW, localTij, g2["recip_dxC"], recip_deepFacC)  # :412-415
            if act[f"fXa{ipass}"]:
                localTij = _gather(localTij, tables[f"fXa{ipass}"])       # :452-455 FILL_CS_CORNER_TR_RL(2)
            if act[f"mX{ipass}"]:
                localTij = update(localTij, ddx(af), dUx, tables[f"mX{ipass}"])
        # --- Y direction (:568-774)
        if act[f"mY{ipass}"] or act[f"fYb{ipass}"] or act[f"fYa{ipass}"]:
            if act[f"fYb{ipass}"]:
                localTij = _gather(localTij, tables[f"fYb{ipass}"])       # :593-596 FILL_CS_CORNER_TR_RL(2)
            af = dst3_adv_y(L, oneSixth, dt, vTrans, vFld, maskLocS, localTij, g2["recip_dyC"], recip_deepFacC)  # :621-624
            if act[f"fYa{ipass}"]:
                localTij = _gather(localTij, tables[f"fYa{ipass}"])       # :661-664 FILL_CS_CORNER_TR_RL(1)
            if act[f"mY{ipass}"]:
                localTij = update(localTij, ddy(af), dVy, tables[f"mY{ipass}"])

    # :779-789 implicitAdvection: gTracer = (localTij - tracer)/deltaTLev(k) over the full tile array
    # (bitwise only with XLA's algsimp pass off, conftest.py: it rewrites x / broadcast(d) as x * broadcast(1/d))
    return (localTij - tracer) / dt


GRID2D = ("dyG", "dxG", "recip_dxC", "recip_dyC", "recip_rA", "maskInC", "maskW", "maskS")
VGRID = ("drF", "recip_drF", "rhoFacC")


def gad_advection(params, g, uFld, vFld, wFld, tracer, hFacW, hFacS, recip_hFacC):
    """GAD_ADVECTION(implicitAdvection=T, advectionScheme=ENUM_DST3, ...) on all tiles -> gTracer [T,Nr,ny,nx].

    Fortran inputs: uFld, vFld, wFld = residual (Eulerian + GM bolus) velocity of THERMODYNAMICS
    (`thermodynamics.F:255-268`); tracer = theta or salt (state at THERMODYNAMICS); hFacW, hFacS, recip_hFacC = the
    current hFac arrays (UPDATE_R_STAR at `forward_step.F:855`); geometry from `g` (dyG, dxG, recip_dxC, recip_dyC,
    recip_rA, maskInC, maskW, maskS, drF, recip_drF, rhoFacC). wFld is not read (implicit vertical advection)."""
    del wFld  # gad_advection.F:835: vertical block skipped when implicitAdvection
    if params.advectionScheme != ENUM_DST3 or not params.implicitAdvection or not params.useCubedSphereExchange:
        raise NotImplementedError("GADParams outside the ported branches")
    grid2d = {k: getattr(g, k) for k in GRID2D}
    vgrid = {k: getattr(g, k) for k in VGRID}
    tables = {k: tile_rows(v, tile_index(g)) for k, v in gad_tables(params).items()}  # rows of g's tiles
    return advect_tiles(params, tables, grid2d, vgrid, jnp.asarray(uFld), jnp.asarray(vFld),
                        jnp.asarray(tracer), jnp.asarray(hFacW), jnp.asarray(hFacS), jnp.asarray(recip_hFacC))
