"""pkg/seaice SEAICE_ADVDIFF (advection + diffusion of HEFF, AREA, HSNOW) and SEAICE_REG_RIDGE (plan M2.5).

Literal port of c66g `pkg/seaice/seaice_advdiff.F`, `seaice_advection.F`, `seaice_diffusion.F`, `seaice_reg_ridge.F`
(no V4r4 override of any of them: the V4r4 `code/` tree overrides only seaice_growth.F, seaice_diagnostics_init.F and
SEAICE_OPTIONS.h) with `pkg/generic_advdiff/gad_dst3fl_adv_x.F`, `gad_dst3fl_adv_y.F`, `gad_diff_x.F`, `gad_diff_y.F`,
and the FILL_CS_CORNER_TR_RL / FILL_CS_CORNER_UV_RS gathers of `mitgcm_jax.pkgs.gad`. Called from SEAICE_MODEL
(`seaice_model.F:184` SEAICE_ADVDIFF(uIce, vIce), `:194` SEAICE_REG_RIDGE) after SEAICE_DYNSOLVER.

Branches the full V4r4 build executes (checked in the preprocessed `bld/*.f` of the oracle build; the only ones
ported, anything else raises at setup):
  * SEAICE_OPTIONS.h (V4r4 `code/`): SEAICE_CGRID defined -> uc, vc are UICE, VICE themselves, no B-grid averaging and
    no EXCH_UV_XY_RL (`seaice_advdiff.F:115-138`); SEAICE_ITD, SEAICE_VARIABLE_SALINITY, ALLOW_SITRACER undefined ->
    no ITD copies, no opnWtrFrac, no HSALT, no SItracer blocks. EXF_SEAICE_FRACTION undefined (EXF_OPTIONS.h) and
    DISABLE_AREA_FLOOR undefined: SEAICE_REG_RIDGE parts (1)-(4) + Hibler capping (`seaice_reg_ridge.F:184-192`,
    `:217-233`, `:252-257`, `:283-289`, `:369-382`). SEAICE_MODIFY_GROWTH_ADJ undefined.
  * ALLOW_AUTODIFF_TAMC defined (pkg/autodiff compiled): gFld = 0 before the fields (`seaice_advdiff.F:163-167`),
    localTij, afx, afy = 0 at the start of SEAICE_ADVECTION (`seaice_advection.F:197-201`, `:269-274`).
  * SEAICEadvScheme = 33 = ENUM_DST3_FLUX_LIMIT (`GAD.h:52`) for HEFF, AREA, HSNOW (data.seaice; per-field schemes
    unset -> `seaice_readparms.F:923-934`): SEAICEmultiDimAdvection = .TRUE. (`seaice_advdiff.F:145-150`), the
    multi-dimensional branch with GAD_DST3FL_ADV_X/Y (`seaice_advection.F:385-388`, `:587-590`).
  * extensiveFld = .TRUE. (PARAMETER, `seaice_advection.F:68-69`): the updates without the r_hFld / div(u) term
    (`seaice_advection.F:425-434`, `:447-456`, `:491-500`, `:627-636`, `:649-658`, `:693-702`).
  * ALLOW_OBCS undefined: maskLocW/S = maskW/S(k=1) (`seaice_advection.F:261-262`).
  * useCubedSphereExchange = T with exch2: nipass = 3 and the per-facet pass table of SEAICE_ADVECTION
    (`seaice_advection.F:205-221`, `:299-315`). NOTE: this table is NOT gad_advection.F's: in pass 2 and 3
    interiorOnly stays .FALSE. (`:308-315`; GAD_ADVECTION sets it, `gad_advection.F:344`, `:348`), and the corner fill
    before a flux runs for `overlapOnly .OR. ipass.EQ.1` (`:352-356`, `:554-558`; GAD: overlapOnly only), while the
    fill after the flux sits outside the `doflux` block (`:408-411`, `:610-613`).
  * SEAICEdiffKhHeff = SEAICEdiffKhArea = SEAICEdiffKhSnow = 400 > 0: SEAICE_DIFFUSION(fac = ONE) is called
    (`seaice_advdiff.F:228-236`, `:271-279`, `:338-346`); deepAtmosphere = F (recip_deepFacC = 1,
    `set_grid_factors.F:54`); cosFacU = 1 (curvilinear grid, `ini_grid.F:108`); ISOTROPIC_COS_SCALING undefined.
  * SEAICE_multDim = 1 (data.seaice): only TICES(:,:,1) is reset in SEAICE_REG_RIDGE (`seaice_reg_ridge.F:224-226`).

Order and independence of the three fields (`seaice_advdiff.F:213-363`): inside the bi,bj loop, HEFF is advected,
diffused and stepped on the tile interior, then AREA, then HSNOW. Each field's tendency reads only that field
(recip_heff = 1 is never used for an extensive field) and the explicit step writes only the interior
(`:238-244`), so no field sees another field's update and no tile sees another tile's update (halos are not
exchanged inside SEAICE_ADVDIFF). The fields are therefore advected together, stacked on axis 1 of a
[tile, field, j, i] array: every element sees exactly its Fortran operations (bitwise identical to one call per
field). The halos of HEFF, AREA, HSNOW keep their input values (SEAICE_MODEL exchanges them after SEAICE_GROWTH,
`seaice_model.F:224-226`).

SPMD design (as mitgcm_jax.pkgs.gad, PORTING_LESSONS Task 16a): every pass evaluates the X and the Y block on all
tiles; per-tile static tables built in numpy from the exch2 topology decide where each block updates localTij and
keeps afx/afy (empty for a tile that does not sweep that direction in that pass), and which corner fills run
(identity gathers elsewhere). Per facet (pass 1 | pass 2 | pass 3; o = overlapOnly, a = all points, i = interiorOnly):
    facet 1: X i (fill1 before) | Y a              | -             facet 4: Y i (fill2 before) | X a | -
    facet 2: X i (fill1 before) | X o (edges only) | Y a           facet 5: Y i (fill2 before) | Y o | X a
    facet 3: Y o (fill2 before, fill1 after)       | X a | Y a
The tables carry the tile axis, so under shard_map they are sharded with the fields (`advdiff_tiles` is the
per-shard kernel).

Differentiability: the flux limiter (min/max of psi, the |Rj|*thetaMax test, ABS of the transports and CFL) makes the
advective tendency piecewise smooth; derivatives are exact inside a piece, and one-sided/undefined at the switches
(documented in the tests: FD checks at points away from the switches). Divisions are guarded so that masked lanes and
the unselected branch of every select stay finite in the backward pass (Rjm/Rj only where |Rj|*thetaMax > |Rjm|,
i.e. Rj != 0), and the quotient Rjm/Rj carries a quotient-rule JVP (`_ratio`) because JAX's rule for division
underflows for tiny Rj (a real case: HSNOW = -1.2e-240 in the oracle state). SEAICE_REG_RIDGE is piecewise linear
(MAX/MIN/IF on HEFF, HSNOW, AREA).
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.layout import Layout
from mitgcm_jax.parallel.tiles import tile_index, tile_rows
from mitgcm_jax.params_io import params_pytree
from mitgcm_jax.pkgs.gad import (TileTopology, _box, _dTtracerLev, _fill_tr_table, _fill_uv_table, _gather,
                                 _nml_indexed)

# GAD.h:28,32,36 ENUM_CENTERED_2ND/UPWIND_3RD/CENTERED_4TH (SEAICEmultiDimAdvection = F for these,
# seaice_advdiff.F:146-150); GAD.h:52 ENUM_DST3_FLUX_LIMIT = 33
ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH = 2, 3, 4
ENUM_DST3_FLUX_LIMIT = 33
# SEAICE_PARAMS.h:579-581 tracer identities
GAD_HEFF, GAD_AREA, GAD_SNOW = 1, 2, 3
# GAD.h:100 oneSixth = 1.D0/6.D0
ONE_SIXTH = 1.0 / 6.0
# gad_dst3fl_adv_x.F:39 / gad_dst3fl_adv_y.F:39 PARAMETER( thetaMax = 1.D+20 )
THETA_MAX = 1.0e20
# SEAICE_PARAMS.h:566-567 PARAMETER ( siEps = 1. _d -5 )
SI_EPS = 1.0e-5
# set_grid_factors.F:54 recip_deepFacC(k) = 1 (deepAtmosphere = F); ini_grid.F:108 cosFacU = 1 (curvilinear grid)
RECIP_DEEPFACC = 1.0
COSFACU = 1.0
# seaice_advdiff.F: the fields in call order (HEFF :213, AREA :256, HSNOW :323) and their dump stages
FIELDS = ("HEFF", "AREA", "HSNOW")

NPASS = 3  # seaice_advection.F:206 nipass = 3 (useCubedSphereExchange)


# ----------------------------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------------------------
def _topology(nml, L):
    """exch2 topology of the run (same checks as gad.GADParams.from_namelists; seaice_advection.F:215-221 reads
    exch2_myFace and exch2_is[N,S,E,W]edge)."""
    # eeset_parms.F:104 useCubedSphereExchange = .FALSE.
    if not bool(nml.get("eedata", "eeparms", "useCubedSphereExchange", default=False)):
        raise NotImplementedError("useCubedSphereExchange=F (nipass=2 branch, seaice_advection.F:229-236) not ported")
    # w2_readparms.F:68-69 default preDefTopol = 3 with useCubedSphereExchange; :74-75 blankList = 0
    if int(nml.get("data.exch2", "W2_EXCH2_PARM01", "preDefTopol", default=3)) != 0:
        raise NotImplementedError("exch2 preDefTopol != 0 not ported")
    if any(int(x) != 0 for x in (_nml_indexed(nml, "data.exch2", "W2_EXCH2_PARM01", "blankList") or [])):
        raise NotImplementedError("exch2 blankList not ported")
    dims = _nml_indexed(nml, "data.exch2", "W2_EXCH2_PARM01", "dimsFacets")
    return TileTopology.from_facets([int(x) for x in dims], L)


@params_pytree
@dataclass(frozen=True)
class SeaiceAdvDiffParams:
    """Parameters of SEAICE_ADVDIFF. Fields annotated `float` are traced pytree leaves (params_io.params_pytree):
    pass the object as a jit argument, never close over it."""
    topo: TileTopology
    layout: Layout
    advHeff: bool                # SEAICEadvHeff
    advArea: bool                # SEAICEadvArea
    advSnow: bool                # SEAICEadvSnow
    advSalt: bool                # SEAICEadvSalt (only decides whether SEAICE_ADVDIFF is called, seaice_model.F:179-180)
    diffuse: tuple               # static (SEAICEdiffKhHeff > 0, SEAICEdiffKhArea > 0, SEAICEdiffKhSnow > 0)
    SEAICE_deltaTtherm: float
    SEAICEdiffKhHeff: float
    SEAICEdiffKhArea: float
    SEAICEdiffKhSnow: float
    oneSixth: float              # GAD.h:100
    thetaMax: float              # gad_dst3fl_adv_x.F:39

    @property
    def called(self):
        """seaice_model.F:179-185: SEAICE_ADVDIFF runs only if some field is advected."""
        return self.advHeff or self.advArea or self.advSnow or self.advSalt

    @property
    def advected(self):
        """The advected fields in call order (seaice_advdiff.F:213, :256, :323)."""
        return tuple(f for f, a in zip(FIELDS, (self.advHeff, self.advArea, self.advSnow)) if a)

    @classmethod
    def from_namelists(cls, nml, layout=None):
        L = layout or Layout()
        s = ("data.seaice", "SEAICE_PARM01")
        # packages_readparms.F (useThSIce default .FALSE.); seaice_readparms.F:912-921 resets the adv flags
        if bool(nml.get("data.pkg", "packages", "useThSIce", default=False)):
            raise NotImplementedError("useThSIce=T (pkg/thsice advection, seaice_model.F:164-170) not ported")
        # seaice_readparms.F:268-274 (SEAICE_VARIABLE_SALINITY undefined: SEAICEadvSalt = .FALSE.)
        advHeff = bool(nml.get(*s, "SEAICEadvHeff", default=True))
        advArea = bool(nml.get(*s, "SEAICEadvArea", default=True))
        advSnow = bool(nml.get(*s, "SEAICEadvSnow", default=True))
        advSalt = bool(nml.get(*s, "SEAICEadvSalt", default=False))
        # seaice_readparms.F:281-285 defaults (UNSET_I = 123456789, EEPARAMS.h), then :923-934
        UNSET_I = 123456789
        advScheme = int(nml.get(*s, "SEAICEadvScheme", default=2))
        schArea = int(nml.get(*s, "SEAICEadvSchArea", default=UNSET_I))
        schHeff = int(nml.get(*s, "SEAICEadvSchHeff", default=UNSET_I))
        schSnow = int(nml.get(*s, "SEAICEadvSchSnow", default=UNSET_I))
        if schArea == UNSET_I:                  # :923-924
            schArea = schHeff
        if schArea == UNSET_I:                  # :925-926
            schArea = advScheme
        if advScheme != schArea:                # :927-928
            advScheme = schArea
        if schHeff == UNSET_I:                  # :929-930
            schHeff = schArea
        if schSnow == UNSET_I:                  # :931-932
            schSnow = schHeff
        if advScheme in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH):   # seaice_advdiff.F:146-150
            raise NotImplementedError(f"SEAICEadvScheme={advScheme}: the ADVECT branch (seaice_advdiff.F:547-729) "
                                      "is not ported")
        for name, sch, on in (("Heff", schHeff, advHeff), ("Area", schArea, advArea), ("Snow", schSnow, advSnow)):
            if on and sch != ENUM_DST3_FLUX_LIMIT:
                raise NotImplementedError(f"SEAICEadvSch{name}={sch}: only ENUM_DST3_FLUX_LIMIT=33 is ported "
                                          "(seaice_advection.F:366-401)")
        # seaice_readparms.F:286-288 defaults UNSET_RL (1.234567D5, EEPARAMS.h), then :936-943
        UNSET_RL = 1.234567e5
        kArea = float(nml.get(*s, "SEAICEdiffKhArea", default=UNSET_RL))
        kHeff = float(nml.get(*s, "SEAICEdiffKhHeff", default=UNSET_RL))
        kSnow = float(nml.get(*s, "SEAICEdiffKhSnow", default=UNSET_RL))
        if kArea == UNSET_RL:                   # :936-937
            kArea = kHeff
        if kArea == UNSET_RL:                   # :938-939
            kArea = 0.0
        if kHeff == UNSET_RL:                   # :940-941
            kHeff = kArea
        if kSnow == UNSET_RL:                   # :942-943
            kSnow = kHeff
        # seaice_readparms.F:293 SEAICE_deltaTtherm = dTtracerLev(1); :683-693 STOP unless equal (and deltaTdyn a
        # multiple of it)
        dTl1 = _dTtracerLev(nml, L.Nr)[0]
        dTtherm = float(nml.get(*s, "SEAICE_deltaTtherm", default=dTl1))
        dTdyn = float(nml.get(*s, "SEAICE_deltaTdyn", default=dTl1))      # :294
        if dTtherm != dTl1 or dTdyn < dTtherm or (dTdyn / dTtherm) != int(dTdyn / dTtherm):
            raise ValueError("Unsupported combination of SEAICE_deltaTtherm, SEAICE_deltaTdyn, dTtracerLev(1) "
                             "(seaice_readparms.F:683-693 STOP)")
        # set_defaults.F:73 deepAtmosphere = .FALSE. (recip_deepFacC = 1)
        if bool(nml.get("data", "parm04", "deepAtmosphere", default=False)):
            raise NotImplementedError("deepAtmosphere=T (set_grid_factors.F:64-93) not ported")
        topo = _topology(nml, L)
        return cls(topo, L, advHeff, advArea, advSnow, advSalt, (kHeff > 0.0, kArea > 0.0, kSnow > 0.0),
                   dTtherm, kHeff, kArea, kSnow, ONE_SIXTH, THETA_MAX)


@params_pytree
@dataclass(frozen=True)
class SeaiceRegRidgeParams:
    """Parameters of SEAICE_REG_RIDGE (non-ITD branch)."""
    layout: Layout
    SEAICE_multDim: int
    siEps: float                 # SEAICE_PARAMS.h:567
    SEAICE_area_floor: float
    SEAICE_area_max: float
    celsius2K: float

    @classmethod
    def from_namelists(cls, nml, layout=None):
        L = layout or Layout()
        s = ("data.seaice", "SEAICE_PARM01")
        # seaice_readparms.F:433 SEAICE_multDim = 1 (SEAICE_ITD undefined)
        multDim = int(nml.get(*s, "SEAICE_multDim", default=1))
        # seaice_readparms.F:486 SEAICE_area_floor = siEPS; :489 SEAICE_area_max = 1.00
        floor = float(nml.get(*s, "SEAICE_area_floor", default=SI_EPS))
        amax = float(nml.get(*s, "SEAICE_area_max", default=1.0))
        # set_defaults.F:270 celsius2K = 273.15 (data PARM01, ini_parms.F:199)
        c2k = float(nml.get("data", "parm01", "celsius2K", default=273.15))
        return cls(L, multDim, SI_EPS, floor, amax, c2k)


# ----------------------------------------------------------------------------------------------------------------
# per-tile static tables (numpy): SEAICE_ADVECTION pass table, update / keep regions, corner-fill gathers
# ----------------------------------------------------------------------------------------------------------------
def pass_flags(nCFace, ipass):
    """seaice_advection.F:299-315 (useCubedSphereExchange): (overlapOnly, interiorOnly, calc_fluxes_X,
    calc_fluxes_Y). Unlike gad_advection.F:333-351, interiorOnly is set only in pass 1."""
    interiorOnly = False                                           # :299
    overlapOnly = False                                            # :300
    if ipass == 1:
        overlapOnly = nCFace % 3 == 0                              # :304
        interiorOnly = nCFace % 3 != 0                             # :305
        cX = nCFace in (6, 1, 2)                                   # :306
        cY = nCFace in (3, 4, 5)                                   # :307
    elif ipass == 2:
        overlapOnly = nCFace % 3 == 2                              # :309
        cX = nCFace in (2, 3, 4)                                   # :310
        cY = nCFace in (5, 6, 1)                                   # :311
    else:
        cX = nCFace in (5, 6)                                      # :313
        cY = nCFace in (2, 3)                                      # :314
    return overlapOnly, interiorOnly, cX, cY


@functools.lru_cache(maxsize=8)
def _tables_np(topo, layout):
    """Per pass p: fXb<p>/fXa<p> (FILL_CS_CORNER_TR_RL before/after the X flux), mX<p> (X update region of localTij),
    kX<p> (afx keep region), and the Y analogues; uvfill_u/v (FILL_CS_CORNER_UV_RS on maskLocW/S)."""
    L = layout
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    T = L.nTiles
    tab = {}
    tab["uvfill_u"], tab["uvfill_v"] = _fill_uv_table(L, topo)       # seaice_advection.F:278-282
    active = {}
    for ipass in range(1, NPASS + 1):                                  # :287
        mX, kX, mY, kY = (np.zeros(L.shape2d, bool) for _ in range(4))
        fXb, fXa, fYb, fYa = (np.zeros(T, bool) for _ in range(4))
        for t in range(T):
            N, S, E, W = topo.N_edge[t], topo.S_edge[t], topo.E_edge[t], topo.W_edge[t]
            overlapOnly, interiorOnly, cX, cY = pass_flags(topo.face[t], ipass)
            if cX:                                                                  # :336
                doflux = (not overlapOnly) or N or S                                # :341
                fXb[t] = doflux and (overlapOnly or ipass == 1)                     # :352-356
                fXa[t] = overlapOnly and ipass == 1                                 # :408-411
                if overlapOnly:                                                     # :417
                    iMinUpd, iMaxUpd = 1 - OLx + 1, sNx + OLx - 1                   # :418-419
                    if W: iMinUpd = 1                                               # :422
                    if E: iMaxUpd = sNx                                             # :423
                    if S: _box(L, mX[t], iMinUpd, iMaxUpd, 1 - OLy, 0)              # :425-434
                    if N: _box(L, mX[t], iMinUpd, iMaxUpd, sNy + 1, sNy + OLy)      # :447-456
                    if S: _box(L, kX[t], 1 - OLx + 1, sNx + OLx, 1 - OLy, 0)        # :470-476
                    if N: _box(L, kX[t], 1 - OLx + 1, sNx + OLx, sNy + 1, sNy + OLy)  # :477-483
                else:
                    jMinUpd, jMaxUpd = 1 - OLy, sNy + OLy                           # :487-488
                    if interiorOnly and S: jMinUpd = 1                              # :489
                    if interiorOnly and N: jMaxUpd = sNy                            # :490
                    _box(L, mX[t], 1 - OLx + 1, sNx + OLx - 1, jMinUpd, jMaxUpd)    # :491-500
                    _box(L, kX[t], 1 - OLx + 1, sNx + OLx, jMinUpd, jMaxUpd)        # :514-518
                # a tile without fluxes keeps af of the previous block (no reset, :343-348 inside the IF): it must
                # read none of it
                assert doflux or not (mX[t].any() or kX[t].any())
            if cY:                                                                  # :538
                doflux = (not overlapOnly) or E or W                                # :543
                fYb[t] = doflux and (overlapOnly or ipass == 1)                     # :554-558
                fYa[t] = overlapOnly and ipass == 1                                 # :610-613
                if overlapOnly:                                                     # :619
                    jMinUpd, jMaxUpd = 1 - OLy + 1, sNy + OLy - 1                   # :620-621
                    if S: jMinUpd = 1                                               # :624
                    if N: jMaxUpd = sNy                                             # :625
                    if W: _box(L, mY[t], 1 - OLx, 0, jMinUpd, jMaxUpd)              # :627-636
                    if E: _box(L, mY[t], sNx + 1, sNx + OLx, jMinUpd, jMaxUpd)      # :649-658
                    if W: _box(L, kY[t], 1 - OLx, 0, 1 - OLy + 1, sNy + OLy)        # :672-678
                    if E: _box(L, kY[t], sNx + 1, sNx + OLx, 1 - OLy + 1, sNy + OLy)  # :679-685
                else:
                    iMinUpd, iMaxUpd = 1 - OLx, sNx + OLx                           # :689-690
                    if interiorOnly and W: iMinUpd = 1                              # :691
                    if interiorOnly and E: iMaxUpd = sNx                            # :692
                    _box(L, mY[t], iMinUpd, iMaxUpd, 1 - OLy + 1, sNy + OLy - 1)    # :693-702
                    _box(L, kY[t], iMinUpd, iMaxUpd, 1 - OLy + 1, sNy + OLy)        # :716-720
                assert doflux or not (mY[t].any() or kY[t].any())
        for name, m in ((f"mX{ipass}", mX), (f"kX{ipass}", kX), (f"mY{ipass}", mY), (f"kY{ipass}", kY)):
            tab[name] = m
            active[name] = bool(m.any())
        for name, fl, d in ((f"fXb{ipass}", fXb, 1), (f"fXa{ipass}", fXa, 2),
                            (f"fYb{ipass}", fYb, 2), (f"fYa{ipass}", fYa, 1)):
            active[name] = bool(fl.any())
            if fl.any():
                tab[name] = _fill_tr_table(L, topo, d, fl)
    return tab, tuple(sorted(active.items()))


def advection_tables(params):
    """Per-tile tables (numpy, leading tile axis) for `advdiff_tiles`; shard them with the fields under shard_map."""
    tab, _ = _tables_np(params.topo, params.layout)
    return dict(tab)


def _active(params):
    return dict(_tables_np(params.topo, params.layout)[1])


# ----------------------------------------------------------------------------------------------------------------
# kernels (fields [T, F, ny, nx]; grid [T, 1, ny, nx])
# ----------------------------------------------------------------------------------------------------------------
def _limited(d0, d1, theta, cfl):
    """gad_dst3fl_adv_x.F:88-93: psi = MAX(0, MIN(MIN(1, d0+d1*theta), theta*(1-CFL)/(CFL+1.d-20)))."""
    psi = d0 + d1 * theta
    return jnp.maximum(0.0, jnp.minimum(jnp.minimum(1.0, psi), theta * (1.0 - cfl) / (cfl + 1.0e-20)))


@jax.custom_jvp
def _ratio(a, b):
    """a/b. The value is the plain IEEE division (forward bitwise unchanged); only the derivative is written as the
    quotient rule (da - (a/b)*db)/b. JAX's own rule for division forms b**-2, which underflows to 0 for
    |b| < ~1e-154 and turns into 0*inf = NaN in the backward pass: measured on the oracle state (it 1), one
    HSNOW = -1.2e-240 made d/d(uIce) NaN on 197 lanes (the stacked fields share uTrans)."""
    return a / b


@_ratio.defjvp
def _ratio_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    q = a / b
    return q, (da - q * db) / b


def _theta(Rj, Rother, thetaMax):
    """gad_dst3fl_adv_x.F:77-86: IF (ABS(Rj)*thetaMax .LE. ABS(R)) THEN SIGN(thetaMax, R*Rj) ELSE R/Rj.
    SIGN honours a negative zero (gfortran -fsign-zero default) = copysign. Rj is replaced by 1 where the quotient is
    not selected (there Rj may be 0), so the backward pass stays finite; where it is selected, Rj != 0 but may be
    tiny (see _ratio)."""
    big = jnp.abs(Rj) * thetaMax <= jnp.abs(Rother)
    return jnp.where(big, jnp.copysign(thetaMax, Rother * Rj), _ratio(Rother, jnp.where(big, 1.0, Rj)))


def dst3fl_adv_x(L, oneSixth, thetaMax, deltaTloc, uTrans, uFld, maskLocW, tracer, recip_dxC):
    """GAD_DST3FL_ADV_X(bi,bj,k=1, calcCFL=.TRUE., ...) (gad_dst3fl_adv_x.F:45-102), all tiles and fields at once.
    tracer: [T,F,ny,nx]; uTrans, uFld, maskLocW, recip_dxC: [T,1,ny,nx]."""
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    J = L.js(1 - OLy, sNy + OLy)                                    # :50
    I = L.is_(1 - OLx + 2, sNx + OLx - 1)                           # :51
    Ip1 = L.is_(1 - OLx + 3, sNx + OLx)
    Im1 = L.is_(1 - OLx + 1, sNx + OLx - 2)
    Im2 = L.is_(1 - OLx, sNx + OLx - 3)
    t = tracer
    m = maskLocW
    Rjp = (t[..., J, Ip1] - t[..., J, I]) * m[..., J, Ip1]         # :57
    Rj = (t[..., J, I] - t[..., J, Im1]) * m[..., J, I]            # :58
    Rjm = (t[..., J, Im1] - t[..., J, Im2]) * m[..., J, Im1]       # :59
    uCFL = jnp.abs(uFld[..., J, I] * deltaTloc * recip_dxC[..., J, I] * RECIP_DEEPFACC)   # :61-63 (calcCFL=T)
    d0 = (2.0 - uCFL) * (1.0 - uCFL) * oneSixth                    # :64
    d1 = (1.0 - uCFL * uCFL) * oneSixth                            # :65
    thetaP = _theta(Rj, Rjm, thetaMax)                             # :77-81
    thetaM = _theta(Rj, Rjp, thetaMax)                             # :82-86
    psiP = _limited(d0, d1, thetaP, uCFL)                          # :88-90
    psiM = _limited(d0, d1, thetaM, uCFL)                          # :91-93
    uTr = uTrans[..., J, I]
    uT = (0.5 * (uTr + jnp.abs(uTr)) * (t[..., J, Im1] + psiP * Rj)              # :95-99
          + 0.5 * (uTr - jnp.abs(uTr)) * (t[..., J, I] - psiM * Rj))
    # uT(1-OLx,j) = uT(2-OLx,j) = uT(sNx+OLx,j) = 0 (:45-49): the zeros of the output array
    return jnp.zeros(tracer.shape, tracer.dtype).at[..., J, I].set(uT)


def dst3fl_adv_y(L, oneSixth, thetaMax, deltaTloc, vTrans, vFld, maskLocS, tracer, recip_dyC):
    """GAD_DST3FL_ADV_Y (gad_dst3fl_adv_y.F:41-102)."""
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    J = L.js(1 - OLy + 2, sNy + OLy - 1)                            # :46
    Jp1 = L.js(1 - OLy + 3, sNy + OLy)
    Jm1 = L.js(1 - OLy + 1, sNy + OLy - 2)
    Jm2 = L.js(1 - OLy, sNy + OLy - 3)
    I = L.is_(1 - OLx, sNx + OLx)                                   # :47
    t = tracer
    m = maskLocS
    Rjp = (t[..., Jp1, I] - t[..., J, I]) * m[..., Jp1, I]         # :53
    Rj = (t[..., J, I] - t[..., Jm1, I]) * m[..., J, I]            # :54
    Rjm = (t[..., Jm1, I] - t[..., Jm2, I]) * m[..., Jm1, I]       # :55
    vCFL = jnp.abs(vFld[..., J, I] * deltaTloc * recip_dyC[..., J, I] * RECIP_DEEPFACC)   # :57-59
    d0 = (2.0 - vCFL) * (1.0 - vCFL) * oneSixth                    # :60
    d1 = (1.0 - vCFL * vCFL) * oneSixth                            # :61
    thetaP = _theta(Rj, Rjm, thetaMax)                             # :73-77
    thetaM = _theta(Rj, Rjp, thetaMax)                             # :78-82
    psiP = _limited(d0, d1, thetaP, vCFL)                          # :84-86
    psiM = _limited(d0, d1, thetaM, vCFL)                          # :87-89
    vTr = vTrans[..., J, I]
    vT = (0.5 * (vTr + jnp.abs(vTr)) * (t[..., Jm1, I] + psiP * Rj)              # :91-95
          + 0.5 * (vTr - jnp.abs(vTr)) * (t[..., J, I] - psiM * Rj))
    # vT(i,1-OLy) = vT(i,2-OLy) = vT(i,sNy+OLy) = 0 (:41-45)
    return jnp.zeros(tracer.shape, tracer.dtype).at[..., J, I].set(vT)


def seaice_advection_tiles(params, tables, g2, uFld, vFld, uTrans, vTrans, iceFld, flux_x=None, flux_y=None):
    """SEAICE_ADVECTION (extensiveFld, multi-dimensional, cubed sphere) for all tiles and the stacked fields.

    g2: dict of [T,1,ny,nx] recip_dxC, recip_dyC, recip_rA, maskInC, maskW, maskS (k=1). uFld, vFld, uTrans, vTrans:
    [T,1,ny,nx]; iceFld: [T,F,ny,nx]. Returns (gFld, afx, afy), each [T,F,ny,nx]. flux_x/flux_y replace the flux
    routines (negative controls only)."""
    L = params.layout
    act = _active(params)
    dt = params.SEAICE_deltaTtherm
    fx = flux_x or dst3fl_adv_x
    fy = flux_y or dst3fl_adv_y
    OLx, OLy, sNx, sNy = L.OLx, L.OLy, L.sNx, L.sNy
    # :254-265 localTij = iceFld; maskLocW/S = maskW/S (no OBCS); :278-282 FILL_CS_CORNER_UV_RS(.FALSE., ...)
    localTij = iceFld
    maskLocW = _gather([g2["maskW"], g2["maskS"]], tables["uvfill_u"])
    maskLocS = _gather([g2["maskW"], g2["maskS"]], tables["uvfill_v"])
    # :269-274 afx = afy = 0 (ALLOW_AUTODIFF_TAMC)
    afx = jnp.zeros_like(iceFld)
    afy = jnp.zeros_like(iceFld)
    # update coefficient SEAICE_deltaTtherm*maskInC*recip_rA, Fortran left-to-right order (:494-497)
    coef = dt * g2["maskInC"] * g2["recip_rA"]

    Jall, Iall = L.js(1 - OLy, sNy + OLy), L.is_(1 - OLx, sNx + OLx)
    IX, IXp1 = L.is_(1 - OLx, sNx + OLx - 1), L.is_(2 - OLx, sNx + OLx)
    JY, JYp1 = L.js(1 - OLy, sNy + OLy - 1), L.js(2 - OLy, sNy + OLy)

    def ddx(a):  # af(i+1,j)-af(i,j) wherever an X update loop may run
        return jnp.zeros_like(a).at[..., Jall, IX].set(a[..., Jall, IXp1] - a[..., Jall, IX])

    def ddy(a):  # af(i,j+1)-af(i,j)
        return jnp.zeros_like(a).at[..., JY, Iall].set(a[..., JYp1, Iall] - a[..., JY, Iall])

    def sel(mask, new, old):
        return jnp.where(jnp.asarray(mask)[:, None], new, old)

    for ipass in range(1, NPASS + 1):                                   # :287
        # --- X direction (:336-524). Tiles that do not compute X fluxes in this pass have empty update/keep regions
        # (asserted in _tables_np), so af is read only where the Fortran computes it.
        if any(act[f"{k}X{ipass}"] for k in "mk") or act[f"fXb{ipass}"] or act[f"fXa{ipass}"]:
            if act[f"fXb{ipass}"]:
                localTij = _gather(localTij, tables[f"fXb{ipass}"])       # :352-356 FILL_CS_CORNER_TR_RL(1)
            af = fx(L, params.oneSixth, params.thetaMax, dt, uTrans, uFld, maskLocW, localTij,
                    g2["recip_dxC"])                                      # :385-388
            if act[f"fXa{ipass}"]:
                localTij = _gather(localTij, tables[f"fXa{ipass}"])       # :408-411 FILL_CS_CORNER_TR_RL(2)
            if act[f"mX{ipass}"]:                                         # :425-456, :491-500
                localTij = sel(tables[f"mX{ipass}"], localTij - coef * ddx(af), localTij)
            if act[f"kX{ipass}"]:                                         # :470-483, :514-518
                afx = sel(tables[f"kX{ipass}"], af, afx)
        # --- Y direction (:538-726)
        if any(act[f"{k}Y{ipass}"] for k in "mk") or act[f"fYb{ipass}"] or act[f"fYa{ipass}"]:
            if act[f"fYb{ipass}"]:
                localTij = _gather(localTij, tables[f"fYb{ipass}"])       # :554-558 FILL_CS_CORNER_TR_RL(2)
            af = fy(L, params.oneSixth, params.thetaMax, dt, vTrans, vFld, maskLocS, localTij,
                    g2["recip_dyC"])                                      # :587-590
            if act[f"fYa{ipass}"]:
                localTij = _gather(localTij, tables[f"fYa{ipass}"])       # :610-613 FILL_CS_CORNER_TR_RL(1)
            if act[f"mY{ipass}"]:                                         # :627-658, :693-702
                localTij = sel(tables[f"mY{ipass}"], localTij - coef * ddy(af), localTij)
            if act[f"kY{ipass}"]:                                         # :672-685, :716-720
                afy = sel(tables[f"kY{ipass}"], af, afy)

    # :732-736 gFld = (localTij - iceFld)/SEAICE_deltaTtherm on the whole tile array (bitwise only with XLA's algsimp
    # pass off, conftest.py: it rewrites x / d as x * (1/d))
    gFld = (localTij - iceFld) / dt
    return gFld, afx, afy


def seaice_diffusion(L, diffKh, fac, iceFld, iceMask, xA, yA, recip_dxC, recip_dyC, recip_rA, gFld):
    """SEAICE_DIFFUSION (seaice_diffusion.F:72-111) with GAD_DIFF_X/Y (gad_diff_x.F, gad_diff_y.F; k = 1).
    diffKh: [1,F,1,1] (or scalar); fac: scalar; iceFld, gFld: [T,F,ny,nx]; iceMask (HEFFM), xA, yA, recip_*:
    [T,1,ny,nx]."""
    OLx, OLy, sNx, sNy = L.OLx, L.OLy, L.sNx, L.sNy
    t = iceFld
    zeros = jnp.zeros_like(iceFld)
    # :81-86 fZon = fMer = 0; gad_diff_x.F:51-59: dfx(1-OLx,j) = 0, i = 2-OLx..sNx+OLx
    Ix, Ixm = L.is_(2 - OLx, sNx + OLx), L.is_(1 - OLx, sNx + OLx - 1)
    fZon = zeros.at[..., :, Ix].set(
        -(diffKh * xA[..., :, Ix] * recip_dxC[..., :, Ix] * RECIP_DEEPFACC
          * (t[..., :, Ix] - t[..., :, Ixm]) * COSFACU))
    # gad_diff_y.F:51-63: dfy(i,1-OLy) = 0, j = 2-OLy..sNy+OLy (ISOTROPIC_COS_SCALING undefined)
    Jy, Jym = L.js(2 - OLy, sNy + OLy), L.js(1 - OLy, sNy + OLy - 1)
    fMer = zeros.at[..., Jy, :].set(
        -(diffKh * yA[..., Jy, :] * recip_dyC[..., Jy, :] * RECIP_DEEPFACC * (t[..., Jy, :] - t[..., Jym, :])))
    # :92-99 on j = 1-OLy..sNy+OLy-1, i = 1-OLx..sNx+OLx-1
    J, I = L.js(1 - OLy, sNy + OLy - 1), L.is_(1 - OLx, sNx + OLx - 1)
    Jp, Ip = L.js(2 - OLy, sNy + OLy), L.is_(2 - OLx, sNx + OLx)
    upd = gFld[..., J, I] - fac * iceMask[..., J, I] * recip_rA[..., J, I] * (
        (fZon[..., J, Ip] - fZon[..., J, I]) + (fMer[..., Jp, I] - fMer[..., J, I]))
    return gFld.at[..., J, I].set(upd)


GRID2D = ("dyG", "dxG", "recip_dxC", "recip_dyC", "recip_rA", "maskInC", "maskW", "maskS")


def advdiff_tiles(params, tables, g2, uIce, vIce, HEFF, AREA, HSNOW, HEFFM):
    """SEAICE_ADVDIFF (multi-dimensional branch, SEAICE_CGRID) on a set of tiles.

    tables: `advection_tables(params)` rows of the tiles held; g2: dict of [T,ny,nx] dyG, dxG, recip_dxC, recip_dyC,
    recip_rA, maskInC, maskW, maskS (maskW/S at k=1); uIce, vIce, HEFF, AREA, HSNOW, HEFFM: [T,ny,nx].
    Returns (out, diag): out = dict HEFF, AREA, HSNOW after the explicit step (interior updated, halos = input);
    diag = uTrans, vTrans [T,ny,nx] and per advected field f: f+'_gAdv' (SEAICE_ADVECTION tendency), f+'_gFld'
    (after SEAICE_DIFFUSION), f+'_afx', f+'_afy' ([T,ny,nx]), i.e. the A01-A06 dump fields."""
    L = params.layout
    OLx, OLy, sNx, sNy = L.OLx, L.OLy, L.sNx, L.sNy
    gg = {k: jnp.asarray(v)[:, None] for k, v in g2.items()}
    uc = jnp.asarray(uIce)[:, None]                          # SEAICE_CGRID: uc = UICE (seaice_advdiff.F:94, :115)
    vc = jnp.asarray(vIce)[:, None]
    heffm = jnp.asarray(HEFFM)[:, None]
    dt = params.SEAICE_deltaTtherm
    # seaice_advdiff.F:126-131 xA = dyG*maskW(k=1), yA = dxG*maskS(k=1)
    xA = gg["dyG"] * gg["maskW"]
    yA = gg["dxG"] * gg["maskS"]
    # :205-210 uTrans = uc*xA, vTrans = vc*yA
    uTrans = uc * xA
    vTrans = vc * yA
    inputs = {"HEFF": HEFF, "AREA": AREA, "HSNOW": HSNOW}
    names = params.advected
    out = {k: jnp.asarray(v) for k, v in inputs.items()}
    diag = {"uTrans": uTrans[:, 0], "vTrans": vTrans[:, 0]}
    if not names:
        return out, diag
    fld = jnp.stack([jnp.asarray(inputs[n]) for n in names], axis=1)          # [T, F, ny, nx]
    # :222-227, :265-270, :332-337 SEAICE_ADVECTION (recip_heff = 1 unused: extensiveFld)
    gAdv, afx, afy = seaice_advection_tiles(params, tables, gg, uc, vc, uTrans, vTrans, fld)
    # :228-236, :271-279, :338-346 SEAICE_DIFFUSION(fac = ONE) where SEAICEdiffKh* > 0
    kh = dict(zip(FIELDS, (params.SEAICEdiffKhHeff, params.SEAICEdiffKhArea, params.SEAICEdiffKhSnow)))
    on = dict(zip(FIELDS, params.diffuse))
    diffKh = jnp.stack([jnp.asarray(kh[n], jnp.float64) for n in names]).reshape(1, len(names), 1, 1)
    gFld = gAdv
    if any(on[n] for n in names):
        gDif = seaice_diffusion(L, diffKh, 1.0, fld, heffm, xA, yA, gg["recip_dxC"], gg["recip_dyC"],
                                gg["recip_rA"], gAdv)
        gFld = jnp.where(np.array([on[n] for n in names])[None, :, None, None], gDif, gAdv)
    # :238-244 (HEFF), :281-287 (AREA), :348-354 (HSNOW): fld = HEFFM*(fld + SEAICE_deltaTtherm*gFld), interior
    J, I = L.js(1, sNy), L.is_(1, sNx)
    new = fld.at[..., J, I].set(heffm[..., J, I] * (fld[..., J, I] + dt * gFld[..., J, I]))
    for n_, name in enumerate(names):
        out[name] = new[:, n_]
        diag[name + "_gAdv"] = gAdv[:, n_]
        diag[name + "_gFld"] = gFld[:, n_]
        diag[name + "_afx"] = afx[:, n_]
        diag[name + "_afy"] = afy[:, n_]
    return out, diag


def _grid2d(g):
    """GRID2D fields of a Grid as [T,ny,nx] (maskW/S at k = 1: _maskW(i,j,ks=1,bi,bj), seaice_advdiff.F:109,128)."""
    out = {}
    for k in GRID2D:
        a = jnp.asarray(getattr(g, k))
        out[k] = a[:, 0] if a.ndim == 4 else a
    return out


def seaice_advdiff(params, g, uIce, vIce, HEFF, AREA, HSNOW, HEFFM):
    """SEAICE_ADVDIFF(uIce, vIce) on all tiles of `g` (or the tiles of a shard, via tile_index(g)).

    Inputs: uIce, vIce after SEAICE_DYNSOLVER (stage I01_dynsolver), HEFF, AREA, HSNOW at SEAICE_MODEL entry
    (I00_seaice_begin; SEAICE_DYNSOLVER does not change them), HEFFM (SEAICE_INIT_FIXED, G01_seaice_geometry), all
    [T,ny,nx] with halos; geometry from `g` (dyG, dxG, recip_dxC, recip_dyC, recip_rA, maskInC, maskW, maskS).
    Returns (out, diag), see `advdiff_tiles`."""
    if not params.called:
        return {"HEFF": HEFF, "AREA": AREA, "HSNOW": HSNOW}, {}
    tables = {k: tile_rows(v, tile_index(g)) for k, v in advection_tables(params).items()}
    return advdiff_tiles(params, tables, _grid2d(g), uIce, vIce, HEFF, AREA, HSNOW, HEFFM)


# ----------------------------------------------------------------------------------------------------------------
# SEAICE_REG_RIDGE
# ----------------------------------------------------------------------------------------------------------------
def seaice_reg_ridge(params, HEFF, AREA, HSNOW, TICES):
    """SEAICE_REG_RIDGE (seaice_reg_ridge.F, non-ITD branch): regularisation after advection + Hibler capping.

    HEFF, AREA, HSNOW: [T,ny,nx]; TICES: [T,nITD,ny,nx]. Only the tile interior changes (loops 1..sNx, 1..sNy);
    d_HEFFbyNEG, d_HSNWbyNEG are 0 on the halos (:102-114). Returns dict HEFF, AREA, HSNOW, TICES, d_HEFFbyNEG,
    d_HSNWbyNEG."""
    L = params.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    HEFF, AREA, HSNOW, TICES = (jnp.asarray(a) for a in (HEFF, AREA, HSNOW, TICES))
    h, a, s = HEFF[..., J, I], AREA[..., J, I], HSNOW[..., J, I]
    # (1) negative values (:184-192)
    dH = jnp.maximum(-h, 0.0)                              # :186
    h = h + dH                                             # :187
    dS = jnp.maximum(-s, 0.0)                              # :188
    s = s + dS                                             # :189
    a = jnp.maximum(a, 0.0)                                # :190
    # (2) very thin ice (:217-233)
    thin = h <= params.siEps                               # :221
    t1 = jnp.where(thin, -h, 0.0)                          # :219, :222
    t2 = jnp.where(thin, -s, 0.0)                          # :220, :223
    h = h + t1                                             # :228
    s = s + t2                                             # :229
    dH = dH + t1                                           # :230
    dS = dS + t2                                           # :231
    nd = params.SEAICE_multDim                             # :224-226 TICES(i,j,IT) = celsius2K, IT = 1..multDim
    ti = TICES[:, :nd][..., J, I]
    TICES = TICES.at[:, :nd, J, I].set(jnp.where(thin[:, None], params.celsius2K, ti))
    # (3) area but no ice/snow (:252-257)
    a = jnp.where((h == 0.0) & (s == 0.0), 0.0, a)         # :254-255
    # (4) very small area (:283-289)
    a = jnp.where((h > 0.0) | (s > 0.0), jnp.maximum(a, params.SEAICE_area_floor), a)
    # Hibler (1979) capping (:369-382)
    a = jnp.minimum(a, params.SEAICE_area_max)             # :380
    zeros = jnp.zeros_like(HEFF)                           # :102-114
    return dict(HEFF=HEFF.at[..., J, I].set(h), AREA=AREA.at[..., J, I].set(a), HSNOW=HSNOW.at[..., J, I].set(s),
                TICES=TICES, d_HEFFbyNEG=zeros.at[..., J, I].set(dH), d_HSNWbyNEG=zeros.at[..., J, I].set(dS))
