"""SEAICE_ADVDIFF (DST3 flux-limited multi-dimensional advection + Laplacian diffusion of HEFF, AREA, HSNOW) and
SEAICE_REG_RIDGE (plan M2.5): replay gates against the full-V4r4 oracle (oracle.FULL, dumps at iterations 1-3).

Replay inputs (what SEAICE_ADVDIFF reads, `seaice_model.F:184`): HEFF, AREA, HSNOW at SEAICE_MODEL entry
(I00_seaice_begin; SEAICE_DYNSOLVER does not write them, checked: I02 halos == I00 halos), uIce, vIce after the
dynamics (I01_dynsolver), HEFFM (G01_seaice_geometry), geometry (G00_geometry). SEAICE_REG_RIDGE is replayed from the
dumped I02_advdiff state (HEFF, AREA, HSNOW, TICES) and also chained from the JAX SEAICE_ADVDIFF output.
Compared (whole tile arrays, halos included, since the Fortran writes or keeps every point): A01 uTrans, vTrans;
A01/A03/A05 advective tendency gFld and fluxes afx, afy; A02/A04/A06 gFld after diffusion; I02 HEFF, AREA, HSNOW;
I03 HEFF, AREA, HSNOW, TICES (all 7 levels), d_HEFFbyNEG, d_HSNWbyNEG.

Measured (test docstrings): bitwise equal at every point of every compared field, iterations 1-3.
"""

import dataclasses
import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import Grid, stack_tiles
from mitgcm_jax.io.dump import DumpSet, read_file
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import gad
from mitgcm_jax.pkgs import seaice_advdiff as sa
from mitgcm_jax.tests import oracle

L = Layout()
RUN = oracle.FULL
ITS = (1, 2, 3)
TOL = 1e-13  # stencil class (KERNEL_GUIDE); achieved: 0 (bitwise)
INTERIOR = (slice(None), L.js(1, L.sNy), L.is_(1, L.sNx))
ADV = {"HEFF": ("A01_heff_adv", "A02_heff_diff"), "AREA": ("A03_area_adv", "A04_area_diff"),
       "HSNOW": ("A05_snow_adv", "A06_snow_diff")}
GEOM = ("dyG", "dxG", "recip_dxC", "recip_dyC", "recip_rA", "maskInC", "maskW", "maskS", "rA", "maskC")


@functools.lru_cache(maxsize=1)
def dumpset():
    """DumpSet of the full oracle with the record headers read in parallel threads (serial indexing of a full-tree
    dump takes ~110 s on cold Lustre, this ~5 s; same approach as scripts/tests/test_reference.py)."""
    d = oracle.run_dir(RUN) / "jaxdump"
    files = sorted(Path(d).glob("jd_*_t*.bin"))
    assert files, d
    with ThreadPoolExecutor(min(len(files), 64)) as ex:
        per_file = list(ex.map(lambda f: read_file(f, lazy=True), files))
    ds = DumpSet.__new__(DumpSet)
    ds.dir, ds.index = Path(d), {}
    for recs in per_file:
        for r in recs:
            ds.index.setdefault((r.iter, r.stage, r.field), {})[r.tile] = r
    ds.order = list(ds.index)
    return ds


@functools.lru_cache(maxsize=1)
def params():
    return sa.SeaiceAdvDiffParams.from_namelists(RunNamelists(oracle.run_dir(RUN)))


@functools.lru_cache(maxsize=1)
def rr_params():
    return sa.SeaiceRegRidgeParams.from_namelists(RunNamelists(oracle.run_dir(RUN)))


@functools.lru_cache(maxsize=3)
def case(it):
    ds = dumpset()
    f = functools.partial(oracle.field, ds, it)
    g = {}
    for k in GEOM:
        a = stack_tiles(ds, it, "G00_geometry", k, L)
        g[k] = a[:, :1] if a.ndim == 4 else a   # 3-D masks: level 1 only is read ([T,1,ny,nx])
    c = dict(g=Grid(g, L), HEFFM=f("G01_seaice_geometry", "HEFFM"),
             uIce=f("I01_dynsolver", "UICE"), vIce=f("I01_dynsolver", "VICE"))
    for n in sa.FIELDS:
        c[n] = f("I00_seaice_begin", n)
        s_adv, s_dif = ADV[n]
        c["ref_" + n + "_gAdv"] = f(s_adv, "gFld")
        c["ref_" + n + "_afx"] = f(s_adv, "afx")
        c["ref_" + n + "_afy"] = f(s_adv, "afy")
        c["ref_" + n + "_gFld"] = f(s_dif, "gFld")
        c["ref_I02_" + n] = f("I02_advdiff", n)
    c["ref_uTrans"] = f("A01_heff_adv", "uTrans")
    c["ref_vTrans"] = f("A01_heff_adv", "vTrans")
    c["TICES"] = f("I02_advdiff", "TICES")
    for n in ("HEFF", "AREA", "HSNOW", "TICES", "d_HEFFbyNEG", "d_HSNWbyNEG"):
        c["ref_I03_" + n] = f("I03_reg_ridge", n)
    return c


_advdiff = jax.jit(sa.seaice_advdiff)       # params is a pytree argument (float leaves traced), never closed over
_reg_ridge = jax.jit(sa.seaice_reg_ridge)


def run_advdiff(p, c, fn=None, **over):
    a = {k: c[k] for k in ("uIce", "vIce", "HEFF", "AREA", "HSNOW", "HEFFM")}
    a.update(over)
    out, diag = (fn or _advdiff)(p, c["g"], a["uIce"], a["vIce"], a["HEFF"], a["AREA"], a["HSNOW"], a["HEFFM"])
    return {k: np.asarray(v) for k, v in out.items()}, {k: np.asarray(v) for k, v in diag.items()}


def rel_err(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    d = np.abs(got - ref).max()
    s = np.abs(ref).max()
    return d / s if s > 0 else d


def stage_errors(c, out, diag):
    """{name: (max rel. error, number of differing values)} of every compared advdiff output."""
    res = {"A01.uTrans": (diag["uTrans"], c["ref_uTrans"]), "A01.vTrans": (diag["vTrans"], c["ref_vTrans"])}
    for n in sa.FIELDS:
        s_adv, s_dif = ADV[n]
        res[f"{s_adv}.gFld"] = (diag[n + "_gAdv"], c["ref_" + n + "_gAdv"])
        res[f"{s_adv}.afx"] = (diag[n + "_afx"], c["ref_" + n + "_afx"])
        res[f"{s_adv}.afy"] = (diag[n + "_afy"], c["ref_" + n + "_afy"])
        res[f"{s_dif}.gFld"] = (diag[n + "_gFld"], c["ref_" + n + "_gFld"])
        res[f"I02.{n}"] = (out[n], c["ref_I02_" + n])
    return {k: (rel_err(a, b), int((np.asarray(a) != np.asarray(b)).sum())) for k, (a, b) in res.items()}


# ------------------------------------------------------------------------------------------------ setup checks
def test_params_from_namelists():
    """data.seaice: SEAICEadvScheme = 33 for all fields, diffKh 400 for all, SEAICE_deltaTtherm = dTtracerLev(1) =
    3600 s, SEAICE_area_max 0.97, area floor siEps; topology as GAD's (same exch2)."""
    p = params()
    assert p.advected == ("HEFF", "AREA", "HSNOW") and p.called and p.diffuse == (True, True, True)
    assert (p.SEAICE_deltaTtherm, p.SEAICEdiffKhHeff, p.SEAICEdiffKhArea, p.SEAICEdiffKhSnow) == \
        (3600.0, 400.0, 400.0, 400.0)
    r = rr_params()
    assert (r.SEAICE_multDim, r.siEps, r.SEAICE_area_floor, r.SEAICE_area_max, r.celsius2K) == \
        (1, 1e-5, 1e-5, 0.97, 273.15)
    assert p.topo == gad.GADParams.from_namelists(RunNamelists(oracle.run_dir(RUN)), "temp").topo


def test_unported_options_raise():
    for key, val in (("seaiceadvscheme", 30), ("seaiceadvscheme", 2), ("seaiceadvschsnow", 77)):
        nml = RunNamelists(oracle.run_dir(RUN))
        nml.file("data.seaice")["seaice_parm01"][key] = [val]
        with pytest.raises(NotImplementedError):
            sa.SeaiceAdvDiffParams.from_namelists(nml)
    nml = RunNamelists(oracle.run_dir(RUN))
    nml.file("data.pkg")["packages"]["usethsice"] = [True]
    with pytest.raises(NotImplementedError):
        sa.SeaiceAdvDiffParams.from_namelists(nml)


def test_pass_table_differs_from_gad():
    """SEAICE_ADVECTION's pass table (seaice_advection.F:299-315) is not GAD_ADVECTION's: interiorOnly only in
    pass 1. Facets 1 and 4 therefore update all rows/columns in pass 2 (GAD: interior only) and facets 2, 3, 5 in
    pass 3."""
    diff = [(f, ip) for f in range(1, 6) for ip in (1, 2, 3)
            if sa.pass_flags(f, ip) != gad.pass_flags(f, ip) and any(sa.pass_flags(f, ip)[2:])]
    assert diff == [(1, 2), (2, 3), (3, 3), (4, 2), (5, 3)], diff
    assert all(sa.pass_flags(f, ip)[2:] == gad.pass_flags(f, ip)[2:] for f in range(1, 6) for ip in (1, 2, 3))


# ------------------------------------------------------------------------------------------------ replay gates
@pytest.mark.parametrize("it", ITS)
def test_gate_advdiff(it):
    """A01-A06 (uTrans, vTrans, advective gFld, afx, afy, gFld after diffusion) and I02 (HEFF, AREA, HSNOW) for the
    three fields, whole tile arrays incl. halos. Achieved: every value bitwise equal (max rel. error 0, 0 differing
    values) at iterations 1, 2, 3."""
    c = case(it)
    out, diag = run_advdiff(params(), c)
    errs = stage_errors(c, out, diag)
    for k, (e, nd) in errs.items():
        print(f"it{it} {k}: rel err {e:.3e}, {nd} values differ")
    for k in out:
        assert np.isfinite(out[k]).all()
    bad = {k: v for k, v in errs.items() if v[0] > TOL}
    assert not bad, bad


@pytest.mark.parametrize("it", ITS)
def test_gate_reg_ridge(it):
    """I03 from the dumped I02 state (replay) and from the JAX SEAICE_ADVDIFF output (chain): HEFF, AREA, HSNOW,
    TICES (7 levels), d_HEFFbyNEG, d_HSNWbyNEG, whole arrays. Achieved: bitwise at iterations 1-3, both ways. Active
    branches (interior points of the I02 input at it 1/2/3): HSNOW < 0 at 1166/820/1004, AREA > SEAICE_area_max at
    2312/2138/2175, HEFF <= siEps (thin-ice reset, TICES = celsius2K; includes open water) at ~93300; HEFF < 0 and
    AREA < 0 never occur."""
    c = case(it)
    rp = rr_params()
    ins = {n: c["ref_I02_" + n] for n in sa.FIELDS}
    got = {k: np.asarray(v) for k, v in _reg_ridge(rp, ins["HEFF"], ins["AREA"], ins["HSNOW"], c["TICES"]).items()}
    out, _ = run_advdiff(params(), c)
    chain = {k: np.asarray(v) for k, v in _reg_ridge(rp, out["HEFF"], out["AREA"], out["HSNOW"], c["TICES"]).items()}
    J = INTERIOR
    h, a, s = (ins[n][J] for n in ("HEFF", "AREA", "HSNOW"))
    print(f"it{it}: HEFF<0 {(h < 0).sum()}, HSNOW<0 {(s < 0).sum()}, AREA<0 {(a < 0).sum()}, "
          f"AREA>area_max {(a > rp.SEAICE_area_max).sum()}, thin {((h <= rp.siEps)).sum()}")
    for name, res in (("replay", got), ("chain", chain)):
        for k in ("HEFF", "AREA", "HSNOW", "TICES", "d_HEFFbyNEG", "d_HSNWbyNEG"):
            e = rel_err(res[k], c["ref_I03_" + k])
            nd = int((res[k] != c["ref_I03_" + k]).sum())
            print(f"it{it} {name} I03.{k}: rel err {e:.3e}, {nd} values differ")
            assert e <= TOL, (name, k, e)


# ------------------------------------------------------------------------------------------------ negative controls
def _gad_table_np(topo, layout):
    """The per-tile tables built with GAD_ADVECTION's pass table instead of SEAICE_ADVECTION's (the tempting wrong
    reuse): interiorOnly in pass 2/3."""
    orig = sa.pass_flags
    try:
        sa.pass_flags = gad.pass_flags
        sa._tables_np.cache_clear()
        tab, act = sa._tables_np(topo, layout)
    finally:
        sa.pass_flags = orig
        sa._tables_np.cache_clear()
    return tab, act


def _run_tables(p, c, tab, act=None, flux_x=None, flux_y=None):
    """advdiff_tiles with explicit tables (and optionally other flux routines), eager-free jit."""
    g = c["g"]
    g2 = {k: np.asarray(getattr(g, k))[:, 0] if np.ndim(getattr(g, k)) == 4 else np.asarray(getattr(g, k))
          for k in sa.GRID2D}
    orig = sa._active
    if act is not None:
        sa._active = lambda params: dict(act)
    orig_tiles = sa.seaice_advection_tiles
    if flux_x is not None:
        sa.seaice_advection_tiles = functools.partial(orig_tiles, flux_x=flux_x, flux_y=flux_y)
    try:
        f = jax.jit(lambda p, tab, g2, *a: sa.advdiff_tiles(p, tab, g2, *a))
        out, diag = f(p, tab, g2, c["uIce"], c["vIce"], c["HEFF"], c["AREA"], c["HSNOW"], c["HEFFM"])
    finally:
        sa._active = orig
        sa.seaice_advection_tiles = orig_tiles
    return {k: np.asarray(v) for k, v in out.items()}, {k: np.asarray(v) for k, v in diag.items()}


def test_gate_negative_controls():
    """Each planted error makes the gate fail (it 1). Measured worst stage rel. error; in brackets the effect on the
    I02 state alone: (1) oneSixth*(1+1e-6): 3.7e-7 [1.0e-8]; (2) no flux limiter (plain DST3, gad.dst3_adv_x/y):
    0.24 [7.5e-3]; (3) GAD_ADVECTION's pass table (interiorOnly in passes 2, 3) instead of SEAICE_ADVECTION's: 0.83
    [0: the table differs only in each facet's LAST pass, whose halo-lane updates reach gFld/afx/afy halos but no
    state]; (4) facet 3 swept in facet 1's order: 0.65 [1.4e-4]; (5) SEAICEdiffKh*(1+1e-6): 4.7e-8 [2.4e-9];
    (6) the pass-2 overlap-only X update of facet 2 dropped: 0.51 [0 at it 1; its effect on the state is shown on
    the all-wet case, test_facet_corners_all_wet]; SEAICE_REG_RIDGE with (7) SEAICE_area_max*(1+1e-6): 1e-6,
    (8) celsius2K*(1+1e-6) (thin-ice TICES reset): 1e-6. Whole-array (halo-inclusive) gates are what catch (3), (6)."""
    c = case(1)
    p = params()

    def worst(out, diag):
        errs = stage_errors(c, out, diag)
        k = max(errs, key=lambda k: errs[k][0])
        return k, errs[k][0]

    def fails(name, out, diag):
        k, e = worst(out, diag)
        e2 = max(rel_err(out[n], c["ref_I02_" + n]) for n in sa.FIELDS)
        print(f"  {name}: worst {k} rel err {e:.3e}; I02 (state) {e2:.3e}")
        return e > TOL

    assert fails("oneSixth", *run_advdiff(dataclasses.replace(p, oneSixth=p.oneSixth * (1 + 1e-6)), c))

    def plain_x(L_, oneSixth, thetaMax, dt, uTrans, uFld, m, t, rdx):
        return gad.dst3_adv_x(L_, oneSixth, dt, jnp.broadcast_to(uTrans, t.shape), uFld, m, t, rdx, 1.0)

    def plain_y(L_, oneSixth, thetaMax, dt, vTrans, vFld, m, t, rdy):
        return gad.dst3_adv_y(L_, oneSixth, dt, jnp.broadcast_to(vTrans, t.shape), vFld, m, t, rdy, 1.0)

    tab = sa.advection_tables(p)
    assert fails("no limiter", *_run_tables(p, c, tab, flux_x=plain_x, flux_y=plain_y))
    gtab, gact = _gad_table_np(p.topo, p.layout)
    assert fails("GAD pass table", *_run_tables(p, c, dict(gtab), act=dict(gact)))
    topo = dataclasses.replace(p.topo, face=tuple(1 if t == 6 else f for t, f in enumerate(p.topo.face)))
    assert fails("facet 3 as facet 1", *run_advdiff(dataclasses.replace(p, topo=topo), c))
    assert fails("diffKh", *run_advdiff(dataclasses.replace(
        p, SEAICEdiffKhHeff=p.SEAICEdiffKhHeff * (1 + 1e-6), SEAICEdiffKhArea=p.SEAICEdiffKhArea * (1 + 1e-6),
        SEAICEdiffKhSnow=p.SEAICEdiffKhSnow * (1 + 1e-6)), c))
    tab2 = dict(tab)
    m = tab2["mX2"].copy()
    m[3:6] = False           # tiles 4-6 = facet 2
    tab2["mX2"] = m
    assert fails("facet-2 overlap update dropped", *_run_tables(p, c, tab2))

    rp = rr_params()
    ins = [c["ref_I02_" + n] for n in ("HEFF", "AREA", "HSNOW")] + [c["TICES"]]

    def rr_fails(name, rp_bad):
        res = _reg_ridge(rp_bad, *ins)
        e = max(rel_err(np.asarray(res[k]), c["ref_I03_" + k]) for k in ("HEFF", "AREA", "HSNOW", "TICES"))
        print(f"  {name}: I03 max rel err {e:.3e}")
        return e > TOL

    assert rr_fails("area_max", dataclasses.replace(rp, SEAICE_area_max=rp.SEAICE_area_max * (1 + 1e-6)))
    assert rr_fails("celsius2K", dataclasses.replace(rp, celsius2K=rp.celsius2K * (1 + 1e-6)))


# ------------------------------------------------------------------------------------------------ properties
def _volume_residual(g, gFld, dt):
    """sum over tile interiors of rA*gFld*dt (the change of ice volume by advection/diffusion; every face flux enters
    two cells with opposite signs, so it telescopes to 0 when the fluxes match across tile and facet edges and none
    crosses the closed boundary) and the scale sum|rA*gFld*dt|."""
    a = (np.asarray(g.rA) * np.asarray(gFld) * dt)[INTERIOR]
    return a.sum(), np.abs(a).sum()


def test_volume_conservation():
    """Ice volume (HEFF), area, snow: sum_interior rA*gFld*dt / sum|.| for the advective tendency (A01/A03/A05) and
    after diffusion (A02/...): Fortran and JAX identical and at round-off (it 1: HEFF 8.8e-17 / 1.1e-16, AREA
    6.7e-16 / 6.6e-16, HSNOW 6.3e-16 / 5.9e-16; asserted < 1e-13). Negative control: facet 3 swept in facet 1's order
    (face fluxes no longer match across facet-3 edges): 2.1e-5 .. 1.2e-4, asserted > 1e-10."""
    c = case(1)
    p = params()
    out, diag = run_advdiff(p, c)
    topo = dataclasses.replace(p.topo, face=tuple(1 if t == 6 else f for t, f in enumerate(p.topo.face)))
    _, bad = run_advdiff(dataclasses.replace(p, topo=topo), c)
    dt = p.SEAICE_deltaTtherm
    for n in sa.FIELDS:
        for kind in ("gAdv", "gFld"):
            r_f, s = _volume_residual(c["g"], c["ref_" + n + "_" + kind], dt)
            r_j, _ = _volume_residual(c["g"], diag[n + "_" + kind], dt)
            r_b, _ = _volume_residual(c["g"], bad[n + "_" + kind], dt)
            print(f"{n} {kind}: residual/scale Fortran {r_f / s:.3e} JAX {r_j / s:.3e} negative control {r_b / s:.3e}")
            assert abs(r_f) / s < 1e-13 and abs(r_j) / s < 1e-13
            assert abs(r_b) / s > 1e-10


def _allwet_case():
    """Synthetic case with wet facet corners (land in LLC90, so the oracle cannot see FILL_CS_CORNER_*): masks 1 on
    every lane, uniform flow in tile coordinates with halos from the signed vector exchange (no flow through the open
    facet edges), a random ice field with exchanged halos and 0 in the facet-corner halos (as exch2 leaves them)."""
    from mitgcm_jax.parallel.exchange import default_exchanger

    ex = default_exchanger()
    c0 = case(1)
    p = params()
    shape = L.shape2d
    u = np.zeros(shape)
    v = np.zeros(shape)
    u[INTERIOR] = 0.2
    v[INTERIOR] = -0.15
    for t in range(L.nTiles):
        if p.topo.face[t] in (1, 2) and p.topo.S_edge[t]:
            v[t, L.jj(1), :] = 0.0      # closed S edge of facets 1, 2 (data.exch2: no neighbour)
    u, v = (np.asarray(a) for a in ex.exch_uv_xy(u, v, True))
    ones = np.ones(shape)
    g = c0["g"].replace(maskW=ones[:, None], maskS=ones[:, None], maskInC=ones, maskC=ones[:, None])
    corners = np.zeros(shape, bool)
    for t in range(L.nTiles):
        sw, se, nw, ne = gad._corners(p.topo, t)
        lo, hiy, hix = slice(0, L.OLy), slice(-L.OLy, None), slice(-L.OLx, None)
        for flag, jj, ii in ((sw, lo, slice(0, L.OLx)), (se, lo, hix), (nw, hiy, slice(0, L.OLx)), (ne, hiy, hix)):
            if flag:
                corners[t, jj, ii] = True
    rnd = np.abs(np.random.default_rng(5).normal(size=shape)) + 0.1
    rnd = np.where(corners, 0.0, np.asarray(ex.exch_xy(rnd)))
    return dict(g=g, uIce=u, vIce=v, HEFF=rnd, AREA=rnd, HSNOW=rnd, HEFFM=ones, corners=corners)


def test_facet_corners_all_wet():
    """FILL_CS_CORNER_TR_RL / _UV_RS inside SEAICE_ADVECTION at work (all-wet synthetic case, no oracle): the global
    ice volume change by advection is at round-off only when the corner fills and the overlap-only updates run.
    Measured: residual/scale 9.2e-18 with the fills, 2.6e-7 without (all fills identity), 5.2e-7 without facet 2's
    pass-2 overlap-only update; asserted < 1e-13 and > 1e-10. The additional fill before the
    pass-1 flux of interior-only tiles (seaice_advection.F:352-356, `ipass.EQ.1`; GAD_ADVECTION fills only for
    overlapOnly) changes halo lanes of gFld/afy only, never a tile interior (and in LLC90 the corners are land, so the
    oracle cannot see it)."""
    c = _allwet_case()
    p = params()
    dt = p.SEAICE_deltaTtherm
    _, diag = run_advdiff(p, c)
    r, s = _volume_residual(c["g"], diag["HEFF_gAdv"], dt)
    tab = sa.advection_tables(p)
    n = L.ny * L.nx
    ident = np.tile(np.arange(n, dtype=np.int32), (L.nTiles, 1))
    nofill = {k: (ident if k[0] == "f" else v) for k, v in tab.items()}
    _, dn = _run_tables(p, c, nofill)
    r_n, _ = _volume_residual(c["g"], dn["HEFF_gAdv"], dt)
    # GAD_ADVECTION's condition: fill before the flux only for overlapOnly (pass 1: facet 3 = tile 7 only)
    gadfill = dict(tab)
    for k in ("fXb1", "fYb1"):
        m = np.array(tab[k])
        keep = np.arange(L.nTiles) == 6
        gadfill[k] = np.where(keep[:, None], m, ident)
    _, dg = _run_tables(p, c, gadfill)
    r_g, _ = _volume_residual(c["g"], dg["HEFF_gAdv"], dt)
    diff = np.zeros(L.shape2d, bool)
    for k in diag:
        diff |= dg[k] != diag[k]
    print(f"all-wet: residual/scale {r / s:.3e}; without corner fills {r_n / s:.3e}; GAD fill conditions "
          f"{r_g / s:.3e} ({diff.sum()} output values differ, {diff[INTERIOR].sum()} in tile interiors)")
    assert abs(r) / s < 1e-13 and abs(r_n) / s > 1e-10
    # the extra fills before the pass-1 fluxes of interior-only tiles (seaice_advection.F:352-356, ipass.EQ.1) change
    # halo lanes only (the filled corner values reach the halo columns/rows updated in the later all-points passes,
    # and gFld = (localTij - iceFld)/dt covers the whole array, :732-736), never a tile interior
    assert diff.any() and not diff[INTERIOR].any()
    # the pass-2 overlap-only X update of facet 2 (invisible in the LLC90 oracle state, whose facet-2 edge halos carry
    # no zonal ice-flux divergence at it 1) is needed for conservation once the halos carry ice
    tab2 = dict(tab)
    m = tab2["mX2"].copy()
    m[3:6] = False
    tab2["mX2"] = m
    _, d2 = _run_tables(p, c, tab2)
    r_2, _ = _volume_residual(c["g"], d2["HEFF_gAdv"], dt)
    print(f"all-wet: facet-2 pass-2 overlap update dropped: residual/scale {r_2 / s:.3e}")
    assert abs(r_2) / s > 1e-10


def test_gradient_vs_finite_difference():
    """d/d(HEFF) and d/d(uIce) of J = sum(w*HEFF_new) + sum(w*AREA_new) (interior, fixed random w on ice points)
    through SEAICE_ADVDIFF (+ SEAICE_REG_RIDGE): finite on every lane (halos, land, corners). The flux limiter makes
    the map piecewise smooth (MIN/MAX in psi, the |Rj|*thetaMax switch, |uTrans|, |CFL|): central differences agree
    only while +-h stays in one piece, so the check uses points in thick ice, away from u = 0, and an h-sweep (the
    best of h/|x| = 1e-3..1e-6 is reported and asserted; the switches are why larger h can fail). Measured (it 1,
    tiles 1, 7, 13): d/dHEFF best rel. err 3.2e-11, 1.8e-12, 5.4e-12; d/duIce 3.8e-8, 1.1e-10, 5.9e-10 (asserted
    < 1e-6). Gradients finite everywhere; with plain JAX division in Rjm/Rj instead of `_ratio`, 197 NaN lanes
    (negative control)."""
    c = case(1)
    p = params()
    rp = rr_params()
    g = c["g"]
    w = np.zeros(L.shape2d)
    ice = (np.asarray(c["HEFF"]) > 0.05) & (np.asarray(c["HEFFM"]) > 0)
    w[INTERIOR] = (np.random.default_rng(0).normal(size=L.shape2d) * ice)[INTERIOR]
    w = jnp.asarray(w)

    def J(heff, u, p, rp, g):
        out, _ = sa.seaice_advdiff(p, g, u, c["vIce"], heff, c["AREA"], c["HSNOW"], c["HEFFM"])
        rr = sa.seaice_reg_ridge(rp, out["HEFF"], out["AREA"], out["HSNOW"], c["TICES"])
        return jnp.sum(w * rr["HEFF"]) + jnp.sum(w * rr["AREA"])

    Jc = jax.jit(J)
    gh, gu = jax.jit(jax.grad(J, argnums=(0, 1)))(jnp.asarray(c["HEFF"]), jnp.asarray(c["uIce"]), p, rp, g)
    gh, gu = np.asarray(gh), np.asarray(gu)
    assert np.isfinite(gh).all() and np.isfinite(gu).all()
    # negative control of the finiteness check: JAX's own division rule for Rjm/Rj (b**-2 underflows for the
    # HSNOW = -1.2e-240 point) gives NaN lanes in d/d(uIce)
    orig = sa._ratio
    try:
        sa._ratio = lambda a, b: a / b
        gu_plain = np.asarray(jax.jit(jax.grad(J, argnums=1))(jnp.asarray(c["HEFF"]), jnp.asarray(c["uIce"]), p, rp, g))
    finally:
        sa._ratio = orig
    print(f"non-finite d/d(uIce) lanes: {(~np.isfinite(gu)).sum()} with _ratio, {(~np.isfinite(gu_plain)).sum()} "
          "with plain division")
    assert (~np.isfinite(gu_plain)).sum() > 0
    h0, u0 = np.asarray(c["HEFF"]), np.asarray(c["uIce"])
    heff = np.asarray(c["HEFF"])
    for t in (0, 6, 12):
        # the first interior point of tile t (in storage order) with thick ice, |u| > 0.02 m/s and a nonzero gradient
        cand = [(t, jj, ii) for jj in range(L.jj(1), L.jj(L.sNy) + 1) for ii in range(L.ii(1), L.ii(L.sNx) + 1)]
        pt = next((q for q in cand if heff[q] > 0.5 and abs(u0[q]) > 0.02 and gh[q] != 0 and gu[q] != 0), None)
        if pt is None:
            print(f"tile {t}: no thick moving ice point")
            continue
        for name, x0, gx in (("HEFF", h0, gh), ("uIce", u0, gu)):
            best = np.inf
            for rh in (1e-3, 1e-4, 1e-5, 1e-6):
                h = rh * abs(x0[pt])
                xp, xm = x0.copy(), x0.copy()
                xp[pt] += h
                xm[pt] -= h
                args = (xp, u0) if name == "HEFF" else (h0, xp)
                argm = (xm, u0) if name == "HEFF" else (h0, xm)
                fd = (float(Jc(*args, p, rp, g)) - float(Jc(*argm, p, rp, g))) / (2 * h)
                e = abs(fd - gx[pt]) / abs(gx[pt])
                print(f"{name} {pt} h={h:.1e}: ad {gx[pt]:.10e} fd {fd:.10e} rel {e:.2e}")
                best = min(best, e)
            assert best < 1e-6, (name, pt, best)


def test_sharded_p4_equals_p1():
    """The kernel is tile-local: under shard_map over 4 devices (tile axis padded 13 -> 16 with inert tiles whose
    tables are identity/empty) the result equals the P=1 run bitwise."""
    from jax.sharding import Mesh, PartitionSpec as P

    c = case(1)
    p = params()
    ref_out, ref_diag = run_advdiff(p, c)
    npad = 16 - L.nTiles

    def pad(a, fill=0):
        a = np.asarray(a)
        return np.concatenate([a, np.full((npad,) + a.shape[1:], fill, a.dtype)])

    n = L.ny * L.nx
    tab = {}
    for k, v in sa.advection_tables(p).items():
        if v.dtype == bool:
            tab[k] = pad(v, False)
        else:
            ident = np.arange(n, dtype=np.int32) + (n if k == "uvfill_v" else 0)
            tab[k] = np.concatenate([v, np.tile(ident, (npad, 1))])
    g = c["g"]
    g2 = {k: pad(np.asarray(getattr(g, k))[:, 0] if np.ndim(getattr(g, k)) == 4 else getattr(g, k))
          for k in sa.GRID2D}
    fields = [pad(c[k]) for k in ("uIce", "vIce", "HEFF", "AREA", "HSNOW", "HEFFM")]
    mesh = Mesh(np.array(jax.devices("cpu")[:4]), ("tile",))
    spec = P("tile")

    def kern(p, tab, g2, *a):
        return sa.advdiff_tiles(p, tab, g2, *a)

    f = jax.jit(jax.shard_map(kern, mesh=mesh, in_specs=(P(), spec, spec) + (spec,) * 6, out_specs=(spec, spec),
                              check_vma=True))
    out, diag = f(p, tab, g2, *fields)
    for k in ref_out:
        np.testing.assert_array_equal(np.asarray(out[k])[:L.nTiles], ref_out[k])
    for k in ref_diag:
        np.testing.assert_array_equal(np.asarray(diag[k])[:L.nTiles], ref_diag[k])
