"""GAD_ADVECTION, multi-dimensional DST3 on the LLC grid (plan Task 16a): replay gates against the Fortran oracle.

Inputs of the replay (what GAD_ADVECTION reads inside THERMODYNAMICS, `temp_integrate.F:277`, `salt_integrate.F:277`):
uFld/vFld (residual flow, T01_residual_flow), theta/salt (last dumped state before THERMODYNAMICS: S04_oceanic_phys;
nothing between DO_OCEANIC_PHYS and THERMODYNAMICS writes theta/salt, staggerTimeStep, implicitIntGravWave=F),
hFacW/hFacS/recip_hFacC (S11_calc_rstar == S06_update_rstar_T, checked below), geometry (G00).
Output: gT_loc (T10_temp_adv) / gS_loc (T20_salt_adv), compared on the whole tile array, halos included: GAD_ADVECTION
writes gTracer on every point (`gad_advection.F:784-789`).

Measured (see test docstrings): JAX == Fortran bitwise on every point, all tiles, both tracers, SMOKE it 1-2 and
FORCED it 1-3 (max rel. error 0).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import gad
from mitgcm_jax.tests import oracle

L = Layout()
# SMOKE it 1 and FORCED it 1 last: the later tests reuse them from the case() cache (dump reads dominate the runtime)
CASES = [(oracle.SMOKE, 2), (oracle.FORCED, 2), (oracle.FORCED, 3), (oracle.FORCED, 1), (oracle.SMOKE, 1)]
TRACERS = {"temp": ("theta", "T10_temp_adv", "gT_loc"), "salt": ("salt", "T20_salt_adv", "gS_loc")}
TR_STAGE = "S04_oceanic_phys"
HFAC_STAGE = "S11_calc_rstar"
INTERIOR = (slice(None), slice(None), L.js(1, L.sNy), L.is_(1, L.sNx))
TOL = 1e-13  # stencil class (KERNEL_GUIDE); achieved: 0


@functools.lru_cache(maxsize=2)
def params(run, tracer):
    return gad.GADParams.from_namelists(RunNamelists(oracle.run_dir(run)), tracer)


@functools.lru_cache(maxsize=3)
def case(run, it):
    ds = oracle.dumpset(run)
    f = functools.partial(oracle.field, ds, it)
    c = dict(g=grid_from_dump(ds, it),
             uFld=f("T01_residual_flow", "uFld"), vFld=f("T01_residual_flow", "vFld"),
             wFld=f("T01_residual_flow", "wFld"),
             hFacW=f(HFAC_STAGE, "hFacW"), hFacS=f(HFAC_STAGE, "hFacS"), recip_hFacC=f(HFAC_STAGE, "recip_hFacC"),
             hFacC=f(HFAC_STAGE, "hFacC"))
    for tr, (state, stage, name) in TRACERS.items():
        c[tr] = f(TR_STAGE, state)
        c["ref_" + tr] = f(stage, name)
    return c


_jit = jax.jit(gad.gad_advection)  # params is a pytree argument (float leaves traced), never closed over


def run_kernel(p, c, tracer_field, fn=None):
    fn = fn or _jit
    return np.asarray(fn(p, c["g"], c["uFld"], c["vFld"], c["wFld"], tracer_field, c["hFacW"], c["hFacS"],
                         c["recip_hFacC"]))


def rel_err(got, ref, region=(Ellipsis,)):
    d = np.abs(got[region] - ref[region]).max()
    return d / np.abs(ref[region]).max()


def gate_errors(got, ref):
    return rel_err(got, ref), rel_err(got, ref, INTERIOR)


# ------------------------------------------------------------------------------------------------ setup checks
def test_topology_matches_exch2_of_the_run():
    """Tile facets and edge flags derived from data.exch2 (w2_set_map_tiles.F, w2_set_tile2tiles.F) agree with
    exch2_myFace / tBasex / tBasey written by the oracle in every dump record header."""
    p = params(oracle.SMOKE, "temp")
    recs = oracle.dumpset(oracle.SMOKE).tiles(1, "T10_temp_adv", "gT_loc")
    facet_nx = {1: 90, 2: 90, 3: 90, 4: 270, 5: 270}
    facet_ny = {1: 270, 2: 270, 3: 90, 4: 90, 5: 90}
    for t in range(L.nTiles):
        r = recs[t + 1]
        f = p.topo.face[t]
        assert r.face == f
        assert p.topo.S_edge[t] == (r.tby == 0) and p.topo.W_edge[t] == (r.tbx == 0)
        assert p.topo.N_edge[t] == (r.tby + L.sNy == facet_ny[f])
        assert p.topo.E_edge[t] == (r.tbx + L.sNx == facet_nx[f])
    assert p.dTtracerLev == (3600.0,) * L.Nr
    assert p == params(oracle.FORCED, "temp")


@pytest.mark.parametrize("run", [oracle.SMOKE, oracle.FORCED])
def test_hfac_unchanged_between_update_rstar_and_calc_rstar(run):
    """CALC_R_STAR (S11) does not touch hFac: the arrays GAD_ADVECTION reads are those of UPDATE_R_STAR(.TRUE.)."""
    ds = oracle.dumpset(run)
    for name in ("hFacW", "hFacS", "recip_hFacC"):
        np.testing.assert_array_equal(oracle.field(ds, 1, "S06_update_rstar_T", name),
                                      oracle.field(ds, 1, HFAC_STAGE, name))


def test_unported_options_raise():
    nml = RunNamelists(oracle.run_dir(oracle.SMOKE))
    nml.file("data")["parm01"]["tempadvscheme"] = [33]
    with pytest.raises(NotImplementedError):
        gad.GADParams.from_namelists(nml, "temp")
    nml = RunNamelists(oracle.run_dir(oracle.SMOKE))
    nml.file("data")["parm01"]["saltimplvertadv"] = [False]
    with pytest.raises(NotImplementedError):
        gad.GADParams.from_namelists(nml, "salt")


# ------------------------------------------------------------------------------------------------ replay gates
@pytest.mark.parametrize("run,it", CASES)
def test_gate_advective_tendency(run, it):
    """T10_temp_adv gT_loc and T20_salt_adv gS_loc: whole tile arrays incl. halos and facet-corner halos.
    Achieved max relative error (whole array, interior): 0, 0 for both tracers at SMOKE it 1-2, FORCED it 1-3."""
    c = case(run, it)
    for tr in TRACERS:
        got = run_kernel(params(run, tr), c, c[tr])
        e_all, e_int = gate_errors(got, c["ref_" + tr])
        print(f"{run} it{it} {tr}: rel err all {e_all:.3e} interior {e_int:.3e}")
        assert np.isfinite(got).all()
        assert e_all <= TOL and e_int <= TOL, (tr, e_all, e_int)


def _identity(tab):
    n = L.ny * L.nx
    return np.tile(np.arange(n, dtype=np.int32), (L.nTiles, 1))


def _run_tables(p, c, tracer_field, tables):
    g = c["g"]
    f = jax.jit(lambda p, tab, g2, vg, *a: gad.advect_tiles(p, tab, g2, vg, *a))
    return np.asarray(f(p, tables, {k: getattr(g, k) for k in gad.GRID2D}, {k: getattr(g, k) for k in gad.VGRID},
                        c["uFld"], c["vFld"], tracer_field, c["hFacW"], c["hFacS"], c["recip_hFacC"]))


def test_gate_negative_controls():
    """Each planted error makes the T10 gate fail (SMOKE it 1, theta): oneSixth * (1+1e-6) (rel. err. 5e-8 / 1e-6
    whole array / interior); facet 3 (tile 7) given facet 1's sweep order; the overlap-only update of facet 5's
    pass 2 dropped; one row missing from an update region. (Dropping FILL_CS_CORNER_TR_RL changes nothing here: the
    LLC90 facet corners are land, tracer 0 and no flow; the fills are tested on an all-wet case below.)"""
    c = case(oracle.SMOKE, 1)
    p = params(oracle.SMOKE, "temp")
    ref = c["ref_temp"]

    def fails(got):
        e_all, e_int = gate_errors(got, ref)
        print(f"  planted error: rel err all {e_all:.3e} interior {e_int:.3e}")
        return e_all > TOL or e_int > TOL   # the gate asserts both

    # 1) constant perturbed by 1e-6 relative
    assert fails(run_kernel(dataclasses.replace(p, oneSixth=p.oneSixth * (1 + 1e-6)), c, c["temp"]))
    # 2) sweep order: tile 7 (facet 3) treated as facet 1
    topo = dataclasses.replace(p.topo, face=tuple(1 if t == 6 else f for t, f in enumerate(p.topo.face)))
    assert fails(run_kernel(dataclasses.replace(p, topo=topo), c, c["temp"]))
    # 3) overlap-only halo update of facet 5, pass 2 (gad_advection.F:681-734) dropped
    tab = gad.gad_tables(p)
    tab["mY2"] = np.zeros_like(tab["mY2"])
    assert fails(_run_tables(p, c, c["temp"], tab))
    # 4) index shift: the X update of pass 1 misses its last row on every tile
    tab = gad.gad_tables(p)
    m = tab["mX1"].copy()
    m[:, L.jj(L.sNy + L.OLy)] = False
    m[:, L.jj(L.sNy)] = False
    tab["mX1"] = m
    assert fails(_run_tables(p, c, c["temp"], tab))


# ------------------------------------------------------------------------------------------------ properties
def _conservation_residual(c, gT, tracer):
    """sum_wet V*gT*dt - dt*sum_wet tracer*div_h(U) over tile interiors. Each wet cell gets
    V*(localTij - tracer) = -dt*(div F - tracer*div U) (gad_advection.F:544-550, :753-759, V*recip_V = 1 to round-off),
    so the remainder is -dt times the global sum of flux divergences, which telescopes to 0 when every face flux is
    the same on both sides of a tile edge. Returns (residual, scale)."""
    g = c["g"]
    dt = 3600.0
    V = g.rA[:, None] * g.drF[None, :, None, None] * c["hFacC"]
    xA = g.dyG[:, None] * g.drF[None, :, None, None] * c["hFacW"]
    yA = g.dxG[:, None] * g.drF[None, :, None, None] * c["hFacS"]
    uT, vT = c["uFld"] * xA, c["vFld"] * yA
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    div = (uT[..., J, L.is_(2, L.sNx + 1)] - uT[..., J, I]) + (vT[..., L.js(2, L.sNy + 1), I] - vT[..., J, I])
    wet = g.maskC[..., J, I] > 0
    a = (V[..., J, I] * gT[..., J, I] * dt)[wet]
    b = (tracer[..., J, I] * div * dt)[wet]
    return a.sum() - b.sum(), np.abs(a).sum() + np.abs(b).sum()


@pytest.mark.parametrize("run", [oracle.SMOKE, oracle.FORCED])
def test_global_conservation(run):
    """Flux form: the global volume integral of the tendency equals sum(T * div_h U) (the non-divergence correction
    of the multi-dim scheme), i.e. face fluxes match across every tile and facet edge. Residual / (sum|a|+sum|b|)
    measured at it 1, Fortran and JAX identical: SMOKE theta 3.6e-16, salt 9.1e-16; FORCED theta 5.9e-17, salt 8.2e-16;
    asserted < 1e-12. Negative control: facet 3 swept in facet 1's order (fluxes across facet-3 edges no longer
    match): 2.9e-9 (SMOKE), 1.6e-8 (FORCED), asserted > 1e-10."""
    c = case(run, 1)
    for tr in TRACERS:
        p = params(run, tr)
        got = run_kernel(p, c, c[tr])
        r_f, s = _conservation_residual(c, c["ref_" + tr], c[tr])
        r_j, _ = _conservation_residual(c, got, c[tr])
        print(f"{run} {tr}: residual/scale Fortran {r_f / s:.3e} JAX {r_j / s:.3e}")
        assert abs(r_f) / s < 1e-12 and abs(r_j) / s < 1e-12
        if tr == "temp":
            topo = dataclasses.replace(p.topo, face=tuple(1 if t == 6 else f for t, f in enumerate(p.topo.face)))
            bad = run_kernel(dataclasses.replace(p, topo=topo), c, c[tr])
            r_b, _ = _conservation_residual(c, bad, c[tr])
            print(f"  negative control {r_b / s:.3e}")
            assert abs(r_b) / s > 1e-10


def test_constant_tracer_stays_constant():
    """DST3 flux of a constant c is exactly uTrans*c (0.5*(u+|u|) is u or 0 exactly), so the update is
    c - dt*r*(u1*c - u0*c - c*(u1-u0)): zero up to round-off on the tile interior. (Not on the outermost halo lanes:
    there the Fortran flux is forced to 0, gad_dst3_adv_x.F:74-78, so i = 2-OLx, sNx+OLx-1 change; Fortran does the
    same.) Real flow, c = 35 everywhere: interior max |gS| dt / c measured 0 (exact), asserted < 1e-13."""
    c = case(oracle.SMOKE, 1)
    p = params(oracle.SMOKE, "salt")
    cst = 35.0
    got = run_kernel(p, c, np.full_like(c["salt"], cst))
    e = np.abs(got[INTERIOR]).max() * 3600.0 / cst
    print(f"constant: interior max |gS| dt / c = {e:.3e}")
    assert e < 1e-13


def _allwet_case():
    """Synthetic case that exercises the facet corners (land in LLC90): every cell wet (hFac = 1, masks 1 on every
    lane), flow uniform in tile coordinates varying with k, halos from the signed vector exchange (so face transports
    agree across tile edges); facet-corner halos stay 0 as exch2 leaves them."""
    from mitgcm_jax.parallel.exchange import default_exchanger

    ex = default_exchanger()
    c0 = case(oracle.SMOKE, 1)
    g = c0["g"]
    shape = c0["temp"].shape
    ones = np.ones(shape)
    k = np.arange(L.Nr)[None, :, None, None]
    u = np.zeros(shape)
    v = np.zeros(shape)
    u[INTERIOR] = np.broadcast_to(0.3 * np.cos(0.1 * k), shape)[INTERIOR]
    v[INTERIOR] = np.broadcast_to(-0.2 + 0.01 * k, shape)[INTERIOR]
    # closed domain: no flow through the open facet edges (data.exch2 facetEdgeLink = 0: S of facets 1, 2 -> the
    # interior v(i,1); E of facets 4, 5 -> the halo u(sNx+1,j), which the exchange leaves at 0)
    topo = params(oracle.SMOKE, "temp").topo
    for t in range(L.nTiles):
        if topo.face[t] in (1, 2) and topo.S_edge[t]:
            v[t, :, L.jj(1), :] = 0.0
    u, v = (np.asarray(a) for a in ex.exch_uv_xy(u, v, True))
    g1 = g.replace(maskW=ones, maskS=ones, maskC=ones, maskInC=np.ones(L.shape2d))
    c = dict(g=g1, uFld=u, vFld=v, wFld=np.zeros(shape), hFacW=ones, hFacS=ones, recip_hFacC=ones, hFacC=ones)
    corners = np.zeros(L.shape2d, bool)
    for t in range(L.nTiles):
        sw, se, nw, ne = gad._corners(params(oracle.SMOKE, "temp").topo, t)
        for flag, jj, ii in ((sw, slice(0, L.OLy), slice(0, L.OLx)), (se, slice(0, L.OLy), slice(-L.OLx, None)),
                             (nw, slice(-L.OLy, None), slice(0, L.OLx)), (ne, slice(-L.OLy, None), slice(-L.OLx, None))):
            if flag:
                corners[t, jj, ii] = True
    return c, ex, corners


def test_facet_corners_all_wet():
    """FILL_CS_CORNER_TR_RL / _UV_RS at work (all-wet synthetic case, no oracle; properties of the Fortran scheme):
    (1) a constant tracer whose facet-corner halos are 0 (never written by exch2) stays constant in every tile interior,
    because every corner point a flux reads is filled first; without the fills of facet 3 (pass 1) and facets 2, 5
    (pass 2) it does not; (2) a random tracer (halos exchanged) is conserved globally (face fluxes identical on both
    sides of every tile and facet edge, corners included). Measured: (1) 7.4e-16 with fills, 3.4e-3 without;
    (2) residual/scale 1.3e-18 with fills, 9.9e-9 without."""
    c, ex, corners = _allwet_case()
    p = params(oracle.SMOKE, "temp")
    cst = 12.0
    tr0 = np.where(corners[:, None], 0.0, cst)
    got = run_kernel(p, c, tr0)
    e = np.abs(got[INTERIOR]).max() * 3600.0 / cst
    tab = gad.gad_tables(p)
    for k in ("fYb1", "fYa1", "fXb2", "fYb2"):
        tab[k] = _identity(tab)
    e_bad = np.abs(_run_tables(p, c, tr0, tab)[INTERIOR]).max() * 3600.0 / cst
    print(f"all-wet constant, zero corners: {e:.3e}; without corner fills: {e_bad:.3e}")
    assert e < 1e-13 and e_bad > 1e-6
    rnd = np.random.default_rng(3).normal(size=tr0.shape)
    rnd = np.where(corners[:, None], 0.0, np.asarray(ex.exch_xy(rnd)))
    got = run_kernel(p, c, rnd)
    r, s = _conservation_residual(c, got, rnd)
    r_bad, _ = _conservation_residual(c, _run_tables(p, c, rnd, tab), rnd)
    print(f"all-wet conservation residual/scale {r / s:.3e}; without corner fills {r_bad / s:.3e}")
    assert abs(r) / s < 1e-12 and abs(r_bad) / s > 1e-10


def test_gradient_vs_finite_difference():
    """d/d(theta) and d/d(uFld) of J = sum(w * gT[interior]) (fixed random w): finite on every lane (halos, dry
    points, corners) and equal to central differences at wet points on facets 1, 3, 5. gT is linear in the tracer
    (one h = 1e-2: rel. err 6.6e-12, 8.6e-13, 1.7e-10; asserted < 1e-7); in uFld it is smooth away from u = 0
    (CFL-dependent weights): h/|u| in 1e-2, 1e-3, 1e-4, plateau at the largest h, best rel. err 2.2e-10, 1.1e-10,
    2.4e-9 (asserted < 1e-6)."""
    c = case(oracle.SMOKE, 1)
    p = params(oracle.SMOKE, "temp")
    g = c["g"]
    wi = np.zeros(c["temp"].shape)
    wi[INTERIOR] = (np.random.default_rng(0).normal(size=c["temp"].shape) * g.maskC)[INTERIOR]
    wi = jnp.asarray(wi)

    def J(tr, u, p, g):
        return jnp.sum(wi * gad.gad_advection(p, g, u, c["vFld"], c["wFld"], tr, c["hFacW"], c["hFacS"],
                                              c["recip_hFacC"]))

    Jc = jax.jit(J)

    def Jj(tr, u):
        return Jc(tr, u, p, g)

    gtr, gu = jax.jit(jax.grad(J, argnums=(0, 1)))(jnp.asarray(c["temp"]), jnp.asarray(c["uFld"]), p, g)
    gtr, gu = np.asarray(gtr), np.asarray(gu)
    assert np.isfinite(gtr).all() and np.isfinite(gu).all()
    tr0, u0 = c["temp"], c["uFld"]
    for t, k, j0, i0 in ((0, 5, 45, 30), (6, 3, 88, 2), (11, 10, 40, 60)):
        # first point at/after (j0, i0) in the tile interior that is wet at C and W with |u| > 1e-3 m/s
        cand = [(t, k, L.jj(j), L.ii(i)) for j in range(j0, L.sNy + 1) for i in range(1, L.sNx + 1)
                if (j > j0 or i >= i0)]
        pt = next(q for q in cand if g.maskC[q] > 0 and g.maskW[q] > 0 and abs(u0[q]) > 1e-3)
        h = 1e-2
        tp, tm = tr0.copy(), tr0.copy()
        tp[pt] += h
        tm[pt] -= h
        fd = (float(Jj(tp, u0)) - float(Jj(tm, u0))) / (2 * h)
        e = abs(fd - gtr[pt]) / abs(gtr[pt])
        print(f"theta {pt}: ad {gtr[pt]:.10e} fd {fd:.10e} rel {e:.2e}")
        assert e < 1e-7
        best = np.inf
        for rh in (1e-2, 1e-3, 1e-4):
            h = rh * abs(u0[pt])
            up, um = u0.copy(), u0.copy()
            up[pt] += h
            um[pt] -= h
            fd = (float(Jj(tr0, up)) - float(Jj(tr0, um))) / (2 * h)
            e = abs(fd - gu[pt]) / abs(gu[pt])
            print(f"uFld {pt} h={h:.1e}: ad {gu[pt]:.10e} fd {fd:.10e} rel {e:.2e}")
            best = min(best, e)
        assert best < 1e-6


# ------------------------------------------------------------------------------------------------ P=4
def test_sharded_p4_equals_p1():
    """The kernel is tile-local: under shard_map over 4 devices (tile axis padded 13 -> 16 with inert tiles whose
    tables are identity/empty) the result equals the P=1 run bitwise."""
    from jax.sharding import Mesh, PartitionSpec as P

    c = case(oracle.SMOKE, 1)
    p = params(oracle.SMOKE, "temp")
    g = c["g"]
    ref = run_kernel(p, c, c["temp"])
    npad = 16 - L.nTiles

    def pad(a, fill=0):
        a = np.asarray(a)
        return np.concatenate([a, np.full((npad,) + a.shape[1:], fill, a.dtype)])

    # padded tiles: empty update masks, identity gathers (uvfill_v indexes the v half of concat(u, v))
    n = L.ny * L.nx
    tab = {}
    for k, v in gad.gad_tables(p).items():
        if v.dtype == bool:
            tab[k] = pad(v, False)
        else:
            ident = np.arange(n, dtype=np.int32) + (n if k == "uvfill_v" else 0)
            tab[k] = np.concatenate([v, np.tile(ident, (npad, 1))])
    g2 = {k: pad(getattr(g, k)) for k in gad.GRID2D}
    vg = {k: np.asarray(getattr(g, k)) for k in gad.VGRID}
    fields = [pad(c[k]) for k in ("uFld", "vFld", "temp", "hFacW", "hFacS", "recip_hFacC")]
    mesh = Mesh(np.array(jax.devices("cpu")[:4]), ("tile",))
    spec = P("tile")

    def kern(p, vg, tab, g2, *a):
        return gad.advect_tiles(p, tab, g2, vg, *a)

    f = jax.jit(jax.shard_map(kern, mesh=mesh, in_specs=(P(), P(), spec, spec) + (spec,) * 6, out_specs=spec,
                              check_vma=True))
    got = np.asarray(f(p, vg, tab, g2, *fields))[:L.nTiles]
    np.testing.assert_array_equal(got, ref)
