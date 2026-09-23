"""MOM_VECINV (plan Task 14b): replay gate against the Fortran oracle, negative controls, gradient.

Replay: the kernel gets the dumped inputs of DYNAMICS (uVel, vVel, wVel from S01_update_rstar_F; hFacC/W/S,
recip_hFacC from S04_oceanic_phys; recip_hFacW/S rebuilt as update_r_star.F:76-79 does; kappaRU/RV from
D01_before_impl_visc; geometry from G00_geometry) and must reproduce the per-level dumps of D00b_mom_vecinv
(gU, gV, guDissip, gvDissip, all points incl. halos) at every dumped iteration of SMOKE (1, 2) and FORCED (1, 2, 3).
Measured (2026-09-23): bitwise equal (0 differing points of 6242600 per field) at all 5 iterations under the
conftest flags (--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp; parameters passed as traced jit arguments);
with XLA's default FMA contraction max rel. error 4.8e-16 (gV, SMOKE it 1). The gate asserts
max|diff| <= 1e-15 * max|ref| and, when the no-FMA flag is set, bitwise equality.
"""

import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import Grid, stack_tiles, unpack_vertical
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import mom_common as mc
from mitgcm_jax.pkgs import mom_vecinv as mv
from mitgcm_jax.tests import oracle

CASES = [(oracle.SMOKE, 1), (oracle.SMOKE, 2), (oracle.FORCED, 1), (oracle.FORCED, 2), (oracle.FORCED, 3)]
OUT = ("gU", "gV", "guDissip", "gvDissip")
IN = ("uVel", "vVel", "wVel", "hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS",
      "kappaRU", "kappaRV")
GRID_FIELDS = ["dxC", "dyC", "dxG", "dyG", "dxV", "dyU", "rA", "rAw", "rAs", "rAz", "recip_dxC", "recip_dyC",
               "recip_dxG", "recip_dyG", "recip_dxV", "recip_dyU", "recip_dxF", "recip_dyF", "recip_rA",
               "recip_rAw", "recip_rAs", "recip_rAz", "fCoriG", "xC", "yC", "xG", "yG", "maskC", "maskW",
               "maskS", "h0FacW", "h0FacS", "viscA4Dfld", "viscA4Zfld", "viscAhDfld", "viscAhZfld", "drF",
               "recip_drF", "recip_drC", "rhoFacF", "rhoFacC"]
RTOL = 1e-15
NO_FMA = "xla_cpu_max_isa=AVX" in os.environ.get("XLA_FLAGS", "")


@functools.lru_cache(maxsize=2)
def params(name):
    return mv.MomVecinvParams.from_namelists(RunNamelists(oracle.run_dir(name)))


@functools.lru_cache(maxsize=2)
def grid(name):
    """The geometry MOM_VECINV reads (G00_geometry of iteration 1; only the needed fields)."""
    ds, L = oracle.dumpset(name), Layout()
    vert = unpack_vertical(ds.tiles(1, "G00_geometry", "vertical")[1].data[0], L)
    f = {n: jnp.asarray(vert[n] if n in vert else stack_tiles(ds, 1, "G00_geometry", n, L)) for n in GRID_FIELDS}
    return Grid(f, L)


@functools.lru_cache(maxsize=1)
def case(name, it):
    """(inputs, reference outputs) of MOM_VECINV at iteration it: see the module docstring for the stages."""
    ds = oracle.dumpset(name)
    f = lambda st, n: oracle.field(ds, it, st, n)  # noqa: E731
    inp = {n: f("S01_update_rstar_F", n) for n in ("uVel", "vVel", "wVel")}
    for n in ("hFacC", "hFacW", "hFacS", "recip_hFacC"):
        inp[n] = f("S04_oceanic_phys", n)
    g = grid(name)
    for n, h, m in (("recip_hFacW", "hFacW", "maskW"), ("recip_hFacS", "hFacS", "maskS")):
        # update_r_star.F:76-79: recip_hFacW = 1/hFacW where maskW != 0; other points keep their value (static,
        # taken from the G00_geometry dump of the same iteration)
        mask = np.asarray(g.f[m])
        inp[n] = np.where(mask != 0.0, 1.0 / np.where(mask != 0.0, inp[h], 1.0), f("G00_geometry", n))
    inp["kappaRU"] = f("D01_before_impl_visc", "kappaRU")
    inp["kappaRV"] = f("D01_before_impl_visc", "kappaRV")
    ref = {n: f("D00b_mom_vecinv", n) for n in OUT}
    return {k: jnp.asarray(v) for k, v in inp.items()}, ref


def make_kernel():
    """A fresh jit of MOM_VECINV(p, g, inputs) with the parameters as an argument (their float leaves traced)."""
    return jax.jit(lambda p, g, inp, visc=None: mv.mom_vecinv(p, g, *(inp[n] for n in IN), visc=visc))


KERNEL = make_kernel()


def errors(out, ref):
    """{field: (max|diff| / max|ref|, number of points not bitwise equal)}"""
    res = {}
    for n in OUT:
        a, b = np.asarray(out[n]), ref[n]
        res[n] = (float(np.max(np.abs(a - b)) / np.max(np.abs(b))), int(np.count_nonzero(a != b)))
    return res


def wet_point(mask, t, k, j, i):
    """The wet point of tile t, level k nearest to (j, i)."""
    jj, ii = np.nonzero(mask[t, k])
    n = np.argmin((jj - j) ** 2 + (ii - i) ** 2)
    return t, k, int(jj[n]), int(ii[n])


def test_params_from_namelists():
    """The V4r4 branch switches come out of the run's namelists as the Fortran derives them; both oracles run
    the same momentum configuration (they differ only in forcing)."""
    p = params(oracle.SMOKE)
    assert p.visc.useVariableVisc and p.visc.useHarmonicVisc and p.visc.useBiharmonicVisc
    assert (p.visc.viscAhD, p.visc.viscAhZ, p.visc.viscAhGrid, p.visc.viscA4D) == (1.0, 1.0, 0.02, 0.0)
    assert p.drag.selectBotDragQuadr == 0 and p.drag.bottomDragQuadratic == 1e-3 and p.bottomDragTerms
    assert p.useJamartWetPoints and p.useCubedSphereExchange and p.selectVortScheme == 1
    q = params(oracle.FORCED)
    assert (q.visc, q.drag, q.useJamartWetPoints, q.selectVortScheme) == (p.visc, p.drag, p.useJamartWetPoints,
                                                                         p.selectVortScheme)


def test_tile_topology_matches_dump():
    """Corner flags / facet numbers of cs_tile_topology agree with the facet and tile offsets the Fortran wrote
    into every dump record (face, tBasex, tBasey)."""
    ds = oracle.dumpset(oracle.SMOKE)
    L = grid(oracle.SMOKE).layout
    topo = mc.cs_tile_topology(L)
    from mitgcm_jax.io.llc import FACET_SHAPE
    recs = ds.tiles(1, "S00_begin", "etaN")
    for t in range(L.nTiles):
        r = recs[t + 1]
        ny_f, nx_f = FACET_SHAPE[r.face]
        assert topo["face"][t] == r.face
        isW, isE, isS, isN = r.tbx == 0, r.tbx + L.sNx == nx_f, r.tby == 0, r.tby + L.sNy == ny_f
        assert (topo["sw"][t], topo["se"][t], topo["nw"][t], topo["ne"][t]) == \
            (isW and isS, isE and isS, isW and isN, isE and isN)


def _fortran_fill(fld, fill4dir, topo, L):
    """Element-by-element transcription of FILL_CS_CORNER_TR_RL (fill_cs_corner_tr_rl.F:165-262, withSigns=F)."""
    out = fld.copy()
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy
    P = lambda i, j: (L.jj(j), L.ii(i))  # noqa: E731  Fortran (i, j) -> python [j, i]
    for t in range(L.nTiles):
        a = out[t]
        for j in range(1, OLy + 1):
            for i in range(1, OLx + 1):
                if fill4dir == 1:
                    if topo["sw"][t]:
                        a[(...,) + P(1 - i, 1 - j)] = a[(...,) + P(1 - j, i)]
                    if topo["se"][t]:
                        a[(...,) + P(sNx + i, 1 - j)] = a[(...,) + P(sNx + j, i)]
                    if topo["nw"][t]:
                        a[(...,) + P(1 - i, sNy + j)] = a[(...,) + P(1 - j, sNy + 1 - i)]
                    if topo["ne"][t]:
                        a[(...,) + P(sNx + i, sNy + j)] = a[(...,) + P(sNx + j, sNy + 1 - i)]
                else:
                    if topo["sw"][t]:
                        a[(...,) + P(1 - i, 1 - j)] = a[(...,) + P(j, 1 - i)]
                    if topo["se"][t]:
                        a[(...,) + P(sNx + i, 1 - j)] = a[(...,) + P(sNx + 1 - j, 1 - i)]
                    if topo["nw"][t]:
                        a[(...,) + P(1 - i, sNy + j)] = a[(...,) + P(j, sNy + i)]
                    if topo["ne"][t]:
                        a[(...,) + P(sNx + i, sNy + j)] = a[(...,) + P(sNx + 1 - j, sNy + i)]
    return out


def test_cs_corner_code_matches_scalar_transcription():
    """The cubed-sphere corner code (FILL_CS_CORNER_TR_RL on hDiv, corner formulas of MOM_CALC_RELVORT3) has no
    effect on the V4r4 oracle outputs: all 8 cube corners and their neighbouring u/v points are dry at every level,
    so velocities there are 0 (measured: switching the corner code off changes no bit of any output or
    intermediate field). No oracle-based negative control can exist for it; it is gated here on random fields
    against an element-by-element transcription of the Fortran loops (negative control: the other direction)."""
    from mitgcm_jax.layout import Layout
    L = Layout()
    topo = mc.cs_tile_topology(L)
    rng = np.random.default_rng(1)
    fld = rng.normal(size=(L.nTiles, 2, L.ny, L.nx))
    for d in (1, 2):
        got = np.asarray(mc.fill_cs_corner_tr_rl(jnp.asarray(fld), d, False, L))
        np.testing.assert_array_equal(got, _fortran_fill(fld, d, topo, L))
        assert not np.array_equal(got, _fortran_fill(fld, 3 - d, topo, L))
    # MOM_CALC_RELVORT3 corner points (mom_calc_relvort3.F:113-286), scalar transcription
    u, v = (rng.normal(size=(L.nTiles, 2, L.ny, L.nx)) for _ in range(2))
    geo = {n: rng.uniform(1.0, 2.0, size=L.shape2d) for n in ("dxC", "dyC", "recip_rAz")}
    vort = np.asarray(mc.mom_calc_relvort3(Grid({n: jnp.asarray(a) for n, a in geo.items()}, L),
                                           jnp.asarray(u), jnp.asarray(v), True))
    dxC, dyC, rAz = geo["dxC"], geo["dyC"], geo["recip_rAz"]
    for t in range(L.nTiles):
        f = topo["face"][t]
        U = lambda i, j: u[t, :, L.jj(j), L.ii(i)] * dxC[t, L.jj(j), L.ii(i)]  # noqa: E731
        V = lambda i, j: v[t, :, L.jj(j), L.ii(i)] * dyC[t, L.jj(j), L.ii(i)]  # noqa: E731
        R = lambda i, j: rAz[t, L.jj(j), L.ii(i)]  # noqa: E731
        N, M = L.sNx + 1, L.sNy + 1
        want = {}
        if topo["sw"][t]:
            want[(1, 1)] = R(1, 1) * ((V(1, 1) - U(1, 1)) + U(1, 0))
        if topo["se"][t]:
            want[(N, 1)] = R(N, 1) * (((-U(N, 1) - V(N - 1, 1)) + U(N, 0)) if f == 2 else
                                      ((-V(N - 1, 1) + U(N, 0)) - U(N, 1)) if f == 4 else
                                      ((U(N, 0) - U(N, 1)) - V(N - 1, 1)))
        if topo["nw"][t]:
            want[(1, M)] = R(1, M) * (((U(1, M - 1) + V(1, M)) - U(1, M)) if f == 1 else
                                      ((-U(1, M) + U(1, M - 1)) + V(1, M)) if f == 3 else
                                      ((V(1, M) - U(1, M)) + U(1, M - 1)))
        if topo["ne"][t]:
            want[(N, M)] = R(N, M) * (((-U(N, M) - V(N - 1, M)) + U(N, M - 1)) if f % 2 == 1 else
                                      ((U(N, M - 1) - U(N, M)) - V(N - 1, M)))
        # an ordinary point for reference (mom_calc_relvort3.F:57-63)
        want[(5, 7)] = R(5, 7) * ((V(5, 7) - V(4, 7)) - (U(5, 7) - U(5, 6)))
        for (i, j), w in want.items():
            np.testing.assert_array_equal(vort[t, :, L.jj(j), L.ii(i)], w)
    plain = np.asarray(mc.mom_calc_relvort3(Grid({n: jnp.asarray(a) for n, a in geo.items()}, L),
                                            jnp.asarray(u), jnp.asarray(v), False))
    assert not np.array_equal(plain, vort)


def test_negative_controls_fail():
    """Planted errors must fail the same comparison: bottomDragQuadratic x (1 + 1e-6); viscAhGrid x (1 + 1e-6);
    the vertical-shear term dropped. (Traced parameters: the first two reuse the compiled kernel.)"""
    name, it = oracle.SMOKE, 1
    p = params(name)
    inp, ref = case(name, it)
    g = grid(name)
    bad_cd = dataclasses.replace(p, drag=dataclasses.replace(
        p.drag, bottomDragQuadratic=p.drag.bottomDragQuadratic * (1 + 1e-6)))
    err = errors(KERNEL(bad_cd, g, inp), ref)
    assert err["guDissip"][0] > RTOL and err["gvDissip"][0] > RTOL, err
    assert err["gU"][1] == 0 or not NO_FMA  # gU does not see the drag
    bad_visc = dataclasses.replace(p, visc=dataclasses.replace(p.visc, viscAhGrid=p.visc.viscAhGrid * (1 + 1e-6)))
    err = errors(KERNEL(bad_visc, g, inp), ref)
    assert err["guDissip"][0] > RTOL and err["gvDissip"][0] > RTOL, err
    orig = mv.mom_vi_u_vertshear
    try:
        mv.mom_vi_u_vertshear = lambda p_, g_, u, w, r: jnp.zeros_like(u)
        err = errors(make_kernel()(p, g, inp), ref)  # fresh jit: retraces with the planted error
    finally:
        mv.mom_vi_u_vertshear = orig
    assert err["gU"][0] > RTOL, err


def test_gradient_finite_and_matches_fd():
    """jax.grad of a weighted sum of all four outputs w.r.t. every input is finite on every lane (dry, halo,
    corner); d/d uVel matches a central finite difference at two wet points (h sweep; the FD sums the output
    differences with the same weights, so the full sum does not cancel). Measured rel. diff (h = 1e-3, 1e-4,
    1e-5 m/s): 4.9e-15, 3.3e-15, 6.8e-13 (Arctic cap, k=1) and 1.4e-16, 1.9e-15, 6.2e-14 (facet 1, k=21)."""
    name, it = oracle.SMOKE, 1
    p = params(name)
    inp, _ = case(name, it)
    g = grid(name)
    rng = np.random.default_rng(0)
    W = {n: jnp.asarray(rng.normal(size=inp["uVel"].shape)) for n in OUT}

    def loss(i, p, g, W):
        o = mv.mom_vecinv(p, g, *(i[n] for n in IN))
        return sum(jnp.sum(W[n] * o[n]) for n in OUT)

    @jax.jit
    def fd(i, p, g, W, e, h):
        op = mv.mom_vecinv(p, g, *(dict(i, uVel=i["uVel"] + h * e)[n] for n in IN))
        om = mv.mom_vecinv(p, g, *(dict(i, uVel=i["uVel"] - h * e)[n] for n in IN))
        return sum(jnp.sum(W[n] * (op[n] - om[n])) for n in OUT) / (2 * h)

    grads = jax.jit(jax.grad(loss))(inp, p, g, W)
    for n in IN:
        assert np.all(np.isfinite(np.asarray(grads[n]))), n
    gu = np.asarray(grads["uVel"])
    assert np.count_nonzero(gu) > 1000
    L = g.layout
    maskW = np.asarray(g.f["maskW"])
    pts = [wet_point(maskW, 6, 0, L.jj(45), L.ii(45)),  # Arctic cap, surface
           wet_point(maskW, 2, 20, L.jj(30), L.ii(60))]  # facet 1, level 21
    for (t, k, j, i) in pts:
        e = jnp.zeros_like(inp["uVel"]).at[t, k, j, i].set(1.0)
        rels = [abs(float(fd(inp, p, g, W, e, h)) - gu[t, k, j, i]) / abs(gu[t, k, j, i])
                for h in (1e-3, 1e-4, 1e-5)]
        print("grad", (t, k, j, i), gu[t, k, j, i], rels)
        assert min(rels) < 1e-8, rels


@pytest.mark.parametrize("name,it", CASES)
def test_replay_mom_vecinv(name, it):
    inp, ref = case(name, it)
    err = errors(KERNEL(params(name), grid(name), inp), ref)
    print(name, it, err)
    for n, (rel, nbit) in err.items():
        assert rel <= RTOL, (n, rel)
        if NO_FMA:
            assert nbit == 0, (n, nbit)
