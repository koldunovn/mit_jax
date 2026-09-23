"""Sharded FORWARD_STEP and CG2D (plan Task 7): the single-device kernels under jax.shard_map(check_vma=True) over the
tile axis (4 fake CPU devices, conftest.py), with the sharded exchanger and Fortran-order global sums, are BITWISE
equal to the single-device run.

  - CG2D from the FORCED oracle's C01 inputs at P = 1, 2, 4: solution (interior + halos as CG2D leaves them) bitwise
    equal to the single-device solve and to the oracle's C02, 164 iterations at every P; the implicit-derivative
    gradient d/d(cg2d_b) at P=4 equals the single-device gradient to rounding (tight transpose CG, tree sums per tile)
    and is exactly 0 on the padding tiles (the padded operator is not symmetric: cg2d zeroes them, ex.zero_padding).
  - FORWARD_STEP (flux-forced, FORCED oracle iteration 1, grid from the run's files as test_step_fluxforced.py) at
    P = 4 equals the single-device step on EVERY state field bitwise (90 fields), cg2d 164 iterations, padding tiles
    are bitwise replicas of tile 1; the compiled step stays within its collective budget and has no all-gather. (P=2:
    also bitwise, measured 2026-09-23; not repeated here to save a compile.)
  - Gradient of the step (d/d theta0 of a weighted sum of theta and etaN after the step) at P=4 equals the
    single-device gradient to rounding (measured max rel. 1.6e-16: scatter-add order of the transposes) and is exactly
    0 on the padding tiles.
  - Halo poison, GAD advection at P=4: NaN on the theta halo points no exchange writes or reads (open Antarctic facet
    edges) changes no interior wet point; the NaN stays within 2 points of the poisoned points, in the 4 edge tiles.
Measured 2026-09-23 (dev node, 16 cores): step 6.7 s single device, 5.5 s at P=4 (fake devices share the cores).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as PS

from mitgcm_jax.core import cg2d as cg
from mitgcm_jax.core import free_surface as fs
from mitgcm_jax.core.forward_step import forward_step
from mitgcm_jax.model import setup
from mitgcm_jax.parallel.exchange import MAP_DIR, ExchangeMaps, Exchanger
from mitgcm_jax.parallel.shard import ShardedModel, tile_mesh
from mitgcm_jax.parallel.sharded_exchange import AXIS, ShardedExchanger, TileBlocks
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.pkgs import gad as gad_mod
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle
from mitgcm_jax.tests.test_cg2d import OPS, load_grid
from mitgcm_jax.tests.test_sharded import untouched

# compiled-step collective budget at P=4 (static HLO op counts, measured 2026-09-23): 23 exchange sites x 3 ppermute
# rounds (the cg2d loop body counted once), 9 all-reduces after XLA's combiner (global sums/max, diagnostics). A change
# is a regression (an extra exchange, an all-gather fallback) or needs a deliberate update here.
STEP_BUDGET_P4 = {"collective-permute": 69, "all-reduce": 9}


@pytest.fixture(scope="module")
def maps():
    return ExchangeMaps.load(MAP_DIR / "exch_maps_13x90x90.npz")


def hlo_count(txt, op):
    return sum(1 for line in txt.splitlines() if f" {op}(" in line or f" {op}-start(" in line)


# ------------------------------------------------------------------------------------------------ CG2D
@pytest.fixture(scope="module")
def cg_case():
    ds = oracle.dumpset(oracle.FORCED)
    nml = RunNamelists(oracle.run_dir(oracle.FORCED))
    g = load_grid(ds, 1)
    p = fs.FreeSurfParams.from_namelists(nml)
    cp = cg.Cg2dParams.from_namelists(nml, cg.ini_cg2d_norm(g, g.h0FacW, g.h0FacS, p.implicSurfPress,
                                                             p.implicDiv2DFlow))
    f = lambda stage, n: oracle.field(ds, 1, stage, n)  # noqa: E731
    return (cp, tuple(f("C01_cg2d_inputs", n) for n in OPS), f("C01_cg2d_inputs", "cg2d_b"),
            f("C01_cg2d_inputs", "cg2d_x"), f("C02_cg2d_solution", "cg2d_x"))


def test_cg2d_sharded_every_P(maps, cg_case):
    cp, ops, b, x0, c02 = cg_case
    ex1 = Exchanger(maps)
    one = jax.jit(lambda cp, ops, b, x0: cg.cg2d_solve(cp, ex1, ops, b, x0))
    x1, d1 = one(cp, ops, b, x0)
    np.testing.assert_array_equal(np.asarray(d1["x_fortran"]), c02)
    assert int(d1["numIters"]) == 164
    w = np.random.default_rng(0).normal(size=b.shape)
    g1 = np.asarray(jax.grad(lambda bb: jnp.sum(w * one(cp, ops, bb, x0)[0]))(jnp.asarray(b)))
    for P in (1, 2, 4):
        blk = TileBlocks(13, P)
        mesh = tile_mesh(P)
        exs = ShardedExchanger.build(maps, blk, kinds=["T"]).device_arrays(mesh)

        def body(cp, ex, ops, bb, xx):
            x, d = cg.cg2d_solve(cp, ex, ops, bb, xx)
            return x, d["x_fortran"], ex.first_device(d["numIters"])

        fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(PS(),) + (PS(AXIS),) * 4,
                                   out_specs=(PS(AXIS), PS(AXIS), PS()), check_vma=True))
        pad = lambda a: blk.pad(np.asarray(a))  # noqa: E731
        opsP = tuple(pad(a) for a in ops)
        x, xf, n = fn(cp, exs, opsP, pad(b), pad(x0))
        np.testing.assert_array_equal(np.asarray(x)[:13], np.asarray(x1), err_msg=f"P={P}")
        np.testing.assert_array_equal(np.asarray(xf)[:13], c02, err_msg=f"P={P}")
        assert int(n) == 164, (P, int(n))
        if P == 4:
            wP = np.zeros((blk.Tpad,) + b.shape[1:])
            wP[:13] = w
            gP = np.asarray(jax.grad(lambda bb: jnp.sum(wP * fn(cp, exs, opsP, bb, pad(x0))[0]))(jnp.asarray(pad(b))))
            rel = np.abs(gP[:13] - g1).max() / np.abs(g1).max()
            assert rel < 1e-11, rel
            assert np.all(gP[13:] == 0.0)


# ------------------------------------------------------------------------------------------------ FORWARD_STEP
@pytest.fixture(scope="module")
def run():
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir)
    st = state_from_dump(ds, 1)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    nml = RunNamelists(rundir)
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    myTime, myIter = exf_mod.model_time(nml, 1)
    bufs, facs, _ = loader.load(myTime, myIter)
    exf_in = {"bufs": bufs, "facs": facs, "myTime": myTime}
    st1, aux = jax.jit(lambda P, g, kLowC, st, exf_in: forward_step(P, g, ex, kLowC, st, exf_in))(
        P, g, kLowC, st, exf_in)
    assert int(aux["cg2d"]["numIters"]) == 164
    ref = {k: np.asarray(v) for k, v in st1.f.items()}
    return ds, P, g, kLowC, st, exf_in, ref


def test_step_sharded_equals_single_device(run):
    ds, P, g, kLowC, st, exf_in, ref = run
    nproc = 4
    sm = ShardedModel(g, nproc)
    f, it = sm.shard_state(st)
    k, e = sm.shard_tiles(kLowC), sm.shard_exf(exf_in)
    f1, it1, info = sm.step(P, k, f, it, e)
    assert int(it1) == 2 and int(info["numIters"]) == 164
    assert all(int(info[c]) == 0 for c in ("icntc1", "icntw", "icnts", "icntc2"))
    assert set(f1) == set(ref)
    for name, a in f1.items():
        a = np.asarray(a)
        np.testing.assert_array_equal(a[:13], ref[name], err_msg=f"P={nproc} {name}")
        for t in range(13, a.shape[0]):
            np.testing.assert_array_equal(a[t], a[0], err_msg=f"P={nproc} {name} padding tile {t}")
    for name in ("theta", "salt", "uVel", "vVel", "etaN", "GGL90TKE"):
        np.testing.assert_array_equal(np.asarray(f1[name])[:13], oracle.field(ds, 2, "S00_begin", name))
    if nproc == 4:
        txt = sm.step_fn(tuple(e)).lower(P, sm.g, sm.ex, k, f, it, e).compile().as_text()
        counts = {op: hlo_count(txt, op) for op in STEP_BUDGET_P4}
        print("P=4 step collectives:", counts)
        assert counts["collective-permute"] == STEP_BUDGET_P4["collective-permute"], counts
        assert counts["all-reduce"] <= STEP_BUDGET_P4["all-reduce"], counts
        for op in ("all-gather", "all-to-all", "ragged-all-to-all", "reduce-scatter"):
            assert hlo_count(txt, op) == 0, op


def test_step_sharded_gradient(run):
    ds, P, g, kLowC, st, exf_in, ref = run
    from mitgcm_jax.state import State
    ex1 = Exchanger(ExchangeMaps.load(MAP_DIR / "exch_maps_13x90x90.npz"))
    L = g.layout
    rng = np.random.default_rng(0)
    inner = np.zeros(L.shape2d, bool)
    inner[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = True
    wT = rng.normal(size=st.f["theta"].shape) * np.asarray(g.maskC) * inner[:, None]
    wE = rng.normal(size=st.f["etaN"].shape) * np.asarray(g.maskInC) * inner

    def loss1(th, P, g, kLowC, f0, it, exf_in):
        s1, _ = forward_step(P, g, ex1, kLowC, State(dict(f0, theta=th), it), exf_in)
        return jnp.sum(wT * s1.f["theta"]) + jnp.sum(wE * s1.f["etaN"])

    g1 = np.asarray(jax.jit(jax.grad(loss1))(jnp.asarray(st.f["theta"]), P, g, kLowC, st.f, st.it, exf_in))
    sm = ShardedModel(g, 4)
    f, it = sm.shard_state(st)
    k, e = sm.shard_tiles(kLowC), sm.shard_exf(exf_in)
    step = sm.step_fn(tuple(e))
    wTp, wEp = sm.blocks.pad(wT), sm.blocks.pad(wE)
    wTp[13:], wEp[13:] = 0.0, 0.0

    def lossP(th, P, gg, exs, k, f0, it, e):
        f1, _, _ = step(P, gg, exs, k, dict(f0, theta=th), it, e)
        return jnp.sum(wTp * f1["theta"]) + jnp.sum(wEp * f1["etaN"])

    gP = np.asarray(jax.jit(jax.grad(lossP))(f["theta"], P, sm.g, sm.ex, k, f, it, e))
    assert np.isfinite(g1).all() and np.abs(g1).max() > 0
    rel = np.abs(gP[:13] - g1).max() / np.abs(g1).max()
    assert rel < 1e-13, rel
    assert np.all(gP[13:] == 0.0)


# ------------------------------------------------------------------------------------------------ halo poison
def test_gad_halo_poison_p4(run, maps):
    """GAD_ADVECTION at P=4 with NaN on the theta halo points no exchange writes or reads: the interior wet points are
    unchanged, and every changed point lies within 2 points of a poisoned point (tile-local stencil; nothing reaches
    another tile), in the four tiles with an open facet edge."""
    ds, P, g, kLowC, st, exf_in, ref = run
    sm = ShardedModel(g, 4)
    L = g.layout
    nw = untouched(maps, "T")[0][0]

    def body(p, g, u, v, w, tr, hW, hS, rhC):
        return gad_mod.gad_advection(p, g, u, v, w, tr, hW, hS, rhC)

    fn = jax.jit(jax.shard_map(body, mesh=sm.mesh, in_specs=(PS(), sm._gspec) + (PS(AXIS),) * 7,
                               out_specs=PS(AXIS), check_vma=True))
    f = st.f
    args = [f["uVel"], f["vVel"], f["wVel"], f["theta"], f["hFacW"], f["hFacS"], f["recip_hFacC"]]
    clean = np.asarray(fn(P.gadT, sm.g, *(sm.put_tiles(a) for a in args)))[:13]
    args[3] = np.where(nw[:, None], np.nan, f["theta"])
    dirty = np.asarray(fn(P.gadT, sm.g, *(sm.put_tiles(a) for a in args)))[:13]
    changed = (~((dirty == clean) | (np.isnan(dirty) & np.isnan(clean)))).any(axis=1)   # [13, ny, nx]
    assert changed.any()
    J, I = slice(L.OLy, L.OLy + L.sNy), slice(L.OLx, L.OLx + L.sNx)
    wet = np.asarray(g.maskC).max(axis=1) > 0
    assert not (changed[:, J, I] & wet[:, J, I]).any()
    near = nw.copy()
    for _ in range(2):   # chessboard dilation by 2 points, within each tile
        p = np.pad(near, ((0, 0), (1, 1), (1, 1)))
        near = np.max([p[:, 1 + dj:1 + dj + L.ny, 1 + di:1 + di + L.nx] for dj in (-1, 0, 1) for di in (-1, 0, 1)],
                      axis=0)
    assert not (changed & ~near).any()
    assert sorted(np.nonzero(changed.any(axis=(1, 2)))[0] + 1) == [1, 4, 10, 13]
