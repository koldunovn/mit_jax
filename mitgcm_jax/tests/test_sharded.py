"""Sharded exch2 exchanges and Fortran-order global sums (plan Task 7), on 4 fake CPU devices (conftest.py).

The sharded exchanger (parallel/sharded_exchange.py: contiguous tile blocks, padding tiles = replicas of tile 1,
coloured ppermute rounds built from the probed Fortran maps) must be BITWISE equal to the single-device gather
exchanger for every kind at P = 1, 2, 4; its JAX-derived transpose must satisfy the adjoint identity; the global sum
must add the 13 tile partials in the Fortran order whatever P (global_sum_tile.F, GLOBAL_SUM_ORDER_TILES). Negative
controls: a dropped sign, a stale halo (rounds skipped) and a wrong transpose each fail; a psum of device-local sums
(the planted order) differs from the Fortran-order sum. Measured 2026-09-23: rounds per exchange 0 / 1 / 3 at
P = 1 / 2 / 4 (every kind), one collective-permute per round in the compiled exchange, one all-reduce per global sum.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import PartitionSpec as PS

from mitgcm_jax.parallel.exchange import MAP_DIR, SCALAR_KINDS, VECTOR_KINDS, ExchangeMaps, Exchanger
from mitgcm_jax.parallel.global_sum import global_sum_tile
from mitgcm_jax.parallel.shard import tile_mesh
from mitgcm_jax.parallel.sharded_exchange import AXIS, ShardedExchanger, TileBlocks, build_kind, colour_pairs

KINDS = list(SCALAR_KINDS) + list(VECTOR_KINDS)
PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def maps():
    return ExchangeMaps.load(MAP_DIR / "exch_maps_13x90x90.npz")


@pytest.fixture(scope="module")
def ex1(maps):
    return Exchanger(maps)


_SETUPS = {}


def setup_p(maps, P):
    """(blocks, mesh, sharded exchanger placed on the mesh), cached per P."""
    if P not in _SETUPS:
        b = TileBlocks(maps.layout.nTiles, P)
        mesh = tile_mesh(P)
        _SETUPS[P] = (b, mesh, ShardedExchanger.build(maps, b).device_arrays(mesh))
    return _SETUPS[P]


def sharded(fn, mesh, nin, out=PS(AXIS)):
    """jit(shard_map(fn(ex, *arrays))) with the exchanger and every array sharded over tiles."""
    return jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=(PS(AXIS),) * (nin + 1), out_specs=out, check_vma=True))


def exchange_fn(kind):
    if kind in SCALAR_KINDS:
        return lambda e, a: e.scalar(a, kind)
    return lambda e, u, v: e.vector(u, v, kind)


def untouched(maps, kind):
    """Per input array of `kind`: halo points the Fortran exchange neither writes nor reads (as the source of a
    written point). Returns (masks [13, ny, nx] per input, number of never-written points that ARE read)."""
    L = maps.layout
    halo = np.ones(L.shape2d, bool)
    halo[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = False
    names = (kind,) if kind in SCALAR_KINDS else (kind + "_u", kind + "_v")
    read = [np.zeros(L.shape2d, bool).reshape(-1) for _ in names]
    for n in names:
        src, comp, _ = maps.maps[n]
        for c in range(1, len(names) + 1):
            read[c - 1][src[comp == c]] = True
    out, nread = [], 0
    for n, r in zip(names, read):
        nw = (maps.maps[n][1] == 0).reshape(L.shape2d) & halo
        out.append(nw & ~r.reshape(L.shape2d))
        nread += int((nw & r.reshape(L.shape2d)).sum())
    return out, nread


# ------------------------------------------------------------------------------------------------ static checks
def test_no_ragged_a2a_in_package():
    """Banned collective (CLAUDE.md: its JAX transpose is wrong). No source file of the package may name it."""
    token = "ragged_" + "all_to_all"
    hits = [str(p.relative_to(PKG)) for p in PKG.rglob("*.py") if token in p.read_text()]
    assert not hits, hits


def test_blocks_and_rounds(maps):
    """Tile blocks (13 tiles, contiguous, padded with replicas of tile 1) and the ppermute round structure: every round
    is a partial permutation; rounds per exchange = 0 (P=1), 1 (P=2), 3 (P=4) for every kind; every remote value is
    requested once per (sender, receiver) pair."""
    for P, Tloc, rounds in ((1, 13, 0), (2, 7, 1), (4, 4, 3)):
        b = TileBlocks(13, P)
        assert (b.Tloc, b.Tpad) == (Tloc, P * Tloc)
        np.testing.assert_array_equal(b.source_tile, list(range(13)) + [0] * (b.Tpad - 13))
        for kind in KINDS:
            outs = (kind,) if kind in SCALAR_KINDS else (kind + "_u", kind + "_v")
            plan, tab = build_kind(maps.maps, outs, b, maps.layout)
            assert len(plan.perms) == rounds, (P, kind, plan.perms)
            for perm in plan.perms:
                assert len({s for s, _ in perm}) == len(perm) == len({d for _, d in perm})
            assert tab["send_ok"].sum() <= plan.total * P
    assert colour_pairs([(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)], 3) == [((0, 1), (1, 2), (2, 0)),
                                                                               ((0, 2), (1, 0), (2, 1))]


# ------------------------------------------------------------------------------------------------ exchanges
def test_sharded_exchange_equals_gather(maps, ex1):
    """Every kind, 2-D and 3-D fields (random values on every point, halos included), P = 1, 2, 4: the sharded
    exchange is bitwise the single-device exchange on the 13 real tiles, and every padding tile is a bitwise copy of
    tile 1's result."""
    L = maps.layout
    rng = np.random.default_rng(1)
    for P in (1, 2, 4):
        b, mesh, exs = setup_p(maps, P)
        for kind in KINDS:
            f = sharded(exchange_fn(kind), mesh, 1 if kind in SCALAR_KINDS else 2)
            for shape in (L.shape2d, (L.nTiles, 3, L.ny, L.nx)):
                ins = [rng.normal(size=shape) for _ in range(1 if kind in SCALAR_KINDS else 2)]
                ref = ex1.scalar(*ins, kind) if kind in SCALAR_KINDS else ex1.vector(*ins, kind)
                got = f(exs, *(b.pad(a) for a in ins))
                for r, gt in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
                    gt = np.asarray(gt)
                    np.testing.assert_array_equal(gt[:13], np.asarray(r), err_msg=f"P={P} {kind} {shape}")
                    for t in range(13, b.Tpad):
                        np.testing.assert_array_equal(gt[t], gt[0], err_msg=f"P={P} {kind} padding tile {t}")


def test_halo_poison_exchange(maps, ex1):
    """NaN on the halo points the Fortran exchange neither writes nor reads (open facet edges; some never-written
    edge-halo points ARE read: exch2's corner pass copies them into a neighbour's corner halo, 128 for T): the sharded
    exchange (P=4) leaves every other point bitwise at the clean result, and the poisoned points stay NaN."""
    L = maps.layout
    b, mesh, exs = setup_p(maps, 4)
    rng = np.random.default_rng(2)
    for kind in ("T", "UVs", "UVn", "Z"):
        names = (kind,) if kind in SCALAR_KINDS else (kind + "_u", kind + "_v")
        f = sharded(exchange_fn(kind), mesh, len(names))
        clean = [rng.normal(size=L.shape2d) for _ in names]
        nw, _ = untouched(maps, kind)
        assert all(m.sum() > 600 for m in nw), kind
        dirty = [np.where(m, np.nan, a) for a, m in zip(clean, nw)]
        ref = jax.tree.leaves(f(exs, *(b.pad(a) for a in clean)))
        got = jax.tree.leaves(f(exs, *(b.pad(a) for a in dirty)))
        for r, gt, m in zip(ref, got, nw):
            r, gt = np.asarray(r)[:13], np.asarray(gt)[:13]
            np.testing.assert_array_equal(gt[~m], r[~m], err_msg=kind)
            assert np.all(np.isnan(gt[m])), kind


def test_sharded_exchange_adjoint_identity(maps, ex1):
    """<E x, y> = <x, E^T y> for the sharded scalar and vector exchanges (P=4; E^T is JAX's transpose of gather +
    ppermute), and E^T y on the real tiles equals the single-device transpose when y is 0 on the padding tiles."""
    L = maps.layout
    b, mesh, exs = setup_p(maps, 4)
    rng = np.random.default_rng(3)
    for kind in ("T", "UVs"):
        n = 1 if kind == "T" else 2
        f = sharded(exchange_fn(kind), mesh, n)
        xs = [jnp.asarray(b.pad(rng.normal(size=L.shape2d))) for _ in range(n)]
        ys = [np.zeros((b.Tpad,) + L.shape2d[1:]) for _ in range(n)]
        for y in ys:
            y[:13] = rng.normal(size=L.shape2d)
        ex_x, vjp = jax.vjp(lambda *a: f(exs, *a), *xs)
        ex_x = jax.tree.leaves(ex_x)
        eTy = vjp(tuple(jnp.asarray(y) for y in ys) if n == 2 else jnp.asarray(ys[0]))
        lhs = sum(float(jnp.vdot(a, y)) for a, y in zip(ex_x, ys))
        rhs = sum(float(jnp.vdot(x, t)) for x, t in zip(xs, eTy))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-13)
        fn1 = (lambda a: ex1.scalar(a, kind)) if n == 1 else (lambda u, v: ex1.vector(u, v, kind))
        _, vjp1 = jax.vjp(fn1, *(x[:13] for x in xs))
        ref = vjp1(tuple(jnp.asarray(y[:13]) for y in ys) if n == 2 else jnp.asarray(ys[0][:13]))
        for r, t in zip(ref, eTy):
            np.testing.assert_allclose(np.asarray(t)[:13], np.asarray(r), rtol=1e-15, atol=1e-15)


def test_exchange_negative_controls(maps, ex1):
    """Each planted error fails its gate: (a) a dropped sign in the sharded tables -> differs from the gather
    exchange; (b) a stale halo (the ppermute rounds skipped: remote halo points keep their old values) -> differs;
    (c) a wrong transpose (custom_vjp whose backward is the forward exchange) -> adjoint identity fails."""
    L = maps.layout
    b, mesh, exs = setup_p(maps, 4)
    rng = np.random.default_rng(4)
    u, v = rng.normal(size=L.shape2d), rng.normal(size=L.shape2d)
    ref_u, ref_v = (np.asarray(a) for a in ex1.vector(u, v, "UVs"))
    f = sharded(exchange_fn("UVs"), mesh, 2)
    # (a) dropped sign
    bad = jax.tree.map(lambda a: a, exs)
    bad.tabs = dict(bad.tabs)
    bad.tabs["UVs"] = dict(bad.tabs["UVs"], sign=jnp.abs(bad.tabs["UVs"]["sign"]))
    gu, gv = (np.asarray(a)[:13] for a in f(bad, b.pad(u), b.pad(v)))
    assert not (np.array_equal(gu, ref_u) and np.array_equal(gv, ref_v))
    # (b) stale halo: points whose source is on another device keep their own value
    stale = jax.tree.map(lambda a: a, exs)
    stale.tabs = dict(stale.tabs)
    t = stale.tabs["UVs"]
    N = t["loc"].shape[-1]
    stale.tabs["UVs"] = dict(t, keep=t["keep"] | (t["loc"] >= 2 * N))
    gu, gv = (np.asarray(a)[:13] for a in f(stale, b.pad(u), b.pad(v)))
    assert not (np.array_equal(gu, ref_u) and np.array_equal(gv, ref_v))
    # (c) wrong transpose
    fT = sharded(exchange_fn("T"), mesh, 1)

    @jax.custom_vjp
    def wrong(x):
        return fT(exs, x)

    wrong.defvjp(lambda x: (fT(exs, x), None), lambda _, ct: (fT(exs, ct),))
    x = jnp.asarray(b.pad(rng.normal(size=L.shape2d)))
    y = np.zeros((b.Tpad,) + L.shape2d[1:])
    y[:13] = rng.normal(size=L.shape2d)
    ex_x, vjp = jax.vjp(wrong, x)
    lhs, rhs = float(jnp.vdot(ex_x, y)), float(jnp.vdot(x, vjp(jnp.asarray(y))[0]))
    assert abs(lhs - rhs) > 1e-6 * abs(lhs)


def test_exchange_collective_budget(maps):
    """Compiled HLO (P=4): one scalar or vector exchange = exactly one collective-permute per round (3), nothing else
    collective; one global sum = exactly one all-reduce. Guards against silent all-gather fallbacks and the banned
    ragged all-to-all."""
    L = maps.layout
    b, mesh, exs = setup_p(maps, 4)

    def count(txt, op):
        return sum(1 for line in txt.splitlines() if f" {op}(" in line or f" {op}-start(" in line)

    x = b.pad(np.zeros(L.shape2d))
    for kind, n in (("T", 1), ("UVs", 2)):
        txt = sharded(exchange_fn(kind), mesh, n).lower(exs, *([x] * n)).compile().as_text()
        assert count(txt, "collective-permute") == exs.rounds(kind) == 3, kind
        for op in ("all-gather", "all-to-all", "all-reduce", "ragged-all-to-all", "reduce-scatter"):
            assert count(txt, op) == 0, (kind, op)
    gs = sharded(lambda e, a: e.global_sum_tile(a), mesh, 1, out=PS())
    txt = gs.lower(exs, np.zeros(b.Tpad)).compile().as_text()
    assert count(txt, "all-reduce") == 1 and count(txt, "collective-permute") == 0


# ------------------------------------------------------------------------------------------------ global sums
def test_global_sum_fortran_order_every_P(maps):
    """GLOBAL_SUM_TILE_RL: 0 + tile 1 + ... + tile 13 whatever P. Per-tile partials with a wide dynamic range
    (the order matters at 1 ulp), 20 random draws: the sharded sum at P = 1, 2, 4 is bitwise the serial ordered sum,
    and global_max is the serial max. Negative control (planted order): a psum of device-local sums at P=4 differs
    from the ordered sum on at least one draw."""
    rng = np.random.default_rng(5)
    draws = [rng.normal(size=13) * 10.0 ** rng.integers(-8, 9, size=13) for _ in range(20)]
    ref = [float(global_sum_tile(jnp.asarray(d))) for d in draws]
    planted_differs = False
    for P in (1, 2, 4):
        b, mesh, exs = setup_p(maps, P)
        gs = sharded(lambda e, a: (e.global_sum_tile(a), e.global_max(a[:, None, None])), mesh, 1, out=(PS(), PS()))
        naive = jax.jit(jax.shard_map(lambda a: lax.psum(jnp.sum(a[:b.Tloc] * (jnp.arange(b.Tloc) + lax.axis_index(
            AXIS) * b.Tloc < 13)), AXIS), mesh=mesh, in_specs=PS(AXIS), out_specs=PS(), check_vma=True))
        for d, r in zip(draws, ref):
            s, m = gs(exs, b.pad(d))
            assert float(s) == r, (P, float(s), r)
            assert float(m) == float(np.max(d))
            if P == 4 and float(naive(b.pad(d))) != r:
                planted_differs = True
    assert planted_differs
