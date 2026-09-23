"""exch2 halo exchanges (plan Task 7): the gather maps reproduce the Fortran exchanges on real model fields.

The maps come from the Fortran exchange routines themselves (index-coded probe, scripts/make_exch_maps.py); these
tests check them on fields the model exchanged during the run: zero every halo of a dumped field, exchange it in
JAX, and compare with the dumped field bitwise (untouched halo points keep the dumped value, as in Fortran).
Negative controls: a flipped sign and a wrong source both fail the same comparison.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import stack_tiles
from mitgcm_jax.layout import Layout
from mitgcm_jax.parallel.exchange import ExchangeMaps, Exchanger, MAP_DIR
from mitgcm_jax.tests import oracle

L = Layout()


@pytest.fixture(scope="module")
def ds():
    return oracle.dumpset(oracle.SMOKE)


@pytest.fixture(scope="module")
def maps():
    return ExchangeMaps.load(MAP_DIR / "exch_maps_13x90x90.npz")


@pytest.fixture(scope="module")
def ex(maps):
    return Exchanger(maps)


def _halo_mask(nd):
    m = np.ones(L.shape2d, bool)
    m[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = False
    return m if nd == 2 else m[:, None]


def _written(maps, kind):
    return (maps.maps[kind][1] > 0).reshape(L.shape2d)


def _refill(ex, maps, kind, fields):
    """Zero the halo points the exchange writes, exchange, return JAX result(s)."""
    out = []
    for name, a in fields:
        w = _written(maps, kind + ("_u" if len(fields) == 2 and name == fields[0][0] else
                                   "_v" if len(fields) == 2 else ""))
        w = w if a.ndim == 3 else w[:, None]
        out.append(np.where(w, 0.0, a))
    if len(out) == 1:
        return [np.asarray(ex.scalar(out[0], kind))]
    return [np.asarray(x) for x in ex.vector(out[0], out[1], kind)]


def test_probe_decodes_consistently(maps):
    """Every written halo point has a source inside a tile interior; scalar maps never swap or flip."""
    for k, (src, comp, sign) in maps.maps.items():
        t, rem = np.divmod(src, L.ny * L.nx)
        j, i = np.divmod(rem, L.nx)
        w = comp > 0
        assert np.all((j[w] >= L.OLy) & (j[w] < L.OLy + L.sNy) & (i[w] >= L.OLx) & (i[w] < L.OLx + L.sNx)), k
        if k in ("T", "Z", "3D"):
            assert np.all(comp[w] == 1) and np.all(sign == 1), k
    # every C-point halo point is written except the four open Antarctic facet edges (facets 1, 2 south; 4, 5 east:
    # 4 x 90 x OL) and 8 facet-corner blocks (OL x OL): 1440 + 128
    unw = ~_written(maps, "T") & _halo_mask(2)
    assert unw.sum() == 4 * 90 * 4 + 8 * 16, unw.sum()


@pytest.mark.parametrize("stage,name", [("S00_begin", "theta"), ("S00_begin", "salt"), ("S00_begin", "etaN")])
def test_scalar_exchange_reproduces_fortran_halos(ds, ex, maps, stage, name):
    a = stack_tiles(ds, 1, stage, name, L)
    got, = _refill(ex, maps, "T", [(name, a)])
    np.testing.assert_array_equal(got, a)


def test_vector_exchange_reproduces_fortran_halos(ds, ex, maps):
    u = stack_tiles(ds, 1, "S00_begin", "uVel", L)
    v = stack_tiles(ds, 1, "S00_begin", "vVel", L)
    gu, gv = _refill(ex, maps, "UVs", [("u", u), ("v", v)])
    np.testing.assert_array_equal(gu, u)
    np.testing.assert_array_equal(gv, v)


def test_negative_controls_fail(ds, maps):
    u = stack_tiles(ds, 1, "S00_begin", "uVel", L)
    v = stack_tiles(ds, 1, "S00_begin", "vVel", L)
    src, comp, sign = maps.maps["UVs_u"]
    # flip every sign of the u map (only matters where a swapped/negated source exists)
    bad = dict(maps.maps)
    bad["UVs_u"] = (src, comp, -sign)
    gu, _ = _refill(Exchanger(ExchangeMaps(L, bad)), maps, "UVs", [("u", u), ("v", v)])
    assert not np.array_equal(gu, u)
    # shift every source by one point in i
    bad = dict(maps.maps)
    bad["UVs_u"] = (np.where(comp > 0, src + 1, src).astype(np.int32), comp, sign)
    gu, _ = _refill(Exchanger(ExchangeMaps(L, bad)), maps, "UVs", [("u", u), ("v", v)])
    assert not np.array_equal(gu, u)


def test_exchange_adjoint_identity(ex):
    """<E x, y> == <x, E^T y> for the scalar and vector exchanges (the JAX transpose of the gather)."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=L.shape2d))
    y = jnp.asarray(rng.normal(size=L.shape2d))
    ex_x, vjp = jax.vjp(ex.exch_xy, x)
    lhs = jnp.vdot(ex_x, y)
    rhs = jnp.vdot(x, vjp(y)[0])
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13)
    xu, xv, yu, yv = (jnp.asarray(rng.normal(size=L.shape2d)) for _ in range(4))
    (eu, ev), vjp = jax.vjp(lambda a, b: ex.exch_uv_xy(a, b), xu, xv)
    bu, bv = vjp((yu, yv))
    np.testing.assert_allclose(jnp.vdot(eu, yu) + jnp.vdot(ev, yv), jnp.vdot(xu, bu) + jnp.vdot(xv, bv), rtol=1e-13)
