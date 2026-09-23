"""Gate for plan Task 6: grid_from_files (literal port of INI_GRID ... INI_MIXING, mitgcm_jax/grid/load.py) equals
the Fortran oracle's own geometry (grid_from_dump, stage G00_geometry of the SMOKE run) field by field, halos
included.

Exchange gap (Task 7, mitgcm_jax/parallel/exchange.py): the exch2 maps come from a probe with ZERO halos, so they
cannot represent exch2 copies whose source is a halo point: at a halo point no probed exchange writes, the Exchanger
keeps the point's own value. exch2 does write some of them from halo sources, e.g. exch2_z_3d_rx.template:108-116
(east edge of facets 2/4: phi(i,j) = phi(i,j-1) for i > sNx), and the UPDATE_CORNERS pass
(exch2_uv_3d_rx.template:80) copies a neighbour's halo point that the first pass left unfilled. For the grid those
halos hold the extra i=sNx+1 / j=sNy+1 row of the mitgrid files, so results differ there, and through
CALC_GRID_ANGLES also at interior points of tile 10, i=90 (land next to the open facet-4 edge).
The gate therefore compares bitwise every point that does NOT depend on such "unwritten" halo points. That set
(`taint`) is computed exactly by re-running the port with those halo points poisoned (NaN, 0, +/-1e5) and marking
every output that changes. The full comparison is kept as a strict xfail until the maps include halo sources.

Achieved (SMOKE oracle): every field of G2D, G3D, R3D and the vertical rows is BITWISE equal outside the taint set
(0 differing bits), including the trigonometric ones: angles/u2zonDir/v2zonDir via glibc sin/cos, fCori/fCoriCos
via glibc sincos (gfortran fuses the pair; plain cos differs by <= 2 ulp at 70 points).
Not compared: dBdrRef (needs the JMD95Z EOS, not ported in load.py).
Negative controls: hFacMin * (1 + 1e-6) and a dropped exchange each make the same comparison fail.
"""

import dataclasses

import numpy as np
import pytest

from mitgcm_jax.grid.geometry import G2D, G3D, R3D, VROWS, grid_from_dump
from mitgcm_jax.grid.load import GridParams, grid_from_files
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.parallel.exchange import MAP_DIR, ExchangeMaps, Exchanger
from mitgcm_jax.tests import oracle

L = Layout()
VERT = [n for n, _ in VROWS.values() if n != "dBdrRef"] + ["phiRef"]
NOT_PORTED = {"dBdrRef"}
GROUPS = {"G2D": G2D, "G3D": G3D, "R3D": R3D, "vertical": VERT}
FIELDS = G2D + G3D + R3D
POISONS = (np.nan, 0.0, 1.0e5, -1.0e5)


def _halo():
    m = np.ones(L.shape2d, bool)
    m[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = False
    return m


class PoisonExchanger:
    """The Exchanger, but every halo point the map of that exchange kind does not write gets `value`."""

    def __init__(self, maps, value):
        self.ex = Exchanger(maps)
        self.value = value
        halo = _halo()
        self.unw = {k: (c == 0).reshape(L.shape2d) & halo for k, (s, c, g) in maps.maps.items()}

    def _p(self, a, key):
        a = np.array(a, np.float64)
        m = self.unw[key]
        return np.where(m if a.ndim == 3 else m[:, None], self.value, a)

    def scalar(self, a, kind="T"):
        return self._p(self.ex.scalar(a, kind), kind)

    def vector(self, u, v, kind="UVs"):
        a, b = self.ex.vector(u, v, kind)
        return self._p(a, kind + "_u"), self._p(b, kind + "_v")


@pytest.fixture(scope="module")
def rundir():
    return oracle.run_dir(oracle.SMOKE)


@pytest.fixture(scope="module")
def gd():
    return grid_from_dump(oracle.dumpset(oracle.SMOKE), 1)


@pytest.fixture(scope="module")
def gf(rundir):
    return grid_from_files(rundir)


@pytest.fixture(scope="module")
def taint(rundir):
    """{field: bool array}: outputs that depend on a halo point no probed exchange writes."""
    maps = ExchangeMaps.load(MAP_DIR / f"exch_maps_{L.nTiles}x{L.sNx}x{L.sNy}.npz")
    runs = [grid_from_files(rundir, exchanger=PoisonExchanger(maps, v)) for v in POISONS]
    out = {}
    for n in FIELDS:
        ref = np.ascontiguousarray(runs[1].f[n], np.float64).view(np.int64)
        t = np.zeros(ref.shape, bool)
        for r in runs:
            t |= np.ascontiguousarray(r.f[n], np.float64).view(np.int64) != ref
        out[n] = t
    return out


def _compare(gf, gd, names, exclude=None):
    """{name: (n differing bit patterns, max ulp distance, max abs diff / max|oracle|)} over the compared points."""
    out = {}
    for n in names:
        a = np.ascontiguousarray(gf.f[n], np.float64)
        b = np.ascontiguousarray(gd.f[n], np.float64)
        assert a.shape == b.shape, (n, a.shape, b.shape)
        keep = ~exclude[n] if exclude is not None and n in exclude else np.ones(a.shape, bool)
        a, b = a[keep], b[keep]
        ai, bi = a.view(np.int64), b.view(np.int64)
        nd = int(np.count_nonzero(ai != bi))
        ulp = int(np.max(np.where(np.signbit(a) == np.signbit(b), np.abs(ai - bi), 2 ** 62))) if nd else 0
        scale = float(np.max(np.abs(b))) if b.size else 0.0
        diff = float(np.max(np.abs(a - b))) if b.size else 0.0
        out[n] = (nd, ulp, diff / scale if scale > 0 else diff)
    return out


def _failures(res):
    return {n: r for n, r in res.items() if r[0]}


def _msg(bad):
    return "\n".join(f"{n}: {nd} differing values, max {u} ulp, max rel {r:.3e}" for n, (nd, u, r) in bad.items())


def test_every_oracle_field_is_compared(gd, gf):
    """The gate covers every geometry field the oracle dumps (nothing silently skipped)."""
    dumped = set(G2D + G3D + R3D + [n for n, _ in VROWS.values()] + ["phiRef"])
    assert dumped <= set(gd.f), sorted(dumped - set(gd.f))
    missing = dumped - set(gf.f) - NOT_PORTED
    assert not missing, sorted(missing)


@pytest.mark.parametrize("group", list(GROUPS))
def test_fields_bitwise_outside_exchange_gap(gd, gf, taint, group):
    bad = _failures(_compare(gf, gd, GROUPS[group], exclude=taint))
    assert not bad, _msg(bad)


def test_exchange_gap_is_small_and_dry(gd, taint):
    """The excluded set is confined: halo points plus a few interior points, all of them dry (maskInC = 0)."""
    halo = _halo()
    for n, t in taint.items():
        t2 = t.any(axis=1) if t.ndim == 4 else t
        assert t2.sum() <= 2000, (n, int(t2.sum()))
        inner = t2 & ~halo
        assert not np.any(np.asarray(gd.maskInC)[inner]), (n, np.argwhere(inner)[:5])


# Fields where the exchange gap changes values (measured): the mitgrid Z-point, C-grid and B-grid pairs, their
# reciprocals, and what CALC_GRID_ANGLES / INI_CORI compute from them. All other fields are bitwise everywhere.
GAP_FIELDS = ["xG", "yG", "rAz", "dxC", "dyC", "dxG", "dyG", "dxV", "dyU", "rAw", "rAs", "recip_dxC", "recip_dyC",
              "recip_dxG", "recip_dyG", "recip_dxV", "recip_dyU", "recip_rAw", "recip_rAs", "recip_rAz",
              "angleCosC", "angleSinC", "u2zonDir", "v2zonDir", "fCoriG"]


def test_other_fields_bitwise_everywhere(gd, gf):
    """Every field outside GAP_FIELDS is bitwise equal including the excluded (taint) points."""
    bad = _failures(_compare(gf, gd, [n for n in FIELDS if n not in GAP_FIELDS]))
    assert not bad, _msg(bad)


@pytest.mark.xfail(strict=True, reason="exch2 maps lack halo-sourced copies (probe with zero halos; Task 7)")
def test_gap_fields_bitwise_everywhere(gd, gf):
    """Fails today (documented gap). Once the maps carry halo sources this XPASSes: then move GAP_FIELDS into the
    full comparison and drop the taint exclusion."""
    bad = _failures(_compare(gf, gd, GAP_FIELDS))
    assert not bad, _msg(bad)


def test_wet_columns(gf):
    """60,646 wet columns in the V4r4 bathymetry (docs/DATA.md)."""
    assert int(gf.maskInC[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx].sum()) == 60646


def test_level_indices_consistent_with_oracle_hfac(gd, gf):
    """kSurfC/W/S, kLowC (not dumped) agree with the oracle's h0Fac: first / last level with hFac != 0."""
    k = np.arange(1, L.Nr + 1)[None, :, None, None]
    for kname, hname, last in (("kSurfC", "h0FacC", False), ("kSurfW", "h0FacW", False),
                               ("kSurfS", "h0FacS", False), ("kLowC", "h0FacC", True)):
        wet = np.asarray(gd.f[hname]) != 0.0
        ref = np.max(np.where(wet, k, 0), axis=1) if last else np.min(np.where(wet, k, L.Nr + 1), axis=1)
        np.testing.assert_array_equal(gf.f[kname], ref, err_msg=kname)


def test_negative_control_hfacmin(gd, rundir, taint):
    """A planted error (hFacMin * (1 + 1e-6)) makes the same comparison fail."""
    p = GridParams.from_namelists(RunNamelists(rundir), L)
    bad = grid_from_files(rundir, params=dataclasses.replace(p, hFacMin=p.hFacMin * (1 + 1e-6)))
    fails = _failures(_compare(bad, gd, G3D + R3D, exclude=taint))
    assert "h0FacC" in fails, fails


def test_negative_control_dropped_exchange(gd, rundir, taint):
    """Replacing the exchanger by the identity (no halo fill) makes the same comparison fail."""

    class NoExchange:
        def scalar(self, a, kind="T"):
            return np.asarray(a)

        def vector(self, u, v, kind="UVs"):
            return np.asarray(u), np.asarray(v)

    names = ["xC", "dxC", "dyG", "angleCosC", "R_low", "diffKr"]
    fails = _failures(_compare(grid_from_files(rundir, exchanger=NoExchange()), gd, names, exclude=taint))
    assert set(fails) == set(names), fails
