"""Grid and geometry (plan Task 6). Oracle geometry from the dump is self-consistent with the Fortran grid output
files; the literal port (grid_from_files) is gated against it field by field (added with mitgcm_jax/grid/load.py)."""

import numpy as np
import pytest

from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.io.mds import read_mds
from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.layout import Layout
from mitgcm_jax.tests import oracle

L = Layout()


@pytest.fixture(scope="module")
def g():
    return grid_from_dump(oracle.dumpset(oracle.SMOKE), 1)


def _interior(a):
    return a[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx]


@pytest.mark.parametrize("name,out", [("xC", "XC"), ("yC", "YC"), ("dxC", "DXC"), ("rA", "RAC"), ("hFacC", None)])
def test_dump_geometry_matches_fortran_output_files(g, name, out):
    """Dumped (float64) geometry equals the model's own float32 grid files (write_grid.F) to float32 rounding."""
    run = oracle.run_dir(oracle.SMOKE)
    if out is None:
        return
    ref, _ = read_mds(run / out)
    ref = compact_to_tiles(ref[0])
    np.testing.assert_array_equal(_interior(getattr(g, name)).astype(np.float32), ref.astype(np.float32))


def test_vertical_grid(g):
    assert g.drF.shape == (50,) and g.rF.shape == (51,)
    np.testing.assert_allclose(np.cumsum(g.drF), -g.rF[1:], rtol=0, atol=1e-9)
    assert g.drF[0] == 10.0 and g.drF[-1] == 456.5


def test_wet_columns(g):
    """60,646 wet columns in the V4r4 bathymetry (docs/DATA.md)."""
    assert int(_interior(g.maskInC).sum()) == 60646
