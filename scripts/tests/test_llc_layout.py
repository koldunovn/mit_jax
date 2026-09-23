"""LLC90 compact <-> tiles mapping, checked against the PO.DAAC V4r4 geometry product (the comparator for every
Fortran-vs-product check, and the layout the JAX grid loader will use)."""

from pathlib import Path

import netCDF4
import numpy as np

from mitgcm_jax.io.llc import (FACET_SHAPE, compact_to_facets, compact_to_tiles, facets_to_tiles,
                               tiles_to_compact)
from mitgcm_jax.io.mds import read_bin

DATA = Path("/work/ab0995/a270088/MIT/data/eccov4r4")
GEOM = DATA / "products_fixed/ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"


def test_roundtrip_with_leading_axes():
    a = np.random.default_rng(0).normal(size=(2, 3, 1170, 90))
    t = compact_to_tiles(a)
    assert t.shape == (2, 3, 13, 90, 90)
    np.testing.assert_array_equal(tiles_to_compact(t), a)


def test_against_geometry_product():
    g = netCDF4.Dataset(GEOM)
    XC, YC, depth = (np.asarray(g[v][:]) for v in ("XC", "YC", "Depth"))
    # grid: tile*.mitgrid facets (16 float64 fields on (m+1, n+1) corner-inclusive points) -> tiles
    xf, yf = {}, {}
    for f, (m, n) in FACET_SHAPE.items():
        a = np.fromfile(DATA / f"native_grid_files/tile00{f}.mitgrid", ">f8").reshape(16, m + 1, n + 1)
        xf[f], yf[f] = a[0, :m, :n], a[1, :m, :n]
    dx = (facets_to_tiles(xf) - XC + 180) % 360 - 180
    assert np.abs(dx).max() < 2e-5 and np.abs(facets_to_tiles(yf) - YC).max() < 1e-5  # product is float32
    # compact layout: the bathymetry file's wet mask equals the product's Depth > 0 everywhere
    bathy = read_bin(DATA / "input_init/bathy_eccollc_90x50_min2pts.bin")[0]
    assert ((compact_to_tiles(bathy) < 0) == (depth > 0)).all()
    # negative control: reading facet 4 without the (90, 270) reshape breaks the mask agreement
    wrong = bathy[630:900].reshape(270, 90)[:90]
    assert ((wrong < 0) == (depth[7] > 0)).mean() < 0.9
    assert compact_to_facets(bathy)[4].shape == (90, 270)
