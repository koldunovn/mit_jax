"""LLC90 layouts: MITgcm compact global arrays <-> 5 facets <-> the 13 tiles of the ECCO/PO.DAAC netCDF products.

Compact (MITgcm global files, W2_mapIO=1): shape (..., 1170, 90). Rows 0:270 facet 1, 270:540 facet 2, 540:630
facet 3 (Arctic cap), each (ny, nx) as-is; rows 630:900 and 900:1170 hold facets 4 and 5, which are 270 wide and 90
tall: their bytes are a (90, 270) array written as 270 rows of 90, i.e. a plain reshape, not a rotation.
Facets: 1, 2: (270, 90); 3: (90, 90); 4, 5: (90, 270), all (j, i) in model index order.
Tiles (PO.DAAC native netCDF, dims (tile, j, i)): 0-2 facet 1 rows 0:90, 90:180, 180:270; 3-5 facet 2; 6 facet 3;
7-9 facet 4 columns 0:90, 90:180, 180:270; 10-12 facet 5. No rotation: tile (j, i) is model (j, i).
Verified against the V4r4 geometry product (XC/YC to float32 rounding; wet mask from the bathymetry file identical,
scripts/tests/test_llc_layout.py). Leading axes (records, levels) pass through.
"""

import numpy as np

NF = 90
FACET_SHAPE = {1: (270, 90), 2: (270, 90), 3: (90, 90), 4: (90, 270), 5: (90, 270)}
_ROWS = {1: (0, 270), 2: (270, 540), 3: (540, 630), 4: (630, 900), 5: (900, 1170)}


def compact_to_facets(a):
    a = np.asarray(a)
    lead = a.shape[:-2]
    assert a.shape[-2:] == (1170, NF), a.shape
    return {f: a[..., r0:r1, :].reshape(*lead, *FACET_SHAPE[f]) for f, (r0, r1) in _ROWS.items()}


def facets_to_compact(fa):
    lead = fa[1].shape[:-2]
    return np.concatenate([fa[f].reshape(*lead, -1, NF) for f in range(1, 6)], axis=-2)


def facets_to_tiles(fa):
    t = []
    for f in (1, 2):
        t += [fa[f][..., NF * k:NF * (k + 1), :] for k in range(3)]
    t.append(fa[3])
    for f in (4, 5):
        t += [fa[f][..., :, NF * k:NF * (k + 1)] for k in range(3)]
    return np.stack(t, axis=-3)


def tiles_to_facets(t):
    t = np.asarray(t)
    fa = {1: np.concatenate([t[..., k, :, :] for k in (0, 1, 2)], axis=-2),
          2: np.concatenate([t[..., k, :, :] for k in (3, 4, 5)], axis=-2),
          3: t[..., 6, :, :],
          4: np.concatenate([t[..., k, :, :] for k in (7, 8, 9)], axis=-1),
          5: np.concatenate([t[..., k, :, :] for k in (10, 11, 12)], axis=-1)}
    return fa


def compact_to_tiles(a):
    """(..., 1170, 90) -> (..., 13, 90, 90) in PO.DAAC tile order."""
    return facets_to_tiles(compact_to_facets(a))


def tiles_to_compact(t):
    return facets_to_compact(tiles_to_facets(t))
