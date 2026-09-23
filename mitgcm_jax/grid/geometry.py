"""Model geometry: every GRID.h / SURFACE.h field the ported kernels read, as `[tile, (k,) j, i]` arrays with halos.

`Grid` is a pytree (a dict of arrays + a static Layout); fields are accessed by their Fortran names (`g.dxC`,
`g.hFacC`, `g.drF`, ...). Two constructors:
  - `grid_from_dump(dumpset, it)`: the Fortran oracle's own values (stage G00_geometry, groups G/V/R, plus hFac of
    S01_update_rstar_F). Used by kernel gates, so a kernel is tested against exactly the geometry Fortran used.
  - `grid_from_files(...)` (mitgcm_jax/grid/load.py): the literal port of INI_GRID/INI_MASKS_ETC (plan Task 6),
    gated against `grid_from_dump` field by field.
1-D vertical arrays (drF, drC, rC, rF, recip_drF, recip_drC, tRef, sRef, rVel2wUnit, wUnit2rVel, rhoFacC, rhoFacF,
dBdrRef, phiRef) are plain [Nr] / [Nr+1] vectors.
"""

from dataclasses import dataclass, field

import jax
import numpy as np

from mitgcm_jax.layout import Layout

# 2-D and 3-D fields of the G group (jaxdump.F JAXDUMP_GEOM), R group, and hFac from the r group
G2D = ["xC", "yC", "xG", "yG", "dxC", "dyC", "dxF", "dyF", "dxG", "dyG", "dxV", "dyU", "rA", "rAw", "rAs", "rAz",
       "recip_dxC", "recip_dyC", "recip_dxF", "recip_dyF", "recip_dxG", "recip_dyG", "recip_dxV", "recip_dyU",
       "recip_rA", "recip_rAw", "recip_rAs", "recip_rAz", "R_low", "rLowW", "rLowS", "Ro_surf", "rSurfW", "rSurfS",
       "recip_Rcol", "maskInC", "maskInW", "maskInS", "angleCosC", "angleSinC", "u2zonDir", "v2zonDir", "fCori",
       "fCoriG", "fCoriCos", "tanPhiAtU", "tanPhiAtV"]
G3D = ["maskC", "maskW", "maskS", "diffKr", "kapGM", "kapRedi", "viscA4Dfld", "viscA4Zfld", "viscAhDfld",
       "viscAhZfld"]
R3D = ["h0FacC", "h0FacW", "h0FacS"]
# packed vertical record (jaxdump.F JAXDUMP_GEOM, group V): row -> (name, length: 'Nr' or 'Nr1')
VROWS = {1: ("drF", "Nr"), 2: ("drC", "Nr1"), 3: ("rC", "Nr"), 4: ("rF", "Nr1"), 5: ("recip_drF", "Nr"),
         6: ("recip_drC", "Nr1"), 7: ("tRef", "Nr"), 8: ("sRef", "Nr"), 9: ("rVel2wUnit", "Nr1"),
         10: ("wUnit2rVel", "Nr1"), 11: ("rhoFacC", "Nr"), 12: ("rhoFacF", "Nr1"), 13: ("dBdrRef", "Nr")}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Grid:
    f: dict
    layout: Layout = field(default_factory=Layout)

    def __getattr__(self, name):
        f = object.__getattribute__(self, "f")
        if name in f:
            return f[name]
        raise AttributeError(name)

    def tree_flatten(self):
        keys = tuple(sorted(self.f))
        return tuple(self.f[k] for k in keys), (keys, self.layout)

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, layout = aux
        return cls(dict(zip(keys, children)), layout)

    def replace(self, **kw):
        f = dict(self.f)
        f.update(kw)
        return Grid(f, self.layout)


def stack_tiles(dumpset, it, stage, name, layout):
    """[T, nz, ny, nx] (nz squeezed to [T, ny, nx] when 1) of one dumped field, tiles in W2 order."""
    recs = dumpset.tiles(it, stage, name)
    a = np.stack([recs[t + 1].data for t in range(layout.nTiles)])
    return a[:, 0] if a.shape[1] == 1 else a


def unpack_vertical(v, layout):
    """Packed vertical record (tile-1 2-D array with halos) -> dict of 1-D arrays."""
    L = layout
    out = {}
    for row, (name, n) in VROWS.items():
        m = L.Nr if n == "Nr" else L.Nr + 1
        out[name] = np.array(v[L.jj(row), L.ii(1):L.ii(1) + m])
    phiRef = np.concatenate([v[L.jj(14), L.ii(1):L.ii(1) + L.Nr + 1], v[L.jj(15), L.ii(1):L.ii(1) + L.Nr]])
    out["phiRef"] = phiRef
    return out


def grid_from_dump(dumpset, it, layout=None, stage="G00_geometry"):
    L = layout or Layout()
    f = {}
    for n in G2D + G3D + R3D:
        try:
            f[n] = stack_tiles(dumpset, it, stage, n, L)
        except KeyError:
            pass
    recs = dumpset.tiles(it, stage, "vertical")
    f.update(unpack_vertical(recs[1].data[0], L))
    return Grid(f, L)
