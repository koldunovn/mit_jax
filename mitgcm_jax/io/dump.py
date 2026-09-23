"""Reader for jaxdump per-substep dumps (reference/jaxdump/jaxdump.F).

Files: <dir>/jd_<myIter %010d>_t<tile %04d>.bin, a stream of big-endian records
    int32 magic=1245990224, version, myIter, seq | char32 stage | char32 field | char4 kind |
    int32 nz, sNx, sNy, OLx, OLy, tile, face, tBasex, tBasey | float64 values[nz][sNy+2*OLy][sNx+2*OLx]
Values include halos. A value at array index (k, j, i) of a record sits at global facet point
(face, jG = tBasey + j - OLy + 1, iG = tBasex + i - OLx + 1) in 1-based model indices; halo points fall outside the
tile's interior (and may fall outside the facet).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mitgcm_jax.io.llc import FACET_SHAPE, facets_to_compact

MAGIC = 1245990224
_HDR = np.dtype([("magic", ">i4"), ("version", ">i4"), ("iter", ">i4"), ("seq", ">i4"),
                 ("stage", "S32"), ("field", "S32"), ("kind", "S4"),
                 ("nz", ">i4"), ("snx", ">i4"), ("sny", ">i4"), ("olx", ">i4"), ("oly", ">i4"),
                 ("tile", ">i4"), ("face", ">i4"), ("tbx", ">i4"), ("tby", ">i4")])


@dataclass
class Record:
    iter: int
    seq: int
    stage: str
    field: str
    kind: str
    tile: int
    face: int
    tbx: int
    tby: int
    olx: int
    oly: int
    _data: np.ndarray = None  # (nz, sNy+2*OLy, sNx+2*OLx), float64, halos included; loaded lazily
    path: str = None
    offset: int = 0
    shape: tuple = None

    @property
    def data(self):
        if self._data is None:
            n = int(np.prod(self.shape))
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                buf = fh.read(8 * n)
            self._data = np.frombuffer(buf, ">f8", count=n).reshape(self.shape).astype(np.float64)
        return self._data

    def drop(self):
        """Forget the loaded values (they are re-read on the next access)."""
        if self.path is not None:
            self._data = None

    @property
    def interior(self):
        return self.data[:, self.oly:self.data.shape[1] - self.oly, self.olx:self.data.shape[2] - self.olx]


def read_file(path, lazy=False):
    """All records of one dump file. lazy=True reads only the headers (values are read on first access)."""
    path = str(path)
    size = Path(path).stat().st_size
    out, pos = [], 0
    with open(path, "rb") as fh:
        while pos < size:
            fh.seek(pos)
            h = np.frombuffer(fh.read(_HDR.itemsize), _HDR, count=1)[0]
            if h["magic"] != MAGIC:
                raise ValueError(f"{path}: bad magic at byte {pos}")
            pos += _HDR.itemsize
            nz, sny, snx, oly, olx = (int(h[k]) for k in ("nz", "sny", "snx", "oly", "olx"))
            shape = (nz, sny + 2 * oly, snx + 2 * olx)
            r = Record(int(h["iter"]), int(h["seq"]), h["stage"].decode().strip(), h["field"].decode().strip(),
                       h["kind"].decode().strip(), int(h["tile"]), int(h["face"]), int(h["tbx"]), int(h["tby"]),
                       olx, oly, path=path, offset=pos, shape=shape)
            if not lazy:
                r.data  # noqa: B018 (load now)
            out.append(r)
            pos += 8 * int(np.prod(shape))
    return out


class DumpSet:
    """All records under a dump directory, indexed by (iter, stage, field) -> {tile: Record}."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.index = {}
        self.order = []  # (iter, stage, field) in first-seen call order per iteration
        files = sorted(self.dir.glob("jd_*_t*.bin"))
        if not files:
            raise FileNotFoundError(f"no jaxdump files in {self.dir}")
        per_iter_tile = {}
        for f in files:
            m = re.fullmatch(r"jd_(\d{10})_t(\d{4})\.bin", f.name)
            per_iter_tile.setdefault(int(m.group(1)), []).append(f)
        for it in sorted(per_iter_tile):
            first = True
            for f in sorted(per_iter_tile[it]):
                for r in read_file(f, lazy=True):
                    key = (r.iter, r.stage, r.field)
                    if key not in self.index:
                        self.index[key] = {}
                        if first:
                            self.order.append(key)
                    self.index[key][r.tile] = r
                first = False
        # keys first seen in a later tile file (tile-scoped stages) keep their relative order at the end
        seen = set(self.order)
        self.order += [k for k in self.index if k not in seen]

    def keys(self):
        return list(self.order)

    def tiles(self, it, stage, field):
        return self.index[(it, stage, field)]

    def compact(self, it, stage, field):
        """Interior values assembled into the compact global layout (nz, 1170, 90). Missing tiles stay NaN."""
        recs = self.index[(it, stage, field)]
        nz = next(iter(recs.values())).data.shape[0]
        facets = {f: np.full((nz, *s), np.nan) for f, s in FACET_SHAPE.items()}
        for r in recs.values():
            v = r.interior
            facets[r.face][:, r.tby:r.tby + v.shape[1], r.tbx:r.tbx + v.shape[2]] = v
        return facets_to_compact(facets)
