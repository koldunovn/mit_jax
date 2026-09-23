"""Readers for MITgcm binary files: MDS `.meta`/`.data` pairs and raw `.bin` inputs.

Files are big-endian (MITgcm writes with `-byteswapio`/`-convert big_endian`); `dataprec` in the `.meta` gives float32
or float64. Global files use the LLC90 compact layout: x (90) fastest, then y (1170 = 5 facets stacked), then z, then
record — returned here as numpy arrays of shape `(nrecords, [nz,] ny, nx)` in native float64.
Raw `.bin` inputs (bathymetry, 3-D K fields, forcing) have no `.meta`: pass the precision and shape explicitly
(`readBinaryPrec=32` in the V4r4 `data`).
"""

import re
from pathlib import Path

import numpy as np

NX, NY = 90, 1170  # LLC90 compact global layout


def read_meta(path):
    """Parse a `.meta` file into a dict: nDims, dimList (list of (n, i0, i1)), dataprec, nrecords, fldList, ..."""
    text = Path(path).read_text()
    meta = {}
    for key, body in re.findall(r"(\w+)\s*=\s*[\[{](.*?)[\]}]\s*;", text, re.S):
        strings = re.findall(r"'([^']*)'", body)
        if strings:
            vals = [s.strip() for s in strings]
        else:
            vals = [float(x) if re.search(r"[.eE]", x) else int(x) for x in re.findall(r"[-+.\dEe]+", body)]
        meta[key] = vals
    dims = meta["dimList"]
    meta["dimList"] = [tuple(dims[i:i + 3]) for i in range(0, len(dims), 3)]
    meta["dataprec"] = meta["dataprec"][0]
    meta["nrecords"] = meta["nrecords"][0]
    meta["nDims"] = meta["nDims"][0]
    return meta


def _dtype(prec):
    return {"float32": ">f4", "float64": ">f8", 32: ">f4", 64: ">f8"}[prec]


def read_mds(prefix):
    """Read `<prefix>.data` using `<prefix>.meta`. Returns (array (nrec, ..., ny, nx), meta)."""
    prefix = str(prefix)
    if prefix.endswith((".data", ".meta")):
        prefix = prefix[:-5]
    meta = read_meta(prefix + ".meta")
    shape = [n for n, _, _ in reversed(meta["dimList"])]  # (nz, ny, nx) from Fortran (nx, ny, nz)
    arr = np.fromfile(prefix + ".data", dtype=_dtype(meta["dataprec"]))
    expected = meta["nrecords"] * int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{prefix}.data has {arr.size} values, meta implies {expected}")
    return arr.reshape([meta["nrecords"], *shape]).astype(np.float64), meta


def read_bin(path, nz=None, prec=32, nx=NX, ny=NY):
    """Read a raw big-endian `.bin` in compact layout. nz=None infers it from the size (must be whole)."""
    arr = np.fromfile(path, dtype=_dtype(prec))
    per = nx * ny
    if arr.size % per:
        raise ValueError(f"{path}: {arr.size} values is not a multiple of {nx}x{ny}")
    n = arr.size // per
    if nz is not None and n % nz:
        raise ValueError(f"{path}: {n} levels/records is not a multiple of nz={nz}")
    nrec = n if nz is None else n // nz
    shape = (nrec, ny, nx) if nz is None else (nrec, nz, ny, nx)
    return arr.reshape(shape).astype(np.float64)
