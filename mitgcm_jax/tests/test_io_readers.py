"""Namelist and MDS readers on synthetic inputs, including the V4r4 quirks they must survive."""

import numpy as np
import pytest

from mitgcm_jax.io.mds import read_bin, read_mds
from mitgcm_jax.io.namelist import parse_namelist

NML = """\
# comment line with key = 'x'
 &PARM01
 tRef = 3*23.,2*22.,
        21.,
 no_slip_sides  = .TRUE.,
 eosType='JMD95Z',
 name = 'it''s, a list',
 exf_inscal_sflux   = -1.d-3,
#skipped = 5,
 /
 &GM_PARM01
  GM_background_K3dFile='total_kapgm.bin'
  GM_Small_Number  = 1.D-20
 /
 &SALT_PLUME_PARM01
 SPsalFRAC= 0.5D0,
 &
 &CTRL_NML_GENARR
 xx_genarr3d_file(3)='xx_kapgm',
 xx_genarr2d_bounds(1:5,1)=-9.0,-8.9,8.9,9.0,0.,
 xx_genarr3d_preproc_i(1,3)=1,
 &end
"""


def test_namelist_quirks():
    g = parse_namelist(NML)
    p = g["parm01"]
    assert p["tref"] == [23.0, 23.0, 23.0, 22.0, 22.0, 21.0]
    assert p["no_slip_sides"] == [True]
    assert p["eostype"] == ["JMD95Z"]
    assert p["name"] == ["it's, a list"]
    assert p["exf_inscal_sflux"] == [-1e-3]
    assert "skipped" not in p
    assert g["gm_parm01"] == {"gm_background_k3dfile": ["total_kapgm.bin"], "gm_small_number": [1e-20]}
    assert g["salt_plume_parm01"] == {"spsalfrac": [0.5]}            # group closed by a bare '&'
    c = g["ctrl_nml_genarr"]                                          # closed by '&end'
    assert c["xx_genarr3d_file(3)"] == ["xx_kapgm"]
    assert c["xx_genarr2d_bounds(1:5,1)"] == [-9.0, -8.9, 8.9, 9.0, 0.0]
    assert c["xx_genarr3d_preproc_i(1,3)"] == [1]


def _write_mds(tmp_path, arr, prec, nrec, dims):
    (tmp_path / "f.data").write_bytes(arr.astype(">f8" if prec == "float64" else ">f4").tobytes())
    dl = ",\n".join(f"{n:6d},{1:6d},{n:6d}" for n in dims)
    (tmp_path / "f.meta").write_text(
        f" nDims = [ {len(dims)} ];\n dimList = [\n{dl}\n ];\n dataprec = [ '{prec}' ];\n"
        f" nrecords = [ {nrec} ];\n timeStepNumber = [ 7 ];\n nFlds = [ 2 ];\n fldList = {{\n 'A       ' 'B       '\n }};\n")
    return tmp_path / "f"


def test_mds_roundtrip_and_layout(tmp_path):
    nx, ny, nz, nrec = 3, 4, 2, 2
    ref = np.arange(nrec * nz * ny * nx, dtype=np.float64).reshape(nrec, nz, ny, nx) * 0.5
    prefix = _write_mds(tmp_path, ref, "float64", nrec, (nx, ny, nz))
    arr, meta = read_mds(prefix)
    np.testing.assert_array_equal(arr, ref)                          # x fastest (Fortran order)
    assert meta["fldList"] == ["A", "B"] and meta["timeStepNumber"] == [7] and meta["dataprec"] == "float64"
    with open(str(prefix) + ".data", "ab") as f:                      # negative control: size mismatch
        f.write(b"\0" * 8)
    with pytest.raises(ValueError, match="meta implies"):
        read_mds(prefix)


def test_read_bin_big_endian_and_size_checks(tmp_path):
    ref = -np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    p = tmp_path / "b.bin"
    p.write_bytes(ref.astype(">f4").tobytes())
    np.testing.assert_array_equal(read_bin(p, nz=2, nx=3, ny=4)[0], ref)
    with pytest.raises(ValueError, match="multiple of nz"):
        read_bin(p, nz=5, nx=3, ny=4)
    with pytest.raises(ValueError, match="not a multiple"):
        read_bin(p, nx=5, ny=5)
