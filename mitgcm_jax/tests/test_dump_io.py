"""jaxdump reader round-trip and diffdump negative controls, on synthetic dumps written in the Fortran record format
(reference/jaxdump/jaxdump.F). The real-dump checks (byte-identical output with dumps on/off, assembly vs the model's
own output, halo semantics) are recorded in docs/REFERENCE_RUNS.md."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mitgcm_jax.io.dump import MAGIC, DumpSet, read_file

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("diffdump", REPO / "tools" / "diffdump.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

SNX = SNY = 4
OL = 2


def _rec(it, seq, stage, field, kind, data, tile, face=1, tbx=0, tby=0):
    hdr = np.array([MAGIC, 1, it, seq], ">i4").tobytes()
    txt = stage.ljust(32).encode() + field.ljust(32).encode() + kind.ljust(4).encode()
    dims = np.array([data.shape[0], SNX, SNY, OL, OL, tile, face, tbx, tby], ">i4").tobytes()
    return hdr + txt + dims + data.astype(">f8").tobytes()


def _write_set(d, fields, it=1):
    """fields: list of (stage, field, kind, {tile: array (nz, SNY+2OL, SNX+2OL)})"""
    d.mkdir(parents=True, exist_ok=True)
    blobs = {}
    for seq, (stage, field, kind, tiles) in enumerate(fields, 1):
        for tile, arr in tiles.items():
            tby = 0 if tile == 1 else SNY
            blobs.setdefault(tile, []).append(_rec(it, seq, stage, field, kind, arr, tile, tby=tby))
    for tile, b in blobs.items():
        (d / f"jd_{it:010d}_t{tile:04d}.bin").write_bytes(b"".join(b))


def _fields(rng, planted=None, zero=False, drop=None):
    shape = (2, SNY + 2 * OL, SNX + 2 * OL)
    h = {t: np.ones(shape) for t in (1, 2)}
    h[2][:, OL, OL] = 0.0                                          # one dry point
    th = {t: rng.normal(size=shape) for t in (1, 2)}
    if planted:
        th = {t: a.copy() for t, a in th.items()}
        th[2][1, OL + 1, OL + 1] += planted                        # wet interior point
    z = {t: np.zeros(shape) for t in (1, 2)}
    out = [("S00_begin", "hFacC", "C", h), ("S00_begin", "theta", "C", th)]
    if zero:
        out.append(("S04_oceanic_phys", "GGL90TKE", "C", z))
    if drop:
        out = [f for f in out if f[1] != drop]
    return out


def test_roundtrip_and_assembly(tmp_path):
    rng = np.random.default_rng(1)
    f = _fields(rng)
    _write_set(tmp_path / "a", f)
    recs = read_file(tmp_path / "a" / "jd_0000000001_t0002.bin")
    assert [r.field for r in recs] == ["hFacC", "theta"] and recs[1].tby == SNY
    np.testing.assert_array_equal(recs[1].data, f[1][3][2])
    np.testing.assert_array_equal(recs[1].interior, f[1][3][2][:, OL:-OL, OL:-OL])
    ds = DumpSet(tmp_path / "a")
    assert ds.keys() == [(1, "S00_begin", "hFacC"), (1, "S00_begin", "theta")]


def test_diffdump_negative_controls(tmp_path):
    rng = np.random.default_rng(2)
    base = _fields(np.random.default_rng(2))
    _write_set(tmp_path / "ref", base)
    _write_set(tmp_path / "same", _fields(np.random.default_rng(2)))
    rows = dd.compare(DumpSet(tmp_path / "ref"), DumpSet(tmp_path / "same"))
    assert all(r[1] == "ok" and r[2] == 0.0 for r in rows)
    # planted 1e-9 difference at one wet point is caught at the right key
    _write_set(tmp_path / "plant", _fields(np.random.default_rng(2), planted=1e-9))
    bad = [r for r in dd.compare(DumpSet(tmp_path / "ref"), DumpSet(tmp_path / "plant")) if r[1] != "ok"]
    assert [(r[0][2], r[1]) for r in bad] == [("theta", "FAIL")]
    assert dd.main([str(tmp_path / "ref"), str(tmp_path / "plant")]) == 1
    # a difference at a DRY point is not a failure (masked), but it is still one with --halos off? -> dry is skipped
    dry = _fields(np.random.default_rng(2))
    dry[1][3][2][:, OL, OL] += 5.0
    _write_set(tmp_path / "dry", dry)
    assert all(r[1] == "ok" for r in dd.compare(DumpSet(tmp_path / "ref"), DumpSet(tmp_path / "dry")))
    # all-zero on both sides is flagged, missing field is flagged
    _write_set(tmp_path / "z1", _fields(np.random.default_rng(2), zero=True))
    _write_set(tmp_path / "z2", _fields(np.random.default_rng(2), zero=True))
    assert ("GGL90TKE", "ZERO") in [(r[0][2], r[1]) for r in dd.compare(DumpSet(tmp_path / "z1"),
                                                                         DumpSet(tmp_path / "z2"))]
    _write_set(tmp_path / "miss", _fields(np.random.default_rng(2), drop="theta"))
    assert ("theta", "MISSING") in [(r[0][2], r[1]) for r in dd.compare(DumpSet(tmp_path / "ref"),
                                                                         DumpSet(tmp_path / "miss"))]


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "jd_0000000001_t0001.bin"
    p.write_bytes(b"\0" * 200)
    with pytest.raises(ValueError, match="bad magic"):
        read_file(p)
