"""Staged ECCO v4r4 inputs (plan Task 3): bytes as recorded, shapes and byte order as the model reads them, and an
input audit that fails when a file is missing.

Needs the data under $MITJAX_DATA (mitgcm_jax/paths.py, docs/DATA.md); it FAILS rather than skips when they are
absent, because on Levante an absent file means the staging is broken.
"""

import hashlib
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from mitgcm_jax import paths
from mitgcm_jax.io.mds import read_bin, read_mds

REPO = Path(__file__).resolve().parents[2]
DATA = paths.DATA                  # $MITJAX_DATA
V4R4 = REPO / "ECCO-v4-Configurations" / "ECCOv4 Release 4"
HASH_LIMIT = 2 * 2**30  # the 90-190 GB forcing archives are verified at fetch time (sha512), not here

_spec = importlib.util.spec_from_file_location("audit_run_inputs", REPO / "scripts" / "audit_run_inputs.py")
air = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(air)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_entries():
    out = []
    for man in sorted(DATA.glob("*/MANIFEST.sha256")) + [DATA / "MANIFEST.extracted.sha256"]:
        assert man.exists(), f"missing manifest {man}"
        for line in man.read_text().splitlines():
            if line.strip():
                digest, rel = line.split()[:2]
                out.append((man.parent / rel, digest))
    return out


def test_staged_files_match_their_recorded_sha256():
    entries = _manifest_entries()
    names = {p.name for p, _ in entries}
    for must in ("pickup.0000000001.data", "tile001.mitgrid", "bathy_eccollc_90x50_min2pts.bin",
                 "ancillary_data_input_init_ECCO_V4r4.tar.gz",
                 "OCEAN_TEMPERATURE_SALINITY_snap_1992-01-02T000000_ECCO_V4r4_native_llc0090.nc"):
        assert must in names, f"{must} not staged"
    checked = 0
    for path, digest in entries:
        assert path.exists(), f"recorded but missing: {path}"
        if path.stat().st_size <= HASH_LIMIT:
            assert _sha256(path) == digest, f"bytes changed: {path}"
            checked += 1
    assert checked >= 90


def test_shapes_and_byte_order():
    ii = DATA / "input_init"
    # every MDS pair is self-consistent in the LLC90 compact layout
    for meta in sorted(ii.glob("*.meta")):
        arr, m = read_mds(meta.with_suffix(""))
        assert arr.shape[-2:] == (1170, 90), meta.name
    pickup, m = read_mds(ii / "pickup.0000000001")
    assert m["fldList"][:4] == ["Uvel", "Vvel", "Theta", "Salt"] and pickup.shape == (403, 1170, 90)
    # raw inputs: big-endian float32, 2-D or 50-level
    for name, nz in (("bathy_eccollc_90x50_min2pts.bin", None), ("total_kapgm_r009bit11.bin", 50),
                     ("total_kapredi_r009bit11.bin", 50), ("total_diffkr_r009bit11.bin", 50),
                     ("fenty_biharmonic_visc_v11.bin", 50)):
        a = read_bin(ii / name, nz=nz)
        assert a.shape == ((1, 1170, 90) if nz is None else (1, 50, 1170, 90)) and np.isfinite(a).all(), name
    bathy = read_bin(ii / "bathy_eccollc_90x50_min2pts.bin")[0]
    wet = bathy < 0
    assert wet.sum() == 60646 and -7000 < bathy.min() < -5000 and bathy.max() == 0.0
    theta, salt = pickup[100], pickup[150]                       # surface level of Theta, Salt
    assert -3 < theta[wet].min() and theta[wet].max() < 35 and 15 < salt[wet].min() and salt[wet].max() < 42
    # negative control: the same bytes read little-endian are nonsense
    le = np.fromfile(ii / "bathy_eccollc_90x50_min2pts.bin", dtype="<f4")
    assert np.nanmax(np.abs(le)) > 1e30
    # grid: 16 float64 fields on (n+1) x (m+1) points per facet
    for t, (n, m_) in {1: (90, 270), 2: (90, 270), 3: (90, 90), 4: (270, 90), 5: (270, 90)}.items():
        assert (DATA / "native_grid_files" / f"tile00{t}.mitgrid").stat().st_size == 16 * 8 * (n + 1) * (m_ + 1)


def _mini_rundir(tmp_path, ntimesteps):
    run = tmp_path / "run"
    run.mkdir()
    for f in ("data", "data.pkg", "data.cal", "data.exf", "data.gmredi", "data.ctrl", "data.optim", "data.smooth",
              "data.ecco", "data.exch2", "eedata"):
        shutil.copy(V4R4 / "flux-forced" / "namelist" / f, run / f)
    text = (run / "data").read_text().replace("nTimeSteps=227903", f"nTimeSteps={ntimesteps}")
    assert text != (run / "data").read_text() or ntimesteps == 227903
    (run / "data").write_text(text)
    return run


def test_audit_years_and_negative_control(tmp_path):
    run = _mini_rundir(tmp_path, 744)                            # 31 days from 1992-01-01 13:00
    items, unresolved = air.needs(run)
    names = {n.name for n in items}
    assert "TFLUX_6hourlyavg_1992" in names and "TFLUX_6hourlyavg_1993" not in names
    assert "pickup.0000000001.data" in names and "xx_qnet.0000000129.data" in names
    assert "smooth3DscalesH001" in names and unresolved == []
    assert {f"tile00{i}.mitgrid" for i in range(1, 6)} <= names
    # year boundary: a run ending after the last 1992 record (Dec 31 21:00) needs the 1993 file
    s, e = air.run_window({"parm03": {"niter0": [1], "ntimesteps": [8784], "deltatclock": [3600.0]}},
                          {"cal_nml": {"startdate_1": [19920101], "startdate_2": [120000]}})
    assert air.exf_years(s, e, 19920101, 30000, 21600.0) == [1992, 1993]
    assert air.exf_years(s, s, 19920101, 30000, 21600.0) == [1992]
    # all inputs present -> pass; one planted absence -> fail, and the missing file is named
    for n in items:
        if n.required and n.klass == "input":
            (run / n.name).write_bytes(b"x")
    for n in items:
        if n.required and n.klass == "cost":
            (run / (n.name + ".bin")).write_bytes(b"x")
    missing, _ = air.audit(run)
    assert missing == []
    (run / "TFLUX_6hourlyavg_1992").rename(run / "moved_away")
    missing, _ = air.audit(run)
    assert [n.name for n in missing] == ["TFLUX_6hourlyavg_1992"]
    assert air.main([str(run)]) == 1
