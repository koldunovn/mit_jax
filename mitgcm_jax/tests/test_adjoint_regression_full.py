"""Plan Task 22 (tier 2, one GPU): regression of the full-V4r4 multi-week-gradient set-up (M2 adjoint acceptance) on a
short window.

scripts/adjoint/fullgrad.py over ONE day (24 steps, two 12-step chunks) of the production full V4r4 run (bulk-formula
EXF + sea ice, useCTRL=T; initial state from the cache that multiweek_grad.py --actions cache builds for the full run
directory, bitwise the Fortran start-of-run state), J = box-mean theta (adjsen box, K) + Arctic mean ice thickness at
the end (m), controls theta0, kapGM, HEFF0 and the EXF atmospheric-state adjustments, in the ecco mode and the exact mode
with the full sea-ice derivative. J and the gradient (norm per control, directional derivatives) must reproduce the
values recorded on the same GPU kind.

Recorded per device kind: with sea ice the production XLA flags make runs on different GPU kinds differ beyond
round-off (SEAICE_GROWTH exact-zero branches, docs/VALIDATION.md "Acceptance criteria for runs with sea ice"), so a GPU
kind without a record is skipped with the row to record. Tolerances: the GPU forward is deterministic (J bitwise in
repeats); gradient repeats differ by <= 1e-14 (ecco) / 5e-11 (exact modes, 7-day window) relative: 1e-9 here.
Negative control: the two modes have the same J and directional derivatives that differ by far more than the
tolerance (the ecco mode has no sea-ice adjoint: d J / d atemp_arctic ~ 0).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import pytest

from mitgcm_jax import paths

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "adjoint" / "fullgrad.py"
CACHE = paths.RUNS / "adjoint_m2" / "init_ref_full_serial13_1day"   # fullgrad.CACHE
OUT_ROOT = paths.RUNS_JAX / "tier2"

# {device_kind: {mode: recorded row subset}}; default XLA flags, sum_unroll=5, one day, chunk 12
RECORDED = {
    # job 27657600 (GH200, dolpung): J bitwise in both modes, negative-control assertions passed. Left out: the
    # ecco values of atemp_arctic / atemp_arctic_pt (round-off-level numbers: no sea-ice adjoint in ecco mode)
    "NVIDIA GH200 120GB": {
        "ecco": {
            "J": 13.68031412817452,
            "grad_norm":
                {"aqh": 0.0013717908963728263, "atemp": 4.090562885855107e-07, "heff": 0.012840996204107552,
                 "kapGM": 5.6810902237176114e-08, "lwdown": 2.206175136208766e-08, "precip": 80.16092273915794,
                 "swdown": 2.366199764004705e-08, "tauu": 0.004266690193651703, "tauv": 0.004847089948363971,
                 "theta": 0.017383618718426395},
            "dirderiv":
                {"theta_A_centre": 0.00023573140221861864, "theta_B_above": 1.0429481536016608e-05,
                 "theta_C_south": 2.5844308633761593e-06, "kapGM_scale": 0.00023027265498145078,
                 "atemp_box": -4.89242792301795e-07, "tauu_box": -0.0019874885841498066,
                 "heff_arctic": 1.0369654175979914, "atemp_pt": 6.197053450921258e-09,
                 "heff_pt": 0.00016648997673671475},
        },
        "exact_full": {
            "J": 13.68031412817452,
            "grad_norm":
                {"aqh": 0.046395819002884274, "atemp": 1.8922331033998037e-05, "heff": 0.011862890858605604,
                 "kapGM": 5.699762971418107e-08, "lwdown": 8.220176706719961e-07, "precip": 156.1499838920965,
                 "swdown": 4.0622429164682284e-07, "tauu": 0.004293085660925946, "tauv": 0.0048859739939973845,
                 "theta": 0.01744968517458154},
            "dirderiv":
                {"theta_A_centre": 0.00023575073424959215, "theta_B_above": 9.84103752930497e-06,
                 "theta_C_south": 2.5293452721339116e-06, "kapGM_scale": 0.0002702791673034752,
                 "atemp_box": -5.751231559230375e-07, "tauu_box": -0.0020294567058242764,
                 "atemp_arctic": -0.0006851184701818948, "heff_arctic": 1.0312137231643692,
                 "atemp_pt": 6.13640496657793e-09, "atemp_arctic_pt": -2.785895941356301e-08,
                 "heff_pt": 0.00016630875749589898},
        },
    },
}


def _run(mode):
    out = OUT_ROOT / (f"adjoint_regression_full_{mode}_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}_"
                      f"{int(time.time())}")
    env = dict(os.environ)
    env["XLA_FLAGS"] = ""                    # production GPU flags (test_adjoint_regression.py docstring)
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), "--days", "1", "--chunk", "12", "--mode", mode,
                        "--actions", "grad"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    return [x for x in rows if x["action"] == "grad"][0]


def _check(row, rec):
    assert row["grad_finite"]
    assert row["nsteps"] == 24 and row["xla_flags"] == ""
    assert row["J"] == pytest.approx(rec["J"], rel=1e-14, abs=0.0)
    for group in ("grad_norm", "dirderiv"):
        for k, v in rec[group].items():
            assert row[group][k] == pytest.approx(v, rel=1e-9, abs=1e-20), (group, k)


def test_adjoint_regression_full_one_day():
    """ecco and exact (full sea ice) over one day: J and gradients as recorded; same J, gradients that differ."""
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    assert gpus, f"tier-2 GPU test needs a GPU, found {jax.devices()}"
    assert (CACHE / "state.npz").exists(), (
        f"initial-state cache {CACHE} missing: build it with multiweek_grad.py --actions cache --rundir "
        f"<full run dir> --cache {CACHE} (CPU, gate flags)")
    kind = str(gpus[0].device_kind)
    rows = {m: _run(m) for m in ("ecco", "exact_full")}
    print(json.dumps({kind: {m: {k: r[k] for k in ("J", "grad_norm", "dirderiv", "job")} for m, r in rows.items()}}))
    assert rows["ecco"]["J"] == rows["exact_full"]["J"]                     # the forward is mode-independent
    ge, gc = rows["exact_full"]["dirderiv"], rows["ecco"]["dirderiv"]
    assert abs(gc["atemp_arctic"]) < 1e-6 * abs(ge["atemp_arctic"]), (ge, gc)   # no sea-ice adjoint in ecco mode
    assert abs(ge["kapGM_scale"] - gc["kapGM_scale"]) / abs(ge["kapGM_scale"]) > 1e-3, (ge, gc)
    if kind not in RECORDED:
        pytest.skip(f"no recorded values for {kind}: record the printed row in RECORDED")
    for m in ("ecco", "exact_full"):
        _check(rows[m], RECORDED[kind][m])
