"""Plan Task 21 (tier 2, one GPU): regression of the multi-week-gradient set-up on a short window.

scripts/adjoint/multiweek_grad.py over ONE day (24 steps, two 12-step chunks) of the production flux-forced V4r4 run
(useCTRL=T, geothermal; initial state from the cache that script builds, bitwise the Fortran start-of-run state),
cost = box-mean theta of the ECCO adjoint-sensitivity experiment (namelist_adjsen), controls theta0, kapGM and the
TFLUX / oceTAUX / oceTAUY adjustments, in the exact and the ecco mode. J and the gradient (norm per control, values
at the named points, directional derivatives) must reproduce the values recorded on an A100-80.

The script runs in a subprocess with the production GPU XLA flags (XLA's defaults). conftest.py's gate flags are for
the CPU oracle gates: with --xla_disable_hlo_passes=algsimp the one-day reverse pass took ~20 min on the GPU instead of
~1.5 min (job 27641176; its J was bitwise the same and its gradient within 3e-13).
Tolerances: the GPU forward is deterministic (J bitwise in repeats, Tasks 20/21); gradient repeats differ by <= 1.3e-12
(exact) / 3e-14 (ecco) relative, so the gradient is compared to 1e-9. Negative control: the two modes have the same J
and gradients that differ by far more than the tolerance (live backward switches), so the check catches a lost or
swapped mode.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "adjoint" / "multiweek_grad.py"
CACHE = Path("/work/ab0995/a270088/MIT/runs/adjoint/init_ref_ff_serial13_1day")   # multiweek_grad.CACHE
OUT_ROOT = Path("/work/ab0995/a270088/MIT/runs_jax/tier2")

# NVIDIA A100-SXM4-80GB, default XLA flags, sum_unroll=5, repeat r0 of job 27640768 (its r1: J bitwise, gradient
# within 1.3e-12 relative).
RECORDED = {
    "J": 12.625360011875072,
    "grad_norm": {"kapGM": 5.6945909824180237e-08, "taux": 0.00537294324980697, "tauy": 0.004740636386815274,
                  "tflux": 2.27951920080843e-08, "theta": 0.01737974522776973},
    "samples": {"theta_A_centre": 0.00023574252461886557, "theta_B_above": 9.934044049503181e-06,
                "theta_C_south": 2.528885957148426e-06},
    "dirderiv": {"theta_A_centre": 0.00023574252461886557, "theta_B_above": 9.934044049503181e-06,
                 "theta_C_south": 2.528885957148426e-06, "kapGM_scale": 0.0002309557511629183,
                 "tflux_box": -1.7847992934417733e-07},
}
# the same, ecco mode (AdjointConfig.ecco of the run directory), job 27644489
RECORDED_ECCO = {
    "J": 12.625360011875072,
    "grad_norm": {"kapGM": 5.680904279325833e-08, "taux": 0.005368278929695501, "tauy": 0.004740824420998291,
                  "tflux": 2.310650191792681e-08, "theta": 0.01738356628963292},
    "samples": {"theta_A_centre": 0.00023572316446004887, "theta_B_above": 1.0521747426151244e-05,
                "theta_C_south": 2.5839685024800945e-06},
    "dirderiv": {"theta_A_centre": 0.00023572316446004887, "theta_B_above": 1.0521747426151244e-05,
                 "theta_C_south": 2.5839685024800945e-06, "kapGM_scale": 0.00023026136954146047,
                 "tflux_box": -1.7663697878696547e-07, "taux_box": 0.010641482772762263},
}


def _run(mode):
    out = OUT_ROOT / (f"adjoint_regression_{mode}_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}_"
                      f"{int(time.time())}")
    env = dict(os.environ)
    env["XLA_FLAGS"] = ""                    # production GPU flags (see the module docstring)
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), "--days", "1", "--chunk", "12", "--mode", mode,
                        "--actions", "grad"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    return [x for x in rows if x["action"] == "grad"][0]


def _check(row, rec):
    assert row["grad_finite"]
    assert row["nsteps"] == 24 and row["xla_flags"] == ""
    assert rec is not None, f"no recorded values for mode {row['mode']}; this run: {json.dumps(row)[:1500]}"
    assert row["J"] == pytest.approx(rec["J"], rel=1e-14, abs=0.0)
    for group in ("grad_norm", "samples", "dirderiv"):
        for k, v in rec[group].items():
            assert row[group][k] == pytest.approx(v, rel=1e-9, abs=1e-20), (group, k)


def test_adjoint_regression_one_day():
    """exact and ecco mode over one day: J and gradients as recorded; same J in both modes, gradients that differ."""
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    assert gpus, f"tier-2 GPU test needs a GPU, found {jax.devices()}"
    assert (CACHE / "state.npz").exists(), (
        f"initial-state cache {CACHE} missing: build it with multiweek_grad.py --actions cache (CPU, gate flags)")
    rows = {m: _run(m) for m in ("exact", "ecco")}
    print(json.dumps({m: {k: r[k] for k in ("J", "grad_norm", "samples", "dirderiv", "job")}
                      for m, r in rows.items()}))
    assert rows["exact"]["J"] == rows["ecco"]["J"]                          # the forward is mode-independent
    for m, rec in (("exact", RECORDED), ("ecco", RECORDED_ECCO)):
        _check(rows[m], rec)
    ge, gc = rows["exact"]["dirderiv"], rows["ecco"]["dirderiv"]
    assert abs(ge["theta_B_above"] - gc["theta_B_above"]) / abs(ge["theta_B_above"]) > 1e-4, (ge, gc)
