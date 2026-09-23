"""%MON dynstat statistics (host-side, mitgcm_jax/diagnostics/monitor.py) reproduce the Fortran monitor lines of the
forced oracle run from its dumped states (S00_begin of iteration n == the model state the monitor printed at
time_tsnumber n). Printed values have 14 significant digits; the comparison allows 2e-13 relative (and an absolute
floor for values that are ~0, e.g. wvel mean)."""

import numpy as np
import pytest

from mitgcm_jax.diagnostics.monitor import dynstat
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.io.monitor import read_monitor
from mitgcm_jax.layout import Layout
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle

L = Layout()


@pytest.mark.parametrize("it", [1, 2, 3])
def test_dynstat_matches_fortran_monitor(it):
    ds = oracle.dumpset(oracle.FORCED)
    g = grid_from_dump(ds, it)
    st = state_from_dump(ds, it)
    mon = read_monitor(oracle.run_dir(oracle.FORCED) / "STDOUT.0000")[it]
    stats = dynstat(st, g, L)
    worst = 0.0
    for name, s in stats.items():
        for k in ("max", "min", "mean", "sd", "del2"):
            ref = mon[f"dynstat_{name}_{k}"]
            err = abs(s[k] - ref) / max(abs(ref), 1e-12 if name != "wvel" else 1e-9)
            worst = max(worst, err)
            assert err < 2e-13, (name, k, s[k], ref, err)


def test_negative_control_detects_perturbed_state():
    ds = oracle.dumpset(oracle.FORCED)
    g = grid_from_dump(ds, 1)
    st = state_from_dump(ds, 1)
    mon = read_monitor(oracle.run_dir(oracle.FORCED) / "STDOUT.0000")[1]
    bad = st.replace(theta=np.asarray(st.theta) * (1 + 1e-9))
    s = dynstat(bad, g, L)["theta"]
    assert abs(s["mean"] - mon["dynstat_theta_mean"]) / abs(mon["dynstat_theta_mean"]) > 2e-13
