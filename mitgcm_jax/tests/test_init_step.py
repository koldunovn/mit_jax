"""Initialisation gates, extended (plan Task 8, tier1x): `init.state_from_pickup` is bitwise equal to the SMOKE
oracle's state at iteration 1 (all 88 fields, halos included), and one FORWARD_STEP from the FORCED pickup state
equals the Fortran S00_begin of iteration 2 bitwise -- every S00_begin/G00 field the step carries, cg2d 164 iterations
(measured 2026-09-23: 0 differing values). Negative control: the step with a wrong Adams-Bashforth start level
(mom_StartAB=0 instead of CHECK_PICKUP's 1) changes uVel at iteration 2.

model.setup cannot build the SMOKE parameters (the EXF module rejects useEXF=F); the SMOKE run directory differs from
FORCED only in useEXF and nTimeSteps (data, data.pkg), and the initialisation reads P.exf only when useEXF=T, so the
SMOKE gate uses the FORCED parameters and grid with the SMOKE run directory (namelists, pickups).
"""

import dataclasses

import jax
import numpy as np
import pytest

from mitgcm_jax import init
from mitgcm_jax.core.forward_step import forward_step
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle


@pytest.fixture(scope="module")
def run():
    rundir = oracle.run_dir(oracle.FORCED)
    ds = oracle.dumpset(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir)
    st = init.state_from_pickup(P, g, ex, kLowC, rundir)
    nml = RunNamelists(rundir)
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    myTime, myIter = exf_mod.model_time(nml, 1)
    bufs, facs, _ = loader.load(myTime, myIter)
    exf_in = {"bufs": bufs, "facs": facs, "myTime": myTime}
    step = jax.jit(lambda P, g, kLowC, st, exf_in: forward_step(P, g, ex, kLowC, st, exf_in))
    return ds, P, g, ex, kLowC, st, exf_in, step


def _assert_state_equal(st, ds, it):
    ref = state_from_dump(ds, it)
    assert set(ref.f) <= set(st.f), sorted(set(ref.f) - set(st.f))
    for k in sorted(ref.f):
        np.testing.assert_array_equal(np.asarray(st.f[k]), ref.f[k], err_msg=k)


def test_init_smoke_bitwise(run):
    _, P, g, ex, kLowC, _, _, _ = run
    st = init.state_from_pickup(P, g, ex, kLowC, oracle.run_dir(oracle.SMOKE))
    _assert_state_equal(st, oracle.dumpset(oracle.SMOKE), 1)


def test_step1_from_pickup_bitwise(run):
    """The strongest check: FORWARD_STEP from the pickup state lands on the Fortran state of iteration 2."""
    ds, P, g, ex, kLowC, st, exf_in, step = run
    st1, a = step(P, g, kLowC, st, exf_in)
    assert int(a["cg2d"]["numIters"]) == 164
    _assert_state_equal(st1, ds, 2)


def test_negative_control_ab_start_step(run):
    ds, P, g, ex, kLowC, st, exf_in, step = run
    bad = P._replace(dyn=dataclasses.replace(P.dyn, ts=dataclasses.replace(P.dyn.ts, mom_StartAB=0)))
    st1, _ = step(bad, g, kLowC, st, exf_in)
    assert not np.array_equal(np.asarray(st1.f["uVel"]), oracle.field(ds, 2, "S00_begin", "uVel"))
