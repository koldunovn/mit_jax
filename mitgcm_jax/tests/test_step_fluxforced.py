"""Integrated FORWARD_STEP gate (plan Task 19): the composed JAX step, set up from the run's namelists and the grid
files (no dumped geometry), started from the Fortran state at iteration 1 of the FORCED oracle, is bitwise equal to
the Fortran at the dumped intermediate stages of step 1 and at the start of iteration 2 (conftest XLA flags: no FMA,
no algsimp). Measured 2026-09-23: 0 on every field checked, cg2d 164 iterations as in STDOUT.
Negative control: the same step with the GGL90 alpha constant perturbed by 1e-6 differs at the end of the step."""

import dataclasses

import jax
import numpy as np
import pytest

from mitgcm_jax.core.forward_step import forward_step
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle

STAGES = [("S02_load_fields", "Qnet", "S02_load_fields", "Qnet"),
          ("S04_oceanic_phys", "GGL90TKE", "P04_ggl90", "GGL90TKE"),
          ("S04_oceanic_phys", "Kwx", "P06_gmredi_exch", "Kwx"),
          ("S05_dynamics", "gU", "S05_dynamics", "gU"),
          ("S12_stagger_exchanges", "etaN", "S12_stagger_exchanges", "etaN"),
          ("S13_thermodynamics", "theta", "S13_thermodynamics", "theta")]
END = ("theta", "salt", "uVel", "vVel", "wVel", "etaN", "etaH", "GGL90TKE", "guNm_1", "rStarFacC")


@pytest.fixture(scope="module")
def run():
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir)
    st = state_from_dump(ds, 1)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    nml = RunNamelists(rundir)
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    myTime, myIter = exf_mod.model_time(nml, 1)
    bufs, facs, _ = loader.load(myTime, myIter)
    exf_in = {"bufs": bufs, "facs": facs, "myTime": myTime}
    step = jax.jit(lambda P, g, kLowC, st, exf_in: forward_step(P, g, ex, kLowC, st, exf_in))
    return ds, P, g, kLowC, st, exf_in, step


def test_step1_bitwise(run):
    ds, P, g, kLowC, st, exf_in, step = run
    st1, aux = step(P, g, kLowC, st, exf_in)
    for key, fld, stage, dfld in STAGES:
        np.testing.assert_array_equal(np.asarray(aux[key][fld]), oracle.field(ds, 1, stage, dfld), err_msg=stage)
    assert int(aux["cg2d"]["numIters"]) == 164
    for k in END:
        np.testing.assert_array_equal(np.asarray(st1.f[k]), oracle.field(ds, 2, "S00_begin", k), err_msg=k)


def test_negative_control_step(run):
    ds, P, g, kLowC, st, exf_in, step = run
    bad = P._replace(ggl=dataclasses.replace(P.ggl, GGL90alpha=P.ggl.GGL90alpha * (1 + 1e-6)))
    st1, _ = step(bad, g, kLowC, st, exf_in)
    assert not np.array_equal(np.asarray(st1.f["GGL90TKE"]), oracle.field(ds, 2, "S00_begin", "GGL90TKE"))
