"""Initialisation gate (plan Task 8, tier 1): `init.state_from_pickup` (INITIALISE_VARIA from the run directory's
pickup files, literal port) equals the Fortran state at the start of iteration 1 of the FORCED oracle -- every field of
`state_from_dump` (S00_begin + G00_geometry group R), halos included -- bitwise (conftest XLA flags: no FMA, no
algsimp). Measured 2026-09-23: 0 differing values in all 88 fields.
Negative controls: a pickup theta value perturbed by 1e-6 (relative) at one wet point changes theta and totPhiHyd; a
wrong Adams-Bashforth start level (mom_StartAB=0 instead of CHECK_PICKUP's 1) is refused.
The SMOKE oracle, one FORWARD_STEP from the pickup state and the AB-flag step control are in test_init_step.py
(tier1x).
"""

import dataclasses

import numpy as np
import pytest

from mitgcm_jax import init
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle


@pytest.fixture(scope="module")
def forced():
    rundir = oracle.run_dir(oracle.FORCED)
    ds = oracle.dumpset(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir)
    st, aux = init.state_from_pickup(P, g, ex, kLowC, rundir, return_aux=True)
    return ds, rundir, P, g, ex, kLowC, st, aux


def test_init_forced_bitwise(forced):
    ds, rundir, P, g, ex, kLowC, st, aux = forced
    assert st.it == 1
    assert aux["pickup"].mom_StartAB == 1 and aux["pickup"].missing == ()
    assert float(aux["cg2dNorm"]) == P.cg.cg2dNorm        # INI_CG2D myNorm (ini_cg2d.F:82-87) == setup's cg2dNorm
    ref = state_from_dump(ds, 1)
    assert set(ref.f) <= set(st.f), sorted(set(ref.f) - set(st.f))
    for k in sorted(ref.f):
        np.testing.assert_array_equal(np.asarray(st.f[k]), ref.f[k], err_msg=k)
    np.testing.assert_array_equal(np.asarray(st.f["runoff"]),
                                  np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))


def test_negative_control_pickup_value(forced):
    ds, rundir, P, g, ex, kLowC, st, aux = forced
    cfg = init.InitConfig.from_namelists(RunNamelists(rundir))
    pk, tke, exf, _ = init.init_inputs(P, rundir, cfg, g.layout)
    wet = np.argwhere(pk["theta"] != 0.0)
    t, k, j, i = wet[len(wet) // 2]                          # a wet interior point
    pk["theta"] = pk["theta"].copy()
    pk["theta"][t, k, j, i] *= 1 + 1e-6
    f, _ = init.initialise_varia_jit(ex)(P, g, kLowC, pk, tke, exf)
    for name in ("theta", "totPhiHyd"):
        assert not np.array_equal(np.asarray(f[name]), oracle.field(ds, 1, "S00_begin", name)), name


def test_negative_control_ab_start_refused(forced):
    ds, rundir, P, g, ex, kLowC, st, aux = forced
    bad = P._replace(dyn=dataclasses.replace(P.dyn, ts=dataclasses.replace(P.dyn.ts, mom_StartAB=0)))
    with pytest.raises(ValueError, match="mom_StartAB"):
        init.state_from_pickup(bad, g, ex, kLowC, rundir)


def test_check_pickup_start_levels():
    """CHECK_PICKUP (check_pickup.F:60-214) on synthetic missing-field lists (nIter0 = 5)."""
    cfg = init.InitConfig(nIter0=5, pickup="pickup.0000000005", pickup_ggl90="pickup_ggl90.0000000005",
                          pickupStrictlyMatch=False, readGuNm1=True, readGuNm2=True, m1=1, m2=2, useEXF=True,
                          useGGL90=True, useGMRedi=True, useSALT_PLUME=True)
    assert init.check_pickup(cfg, 11, (), ()).mom_StartAB == 5
    assert init.check_pickup(cfg, 9, (), ("GuNm2", "GvNm2")).mom_StartAB == 1
    assert init.check_pickup(cfg, 7, (), ("GuNm1", "GuNm2", "GvNm1", "GvNm2")).mom_StartAB == 0
    for fatal in ("Theta", "EtaH", "dEtaHdt", "Foo"):
        with pytest.raises(ValueError):
            init.check_pickup(cfg, 10, (), (fatal,))
    with pytest.raises(ValueError):
        init.check_pickup(dataclasses.replace(cfg, pickupStrictlyMatch=True), 9, (), ("GuNm2",))
