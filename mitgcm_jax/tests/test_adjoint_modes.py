"""Backward-mode semantics (plan Task 17): fast checks (tier 1). Gradient effect tests: test_adjoint_modes_grad.py.

1. AdjointConfig.ecco from the flux-forced run's data.autodiff / data.pkg gives the TAF semantics of that build
   (docs/ADJOINT_MODES.md; STDOUT.0000 of the FORCED oracle prints useGGL90inAdMode=F, useSALT_PLUMEinAdMode=T,
   useGMRediInAdMode=T, inAdExact=T, viscFacInAd=1); the published full-V4r4 namelists give the full-tree config.
2. Seam census: the exact mode inserts no seam primitive; each switch adds exactly its own stop_gradient /
   custom_jvp_call equations to the traced step (so every seam is wired, and nothing else changes).
3. Forward byte-identical: the step with EVERY switch on (GGL90 frozen, stable sigma, salt plume off, passive cg2d
   operator, viscFacInAd = 2) is bitwise equal to the Fortran at the dumped stages of step 1 and at the start of
   iteration 2 of the FORCED oracle, cg2d 164 iterations (measured 2026-09-23: 0 difference). The exact mode is
   test_step_fluxforced.py.
"""

import jax
import numpy as np
import pytest

from mitgcm_jax.adjoint.modes import AdjointConfig
from mitgcm_jax.core.forward_step import forward_step
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle
from mitgcm_jax.tests.test_step_fluxforced import END, STAGES

FULL_V4R4_NAMELISTS = oracle.REPO / "ECCO-v4-Configurations" / "ECCOv4 Release 4" / "namelist"
ALL_ON = AdjointConfig(ggl90="frozen", gm_sigma="stable", salt_plume="off", cg2d="passive", visc_fac_in_ad=2.0)


@pytest.fixture(scope="module")
def run():
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir, grid=grid_from_dump(ds, 1))
    st = state_from_dump(ds, 1)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    nml = RunNamelists(rundir)
    myTime, myIter = exf_mod.model_time(nml, 1)
    bufs, facs, _ = exf_mod.ExfRecordLoader(P.exf, g, rundir).load(myTime, myIter)
    exf_in = {"bufs": bufs, "facs": facs, "myTime": myTime}
    return ds, nml, P, g, ex, kLowC, st, exf_in


def test_config_from_namelists(run):
    nml = run[1]
    ecco = AdjointConfig.ecco(nml)
    assert ecco == AdjointConfig(ggl90="frozen", gm_sigma="stable", salt_plume="exact", cg2d="passive",
                                 visc_fac_in_ad=1.0)
    assert AdjointConfig().is_exact and not ecco.is_exact
    assert hash(ecco) == hash(AdjointConfig.ecco(nml))            # static: usable as a jit static argument
    with pytest.raises(ValueError):
        AdjointConfig(ggl90="off")
    assert ecco.seaice == "ecco" and AdjointConfig().seaice == "ecco"   # the sea-ice default (no sea ice here)
    # the full V4r4 tree (useSEAICE=T, plan M2.6b-2): its data.autodiff switches sea ice, GGL90 and the salt plume
    # off in the adjoint (tests/test_adjoint_modes_full.py gates the full-tree seams)
    assert AdjointConfig.ecco(RunNamelists(FULL_V4R4_NAMELISTS)) == AdjointConfig(
        ggl90="frozen", gm_sigma="stable", salt_plume="off", cg2d="passive", visc_fac_in_ad=1.0, seaice="ecco")


def _census(run, adj):
    ds, nml, P, g, ex, kLowC, st, exf_in = run
    jp = jax.make_jaxpr(lambda P, g, kLowC, st, exf_in: forward_step(P, g, ex, kLowC, st, exf_in, adj=adj)[0])(
        P, g, kLowC, st, exf_in)
    s = str(jp)
    return s.count(" stop_gradient "), s.count("custom_jvp_call")


def test_seam_census(run):
    """stop_gradient seams: gm_sigma 3 (sigmaX/Y/R; "stable" and "gm_only" alike), ggl90 4 (TKE, viscArU/V, diffKr), cg2d 3 (aW2d, aS2d, aC2d),
    salt_plume 2 (saltPlumeFlux, saltPlumeDepth); visc_fac_in_ad: 1 custom_jvp_call (MOM_VECINV)."""
    sg0, cj0 = _census(run, AdjointConfig())
    assert cj0 == 1   # the CG2D implicit derivative (core/cg2d.py _cg2d_implicit, custom_jvp since plan Task 18)
    expect = {AdjointConfig(gm_sigma="stable"): (3, 0), AdjointConfig(gm_sigma="gm_only"): (3, 0),
              AdjointConfig(ggl90="frozen"): (4, 0),
              AdjointConfig(cg2d="passive"): (3, 0), AdjointConfig(salt_plume="off"): (2, 0),
              AdjointConfig(visc_fac_in_ad=1.0): (0, 1), AdjointConfig.ecco(run[1]): (10, 1), ALL_ON: (12, 1)}
    for adj, (dsg, dcj) in expect.items():
        sg, cj = _census(run, adj)
        assert (sg - sg0, cj - cj0) == (dsg, dcj), adj


def test_forward_bitwise_all_switches(run):
    ds, nml, P, g, ex, kLowC, st, exf_in = run
    step = jax.jit(lambda P, g, kLowC, st, exf_in: forward_step(P, g, ex, kLowC, st, exf_in, adj=ALL_ON))
    st1, aux = step(P, g, kLowC, st, exf_in)
    for key, fld, stage, dfld in STAGES:
        np.testing.assert_array_equal(np.asarray(aux[key][fld]), oracle.field(ds, 1, stage, dfld), err_msg=stage)
    assert int(aux["cg2d"]["numIters"]) == 164
    for k in END:
        np.testing.assert_array_equal(np.asarray(st1.f[k]), oracle.field(ds, 2, "S00_begin", k), err_msg=k)
