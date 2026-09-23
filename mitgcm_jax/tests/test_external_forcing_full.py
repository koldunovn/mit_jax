"""Full V4r4 tree (plan M2.6a): the ocean-side forcing chain around SEAICE_MODEL, gated against oracle.FULL
(full_jaxdump_v5, iterations 1-3). The sea-ice and EXF-bulk kernels themselves are other modules; these gates replay
the dumped inputs of each ocean stage.

  - CTRL_MAP_FORCING (forward_step.F:524-530, useCTRL = T): S02_load_fields -> S03_ctrl_map_forcing. The full
    data.ctrl has no xx_gentim2d control (gcov ref_full_serial13_gcov_1day: ctrl_map_forcing.f:4131-4147 never run,
    CTRL_GET_GEN not called), so the call is the FFIELDS exchanges (ctrl_map_forcing.F:365-372) only, and those change
    no value here: c66g EXF_MAPFIELDS has already exchanged Qnet, EmPmR, fu/fv, Qsw, pLoad (exf_mapfields.F:354-370)
    and sets saltFlux = saltflx on the full array (:331-337; saltflxfile unset in the full data.exf: the constant 0),
    unlike the flux-forced override, whose saltFlux halos CTRL_MAP_FORCING rewrites. S03 == S02 bitwise.
  - DO_OCEANIC_PHYS before SEAICE_MODEL (c66g do_oceanic_phys.F:286-298, no READIN_SALT_PLUME_FLUX in the full
    build): saltPlumeDepth AND saltPlumeFlux are zeroed; SEAICE_GROWTH then writes saltPlumeFlux on the interior only
    (V4r4 seaice_growth.F:2050-2127, i = 1..sNx, j = 1..sNy) and SEAICE_MODEL does not exchange it, so the halos of
    saltPlumeFlux at P00_seaice_model are the zeros of :293 -- the observable trace of the zeroing (the S02 input of
    iteration 2 holds the previous step's exchanged, non-zero halos).
  - DO_OCEANIC_PHYS after SEAICE_MODEL (c66g :573-608): SALT_PLUME_DO_EXCH + EXTERNAL_FORCING_SURF, P00 -> P01:
    surfaceForcingU/V/T/S, phi0surf (with the sea-ice load sIceLoad of SEAICE_GROWTH), saltPlumeFlux exchanged.
    In the full tree temp_EvPrRn is unset (UNSET_RL): surfaceForcingT has no PmEpR*(temp_EvPrRn - theta) term
    (external_forcing_surf.F:257-266; the flux-forced data sets temp_EvPrRn = 0.).
Achieved (2026-09-23): bitwise at every point, halos included, all three iterations, jitted and eager.
Negative controls: temp_EvPrRn set to 0 (the ff value), the salt-plume term dropped, the ff (READIN) behaviour of the
pre-seaice zeroing, and the CTRL_MAP_FORCING fu/fv exchange without signs: each fails its comparison.
Gradient: d(sum W*P01 outputs)/d(Qnet, EmPmR, saltPlumeFlux, sIceLoad, salt) finite and equal to central differences.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core import external_forcing as ef
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.pkgs import ctrl as ctrl_mod
from mitgcm_jax.tests import oracle

ORACLE = oracle.FULL
ITERS = (1, 2, 3)
FFIELDS = ["fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad", "sIceLoad", "saltPlumeFlux"]
SURF_OUT = ["surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "phi0surf"]
CTRL_FIELDS = ["fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad"]


@pytest.fixture(scope="module")
def ds():
    return oracle.dumpset(ORACLE)


@pytest.fixture(scope="module")
def nml():
    return RunNamelists(oracle.run_dir(ORACLE))


@pytest.fixture(scope="module")
def g(ds):
    return grid_from_dump(ds, 1)


@pytest.fixture(scope="module")
def ex():
    return default_exchanger()


@pytest.fixture(scope="module")
def q(nml):
    return ef.SurfForcingParams.from_namelists(nml)


def F(ds, it, stage, name):
    return oracle.field(ds, it, stage, name)


def post(q, g, ex, ds, it, jit=True):
    ff = {n: jnp.asarray(F(ds, it, "P00_seaice_model", n)) for n in FFIELDS}
    theta = jnp.asarray(F(ds, it, "S00_begin", "theta"))
    salt = jnp.asarray(F(ds, it, "S00_begin", "salt"))
    fn = lambda q, g, ff, theta, salt: ef.oceanic_phys_post_seaice(q, g, ex, ff, theta, salt)  # noqa: E731
    return (jax.jit(fn) if jit else fn)(q, g, ff, theta, salt)


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def halo(a, L):
    m = np.ones(a.shape, bool)
    m[..., L.js(1, L.sNy), L.is_(1, L.sNx)] = False
    return np.asarray(a)[m]


def test_params_full_tree(q, nml):
    """Full tree: useSEAICE, saltPlumeFlux zeroed before SEAICE_MODEL (no READIN_SALT_PLUME_FLUX), temp_EvPrRn unset
    (set_defaults.F:259 UNSET_RL), salt_EvPrRn = 0 (set_defaults.F:260); flux-forced: the opposite."""
    assert q.useSEAICE and q.zero_salt_plume_flux and not ef.readin_salt_plume_flux(nml)
    assert not q.temp_EvPrRn_set and q.salt_EvPrRn_set and q.salt_EvPrRn == 0.0
    qf = ef.SurfForcingParams.from_namelists(RunNamelists(oracle.run_dir(oracle.FORCED)))
    assert not qf.useSEAICE and not qf.zero_salt_plume_flux and qf.temp_EvPrRn_set
    with pytest.raises(ValueError):          # the ff composition refuses the full tree (SEAICE_MODEL in between)
        ef.oceanic_phys_forcing(q, None, None, {}, None, None, None)


@pytest.mark.parametrize("it", ITERS)
def test_external_forcing_surf_P01(q, g, ex, ds, it):
    """P00_seaice_model (FFIELDS after SEAICE_MODEL) + S00 theta/salt -> P01_external_forcing_surf, every point."""
    for jit in (True, False):
        ff, out = post(q, g, ex, ds, it, jit=jit)
        errs = {n: maxdiff(out[n], F(ds, it, "P01_external_forcing_surf", n)) for n in SURF_OUT}
        for n in FFIELDS:
            errs[n] = maxdiff(ff[n], F(ds, it, "P01_external_forcing_surf", n))
        assert max(errs.values()) == 0.0, (jit, errs)
    # not vacuous: ice load and brine-rejection plume flux present, halos of saltPlumeFlux filled by the exchange
    assert np.abs(F(ds, it, "P01_external_forcing_surf", "sIceLoad")).max() > 100.0
    ref = F(ds, it, "P01_external_forcing_surf", "saltPlumeFlux")
    assert np.abs(ref).max() > 0 and np.abs(halo(ref, g.layout)).max() > 0


@pytest.mark.parametrize("it", (2, 3))
def test_pre_seaice_zeroing(q, g, ds, it):
    """oceanic_phys_pre_seaice on the S02 fields (DO_OCEANIC_PHYS entry): its saltPlumeFlux halos equal P00's (the
    zeros SEAICE_GROWTH does not overwrite); the input halos differ (so the check is not vacuous); saltPlumeDepth
    zeroed (P01 holds 0 everywhere)."""
    L = g.layout
    ff_in = {n: jnp.asarray(F(ds, it, "S02_load_fields", n)) for n in ("saltPlumeFlux",)}
    ff, depth = jax.jit(ef.oceanic_phys_pre_seaice)(q, ff_in, jnp.asarray(F(ds, it, "S02_load_fields",
                                                                            "saltPlumeDepth")))
    p00 = F(ds, it, "P00_seaice_model", "saltPlumeFlux")
    np.testing.assert_array_equal(halo(ff["saltPlumeFlux"], L), halo(p00, L))
    assert np.abs(halo(ff_in["saltPlumeFlux"], L)).max() > 0
    assert np.all(np.asarray(depth) == 0.0)
    np.testing.assert_array_equal(np.asarray(depth), F(ds, it, "P01_external_forcing_surf", "saltPlumeDepth"))


@pytest.mark.parametrize("it", ITERS)
def test_ctrl_map_forcing_S03(nml, g, ex, ds, it):
    """CTRL_MAP_FORCING with no forcing control: S02 -> S03 (FFIELDS exchanges, ctrl_map_forcing.F:365-372)."""
    cc = ctrl_mod.CtrlConfig.from_namelists(nml)
    assert cc.tim2d == ()                   # no xx_gentim2d in the full data.ctrl
    ff = {n: jnp.asarray(F(ds, it, "S02_load_fields", n)) for n in CTRL_FIELDS}
    out = jax.jit(lambda ff: ctrl_mod.ctrl_map_forcing(cc, g, ex, ff))(ff)
    for n in CTRL_FIELDS:
        np.testing.assert_array_equal(np.asarray(out[n]), F(ds, it, "S03_ctrl_map_forcing", n), err_msg=n)
        # full tree: the input is already exchanged (c66g EXF_MAPFIELDS), CTRL_MAP_FORCING changes no value
        np.testing.assert_array_equal(F(ds, it, "S02_load_fields", n), F(ds, it, "S03_ctrl_map_forcing", n))


def test_negative_controls(q, g, ex, ds, nml):
    it = 1
    ref = {n: F(ds, it, "P01_external_forcing_surf", n) for n in SURF_OUT}
    # (1) temp_EvPrRn = 0 as in the flux-forced data: surfaceForcingT gains the PmEpR*(0 - theta) term
    bad = dataclasses.replace(q, temp_EvPrRn=0.0, temp_EvPrRn_set=True)
    _, out = post(bad, g, ex, ds, it)
    assert maxdiff(out["surfaceForcingT"], ref["surfaceForcingT"]) > 1e-12
    # (2) salt-plume term dropped (SALT_PLUME_FORCING_SURF)
    _, out = post(dataclasses.replace(q, useSALT_PLUME=False), g, ex, ds, it)
    assert maxdiff(out["surfaceForcingS"], ref["surfaceForcingS"]) > 1e-12
    # (3) the flux-forced pre-seaice behaviour (saltPlumeFlux kept): P00 halos not reproduced
    L = g.layout
    ff_in = {"saltPlumeFlux": jnp.asarray(F(ds, 2, "S02_load_fields", "saltPlumeFlux"))}
    ff, _ = ef.oceanic_phys_pre_seaice(dataclasses.replace(q, zero_salt_plume_flux=False), ff_in,
                                       jnp.zeros(L.shape2d))
    assert not np.array_equal(halo(ff["saltPlumeFlux"], L), halo(F(ds, 2, "P00_seaice_model", "saltPlumeFlux"), L))
    # (4) CTRL_MAP_FORCING with EXCH_UV_XY_RS(fu, fv) without signs (ctrl_map_forcing.F:372 passes .TRUE.)
    class NoSignEx:
        def exch_xy(self, a):
            return ex.exch_xy(a)

        def exch_uv_xy(self, u, v, withSigns):
            return ex.exch_uv_xy(u, v, not withSigns)

    cc = ctrl_mod.CtrlConfig.from_namelists(nml)
    ff = {n: jnp.asarray(F(ds, it, "S02_load_fields", n)) for n in CTRL_FIELDS}
    out = ctrl_mod.ctrl_map_forcing(cc, g, NoSignEx(), ff)
    assert not np.array_equal(np.asarray(out["fu"]), F(ds, it, "S03_ctrl_map_forcing", "fu"))


def test_gradient_fd(q, g, ex, ds):
    """J = sum(W * (surfaceForcingT, surfaceForcingS, surfaceForcingU, phi0surf)): dJ/d(Qnet, EmPmR, saltPlumeFlux,
    sIceLoad, fu) and dJ/dsalt(k=1) finite everywhere; central differences at a few wet points (h-sweep, plateau)."""
    it = 1
    rng = np.random.default_rng(3)
    ff0 = {n: jnp.asarray(F(ds, it, "P00_seaice_model", n)) for n in FFIELDS}
    theta = jnp.asarray(F(ds, it, "S00_begin", "theta"))
    salt0 = jnp.asarray(F(ds, it, "S00_begin", "salt"))
    W = {n: jnp.asarray(rng.normal(size=g.layout.shape2d)) for n in ("surfaceForcingT", "surfaceForcingS",
                                                                     "surfaceForcingU", "phi0surf")}

    def J(q, g, ff, salt):
        _, out = ef.oceanic_phys_post_seaice(q, g, ex, ff, theta, salt)
        return sum(jnp.sum(W[n] * out[n]) for n in W)

    Jj = jax.jit(J)
    gff, gsalt = jax.jit(jax.grad(J, argnums=(2, 3)))(q, g, ff0, salt0)
    for n in ("Qnet", "EmPmR", "saltPlumeFlux", "sIceLoad", "fu"):
        assert np.all(np.isfinite(np.asarray(gff[n]))), n
    assert np.all(np.isfinite(np.asarray(gsalt)))
    L = g.layout
    wet = np.argwhere(np.asarray(g.maskC)[:, 0, L.js(1, L.sNy), L.is_(1, L.sNx)] > 0)
    worst = 0.0
    for t, j, i in wet[rng.choice(len(wet), 4, replace=False)]:
        pt = (t, j + L.OLy, i + L.OLx)
        for n in ("Qnet", "EmPmR", "sIceLoad"):
            a = float(gff[n][pt])
            scale = max(1.0, float(np.abs(np.asarray(ff0[n])).max()))
            fd = []
            for h in (1e-2, 1e-3, 1e-4):
                hh = h * scale
                jp = float(Jj(q, g, dict(ff0, **{n: ff0[n].at[pt].add(hh)}), salt0))
                jm = float(Jj(q, g, dict(ff0, **{n: ff0[n].at[pt].add(-hh)}), salt0))
                fd.append((jp - jm) / (2 * hh))
            e = min(abs(f - a) for f in fd) / max(abs(a), 1e-30)
            worst = max(worst, e) if a != 0.0 else worst
    assert worst < 1e-6, worst
