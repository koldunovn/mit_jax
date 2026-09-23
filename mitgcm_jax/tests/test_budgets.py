"""Global volume / SSH / heat / salt budgets close to round-off (plan Task 19; mitgcm_jax/diagnostics/budgets.py has
the discrete equations with their Fortran citations).

Set-up: the FORCED oracle state at iteration 1 with the production geothermal flux (ref_ff_jaxdump_v4 run
directory's geothermalFile) put into the grid, so every budget term of V4r4 is live; two JAX FORWARD_STEPs.
Closure criterion: |residual| <= TOL[budget] x the round-off floor (`floor_*` = eps x the quadrature sum of the cell
contents: the size of one random rounding error per cell). Measured (2026-09-23, 16 cores, conftest XLA flags), 48
steps (24 with, 24 without geothermal, dev/budgets/run24_*.txt): residual / floor has random sign, no drift (the
24-step sum is -75 floors, a random walk of sd 19 predicts 93), max |ratio| volume 5.4, SSH 0.76, heat 37, salt 7.7
(sd 2, 0.3, 19, 4). Relative to the step's gross forcing: volume ~1e-13, SSH ~5e-16, heat ~1e-13..6e-13, salt
~1e-10 (salt content is 35x larger than its forcing). The Fortran's own steps give the same (test_fortran_steps_close:
ratio <= 3.5 in 2+2 steps). The heat ratios are larger than salt/volume at the same floor; not localised (the floor
counts one rounding per cell, the implicit vertical solve does more where the diffusivity is large). Tolerances are
~5-10x the measured maxima; every negative control misses by >= FAIL_FLOOR floors (measured >= 2e5).
Negative controls (each must fail): budget side - drop the fresh-water flux, drop the geothermal flux, count the salt
plume only at the surface, add the fresh-water heat/salt at the surface values, pair the tracer with the State's
hFacC (one step behind); model side - the model runs with the geothermal flux x (1 + 1e-5) (the budget keeps the
original), and the model runs with temp_EvPrRn/salt_EvPrRn UNSET (fresh water at the local surface values; the
default closure fails, the closure with the fresh-water terms holds again).
Scan composition: integrate_with_diagnostics (lax.scan: means in the carry, budgets and monitor as outputs) matches
the Python loop."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.adjoint import checkpoint as ck
from mitgcm_jax.diagnostics import budgets as bd
from mitgcm_jax.diagnostics import means as mn
from mitgcm_jax.diagnostics.monitor import dynstat_device
from mitgcm_jax.model import _extra_grid_fields, setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import State, state_from_dump
from mitgcm_jax.tests import oracle

NSTEPS = 2
TOL = {"volume": 30.0, "ssh": 10.0, "heat": 200.0, "salt": 50.0}  # floors; measured max 5.4, 0.76, 37, 7.7
FAIL_FLOOR = 1.0e4    # every negative control must miss by at least this many floors (measured >= 2e5)
GEO_PERTURB = 1e-5    # relative perturbation of the geothermal flux the model sees (model-side control)
GEO_RUN = "ref_ff_jaxdump_v4"
BUDGETS = ("volume", "ssh", "heat", "salt")


def _floor_ratio(b, k):
    return abs(float(b["resid_" + k])) / float(b["floor_" + k])


@pytest.fixture(scope="module")
def run():
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    P, g0, ex, kLowC = setup(rundir)
    bd.check_config(P)
    geodir = oracle.run_dir(GEO_RUN)
    geo = _extra_grid_fields(RunNamelists(geodir), g0, ex, geodir)["geothermalFlux"]
    g = g0.replace(geothermalFlux=jnp.asarray(geo))
    nml = RunNamelists(rundir)
    st = state_from_dump(ds, 1)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    st0 = State({k: jnp.asarray(v) for k, v in st.f.items()}, jnp.asarray(1))
    xs = ck.exf_window(ck.exf_loader_at(P, g, rundir, nml, 1), nml, 1, NSTEPS)
    model = ck.Model(P, g, kLowC, ex)
    step = ck.make_step()
    st0 = ck.prepare_state(step, model, st0, jax.tree.map(lambda a: a[0], xs))
    step_jit = jax.jit(step)
    sts = [st0]
    for n in range(NSTEPS):
        sts.append(step_jit(model, sts[-1], jax.tree.map(lambda a: a[n], xs)))
    bud = jax.jit(bd.step_budget, static_argnames=("pair", "drop", "add"))
    return dict(P=P, g=g, g0=g0, ex=ex, kLowC=kLowC, model=model, step=step, step_jit=step_jit, xs=xs, sts=sts,
                bud=bud)


def test_budgets_close(run):
    """Every step and the 2-step sum: volume, SSH, heat and salt residuals within TOL round-off floors; the salt
    plume redistribution and the shortwave column weights conserve (plume_net, sw_bottom at round-off); every
    forcing term is live (geothermal, fresh water, salt plume nonzero)."""
    P, g, kLowC, sts, bud = run["P"], run["g"], run["kLowC"], run["sts"], run["bud"]
    acc = bd.budget_acc_init()
    floors = {k: 0.0 for k in BUDGETS}
    for a, b in zip(sts[:-1], sts[1:]):
        r = bud(P, g, kLowC, a, b)
        for k in BUDGETS:
            ratio = _floor_ratio(r, k)
            print(f"{k}: resid {float(r['resid_' + k]): .3e} floor {float(r['floor_' + k]):.3e} ratio {ratio:.2f}")
            assert ratio <= TOL[k], (k, ratio)
            floors[k] += float(r["floor_" + k]) ** 2
        assert abs(float(r["plume_net"])) <= 1e-12 * float(r["plume_gross"])
        assert abs(float(r["sw_bottom"])) <= 1e-12 * abs(float(r["qnet"]))
        for k, fl in (("geo", "heat"), ("qnet", "heat"), ("fw", "volume"), ("plume_gross", "salt"),
                      ("saltflux", "salt")):
            assert abs(float(r[k])) > FAIL_FLOOR * float(r["floor_" + fl]), k
        acc = bd.budget_acc_add(acc, r)
    for k in BUDGETS:
        assert abs(float(acc["resid_" + k])) <= TOL[k] * np.sqrt(floors[k]), k


@pytest.mark.parametrize("drop,add,pair,fails", [
    (("fw",), (), "rstar", ("volume", "ssh")),
    (("geo",), (), "rstar", ("heat",)),
    (("plume_at_depth",), (), "rstar", ("salt",)),
    ((), ("fw_heat",), "rstar", ("heat",)),
    ((), ("fw_salt",), "rstar", ("salt",)),
    ((), (), "hFacC", ("volume", "heat", "salt")),
])
def test_budget_side_negative_controls(run, drop, add, pair, fails):
    """Dropping or adding one term, or pairing the tracer with the State's own (lagging) hFacC, breaks the closure by
    >= FAIL_FLOOR floors in both steps, and only in the budgets that contain the term."""
    P, g, kLowC, sts, bud = run["P"], run["g"], run["kLowC"], run["sts"], run["bud"]
    for a, b in zip(sts[:-1], sts[1:]):
        r = bud(P, g, kLowC, a, b, pair=pair, drop=drop, add=add)
        for k in BUDGETS:
            ratio = _floor_ratio(r, k)
            if k in fails:
                assert ratio >= FAIL_FLOOR, (k, ratio)
            else:
                assert ratio <= TOL[k], (k, ratio)


def test_model_side_negative_controls(run):
    """(1) The model steps with geothermalFlux x (1 + GEO_PERTURB), the budget uses the original flux: the heat budget
    fails, the others close. (2) The model runs with temp_EvPrRn and salt_EvPrRn UNSET (external_forcing_surf.F:
    257-277 skipped: fresh water enters/leaves at the local surface temperature and salinity): the V4r4 closure fails
    for heat and salt, the closure with the fresh-water terms (add=fw_heat, fw_salt) holds."""
    P, g, kLowC, sts, bud, xs = run["P"], run["g"], run["kLowC"], run["sts"], run["bud"], run["xs"]
    step_jit, ex = run["step_jit"], run["ex"]
    x0 = jax.tree.map(lambda a: a[0], xs)
    gbad = g.replace(geothermalFlux=g.geothermalFlux * (1.0 + GEO_PERTURB))
    s1 = step_jit(ck.Model(P, gbad, kLowC, ex), sts[0], x0)
    r = bud(P, g, kLowC, sts[0], s1)
    print("geothermal x (1+GEO_PERTURB): heat ratio", _floor_ratio(r, "heat"))
    assert _floor_ratio(r, "heat") >= FAIL_FLOOR
    for k in ("volume", "ssh", "salt"):
        assert _floor_ratio(r, k) <= TOL[k], k
    sf = dataclasses.replace(P.sf, temp_EvPrRn_set=False, salt_EvPrRn_set=False)
    Pu = P._replace(sf=sf)
    s1 = step_jit(ck.Model(Pu, g, kLowC, ex), sts[0], x0)
    r = bud(P, g, kLowC, sts[0], s1)
    print("EvPrRn unset: heat ratio", _floor_ratio(r, "heat"), "salt ratio", _floor_ratio(r, "salt"))
    assert _floor_ratio(r, "heat") >= FAIL_FLOOR and _floor_ratio(r, "salt") >= FAIL_FLOOR
    r = bud(P, g, kLowC, sts[0], s1, add=("fw_heat", "fw_salt"))
    print("EvPrRn unset, fw terms: heat ratio", _floor_ratio(r, "heat"), "salt ratio", _floor_ratio(r, "salt"))
    for k in BUDGETS:
        assert _floor_ratio(r, k) <= TOL[k], k


def test_fortran_steps_close():
    """The same closure on the Fortran's own steps: S00_begin dumps of ref_ff_jaxdump_v4 (production V4r4 flux-forced:
    useCTRL=T, geothermal flux on) at iterations 1, 2, 3. No JAX step is involved: this checks the derivation against
    MITgcm itself. Measured residual/floor <= 3.5 (both steps, all four budgets)."""
    ds = oracle.dumpset(GEO_RUN)
    rundir = oracle.run_dir(GEO_RUN)
    P, g, ex, kLowC = setup(oracle.run_dir(oracle.FORCED))
    np.testing.assert_array_equal(np.asarray(g.h0FacC), oracle.field(ds, 1, "G00_geometry", "h0FacC"))
    g = g.replace(**{k: jnp.asarray(v) for k, v in _extra_grid_fields(RunNamelists(rundir), g, ex, rundir).items()})
    assert float(jnp.max(g.geothermalFlux)) > 0.0
    names = ("theta", "salt", "etaH", "rStarFacC", "EmPmR", "Qnet", "Qsw", "saltFlux", "saltPlumeFlux",
             "saltPlumeDepth")
    sts = [State({n: jnp.asarray(oracle.field(ds, it, "S00_begin", n)) for n in names}, it) for it in (1, 2, 3)]
    bud = jax.jit(bd.step_budget)
    for a, b in zip(sts[:-1], sts[1:]):
        r = bud(P, g, kLowC, a, b)
        for k in BUDGETS:
            ratio = _floor_ratio(r, k)
            print(f"Fortran {a.it}->{b.it} {k}: ratio {ratio:.2f} rel {float(r.get('rel_' + k, np.nan)):.2e}")
            assert ratio <= TOL[k], (k, ratio)
        assert float(r["geo"]) > 1e3 * float(r["floor_heat"])


def test_scan_diagnostics_match_loop(run):
    """integrate_with_diagnostics (lax.scan with the time means in the carry and the budgets + on-device monitor as
    per-step outputs) gives the loop's final State, the mean of the loop's snapshots, and the loop's budgets."""
    P, g, kLowC, sts, bud = run["P"], run["g"], run["kLowC"], run["sts"], run["bud"]
    model, step, xs = run["model"], run["step"], run["xs"]

    @jax.jit
    def scan(model, st, xs):
        budget = functools.partial(bd.step_budget, model.P, model.g, model.kLowC)
        monitor = functools.partial(dynstat_device, g=model.g)
        return mn.integrate_with_diagnostics(step, model, st, xs, budget=budget, monitor=monitor)

    st_n, acc, ys = scan(model, sts[0], xs)
    for k in ("theta", "salt", "etaN", "uVel", "rStarFacC"):
        np.testing.assert_allclose(np.asarray(st_n.f[k]), np.asarray(sts[-1].f[k]), rtol=0, atol=1e-12, err_msg=k)
    m = mn.means_finish(acc)
    for k in mn.MEAN_FIELDS:
        snaps = np.stack([np.asarray(mn._get(s, k)) for s in sts[1:]])
        np.testing.assert_allclose(np.asarray(m[k]), snaps.mean(axis=0), rtol=1e-13, atol=1e-13, err_msg=k)
    assert float(acc["w"]) == NSTEPS
    for n, (a, b) in enumerate(zip(sts[:-1], sts[1:])):
        r = bud(P, g, kLowC, a, b)
        for k in ("dV", "dH", "dS", "qnet", "geo", "fw"):
            np.testing.assert_allclose(float(ys["budget"][k][n]), float(r[k]), rtol=1e-10, err_msg=k)
        for k in BUDGETS:
            assert abs(float(ys["budget"]["resid_" + k][n])) <= TOL[k] * float(r["floor_" + k]), k
        np.testing.assert_allclose(float(ys["monitor"]["theta"]["mean"][n]),
                                   float(dynstat_device(b, g)["theta"]["mean"]), rtol=1e-13)
