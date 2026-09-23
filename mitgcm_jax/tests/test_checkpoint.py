"""Gradient drivers and checkpointing schedules (plan Task 18), on the live FORCED oracle state (CPU, tier1x).

- scan == loop: `checkpoint.integrate` (lax.scan, every schedule) is bitwise equal to the Python loop of a jitted
  FORWARD_STEP that scripts/run_jax.py runs, over 2 steps (conftest XLA flags).
- schedules: the gradient of J = final cost + running cost over a 2-step window w.r.t. the initial theta, salt, uVel,
  vVel, etaN is the same for per-step remat, sqrt(N) segments and chunked reverse accumulation (1-step chunks,
  boundary stride 1 and 2).
- dot test: <J v, w> (jax.jvp, tangent-linear) == <v, J^T w> (jax.vjp, adjoint) for the 2-step map of the five fields.
The driver logic on a toy step (every schedule, chunk length, stride) and the trust utilities: test_checkpoint_drivers.py.
Measured numbers (2026-09-23, 16 cores): in each test's docstring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.adjoint import checkpoint as ck
from mitgcm_jax.adjoint import grad as gr
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import State, state_from_dump
from mitgcm_jax.tests import oracle

NSTEPS = 2
CONTROLS = ("theta", "salt", "uVel", "vVel", "etaN")


@pytest.fixture(scope="module")
def win():
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir)
    nml = RunNamelists(rundir)
    st = state_from_dump(ds, 1)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    st0 = State({k: jnp.asarray(v) for k, v in st.f.items()}, jnp.asarray(1))
    xs = ck.exf_window(ck.exf_loader_at(P, g, rundir, nml, 1), nml, 1, NSTEPS)
    L = g.layout
    wC = np.zeros(L.shape2d)
    inner = (slice(None), slice(L.OLy, L.OLy + L.sNy), slice(L.OLx, L.OLx + L.sNx))
    wC[inner] = (np.asarray(g.rA) * np.asarray(g.maskC)[:, 0])[inner]
    wC = jnp.asarray(wC / wC.sum())
    return dict(P=P, g=g, ex=ex, model=ck.Model(P, g, kLowC, ex), st0=st0, xs=xs, step=ck.make_step(), wC=wC)


def _costs(wC):
    def final_cost(m, s):
        return jnp.sum(wC * s.theta[:, 0] ** 2)

    def cost(m, s, x):
        return jnp.sum(wC * s.uVel[:, 0] ** 2) * 100.0
    return final_cost, cost


def test_scan_equals_loop(win):
    """The lax.scan integrator (schedules none, step, sqrt) and the Python loop of a jitted step give bitwise equal
    States after 2 steps (every field, and the iteration counter)."""
    step, model, st0, xs = win["step"], win["model"], win["st0"], win["xs"]
    step_jit = jax.jit(step)
    s_loop = ck.run_loop(step_jit, model, st0, [jax.tree.map(lambda a: a[i], xs) for i in range(NSTEPS)])
    for sch in ck.SCHEDULES:
        s_scan, _ = jax.jit(lambda m, s, x: ck.integrate(step, m, s, x, schedule=sch, segments=2))(model, st0, xs)
        assert int(s_scan.it) == int(s_loop.it) == 1 + NSTEPS
        bad = [k for k in s_loop.f if not np.array_equal(np.asarray(s_loop.f[k]), np.asarray(s_scan.f[k]))]
        assert not bad, (sch, bad)


def test_schedules_same_gradient(win):
    """J and dJ/d(theta, salt, uVel, vVel, etaN at the start) agree between per-step remat (reference), sqrt(N) segments
    (2 x 1) and chunked reverse accumulation (1-step chunks, boundary stride 1 and 2): J bitwise, gradients within
    1e-13 of each field's max (measured 2026-09-23: see the assertion message on failure). The plain scan ("none")
    is left out: 2 steps without remat exceed the 64 GB test allocation."""
    step, model, st0, xs = win["step"], win["model"], win["st0"], win["xs"]
    final_cost, cost = _costs(win["wC"])
    theta = {k: st0.f[k] for k in CONTROLS}
    res = {}
    for sch in ("step", "sqrt"):
        J, g_ = gr.value_and_grad(step, theta, model, st0, xs, final_cost=final_cost, cost=cost, schedule=sch,
                                  segments=2)
        res[sch] = (float(J), jax.tree.map(np.asarray, g_))
    xs_fn, nch = gr.chunks_of(xs, 1)
    for stride in (1, 2):
        r = gr.chunked_value_and_grad(step, theta, model, st0, n_chunks=nch, chunk_steps=1, xs_fn=xs_fn,
                                      final_cost=final_cost, cost=cost, boundary_stride=stride)
        res[f"chunked{stride}"] = (r.loss, jax.tree.map(np.asarray, r.grad))
        assert len(r.trace) == nch + 1 and all(np.isfinite(r.trace))
    J0, g0 = res["step"]
    for k in CONTROLS:
        assert np.all(np.isfinite(g0[k])), k
        if k != "etaN":   # etaN only seeds the cg2d first guess (stop_gradient): dJ/detaN = 0 exactly
            assert np.count_nonzero(g0[k]) > 0, k
    for name, (J, g_) in res.items():
        assert J == J0, name
        for k in CONTROLS:
            err = np.max(np.abs(g_[k] - g0[k])) / max(np.max(np.abs(g0[k])), 1e-300)
            assert err <= 1e-13, (name, k, err)


def test_dot_test_two_steps(win):
    """Tangent-linear (jax.jvp) vs adjoint (jax.vjp) of the 2-step map (theta, salt, uVel, vVel, etaN) -> the same
    fields, random masked v, w: |<J v, w> - <v, J^T w>| / |<J v, w>| <= 1e-10 at tangent amplitude 1 and 1e-6.
    Measured 2026-09-23 (CPU, conftest flags): 1.3e-11 (amp 1), 6.2e-12 (amp 1e-6); the same with the cg2d adjoint
    tolerance 1e-16 instead of 1e-13 (1.33e-11), so this is the round-off floor of the 2-step TL/adjoint on O(1) random
    fields, not the solver. Discriminates: the pre-Task-18 cg2d tangent (literal solve on the tangent from the primal
    first guess at the Fortran tolerance) gave 1.9e-9 (amp 1) and 4.4e-9 (amp 1e-6)."""
    step, model, st0, xs, g = win["step"], win["model"], win["st0"], win["xs"], win["g"]

    def f(x):
        s, _ = ck.integrate(step, model, st0.replace(**x), xs, schedule="step")
        return {k: s.f[k] for k in CONTROLS}
    rng = np.random.default_rng(3)
    mask = {"theta": g.maskC, "salt": g.maskC, "uVel": g.maskW, "vVel": g.maskS, "etaN": g.maskC[:, 0]}
    x = {k: st0.f[k] for k in CONTROLS}
    v = {k: jnp.asarray(rng.standard_normal(x[k].shape)) * mask[k] for k in CONTROLS}
    w = {k: jnp.asarray(rng.standard_normal(x[k].shape)) * mask[k] for k in CONTROLS}
    for amp, lhs, rhs, rel in gr.dot_test(f, x, v, w, amps=(1.0, 1e-6)):
        assert rel <= 1e-10, (amp, lhs, rhs, rel)
