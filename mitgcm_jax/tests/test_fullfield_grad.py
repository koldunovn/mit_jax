"""Standing full-field gradient gate (plan Task 8, rerun in every later task): the gradient of a scalar of the state
after one FORWARD_STEP w.r.t. the WHOLE initial theta, salt, uVel, vVel, etaN fields (halos included) is finite
everywhere, exactly zero on dry tracer / surface points (maskC / maskInC = 0), and non-zero on most wet surface points.
Dry VELOCITY points are not required to be zero: the vorticity and no-slip side-drag stencils of MOM_VECINV read the
(always zero) velocities at dry W/S points, so the Fortran operator itself is sensitive to them (measured: 29,029 dry
uVel points with non-zero sensitivity) — a property of the discretisation, not a masked-lane leak.
Halo lanes are NOT required to be zero: MITgcm's step reads the (exchanged) halo copies directly — a halo value is an
independent input of the step, so its sensitivity is legitimately non-zero.
Negative control: a planted 0*inf (sqrt at 0 on dry lanes) would put NaN into the dry-lane gradient; checked on a toy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core.forward_step import forward_step
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import State, state_from_dump
from mitgcm_jax.tests import oracle

KEYS = ("theta", "salt", "uVel", "vVel", "etaN")


@pytest.fixture(scope="module")
def grads():
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir)
    st = state_from_dump(ds, 1)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    st = State({k: jnp.asarray(v) for k, v in st.f.items()}, jnp.asarray(1))
    nml = RunNamelists(rundir)
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    myTime, myIter = exf_mod.model_time(nml, 1)
    bufs, facs, _ = loader.load(myTime, myIter)
    exf_in = {"bufs": bufs, "facs": facs, "myTime": myTime}
    L = g.layout
    I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
    w = np.zeros(L.shape2d)
    w[:, J, I] = np.asarray(g.rA)[:, J, I] * np.asarray(g.maskC)[:, 0, J, I]
    w = jnp.asarray(w / w.sum())

    def cost(x, P, g, ex, kLowC, st, exf_in):
        s1 = forward_step(P, g, ex, kLowC, st.replace(**x), exf_in)[0]
        return (jnp.sum(w * s1.theta[:, 0]) + 1e-2 * jnp.sum(w * s1.salt[:, 0])
                + 1e-1 * jnp.sum(w * s1.etaN) + jnp.sum(w * s1.uVel[:, 0] ** 2))

    x0 = {k: st.f[k] for k in KEYS}
    gr = jax.jit(jax.grad(cost))(x0, P, g, ex, kLowC, st, exf_in)
    return {k: np.asarray(v) for k, v in gr.items()}, g


def test_finite_everywhere(grads):
    gr, _ = grads
    for k, v in gr.items():
        assert np.all(np.isfinite(v)), k


def test_zero_on_dry_points(grads):
    gr, g = grads
    masks = {"theta": g.maskC, "salt": g.maskC, "etaN": g.maskInC}
    for k, m in masks.items():
        dry = np.asarray(m) == 0
        assert np.count_nonzero(gr[k][dry]) == 0, (k, np.count_nonzero(gr[k][dry]))


def test_nonzero_on_wet_surface(grads):
    gr, g = grads
    L = g.layout
    I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
    wet = np.asarray(g.maskC)[:, 0, J, I] > 0
    frac = np.mean(gr["theta"][:, 0, J, I][wet] != 0)
    assert frac > 0.95, frac


def test_negative_control_masked_nan():
    """A forward `where` does not stop a backward 0*inf: sqrt taken at 0 on a masked lane poisons the gradient."""
    x = jnp.array([0.0, 4.0])
    m = jnp.array([0.0, 1.0])
    bad = jax.grad(lambda x: jnp.sum(jnp.where(m > 0, jnp.sqrt(x), 0.0)))(x)
    good = jax.grad(lambda x: jnp.sum(jnp.where(m > 0, jnp.sqrt(jnp.where(m > 0, x, 1.0)), 0.0)))(x)
    assert not np.all(np.isfinite(np.asarray(bad))) and np.all(np.isfinite(np.asarray(good)))
