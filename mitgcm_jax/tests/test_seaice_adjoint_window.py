"""Multi-step adjoint tests of the sea-ice model alone (plan M2, before ocean coupling): the fast checks.

Harness: scripts/adjoint/seaice_window.py (its docstring defines it): SEAICE_MODEL stepped on its own carried state
(ICE_STATE + DYN_CARRY + sIceLoad), every other input prescribed from the full-V4r4 oracle (oracle.FULL, iterations
1-3), controls on the initial HEFF/AREA/UICE/VICE and time-constant shifts of atemp, fu, fv; costs J1 (Arctic ice
volume north of 70N), J2 (Southern Ocean ice volume south of 60S), J3 (Arctic ice area). The full experiment (windows
of 6, 24, 48 steps, FD h-sweeps, repeats, amplification screens) is run by that script; docs/ADJOINT_RESULTS.md.

Checks (measured values in each docstring; conftest gate flags, CPU):
  - harness gate: with the "cycle" schedule the harness carry after steps 1, 2, 3 is BITWISE the oracle's P00 at
    iterations 1-3 (ICE_STATE, sIceLoad; every point incl. halos): the harness steps the literal model.
  - ecco level over a 6-step window: forward bitwise equal to the "full" forward (every carried field), and the
    reverse of the WHOLE window is exactly the identity on the carried state and zero on the input shifts
    (dJ1/dHEFF0 == W1, dJ3/dAREA0 == W3 bitwise, all else 0).
  - the loop drivers (per-step jitted VJP/JVP) == jax.grad / jax.jvp of the whole window as lax.scan with
    jax.checkpoint per step (no_dynamics, 6 steps): bitwise.
  - TL vs adjoint dot tests: no_dynamics over 6 steps with random v on every control and random w on all 18 carried
    fields; full (implicit LSR derivative) over N_FULL = 3 steps with random v and w = the J1 seed.
  - one FD plateau: J1 along the relative HEFF direction, N_FULL steps, production LSR tolerance, vs the full
    adjoint (negative control: the no_dynamics adjoint fails the bound).
Cost: dominated by compilation and the full level (GMRES 40 x 8 per LSR pass, ~20-40 s per step and seed on CPU):
11-12 min on one compute node (job 27651707; job 27654164 after the Pallas-LSR merge 45875d8: identical values;
the 6-step version of the full checks: 17 min, job 27651045).
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("seaice_window", REPO / "scripts" / "adjoint" / "seaice_window.py")
sw = importlib.util.module_from_spec(_spec)
sys.modules["seaice_window"] = sw
_spec.loader.exec_module(sw)

N = 6
N_FULL = 3  # full-level checks (GMRES 40 x 8 per LSR pass, ~20-40 s per step and seed): 3 steps keep it short


@pytest.fixture(scope="module")
def env():
    return sw.load_env(log=print)


@pytest.fixture(scope="module")
def base(env):
    """The 6-step base window (production LSR), carries kept for the reverse loops."""
    return sw.forward(env.M, env, sw.zero_controls(), N, keep=True)


@pytest.fixture(scope="module")
def base_full(env):
    """The N_FULL-step base window for the full-level checks."""
    return sw.forward(env.M, env, sw.zero_controls(), N_FULL, keep=True)


def test_harness_gate_cycle_bitwise(env):
    """cycle schedule: harness steps 1-3 == the Fortran's SEAICE_MODEL calls at iterations 1-3 (P00, every point).
    Measured: 0 differing values in AREA, HEFF, HSNOW, TICES, UICE, VICE, sIceLoad at each iteration.
    Not vacuous: the fixed schedule (iteration-1 inputs at step 2) differs from P00 at iteration 2."""
    ctl = sw.zero_controls()
    carry, shift = sw.BIND(env.M, env.carry0, ctl)
    for it in sw.ITS:
        carry = sw.STEP(env.M, carry, sw.take(env.X3, it - 1), shift, ad="ecco")
        for k in sw.sm.ICE_STATE + ("sIceLoad",):
            np.testing.assert_array_equal(np.asarray(carry[k]), env.P00[it][k], err_msg=f"it {it} {k}")
    f = sw.forward(env.M, env, ctl, 2, "fixed")
    assert np.sum(np.asarray(f.carry["HEFF"]) != env.P00[2]["HEFF"]) > 1000


def test_ecco_window_identity(env, base):
    """ad="ecco", 6 steps: the forward is bitwise the "full" forward, and the reverse of the whole window is exactly
    the identity on the carried state and zero on the atemp/fu/fv shifts (TAF skips SEAICE_MODEL in reverse:
    useSEAICEinAdMode = F). Measured: exact (0 mismatches)."""
    ctl = sw.zero_controls()
    fe = sw.forward(env.M, env, ctl, N, ad="ecco")
    for k in sw.CARRY:
        np.testing.assert_array_equal(np.asarray(fe.carry[k]), np.asarray(base.carry[k]), err_msg=k)
    seeds = sw.seeds_for_costs(env)
    a = sw.adjoint(env.M, env, ctl, N, seeds, ad="ecco", fwd=base)
    for k in sw.CARRY:
        np.testing.assert_array_equal(np.asarray(a.ct0[k]), np.asarray(seeds[k]), err_msg=k)
    exp = {k: np.zeros((3,) + sw.L.shape2d) for k in sw.CONTROLS}
    exp["HEFF"][0], exp["HEFF"][1], exp["AREA"][2] = (np.asarray(env.W[j]) for j in sw.COSTS)
    for k in sw.CONTROLS:
        np.testing.assert_array_equal(a.grad[k], exp[k], err_msg=k)
    # not vacuous: the model changes HEFF over the window, so the identity is not the true derivative
    assert np.abs(np.asarray(base.carry["HEFF"]) - np.asarray(env.carry0["HEFF"])).max() > 1e-3


def test_loop_drivers_equal_scan(env, base):
    """The per-step loop drivers (jitted step-VJP / step-JVP, carries on the host) == one jax.grad / jax.jvp of the
    whole window as lax.scan with jax.checkpoint per step (window_scan), no_dynamics level, 6 steps: J and the three
    gradients (every control). Measured: bitwise equal (and the TL along HEFF, bitwise, checked once: dev bench
    scan_check.py; not repeated here to save its compilation)."""
    ctl = sw.zero_controls()
    idx = sw.schedule_index("fixed", N)
    f = jax.jit(lambda M, W, c0, X3, c: sw.window_scan(M, W, c0, X3, idx, c, "no_dynamics"))
    np.testing.assert_array_equal(np.asarray(f(env.M, env.W, env.carry0, env.X3, ctl)), base.J)
    a = sw.adjoint(env.M, env, ctl, N, sw.seeds_for_costs(env), ad="no_dynamics", fwd=base)
    jac = jax.jit(lambda M, W, c0, X3, c, e: jax.grad(
        lambda z: jnp.vdot(sw.window_scan(M, W, c0, X3, idx, z, "no_dynamics"), e))(c))
    for s in range(3):
        g = jac(env.M, env.W, env.carry0, env.X3, ctl, jnp.zeros(3).at[s].set(1.0))
        for k in sw.CONTROLS:
            np.testing.assert_array_equal(np.asarray(g[k]), a.grad[k][s], err_msg=f"J{s + 1} {k}")


def test_dot_6steps_no_dynamics(env, base):
    """no_dynamics level, TL (jitted step-JVP loop) vs adjoint (step-VJP loop) over 6 steps: <J'v, w> vs
    <v, J'^T w>, random v on all 7 controls, random w on all 18 carried fields at the window end, amplitudes 1 and
    1e-6. Measured (job 27650296, CPU): 3.8e-15, 5.9e-15."""
    ctl = sw.zero_controls()
    v = sw.random_controls(env, 0)
    w = sw.random_carry(env, 1)
    a = sw.adjoint(env.M, env, ctl, N, jax.tree.map(lambda x: x[None], w), ad="no_dynamics", fwd=base)
    rhs = sw.tree_vdot(v, {k: a.grad[k][0] for k in sw.CONTROLS})
    for amp in (1.0, 1e-6):
        _, _, dcarry = sw.tl(env.M, env, ctl, jax.tree.map(lambda z: z * amp, v), N, ad="no_dynamics")
        lhs = sw.tree_vdot(dcarry, w)
        rel = abs(lhs - amp * rhs) / max(abs(lhs), abs(amp * rhs))
        print(f"no_dynamics amp {amp}: lhs {lhs:.16e} rhs {amp * rhs:.16e} rel {rel:.2e}")
        assert rel < 1e-12, (amp, lhs, amp * rhs, rel)


@pytest.fixture(scope="module")
def full_grad_J1(env, base_full):
    """ad="full" (implicit LSR derivative, GMRES 40 x 8) gradient of J1 over the N_FULL-step base window."""
    return sw.adjoint(env.M, env, sw.zero_controls(), N_FULL, sw.seeds_for_costs(env, ("J1",)), ad="full",
                      fwd=base_full)


def test_full_dot(env, full_grad_J1):
    """full level, N_FULL steps: TL of J1 along a random v on all 7 controls (GMRES on A in every LSR pass) vs
    <v, dJ1/dctl> (GMRES on A^T), and the gradient finite everywhere. Measured: 2.5e-14 (job 27651707; 6 steps:
    7.2e-14, job 27651045). The experiment script's dot test with a random w on all 18 carried fields: 1.4e-14 /
    6.1e-15 at amplitudes 1 / 1e-6 over 6 steps (job 27650296), 7.6e-14 over 24 steps (job 27650423), 2.9e-14 over
    48 steps (job 27650424)."""
    ctl = sw.zero_controls()
    v = sw.random_controls(env, 0)
    for k in sw.CONTROLS:
        assert np.all(np.isfinite(full_grad_J1.grad[k])), k
    rhs = sw.tree_vdot(v, {k: full_grad_J1.grad[k][0] for k in sw.CONTROLS})
    _, dJ, _ = sw.tl(env.M, env, ctl, v, N_FULL, ad="full")
    rel = abs(dJ[0] - rhs) / max(abs(dJ[0]), abs(rhs))
    print(f"full: TL {dJ[0]:.16e} adjoint {rhs:.16e} rel {rel:.2e}")
    assert rel < 1e-11, (dJ[0], rhs, rel)


def test_full_fd_plateau_heff(env, base_full, full_grad_J1):
    """FD plateau: J1 (Arctic ice volume after N_FULL steps) along the relative HEFF direction (x + h d = (1 + h)
    HEFF0), production LSR tolerance, central differences at h = 1e-4 and 1e-6 vs the full adjoint. Measured
    (job 27651707): 2.9e-6, 1.7e-7; the no_dynamics adjoint differs from the h = 1e-6 FD by 1.5e-6. (6-step sweep
    h = 1e-2 .. 1e-8, job 27650297: 2.0e-5, 1.5e-6, 2.6e-6, 1.4e-7, 1.4e-7, 1.4e-7, 1.3e-7, a plateau at every h;
    switch flips between +h and -h: 174 cell-steps at h = 1e-2, 3-6 below 1e-4.) Negative control: the no_dynamics
    adjoint (the LSR skipped in reverse) fails the same bound (bound 5e-7: 3x margin on both sides)."""
    d = sw.direction(env, "HEFF")

    def dirder(a):
        return sum(float(np.vdot(a.grad[k][0], np.asarray(v))) for k, v in env.dirs["HEFF"].items())

    ad = dirder(full_grad_J1)
    fd = {}
    for h in (1e-4, 1e-6):
        fp = sw.forward(env.M, env, jax.tree.map(lambda a: h * a, d), N_FULL)
        fm = sw.forward(env.M, env, jax.tree.map(lambda a: -h * a, d), N_FULL)
        fd[h] = (fp.J[0] - fm.J[0]) / (2 * h)
    err = {h: abs(v - ad) / abs(ad) for h, v in fd.items()}
    print(f"FD {fd} AD {ad:.12e} rel {err}")
    assert err[1e-4] < 1e-5 and err[1e-6] < 5e-7, err
    nd = sw.adjoint(env.M, env, sw.zero_controls(), N_FULL, sw.seeds_for_costs(env, ("J1",)), ad="no_dynamics",
                    fwd=base_full)
    err_nd = abs(fd[1e-6] - dirder(nd)) / abs(dirder(nd))
    print(f"negative control, no_dynamics: rel {err_nd:.2e}")
    assert err_nd > 5e-7, err_nd
