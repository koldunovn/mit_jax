"""Time-mean accumulators (mitgcm_jax/diagnostics/means.py, plan Task 19) on a toy step (small arrays, seconds):
the mean over N steps equals the average of the N end-of-step snapshots, in a jitted Python loop and in a lax.scan
carry (integrate_with_diagnostics), with uniform and non-uniform weights. The LLC90 model step in the scan is checked
in test_budgets.py::test_scan_diagnostics_match_loop.
Negative control: accumulating the state BEFORE each step (off by one step) does not match the snapshot average."""

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.diagnostics import means as mn
from mitgcm_jax.state import State

N = 5


def _state(seed=0):
    r = np.random.default_rng(seed)
    f = {"theta": r.normal(10.0, 3.0, (2, 3, 6, 7)), "salt": r.normal(35.0, 0.5, (2, 3, 6, 7)),
         "etaN": r.normal(0.0, 0.3, (2, 6, 7)), "uVel": r.normal(0.0, 0.1, (2, 3, 6, 7)),
         "vVel": r.normal(0.0, 0.1, (2, 3, 6, 7))}
    return State({k: jnp.asarray(v) for k, v in f.items()}, jnp.asarray(1))


def _step(model, st, x):
    """Toy nonlinear step: every field changes with the per-step input x."""
    a = model["a"]
    f = {k: v * a + x * jnp.sin(v) for k, v in st.f.items()}
    return State(f, st.it + 1)


def _snapshots(st, xs, model):
    out = []
    for x in xs:
        st = _step(model, st, x)
        out.append(st)
    return out


def test_loop_mean_equals_snapshot_average():
    model = {"a": jnp.asarray(0.97)}
    st0 = _state()
    xs = jnp.linspace(0.1, 0.5, N)
    snaps = _snapshots(st0, xs, model)
    acc_jit = jax.jit(mn.means_accumulate)
    acc = mn.means_init(st0)
    st = st0
    for x in xs:
        st = _step(model, st, x)
        acc = acc_jit(acc, st, 1.0)
    m = mn.means_finish(acc)
    assert set(m) == set(mn.MEAN_FIELDS)
    for k in mn.MEAN_FIELDS:
        ref = np.mean([np.asarray(mn._get(s, k)) for s in snaps], axis=0)
        np.testing.assert_allclose(np.asarray(m[k]), ref, rtol=1e-14, atol=1e-14, err_msg=k)
    np.testing.assert_array_equal(np.asarray(m["sst"]), np.asarray(m["theta"])[:, 0])
    # negative control: the off-by-one mean (states at the START of each step) differs
    ref_off = np.mean([np.asarray(s.theta) for s in [st0] + snaps[:-1]], axis=0)
    assert np.max(np.abs(np.asarray(m["theta"]) - ref_off)) > 1e-3


def test_scan_mean_weights_and_outputs():
    """integrate_with_diagnostics: the scan carry mean equals the loop's snapshot average (uniform weights) and the
    weighted average (non-uniform weights); per-step budget/monitor outputs are the loop's values."""
    model = {"a": jnp.asarray(0.97)}
    st0 = _state(1)
    xs = jnp.linspace(0.1, 0.5, N)
    snaps = _snapshots(st0, xs, model)
    budget = lambda s0, s1: {"d": jnp.sum(s1.theta - s0.theta)}  # noqa: E731
    monitor = lambda s1: {"max": jnp.max(s1.salt)}  # noqa: E731

    @jax.jit
    def run(model, st, xs, w):
        return mn.integrate_with_diagnostics(_step, model, st, xs, weights=w, budget=budget, monitor=monitor)

    w = jnp.ones(N)
    st_n, acc, ys = run(model, st0, xs, w)
    np.testing.assert_allclose(np.asarray(st_n.theta), np.asarray(snaps[-1].theta), rtol=1e-15)
    m = mn.means_finish(acc)
    for k in mn.MEAN_FIELDS:
        ref = np.mean([np.asarray(mn._get(s, k)) for s in snaps], axis=0)
        np.testing.assert_allclose(np.asarray(m[k]), ref, rtol=1e-14, atol=1e-14, err_msg=k)
    prev = [st0] + snaps[:-1]
    np.testing.assert_allclose(np.asarray(ys["budget"]["d"]),
                               [float(jnp.sum(b.theta - a.theta)) for a, b in zip(prev, snaps)], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ys["monitor"]["max"]), [float(jnp.max(s.salt)) for s in snaps],
                               rtol=0)
    wn = jnp.asarray([0.5, 1.0, 2.0, 1.0, 0.25])
    _, acc, _ = run(model, st0, xs, wn)
    m = mn.means_finish(acc)
    ref = np.average(np.stack([np.asarray(s.etaN) for s in snaps]), axis=0, weights=np.asarray(wn))
    np.testing.assert_allclose(np.asarray(m["etaN"]), ref, rtol=1e-14, atol=1e-14)
    assert float(acc["w"]) == float(jnp.sum(wn))
