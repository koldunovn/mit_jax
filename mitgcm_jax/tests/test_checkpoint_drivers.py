"""Gradient drivers and trust utilities of plan Task 18 (mitgcm_jax/adjoint/{checkpoint,grad}.py) on a toy step
(seconds; tier 1). The LLC90 versions of these checks are in test_checkpoint.py (tier1x).

Toy model: a State with two fields and the iteration counter; the step is nonlinear, reads a per-step input and
adds a field the initial State lacks (as FORWARD_STEP adds PmEpR), so `prepare_state` is exercised.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.adjoint import checkpoint as ck
from mitgcm_jax.adjoint import grad as gr
from mitgcm_jax.state import State

N = 12


def _step(model, st, x):
    f = dict(st.f)
    a, b = f["a"], f["b"]
    f["a"] = jnp.sin(a) * model["k"] + x["f"] * b
    f["b"] = b + 0.1 * a ** 2
    f["c"] = a * 2.0
    return State(f, st.it + 1)


def _setup():
    model = {"k": jnp.asarray(0.9)}
    st0 = State({"a": jnp.linspace(0.0, 1.0, 5), "b": jnp.ones(5)}, jnp.asarray(1))
    xs = {"f": np.linspace(0.1, 0.5, N)}
    return model, st0, xs


def _final_cost(m, s):
    return jnp.sum(s.a ** 2) + jnp.sum(s.c)


def _cost(m, s, x):
    return jnp.sum(s.b) * 0.01


def test_toy_scan_equals_loop():
    """integrate (every schedule) == the Python loop of the jitted step, bitwise; the added field is carried."""
    model, st0, xs = _setup()
    s_loop = ck.run_loop(jax.jit(_step), model, st0, [jax.tree.map(lambda a: a[i], xs) for i in range(N)])
    for sch in ck.SCHEDULES:
        s, _ = jax.jit(lambda m, s, x: ck.integrate(_step, m, s, x, schedule=sch))(model, st0, xs)
        assert int(s.it) == 1 + N
        for k in ("a", "b", "c"):
            np.testing.assert_array_equal(np.asarray(s.f[k]), np.asarray(s_loop.f[k]), err_msg=f"{sch} {k}")


def test_toy_all_drivers_same_gradient():
    """value_and_grad with schedules none/step/sqrt and chunked_value_and_grad over chunk lengths 1, 3, 4, 12, boundary
    strides 1-3 and in-chunk schedules step/sqrt/none: J and the initial-state gradient bitwise equal."""
    model, st0, xs = _setup()
    th = {"a": st0.f["a"], "b": st0.f["b"]}
    J0, g0 = gr.value_and_grad(_step, th, model, st0, xs, final_cost=_final_cost, cost=_cost, schedule="none")
    runs = {}
    for sch in ("step", "sqrt"):
        runs[sch] = gr.value_and_grad(_step, th, model, st0, xs, final_cost=_final_cost, cost=_cost, schedule=sch)
    for K, stride in ((1, 1), (3, 1), (3, 2), (4, 3), (12, 1)):
        xs_fn, nch = gr.chunks_of(xs, K)
        for inner in ("step", "sqrt", "none"):
            r = gr.chunked_value_and_grad(_step, th, model, st0, n_chunks=nch, chunk_steps=K, xs_fn=xs_fn,
                                          final_cost=_final_cost, cost=_cost, boundary_stride=stride, schedule=inner)
            assert len(r.trace) == nch + 1
            runs[f"chunk{K} stride{stride} {inner}"] = (r.loss, r.grad)
    for name, (J, g) in runs.items():
        assert float(J) == float(J0), name
        for k in th:
            np.testing.assert_array_equal(np.asarray(g[k]), np.asarray(g0[k]), err_msg=name)


def test_toy_parameter_gradient_vs_fd():
    """A control in the Model (params_fn): chunked and one-shot gradients agree bitwise and match a central difference
    (measured 4e-11 relative at h = 1e-6)."""
    model, st0, xs = _setup()
    th = {"k": jnp.asarray(0.9)}
    pf = lambda th, m: {"k": th["k"]}  # noqa: E731
    keep = lambda th, s: s  # noqa: E731
    kw = dict(final_cost=_final_cost, cost=_cost, init_fn=keep, params_fn=pf)
    J, g = gr.value_and_grad(_step, th, model, st0, xs, **kw)
    xs_fn, nch = gr.chunks_of(xs, 3)
    r = gr.chunked_value_and_grad(_step, th, model, st0, n_chunks=nch, chunk_steps=3, xs_fn=xs_fn,
                                  boundary_stride=2, **kw)
    assert float(r.grad["k"]) == float(g["k"])
    J_ = lambda t: gr.objective(_step, t, model, st0, xs, **kw)  # noqa: E731
    rows = gr.fd_sweep(lambda t: J_(t), th, {"k": jnp.asarray(1.0)}, [1e-4, 1e-6], float(g["k"]))
    assert min(r_.rel_err for r_ in rows) < 1e-8, rows


def test_toy_dot_test():
    """TL (jax.jvp) vs adjoint (jax.vjp) of the 12-step map through integrate: equal to round-off."""
    model, st0, xs = _setup()
    th = {"a": st0.f["a"], "b": st0.f["b"]}

    def f(x):
        s, _ = ck.integrate(_step, model, st0.replace(**x), xs, schedule="step")
        return {"a": s.a, "b": s.b}
    v = {"a": jnp.linspace(-1.0, 1.0, 5), "b": jnp.cos(jnp.arange(5.0))}
    w = {"a": jnp.sin(jnp.arange(5.0)), "b": jnp.linspace(2.0, 0.5, 5)}
    assert gr.dot_test(f, th, v, w)[2] < 1e-14


def test_dot_test_utility_detects_wrong_transpose():
    """dot_test passes a linear map with its true transpose and fails one with a wrong transpose (a
    custom_linear_solve whose JVP applies A and whose VJP applies the `transpose_solve` given: A^T, or A itself)."""
    A = jnp.asarray(np.random.default_rng(0).standard_normal((4, 4)))
    x, v, w = jnp.ones(4), jnp.arange(4.0), jnp.arange(4.0) - 2.0

    def op(tr):
        return lambda z: jax.lax.custom_linear_solve(lambda y: y, z, lambda _, b: A @ b, lambda _, b: tr @ b)
    assert gr.dot_test(lambda z: A @ z, x, v, w)[2] < 1e-14
    assert gr.dot_test(op(A.T), x, v, w)[2] < 1e-14
    assert gr.dot_test(op(A), x, v, w)[2] > 1e-3


def test_fd_sweep_and_amplification():
    """fd_sweep finds the plateau of a smooth function (cubic: truncation error ~h^2, round-off ~eps/h); amplification
    gives the per-step rates of a synthetic trace and applies the three bars."""
    def J(x):
        return jnp.sum(x ** 3)
    x = jnp.asarray([0.5, -1.0, 2.0])
    d = jnp.asarray([1.0, 2.0, -0.5])
    ad = float(jnp.vdot(jax.grad(J)(x), d))
    rows = gr.fd_sweep(jax.jit(J), x, d, [1e-1, 1e-3, 1e-5, 1e-7], ad)
    assert rows[0].rel_err > 1e-3            # truncation
    assert min(r.rel_err for r in rows) < 1e-8
    assert gr.plateau(rows, 1e-6)
    tr = [1.0 * 1.005 ** (4 * i) for i in range(6)]
    a = gr.amplification(tr, 4)
    assert abs(a.median - 1.005) < 1e-12 and a.log_spread < 1e-12 and a.passes
    tr2 = tr[:3] + [tr[3] * 1e3] + [t * 1e3 for t in tr[4:]]
    assert not gr.amplification(tr2, 4).passes


def test_chunked_field_trace():
    """ChunkedGrad.field_trace (Task 21): per-field cotangent norms at every chunk boundary, same order as `trace`, and
    their root-sum-square is the total norm; a field the step passes through unchanged accumulates (toy: none)."""
    model, st0, xs = _setup()
    th = {"a": st0.f["a"], "b": st0.f["b"]}
    xs_fn, nch = gr.chunks_of(xs, 3)
    r = gr.chunked_value_and_grad(_step, th, model, st0, n_chunks=nch, chunk_steps=3, xs_fn=xs_fn,
                                  final_cost=_final_cost, cost=_cost)
    assert len(r.field_trace) == len(r.trace) == nch + 1
    for tot, per in zip(r.trace, r.field_trace):
        assert set(per) == {"a", "b", "c"}
        np.testing.assert_allclose(np.sqrt(sum(v ** 2 for v in per.values())), tot, rtol=1e-12)
