"""Gradient drivers over a window of FORWARD_STEPs and the trust utilities (plan Task 18).

    theta -> J:   st0 = init_fn(theta, st0_base); model = params_fn(theta, model_base);
                  (st_n, acc) = integrate(step, model, st0, xs); J = final_cost(model, st_n) + acc

`theta` is any pytree of controls. `init_fn(theta, st)` puts the initial-state controls into the State (default: a
dict of State fields replaced by theta's entries); `params_fn(theta, model)` puts parameter/field controls into the
Model (default: the identity). Both run inside the differentiated function, so any State or Model leaf can be a
control. `cost(model, st, x)` is accumulated after every step (checkpoint.integrate), `final_cost(model, st_n)` is
applied to the last state; either may be None.

Drivers (same J, same gradient; test_checkpoint.py):
  value_and_grad(...)           one jitted jax.value_and_grad of the whole window, schedule "none"|"step"|"sqrt"
  chunked_value_and_grad(...)   the window in chunks of `chunk_steps` steps: forward with every `boundary_stride`-th
                                chunk-boundary State parked on the HOST, then a reverse walk over the chunks, one
                                jitted jax.vjp per chunk (in-chunk schedule "step"/"sqrt"), carrying the State
                                cotangent; spans between kept boundaries are recomputed (one extra forward at most).
                                Device memory: one chunk, whatever the window length (fesom_jax adjoint.py).
                                Returns ChunkedGrad with the per-chunk cotangent-norm trace.

Trust utilities (lessons_adjoint.md section 5; a gradient is not trusted until these pass):
  dot_test(f, x, v, w)          <J v, w> vs <v, J^T w> (jax.jvp vs jax.vjp of the same f)
  fd_sweep(J, x, d, hs, ...)    central differences over step sizes h against the AD directional derivative, with the
                                forward noise floor measured from repeated evaluations
  cotangent_norm(tree), amplification(trace, steps_per_chunk)   per-step reverse growth: median, log-spread, worst
                                3 consecutive chunks (the fesom_jax screen: median <= 1.010/step, log-spread <= 0.020,
                                worst-3 <= 1.030/step)
"""

import time
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.adjoint.checkpoint import SAVE_NAMES, integrate, n_steps, prepare_state, take_steps

# ---------------------------------------------------------------------------------------------------------------
# control binding


def replace_fields(theta, st):
    """Default init_fn: theta is a dict {State field: array}; the State with those fields replaced."""
    return st.replace(**theta) if theta else st


def keep_model(theta, model):
    """Default params_fn: the Model does not depend on theta."""
    return model


def _objective(step, *, schedule, segments, cost, final_cost, init_fn, params_fn, save_names=SAVE_NAMES):
    def J(theta, model, st0, xs):
        m = params_fn(theta, model)
        s0 = init_fn(theta, st0)
        s_n, acc = integrate(step, m, s0, xs, schedule=schedule, segments=segments, cost=cost, save_names=save_names)
        out = acc
        if final_cost is not None:
            out = out + final_cost(m, s_n)
        return out
    return J


def value_and_grad(step, theta, model, st0, xs, *, final_cost=None, cost=None, schedule="step", segments=None,
                   init_fn=replace_fields, params_fn=keep_model, save_names=SAVE_NAMES, jit=True):
    """(J, dJ/dtheta) over the whole window in one jax.value_and_grad (xs: stacked step inputs, on host or device).
    Memory grows with the window (schedule "step": one State per step; "sqrt": ~2 sqrt(N)); for long windows use
    chunked_value_and_grad."""
    J = _objective(step, schedule=schedule, segments=segments, cost=cost, final_cost=final_cost, init_fn=init_fn,
                   params_fn=params_fn, save_names=save_names)
    vg = jax.value_and_grad(J)
    if jit:
        vg = jax.jit(vg)
    return vg(theta, model, st0, xs)


def objective(step, theta, model, st0, xs, *, final_cost=None, cost=None, init_fn=replace_fields,
              params_fn=keep_model, jit=True):
    """J(theta) alone (forward only, schedule "none"): for finite differences."""
    J = _objective(step, schedule="none", segments=None, cost=cost, final_cost=final_cost, init_fn=init_fn,
                   params_fn=params_fn)
    if jit:
        J = jax.jit(J)
    return J(theta, model, st0, xs)


# ---------------------------------------------------------------------------------------------------------------
# chunked reverse accumulation


class ChunkedGrad(NamedTuple):
    loss: float
    grad: Any
    trace: list            # norm of the State cotangent at every chunk boundary, end of window first (the growth
                           # screen `amplification` reads it; with a running cost the end-of-window seed is 0)
    host_gb: float         # chunk-boundary States parked on the host
    forward_seconds: float
    reverse_seconds: float
    n_chunks: int
    chunk_steps: int
    boundary_stride: int
    schedule: str
    field_trace: list = None   # per chunk boundary (same order as `trace`): {State field: cotangent norm}, so a screen
                               # can say WHICH field grows (the total norm mixes units: K, psu, m/s, m, hFac, ...)


def field_norms(ct):
    """{field: Euclidean norm} of a State cotangent (floating fields; {} for a cotangent without a field dict)."""
    f = getattr(ct, "f", None)
    if not isinstance(f, dict):
        return {}
    return {k: float(jnp.sqrt(jnp.sum(jnp.square(v)))) for k, v in sorted(f.items())
            if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating)}


def cotangent_norm(tree):
    """Euclidean norm over every floating leaf of a cotangent pytree (float0 / integer leaves skipped)."""
    leaves = [x for x in jax.tree.leaves(tree)
              if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)]
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def _float0_like(x):
    return np.zeros(np.shape(x), jax.dtypes.float0)


def _state_ct(ct_st):
    """Cotangent of a State to feed back into a VJP: integer leaves (the iteration counter) get float0 zeros."""
    return jax.tree.map(lambda x: x if jnp.issubdtype(jnp.result_type(x), jnp.floating) else _float0_like(x), ct_st)


def _add(a, b):
    return b if a is None else jax.tree.map(jnp.add, a, b)


# Jitted chunk forward/VJP per configuration, kept across calls (a repeat or an FD sweep would otherwise recompile
# the chunk, minutes at LLC90). SMALL on purpose: each entry holds GPU executables (fesom_jax: 8 resident
# executables OOM'd a sweep); one configuration per process is the intended use.
_CHUNK_CACHE = {}
_CHUNK_CACHE_MAX = 2


def _chunk_fns(step, schedule, segments, cost, params_fn, save_names):
    key = (step, schedule, segments, cost, params_fn, save_names)
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]

    def chunk(theta, model, carry, xs):
        return integrate(step, params_fn(theta, model), carry[0], xs, schedule=schedule, segments=segments,
                         cost=cost, acc0=carry[1], save_names=save_names)

    @jax.jit
    def bwd(theta, model, carry, xs, cot):
        _, pull = jax.vjp(lambda th, c: chunk(th, model, c, xs), theta, carry)
        return pull(cot)

    while len(_CHUNK_CACHE) >= _CHUNK_CACHE_MAX:
        _CHUNK_CACHE.pop(next(iter(_CHUNK_CACHE)))
    _CHUNK_CACHE[key] = (jax.jit(chunk), bwd)
    return _CHUNK_CACHE[key]


def chunked_value_and_grad(step, theta, model, st0, *, n_chunks, chunk_steps, xs_fn, final_cost=None, cost=None,
                           schedule="step", segments=None, init_fn=replace_fields, params_fn=keep_model,
                           boundary_stride=1, save_names=SAVE_NAMES, on_chunk=None, log=None):
    """(J, dJ/dtheta) with chunked reverse accumulation. xs_fn(c) -> the stacked inputs of chunk c (host numpy,
    `chunk_steps` steps, built fresh per call: it is called again in the reverse walk). Boundaries every
    `boundary_stride` chunks are kept on the host (always the window start and end); the spans between are rebuilt
    once each in the reverse walk. on_chunk(c, cot_norm) after each reverse chunk. Returns ChunkedGrad."""
    stride = max(1, int(boundary_stride))
    say = log or (lambda *a: None)
    fwd, bwd = _chunk_fns(step, schedule, segments, cost, params_fn, tuple(save_names))

    def seed(theta, model, carry):
        s, a = carry
        out = a
        if final_cost is not None:
            out = out + final_cost(params_fn(theta, model), s)
        return out

    # --- forward, boundaries to the host
    t0 = time.time()
    x0 = jax.tree.map(lambda a: a[0], xs_fn(0))
    st0 = prepare_state(step, params_fn(theta, model), st0, x0)   # every chunk carries the same structure
    s0 = jax.jit(init_fn)(theta, st0)
    carry = (s0, jnp.zeros((), jnp.float64))
    kept = {0: jax.device_get(carry)}
    for c in range(n_chunks):
        carry = jax.block_until_ready(fwd(theta, model, carry, jax.device_put(xs_fn(c))))
        if (c + 1) % stride == 0 or c == n_chunks - 1:
            kept[c + 1] = jax.device_get(carry)
    host_gb = sum(np.asarray(a).nbytes for a in jax.tree.leaves(kept)) / 1e9
    loss, pull = jax.vjp(lambda th, c: seed(th, model, c), theta, carry)
    g_th, cot = pull(jnp.ones((), jnp.float64))
    grad = g_th if final_cost is not None else None
    cot = (_state_ct(cot[0]), cot[1])
    fwd_t = time.time() - t0
    trace = [cotangent_norm(cot[0])]
    ftrace = [field_norms(cot[0])]
    say(f"chunked forward: {n_chunks} chunks x {chunk_steps} steps, J = {float(loss):.16e}, host {host_gb:.2f} GB, "
        f"{fwd_t:.1f} s")
    del carry

    # --- reverse walk
    t1 = time.time()
    for lo in reversed(range(0, n_chunks, stride)):
        hi = min(lo + stride, n_chunks)
        span = {lo: kept[lo]}
        c_in = jax.device_put(kept[lo])
        for c in range(lo, hi - 1):   # rebuild the span between kept boundaries (stride > 1)
            c_in = jax.block_until_ready(fwd(theta, model, c_in, jax.device_put(xs_fn(c))))
            span[c + 1] = jax.device_get(c_in)
        del c_in
        for c in reversed(range(lo, hi)):
            g_c, cot = bwd(theta, model, jax.device_put(span[c]), jax.device_put(xs_fn(c)), cot)
            cot = (_state_ct(cot[0]), cot[1])
            grad = _add(grad, g_c)
            trace.append(cotangent_norm(cot[0]))
            ftrace.append(field_norms(cot[0]))
            if on_chunk is not None:
                on_chunk(c, trace[-1])
        del span
    # initial state: theta -> st0
    _, pull0 = jax.vjp(lambda th: init_fn(th, st0), theta)
    grad = _add(grad, pull0(cot[0])[0])
    rev_t = time.time() - t1
    say(f"chunked reverse: {rev_t:.1f} s")
    return ChunkedGrad(float(loss), grad, trace, host_gb, fwd_t, rev_t, n_chunks, chunk_steps, stride, schedule,
                       ftrace)


def chunks_of(xs, chunk_steps):
    """xs_fn for a window whose stacked inputs are already on the host: chunk c = steps c*K .. (c+1)*K-1."""
    n = n_steps(xs)
    if n % chunk_steps:
        raise ValueError(f"{n} steps is not a multiple of chunk_steps={chunk_steps}")
    return (lambda c: take_steps(xs, c * chunk_steps, (c + 1) * chunk_steps)), n // chunk_steps


# ---------------------------------------------------------------------------------------------------------------
# trust utilities


def tree_vdot(a, b):
    """sum over floating leaves of <a, b> (float64)."""
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    tot = 0.0
    for x, y in zip(la, lb):
        if jnp.issubdtype(jnp.result_type(x), jnp.floating):
            tot += float(jnp.vdot(jnp.ravel(x), jnp.ravel(y)))
    return tot


def dot_test(f, x, v, w, amps=None, jit=True):
    """Adjoint dot test of y = f(x): <J v, w> (jax.jvp, the tangent-linear model) vs <v, J^T w> (jax.vjp, the
    adjoint). v: tangent like x; w: cotangent like f(x). Returns (lhs, rhs, rel); with `amps` (tangent amplitudes,
    e.g. (1.0, 1e-6): a TL that is not linear in the tangent fails at small amplitude) a list of (amp, lhs, rhs, rel)
    computed from ONE adjoint and one compiled tangent-linear function."""
    jvp = lambda x, v: jax.jvp(f, (x,), (v,))[1]  # noqa: E731
    vjp = lambda x, w: jax.vjp(f, x)[1](w)[0]     # noqa: E731
    if jit:
        jvp, vjp = jax.jit(jvp), jax.jit(vjp)
    rhs1 = tree_vdot(v, vjp(x, w))
    out = []
    for amp in ((1.0,) if amps is None else amps):
        lhs = tree_vdot(jvp(x, jax.tree.map(lambda z: z * amp, v)), w)
        rhs = amp * rhs1
        out.append((amp, lhs, rhs, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)))
    return out[0][1:] if amps is None else out


class FDRow(NamedTuple):
    h: float
    fd: float
    ad: float
    rel_err: float
    noise: float    # forward noise floor / h (the FD error the noise alone can cause)


def fd_sweep(J, x, d, hs, ad, repeats=2):
    """Central differences (J(x + h d) - J(x - h d)) / 2h for every h in hs against `ad` = <dJ/dx, d>. The forward
    noise floor is the spread of `repeats` evaluations of J(x) (0 on a deterministic CPU run; GPU atomics make it
    nonzero); a row is meaningful only where |fd - ad| is well above noise. Returns [FDRow]."""
    j0 = [float(J(x)) for _ in range(max(1, repeats))]
    spread = max(j0) - min(j0)
    rows = []
    for h in hs:
        xp = jax.tree.map(lambda a, b: a + h * b, x, d)
        xm = jax.tree.map(lambda a, b: a - h * b, x, d)
        fd = (float(J(xp)) - float(J(xm))) / (2.0 * h)
        rows.append(FDRow(float(h), fd, float(ad), abs(fd - ad) / max(abs(ad), 1e-300), spread / h))
    return rows


def plateau(rows, rtol):
    """The rows whose relative error is below rtol and above their noise floor (the FD plateau)."""
    return [r for r in rows if r.rel_err <= rtol]


class Amplification(NamedTuple):
    median: float        # per-step growth of the cotangent norm, median over chunks
    log_spread: float    # std of log per-step rates
    worst3: float        # worst per-step growth over 3 consecutive chunks
    rates: tuple         # per-chunk per-step growth, end of window first
    passes: bool         # median <= 1.010, log-spread <= 0.020, worst-3 <= 1.030 (fesom_jax bars)


def amplification(trace, steps_per_chunk, bars=(1.010, 0.020, 1.030)):
    """Per-step reverse growth from a chunk-boundary cotangent-norm trace (trace[0] = seed at the window end, each
    further entry one chunk earlier)."""
    tr = [float(t) for t in trace]
    k = max(1, int(steps_per_chunk))
    rates = [(tr[i + 1] / tr[i]) ** (1.0 / k) if tr[i] > 0 else float("nan") for i in range(len(tr) - 1)]
    fin = np.array([r for r in rates if np.isfinite(r) and r > 0])
    if fin.size == 0:
        return Amplification(float("nan"), float("nan"), float("nan"), tuple(rates), False)
    w = max(1, min(3, len(tr) - 1))
    sus = [(tr[i + w] / tr[i]) ** (1.0 / (k * w)) for i in range(len(tr) - w) if tr[i] > 0]
    worst = float(np.max(sus)) if sus else float("nan")
    med, spread = float(np.median(fin)), float(np.std(np.log(fin)))
    ok = med <= bars[0] and spread <= bars[1] and worst <= bars[2]
    return Amplification(med, spread, worst, tuple(rates), bool(ok))


def gpu_memory_gb():
    """Peak and current device memory of device 0 in GB (None where the backend has no memory stats)."""
    try:
        s = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001  (CPU backends may not implement it)
        return None
    if not s:
        return None
    return {"peak": s.get("peak_bytes_in_use", 0) / 1e9, "in_use": s.get("bytes_in_use", 0) / 1e9,
            "limit": s.get("bytes_limit", 0) / 1e9}


def memory_note(tag, log=print):
    m = gpu_memory_gb()
    if m is not None:
        log(f"{tag}: device peak {m['peak']:.2f} GB, in use {m['in_use']:.2f} GB, limit {m['limit']:.2f} GB")
    return m
