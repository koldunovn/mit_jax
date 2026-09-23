"""Time means of model fields accumulated every step (plan Task 19), for a Python-loop driver or a lax.scan carry.

    acc = means_init(st)                        # zero sums of the selected fields (default MEAN_FIELDS)
    acc = means_accumulate(acc, st, weight)     # after every step (jit-able; weight = the step's share of the period)
    m = means_finish(acc)                       # dict of time means (sum / total weight)

A simple (not thickness-weighted) time average of the State after each step, like the MITgcm diagnostics package
(each accumulated snapshot is the state at the end of a time step; the period mean is sum/count). Accumulating every
step instead of saving snapshots avoids aliasing the diurnal cycle into the means (fesom_jax: 00 UTC snapshots put a
wavenumber-1 pattern into the SST). Sums are float64; `acc` is a pytree of arrays (fields + total weight), so it can
sit in a scan carry next to the State; `integrate_with_diagnostics` does that together with per-step budgets and
monitor statistics.

Field names are State fields, plus derived 2-D fields in DERIVED ("sst" = theta at level 1).
"""

import jax
import jax.numpy as jnp
from jax import lax

MEAN_FIELDS = ("theta", "salt", "etaN", "uVel", "vVel", "sst")
DERIVED = {"sst": lambda st: st.theta[:, 0]}


def _get(st, name):
    if name in DERIVED:
        return DERIVED[name](st)
    return getattr(st, name)


def means_init(st, names=MEAN_FIELDS):
    """Zero accumulator for the fields `names` of State `st` (shapes taken from st). Returns
    {"sum": {name: zeros float64}, "w": 0.0}."""
    return {"sum": {n: jnp.zeros(jnp.shape(_get(st, n)), jnp.float64) for n in names},
            "w": jnp.zeros((), jnp.float64)}


def means_accumulate(acc, st, weight=1.0):
    """acc + weight*fields(st). Pure and jit-able; `weight` may be traced."""
    w = jnp.asarray(weight, jnp.float64)
    return {"sum": {n: s + w * _get(st, n) for n, s in acc["sum"].items()}, "w": acc["w"] + w}


def means_finish(acc):
    """{name: time mean} = sum / total weight (NaN if nothing was accumulated)."""
    return {n: s / acc["w"] for n, s in acc["sum"].items()}


def integrate_with_diagnostics(step, model, st, xs, *, mean_names=MEAN_FIELDS, weights=None, budget=None,
                               monitor=None):
    """lax.scan of `step(model, st, x) -> st` over the stacked per-step inputs xs, with the time means in the carry
    and per-step diagnostics as scan outputs:
      budget(st0, st1) -> dict of scalars  (e.g. partial(budgets.step_budget, P, g, kLowC))
      monitor(st1)     -> pytree of scalars (e.g. partial(monitor.dynstat_device, g=g))
    weights: per-step weights [n] (default 1). Returns (st_n, means_acc, ys) with ys = {"budget": ..., "monitor":
    ...} stacked over steps (absent keys omitted). `st` must already have every field the step writes
    (adjoint.checkpoint.prepare_state)."""
    n = int(jax.tree.leaves(xs)[0].shape[0])
    ws = jnp.ones((n,), jnp.float64) if weights is None else jnp.asarray(weights, jnp.float64)
    acc = means_init(st, mean_names)

    def body(carry, xw):
        s, a = carry
        x, w = xw
        s1 = step(model, s, x)
        a = means_accumulate(a, s1, w)
        ys = {}
        if budget is not None:
            ys["budget"] = budget(s, s1)
        if monitor is not None:
            ys["monitor"] = monitor(s1)
        return (s1, a), ys

    (st, acc), ys = lax.scan(body, (st, acc), (xs, ws))
    return st, acc, ys
