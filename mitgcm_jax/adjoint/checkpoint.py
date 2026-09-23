"""Time integration for reverse mode: FORWARD_STEP in a `lax.scan` with rematerialization schedules (plan Task 18).

    xs = exf_window(loader, nml, it0, n)                    # host: the EXF record inputs of steps it0 .. it0+n-1
    model = Model(P, g, kLowC, ex)                          # pytree, a jit ARGUMENT (never closed over)
    step = make_step(adj)                                   # (model, st, x) -> st, one FORWARD_STEP
    st_n, acc = integrate(step, model, st0, xs, schedule="step")

Schedules (what the reverse pass stores; the forward values are the same in every schedule, tested):
  "none"   plain scan: every intermediate of every step is kept (only for tiny windows / tests)
  "step"   per-step remat (L2): the State carry of every step is kept, each step is recomputed once in the reverse
           pass and its VJP taken with the step's intermediates live (the per-step reverse working set)
  "sqrt"   two-level remat (L3): S ~ sqrt(N) outer segments, each a checkpointed scan of per-step-checkpointed
           steps: S + N/S carries stored, every step recomputed twice
Long windows: `grad.chunked_value_and_grad` (chunks of this integrator, boundaries parked on the host).

The carry is (State, acc): `acc` accumulates an optional running cost `cost(model, st_new, x)` after every step, so
a time-integrated objective (box-mean theta over a window) needs no stored trajectory. With no cost, acc stays 0.

EXF inputs: `exf_window` runs the host-side record loader (ExfRecordLoader, the literal exf_set_fld.F record logic)
for the window's steps and stacks what FORWARD_STEP consumes per step (fld0/fld1 buffers, interpolation weights,
myTime) on a leading step axis, as numpy (device_put per chunk by the caller). ~16 MB per step at LLC90.

Why a scan (fesom_jax lesson): reverse mode through a Python loop of jitted steps cannot be rematerialized and
keeps every step's intermediates; a scan with a checkpointed body keeps only the carries. The scan is bitwise equal
to the Python loop the forward driver (scripts/run_jax.py) runs (test_checkpoint.py).
"""

import math
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitgcm_jax.adjoint.modes import EXACT
from mitgcm_jax.core.forward_step import forward_step
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod

SCHEDULES = ("none", "step", "sqrt")
# Intermediates the rematerialized reverse pass keeps instead of recomputing (jax.ad_checkpoint.checkpoint_name in the
# kernels): the CG2D solution. Recomputing a step in the reverse pass would otherwise re-run the literal CG2D iteration
# (Fortran-order sums, ~160 iterations: most of a GPU step); keeping it costs one 2-D field per stored step.
SAVE_NAMES = ("cg2d_x",)


class Model(NamedTuple):
    """Everything FORWARD_STEP reads besides the State and the per-step EXF inputs, as one pytree. Pass it as a jit
    ARGUMENT: closed-over arrays become compile-time constants that XLA constant-folds (measured, one-step
    value_and_grad on CPU: grid and parameters closed over 275 s compile vs 59 s as arguments; the exchange maps
    closed over 64 s vs 26 s), and parameters must stay traced for the bitwise gates (params_io.params_pytree)."""
    P: Any       # forward_step.ModelParams
    g: Any       # grid.geometry.Grid
    kLowC: Any   # [T, ny, nx] int
    ex: Any      # parallel.exchange.Exchanger (a pytree: its maps are leaves)


def make_step(adj=EXACT):
    """(model, st, x) -> st: one FORWARD_STEP with backward-mode semantics `adj` (static AdjointConfig); x = one step
    of `exf_window` inputs. The aux outputs are dropped."""
    def step(model, st, x):
        st1, _ = forward_step(model.P, model.g, model.ex, model.kLowC, st, x, adj=adj)
        return st1
    return step


# ---------------------------------------------------------------------------------------------------------------
# EXF inputs


def exf_step_input(bufs, facs, myTime):
    """One step's EXF input as FORWARD_STEP takes it, from ExfRecordLoader.load (numpy, float64)."""
    return {"bufs": {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in bufs.items()},
            "facs": {k: np.float64(v) for k, v in facs.items()},
            "myTime": np.float64(myTime)}


def exf_window(loader, nml, it0, n):
    """EXF inputs of the n steps starting at iteration it0, stacked on a leading step axis (numpy).

    `loader` must be positioned at it0: ExfRecordLoader.load has been called for every iteration nIter0 .. it0-1
    (a fresh loader for it0 = nIter0; see `exf_loader_at`). Advances the loader by n steps."""
    nIter0 = int(nml.get("data", "parm03", "nIter0", default=0))
    steps = []
    for it in range(it0, it0 + n):
        myTime, myIter = exf_mod.model_time(nml, it - nIter0 + 1)
        bufs, facs, _ = loader.load(myTime, myIter)
        steps.append(exf_step_input(bufs, facs, myTime))
    return stack_steps(steps)


def exf_loader_at(P, g, rundir, nml, it0):
    """A fresh ExfRecordLoader advanced to iteration it0 (replays the record logic of nIter0 .. it0-1: reads only
    the records those steps load; no model work), as scripts/run_jax.py does on a restart."""
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    nIter0 = int(nml.get("data", "parm03", "nIter0", default=0))
    for it in range(nIter0, it0):
        loader.load(*exf_mod.model_time(nml, it - nIter0 + 1))
    return loader


def stack_steps(steps):
    """list of per-step pytrees -> one pytree with a leading step axis (numpy)."""
    return jax.tree.map(lambda *a: np.stack([np.asarray(x) for x in a]), *steps)


def take_steps(xs, lo, hi):
    """Steps lo..hi-1 of a stacked input (a fresh numpy copy: safe to device_put and free)."""
    return jax.tree.map(lambda a: np.array(a[lo:hi]), xs)


def n_steps(xs):
    return int(jax.tree.leaves(xs)[0].shape[0])


# ---------------------------------------------------------------------------------------------------------------
# the integrator


def prepare_state(step, model, st, x0):
    """The State with every field FORWARD_STEP produces: a scan carry must keep its structure, and the step adds
    fields its input may lack (PmEpR on the oracle's initial state). A field the input lacks cannot be read in the
    first step before it is written (the Python loop would fail on it), so it is added as zeros; later steps read
    what the previous step wrote, in the loop and in the scan alike. Also casts the iteration counter and every field
    to the dtype the step returns (the carry types must match exactly). x0: one step's inputs."""
    out = jax.eval_shape(step, model, st, x0)
    f = dict(st.f)
    for k, sd in out.f.items():
        if k not in f:
            f[k] = jnp.zeros(sd.shape, sd.dtype)
        elif jnp.shape(f[k]) != sd.shape:
            raise ValueError(f"State field {k}: shape {jnp.shape(f[k])} in, {sd.shape} out")
        else:
            f[k] = jnp.asarray(f[k], sd.dtype)
    extra = set(f) - set(out.f)
    if extra:
        raise ValueError(f"State fields the step drops: {sorted(extra)}")
    return type(st)(f, jnp.asarray(st.it, out.it.dtype))


def sqrt_segments(n):
    """Outer segment count S of the two-level scheme: minimises the carries it stores, S outer + n//S inner + the
    n mod S per-step tail (S ~ sqrt(n); a divisor of n when one is near: 24 steps -> 4 x 6, not 5 x 4 + 4)."""
    n = max(1, int(n))
    return min(range(1, n + 1), key=lambda S: (S + n // S + n % S, abs(S - math.sqrt(n))))


def integrate(step: Callable, model, st, xs, *, schedule="step", segments=None, cost=None, acc0=None,
              save_names=SAVE_NAMES):
    """Run n = len(xs) steps of `step` from `st` in a lax.scan; returns (st_n, acc).

    schedule: "none" | "step" | "sqrt" (module docstring). segments: outer segment count for "sqrt" (default
    sqrt_segments(n)); the remainder n mod S runs as a per-step-checkpointed tail. cost(model, st_new, x) -> scalar
    is accumulated into acc (acc0, default 0.0) after every step. save_names: named intermediates the per-step
    checkpoint keeps (SAVE_NAMES; () recomputes everything)."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule {schedule!r}; allowed {SCHEDULES}")
    n = n_steps(xs)
    acc = jnp.zeros((), jnp.float64) if acc0 is None else acc0
    st = prepare_state(step, model, st, jax.tree.map(lambda a: a[0], xs))

    def body(carry, x):
        s, a = carry
        s1 = step(model, s, x)
        if cost is not None:
            a = a + cost(model, s1, x)
        return (s1, a), None

    carry = (st, acc)
    if schedule == "none":
        carry, _ = lax.scan(body, carry, xs)
        return carry
    policy = jax.checkpoint_policies.save_only_these_names(*save_names) if save_names else None
    step_body = jax.checkpoint(body, prevent_cse=False, policy=policy)
    if schedule == "step":
        carry, _ = lax.scan(step_body, carry, xs)
        return carry
    S = sqrt_segments(n) if segments is None else int(segments)
    M = n // S if S > 0 else 0
    if S <= 1 or M < 1:
        carry, _ = lax.scan(step_body, carry, xs)
        return carry
    used = S * M

    def seg_body(c, seg_xs):
        c, _ = lax.scan(step_body, c, seg_xs)
        return c, None

    main = jax.tree.map(lambda a: a[:used].reshape((S, M) + a.shape[1:]), xs)
    carry, _ = lax.scan(jax.checkpoint(seg_body, prevent_cse=False), carry, main)
    if used < n:
        carry, _ = lax.scan(step_body, carry, jax.tree.map(lambda a: a[used:], xs))
    return carry


def run_loop(step_jit, model, st, xs_steps):
    """Reference driver: a Python loop over a jitted step (as scripts/run_jax.py), xs_steps = list of per-step
    inputs. For the scan == loop gate and for forward runs that must not be differentiated."""
    for x in xs_steps:
        st = step_jit(model, st, x)
    return st
