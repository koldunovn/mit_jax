"""Derivative of a code block that TAF skips in the reverse sweep (plan M2.6b-1; sea-ice adjoint levels).

When a package switch is .FALSE. only in the adjoint (data.autodiff useSEAICEinAdMode = .FALSE.,
autodiff_inadmode_set_ad.F:37; SEAICEuseDYNAMICSswitchInAd, :49-51), TAF's reverse sweep jumps over the IF block: the
adjoint variables are not touched there. For every variable X the block overwrites (reads and writes, or only writes a
part of it) the adjoint passes through unchanged, adX_before = adX_after; a variable the block only reads receives no
contribution from it. That is the derivative of the IDENTITY map on the overwritten variables, not a stop_gradient
(which would zero adX_before).

`skipped_in_reverse(fn, inout, n_static)` wraps a block written as `fn(*static, ins, rest) -> (outs, aux)`:
  - static: hashable arguments (flags, callables), passed through unchanged;
  - ins: dict of the block's differentiable inputs; `rest`: everything else it reads (params pytree, grid, fixed
    fields, exchanger) -- carries no derivative;
  - outs: dict of the block's results, aux: any pytree of extra results (dump-stage records, counts).
The forward is `fn` itself (same operations; the custom_jvp is inlined when lowered, so the values are byte-identical to
calling fn). The tangent rule: d outs[k] = d ins[k] for k in `inout` (the overwritten variables, present in both dicts
with the same shape), 0 for every other output (outputs no later code reads before overwriting them, e.g. viscosities
that the block recomputes from scratch every step) and for aux; tangents of read-only inputs and of `rest` are
dropped. The rule is linear, so JAX transposes it: the VJP is ct_ins[k] = ct_outs[k] for k in inout, 0 for the other
inputs. As a custom_jvp (not custom_vjp) it also supports forward mode, with the same approximation (TAF's tangent-
linear model keeps every package; use the exact level for tangent-linear work).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _zero_tangent(x):
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros(x.shape, x.dtype)
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


def skipped_in_reverse(fn, inout, n_static=0):
    """fn(*static, ins, rest) -> (outs, aux) with the derivative of TAF's skipped block (module docstring)."""
    inout = tuple(inout)

    @partial(jax.custom_jvp, nondiff_argnums=tuple(range(n_static)))
    def wrapped(*args):
        return fn(*args)

    @wrapped.defjvp
    def _jvp(*args):
        static, (primals, tangents) = args[:n_static], args[n_static:]
        ins, rest = primals
        dins = tangents[0]
        outs, aux = fn(*static, ins, rest)
        missing = [k for k in inout if k not in ins or k not in outs]
        if missing:
            raise ValueError(f"skipped_in_reverse: {missing} must be both an input and an output of the block")
        douts = {k: (jnp.asarray(dins[k], jnp.asarray(v).dtype) if k in inout else _zero_tangent(v))
                 for k, v in outs.items()}
        return (outs, aux), (douts, jax.tree.map(_zero_tangent, aux))

    return wrapped
