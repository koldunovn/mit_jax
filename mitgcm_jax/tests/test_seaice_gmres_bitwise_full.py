"""Bit-identity gate of the implicit-LSR GMRES (pkgs/seaice_lsr.py `_gmres`) against the frozen copy of commit
ef3ea4e (test_seaice_gmres_bitwise._gmres_frozen): the full part (nightly).

Every comparison counts values whose 64-bit patterns differ (+0 and -0 distinguished); the gate is zero everywhere.
Oracle full_jaxdump_v5, iterations 1-3; CPU with the gate XLA flags (conftest.py).
  1. GMRES alone, production setting (restart 40, 8 cycles), on the LSOR system of each Picard pass of each
     iteration (6 systems), random right-hand side on the wet points: tangent solve (A, P) and transpose solve
     (A^T, P^T). Negative control: 7 cycles instead of 8 differ.
  2. lsor_solve's custom_jvp (the production path through lax.custom_linear_solve) on the same 6 systems: the tangent
     (du, dv) along a random direction of all 14 coefficient fields, and the VJP (all 14 coefficient cotangents) of
     random output cotangents.
  3. SEAICE_DYNSOLVER (two Picard passes, clipping): gradient and tangent of J = sum(w*UICE) + 1e-3 sum(fu_out)
     w.r.t. the ocean stress fu, fv at iterations 1-3.
  4. SEAICE_MODEL, adjoint levels "full" (implicit LSR derivative) and "no_dynamics": gradient of J = HEFF, Qnet, fu,
     UICE weighted sums w.r.t. every input at iterations 1-3 ("no_dynamics" never calls GMRES: its identity is the
     control that the level switch is unaffected), and the "full" tangent of every output at iterations 1-3.
The reference side of 2-4 runs the same code with seaice_lsr._gmres replaced by the frozen copy (fresh jit traces
after jax.clear_caches(); a planted 7-cycle GMRES in the same slot must change the result, so the swap is effective).
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.pkgs import seaice_dyn as sd
from mitgcm_jax.pkgs import seaice_lsr as sl
from mitgcm_jax.pkgs import seaice_model as sm
from mitgcm_jax.tests import test_seaice_dyn as TD
from mitgcm_jax.tests import test_seaice_gmres_bitwise as G

ITS = (1, 2, 3)
L = TD.L
EX = TD.EX


@contextlib.contextmanager
def gmres_impl(fn):
    """seaice_lsr._gmres replaced by fn for the duration (traces made inside use fn)."""
    orig = sl._gmres
    jax.clear_caches()
    sl._gmres = fn
    try:
        yield
    finally:
        sl._gmres = orig
        jax.clear_caches()


def seven_cycles(apply_A, apply_M, b, restart, cycles, gsum, L):
    """Negative control: the frozen copy with one restart cycle less."""
    return G._gmres_frozen(apply_A, apply_M, b, restart, cycles - 1, gsum)


def diffs(new, ref, prefix=""):
    """{path: differing values} over two equal pytrees."""
    fn, fr = jax.tree_util.tree_flatten_with_path(new)[0], jax.tree_util.tree_flatten_with_path(ref)[0]
    assert [p for p, _ in fn] == [p for p, _ in fr]
    return {prefix + jax.tree_util.keystr(p): G.bitdiff(a, b) for (p, a), (_, b) in zip(fn, fr)}


def nonzero_finite(tree):
    leaves = [np.asarray(x) for x in jax.tree.leaves(tree)]
    return all(np.all(np.isfinite(x)) for x in leaves) and any(np.any(x != 0) for x in leaves)


@pytest.fixture(scope="module")
def env():
    return TD.Env()


@pytest.fixture(scope="module")
def probs(env):
    return G.lsr_problems(env, ITS)


# ---------------------------------------------------------------------------------------------------------------
# 1. GMRES alone, 8 cycles

_solve = jax.jit(G.solve_both, static_argnums=(0, 4, 5))


def test_gmres_bitwise_all_states(env, probs):
    """6 LSOR systems x (tangent, transpose) solve, restart 40, 8 cycles: zero differing bits; 7 cycles differ."""
    p = env.p
    bad = {}
    for (it, ip), (co, _, _) in probs.items():
        b = G.random_rhs(co, 10 * it + ip)
        ref = _solve(G.frozen, p, co, b, p.lsr_ad_cycles)
        new = _solve(sl._gmres, p, co, b, p.lsr_ad_cycles)
        assert nonzero_finite(ref)
        bad.update(diffs(new, ref, f"it{it}/p{ip}"))
        if (it, ip) == (1, 1):
            neg = _solve(G.frozen, p, co, b, p.lsr_ad_cycles - 1)
            assert G.bitdiff(neg[0][0], ref[0][0]) > 10000 and G.bitdiff(neg[1][1], ref[1][1]) > 10000
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}


# ---------------------------------------------------------------------------------------------------------------
# 2. lsor_solve custom_jvp: tangent and VJP


def _lsor_derivs(p, co, u, v, dco, wu, wv):
    def F(c):
        uu, vv, _ = sl.lsor_solve(p, EX, c, u, v)
        return uu, vv

    _, tan = jax.jvp(F, (co,), (dco,))
    _, pull = jax.vjp(F, co)
    (ct,) = pull((wu, wv))
    return tan, ct


def _direction(co, seed):
    rng = np.random.default_rng(seed)
    J, I = sl._interior(L)
    d = {}
    for k, v in co.items():
        r = np.zeros(L.shape2d)
        r[:, J, I] = rng.standard_normal((L.nTiles, L.sNy, L.sNx))
        d[k] = jnp.asarray(r) * v * 1e-3
    wu = jnp.asarray(rng.standard_normal(L.shape2d)) * co["seaiceMaskU"]
    wv = jnp.asarray(rng.standard_normal(L.shape2d)) * co["seaiceMaskV"]
    return d, wu, wv


def _run_lsor_derivs(p, probs, impl):
    out = {}
    with gmres_impl(impl):
        f = jax.jit(lambda p, co, u, v, d, wu, wv: _lsor_derivs(p, co, u, v, d, wu, wv))
        for (it, ip), (co, u, v) in probs.items():
            out[(it, ip)] = jax.block_until_ready(f(p, co, u, v, *_direction(co, 100 * it + ip)))
    return out


def test_lsor_solve_jvp_vjp_bitwise(env, probs):
    """lsor_solve tangent (du, dv) and VJP (14 coefficient cotangents) on the 6 oracle LSOR systems: zero differing
    bits between the current _gmres and the frozen copy; the planted 7-cycle GMRES in the same slot changes them."""
    ref = _run_lsor_derivs(env.p, probs, G.frozen)
    new = _run_lsor_derivs(env.p, probs, sl._gmres)
    bad = {}
    for key in probs:
        assert nonzero_finite(ref[key])
        bad.update(diffs(new[key], ref[key], f"it{key[0]}/p{key[1]}"))
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    neg = _run_lsor_derivs(env.p, {(1, 1): probs[(1, 1)]}, seven_cycles)
    assert G.bitdiff(neg[(1, 1)][0][0], ref[(1, 1)][0][0]) > 10000


# ---------------------------------------------------------------------------------------------------------------
# 3. SEAICE_DYNSOLVER gradient and tangent


def _dyn_derivs(p, g, sg, ex, st, wu, du, dv):
    def J(fu, fv):
        out, _ = sd.dynsolver(p, g, sg, ex, dict(st, fu=fu, fv=fv))
        return jnp.sum(wu * out["UICE"]) + 1e-3 * jnp.sum(out["fu"])

    grad = jax.grad(J, argnums=(0, 1))(st["fu"], st["fv"])
    _, tan = jax.jvp(lambda a, b: sd.dynsolver(p, g, sg, ex, dict(st, fu=a, fv=b))[0], (st["fu"], st["fv"]),
                     (du, dv))
    return grad, {k: tan[k] for k in ("UICE", "VICE", "fu", "fv")}


def _run_dyn(env, impl):
    out = {}
    with gmres_impl(impl):
        f = jax.jit(lambda p, g, sg, ex, st, wu, du, dv: _dyn_derivs(p, g, sg, ex, st, wu, du, dv))
        for it in ITS:
            st = {k: jnp.asarray(v) for k, v in env.dyn_state(it).items()}
            sg = env.sg(it)
            rng = np.random.default_rng(200 + it)
            wu = jnp.asarray(rng.standard_normal(L.shape2d)) * sg["seaiceMaskU"]
            du = jnp.asarray(rng.standard_normal(L.shape2d)) * env.g.maskW[:, 0] * 1e-2
            dv = jnp.asarray(rng.standard_normal(L.shape2d)) * env.g.maskS[:, 0] * 1e-2
            out[it] = jax.block_until_ready(f(env.p, env.g, sg, EX, st, wu, du, dv))
    return out


def test_dynsolver_derivatives_bitwise(env):
    """Whole SEAICE_DYNSOLVER at iterations 1-3: gradient d J / d(fu, fv) and the tangent of UICE, VICE, fu, fv:
    zero differing bits."""
    ref = _run_dyn(env, G.frozen)
    new = _run_dyn(env, sl._gmres)
    bad = {}
    for it in ITS:
        assert nonzero_finite(ref[it])
        bad.update(diffs(new[it], ref[it], f"it{it}"))
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}


# ---------------------------------------------------------------------------------------------------------------
# 4. SEAICE_MODEL adjoint levels


def _model_grads(levels, impl):
    from mitgcm_jax.tests import test_seaice_model as TM

    e = TM.env()
    out = {}
    with gmres_impl(impl):
        for ad in levels:
            def J(i, P, g, sg, ex, w, ad=ad):
                o, _ = sm.seaice_model(P, g, sg, ex, i, ad=ad)
                return sum(jnp.sum(w[k] * o[k]) for k in w)

            f = jax.jit(jax.grad(J))
            tl = jax.jit(lambda i, d, P, g, sg, ex, ad=ad: jax.jvp(
                lambda x: sm.seaice_model(P, g, sg, ex, x, ad=ad)[0], (i,), (d,))[1])
            for it in ITS:
                ins = TM.inputs(it)
                rng = np.random.default_rng(300 + it)
                J_, I_ = L.js(1, L.sNy), L.is_(1, L.sNx)
                w = {}
                for k, m in (("HEFF", "HEFFM"), ("Qnet", "HEFFM"), ("fu", "seaiceMaskU"), ("UICE", "seaiceMaskU")):
                    a = np.zeros(L.shape2d)
                    a[:, J_, I_] = rng.standard_normal((L.nTiles, L.sNy, L.sNx))
                    w[k] = jnp.asarray(a) * e.sg[m]
                out[(ad, it, "grad")] = jax.block_until_ready(f(ins, e.P, e.g, e.sg, TM.EX, w))
                if ad == "full":  # tangent of every output along a random direction of every input
                    d = {k: jnp.asarray(rng.standard_normal(np.shape(v))) * 1e-3 for k, v in ins.items()}
                    out[(ad, it, "tangent")] = jax.block_until_ready(tl(ins, d, e.P, e.g, e.sg, TM.EX))
    return out


def test_seaice_model_levels_bitwise():
    """SEAICE_MODEL, iterations 1-3: gradients of every input at levels "full" and "no_dynamics", and the "full"
    tangent of every output along a random direction of every input: zero differing bits."""
    levels = ("full", "no_dynamics")
    ref = _model_grads(levels, G.frozen)
    new = _model_grads(levels, sl._gmres)
    bad = {}
    for key in ref:
        assert nonzero_finite(ref[key])
        bad.update(diffs(new[key], ref[key], f"{key[0]}/it{key[1]}/{key[2]}"))
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
