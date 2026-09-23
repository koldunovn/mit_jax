"""Bit-identity gate of the implicit-LSR GMRES (pkgs/seaice_lsr.py `_gmres`) against a FROZEN copy of the
implementation of commit ef3ea4e (`_gmres_frozen` below, verbatim): the fast part (tier1x).

Any rewrite of `_gmres` made for speed must give the same 64-bit patterns (+0 and -0 distinguished) as the frozen
copy on CPU with the gate XLA flags (conftest.py: no FMA, no algsimp). Here: the oracle LSOR system of iteration 1,
Picard pass 1 (full_jaxdump_v5; operator A, preconditioner P = one line-SOR sweep, built as _lsor_implicit_jvp builds
them), one GMRES(40) restart cycle, a random right-hand side on the wet points: the solution of A x = b (tangent solve)
and of A^T x = b with P^T (transpose solve, both from jax.linear_transpose). One cycle runs every code path (Arnoldi
with CGS2, the normalisation, the least-squares solve, the update); the full 8 cycles, all oracle states
(iterations 1-3, both Picard passes), the custom_jvp tangent/adjoint of lsor_solve and the dynsolver / SEAICE_MODEL
derivatives are in test_seaice_gmres_bitwise_full.py (nightly).
Negative controls: the frozen copy with the tile partial sums of the inner products added in reverse tile order
(round-off only), and with 8 -> 7 restart cycles (in the nightly file), must differ from the frozen copy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from mitgcm_jax.parallel.global_sum import global_sum_tile
from mitgcm_jax.pkgs import seaice_lsr as sl
from mitgcm_jax.tests import test_seaice_dyn as TD

FROZEN_COMMIT = "ef3ea4e"


def _gmres_frozen(apply_A, apply_M, b, restart, cycles, gsum):
    """seaice_lsr._gmres as of commit ef3ea4e (verbatim; do not edit)."""
    def dot_many(Vs, w):  # Vs: pytree with leading basis axis [k, T, ...]; w: [T, ...] -> [k]
        part = sum(jnp.sum(Va * wa[None], axis=(-2, -1)) for Va, wa in zip(Vs, w))  # [k, T]
        return gsum(part.T)

    def dot(a, c):
        part = sum(jnp.sum(x * y, axis=(-2, -1)) for x, y in zip(a, c))  # [T]
        return gsum(part)

    def scale(a, s):
        return tuple(x * s for x in a)

    def safe_normalize(w):
        nrm = jnp.sqrt(dot(w, w))
        ok = nrm > 0.0
        return scale(w, jnp.where(ok, 1.0 / jnp.where(ok, nrm, 1.0), 0.0)), nrm

    m = restart

    def cycle(_, x):
        Ax = apply_A(x)
        r0 = apply_M((b[0] - Ax[0], b[1] - Ax[1]))
        v0, beta = safe_normalize(r0)
        V = tuple(jnp.zeros((m + 1,) + a.shape, a.dtype).at[0].set(a) for a in v0)
        H = jnp.zeros((m + 1, m), b[0].dtype)

        def arnoldi(j, carry):
            V, H = carry
            vj = tuple(Va[j] for Va in V)
            w = apply_M(apply_A(vj))
            act = (jnp.arange(m + 1) <= j).astype(w[0].dtype)
            h = dot_many(V, w) * act
            w = tuple(wa - jnp.tensordot(h, Va, axes=1) for wa, Va in zip(w, V))
            h2 = dot_many(V, w) * act  # re-orthogonalisation (CGS2)
            w = tuple(wa - jnp.tensordot(h2, Va, axes=1) for wa, Va in zip(w, V))
            h = h + h2
            w, nrm = safe_normalize(w)
            H = H.at[:, j].set(h.at[j + 1].set(nrm))
            V = tuple(Va.at[j + 1].set(wa) for Va, wa in zip(V, w))
            return V, H

        V, H = lax.fori_loop(0, m, arnoldi, (V, H))
        e1 = jnp.zeros((m + 1,), H.dtype).at[0].set(beta)
        y = jnp.linalg.lstsq(H, e1)[0]
        return tuple(xa + jnp.tensordot(y, Va[:m], axes=1) for xa, Va in zip(x, V))

    return lax.fori_loop(0, cycles, cycle, (jnp.zeros_like(b[0]), jnp.zeros_like(b[1])))


def frozen(apply_A, apply_M, b, restart, cycles, gsum, L):
    """_gmres_frozen with the current signature of seaice_lsr._gmres (the layout argument is not used)."""
    return _gmres_frozen(apply_A, apply_M, b, restart, cycles, gsum)


def natural(apply_A, apply_M, b, restart, cycles, gsum, L):
    """seaice_lsr._gmres_natural (the non-CPU form) with the signature of _gmres."""
    return sl._gmres_natural(apply_A, apply_M, b, restart, cycles, gsum)


def reversed_tile_gsum(phiTile):
    """Negative control: the tile partial sums added in reverse tile order (round-off-level change)."""
    return global_sum_tile(phiTile[::-1])


def bitdiff(a, b):
    """Number of values whose 64-bit patterns differ (+0 and -0 count as different)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype == np.float64, (a.shape, b.shape, a.dtype, b.dtype)
    return int(np.sum(a.view(np.int64) != b.view(np.int64)))


def operators(p, ex, co, L):
    """A (matvec), P (preconditioner) exactly as _lsor_implicit_jvp builds them."""
    J, I = sl._interior(L)
    T = co["AU"].shape[0]
    inner = jnp.zeros(co["AU"].shape, bool).at[:, J, I].set(True)
    ln = sl._lines(co, L)
    w = jnp.concatenate([jnp.full((T,), p.SEAICE_LSRrelaxU), jnp.full((T,), p.SEAICE_LSRrelaxV)])
    return (lambda x: sl._matvec(ex, co, inner, x)), sl._precond_impl(p, ln, L, w)


def solve_both(impl, p, co, b, cycles, gsum=None, ex=TD.EX):
    """(x, xt): impl on A x = b with M = P, and on A^T x = b with M = P^T (jax.linear_transpose), m = lsr_ad_restart."""
    A, P = operators(p, ex, co, ex.L)
    gsum = ex.global_sum_tile if gsum is None else gsum
    m = p.lsr_ad_restart
    x = impl(A, P, b, m, cycles, gsum, ex.L)
    AT, PT = jax.linear_transpose(A, b), jax.linear_transpose(P, b)
    xt = impl(lambda r: AT(r)[0], lambda r: PT(r)[0], b, m, cycles, gsum, ex.L)
    return x, xt


def random_rhs(co, seed):
    """Random interior values on the wet points (seaiceMaskU/V), zero elsewhere."""
    rng = np.random.default_rng(seed)
    L = TD.L
    J, I = sl._interior(L)
    out = []
    for mk in ("seaiceMaskU", "seaiceMaskV"):
        a = np.zeros(L.shape2d)
        a[:, J, I] = rng.standard_normal((L.nTiles, L.sNy, L.sNx))
        out.append(jnp.asarray(a) * co[mk])
    return tuple(out)


def lsr_problems(env, its):
    """{(it, pass): (co, u, v)}: the LSOR system of each Picard pass of SEAICE_LSR replayed from the oracle, and the
    iterate lsor_solve returns (stage L04, before the masking)."""
    out = {}
    for it in its:
        _, passes = env.lsr(it)
        for ip, pr in enumerate(passes, 1):
            co = {k: jnp.asarray(v) for k, v in pr["co"].items()}
            out[(it, ip)] = (co, jnp.asarray(pr["L04"]["UICE"]), jnp.asarray(pr["L04"]["VICE"]))
    return out


@pytest.fixture(scope="module")
def env():
    return TD.Env()


@pytest.fixture(scope="module")
def prob(env):
    return lsr_problems(env, (1,))[(1, 1)]


_solve = jax.jit(solve_both, static_argnums=(0, 4, 5))


@pytest.mark.parametrize("impl", ["_gmres", "_gmres_windowed", "natural"])
def test_gmres_one_cycle_bitwise(env, prob, impl):
    """_gmres (the CPU dispatch: _gmres_windowed), _gmres_windowed and _gmres_natural (the non-CPU form) == the frozen
    copy, every bit, on the oracle LSOR system (iteration 1, pass 1), one restart cycle, for the tangent solve (A, P)
    and the transpose solve (A^T, P^T); the solution is nonzero and finite."""
    co = prob[0]
    b = random_rhs(co, 0)
    fn = natural if impl == "natural" else getattr(sl, impl)
    ref = _solve(frozen, env.p, co, b, 1)
    new = _solve(fn, env.p, co, b, 1)
    diffs = {f"{name}/{c}": bitdiff(n[i], r[i]) for name, n, r in (("tangent", new[0], ref[0]),
                                                                   ("transpose", new[1], ref[1])) for i, c in
             enumerate("uv")}
    assert not any(diffs.values()), diffs
    for x in jax.tree.leaves(ref):
        assert np.all(np.isfinite(np.asarray(x))) and np.any(np.asarray(x) != 0)


def test_gmres_gate_negative_control(env, prob):
    """The comparison sees a round-off-level change: the frozen copy with the tile partial sums added in reverse tile
    order differs from the frozen copy in (almost) every wet value."""
    co = prob[0]
    b = random_rhs(co, 0)
    ref = _solve(frozen, env.p, co, b, 1)
    bad = jax.jit(solve_both, static_argnums=(0, 4, 5))(frozen, env.p, co, b, 1, reversed_tile_gsum)
    n = bitdiff(bad[0][0], ref[0][0])
    assert n > 10000, n
    assert bitdiff(bad[1][1], ref[1][1]) > 10000
