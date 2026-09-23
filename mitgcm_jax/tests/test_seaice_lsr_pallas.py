"""GPU path of the LSOR sweep (pkgs/seaice_lsr_pallas.py, SeaiceDynParams.lsr_impl = "pallas"), gated on CPU: the
same Pallas kernel run by the Pallas interpreter ("pallas_interpret") must be bitwise equal to the XLA sweep
(seaice_lsr._sweep) and, inside SEAICE_LSR, to the Fortran oracle (full_jaxdump_v5, iteration 1: L01-L04 of both
Picard passes and Y05 bitwise, LSOR counts 178/118). Also the unrolled XLA path ("xla_unrolled", the non-CUDA GPU
default, and partial unrolls). The interpreter runs the kernel's own jaxpr (its loads, stores and arithmetic in the
kernel's order); the compiled Triton kernel is gated on the GPU by scripts/runs/lsr_perf_bench.py --full (every output
of the whole dynsolver at iterations 1-3 against the XLA path and the Fortran; compiler contraction into FMA is not
covered by the interpreter). The interpreter cannot run under shard_map(check_vma=True) (its internal scans mix
varying and invariant carries); the compiled kernel can: scripts/runs/lsr_sharded_gpu_check.py (4 GPUs == 1 GPU).
Negative controls: one bet factor moved by 1 ulp, the relaxation factor x(1+2^-52), a reassociated right-hand side in
the kernel — each makes the sweep differ.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.pkgs import seaice_lsr as sl
from mitgcm_jax.pkgs import seaice_lsr_pallas as slp
from mitgcm_jax.tests import test_seaice_dyn as T


def _synthetic_lines(seed=0, nline=90, nal=90, lanes=26):
    """Random diagonally dominant lines with the elimination factors of seaice_lsr._lines."""
    rng = np.random.default_rng(seed)

    def R(*s):
        return jnp.asarray(rng.standard_normal(s))

    A, C = R(nline, nal, lanes) * 0.2, R(nline, nal, lanes) * 0.2
    B = 1.0 + jnp.abs(R(nline, nal, lanes))
    CUU0 = C[:, 0] / B[:, 0]

    def elim(c, xs):
        a, b, cc = xs
        bet = b - a * c
        cu = cc / bet
        return cu, (bet, cu)

    _, (bet, cuu) = jax.lax.scan(elim, CUU0, tuple(jnp.moveaxis(x[:, 1:], 1, 0) for x in (A, B, C)))
    ln = dict(A=A, B0=B[:, 0], Clast=C[:, -1], bet=jnp.concatenate([B[:, :1], jnp.moveaxis(bet, 0, 1)], axis=1),
              CUU=jnp.concatenate([CUU0[:, None], jnp.moveaxis(cuu, 0, 1)], axis=1), Rt1=R(nline, nal, lanes) * 0.1,
              Rt2=R(nline, nal, lanes) * 0.1, rhs=R(nline, nal, lanes),
              mask=jnp.asarray((rng.random((nline, nal, lanes)) > 0.2).astype(float)))
    args = (R(nline, nal, lanes), R(nline, lanes), R(nline, lanes), R(nal, lanes), R(nline, nal, lanes),
            jnp.full((lanes,), 0.95))  # tmp, lo, hi, prev0, nxt, w
    return ln, args


def _pallas(lp, args):
    return np.asarray(jax.jit(lambda lp, *a: slp.sweep(lp, *a, interpret=True))(lp, *args))


def test_pallas_sweep_interpret_bitwise(monkeypatch):
    """One sweep: Pallas kernel (interpreted) == XLA sweep == XLA sweep with unrolled Thomas scans, at every point;
    negative controls: a 1-ulp change of one elimination factor, the relaxation factor x(1+2^-52), a reassociated
    right-hand side in the kernel."""
    ln, args = _synthetic_lines()
    ref = np.asarray(jax.jit(sl._sweep)(ln, *args))
    lp = slp.pack_lines(ln)
    assert int(np.sum(_pallas(lp, args) != ref)) == 0
    for u in (2, 8, True):
        assert int(np.sum(np.asarray(jax.jit(lambda *a: sl._sweep(*a, unroll=u))(ln, *args)) != ref)) == 0, u
    # negative controls. (1) one elimination factor bet moved by 1 ulp: the lines damp it (|CUU|, |Rt| < 1), so only
    # a few points change, but they do
    bad = dict(lp, big=lp["big"].at[5, 40, 30, 3].set(jnp.nextafter(lp["big"][5, 40, 30, 3], jnp.inf)))
    assert int(np.sum(_pallas(bad, args) != ref)) > 0
    # (2) relaxation factor x(1+2^-52)
    w = args[-1] * (1.0 + 2.0 ** -52)
    assert int(np.sum(_pallas(lp, args[:-1] + (w,)) != ref)) > 1000
    # (3) an operation-order change inside the kernel: (rhs + AA3) + (Rt1*x(J-1) + Rt2*xold(J+1)) instead of
    # ((rhs + AA3) + Rt1*x(J-1)) + Rt2*xold(J+1)
    monkeypatch.setattr(slp, "_rhs", lambda rhs, aa3, rt1, xp, rt2, xn, m: ((rhs + aa3) + (rt1 * xp + rt2 * xn)) * m)
    assert int(np.sum(_pallas(lp, args) != ref)) > 1000


@pytest.fixture(scope="module")
def env():
    return T.Env()


@pytest.mark.parametrize("impl", ["pallas_interpret", "xla_unrolled"])
def test_lsr_replay_bitwise_impl(env, impl):
    """SEAICE_LSR at iteration 1 with the GPU kernel (interpreted) or the fully unrolled XLA sweep: L01-L04 of both
    Picard passes and Y05 bitwise vs the Fortran, LSOR counts 178/118."""
    it = 1
    p = dataclasses.replace(env.p, lsr_impl=impl)
    out, passes = T._run_lsr(p, env.g, env.sg(it), T.EX, env.lsr_state(it))
    bad = T._lsr_mismatches(env, it, out, passes)
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    assert tuple(int(pr["L04"]["ICOUNT1"]) for pr in passes) == T.COUNTS[it]

