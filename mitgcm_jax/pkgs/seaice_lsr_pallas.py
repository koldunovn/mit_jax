"""One LSOR sweep of SEAICE_LSR (seaice_lsr._sweep) as a single Pallas-Triton GPU kernel (forward primal only).

Why: the XLA form of the sweep (lax.scan over 90 lines, two inner Thomas scans of 89 steps each) is ~16k sequential
scan steps per sweep, and on a GPU every step costs at least one kernel launch (A100-40 181 ms/sweep, GH200
92 ms/sweep). Here the whole sweep is one launch: one program (grid of 1), lanes = 13 u tiles + 13 v tiles padded to
32, one lane per thread; the line loop and both Thomas substitutions run sequentially inside the kernel.

Operation order: literally that of seaice_lsr._sweep / _thomas (seaice_lsr.F:1808-1870 TRIDIAGU, :1958-2022
TRIDIAGV), element by element: AA3(iMin) = 0 - A(iMin)*lo, AA3(iMax) = 0 - C(iMax)*hi, AA3 = 0 elsewhere;
r = (((rhs + AA3) + Rt1*x(J-1)) + Rt2*xold(J+1))*mask; y(iMin) = r/B(iMin); y(I) = (r - A*y(I-1))/bet(I);
v(iMax) = y(iMax), v(I) = y(I) - CUU(I)*v(I+1) (I = iMax-1 .. iMin); x = tmp + w*(v - tmp); bet and CUU are the
factors seaice_lsr._lines computes once per Picard pass. Only the memory traffic is organised for the GPU:
  - the parts of a line without a sequential dependence (r, and the relaxation x) are computed as [ROWS, lanes] blocks
    by all NUM_WARPS warps (the along axis padded to a multiple of ROWS; padded rows are never read by the
    substitutions and never stored into x), with a CTA barrier between phases;
  - the two substitutions issue the loads of the next CHUNK (CHUNK_B) steps before the current chunk's dependent
    arithmetic (_chain), so the load latency overlaps the chain.
XLA's Triton pipeline emits mul.rn/sub.rn/div.rn.f64 (no FMA contraction; checked in the PTX): on the A100 and the
GH200 the kernel is bitwise equal to the XLA sweep and to the Fortran (scripts/runs/lsr_perf_bench.py). On CPU the
Pallas interpreter runs the same kernel ("pallas_interpret"), bitwise equal to _sweep (tests/test_seaice_lsr_pallas.py,
with an operation-order negative control).

Cost (kernel only, measured 2026-09-23): GH200 2.44-2.47 ms/sweep, A100-40 3.6-3.8 ms/sweep (+ ~0.1 ms of XLA work
per sweep outside the kernel). Floors measured on the same GPUs: a dependent (mul, sub, div.rn.f64) step costs
151 ns (A100) / 64 ns (GH200), so the forward substitution alone needs >= 1.21 / 0.51 ms per sweep; the rest is load
latency the chains do not hide. Tried and not kept: loads inline, 1 warp (5.0 / 8.4 ms), prefetch distance 4 or 16,
backward distance 16, 1-2 warps with 8-16-row blocks, unrolled block loops, staging A/bet/CUU through the block pass.

Only the forward sweep uses this kernel. The implicit derivative (seaice_lsr._lsor_implicit_jvp) keeps the traceable
XLA preconditioner, whose transpose comes from jax.linear_transpose (fully unrolled Thomas scans off the CPU:
seaice_lsr._precond_impl).

Backends: SeaiceDynParams.lsr_impl = "auto" uses this kernel only when the computation is lowered for CUDA. Pallas'
Triton lowering also targets ROCm, but the kernel is untested on AMD GPUs (no hardware here: 64-thread wavefronts,
the barrier and the absence of FMA contraction are unverified), so "auto" runs "xla_unrolled" there. To try it on
ROCm: lsr_impl="pallas", then `python scripts/runs/lsr_perf_bench.py --variants xla_unrolled,pallas --full` (bitwise
vs the XLA path and the Fortran, sweep counts) and tests/test_seaice_lsr_pallas.py (interpreter, any backend).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

# fields of the padded line-coefficient pack (seaice_lsr._lines), in this order: [NF, line, along, lanes]
FIELDS = ("rhs", "A", "Rt1", "Rt2", "mask", "bet", "CUU")
# padding values of the extra lanes/rows: finite everywhere (bet = B = 1: padding lanes divide by 1)
_PAD = dict(rhs=0.0, A=0.0, Rt1=0.0, Rt2=0.0, mask=0.0, bet=1.0, CUU=0.0, B0=1.0, Clast=0.0)

CHUNK = 8  # prefetch distance of the forward substitution (steps)
CHUNK_B = 8  # prefetch distance of the back substitution (steps)
ROWS = 32  # along-line points per block of the vectorised passes
NUM_WARPS = 4


def n_lanes(lanes):
    """Lanes padded to a power of 2 (Triton tensor sizes)."""
    n = 1
    while n < lanes:
        n *= 2
    return n


def _pad_axis(a, n, axis, value=0.0):
    axis = axis % a.ndim
    k = n - a.shape[axis]
    if k == 0:
        return a
    shp = list(a.shape)
    shp[axis] = k
    return jnp.concatenate([a, jnp.full(shp, value, a.dtype)], axis=axis)


def _nalp(nal):
    return -(-nal // ROWS) * ROWS


def pack_lines(ln):
    """Once per Picard pass (outside the sweep loop): the line coefficients of seaice_lsr._lines with the lanes padded
    to n_lanes and the along axis to a multiple of ROWS: big [7, line, along, lanes] (FIELDS), edge [2, line, lanes]
    (B(iMin), C(iMax))."""
    n = n_lanes(ln["A"].shape[-1])
    nalp = _nalp(ln["A"].shape[1])
    big = jnp.stack([_pad_axis(_pad_axis(ln[k], n, -1, _PAD[k]), nalp, 1, _PAD[k]) for k in FIELDS])
    edge = jnp.stack([_pad_axis(ln["B0"], n, -1, _PAD["B0"]), _pad_axis(ln["Clast"], n, -1, _PAD["Clast"])])
    return dict(big=big, edge=edge)


def _rhs(rhs, aa3, rt1, xprev, rt2, xnext, mask):
    """:1815-1819 / :1965-1969  (((rhs + AA3) + Rt1*x(J-1)) + Rt2*xold(J+1)) * mask."""
    return (((rhs + aa3) + rt1 * xprev) + rt2 * xnext) * mask


def _chain(first, step_dir, n, load, step, carry, K):
    """carry = step(index, loaded, carry) for the n indices first, first+step_dir, ...; the loads of chunk c+1 (K
    steps) are issued before the K dependent steps of chunk c (indices past the end are clamped: loaded, unused)."""
    last = first + step_dir * (n - 1)

    def idx(s):
        return first + step_dir * s

    def clamp(i):
        return jnp.minimum(i, last) if step_dir > 0 else jnp.maximum(i, last)

    nch, rem = divmod(n, K)
    cur = tuple(load(clamp(idx(k))) for k in range(K))

    def body(c, st):
        carry, cur = st
        s0 = c * K
        nxt = tuple(load(clamp(idx(s0 + K + k))) for k in range(K))
        for k in range(K):
            carry = step(idx(s0 + k), cur[k], carry)
        return carry, nxt

    carry, cur = lax.fori_loop(0, nch, body, (carry, cur))
    for k in range(rem):  # tail: already loaded by the last chunk's prefetch
        carry = step(idx(nch * K + k), cur[k], carry)
    return carry


def _kernel(nal, barrier, big_ref, edge_ref, lohi_ref, prev0_ref, nxt_ref, tmp_ref, w_ref, xs_ref, r_ref, y_ref,
            v_ref):
    """xs_ref[0] = prev0 (the halo line before the first line), xs_ref[J+1] = the new line J. r_ref, y_ref, v_ref:
    per-line scratch (right-hand side, forward substitution, back substitution). Along axis padded to nalp."""
    nline, nalp, n = tmp_ref.shape
    RHS, A, RT1, RT2, MASK, BET, CUU = range(len(FIELDS))
    w = w_ref[...]
    rows = lax.broadcasted_iota(jnp.int32, (ROWS, n), 0)

    def blk(r0):
        return pl.ds(r0, ROWS)

    def blocks(body):
        lax.fori_loop(0, nalp // ROWS, lambda b, c: (body(b * ROWS), c)[1], 0)

    def copy_prev0(r0):
        xs_ref[0, blk(r0), :] = prev0_ref[blk(r0), :]

    blocks(copy_prev0)
    barrier()

    def line(J, c):
        # (1) r of the whole line: AA3(iMin) = 0 - A(iMin)*lo, AA3(iMax) = 0 - C(iMax)*hi, 0 elsewhere (:1811-1813)
        a_lo = jnp.zeros((ROWS, n), w.dtype) - big_ref[A, J, blk(0), :] * lohi_ref[0, J, :][None, :]
        a_hi = jnp.zeros((ROWS, n), w.dtype) - edge_ref[1, J, :][None, :] * lohi_ref[1, J, :][None, :]

        def rblock(r0):
            gi = rows + r0
            aa3 = jnp.where(gi == 0, a_lo, jnp.where(gi == nal - 1, a_hi, 0.0))
            r_ref[blk(r0), :] = _rhs(big_ref[RHS, J, blk(r0), :], aa3, big_ref[RT1, J, blk(r0), :],
                                     xs_ref[J, blk(r0), :], big_ref[RT2, J, blk(r0), :], nxt_ref[J, blk(r0), :],
                                     big_ref[MASK, J, blk(r0), :])

        blocks(rblock)
        barrier()
        # (2) forward substitution y(0) = r(0)/B(0) (:1826), y(I) = (r(I) - A(I)*y(I-1))/bet(I), I = 1..nal-1 (:1841)
        y0 = r_ref[0, :] / edge_ref[0, J, :]
        y_ref[0, :] = y0

        def fwd(i, ld, yp):
            r, a, b = ld
            y = (r - a * yp) / b
            y_ref[i, :] = y
            return y

        ylast = _chain(1, 1, nal - 1, lambda i: (r_ref[i, :], big_ref[A, J, i, :], big_ref[BET, J, i, :]), fwd, y0,
                       CHUNK)
        # (3) back substitution v(nal-1) = y(nal-1), v(I) = y(I) - CUU(I)*v(I+1), I = nal-2..0 (:1846-1855)
        v_ref[nal - 1, :] = ylast

        def bwd(i, ld, vn):
            y, cu = ld
            v = y - cu * vn
            v_ref[i, :] = v
            return v

        _chain(nal - 2, -1, nal - 1, lambda i: (y_ref[i, :], big_ref[CUU, J, i, :]), bwd, ylast, CHUNK_B)
        barrier()

        # (4) relaxation x = tmp + w*(v - tmp) (:1866-1867 / :2019-2020)
        def xblock(r0):
            t = tmp_ref[J, blk(r0), :]
            x = t + w[None, :] * (v_ref[blk(r0), :] - t)
            xs_ref[J + 1, blk(r0), :] = jnp.where(rows + r0 < nal, x, 0.0)

        blocks(xblock)
        barrier()
        return c

    lax.fori_loop(0, nline, line, 0)


def sweep(lp, tmp, lo, hi, prev0, nxt, w, interpret=False):
    """Drop-in for seaice_lsr._sweep (same arguments; `lp` = pack_lines(ln) instead of ln). Returns the new interior
    [line, along, lanes]. interpret: run the kernel with the Pallas interpreter (CPU tests)."""
    nline, nal, lanes = tmp.shape
    n = lp["big"].shape[-1]
    nalp = lp["big"].shape[2]
    assert nalp == _nalp(nal), (nalp, nal)
    lohi = jnp.stack([_pad_axis(lo, n, -1), _pad_axis(hi, n, -1)])
    args = (lp["big"], lp["edge"], lohi, _pad_axis(_pad_axis(prev0, n, -1), nalp, 0),
            _pad_axis(_pad_axis(nxt, n, -1), nalp, 1), _pad_axis(_pad_axis(tmp, n, -1), nalp, 1),
            _pad_axis(w, n, -1))
    kw = {}
    if interpret:
        barrier = lambda: None  # noqa: E731  (the interpreter runs the kernel as one sequential program)
    else:
        from jax.experimental.pallas import triton as plgpu
        barrier = plgpu.debug_barrier
        kw["compiler_params"] = plgpu.CompilerParams(num_warps=NUM_WARPS, num_stages=1)
    # inside shard_map(check_vma=True) the outputs must say how they vary: like the iterate
    mat = jax.typeof(tmp).manual_axis_type

    def out(shape):
        return jax.ShapeDtypeStruct(shape, tmp.dtype, manual_axis_type=mat)

    xs, _, _, _ = pl.pallas_call(
        partial(_kernel, nal, barrier),
        out_shape=(out((nline + 1, nalp, n)), out((nalp, n)), out((nalp, n)), out((nalp, n))),
        interpret=interpret, name="lsor_sweep", **kw)(*args)
    return xs[1:, :nal, :lanes]
