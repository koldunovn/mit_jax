"""Implicit vertical advection + diffusion of a tracer: GAD_IMPLICIT_R and SOLVE_PENTADIAGONAL (plan Task 16b).

Literal translation of c66g `pkg/generic_advdiff/gad_implicit_r.F`, `gad_u3c4_impl_r.F` and
`model/src/solve_pentadiagonal.F` as compiled in the V4r4 ff build: SOLVE_DIAGONAL_LOWMEMORY undef,
SOLVE_DIAGONAL_KINNER defined (ff `code/CPP_OPTIONS.h:100-102`), TARGET_NEC_SX undef. V4r4 uses implicitDiffusion = T
and implicit vertical advection with the 3rd-order upwind scheme (tempVertAdvScheme = saltVertAdvScheme = 3), so the
matrix is pentadiagonal (gad_implicit_r.F:234-242); other schemes raise NotImplementedError.

The matrix is set on the tile interior (gad_implicit_r.F:78-79 iMin..iMax = 1..sNx, jMin..jMax = 1..sNy); halo
columns keep the initial identity rows (a=b=d=e=0, c=1, gad_implicit_r.F:108-118). SOLVE_DIAGONAL_KINNER sweeps every
column including halos (solve_pentadiagonal.F:213-293), which leaves halo values unchanged (identity rows).

AD: the direct solve is two `lax.scan` recurrences over k (forward elimination, back substitution), differentiated by
JAX as written; there is no iteration count. The division 1/c' is guarded (tmpVar .NE. 0 branch, solve_pentadiagonal.F
:256-267) so a zero pivot gives finite values and a zero cotangent, as the Fortran errCode path gives zeros.
"""

import jax
import jax.numpy as jnp
import numpy as np

# GAD.h
ENUM_UPWIND_3RD = 3
ENUM_CENTERED_4TH = 4
ENUM_DST3 = 30
ONE_SIXTH = 1.0 / 6.0  # GAD.h:100 oneSixth = 1.D0/6.D0
DEEPFAC = 1.0  # set_grid_factors.F:52-61 (deepAtmosphere = F)
RHOFAC = 1.0  # set_ref_state.F:75-80 (not anelastic)


def _col(v):
    return jnp.asarray(v, dtype=jnp.float64)[None, :, None, None]


def gad_implicit_r_matrix(p, g, implicitAdvection, advectionScheme, kappaRX, recip_hFac, wFld):
    """gad_implicit_r.F:104-259: the five diagonals a5d..e5d, each [T, Nr, j, i]."""
    L = g.layout
    Nr = kappaRX.shape[1]
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)  # gad_implicit_r.F:78-79
    dt = _col(p.dTtracerLev)
    maskC = g.maskC[..., J, I]
    rh = recip_hFac[..., J, I]
    kap = kappaRX[..., J, I]
    rdrF = _col(g.recip_drF)
    rdrC = jnp.asarray(g.recip_drC)
    shp = rh.shape
    zero = jnp.zeros(shp, rh.dtype)
    # gad_implicit_r.F:108-118
    a = zero
    b = zero
    c = jnp.ones(shp, rh.dtype)
    d = zero
    e = zero
    if not p.implicitDiffusion:
        raise NotImplementedError("GAD_IMPLICIT_R without implicitDiffusion: the V4r4 run always has it")
    # gad_implicit_r.F:125-135 (k = 2..Nr): 1rst lower diagonal
    bk = -(dt[:, 1:] * maskC[:, :-1] * rh[:, 1:] * rdrF[:, 1:] * DEEPFAC * RHOFAC
           * kap[:, 1:] * rdrC[1:Nr][None, :, None, None] * DEEPFAC * RHOFAC)
    b = b.at[:, 1:].set(bk)
    # gad_implicit_r.F:137-147 (k = 1..Nr-1): 1rst upper diagonal
    dk = -(dt[:, :-1] * maskC[:, 1:] * rh[:, :-1] * rdrF[:, :-1] * DEEPFAC * RHOFAC
           * kap[:, 1:] * rdrC[1:Nr][None, :, None, None] * DEEPFAC * RHOFAC)
    d = d.at[:, :-1].set(dk)
    # gad_implicit_r.F:149-157: main diagonal
    c = 1.0 - (b + d)

    if implicitAdvection:
        if advectionScheme != ENUM_UPWIND_3RD:
            # gad_implicit_r.F:214-251: only the U3 branch of GAD_U3C4_IMPL_R is ported
            raise NotImplementedError(f"GAD_IMPLICIT_R: vertical scheme {advectionScheme} not ported (V4r4: 3)")
        # gad_implicit_r.F:186-256, k = Nr..2 (k = 1 has no call: gad_implicit_r.F:211)
        # rTrans (gad_implicit_r.F:196-202) of interface k = 2..Nr (index 0 <-> k = 2)
        rA = g.rA[:, None, J, I]
        rTrans = wFld[:, 1:][..., J, I] * rA * DEEPFAC * RHOFAC * maskC[:, :-1]
        # gad_u3c4_impl_r.F:86-155 (not TARGET_NEC_SX), interfaces k = 2..Nr
        ks = np.arange(2, Nr + 1)
        km2 = np.maximum(1, ks - 2) - 1
        kp1 = np.minimum(Nr, ks + 1) - 1
        maskP1 = np.where(ks >= Nr, 0.0, 1.0)  # gad_u3c4_impl_r.F:90,93
        maskM2 = np.where(ks <= 2, 0.0, 1.0)  # gad_u3c4_impl_r.F:91-92
        flagC4 = (advectionScheme == ENUM_CENTERED_4TH)  # False for U3 (gad_u3c4_impl_r.F:94-95)
        assert not flagC4
        rCenter = 0.5 * rTrans * g.recip_rA[:, None, J, I] * p.rkSign  # :135
        mskM = maskC[:, km2] * _col(maskM2)  # :136
        mskP = maskC[:, kp1] * _col(maskP1)  # :137
        # :151-154 (advectionScheme .NE. ENUM_DST3, not flagC4)
        rUpwind = 2.0 * ONE_SIXTH * jnp.abs(rCenter)
        rC4km = ONE_SIXTH * (rCenter + jnp.abs(rCenter)) * mskM
        rC4kp = ONE_SIXTH * (rCenter - jnp.abs(rCenter)) * mskP

        def sc(x, lev):  # x * deltaTarg(lev) * recip_hFac(lev) * recip_drF(lev) * recip_deepFac2C * recip_rhoFacC
            return x * dt[:, lev] * rh[:, lev] * rdrF[:, lev] * DEEPFAC * RHOFAC

        up = slice(1, Nr)   # levels k   = 2..Nr   (interface k updates level k)
        lo = slice(0, Nr - 1)  # levels k-1 = 1..Nr-1 (interface k updates level k-1)
        # The Fortran loop runs k = Nr..2: level m first receives the (k-1)-updates of interface m+1 (k = m+1), then
        # the k-updates of interface m. Updates of one kind for all interfaces are independent, so: first all
        # (k-1)-updates (gad_u3c4_impl_r.F:177-196), then all k-updates (:157-176), each in Fortran expression order.
        # Level-m values after interface m+1's step (levels 1..Nr-1):
        b = b.at[:, lo].set(b[:, lo] - sc(rC4km, lo))                                   # :177-181
        c = c.at[:, lo].set(c[:, lo] + sc((rCenter + rUpwind) + rC4km, lo))              # :182-186
        d = d.at[:, lo].set(d[:, lo] + sc((rCenter - rUpwind) + rC4kp, lo))              # :187-191
        e = e.at[:, lo].set(e[:, lo] - sc(rC4kp, lo))                                    # :192-196
        # then interface m's own step (levels 2..Nr)
        a = a.at[:, up].set(a[:, up] + sc(rC4km, up))                                    # :157-161
        b = b.at[:, up].set(b[:, up] - sc((rCenter + rUpwind) + rC4km, up))              # :162-166
        c = c.at[:, up].set(c[:, up] - sc((rCenter - rUpwind) + rC4kp, up))              # :167-171
        d = d.at[:, up].set(d[:, up] + sc(rC4kp, up))                                    # :172-176

    def full(x, init):
        return jnp.full(recip_hFac.shape, init, recip_hFac.dtype).at[..., J, I].set(x)

    return full(a, 0.0), full(b, 0.0), full(c, 1.0), full(d, 0.0), full(e, 0.0)


def solve_pentadiagonal(a5d, b5d, c5d, d5d, e5d, y5d):
    """solve_pentadiagonal.F:198-293 (SOLVE_DIAGONAL_KINNER, not LOWMEMORY) on every column (k is axis 1)."""
    Nr = y5d.shape[1]
    a, b, c, d, e, y = (jnp.moveaxis(x, 1, 0) for x in (a5d, b5d, c5d, d5d, e5d, y5d))

    def norm(cp, dp, ep, yp):
        # solve_pentadiagonal.F:256-267
        nz = cp != 0.0
        recVar = 1.0 / jnp.where(nz, cp, 1.0)
        return (jnp.where(nz, dp * recVar, 0.0), jnp.where(nz, ep * recVar, 0.0), jnp.where(nz, yp * recVar, 0.0))

    # k = 1: solve_pentadiagonal.F:227-232 (copy terms)
    d1, e1, y1 = norm(c[0], d[0], e[0], y[0])
    # k = 2: solve_pentadiagonal.F:233-241 (subtract one term)
    c2 = c[1] - b[1] * d1
    d2 = d[1] - b[1] * e1
    y2 = y[1] - b[1] * y1
    d2, e2, y2 = norm(c2, d2, e[1], y2)

    def fwd(carry, xs):
        dm2, em2, ym2, dm1, em1, ym1 = carry
        ak, bk, ck, dk, ek, yk = xs
        # solve_pentadiagonal.F:242-253 (subtract two terms)
        cp = ck - ak * em2 - (bk - ak * dm2) * dm1
        dp = dk - (bk - ak * dm2) * em1
        ep = ek
        yp = yk - ak * ym2 - (bk - ak * dm2) * ym1
        dp, ep, yp = norm(cp, dp, ep, yp)
        return (dm1, em1, ym1, dp, ep, yp), (dp, ep, yp)

    _, (dr, er, yr) = jax.lax.scan(fwd, (d1, e1, y1, d2, e2, y2), (a[2:], b[2:], c[2:], d[2:], e[2:], y[2:]))
    dpr = jnp.concatenate([d1[None], d2[None], dr])
    epr = jnp.concatenate([e1[None], e2[None], er])
    ypr = jnp.concatenate([y1[None], y2[None], yr])

    # backward sweep solve_pentadiagonal.F:273-284
    uN = ypr[Nr - 1]
    uN1 = ypr[Nr - 2] - uN * dpr[Nr - 2]

    def bwd(carry, xs):
        up1, up2 = carry  # y5d_update(k+1), y5d_update(k+2)
        yp, dp, ep = xs
        u = yp - up1 * dp - up2 * ep
        return (u, up1), u

    _, ur = jax.lax.scan(bwd, (uN1, uN), (ypr[:Nr - 2], dpr[:Nr - 2], epr[:Nr - 2]), reverse=True)
    out = jnp.concatenate([ur, uN1[None], uN[None]])
    return jnp.moveaxis(out, 0, 1)


def gad_implicit_r(p, g, implicitAdvection, advectionScheme, kappaRX, recip_hFac, wFld, gTracer):
    """GAD_IMPLICIT_R (gad_implicit_r.F:104-285): gTracer (T + dt*gT, all levels) -> tracer after the implicit
    vertical advection + diffusion solve. Diagnostics (:287-446) are off."""
    if gTracer.shape[1] <= 1:  # gad_implicit_r.F:105
        return gTracer
    a, b, c, d, e = gad_implicit_r_matrix(p, g, implicitAdvection, advectionScheme, kappaRX, recip_hFac, wFld)
    # gad_implicit_r.F:261-285: diagonalNumber = 5 (U3 implicit advection)
    return solve_pentadiagonal(a, b, c, d, e, gTracer)
