"""Two-dimensional conjugate-gradient solver CG2D (plan Task 15) and its adjoint seam.

Literal port of c66g `model/src/cg2d.F` (the solver SOLVE_FOR_PRESSURE calls in V4r4: DISCONNECTED_TILES,
ALLOW_CG2D_NSA and ALLOW_SRCG are all undefined, solve_for_pressure.F:275-311) and of the solver normalisation of
`ini_cg2d.F:90-135`. The preconditioned CG iterates exactly as the Fortran does: RHS normalised by its global max
(cg2dNormaliseRHS, cg2dTargetResWunit <= 0), same preconditioner (pW, pS, pC from UPDATE_CG2D), same order of the
updates, same stopping rule `err_sq < cg2dTolerance**2` tested after every iteration, `cg2dUseMinResSol=0`
(nIterMin = -1: no min-residual solution is kept).

Global sums follow `global_sum_tile.F` (serial and GLOBAL_SUM_ORDER_TILES paths: tile partial sums added in fixed tile
order starting from 0). The per-tile partial sums are the Fortran `errTile(bi,bj) = errTile(bi,bj) + r*r` loops over
j (outer) and i (inner): `sum_order="fortran"` reproduces that sequential order exactly (a `lax.scan` over j with the
i-chain unrolled); `sum_order="tile"` uses an XLA tree reduction inside each tile (same tile order across tiles).
Measured on both oracles (5 solves, 158-179 iterations): "fortran" gives cg2d_x bitwise equal to the Fortran and the
same iteration counts (needs the parameters traced and XLA_FLAGS --xla_cpu_max_isa=AVX
--xla_disable_hlo_passes=algsimp, as conftest.py sets); "tile" gives the same counts and cg2d_x within 4e-9 relative.
CPU cost per solve (164 iterations, 12 cores): 0.146 s "fortran", 0.110 s "tile".

Arrays: the Fortran work arrays cg2d_r, cg2d_s, cg2d_q are (0:sNx+1, 0:sNy+1) (CG2D.h:53-59); here every field is a
full `[tile, j, i]` array with OLx=OLy=4 halos and only the points the Fortran touches are read. EXCH_S3D_RL (width-1
exchange ignoring corners, exch2_s3d_rl.F) is the scalar exch2 map restricted to the points the 5-point stencil
reads (i=0, sNx+1 for j=1..sNy and j=0, sNy+1 for i=1..sNx): the full-width exch2 map writes the same values there.
Halo points no exchange writes (open facet edges) keep 0, as the common-block arrays do (ini_cg2d.F:65-78).

Differentiation (CLAUDE.md: never through solver iterations): `cg2d_solve` wraps the literal forward iteration in
`jax.lax.custom_linear_solve(symmetric=True)`. The unknown is the interior of the tile arrays (halo lanes are 0 in
the solution and ignored in the operator), so the operator is the symmetric 5-point matrix. The transpose solve is
the same preconditioned CG from a zero first guess to `adj_tolerance` (normalised residual). The first guess enters
only through `solve`'s closure (stop_gradient), so it carries no derivative. `stop_coeff_grad=True` (ECCO/TAF
semantics of pkg/autodiff/cg2d.flow: only cg2d_b and cg2d_x are active) stops the derivative with respect to the
operator coefficients aW2d, aS2d, aC2d; the forward is unchanged.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitgcm_jax.params_io import params_pytree


@params_pytree
@dataclass(frozen=True)
class Cg2dParams:
    """CG2D parameters. Floats are traced pytree leaves: pass the object as a jit argument (params_io.params_pytree:
    with cg2dNorm a compile-time constant XLA reassociates `(cg2d_b*cg2dNorm)*rhsNorm` and the iterate is no longer
    the Fortran one; traced, the solve is bitwise equal to the oracle)."""
    cg2dMaxIters: int        # data PARM02 cg2dMaxIters (set_defaults.F:280 default 150); V4r4 300
    cg2dTolerance: float     # ini_cg2d.F:165 cg2dTolerance = cg2dTargetResidual (set_defaults.F:281 1.e-7)
    cg2dNorm: float          # ini_cg2d.F:160 cg2dNorm = myNorm (see ini_cg2d_norm)
    cg2dpcOffDFac: float = 0.51  # set_defaults.F:285 (UPDATE_CG2D preconditioner, update_cg2d.F:181, 188)
    nIterMin: int = -1       # solve_for_pressure.F:273 nIterMin = cg2dUseMinResSol - 1 (ini_parms.F:1451-1455: 0)
    sum_order: str = "fortran"   # "fortran" (sequential j/i per tile, as cg2d.F) or "tile" (XLA tree per tile)
    adj_tolerance: float = 1e-13  # transpose (adjoint) solve: normalised residual target
    adj_max_iters: int = 2000
    stop_coeff_grad: bool = False  # ECCO semantics (cg2d.flow ACTIVE = cg2d_b, cg2d_x only)

    @classmethod
    def from_namelists(cls, nml, cg2dNorm, **kw):
        g = lambda k, d: nml.get("data", "parm02", k, default=d)  # noqa: E731
        maxit = int(g("cg2dMaxIters", 150))                     # set_defaults.F:280
        target = float(g("cg2dTargetResidual", 1.0e-7))         # set_defaults.F:281
        wunit = float(g("cg2dTargetResWunit", -1.0))            # set_defaults.F:282
        if not wunit <= 0.0:
            # ini_cg2d.F:162-170: cg2dNormaliseRHS = cg2dTargetResWunit.LE.0 ; the W-unit tolerance branch is not ported
            raise NotImplementedError("cg2dTargetResWunit > 0 (cg2dNormaliseRHS=F) is not ported")
        if nml.has("data", "parm02", "cg2dUseMinResSol"):
            minres = int(nml.get("data", "parm02", "cg2dUseMinResSol"))
        else:
            # ini_parms.F:1451-1455: unset -> 0, or 1 if topoFile = bathyFile = ' ' and usingCartesianGrid
            blank = lambda k: str(nml.get("data", "parm05", k, default=" ")).strip() == ""  # noqa: E731
            cart = bool(nml.get("data", "parm04", "usingCartesianGrid", default=False))
            minres = 1 if (blank("topoFile") and blank("bathyFile") and cart) else 0
        if int(minres) != 0:
            raise NotImplementedError("cg2dUseMinResSol=1 (min-residual solution, cg2d.F:347-378) is not ported")
        if int(g("cg2dPreCondFreq", 1)) != 1:                    # set_defaults.F:286
            # update_cg2d.F:60-64: with cg2dPreCondFreq=1 the preconditioner is rebuilt every step; the other
            # frequencies (keep the ini_cg2d.F / previous preconditioner) are not ported
            raise NotImplementedError("cg2dPreCondFreq != 1 is not ported")
        return cls(cg2dMaxIters=maxit, cg2dTolerance=target, cg2dNorm=float(cg2dNorm),
                   cg2dpcOffDFac=float(g("cg2dpcOffDFac", 0.51)),  # set_defaults.F:285
                   nIterMin=int(minres) - 1, **kw)


def _interior(L):
    return L.js(1, L.sNy), L.is_(1, L.sNx)


def interior_mask(L):
    """[ny, nx] bool: Fortran interior points i=1..sNx, j=1..sNy."""
    m = np.zeros((L.ny, L.nx), bool)
    J, I = _interior(L)
    m[J, I] = True
    return m


def ini_cg2d_norm(g, hFacW, hFacS, implicSurfPress, implicDiv2DFlow):
    """cg2dNorm = 1/max|aW2d, aS2d| of the initial operator (ini_cg2d.F:90-135), float.

    Called with the hFac of INITIALISE_FIXED (h0FacW/S: INI_CG2D runs in initialise_fixed.F:249, before any r* update).
    """
    L = g.layout
    J, I = _interior(L)
    return float(_ini_cg2d_norm(L, jnp.asarray(g.dyG[:, J, I]), jnp.asarray(g.dxG[:, J, I]),
                                jnp.asarray(g.recip_dxC[:, J, I]), jnp.asarray(g.recip_dyC[:, J, I]),
                                jnp.asarray(g.drF), jnp.asarray(hFacW[:, :, J, I]), jnp.asarray(hFacS[:, :, J, I]),
                                jnp.float64(implicSurfPress), jnp.float64(implicDiv2DFlow)))


@partial(jax.jit, static_argnums=0)
def _ini_cg2d_norm(L, dyG, dxG, recip_dxC, recip_dyC, drF, hFacW, hFacS, implicSurfPress, implicDiv2DFlow):
    drF = drF[None, :, None, None]
    fac = implicSurfPress * implicDiv2DFlow
    tW = fac * (dyG[:, None] * drF * hFacW) * recip_dxC[:, None]     # ini_cg2d.F:103-107
    tS = fac * (dxG[:, None] * drF * hFacS) * recip_dyC[:, None]     # ini_cg2d.F:108-112

    def acc(a, t):                                                   # ini_cg2d.F:93-115, k = 1..Nr in order
        return a + t, None
    aW, _ = lax.scan(acc, jnp.zeros(dyG.shape), jnp.moveaxis(tW, 1, 0))
    aS, _ = lax.scan(acc, jnp.zeros(dyG.shape), jnp.moveaxis(tS, 1, 0))
    myNorm = jnp.maximum(jnp.max(jnp.abs(aW)), jnp.max(jnp.abs(aS)))  # ini_cg2d.F:124-130 (MAX: order-free)
    nz = myNorm != 0.0
    return jnp.where(nz, 1.0 / jnp.where(nz, myNorm, 1.0), 1.0)      # ini_cg2d.F:131-135


# ---------------------------------------------------------------------------------------------------------------
# sums


def tile_sum(a, order):
    """Per-tile partial sums of an interior field a [T, sNy, sNx] -> [T].

    "fortran": `s = 0; DO j; DO i; s = s + a(i,j)` (cg2d.F:164-186 and friends), strictly sequential.
    "tile": XLA reduction inside each tile (different rounding order).
    """
    if order == "tile":
        return jnp.sum(a, axis=(1, 2))
    if order != "fortran":
        raise ValueError(order)
    nI = a.shape[2]

    def row(s, arow):  # arow [T, sNx]: i-chain in Fortran order
        for i in range(nI):
            s = s + arow[:, i]
        return s, None

    s, _ = lax.scan(row, jnp.zeros(a.shape[0], a.dtype), jnp.moveaxis(a, 1, 0))
    return s


def global_sum_tile(phiTile):
    """GLOBAL_SUM_TILE_RL (global_sum_tile.F, serial / GLOBAL_SUM_ORDER_TILES): 0 + tile 1 + tile 2 + ... in order."""
    s = jnp.zeros((), phiTile.dtype)
    for t in range(phiTile.shape[0]):
        s = s + phiTile[t]
    return s


def global_sum(a_interior, order):
    return global_sum_tile(tile_sum(a_interior, order))


# ---------------------------------------------------------------------------------------------------------------
# stencils (interior results [T, sNy, sNx])


def _slices(L):
    J, I = _interior(L)
    return J, I, L.js(0, L.sNy - 1), L.js(2, L.sNy + 1), L.is_(0, L.sNx - 1), L.is_(2, L.sNx + 1)


def apply_operator(L, aW2d, aS2d, aC2d, x):
    """aW2d(i)x(i-1) + aW2d(i+1)x(i+1) + aS2d(j)x(j-1) + aS2d(j+1)x(j+1) + aC2d x (cg2d.F:172-176, 286-290)."""
    J, I, Jm, Jp, Im, Ip = _slices(L)
    return (aW2d[:, J, I] * x[:, J, Im]
            + aW2d[:, J, Ip] * x[:, J, Ip]
            + aS2d[:, J, I] * x[:, Jm, I]
            + aS2d[:, Jp, I] * x[:, Jp, I]
            + aC2d[:, J, I] * x[:, J, I])


def apply_preconditioner(L, pW, pS, pC, r):
    """pC r + pW(i) r(i-1) + pW(i+1) r(i+1) + pS(j) r(j-1) + pS(j+1) r(j+1) (cg2d.F:228-233)."""
    J, I, Jm, Jp, Im, Ip = _slices(L)
    return (pC[:, J, I] * r[:, J, I]
            + pW[:, J, I] * r[:, J, Im]
            + pW[:, J, Ip] * r[:, J, Ip]
            + pS[:, J, I] * r[:, Jm, I]
            + pS[:, Jp, I] * r[:, Jp, I])


# ---------------------------------------------------------------------------------------------------------------
# the literal solver


def cg2d_fortran(ex, ops, cg2d_b, cg2d_x, *, cg2dNorm, tolerance, max_iters, sum_order, nIterMin=-1):
    """CG2D (cg2d.F:110-395). ops = (aW2d, aS2d, aC2d, pW, pS, pC); cg2d_b, cg2d_x full [T, ny, nx].

    Returns (cg2d_x, diag): cg2d_x exactly as the Fortran leaves it (interior = solution; halos = the exchanged
    normalised first guess, cg2d.F:147, which SOLVE_FOR_PRESSURE's EXCH_XY_RL overwrites), and diag with
    firstResidual, lastResidual, numIters, sumRHS, rhsMax (the values cg2d.F prints / returns).
    """
    if nIterMin >= 0:
        raise NotImplementedError("nIterMin >= 0 (min-residual solution) is not ported")
    L = ex.L
    aW2d, aS2d, aC2d, pW, pS, pC = (jnp.asarray(a) for a in ops)
    cg2d_b, cg2d_x = jnp.asarray(cg2d_b), jnp.asarray(cg2d_x)
    J, I = _interior(L)
    gsum = lambda a: global_sum(a, sum_order)  # noqa: E731
    cg2dTolerance_sq = tolerance * tolerance                        # cg2d.F:111
    eta_qrNM1 = jnp.ones((), cg2d_b.dtype)                           # cg2d.F:113

    b = cg2d_b[:, J, I] * cg2dNorm                                   # cg2d.F:121
    rhsMax = jnp.max(jnp.abs(b))                                     # cg2d.F:122, 130 (_GLOBAL_MAX_RL)
    nz = rhsMax != 0.0
    rhsNorm = jnp.where(nz, 1.0 / jnp.where(nz, rhsMax, 1.0), 1.0)   # cg2d.F:131-132
    b = b * rhsNorm                                                  # cg2d.F:137
    x = cg2d_x.at[:, J, I].set(cg2d_x[:, J, I] * rhsNorm)            # cg2d.F:138
    x = ex.exch_xy(x)                                                # cg2d.F:147 EXCH_XY_RL

    zero = jnp.zeros_like(cg2d_b)
    s = zero                                                         # cg2d.F:159-163 (whole 0:sNx+1 array)
    r_i = b - apply_operator(L, aW2d, aS2d, aC2d, x)                 # cg2d.F:171-177
    err_sq = gsum(r_i * r_i)                                         # cg2d.F:181-182, 194
    sumRHS = gsum(b)                                                 # cg2d.F:183, 195
    r = ex.exch_xy(zero.at[:, J, I].set(r_i))                        # cg2d.F:189 EXCH_S3D_RL
    firstResidual = jnp.sqrt(err_sq)                                 # cg2d.F:198

    def cond(st):
        it, x, r, s, eta_qrNM1, err_sq = st
        # cg2d.F:213 (before the loop) and :346 (after each iteration): stop when err_sq < tol^2; DO it2d=1,numIters
        return (it < max_iters) & jnp.logical_not(err_sq < cg2dTolerance_sq)

    def body(st):
        it, x, r, s, eta_qrNM1, err_sq = st
        q = apply_preconditioner(L, pW, pS, pC, r)                   # cg2d.F:228-233
        eta_qrN = gsum(q * r[:, J, I])                               # cg2d.F:241-242, 252
        cgBeta = eta_qrN / eta_qrNM1                                 # cg2d.F:254
        eta_qrNM1 = eta_qrN                                          # cg2d.F:259
        s_i = q + cgBeta * s[:, J, I]                                # cg2d.F:265-266
        s = ex.exch_xy(s.at[:, J, I].set(s_i))                       # cg2d.F:273 EXCH_S3D_RL
        q = apply_operator(L, aW2d, aS2d, aC2d, s)                   # cg2d.F:285-290
        alpha = gsum(s_i * q)                                        # cg2d.F:294-295, 304
        alpha = eta_qrN / alpha                                      # cg2d.F:310
        x_i = x[:, J, I] + alpha * s_i                               # cg2d.F:319
        r_i = r[:, J, I] - alpha * q                                 # cg2d.F:320
        err_sq = gsum(r_i * r_i)                                     # cg2d.F:324-325, 336
        x = x.at[:, J, I].set(x_i)
        # cg2d.F:362 EXCH_S3D_RL(cg2d_r) runs only when not converged; after convergence r is dead, so the
        # unconditional exchange changes no output
        r = ex.exch_xy(r.at[:, J, I].set(r_i))
        return it + 1, x, r, s, eta_qrNM1, err_sq                    # cg2d.F:331 actualIts = it2d

    it, x, r, s, eta_qrNM1, err_sq = lax.while_loop(
        cond, body, (jnp.zeros((), jnp.int32), x, r, s, eta_qrNM1, err_sq))

    x = x.at[:, J, I].set(x[:, J, I] / rhsNorm)                      # cg2d.F:380-391 un-normalise
    diag = dict(firstResidual=firstResidual, lastResidual=jnp.sqrt(err_sq),  # cg2d.F:394
                numIters=it, sumRHS=sumRHS, rhsMax=rhsMax)            # cg2d.F:395
    return x, diag


# ---------------------------------------------------------------------------------------------------------------
# differentiable wrapper


def cg2d_solve(p: Cg2dParams, ex, ops, cg2d_b, cg2d_x):
    """Solve A x = b with the literal CG2D forward and implicit (custom_linear_solve) derivatives.

    Returns (x, diag): x [T, ny, nx] with the Fortran solution on the interior and 0 on halo lanes (the linear-solve
    unknown); diag = cg2d_fortran's diagnostics plus `x_fortran`, the full array exactly as CG2D returns it (halos
    included; for the C02 gate, carries no derivative).
    """
    L = ex.L
    inner = jnp.asarray(interior_mask(L))
    aW2d, aS2d, aC2d, pW, pS, pC = ops
    if p.stop_coeff_grad:
        aW2d, aS2d, aC2d = (lax.stop_gradient(a) for a in (aW2d, aS2d, aC2d))
    x_first = lax.stop_gradient(cg2d_x)
    J, I = _interior(L)

    def matvec(v):
        # CG2D solves (cg2dNorm-scaled operator) x = cg2dNorm * cg2d_b (cg2d.F:121), i.e. (A/cg2dNorm) x = cg2d_b
        xe = ex.exch_xy(jnp.where(inner, v, 0.0))
        return jnp.zeros_like(v).at[:, J, I].set(apply_operator(L, aW2d, aS2d, aC2d, xe) / p.cg2dNorm)

    def solve(_, b):
        x, diag = cg2d_fortran(ex, (aW2d, aS2d, aC2d, pW, pS, pC), b, x_first, cg2dNorm=p.cg2dNorm,
                               tolerance=p.cg2dTolerance, max_iters=p.cg2dMaxIters, sum_order=p.sum_order,
                               nIterMin=p.nIterMin)
        diag = dict(diag, x_fortran=x)
        return jnp.where(inner, x, 0.0), diag

    def transpose_solve(_, b):
        x, diag = cg2d_fortran(ex, (aW2d, aS2d, aC2d, pW, pS, pC), jnp.where(inner, b, 0.0), jnp.zeros_like(b),
                               cg2dNorm=p.cg2dNorm, tolerance=p.adj_tolerance, max_iters=p.adj_max_iters,
                               sum_order="tile")
        diag = dict(diag, x_fortran=x)
        return jnp.where(inner, x, 0.0), diag

    return lax.custom_linear_solve(matvec, cg2d_b, solve, transpose_solve, symmetric=True, has_aux=True)
