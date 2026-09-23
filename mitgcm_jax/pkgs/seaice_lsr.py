"""pkg/seaice: SEAICE_LSR (Picard solver with line successive over-relaxation, C-grid) as ECCO v4r4 runs it (M2.4).

Literal port of c66g `pkg/seaice/seaice_lsr.F` (SEAICE_LSR, SEAICE_LSR_CALC_COEFFS, SEAICE_LSR_RHSU/V,
SEAICE_LSR_TRIDIAGU/V), `seaice_calc_strainrates.F`, `seaice_calc_viscosities.F`, `seaice_oceandrag_coeffs.F`; no V4r4
override. CPP branches from `ECCOv4 Release 4/code/SEAICE_OPTIONS.h` + CPP_OPTIONS.h, confirmed in the preprocessed
build (reference/build/full_serial13_jaxdump_*/bld/seaice_lsr.f): SEAICE_CGRID, SEAICE_ALLOW_DYNAMICS,
SEAICE_ALLOW_FREEDRIFT, SEAICE_ALLOW_EVP, ALLOW_AUTODIFF_TAMC defined; SEAICE_LSR_ZEBRA, SEAICE_VECTORIZE_LSR,
SEAICE_GLOBAL_3DIAG_SOLVER, SEAICE_ALLOW_BOTTOMDRAG, SEAICE_LSR_ADJOINT_ITER, SEAICE_ALLOW_CHECK_LSR_CONVERGENCE,
SEAICE_ZETA_SMOOTHREG, SEAICE_ALLOW_TEM undefined. Runtime branches (data.seaice + seaice_readparms.F defaults, STDOUT
of the oracle): SEAICE_OLx = SEAICE_OLy = 0 (iMin=jMin=1, iMax=sNx, jMax=sNy), SEAICEnonLinIterMax = 2 (two Picard
passes, MPSEUDOTIMESTEPS=2), SEAICElinearIterMax = 1500, SOLV_NCHECK = 2, LSR_ERROR = 2e-4, relaxation 0.95,
useCubedSphereExchange, SEAICE_no_slip, SEAICEetaZmethod=0, no BDF2 / strong implicit coupling / surface-stress scaling
/ HB87 coupling, LSR_mixIniGuess = 0 (SEAICE_RESIDUAL only prints: debugLevel=1 and SEAICE_monFreq=deltaTClock make
printResidual true, but its outputs are overwritten or unused). Any other value raises NotImplementedError in
`seaice_dyn.SeaiceDynParams.from_namelists`.

Layout: `[tile, j, i]` with halos (layout.py). LSR is tile-local: every sweep solves tridiagonal systems along lines
inside each tile (SEAICE_OLx=0), neighbours across tile edges enter through the halos of the last exchange; the result
therefore depends on the tile size (13 tiles of 90x90 here, as the oracle), not on the number of devices. Measured:
inside shard_map on 4 devices (tiles in blocks, padding = replicas of tile 1) uIce, vIce and the sweep counts are
bitwise those of P = 1 (global max via Exchanger.global_max; lane counts from the local tile axis).

Gates (mitgcm_jax/tests/test_seaice_dyn.py, oracle full_jaxdump_v5, iterations 1-3): every LSR stage (L01-L04 of both
Picard passes, Y05) bitwise at every point, LSOR sweeps 178/118, 112/82, 84/58 as in the Fortran.

Implementations (SeaiceDynParams.lsr_impl, static; all perform the same operations in the same order and are bitwise
equal to each other and to the Fortran on CPU, A100 and GH200: whole dynsolver, iterations 1-3, sweep counts 178/118,
112/82, 84/58; scripts/runs/lsr_perf_bench.py --full). Each sweep is 90 sequential lines x (89 + 89) sequential Thomas
steps, ~16k sequential scan steps: on a GPU every XLA scan step costs at least one kernel launch.
  - "xla": lax.scan (unroll XLA_UNROLL = 1). CPU 16 cores: ~3.5 ms/sweep, whole dynsolver 1.2-1.6 s at iteration 1
    (296 sweeps; shared node). GPU: whole dynsolver 53 / 35 / 26 s per step (A100-40) and 27.6 / 18.1 / 13.2 s
    (GH200) at iterations 1 / 2 / 3 (180 / 93 ms per sweep). XLA_UNROLL = 2 is bitwise too and measured 0-12% faster
    on CPU, within the noise of the shared node: not changed.
  - "xla_unrolled": the Thomas scans fully unrolled (lax.scan unroll=True): one XLA loop over the 90 lines. Whole
    dynsolver 2.87 / 1.89 / 1.39 s (A100-40), 1.70 / 1.12 / 0.82 s (GH200); compile 2.5-4 s; CPU 6.5 ms/sweep (slower
    than "xla").
  - "pallas": the forward sweep as one Pallas-Triton kernel launch (seaice_lsr_pallas.py; CUDA). Whole dynsolver
    1.18 / 0.78 / 0.58 s (A100-40, 4.0 ms per sweep) and 0.78 / 0.51 / 0.38 s (GH200, 2.6 ms per sweep); first call
    (compile) 6.5 / 4.3 s.
  - "auto" (default): lax.platform_dependent at lowering: cpu -> "xla", cuda -> "pallas", other platforms (rocm, tpu)
    -> "xla_unrolled" (Pallas on ROCm untested: seaice_lsr_pallas.py).
The implicit derivative never uses the Pallas kernel: its preconditioner sweep (_precond, transposed by
jax.linear_transpose) is XLA code, with the Thomas scans fully unrolled off the CPU (_precond_impl). Tangent / gradient
of one LSOR solve (GMRES 40 x 8 cycles) on a GH200: 2.7 / 3.0 s unrolled, 31 / 31 s with lax.scan; A100-40
3.3 / 6.6 s unrolled; CPU 22 / 20 s.

LSOR sweep (SEAICE_LSR_TRIDIAGU/V, non-zebra, non-vectorised): for u the rows J=1..sNy are visited in order and row J
uses the NEW row J-1 (Gauss-Seidel by lines), the OLD row J+1 and the halo columns i=0, sNx+1; each row is a Thomas
solve along i followed by relaxation uIce = uTmp + WFAU*(URT - uTmp). For v the columns are visited in order. Here u
rows and (transposed) v columns form one [line, along, lane] problem (lanes = 13 u tiles + 13 v tiles), a `lax.scan`
over lines with two inner scans (forward elimination, back substitution) in the Fortran operation order. The
elimination factors bet and CUU do not depend on the iterate; they are computed once per Picard pass with the same
operations (hoisting, bitwise the same values).

Stopping rule (seaice_lsr.F:617-836): every SOLV_NCHECK sweeps S1 = global max |uIce-uTmp|*seaiceMaskU (interior,
_GLOBAL_MAX_RL), WFAU := 0 if S1 grew since the last check, u converged if S1 < LSR_ERROR; with the cubed-sphere
exchange both components keep iterating (:657-660) until both converge at the same check. Reproduced literally in a
`lax.while_loop` (the counts ICOUNT1/2 are a gate).

DESIGN DECISION (AD; docs/PORTING_RULES.md "never differentiate through solver iterations"): `lsor_solve` is a jax.custom_jvp.
  - Primal = the literal Fortran loop above (data-dependent count, bitwise with the oracle).
  - Tangent = implicit derivative of the linear system the LSOR iterates towards, A(c) x = b(c) (for the current
    Picard pass: A x = AU x(i-1) + BU x + CU x(i+1) - mask*(uRt1 x(j-1) + uRt2 x(j+1)) on the interior, halos of x from
    EXCH_UV_XY_RL, b = mask*rhsU; likewise for v), evaluated at the returned iterate x:
        dx = A^-1 (db - dA x),
    solved inside `lax.custom_linear_solve` by GMRES(lsr_ad_restart) with a FIXED number lsr_ad_cycles of restart
    cycles (no data-dependent stopping), left-preconditioned with P = omega (D - omega L)^-1 = one line-SOR sweep from
    zero, i.e. the Fortran sweep in operator form (x_{k+1} = SOR(x_k) is exactly x_k + P(b - A x_k)). The transpose
    solve (reverse mode) is the same GMRES on A^T with M = P^T, both from jax.linear_transpose. Inner products are tile
    partial sums in tile order (P-independent). The plain stationary iteration converges too slowly for this
    (measured at LLC90: ~415 sweeps per decade of residual). Nothing differentiates through the iterations; the first
    guess carries no derivative; the stopping rule, counts and relaxation switch carry none.
  - Consequence: the derivative is that of the exactly solved linear system. The Fortran iterate stops at
    |dU| < LSR_ERROR = 2e-4 m/s per 2 sweeps, far from the exact solution; FD of the literal forward is therefore not
    the implicit derivative. The FD gate runs the forward with a tight LSR_ERROR (same code path) where both agree.
  - Picard passes (2), coefficient computations, masking and clipping are ordinary differentiable JAX code.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from mitgcm_jax.core.cg2d import _join_static, _split_static, _zero_tangent

# SEAICE_PARAMS.h:562-565 (PARAMETER)
ZERO = 0.0
ONE = 1.0
TWO = 2.0
HALF = 0.5


def _m2(g, name):
    """Surface level (k=1) of a 3-D grid mask."""
    return g.f[name][:, 0]


# ---------------------------------------------------------------------------------------------------------------
# SEAICE_CALC_STRAINRATES (seaice_calc_strainrates.F, SEAICE_CGRID)


def calc_strainrates(p, g, sg, uFld, vFld, e11, e22, e12):
    """e11, e22 on j,i = 1-OL..sN+OL-1 (:83-107), e12 on 2-OL..sN+OL (:111-151); other points keep the values passed
    in (the common-block arrays keep them across calls)."""
    L = g.layout
    OLx, OLy, sNx, sNy = L.OLx, L.OLy, L.sNx, L.sNy
    maskW, maskS, maskC = _m2(g, "maskW"), _m2(g, "maskS"), _m2(g, "maskC")
    noSlipFac = 1.0 if p.SEAICE_no_slip else 0.0  # seaice_calc_strainrates.F:76-77
    # :83-88, :89-94  (dudx, uave, dvdy, vave at C points)
    J, I = L.js(1 - OLy, sNy + OLy - 1), L.is_(1 - OLx, sNx + OLx - 1)
    Jp, Ip = L.js(2 - OLy, sNy + OLy), L.is_(2 - OLx, sNx + OLx)
    dudx = g.recip_dxF[:, J, I] * (uFld[:, J, Ip] - uFld[:, J, I])  # :85-86
    uave = 0.5 * (uFld[:, J, I] + uFld[:, J, Ip])  # :87
    dvdy = g.recip_dyF[:, J, I] * (vFld[:, Jp, I] - vFld[:, J, I])  # :91-92
    vave = 0.5 * (vFld[:, J, I] + vFld[:, Jp, I])  # :93
    e11 = e11.at[:, J, I].set(dudx + vave * sg["k2AtC"][:, J, I])  # :99
    e22 = e22.at[:, J, I].set(dvdy + uave * sg["k1AtC"][:, J, I])  # :100
    # :111-124 (Z points), :126-151
    J, I = L.js(2 - OLy, sNy + OLy), L.is_(2 - OLx, sNx + OLx)
    Jm, Im = L.js(1 - OLy, sNy + OLy - 1), L.is_(1 - OLx, sNx + OLx - 1)
    dudy = (uFld[:, J, I] - uFld[:, Jm, I]) * g.recip_dyU[:, J, I]  # :113-114
    uave = 0.5 * (uFld[:, J, I] + uFld[:, Jm, I])  # :115
    dvdx = (vFld[:, J, I] - vFld[:, J, Im]) * g.recip_dxV[:, J, I]  # :120-121
    vave = 0.5 * (vFld[:, J, I] + vFld[:, J, Im])  # :122
    hFacU = maskW[:, J, I] - maskW[:, Jm, I]  # :128
    hFacV = maskS[:, J, I] - maskS[:, J, Im]  # :129
    t1 = 0.5 * (((dudy + dvdx) - sg["k1AtZ"][:, J, I] * vave) - sg["k2AtZ"][:, J, I] * uave)  # :130-134
    t1 = (((t1 * maskC[:, J, I]) * maskC[:, J, Im]) * maskC[:, Jm, I]) * maskC[:, Jm, Im]  # :135-136
    t2 = noSlipFac * (((2.0 * uave) * g.recip_dyU[:, J, I]) * hFacU
                      + ((2.0 * vave) * g.recip_dxV[:, J, I]) * hFacV)  # :137-140
    e12 = e12.at[:, J, I].set(t1 + t2)
    return e11, e22, e12


# ---------------------------------------------------------------------------------------------------------------
# SEAICE_CALC_VISCOSITIES (seaice_calc_viscosities.F, etaZmethod=0, no TEM / smooth regularisation / zeta clipping)


def calc_viscosities(p, g, sg, e11, e22, e12, zMin, zMax, hEffM, press0, tnsFac):
    """(eta, etaZ, zeta, zetaZ, press, deltaC). SEAICE_LSR zeroes them on every point before the Picard loop
    (seaice_lsr.F:178-195, ALLOW_AUTODIFF_TAMC); this routine writes j,i = 2-OL..sN+OL-1 only."""
    L = g.layout
    OLx, OLy, sNx, sNy = L.OLx, L.OLy, L.sNx, L.sNy
    maskC = _m2(g, "maskC")
    # :92-94 recip_e2 = ONE/(SEAICE_eccen**2) (eccen != 0 checked at setup; gfortran x**2 = x*x)
    recip_e2 = ONE / (p.SEAICE_eccen * p.SEAICE_eccen)
    J, I = L.js(2 - OLy, sNy + OLy - 1), L.is_(2 - OLx, sNx + OLx - 1)
    Jp, Ip = L.js(3 - OLy, sNy + OLy), L.is_(3 - OLx, sNx + OLx)
    Jm, Im = L.js(1 - OLy, sNy + OLy - 2), L.is_(1 - OLx, sNx + OLx - 2)
    # :108-116 (SEAICEetaZmethod = 0); 0.25 is a REAL*4 literal (exact)
    tmp = 0.25 * (((e12[:, J, I] + e12[:, J, Ip]) + e12[:, Jp, I]) + e12[:, Jp, Ip])
    e12Csq = tmp * tmp
    ep = e11[:, J, I] + e22[:, J, I]  # :137
    em = e11[:, J, I] - e22[:, J, I]  # :138
    deltaCsq = ep * ep + recip_e2 * (em * em + 4.0 * e12Csq)  # :139
    pos = deltaCsq > 0.0
    dC = jnp.where(pos, jnp.sqrt(jnp.where(pos, deltaCsq, 1.0)), 0.0)  # :147-149 (safe sqrt: finite derivative)
    deltaCreg = jnp.maximum(dC, p.SEAICE_deltaMin)  # :150
    t = tnsFac[:, J, I]
    z = (HALF * (press0[:, J, I] * (1.0 + t))) / deltaCreg  # :151-153
    z = jnp.minimum(zMax[:, J, I], z)  # :160 (no SEAICE_ZETA_SMOOTHREG)
    z = jnp.maximum(zMin[:, J, I], z)  # :161
    z = z * hEffM[:, J, I]  # :163
    e = recip_e2 * z  # :164
    pr = (press0[:, J, I] * (1.0 - p.SEAICEpressReplFac)
          + (((TWO * z) * dC) * p.SEAICEpressReplFac) / (1.0 + t)) * (1.0 - t)  # :166-170
    zero = jnp.zeros_like(e11)
    deltaC = zero.at[:, J, I].set(dC)
    zeta = zero.at[:, J, I].set(z)
    eta = zero.at[:, J, I].set(e)
    press = zero.at[:, J, I].set(pr)
    # :178-189 viscosities at Z points (simple average)
    sumNorm = ((maskC[:, J, I] + maskC[:, J, Im]) + maskC[:, Jm, I]) + maskC[:, Jm, Im]  # :180-181
    pos = sumNorm > 0.0
    sumNorm = jnp.where(pos, 1.0 / jnp.where(pos, sumNorm, 1.0), sumNorm)  # :182
    etaZ = zero.at[:, J, I].set(sumNorm * (((eta[:, J, I] + eta[:, J, Im]) + eta[:, Jm, I]) + eta[:, Jm, Im]))
    zetaZ = zero.at[:, J, I].set(sumNorm * (((zeta[:, J, I] + zeta[:, J, Im]) + zeta[:, Jm, I])
                                            + zeta[:, Jm, Im]))  # :183-188
    # :194-203 masking only with free slip (SEAICE_no_slip = T in V4r4: not executed)
    return eta, etaZ, zeta, zetaZ, press, deltaC


# ---------------------------------------------------------------------------------------------------------------
# SEAICE_OCEANDRAG_COEFFS (seaice_oceandrag_coeffs.F)


def oceandrag_coeffs(p, g, uIceLoc, vIceLoc, uVel, vVel, CwatC):
    """DWATN (CwatC) on j = 1-OLy..sNy+OLy-1, i = 1-OLx..sNx+OLx-1 (:77-110); other points keep the passed values.
    uVel, vVel: surface level (kSrf=1) [T, ny, nx]."""
    L = g.layout
    OLx, OLy, sNx, sNy = L.OLx, L.OLy, L.sNx, L.sNy
    J, I = L.js(1 - OLy, sNy + OLy - 1), L.is_(1 - OLx, sNx + OLx - 1)
    Jp, Ip = L.js(2 - OLy, sNy + OLy), L.is_(2 - OLx, sNx + OLx)
    mW, mS = g.maskInW, g.maskInS
    a = ((uIceLoc[:, J, I] - uVel[:, J, I]) * mW[:, J, I]
         + (uIceLoc[:, J, Ip] - uVel[:, J, Ip]) * mW[:, J, Ip])  # :80-83
    b = ((vIceLoc[:, J, I] - vVel[:, J, I]) * mS[:, J, I]
         + (vIceLoc[:, Jp, I] - vVel[:, Jp, I]) * mS[:, Jp, I])  # :84-87
    TEMPVAR = 0.25 * (a * a + b * b)  # :79-87 (x**2 = x*x)
    south = g.yC[:, J, I] < ZERO  # :88
    Cd = jnp.where(south, p.SEAICE_waterDrag_south, p.SEAICE_waterDrag)
    thr = (0.25 / Cd) * (0.25 / Cd)  # :91-92 / :99  ((0.25/Cd)**2)
    big = TEMPVAR > thr
    c = jnp.where(big, Cd * jnp.sqrt(jnp.where(big, TEMPVAR, 1.0)), 0.25)  # :91-95 / :99-103
    c = jnp.where(Cd <= 0.0, 0.0, c)  # :89-90 / :97-98 (not taken: SEAICE_waterDrag > 0)
    return CwatC.at[:, J, I].set(c * _m2(g, "maskC")[:, J, I])  # :106


# ---------------------------------------------------------------------------------------------------------------
# SEAICE_LSR_RHSU / RHSV and SEAICE_LSR_CALC_COEFFS (seaice_lsr.F:1470-1728, :1165-1467)


def lsr_rhsu(g, sg, zetaMinusEta, etaPlusZeta, etaZ, pressLoc, uIceC, vIceC, rhsU):
    """seaice_lsr.F:1474-1598 (no SEAICEuseStrImpCpl); rhsU updated on the interior."""
    L = g.layout
    sNx, sNy = L.sNx, L.sNy
    maskC = _m2(g, "maskC")
    mU, mV = sg["seaiceMaskU"], sg["seaiceMaskV"]
    sig11 = jnp.zeros_like(rhsU)  # :1522-1527
    sig12 = jnp.zeros_like(rhsU)
    J, I = L.js(1, sNy), L.is_(0, sNx)  # :1529-1538
    Jp = L.js(2, sNy + 1)
    s11 = ((zetaMinusEta[:, J, I] * (vIceC[:, Jp, I] - vIceC[:, J, I])) * g.recip_dyF[:, J, I]
           + ((etaPlusZeta[:, J, I] * sg["k2AtC"][:, J, I]) * 0.5) * (vIceC[:, Jp, I] + vIceC[:, J, I])) \
        - 0.5 * pressLoc[:, J, I]
    sig11 = sig11.at[:, J, I].set(s11)
    J, I = L.js(1, sNy + 1), L.is_(1, sNx)  # :1540-1559
    Jm, Im = L.js(0, sNy), L.is_(0, sNx - 1)
    hFacM = mV[:, J, I] - mV[:, J, Im]  # :1542
    s12 = etaZ[:, J, I] * ((vIceC[:, J, I] - vIceC[:, J, Im]) * g.recip_dxV[:, J, I]
                           - (sg["k1AtZ"][:, J, I] * 0.5) * (vIceC[:, J, I] + vIceC[:, J, Im]))
    s12 = (((s12 * maskC[:, J, I]) * maskC[:, J, Im]) * maskC[:, Jm, I]) * maskC[:, Jm, Im]
    s12 = s12 + (((etaZ[:, J, I] * g.recip_dxV[:, J, I]) * (vIceC[:, J, I] + vIceC[:, J, Im])) * hFacM) * 2.0
    sig12 = sig12.at[:, J, I].set(s12)
    J, I = L.js(1, sNy), L.is_(1, sNx)  # :1585-1595
    Jp, Im = L.js(2, sNy + 1), L.is_(0, sNx - 1)
    add = (g.recip_rAw[:, J, I] * mU[:, J, I]) * (
        ((g.dyF[:, J, I] * sig11[:, J, I] - g.dyF[:, J, Im] * sig11[:, J, Im])
         + g.dxV[:, Jp, I] * sig12[:, Jp, I]) - g.dxV[:, J, I] * sig12[:, J, I])
    return rhsU.at[:, J, I].set(rhsU[:, J, I] + add)


def lsr_rhsv(g, sg, zetaMinusEta, etaPlusZeta, etaZ, pressLoc, uIceC, vIceC, rhsV):
    """seaice_lsr.F:1605-1728 (no SEAICEuseStrImpCpl); rhsV updated on the interior."""
    L = g.layout
    sNx, sNy = L.sNx, L.sNy
    maskC = _m2(g, "maskC")
    mU, mV = sg["seaiceMaskU"], sg["seaiceMaskV"]
    sig22 = jnp.zeros_like(rhsV)  # :1653-1658
    sig12 = jnp.zeros_like(rhsV)
    J, I = L.js(0, sNy), L.is_(1, sNx)  # :1661-1670
    Ip = L.is_(2, sNx + 1)
    s22 = ((zetaMinusEta[:, J, I] * (uIceC[:, J, Ip] - uIceC[:, J, I])) * g.recip_dxF[:, J, I]
           + ((etaPlusZeta[:, J, I] * sg["k1AtC"][:, J, I]) * 0.5) * (uIceC[:, J, Ip] + uIceC[:, J, I])) \
        - 0.5 * pressLoc[:, J, I]
    sig22 = sig22.at[:, J, I].set(s22)
    J, I = L.js(1, sNy), L.is_(1, sNx + 1)  # :1672-1691
    Jm, Im = L.js(0, sNy - 1), L.is_(0, sNx)
    hFacM = mU[:, J, I] - mU[:, Jm, I]  # :1674
    s12 = etaZ[:, J, I] * ((uIceC[:, J, I] - uIceC[:, Jm, I]) * g.recip_dyU[:, J, I]
                           - (sg["k2AtZ"][:, J, I] * 0.5) * (uIceC[:, J, I] + uIceC[:, Jm, I]))
    s12 = (((s12 * maskC[:, J, I]) * maskC[:, J, Im]) * maskC[:, Jm, I]) * maskC[:, Jm, Im]
    s12 = s12 + (((etaZ[:, J, I] * g.recip_dyU[:, J, I]) * (uIceC[:, J, I] + uIceC[:, Jm, I])) * hFacM) * 2.0
    sig12 = sig12.at[:, J, I].set(s12)
    J, I = L.js(1, sNy), L.is_(1, sNx)  # :1716-1725
    Jm, Ip = L.js(0, sNy - 1), L.is_(2, sNx + 1)
    add = (g.recip_rAs[:, J, I] * mV[:, J, I]) * (
        ((g.dyU[:, J, Ip] * sig12[:, J, Ip] - g.dyU[:, J, I] * sig12[:, J, I])
         + g.dxF[:, J, I] * sig22[:, J, I]) - g.dxF[:, Jm, I] * sig22[:, Jm, I])
    return rhsV.at[:, J, I].set(rhsV[:, J, I] + add)


def lsr_calc_coeffs(p, g, sg, etaPlusZeta, zetaMinusEta, etaZ, zetaZ, dragSym, seaiceMassU, seaiceMassV):
    """seaice_lsr.F:1165-1467: dict AU, BU, CU, AV, BV, CV, uRt1, uRt2, vRt1, vRt2 on the interior (0 elsewhere; the
    Fortran local arrays are undefined there and never read)."""
    L = g.layout
    sNx, sNy = L.sNx, L.sNy
    mU, mV = sg["seaiceMaskU"], sg["seaiceMaskV"]
    bdfAlphaOverDt = 1.0 / p.SEAICE_deltaTdyn  # :1243-1252 (no BDF2)
    strImpCplFac = 0.0  # :1254-1255 (SEAICEuseStrImpCpl = F)
    areaW = 1.0  # :1266-1272 (SEAICEscaleSurfStress = F)
    areaS = 1.0
    zero = jnp.zeros_like(etaZ)
    # UXX, UXM: J=1..sNy, I=0..sNx (:1277-1286)
    J, I = L.js(1, sNy), L.is_(0, sNx)
    UXX = zero.at[:, J, I].set((g.dyF[:, J, I] * etaPlusZeta[:, J, I]) * g.recip_dxF[:, J, I])
    UXM = zero.at[:, J, I].set(((g.dyF[:, J, I] * zetaMinusEta[:, J, I]) * sg["k1AtC"][:, J, I]) * 0.5)
    # UYY, UYM: J=1..sNy+1, I=1..sNx (:1287-1297)
    J, I = L.js(1, sNy + 1), L.is_(1, sNx)
    UYY = zero.at[:, J, I].set((g.dxV[:, J, I] * (etaZ[:, J, I] + strImpCplFac * zetaZ[:, J, I]))
                               * g.recip_dyU[:, J, I])
    UYM = zero.at[:, J, I].set(((g.dxV[:, J, I] * etaZ[:, J, I]) * sg["k2AtZ"][:, J, I]) * 0.5)
    # VXX, VXM: J=1..sNy, I=1..sNx+1 (:1298-1308)
    J, I = L.js(1, sNy), L.is_(1, sNx + 1)
    VXX = zero.at[:, J, I].set((g.dyU[:, J, I] * (etaZ[:, J, I] + strImpCplFac * zetaZ[:, J, I]))
                               * g.recip_dxV[:, J, I])
    VXM = zero.at[:, J, I].set(((g.dyU[:, J, I] * etaZ[:, J, I]) * sg["k1AtZ"][:, J, I]) * 0.5)
    # VYY, VYM: J=0..sNy, I=1..sNx (:1309-1318)
    J, I = L.js(0, sNy), L.is_(1, sNx)
    VYY = zero.at[:, J, I].set((g.dxF[:, J, I] * etaPlusZeta[:, J, I]) * g.recip_dyF[:, J, I])
    VYM = zero.at[:, J, I].set(((g.dxF[:, J, I] * zetaMinusEta[:, J, I]) * sg["k2AtC"][:, J, I]) * 0.5)

    J, I = L.js(1, sNy), L.is_(1, sNx)
    Jm, Jp, Im, Ip = L.js(0, sNy - 1), L.js(2, sNy + 1), L.is_(0, sNx - 1), L.is_(2, sNx + 1)
    m = mU[:, J, I]
    AU = (-UXX[:, J, Im] + UXM[:, J, Im]) * m  # :1328-1329
    CU = (-UXX[:, J, I] - UXM[:, J, I]) * m  # :1331-1332
    BU = (ONE - m) + (((((((UXX[:, J, Im] + UXX[:, J, I]) + UYY[:, Jp, I]) + UYY[:, J, I]) + UXM[:, J, Im])
                        - UXM[:, J, I]) + UYM[:, Jp, I]) - UYM[:, J, I]) * m  # :1334-1337
    uRt1 = UYY[:, J, I] + UYM[:, J, I]  # :1339
    uRt2 = UYY[:, Jp, I] - UYM[:, Jp, I]  # :1341
    hFacM = mU[:, Jm, I]  # :1350
    hFacP = mU[:, Jp, I]  # :1351
    BU = BU + m * ((1.0 - hFacM) * (UYY[:, J, I] + UYM[:, J, I])
                   + (1.0 - hFacP) * (UYY[:, Jp, I] - UYM[:, Jp, I]))  # :1355-1357
    uRt1 = uRt1 * hFacM  # :1359
    uRt2 = uRt2 * hFacP  # :1360
    rAw = g.recip_rAw[:, J, I]
    AU = AU * rAw  # :1367
    CU = CU * rAw  # :1368
    BU = BU * rAw + m * (bdfAlphaOverDt * seaiceMassU[:, J, I]
                         + (0.5 * (dragSym[:, J, I] + dragSym[:, J, Im])) * areaW)  # :1372-1377
    uRt1 = uRt1 * rAw  # :1378
    uRt2 = uRt2 * rAw  # :1379

    m = mV[:, J, I]
    AV = (-VYY[:, Jm, I] + VYM[:, Jm, I]) * m  # :1391-1392
    CV = (-VYY[:, J, I] - VYM[:, J, I]) * m  # :1394-1395
    BV = (ONE - m) + (((((((VXX[:, J, I] + VXX[:, J, Ip]) + VYY[:, J, I]) + VYY[:, Jm, I]) - VXM[:, J, I])
                        + VXM[:, J, Ip]) - VYM[:, J, I]) + VYM[:, Jm, I]) * m  # :1397-1400
    vRt1 = VXX[:, J, I] + VXM[:, J, I]  # :1402
    vRt2 = VXX[:, J, Ip] - VXM[:, J, Ip]  # :1404
    hFacM = mV[:, J, Im]  # :1413
    hFacP = mV[:, J, Ip]  # :1414
    BV = BV + m * ((1.0 - hFacM) * (VXX[:, J, I] + VXM[:, J, I])
                   + (1.0 - hFacP) * (VXX[:, J, Ip] - VXM[:, J, Ip]))  # :1418-1420
    vRt1 = vRt1 * hFacM  # :1422
    vRt2 = vRt2 * hFacP  # :1423
    rAs = g.recip_rAs[:, J, I]
    AV = AV * rAs  # :1430
    CV = CV * rAs  # :1431
    BV = BV * rAs + m * (bdfAlphaOverDt * seaiceMassV[:, J, I]
                         + (0.5 * (dragSym[:, J, I] + dragSym[:, Jm, I])) * areaS)  # :1435-1440
    vRt1 = vRt1 * rAs  # :1441
    vRt2 = vRt2 * rAs  # :1442
    # :1446-1461 BU/BV = 0 fix: only with SEAICE_OLx/OLy > 0 or surface-stress scaling (not in V4r4)
    out = {}
    for k, v in dict(AU=AU, BU=BU, CU=CU, AV=AV, BV=BV, CV=CV, uRt1=uRt1, uRt2=uRt2, vRt1=vRt1, vRt2=vRt2).items():
        out[k] = zero.at[:, J, I].set(v)
    return out


# ---------------------------------------------------------------------------------------------------------------
# LSOR: line layout helpers. u: lines = rows J, along = I;  v: lines = columns I, along = J. Line arrays are
# [line, along, lane] with lanes = tiles (u) followed by tiles (v).


def _to_lines_u(a, L):
    return jnp.transpose(a[:, L.js(1, L.sNy), L.is_(1, L.sNx)], (1, 2, 0))  # [J, I, T]


def _to_lines_v(a, L):
    return jnp.transpose(a[:, L.js(1, L.sNy), L.is_(1, L.sNx)], (2, 1, 0))  # [I, J, T]


def _from_lines_u(x, L):
    return jnp.transpose(x, (2, 0, 1))  # [T, J, I]


def _from_lines_v(x, L):
    return jnp.transpose(x, (2, 1, 0))  # [T, J, I]


def _lines(co, L):
    """Line-layout coefficients (lanes = u tiles, v tiles) + elimination factors for one Picard pass.
    bet(I) = B(I) - A(I)*CUU(I-1), CUU(iMin) = C/B, CUU(I) = C(I)/bet(I) (seaice_lsr.F:1823-1840 / :1972-1993): these
    do not depend on the iterate, so they are formed once per pass with the Fortran operations."""
    cat = lambda a, b: jnp.concatenate([a, b], axis=-1)  # noqa: E731
    A = cat(_to_lines_u(co["AU"], L), _to_lines_v(co["AV"], L))
    B = cat(_to_lines_u(co["BU"], L), _to_lines_v(co["BV"], L))
    C = cat(_to_lines_u(co["CU"], L), _to_lines_v(co["CV"], L))
    Rt1 = cat(_to_lines_u(co["uRt1"], L), _to_lines_v(co["vRt1"], L))
    Rt2 = cat(_to_lines_u(co["uRt2"], L), _to_lines_v(co["vRt2"], L))
    rhs = cat(_to_lines_u(co["rhsU"], L), _to_lines_v(co["rhsV"], L))
    mask = cat(_to_lines_u(co["seaiceMaskU"], L), _to_lines_v(co["seaiceMaskV"], L))
    # scan over the along-line axis (axis 1), vectorised over lines and lanes
    CUU0 = C[:, 0] / B[:, 0]  # :1825 / :1978

    def elim(cuu_prev, xs):
        A_i, B_i, C_i = xs
        bet = B_i - A_i * cuu_prev  # :1839 / :1992
        cuu = C_i / bet  # :1840 / :1993
        return cuu, (bet, cuu)

    _, (bet, cuu) = lax.scan(elim, CUU0, (jnp.moveaxis(A[:, 1:], 1, 0), jnp.moveaxis(B[:, 1:], 1, 0),
                                          jnp.moveaxis(C[:, 1:], 1, 0)))
    bet = jnp.concatenate([B[:, :1], jnp.moveaxis(bet, 0, 1)], axis=1)  # bet(iMin) unused: B(iMin)
    CUU = jnp.concatenate([CUU0[:, None], jnp.moveaxis(cuu, 0, 1)], axis=1)
    return dict(A=A, B0=B[:, 0], Clast=C[:, -1], bet=bet, CUU=CUU, Rt1=Rt1, Rt2=Rt2, rhs=rhs, mask=mask)


def _thomas(r, A_r, B0_r, bet_r, CUU_r, unroll=1):
    """Tridiagonal solve along one line (seaice_lsr.F:1825-1855 / :1978-2009). r, A_r, bet_r, CUU_r: [along, lanes].
    unroll: lax.scan unroll of both substitutions (the same operations in the same order)."""
    y0 = r[0] / B0_r  # :1826 / :1979

    def fwd(yp, xs):
        r_i, A_i, b_i = xs
        y = (r_i - A_i * yp) / b_i  # :1841 / :1994
        return y, y

    _, ys = lax.scan(fwd, y0, (r[1:], A_r[1:], bet_r[1:]), unroll=unroll)
    y = jnp.concatenate([y0[None], ys])

    def bwd(yn, xs):  # DO I=iMin,iMax-1: IM=sNx-I (:1846-1855): IM = sNx-1 .. 1, using the updated URT(IM+1)
        y_i, c_i = xs
        v = y_i - c_i * yn  # :1854 / :2008
        return v, v

    _, yb = lax.scan(bwd, y[-1], (y[:-1], CUU_r[:-1]), reverse=True, unroll=unroll)
    return jnp.concatenate([yb, y[-1:]])


def _sweep(ln, tmp, lo, hi, prev0, nxt, w, unroll=1):
    """One LSOR sweep of u (rows) and v (columns) together, literal TRIDIAGU/V (seaice_lsr.F:1808-1870 /
    :1958-2022). tmp: [line, along, lane] current interior values (uTmp/vTmp); lo/hi: [line, lane] along-line halo
    values (uIce(0,J), uIce(sNx+1,J) / vIce(I,0), vIce(I,sNy+1)); prev0: [along, lane] the halo line before the first
    line (uIce(:,0) / vIce(0,:)); nxt: [line, along, lane] the old next line (uTmp(:,J+1) / vTmp(I+1,:)); w: [lane]
    relaxation WFAU / WFAV. Returns the new interior [line, along, lane]."""

    def line(prev, xs):
        rhs_r, A_r, Rt1_r, Rt2_r, mask_r, bet_r, CUU_r, B0_r, Cl_r, lo_r, hi_r, nx_r, tmp_r = xs
        AA3 = jnp.zeros_like(rhs_r)  # :1811 / :1961
        AA3 = AA3.at[0].set(AA3[0] - A_r[0] * lo_r)  # :1812 / :1962 (I.EQ.iMin)
        AA3 = AA3.at[-1].set(AA3[-1] - Cl_r * hi_r)  # :1813 / :1963 (I.EQ.iMax)
        r = (((rhs_r + AA3) + Rt1_r * prev) + Rt2_r * nx_r) * mask_r  # :1815-1819 / :1965-1969
        y = _thomas(r, A_r, B0_r, bet_r, CUU_r, unroll)
        xn = tmp_r + w * (y - tmp_r)  # :1866-1867 / :2019-2020
        return xn, xn

    _, rows = lax.scan(line, prev0, (ln["rhs"], ln["A"], ln["Rt1"], ln["Rt2"], ln["mask"], ln["bet"], ln["CUU"],
                                     ln["B0"], ln["Clast"], lo, hi, nxt, tmp))
    return rows


def _halo_parts(u, v, L):
    """lo/hi along-line halos, first-line halo and old next lines of u (rows) and v (columns), lane-stacked."""
    sNx, sNy = L.sNx, L.sNy
    J, I = L.js(1, sNy), L.is_(1, sNx)
    cat = lambda a, b: jnp.concatenate([a, b], axis=-1)  # noqa: E731
    lo = cat(u[:, J, L.ii(0)].T, v[:, L.jj(0), I].T)  # [line, lane]
    hi = cat(u[:, J, L.ii(sNx + 1)].T, v[:, L.jj(sNy + 1), I].T)
    prev0 = cat(u[:, L.jj(0), I].T, v[:, J, L.ii(0)].T)  # [along, lane]
    nxt = cat(_to_lines_u(u[:, 1:], L), jnp.transpose(v[:, J, L.is_(2, sNx + 1)], (2, 1, 0)))
    return lo, hi, prev0, nxt


LSR_IMPLS = ("auto", "pallas", "xla_unrolled", "xla", "pallas_interpret")
XLA_UNROLL = 1  # lax.scan unroll of the Thomas substitutions on the "xla" path (module docstring, "Implementations")


def _check_impl(p):
    if p.lsr_impl not in LSR_IMPLS:
        raise ValueError(f"lsr_impl = {p.lsr_impl!r}: one of {LSR_IMPLS}")
    return p.lsr_impl


def _sweep_impl(p, ln):
    """The forward sweep (tmp, lo, hi, prev0, nxt, w) -> new interior selected by the static option p.lsr_impl (module
    docstring, "Implementations"): "xla" = _sweep with lax.scan (unroll XLA_UNROLL), "xla_unrolled" = _sweep with the
    Thomas scans fully unrolled, "pallas" = seaice_lsr_pallas.sweep (one Triton kernel per sweep), "pallas_interpret"
    = that kernel run by the Pallas interpreter (CPU tests), "auto" = lax.platform_dependent: cpu -> "xla",
    cuda -> "pallas", any other platform (rocm, tpu, ...) -> "xla_unrolled" (only the branch of the platform the
    computation is lowered for is compiled). All are the same operations in the same order."""
    impl = _check_impl(p)
    xla = partial(_sweep, ln, unroll=XLA_UNROLL)
    xla_unrolled = partial(_sweep, ln, unroll=True)
    if impl == "xla":
        return xla
    if impl == "xla_unrolled":
        return xla_unrolled
    from mitgcm_jax.pkgs import seaice_lsr_pallas as slp
    lp = slp.pack_lines(ln)  # once per Picard pass, outside the sweep loop (dead code on the other branches)
    pallas = partial(slp.sweep, lp, interpret=impl == "pallas_interpret")
    if impl != "auto":
        return pallas
    return lambda *a: lax.platform_dependent(*a, cpu=xla, cuda=pallas, default=xla_unrolled)


def _precond_impl(p, ln, L, w):
    """The preconditioner sweep of the implicit derivative (traceable XLA code, transposed by jax.linear_transpose):
    Thomas scans as the forward path's XLA form ("xla" -> XLA_UNROLL; "xla_unrolled", "pallas", "pallas_interpret" ->
    fully unrolled; "auto" -> lax.platform_dependent: cpu XLA_UNROLL, any other platform fully unrolled)."""
    impl = _check_impl(p)
    rolled = partial(_precond, ln, L, w, unroll=XLA_UNROLL)
    unrolled = partial(_precond, ln, L, w, unroll=True)
    if impl == "xla":
        return rolled
    if impl != "auto":
        return unrolled
    return lambda b: lax.platform_dependent(b, cpu=rolled, default=unrolled)


def _lsor_fortran(p, ex, co, u, v, max_iter, lsr_error):
    """The LSOR loop (seaice_lsr.F:617-836) from uIce=u, vIce=v. Returns (u, v, info)."""
    L = ex.L
    T = u.shape[0]  # local tiles (sharded: this device's block), not L.nTiles
    ln = _lines(co, L)
    sweep = _sweep_impl(p, ln)
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    mU = co["seaiceMaskU"][:, J, I]
    mV = co["seaiceMaskV"][:, J, I]
    WFAU2 = ZERO  # seaice_lsr.F:263
    WFAV2 = ZERO

    def body(st):
        m, go, u, v, wu, wv, s1, s2, s1a, s2a, ic1, ic2 = st
        # :657-660 useCubedSphereExchange: doIterate4u = doIterate4v = .TRUE.
        uTmp, vTmp = u, v  # :753-758
        tmp = jnp.concatenate([_to_lines_u(uTmp, L), _to_lines_v(vTmp, L)], axis=-1)
        lo, hi, prev0, nxt = _halo_parts(uTmp, vTmp, L)
        w = jnp.concatenate([jnp.full((T,), wu), jnp.full((T,), wv)])
        rows = sweep(tmp, lo, hi, prev0, nxt, w)  # :760-774
        un = uTmp.at[:, J, I].set(_from_lines_u(rows[..., :T], L))
        vn = vTmp.at[:, J, I].set(_from_lines_v(rows[..., T:], L))
        check = (m % p.SOLV_NCHECK) == 0  # :784, :809
        # :785-797 / :810-822  S1 = max |UERR| over the interior, _GLOBAL_MAX_RL
        S1n = ex.global_max(jnp.max(jnp.abs((un[:, J, I] - uTmp[:, J, I]) * mU), axis=(1, 2)))
        S2n = ex.global_max(jnp.max(jnp.abs((vn[:, J, I] - vTmp[:, J, I]) * mV), axis=(1, 2)))
        wu = jnp.where(check & (m > 1) & (S1n > s1a), WFAU2, wu)  # :801
        s1a = jnp.where(check, S1n, s1a)  # :802
        s1 = jnp.where(check, S1n, s1)
        cu = check & (S1n < lsr_error)  # :803-806
        ic1 = jnp.where(cu, m, ic1)
        wv = jnp.where(check & (m > 1) & (S2n > s2a), WFAV2, wv)  # :824
        s2a = jnp.where(check, S2n, s2a)  # :825
        s2 = jnp.where(check, S2n, s2)
        cv = check & (S2n < lsr_error)  # :826-829
        ic2 = jnp.where(cv, m, ic2)
        un, vn = ex.exch_uv_xy(un, vn, True)  # :832 EXCH_UV_XY_RL(uIce, vIce, .TRUE.)
        go = jnp.logical_not(cu) | jnp.logical_not(cv)  # doIterate4u .OR. doIterate4v (:655)
        return (m + 1, go, un, vn, wu, wv, s1, s2, s1a, s2a, ic1, ic2)

    def cond(st):
        return (st[0] <= max_iter) & st[1]  # DO m = 1, SOLV_MAX_TMP ; IF (doIterate4u .OR. doIterate4v)

    f = jnp.float64
    st0 = (jnp.int32(1), jnp.bool_(True), u, v, f(p.SEAICE_LSRrelaxU), f(p.SEAICE_LSRrelaxV), f(0.0), f(0.0),
           f(0.80), f(0.80), jnp.int32(max_iter), jnp.int32(max_iter))  # :261-268, :638-639
    st = lax.while_loop(cond, body, st0)
    m, go, u, v, wu, wv, s1, s2, _, _, ic1, ic2 = st
    info = dict(ICOUNT1=ic1.astype(f), ICOUNT2=ic2.astype(f), S1=s1, S2=s2, WFAU=wu, WFAV=wv,
                sweeps=(m - 1).astype(f), converged=jnp.logical_not(go).astype(f))
    return u, v, info


# ---------------------------------------------------------------------------------------------------------------
# linear operator of the LSOR fixed point (for the implicit derivative)


def _interior(L):
    return L.js(1, L.sNy), L.is_(1, L.sNx)


def _residual(ex, co, u, v):
    """F(x; c) = mask*(rhs + Rt1 x(-1) + Rt2 x(+1)) - (A x(i-1) + B x + C x(i+1)) on the interior (0 on halo lanes)
    for u and v; u, v carry their exchanged halos. F = 0 is the fixed point of the literal line-SOR sweep."""
    L = ex.L
    J, I = _interior(L)
    Jm, Jp, Im, Ip = L.js(0, L.sNy - 1), L.js(2, L.sNy + 1), L.is_(0, L.sNx - 1), L.is_(2, L.sNx + 1)
    Fu = co["seaiceMaskU"][:, J, I] * (co["rhsU"][:, J, I] + co["uRt1"][:, J, I] * u[:, Jm, I]
                                       + co["uRt2"][:, J, I] * u[:, Jp, I]) \
        - (co["AU"][:, J, I] * u[:, J, Im] + co["BU"][:, J, I] * u[:, J, I] + co["CU"][:, J, I] * u[:, J, Ip])
    Fv = co["seaiceMaskV"][:, J, I] * (co["rhsV"][:, J, I] + co["vRt1"][:, J, I] * v[:, J, Im]
                                       + co["vRt2"][:, J, I] * v[:, J, Ip]) \
        - (co["AV"][:, J, I] * v[:, Jm, I] + co["BV"][:, J, I] * v[:, J, I] + co["CV"][:, J, I] * v[:, Jp, I])
    z = jnp.zeros_like(u)
    return z.at[:, J, I].set(Fu), z.at[:, J, I].set(Fv)


def _embed_exchange(ex, xu, xv, inner):
    """Interior-only (u, v) -> full arrays with exchanged halos (halo lanes of the input ignored)."""
    return ex.exch_uv_xy(jnp.where(inner, xu, 0.0), jnp.where(inner, xv, 0.0), True)


def _matvec(ex, co, inner, x):
    """A x (interior only) for x = (xu, xv) interior-only: A = -dF/dx."""
    zc = {k: (jnp.zeros_like(co[k]) if k in ("rhsU", "rhsV") else co[k]) for k in co}
    u, v = _embed_exchange(ex, x[0], x[1], inner)
    Fu, Fv = _residual(ex, zc, u, v)
    return (-Fu, -Fv)


def _precond(ln, L, w, b, unroll=1):
    """P b = omega (D - omega L)^-1 b: one line-SOR sweep from a zero iterate with zero halos (linear in b).
    unroll: lax.scan unroll of the Thomas substitutions (_precond_impl)."""
    T = b[0].shape[0]
    bl = jnp.concatenate([_to_lines_u(b[0], L), _to_lines_v(b[1], L)], axis=-1)
    n_along, lanes = bl.shape[1], bl.shape[2]

    def line(prev, xs):
        b_r, A_r, Rt1_r, mask_r, bet_r, CUU_r, B0_r = xs
        r = (b_r + Rt1_r * prev) * mask_r
        y = _thomas(r, A_r, B0_r, bet_r, CUU_r, unroll)
        xn = w * y
        return xn, xn

    # the zero first iterate must vary like the lines inside shard_map(check_vma=True) (the carry the scan returns is
    # tile-varying); typed as bl's mesh axes (none on one device: then nothing changes)
    x0 = jnp.zeros((n_along, lanes), bl.dtype)
    vary = tuple(sorted(getattr(getattr(jax.typeof(bl), "mat", None), "varying", frozenset())))
    if vary:
        x0 = lax.pcast(x0, vary[0] if len(vary) == 1 else vary, to="varying")
    _, rows = lax.scan(line, x0, (bl, ln["A"], ln["Rt1"], ln["mask"], ln["bet"], ln["CUU"], ln["B0"]))
    J, I = _interior(L)
    z = jnp.zeros_like(b[0])
    return (z.at[:, J, I].set(_from_lines_u(rows[..., :T], L)), z.at[:, J, I].set(_from_lines_v(rows[..., T:], L)))


def _stationary(apply_A, apply_P, b, n_iter):
    """n_iter preconditioned stationary iterations x <- x + P(b - A x) from x = 0 (the Fortran LSOR in operator form).
    Kept for reference/tests: its rate (~415 sweeps per decade at LLC90) is too slow for the derivative solves."""
    def step(_, x):
        r = apply_A(x)
        d = apply_P((b[0] - r[0], b[1] - r[1]))
        return (x[0] + d[0], x[1] + d[1])

    return lax.fori_loop(0, n_iter, step, (jnp.zeros_like(b[0]), jnp.zeros_like(b[1])))


def _gmres(apply_A, apply_M, b, restart, cycles, gsum):
    """Left-preconditioned restarted GMRES(restart) with a FIXED number of cycles (no convergence test): solves
    A x = b through min ||M(b - A x)||. x = (xu, xv) interior-only arrays [T, ny, nx]. Inner products are per-tile
    partial sums added in tile order (gsum = Exchanger.global_sum_tile: P-independent). Classical Gram-Schmidt applied
    twice; a zero Krylov vector (exact convergence) is kept at zero instead of dividing by 0."""
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


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _lsor_implicit(static, dyn, co, u0, v0):
    p, ex, max_iter, lsr_error = _join_static(static, dyn)
    return _lsor_fortran(p, ex, co, u0, v0, max_iter, lsr_error)


@_lsor_implicit.defjvp
def _lsor_implicit_jvp(static, primals, tangents):
    dyn, co, u0, v0 = primals
    _, dco, _, _ = tangents
    p, ex, max_iter, lsr_error = _join_static(static, dyn)
    u, v, info = _lsor_implicit(static, dyn, co, u0, v0)
    L = ex.L
    J, I = _interior(L)
    inner = jnp.zeros(u.shape, bool).at[:, J, I].set(True)
    # dF/dc . dc at the returned iterate (u, v with their exchanged halos): linear in dco
    _, rhs = jax.jvp(lambda c: _residual(ex, c, u, v), (co,), (dco,))
    ln = _lines(co, L)
    T = u.shape[0]
    w = jnp.concatenate([jnp.full((T,), p.SEAICE_LSRrelaxU), jnp.full((T,), p.SEAICE_LSRrelaxV)])
    A = partial(_matvec, ex, co, inner)
    P = _precond_impl(p, ln, L, w)
    m, cycles = p.lsr_ad_restart, p.lsr_ad_cycles
    gsum = ex.global_sum_tile

    def solve(matvec, b):
        return _gmres(matvec, P, b, m, cycles, gsum)

    def transpose_solve(vecmat, b):
        PT = jax.linear_transpose(P, b)
        return _gmres(vecmat, lambda r: PT(r)[0], b, m, cycles, gsum)

    dx = lax.custom_linear_solve(A, rhs, solve, transpose_solve)
    du, dv = _embed_exchange(ex, dx[0], dx[1], inner)
    return (u, v, info), (du, dv, jax.tree.map(_zero_tangent, info))


def lsor_solve(p, ex, co, u0, v0, max_iter=None, lsr_error=None):
    """The LSOR loop of one Picard pass: literal forward, implicit derivative (module docstring).
    co: dict AU, BU, CU, AV, BV, CV, uRt1, uRt2, vRt1, vRt2, rhsU, rhsV, seaiceMaskU, seaiceMaskV ([T, ny, nx]).
    Returns (uIce, vIce, info) after the loop (before the masking of seaice_lsr.F:898-907); info: ICOUNT1/2, S1, S2,
    WFAU/V (as dumped in stage L04), the number of sweeps and a converged flag."""
    max_iter = p.SEAICElinearIterMax if max_iter is None else max_iter
    lsr_error = p.LSR_ERROR if lsr_error is None else lsr_error
    static, dyn = _split_static((p, ex, max_iter, lsr_error))
    return _lsor_implicit(static, dyn, co, lax.stop_gradient(u0), lax.stop_gradient(v0))


# ---------------------------------------------------------------------------------------------------------------
# SEAICE_LSR (seaice_lsr.F:17-1015)


def lsr_forcing(p, g, sg, st, uIceC, vIceC, uIceNm1, vIceNm1, DWATN, FORCEX, FORCEY):
    """FORCEX/FORCEY of one Picard pass (seaice_lsr.F:373-430) on the interior; halos keep the passed values."""
    L = g.layout
    sNx, sNy = L.sNx, L.sNy
    J, I = L.js(1, sNy), L.is_(1, sNx)
    Jm, Jp, Im, Ip = L.js(0, sNy - 1), L.js(2, sNy + 1), L.is_(0, sNx - 1), L.is_(2, sNx + 1)
    uVel, vVel, fC = st["uVel"], st["vVel"], g.fCori
    COSWAT, SINWAT = p.COSWAT, p.SINWAT
    areaW = 1.0  # seaice_lsr.F:199-206 (SEAICEscaleSurfStress = F)
    areaS = 1.0
    recip_deltaT = 1.0 / p.SEAICE_deltaTdyn  # :174
    D = DWATN
    sgn = jnp.copysign(SINWAT, fC[:, J, I])  # SIGN(SINWAT, fCori)
    fx = st["FORCEX0"][:, J, I] + (
        (((0.5 * (D[:, J, I] + D[:, J, Im])) * COSWAT) * uVel[:, J, I])
        - (sgn * 0.5) * ((D[:, J, I] * 0.5) * (((vVel[:, J, I] - vIceC[:, J, I]) + vVel[:, Jp, I]) - vIceC[:, Jp, I])
                         + (D[:, J, Im] * 0.5) * (((vVel[:, J, Im] - vIceC[:, J, Im]) + vVel[:, Jp, Im])
                                                  - vIceC[:, Jp, Im]))) * areaW  # :379-389
    fy = st["FORCEY0"][:, J, I] + (
        (((0.5 * (D[:, J, I] + D[:, Jm, I])) * COSWAT) * vVel[:, J, I])
        + (sgn * 0.5) * ((D[:, J, I] * 0.5) * (((uVel[:, J, I] - uIceC[:, J, I]) + uVel[:, J, Ip]) - uIceC[:, J, Ip])
                         + (D[:, Jm, I] * 0.5) * (((uVel[:, Jm, I] - uIceC[:, Jm, I]) + uVel[:, Jm, Ip])
                                                  - uIceC[:, Jm, Ip]))) * areaS  # :390-400
    mC = st["seaiceMassC"]
    fx = fx + HALF * (((mC[:, J, I] * fC[:, J, I]) * 0.5) * (vIceC[:, J, I] + vIceC[:, Jp, I])
                      + ((mC[:, J, Im] * fC[:, J, Im]) * 0.5) * (vIceC[:, J, Im] + vIceC[:, Jp, Im]))  # :406-410
    fy = fy - HALF * (((mC[:, J, I] * fC[:, J, I]) * 0.5) * (uIceC[:, J, I] + uIceC[:, J, Ip])
                      + ((mC[:, Jm, I] * fC[:, Jm, I]) * 0.5) * (uIceC[:, Jm, I] + uIceC[:, Jm, Ip]))  # :411-415
    fx = fx + (st["seaiceMassU"][:, J, I] * recip_deltaT) * uIceNm1[:, J, I]  # :421-423
    fy = fy + (st["seaiceMassV"][:, J, I] * recip_deltaT) * vIceNm1[:, J, I]  # :424-426
    fx = fx * sg["seaiceMaskU"][:, J, I]  # :427
    fy = fy * sg["seaiceMaskV"][:, J, I]  # :428
    return FORCEX.at[:, J, I].set(fx), FORCEY.at[:, J, I].set(fy)


def seaice_lsr(p, g, sg, ex, st, record=False, max_iter=None, lsr_error=None):
    """SEAICE_LSR (seaice_lsr.F:17-1015) with the V4r4 branches.

    st: dict with uIce, vIce (on entry), uVel, vVel (surface level), seaiceMassC/U/V, FORCEX0/Y0, PRESS0, ZMAX, ZMIN,
    and the values on entry of the arrays this routine writes only partly: e11, e22, e12, DWATN, FORCEX, FORCEY.
    sg: seaiceMaskU/V, HEFFM, tensileStrFac, k1AtC, k1AtZ, k2AtC, k2AtZ.
    Returns (out, passes): out = uIce, vIce (masked), uIceNm1, vIceNm1, e11, e22, e12, deltaC, ETA, etaZ, ZETA, zetaZ,
    PRESS, DWATN, FORCEX, FORCEY; passes (record=True) = per Picard pass the stage-L01/L02/L04 values."""
    L = g.layout
    st = {k: jnp.asarray(v) for k, v in st.items()}
    if not p.useCubedSphereExchange:
        raise NotImplementedError("SEAICE_LSR stopping rule ported for useCubedSphereExchange only")
    uIce, vIce = st["uIce"], st["vIce"]
    zero = jnp.zeros_like(uIce)
    # :178-195 (ALLOW_AUTODIFF_TAMC) zero deltaC, press, zeta, zetaZ, eta, etaZ, uIceC, vIceC, uIceNm1, vIceNm1
    uIceNm1, vIceNm1 = zero, zero
    e11, e22, e12 = st["e11"], st["e22"], st["e12"]
    DWATN, FORCEX, FORCEY = st["DWATN"], st["FORCEX"], st["FORCEY"]
    J0, I0 = L.js(0, L.sNy), L.is_(0, L.sNx)
    passes = []
    nonLinIterLoc = p.SEAICEnonLinIterMax  # :168
    ipass0 = 2  # :220
    if nonLinIterLoc > 2:  # :223
        ipass0 = 1
    for ipass in range(1, p.MPSEUDOTIMESTEPS + 1):  # :225 (ALLOW_AUTODIFF_TAMC)
        if ipass > nonLinIterLoc:  # :242
            continue
        if ipass == 1:  # :270-283
            uIceNm1, vIceNm1 = uIce, vIce
            uIceC, vIceC = uIce, vIce
        else:  # :284-297
            uIce = HALF * (uIce + uIceNm1)
            vIce = HALF * (vIce + vIceNm1)
            uIceC, vIceC = uIce, vIce
        if ipass > ipass0:  # :299-311
            uIceNm1, vIceNm1 = uIce, vIce
        e11, e22, e12 = calc_strainrates(p, g, sg, uIceC, vIceC, e11, e22, e12)  # :322-325
        eta, etaZ, zeta, zetaZ, press, deltaC = calc_viscosities(
            p, g, sg, e11, e22, e12, st["ZMIN"], st["ZMAX"], sg["HEFFM"], st["PRESS0"], sg["tensileStrFac"])  # :327
        DWATN = oceandrag_coeffs(p, g, uIceC, vIceC, st["uVel"], st["vVel"], DWATN)  # :332-335
        rec = {}
        if record:
            rec["L01"] = dict(UICE=uIce, VICE=vIce, uIceC=uIceC, vIceC=vIceC, uIceNm1=uIceNm1, vIceNm1=vIceNm1,
                              e11=e11, e22=e22, e12=e12, deltaC=deltaC, ETA=eta, etaZ=etaZ, ZETA=zeta, zetaZ=zetaZ,
                              PRESS=press, DWATN=DWATN, FORCEX=FORCEX, FORCEY=FORCEY)
        # :351-368 (local arrays; outside J,I = 0..sN they are undefined in the Fortran and never read)
        etaPlusZeta = zero.at[:, J0, I0].set(eta[:, J0, I0] + zeta[:, J0, I0])
        zetaMinusEta = zero.at[:, J0, I0].set(zeta[:, J0, I0] - eta[:, J0, I0])
        dragSym = DWATN * p.COSWAT  # :361
        FORCEX, FORCEY = lsr_forcing(p, g, sg, st, uIceC, vIceC, uIceNm1, vIceNm1, DWATN, FORCEX, FORCEY)
        J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
        rhsU = zero.at[:, J, I].set(FORCEX[:, J, I])  # :435-439
        rhsU = lsr_rhsu(g, sg, zetaMinusEta, etaPlusZeta, etaZ, press, uIceC, vIceC, rhsU)  # :440-444
        rhsV = zero.at[:, J, I].set(FORCEY[:, J, I])  # :448-452
        rhsV = lsr_rhsv(g, sg, zetaMinusEta, etaPlusZeta, etaZ, press, uIceC, vIceC, rhsV)  # :453-457
        co = lsr_calc_coeffs(p, g, sg, etaPlusZeta, zetaMinusEta, etaZ, zetaZ, dragSym,
                             st["seaiceMassU"], st["seaiceMassV"])  # :468-471
        co.update(rhsU=rhsU, rhsV=rhsV, seaiceMaskU=sg["seaiceMaskU"], seaiceMaskV=sg["seaiceMaskV"])
        if record:
            rec["L02"] = dict(co, etaPlusZeta=etaPlusZeta, zetaMinusEta=zetaMinusEta, dragSym=dragSym,
                              FORCEX=FORCEX, FORCEY=FORCEY)
            rec["L02_inputs"] = dict(uIce=uIce, vIce=vIce)
        # :475-610 SEAICE_RESIDUAL (print only) and free-drift mixing (LSR_mixIniGuess = 0: not executed)
        uIce, vIce, info = lsor_solve(p, ex, co, uIce, vIce, max_iter=max_iter, lsr_error=lsr_error)  # :617-836
        if record:
            rec["L04"] = dict(UICE=uIce, VICE=vIce, **info)
            rec["co"] = co
        uIce = uIce * sg["seaiceMaskU"]  # :898-907
        vIce = vIce * sg["seaiceMaskV"]
        passes.append(rec)
    # :951-1012 useHB87StressCoupling = F
    out = dict(uIce=uIce, vIce=vIce, uIceNm1=uIceNm1, vIceNm1=vIceNm1, e11=e11, e22=e22, e12=e12, deltaC=deltaC,
               ETA=eta, etaZ=etaZ, ZETA=zeta, zetaZ=zetaZ, PRESS=press, DWATN=DWATN, FORCEX=FORCEX, FORCEY=FORCEY)
    return out, passes
