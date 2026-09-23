"""Implicit vertical diffusion / viscosity: IMPLDIFF (plan Task 14c), for all tiles, usable for momentum and tracers.

Literal port of MITgcm_c66g/model/src/impldiff.F (the flux-forced override `code/impldiff.F` only adds the
Um_Impl/Vm_Impl diagnostics: df3d/impl/maximpl are written, gTracer is not; verified by diff). The tridiagonal system
(1 - dt d_r(kappa d_r)) x = gTracer is set up and solved column by column with the Thomas algorithm exactly as the
Fortran does (impldiff.F:128-306; TARGET_NEC_SX undefined, so every loop runs on iMin..iMax, jMin..jMax only and
points outside keep their input value). The k recursions are `lax.scan`s in the Fortran order; divisions are guarded
so that masked (b=0) lanes stay finite in the backward pass.

Note: SOLVE_DIAGONAL_KINNER (CPP_OPTIONS.h) concerns SOLVE_TRIDIAGONAL (GAD_IMPLICIT_R / MOM_U_IMPLICIT_R), not
IMPLDIFF, which carries its own solver.
"""

import jax
import jax.numpy as jnp
import numpy as np


def vertical_factors(Nr):
    """1-D vertical factors of GRID.h used by the vertical kernels, as set in V4r4 (all 1): the namelist checks
    `check_unit_vertical_factors` guard the options that would make them differ."""
    return dict(
        gravFacC=np.ones(Nr),          # load_ref_files.F:161 gravFacC(k) = 1 (gravityFile=' ')
        gravFacF=np.ones(Nr + 1),      # load_ref_files.F:165 gravFacF(k) = 1
        recip_deepFacC=np.ones(Nr),    # set_grid_factors.F:54 (deepAtmosphere=F)
        recip_deepFac2C=np.ones(Nr),   # set_grid_factors.F:55
        deepFac2F=np.ones(Nr + 1),     # set_grid_factors.F:59
        recip_rhoFacC=np.ones(Nr),     # set_ref_state.F:76 (rhoRefFile=' ': not anelastic)
        rhoFacF=np.ones(Nr + 1),       # set_ref_state.F:79
    )


def check_unit_vertical_factors(nml):
    """Hard error for the options under which `vertical_factors` would not be 1."""
    g = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
    bad = []
    if str(g("gravityFile", " ")).strip():          # set_defaults.F:62 gravityFile = ' '
        bad.append("gravityFile (load_ref_files.F:123)")
    if str(g("rhoRefFile", " ")).strip():           # set_defaults.F:61 rhoRefFile = ' '
        bad.append("rhoRefFile (set_ref_state.F:336 anelastic)")
    if nml.get("data", "parm04", "deepAtmosphere", default=False):  # set_defaults.F:73
        bad.append("deepAtmosphere (set_grid_factors.F:62)")
    if bad:
        raise NotImplementedError("vertical grid factors != 1 not ported: " + ", ".join(bad))


def impldiff_deltaTX(tracerId, deltaTMom, dTtracerLev=None, Nr=50):
    """impldiff.F:96-112: deltaTX(k) = dTtracerLev(k) for a tracer (tracerId >= 1), deltaTMom for momentum (-1, -2)
    and tracerId = 0. PTRACERS (tracerId >= GAD_TR1) are not compiled in V4r4."""
    if tracerId >= 1:
        if dTtracerLev is None:
            raise ValueError("tracer IMPLDIFF needs dTtracerLev")
        return jnp.asarray(dTtracerLev, dtype=jnp.float64)  # impldiff.F:105-107
    return jnp.full((Nr,), deltaTMom, dtype=jnp.float64)    # impldiff.F:109-111 (deltaTMom may be traced)


def _safe_recip(d):
    """1/d where d .NE. 0 (the IF guards of impldiff.F:227, :247-248), 1 elsewhere (the bet initial value, :209);
    the division never sees d = 0, so the backward pass stays finite."""
    nz = d != 0.0
    return jnp.where(nz, 1.0 / jnp.where(nz, d, 1.0), 1.0)


def impldiff(g, tracerId, KappaRX, recip_hFac, gTracer, deltaTX, iMin, iMax, jMin, jMax):
    """IMPLDIFF(bi,bj,iMin,iMax,jMin,jMax, tracerId, KappaRX, recip_hFac, gTracer) on all tiles.

    KappaRX [T,Nr(+1),ny,nx] (levels 1..Nr used), recip_hFac [T,Nr,ny,nx] (recip_hFacW/S/C at call time),
    gTracer [T,Nr,ny,nx]; deltaTX [Nr] (`impldiff_deltaTX`); grid: recip_drF [Nr], recip_drC [Nr+1].
    Returns the new gTracer (input values outside iMin..iMax, jMin..jMax). tracerId only selects deltaTX (and
    diagnostics in Fortran), so it is not used here.
    """
    del tracerId
    L = g.layout
    Nr = L.Nr
    vf = vertical_factors(Nr)
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    col = lambda v: jnp.asarray(v)[None, :, None, None]  # noqa: E731  [Nr] -> [1,Nr,1,1]
    dt = col(deltaTX)
    rdrF = col(g.recip_drF)
    rdrC = col(jnp.asarray(g.recip_drC)[:Nr])        # recip_drC(k), k=1..Nr
    rdrCp = col(jnp.asarray(g.recip_drC)[1:Nr + 1])  # recip_drC(k+1)
    rdf2c, rrfc = col(vf["recip_deepFac2C"]), col(vf["recip_rhoFacC"])
    df2F, rfF = col(vf["deepFac2F"][:Nr]), col(vf["rhoFacF"][:Nr])
    df2Fp, rfFp = col(vf["deepFac2F"][1:Nr + 1]), col(vf["rhoFacF"][1:Nr + 1])

    rh = jnp.asarray(recip_hFac)[:, :, J, I]
    K = jnp.asarray(KappaRX)[:, :Nr, J, I]
    gT = jnp.asarray(gTracer)
    x = gT[:, :, J, I]

    # impldiff.F:136 a(1) = 0 ; :147-151 k=2..Nr (lower diagonal)
    a = -(dt * rh * rdrF * rdf2c * rrfc * K * rdrC * df2F * rfF)
    rh_km1 = jnp.concatenate([jnp.zeros_like(rh[:, :1]), rh[:, :-1]], axis=1)
    a = jnp.where(rh_km1 == 0.0, 0.0, a)
    a = a.at[:, 0].set(0.0)
    # impldiff.F:157-170 k=1..Nr-1 (upper diagonal) ; :180 c(Nr) = 0
    K_kp1 = jnp.concatenate([K[:, 1:], jnp.zeros_like(K[:, :1])], axis=1)
    c = -(dt * rh * rdrF * rdf2c * rrfc * K_kp1 * rdrCp * df2Fp * rfFp)
    rh_kp1 = jnp.concatenate([rh[:, 1:], jnp.zeros_like(rh[:, :1])], axis=1)
    c = jnp.where(rh_kp1 == 0.0, 0.0, c)
    c = c.at[:, Nr - 1].set(0.0)
    # impldiff.F:193 b = 1 - (a + c)
    b = 1.0 - (a + c)

    # forward sweep (impldiff.F:216-278): bet(1) = 1/b(1) if b(1).NE.0; gam(k) = c(k-1)*bet(k-1);
    # bet(k) = 1/(b(k) - a(k)*gam(k)) if nonzero; locTr(1) = gTracer(1)*bet(1);
    # locTr(k) = bet(k)*(gTracer(k) - a(k)*locTr(k-1)). bet/gam do not depend on locTr, so one scan gives the
    # values of the Fortran's two loops.
    km = lambda v: jnp.moveaxis(v, 1, 0)  # noqa: E731  [T,Nr,..] -> [Nr,T,..]
    a_, b_, c_, x_ = km(a), km(b), km(c), km(x)
    bet1 = _safe_recip(b_[0])
    loc1 = x_[0] * bet1

    def fwd(carry, xs):
        bet_m, loc_m, c_m = carry
        ak, bk, ck, xk = xs
        gam = c_m * bet_m
        bet = _safe_recip(bk - ak * gam)
        loc = bet * (xk - ak * loc_m)
        return (bet, loc, ck), (gam, loc)

    _, (gam_, loc_) = jax.lax.scan(fwd, (bet1, loc1, c_[0]), (a_[1:], b_[1:], c_[1:], x_[1:]))
    loc_ = jnp.concatenate([loc1[None], loc_], axis=0)   # locTr(1..Nr)
    # gam_[k-2] = gam(k), k=2..Nr

    # backward sweep (impldiff.F:282-293): locTr(k) = locTr(k) - gam(k+1)*locTr(k+1), k = Nr-1..1
    def bwd(loc_p, xs):
        lk, gk1 = xs
        lk = lk - gk1 * loc_p
        return lk, lk

    _, loc_up = jax.lax.scan(bwd, loc_[Nr - 1], (loc_[:Nr - 1], gam_), reverse=True)
    loc_ = jnp.concatenate([loc_up, loc_[Nr - 1:]], axis=0)
    # impldiff.F:295-306 gTracer = locTr on iMin..iMax, jMin..jMax
    return gT.at[:, :, J, I].set(jnp.moveaxis(loc_, 0, 1))
