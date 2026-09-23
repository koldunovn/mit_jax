"""Global volume / SSH / heat / salt budgets of one FORWARD_STEP, in the discrete form of the V4r4 flux-forced
configuration (plan Task 19): z* (select_rStar=2, nonlinFreeSurf=4), exactConserv, useRealFreshWaterFlux,
staggerTimeStep, temp_EvPrRn=0, salt_EvPrRn=0 (default), convertFW2Salt=-1, implicit vertical advection (U3) and
diffusion, DST3 multi-dimensional horizontal advection without Adams-Bashforth on tracers, shortwave penetration,
geothermal flux, salt plume (READIN_SALT_PLUME_FLUX, SALT_PLUME_VOLUME undef).

    b = step_budget(P, g, kLowC, st0, st1)      # st1 = forward_step(P, g, ex, kLowC, st0, exf_in)[0]
    b["resid_heat"], b["rel_heat"], ...         # jnp scalars; jit-able, usable in a lax.scan body

Everything is computed from the model's own discrete quantities (State fields and grid); nothing is re-derived
from the dynamics. All sums run over the interior points (i=1..sNx, j=1..sNy) of every tile; content changes are
differenced cell by cell before summing (no cancellation of two ~1e18 m^3 totals).

Which thickness goes with which tracer (the one trap)
-----------------------------------------------------
Let F^n = rStarFacC of the State at the start of iteration n (CALC_R_STAR of step n-1 computed it from etaH^n:
calc_r_star.F:103-113, F = (etaH + Ro_surf - R_low)*recip_Rcol). During step n, UPDATE_R_STAR(.TRUE.)
(ff forward_step.F:855; update_r_star.F: hFacC = h0FacC*rStarFacC) sets hFacC = h0FacC*F^n; CALC_R_STAR (:980)
then advances rStarFacC to F^{n+1} and sets rStarExpC = F^{n+1}/F^n; THERMODYNAMICS steps theta^n -> theta^{n+1}
with recip_hFacC = 1/(h0FacC F^n) and recip_hFacNew = recip_hFacC/rStarExpC (thermodynamics.F:189-211). The
State's own `hFacC` at the start of iteration n+1 is therefore h0FacC*F^n, one step BEHIND its theta: the tracer
content is  C(st) = sum rA*drF*h0FacC*rStarFacC*theta  (not ... *hFacC*theta; `content(..., pair="hFacC")` exists
only as the negative control).

Discrete budgets (cell volume V^n = rA drF h0FacC F^n, dt = deltaTtracer = deltaTfreesurf = 3600 s)
--------------------------------------------------------------------------------------------------
Tracer update of one wet cell (c66g, temp_integrate.F:285-547 with the ff apply_forcing.F):
    theta* = theta^n + dt*G/rStarExpC                        (freesurf_rescale_g.F:42-50, timestep_tracer.F:54-69)
    theta^{n+1} + dt/V^{n+1} * D_r(Fv(theta^{n+1})) = theta* (gad_implicit_r.F:104-285, flux form: every interface
                                                              term enters the rows above and below with opposite
                                                              sign, gad_u3c4_impl_r.F:157-196, :125-157 diffusion)
    V^n G = -div(F_adv + F_diff + F_GM) + theta^n*(dx uTrans + dy vTrans + rA*(w_k - w_{k+1})) + V^n*forcing
            gad_advection.F:544-550,753-759,786-787 (horizontal: "af(i+1)-af(i) - tracer*(uTrans(i+1)-uTrans(i))");
            gad_calc_rhs.F:773-787 with advFac=0, rAdvFac=rkSign (vertical "-tracer*(rTransKp-rTrans)"; rTrans=0
            at k=1, calc_adv_flow.F:116-134)
Multiplying by V^{n+1} (= V^n*rStarExpC) and using the discrete continuity of INTEGR_CONTINUITY/INTEGRATE_FOR_W
(forward_step.F:965; integr_continuity.F:170-182: dEtaHdt = -hDivFlow/rA - mass2rUnit*EmPmR; integrate_for_w.F:123-147:
rA(w_k - w_{k+1}) = conv2d_k - rA*drF*h0FacC_k*rStarDhDt, rStarDhDt = dEtaHdt*recip_Rcol,
integr_continuity.F:239-246; the GM bolus part of uFld/wFld is divergence-free):
    V^{n+1}theta^{n+1} - V^n theta^n = -dt*div(all fluxes) - dt*delta_{k,1}*rA*mass2rUnit*EmPmR*theta^n_1
                                       + dt*V^n*forcing
The surface term cancels against the fresh-water part of surfaceForcingT/S: with staggerTimeStep PmEpR = -EmPmR of
the SAME step (external_forcing_surf.F:122-132) and surfaceForcingT += PmEpR*(temp_EvPrRn - theta_1)*mass2rUnit
(:257-266; salt :268-277). With temp_EvPrRn = salt_EvPrRn = 0 fresh water carries no heat or salt: the
EmPmR*theta_surf terms close EXACTLY in this configuration (algebraically; their size is reported as fw_heat,
fw_salt to show what a wrong closure would miss). Interior flux divergences telescope; what is left:

  VOLUME   sum V^{n+1} - V^n            = -dt*mass2rUnit*sum rA*maskInC*EmPmR                     (dV vs fw)
  SSH      sum rA*maskInC*(etaH^{n+1}-etaH^n) = same (the column form; integr_continuity.F:213-216,
           update_etah.F:62-67: etaH^{n+1} = etaH^n + dt*dEtaHdt with implicDiv2DFlow = 1)
  HEAT     sum V^{n+1}theta^{n+1} - V^n theta^n = dt/(rhoConst*Cp) * sum rA*maskInC*(-Qnet)
           + dt/(rhoConst*Cp) * sum rA*geothermalFlux (k = kLowC, apply_forcing.F(ff):682-695)
           The penetrating shortwave (apply_forcing.F(ff):697-721) is -Qsw*(swfrac(k)*maskC(k) -
           swfrac(k+1)*maskC(k+1)); its column sum telescopes to swfrac(|rF(1)|) = 0.62 + 0.38: nothing leaves
           through the sea floor (maskC below the deepest wet cell is 0, the bottom cell absorbs the rest). Reported
           as sw_bottom = dt/(rho Cp) sum rA*Qsw*(1 - column sum) (0 to round-off); the closure uses -Qnet.
  SALT     sum V^{n+1}S^{n+1} - V^n S^n = -dt*mass2rUnit * sum rA*maskInC*saltFlux
           The salt plume is a redistribution: SALT_PLUME_FORCING_SURF removes saltPlumeFlux at the surface
           (salt_plume_forcing_surf.F:72-73) and SALT_PLUME_TENDENCY_APPLY_S deposits saltPlumeFlux*plumefrac(k)
           below (salt_plume_tendency_apply_s.F:121-156); the fractions telescope to 1 over the levels above the plume
           depth (which is <= the column depth, salt_plume_calc_depth.F:185-191). Reported: plume_gross (the amount
           moved) and plume_net = mass2rUnit*dt*sum rA*saltPlumeFlux*(sum_k plumefrac - 1) (0 to round-off).

Residuals are reported absolute, relative to the gross forcing of the step (sum of |cell terms|): rel_* =
|resid| / gross, and against the round-off floor floor_* = eps*sqrt(sum (V^n c^n)^2 + sum (V^{n+1} c^{n+1})^2) (one
random rounding per cell content). Units: volume m^3; heat K m^3 (x rhoConst*Cp for J); salt (g/kg) m^3.
Measured (mitgcm_jax/tests/test_budgets.py, LLC90, January 1992 forcing, geothermal on): residuals of random sign
without drift; max |resid|/floor over 48 steps: volume 5.4, SSH 0.8, heat 37, salt 7.7 (rel ~1e-13, ~5e-16, 1e-13 to
6e-13, ~1e-10); the Fortran's own steps (ref_ff_jaxdump_v4 dumps) the same (<= 3.5). The terms a wrong closure would
miss are 1e10-1e12 floors (fw_heat ~10 % of the net surface heat flux, fw_salt ~15x the salt flux per step).
The sums are plain jnp reductions over global [tile, ...] arrays (jit on P=1 or on sharded global arrays); inside a
shard_map they would need the exchanger's global sum.

Not included (V4r4 does not execute them; setup raises for each): balanceEmPmR/Qnet, surface relaxation, OBCS,
AddFluid, linear free surface (linFSConserveTr), KPP, sea ice, Adams-Bashforth on tracers. Per-level tracer time steps
must equal deltaTfreesurf (V4r4: all 3600 s) or the continuity term does not cancel; `check_config` tests it.
"""

import jax.numpy as jnp
import numpy as np

from mitgcm_jax.pkgs import salt_plume as sp_mod


def _interior(a, L):
    return a[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx]


def _gsum(x):
    """Sum of a [T, ...] interior array: per tile, then over tiles in tile order."""
    per_tile = jnp.sum(x.reshape(x.shape[0], -1), axis=1)
    return jnp.sum(per_tile)


def _gabs(x):
    return _gsum(jnp.abs(x))


def check_config(P):
    """Host-side: the budget algebra needs dTtracerLev(k) == deltaTFreeSurf for every k (V4r4: 3600 s). Raises
    otherwise (only with concrete parameters)."""
    dts = [float(x) for x in P.th.dTtracerLev]
    dfs = float(P.fs.deltaTFreeSurf)
    if any(d != dfs for d in dts):
        raise NotImplementedError(f"budgets assume dTtracerLev == deltaTFreeSurf ({dfs}); got {sorted(set(dts))}")
    if not P.th.useSALT_PLUME:
        raise NotImplementedError("budgets: useSALT_PLUME=F not handled (V4r4 uses it)")


def cell_volume(g, st, pair="rstar"):
    """Interior cell volumes [T, Nr, sNy, sNx] that pair with st.theta/st.salt: rA*drF*h0FacC*rStarFacC.
    pair="hFacC" uses the State's hFacC (one step behind: the NEGATIVE CONTROL, see the module docstring)."""
    L = g.layout
    rA = _interior(g.rA, L)[:, None]
    drF = jnp.asarray(g.drF)[None, :, None, None]
    if pair == "rstar":
        h = _interior(g.h0FacC, L) * _interior(st.rStarFacC, L)[:, None]
    elif pair == "hFacC":
        h = _interior(st.hFacC, L)
    else:
        raise ValueError(pair)
    return rA * drF * h


def content(g, st, pair="rstar"):
    """Global totals of one State: volume (m^3), heat (K m^3), salt (g/kg m^3), and the SSH integral
    sum rA*maskInC*etaH (m^3) with the wet area (m^2)."""
    L = g.layout
    V = cell_volume(g, st, pair)
    th, s = _interior(st.theta, L), _interior(st.salt, L)
    rAm = _interior(g.rA * g.maskInC, L)
    return {"volume": _gsum(V), "heat": _gsum(V * th), "salt": _gsum(V * s),
            "ssh_int": _gsum(rAm * _interior(st.etaH, L)), "area": _gsum(rAm)}


def _sw_column_fraction(P, g):
    """Column sum of the penetrating-shortwave weights of APPLY_FORCING_T (apply_forcing.F(ff):697-721, as in
    thermodynamics.apply_forcing_t): sum_k swfrac1(k)*maskC(k) - swfrac2(k)*maskC(kp1(k)), [T, sNy, sNx]."""
    L = g.layout
    mC = _interior(g.maskC, L)
    sw1 = jnp.asarray(P.th.swfrac1)[None, :, None, None]
    sw2 = jnp.asarray(P.th.swfrac2)[None, :, None, None]
    kp1 = np.array(P.th.swkp1) - 1
    return jnp.sum(sw1 * mC - sw2 * mC[:, kp1], axis=1)


def _plume_column_fraction(P, g, saltPlumeDepth):
    """Column sum of the salt-plume fractions SALT_PLUME_TENDENCY_APPLY_S deposits (salt_plume_tendency_apply_s.F:
    121-156 via salt_plume.salt_plume_tendency_s's formula): sum over k with SPD > |rF(k)| of
    (FRAC(|rF(k+1)|) - FRAC(|rF(k)|))*maskC(k), [T, sNy, sNx]."""
    L = g.layout
    sp = P.th.sp
    rF = jnp.asarray(g.rF)
    spd = _interior(saltPlumeDepth, L)
    mC = _interior(g.maskC, L)
    tot = jnp.zeros_like(spd)
    for k in range(1, L.Nr + 1):
        a, b = jnp.abs(rF[k - 1]), jnp.abs(rF[k])
        kb1 = sp_mod.salt_plume_frac(sp, -1.0, spd, a)
        kb2 = sp_mod.salt_plume_frac(sp, -1.0, spd, b)
        tot = tot + jnp.where(spd > a, (kb2 - kb1) * mC[:, k - 1], 0.0)
    return tot


def step_budget(P, g, kLowC, st0, st1, pair="rstar", drop=(), add=()):
    """Budgets of the step st0 -> st1 (st1 = FORWARD_STEP(st0)). The step's forcing (EmPmR, Qnet, Qsw, saltFlux,
    saltPlumeFlux, saltPlumeDepth) is read from st1: FORWARD_STEP stores the fields it used.

    Negative-control switches (static tuples; default = the correct closure): drop="fw" drops the fresh-water volume
    flux, "geo" the geothermal heat, "plume_at_depth" counts the salt plume only as its surface removal;
    add="fw_heat" / "fw_salt" adds the heat / salt the fresh water would carry at the local surface values
    (-dt*mass2rUnit*sum rA*EmPmR*theta_1: the right term only for temp_EvPrRn / salt_EvPrRn UNSET; in V4r4 it is
    cancelled exactly). `pair`: which thickness pairs with the tracer (content docstring).
    Returns a dict of float64 jnp scalars (jit with static_argnames=("pair", "drop", "add"))."""
    unknown = (set(drop) - {"fw", "geo", "plume_at_depth"}) | (set(add) - {"fw_heat", "fw_salt"})
    if unknown:
        raise ValueError(f"unknown budget switches {sorted(unknown)}")
    t = {"fw": "fw" not in drop, "geo": "geo" not in drop, "plume_at_depth": "plume_at_depth" not in drop,
         "fw_heat": "fw_heat" in add, "fw_salt": "fw_salt" in add}
    L = g.layout
    dt_fs = P.fs.deltaTFreeSurf
    dt = jnp.asarray(P.th.dTtracerLev)[0]            # == every level and deltaTFreeSurf (check_config)
    m2r = P.fs.mass2rUnit                             # recip_rhoConst (ini_parms.F:1439)
    rcp = P.th.recip_Cp                               # 1/HeatCapacity_Cp
    rA = _interior(g.rA, L)
    rAm = rA * _interior(g.maskInC, L)
    # --- content changes, cell by cell
    V0, V1 = cell_volume(g, st0, pair), cell_volume(g, st1, pair)
    th0, th1 = _interior(st0.theta, L), _interior(st1.theta, L)
    s0, s1 = _interior(st0.salt, L), _interior(st1.salt, L)
    dVc, dHc, dSc = V1 - V0, V1 * th1 - V0 * th0, V1 * s1 - V0 * s0
    detac = rAm * (_interior(st1.etaH, L) - _interior(st0.etaH, L))
    # --- the step's surface fluxes (as stored by FORWARD_STEP)
    EmPmR = _interior(st1.EmPmR, L)
    Qnet, Qsw = _interior(st1.Qnet, L), _interior(st1.Qsw, L)
    sF, spF = _interior(st1.saltFlux, L), _interior(st1.saltPlumeFlux, L)
    geo = _interior(g.geothermalFlux, L) * (_interior(kLowC, L) > 0)
    fwc = -dt_fs * m2r * rAm * EmPmR
    qnetc = -dt * rcp * m2r * rAm * Qnet
    geoc = dt * rcp * m2r * rA * geo
    swcol = _sw_column_fraction(P, g)
    swbot = -dt * rcp * m2r * rAm * Qsw * (1.0 - swcol)       # heat the SW weights would lose below the bottom
    # heat / salt the fresh water would carry if it had the local surface values (temp_EvPrRn, salt_EvPrRn UNSET):
    # in V4r4 (both 0) this is exactly cancelled by the PmEpR*(EvPrRn - tracer) terms; reported for its size
    fwhc = -dt * m2r * rAm * EmPmR * th0[:, 0]
    fwsc = -dt * m2r * rAm * EmPmR * s0[:, 0]
    sfc = -dt * m2r * rAm * sF
    plcol = _plume_column_fraction(P, g, st1.saltPlumeDepth)
    plgross = dt * m2r * rAm * spF
    plnet = plgross * (plcol - 1.0)                            # surface removal + deposit at depth
    out = {}
    # volume / SSH
    fw = _gsum(fwc) if t["fw"] else jnp.zeros(())
    out.update(dV=_gsum(dVc), fw=_gsum(fwc), dEta=_gsum(detac))
    out["resid_volume"] = out["dV"] - fw
    out["resid_ssh"] = out["dEta"] - fw
    out["gross_volume"] = _gabs(fwc) + _gabs(dVc)
    # heat
    heat_in = _gsum(qnetc) + (_gsum(geoc) if t["geo"] else 0.0) + (_gsum(fwhc) if t["fw_heat"] else 0.0)
    out.update(dH=_gsum(dHc), qnet=_gsum(qnetc), geo=_gsum(geoc), sw_bottom=_gsum(swbot), fw_heat=_gsum(fwhc))
    out["resid_heat"] = out["dH"] - heat_in
    out["gross_heat"] = _gabs(qnetc) + _gabs(geoc)
    # salt
    salt_in = _gsum(sfc) + (_gsum(fwsc) if t["fw_salt"] else 0.0)
    if not t["plume_at_depth"]:
        salt_in = salt_in - _gsum(plgross)                     # plume removed at the surface, never deposited
    out.update(dS=_gsum(dSc), saltflux=_gsum(sfc), plume_gross=_gsum(plgross), plume_net=_gsum(plnet),
               fw_salt=_gsum(fwsc))
    out["resid_salt"] = out["dS"] - salt_in
    out["gross_salt"] = _gabs(sfc) + _gabs(plgross)
    for k in ("volume", "heat", "salt"):
        out["rel_" + k] = jnp.abs(out["resid_" + k]) / out["gross_" + k]
    out["rel_ssh"] = jnp.abs(out["resid_ssh"]) / out["gross_volume"]
    # round-off floor: one rounding error (eps) per cell content, added in quadrature (random signs)
    eps = jnp.finfo(jnp.float64).eps
    rms = lambda a, b: jnp.sqrt(_gsum(a * a) + _gsum(b * b))  # noqa: E731
    out["floor_volume"] = eps * rms(V0, V1)
    out["floor_ssh"] = eps * rms(rAm * _interior(st0.etaH, L), rAm * _interior(st1.etaH, L))
    out["floor_heat"] = eps * rms(V0 * th0, V1 * th1)
    out["floor_salt"] = eps * rms(V0 * s0, V1 * s1)
    return out


def budget_acc_init():
    """Accumulator for budgets over many steps (lax.scan carry or Python loop): the additive terms only."""
    return {k: jnp.zeros((), jnp.float64) for k in ADDITIVE}


ADDITIVE = ("dV", "fw", "dEta", "resid_volume", "resid_ssh", "dH", "qnet", "geo", "sw_bottom", "fw_heat",
            "resid_heat", "dS", "saltflux", "plume_gross", "plume_net", "fw_salt", "resid_salt")


def budget_acc_add(acc, b):
    return {k: acc[k] + b[k] for k in ADDITIVE}


def format_budget(b, it):
    """One text line per budget (for run logs)."""
    f = {k: float(v) for k, v in b.items()}
    return (f"%BUDGET it={it} vol dV={f['dV']: .6e} fw={f['fw']: .6e} res={f['resid_volume']: .3e} | "
            f"ssh dEta={f['dEta']: .6e} res={f['resid_ssh']: .3e} | "
            f"heat dH={f['dH']: .6e} qnet={f['qnet']: .6e} geo={f['geo']: .6e} res={f['resid_heat']: .3e} | "
            f"salt dS={f['dS']: .6e} sflx={f['saltflux']: .6e} plume={f['plume_gross']: .3e} "
            f"res={f['resid_salt']: .3e}")
