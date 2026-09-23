"""THERMODYNAMICS / TEMP_INTEGRATE / SALT_INTEGRATE / TRACERS_CORRECTION_STEP (plan Task 16b) against the oracle.

Replay gates: the kernel gets the dumped inputs of the step (theta, salt at S00_begin — unchanged until THERMODYNAMICS;
r* fields after CALC_R_STAR, S11; forcing, IVDConvCount, GGL90diffKr and the GM/Redi tensor after DO_OCEANIC_PHYS,
S04; the residual flow of GMREDI_RESIDUAL_FLOW, T01; the advective tendency of GAD_ADVECTION, T10/T20) and every
intermediate is compared with its dump at every point of the tile arrays (halos included: the Fortran computes and
stores gT_loc and the new theta/salt on the full arrays):
    recip_hFacNew, kappaRk           T13/T23 recip_hFac, kappaRk          (pointwise)
    explicit tendency                T11_temp_gT / T21_salt_gS             (before TIMESTEP_TRACER)
    T + dt*gT                        T12_temp_step / T22_salt_step
    after GAD_IMPLICIT_R             T13_temp_impl / T23_salt_impl
    theta/salt after the routines    T02_temp_integrate / T03_salt_integrate, S13_thermodynamics
    TRACERS_CORRECTION_STEP          S14_tracers_correction
Both oracles, every dumped iteration: SMOKE (no surface forcing; iterations 1, 2) and FORCED (1992 flux forcing,
salt plume active; iterations 1, 2, 3). rStarExpC = rStarFacC(S11)/rStarFacC(S06) is the Fortran expression
(calc_r_star.F:96, 309); it equals the dumped rStarExpC of the next step's G00 bitwise (checked below).

Achieved (2026-09-23, conftest XLA flags --xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp): every one of the 16
compared fields is BITWISE equal to the dump in all 5 cases (max rel. error 0), so the gate asserts bitwise equality;
the tolerance classes (1e-15 pointwise, 1e-13 stencil) remain the bound the negative controls must exceed.
Without --xla_disable_hlo_passes=algsimp the r* divisions (recip_hFacC/rStarExpC, gT/rStarExpC) become multiplications
by 1/rStarExpC: 1-ulp differences, grown to ~6e-14 relative by the implicit solve (measured).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core import thermodynamics as th
from mitgcm_jax.core import tracer_implicit as ti
from mitgcm_jax.core.tracers_correction import TracersCorrectionParams, tracers_correction_step
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.tests import oracle

# FORCED it=1 last: the later tests reuse its cached run
CASES = [(oracle.SMOKE, 1), (oracle.SMOKE, 2), (oracle.FORCED, 2), (oracle.FORCED, 3), (oracle.FORCED, 1)]
TOL_POINTWISE = 1e-15
TOL_STENCIL = 1e-13

S04_FIELDS = ["surfaceForcingT", "surfaceForcingS", "Qsw", "saltPlumeDepth", "saltPlumeFlux", "IVDConvCount",
              "GGL90diffKr", "Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz"]


@functools.lru_cache(maxsize=2)
def _setup(name):
    ds = oracle.dumpset(name)
    nml = RunNamelists(oracle.run_dir(name))
    g = grid_from_dump(ds, min(k[0] for k in ds.index))
    p = th.ThermoParams.from_namelists(nml, g)
    return ds, nml, g, p


def _F(ds, it, stage, name):
    """oracle.field, then forget the records' cached values (DumpSet keeps every loaded record otherwise: ~50 MB per
    3-D field, gigabytes over the gate cases)."""
    a = oracle.field(ds, it, stage, name)
    rec = ds.index.get((it, stage, name))
    for r in (rec or {}).values():
        r.drop()
    return a


def _geothermal(nml, g):
    # geothermalFile = ' ' in both oracles: geothermalFlux stays 0 (ini_forcing / FFIELDS.h); a later oracle with a
    # geothermal file must load it here
    f = nml.get("data", "parm05", "geothermalFile", default=" ").strip()
    assert f == "", f"geothermalFile={f!r}: load the field for this oracle"
    return np.zeros(g.rA.shape)


@functools.lru_cache(maxsize=1)
def _inputs(name, it):
    """Dumped inputs (numpy, shared by the tests: callers replace dict entries, never modify arrays in place)."""
    ds, nml, g, p = _setup(name)
    F = lambda st, n: _F(ds, it, st, n)
    f = {"theta": F("S00_begin", "theta"), "salt": F("S00_begin", "salt")}
    for n in ("recip_hFacC", "hFacW", "hFacS"):
        f[n] = F("S11_calc_rstar", n)
    f["rStarExpC"] = F("S11_calc_rstar", "rStarFacC") / F("S06_update_rstar_T", "rStarFacC")
    for n in S04_FIELDS:
        f[n] = F("S04_oceanic_phys", n)
    f["diffKr"] = g.diffKr
    f["geothermalFlux"] = _geothermal(nml, g)
    flow = [F("T01_residual_flow", n) for n in ("uFld", "vFld", "wFld")]
    adv = [F("T10_temp_adv", "gT_loc"), F("T20_salt_adv", "gS_loc")]
    return f, flow, adv


# params are a jit ARGUMENT (params_pytree: float fields traced), never closed over (KERNEL_GUIDE)
THERMO = jax.jit(th.thermodynamics)
IMPL = jax.jit(ti.gad_implicit_r, static_argnums=(2, 3))


@functools.lru_cache(maxsize=2)
def _run(name, it):
    ds, nml, g, p = _setup(name)
    f, (u, v, w), (gt, gs) = _inputs(name, it)
    out = THERMO(p, g, f, u, v, w, gt, gs)
    return {k: np.asarray(x) for k, x in out.items()}


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / (scale if scale > 0 else 1.0))


# (output key, stage, dumped name, tolerance)
GATES = [
    ("recip_hFacNew", "T13_temp_impl", "recip_hFac", TOL_POINTWISE),
    ("recip_hFacNew", "T23_salt_impl", "recip_hFac", TOL_POINTWISE),
    ("T_kappaRk", "T13_temp_impl", "kappaRk", TOL_POINTWISE),
    ("S_kappaRk", "T23_salt_impl", "kappaRk", TOL_POINTWISE),
    ("T_gExplicit", "T11_temp_gT", "gT_loc", TOL_STENCIL),
    ("S_gExplicit", "T21_salt_gS", "gS_loc", TOL_STENCIL),
    ("T_gStep", "T12_temp_step", "gT_loc", TOL_STENCIL),
    ("S_gStep", "T22_salt_step", "gS_loc", TOL_STENCIL),
    ("T_gImpl", "T13_temp_impl", "gT_loc", TOL_STENCIL),
    ("S_gImpl", "T23_salt_impl", "gS_loc", TOL_STENCIL),
    ("theta", "T02_temp_integrate", "theta", TOL_STENCIL),
    ("salt", "T03_salt_integrate", "salt", TOL_STENCIL),
    ("theta", "S13_thermodynamics", "theta", TOL_STENCIL),
    ("salt", "S13_thermodynamics", "salt", TOL_STENCIL),
]


@pytest.mark.parametrize("name,it", CASES)
def test_gates(name, it):
    """Every intermediate of THERMODYNAMICS + TRACERS_CORRECTION_STEP vs the dumps, full arrays incl. halos."""
    ds, nml, g, p = _setup(name)
    out = _run(name, it)
    errs = {}
    for key, stage, dname, tol in GATES:
        ref = _F(ds, it, stage, dname)
        errs[f"{stage}:{dname}"] = (_relerr(out[key], ref), np.array_equal(out[key], ref), tol)
    # TRACERS_CORRECTION_STEP
    tc = TracersCorrectionParams.from_namelists(nml)
    t14, s14 = tracers_correction_step(tc, out["theta"], out["salt"])
    for arr, n in ((t14, "theta"), (s14, "salt")):
        ref = _F(ds, it, "S14_tracers_correction", n)
        errs[f"S14_tracers_correction:{n}"] = (_relerr(arr, ref), np.array_equal(np.asarray(arr), ref), TOL_STENCIL)
    print(f"\n[{name} it={it}]")
    for k, (e, eq, tol) in errs.items():
        print(f"  {k:40s} rel={e:.3e} bitwise={eq}")
    bad = {k: e for k, (e, eq, tol) in errs.items() if not (e <= tol and eq)}
    assert not bad, bad


def test_rstarexp_matches_next_step_dump():
    """rStarExpC used by the gates (S11/S06 quotient) is the value CALC_R_STAR stored (dumped at G00 of it+1)."""
    ds, nml, g, p = _setup(oracle.FORCED)
    for it in (1, 2):
        mine = oracle.field(ds, it, "S11_calc_rstar", "rStarFacC") / oracle.field(ds, it, "S06_update_rstar_T",
                                                                                   "rStarFacC")
        np.testing.assert_array_equal(mine, oracle.field(ds, it + 1, "G00_geometry", "rStarExpC"))


def test_forcing_terms_active_in_forced_oracle():
    """The FORCED gate exercises surface heat/salt flux, shortwave penetration and the salt plume (non-zero there);
    geothermal flux is 0 in both oracles (geothermalFile = ' ')."""
    ds, nml, g, p = _setup(oracle.FORCED)
    for n in ("surfaceForcingT", "surfaceForcingS", "Qsw", "saltPlumeFlux"):
        assert np.abs(oracle.field(ds, 1, "S04_oceanic_phys", n)).max() > 0, n
    out = _run(oracle.FORCED, 1)
    assert np.abs(out["gsForc"][:, 1:]).max() > 0  # salt plume below the surface level
    assert np.abs(out["gtForc"][:, 1:]).max() > 0  # penetrating shortwave below the surface level


# ------------------------------------------------------------------------------------------------ negative controls
def _replay(name, it, p=None, mutate=None):
    ds, nml, g, p0 = _setup(name)
    p = p or p0
    f, (u, v, w), (gt, gs) = _inputs(name, it)
    f = dict(f)
    if mutate:
        mutate(f)
    out = THERMO(p, g, f, u, v, w, gt, gs)
    return ds, {k: np.asarray(x) for k, x in out.items()}


def test_negative_controls():
    """Planted errors must fail the same comparisons (params are traced: no recompilation per control)."""
    name, it = oracle.FORCED, 1
    ds, nml, g, p = _setup(name)
    ref = lambda st, n: _F(ds, it, st, n)
    # 1) diffKhT perturbed by 1e-6 relative -> explicit temperature tendency fails
    _, o = _replay(name, it, p=dataclasses.replace(p, diffKhT=p.diffKhT * (1 + 1e-6)))
    assert _relerr(o["T_gExplicit"], ref("T11_temp_gT", "gT_loc")) > TOL_STENCIL
    # 2) Redi fluxes dropped (horizontal and off-diagonal tensor components zeroed) -> salt tendency fails
    def no_redi(f):
        for n in ("Kux", "Kvy", "Kuz", "Kvz", "Kwx", "Kwy"):
            f[n] = np.zeros_like(f[n])
    _, o = _replay(name, it, mutate=no_redi)
    assert _relerr(o["S_gExplicit"], ref("T21_salt_gS", "gS_loc")) > TOL_STENCIL
    # 3) salt plume dropped (saltPlumeFlux = 0) -> salt tendency fails (forced oracle only)
    _, o = _replay(name, it, mutate=lambda f: f.update(saltPlumeFlux=np.zeros_like(f["saltPlumeFlux"])))
    assert _relerr(o["S_gExplicit"], ref("T21_salt_gS", "gS_loc")) > TOL_STENCIL
    # 4) implicit solve with the old-time recip_hFacC instead of recip_hFacNew (rStarExpC = 1 in the set-up only)
    o = _run(name, it)
    f, (u, v, w), _ = _inputs(name, it)
    kap = o["T_kappaRk"]
    good = IMPL(p, g, True, p.tempVertAdvScheme, kap, o["recip_hFacNew"], w, o["T_gStep"])
    assert _relerr(good, ref("T13_temp_impl", "gT_loc")) <= TOL_STENCIL
    bad = IMPL(p, g, True, p.tempVertAdvScheme, kap, f["recip_hFacC"], w, o["T_gStep"])
    assert _relerr(bad, ref("T13_temp_impl", "gT_loc")) > TOL_STENCIL
    # 5) kappaRk perturbed by 1e-6 relative -> implicit solution fails
    bad = IMPL(p, g, True, p.tempVertAdvScheme, kap * (1 + 1e-6), o["recip_hFacNew"], w, o["T_gStep"])
    assert _relerr(bad, ref("T13_temp_impl", "gT_loc")) > TOL_STENCIL
    # 6) ivdc_kappa perturbed -> kappaRk fails the pointwise gate
    k2 = th.calc_3d_diffusivity(dataclasses.replace(p, ivdc_kappa=p.ivdc_kappa * (1 + 1e-6)), g, th.GAD_TEMPERATURE,
                                f["IVDConvCount"], f["diffKr"], f["GGL90diffKr"], f["Kwz"])
    assert _relerr(k2, ref("T13_temp_impl", "kappaRk")) > TOL_POINTWISE


# ------------------------------------------------------------------------------------------------ budget closure
def _column_residual(g, recip_hFacNew, before, after):
    """max over interior columns of |sum_k (after - before)*drF/recip_hFacNew| / sum_k |before|*drF/recip_hFacNew
    (wet levels): the column-content change of the implicit step relative to the content (round-off ~1e-16)."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    rh = np.asarray(recip_hFacNew)[..., J, I]
    wet = rh != 0.0
    h = np.where(wet, 1.0 / np.where(wet, rh, 1.0), 0.0) * np.asarray(g.drF)[None, :, None, None]
    b = np.asarray(before)[..., J, I]
    dlt = np.asarray(after)[..., J, I] - b
    res = np.abs(np.sum(dlt * h, axis=1))
    tot = np.sum(np.abs(b) * h, axis=1)
    return float(np.max(res / np.where(tot > 0, tot, 1.0)))


def test_implicit_budget_closure():
    """GAD_IMPLICIT_R (U3 implicit advection + implicit diffusion, flux form) conserves the column content
    sum_k T*hFacNew*drF exactly up to round-off; a planted non-conservative error (drop the e5d diagonal) breaks it."""
    name, it = oracle.FORCED, 1
    ds, nml, g, p = _setup(name)
    o = _run(name, it)
    rel = _column_residual(g, o["recip_hFacNew"], o["T_gStep"], o["T_gImpl"])
    print(f"\nimplicit column budget residual (rel): {rel:.3e}")
    # round-off of the solve scales with the matrix norm (dt*kappaRk/(h*drF*drC) reaches ~1e2 with ivdc_kappa = 10)
    assert rel < 1e-12
    f, (u, v, w), _ = _inputs(name, it)
    a, b, c, d, e = ti.gad_implicit_r_matrix(p, g, True, p.tempVertAdvScheme, o["T_kappaRk"], o["recip_hFacNew"], w)
    bad = ti.solve_pentadiagonal(a, b, c, d, jnp.zeros_like(e), o["T_gStep"])
    rel_bad = _column_residual(g, o["recip_hFacNew"], o["T_gStep"], bad)
    print(f"  negative control (e5d dropped): {rel_bad:.3e}")
    assert rel_bad > 1e-9


# ------------------------------------------------------------------------------------------------ gradients
def _tile0(g, arrays):
    """Restrict a Grid and arrays to tile 0 (THERMODYNAMICS has no exchange: tiles are independent)."""
    from mitgcm_jax.grid.geometry import Grid

    T = g.rA.shape[0]
    gf = {k: (v[:1] if (np.ndim(v) >= 3 and np.shape(v)[0] == T) else v) for k, v in g.f.items()}
    cut = lambda x: x[:1] if (np.ndim(x) >= 3 and np.shape(x)[0] == T) else x
    return Grid(gf, g.layout), jax.tree_util.tree_map(cut, arrays)


def _wet_points(g1, levels):
    """One interior wet point of tile 0 per (0-based) level, deterministic."""
    L = g1.layout
    m = np.asarray(g1.maskC)[0]
    pts = []
    for k in levels:
        jj, ii = np.nonzero(m[k, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx])
        n = len(jj) // 2
        pts.append((0, k, int(jj[n]) + L.OLy, int(ii[n]) + L.OLx))
    return pts


def _fd_check(fun, x, points, hs):
    """Central differences of fun at x along unit vectors at `points` for each h; returns (grad values, fd table)."""
    gr = np.asarray(jax.grad(fun)(x))
    table = []
    for pt in points:
        row = []
        for h in hs:
            e = np.zeros(x.shape)
            e[pt] = 1.0
            row.append((float(fun(x + h * e)) - float(fun(x - h * e))) / (2 * h))
        table.append(row)
    return gr, np.array(table)


def test_implicit_solver_gradient():
    """d/d(kappaRk) and d/d(rhs) of a weighted sum of the GAD_IMPLICIT_R solution: finite on every lane (halo, dry)
    and equal to central finite differences at wet points (plateau reported)."""
    name, it = oracle.FORCED, 1
    ds, nml, g, p = _setup(name)
    o = _run(name, it)
    f, (u, v, w), _ = _inputs(name, it)
    g1, (kap, rh, ww, y) = _tile0(g, (o["T_kappaRk"], o["recip_hFacNew"], w, o["T_gStep"]))
    rng = np.random.default_rng(1)
    wts = jnp.asarray(rng.normal(size=y.shape))
    solve = jax.jit(lambda kap_, y_: ti.gad_implicit_r(p, g1, True, p.tempVertAdvScheme, kap_, rh, ww, y_))
    Lk = lambda kap_: jnp.sum(wts * solve(kap_, y))
    Ly = lambda y_: jnp.sum(wts * solve(kap, y_))
    pts = _wet_points(g1, [1, 5, 20])
    hs_k = [1e-4, 1e-5, 1e-6]
    gk, tk = _fd_check(Lk, jnp.asarray(kap), pts, hs_k)
    assert np.all(np.isfinite(gk))
    print("\nkappaRk: grad vs FD(h=1e-4,1e-5,1e-6)")
    for pt, row in zip(pts, tk):
        print(f"  {pt}: grad={gk[pt]:.10e} fd={row}")
        best = min(abs(r - gk[pt]) for r in row)
        assert best <= 1e-6 * max(abs(gk[pt]), 1e-12), (pt, gk[pt], row)
    gy, ty = _fd_check(Ly, jnp.asarray(y), pts, [1e-2, 1e-3])
    assert np.all(np.isfinite(gy))
    for pt, row in zip(pts, ty):  # linear in y: FD exact up to round-off
        assert min(abs(r - gy[pt]) for r in row) <= 1e-8 * max(abs(gy[pt]), 1e-12), (pt, gy[pt], row)


def test_thermodynamics_gradient_finite():
    """Weighted sum of the new theta and salt of THERMODYNAMICS on tile 0: gradient w.r.t. the input theta (affine map:
    central FD exact up to round-off) and w.r.t. the Redi coefficient Kwz (nonlinear, through kappaRk and the implicit
    solve) is finite on every lane and matches central finite differences at wet points."""
    name, it = oracle.FORCED, 1
    ds, nml, g, p = _setup(name)
    f, (u, v, w), (gt, gs) = _inputs(name, it)
    g1, (f1, u1, v1, w1, gt1, gs1) = _tile0(g, (f, u, v, w, gt, gs))
    f1 = {k: jnp.asarray(x) for k, x in f1.items()}
    u1, v1, w1, gt1, gs1 = (jnp.asarray(x) for x in (u1, v1, w1, gt1, gs1))
    rng = np.random.default_rng(2)
    wt = jnp.asarray(rng.normal(size=f1["theta"].shape))
    ws = jnp.asarray(rng.normal(size=f1["theta"].shape))

    def loss(pp, theta, Kwz, wt_, ws_):
        ff = dict(f1, theta=theta, Kwz=Kwz)
        out = th.thermodynamics(pp, g1, ff, u1, v1, w1, gt1, gs1)
        return jnp.sum(wt_ * out["theta"]) + jnp.sum(ws_ * out["salt"])

    vg = jax.jit(jax.value_and_grad(loss, argnums=(1, 2)))
    _, (gth, gk) = vg(p, f1["theta"], f1["Kwz"], wt, ws)
    gth, gk = np.asarray(gth), np.asarray(gk)
    assert np.all(np.isfinite(gth)) and np.all(np.isfinite(gk))
    val = lambda th_, k_, a, b: float(vg(p, th_, k_, a, b)[0])
    for pt in _wet_points(g1, [0, 3, 30]):
        e = jnp.zeros_like(f1["theta"]).at[pt].set(1.0)
        rows = [(val(f1["theta"] + h * e, f1["Kwz"], wt, ws) - val(f1["theta"] - h * e, f1["Kwz"], wt, ws)) / (2 * h)
                for h in (1e-1, 1e-2)]
        print(f"\ntheta {pt}: grad={gth[pt]:.12e} fd={rows}")
        assert min(abs(r - gth[pt]) for r in rows) <= 1e-8 * abs(gth[pt])
    for pt in _wet_points(g1, [5, 20]):
        # Kwz(i,j,k) acts on its own column only (kappaRk): localise the weights there so that the loss is O(1) and
        # the FD round-off (eps*|loss|/h) stays below the truncation error
        col = jnp.zeros_like(wt).at[pt[0], :, pt[2], pt[3]].set(1.0)
        _, (_, gkc) = vg(p, f1["theta"], f1["Kwz"], wt * col, ws * col)
        gkc = float(np.asarray(gkc)[pt])
        e = jnp.zeros_like(f1["Kwz"]).at[pt].set(1.0)
        rows = [(val(f1["theta"], f1["Kwz"] + h * e, wt * col, ws * col)
                 - val(f1["theta"], f1["Kwz"] - h * e, wt * col, ws * col)) / (2 * h) for h in (1e-4, 1e-5, 1e-6, 1e-7)]
        print(f"Kwz {pt}: grad={gkc:.12e} (full-weight grad {gk[pt]:.12e}) fd={rows}")
        np.testing.assert_allclose(gkc, gk[pt], rtol=1e-12)
        # h-sweep: truncation O(h^2) above 1e-5, round-off eps*|loss|/h below 1e-6; measured plateau ~1.4e-7 rel.
        assert min(abs(r - gkc) for r in rows) <= 1e-6 * abs(gkc)
