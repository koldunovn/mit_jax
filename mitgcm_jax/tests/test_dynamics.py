"""DYNAMICS gates (plan Tasks 14b/14c): CALC_VISCOSITY, TIMESTEP (AB3), IMPLDIFF and the DYNAMICS driver, replayed
against the Fortran oracle on every point of every tile (halos included), both oracles, every dumped iteration.

Inputs: `test_phi_hyd.dynamics_inputs` (fields at DYNAMICS entry: S00/S01/S04 dumps), the momentum tendencies of
MOM_VECINV replayed from D00b_mom_vecinv (gU, gV, guDissip, gvDissip per level), the hydrostatic pressure gradient
from D00a_phi_hyd where TIMESTEP is gated alone. Outputs compared with
  D01_before_impl_visc  kappaRU/kappaRV, gU/gV after TIMESTEP, AB histories guNm_1/2, gvNm_1/2
  D02_after_impl_visc   gU/gV after IMPLDIFF
  S05_dynamics          gU/gV, histories, totPhiHyd, phiHydLow at the end of DYNAMICS (driver).
The oracle's nIter0 = 1 with a complete pickup gives mom_StartAB = 1 (check_pickup.F:66), so iteration 1 is an AB2
step and iterations >= 2 are AB3 steps (adams_bashforth3.F:91-99): both weight sets are gated.

Achieved (XLA_FLAGS --xla_cpu_max_isa=AVX from conftest.py, parameters as traced pytree leaves): bitwise equality of
kappaRU/kappaRV, gU/gV after TIMESTEP, guNm/gvNm, gU/gV after IMPLDIFF and the driver's S05 fields, at iterations 1-2
(SMOKE) and 1-3 (FORCED). Before the parameters were traced, XLA simplified viscArNr + (GGL90viscArU - viscArNr) to
GGL90viscArU (720 points 1 ulp off) and reassociated the constant deltaTMom in IMPLDIFF's a/c (59675 points, 5e-16).
Full V4r4 tree (oracle.FULL, iterations 1-3; M2.6a): the same gates. The momentum forcing surfaceForcingU/V (S04)
there includes the sea-ice ocean stress (SEAICE_OCEAN_STRESS inside SEAICE_MODEL); DYNAMICS / TIMESTEP / IMPLDIFF
take the c66g branches of the flux-forced port (the full tree's dynamics.F override adds two diagnostics fills; the
ff overrides of apply_forcing / impldiff / momentum_correction_step are diagnostics-only and not used by the full
tree). Bitwise (measured 2026-09-23); negative control: the momentum forcing without the ice stress (S02 values,
before SEAICE_MODEL) fails the TIMESTEP gate.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core.dynamics import DynamicsParams, calc_viscosity, dynamics
from mitgcm_jax.core.implicit import impldiff, impldiff_deltaTX
from mitgcm_jax.core.phi_hyd import kLowC_from_hFac
from mitgcm_jax.core.timestep import ab3_weights, adams_bashforth3, apply_forcing_uv, timestep
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.tests import oracle
from mitgcm_jax.tests.test_phi_hyd import dynamics_inputs, fd_check, relerr, wet_points

ORACLES = [oracle.SMOKE, oracle.FORCED, oracle.FULL]
TOL = 0.0           # achieved: bitwise on every gate (see module docstring)
TOL_POINT = 1e-15   # KERNEL_GUIDE class of pointwise maps (kappa, AB3 + TIMESTEP): planted errors must exceed it
TOL_CLASS = 1e-13   # KERNEL_GUIDE class of the tridiagonal solve / vertical integrals


def make_case(name):
    ds = oracle.dumpset(name)
    nml = RunNamelists(oracle.run_dir(name))
    its = sorted({k[0] for k in ds.index if k[1] == "D01_before_impl_visc"})
    g = jax.tree_util.tree_map(jnp.asarray, grid_from_dump(ds, its[0]))
    return dict(name=name, ds=ds, nml=nml, its=its, g=g, p=DynamicsParams.from_namelists(nml),
                kLowC=jnp.asarray(kLowC_from_hFac(g.h0FacC)), cache={})


@pytest.fixture(scope="module")
def cases():
    return {}  # module-scoped: the loaded oracle data is released after this module


@pytest.fixture(params=ORACLES)
def case(request, cases):
    if request.param not in cases:
        cases[request.param] = make_case(request.param)
    return cases[request.param]


@pytest.fixture
def forced(cases):
    """Negative controls, gradients and the rest state run on the forced oracle only (AB2 and AB3 steps)."""
    if oracle.FORCED not in cases:
        cases[oracle.FORCED] = make_case(oracle.FORCED)
    return cases[oracle.FORCED]


def inputs(case, it):
    """Fields at DYNAMICS entry + MOM_VECINV tendencies (D00b) + phi_hyd gradients (D00a) of step `it` (cached)."""
    if it not in case["cache"]:
        ds = case["ds"]
        s = dynamics_inputs(ds, it)
        for n in ("gU", "gV", "guDissip", "gvDissip"):
            s["D00b_" + n] = oracle.field(ds, it, "D00b_mom_vecinv", n)
        for n in ("dPhiHydX", "dPhiHydY"):
            s["D00a_" + n] = oracle.field(ds, it, "D00a_phi_hyd", n)
        case["cache"][it] = s
    return case["cache"][it]


def ref(case, it, stage, name):
    return oracle.field(case["ds"], it, stage, name)


def hist(case, it, stage, c):
    return np.stack([ref(case, it, stage, f"g{c}Nm_1"), ref(case, it, stage, f"g{c}Nm_2")])


@functools.partial(jax.jit, static_argnames=("myIter",))
def _timestep(ts, g, myIter, s):
    guExt, gvExt = apply_forcing_uv(ts, g, s["surfaceForcingU"], s["surfaceForcingV"], s["recip_hFacW"],
                                    s["recip_hFacS"])
    z = jnp.zeros(s["etaH"].shape)
    return timestep(ts, g, myIter, s["D00b_gU"], s["D00b_gV"], s["guNm"], s["gvNm"], s["uVel"], s["vVel"],
                    s["D00a_dPhiHydX"], s["D00a_dPhiHydY"], z, z, s["D00b_guDissip"], s["D00b_gvDissip"],
                    guExt, gvExt)


def timestep_errors(case, it, ts=None, s=None):
    s = inputs(case, it) if s is None else s
    gU, gV, guNm, gvNm = _timestep(ts or case["p"].ts, case["g"], it, s)
    D01 = "D01_before_impl_visc"
    return dict(gU=relerr(gU, ref(case, it, D01, "gU")), gV=relerr(gV, ref(case, it, D01, "gV")),
                guNm=relerr(guNm, hist(case, it, D01, "u")), gvNm=relerr(gvNm, hist(case, it, D01, "v")))


@jax.jit
def _impldiff_uv(g, deltaTMom, kU, kV, rhW, rhS, gU, gV):
    L = g.layout
    dTX = impldiff_deltaTX(-1, deltaTMom, Nr=L.Nr)
    rng = dict(iMin=0, iMax=L.sNx + 1, jMin=0, jMax=L.sNy + 1)
    return impldiff(g, -1, kU, rhW, gU, dTX, **rng), impldiff(g, -2, kV, rhS, gV, dTX, **rng)


def impldiff_errors(case, it, shift=0):
    s = inputs(case, it)
    D01, D02 = "D01_before_impl_visc", "D02_after_impl_visc"
    kU, kV = ref(case, it, D01, "kappaRU"), ref(case, it, D01, "kappaRV")
    if shift:
        kU = np.roll(kU, shift, axis=1)
    gU, gV = _impldiff_uv(case["g"], case["p"].ts.deltaTMom, kU, kV, s["recip_hFacW"], s["recip_hFacS"],
                          ref(case, it, D01, "gU"), ref(case, it, D01, "gV"))
    return dict(gU=relerr(gU, ref(case, it, D02, "gU")), gV=relerr(gV, ref(case, it, D02, "gV")))


def test_calc_viscosity_matches_fortran(case):
    """kappaRU/kappaRV (CALC_VISCOSITY + GGL90_CALC_VISC, Nr+1 levels) vs D01_before_impl_visc."""
    f = jax.jit(calc_viscosity)
    for it in case["its"]:
        s = inputs(case, it)
        kU, kV = f(case["p"], case["g"], s["GGL90viscArU"], s["GGL90viscArV"])
        e = dict(kappaRU=relerr(kU, ref(case, it, "D01_before_impl_visc", "kappaRU")),
                 kappaRV=relerr(kV, ref(case, it, "D01_before_impl_visc", "kappaRV")))
        print(case["name"], it, e)
        assert all(v <= TOL for v in e.values()), (it, e)


def test_timestep_matches_fortran(case):
    """TIMESTEP (APPLY_FORCING + AB3 + step) replayed from D00b tendencies and D00a gradients vs D01 gU, gV and the
    rotated AB histories; iteration 1 exercises the AB2 start-up weights, later iterations AB3."""
    for it in case["its"]:
        e = timestep_errors(case, it)
        print(case["name"], it, {k: f"{v:.2e}" for k, v in e.items()})
        assert all(v <= TOL for v in e.values()), (it, e)


def test_impldiff_matches_fortran(case):
    """IMPLDIFF(-1, kappaRU, recip_hFacW, gU) and (-2, kappaRV, recip_hFacS, gV): D01 -> D02."""
    for it in case["its"]:
        e = impldiff_errors(case, it)
        print(case["name"], it, {k: f"{v:.2e}" for k, v in e.items()})
        assert all(v <= TOL for v in e.values()), (it, e)


@functools.partial(jax.jit, static_argnames=("myIter",))
def _dynamics_replay(p, g, kLowC, s, myIter):
    """DYNAMICS with MOM_VECINV replayed from the D00b dumps carried in s (all arrays are jit arguments)."""
    replay = lambda kU, kV: (s["D00b_gU"], s["D00b_gV"], s["D00b_guDissip"], s["D00b_gvDissip"])  # noqa: E731
    return dynamics(p, g, kLowC, s, replay, myIter)


def test_dynamics_driver_matches_fortran(case):
    """The DYNAMICS driver (MOM_VECINV replayed from D00b) vs D01, D02 and S05_dynamics."""
    p, g = case["p"], case["g"]
    for it in case["its"]:
        o = _dynamics_replay(p, g, case["kLowC"], inputs(case, it), it)
        D01, S05 = "D01_before_impl_visc", "S05_dynamics"
        e = dict(kappaRU=relerr(o["kappaRU"], ref(case, it, D01, "kappaRU")),
                 gU_explicit=relerr(o["gU_explicit"], ref(case, it, D01, "gU")),
                 gV_explicit=relerr(o["gV_explicit"], ref(case, it, D01, "gV")),
                 gU=relerr(o["gU"], ref(case, it, S05, "gU")), gV=relerr(o["gV"], ref(case, it, S05, "gV")),
                 guNm=relerr(o["guNm"], hist(case, it, S05, "u")), gvNm=relerr(o["gvNm"], hist(case, it, S05, "v")),
                 totPhiHyd=relerr(o["totPhiHyd"], ref(case, it, S05, "totPhiHyd")),
                 phiHydLow=relerr(o["phiHydLow"], ref(case, it, S05, "phiHydLow")))
        print(case["name"], it, {k: f"{v:.2e}" for k, v in e.items()})
        assert all(v <= TOL for v in e.values()), (it, e)


def test_negative_controls(forced):
    """Planted errors fail the same comparisons: beta_AB * (1 + 1e-6); swapped history slots; AB3 weights at the
    AB2 start-up step (mom_StartAB wrong); kappaRU shifted one level in IMPLDIFF."""
    case = forced
    ts = case["p"].ts
    it = case["its"][-1]
    assert it >= ts.nIter0 + 1  # an AB3 step, where beta_AB matters
    e = timestep_errors(case, it, dataclasses.replace(ts, beta_AB=ts.beta_AB * (1 + 1e-6)))
    assert e["gU"] > TOL_POINT and e["gV"] > TOL_POINT, e
    s = inputs(case, it)
    e = timestep_errors(case, it, s=dict(s, guNm=s["guNm"][::-1], gvNm=s["gvNm"][::-1]))
    assert e["gU"] > TOL_POINT and e["guNm"] > TOL_POINT, e
    e = impldiff_errors(case, it, shift=1)
    assert e["gU"] > TOL_CLASS, e
    it0 = case["its"][0]
    assert it0 == ts.nIter0 and ts.mom_StartAB == 1  # the oracle's AB2 start-up step
    e = timestep_errors(case, it0, dataclasses.replace(ts, mom_StartAB=ts.nIter0 + 7))
    assert e["gU"] > TOL_POINT and e["gV"] > TOL_POINT, e


def test_ab3_weights_startup():
    """adams_bashforth3.F:87-100 for the three start-up cases, Python-int and traced myIter agree."""
    a, b = 0.5, 0.281105
    ab3 = (a + b, -a - 2. * b, b)
    cases = [(0, 0, 0, (0., 0., 0.)), (0, 0, 1, (a, -a, 0.)), (0, 0, 2, ab3),
             (1, 1, 1, (a, -a, 0.)), (1, 1, 2, ab3), (5, 5, 5, ab3), (5, 1, 5, (a, -a, 0.)), (5, 0, 6, (a, -a, 0.))]
    f = jax.jit(ab3_weights, static_argnums=(1, 2, 3, 4))
    for nIter0, startAB, myIter, want in cases:
        assert ab3_weights(myIter, nIter0, startAB, a, b) == want, (nIter0, startAB, myIter)
        got = f(jnp.int32(myIter), nIter0, startAB, a, b)
        assert tuple(float(x) for x in got) == want


def test_ab3_history_rotation():
    """Consecutive calls: slot m2 = 1+MOD(myIter,2) receives the current tendency, slot m1 holds n-1
    (adams_bashforth3.F:83-84, :121-125)."""
    rng = np.random.default_rng(0)
    gm2, gm1, g2, g3 = (rng.normal(size=(3, 4)) for _ in range(4))
    a, b = 0.5, 0.281105
    ab0, ab1, ab2 = a + b, -a - 2. * b, b
    h0 = np.stack([gm2, gm1])  # myIter=2: m1 = 2 (index 1) holds n-1, m2 = 1 (index 0) holds n-2
    out, h, _ = adams_bashforth3(g2, h0, 2, 0, 0, a, b)
    np.testing.assert_array_equal(out, g2 + (ab0 * g2 + ab1 * gm1 + ab2 * gm2))
    np.testing.assert_array_equal(h[0], g2)
    np.testing.assert_array_equal(h[1], gm1)
    out, h, _ = adams_bashforth3(g3, h, 3, 0, 0, a, b)  # myIter=3: m1 = index 0 (n-1 = g2), m2 = index 1 (gm1)
    np.testing.assert_array_equal(out, g3 + (ab0 * g3 + ab1 * g2 + ab2 * gm1))
    np.testing.assert_array_equal(h[1], g3)


@jax.jit
def _impldiff_u(g, deltaTMom, kU, rhW, x):
    L = g.layout
    return impldiff(g, -1, kU, rhW, x, impldiff_deltaTX(-1, deltaTMom, Nr=L.Nr), 0, L.sNx + 1, 0, L.sNy + 1)


def test_impldiff_conservation_rest_and_gradient(forced):
    """IMPLDIFF on the oracle's kappaRU / recip_hFacW: a zero field stays zero (rest stays at rest); the column
    integral sum_k drF hFacW x is conserved (no flux through the surface, the bottom or dry interfaces);
    d/d(KappaRX, gTracer) of a weighted sum is finite on every lane and matches central finite differences at wet
    points."""
    case = forced
    it = case["its"][0]
    g = case["g"]
    L = g.layout
    s = inputs(case, it)
    dtm = case["p"].ts.deltaTMom
    D01 = "D01_before_impl_visc"
    kU, x, rh = ref(case, it, D01, "kappaRU"), ref(case, it, D01, "gU"), s["recip_hFacW"]
    assert np.all(np.asarray(_impldiff_u(g, dtm, kU, rh, np.zeros_like(x))) == 0.0)
    y = np.asarray(_impldiff_u(g, dtm, kU, rh, x))
    w = np.asarray(g.drF)[None, :, None, None] * s["hFacW"] * np.asarray(g.maskW)
    J, I = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
    before, after = np.sum(w * x, axis=1)[:, J, I], np.sum(w * y, axis=1)[:, J, I]
    col = np.sum(np.abs(w * x), axis=1)[:, J, I]
    cons = float(np.max(np.abs(after - before) / np.where(col > 0, col, 1.0)))
    print("column-integral change / column abs integral:", cons)
    assert cons < 1e-13
    wt = jnp.asarray(np.random.default_rng(2).normal(size=x.shape))
    obj = lambda k_, x_, wt_, g_, rh_, dt_: jnp.sum(wt_ * _impldiff_u(g_, dt_, k_, rh_, x_))  # noqa: E731
    gk, gx = jax.jit(jax.grad(obj, argnums=(0, 1)))(kU, x, wt, g, rh, dtm)
    assert np.all(np.isfinite(gk)) and np.all(np.isfinite(gx))
    for (t, k, j, i) in wet_points(g, grid="W", levels=(5, 20)):
        k0 = float(kU[t, k, j, i])
        best, rows = fd_check(lambda v: (_impldiff_u(g, dtm, v, rh, x),), kU, gk, (t, k, j, i),
                              tuple(k0 * r for r in (1e-2, 1e-3, 1e-4, 1e-5)), (wt,))
        print("kappaRU", (t, k, j, i), rows)
        assert best < 1e-6, rows
        best, rows = fd_check(lambda v: (_impldiff_u(g, dtm, kU, rh, v),), x, gx, (t, k, j, i),
                              (1e-2, 1e-3, 1e-4), (wt,))
        print("gU", (t, k, j, i), rows)
        assert best < 1e-8, rows


def test_dynamics_rest_state(forced):
    """Ocean at rest, level-only density, flat surface, no forcing, zero tendencies and histories: DYNAMICS returns
    gU = gV = 0 exactly (the predicted velocity stays at rest)."""
    case = forced
    p, g = case["p"], case["g"]
    it = case["its"][-1]
    s = {k: v for k, v in inputs(case, it).items() if not k.startswith("D00a_")}
    rho = s["rhoInSitu"]
    zero3 = np.zeros_like(rho)
    rhoK = np.broadcast_to(np.linspace(1.0, 30.0, rho.shape[1])[None, :, None, None], rho.shape)
    s.update(uVel=zero3, vVel=zero3, guNm=np.zeros_like(s["guNm"]), gvNm=np.zeros_like(s["gvNm"]),
             rhoInSitu=np.array(rhoK), etaH=np.zeros_like(s["etaH"]), rStarFacC=np.ones_like(s["rStarFacC"]),
             phi0surf=np.zeros_like(s["phi0surf"]), surfaceForcingU=np.zeros_like(s["etaH"]),
             surfaceForcingV=np.zeros_like(s["etaH"]))
    s.update(D00b_gU=zero3, D00b_gV=zero3, D00b_guDissip=zero3, D00b_gvDissip=zero3)
    o = _dynamics_replay(p, g, case["kLowC"], s, it)
    assert np.all(np.asarray(o["gU"]) == 0.0) and np.all(np.asarray(o["gV"]) == 0.0)
    assert np.all(np.asarray(o["dPhiHydX"]) == 0.0) and np.all(np.asarray(o["dPhiHydY"]) == 0.0)


def test_dynamics_gradient(forced):
    """d/d(uVel, rhoInSitu, GGL90viscArU) of sum(w*gU) + sum(w*gV) through the whole driver (phi_hyd, TIMESTEP,
    IMPLDIFF; MOM_VECINV replayed as a constant): finite on every lane, and equal to central finite differences
    at a wet point for the nonlinear dependence on GGL90viscArU (via kappaRU in IMPLDIFF)."""
    case = forced
    p, g = case["p"], case["g"]
    it = case["its"][-1]
    s = {k: v for k, v in inputs(case, it).items() if not k.startswith("D00a_")}
    rng = np.random.default_rng(3)
    wU = jnp.asarray(rng.normal(size=s["uVel"].shape))
    wV = jnp.asarray(rng.normal(size=s["vVel"].shape))

    def outs(u, rho, vU, s_, p_, g_, kLowC_):
        s2 = dict(s_, uVel=u, rhoInSitu=rho, GGL90viscArU=vU)
        o = dynamics(p_, g_, kLowC_, s2, lambda kU, kV: (s2["D00b_gU"], s2["D00b_gV"], s2["D00b_guDissip"],
                                                        s2["D00b_gvDissip"]), it)
        return o["gU"], o["gV"]

    def J(u, rho, vU, wU_, wV_, *rest):
        a, b = outs(u, rho, vU, *rest)
        return jnp.sum(wU_ * a) + jnp.sum(wV_ * b)

    rest = (s, p, g, case["kLowC"])
    grads = jax.jit(jax.grad(J, argnums=(0, 1, 2)))(s["uVel"], s["rhoInSitu"], s["GGL90viscArU"], wU, wV, *rest)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in grads)
    outs_j = jax.jit(outs)
    (t, k, j, i), = wet_points(g, grid="W", levels=(10,), tiles=(2,))
    v0 = float(s["GGL90viscArU"][t, k, j, i])
    best, rows = fd_check(lambda v: outs_j(s["uVel"], s["rhoInSitu"], v, *rest), s["GGL90viscArU"], grads[2],
                          (t, k, j, i), tuple(v0 * r for r in (1e-2, 1e-3, 1e-4, 1e-5)), (wU, wV))
    print("GGL90viscArU", (t, k, j, i), rows)
    assert best < 1e-6, rows


def test_full_negative_control_ice_stress(cases):
    """Full tree: TIMESTEP with surfaceForcingU/V of S02_load_fields (the EXF stress before SEAICE_MODEL replaced it
    under ice) fails the D01 gU/gV comparison; with the S04 values it passes."""
    if oracle.FULL not in cases:
        cases[oracle.FULL] = make_case(oracle.FULL)
    case = cases[oracle.FULL]
    it = case["its"][0]
    s = inputs(case, it)
    assert max(timestep_errors(case, it, s=s).values()) <= TOL
    ds = case["ds"]
    pre = {n: oracle.field(ds, it, "S02_load_fields", n) for n in ("surfaceForcingU", "surfaceForcingV")}
    assert not np.array_equal(pre["surfaceForcingU"], s["surfaceForcingU"])
    e = timestep_errors(case, it, s=dict(s, **pre))
    assert e["gU"] > TOL_POINT and e["gV"] > TOL_POINT, e
