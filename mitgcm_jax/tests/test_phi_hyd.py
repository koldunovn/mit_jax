"""CALC_PHI_HYD gates (plan Task 14b-phi_hyd): replay of the Fortran oracle's hydrostatic pressure, all levels.

Inputs are the oracle's own fields at DYNAMICS entry (`dynamics_inputs`): rhoInSitu, totPhiHyd, phi0surf,
rStarFacC from S04_oceanic_phys (after DO_OCEANIC_PHYS), etaH from S01_update_rstar_F, grid from G00_geometry.
Compared with D00a_phi_hyd (per level, after each CALL CALC_PHI_HYD: dPhiHydX, dPhiHydY, phiHydC, phiHydF) and with
S05_dynamics (totPhiHyd, phiHydLow), on every point of every tile (halos included: points outside the Fortran loop
range must hold what the Fortran arrays hold). Both oracles (SMOKE: no forcing, phi0surf = 0; FORCED: 1992 flux
forcing with atmospheric loading, phi0surf != 0), every dumped iteration.

Achieved (XLA_FLAGS --xla_cpu_max_isa=AVX from conftest.py, parameters traced): bitwise equality of phiHydC,
phiHydF, dPhiHydX, dPhiHydY (all 50 levels), totPhiHyd and phiHydLow at iterations 1-2 (SMOKE) and 1-3 (FORCED).
With XLA's default FMA contraction dPhiHydX/Y differ by 2e-13 relative (1-ulp phiHydC differences amplified by the
horizontal difference of large phiHydC*rStarFacC), which is why the gate needs the no-FMA flag.
Full V4r4 tree (oracle.FULL, iterations 1-3; M2.6a): the same gate. phi0surf there carries the sea-ice load
(phi0surf = (pLoad + sIceLoad*gravity)/rhoConst, external_forcing_surf.F:356-366, sIceLoad from SEAICE_GROWTH); the
full-tree dynamics.F override differs from c66g only in two diagnostics fills. Bitwise (measured 2026-09-23);
negative control: phi0surf without its sIceLoad term fails totPhiHyd/phiHydLow.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core.phi_hyd import PhiHydParams, calc_phi_hyd, kLowC_from_hFac
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.tests import oracle

ORACLES = [oracle.SMOKE, oracle.FORCED, oracle.FULL]
TOL = 0.0          # achieved: bitwise equality on every field, point, level and iteration of both oracles
TOL_CLASS = 1e-13  # KERNEL_GUIDE class of stencils / vertical integrals: planted errors must exceed it


def iterations(ds):
    return sorted({k[0] for k in ds.index if k[1] == "D00a_phi_hyd"})


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / (scale if scale > 0 else 1.0))


def dynamics_inputs(ds, it):
    """The fields DYNAMICS reads, as the oracle held them at DYNAMICS entry of step `it` (Fortran names)."""
    f = lambda st, n: oracle.field(ds, it, st, n)  # noqa: E731
    S04, S01, S00 = "S04_oceanic_phys", "S01_update_rstar_F", "S00_begin"
    s = {n: f(S01, n) for n in ("uVel", "vVel", "etaH", "etaN")}
    s.update({n: f(S04, n) for n in ("rStarFacC", "hFacW", "hFacS", "rhoInSitu", "totPhiHyd", "phi0surf",
                                     "surfaceForcingU", "surfaceForcingV", "GGL90viscArU", "GGL90viscArV")})
    s["guNm"] = np.stack([f(S00, "guNm_1"), f(S00, "guNm_2")])
    s["gvNm"] = np.stack([f(S00, "gvNm_1"), f(S00, "gvNm_2")])
    # recip_hFacW/S as UPDATE_R_STAR (update_r_star.F:76-77, :111-112; USE_MASK_AND_NO_IF undefined) left them for
    # this step: 1/hFacW where maskW .NE. 0, else the value held since INI_MASKS_ETC (G00 dump: 0 on dry points)
    for c in ("W", "S"):
        m = f("G00_geometry", "mask" + c)
        h = s["hFac" + c]
        s["recip_hFac" + c] = np.where(m != 0.0, 1.0 / np.where(m != 0.0, h, 1.0), f("G00_geometry", "recip_hFac" + c))
    return s


def make_case(name, extra=None):
    ds = oracle.dumpset(name)
    nml = RunNamelists(oracle.run_dir(name))
    its = iterations(ds)
    g = jax.tree_util.tree_map(jnp.asarray, grid_from_dump(ds, its[0]))
    c = dict(name=name, ds=ds, nml=nml, its=its, g=g, p=PhiHydParams.from_namelists(nml),
             kLowC=jnp.asarray(kLowC_from_hFac(g.h0FacC)))
    c.update(extra(c) if extra else {})
    return c


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
    """Negative controls and gradients run on the forced oracle only (phi0surf != 0, three iterations)."""
    if oracle.FORCED not in cases:
        cases[oracle.FORCED] = make_case(oracle.FORCED)
    return cases[oracle.FORCED]


_phi = jax.jit(calc_phi_hyd, static_argnames=("myIter",))  # params are a jit argument (KERNEL_GUIDE)


def phi_inputs(case, it):
    """The five fields CALC_PHI_HYD reads (see dynamics_inputs), cached per iteration."""
    cache = case.setdefault("phi_cache", {})
    if it not in cache:
        f = lambda st, n: oracle.field(case["ds"], it, st, n)  # noqa: E731
        cache[it] = dict(etaH=f("S01_update_rstar_F", "etaH"),
                         **{n: f("S04_oceanic_phys", n) for n in ("rStarFacC", "rhoInSitu", "totPhiHyd", "phi0surf")})
    return cache[it]


def run_phi(case, it, p=None, **override):
    s = dict(phi_inputs(case, it))
    s.update(override)
    return _phi(p or case["p"], case["g"], case["kLowC"], s["rhoInSitu"], s["rStarFacC"], s["etaH"], s["phi0surf"],
                s["totPhiHyd"], myIter=it)


def errors(case, out, it):
    ds = case["ds"]
    e = {n: relerr(out[n], oracle.field(ds, it, "D00a_phi_hyd", n))
         for n in ("phiHydC", "phiHydF", "dPhiHydX", "dPhiHydY")}
    e.update({n: relerr(out[n], oracle.field(ds, it, "S05_dynamics", n)) for n in ("totPhiHyd", "phiHydLow")})
    return e


def test_phi_hyd_matches_fortran(case):
    """Replay gate D00a_phi_hyd (per level) + S05_dynamics totPhiHyd/phiHydLow at every dumped iteration."""
    for it in case["its"]:
        e = errors(case, run_phi(case, it), it)
        print(case["name"], it, {k: f"{v:.2e}" for k, v in e.items()})
        assert all(v <= TOL for v in e.values()), (it, e)


def test_phi_hyd_negative_controls(forced):
    """The same comparison fails for a planted error: gravity perturbed by 1e-6 relative; z* slope term dropped
    (etaH = 0 removes calc_grad_phi_hyd.F:164-189 only from dPhiHydX/Y); rhoInSitu shifted one level down."""
    case = forced
    it = case["its"][-1]
    p = case["p"]
    bad = errors(case, run_phi(case, it, p=dataclasses.replace(p, gravity=p.gravity * (1 + 1e-6))), it)
    assert all(bad[n] > TOL_CLASS for n in ("phiHydC", "phiHydF", "totPhiHyd", "phiHydLow")), bad
    s = phi_inputs(case, it)
    bad = errors(case, run_phi(case, it, etaH=np.zeros_like(s["etaH"])), it)
    assert bad["dPhiHydX"] > TOL_CLASS and bad["dPhiHydY"] > TOL_CLASS, bad
    assert bad["phiHydC"] <= TOL, bad  # the slope term does not enter phiHydC
    bad = errors(case, run_phi(case, it, rhoInSitu=np.roll(s["rhoInSitu"], 1, axis=1)), it)
    assert all(v > TOL_CLASS for v in bad.values()), bad


def wet_points(g, grid="C", levels=(3, 20), tiles=(1, 5, 9)):
    """A few wet interior points (t, k, j, i) spread over tiles and levels."""
    L = g.layout
    m = np.asarray(getattr(g, "mask" + grid))
    pts = []
    for t, k in zip(tiles, levels * len(tiles)):
        w = np.argwhere(m[t, k, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] == 1.0)
        j, i = w[len(w) // 2]
        pts.append((t, k, j + L.OLy, i + L.OLx))
    return pts


def fd_check(f, x0, gx, idx, hs, weights):
    """Best relative difference between gx[idx] and central differences of sum(w * f(x)) over the h-sweep hs.
    The output difference is formed before the weighted sum, so the FD sees no cancellation of the large total."""
    best, rows = np.inf, []
    for h in hs:
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += h
        xm[idx] -= h
        op, om = f(xp), f(xm)
        fd = sum(float(jnp.sum(w * (a - b))) for w, a, b in zip(weights, op, om)) / (2 * h)
        err = abs(fd - float(gx[idx])) / max(abs(fd), 1e-300)
        rows.append((h, err))
        best = min(best, err)
    return best, rows


def test_phi_hyd_gradient(forced):
    """d/d(rhoInSitu, rStarFacC) of sum(wX*dPhiHydX + wT*totPhiHyd + phiHydLow): finite everywhere (dry and halo
    lanes included) and equal to central finite differences at wet points (h-sweep printed; the functional is linear
    in each argument, so the plateau spans the sweep)."""
    case = forced
    it = case["its"][0]
    s = phi_inputs(case, it)
    g, p, kLowC = case["g"], case["p"], case["kLowC"]
    rng = np.random.default_rng(1)
    wX = jnp.asarray(rng.normal(size=s["rhoInSitu"].shape))
    wT = jnp.asarray(rng.normal(size=s["rhoInSitu"].shape))
    one = jnp.ones(s["etaH"].shape)
    fixed = (p, g, kLowC, s["etaH"], s["phi0surf"], s["totPhiHyd"])  # jit arguments: no captured constants

    def outs(rho, rsf, p_, g_, kLowC_, etaH, phi0surf, totPhiHyd):
        o = calc_phi_hyd(p_, g_, kLowC_, rho, rsf, etaH, phi0surf, totPhiHyd, myIter=it)
        return o["dPhiHydX"], o["totPhiHyd"], o["phiHydLow"]

    def J(rho, rsf, wX_, wT_, one_, *rest):
        a, b, c = outs(rho, rsf, *rest)
        return jnp.sum(wX_ * a) + jnp.sum(wT_ * b) + jnp.sum(one_ * c)

    outs_j = jax.jit(outs)
    gr, gs = jax.jit(jax.grad(J, argnums=(0, 1)))(s["rhoInSitu"], s["rStarFacC"], wX, wT, one, *fixed)
    assert np.all(np.isfinite(gr)) and np.all(np.isfinite(gs))
    for (t, k, j, i) in wet_points(g):
        best, rows = fd_check(lambda x: outs_j(x, s["rStarFacC"], *fixed), s["rhoInSitu"], gr, (t, k, j, i),
                              (1.0, 1e-1, 1e-2, 1e-3), (wX, wT, one))
        print("rho", (t, k, j, i), rows)
        assert best < 1e-8, rows
        best, rows = fd_check(lambda x: outs_j(s["rhoInSitu"], x, *fixed), s["rStarFacC"], gs, (t, j, i),
                              (1e-2, 1e-3, 1e-4, 1e-5), (wX, wT, one))
        print("rStarFacC", (t, j, i), rows)
        assert best < 1e-8, rows


def test_full_ice_load_enters_phi_hyd(cases):
    """Full tree: the replay passes with the dumped phi0surf; removing the sea-ice load from it (phi0surf -
    sIceLoad*gravity/rhoConst, i.e. pLoad/rhoConst) fails totPhiHyd and phiHydLow (the load is present)."""
    if oracle.FULL not in cases:
        cases[oracle.FULL] = make_case(oracle.FULL)
    case = cases[oracle.FULL]
    it = case["its"][0]
    assert max(errors(case, run_phi(case, it), it).values()) <= TOL
    ds, nml = case["ds"], case["nml"]
    sIceLoad = oracle.field(ds, it, "S04_oceanic_phys", "sIceLoad")
    pLoad = oracle.field(ds, it, "S04_oceanic_phys", "pLoad")
    rhoConst = float(nml.get("data", "parm01", "rhoConst", default=nml.get("data", "parm01", "rhoNil", default=999.8)))
    assert np.abs(sIceLoad).max() > 100.0
    bad = errors(case, run_phi(case, it, phi0surf=pLoad * (1.0 / rhoConst)), it)
    assert bad["totPhiHyd"] > TOL_CLASS and bad["phiHydLow"] > TOL_CLASS, bad
