"""r* / free-surface part of FORWARD_STEP (plan Task 15): replay gates against both oracles (SMOKE its 1-2, FORCED its
1-3; EmPmR is non-zero only in FORCED), the chain DYNAMICS -> THERMODYNAMICS, volume conservation, gradients.

Replay gates feed each kernel the dumped inputs of the previous stage and compare the full arrays (halos included)
with the dumped outputs. Measured (XLA_FLAGS=--xla_cpu_max_isa=AVX, params traced): every field bitwise equal —
S01/S06 hFacC/W/S + recip_hFacC, S07 aW2d aS2d aC2d pW pS pC, S09 uVel vVel, S10 wVel etaN etaH dEtaHdt, S11
rStarFacC/W/S, G00(it+1) rStarFacNm1*, rStarExp*, rStarDh*Dt, pStarFacK, etaHnm1, S12 uVel vVel wVel, and the whole
chain S05 -> S12 (cg2d included, same iteration counts as the Fortran STDOUT). With the parameters as compile-time
constants XLA rewrites `x/deltaTFreeSurf` (rStarDh*Dt: 1e-16 relative) — see params_io.params_pytree.
Full V4r4 tree (oracle.FULL, iterations 1-3; M2.6a): the same gates (EmPmR with the sea-ice fresh-water flux; the
r*/free-surface code takes the same branches: the sea-ice load enters only phi0surf). Bitwise (measured 2026-09-23);
negative control on the full tree: the EmPmR term dropped (facEmP = 0) fails dEtaHdt, rStarFacNm1C * (1 + 1e-6)
fails hFacC.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core import cg2d as cg
from mitgcm_jax.core import free_surface as fs
from mitgcm_jax.core import solve_for_pressure as sfp
from mitgcm_jax.grid.geometry import Grid, stack_tiles, unpack_vertical
from mitgcm_jax.layout import Layout
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.tests import oracle

L = Layout()
EX = default_exchanger()
ORACLES = (oracle.SMOKE, oracle.FORCED, oracle.FULL)
# the geometry these kernels read (grid_from_dump minus the 3-D mixing fields they never use)
GRID_FIELDS = ("dxG", "dyG", "recip_dxC", "recip_dyC", "rA", "recip_rA", "recip_rAw", "recip_rAs", "R_low", "rLowW",
               "rLowS", "Ro_surf", "rSurfW", "rSurfS", "recip_Rcol", "maskInC", "maskC", "maskW", "maskS",
               "h0FacC", "h0FacW", "h0FacS")


def load_grid(ds, it):
    f = {n: stack_tiles(ds, it, "G00_geometry", n, L) for n in GRID_FIELDS}
    f.update(unpack_vertical(ds.tiles(it, "G00_geometry", "vertical")[1].data[0], L))
    return Grid(f, L)


class Case:
    def __init__(self, name, it, ds, g, p, cp, its):
        self.name, self.it, self.ds, self.g, self.p, self.cp, self.its = name, it, ds, g, p, cp, its

    def f(self, stage, name, it=None):
        return oracle.field(self.ds, self.it if it is None else it, stage, name)

    @property
    def has_next(self):
        return self.it + 1 in self.its


@pytest.fixture(scope="module")
def cases():
    out = []
    for name in ORACLES:
        ds = oracle.dumpset(name)
        its = sorted({k[0] for k in ds.index})
        nml = RunNamelists(oracle.run_dir(name))
        g = load_grid(ds, its[0])  # static geometry
        p = fs.FreeSurfParams.from_namelists(nml)
        cp = cg.Cg2dParams.from_namelists(nml, cg.ini_cg2d_norm(g, g.h0FacW, g.h0FacS, p.implicSurfPress,
                                                                   p.implicDiv2DFlow))
        out += [Case(name, it, ds, g, p, cp, its) for it in its]
    assert [c.name for c in out].count(oracle.FORCED) == 3 and [c.name for c in out].count(oracle.FULL) == 3
    assert len(out) == 8
    return out


# jitted kernels: params and grid are ARGUMENTS (traced), the exchanger is closed over
J_UPDATE_R_STAR = jax.jit(fs.update_r_star)
J_UPDATE_CG2D = jax.jit(lambda p, cp, g, *a: sfp.update_cg2d(p, cp, g, EX, *a))
J_MOM_CORR = jax.jit(sfp.momentum_correction_step)
J_CONT = jax.jit(lambda p, g, *a: fs.integr_continuity(p, g, EX, *a))
J_RSTAR = jax.jit(lambda p, g, *a: fs.calc_r_star(p, g, EX, *a))
J_STAGGER = jax.jit(lambda *a: fs.do_stagger_fields_exchanges(EX, *a))
J_CHAIN = jax.jit(lambda p, cp, g, s: sfp.step_after_dynamics(p, cp, g, EX, s))

R_FIELDS = ("rStarFacNm1C", "rStarFacNm1W", "rStarFacNm1S", "rStarExpC", "rStarExpW", "rStarExpS",
            "rStarDhCDt", "rStarDhWDt", "rStarDhSDt", "pStarFacK")


def _eq(got, ref, what):
    got, ref = np.asarray(got), np.asarray(ref)
    if not np.array_equal(got, ref):
        d = np.abs(got - ref)
        rel = d.max() / max(np.abs(ref).max(), 1e-300)
        raise AssertionError(f"{what}: not bitwise; max|d|={d.max():.3e} rel={rel:.3e}"
                             f" at {np.unravel_index(np.argmax(d), d.shape)}")


def _rstar_F(c):
    G = lambda n: c.f("G00_geometry", n)  # noqa: E731
    return J_UPDATE_R_STAR(c.g, G("rStarFacNm1C"), G("rStarFacNm1W"), G("rStarFacNm1S"),
                           c.f("S00_begin", "recip_hFacC"), G("recip_hFacW"), G("recip_hFacS"))


def _rstar_T(c, recip_prev):
    return J_UPDATE_R_STAR(c.g, *[c.f("S01_update_rstar_F", n) for n in ("rStarFacC", "rStarFacW", "rStarFacS")],
                           c.f("S01_update_rstar_F", "recip_hFacC"), recip_prev[1], recip_prev[2])


def _cg2d_op(c, p=None):
    return J_UPDATE_CG2D(p or c.p, c.cp, c.g, c.f("S06_update_rstar_T", "hFacW"), c.f("S06_update_rstar_T", "hFacS"),
                         c.f("G00_geometry", "recip_Bo"), *[c.f("S00_begin", n) for n in ("pW", "pS", "pC")])


def test_update_rstar_and_cg2d_operator(cases):
    """S01 (RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.)), S06 (UPDATE_R_STAR(.TRUE.)), S07 (UPDATE_CG2D): bitwise."""
    for c in cases:
        out = _rstar_F(c)
        for n, a in zip(("hFacC", "hFacW", "hFacS", "recip_hFacC"), out[:4]):
            _eq(a, c.f("S01_update_rstar_F", n), f"{c.name} it{c.it} S01 {n}")
        _eq(fs.reset_nlfs_vars(c.f("S00_begin", "rStarFacC")), c.f("G00_geometry", "pStarFacK"), "pStarFacK")
        outT = _rstar_T(c, (None, out[4], out[5]))
        for n, a in zip(("hFacC", "hFacW", "hFacS", "recip_hFacC"), outT[:4]):
            _eq(a, c.f("S06_update_rstar_T", n), f"{c.name} it{c.it} S06 {n}")
        for n, a in zip(("aW2d", "aS2d", "aC2d", "pW", "pS", "pC"), _cg2d_op(c)):
            _eq(a, c.f("S07_update_cg2d", n), f"{c.name} it{c.it} S07 {n}")


def test_momentum_correction_and_stagger_exchanges(cases):
    """S09 (MOMENTUM_CORRECTION_STEP from S08 etaN, S05 gU/gV) and S12 (DO_STAGGER_FIELDS_EXCHANGES): bitwise."""
    for c in cases:
        u, v = J_MOM_CORR(c.p, c.g, c.f("S08_solve_for_pressure", "etaN"), c.f("S05_dynamics", "gU"),
                          c.f("S05_dynamics", "gV"), c.f("S08_solve_for_pressure", "uVel"),
                          c.f("S08_solve_for_pressure", "vVel"), c.f("G00_geometry", "Bo_surf"))
        _eq(u, c.f("S09_momentum_correction", "uVel"), f"{c.name} it{c.it} S09 uVel")
        _eq(v, c.f("S09_momentum_correction", "vVel"), f"{c.name} it{c.it} S09 vVel")
        out = J_STAGGER(*[c.f("S10_integr_continuity", n) for n in ("uVel", "vVel", "wVel")])
        for n, a in zip(("uVel", "vVel", "wVel"), out):
            _eq(a, c.f("S12_stagger_exchanges", n), f"{c.name} it{c.it} S12 {n}")


def _cont(c, p=None):
    S = lambda n: c.f("S09_momentum_correction", n)  # noqa: E731
    return J_CONT(p or c.p, c.g, S("uVel"), S("vVel"), c.f("S06_update_rstar_T", "hFacW"),
                  c.f("S06_update_rstar_T", "hFacS"), c.f("S04_oceanic_phys", "EmPmR"), S("etaN"), S("etaH"),
                  S("dEtaHdt"), S("wVel"))


def test_integr_continuity(cases):
    """S10 (INTEGR_CONTINUITY + UPDATE_ETAH) and etaHnm1 (G00 of the next step): bitwise."""
    for c in cases:
        out = _cont(c)
        for n in ("wVel", "etaN", "etaH", "dEtaHdt"):
            _eq(out[n], c.f("S10_integr_continuity", n), f"{c.name} it{c.it} S10 {n}")
        if c.has_next:
            _eq(out["etaHnm1"], c.f("G00_geometry", "etaHnm1", c.it + 1), f"{c.name} it{c.it} etaHnm1")


def _rstar(c, p=None, etaH=None):
    return J_RSTAR(p or c.p, c.g, c.f("S10_integr_continuity", "etaH") if etaH is None else etaH,
                   *[c.f("S06_update_rstar_T", n) for n in ("rStarFacC", "rStarFacW", "rStarFacS")])


def test_calc_r_star(cases):
    """S11 rStarFacC/W/S and the r* fields of G00 at the next step (rStarFacNm1*, rStarExp*, rStarDh*Dt, pStarFacK):
    bitwise; the hFacInf/hFacSup checks (calc_r_star.F:182-202) find nothing."""
    for c in cases:
        out = _rstar(c)
        for n in ("rStarFacC", "rStarFacW", "rStarFacS"):
            _eq(out[n], c.f("S11_calc_rstar", n), f"{c.name} it{c.it} S11 {n}")
        assert int(out["icntc1"]) + int(out["icntw"]) + int(out["icnts"]) + int(out["icntc2"]) == 0
        if c.has_next:
            for n in R_FIELDS:
                _eq(out[n], c.f("G00_geometry", n, c.it + 1), f"{c.name} it{c.it} G00(it+1) {n}")


def _chain_inputs(c):
    s = {n: c.f("S05_dynamics", n) for n in ("gU", "gV", "uVel", "vVel", "wVel", "etaN", "etaH", "dEtaHdt")}
    s.update({n: c.f("S01_update_rstar_F", n) for n in ("rStarFacC", "rStarFacW", "rStarFacS", "recip_hFacC")})
    s.update({n: c.f("G00_geometry", n) for n in ("recip_hFacW", "recip_hFacS", "Bo_surf", "recip_Bo")})
    s.update({n: c.f("S00_begin", n) for n in ("pW", "pS", "pC")})
    s["EmPmR"] = c.f("S04_oceanic_phys", "EmPmR")
    return s


def _stdout_cg2d(name):
    lines = (oracle.run_dir(name) / "STDOUT.0000").read_text().splitlines()
    return [int(lines[i + 2].split("=")[1].split()[1]) for i, t in enumerate(lines) if "cg2d: Sum(rhs),rhsMax" in t]


def test_chain_after_dynamics(cases):
    """forward_step.F:846-1016 as one jitted function from the S05 state: every output bitwise equal to the dumps
    (S06 hFac, S07 operator, S10/S12 dynamics fields, S11 r*, G00(it+1) r* fields and etaHnm1); cg2d iteration count
    equal to the Fortran STDOUT."""
    for c in cases:
        out = J_CHAIN(c.p, c.cp, c.g, _chain_inputs(c))
        tag = f"{c.name} it{c.it}"
        for n in ("hFacC", "hFacW", "hFacS", "recip_hFacC"):
            _eq(out[n], c.f("S11_calc_rstar", n), f"{tag} chain {n}")
        for n in ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC"):
            _eq(out[n], c.f("S07_update_cg2d", n), f"{tag} chain {n}")
        _eq(out["cg2d"]["x_fortran"], c.f("C02_cg2d_solution", "cg2d_x"), f"{tag} chain C02 cg2d_x")
        for n in ("uVel", "vVel", "wVel", "etaN", "etaH", "dEtaHdt"):
            _eq(out[n], c.f("S12_stagger_exchanges", n), f"{tag} chain {n}")
        for n in ("rStarFacC", "rStarFacW", "rStarFacS"):
            _eq(out[n], c.f("S11_calc_rstar", n), f"{tag} chain {n}")
        if c.has_next:
            for n in R_FIELDS + ("etaHnm1",):
                _eq(out[n], c.f("G00_geometry", n, c.it + 1), f"{tag} chain G00(it+1) {n}")
        assert int(out["cg2d"]["numIters"]) == _stdout_cg2d(c.name)[c.it - 1], tag


def _fails(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


def test_negative_controls(cases):
    """Each gate fails on a planted error: rStarFacNm1C * (1+1e-6); cg2dpcOffDFac 0.51 -> 0.51*(1+1e-6);
    deltaTFreeSurf * (1+1e-6) in INTEGR_CONTINUITY (etaN) and CALC_R_STAR (rStarDhCDt); EmPmR term dropped
    (facEmP=0, FORCED only); etaH shifted by one point in i for CALC_R_STAR."""
    c = next(x for x in cases if x.name == oracle.FORCED)
    G = lambda n: c.f("G00_geometry", n)  # noqa: E731
    bad = J_UPDATE_R_STAR(c.g, G("rStarFacNm1C") * (1 + 1e-6), G("rStarFacNm1W"), G("rStarFacNm1S"),
                          c.f("S00_begin", "recip_hFacC"), G("recip_hFacW"), G("recip_hFacS"))
    assert _fails(lambda: _eq(bad[0], c.f("S01_update_rstar_F", "hFacC"), "planted"))
    cpb = dataclasses.replace(c.cp, cg2dpcOffDFac=0.51 * (1 + 1e-6))
    badop = J_UPDATE_CG2D(c.p, cpb, c.g, c.f("S06_update_rstar_T", "hFacW"), c.f("S06_update_rstar_T", "hFacS"),
                          G("recip_Bo"), *[c.f("S00_begin", n) for n in ("pW", "pS", "pC")])
    assert _fails(lambda: _eq(badop[3], c.f("S07_update_cg2d", "pW"), "planted"))
    pb = dataclasses.replace(c.p, deltaTFreeSurf=c.p.deltaTFreeSurf * (1 + 1e-6))
    assert _fails(lambda: _eq(_cont(c, pb)["etaN"], c.f("S10_integr_continuity", "etaN"), "planted"))
    assert _fails(lambda: _eq(_rstar(c, pb)["rStarDhCDt"], c.f("G00_geometry", "rStarDhCDt", c.it + 1), "planted"))
    pe = dataclasses.replace(c.p, facEmP=0.0)
    assert _fails(lambda: _eq(_cont(c, pe)["dEtaHdt"], c.f("S10_integr_continuity", "dEtaHdt"), "planted"))
    shifted = np.roll(c.f("S10_integr_continuity", "etaH"), 1, axis=-1)
    assert _fails(lambda: _eq(_rstar(c, etaH=shifted)["rStarFacC"], c.f("S11_calc_rstar", "rStarFacC"), "planted"))


def test_full_negative_controls(cases):
    """Full tree: facEmP = 0 (EmPmR term dropped, incl. the sea-ice fresh-water flux) fails dEtaHdt at S10;
    rStarFacNm1C * (1 + 1e-6) fails hFacC at S01."""
    c = next(x for x in cases if x.name == oracle.FULL)
    _eq(_cont(c)["dEtaHdt"], c.f("S10_integr_continuity", "dEtaHdt"), "full dEtaHdt")
    pe = dataclasses.replace(c.p, facEmP=0.0)
    assert _fails(lambda: _eq(_cont(c, pe)["dEtaHdt"], c.f("S10_integr_continuity", "dEtaHdt"), "planted"))
    G = lambda n: c.f("G00_geometry", n)  # noqa: E731
    bad = J_UPDATE_R_STAR(c.g, G("rStarFacNm1C") * (1 + 1e-6), G("rStarFacNm1W"), G("rStarFacNm1S"),
                          c.f("S00_begin", "recip_hFacC"), G("recip_hFacW"), G("recip_hFacS"))
    assert _fails(lambda: _eq(bad[0], c.f("S01_update_rstar_F", "hFacC"), "planted"))


def test_volume_conservation(cases):
    """exactConserv + r*: (a) surface w of every wet column equals the fresh-water flux mass2rUnit*EmPmR
    (integrate_for_w.F: the column sum of -div - rStarDhDt*drF*h0FacC telescopes to -hDiv/rA - dEtaHdt); (b) the
    global volume change sum(rA*(etaH_new-etaH_old))/dt equals -mass2rUnit*sum(rA*EmPmR) over wet columns (the
    divergence sums to zero across tile edges); (c) the r* column thickness change equals the etaH change. Measured
    (5 cases): (a) <= 6.7e-15 of max|dEtaHdt|, (b) <= 5e-17 of sum|rA*dEtaHdt| (FORCED net fresh-water volume flux
    -4.5e6..-5.1e6 m3/s reproduced), (c) <= 1.53 ulp of the column thickness. Asserted: 1e-13, 1e-12, 16 ulp.
    Negative control: with the EmPmR term dropped from dEtaHdt (facEmP=0) (a) and (b) fail on FORCED."""
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)

    def check(c, p):
        out = _cont(c, p)
        g = c.g
        E = c.f("S04_oceanic_phys", "EmPmR")[:, J, I]
        wet = g.maskC[:, 0, J, I] != 0
        w1 = np.asarray(out["wVel"])[:, 0, J, I]
        scale_a = np.abs(np.asarray(out["dEtaHdt"])[:, J, I]).max()   # size of the terms that cancel
        ea = np.abs(np.where(wet, w1 - c.p.mass2rUnit * E, 0.0)).max() / scale_a
        rA = g.rA[:, J, I]
        dV = np.sum(rA * (np.asarray(out["etaH"])[:, J, I] - c.f("S09_momentum_correction", "etaH")[:, J, I]))
        dV /= c.p.deltaTFreeSurf
        src = -c.p.mass2rUnit * np.sum(np.where(wet, rA * E, 0.0))
        scale_b = np.sum(np.abs(rA * np.asarray(out["dEtaHdt"])[:, J, I]))
        eb = abs(dV - src) / scale_b
        return ea, eb, out

    for c in cases:
        ea, eb, out = check(c, c.p)
        assert ea < 1e-13, (c.name, c.it, ea)
        assert eb < 1e-12, (c.name, c.it, eb)
        rs = _rstar(c)
        g = c.g
        Rcol = g.Ro_surf[:, J, I] - g.R_low[:, J, I]
        wet = g.maskInC[:, J, I] != 0
        dcol = np.where(wet, (np.asarray(rs["rStarFacC"]) - c.f("S06_update_rstar_T", "rStarFacC"))[:, J, I] * Rcol,
                        0.0)
        deta = np.where(wet, np.asarray(out["etaH"])[:, J, I] - c.f("S09_momentum_correction", "etaH")[:, J, I], 0.0)
        # in ulps of the column thickness: rStarFac = (eta + Ro_surf - R_low)*recip_Rcol rounds at eps*Rcol
        ec = np.abs(dcol - deta).max() / (np.finfo(float).eps * np.abs(Rcol).max())
        assert ec < 16, (c.name, c.it, ec)
    for c in (x for x in cases if x.name == oracle.FORCED):
        ea, eb, _ = check(c, dataclasses.replace(c.p, facEmP=0.0))
        assert ea > 1e-3 and eb > 1e-3, (ea, eb)


def test_gradients(cases):
    """Reverse-mode gradients of the chain S05 -> S12 (through UPDATE_R_STAR, UPDATE_CG2D, the cg2d
    custom_linear_solve, the correction, continuity and CALC_R_STAR): finite everywhere (halo and dry lanes too),
    and d(J)/d(gU), d(J)/d(etaH) match central differences at wet points. The FD forward uses cg2dTolerance=1e-13
    so the solver noise is below the FD truncation error; J = sum(w1*etaN) + sum(w2*rStarExpC) + sum(w3*wVel)."""
    c = next(x for x in cases if x.name == oracle.FORCED and x.it == 2)
    s0 = _chain_inputs(c)
    rng = np.random.default_rng(3)
    wts = [rng.normal(size=s0["etaN"].shape), rng.normal(size=s0["etaN"].shape), rng.normal(size=s0["wVel"].shape)]
    inner = cg.interior_mask(L)
    wts = [w * inner for w in wts]
    cpt = dataclasses.replace(c.cp, cg2dTolerance=1e-13, cg2dMaxIters=1000)

    def J(p, cp, g, s, wts, gU, etaH):
        o = sfp.step_after_dynamics(p, cp, g, EX, dict(s, gU=gU, etaH=etaH))
        return (jnp.sum(wts[0] * o["etaN"]) + jnp.sum(wts[1] * o["rStarExpC"]) + 1e3 * jnp.sum(wts[2] * o["wVel"]))

    s0 = {k: jnp.asarray(v) for k, v in s0.items()}
    wts = [jnp.asarray(w) for w in wts]
    Jj = jax.jit(J)
    Jt = lambda cp, gU, etaH: Jj(c.p, cp, c.g, s0, wts, gU, etaH)  # noqa: E731
    gU0, eta0 = s0["gU"], s0["etaH"]
    gg, ge = jax.jit(jax.grad(J, argnums=(5, 6)))(c.p, c.cp, c.g, s0, wts, gU0, eta0)
    gg, ge = np.asarray(gg), np.asarray(ge)
    assert np.all(np.isfinite(gg)) and np.all(np.isfinite(ge))
    maskW, maskC = c.g.maskW, c.g.maskC
    # wet u points at levels 1, 21 and two wet columns, deterministic, on different tiles. J is linear in gU and
    # etaH, so large steps are exact up to rounding; the FD noise is ~1e-9/h (measured h-sweep: gU 1e-2..1e-5,
    # etaH 1e-1..1e-4 m, the FD converges to the gradient as h grows; at h=1e-2 / 1e-1 within 2e-7 relative)
    wetW = np.argwhere(maskW[:, 20, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] != 0)
    pts_u = [(t, k, j, i) for (t, j, i), k in zip(wetW[::len(wetW) // 2][:2], (0, 20))]
    wetC = np.argwhere(maskC[:, 0, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] != 0)
    pts_e = [tuple(x) for x in wetC[len(wetC) // 7::len(wetC) // 2][:2]]
    for (t, k, j, i) in pts_u:
        j, i = j + L.OLy, i + L.OLx
        h = 1e-2
        e = np.zeros(gU0.shape)
        e[t, k, j, i] = h
        fd = (float(Jt(cpt, gU0 + e, eta0)) - float(Jt(cpt, gU0 - e, eta0))) / (2 * h)
        assert abs(fd - gg[t, k, j, i]) <= 1e-6 * abs(fd), ("gU", t, k, j, i, fd, gg[t, k, j, i])
    for (t, j, i) in pts_e:
        j, i = j + L.OLy, i + L.OLx
        h = 1e-1
        e = np.zeros(eta0.shape)
        e[t, j, i] = h
        fd = (float(Jt(cpt, gU0, eta0 + e)) - float(Jt(cpt, gU0, eta0 - e))) / (2 * h)
        assert abs(fd - ge[t, j, i]) <= 1e-6 * abs(fd), ("etaH", t, j, i, fd, ge[t, j, i])
