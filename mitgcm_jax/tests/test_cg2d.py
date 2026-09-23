"""CG2D and SOLVE_FOR_PRESSURE (plan Task 15): replay gates against both oracles, the Fortran solver log, the
custom_linear_solve adjoint.

Measured (XLA_FLAGS=--xla_cpu_max_isa=AVX, params traced, sum_order="fortran"): cg2dNorm equals the Fortran value
exactly (7.1522409280111305E-05); C01 cg2d_b, cg2d_x bitwise; C02 cg2d_x (the solver output with its halos) bitwise;
S08 etaN bitwise; iteration counts SMOKE 179, 172 / FORCED 164, 161, 158 as in STDOUT (tolerance 1e-7; residual at
stop 9.29e-8, 9.90e-8 / 9.22e-8, 9.92e-8, 9.88e-8, margins 7 %, 1 %, 8 %, 0.8 %, 1.2 %). With the XLA tree sum inside
each tile (sum_order="tile") the iteration counts are the same and cg2d_x differs by up to 4e-9 relative (CG amplifies
the 1-ulp differences of the sums).
Full V4r4 tree (oracle.FULL, iterations 1-3; M2.6a): the same gates; EmPmR (S04) there includes the sea-ice fresh-water
flux of SEAICE_GROWTH; the solver and SOLVE_FOR_PRESSURE take the same branches. Bitwise, iteration counts 165, 162,
158 as in its STDOUT (residual at stop 9.41e-8, 9.96e-8, 9.72e-8: margins 6 %, 0.4 %, 3 %). Negative control on the
full tree: cg2dNorm * (1 + 1e-6) fails C02 and the EmPmR term dropped fails cg2d_b.
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
OPS = ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC")
J, I = L.js(1, L.sNy), L.is_(1, L.sNx)


def stdout_cg2d(name):
    """Per step: (normalisation factor, [dict(sumRHS, rhsMax, first, n, last)]) from the Fortran STDOUT."""
    lines = (oracle.run_dir(name) / "STDOUT.0000").read_text().splitlines()
    norm, steps = None, []
    for i, t in enumerate(lines):
        if "CG2D normalisation factor" in t:
            norm = float(t.split("=")[-1])
        if "cg2d: Sum(rhs),rhsMax" in t:
            s, m = (float(x) for x in t.split("=")[1].split())
            steps.append(dict(sumRHS=s, rhsMax=m, first=float(lines[i + 1].split("=")[1]),
                              n=int(lines[i + 2].split("=")[1].split()[1]), last=float(lines[i + 3].split("=")[1])))
    return norm, steps


class Case:
    def __init__(self, name, it, ds, g, p, cp, log):
        self.name, self.it, self.ds, self.g, self.p, self.cp, self.log = name, it, ds, g, p, cp, log

    def f(self, stage, name):
        return oracle.field(self.ds, self.it, stage, name)

    def ops(self):
        return tuple(self.f("C01_cg2d_inputs", n) for n in OPS)


@pytest.fixture(scope="module")
def cases():
    out = []
    for name in ORACLES:
        ds = oracle.dumpset(name)
        its = sorted({k[0] for k in ds.index})
        nml = RunNamelists(oracle.run_dir(name))
        g = load_grid(ds, its[0])
        p = fs.FreeSurfParams.from_namelists(nml)
        norm = cg.ini_cg2d_norm(g, g.h0FacW, g.h0FacS, p.implicSurfPress, p.implicDiv2DFlow)
        cp = cg.Cg2dParams.from_namelists(nml, norm)
        fnorm, steps = stdout_cg2d(name)
        assert norm == fnorm, (name, norm, fnorm)  # ini_cg2d.F:160, printed with 17 digits (:175-177)
        assert len(steps) == len(its)
        out += [Case(name, it, ds, g, p, cp, steps[it - 1]) for it in its]
    return out


def _solver(cp, ops, b, x):
    return cg.cg2d_fortran(EX, ops, b, x, cg2dNorm=cp.cg2dNorm, tolerance=cp.cg2dTolerance,
                           max_iters=cp.cg2dMaxIters, sum_order=cp.sum_order, nIterMin=cp.nIterMin)


J_SOLVE = jax.jit(_solver)
J_RHS = jax.jit(sfp.cg2d_rhs)
J_SFP = jax.jit(lambda p, cp, g, ops, *a: sfp.solve_for_pressure(p, cp, g, EX, ops, *a))


def _eq(got, ref, what):
    got, ref = np.asarray(got), np.asarray(ref)
    if not np.array_equal(got, ref):
        d = np.abs(got - ref)
        raise AssertionError(f"{what}: not bitwise; max|d|={d.max():.3e} rel={d.max() / np.abs(ref).max():.3e}")


def _rhs(c, p=None):
    return J_RHS(p or c.p, c.g, c.f("S05_dynamics", "gU"), c.f("S05_dynamics", "gV"),
                 c.f("S06_update_rstar_T", "hFacW"), c.f("S06_update_rstar_T", "hFacS"),
                 c.f("S04_oceanic_phys", "EmPmR"), c.f("S05_dynamics", "etaN"), c.f("S05_dynamics", "etaH"),
                 c.f("G00_geometry", "Bo_surf"))


def test_cg2d_rhs_and_operator(cases):
    """C01: cg2d_b (EmPmR + CALC_DIV_GHAT + exactConserv etaH term) and the first guess Bo_surf*etaN, bitwise; the
    operator/preconditioner handed to CG2D is the one UPDATE_CG2D produced (S07)."""
    for c in cases:
        b, x = _rhs(c)
        _eq(b, c.f("C01_cg2d_inputs", "cg2d_b"), f"{c.name} it{c.it} cg2d_b")
        _eq(x, c.f("C01_cg2d_inputs", "cg2d_x"), f"{c.name} it{c.it} cg2d_x")
        for n in OPS:
            _eq(c.f("C01_cg2d_inputs", n), c.f("S07_update_cg2d", n), f"{c.name} it{c.it} {n}")


def test_cg2d_solution_and_iterations(cases):
    """C02: the literal CG2D from the C01 inputs gives cg2d_x bitwise (halos included), the Fortran iteration count,
    and the logged sumRHS, rhsMax, first and last residuals (printed with 15 digits: rel 1e-14). The XLA tree sum
    inside tiles (sum_order='tile') keeps the iteration count; its solution is within 1e-8 relative."""
    tile = None
    for c in cases:
        x, d = J_SOLVE(c.cp, c.ops(), c.f("C01_cg2d_inputs", "cg2d_b"), c.f("C01_cg2d_inputs", "cg2d_x"))
        ref = c.f("C02_cg2d_solution", "cg2d_x")
        _eq(x, ref, f"{c.name} it{c.it} C02 cg2d_x")
        assert int(d["numIters"]) == c.log["n"], (c.name, c.it, int(d["numIters"]), c.log["n"])
        for k, kk in (("sumRHS", "sumRHS"), ("rhsMax", "rhsMax"), ("firstResidual", "first"),
                      ("lastResidual", "last")):
            np.testing.assert_allclose(float(d[k]), c.log[kk], rtol=1e-14, atol=0, err_msg=f"{c.name} {c.it} {k}")
        assert float(d["lastResidual"]) < c.cp.cg2dTolerance
        tile = dataclasses.replace(c.cp, sum_order="tile")
        xt, dt = J_SOLVE(tile, c.ops(), c.f("C01_cg2d_inputs", "cg2d_b"), c.f("C01_cg2d_inputs", "cg2d_x"))
        assert int(dt["numIters"]) == c.log["n"]
        rel = np.abs(np.asarray(xt) - ref).max() / np.abs(ref).max()
        assert rel < 1e-8, (c.name, c.it, rel)


def test_solve_for_pressure_etaN(cases):
    """S08: SOLVE_FOR_PRESSURE through the custom_linear_solve wrapper (from S05/S06 inputs, operator of S07): etaN
    bitwise, C02 cg2d_x bitwise, same iteration count."""
    for c in cases:
        etaN, info = J_SFP(c.p, c.cp, c.g, tuple(c.f("S07_update_cg2d", n) for n in OPS),
                           c.f("S05_dynamics", "gU"), c.f("S05_dynamics", "gV"), c.f("S06_update_rstar_T", "hFacW"),
                           c.f("S06_update_rstar_T", "hFacS"), c.f("S04_oceanic_phys", "EmPmR"),
                           c.f("S05_dynamics", "etaN"), c.f("S05_dynamics", "etaH"), c.f("G00_geometry", "Bo_surf"),
                           c.f("G00_geometry", "recip_Bo"))
        _eq(etaN, c.f("S08_solve_for_pressure", "etaN"), f"{c.name} it{c.it} S08 etaN")
        _eq(info["x_fortran"], c.f("C02_cg2d_solution", "cg2d_x"), f"{c.name} it{c.it} C02")
        assert int(info["numIters"]) == c.log["n"]


def _fails(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


def test_negative_controls(cases):
    """The gates fail on planted errors: cg2dNorm * (1+1e-6) (C01 operator scaling enters cg2d_b via the solver
    normalisation -> C02), one preconditioner value changed by 1e-6 (C02: the iterate changes, the count may not),
    the EmPmR term dropped from cg2d_b (FORCED), and a target residual 0.99 x the Fortran last residual (one more
    iteration)."""
    c = next(x for x in cases if x.name == oracle.FORCED and x.it == 2)
    b0, x0, ref = c.f("C01_cg2d_inputs", "cg2d_b"), c.f("C01_cg2d_inputs", "cg2d_x"), c.f("C02_cg2d_solution", "cg2d_x")
    x, _ = J_SOLVE(dataclasses.replace(c.cp, cg2dNorm=c.cp.cg2dNorm * (1 + 1e-6)), c.ops(), b0, x0)
    assert _fails(lambda: _eq(x, ref, "planted"))
    ops = list(c.ops())
    pW = ops[3].copy()
    pW[4, L.jj(45), L.ii(45)] *= 1 + 1e-6
    ops[3] = pW
    x, _ = J_SOLVE(c.cp, tuple(ops), b0, x0)
    assert _fails(lambda: _eq(x, ref, "planted"))
    b, _ = _rhs(c, dataclasses.replace(c.p, useRealFreshWaterFlux=False))
    assert _fails(lambda: _eq(b, b0, "planted"))
    # the stopping rule: a target just below the residual the Fortran stopped at needs one more iteration
    x, d = J_SOLVE(dataclasses.replace(c.cp, cg2dTolerance=0.99 * c.log["last"]), c.ops(), b0, x0)
    assert int(d["numIters"]) == c.log["n"] + 1


def test_full_negative_controls(cases):
    """Full tree (iteration 2, the tightest stopping margin): cg2dNorm * (1 + 1e-6) fails C02; dropping the fresh-water
    term (useRealFreshWaterFlux=F: EmPmR incl. the sea-ice melt/freeze flux) fails cg2d_b."""
    c = next(x for x in cases if x.name == oracle.FULL and x.it == 2)
    b0, x0, ref = c.f("C01_cg2d_inputs", "cg2d_b"), c.f("C01_cg2d_inputs", "cg2d_x"), c.f("C02_cg2d_solution", "cg2d_x")
    x, d = J_SOLVE(c.cp, c.ops(), b0, x0)
    _eq(x, ref, "full C02")
    x, _ = J_SOLVE(dataclasses.replace(c.cp, cg2dNorm=c.cp.cg2dNorm * (1 + 1e-6)), c.ops(), b0, x0)
    assert _fails(lambda: _eq(x, ref, "planted"))
    b, _ = _rhs(c, dataclasses.replace(c.p, useRealFreshWaterFlux=False))
    assert _fails(lambda: _eq(b, b0, "planted"))


def _matvec(cp, ops):
    inner = jnp.asarray(cg.interior_mask(L))

    def mv(v):
        xe = EX.exch_xy(jnp.where(inner, v, 0.0))
        return jnp.zeros_like(v).at[:, J, I].set(cg.apply_operator(L, *ops[:3], xe) / cp.cg2dNorm)
    return jax.jit(mv)


def test_operator_symmetric_and_transpose_solve(cases):
    """custom_linear_solve(symmetric=True) needs a symmetric operator: <A u, v> = <u, A v> for random interior u, v
    (exchange-coupled 5-point operator on the 13-tile LLC90). The VJP of the solve w.r.t. cg2d_b is the transpose
    solve: z = dJ/db for J = <w, x(b)> satisfies A z = w (relative residual < 1e-11) and <z, b> = <w, x> (x from a
    forward solve at 1e-13) to 1e-9."""
    c = next(x for x in cases if x.name == oracle.FORCED and x.it == 1)
    ops = tuple(jnp.asarray(a) for a in c.ops())
    mv = _matvec(c.cp, ops)
    inner = cg.interior_mask(L)
    rng = np.random.default_rng(1)
    u, v = (jnp.asarray(rng.normal(size=L.shape2d) * inner) for _ in range(2))
    lhs, rhs = float(jnp.vdot(mv(u), v)), float(jnp.vdot(u, mv(v)))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs), (lhs, rhs)
    b = jnp.asarray(c.f("C01_cg2d_inputs", "cg2d_b"))
    x0 = jnp.asarray(c.f("C01_cg2d_inputs", "cg2d_x"))
    w = jnp.asarray(rng.normal(size=L.shape2d) * inner)
    fwd = jax.jit(lambda cp, b: cg.cg2d_solve(cp, EX, ops, b, x0)[0])
    z = jax.jit(jax.grad(lambda b: jnp.vdot(w, fwd(c.cp, b))))(b)
    assert np.all(np.isfinite(np.asarray(z)))
    assert np.all(np.asarray(z)[:, ~inner] == 0.0)                  # b's halos never enter CG2D
    res = np.abs(np.asarray(mv(z) - w)).max() / np.abs(np.asarray(w)).max()
    assert res < 1e-11, res
    x = fwd(dataclasses.replace(c.cp, cg2dTolerance=1e-13, cg2dMaxIters=1000), b)
    np.testing.assert_allclose(float(jnp.vdot(z, b)), float(jnp.vdot(w, x)), rtol=1e-9)


def test_coefficient_and_rhs_gradients_fd(cases):
    """d/d(aW2d), d/d(aC2d), d/d(cg2d_b) of J = <w, x> (reverse mode through custom_linear_solve) vs central
    differences along a random direction on a 3x3 window. The FD forward runs at cg2dTolerance=1e-13 (287
    iterations) so the solver noise is below the truncation error. Measured best relative error over the h-sweep:
    aW2d 3e-8 (h=1e-4, quadratic convergence from 2e-4 at h=1e-2), aC2d 1.5e-6 (h=1e-5; the window direction nearly
    cancels, so the noise floor of J (~4e-12) shows), cg2d_b 4e-10 (J is linear in b)."""
    c = next(x for x in cases if x.name == oracle.FORCED and x.it == 3)
    ops0 = tuple(jnp.asarray(a) for a in c.ops())
    b0 = jnp.asarray(c.f("C01_cg2d_inputs", "cg2d_b"))
    x0 = jnp.asarray(c.f("C01_cg2d_inputs", "cg2d_x"))
    rng = np.random.default_rng(2)
    w = jnp.asarray(rng.normal(size=L.shape2d) * cg.interior_mask(L))
    tight = dataclasses.replace(c.cp, cg2dTolerance=1e-13, cg2dMaxIters=1000)

    def J(cp, aW, aC, b):
        ops = (aW, ops0[1], aC) + ops0[3:]
        return jnp.vdot(w, cg.cg2d_solve(cp, EX, ops, b, x0)[0])

    Jt = jax.jit(J)
    grads = jax.jit(jax.grad(J, argnums=(1, 2, 3)))(c.cp, ops0[0], ops0[2], b0)
    for a in grads:
        assert np.all(np.isfinite(np.asarray(a)))
    t, j0, i0 = 4, L.jj(40), L.ii(40)                   # a wet 3x3 window on tile 5
    base = [ops0[0], ops0[2], b0]
    assert np.all(np.asarray(ops0[2])[t, j0:j0 + 3, i0:i0 + 3] != 0)
    for k, name, hs, tol in ((0, "aW2d", (1e-2, 1e-3, 1e-4, 1e-5), 1e-6), (1, "aC2d", (1e-3, 1e-4, 1e-5), 1e-5),
                             (2, "cg2d_b", (1e-1, 1e-2, 1e-3), 1e-8)):
        win = np.asarray(base[k])[t, j0:j0 + 3, i0:i0 + 3]
        d = np.zeros(L.shape2d)
        d[t, j0:j0 + 3, i0:i0 + 3] = rng.normal(size=(3, 3)) * np.abs(win).max()
        gd = float(jnp.vdot(grads[k], jnp.asarray(d)))
        errs = []
        for h in hs:
            plus = [a + h * d if n == k else a for n, a in enumerate(base)]
            minus = [a - h * d if n == k else a for n, a in enumerate(base)]
            fd = (float(Jt(tight, *plus)) - float(Jt(tight, *minus))) / (2 * h)
            errs.append(abs(fd - gd) / abs(gd))
        assert min(errs) < tol, (name, errs)


def test_ecco_mode_stops_coefficient_gradient(cases):
    """stop_coeff_grad=True (TAF cg2d.flow: only cg2d_b, cg2d_x active): forward bitwise identical to the exact mode;
    d/d(aC2d) is exactly 0 (effect), d/d(cg2d_b) unchanged."""
    c = next(x for x in cases if x.name == oracle.SMOKE and x.it == 2)
    ops0 = tuple(jnp.asarray(a) for a in c.ops())
    b0 = jnp.asarray(c.f("C01_cg2d_inputs", "cg2d_b"))
    x0 = jnp.asarray(c.f("C01_cg2d_inputs", "cg2d_x"))
    w = jnp.asarray(np.random.default_rng(4).normal(size=L.shape2d) * cg.interior_mask(L))
    ecco = dataclasses.replace(c.cp, stop_coeff_grad=True)

    def J(cp, aC, b):
        return jnp.vdot(w, cg.cg2d_solve(cp, EX, ops0[:2] + (aC,) + ops0[3:], b, x0)[0])

    xe = jax.jit(lambda cp: cg.cg2d_solve(cp, EX, ops0, b0, x0)[0])
    _eq(xe(ecco), xe(c.cp), "ecco forward")
    grad = jax.jit(jax.grad(J, argnums=(1, 2)), static_argnums=())
    gC_exact, gb_exact = grad(c.cp, ops0[2], b0)
    gC_ecco, gb_ecco = grad(ecco, ops0[2], b0)
    assert np.abs(np.asarray(gC_exact)).max() > 0
    assert np.all(np.asarray(gC_ecco) == 0)
    _eq(gb_ecco, gb_exact, "d/db ecco vs exact")
