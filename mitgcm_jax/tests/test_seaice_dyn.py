"""SEAICE_DYNSOLVER (plan M2.4): replay gates against the full-V4r4 oracle, negative controls, derivative checks.

Oracle `oracle.FULL` (full_jaxdump_v5: 13 tiles of 90x90, dumps at iterations 1-3; stages Y01-Y06, L01-L04 per Picard
pass, I00, I01 in reference/jaxdump/SUBSTEPS.md). Kernels: mitgcm_jax/pkgs/seaice_dyn.py, seaice_lsr.py.

Measured (conftest XLA flags: no FMA, no algsimp; parameters traced), iterations 1, 2, 3:
  - SEAICE_LSR replayed from its dumped inputs (Y04_before_lsr): every point of L01, L02 (locals on their computed
    range), L03 (first LSOR sweep + exchange), L04 and Y05 BITWISE for both Picard passes; LSOR sweep counts
    ICOUNT1 = ICOUNT2 = 178/118, 112/82, 84/58 (pass 1/2) and S1, S2, WFAU, WFAV equal to the Fortran values (the
    stopping margin is small: S2 = 1.986e-4 against LSR_ERROR = 2e-4 at iteration 1, pass 1).
  - Whole SEAICE_DYNSOLVER from SEAICE_MODEL's inputs (I00, EXF fu/fv of S03) to I01: every field BITWISE at every
    point (Y01-Y06 and all LSR stages too), except uice_fd/vice_fd: 1 ulp at 27-45 points (SEAICE_FREEDRIFT: gcc
    fuses SIN/COS into glibc sincos; uice_fd is read by nothing in V4r4: LSR_mixIniGuess = 0).
  - The EXP of SEAICE_CALC_ICE_STRENGTH is gfortran-vectorised into libmvec _ZGVbN2v_exp; with XLA's exp PRESS0
    differs at ~3100 points (1-2 ulp), with glibc scalar exp at 2978, with seaice_dyn.exp_libmvec at 0.
Negative controls: relaxation factor x(1+1e-6), SOLV_NCHECK = 4, free-slip strain rates, SEAICEstressFactor x(1+1e-6),
waterDrag x(1+1e-6), XLA's exp in the ice strength: each makes the corresponding gate fail.
Derivatives (seaice_lsr.py docstring, AD design), measured: implicit tangent of one LSOR solve (GMRES 40 x 8 cycles)
vs central FD of the literal solve run to LSR_ERROR = 1e-12: right-hand-side direction 8.4e-10 relative; operator
direction (AU, BU, CU.. perturbed) 7.0e-4 / 1.7e-4 at h = 0.1 / 0.05 (O(h^2)), Richardson 9.0e-7; adjoint (GMRES on
A^T with P^T) vs tangent 3e-15 / 3e-13; whole dynsolver (2 Picard passes) d/d(fu, fv): finite on every lane, adjoint
vs tangent 1.1e-14; ocean stress w.r.t. ice velocity vs FD. P = 4 (shard_map, 4 fake CPU devices): LSR bitwise = P = 1.
Runtime ~8 min (3 FD/gradient tests ~2 min each): tier1x.
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.io.dump import DumpSet, read_file
from mitgcm_jax.layout import Layout
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import seaice_dyn as sd
from mitgcm_jax.pkgs import seaice_lsr as sl
from mitgcm_jax.tests import oracle

L = Layout()
EX = default_exchanger()
ITS = (1, 2, 3)
INT = (L.js(1, L.sNy), L.is_(1, L.sNx))
R0 = (L.js(0, L.sNy), L.is_(0, L.sNx))  # etaPlusZeta/zetaMinusEta: J,I = 0..sN (seaice_lsr.F:353-358)
COUNTS = {1: (178, 118), 2: (112, 82), 3: (84, 58)}  # LSOR sweeps per Picard pass (L04 ICOUNT1 = ICOUNT2)
LSR_IN = ("seaiceMassC", "seaiceMassU", "seaiceMassV", "FORCEX0", "FORCEY0", "PRESS0", "ZMAX", "ZMIN", "e11", "e22",
          "e12", "DWATN", "FORCEX", "FORCEY")
L02_INTERIOR = ("AU", "BU", "CU", "AV", "BV", "CV", "uRt1", "uRt2", "vRt1", "vRt2", "rhsU", "rhsV")


def _dumpset(directory):
    """DumpSet with the record headers read in parallel threads (the serial index of a full-tree oracle takes ~110 s
    on cold Lustre; scripts/tests/test_reference.py::_dumpset_parallel)."""
    files = sorted(Path(directory).glob("jd_*_t*.bin"))
    assert files, directory
    with ThreadPoolExecutor(min(len(files), 64)) as pool:
        per_file = list(pool.map(lambda f: read_file(f, lazy=True), files))
    ds = DumpSet.__new__(DumpSet)
    ds.dir, ds.index = Path(directory), {}
    for recs in per_file:
        for r in recs:
            ds.index.setdefault((r.iter, r.stage, r.field), {})[r.tile] = r
    ds.order = list(ds.index)
    return ds


class Env:
    def __init__(self):
        self.ds = _dumpset(oracle.run_dir(oracle.FULL) / "jaxdump")
        self.nml = RunNamelists(oracle.run_dir(oracle.FULL))
        self.p = sd.SeaiceDynParams.from_namelists(self.nml)
        self.g = grid_from_dump(self.ds, 1)
        self._cache = {}

    def f(self, it, stage, name):
        return oracle.field(self.ds, it, stage, name)

    def sg(self, it):
        s = {n: self.f(1, "G01_seaice_geometry", n) for n in ("HEFFM", "k1AtC", "k1AtZ", "k2AtC", "k2AtZ")}
        for n in ("seaiceMaskU", "seaiceMaskV", "tensileStrFac"):
            s[n] = self.f(it, "Y01_get_dynforcing", n)
        return s

    def surf(self, it):
        return (self.f(it, "S01_update_rstar_F", "uVel")[:, 0], self.f(it, "S01_update_rstar_F", "vVel")[:, 0])

    def dyn_state(self, it):
        """SEAICE_DYNSOLVER inputs: SEAICE_MODEL's inputs (I00), EXF fu/fv (S03), and the values on entry of the
        arrays it writes only partly (seaiceMass*: previous step's I01, or 1000 from seaice_init_varia.F:430-432 at
        the first step; FORCEX0/Y0 on entry = Y01; e11.., DWATN, FORCEX/Y on entry = Y04: not touched before)."""
        uVel, vVel = self.surf(it)
        st = dict(HEFF=self.f(it, "I00_seaice_begin", "HEFF"), AREA=self.f(it, "I00_seaice_begin", "AREA"),
                  uIce=self.f(it, "I00_seaice_begin", "UICE"), vIce=self.f(it, "I00_seaice_begin", "VICE"),
                  uVel=uVel, vVel=vVel, fu=self.f(it, "S03_ctrl_map_forcing", "fu"),
                  fv=self.f(it, "S03_ctrl_map_forcing", "fv"))
        for n in ("seaiceMassC", "seaiceMassU", "seaiceMassV"):
            st[n] = self.f(it - 1, "I01_dynsolver", n) if it > 1 else np.full(L.shape2d, 1000.0)
        for n in ("FORCEX0", "FORCEY0"):
            st[n] = self.f(it, "Y01_get_dynforcing", n)
        for n in ("e11", "e22", "e12", "DWATN", "FORCEX", "FORCEY"):
            st[n] = self.f(it, "Y04_before_lsr", n)
        return st

    def lsr_state(self, it):
        uVel, vVel = self.surf(it)
        ls = dict(uIce=self.f(it, "Y04_before_lsr", "UICE"), vIce=self.f(it, "Y04_before_lsr", "VICE"),
                  uVel=uVel, vVel=vVel)
        ls.update({n: self.f(it, "Y04_before_lsr", n) for n in LSR_IN})
        return ls

    def lsr(self, it, p=None):
        key = ("lsr", it) if p is None else None
        if key in self._cache:
            return self._cache[key]
        r = _run_lsr(p or self.p, self.g, self.sg(it), EX, self.lsr_state(it))
        if key is not None:
            self._cache[key] = r
        return r


@pytest.fixture(scope="module")
def env():
    return Env()


@jax.jit
def _run_lsr(p, g, sg, ex, ls):
    return sl.seaice_lsr(p, g, sg, ex, ls, record=True)


@partial(jax.jit, static_argnames=("max_iter",))
def _run_lsor(p, ex, co, u, v, max_iter=None, lsr_error=None):
    return sl.lsor_solve(p, ex, co, u, v, max_iter=max_iter, lsr_error=lsr_error)


@jax.jit
def _run_dyn(p, g, sg, ex, st):
    return sd.dynsolver(p, g, sg, ex, st, record=True)


def _ndiff(a, b, region=None):
    a, b = np.asarray(a), np.asarray(b)
    if region is not None:
        a, b = a[(slice(None),) + region], b[(slice(None),) + region]
    return int(np.sum(a != b))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.abs(b).max()
    return float(np.abs(a - b).max() / (m if m > 0 else 1.0))


def _lsr_mismatches(env, it, out, passes):
    """{stage/field: number of differing points} over every gated point of the LSR stages."""
    bad = {}
    for ip, pr in enumerate(passes, 1):
        for k, v in pr["L01"].items():
            bad[f"L01_p{ip}/{k}"] = _ndiff(v, env.f(it, f"L01_lsr_visc_drag_p{ip}", k))
        for k, v in pr["L02"].items():
            if k in ("seaiceMaskU", "seaiceMaskV"):
                continue
            reg = R0 if k in ("etaPlusZeta", "zetaMinusEta") else (INT if k in L02_INTERIOR else None)
            bad[f"L02_p{ip}/{k}"] = _ndiff(v, env.f(it, f"L02_lsr_coeffs_p{ip}", k), reg)
        for k in ("UICE", "VICE"):
            bad[f"L04_p{ip}/{k}"] = _ndiff(pr["L04"][k], env.f(it, f"L04_lsor_end_p{ip}", k))
        for k in ("ICOUNT1", "ICOUNT2", "S1", "S2", "WFAU", "WFAV"):
            bad[f"L04_p{ip}/{k}"] = int(float(pr["L04"][k]) != float(env.f(it, f"L04_lsor_end_p{ip}", k)[0, 0, 0]))
    for k, n in (("uIce", "UICE"), ("vIce", "VICE"), ("e11", "e11"), ("e22", "e22"), ("e12", "e12"),
                 ("deltaC", "deltaC"), ("ETA", "ETA"), ("etaZ", "etaZ"), ("ZETA", "ZETA"), ("zetaZ", "zetaZ"),
                 ("PRESS", "PRESS"), ("DWATN", "DWATN"), ("FORCEX", "FORCEX"), ("FORCEY", "FORCEY"),
                 ("uIceNm1", "uIceNm1"), ("vIceNm1", "vIceNm1")):
        bad[f"Y05/{n}"] = _ndiff(out[k], env.f(it, "Y05_lsr", n))
    return bad


# ---------------------------------------------------------------------------------------------------------------
# gates


@pytest.mark.parametrize("it", ITS)
def test_lsr_replay_bitwise(env, it):
    """SEAICE_LSR from its dumped inputs: L01-L04 of both Picard passes and Y05 bitwise, same sweep counts."""
    out, passes = env.lsr(it)
    bad = _lsr_mismatches(env, it, out, passes)
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    assert tuple(int(pr["L04"]["ICOUNT1"]) for pr in passes) == COUNTS[it]
    for ip, pr in enumerate(passes, 1):  # L03: one LSOR sweep (+ exchange) from the pass's first guess
        u1, v1, _ = _run_lsor(env.p, EX, pr["co"], pr["L02_inputs"]["uIce"], pr["L02_inputs"]["vIce"], max_iter=1)
        assert _ndiff(u1, env.f(it, f"L03_lsor_sweep1_p{ip}", "UICE")) == 0
        assert _ndiff(v1, env.f(it, f"L03_lsor_sweep1_p{ip}", "VICE")) == 0


@pytest.mark.parametrize("it", ITS)
def test_dynsolver_kernels_replay(env, it):
    """GET_DYNFORCING, ice masses, FORCEX0/Y0, ice strength, FREEDRIFT, OCEAN_STRESS and clipping from dumped inputs."""
    p, g, f = env.p, env.g, env.f
    st = env.dyn_state(it)
    TAUX, TAUY = sd.get_dynforcing(p, g, st["fu"], st["fv"])
    assert _ndiff(TAUX, f(it, "Y01_get_dynforcing", "TAUX")) == 0
    assert _ndiff(TAUY, f(it, "Y01_get_dynforcing", "TAUY")) == 0
    for n, a in zip(("seaiceMassC", "seaiceMassU", "seaiceMassV"),
                    sd.ice_mass(p, g, st["HEFF"], st["seaiceMassC"], st["seaiceMassU"], st["seaiceMassV"])):
        assert _ndiff(a, f(it, "Y01_get_dynforcing", n)) == 0, n
    PRESS0, ZMAX, ZMIN = jax.jit(sd.calc_ice_strength)(p, st["HEFF"], st["AREA"], env.sg(it)["HEFFM"])
    for n, a in (("PRESS0", PRESS0), ("ZMAX", ZMAX), ("ZMIN", ZMIN)):
        assert _ndiff(a, f(it, "Y02_ice_strength", n)) == 0, n
    out, rec = _run_dyn(p, g, env.sg(it), EX, st)
    for n in ("FORCEX0", "FORCEY0"):
        assert _ndiff(rec["Y02"][n], f(it, "Y02_ice_strength", n)) == 0, n
    # SEAICE_FREEDRIFT from the dumped Y02 state
    TX, TY, ufd, vfd = jax.jit(sd.freedrift)(p, g, EX, f(it, "Y02_ice_strength", "TAUX"),
                                             f(it, "Y02_ice_strength", "TAUY"), f(it, "Y02_ice_strength", "FORCEX0"),
                                             f(it, "Y02_ice_strength", "FORCEY0"), st["HEFF"], st["uVel"], st["vVel"])
    assert _ndiff(TX, f(it, "Y03_freedrift", "TAUX")) == 0
    assert _ndiff(TY, f(it, "Y03_freedrift", "TAUY")) == 0
    for n, a in (("uice_fd", ufd), ("vice_fd", vfd)):  # glibc sincos vs XLA cos/sin: a few 1-ulp points
        ref = f(it, "Y03_freedrift", n)
        assert _rel(a, ref) < 1e-15 and _ndiff(a, ref) < 200, (n, _rel(a, ref), _ndiff(a, ref))
    # SEAICE_OCEAN_STRESS from the dumped LSR result
    fu, fv = jax.jit(sd.ocean_stress)(p, g, EX, st["fu"], st["fv"], f(it, "Y05_lsr", "UICE"), f(it, "Y05_lsr", "VICE"),
                                      f(it, "Y05_lsr", "DWATN"), st["AREA"], st["uVel"], st["vVel"])
    assert _ndiff(fu, f(it, "Y06_ocean_stress", "fu")) == 0
    assert _ndiff(fv, f(it, "Y06_ocean_stress", "fv")) == 0
    # clipping (seaice_dynsolver.F:365-385)
    for n in ("UICE", "VICE"):
        clip = np.maximum(np.minimum(f(it, "Y05_lsr", n), 0.40), -0.40)
        assert _ndiff(clip, f(it, "I01_dynsolver", n)) == 0


@pytest.mark.parametrize("it", ITS)
def test_dynsolver_chain(env, it):
    """Whole SEAICE_DYNSOLVER from SEAICE_MODEL's inputs to I01: every stage bitwise (uice_fd/vice_fd: 1 ulp at a few
    points, glibc sincos), same LSOR counts."""
    out, rec = _run_dyn(env.p, env.g, env.sg(it), EX, env.dyn_state(it))
    bad = {}
    for stage, key in (("Y01_get_dynforcing", "Y01"), ("Y02_ice_strength", "Y02"), ("Y03_freedrift", "Y03"),
                       ("Y05_lsr", "Y05"), ("Y06_ocean_stress", "Y06")):
        for k, v in rec[key].items():
            if k not in ("uice_fd", "vice_fd"):
                bad[f"{key}/{k}"] = _ndiff(v, env.f(it, stage, k))
    for k, v in out.items():
        ref = env.f(it, "I01_dynsolver", k)
        if k in ("uice_fd", "vice_fd"):
            assert _rel(v, ref) < 1e-15 and _ndiff(v, ref) < 200, (k, _rel(v, ref), _ndiff(v, ref))
        else:
            bad[f"I01/{k}"] = _ndiff(v, ref)
    bad.update({f"LSR/{k}": v for k, v in _lsr_mismatches(env, it, {"uIce": out["UICE"], "vIce": out["VICE"]}
                                                             | {k: out[k] for k in out}, rec["passes"]).items()
                if not k.startswith("Y05/")})
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    assert tuple(int(pr["L04"]["ICOUNT1"]) for pr in rec["passes"]) == COUNTS[it]
    assert tuple(int(pr["L04"]["ICOUNT2"]) for pr in rec["passes"]) == COUNTS[it]


# ---------------------------------------------------------------------------------------------------------------
# negative controls


def test_negative_controls(env):
    it = 1
    p = env.p
    # relaxation factor x(1+1e-6): the LSOR iterate is no longer the Fortran one
    out, passes = env.lsr(it, dataclasses.replace(p, SEAICE_LSRrelaxU=p.SEAICE_LSRrelaxU * (1 + 1e-6)))
    assert _ndiff(passes[0]["L04"]["UICE"], env.f(it, "L04_lsor_end_p1", "UICE")) > 1000
    # convergence check every 4th sweep instead of every SOLV_NCHECK=2: other sweep counts
    out, passes = env.lsr(it, dataclasses.replace(p, SOLV_NCHECK=4))
    assert int(passes[0]["L04"]["ICOUNT1"]) != COUNTS[it][0]
    # free-slip strain rates (noSlipFac = 0): e12 differs
    out, passes = env.lsr(it, dataclasses.replace(p, SEAICE_no_slip=False))
    assert _ndiff(passes[0]["L01"]["e12"], env.f(it, "L01_lsr_visc_drag_p1", "e12")) > 1000
    st = env.dyn_state(it)
    f = env.f
    # ocean stress scaling x(1+1e-6)
    fu, _ = jax.jit(sd.ocean_stress)(dataclasses.replace(p, SEAICEstressFactor=1 + 1e-6), env.g, EX, st["fu"],
                                     st["fv"], f(it, "Y05_lsr", "UICE"), f(it, "Y05_lsr", "VICE"),
                                     f(it, "Y05_lsr", "DWATN"), st["AREA"], st["uVel"], st["vVel"])
    assert _ndiff(fu, f(it, "Y06_ocean_stress", "fu")) > 1000
    # free-drift water drag x(1+1e-6): uice_fd outside its 1-ulp tolerance
    _, _, ufd, _ = jax.jit(sd.freedrift)(dataclasses.replace(p, SEAICE_waterDrag=p.SEAICE_waterDrag * (1 + 1e-6)),
                                         env.g, EX, f(it, "Y02_ice_strength", "TAUX"),
                                         f(it, "Y02_ice_strength", "TAUY"), f(it, "Y02_ice_strength", "FORCEX0"),
                                         f(it, "Y02_ice_strength", "FORCEY0"), st["HEFF"], st["uVel"], st["vVel"])
    assert _rel(ufd, f(it, "Y03_freedrift", "uice_fd")) > 1e-12
    # XLA's exp instead of libmvec's in the ice strength: PRESS0 no longer bitwise
    orig = sd.exp_libmvec
    try:
        sd.exp_libmvec = jnp.exp
        P0, _, _ = sd.calc_ice_strength(p, st["HEFF"], st["AREA"], env.sg(it)["HEFFM"])
    finally:
        sd.exp_libmvec = orig
    assert _ndiff(P0, f(it, "Y02_ice_strength", "PRESS0")) > 1000


# ---------------------------------------------------------------------------------------------------------------
# derivatives


def _masked_direction(co, seed, keys, scale):
    """Random pointwise-relative perturbation of the coefficient arrays `keys` (interior, wet points)."""
    rng = np.random.default_rng(seed)
    d = {k: jnp.zeros_like(v) for k, v in co.items()}
    J, I = INT
    for k in keys:
        r = np.zeros(L.shape2d)
        r[:, J, I] = rng.standard_normal((L.nTiles, L.sNy, L.sNx))
        d[k] = jnp.asarray(r) * co[k] * scale
    return d


@pytest.mark.parametrize("keys,scale,hs", [(("rhsU", "rhsV"), 1e-2, (1.0,)),
                                           (("BU", "BV", "AU", "CV"), 1e-3, (0.1, 0.05))])
def test_lsor_implicit_derivative_vs_fd(env, keys, scale, hs):
    """d/deps J(x(c + eps*d)), x = LSOR solution of pass 1 (iteration 1), J = sum(w*u) + sum(w'*v): the implicit
    tangent (GMRES on A) against a central FD of the literal solve run to LSR_ERROR = 1e-12 (the Fortran tolerance
    2e-4 leaves the iterate far from the solution: its FD is not the implicit derivative), for right-hand-side and for
    operator perturbations; and the adjoint (transpose GMRES on A^T) against the tangent."""
    it = 1
    out, passes = env.lsr(it)
    co = {k: jnp.asarray(v) for k, v in passes[0]["co"].items()}
    p = env.p
    d = _masked_direction(co, 0, keys, scale)
    rng = np.random.default_rng(1)
    wu = jnp.asarray(rng.standard_normal(L.shape2d)) * co["seaiceMaskU"]
    wv = jnp.asarray(rng.standard_normal(L.shape2d)) * co["seaiceMaskV"]
    J, I = INT

    def Jfun(u, v):
        return jnp.sum(wu[:, J, I] * u[:, J, I]) + jnp.sum(wv[:, J, I] * v[:, J, I])

    tight = dict(max_iter=20000, lsr_error=1e-12)
    if "tight" not in env._cache:
        env._cache["tight"] = _run_lsor(p, EX, co, passes[0]["L02_inputs"]["uIce"], passes[0]["L02_inputs"]["vIce"],
                                        **tight)
    u_t, v_t, info = env._cache["tight"]
    assert float(info["converged"]) == 1.0

    def F(c):
        u, v, _ = sl.lsor_solve(p, EX, c, u_t, v_t)
        return Jfun(u, v)

    tan = float(jax.jit(lambda c, dc: jax.jvp(F, (c,), (dc,))[1])(co, d))
    fd = {}
    for h in hs:
        up = _run_lsor(p, EX, {k: co[k] + h * d[k] for k in co}, u_t, v_t, **tight)
        um = _run_lsor(p, EX, {k: co[k] - h * d[k] for k in co}, u_t, v_t, **tight)
        assert float(up[2]["converged"]) == 1.0 and float(um[2]["converged"]) == 1.0
        fd[h] = float((Jfun(up[0], up[1]) - Jfun(um[0], um[1])) / (2 * h))
    if len(hs) == 2:  # operator perturbation: FD error O(h^2) (measured 7e-2 at h=1, 6e-3 at h=0.3): Richardson
        h1, h2 = hs
        fd["R"] = (h1 * h1 * fd[h2] - h2 * h2 * fd[h1]) / (h1 * h1 - h2 * h2)
    err = {h: abs(v - tan) / abs(tan) for h, v in fd.items()}
    assert min(err.values()) < 1e-5, (tan, fd, err)
    gr = jax.jit(jax.grad(F))(co)
    adj = float(sum(jnp.vdot(gr[k], d[k]) for k in co))
    assert abs(adj - tan) <= 1e-9 * abs(tan), (adj, tan)


def test_dynsolver_gradient_finite_and_dot(env):
    """Gradient of the whole dynsolver (2 Picard passes, implicit LSOR derivatives) w.r.t. the ocean surface stress
    fu, fv: finite on every lane (halos, land), and the adjoint equals the tangent (dot test)."""
    it = 2
    st = {k: jnp.asarray(v) for k, v in env.dyn_state(it).items()}
    p, g, sg = env.p, env.g, env.sg(it)
    rng = np.random.default_rng(2)
    wu = jnp.asarray(rng.standard_normal(L.shape2d)) * sg["seaiceMaskU"]

    def J(fu, fv):
        out, _ = sd.dynsolver(p, g, sg, EX, dict(st, fu=fu, fv=fv))
        return jnp.sum(wu * out["UICE"]) + 1e-3 * jnp.sum(out["fu"])

    du = jnp.asarray(rng.standard_normal(L.shape2d)) * g.maskW[:, 0] * 1e-2
    dv = jnp.asarray(rng.standard_normal(L.shape2d)) * g.maskS[:, 0] * 1e-2
    gu, gv = jax.jit(jax.grad(J, argnums=(0, 1)))(st["fu"], st["fv"])
    assert np.all(np.isfinite(np.asarray(gu))) and np.all(np.isfinite(np.asarray(gv)))
    tan = float(jax.jit(lambda a, b, c, e: jax.jvp(J, (a, b), (c, e))[1])(st["fu"], st["fv"], du, dv))
    adj = float(jnp.vdot(gu, du) + jnp.vdot(gv, dv))
    assert abs(adj - tan) <= 1e-8 * abs(tan), (adj, tan)


def test_ocean_stress_gradient_vs_fd(env):
    """SEAICE_OCEAN_STRESS: d(sum w*fu_out)/d(uIce) at wet points against central FD (linear: exact up to rounding)."""
    it = 1
    f, p, g = env.f, env.p, env.g
    st = env.dyn_state(it)
    u0, v0, D = f(it, "Y05_lsr", "UICE"), f(it, "Y05_lsr", "VICE"), f(it, "Y05_lsr", "DWATN")
    w = jnp.asarray(np.random.default_rng(3).standard_normal(L.shape2d))

    def J(u):
        fu, fv = sd.ocean_stress(p, g, EX, st["fu"], st["fv"], u, v0, D, st["AREA"], st["uVel"], st["vVel"])
        return jnp.sum(w * fu) + jnp.sum(w * fv)

    gr = np.asarray(jax.jit(jax.grad(J))(jnp.asarray(u0)))
    assert np.all(np.isfinite(gr))
    Jj = jax.jit(J)
    ice = np.argwhere((np.asarray(st["AREA"]) > 0.5) & (np.asarray(g.maskW[:, 0]) > 0))
    ice = ice[(ice[:, 1] >= L.jj(2)) & (ice[:, 1] <= L.jj(L.sNy - 1)) & (ice[:, 2] >= L.ii(2))
              & (ice[:, 2] <= L.ii(L.sNx - 1))]
    assert len(ice) > 100
    for t, j, i in ice[:: max(1, len(ice) // 5)][:5]:
        h = 1e-4
        e = np.zeros(L.shape2d)
        e[t, j, i] = h
        fd = (float(Jj(jnp.asarray(u0 + e))) - float(Jj(jnp.asarray(u0 - e)))) / (2 * h)
        assert abs(fd - gr[t, j, i]) <= 1e-7 * max(abs(gr[t, j, i]), 1e-12), (t, j, i, fd, gr[t, j, i])


# ---------------------------------------------------------------------------------------------------------------
# sharding: LSR is tile-local, so the tile size (90x90) is fixed and P only distributes tiles


def test_lsr_sharded_p4_bitwise(env):
    """SEAICE_LSR inside shard_map on 4 fake CPU devices (conftest.py), tiles in contiguous blocks padded with replicas
    of tile 1 (parallel/sharded_exchange.py): uIce, vIce and the LSOR counts bitwise equal to P = 1."""
    from jax.sharding import PartitionSpec as PS

    from mitgcm_jax.grid.geometry import Grid
    from mitgcm_jax.parallel.exchange import MAP_DIR, ExchangeMaps
    from mitgcm_jax.parallel.shard import tile_mesh
    from mitgcm_jax.parallel.sharded_exchange import AXIS, ShardedExchanger, TileBlocks

    it = 1
    out1, passes1 = env.lsr(it)
    P = 4
    maps = ExchangeMaps.load(MAP_DIR / "exch_maps_13x90x90.npz")
    blocks = TileBlocks(L.nTiles, P)
    mesh = tile_mesh(P)
    exs = ShardedExchanger.build(maps, blocks).device_arrays(mesh)
    names = ("maskW", "maskS", "maskC", "maskInW", "maskInS", "yC", "fCori", "recip_dxF", "recip_dyF", "recip_dxV",
             "recip_dyU", "dxF", "dyF", "dxV", "dyU", "recip_rAw", "recip_rAs")
    gf = {n: blocks.pad(np.asarray(env.g.f[n])[:, :1] if np.ndim(env.g.f[n]) == 4 else np.asarray(env.g.f[n]))
          for n in names}
    sg = {k: blocks.pad(np.asarray(v)) for k, v in env.sg(it).items()}
    ls = {k: blocks.pad(np.asarray(v)) for k, v in env.lsr_state(it).items()}

    def fn(p, ex, gf, sg, ls):
        out, passes = sl.seaice_lsr(p, Grid(gf, L), sg, ex, ls, record=True)
        counts = jnp.stack([jnp.stack([pr["L04"]["ICOUNT1"], pr["L04"]["ICOUNT2"]]) for pr in passes])
        return out["uIce"], out["vIce"], counts

    run = jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=(PS(), PS(AXIS), PS(AXIS), PS(AXIS), PS(AXIS)),
                                out_specs=(PS(AXIS), PS(AXIS), PS()), check_vma=True))
    u4, v4, counts = run(env.p, exs, gf, sg, ls)
    assert _ndiff(blocks.unpad(np.asarray(u4)), out1["uIce"]) == 0
    assert _ndiff(blocks.unpad(np.asarray(v4)), out1["vIce"]) == 0
    assert [tuple(int(c) for c in row) for row in np.asarray(counts)] == [(n, n) for n in COUNTS[it]]
