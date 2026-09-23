"""GGL90_CALC (plan Task 12): replay gate against the Fortran oracle, negative controls, effect and gradient tests.

Replay: the kernel gets exactly what GGL90_CALC reads in the V4r4 build — GGL90TKE and uVel/vVel on entry to the step
(S00_begin; nothing touches them before GGL90_CALC), sigmaR from GRAD_SIGMA (P02 tile dump), surfaceForcingU/V
(P01), recip_hFacC after UPDATE_R_STAR (S01), static geometry (G00) — and its four outputs are compared with the
dump right after the call (P04_ggl90) over the whole arrays: GGL90TKE on 2-OLx..sNx+OLx-1 (the outermost halo ring
must keep its input value), GGL90viscArU/V and GGL90diffKr on their loop ranges (zero elsewhere, as the V4r4 caller
zeroes them). Both oracles, every dumped iteration: SMOKE (no surface forcing: surface TKE = GGL90TKEsurfMin) and
FORCED (1992 wind stress drives the surface TKE).
Achieved max relative error (vs the field's max abs), measured 2026-09-23 on the dev node under conftest's
XLA_FLAGS (--xla_cpu_max_isa=AVX: no FMA contraction, like the -ffp-contract=off oracle;
--xla_disable_hlo_passes=algsimp) with the params passed as a jit argument: 0 (bitwise) for all four fields at
SMOKE it 1, 2 and FORCED it 1, 2, 3. For reference, with XLA's defaults (FMA contraction, algsimp on) the same
gate gave GGL90TKE <= 1.3e-15, GGL90viscArU 1.4e-16, GGL90viscArV 1.6e-16, GGL90diffKr 2.6e-16: algsimp
rewrites SQRTTWO*SQRTTKE/SQRT(N2) (ggl90_calc.F:223) as a multiply by RSQRT (1-ulp mixing lengths, ~28000
coefficient points), FMA contraction does the rest (TKE solve, Prandtl number).
Full V4r4 tree (oracle.FULL, iterations 1-3; M2.6a): the same replay; surfaceForcingU/V (P01) there include the
sea-ice ocean stress of SEAICE_MODEL, which runs before EXTERNAL_FORCING_SURF; GGL90_CALC itself takes no other
branch (gcov: only its diagnostics fills differ). Bitwise on all four fields (measured 2026-09-23); negative control:
the m2 plant fails on the full tree too.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import ggl90
from mitgcm_jax.pkgs.ggl90 import GGL90Params, ggl90_calc
from mitgcm_jax.tests import oracle

L = Layout()
OUT = ("GGL90TKE", "GGL90viscArU", "GGL90viscArV", "GGL90diffKr")
CASES = [(oracle.SMOKE, 1), (oracle.SMOKE, 2), (oracle.FORCED, 1), (oracle.FORCED, 2), (oracle.FORCED, 3),
         (oracle.FULL, 1), (oracle.FULL, 2), (oracle.FULL, 3)]
GATE_TOL = 0.0  # bitwise: measured 0 on every field and iteration (needs conftest XLA_FLAGS)
NEG_MIN = 1e-12  # a planted 1e-6 relative error moves its field by more than this (measured >= 6.4e-11)


@functools.lru_cache(maxsize=None)
def params(name):
    return GGL90Params.from_namelists(RunNamelists(oracle.run_dir(name)), Nr=L.Nr)


@functools.lru_cache(maxsize=None)
def case(name, it):
    ds = oracle.dumpset(name)

    def f(st, n):
        return oracle.field(ds, it, st, n)

    inp = dict(GGL90TKE=f("S00_begin", "GGL90TKE"), uVel=f("S00_begin", "uVel"), vVel=f("S00_begin", "vVel"),
               sigmaR=f("P02_rho_sigma_ivdc_mxlayer", "sigmaR"),
               surfaceForcingU=f("P01_external_forcing_surf", "surfaceForcingU"),
               surfaceForcingV=f("P01_external_forcing_surf", "surfaceForcingV"),
               recip_hFacC=f("S01_update_rstar_F", "recip_hFacC"))
    ref = {n: f("P04_ggl90", n) for n in OUT}
    return grid_from_dump(ds, it), inp, ref


_calc = jax.jit(ggl90_calc)  # params are a jit ARGUMENT (params_pytree leaves), never closed over


def run(p, g, inp, fn=None):
    return dict(zip(OUT, (np.asarray(x) for x in (fn or _calc)(p, g, **inp))))


def rel_errors(out, ref):
    """max |out - ref| / max |ref| per field, over the whole array (halos and the points GGL90_CALC does not
    write included: those must hold the value the Fortran array holds)."""
    return {n: float(np.max(np.abs(out[n] - ref[n])) / max(np.max(np.abs(ref[n])), 1e-300)) for n in OUT}


@pytest.mark.parametrize("name,it", CASES)
def test_replay_gate(name, it):
    g, inp, ref = case(name, it)
    err = rel_errors(run(params(name), g, inp), ref)
    print(name, it, {k: f"{v:.2e}" for k, v in err.items()})
    for n in OUT:
        assert err[n] <= GATE_TOL, (n, err[n])


def test_negative_controls_fail():
    """Each planted error (one constant perturbed by 1e-6 relative) fails the gate on the field it feeds:
    alpha -> TKE diffusion, ck -> all four outputs, m2 -> surface TKE (wind, FORCED), TKEbottom -> bottom Dirichlet
    value, p16 -> the ALLOW_GGL90_SMOOTH stencil (viscArU/V are not built with p16: they must stay bitwise).
    (A mskCor planted error cannot be tested on this grid: every facet-corner halo point is dry, see
    test_parameters_and_fixed_fields.)"""
    name, it = oracle.FORCED, 1
    g, inp, ref = case(name, it)
    p = params(name)
    plants = {
        "GGL90alpha": (dataclasses.replace(p, GGL90alpha=p.GGL90alpha * (1 + 1e-6)), ["GGL90TKE"]),
        "GGL90ck": (dataclasses.replace(p, GGL90ck=p.GGL90ck * (1 + 1e-6)), list(OUT)),
        "GGL90m2": (dataclasses.replace(p, GGL90m2=p.GGL90m2 * (1 + 1e-6)), ["GGL90TKE"]),
        "GGL90TKEbottom": (dataclasses.replace(p, GGL90TKEbottom=p.GGL90TKEbottom * (1 + 1e-6)), ["GGL90TKE"]),
    }
    for label, (pp, fields) in plants.items():
        err = rel_errors(run(pp, g, inp), ref)
        print(label, {k: f"{v:.2e}" for k, v in err.items()})
        for n in fields:
            assert err[n] > NEG_MIN, (label, n, err[n])
    orig = ggl90.p16
    try:
        ggl90.p16 = orig * (1 + 1e-6)
        err = rel_errors(run(p, g, inp, fn=jax.jit(lambda *a, **k: ggl90_calc(*a, **k))), ref)  # retrace
    finally:
        ggl90.p16 = orig
    print("p16", {k: f"{v:.2e}" for k, v in err.items()})
    assert err["GGL90diffKr"] > NEG_MIN, err
    assert err["GGL90viscArU"] <= GATE_TOL and err["GGL90viscArV"] <= GATE_TOL and err["GGL90TKE"] <= GATE_TOL, err


def test_parameters_and_fixed_fields():
    """V4r4 parameters from the namelists; facet corners of the exch2 layout equal the dumped tile headers;
    kLowC from maskC equals kLowC from h0FacC; unported options raise."""
    p = params(oracle.FORCED)
    assert (p.GGL90alpha, p.GGL90TKEmin, p.GGL90TKEbottom, p.GGL90TKEsurfMin) == (30.0, 1e-7, 1e-6, 1e-4)
    assert (p.mxlMaxFlag, p.mxlSurfFlag, p.GGL90_dirichlet, p.calcMeanVertShear) == (2, True, True, False)
    assert (p.GGL90ck, p.GGL90ceps, p.GGL90m2, p.GGL90mixingLengthMin) == (0.1, 0.7, 3.75, 1e-8)
    assert p.deltaTggl90 == 3600.0 and p.viscArNr == (5e-5,) * 50 and p.diffKrNrS == (1e-5,) * 50
    assert params(oracle.SMOKE) == p
    ds = oracle.dumpset(oracle.FORCED)
    recs = ds.tiles(1, "S00_begin", "theta")
    dims = p.dimsFacets
    corners = ggl90.tile_corners(dims, L)
    for t in range(L.nTiles):
        r = recs[t + 1]
        fNx, fNy = dims[2 * (r.face - 1)], dims[2 * (r.face - 1) + 1]
        W, S = r.tbx == 0, r.tby == 0
        E, N = r.tbx + L.sNx == fNx, r.tby + L.sNy == fNy
        assert tuple(corners[t]) == (W and S, E and S, W and N, E and N), t
    assert corners.sum() == 20  # 13x90x90 llc: 2+0+2 (facets 1,2), 4 (facet 3), 2+0+2 (facets 4,5)
    g = grid_from_dump(ds, 1)
    mc = ggl90.mskcor(p, L)
    assert np.count_nonzero(mc == 0.0) == 20 * L.OLx * L.OLy
    # every facet-corner halo point is dry at every level, so mskCor cannot change any output on this grid
    # (measured: mskCor = 1 everywhere gives bitwise-identical outputs)
    assert np.all(np.asarray(g.maskC)[:, :, :][np.broadcast_to((mc == 0.0)[:, None], g.maskC.shape)] == 0.0)
    k = np.arange(1, L.Nr + 1)[None, :, None, None]
    kl_h0 = np.max(np.where(g.h0FacC != 0.0, k, 0), axis=1)
    np.testing.assert_array_equal(np.asarray(ggl90.klowc(jnp.asarray(g.maskC))), kl_h0)
    for bad in (dict(mxlMaxFlag=0), dict(mxlSurfFlag=False), dict(calcMeanVertShear=True),
                dict(GGL90_dirichlet=False), dict(useIDEMIX=True)):
        with pytest.raises(NotImplementedError):
            dataclasses.replace(p, **bad).check()


def test_effect():
    """TKE >= GGL90TKEmin on wet points, 0 on dry ones. GGL90 raises Kv above the background: at k=2 on every wet
    point (mxlSurfFlag sets the mixing length to drF(1) = 10 m, so KappaM >= 0.1*10*sqrt(TKEmin) = 3.2e-4 and
    diffKr >= 3.2e-5 > diffKrNrS), and on ~11.5% of all wet points (measured 0.114-0.118 for diffKr, 0.119-0.123
    for viscArU on wet W points, all 5 dumped iterations; asserted > 0.08). The surface TKE equals GGL90TKEsurfMin
    without wind (SMOKE) and exceeds it on 78% of wet surface points under the 1992 wind (FORCED; asserted > 0.6)."""
    J, I = L.js(2 - L.OLy, L.sNy + L.OLy - 1), L.is_(2 - L.OLx, L.sNx + L.OLx - 1)
    Jc, Ic = L.js(1, L.sNy), L.is_(1, L.sNx)
    for name, it in [(oracle.SMOKE, 1), (oracle.FORCED, 1)]:
        g, inp, ref = case(name, it)
        p = params(name)
        out = run(p, g, inp)
        m = np.asarray(g.maskC)[:, :, J, I]
        tke = out["GGL90TKE"][:, :, J, I]
        assert np.all(tke[m == 1] >= p.GGL90TKEmin) and np.all(tke[m == 0] == 0.0)
        wet = np.asarray(g.maskC)[:, 1:, Jc, Ic] == 1
        wetW = np.asarray(g.maskW)[:, 1:, Jc, Ic] == 1
        kr = out["GGL90diffKr"][:, 1:, Jc, Ic]
        assert np.all(kr[:, 0][wet[:, 0]] > p.diffKrNrS[1])
        frac = np.mean(kr[wet] > p.diffKrNrS[0])
        fracU = np.mean(out["GGL90viscArU"][:, 1:, Jc, Ic][wetW] > p.viscArNr[0])
        print(name, it, "fraction of wet points above background: diffKr", frac, "viscArU", fracU)
        assert frac > 0.08 and fracU > 0.08, (frac, fracU)
        surf = tke[:, 0][m[:, 0] == 1]
        if name == oracle.SMOKE:
            assert np.all(surf == p.GGL90TKEsurfMin)
        else:
            fw = np.mean(surf > p.GGL90TKEsurfMin)
            print("fraction of wet surface points with wind-driven TKE:", fw)
            assert fw > 0.6, fw


GRAD_POINTS = {  # (tile, [k,] j, i) Python indices, FORCED it 1; picked from the data (dev/ggl90/probe.py):
    # uVel: wet W point, its 4 cells wet; local Ri of the 4 affected cells (5.1, 4.9, 11.4, 10.5) and
    #       (0, 1.65, 7.3, 7.8) away from the switches Ri = 0.2 and 5*Ri = 10
    "uVel": [(1, 5, 14, 66), (2, 10, 44, 31)],
    # GGL90TKE: wet, TKE = 4.5e-4 (Ri = 0) and 1.8e-4 (Ri = 0.54), well above GGL90TKEmin
    "GGL90TKE": [(7, 3, 57, 59), (8, 6, 79, 23)],
    # surfaceForcingU: wet W point where GGL90m2*uStar = 3.3e-4..5.6e-4 > 2*GGL90TKEsurfMin in both cells
    "surfaceForcingU": [(8, 56, 57), (8, 80, 48)],
}


def test_gradient_finite_and_matches_fd():
    """J = sum over 7x7 boxes around the GRAD_POINTS (all k) of TKE + 1e3*(diffKr + viscArU), outputs taken relative
    to their baseline values (so J(x0) = 0 and the FD differences are not swamped by the size of J). Weights are 0
    outside the boxes, so most cotangents are exactly 0 (the 0*inf case): dJ/d(every input) must be finite on every
    lane, dry and halo included. At each point the gradient matches a central FD, h = 1e-3, 1e-4 relative
    (measured best agreement 1e-10..2e-6 relative; asserted 1e-5)."""
    name, it = oracle.FORCED, 1
    g, inp, _ = case(name, it)
    p = params(name)
    jinp = {k: jnp.asarray(v) for k, v in inp.items()}
    base = [jnp.asarray(x) for x in _calc(p, g, **jinp)]
    w = np.zeros(L.shape3d)
    for pts in GRAD_POINTS.values():
        for pt in pts:
            t, j, i = pt[0], pt[-2], pt[-1]
            w[t, :, j - 3:j + 4, i - 3:i + 4] = 1.0
    w = jnp.asarray(w)

    def J(d, p, g, w, base):  # everything an argument: closed-over arrays become compile-time constants
        tke, vu, vv, kr = ggl90_calc(p, g, **d)
        return (jnp.sum(w * (tke - base[0])) + 1e3 * jnp.sum(w * (kr - base[3]))
                + 1e3 * jnp.sum(w * (vu - base[1])))

    Jf = functools.partial(jax.jit(J), p=p, g=g, w=w, base=base)
    grads = jax.jit(jax.grad(J))(jinp, p, g, w, base)
    for key, gr in grads.items():
        assert np.all(np.isfinite(np.asarray(gr))), key
    m = np.asarray(g.maskC)
    mW = np.asarray(g.maskW)
    for key, pts in GRAD_POINTS.items():
        x0 = np.asarray(jinp[key])
        gr = np.asarray(grads[key])
        for pt in pts:
            t, j, i = pt[0], pt[-2], pt[-1]
            wetp = mW[t, 0, j, i] if len(pt) == 3 else (mW[pt] if key == "uVel" else m[pt])
            assert wetp == 1, (key, pt)
            fds = []
            for h in (1e-3, 1e-4):
                hh = h * abs(x0[pt])
                xp, xm = x0.copy(), x0.copy()
                xp[pt] += hh
                xm[pt] -= hh
                dp, dm = dict(jinp), dict(jinp)
                dp[key], dm[key] = jnp.asarray(xp), jnp.asarray(xm)
                fds.append(float(Jf(dp) - Jf(dm)) / (2 * hh))
            best = min(abs(fd - gr[pt]) for fd in fds) / abs(gr[pt])
            print(key, pt, "grad", gr[pt], "fd", fds, "best rel", f"{best:.1e}")
            assert gr[pt] != 0.0, (key, pt)
            assert best <= 1e-5, (key, pt, gr[pt], fds)


def test_full_negative_control():
    """Full tree: the gate passes and the GGL90m2 plant (surface TKE from the ice-modified wind stress) fails it."""
    name, it = oracle.FULL, 1
    g, inp, ref = case(name, it)
    p = params(name)
    assert max(rel_errors(run(p, g, inp), ref).values()) <= GATE_TOL
    err = rel_errors(run(dataclasses.replace(p, GGL90m2=p.GGL90m2 * (1 + 1e-6)), g, inp), ref)
    assert err["GGL90TKE"] > NEG_MIN, err
