"""EOS (JMD95Z), GRAD_SIGMA, CALC_IVDC, CALC_OCE_MXLAYER (plan Task 10): replay gate at P02_rho_sigma_ivdc_mxlayer.

Inputs: theta, salt, hMixLayer of S00_begin (nothing between the start of the step and the DO_OCEANIC_PHYS tile loop
changes them in V4r4: no sea ice, no freezing, no frazil); geometry of G00_geometry. Outputs are compared at EVERY
point of the tile, halos included, because the Fortran computes every point (full-tile loop ranges):
rhoInSitu, sigmaX, sigmaY, sigmaR, IVDConvCount, hMixLayer; both oracles (SMOKE: no surface forcing, iterations 1, 2;
FORCED: 1992 flux forcing, iterations 1, 2, 3).

Full V4r4 tree (oracle.FULL, iterations 1-3; M2.6a): the same gate. The full tree's data.diagnostics requests
MXLDEPTH, so CALC_OCE_MXLAYER computes hMixLayer (method 1 with FIND_ALPHA, core/mxlayer.py; the flux-forced tree
leaves it 0); SEAICE_MODEL, which runs before this stage in the full tree, does not change theta/salt, so the inputs
are still those of S00_begin.

Achieved: bitwise (max rel. error 0) for every field, oracle and iteration, with XLA FMA contraction off (full tree
measured 2026-09-23: 0 differing values, hMixLayer included).
Negative controls: one EOS coefficient * (1 + 1e-6), recip_dxC/recip_dyC swapped, recip_drC shifted by one level:
each fails the same comparison; full tree: hMixCriteria * (1 + 1e-6) and FIND_ALPHA * (1 + 1e-6) each fail the
hMixLayer comparison, and kLowC - 1 does too. The cube-sphere corner fill is invisible in this output (masked), so it
is checked against a transcription of the Fortran loops instead.
Gradient: d/d(theta, salt) of weighted sums of rhoInSitu, sigmaX, sigmaY, sigmaR is finite at every point (dry points
hold S = 0: the S**1.5 term is guarded) and matches central finite differences at wet points; full tree: the same for
hMixLayer (FIND_ALPHA and the interpolation factor tmpFac).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core import eos as eos_mod
from mitgcm_jax.core import grad_sigma as gs
from mitgcm_jax.core import mxlayer as mxl_mod
from mitgcm_jax.core.phi_hyd import kLowC_from_hFac
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.tests import oracle

STAGE = "P02_rho_sigma_ivdc_mxlayer"
FIELDS = ("rhoInSitu", "sigmaX", "sigmaY", "sigmaR", "IVDConvCount", "hMixLayer")
FF_CASES = [(oracle.SMOKE, 1), (oracle.SMOKE, 2), (oracle.FORCED, 1), (oracle.FORCED, 2), (oracle.FORCED, 3)]
FULL_CASES = [(oracle.FULL, 1), (oracle.FULL, 2), (oracle.FULL, 3)]
CASES = FF_CASES + FULL_CASES
# max |jax - fortran| / max |fortran| over the whole tile. Achieved: 0 (bitwise) for every field, both oracles, all
# iterations, with FMA contraction off (conftest: --xla_cpu_max_isa=AVX); so the gate is bitwise. With FMA on
# (default AVX2): rhoInSitu 1.6e-14, sigmaX 2.5e-13, sigmaY 3.8e-13, sigmaR 1.0e-13, IVDConvCount 0.
TOL = {f: 0.0 for f in FIELDS}


def fma_contracted():
    """True if XLA fuses a*b+c into one FMA (default on AVX2 CPUs). The oracle is built with -ffp-contract=off, so
    bitwise gates need XLA_FLAGS=--xla_cpu_max_isa=AVX (no FMA3); with FMA the EOS differs from Fortran by a few ulp
    at ~0.1 % of the points (measured: rhoInSitu rel. error 1.6e-14, sigmaX 2.5e-13)."""
    a = jnp.full((4096,), 1.0 + 2.0 ** -30)
    c = jnp.full((4096,), -(1.0 + 2.0 ** -29))
    r = jax.jit(lambda a, b, c: a * b + c)(a, a, c)
    return bool(np.any(np.asarray(r) != 0.0))


def require_exact_fp():
    if fma_contracted():
        pytest.fail("XLA contracts a*b+c into FMA: the Fortran oracle is -ffp-contract=off; run the gates with "
                    "XLA_FLAGS=--xla_cpu_max_isa=AVX (see fma_contracted)")


def relerr(got, ref):
    d = float(np.max(np.abs(np.asarray(got) - ref)))
    m = float(np.max(np.abs(ref)))
    return d / m if m > 0 else d


@functools.lru_cache(maxsize=1)
def case(name, it):
    ds = oracle.dumpset(name)
    g = grid_from_dump(ds, it)
    p = gs.RhoSigmaParams.from_namelists(RunNamelists(oracle.run_dir(name)), g)
    inp = {n: oracle.field(ds, it, "S00_begin", n) for n in ("theta", "salt", "hMixLayer")}
    inp["kLowC"] = np.asarray(kLowC_from_hFac(np.asarray(g.h0FacC)))   # as model.setup (ini_masks_etc.F klowC)
    ref = {n: oracle.field(ds, it, STAGE, n) for n in FIELDS}
    return g, p, inp, ref


def run(p, g, inp):
    """params and grid are jit ARGUMENTS (float parameters traced, never folded as constants)."""
    f = jax.jit(gs.rho_sigma_ivdc_mxlayer)
    return {k: np.asarray(v) for k, v in f(p, g, inp["theta"], inp["salt"], inp["hMixLayer"], inp["kLowC"]).items()}


@pytest.mark.parametrize("name,it", CASES)
def test_p02_replay_gate(name, it):
    require_exact_fp()
    g, p, inp, ref = case(name, it)
    out = run(p, g, inp)
    errs = {f: relerr(out[f], ref[f]) for f in FIELDS}
    print(name, it, errs)
    for f in FIELDS:
        assert errs[f] <= TOL[f], (f, errs[f])
    # not vacuous: stratified, convecting somewhere, horizontal gradients present
    assert np.abs(ref["sigmaX"]).max() > 0 and np.abs(ref["sigmaY"]).max() > 0 and np.abs(ref["sigmaR"]).max() > 0
    assert ref["IVDConvCount"].sum() > 0
    # CALC_OCE_MXLAYER computes only in the full tree (MXLDEPTH diagnostic); ff keeps the INI_DYNVARS zero
    assert p.mxl.calcMixLayerDepth == (name == oracle.FULL)
    if name == oracle.FULL:
        assert (ref["hMixLayer"] > 0).mean() > 0.3
        if it == 1:                                        # INI_DYNVARS zero in: the output is computed, not kept
            assert np.all(inp["hMixLayer"] == 0.0)
    else:
        assert np.all(ref["hMixLayer"] == 0.0)


def test_pref4eos_matches_phiref():
    """pRef4EOS(k) (set_ref_state.F:97) equals rhoConst*phiRef(2k) (dumped vertical record) to rounding."""
    g, p, _, _ = case(*FF_CASES[-1])
    np.testing.assert_allclose(np.array(p.eos.pRef4EOS), p.eos.rhoConst * g.phiRef[1:2 * g.layout.Nr:2],
                               rtol=1e-15, atol=0)


def test_negative_control_eos_coefficient():
    g, p, inp, ref = case(*FF_CASES[-1])
    KP = list(p.eos.KP)
    KP[0] = KP[0] * (1.0 + 1e-6)
    e2 = eos_mod.EOSParams(eosType=p.eos.eosType, rhoConst=p.eos.rhoConst, selectP_inEOS_Zc=0,
                           pRef4EOS=p.eos.pRef4EOS, KP=tuple(KP))
    out = run(gs.RhoSigmaParams(eos=e2, calcConvect=p.calcConvect, mxl=p.mxl), g, inp)
    assert relerr(out["rhoInSitu"], ref["rhoInSitu"]) > 1e-12  # well above rounding (FMA-level: 1.6e-14)


def test_negative_control_swapped_metric():
    """Plant: recip_dyC in place of recip_dxC (and vice versa) in GRAD_SIGMA: sigmaX / sigmaY fail."""
    g, p, inp, ref = case(*FF_CASES[-1])
    out = run(p, g.replace(recip_dxC=g.recip_dyC, recip_dyC=g.recip_dxC), inp)
    assert relerr(out["sigmaX"], ref["sigmaX"]) > 1e-6
    assert relerr(out["sigmaY"], ref["sigmaY"]) > 1e-6


def test_corner_fill_not_observable_in_sigma(monkeypatch):
    """FILL_CS_CORNER_TR_RL cannot be checked through this gate: without it sigmaX/Y are bitwise unchanged, because
    maskW/maskS vanish on every stencil that touches a facet-corner halo block (hFacW/S there come from R_low of
    the never-exchanged corner, = 0). Its index mapping is checked against the Fortran loops in
    test_corner_fill_transcription instead."""
    g, p, inp, ref = case(*FF_CASES[-1])
    monkeypatch.setattr(gs, "fill_cs_corner_tr", lambda a, *args, **kw: a)
    out = run(p, g, inp)
    assert relerr(out["sigmaX"], ref["sigmaX"]) == 0.0 and relerr(out["sigmaY"], ref["sigmaY"]) == 0.0


def test_corner_fill_transcription():
    """fill_cs_corner_tr vs a plain transcription of fill_cs_corner_tr_rl.F:165-192 (dir 1) and :235-262 (dir 2)
    on an index-coded field; corner tiles from w2_set_tile2tiles.F (SW: 1,4,7,8,11; SE: 1,4,7,10,13;
    NE: 3,6,7,10,13; NW: 3,6,7,8,11)."""
    from mitgcm_jax.layout import Layout
    L = Layout()
    c = gs.cs_corners(L)
    assert [t + 1 for t in range(L.nTiles) if c["SW"][t]] == [1, 4, 7, 8, 11]
    assert [t + 1 for t in range(L.nTiles) if c["NE"][t]] == [3, 6, 7, 10, 13]
    a = np.arange(np.prod(L.shape2d), dtype=np.float64).reshape(L.shape2d) + 1.0
    sNx, sNy, OLx, OLy = L.sNx, L.sNy, L.OLx, L.OLy

    def at(T, x, y):  # Fortran trFld(x, y) of one tile -> numpy element index
        return (y - 1 + OLy, x - 1 + OLx)

    for d in (1, 2):
        exp = a.copy()
        for t in range(L.nTiles):
            T = exp[t]
            for corner in ("SW", "SE", "NW", "NE"):
                if not c[corner][t]:
                    continue
                for j in range(1, OLy + 1):
                    for i in range(1, OLx + 1):
                        if d == 1:
                            tgt, src = {"SW": ((1 - i, 1 - j), (1 - j, i)),
                                        "SE": ((sNx + i, 1 - j), (sNx + j, i)),
                                        "NW": ((1 - i, sNy + j), (1 - j, sNy + 1 - i)),
                                        "NE": ((sNx + i, sNy + j), (sNx + j, sNy + 1 - i))}[corner]
                        else:
                            tgt, src = {"SW": ((1 - i, 1 - j), (j, 1 - i)),
                                        "SE": ((sNx + i, 1 - j), (sNx + 1 - j, 1 - i)),
                                        "NW": ((1 - i, sNy + j), (j, sNy + i)),
                                        "NE": ((sNx + i, sNy + j), (sNx + 1 - j, sNy + i))}[corner]
                        T[at(T, *tgt)] = 1.0 * T[at(T, *src)]
        got = np.asarray(gs.fill_cs_corner_tr(jnp.asarray(a), d, False, L))
        np.testing.assert_array_equal(got, exp)
        assert not np.array_equal(got, a)


def test_negative_control_shifted_recip_drC():
    g, p, inp, ref = case(*FF_CASES[-1])
    r = np.asarray(g.recip_drC)
    out = run(p, g.replace(recip_drC=np.concatenate([r[1:], r[-1:]])), inp)
    assert relerr(out["sigmaR"], ref["sigmaR"]) > 1e-12


GRAD_FIELDS = ("rhoInSitu", "sigmaX", "sigmaY", "sigmaR")


def test_gradient_finite_and_fd():
    """(a) Random weights on every point of rhoInSitu, sigmaX, sigmaY, sigmaR (dry, halo and corner points
    included): d/d(theta, salt) is finite everywhere. (b) Per field f, J_f = sum(W_f * f) over a 3x3x3 box around
    each of a few wet points (boxes far apart, so one backward pass and one FD evaluation serve all points):
    dJ_f/dtheta(p), dJ_f/dsalt(p) by AD vs central differences, h-sweep 1e-2..1e-5 (plateau: min over h)."""
    g, p, inp, _ = case(*FF_CASES[-1])
    L = g.layout
    th0, sa0 = inp["theta"], inp["salt"]  # numpy; every jitted call gets numpy in, numpy out (no eager jax ops)
    fwd_pg = jax.jit(lambda p, g, th, sa: {f: v for f, v in gs.rho_sigma_ivdc_mxlayer(p, g, th, sa, inp["hMixLayer"])
                                           .items() if f in GRAD_FIELDS})
    # the backward pass as one jitted function of (params, grid, state, cotangent): jitting the vjp closure instead
    # would capture its residuals (~3 GB) as constants
    vjp_pg = jax.jit(lambda p, g, th, sa, ct: jax.vjp(lambda a, b: fwd_pg(p, g, a, b), th, sa)[1](ct))

    def fwd(th, sa):
        return {f: np.asarray(v) for f, v in fwd_pg(p, g, th, sa).items()}

    def vjp(ct):
        return tuple(np.asarray(x) for x in vjp_pg(p, g, th0, sa0, ct))

    shape = th0.shape
    rng = np.random.default_rng(0)
    gth, gsa = vjp({f: rng.normal(size=shape) for f in GRAD_FIELDS})
    assert np.all(np.isfinite(gth)) and np.all(np.isfinite(gsa))
    assert (np.asarray(g.maskC) == 0).any() and (inp["salt"] <= 0).any()  # guarded lanes exercised

    maskC = np.asarray(g.maskC)
    pts = [pt for pt in [(1, 4, L.jj(45), L.ii(45)), (9, 10, L.jj(30), L.ii(60)), (4, 1, L.jj(20), L.ii(70))]
           if maskC[pt[0], pt[1] - 1:pt[1] + 2, pt[2] - 1:pt[2] + 2, pt[3] - 1:pt[3] + 2].all()]
    assert len(pts) >= 2, pts
    boxes = [(t, slice(k - 1, k + 2), slice(j - 1, j + 2), slice(i - 1, i + 2)) for t, k, j, i in pts]
    W = {f: [rng.normal(size=(3, 3, 3)) for _ in pts] for f in GRAD_FIELDS}
    ad = {}
    for f in GRAD_FIELDS:
        ct = {n: np.zeros(shape) for n in GRAD_FIELDS}
        for b, w in zip(boxes, W[f]):
            ct[f][b] = w
        gth, gsa = vjp(ct)
        ad[f] = [(gth[pt], gsa[pt]) for pt in pts]
    idx = tuple(np.array(pts).T)
    worst = 0.0
    for var in (0, 1):
        x0 = (th0, sa0)[var]
        fd = {f: [[] for _ in pts] for f in GRAD_FIELDS}
        for h in (1e-2, 1e-3, 1e-4, 1e-5):
            outs = []
            for sg in (1.0, -1.0):
                x = x0.copy()
                x[idx] += sg * h
                outs.append(fwd(x, sa0) if var == 0 else fwd(th0, x))
            for f in GRAD_FIELDS:
                for n, b in enumerate(boxes):
                    jp, jm = (float(np.sum(o[f][b] * W[f][n])) for o in outs)
                    fd[f][n].append((jp - jm) / (2 * h))
        for f in GRAD_FIELDS:
            for n, pt in enumerate(pts):
                a = ad[f][n][var]
                if a == 0.0:
                    continue
                e = min(abs(x - a) / abs(a) for x in fd[f][n])
                print(pt, ("theta", "salt")[var], f, a, e)
                worst = max(worst, e)
    assert worst < 1e-6, worst


# ---------------------------------------------------------------------------------------------------------------------
# full tree (M2.6a): CALC_OCE_MXLAYER method 1 with FIND_ALPHA
# ---------------------------------------------------------------------------------------------------------------------
def test_full_mxlayer_negative_controls(monkeypatch):
    """Each plant fails the full-tree hMixLayer comparison: hMixCriteria * (1 + 1e-6) (calc_oce_mxlayer.F:103-104),
    FIND_ALPHA * (1 + 1e-6) (:96-98), kLowC - 1 (the k.LE.klowC test, :118)."""
    import dataclasses

    g, p, inp, ref = case(*FULL_CASES[0])
    assert relerr(run(p, g, inp)["hMixLayer"], ref["hMixLayer"]) == 0.0
    bad = dataclasses.replace(p, mxl=dataclasses.replace(p.mxl, hMixCriteria=p.mxl.hMixCriteria * (1 + 1e-6)))
    assert relerr(run(bad, g, inp)["hMixLayer"], ref["hMixLayer"]) > 0.0
    assert relerr(run(p, g, dict(inp, kLowC=np.maximum(inp["kLowC"] - 1, 0)))["hMixLayer"], ref["hMixLayer"]) > 1e-3
    fa = eos_mod.find_alpha
    monkeypatch.setattr(eos_mod, "find_alpha", lambda *a: fa(*a) * (1.0 + 1e-6))
    fresh = jax.jit(lambda *a: gs.rho_sigma_ivdc_mxlayer(*a))     # a new function: no cached trace of the original
    out = fresh(p, g, inp["theta"], inp["salt"], inp["hMixLayer"], inp["kLowC"])
    assert relerr(np.asarray(out["hMixLayer"]), ref["hMixLayer"]) > 0.0


def test_full_mxlayer_off_without_mxldepth():
    """calcMixLayerDepth follows DIAGNOSTICS_IS_ON('MXLDEPTH') (calc_oce_mxlayer.F:74-76): the full run directory's
    namelists with MXLDEPTH removed from data.diagnostics give .FALSE.; useDiagnostics=F gives .FALSE. (namelist texts
    substituted in memory)."""
    from mitgcm_jax.io.namelist import parse_namelist
    from mitgcm_jax.params_io import diagnostics_is_on

    rundir = oracle.run_dir(oracle.FULL)
    nml = RunNamelists(rundir)
    assert diagnostics_is_on(nml, "MXLDEPTH") and diagnostics_is_on(nml, "DRHODR")
    assert not diagnostics_is_on(nml, "WdRHO_P") and not diagnostics_is_on(nml, "WdRHOdP")   # doDiagsRho = 3
    assert mxl_mod.MxLayerParams.from_namelists(nml).calcMixLayerDepth
    diag = (rundir / "data.diagnostics").read_text()
    pkg = (rundir / "data.pkg").read_text()
    n1 = RunNamelists(rundir)
    n1._cache["data.diagnostics"] = parse_namelist(diag.replace("MXLDEPTH", "XXXXXXXX"))
    assert not mxl_mod.MxLayerParams.from_namelists(n1).calcMixLayerDepth
    n2 = RunNamelists(rundir)
    n2._cache["data.pkg"] = parse_namelist(pkg.replace("useDiagnostics=.TRUE.", "useDiagnostics=.FALSE."))
    assert not mxl_mod.MxLayerParams.from_namelists(n2).calcMixLayerDepth


def test_full_mxlayer_gradient_finite_and_fd():
    """d(sum W*hMixLayer)/d(theta, salt), W random on every point: finite everywhere (guarded tmpFac division, S = 0
    on dry points). At wet columns whose mixed-layer base lies strictly inside a level interval (0 < tmpFac < 1, the
    derivative is smooth there), AD vs central differences with an h-sweep (plateau: min over h)."""
    g, p, inp, ref = case(*FULL_CASES[0])
    fwd = jax.jit(lambda p, g, th, sa: gs.rho_sigma_ivdc_mxlayer(p, g, th, sa, inp["hMixLayer"], inp["kLowC"])
                  ["hMixLayer"])
    vjp = jax.jit(lambda p, g, th, sa, ct: jax.vjp(lambda a, b: fwd(p, g, a, b), th, sa)[1](ct))
    th0, sa0 = inp["theta"], inp["salt"]
    rng = np.random.default_rng(1)
    gth, gsa = (np.asarray(x) for x in vjp(p, g, th0, sa0, rng.normal(size=ref["hMixLayer"].shape)))
    assert np.all(np.isfinite(gth)) and np.all(np.isfinite(gsa))
    # interior points with a mixed-layer base between two wet levels, not at the column bottom
    L = g.layout
    h = ref["hMixLayer"]
    rF, rC = np.asarray(g.rF), np.asarray(g.rC)
    kl = inp["kLowC"]
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    cand = np.argwhere((h[:, J, I] > 20.0) & (h[:, J, I] < 500.0) & (kl[:, J, I] > 20))
    pts = [tuple(c) for c in cand[:: max(1, len(cand) // 5)][:5]]
    worst = 0.0
    for t, j, i in pts:
        j, i = j + L.OLy, i + L.OLx
        ct = np.zeros_like(h)
        ct[t, j, i] = 1.0
        gth, gsa = (np.asarray(x) for x in vjp(p, g, th0, sa0, ct))
        # the level just above the mixed-layer base (it carries the largest derivative)
        k = int(np.searchsorted(-rC, h[t, j, i]))
        for var, gad in ((0, gth), (1, gsa)):
            a = gad[t, k, j, i]
            if a == 0.0:
                continue
            fd = []
            for hh in (1e-3, 1e-4, 1e-5, 1e-6):
                vals = []
                for sg in (1.0, -1.0):
                    x = (th0, sa0)[var].copy()
                    x[t, k, j, i] += sg * hh
                    o = np.asarray(fwd(p, g, x, sa0) if var == 0 else fwd(p, g, th0, x))
                    vals.append(o[t, j, i])
                fd.append((vals[0] - vals[1]) / (2 * hh))
            e = min(abs(f - a) / abs(a) for f in fd)
            print((t, k, j, i), ("theta", "salt")[var], a, fd, e)
            worst = max(worst, e)
    assert len(pts) >= 3
    assert worst < 1e-5, worst
