"""pkg/gmredi gates (plan Task 13): replay of GMREDI_CALC_TENSOR, GMREDI_DO_EXCH, GMREDI_RESIDUAL_FLOW and
GMREDI_CALC_DIFF against the Fortran oracle, at every dumped iteration of both oracles (SMOKE: no surface forcing,
iterations 1-2; FORCED: 1992 flux forcing, iterations 1-3).

Gates (all compare the whole array, halos included: every point is either computed or zero-initialised by the Fortran):
  P05_gmredi_tensor  sigmaX/Y/R (P02) + kapGM/kapRedi (G00) -> Kwx Kwy Kwz Kux Kvy Kuz Kvz GM_PsiX GM_PsiY
  P06_gmredi_exch    P05 GM_PsiX/Y -> EXCH_UV_XYZ_RL -> P06; Kwx..Kvz unchanged; full chain P02 -> P06
  T01_residual_flow  uVel/vVel/wVel (S12) + GM_PsiX/Y (P06) + recip_hFacW/S (from hFacW/S at S11) -> uFld vFld wFld
  T13 kappaRk        CALC_3D_DIFFUSIVITY terms (IVDC, diffKr, GGL90) + GMREDI_CALC_DIFF(Kwz of P05) -> kappaRk
Achieved (jit; conftest.py XLA_FLAGS --xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp; params passed as a jit
argument): bitwise, max relative error 0, for every field, iteration and oracle (the assertions require exact
equality). With algsimp enabled Kuz/Kvz differ by 1 ulp at ~66e3 points (A/sqrt(B) -> A*rsqrt(B)).
Negative controls: a 1e-6 relative change of slopeMax, slopeMaxSpec or GM_Kmin_horiz, a dropped maskp1, an unsigned
exchange, and a dropped GM term each fail the corresponding comparison.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import Grid, stack_tiles, unpack_vertical
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.pkgs import gmredi
from mitgcm_jax.tests import oracle

CASES = [(oracle.SMOKE, 1), (oracle.SMOKE, 2), (oracle.FORCED, 1), (oracle.FORCED, 2), (oracle.FORCED, 3)]
IDS = [f"{'smoke' if n == oracle.SMOKE else 'forced'}-it{it}" for n, it in CASES]
TENSOR = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX", "GM_PsiY")


def _drop(ds, it, stage, name=None):
    for key, recs in ds.index.items():
        if key[0] == it and key[1] == stage and (name is None or key[2] == name):
            for r in recs.values():
                r.drop()


def _fld(ds, it, stage, name):
    """Dumped field [T,(k),j,i]; the per-tile records are released after stacking (memory)."""
    a = oracle.field(ds, it, stage, name)
    _drop(ds, it, stage, name)
    return a


GRID_FIELDS = ("maskC", "maskW", "maskS", "maskInC", "kapGM", "kapRedi", "dxG", "dyG", "recip_rA")


def grid_subset(ds, it):
    """The G00_geometry fields the gmredi kernels read (grid_from_dump restricted to them: less I/O)."""
    L = Layout()
    f = {n: stack_tiles(ds, it, "G00_geometry", n, L) for n in GRID_FIELDS}
    f.update(unpack_vertical(ds.tiles(it, "G00_geometry", "vertical")[1].data[0], L))
    _drop(ds, it, "G00_geometry")
    return Grid(f, L)


def load_case(name, it):
    """Everything one (oracle, iteration) needs."""
    ds = oracle.dumpset(name)
    g = grid_subset(ds, it)
    nml = RunNamelists(oracle.run_dir(name))
    c = dict(g=g, p=gmredi.GMRediParams.from_namelists(nml), ds=ds, it=it, nml=nml)
    for n in ("sigmaX", "sigmaY", "sigmaR"):
        c[n] = _fld(ds, it, "P02_rho_sigma_ivdc_mxlayer", n)
    c["P05"] = {n: _fld(ds, it, "P05_gmredi_tensor", n) for n in TENSOR}
    c["P06"] = {n: _fld(ds, it, "P06_gmredi_exch", n) for n in TENSOR}
    return c


_KEEP = {}  # the SMOKE iteration-1 case serves the gate and the negative-control / effect / gradient tests


def _smoke1():
    if "smoke1" not in _KEEP:
        _KEEP["smoke1"] = load_case(oracle.SMOKE, 1)
    return _KEEP["smoke1"]


@pytest.fixture(scope="module", params=CASES, ids=IDS)
def c(request):
    if request.param == (oracle.SMOKE, 1):
        yield _smoke1()
        return
    d = load_case(*request.param)
    yield d
    d.clear()


@pytest.fixture(scope="module")
def smoke1():
    yield _smoke1()
    _KEEP.clear()


_tensor_jit = jax.jit(gmredi.gmredi_calc_tensor)
_residual_jit = jax.jit(gmredi.gmredi_residual_flow)


def tensor(p, c):
    out = _tensor_jit(p, c["g"], c["sigmaX"], c["sigmaY"], c["sigmaR"], c["g"].kapGM, c["g"].kapRedi)
    return {k: np.asarray(v) for k, v in out.items()}


def residual_inputs(c):
    ds, it, g = c["ds"], c["it"], c["g"]
    u, v, w = (_fld(ds, it, "S12_stagger_exchanges", n) for n in ("uVel", "vVel", "wVel"))
    # recip_hFacW/S as UPDATE_R_STAR(.TRUE.) leaves them (update_r_star.F:76-79; dry points keep the 0 of
    # ini_masks_etc.F:494,501); hFac at S11 = the S06 update (CALC_R_STAR changes rStarFac only)
    hW, hS = _fld(ds, it, "S11_calc_rstar", "hFacW"), _fld(ds, it, "S11_calc_rstar", "hFacS")
    rW = np.where(g.maskW != 0.0, 1.0 / np.where(g.maskW != 0.0, hW, 1.0), 0.0)
    rS = np.where(g.maskS != 0.0, 1.0 / np.where(g.maskS != 0.0, hS, 1.0), 0.0)
    return u, v, w, rW, rS


def assert_same(got, ref, what):
    got, ref = np.asarray(got), np.asarray(ref)
    scale = np.max(np.abs(ref))
    err = np.max(np.abs(got - ref)) / (scale if scale > 0 else 1.0)
    print(f"{what}: max rel. error {err:.3e}, max |ref| {scale:.3e}")
    assert np.array_equal(got, ref), f"{what}: max rel. error {err:.3e}, {np.sum(got != ref)} points differ"


def differs(got, ref):
    return not np.array_equal(np.asarray(got), np.asarray(ref))


# ---------------------------------------------------------------------------------------------------------------- gates
def test_tensor_replay_P05(c):
    """GMREDI_CALC_TENSOR (+ SLOPE_LIMIT, CALC_PSI_B, SLOPE_PSI) replayed from the dumped sigma fields: bitwise."""
    out = tensor(c["p"], c)
    for n in TENSOR:
        assert_same(out[n], c["P05"][n], f"{n} it={c['it']}")
    c["out"] = out


def test_exchange_replay_P06(c):
    """GMREDI_DO_EXCH: EXCH_UV_XYZ_RL(GM_PsiX, GM_PsiY, .TRUE.) of the P05 fields gives P06; the tensor is not
    exchanged; the chain P02 -> tensor -> exchange gives P06 as well. Bitwise."""
    ex = default_exchanger()
    px, py = gmredi.gmredi_do_exch(c["p"], ex, c["P05"]["GM_PsiX"], c["P05"]["GM_PsiY"])
    assert_same(px, c["P06"]["GM_PsiX"], "GM_PsiX")
    assert_same(py, c["P06"]["GM_PsiY"], "GM_PsiY")
    for n in TENSOR[:7]:
        assert_same(c["P06"][n], c["P05"][n], f"{n} untouched by GMREDI_DO_EXCH")
    out = c["out"] if "out" in c else tensor(c["p"], c)
    px, py = gmredi.gmredi_do_exch(c["p"], ex, out["GM_PsiX"], out["GM_PsiY"])
    assert_same(px, c["P06"]["GM_PsiX"], "chain GM_PsiX")
    assert_same(py, c["P06"]["GM_PsiY"], "chain GM_PsiY")


def test_residual_flow_replay_T01(c):
    """GMREDI_RESIDUAL_FLOW inside THERMODYNAMICS: uVel/vVel/wVel after the stagger exchanges (S12) plus the bolus
    velocity of the exchanged GM_PsiX/Y (P06) give the dumped uFld, vFld, wFld. Bitwise."""
    it = c["it"]
    u, v, w, rW, rS = residual_inputs(c)
    uF, vF, wF = _residual_jit(c["p"], c["g"], u, v, w, c["P06"]["GM_PsiX"], c["P06"]["GM_PsiY"], rW, rS)
    for n, got in (("uFld", uF), ("vFld", vF), ("wFld", wF)):
        assert_same(got, _fld(c["ds"], it, "T01_residual_flow", n), f"{n} it={it}")


def kappaRk_expected(c, Kwz, with_gm=True):
    """kappaRk of TEMP_INTEGRATE (CALC_3D_DIFFUSIVITY, GAD_TEMPERATURE, iMin=0..sNx+1: temp_integrate.F:165-168),
    in the Fortran order: IVDConvCount*ivdc_kappa + KbryanLewis79 (calc_3d_diffusivity.F:78-104) + diffKr
    (:112-117) + GMREDI_CALC_DIFF (:195) + GGL90_CALC_DIFF (:231; ggl90_calc_diff.F:52-55)."""
    ds, it, g, nml = c["ds"], c["it"], c["g"], c["nml"]
    L = g.layout
    ivdc_kappa = float(nml.get("data", "PARM01", "ivdc_kappa", default=0.0))  # set_defaults.F:216
    s = float(nml.get("data", "PARM01", "diffKrBL79surf", default=0.0))  # set_defaults.F:158
    d = float(nml.get("data", "PARM01", "diffKrBL79deep", default=0.0))  # set_defaults.F:159
    scl = float(nml.get("data", "PARM01", "diffKrBL79scl", default=200.0))  # set_defaults.F:160
    Ho = float(nml.get("data", "PARM01", "diffKrBL79Ho", default=-2000.0))  # set_defaults.F:161
    if nml.has("data", "PARM01", "diffKrNrS"):
        raise NotImplementedError("diffKrNrS set per level")
    diffKrNrS = float(nml.get("data", "PARM01", "diffKrS"))  # ini_parms.F:574 diffKrNrS(k) = diffKrS
    PI = 3.14159265358979323844  # EEPARAMS.h PI
    rF = np.asarray(g.rF)[:L.Nr]
    KBL79 = s + (d - s) * (np.arctan(-(rF - Ho) / scl) / PI + 0.5)
    ivdc = _fld(ds, it, "S04_oceanic_phys", "IVDConvCount")
    diffKr = _fld(ds, it, "S04_oceanic_phys", "diffKr")
    ggl = _fld(ds, it, "S04_oceanic_phys", "GGL90diffKr")
    k = ivdc * ivdc_kappa + KBL79[None, :, None, None]
    k = k + diffKr
    iMin, iMax, jMin, jMax = 0, L.sNx + 1, 0, L.sNy + 1
    if with_gm:
        k = np.array(gmredi.gmredi_calc_diff(g, jnp.asarray(k), jnp.asarray(Kwz), iMin, iMax, jMin, jMax))
    J, I = L.js(jMin, jMax), L.is_(iMin, iMax)
    k[..., J, I] = k[..., J, I] + (ggl[..., J, I] - diffKrNrS)
    return k


def test_calc_diff_replay_T13(c):
    """GMREDI_CALC_DIFF: kappaRk dumped in TEMP_INTEGRATE equals the CALC_3D_DIFFUSIVITY sum with Kwz*maskInC added
    over i,j = 0..sNx+1. Bitwise."""
    assert_same(kappaRk_expected(c, c["P05"]["Kwz"]), _fld(c["ds"], c["it"], "T13_temp_impl", "kappaRk"), "kappaRk")


# ---------------------------------------------------------------------------------------------- negative controls
def test_negative_controls(smoke1, monkeypatch):
    """Each planted error fails the gate it belongs to (live fixture: SMOKE iteration 1)."""
    c = smoke1
    p = c["p"]
    ref = c["P05"]
    # slopeMax of stableGmAdjTap (gmredi_slope_limit.F:593) * (1 + 1e-6) fails P05 Kwx .. Kvz
    out = tensor(dataclasses.replace(p, slopeMax=p.slopeMax * (1 + 1e-6)), c)
    assert all(differs(out[n], ref[n]) for n in ("Kwx", "Kwy", "Kwz", "Kuz", "Kvz"))
    # slopeMaxSpec (gmredi_slope_psi.F:377) * (1 + 1e-6) fails P05 GM_PsiX/Y
    out = tensor(dataclasses.replace(p, slopeMaxSpec=p.slopeMaxSpec * (1 + 1e-6)), c)
    assert differs(out["GM_PsiX"], ref["GM_PsiX"]) and differs(out["GM_PsiY"], ref["GM_PsiY"])
    # GM_Kmin_horiz * (1 + 1e-6) fails P05 Kux/Kvy
    out = tensor(dataclasses.replace(p, GM_Kmin_horiz=p.GM_Kmin_horiz * (1 + 1e-6)), c)
    assert differs(out["Kux"], ref["Kux"]) and differs(out["Kvy"], ref["Kvy"])
    # exchange without signs (EXCH_UV_XYZ_RL(..., .FALSE.)) fails P06
    ex = default_exchanger()
    px, py = ex.exch_uv_xy(ref["GM_PsiX"], ref["GM_PsiY"], False)
    assert differs(px, c["P06"]["GM_PsiX"]) or differs(py, c["P06"]["GM_PsiY"])
    # residual flow with maskp1 = 1 at k=Nr (gmredi_residual_flow.F:63 dropped) fails T01
    u, v, w, rW, rS = residual_inputs(c)
    with monkeypatch.context() as m:
        m.setattr(gmredi, "_maskp1", lambda Nr: jnp.ones((1, Nr, 1, 1)))
        uF, _, _ = gmredi.gmredi_residual_flow(p, c["g"], u, v, w, c["P06"]["GM_PsiX"], c["P06"]["GM_PsiY"], rW, rS)
    assert differs(uF, _fld(c["ds"], 1, "T01_residual_flow", "uFld"))
    # kappaRk without the GM term fails T13
    assert differs(kappaRk_expected(c, ref["Kwz"], with_gm=False), _fld(c["ds"], 1, "T13_temp_impl", "kappaRk"))


# ------------------------------------------------------------------------------------------------------ effect test
def test_stable_adjoint_taper_is_active(smoke1):
    """On the live fixture both branches of both stableGmAdjTap limiters are taken at wet points: removing a limiter
    (limit -> inf) changes the output at some but not all wet points, and at no dry point."""
    c = smoke1
    p, g = c["p"], c["g"]
    ref = c["P05"]
    free = tensor(dataclasses.replace(p, slopeMax=np.inf), c)
    changed = (free["Kwx"] != ref["Kwx"]) | (free["Kwy"] != ref["Kwy"])
    wet = np.asarray(g.maskC) != 0.0
    wet[:, 0] = False  # level 1 is not computed (k loop Nr..2)
    n_lim, n_wet = int(np.sum(changed & wet)), int(np.sum(wet))
    print(f"tensor slope limited at {n_lim} of {n_wet} wet W points")
    assert 0 < n_lim < n_wet, (n_lim, n_wet)
    assert not np.any(changed & ~wet)
    free = tensor(dataclasses.replace(p, slopeMaxSpec=np.inf), c)
    changedX = free["GM_PsiX"] != ref["GM_PsiX"]
    wetW = np.asarray(g.maskW) != 0.0
    wetW[:, 0] = False
    n_lim, n_wet = int(np.sum(changedX & wetW)), int(np.sum(wetW))
    print(f"bolus slope limited at {n_lim} of {n_wet} wet U points (k>1)")
    assert 0 < n_lim < n_wet, (n_lim, n_wet)
    assert not np.any(changedX & ~wetW)
    # the clamp GM_Kmin_horiz (gmredi_calc_tensor.F:747) is active at some points and not at others
    assert np.any(ref["Kux"] == p.GM_Kmin_horiz) and np.any(ref["Kux"] > p.GM_Kmin_horiz)


# --------------------------------------------------------------------------------------------------- gradients
def test_gradient_kapGM_and_kapRedi(smoke1):
    """d/d kapGM of a weighted sum of the residual velocity (tensor -> exchange -> residual flow) and d/d kapRedi of a
    weighted sum of the tensor: finite on every lane (dry and halo included; also d/d sigmaX,Y,R) and equal to central
    finite differences at random wet points (h-sweep h = 10, 1, 0.1 m^2/s; the FD is formed from output differences,
    so unaffected points cancel exactly). Both functionals are linear in the kappas away from the GM_Kmin_horiz clamp,
    so the plateau is flat; the measured max relative AD-FD difference is printed (-s)."""
    c = smoke1
    p, g = c["p"], c["g"]
    ex = default_exchanger()
    u, v, w, rW, rS = (jnp.asarray(a) for a in residual_inputs(c))
    rng = np.random.default_rng(0)
    ru, rv, rw = (jnp.asarray(rng.normal(size=u.shape)) for _ in range(3))
    rt = {n: jnp.asarray(rng.normal(size=u.shape)) for n in TENSOR[:7]}
    sX, sY, sR = (jnp.asarray(c[n]) for n in ("sigmaX", "sigmaY", "sigmaR"))
    kG, kR = jnp.asarray(g.kapGM), jnp.asarray(g.kapRedi)

    def chain(p, g, kapGM, kapRedi, sX, sY, sR, u, v, w, rW, rS):
        out = gmredi.gmredi_calc_tensor(p, g, sX, sY, sR, kapGM, kapRedi)
        px, py = gmredi.gmredi_do_exch(p, ex, out["GM_PsiX"], out["GM_PsiY"])
        return gmredi.gmredi_residual_flow(p, g, u, v, w, px, py, rW, rS)

    def J_gm(p, g, kapGM, kapRedi, sX, sY, sR, u, v, w, rW, rS, ru, rv, rw):
        uF, vF, wF = chain(p, g, kapGM, kapRedi, sX, sY, sR, u, v, w, rW, rS)
        return jnp.sum(ru * uF) + jnp.sum(rv * vF) + jnp.sum(rw * wF)

    def J_redi(p, g, kapGM, kapRedi, sX, sY, sR, rt):
        out = gmredi.gmredi_calc_tensor(p, g, sX, sY, sR, kapGM, kapRedi)
        return sum(jnp.sum(rt[n] * out[n]) for n in rt)

    gG, gsX, gsY, gsR = jax.jit(jax.grad(J_gm, argnums=(2, 4, 5, 6)))(p, g, kG, kR, sX, sY, sR, u, v, w, rW, rS,
                                                                        ru, rv, rw)
    gR = jax.jit(jax.grad(J_redi, argnums=3))(p, g, kG, kR, sX, sY, sR, rt)
    gG, gR = np.asarray(gG), np.asarray(gR)
    for a in (gG, gR, gsX, gsY, gsR):  # every lane finite: guarded divisions / sqrt, AD-safe where
        assert np.all(np.isfinite(np.asarray(a)))

    # directional FD from output differences (unaffected points cancel exactly)
    def dJ(p, g, ka, kb, kR, sX, sY, sR, u, v, w, rW, rS, ru, rv, rw):
        a = (sX, sY, sR, u, v, w, rW, rS)
        return sum(jnp.sum(r * (x - y)) for r, x, y in zip((ru, rv, rw), chain(p, g, ka, kR, *a),
                                                            chain(p, g, kb, kR, *a)))

    dJ_gm = jax.jit(dJ)
    tens = jax.jit(gmredi.gmredi_calc_tensor)
    maskC = np.asarray(g.maskC) != 0
    ptsG = np.argwhere(maskC & (np.abs(gG) > 0))
    ptsR = np.argwhere(maskC & (np.abs(gR) > 0) & (np.asarray(g.kapRedi) > 2 * p.GM_Kmin_horiz))
    L = g.layout
    worst = {0: 0.0, 1: 0.0}
    for which, grad, pts in ((0, gG, ptsG), (1, gR, ptsR)):
        assert len(pts) > 0
        for pt in (tuple(pts[i]) for i in rng.choice(len(pts), 2, replace=False)):
            ad = float(grad[pt])
            for h in (10.0, 1.0, 0.1):
                e = jnp.zeros(L.shape3d).at[pt].set(h)
                if which == 0:
                    num = float(dJ_gm(p, g, kG + e, kG - e, kR, sX, sY, sR, u, v, w, rW, rS, ru, rv, rw))
                else:
                    ta, tb = tens(p, g, sX, sY, sR, kG, kR + e), tens(p, g, sX, sY, sR, kG, kR - e)
                    num = sum(float(jnp.sum(rt[n] * (ta[n] - tb[n]))) for n in rt)
                fd = num / (2 * h)
                rel = abs(fd - ad) / abs(ad)
                worst[which] = max(worst[which], rel)
                assert rel <= 1e-9, (which, pt, h, fd, ad)
    print("max rel AD-FD difference: kapGM %.2e, kapRedi %.2e" % (worst[0], worst[1]))


def test_params_hard_errors():
    """Unported configurations are hard errors at setup."""
    p = gmredi.GMRediParams.from_namelists(RunNamelists(oracle.run_dir(oracle.SMOKE)))
    assert p.GM_taper_scheme == "stableGmAdjTap" and p.GM_skewflx == 0.0 and p.GM_ExtraDiag and p.useMultiDimAdvec
    for bad in (dict(GM_taper_scheme="gkw91"), dict(GM_AdvForm=False), dict(GM_Visbeck_alpha=0.015),
                dict(GM_ExtraDiag=False), dict(useMultiDimAdvec=False)):
        with pytest.raises(NotImplementedError):
            dataclasses.replace(p, **bad).check()
