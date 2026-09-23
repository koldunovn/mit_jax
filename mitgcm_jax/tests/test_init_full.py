"""Full V4r4 initialisation gate (plan M2.6a, tier1x): `model.setup` + `init.state_from_pickup` on the full-tree run
directory (oracle.FULL = full_jaxdump_v5: useSEAICE, useCTRL with the production xx_*.0000000129 controls,
geothermal flux, bulk-formula EXF) against the Fortran state at the start of iteration 1.

  - grid from the files (grid_from_files) + the ctrl-adjusted mixing fields kapGM, kapRedi, diffKr (INI_MIXING +
    CTRL_MAP_INI_GENARR) == G00_geometry, every G2D/G3D/R3D field and the vertical rows, halos included;
    the static sea-ice fields of SEAICE_INIT_VARIA (HEFFM, k1AtC, k1AtZ, k2AtC, k2AtZ) == G01_seaice_geometry.
  - State == Fortran: every S00_begin field and the r* fields of G00 group R; the sea-ice state AREA, HEFF,
    HSNOW, TICES (7 categories), UICE, VICE == S00i_begin_ice_exf; sIceLoad (SEAICE_INIT_VARIA) in S00_begin; the 28
    EXF_FIELDS arrays of the full EXF_INIT_VARIA (M2.6b-2) == S00i_begin_ice_exf ('b' group) and S00_begin ('x'
    group); DYN_CARRY == seaice_model.dyn_carry_init (== the SEAICE_DYNSOLVER entry values at iteration 1,
    tests/test_seaice_model.py::test_carried_state_entry_values).
`production()` (setup + state_from_pickup, ~4 min) is cached for the session: tests/test_step_full.py reuses it.
Measured 2026-09-23 (16 CPU cores): bitwise, 0 differing values in every compared field, halos included.
Negative controls: one pickup theta value * (1 + 1e-6) changes theta and totPhiHyd; one pickup_seaice HEFF value
* (1 + 1e-6) changes HEFF and sIceLoad; SEAICE_rhoSnow * (1 + 1e-6) changes sIceLoad; without the doMapTice copy
TICES(2..7) fail; without the uIce/vIce vector exchange signs UICE/VICE fail.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax import init
from mitgcm_jax.grid.geometry import G2D, G3D, R3D, VROWS, grid_from_dump
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import ctrl as ctrl_mod
from mitgcm_jax.pkgs import exf_full as exfb_mod
from mitgcm_jax.pkgs import seaice_init as si_mod
from mitgcm_jax.pkgs import seaice_model as sm_mod
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle

ORACLE = oracle.FULL
ICE_STAGE = "S00i_begin_ice_exf"
VERT = [n for n, _ in VROWS.values() if n != "dBdrRef"] + ["phiRef"]   # dBdrRef: not ported (grid/load.py)


@functools.lru_cache(maxsize=1)
def production():
    """model.setup(rundir) + init.state_from_pickup of the full oracle's run directory (the production path: grid from
    the files, ctrl adjustments, pickups), with CTRL_INIT's smoothed controls recorded (negative controls below).
    Cached for the pytest session (tests/test_step_full.py starts its production-path gate from it)."""
    rundir = oracle.run_dir(ORACLE)
    ds = oracle.dumpset(ORACLE)
    P, g, ex, kLowC = setup(rundir)
    calls = []
    orig = ctrl_mod.ctrl_init

    def recording(*a, **k):                  # keep CTRL_INIT's smoothed controls for the negative controls
        calls.append(orig(*a, **k))
        return calls[-1]

    ctrl_mod.ctrl_init = recording
    try:
        st, aux = init.state_from_pickup(P, g, ex, kLowC, rundir, return_aux=True)
    finally:
        ctrl_mod.ctrl_init = orig
    return dict(rundir=rundir, ds=ds, P=P, g=g, ex=ex, kLowC=kLowC, st=st, aux=aux, ctrl_in=calls[-1])


@pytest.fixture(scope="module")
def run():
    return production()


def _eq(a, ref, name):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(ref), err_msg=name)


def test_setup_full_tree(run):
    """Full tree: no flux-forced EXF (P.exf None) but the bulk-formula EXF (P.exfb) and the sea-ice parameters
    (P.seaice); the ocean parameters take the full-tree branches; the grid carries the fixed sea-ice fields and the
    zenith-angle factors (model.setup). A ModelParams mixing the trees is refused by forward_step."""
    P, g = run["P"], run["g"]
    assert P.exf is None and P.ctrl is not None and P.ctrl.tim2d == ()
    assert isinstance(P.exfb, exfb_mod.ExfFullParams) and isinstance(P.seaice, sm_mod.SeaiceParams)
    assert P.sf.useSEAICE and P.sf.zero_salt_plume_flux and not P.sf.temp_EvPrRn_set
    assert P.rs.mxl.calcMixLayerDepth
    from mitgcm_jax.core.forward_step import ZS_KEYS, forward_step
    assert set(si_mod.ICE_FIXED) <= set(g.f) and {"zs_" + k for k in ZS_KEYS} <= set(g.f)
    with pytest.raises(NotImplementedError):
        forward_step(P._replace(seaice=None), g, run["ex"], run["kLowC"], run["st"], None)


def test_tree_detection_and_override(run):
    """The tree is detected from the namelists (spflxfile in data.exf: flux-forced) or declared (Nikolay 2026-09-23:
    setup / state_from_pickup / run_jax --tree); a declaration contradicting the namelists is an error, raised before
    any file is read."""
    from mitgcm_jax.model import detect_tree, resolve_tree
    nml_full, nml_ff = RunNamelists(run["rundir"]), RunNamelists(oracle.run_dir(oracle.FORCED))
    assert detect_tree(nml_full) == "full" and detect_tree(nml_ff) == "ff"
    assert resolve_tree(nml_full, "full") == "full" and resolve_tree(nml_ff, "ff") == "ff"
    assert resolve_tree(nml_full) == "full" and resolve_tree(nml_ff, None) == "ff"
    for nml, bad in ((nml_full, "ff"), (nml_ff, "full")):
        with pytest.raises(ValueError, match="contradicts"):
            resolve_tree(nml, bad)
    with pytest.raises(ValueError, match="one of"):
        resolve_tree(nml_full, "flux-forced")
    with pytest.raises(ValueError, match="contradicts"):
        setup(run["rundir"], tree="ff")
    with pytest.raises(ValueError, match="contradicts"):
        init.state_from_pickup(run["P"], run["g"], run["ex"], run["kLowC"], run["rundir"], tree="ff")


def test_grid_and_ctrl_mixing_bitwise(run):
    """grid_from_files + ctrl-adjusted kapGM/kapRedi/diffKr == G00; sea-ice static fields == G01."""
    ds, g = run["ds"], run["g"]
    ref = grid_from_dump(ds, 1)
    for n in G2D + G3D + R3D + VERT:
        _eq(g.f[n], ref.f[n], n)
    for n in si_mod.ICE_GEOMETRY:
        _eq(g.f[n], oracle.field(ds, 1, "G01_seaice_geometry", n), n)
    assert np.abs(np.asarray(g.k1AtC)).max() > 0 and 0 < np.asarray(g.HEFFM).mean() < 1


def test_init_full_bitwise(run):
    ds, st, aux, P = run["ds"], run["st"], run["aux"], run["P"]
    assert st.it == 1
    assert aux["pickup"].mom_StartAB == 1 and aux["pickup"].missing == ()
    assert float(aux["cg2dNorm"]) == P.cg.cg2dNorm
    ref = state_from_dump(ds, 1)
    compare = sorted(ref.f)
    assert not set(compare) - set(st.f), sorted(set(compare) - set(st.f))
    for k in compare:
        _eq(st.f[k], ref.f[k], k)
    for k in si_mod.ICE_STATE:
        _eq(st.f[k], oracle.field(ds, 1, ICE_STAGE, k), k)
    # the full EXF_INIT_VARIA arrays: 'b' group in S00i_begin_ice_exf, 'x' group in S00_begin (M2.6b-2)
    for k in exfb_mod.EXF_ARRAYS:
        stage = ICE_STAGE if (1, ICE_STAGE, k) in ds.index else "S00_begin"
        _eq(st.f[k], oracle.field(ds, 1, stage, k), k)
    for k, v in sm_mod.dyn_carry_init(run["g"].layout).items():
        _eq(st.f[k], v, k)
    extra = set(st.f) - set(ref.f) - set(exfb_mod.EXF_ARRAYS) - set(sm_mod.SEAICE_CARRIED)
    assert not extra, sorted(extra)
    # not vacuous: ice present, the load non-zero, TICES categories copied, the production controls applied
    assert np.abs(np.asarray(st.f["sIceLoad"])).max() > 100.0 and np.asarray(st.f["AREA"]).max() > 0.5
    assert np.asarray(st.f["TICES"]).shape[1] == si_mod.NITD


def test_negative_controls_ocean(run, monkeypatch):
    """A pickup theta value * (1 + 1e-6) (re-run with the recorded ctrl inputs) changes theta and totPhiHyd."""
    ds = run["ds"]
    orig_read = init.read_pickup

    def planted(*a, **k):
        pk, info = orig_read(*a, **k)
        wet = np.argwhere(pk["theta"] != 0.0)
        t, kk, j, i = wet[len(wet) // 2]
        pk = dict(pk, theta=pk["theta"].copy())
        pk["theta"][t, kk, j, i] *= 1 + 1e-6
        return pk, info

    monkeypatch.setattr(init, "read_pickup", planted)
    monkeypatch.setattr(ctrl_mod, "ctrl_init", lambda *a, **k: run["ctrl_in"])
    st = init.state_from_pickup(run["P"], run["g"], run["ex"], run["kLowC"], run["rundir"])
    for name in ("theta", "totPhiHyd"):
        assert not np.array_equal(np.asarray(st.f[name]), oracle.field(ds, 1, "S00_begin", name)), name
    for name in si_mod.ICE_STATE:                     # the ice part does not see the ocean plant
        _eq(st.f[name], run["st"].f[name], name)


def _ice(run, cfg=None, pk=None, ex=None, fn=None):
    """SEAICE_INIT_VARIA (jitted) on the run's pickup_seaice; returns ((ice, sIceLoad), cfg, pickup interiors)."""
    icfg, ipk = init.seaice_inputs(run["rundir"], init.InitConfig.from_namelists(RunNamelists(run["rundir"])))
    fn = si_mod.seaice_init_varia if fn is None else fn
    ex = run["ex"] if ex is None else ex
    f = jax.jit(lambda c, p: fn(c, run["g"], ex, p))
    return f(icfg if cfg is None else cfg, ipk if pk is None else pk), icfg, ipk


def test_negative_controls_seaice(run, monkeypatch):
    """SEAICE_INIT_VARIA alone vs the dumps: passes as is; fails with one HEFF value * (1 + 1e-6), rhoSnow *
    (1 + 1e-6), without the doMapTice copy, or with the uIce/vIce exchange done without signs."""
    ds = run["ds"]
    ref = {k: oracle.field(ds, 1, ICE_STAGE, k) for k in si_mod.ICE_STATE}
    ref_load = oracle.field(ds, 1, "S00_begin", "sIceLoad")
    (ice, load), icfg, ipk = _ice(run)
    for k in si_mod.ICE_STATE:
        _eq(ice[k], ref[k], k)
    _eq(load, ref_load, "sIceLoad")
    # one HEFF value
    h = np.asarray(ipk["HEFF"]).copy()
    t, j, i = np.argwhere(h > 0.5)[0]
    h[t, j, i] *= 1 + 1e-6
    (ice, load), _, _ = _ice(run, pk=dict(ipk, HEFF=jnp.asarray(h)))
    assert not np.array_equal(np.asarray(ice["HEFF"]), ref["HEFF"])
    assert not np.array_equal(np.asarray(load), ref_load)
    # rhoSnow
    (ice, load), _, _ = _ice(run, cfg=dataclasses.replace(icfg, SEAICE_rhoSnow=icfg.SEAICE_rhoSnow * (1 + 1e-6)))
    assert not np.array_equal(np.asarray(load), ref_load)
    # no doMapTice: categories 2..nITD keep 273
    orig = si_mod.seaice_init_varia

    def no_map(cfg, g, ex, pk):
        L = g.layout
        J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
        out, sl = orig(cfg, g, ex, pk)
        T = out["TICES"].at[:, 1:, J, I].set(273.0)
        return dict(out, TICES=ex.scalar(T, "3D")), sl

    (ice, _), _, _ = _ice(run, fn=no_map)
    assert not np.array_equal(np.asarray(ice["TICES"]), ref["TICES"])

    # vector exchange without signs
    class NoSignEx:
        def __init__(self, ex):
            self.ex = ex

        def exch_uv_xy(self, u, v, withSigns):
            return self.ex.exch_uv_xy(u, v, not withSigns)

        def exch_xy(self, a):
            return self.ex.exch_xy(a)

        def scalar(self, a, kind):
            return self.ex.scalar(a, kind)

    (ice, _), _, _ = _ice(run, ex=NoSignEx(run["ex"]))
    assert not (np.array_equal(np.asarray(ice["UICE"]), ref["UICE"])
                and np.array_equal(np.asarray(ice["VICE"]), ref["VICE"]))


def test_seaice_pickup_checks(run, monkeypatch):
    """SEAICE_READ_PICKUP / SEAICE_CHECK_PICKUP: the V4r4 pickup has all six fields (siTICE a single level); a field
    list without siHEFF stops the run ('cannot restart without field', seaice_check_pickup.F:132-146) even with
    pickupStrictlyMatch = F; a missing unknown field stops it only with pickupStrictlyMatch (:191-200)."""
    rundir = run["rundir"]
    icfg = si_mod.SeaiceInitConfig.from_namelists(RunNamelists(rundir))
    pk = si_mod.read_seaice_pickup(rundir, icfg)
    assert set(pk) == {"TICES1", "AREA", "HEFF", "HSNOW", "UICE", "VICE"}
    orig = si_mod.read_mds

    def renamed(old, new):
        def f(prefix):
            arr, meta = orig(prefix)
            return arr, dict(meta, fldList=[new if x == old else x for x in meta["fldList"]])
        return f

    monkeypatch.setattr(si_mod, "read_mds", renamed("siHEFF", "siXXXX"))
    with pytest.raises(ValueError, match="SEAICE_CHECK_PICKUP"):
        si_mod.read_seaice_pickup(rundir, dataclasses.replace(icfg, pickupStrictlyMatch=False))
