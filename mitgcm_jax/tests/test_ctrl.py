"""Controls gate (plan Task 8b, tier1x): the production useCTRL=T initialisation (ff CTRL_MAP_INI_GENARR with the
pkg/smooth WC01 correlation operator, pkgs/ctrl.py + pkgs/smooth.py) against the oracle ref_ff_jaxdump_v4
(useCTRL=T, ctrlUseGen=T, geothermal on, xx_*.0000000129 controls, dumps at iterations 1, 2, 3).

- model.setup(rundir): kapGM, kapRedi, diffKr (INI_MIXING + xx_kapgm/xx_kapredi/xx_diffkr) == G00_geometry, bitwise
  with halos; the pkg/smooth operators == the smooth3Doperator001 / smooth2Doperator001 files the Fortran run wrote.
- init.state_from_pickup(rundir) == state_from_dump(ds, 1): every field, halos included, bitwise (etaN, theta, salt,
  uVel, vVel carry the controls; wVel, etaH, r* fields follow from them).
- one FORWARD_STEP from that state == S00_begin of iteration 2, bitwise; CTRL_MAP_FORCING (forward_step.F:524-530) is
  composed after LOAD_FIELDS_DRIVER here (see `_step_with_ctrl_map_forcing`) until forward_step.py calls it.
- CTRL_MAP_FORCING replay: S02_load_fields -> S03_ctrl_map_forcing, all fields bitwise, iterations 1-3 (the zero
  forcing controls leave values unchanged; the saltFlux halo exchange is the only difference, 1538 values).
Negative controls (each with its positive counterpart on the same code path): one xx_etan value perturbed by 1e-6
(relative) changes etaN; one pseudo-time step less in the WC01 smoother (149 instead of smooth3Dnbt/2 = 150) changes
theta; a non-zero forcing-control record is refused.
Measured 2026-09-23 (16 CPU cores): 0 differing values in every compared field; maximal adjustments vs the pickup
theta 8.68 K, salt 7.00, uVel 0.36, vVel 0.79 m/s, etaN 0.016 m, kapGM 7097, kapRedi 3652, diffKr 1.6e-4 m2/s.
Without CTRL_MAP_FORCING the step differs only in 1538 halo values of saltFlux and surfaceForcingS. Cost: setup
~160 s + state_from_pickup ~150 s (seven 3-D WC01 smoothings of 150 steps; >50 % exch2 gathers); file ~11-15 min.
"""

import dataclasses

import jax
import numpy as np
import pytest

from mitgcm_jax import init
from mitgcm_jax.core import forward_step as fs_mod
from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.io.mds import read_mds
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import ctrl
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.state import state_from_dump
from mitgcm_jax.tests import oracle

CTRL_ORACLE = "ref_ff_jaxdump_v4"


@pytest.fixture(scope="module")
def run():
    rundir = oracle.run_dir(CTRL_ORACLE)
    ds = oracle.dumpset(CTRL_ORACLE)
    P, g, ex, kLowC = setup(rundir)
    st = init.state_from_pickup(P, g, ex, kLowC, rundir)
    return rundir, ds, P, g, ex, kLowC, st


def _eq(a, ref, name):
    np.testing.assert_array_equal(np.asarray(a), ref, err_msg=name)


def test_mixing_fields_bitwise(run):
    rundir, ds, P, g, ex, kLowC, st = run
    for n in ctrl.MIXING_TARGETS:
        _eq(g.f[n], oracle.field(ds, 1, "G00_geometry", n), n)


def test_smooth_operators_match_fortran_files(run):
    """SMOOTH_INIT3D/2D operators after the real*4 write/read round trip == the files the Fortran run wrote."""
    rundir, ds, P, g, ex, kLowC, st = run
    nml = RunNamelists(rundir)
    ci = ctrl.ctrl_init(nml, g, ex, ("etaN", "theta"), init.ini_recip_hfac(g.h0FacC), smooth=False)
    L = g.layout
    inner = (Ellipsis, slice(L.OLy, L.OLy + L.sNy), slice(L.OLx, L.OLx + L.sNx))
    op3, _ = read_mds(rundir / "smooth3Doperator001")
    for r, n in enumerate(("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "Kuy", "Kvx", "kappaR")):
        _eq(np.asarray(ci.ops["3d"][1][n])[inner], np.moveaxis(compact_to_tiles(op3[r]), -3, 0), n)
    op2, _ = read_mds(rundir / "smooth2Doperator001")
    for r, n in enumerate(("Kux", "Kvy")):
        _eq(np.asarray(ci.ops["2d"][1][n])[inner], compact_to_tiles(op2[r]), "2d " + n)


def test_init_ctrl_bitwise(run):
    rundir, ds, P, g, ex, kLowC, st = run
    assert st.it == 1
    ref = state_from_dump(ds, 1)
    assert set(ref.f) <= set(st.f), sorted(set(ref.f) - set(st.f))
    for k in sorted(ref.f):
        _eq(st.f[k], ref.f[k], k)


def test_step1_from_ctrl_state_bitwise(run):
    rundir, ds, P, g, ex, kLowC, st = run
    nml = RunNamelists(rundir)
    assert P.ctrl is not None and P.ctrl == ctrl.CtrlConfig.from_namelists(nml)   # setup wires CTRL_MAP_FORCING
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    myTime, myIter = exf_mod.model_time(nml, 1)
    bufs, facs, _ = loader.load(myTime, myIter)
    step = jax.jit(lambda P, g, kLowC, st, exf_in: fs_mod.forward_step(P, g, ex, kLowC, st, exf_in))
    st1, a = step(P, g, kLowC, st, {"bufs": bufs, "facs": facs, "myTime": myTime})
    ref = state_from_dump(ds, 2)
    assert set(ref.f) <= set(st1.f), sorted(set(ref.f) - set(st1.f))
    for k in sorted(ref.f):
        _eq(st1.f[k], ref.f[k], k)


def test_ctrl_map_forcing_replay(run):
    """S02_load_fields -> CTRL_MAP_FORCING -> S03_ctrl_map_forcing (all dumped fields, halos included)."""
    rundir, ds, P, g, ex, kLowC, st = run
    cc = ctrl.CtrlConfig.from_namelists(RunNamelists(rundir))
    names = sorted({k[2] for k in ds.index if k[0] == 1 and k[1] == "S03_ctrl_map_forcing"})
    for it in (1, 2, 3):
        ff = {n: oracle.field(ds, it, "S02_load_fields", n) for n in names}
        out = ctrl.ctrl_map_forcing(cc, g, ex, {k: jax.numpy.asarray(v) for k, v in ff.items()})
        for n in names:
            _eq(out[n], oracle.field(ds, it, "S03_ctrl_map_forcing", n), f"it {it} {n}")
    # negative control: without CTRL_MAP_FORCING the saltFlux halos differ
    assert not np.array_equal(ff["saltFlux"], oracle.field(ds, 3, "S03_ctrl_map_forcing", "saltFlux"))


def _raw(run, field):
    """CtrlInit of one control without the ctrl_init smoothing (WC01 runs inside ctrl_map_ini_genarr) and the field
    as INITIALISE_VARIA holds it before CTRL_INIT_VARIABLES (READ_PICKUP + exchange, read_pickup.F:515, 531)."""
    rundir, ds, P, g, ex, kLowC, st = run
    nml = RunNamelists(rundir)
    ci = ctrl.ctrl_init(nml, g, ex, (field,), init.ini_recip_hfac(g.h0FacC), smooth=False)
    pk, _ = init.read_pickup(rundir, init.InitConfig.from_namelists(nml), g.layout)
    L = g.layout
    a = np.zeros(L.shape3d if field == "theta" else L.shape2d)
    a[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = pk[field]
    before = ex.scalar(a, "3D") if field == "theta" else ex.exch_xy(a)
    return ds, g, ex, ci, before


def test_negative_control_xx_value(run):
    """etaN (2-D control, cheap): the raw path reproduces S00_begin etaN; one xx_etan value x (1+1e-6) does not."""
    ds, g, ex, ci, before = _raw(run, "etaN")
    ref = oracle.field(ds, 1, "S00_begin", "etaN")
    f = ctrl.ctrl_map_ini_genarr_jit(ex)
    _eq(f(ci, g, {"etaN": before})["etaN"], ref, "etaN (raw path)")
    xx = np.array(ci.inputs["etaN"]["xx"])
    wet = np.argwhere((xx != 0.0) & (np.asarray(g.maskC)[:, 0] != 0.0))
    t, j, i = wet[len(wet) // 2]
    xx[t, j, i] *= 1 + 1e-6
    ci2 = dataclasses.replace(ci, inputs={"etaN": dict(ci.inputs["etaN"], xx=jax.numpy.asarray(xx))})
    assert not np.array_equal(np.asarray(f(ci2, g, {"etaN": before})["etaN"]), ref)


def test_negative_control_smoothing_steps(run):
    """theta: the raw path (SMOOTH_CORREL3D traced inside ctrl_map_ini_genarr) reproduces S00_begin theta; 149
    pseudo-time steps instead of smooth3Dnbt/2 = 150 (same operator) do not."""
    ds, g, ex, ci, before = _raw(run, "theta")
    ref = oracle.field(ds, 1, "S00_begin", "theta")
    f = ctrl.ctrl_map_ini_genarr_jit(ex)
    _eq(f(ci, g, {"theta": before})["theta"], ref, "theta (raw path)")
    op = ci.scfg.op3d(1)
    scfg = dataclasses.replace(ci.scfg, ops3d=(dataclasses.replace(op, nbt=op.nbt - 2),))   # nbt//2 = 149
    out = f(dataclasses.replace(ci, scfg=scfg), g, {"theta": before})
    assert not np.array_equal(np.asarray(out["theta"]), ref)


def test_nonzero_forcing_control_refused(tmp_path):
    """require_zero_forcing_controls reads every record: one non-zero value -> NotImplementedError; a bound that
    moves 0 -> NotImplementedError; zeros -> accepted. (A new optimcycle per case: new file names.)"""
    base = (" &ctrl_nml\n doMainUnpack=.FALSE.,\n /\n &ctrl_nml_genarr\n"
            " xx_gentim2d_weight(1) = 'w.data',\n xx_gentim2d_file(1)='xx_qnet',\n{extra} /\n")
    (tmp_path / "data.pkg").write_text(" &PACKAGES\n useCTRL=.TRUE.,\n /\n")

    def case(cycle, a, extra=""):
        (tmp_path / "data.optim").write_text(f" &OPTIM\n optimcycle={cycle},\n &\n")
        (tmp_path / "data.ctrl").write_text(base.format(extra=extra))
        a.tofile(tmp_path / f"xx_qnet.{cycle:010d}.data")
        ctrl.require_zero_forcing_controls(RunNamelists(tmp_path))

    a = np.zeros(3 * 90 * 1170, dtype=">f4")
    case(7, a)
    a[2 * 90 * 1170 + 5] = 1e-3
    with pytest.raises(NotImplementedError, match="non-zero"):
        case(8, a)
    with pytest.raises(NotImplementedError, match="bounds"):
        case(9, np.zeros_like(a), " xx_gentim2d_bounds(1:5,1)=1.,2.,3.,4.,0.,\n")
