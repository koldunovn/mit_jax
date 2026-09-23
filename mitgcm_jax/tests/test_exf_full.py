"""Full-V4r4 EXF (plan M2.1 + M2.2): EXF_GETFORCING with the atmospheric-state forcing, EXF_RADIATION +
EXF_ZENITHANGLE, EXF_WIND, EXF_BULKFORMULAE (Large & Yeager 2004, niter_bulk = 2), EXF_GETSURFACEFLUXES and
EXF_MAPFIELDS, gated against the full oracle (oracle.FULL = full_jaxdump_v5, iterations 1, 2, 3; iteration 1 loads
from `first`, iteration 3 swaps to a new 6-hourly record).

Elementary functions. The device kernels take a `Libm` bundle. `GLIBC` (below, test-only) evaluates exp, log, atan,
sin, cos, acos with glibc through a host callback, i.e. the functions the gfortran binary calls. Measured on this CPU:
XLA's log/atan/sin/cos equal glibc's bit for bit; its exp differs in the last bit for ~14 % of arguments, arccos for
~7 %. The production default `exf_full.DEVICE_LIBM` = jnp functions + `exf_full.exp_glibc` (glibc 2.28's exp, FMA
variant, transcribed from libm's machine code with exact emulated FMAs): bit-identical to glibc's exp on 1e6 random
arguments of its main range (test_exp_glibc).

Gates (all points, halos included, every iteration; replay = each stage fed the dumped inputs of the previous one):
  - records: ExfFullRecordLoader buffers == the X01 'e' group (<f>0, <f>1 of the 10 fields) bitwise; record numbers,
    weights and first/changed for the 6-hourly fields and the monthly runoff (cal_GetMonthsRec) at the oracle steps.
  - GLIBC: X01 (EXF_GETFFIELDS, A-grid stress rotation), X02 (zenith angle), X03 (radiation), X04 (wind), X05a/X05b
    (bulk-formula locals before/after the stability iterations, interior), X05 (bulk), X06 (hflux/sflux + A-grid
    exchange), X07, X08 (EXF_MAPFIELDS: fu, fv, Qnet, Qsw, EmPmR, saltFlux, pLoad): max abs difference 0 at every
    stage and iteration, replayed and chained (S00 state -> X08, jitted), and S02 == X08.
  - DEVICE_LIBM (production), chained S00 -> X08, jitted and eager: bitwise at every stage and iteration except
    zen_fsol_daily (XLA's arccos; a diagnostic nothing reads): max relative error 2.1e-16.
  - JNP_LIBM (XLA's exp) for reference: hl 8.0e-16, evap 8.1e-16, hs 2.8e-16, hflux 3.0e-16, sflux 9.7e-17,
    Qnet 3.0e-16, EmPmR 1.0e-16 (max|diff| / max|ref| over the 3 iterations). Gate: 2e-15.
  - Record/weight logic by hand (datetime): 6-hourly records over month/leap-day/year boundaries (as the M1 test),
    monthly runoff records over two years (mid-month switches, leap February, year ends), the zenith-angle date
    scalars (TYEAR, TDAY, table row) over two years.
Negative controls (each makes its comparison fail): weight x (1+1e-6); stress rotation dropped; monthly records of
the wrong month; zenith table row + 1; cvapor_exp x (1+1e-6); one stability iteration instead of two; fu from the
A-grid stress without the average to u points; jnp's exp instead of glibc's (the gate resolves 1 ulp); the hand
monthly-record computation shifted by one day.
Gradient: d(sum w.(hs, hl, evap))/d(atemp, aqh, wspeed) of EXF_BULKFORMULAE (DEVICE_LIBM) is finite everywhere and
matches central differences at wet points away from the clip/switch thresholds; d(sum w.(Qnet, EmPmR, fu, fv))/d(next
atemp, wspeed, ustress records) through the whole EXF_GETFORCING likewise (h-sweep over 3 decades; worst rel. diff
printed).
"""

import calendar as pycal
import ctypes
import ctypes.util
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.io.dump import DumpSet, read_file
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.pkgs import exf_full as X
from mitgcm_jax.pkgs.exf_fluxforced import _replace, cal_getdate, exf_filter_rl, exf_getffieldrec, read_rec_2d
from mitgcm_jax.tests import oracle

ORACLE = oracle.FULL
ITERS = (1, 2, 3)
B = ["uwind", "vwind", "wspeed", "wStress", "cw", "sw", "sh", "atemp", "aqh", "hs", "hl", "lwflux", "evap", "precip",
     "snowprecip", "swdown", "lwdown", "zen_albedo", "zen_fsol_diurnal", "zen_fsol_daily", "runoff"]   # 'b' group
XG = ["ustress", "vstress", "hflux", "swflux", "sflux", "saltflx", "apressure"]                     # 'x' group
FF = ["fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad"]                                     # set by MAPFIELDS
# (stage key of exf_getforcing's return_stages, dump stage, fields dumped there)
STAGES = (("X01", "X01_exf_getffields", B + XG), ("X03", "X03_exf_radiation", B + XG), ("X04", "X04_exf_wind", B),
          ("X05", "X05_exf_bulkformulae", B), ("X06", "X06_exf_hflux_sflux", B + XG),
          ("X07", "X07_exf_getsurfacefluxes", XG))
# fields that differ with XLA's exp (JNP_LIBM: exp in the bulk formulae) or arccos (both: zen_fsol_daily), and the gate
JNP_ULP_FIELDS = {"hs", "hl", "evap", "hflux", "sflux", "Qnet", "EmPmR", "zen_fsol_daily"}
DEVICE_ULP_FIELDS = {"zen_fsol_daily"}
JNP_TOL = 2e-15


# ------------------------------------------------------------------------------------------------ glibc via callback
def _glibc(name):
    lib = ctypes.CDLL(ctypes.util.find_library("m"))
    f = getattr(lib, name)
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    vf = np.frompyfunc(f, 1, 1)

    def host(x):
        x = np.asarray(x)
        return np.asarray(vf(x), dtype=np.float64).reshape(x.shape)

    return lambda x: jax.pure_callback(host, jax.ShapeDtypeStruct(x.shape, x.dtype), x)


GLIBC = X.Libm(exp=_glibc("exp"), log=_glibc("log"), atan=_glibc("atan"), sin=_glibc("sin"), cos=_glibc("cos"),
               acos=_glibc("acos"))


# ---------------------------------------------------------------------------------------------------------- fixtures
def _dumpset_parallel(directory):
    """DumpSet of `directory` with the record headers read in parallel threads (serial indexing of the full oracle
    takes ~110 s on cold Lustre, this ~5 s); same index as DumpSet(directory)."""
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


@pytest.fixture(scope="module")
def ds():
    return _dumpset_parallel(oracle.run_dir(ORACLE) / "jaxdump")


@pytest.fixture(scope="module")
def nml():
    return RunNamelists(oracle.run_dir(ORACLE))


@pytest.fixture(scope="module")
def g(ds):
    return grid_from_dump(ds, 1)


@pytest.fixture(scope="module")
def ex():
    return default_exchanger()


@pytest.fixture(scope="module")
def p(nml):
    return X.ExfFullParams.from_namelists(nml)


@pytest.fixture(scope="module")
def zs(p, g):
    return X.zenith_static(p, g)


@pytest.fixture(scope="module")
def steps(nml, p, g):
    """ExfFullRecordLoader driven through the oracle's steps in order: {myIter: dict(myTime, bufs, facs, recs, zt)}
    with device copies (jb, jf, jzt) for the kernels."""
    loader = X.ExfFullRecordLoader(p, g, oracle.run_dir(ORACLE))
    out = {}
    for iloop in range(1, len(ITERS) + 1):
        myTime, myIter = X.model_time(nml, iloop)
        bufs, facs, recs = loader.load(myTime, myIter)
        zt = X.zenith_time(p, myTime, myIter)
        out[myIter] = dict(myTime=myTime, bufs=bufs, facs=facs, recs=recs, zt=zt,
                           jb={n: (jnp.asarray(a), jnp.asarray(b)) for n, (a, b) in bufs.items()},
                           jf={n: jnp.asarray(f) for n, f in facs.items()},
                           jzt={k: jnp.asarray(v) for k, v in zt.items()})
    out["loaded"] = list(loader.loaded)
    return out


def F(ds, it, stage, name):
    return oracle.field(ds, it, stage, name)


def state(ds, it, bstage, xstage):
    s = {n: jnp.asarray(F(ds, it, bstage, n)) for n in B}
    s.update({n: jnp.asarray(F(ds, it, xstage, n)) for n in XG})
    return s


def start_state(ds, it):
    """EXF arrays, FFIELDS and theta at the start of step `it` (before LOAD_FIELDS_DRIVER)."""
    exf = state(ds, it, "S00i_begin_ice_exf", "S00_begin")
    ff = {n: jnp.asarray(F(ds, it, "S00_begin", n)) for n in FF}
    return exf, ff, jnp.asarray(F(ds, it, "S00_begin", "theta"))


def err(got, ref):
    """(max abs difference, max abs difference / max |ref|)."""
    got, ref = np.asarray(got), np.asarray(ref)
    d = float(np.max(np.abs(got - ref)))
    s = float(np.max(np.abs(ref)))
    return d, (d / s if s > 0 else d)


def diffs(got, ds, it, stage, names):
    return {n: err(got[n], F(ds, it, stage, n)) for n in names}


def nonzero(errs):
    return {n: e for n, e in errs.items() if e[0] != 0.0}


def run_chain(p, g, ex, exf, ff, s, theta, zs, libm, jit=True, **kw):
    fn = lambda p, g, exf, ff, jb, jf, theta, t, zt, zs: X.exf_getforcing(  # noqa: E731
        p, g, ex, exf, ff, jb, jf, theta, t, zt, zs, libm=libm, return_stages=True)
    return (jax.jit(fn) if jit else fn)(p, g, exf, ff, s["jb"], s["jf"], theta, s["myTime"], s["jzt"], zs)


# --------------------------------------------------------------------------------------------- setup and records
def test_params(p):
    """Fields, masks (useSEAICE: stress, wspeed, swdown, apressure unmasked, exf_init_fixed.F:85-114), record start
    03:00 (10800 s), 6-hourly periods, monthly runoff; derived switches."""
    got = [(f.name, f.mask, f.period, f.startTime) for f in p.fields]
    want = [(n, " " if n in ("ustress", "vstress", "wspeed", "swdown", "apressure") else "c",
             -12.0 if n == "runoff" else 21600.0, 0.0 if n == "runoff" else 10800.0) for n in X.FIELDS]
    assert got == want
    assert p.field_map["atemp"].const == 273.15 and p.exf_offset_atemp == 273.15
    assert (p.inscal_ustress, p.inscal_vstress, p.inscal_swdown, p.inscal_lwdown) == (-1.0, -1.0, -1.0, -1.0)
    assert p.solve4Stress and p.useExfZenAlbedo and p.useExfZenIncoming and p.select_ZenAlbedo == 1
    assert (p.ocean_emissivity, p.exf_albedo, p.hq, p.zwln) == (0.97, 0.1, 2.0, 0.0)
    assert p.cal.modelStart == 3600.0 and p.cal.modelStep == 3600.0


def test_oracle_step_records(steps, ds):
    """it=1 (01-01 13:00): first, 6-hourly records 2/3 with fac 1-14400/21600, runoff Dec/Jan (12, 1); it=2: same
    records; it=3 (15:00): 6-hourly records 3/4 (changed, fac 1). Loaded buffers == the X01 'e' group bitwise, halos
    included (halos keep fldConst)."""
    want6 = {1: (2, 3, 1.0 - 14400.0 / 21600.0, True, False), 2: (2, 3, 1.0 - 18000.0 / 21600.0, False, False),
             3: (3, 4, 1.0, False, True)}
    for it in ITERS:
        s = steps[it]
        for name, r in s["recs"].items():
            if name == "runoff":
                mid_jan, mid_dec = 31 * 86400 / 2, -31 * 86400 / 2        # 1992-01-16 12:00, 1991-12-16 12:00
                t = (it - 1) * 3600.0 + 13 * 3600.0
                assert (r.count0, r.count1, r.first, r.changed) == (12, 1, it == 1, False)
                assert r.fac == (mid_jan - t) / (mid_jan - mid_dec), (it, r.fac)
            else:
                c0, c1, fac, first, changed = want6[it]
                assert (r.count0, r.count1, r.year0, r.year1, r.first, r.changed) == (c0, c1, 1992, 1992, first,
                                                                                      changed), (it, name)
                assert r.fac == fac
        for name in X.FIELDS:
            for k in (0, 1):
                np.testing.assert_array_equal(s["bufs"][name][k], F(ds, it, "X01_exf_getffields", f"{name}{k}"),
                                              err_msg=f"it={it} {name}{k}")
    recs = sorted({(it, name, rec) for it, name, rec, _ in steps["loaded"]})
    assert recs == sorted([(1, n, r) for n in X.FIELDS for r in ((12, 1) if n == "runoff" else (2, 3))]
                          + [(3, n, 4) for n in X.FIELDS if n != "runoff"])


def test_init_state(p, g, ds):
    """At the model start (it=1, before EXF_GETFORCING) the dumped EXF arrays equal exf_init_varia (fldConst,
    zeros) and the dumped record buffers equal the loader's initial buffers (fldConst), every point."""
    exf = X.exf_init_varia(p, g.layout)
    for n in B:
        np.testing.assert_array_equal(np.asarray(exf[n]), F(ds, 1, "S00i_begin_ice_exf", n), err_msg=n)
    for n in XG:
        np.testing.assert_array_equal(np.asarray(exf[n]), F(ds, 1, "S00_begin", n), err_msg=n)
    for f in p.fields:
        for k in (0, 1):
            np.testing.assert_array_equal(np.full(g.layout.shape2d, f.const),
                                          F(ds, 1, "S00i_begin_ice_exf", f"{f.name}{k}"), err_msg=f.name)


def _mid(y, m):
    return dt.datetime(y, m, 1) + dt.timedelta(seconds=pycal.monthrange(y, m)[1] * 86400 // 2)


def _hand_month(date, prev):
    """cal_GetMonthsRec by hand: records bracket the month midpoints; changed when the record pair differs from the
    previous step's (only possible at/after the midpoint, cal_getmonthsrec.F:133-200)."""
    y, m = date.year, date.month
    py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
    ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
    cur = _mid(y, m)
    if date < cur:
        c0, c1 = pm, m
        fac = (cur - date).total_seconds() / (cur - _mid(py, pm)).total_seconds()
        prevcount = c0
    else:
        c0, c1 = m, nm
        fac = (_mid(ny, nm) - date).total_seconds() / (_mid(ny, nm) - cur).total_seconds()
        prevcount = pm if (prev.month == m and prev < cur) or prev.month != m else m
    return c0, c1, fac, prevcount != c0


def test_monthly_records_by_hand(p):
    """cal_GetMonthsRec (runoff) over two years of 7-h steps, plus hourly steps around every 1992 month midpoint:
    record pair, weight (bitwise) and `changed` equal the hand computation; negative control: a one-day shift
    breaks it."""
    cal = p.cal
    t0 = dt.datetime(1992, 1, 1, 13)
    dates = [t0 + dt.timedelta(hours=7 * k) for k in range(2 * 366 * 24 // 7)]
    dates += [_mid(1992, m) + dt.timedelta(hours=h) for m in range(1, 13) for h in range(-3, 4)]
    for date in dates:
        myTime = cal.modelStart + (date - t0).total_seconds()
        r = X.cal_getmonthsrec(cal, myTime, 10)
        want = _hand_month(date, date - dt.timedelta(hours=1))
        assert (r.count0, r.count1, r.fac, r.changed) == want, (date, r, want)
        d = cal_getdate(cal, 10, myTime)
        assert (d[0], d[1]) == (int(date.strftime("%Y%m%d")), int(date.strftime("%H%M%S")))
    bad = 0
    for date in dates[:400]:
        myTime = cal.modelStart + (date - t0).total_seconds()
        r = X.cal_getmonthsrec(cal, myTime, 10)
        d1 = date + dt.timedelta(days=1)
        bad += (r.count0, r.count1, r.fac, r.changed) != _hand_month(d1, d1 - dt.timedelta(hours=1))
    assert bad > 0


def test_six_hourly_records_by_hand(p):
    """EXF_GetFFieldRec for every 6-hourly field of the full set over the 1992 leap day and the 1992/93 year end."""
    cal = p.cal
    t0 = dt.datetime(1992, 1, 1, 13)
    for base in (dt.datetime(1992, 2, 28, 12), dt.datetime(1992, 12, 31, 12)):
        for h in range(37):
            date = base + dt.timedelta(hours=h)
            myTime = cal.modelStart + (date - t0).total_seconds()
            year0 = date.year
            secs = (date - dt.datetime(year0, 1, 1)).total_seconds()
            if secs < 10800.0:
                year0 -= 1
                secs += (366.0 if year0 % 4 == 0 else 365.0) * 86400.0
            siy = (366.0 if year0 % 4 == 0 else 365.0) * 86400.0
            tot = secs - 10800.0
            c0 = int((tot + 0.5) / 21600.0) + 1
            y1, c1 = (year0 + 1, 1) if 10800.0 + c0 * 21600.0 >= siy else (year0, c0 + 1)
            fs = tot % 21600.0
            fac = 1.0 - fs / 21600.0 if y1 == year0 else 1.0 - fs / (siy - (c0 - 1) * 21600.0)
            for f in p.fields:
                if f.monthly:
                    continue
                r = exf_getffieldrec(cal, f.startTime, f.period, True, myTime, 10)
                assert (r.count0, r.count1, r.year0, r.year1, r.fac, r.changed) == (c0, c1, year0, y1, fac,
                                                                                    fs - 3600.0 < 0.0), (date, f.name)


def test_zenith_time_by_hand(p):
    """TYEAR, TDAY and the table row of EXF_ZENITHANGLE (useCAL) over two years of 5-h steps equal datetime's."""
    cal = p.cal
    t0 = dt.datetime(1992, 1, 1, 13)
    for k in range(2 * 366 * 24 // 5):
        date = t0 + dt.timedelta(hours=5 * k)
        zt = X.zenith_time(p, cal.modelStart + (date - t0).total_seconds(), 10)
        siy = (366.0 if pycal.isleap(date.year) else 365.0) * 86400.0
        tyear = (date - dt.datetime(date.year, 1, 1)).total_seconds() / siy
        tday = (date - dt.datetime(date.year, date.month, date.day)).total_seconds() / 86400.0
        assert (zt["TYEAR"], zt["TDAY"]) == (tyear, tday), date
        assert zt["iTyear1"] == int(1 + 365.0 * tyear) - 1 and zt["iTyear2"] == zt["iTyear1"] + 1


# --------------------------------------------------------------------------------------------------- replay gates
@pytest.mark.parametrize("it", ITERS)
def test_stage_replay_glibc(p, g, ex, ds, steps, zs, it):
    """Every stage replayed from the dumped inputs of the previous stage, with glibc's elementary functions:
    X01..X08 (all points) and the bulk-formula locals X05a/X05b (interior) have max abs difference 0."""
    s = steps[it]
    exf0, ff0, theta = start_state(ds, it)
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    out = {}
    out["X01"] = nonzero(diffs(jax.jit(lambda p, g, e, b, f, t: X.exf_getffields(p, g, e, b, f, t))(
        p, g, exf0, s["jb"], s["jf"], s["myTime"]), ds, it, "X01_exf_getffields", B + XG))
    s01 = state(ds, it, "X01_exf_getffields", "X01_exf_getffields")
    z = X.exf_zenithangle(p, g, s01, s["jzt"], zs, libm=GLIBC)
    # X02 sits inside EXF_RADIATION after lwflux is set (exf_radiation.F:67-118 before :142): lwflux there is X03's
    out["X02"] = nonzero(diffs(z, ds, it, "X02_exf_zenithangle", [n for n in B if n != "lwflux"]))
    np.testing.assert_array_equal(F(ds, it, "X02_exf_zenithangle", "lwflux"), F(ds, it, "X03_exf_radiation", "lwflux"))
    r = X.exf_radiation(p, g, s01, theta, s["jzt"], zs, libm=GLIBC)
    out["X03"] = nonzero(diffs(r, ds, it, "X03_exf_radiation", B + XG))
    w = jax.jit(X.exf_wind)(p, g, state(ds, it, "X03_exf_radiation", "X03_exf_radiation"))
    out["X04"] = nonzero(diffs(w, ds, it, "X04_exf_wind", B))
    bk, la, lb = X.exf_bulkformulae(p, g, state(ds, it, "X04_exf_wind", "X03_exf_radiation"), theta, libm=GLIBC,
                                    return_locals=True)
    out["X05"] = nonzero(diffs(bk, ds, it, "X05_exf_bulkformulae", B))
    for stage, loc in (("X05a_bulk_init", la), ("X05b_bulk_iter", lb)):
        for n in ("tstar", "qstar", "ustar", "rdn") + (("delq", "deltap") if stage.startswith("X05a") else
                                                       ("tau", "rd")):
            e = err(loc[n], F(ds, it, stage, n)[:, J, I])
            if e[0] != 0.0:
                out.setdefault(stage, {})[n] = e
    s05 = state(ds, it, "X05_exf_bulkformulae", "X03_exf_radiation")
    hs = X.exf_hflux_sflux(p, g, s05)
    hs["ustress"], hs["vstress"] = ex.exch_uv_agrid(hs["ustress"], hs["vstress"], True)
    out["X06"] = nonzero(diffs(hs, ds, it, "X06_exf_hflux_sflux", B + XG))
    out["X07"] = nonzero(diffs(X.exf_getsurfacefluxes(p, state(ds, it, "X06_exf_hflux_sflux", "X06_exf_hflux_sflux")),
                               ds, it, "X07_exf_getsurfacefluxes", XG))
    s07 = state(ds, it, "X06_exf_hflux_sflux", "X07_exf_getsurfacefluxes")
    s07["hflux"] = s07["hflux"] + s07["swflux"]                           # exf_getforcing.F:280-288
    e8, f8 = jax.jit(lambda p, g, e, f, t: X.exf_mapfields(p, g, ex, e, f, t))(p, g, s07, ff0, s["myTime"])
    out["X08"] = nonzero({**diffs(e8, ds, it, "X08_exf_mapfields", XG), **diffs(f8, ds, it, "X08_exf_mapfields", FF)})
    assert not any(out.values()), out


@pytest.mark.parametrize("it", ITERS)
def test_chain_glibc(p, g, ex, ds, steps, zs, it):
    """EXF_GETFORCING chained from the start-of-step state (S00i/S00), jitted, glibc functions: bitwise at X01, X03,
    X04, X05, X06, X07, X08; S02 (after LOAD_FIELDS_DRIVER) holds the same x/f fields as X08, and the f fields
    EXF does not set (surfaceForcing*, phi0surf, sIceLoad) are those of S00."""
    exf0, ff0, theta = start_state(ds, it)
    exf, ff, st = run_chain(p, g, ex, exf0, ff0, steps[it], theta, zs, GLIBC)
    out = {k: nonzero(diffs(st[k], ds, it, stage, names)) for k, stage, names in STAGES}
    out["X08"] = nonzero({**diffs(exf, ds, it, "X08_exf_mapfields", XG), **diffs(ff, ds, it, "X08_exf_mapfields", FF)})
    assert not any(out.values()), out
    for n in XG + FF:
        np.testing.assert_array_equal(F(ds, it, "S02_load_fields", n), F(ds, it, "X08_exf_mapfields", n), err_msg=n)
    for n in ("surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "phi0surf", "sIceLoad"):
        np.testing.assert_array_equal(F(ds, it, "X08_exf_mapfields", n), F(ds, it, "S00_begin", n), err_msg=n)


@pytest.mark.parametrize("libm_name", ["DEVICE_LIBM", "JNP_LIBM"])
def test_chain_production_libm(p, g, ex, ds, steps, zs, libm_name):
    """The same chain with the device libm bundles, jitted (and eager at it=1). DEVICE_LIBM (production): bitwise
    except zen_fsol_daily (arccos). JNP_LIBM (XLA's exp): the exp paths differ below JNP_TOL relative (maxima in the
    module docstring); every other field bitwise."""
    libm, ulp_fields = getattr(X, libm_name), (DEVICE_ULP_FIELDS if libm_name == "DEVICE_LIBM" else JNP_ULP_FIELDS)
    worst = {}
    for it in ITERS:
        exf0, ff0, theta = start_state(ds, it)
        for jit in (True, False) if it == 1 else (True,):
            exf, ff, st = run_chain(p, g, ex, exf0, ff0, steps[it], theta, zs, libm, jit=jit)
            errs = {}
            for k, stage, names in STAGES:
                errs.update(diffs(st[k], ds, it, stage, names))
            errs.update(diffs(exf, ds, it, "X08_exf_mapfields", XG))
            errs.update(diffs(ff, ds, it, "X08_exf_mapfields", FF))
            for n, (d, rel) in errs.items():
                if n not in ulp_fields:
                    assert d == 0.0, (libm_name, it, jit, n, d)
                worst[n] = max(worst.get(n, 0.0), rel)
    print(libm_name, "worst relative errors:", {n: f"{v:.2e}" for n, v in worst.items() if v > 0})
    assert max(worst.values()) <= JNP_TOL, worst
    assert worst["zen_fsol_daily"] > 0.0                  # arccos differs (else DEVICE_ULP_FIELDS is stale)
    if libm_name == "JNP_LIBM":
        assert worst["hl"] > 0.0                          # XLA's exp differs (else JNP_ULP_FIELDS is stale)


def test_exp_glibc():
    """exf_full.exp_glibc: its tables are glibc's (libm-2.28.so .rodata); it equals glibc's exp bit for bit on 3e5
    random arguments of its range (jnp.exp: ~14 % differ, the negative control), falls back to jnp.exp outside, and
    its derivative is exp."""
    blob = Path("/lib64/libm.so.6").resolve().read_bytes()
    coar, fine = X.exp_tables()
    np.testing.assert_array_equal(coar, np.frombuffer(blob[0xfa4a0:0xfa4a0 + 712 * 8], "<f8"))
    np.testing.assert_array_equal(fine, np.frombuffer(blob[0xf84a0:0xf84a0 + 1024 * 8], "<f8"))
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.uniform(-26.0, -14.0, 100000), rng.uniform(-708.0, -1.04, 100000),
                        rng.uniform(1.04, 708.0, 100000)])
    lib = ctypes.CDLL(ctypes.util.find_library("m"))
    lib.exp.argtypes, lib.exp.restype = [ctypes.c_double], ctypes.c_double
    ref = np.asarray(np.frompyfunc(lib.exp, 1, 1)(x), np.float64)
    got = np.asarray(jax.jit(X.exp_glibc)(jnp.asarray(x)))
    assert int(np.sum(got != ref)) == 0
    assert np.mean(np.asarray(jax.jit(jnp.exp)(jnp.asarray(x))) != ref) > 0.05
    small = jnp.asarray(rng.uniform(-1.0, 1.0, 1000))
    np.testing.assert_array_equal(np.asarray(jax.jit(X.exp_glibc)(small)), np.asarray(jax.jit(jnp.exp)(small)))
    dx = jax.jit(jax.vmap(jax.grad(X.exp_glibc)))(jnp.asarray(x[:1000]))
    np.testing.assert_array_equal(np.asarray(dx), got[:1000])


# ----------------------------------------------------------------------------------------------- negative controls
def test_negative_controls(p, g, ex, ds, steps, zs):
    """Planted errors make the bitwise comparisons fail."""
    it = 3
    s = steps[it]
    exf0, ff0, theta = start_state(ds, it)
    L = g.layout
    getff = jax.jit(lambda p, g, e, b, f, t: X.exf_getffields(p, g, e, b, f, t))
    # weight of atemp x (1 + 1e-6)
    jf = dict(s["jf"])
    jf["atemp"] = jf["atemp"] * (1.0 + 1e-6)
    x1 = getff(p, g, exf0, s["jb"], jf, s["myTime"])
    assert err(x1["atemp"], F(ds, it, "X01_exf_getffields", "atemp"))[0] > 0.0
    # stress rotation dropped (angleCosC = 1, angleSinC = 0)
    g_norot = g.replace(angleCosC=jnp.ones_like(g.angleCosC), angleSinC=jnp.zeros_like(g.angleSinC))
    x1 = getff(p, g_norot, exf0, s["jb"], s["jf"], s["myTime"])
    assert err(x1["ustress"], F(ds, it, "X01_exf_getffields", "ustress"))[1] > 1e-3
    # runoff records of the wrong month (Jan/Feb instead of Dec/Jan)
    f = p.field_map["runoff"]
    feb = exf_filter_rl(jnp.asarray(read_rec_2d(oracle.run_dir(ORACLE) / f.file, p.exf_iprec, 2, L)), f.mask, g)
    jb = dict(s["jb"])
    jb["runoff"] = (jb["runoff"][1], jb["runoff"][1].at[:, L.js(1, L.sNy), L.is_(1, L.sNx)].set(
        feb[:, L.js(1, L.sNy), L.is_(1, L.sNx)]))
    x1 = getff(p, g, exf0, jb, s["jf"], s["myTime"])
    assert err(x1["runoff"], F(ds, it, "X01_exf_getffields", "runoff"))[1] > 1e-3
    # zenith table row + 1
    s01 = state(ds, it, "X01_exf_getffields", "X01_exf_getffields")
    zt = dict(s["jzt"], iTyear1=s["jzt"]["iTyear1"] + 1, iTyear2=s["jzt"]["iTyear2"] + 1)
    z = X.exf_zenithangle(p, g, s01, zt, zs, libm=GLIBC)
    assert err(z["zen_albedo"], F(ds, it, "X02_exf_zenithangle", "zen_albedo"))[0] > 0.0
    # bulk formulae: cvapor_exp x (1+1e-6); one stability iteration; jnp's exp instead of glibc's
    s04 = state(ds, it, "X04_exf_wind", "X03_exf_radiation")
    bad = X.exf_bulkformulae(_replace(p, cvapor_exp=p.cvapor_exp * (1.0 + 1e-6)), g, s04, theta, libm=GLIBC)
    assert err(bad["hl"], F(ds, it, "X05_exf_bulkformulae", "hl"))[1] > 1e-9
    niter = X.NITER_BULK
    try:
        X.NITER_BULK = 1
        bad = X.exf_bulkformulae(p, g, s04, theta, libm=GLIBC)
    finally:
        X.NITER_BULK = niter
    assert err(bad["hs"], F(ds, it, "X05_exf_bulkformulae", "hs"))[1] > 1e-6
    bad = X.exf_bulkformulae(p, g, s04, theta, libm=GLIBC._replace(exp=jnp.exp))
    assert err(bad["hl"], F(ds, it, "X05_exf_bulkformulae", "hl"))[0] > 0.0
    # fu without the average of the A-grid stress to u points
    s07 = state(ds, it, "X06_exf_hflux_sflux", "X07_exf_getsurfacefluxes")
    s07["hflux"] = s07["hflux"] + s07["swflux"]
    e8, f8 = X.exf_mapfields(p, g, ex, s07, ff0, s["myTime"])
    fu_bad = ex.exch_uv_xy(p.outscal_ustress * e8["ustress"] * g.maskW[:, 0], f8["fv"], True)[0]
    assert err(f8["fu"], F(ds, it, "X08_exf_mapfields", "fu"))[0] == 0.0
    assert err(fu_bad, F(ds, it, "X08_exf_mapfields", "fu"))[1] > 1e-3


# ---------------------------------------------------------------------------------------------------------- gradient
def _fd_check(fun, x0, names, pts_of, w_keys, rng, hs_rel, label):
    """Central differences of the weighted outputs of fun(x) (pointwise differences first) vs jax.grad."""
    wts = None

    def J(x):
        o = fun(x)
        return sum(jnp.sum(wts[k] * o[k]) for k in w_keys)
    out0 = fun(x0)
    wts = {k: jnp.asarray(rng.normal(size=out0[k].shape)) for k in w_keys}
    grad = jax.jit(jax.grad(J))(x0)
    fj = jax.jit(fun)
    worst = 0.0
    for n in names:
        gn = np.asarray(grad[n])
        assert np.all(np.isfinite(gn)), (label, n)
        assert np.any(gn != 0.0), (label, n)
        scale = float(np.max(np.abs(np.asarray(x0[n])))) or 1.0
        for t, j, i in pts_of(n):
            gad = float(gn[t, j, i])
            rels = []
            for hr in hs_rel:
                h = hr * scale
                xp, xm = dict(x0), dict(x0)
                xp[n] = x0[n].at[t, j, i].add(h)
                xm[n] = x0[n].at[t, j, i].add(-h)
                op, om = fj(xp), fj(xm)
                fd = sum(float(np.sum(np.asarray(wts[k]) * (np.asarray(op[k]) - np.asarray(om[k])))) for k in w_keys)
                fd /= 2 * h
                rels.append(abs(fd - gad) / max(abs(gad), 1e-300))
            best = min(rels)
            worst = max(worst, best)
            assert best < 1e-6, (label, n, (t, j, i), gad, rels)
    print(f"{label}: worst FD rel. difference over the plateau {worst:.1e}")


def test_gradient_bulk(p, g, ds, steps):
    """d(sum w.(hs, hl, evap))/d(atemp, aqh, wspeed) of EXF_BULKFORMULAE at the X04 state of it=1 (theta fixed):
    finite everywhere (land, halo lanes), equal to central differences at 3 wet points per input with |huol| < 9,
    |huol| > 1e-2 and usn > 1.1*umin in both iterations (no clip/switch nearby)."""
    it = 1
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    s04 = state(ds, it, "X04_exf_wind", "X03_exf_radiation")
    theta = jnp.asarray(F(ds, it, "S00_begin", "theta"))
    names = ["atemp", "aqh", "wspeed"]

    def fun(x):
        e = dict(s04)
        e.update(x)
        e["sh"] = e["sh"].at[:, J, I].set(jnp.maximum(e["wspeed"][:, J, I], p.umin))   # exf_wind.F:248
        o = X.exf_bulkformulae(p, g, e, theta)
        return {k: o[k] for k in ("hs", "hl", "evap")}
    x0 = {n: s04[n] for n in names}
    # thresholds: recompute huol and usn of both iterations (interior) from the dumped locals
    lb = X.exf_bulkformulae(p, g, s04, theta, return_locals=True)[2]
    ok = np.ones(L.shape2d, bool)
    tstar, qstar, ustar = (np.asarray(lb[k]) for k in ("tstar", "qstar", "ustar"))
    at, aq = np.asarray(s04["atemp"][:, J, I]), np.asarray(s04["aqh"][:, J, I])
    huol = ((tstar / (at * (1 + p.humid_fac * aq)) + qstar / (1 / p.humid_fac + aq)) * p.hu * 0.4 * p.gravity_mks
            / (ustar * ustar))
    good = (np.abs(huol) < 9.0) & (np.abs(huol) > 1e-2) & (np.asarray(s04["wspeed"][:, J, I]) > 1.1 * p.umin * 3)
    ok[:, J, I] = good & (np.asarray(g.maskC[:, 0, J, I]) > 0)
    cand = np.argwhere(ok)
    rng = np.random.default_rng(3)
    pts = cand[rng.choice(len(cand), 3, replace=False)]
    _fd_check(fun, x0, names, lambda n: pts, ["hs", "hl", "evap"], rng, (1e-7, 1e-6, 1e-5), "bulk")


def test_gradient_getforcing(p, g, ex, ds, steps, zs):
    """d(sum w.(Qnet, EmPmR, fu, fv))/d(next records of atemp, wspeed, ustress) through the whole EXF_GETFORCING at
    it=1 (DEVICE_LIBM): finite everywhere, equal to central differences at 2 wet points per record (|ustress| < 1, far
    from the 2 N/m2 clip)."""
    it = 1
    s = steps[it]
    exf0, ff0, theta = start_state(ds, it)
    L = g.layout
    names = ["atemp", "wspeed", "ustress"]

    def fun(x):
        jb = dict(s["jb"])
        for n in names:
            jb[n] = (s["jb"][n][0], x[n])
        _, ff = X.exf_getforcing(p, g, ex, exf0, ff0, jb, s["jf"], theta, s["myTime"], s["jzt"], zs)
        return {k: ff[k] for k in ("Qnet", "EmPmR", "fu", "fv")}
    x0 = {n: s["jb"][n][1] for n in names}
    wet = np.argwhere(np.asarray(g.maskC[:, 0]) > 0)
    wet = wet[(wet[:, 1] >= L.OLy + 2) & (wet[:, 1] < L.OLy + L.sNy - 2) & (wet[:, 2] >= L.OLx + 2)
              & (wet[:, 2] < L.OLx + L.sNx - 2)]
    u1 = np.asarray(s["jb"]["ustress"][1])
    wet = wet[np.abs(u1[wet[:, 0], wet[:, 1], wet[:, 2]]) < 1.0]
    rng = np.random.default_rng(5)
    pts = wet[rng.choice(len(wet), 2, replace=False)]
    _fd_check(fun, x0, names, lambda n: pts, ["Qnet", "EmPmR", "fu", "fv"], rng, (1e-7, 1e-6, 1e-5), "getforcing")
