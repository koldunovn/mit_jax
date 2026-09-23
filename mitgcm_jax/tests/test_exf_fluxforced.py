"""Flux-forced surface forcing of one time step (plan Task 9): LOAD_FIELDS_DRIVER -> EXF_GETFORCING (pkg/exf, ff
overrides) and EXTERNAL_FORCING_SURF at the start of DO_OCEANIC_PHYS, gated against the forced 3-step oracle.

Gates (oracle.FORCED, iterations 1, 2, 3; iteration 3 loads a new record, iteration 1 starts from `first`):
  - S02_load_fields: records read from the 1992 files by ExfRecordLoader + EXF_GetFFieldRec weights, replayed from
    the S00_begin EXF/FFIELDS arrays -> x group (ustress, vstress, hflux, swflux, sflux, saltflx, apressure, spflx),
    f group (fu, fv, Qnet, Qsw, EmPmR, saltFlux, pLoad; the rest untouched), p group (saltPlumeFlux).
    Achieved: max abs. difference 0 at every point, halos included, all three iterations, jitted and eager (the only
    bit differences: sign of zero at ~1300 halo points of ustress/vstress/fu/fv, see test_load_fields_S02).
  - CTRL_MAP_FORCING is not called (useCTRL=F, ff/forward_step.F:524): the oracle has no S03_ctrl_map_forcing
    record (its anchor sits inside that IF), and S02 is the input of DO_OCEANIC_PHYS.
  - P01_external_forcing_surf: S02 f/p groups + S00 theta/salt -> surfaceForcingU/V/T/S, phi0surf (all points),
    saltPlumeDepth = 0, saltPlumeFlux exchanged. Achieved: bitwise, all three iterations, jitted and eager.
  - Record/weight sequence (EXF_GetFFieldRec with the ported pkg/cal) across a month boundary, the 1992 leap day and
    the 1992/1993 year boundary against a hand computation; the calendar date against Python's datetime.
Negative controls: interpolation weight perturbed by 1e-6 (relative), the next record loaded, recip_Cp perturbed by
1e-6, the salt-plume term dropped, the record start time shifted by 1 s: each makes its comparison fail.
Gradient: d(sum of weighted P01 outputs)/d(hflux, ustress, spflx, apressure record buffers) vs central differences
(h-sweep over four decades): worst rel. difference 5.0e-13.
"""

import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core.external_forcing import SurfForcingParams, oceanic_phys_forcing
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.pkgs import exf_fluxforced as X
from mitgcm_jax.tests import oracle

ORACLE = oracle.FORCED
ITERS = (1, 2, 3)
XFIELDS = ["ustress", "vstress", "hflux", "swflux", "sflux", "saltflx", "apressure", "spflx"]
FFIELDS = ["fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad", "sIceLoad"]
SURF_OUT = ["surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "phi0surf"]
F_GROUP = ["surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "fu", "fv", "Qnet", "Qsw",
           "EmPmR", "saltFlux", "pLoad", "phi0surf", "sIceLoad"]


# ---------------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def ds():
    return oracle.dumpset(ORACLE)


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
    return X.ExfParams.from_namelists(nml)


@pytest.fixture(scope="module")
def q(nml):
    return SurfForcingParams.from_namelists(nml)


@pytest.fixture(scope="module")
def steps(nml, p, g):
    """ExfRecordLoader driven through the oracle's steps in order: {myIter: (myTime, bufs, facs, recs)}."""
    loader = X.ExfRecordLoader(p, g, oracle.run_dir(ORACLE))
    out = {}
    for iloop in range(1, len(ITERS) + 1):
        myTime, myIter = X.model_time(nml, iloop)
        bufs, facs, recs = loader.load(myTime, myIter)
        out[myIter] = (myTime, bufs, facs, recs)
    out["loaded"] = list(loader.loaded)
    return out


def F(ds, it, stage, name):
    return oracle.field(ds, it, stage, name)


def relerr(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    d = np.max(np.abs(got - ref))
    s = np.max(np.abs(ref))
    return d / s if s > 0 else d


def run_load_fields(p, g, ex, ds, it, bufs, facs, myTime, jit=True):
    exf = {n: jnp.asarray(F(ds, it, "S00_begin", n)) for n in XFIELDS}
    exf["runoff"] = X.exf_init_varia(p, g.layout)["runoff"]
    ff = {n: jnp.asarray(F(ds, it, "S00_begin", n)) for n in FFIELDS + ["saltPlumeFlux"]}
    bufs = {n: (jnp.asarray(a), jnp.asarray(b)) for n, (a, b) in bufs.items()}
    facs = {n: jnp.asarray(f) for n, f in facs.items()}
    fn = lambda p, g, exf, ff, bufs, facs, t: X.exf_getforcing(p, g, ex, exf, ff, bufs, facs, t)  # noqa: E731
    return (jax.jit(fn) if jit else fn)(p, g, exf, ff, bufs, facs, myTime)


def run_surf(q, g, ex, ds, it, recip_cp_scale=1.0, jit=True):
    ff = {n: jnp.asarray(F(ds, it, "S02_load_fields", n)) for n in FFIELDS}
    ff["saltPlumeFlux"] = jnp.asarray(F(ds, it, "S02_load_fields", "saltPlumeFlux"))
    depth = jnp.asarray(F(ds, it, "S02_load_fields", "saltPlumeDepth"))
    theta = jnp.asarray(F(ds, it, "S00_begin", "theta"))
    salt = jnp.asarray(F(ds, it, "S00_begin", "salt"))
    if recip_cp_scale != 1.0:
        q = X._replace(q, HeatCapacity_Cp=q.HeatCapacity_Cp / recip_cp_scale)
    fn = lambda q, g, ff, depth, theta, salt: oceanic_phys_forcing(q, g, ex, ff, depth, theta, salt)  # noqa: E731
    return (jax.jit(fn) if jit else fn)(q, g, ff, depth, theta, salt)


# --------------------------------------------------------------------------------------------------- time logic
def test_calendar_and_start_times(p):
    """CAL_SET: startDate 19920101 12:00 + startTime (nIter0*deltaTClock = 3600 s) -> 19920101 13:00, a leap year,
    Wednesday (weekday 6 counting from Friday 15821015 = 1). Every record starts at 03:00 = 10800 s of the year."""
    cal = p.cal
    assert cal.modelStart == 3600.0 and cal.modelStep == 3600.0
    assert cal.modelBaseDate == (19920101, 120000, 2, 6)
    assert cal.modelStartDate[:3] == (19920101, 130000, 2)
    assert {f.startTime for f in p.fields} == {10800.0}
    assert {f.period for f in p.fields} == {21600.0}
    assert [f.name for f in p.fields] == list(X.READ_FIELDS)


def test_oracle_step_records(steps):
    """it=1 (13:00): first (not changed), records 2 (09:00) and 3 (15:00), fac = 1 - 14400/21600; it=2 (14:00):
    same records, fac = 1 - 18000/21600; it=3 (15:00): record change (changed=T), records 3 and 4, fac = 1."""
    want = {1: (2, 3, 1.0 - 14400.0 / 21600.0, True, False),
            2: (2, 3, 1.0 - 18000.0 / 21600.0, False, False),
            3: (3, 4, 1.0, False, True)}
    for it, (c0, c1, fac, first, changed) in want.items():
        for name, r in steps[it][3].items():
            assert (r.count0, r.count1, r.year0, r.year1, r.first, r.changed) == (c0, c1, 1992, 1992, first, changed)
            assert r.fac == fac, (it, name, r.fac, fac)
    # records read from disk: 2 and 3 at it=1, 4 at it=3 (one file per field, 1992)
    recs = sorted({(it, rec) for it, _, rec, _ in steps["loaded"]})
    assert recs == [(1, 2), (1, 3), (3, 4)]
    assert all(f.endswith("_1992") for _, _, _, f in steps["loaded"])


def _hand_rec(date, start=10800.0, period=21600.0, step=3600.0):
    """EXF_GetFFieldRec (useExfYearlyFields) evaluated by hand from a Python datetime."""
    year0 = date.year
    secs = (date - dt.datetime(year0, 1, 1)).total_seconds()
    if secs < start:
        year0 -= 1
    leap = year0 % 4 == 0 and (year0 % 100 != 0 or year0 % 400 == 0)
    siy = (366.0 if leap else 365.0) * 86400.0
    if secs < start:
        secs += siy
    tot = secs - start
    c0 = int((tot + 0.5) / period) + 1
    y1, c1 = year0, c0 + 1
    if start + c0 * period >= siy:
        y1, c1 = year0 + 1, 1
    fs = tot % period
    fac = 1.0 - fs / period if year0 == y1 else 1.0 - fs / (siy - (c0 - 1) * period)
    return c0, c1, year0, y1, fac, fs - step < 0.0


BOUNDARIES = {"month": dt.datetime(1992, 1, 31, 12), "leapday": dt.datetime(1992, 2, 28, 12),
              "year": dt.datetime(1992, 12, 31, 12), "year93": dt.datetime(1993, 12, 31, 12)}


def test_record_sequence_across_boundaries(p):
    """Hourly model steps over 36 h around a month end, the 1992 leap day and two year ends: record numbers, file
    years, weights and the `changed` switch equal the hand computation; the calendar date equals datetime's."""
    cal = p.cal
    t0 = dt.datetime(1992, 1, 1, 13)
    f = p.field_map["hflux"]
    dates = [b + dt.timedelta(hours=h) for b in BOUNDARIES.values() for h in range(37)]
    for date in dates:
        myTime = cal.modelStart + (date - t0).total_seconds()
        d = X.cal_getdate(cal, 10, myTime)
        assert (d[0], d[1]) == (int(date.strftime("%Y%m%d")), int(date.strftime("%H%M%S"))), (date, d)
        r = X.exf_getffieldrec(cal, f.startTime, f.period, True, myTime, 10)
        assert (r.count0, r.count1, r.year0, r.year1, r.fac, r.changed) == _hand_rec(date), (date, r)


def test_record_values_by_hand(p):
    """Explicit values: 1992-02-29 12:00 -> records 238/239, fac 0.5; 1992-12-31 22:00 -> record 1464 of 1992 and
    record 1 of 1993, fac = 1 - 3600/21600; 1993-01-01 01:00 -> the same pair, fac = 1 - 14400/21600, file names
    *_1992 / *_1993; 1993-01-01 03:00 -> records 1/2 of 1993, fac 1, changed."""
    cal = p.cal
    t0 = dt.datetime(1992, 1, 1, 13)
    f = p.field_map["hflux"]

    def rec(*ymdh):
        myTime = cal.modelStart + (dt.datetime(*ymdh) - t0).total_seconds()
        r = X.exf_getffieldrec(cal, f.startTime, f.period, True, myTime, 10)
        return r.count0, r.count1, r.year0, r.year1, r.fac, r.changed

    assert rec(1992, 2, 29, 12) == (238, 239, 1992, 1992, 0.5, False)
    assert rec(1992, 12, 31, 22) == (1464, 1, 1992, 1993, 1.0 - 3600.0 / 21600.0, False)
    assert rec(1993, 1, 1, 1) == (1464, 1, 1992, 1993, 1.0 - 14400.0 / 21600.0, False)
    assert rec(1993, 1, 1, 3) == (1, 2, 1993, 1993, 1.0, True)
    assert X.exf_getyearlyfieldname(True, False, 21600.0, 1993, "TFLUX_6hourlyavg") == "TFLUX_6hourlyavg_1993"


def test_calendar_long_sweep(p):
    """cal_GetDate over three years in 7 h + 13 min steps (cal_TimeInterval + cal_AddTime) equals datetime."""
    cal = p.cal
    t0 = dt.datetime(1992, 1, 1, 13)
    for k in range(0, 3 * 366 * 24 // 7):
        s = k * (7 * 3600 + 13 * 60)
        date = t0 + dt.timedelta(seconds=s)
        d = X.cal_getdate(cal, 10, cal.modelStart + s)
        assert (d[0], d[1]) == (int(date.strftime("%Y%m%d")), int(date.strftime("%H%M%S"))), (date, d)


def test_negative_control_time_logic(p):
    """Record start time shifted by 1 s: the sequence no longer matches the hand computation."""
    cal = p.cal
    t0 = dt.datetime(1992, 1, 1, 13)
    f = p.field_map["hflux"]
    bad = 0
    for h in range(37):
        date = BOUNDARIES["year"] + dt.timedelta(hours=h)
        myTime = cal.modelStart + (date - t0).total_seconds()
        r = X.exf_getffieldrec(cal, f.startTime + 1.0, f.period, True, myTime, 10)
        bad += (r.count0, r.count1, r.year0, r.year1, r.fac, r.changed) != _hand_rec(date)
    assert bad > 0


# --------------------------------------------------------------------------------------------------- replay gates
@pytest.mark.parametrize("it", ITERS)
def test_load_fields_S02(p, g, ex, ds, steps, it):
    """LOAD_FIELDS_DRIVER replayed from S00_begin equals S02_load_fields, every point, halos included: max abs
    difference 0 (jitted and eager; the only bit differences are the signs of 0 at ~1300 halo points of
    ustress/vstress/fu/fv filled by the signed vector exchange from a +0 source: Fortran +0, JAX -0)."""
    myTime, bufs, facs, _ = steps[it]
    for jit in (True, False):
        exf, ff = run_load_fields(p, g, ex, ds, it, bufs, facs, myTime, jit=jit)
        errs = {}
        for n in XFIELDS:
            errs[n] = relerr(exf[n], F(ds, it, "S02_load_fields", n))
        for n in ["fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad", "saltPlumeFlux"]:
            errs[n] = relerr(ff[n], F(ds, it, "S02_load_fields", n))
        assert max(errs.values()) == 0.0, (jit, errs)
    # fields LOAD_FIELDS_DRIVER does not touch
    for n in ["surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "phi0surf", "sIceLoad",
              "saltPlumeDepth"]:
        np.testing.assert_array_equal(F(ds, it, "S02_load_fields", n), F(ds, it, "S00_begin", n), err_msg=n)
    # CTRL_MAP_FORCING not called (useCTRL=F, ff/forward_step.F:524): its dump anchor (inside the IF) never ran
    assert not any(k[1] == "S03_ctrl_map_forcing" for k in ds.index)


@pytest.mark.parametrize("it", ITERS)
def test_external_forcing_surf_P01(q, g, ex, ds, it):
    """DO_OCEANIC_PHYS up to EXTERNAL_FORCING_SURF replayed from S02 equals P01 bitwise (all points, jitted and
    eager). Needs XLA_FLAGS=--xla_cpu_max_isa=AVX (conftest.py) and q passed as a jit argument: with FMA contraction,
    or with q's floats folded as compile-time constants ((x*recip_Cp)*mass2rUnit -> x*(recip_Cp*mass2rUnit)),
    surfaceForcingT/S differ by up to 2.4e-16 relative."""
    for jit in (True, False):
        ff, out, depth = run_surf(q, g, ex, ds, it, jit=jit)
        errs = {n: relerr(out[n], F(ds, it, "P01_external_forcing_surf", n)) for n in SURF_OUT}
        for n in FFIELDS + ["saltPlumeFlux"]:
            errs[n] = relerr(ff[n], F(ds, it, "P01_external_forcing_surf", n))
        errs["saltPlumeDepth"] = relerr(depth, F(ds, it, "P01_external_forcing_surf", "saltPlumeDepth"))
        assert max(errs.values()) == 0.0, (jit, errs)


# ----------------------------------------------------------------------------------------------- negative controls
def test_negative_controls_load_fields(p, g, ex, ds, steps):
    """Weight perturbed by 1e-6 (relative) or the next record loaded: S02 comparison fails."""
    it = 1
    myTime, bufs, facs, _ = steps[it]
    facs_bad = dict(facs)
    facs_bad["hflux"] = facs["hflux"] * (1.0 + 1e-6)
    exf, ff = run_load_fields(p, g, ex, ds, it, bufs, facs_bad, myTime)
    assert relerr(exf["hflux"], F(ds, it, "S02_load_fields", "hflux")) > 1e-9
    assert relerr(ff["Qnet"], F(ds, it, "S02_load_fields", "Qnet")) > 1e-9
    bufs_bad = dict(bufs)
    bufs_bad["spflx"] = (bufs["spflx"][1], steps[3][1]["spflx"][1])   # records 3, 4 instead of 2, 3
    exf, ff = run_load_fields(p, g, ex, ds, it, bufs_bad, facs, myTime)
    assert relerr(ff["saltPlumeFlux"], F(ds, it, "S02_load_fields", "saltPlumeFlux")) > 1e-3


def test_negative_controls_surf(q, g, ex, ds):
    """recip_Cp perturbed by 1e-6, or the salt-plume term dropped: P01 comparison fails."""
    it = 1
    _, out, _ = run_surf(q, g, ex, ds, it, recip_cp_scale=1.0 + 1e-6)
    assert relerr(out["surfaceForcingT"], F(ds, it, "P01_external_forcing_surf", "surfaceForcingT")) > 1e-9
    _, out, _ = run_surf(X._replace(q, useSALT_PLUME=False), g, ex, ds, it)
    assert relerr(out["surfaceForcingS"], F(ds, it, "P01_external_forcing_surf", "surfaceForcingS")) > 1e-9


# ---------------------------------------------------------------------------------------------------------- gradient
def test_gradient_wrt_forcing_records(p, q, g, ex, ds, steps):
    """J = sum(w_T*surfaceForcingT + w_U*surfaceForcingU + w_S*surfaceForcingS + w_P*phi0surf) after LOAD_FIELDS
    + EXTERNAL_FORCING_SURF at it=1, as a function of the fld1 buffers (next record) of hflux, ustress, spflx and
    apressure. jax.grad is finite everywhere (dry and halo lanes included) and equals central differences at wet
    points (differences taken pointwise on the outputs, then weighted); the map is linear away from the wind-stress
    clip, so the h-sweep is flat: worst rel. difference 5.0e-13 over h = 1e-3 ... 1e1 x field scale (ustress:
    1e-5 ... 1e-1 N/m2 at points with |tau| < 1)."""
    it = 1
    myTime, bufs, facs, _ = steps[it]
    L = g.layout
    rng = np.random.default_rng(1)
    w = {n: rng.normal(size=L.shape2d) for n in ["surfaceForcingT", "surfaceForcingU", "surfaceForcingS", "phi0surf"]}
    consts = dict(theta=jnp.asarray(F(ds, it, "S00_begin", "theta")), salt=jnp.asarray(F(ds, it, "S00_begin", "salt")),
                  depth=jnp.asarray(F(ds, it, "S00_begin", "saltPlumeDepth")),
                  exf={n: jnp.asarray(F(ds, it, "S00_begin", n)) for n in XFIELDS},
                  ff={n: jnp.asarray(F(ds, it, "S00_begin", n)) for n in FFIELDS + ["saltPlumeFlux"]},
                  bufs={n: (jnp.asarray(a), jnp.asarray(b)) for n, (a, b) in bufs.items()},
                  facs={n: jnp.asarray(f) for n, f in facs.items()}, g=g, p=p, q=q)
    consts["exf"]["runoff"] = X.exf_init_varia(p, L)["runoff"]
    names = ["hflux", "ustress", "spflx", "apressure"]

    def outputs(recs, c):
        b = dict(c["bufs"])
        for n in names:
            b[n] = (c["bufs"][n][0], recs[n])
        exf, ff = X.exf_getforcing(c["p"], c["g"], ex, c["exf"], c["ff"], b, c["facs"], myTime)
        _, out, _ = oceanic_phys_forcing(c["q"], c["g"], ex, ff, c["depth"], c["theta"], c["salt"])
        return {n: out[n] for n in w}

    def J(recs, c):
        o = outputs(recs, c)
        return sum(jnp.sum(jnp.asarray(w[n]) * o[n]) for n in w)

    x0 = {n: consts["bufs"][n][1] for n in names}
    out_j = jax.jit(outputs)
    grad = jax.jit(jax.grad(J))(x0, consts)
    for n in names:
        assert np.all(np.isfinite(np.asarray(grad[n]))), n
        assert np.any(np.asarray(grad[n]) != 0.0), n
    wet = np.argwhere(np.asarray(g.maskC[:, 0]) > 0)
    wet = wet[(wet[:, 1] >= L.OLy) & (wet[:, 1] < L.OLy + L.sNy) & (wet[:, 2] >= L.OLx) & (wet[:, 2] < L.OLx + L.sNx)]
    u0 = np.asarray(x0["ustress"])
    worst = 0.0
    for n in names:
        cand = wet if n != "ustress" else wet[np.abs(u0[wet[:, 0], wet[:, 1], wet[:, 2]]) < 1.0]
        pts = cand[rng.choice(len(cand), 3, replace=False)]
        scale = float(np.max(np.abs(np.asarray(x0[n])))) or 1.0
        hs = [hr * scale for hr in (1e-3, 1e-2, 1e-1, 1e0, 1e1)] if n != "ustress" else [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for t, j, i in pts:
            gad = float(grad[n][t, j, i])
            for h in hs:
                xp, xm = dict(x0), dict(x0)
                xp[n] = x0[n].at[t, j, i].add(h)
                xm[n] = x0[n].at[t, j, i].add(-h)
                op, om = out_j(xp, consts), out_j(xm, consts)
                fd = sum(np.sum(w[k] * (np.asarray(op[k]) - np.asarray(om[k]))) for k in w) / (2 * h)
                rel = abs(fd - gad) / max(abs(gad), 1e-300)
                worst = max(worst, rel)
                assert rel < 1e-8, (n, (t, j, i), h, gad, fd)
    print("worst FD rel. difference", worst)
