"""SEAICE_GROWTH (V4r4) + SEAICE_SOLVE4TEMP + SEAICE_BUDGET_OCEAN (plan M2.3) against the full-V4r4 oracle.

Oracle: oracle.FULL (full_jaxdump_v5), iterations 1-3. Stages (reference/jaxdump/SUBSTEPS.md): inputs of SEAICE_GROWTH
are I03_reg_ridge (AREA, HEFF, HSNOW, TICES, d_HEFFbyNEG, d_HSNWbyNEG), I00_seaice_begin (EXF: wspeed, atemp, aqh,
lwdown, swdown, evap, precip, snowprecip, runoff), I01_dynsolver (Qnet, Qsw, EmPmR, saltFlux, sIceLoad before growth),
S00_begin (theta, salt at k=1), G01_seaice_geometry (HEFFM); saltPlumeFlux enters as 0 (zeroed at the top of
DO_OCEANIC_PHYS under ALLOW_AUTODIFF, do_oceanic_phys.F:286-297 of the full tree). Outputs: H01..H06 (routine locals,
interior; the dump writes halos as 0) and I04_growth (full arrays, halos included).

Replay gates (each stage fed the dumped inputs of the stage before, all 3 iterations, every interior point; I04 at every
point incl. halos): H01..H06 and I04 are BITWISE (0 differing values, ~12k ice points per iteration). H04
(SEAICE_SOLVE4TEMP, 10 Newton steps + flux recomputation) is bitwise because the kernel's exp is `sg.glibc_exp` (bit
emulation of the oracle's glibc exp; own tests below); with XLA's exp (jnp.exp) H04 differs at ~2000 points by
<= 2.6e-14 relative (measured it 1/2/3: a_FWbySublimMult 2.3e-14/1.9e-14/2.6e-14, a_QbyATMmult_cover
7.5e-15/7.2e-15/1.0e-14, ticeOutMult 2.1e-16) -- asserted too, as the negative control of the exp emulation.
Free-run gate: the composed seaice_growth from the I03 inputs reproduces every H stage and I04 bitwise at every point
(halos pass through).
Categories: SEAICE_multDim = 1, so only level 1 of the nITD = 7 *Mult/TICES levels is computed; levels 2..7 of the
dumped *Mult locals are 0 and TICES(2..7) passes through unchanged (both asserted).

Negative controls (each must make its gate fail): IMAX_TICE 10 -> 9 (H04: 340 points, 9e-15 -- the Newton iteration is
converged to round-off after 9 steps, so only a bitwise gate sees the count); SEAICE_dalton * (1 + 1e-6) (H04);
SWFracB * (1 + 1e-6) (H06 open-water growth); frazilFrac * (1 + 1e-6) (H05 a_QbyOCN, frazil points exist); the
heatConsFix term dropped (I04 Qnet); TICES halos not passed through (I04 TICES).
Gradient: d(weighted sum of HEFF, HSNOW, AREA, TICES, Qnet, EmPmR, saltFlux) / d(HEFF, HSNOW, AREA, theta, atemp,
aqh, lwdown, swdown, wspeed, TICES) vs central differences at ice points away from every branch threshold (listed in
`BRANCHES`), finite gradients on every lane.
"""

import functools
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.io.dump import DumpSet, read_file
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import seaice_growth as sg
from mitgcm_jax.tests import oracle

ITERS = (1, 2, 3)
EXF_NAMES = (("wspeed", "wspeed"), ("atemp", "atemp"), ("aqh", "aqh"), ("lwdown", "lwdown"), ("swdown", "swdown"),
             ("evap", "evap"), ("precip", "precip"), ("snowPrecip", "snowprecip"), ("runoff", "runoff"))
FLX_NAMES = ("Qnet", "Qsw", "EmPmR", "saltFlux", "sIceLoad")
H01 = ("HEFFpreTH", "HSNWpreTH", "AREApreTH", "heffActual", "hsnowActual", "recip_heffActual", "UG", "TmixLoc")
H04 = ("ticeOutMult", "a_QbyATMmult_cover", "a_QSWbyATMmult_cover", "a_FWbySublimMult")
H05 = ("a_QbyATM_cover", "a_QSWbyATM_cover", "a_QbyATM_open", "a_QSWbyATM_open", "r_QbyATM_cover", "r_QbyATM_open",
       "a_FWbySublim", "r_FWbySublim", "a_QbyOCN", "r_QbyOCN")
H06 = ("d_HEFFbyOCNonICE", "d_HEFFbyATMonOCN", "d_HEFFbyFLOODING", "d_HEFFbyATMonOCN_open", "d_HEFFbyATMonOCN_cover",
       "d_HSNWbyATMonSNW", "d_HSNWbyOCNonSNW", "d_HSNWbyRAIN", "d_HFRWbyRAIN", "d_HEFFbySublim", "d_HSNWbySublim",
       "r_QbyATM_cover", "r_QbyATM_open", "r_FWbySublim")
OUT_FLX = ("Qnet", "Qsw", "EmPmR", "saltFlux", "saltPlumeFlux", "sIceLoad")

# Branch thresholds of the V4r4 code path (the gradient test keeps its points away from all of them):
BRANCHES = (
    "HEFFpreTH > 0 (regularisation, seaice_growth.F:660)",
    "wspeed vs SEAICE_EPS (UG = MAX, :736)",
    "HSNOW_ACTUAL > 0 (emissivity, penetrating SW; seaice_solve4temp.F:250, 311)",
    "HSNOW_ACTUAL > SEAICE_snowThick (albedo, seaice_solve4temp.F:297) and MIN(albedo blend, ALB_SNOW) (:303)",
    "TSURFin >= TMELT + SEAICE_wetAlbTemp (wet/dry albedo, seaice_solve4temp.F:279, 287)",
    "tsurf = MIN(tsurf, TMELT) at every Newton step (seaice_solve4temp.F:450)",
    "theta >= tempFrz (McPhee piston vs frazil, seaice_growth.F:1042) and AREApreTH > 0 (:1048)",
    "sublimation MAX(MIN(r_FWbySublim, HSNOW*SNOW2ICE), 0), MAX(MIN(r_FWbySublim, HEFF), 0) (:1242, 1264)",
    "MAX(r_QbyOCN, -HEFF) (:1333); snow melt MAX/MIN(., 0) (:1379-1380, 1555-1556); MAX(-HEFF, ...) (:1440)",
    "a_QbyATM_cover >= 0 (precip falls as snow or rain, :1483)",
    "open-water growth MAX(., -HEFF*facOpenMelt) (:1606); flooding MAX(0, .) (:1718)",
    "AREA: HEFF > 0 or HSNOW > 0, MAX(0, MIN(area_max, .)), MAX(0, d_HEFFbyATMonOCN_open), MIN(0, net melt)",
    "salt flux MAX(0, MIN(salt0, SSS)) (:1996); salt plume MAX(., 0) (:2032)",
)


def _dumpset_parallel(directory):
    """DumpSet of `directory` with the record headers read in parallel threads (a serial index of the full-tree oracle
    takes ~110 s on cold Lustre, this ~3 s)."""
    files = sorted(Path(directory).glob("jd_*_t*.bin"))
    assert files, directory
    with ThreadPoolExecutor(min(len(files), 64)) as ex:
        per_file = list(ex.map(lambda f: read_file(f, lazy=True), files))
    ds = DumpSet.__new__(DumpSet)
    ds.dir, ds.index = Path(directory), {}
    for recs in per_file:
        for r in recs:
            ds.index.setdefault((r.iter, r.stage, r.field), {})[r.tile] = r
    ds.order = list(ds.index)
    return ds


@functools.lru_cache(maxsize=1)
def dumps():
    return _dumpset_parallel(oracle.run_dir(oracle.FULL) / "jaxdump")


@functools.lru_cache(maxsize=1)
def params():
    ds = dumps()
    g = grid_from_dump(ds, 1)
    return sg.SeaiceGrowthParams.from_namelists(RunNamelists(oracle.run_dir(oracle.FULL)), g), g


@functools.lru_cache(maxsize=3)
def case(it):
    ds = dumps()

    def F(st, n):
        return oracle.field(ds, it, st, n)

    g = grid_from_dump(ds, it)
    ice = {n: F("I03_reg_ridge", n) for n in ("AREA", "HEFF", "HSNOW", "TICES", "d_HEFFbyNEG", "d_HSNWbyNEG")}
    ocn = dict(theta_s=F("S00_begin", "theta")[:, 0], salt_s=F("S00_begin", "salt")[:, 0])
    exf = {k: F("I00_seaice_begin", n) for k, n in EXF_NAMES}
    flx = {n: F("I01_dynsolver", n) for n in FLX_NAMES}
    flx["saltPlumeFlux"] = np.zeros_like(flx["Qnet"])  # do_oceanic_phys.F:286-297 (full tree, ALLOW_AUTODIFF)
    ref = {st: {n: F(st, n) for n in names} for st, names in (
        ("H01_growth_pre_budget", H01), ("H02_growth_budget_ocean", ("a_QbyATM_open", "a_QSWbyATM_open")),
        ("H03_growth_pre_solve4temp", ("UG", "heffActualMult", "hsnowActualMult", "ticeInMult")),
        ("H04_growth_solve4temp", H04 + ("ticeInMult",)), ("H05_growth_heat_stocks", H05),
        ("H06_growth_ocean_forcing", H06 + ("AREA", "HEFF", "HSNOW", "TICES")),
        ("I04_growth", ("AREA", "HEFF", "HSNOW", "TICES") + OUT_FLX + ("d_HEFFbyNEG", "d_HSNWbyNEG")))}
    return dict(g=g, HEFFM=F("G01_seaice_geometry", "HEFFM"), ice=ice, ocn=ocn, exf=exf, flx=flx, ref=ref)


def I(a):
    L = params()[1].layout
    return jnp.asarray(sg.interior(L, jnp.asarray(a)))


def ndiff(got, ref):
    """(number of differing values, max |diff| / max |ref|)."""
    got, ref = np.asarray(got), np.asarray(ref)
    d = np.abs(got - ref)
    m = float(np.max(np.abs(ref)))
    return int(np.sum(got != ref)), (float(d.max()) / m if m > 0 else float(d.max()))


def bitwise_exp():
    return sg.default_exp() is not jnp.exp


# tolerances (max rel. error) for results that pass through SEAICE_SOLVE4TEMP's exp
TOL_EXP = 0.0 if bitwise_exp() else 5e-14


def check(name, got, ref, tol=0.0):
    n, e = ndiff(got, ref)
    print(f"{name:34s} differing {n:6d}  max rel err {e:.2e}")
    assert e <= tol, (name, n, e)
    return n, e


# ------------------------------------------------------------------------------------------ glibc exp


def test_glibc_exp_tables_match_libm():
    """The regenerated coar/fine/accurate tables and the 13 constants equal the bytes of the oracle's libm."""
    assert sg.glibc_exp_tables_vs_libm() == []


def test_glibc_exp_bitwise_vs_libm():
    """sg.glibc_exp == glibc exp (Python math.exp: same libm, same CPU model as the oracle run) on the argument ranges
    of SEAICE_SOLVE4TEMP and a log-uniform sweep; XLA's exp differs at ~14 %. Negative control: jnp.exp."""
    rng = np.random.default_rng(7)
    n = 50_000
    xs = np.concatenate([rng.uniform(-15.0, -0.075, n), rng.uniform(1.0, 8.0, n), rng.uniform(20.0, 30.0, n),
                         rng.choice([-1.0, 1.0], n) * np.exp(rng.uniform(np.log(1e-12), np.log(708.0), n))])
    ref = np.fromiter((math.exp(v) for v in xs), np.float64, xs.size)
    got = np.asarray(jax.jit(sg.glibc_exp)(xs))
    bad = int(np.sum(got != ref))
    badx = int(np.sum(np.asarray(jax.jit(jnp.exp)(xs)) != ref))
    print(f"glibc_exp mismatches {bad} / {xs.size}; jnp.exp {badx}")
    assert bad == 0 and badx > 1000
    g = jax.grad(lambda v: jnp.sum(sg.glibc_exp(v)))(jnp.asarray(xs[:100]))
    np.testing.assert_array_equal(np.asarray(g), got[:100])


# ------------------------------------------------------------------------------------------ params


def _stdout_params():
    """name -> first printed value of the seaice/model parameter summary in the oracle's STDOUT.0000."""
    lines = (oracle.run_dir(oracle.FULL) / "STDOUT.0000").read_text().splitlines()
    out = {}
    pat = re.compile(r"^(\w+)\s*=\s*/\*")
    for k, line in enumerate(lines[:-1]):
        m = pat.match(line.split(")", 1)[-1].strip())
        if m:
            val = lines[k + 1].split(")", 1)[-1].strip().rstrip(",").split(",")[0].strip()
            out.setdefault(m.group(1), val)
    return out


def test_params_match_fortran_summary():
    """Every parameter the kernel reads equals the value the Fortran printed (SEAICE_SUMMARY, INI_PARMS)."""
    p, _ = params()
    so = _stdout_params()
    pairs = dict(SEAICE_deltaTtherm="deltaTtherm", SEAICE_rhoIce="rhoIce", SEAICE_rhoSnow="rhoSnow",
                 SEAICE_rhoAir="rhoAir", SEAICE_lhEvap="lhEvap", SEAICE_lhFusion="lhFusion",
                 SEAICE_mcPheePiston="mcPheePiston", SEAICE_mcPheeTaper="mcPheeTaper",
                 SEAICE_frazilFrac="frazilFrac", SEAICE_tempFrz0="tempFrz0", SEAICE_dTempFrz_dS="dTempFrz_dS",
                 HO="HO", HO_south="HO_south", SEAICE_area_max="area_max", SEAICE_salt0="salt0",
                 SEAICE_dryIceAlb="dryIceAlb", SEAICE_wetIceAlb="wetIceAlb", SEAICE_drySnowAlb="drySnowAlb",
                 SEAICE_wetSnowAlb="wetSnowAlb", SEAICE_dryIceAlb_south="dryIceAlb_south",
                 SEAICE_wetIceAlb_south="wetIceAlb_south", SEAICE_drySnowAlb_south="drySnowAlb_south",
                 SEAICE_wetSnowAlb_south="wetSnowAlb_south", SEAICE_wetAlbTemp="wetAlbTemp",
                 SEAICE_snow_emiss="snow_emiss", SEAICE_ice_emiss="ice_emiss", SEAICE_cpAir="cpAir",
                 SEAICE_dalton="dalton", SEAICE_iceConduct="iceConduct", SEAICE_snowConduct="snowConduct",
                 SEAICE_snowThick="snowThick", SEAICE_shortwave="shortwave", MIN_ATEMP="MIN_ATEMP",
                 MIN_LWDOWN="MIN_LWDOWN", SEAICE_EPS="EPS", SEAICE_area_reg="area_reg", SEAICE_hice_reg="hice_reg",
                 HeatCapacity_Cp="HeatCapacity_Cp", celsius2K="celsius2K", rhoConst="rhoConst")
    bad = []
    for fname, attr in pairs.items():
        assert fname in so, fname
        if float(so[fname]) != getattr(p, attr):
            bad.append((fname, so[fname], getattr(p, attr)))
    for fname, val in (("IMAX_TICE", p.IMAX_TICE), ("postSolvTempIter", p.postSolvTempIter),
                       ("SEAICE_multDim", p.multDim), ("SEAICE_PDF", p.SEAICE_PDF[0])):
        if float(so[fname]) != float(val):
            bad.append((fname, so[fname], val))
    for fname, val in (("SEAICE_doOpenWaterGrowth", p.facOpenGrow == 1.0),
                       ("SEAICE_doOpenWaterMelt", p.facOpenMelt == 1.0), ("SEAICEuseFlooding", True),
                       ("SEAICEheatConsFix", True), ("SEAICE_growMeltByConv", False), ("useMaykutSatVapPoly", False),
                       ("SEAICE_mcPheeStepFunc", False), ("usePW79thermodynamics", True)):
        if (so[fname] == "T") != val:
            bad.append((fname, so[fname], val))
    for fname, val in (("SEAICE_areaGainFormula", 1), ("SEAICE_areaLossFormula", 2)):
        if int(so[fname]) != val:
            bad.append((fname, so[fname], val))
    assert not bad, bad


# ------------------------------------------------------------------------------------------ stage replay gates


@pytest.mark.parametrize("it", ITERS)
def test_replay_h01_h02_h03(it):
    p, _ = params()
    c = case(it)
    L = c["g"].layout
    Z = functools.partial(sg.zero_halo, L)
    pre = jax.jit(sg.growth_pre_budget)(p, I(c["ice"]["AREA"]), I(c["ice"]["HEFF"]), I(c["ice"]["HSNOW"]),
                                        I(c["exf"]["wspeed"]), I(c["ocn"]["theta_s"]))
    for n in H01:
        check(f"it{it} H01 {n}", Z(pre[n]), c["ref"]["H01_growth_pre_budget"][n])
    qo, qswo = sg.budget_ocean(I(c["flx"]["Qnet"]), I(c["flx"]["Qsw"]))
    check(f"it{it} H02 a_QbyATM_open", Z(qo), c["ref"]["H02_growth_budget_ocean"]["a_QbyATM_open"])
    check(f"it{it} H02 a_QSWbyATM_open", Z(qswo), c["ref"]["H02_growth_budget_ocean"]["a_QSWbyATM_open"])
    r1 = c["ref"]["H01_growth_pre_budget"]
    hM, sM, tin = sg.category_inputs(p, I(r1["heffActual"]), I(r1["hsnowActual"]), I(c["ice"]["TICES"]))
    r3 = c["ref"]["H03_growth_pre_solve4temp"]
    check(f"it{it} H03 UG", r3["UG"], r1["UG"])
    for n, v in (("heffActualMult", hM), ("hsnowActualMult", sM), ("ticeInMult", tin)):
        check(f"it{it} H03 {n}[1]", Z(v[0]), r3[n][:, 0])
        assert np.all(r3[n][:, p.multDim:] == 0.0), n  # categories 2..nITD are never computed (multDim = 1)


@pytest.mark.parametrize("it", ITERS)
def test_replay_h04_solve4temp(it):
    p, _ = params()
    c = case(it)
    L = c["g"].layout
    r3, r4 = c["ref"]["H03_growth_pre_solve4temp"], c["ref"]["H04_growth_solve4temp"]
    e = c["exf"]
    out = jax.jit(sg.solve4temp)(p, I(r3["UG"]), I(r3["heffActualMult"][:, 0]), I(r3["hsnowActualMult"][:, 0]),
                                 I(r3["ticeInMult"][:, 0]), I(e["lwdown"]), I(e["atemp"]), I(e["swdown"]),
                                 I(e["aqh"]), I(c["ocn"]["salt_s"]), I(c["g"].yC))
    np.testing.assert_array_equal(r4["ticeInMult"], r3["ticeInMult"])  # TSURFin is not modified
    nice = int(np.sum(I(r3["heffActualMult"][:, 0]) > 0))
    print(f"it{it}: {nice} ice points, exp = {'glibc emulation' if bitwise_exp() else 'jnp.exp'}")
    assert nice > 1000
    for n, v in zip(H04, out):
        check(f"it{it} H04 {n}[1]", sg.zero_halo(L, v), r4[n][:, 0], TOL_EXP)
        assert np.all(r4[n][:, p.multDim:] == 0.0), n
    # the same with XLA's exp: ulp-level differences (the gate tells the two exp implementations apart)
    outx = jax.jit(functools.partial(sg.solve4temp, expf=jnp.exp))(
        p, I(r3["UG"]), I(r3["heffActualMult"][:, 0]), I(r3["hsnowActualMult"][:, 0]), I(r3["ticeInMult"][:, 0]),
        I(e["lwdown"]), I(e["atemp"]), I(e["swdown"]), I(e["aqh"]), I(c["ocn"]["salt_s"]), I(c["g"].yC))
    nx = 0
    for n, v in zip(H04, outx):
        k, err = check(f"it{it} H04 {n}[1] (jnp.exp)", sg.zero_halo(L, v), r4[n][:, 0], 5e-14)
        nx += k
    assert nx > 0


@pytest.mark.parametrize("it", ITERS)
def test_replay_h05_h06_i04(it):
    """H05 from the dumped solve4temp outputs (H04), H06 from H05, I04 from H06: bitwise whatever exp is used."""
    p, g = params()
    c = case(it)
    L = c["g"].layout
    Z = functools.partial(sg.zero_halo, L)
    r1, r2, r4, r5, r6 = (c["ref"][s] for s in ("H01_growth_pre_budget", "H02_growth_budget_ocean",
                                                   "H04_growth_solve4temp", "H05_growth_heat_stocks",
                                                   "H06_growth_ocean_forcing"))
    pre = {n: I(r1[n]) for n in H01}
    s4t = [tuple(I(r4[n][:, 0]) for n in H04)]
    maskC1, yC, HEFFM = I(c["g"].maskC[:, 0]), I(c["g"].yC), I(c["HEFFM"])
    theta_s, salt_s = I(c["ocn"]["theta_s"]), I(c["ocn"]["salt_s"])
    hs, tout = jax.jit(sg.heat_stocks)(p, pre, I(r2["a_QbyATM_open"]), I(r2["a_QSWbyATM_open"]), s4t, theta_s,
                                       salt_s, maskC1, c["g"].drF[0])
    for n in H05:
        check(f"it{it} H05 {n}", Z(hs[n]), r5[n])
    # H06 from the dumped H05
    hs_ref = {n: I(r5[n]) for n in H05}
    e = c["exf"]
    th = jax.jit(sg.thickness_updates)(p, pre, hs_ref, I(c["ice"]["AREA"]), I(c["ice"]["HEFF"]), I(c["ice"]["HSNOW"]),
                                       I(e["precip"]), I(e["snowPrecip"]), HEFFM, yC)
    for n in H06:
        check(f"it{it} H06 {n}", Z(th[n]), r6[n])
    for n in ("AREA", "HEFF", "HSNOW"):
        check(f"it{it} H06 {n} (interior)", Z(th[n]), Z(I(r6[n])))
    # I04 from the dumped H06 (+ H05 a_QSWbyATM_*)
    th_ref = {n: I(r6[n]) for n in H06 + ("AREA", "HEFF", "HSNOW")}
    _, of = jax.jit(sg.ocean_forcing)(p, pre, hs_ref, th_ref, I(c["ice"]["d_HEFFbyNEG"]), I(c["ice"]["d_HSNWbyNEG"]),
                                      theta_s, salt_s, I(e["evap"]), I(e["precip"]), I(e["snowPrecip"]),
                                      I(e["runoff"]), HEFFM, maskC1, yC)
    r = c["ref"]["I04_growth"]
    for n in OUT_FLX:
        full = sg.set_interior(L, c["flx"][n], of[n])
        check(f"it{it} I04 {n}", full, r[n])
    # not vacuous: ice grows/melts, snow, flooding, frazil and plume points exist
    assert np.any(r6["d_HEFFbyATMonOCN_open"] > 0) and np.any(r6["d_HEFFbyATMonOCN_cover"] < 0)
    assert np.any(r6["d_HEFFbyFLOODING"] > 0) and np.any(r["saltPlumeFlux"] > 0)


@pytest.mark.parametrize("it", ITERS)
def test_free_run_growth(it):
    """seaice_growth composed, from the I03 inputs: every H stage and I04 at every point (halos included); the
    fields SEAICE_GROWTH does not write (fu, fv, pLoad, TICES levels 2..7, halos) pass through."""
    p, _ = params()
    c = case(it)
    L = c["g"].layout
    Z = functools.partial(sg.zero_halo, L)
    ice_o, flx_o, diag = jax.jit(sg.seaice_growth)(p, c["g"], c["HEFFM"], c["ice"], c["ocn"], c["exf"], c["flx"])
    ref = c["ref"]
    for n in H01:
        check(f"it{it} free H01 {n}", Z(diag["pre"][n]), ref["H01_growth_pre_budget"][n])
    for n, v in zip(H04, diag["s4t"][0]):
        check(f"it{it} free H04 {n}", Z(v), ref["H04_growth_solve4temp"][n][:, 0], TOL_EXP)
    for n in H05:
        check(f"it{it} free H05 {n}", Z(diag["hs"][n]), ref["H05_growth_heat_stocks"][n], TOL_EXP)
    for n in H06:
        check(f"it{it} free H06 {n}", Z(diag["th"][n]), ref["H06_growth_ocean_forcing"][n], TOL_EXP)
    for n in ("AREA", "HEFF", "HSNOW", "TICES"):
        check(f"it{it} free I04 {n}", ice_o[n], ref["I04_growth"][n], TOL_EXP)
    for n in OUT_FLX:
        check(f"it{it} free I04 {n}", flx_o[n], ref["I04_growth"][n], TOL_EXP)
    np.testing.assert_array_equal(np.asarray(ice_o["TICES"])[:, 1:], c["ice"]["TICES"][:, 1:])
    np.testing.assert_array_equal(ref["I04_growth"]["d_HEFFbyNEG"], c["ice"]["d_HEFFbyNEG"])


# ------------------------------------------------------------------------------------------ negative controls


def _replace(p, **kw):
    return type(p)(**{**p.__dict__, **kw})


def test_negative_controls():
    p, _ = params()
    c = case(1)
    L = c["g"].layout
    Z = functools.partial(sg.zero_halo, L)
    ref = c["ref"]
    # (1) one Newton step fewer in SEAICE_SOLVE4TEMP
    r3, e = ref["H03_growth_pre_solve4temp"], c["exf"]
    args = (I(r3["UG"]), I(r3["heffActualMult"][:, 0]), I(r3["hsnowActualMult"][:, 0]), I(r3["ticeInMult"][:, 0]),
            I(e["lwdown"]), I(e["atemp"]), I(e["swdown"]), I(e["aqh"]), I(c["ocn"]["salt_s"]), I(c["g"].yC))
    out = jax.jit(sg.solve4temp)(_replace(p, IMAX_TICE=p.IMAX_TICE - 1), *args)
    n, e1 = ndiff(Z(out[1]), ref["H04_growth_solve4temp"]["a_QbyATMmult_cover"][:, 0])
    print("NC IMAX_TICE-1: differing", n, "max rel err", e1)  # Newton is converged to ~1e-14 after 9 steps
    assert n > 0 and e1 > TOL_EXP
    out = jax.jit(sg.solve4temp)(_replace(p, dalton=p.dalton * (1 + 1e-6)), *args)
    assert ndiff(Z(out[1]), ref["H04_growth_solve4temp"]["a_QbyATMmult_cover"][:, 0])[1] > 1e-9
    # (2)-(4) on the free run
    run = jax.jit(sg.seaice_growth)
    _, _, d2 = run(_replace(p, SWFracB=p.SWFracB * (1 + 1e-6)), c["g"], c["HEFFM"], c["ice"], c["ocn"], c["exf"],
                   c["flx"])
    assert ndiff(Z(d2["th"]["d_HEFFbyATMonOCN_open"]),
                 ref["H06_growth_ocean_forcing"]["d_HEFFbyATMonOCN_open"])[1] > 1e-9
    _, _, d3 = run(_replace(p, frazilFrac=p.frazilFrac * (1 + 1e-6)), c["g"], c["HEFFM"], c["ice"], c["ocn"],
                   c["exf"], c["flx"])
    assert ndiff(Z(d3["hs"]["a_QbyOCN"]), ref["H05_growth_heat_stocks"]["a_QbyOCN"])[1] > 1e-9
    # (4) heatConsFix term dropped: rebuild Qnet from the kernel's H06-stage Qnet (before the term)
    _, flx_o, d = run(p, c["g"], c["HEFFM"], c["ice"], c["ocn"], c["exf"], c["flx"])
    q_bad = sg.set_interior(L, c["flx"]["Qnet"], d["Qnet_H06"])
    assert ndiff(q_bad, ref["I04_growth"]["Qnet"])[1] > 1e-9
    # (5) halos not passed through (TICES written at every point)
    t_bad = np.asarray(sg.zero_halo(L, I(ref["I04_growth"]["TICES"])))
    assert ndiff(t_bad, ref["I04_growth"]["TICES"])[0] > 1000


# ------------------------------------------------------------------------------------------ gradient


def _select_points(c, p, n=24, seed=0):
    """Interior ice points far from every threshold in BRANCHES (margins in the input and in the Fortran's own H-stage
    values of iteration 1)."""
    L = c["g"].layout
    r = {s: {k: np.asarray(I(v)) for k, v in c["ref"][s].items() if np.ndim(v) == 3} for s in c["ref"]}
    r1, r5, r6 = r["H01_growth_pre_budget"], r["H05_growth_heat_stocks"], r["H06_growth_ocean_forcing"]
    t_in = np.asarray(I(c["ice"]["TICES"][:, 0]))
    t_out = np.asarray(I(c["ref"]["H04_growth_solve4temp"]["ticeOutMult"][:, 0]))
    theta = np.asarray(I(c["ocn"]["theta_s"]))
    hs_act = r1["hsnowActual"]
    ok = ((r1["HEFFpreTH"] > 0.3) & (r1["AREApreTH"] > 0.2) & (r1["AREApreTH"] < 0.9)
          & (hs_act > 0.01) & (hs_act < p.snowThick - 0.02)
          & (t_in < p.celsius2K + p.wetAlbTemp - 0.5) & (t_out < p.celsius2K - 0.5)
          & (np.abs(theta - p.tempFrz0) > 0.02) & (np.asarray(I(c["exf"]["wspeed"])) > 1.0)
          & (np.abs(r5["a_QbyATM_cover"] - r5["r_FWbySublim"]) > 1e-6)
          & (r6["d_HEFFbySublim"] == 0.0) & (r6["d_HSNWbySublim"] < -1e-7)      # sublimation fully from snow
          & (r6["d_HSNWbyATMonSNW"] == 0.0) & (r6["d_HSNWbyOCNonSNW"] == 0.0)  # no snow melt (far from 0)
          & (r6["d_HEFFbyFLOODING"] == 0.0) & (r6["d_HEFFbyATMonOCN_open"] > 1e-5)
          & (np.abs(r6["d_HEFFbyATMonOCN_cover"]) > 1e-6)
          & (np.abs(r6["d_HEFFbyATMonOCN_cover"] + r6["d_HEFFbyATMonOCN_open"] + r6["d_HEFFbyOCNonICE"]) > 1e-6))
    pts = np.argwhere(ok)
    assert len(pts) >= n, len(pts)
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), n, replace=False)]


INPUTS = (("ice", "HEFF", 1e-2), ("ice", "HSNOW", 1e-3), ("ice", "AREA", 1e-3), ("ocn", "theta_s", 1e-2),
          ("exf", "atemp", 1e-2), ("exf", "aqh", 1e-5), ("exf", "lwdown", 1e-1), ("exf", "swdown", 1e-1),
          ("exf", "wspeed", 1e-2), ("ice", "TICES", 1e-2))


def test_gradient_vs_fd():
    p, _ = params()
    c = case(1)
    L = c["g"].layout
    pts = _select_points(c, p)
    tt, jj, ii = pts[:, 0], pts[:, 1] + L.OLy, pts[:, 2] + L.OLx
    rng = np.random.default_rng(1)
    w = {k: rng.standard_normal(len(pts)) for k in ("HEFF", "HSNOW", "AREA", "TICES", "Qnet", "EmPmR", "saltFlux")}
    scale = dict(HEFF=1.0, HSNOW=1.0, AREA=1.0, TICES=1.0 / 273.0, Qnet=1e-2, EmPmR=1e4, saltFlux=1e2)

    def outputs(x):
        ice_o, flx_o, _ = sg.seaice_growth(p, c["g"], c["HEFFM"], x["ice"], x["ocn"], x["exf"], c["flx"])
        o = dict(HEFF=ice_o["HEFF"], HSNOW=ice_o["HSNOW"], AREA=ice_o["AREA"], TICES=ice_o["TICES"][:, 0],
                 Qnet=flx_o["Qnet"], EmPmR=flx_o["EmPmR"], saltFlux=flx_o["saltFlux"])
        return {k: v[tt, jj, ii] for k, v in o.items()}

    def J(x):
        o = outputs(x)
        return sum(jnp.sum(w[k] * scale[k] * o[k]) for k in o)

    x0 = {grp: {k: jnp.asarray(v) for k, v in c[grp].items()} for grp in ("ice", "ocn", "exf")}
    grad = jax.jit(jax.grad(J))(x0)
    for grp in grad:
        for k, v in grad[grp].items():
            assert np.all(np.isfinite(np.asarray(v))), (grp, k)
    outj = jax.jit(outputs)
    base = outj(x0)
    worst = 0.0
    for grp, k, h0 in INPUTS:
        v = np.zeros(np.shape(c[grp][k]))
        sign = rng.choice([-1.0, 1.0], len(pts))
        if k == "TICES":
            v[tt, 0, jj, ii] = sign
        else:
            v[tt, jj, ii] = sign
        ad = float(np.sum(np.asarray(grad[grp][k]) * v))
        errs = []
        for h in (h0, h0 * 1e-1, h0 * 1e-2, h0 * 1e-3):
            def xs(sgn):
                x = {g2: dict(d2) for g2, d2 in x0.items()}
                x[grp][k] = x0[grp][k] + sgn * h * v
                return outj(x)
            op, om = xs(1.0), xs(-1.0)
            # subtract outputs pointwise before weighting (a large J loses digits to cancellation)
            fd = sum(float(np.sum(w[q] * scale[q] * (np.asarray(op[q]) - np.asarray(om[q])))) for q in base) / (2 * h)
            errs.append(abs(fd - ad) / max(abs(ad), 1e-2))  # abs. floor: d/dTICES is ~0 after 10 Newton steps
        e = min(errs)
        print(f"d J / d {k:8s}: AD {ad: .6e}  rel err vs FD (h-sweep {h0:g}..{h0 * 1e-3:g}) "
              + " ".join(f"{x:.1e}" for x in errs))
        worst = max(worst, e)
        assert e < 1e-6, (k, errs)
    print("worst plateau rel err", worst)
