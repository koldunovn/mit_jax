"""Full-V4r4 FORWARD_STEP gate (plan M2.6b-2, tier1x): bulk-formula EXF + SEAICE_MODEL + the ocean, composed in
core/forward_step.forward_step (its full-tree branch), against the full-tree oracle oracle.FULL = full_jaxdump_v5
(iterations 1-3, every stage of reference/jaxdump/SUBSTEPS.md) and the Fortran's end-of-run pickups (iteration 4).

Set-up: the production path, tests/test_init_full.production() (cached for the session): model.setup from the run
directory (grid from the files, ctrl-adjusted mixing, ExfFullParams, SeaiceParams, fixed sea-ice fields and zenith
factors in the Grid) and init.state_from_pickup. EXF inputs from ExfFullRecordLoader + zenith_time, driven from the
model start. conftest XLA flags (no FMA, no algsimp). Every comparison is over every point, halos included.

Gates:
  (a) one FORWARD_STEP (record=True) from the Fortran state at the start of iteration 1 (S00_begin + the G00 r* fields,
      the 21 'b' EXF_FIELDS of S00i_begin_ice_exf, the sea-ice state of S00i_begin_ice_exf, DYN_CARRY =
      seaice_model.dyn_carry_init): EVERY dumped field of every stage in Fortran order -- S01, X01 (incl. the 'e'
      record buffers), X03, X04, X05, X06, X07, X08, S02, S03, I00, Y01-Y06, L01/L02/L04 of both Picard passes, I01-I04,
      P00-P06, S04, D00a, D00b, D01, D02, S05, S06-S11, C01, C02, S12, T01, T10-T13, T02, T20-T23, T03, S13, S14 --
      against the JAX value at that point (the step's stage records applied in order to a running view of the fields),
      then the end state vs S00_begin / S00i_begin_ice_exf / G00 of iteration 2 and DYN_CARRY vs I01 of iteration 1.
      Stages not compared (their values are routine locals the composed step does not return; each is replay-gated in
      its kernel test): X02 (test_exf_full), X05a/X05b (bulk-formula locals, test_exf_full), L03 (first LSOR sweep,
      test_seaice_dyn), A01-A06 (advdiff locals, test_seaice_advdiff), H01-H06 (growth locals, test_seaice_growth).
      Dumped fields a compared stage holds but the step never computes at that point (previous-step values of arrays
      overwritten before they are read): UNCOMPARED below, asserted exactly.
      Allowed differences (read by nothing): zen_fsol_daily (XLA arccos vs glibc: <= 2.1e-16 relative, test_exf_full),
      uice_fd/vice_fd (glibc sincos: <= 1 ulp at < 200 points, test_seaice_dyn).
      LSOR sweeps per Picard pass (ICOUNT1/2) and S1, S2, WFAU, WFAV equal; cg2d iterations 165 as STDOUT.0000.
  (b) free run 1 -> 2 -> 3 -> 4 from that state (record=False, the production program): the whole State bitwise vs
      iterations 2 and 3 (as in (a)), cg2d 165/162/158 iterations; after iteration 3 vs the Fortran's pickup.ckptA,
      pickup_seaice.ckptA, pickup_ggl90.ckptA (timeStepNumber 4, float64 interiors, read with the literal pickup readers):
      uVel, vVel, theta, salt, AB histories (slots m1/m2 of myIter = 4), etaN, dEtaHdt, EtaH (= etaHnm1, write_pickup.F:
      319), the six sea-ice fields, GGL90TKE.
  (c) step 1 from init.state_from_pickup (grid from the files, ctrl adjustments, pickups, full EXF_INIT_VARIA,
      SEAICE_INIT_VARIA incl. DYN_CARRY; no dump anywhere) bitwise vs iteration 2.
Measured 2026-09-23 (16 CPU cores): (a) 820 (stage, field) pairs, 0 differing values (zen_fsol_daily within 2.1e-16,
uice_fd/vice_fd within 1 ulp); (b) and (c) 0 differing values in every State field and all 18 pickup fields; step
5-6 s after a ~45 s compile (record=True: ~50 s compile). The same program run 24 steps from the pickup
(scripts/run_jax.py) is bitwise the Fortran 1-day run's end-of-run pickups (M2.6b-2 report).
Negative controls (each makes the end-of-step-1 comparison fail; State fields differing): SEAICE_MODEL skipped (54);
EXTERNAL_FORCING_SURF fed the pre-sea-ice fluxes, i.e. sea ice after the surface forcing instead of before (27: the
ocean, not the sea-ice state); EXF_BULKFORMULAE with one stability iteration instead of niter_bulk = 2 (37).
"""

import dataclasses
import functools
import re
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax import init
from mitgcm_jax.adjoint.modes import EXACT
from mitgcm_jax.core import external_forcing as ef
from mitgcm_jax.core import forward_step as fs_mod
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_full as X
from mitgcm_jax.pkgs import seaice_init as si
from mitgcm_jax.pkgs import seaice_model as sm
from mitgcm_jax.state import S00I_EXF_FIELDS, State, state_from_dump, state_from_dump_full
from mitgcm_jax.tests import oracle
from mitgcm_jax.tests.test_init_full import production
from mitgcm_jax.tests.test_seaice_dyn import COUNTS as LSOR_COUNTS
from mitgcm_jax.tests.test_seaice_dyn import INT, L02_INTERIOR, R0

ORACLE = oracle.FULL
ITS = (1, 2, 3)
CG2D_ITERS = {1: 165, 2: 162, 3: 158}          # STDOUT.0000 of full_jaxdump_v5: cg2d_iters(min,last)
ICE_STAGE = "S00i_begin_ice_exf"
# EXF_FIELDS arrays dumped in the 'b' group (S00i_begin_ice_exf); the other 7 ('x' group) are in S00_begin
EXF_B = ("uwind", "vwind", "wspeed", "wStress", "cw", "sw", "sh", "atemp", "aqh", "hs", "hl", "lwflux", "evap", "precip",
         "snowprecip", "swdown", "lwdown", "zen_albedo", "zen_fsol_diurnal", "zen_fsol_daily", "runoff")
# allowed differences: field -> (max relative difference, max number of differing points)
ULP = {"zen_fsol_daily": (2.1e-16, 10 ** 6), "uice_fd": (1e-15, 200), "vice_fd": (1e-15, 200)}
SKIPPED_STAGES = ("S00_begin", "G00_geometry", "G01_seaice_geometry", ICE_STAGE, "X02_exf_zenithangle",
                  "X05a_bulk_init", "X05b_bulk_iter", "L03_lsor_sweep1_p1", "L03_lsor_sweep1_p2", "A01_heff_adv",
                  "A02_heff_diff", "A03_area_adv", "A04_area_diff", "A05_snow_adv", "A06_snow_diff",
                  "H01_growth_pre_budget", "H02_growth_budget_ocean", "H03_growth_pre_solve4temp",
                  "H04_growth_solve4temp", "H05_growth_heat_stocks", "H06_growth_ocean_forcing")
# dumped fields of compared stages that hold a PREVIOUS step's value there (overwritten before they are read, so the
# step does not carry them): SEAICE_DYNSOLVER's PRESS0/ZMAX/ZMIN (set by CALC_ICE_STRENGTH at Y02) and uice_fd/vice_fd
# (set by FREEDRIFT at Y03) seen at Y01; the growth diagnostics saltWtrIce/frWtrIce (never written in V4r4, 0) and
# the 'v' group arrays of the LSR (ETA .. PRESS, deltaC, uIceNm1/vIceNm1) seen before SEAICE_LSR writes them
# (Y01-Y04); filled in by the test run below
UNCOMPARED = {
    "Y01_get_dynforcing": {"PRESS0", "ZMAX", "ZMIN", "uice_fd", "vice_fd"},
    "Y02_ice_strength": {"uice_fd", "vice_fd"},
    "Y04_before_lsr": {"ETA", "etaZ", "ZETA", "zetaZ", "PRESS", "deltaC", "uIceNm1", "vIceNm1"},
    "I03_reg_ridge": {"saltWtrIce", "frWtrIce"},
    "I04_growth": {"saltWtrIce", "frWtrIce"},
}


def F(it, stage, name):
    return oracle.field(env().ds, it, stage, name)


@functools.lru_cache(maxsize=1)
def env():
    pr = production()
    P, g, ex, kLowC, rundir = pr["P"], pr["g"], pr["ex"], pr["kLowC"], pr["rundir"]
    nml = RunNamelists(rundir)
    loader = X.ExfFullRecordLoader(P.exfb, g, rundir)
    exf_in = {}
    for n, it in enumerate(ITS, 1):                       # from the model start, in order (first = .TRUE. at it 1)
        myTime, myIter = X.model_time(nml, n)
        assert myIter == it
        bufs, facs, _ = loader.load(myTime, myIter)
        exf_in[it] = dict(bufs=bufs, facs=facs, myTime=myTime, zt=X.zenith_time(P.exfb, myTime, myIter))
    return SimpleNamespace(ds=pr["ds"], P=P, g=g, ex=ex, kLowC=kLowC, rundir=rundir, nml=nml, exf_in=exf_in,
                           st_pickup=pr["st"], steps={}, cache={})


def step_fn(adj=EXACT, record=False):
    """The jitted FORWARD_STEP (model arguments passed, not closed over), one per (adj, record)."""
    e = env()
    key = (adj, record)
    if key not in e.steps:
        e.steps[key] = jax.jit(lambda P, g, kLowC, st, x: fs_mod.forward_step(P, g, e.ex, kLowC, st, x, adj=adj,
                                                                              record=record))
    return e.steps[key]


def as_state(f, it):
    """A jnp State; PmEpR (written by EXTERNAL_FORCING_SURF and INTEGR_CONTINUITY before any read in every step, and
    returned by forward_step) is added as 0 when absent so that every step runs the same compiled program; it = int64
    array (the type of the step's own myIter + 1); arrays strongly typed (jnp.full in the initialisation gives weakly
    typed float64, which would compile a second program)."""
    f = dict(f)
    f.setdefault("PmEpR", np.zeros(env().g.layout.shape2d))
    return State({k: jnp.asarray(v, dtype=v.dtype) for k, v in f.items()}, jnp.asarray(it, jnp.int64))


def dumped_state(it):
    """The Fortran state at the start of iteration `it` (state.state_from_dump_full): S00_begin + G00 group R, the 'b'
    EXF_FIELDS and the sea-ice state of S00i_begin_ice_exf, DYN_CARRY (dyn_carry_init at the model start, it = 1;
    else I01_dynsolver of it-1)."""
    ds, L = env().ds, env().g.layout
    assert tuple(EXF_B) == tuple(S00I_EXF_FIELDS)
    st = state_from_dump_full(ds, it, L)
    if it > 1:
        assert (it - 1, "I01_dynsolver", "e11") in ds.index
    return as_state(st.f, it)


def ndiff(a, b, region=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if region is not None:
        a, b = a[(slice(None),) + region], b[(slice(None),) + region]
    return int(np.sum(a != b))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.abs(b).max()
    return float(np.abs(a - b).max() / (m if m > 0 else 1.0))


def check(name, a, ref, region=None):
    """(number of differing values, max rel) or None when within the ULP allowance of `name`."""
    n = ndiff(a, ref, region)
    if n and name in ULP:
        r = rel(a, ref)
        if r <= ULP[name][0] and n <= ULP[name][1]:
            return None
    return (n, rel(a, ref)) if n else None


def end_state_mismatches(st1, it):
    """st1 (JAX State at the start of iteration `it`) vs the Fortran: S00_begin + G00 group R (state_from_dump), the
    EXF 'b' group and sea-ice state of S00i_begin_ice_exf, DYN_CARRY vs I01 of it-1. Returns {field: (n, rel)}; also
    checks that every State field was compared (PmEpR aside: not dumped, never read before written)."""
    ds = env().ds
    ref = state_from_dump(ds, it, env().g.layout).f
    ref.update({k: F(it, ICE_STAGE, k) for k in EXF_B + sm.ICE_STATE})
    ref.update({k: F(it - 1, "I01_dynsolver", k) for k in sm.DYN_CARRY})
    missing = set(st1.f) - set(ref) - {"PmEpR"}
    assert not missing, sorted(missing)
    bad = {k: check(k, st1.f[k], v) for k, v in ref.items()}
    return {k: v for k, v in bad.items() if v}


# ---------------------------------------------------------------------------------------------------------------------
# gate (a): every stage of step 1


def stage_records(aux, x):
    """[(dump stage, dict of JAX values)] in Fortran order: what each stage wrote, as the step recorded it. Names
    starting with '_' are updates of the running view only (no dump stage). x: the step's EXF inputs."""
    sea, op, d, t = aux["seaice"], aux["S04_oceanic_phys"], aux["S05_dynamics"], aux["S13_thermodynamics"]
    dyn = sea["dyn"]
    recs = [("S01_update_rstar_F", aux["S01_update_rstar_F"]),
            ("X01_exf_getffields", dict(aux["X01_exf_getffields"],
                                        **{f"{n}{k}": x["bufs"][n][k] for n in x["bufs"] for k in (0, 1)}))]
    recs += [(s, aux[s]) for s in ("X03_exf_radiation", "X04_exf_wind", "X05_exf_bulkformulae", "X06_exf_hflux_sflux",
                                   "X07_exf_getsurfacefluxes", "X08_exf_mapfields", "S02_load_fields",
                                   "S03_ctrl_map_forcing", "_pre_seaice")]
    # SEAICE_MODEL; stressDivergenceX/Y are zeroed at the start of SEAICE_DYNSOLVER (seaice_dynsolver.F:176-191)
    recs += [("I00_seaice_begin", sea["I00"]),
             ("Y01_get_dynforcing", dict(dyn["Y01"], stressDivergenceX=sea["I01"]["stressDivergenceX"],
                                         stressDivergenceY=sea["I01"]["stressDivergenceY"])),
             ("Y02_ice_strength", dyn["Y02"]), ("Y03_freedrift", dyn["Y03"]), ("Y04_before_lsr", {})]
    for p, pr in enumerate(dyn["passes"], 1):
        recs += [(f"L01_lsr_visc_drag_p{p}", pr["L01"]), (f"L02_lsr_coeffs_p{p}", pr["L02"]),
                 (f"L04_lsor_end_p{p}", pr["L04"])]
    vgrp = ("e11", "e22", "e12", "deltaC", "ETA", "etaZ", "ZETA", "zetaZ", "PRESS", "DWATN", "FORCEX", "FORCEY",
            "uIceNm1", "vIceNm1")
    recs += [("Y05_lsr", dict(dyn["Y05"], **{k: sea["I01"][k] for k in vgrp})),   # nothing writes them Y05 -> I01
             ("Y06_ocean_stress", dyn["Y06"]), ("I01_dynsolver", sea["I01"]), ("I02_advdiff", sea["I02"]),
             ("I03_reg_ridge", sea["I03"]), ("I04_growth", sea["I04"]), ("P00_seaice_model", sea["P00"]),
             ("P01_external_forcing_surf", aux["P01_external_forcing_surf"]),
             ("P02_rho_sigma_ivdc_mxlayer", dict(aux["P02"], **{k: op[k] for k in ("rhoInSitu", "IVDConvCount",
                                                                                 "hMixLayer")})),
             ("P03_salt_plume_depth", {"saltPlumeDepth": op["saltPlumeDepth"]}),
             ("P04_ggl90", {k: op[k] for k in ("GGL90TKE", "GGL90viscArU", "GGL90viscArV", "GGL90diffKr")}),
             ("P05_gmredi_tensor", aux["P05_gmredi_tensor"]),
             ("P06_gmredi_exch", {k: op[k] for k in ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX",
                                                     "GM_PsiY")}),
             ("S04_oceanic_phys", op),
             ("D00a_phi_hyd", {k: d[k] for k in ("phiHydC", "phiHydF", "dPhiHydX", "dPhiHydY")}),
             ("D00b_mom_vecinv", aux["D00b_mom_vecinv"]),
             ("D01_before_impl_visc", dict(gU=d["gU_explicit"], gV=d["gV_explicit"], kappaRU=d["kappaRU"],
                                           kappaRV=d["kappaRV"], guNm_1=d["guNm"][0], guNm_2=d["guNm"][1],
                                           gvNm_1=d["gvNm"][0], gvNm_2=d["gvNm"][1])),
             ("D02_after_impl_visc", dict(gU=d["gU"], gV=d["gV"])),
             ("S05_dynamics", dict(gU=d["gU"], gV=d["gV"], totPhiHyd=d["totPhiHyd"], phiHydLow=d["phiHydLow"]))]
    recs += [(s, aux[s]) for s in ("S06_update_rstar_T", "S07_update_cg2d", "C01_cg2d_inputs", "C02_cg2d_solution",
                                   "S08_solve_for_pressure", "S09_momentum_correction", "S10_integr_continuity",
                                   "S11_calc_rstar", "S12_stagger_exchanges", "T01_residual_flow", "T10_temp_adv",
                                   "T11_temp_gT", "T12_temp_step", "T13_temp_impl", "T02_temp_integrate",
                                   "T20_salt_adv", "T21_salt_gS", "T22_salt_step", "T23_salt_impl",
                                   "T03_salt_integrate")]
    recs += [("S13_thermodynamics", {"theta": t["theta"], "salt": t["salt"]}),
             ("S14_tracers_correction", aux["S14_tracers_correction"])]
    return recs


def dumped_fields(ds, it, stage):
    """Field names dumped at (it, stage); per-level K: records (<name>_kNNN) collapsed to <name>."""
    names = set()
    for (i, s, n) in ds.index:
        if i == it and s == stage:
            m = re.fullmatch(r"(.+)_k\d{3}", n)
            names.add(m.group(1) if m and (i, s, n[:-5]) not in ds.index else n)
    return names


def region_of(stage, name):
    """Points the Fortran computes for the LSR locals (test_seaice_dyn: etaPlusZeta/zetaMinusEta J,I = 0..sN; the
    tridiagonal coefficients and right-hand sides the interior); None = every point."""
    if stage.startswith("L02_"):
        if name in ("etaPlusZeta", "zetaMinusEta"):
            return R0
        if name in L02_INTERIOR:
            return INT
    return None


def compare_stages(it, st0, aux, x):
    """Apply the stage records in order to a running view (start: the step's input State + the static grid fields the
    dumps include) and compare every dumped field of every compared stage. Returns (bad, compared, uncompared,
    stages seen)."""
    e = env()
    ds = e.ds
    view = dict(st0.f)
    view.update(diffKr=e.g.diffKr, **{k: e.g.f[k] for k in si.ICE_FIXED})
    bad, compared, uncompared = {}, 0, {}
    order = []
    for (i, s, _) in ds.order:
        if i == it and s not in order:
            order.append(s)
    recs = stage_records(aux, x)
    rec_stages = [s for s, _ in recs if not s.startswith("_")]
    expected = [s for s in order if s not in SKIPPED_STAGES]
    assert rec_stages == expected, (rec_stages, expected)
    for stage, r in recs:
        view.update(r)
        if stage.startswith("_"):
            continue
        for name in sorted(dumped_fields(ds, it, stage)):
            if name not in view:
                uncompared.setdefault(stage, set()).add(name)
                continue
            ref = F(it, stage, name)
            a = view[name]
            if np.ndim(a) == 0:                            # N: scalars (LSOR counts, residuals, relaxation)
                n = int(float(a) != float(ref[0, 0, 0]))
                res = (n, abs(float(a) - float(ref[0, 0, 0]))) if n else None
            else:
                assert np.shape(a) == ref.shape, (stage, name, np.shape(a), ref.shape)
                res = check(name, a, ref, region_of(stage, name))
            compared += 1
            if res:
                bad[f"{stage}/{name}"] = res
    return bad, compared, uncompared, order


def test_step1_every_stage_bitwise():
    """Gate (a): step 1 from the dumped state, every dumped field of every stage (module docstring); end state."""
    e = env()
    st0 = dumped_state(1)
    t = time.time()
    st1, aux = step_fn(record=True)(e.P, e.g, e.kLowC, st0, e.exf_in[1])
    jax.block_until_ready(st1.f["theta"])
    print(f"\nrecord step (incl. compile): {time.time() - t:.1f} s")
    bad, n, unc, order = compare_stages(1, st0, aux, e.exf_in[1])
    print(f"compared {n} (stage, field) pairs over {len(order) - len(SKIPPED_STAGES)} stages; uncompared {unc}")
    assert not bad, bad
    assert unc == UNCOMPARED, unc
    assert int(aux["cg2d"]["numIters"]) == CG2D_ITERS[1]
    assert tuple(int(p["L04"]["ICOUNT1"]) for p in aux["seaice"]["dyn"]["passes"]) == LSOR_COUNTS[1]
    bad = end_state_mismatches(st1, 2)
    assert not bad, bad
    # not vacuous: sea ice moves, grows and melts; the ice changes the ocean forcing; the plume flux is set
    sea = aux["seaice"]
    assert np.abs(np.asarray(sea["P00"]["UICE"])).max() > 0.05
    assert np.abs(np.asarray(sea["P00"]["HEFF"]) - np.asarray(st0.f["HEFF"])).max() > 1e-4
    assert ndiff(sea["P00"]["Qnet"], aux["S03_ctrl_map_forcing"]["Qnet"]) > 10000
    assert np.abs(np.asarray(sea["P00"]["saltPlumeFlux"])).max() > 0


def _free_run(st, its, adj=EXACT):
    out, e = [], env()
    for it in its:
        t = time.time()
        st, aux = step_fn(adj)(e.P, e.g, e.kLowC, st, e.exf_in[it])
        jax.block_until_ready(st.f["theta"])
        out.append((st, int(aux["cg2d"]["numIters"]), time.time() - t))
    return out


def pickup_mismatches(st):
    """The State at the start of iteration 4 vs the Fortran's pickup.ckptA / pickup_seaice.ckptA / pickup_ggl90.ckptA
    (written at the end of the run, timeStepNumber 4; float64 interiors), read with the literal readers of init.py and
    pkgs/seaice_init.py. AB slots of myIter = 4 (write_pickup.F:139-140): GuNm1 -> slot m1 = 2, GuNm2 -> m2 = 1;
    EtaH is etaHnm1 (write_pickup.F:319)."""
    e = env()
    L = e.g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    cfg = init.InitConfig.from_namelists(e.nml)
    cfg4 = dataclasses.replace(cfg, nIter0=4, pickup="pickup.ckptA", pickup_ggl90="pickup_ggl90.ckptA",
                                    m1=1 + (4 + 1) % 2, m2=1 + 4 % 2)
    pk, info = init.read_pickup(e.rundir, cfg4, L)
    assert info.missing == ()
    pk["etaHnm1"] = pk.pop("etaH")
    pk["GGL90TKE"] = init.read_ggl90_pickup(e.rundir, cfg4, L)
    icfg = dataclasses.replace(si.SeaiceInitConfig.from_namelists(e.nml), pickup_seaice="pickup_seaice.ckptA")
    ice = si.read_seaice_pickup(e.rundir, icfg)
    pk.update({k: v for k, v in ice.items() if k != "TICES1"})
    pk["TICES"] = ice["TICES1"]
    bad = {}
    for k, ref in pk.items():
        a = np.asarray(st.f[k])[..., J, I]
        if k == "TICES":
            a = a[:, 0]
        n = ndiff(a, ref)
        if n:
            bad[k] = (n, rel(a, ref))
    return bad, sorted(pk)


def test_free_run_to_iteration4_bitwise():
    """Gate (b): free run 1 -> 4 from the dumped state (the production program, record=False): the State at the
    start of iterations 2 and 3 bitwise, cg2d 165/162/158; the end of iteration 3 bitwise vs the Fortran's pickups."""
    runs = _free_run(dumped_state(1), ITS)
    env().cache["free"] = runs
    for (st, n, dt), it in zip(runs, ITS):
        print(f"it {it} -> {it + 1}: cg2d {n}, {dt:.1f} s")
        assert n == CG2D_ITERS[it], (it, n)
        if it + 1 in ITS:
            bad = end_state_mismatches(st, it + 1)
            assert not bad, (it + 1, bad)
    bad, fields = pickup_mismatches(runs[-1][0])
    print("pickup fields compared:", fields)
    assert len(fields) == 18 and not bad, bad


def test_step1_from_pickup_bitwise():
    """Gate (c): step 1 from init.state_from_pickup (no dump used) bitwise vs the Fortran at iteration 2."""
    e = env()
    st0 = as_state(e.st_pickup.f, 1)
    assert set(st0.f) == set(dumped_state(1).f)
    (st1, n, _), = _free_run(st0, (1,))
    assert n == CG2D_ITERS[1]
    bad = end_state_mismatches(st1, 2)
    assert not bad, bad


# ---------------------------------------------------------------------------------------------------------------------
# negative controls


def _planted_step(monkeypatch=None):
    """Step 1 from the dumped state with a fresh jit (jax caches a trace per function object: re-jitting the same
    callable after a monkeypatch would replay the old trace)."""
    e = env()
    f = jax.jit(lambda P, g, kLowC, st, x: fs_mod.forward_step(P, g, e.ex, kLowC, st, x))
    st1, _ = f(e.P, e.g, e.kLowC, dumped_state(1), e.exf_in[1])
    return end_state_mismatches(st1, 2)


def test_negative_controls(monkeypatch):
    """Each planted error makes the end-of-step-1 gate fail: SEAICE_MODEL skipped (sea-ice state and ocean fluxes
    wrong), the surface forcing computed from the pre-sea-ice fluxes (sea ice after EXTERNAL_FORCING_SURF instead of
    before: the sea-ice state is right, the ocean is not), one bulk-formula stability iteration instead of niter_bulk
    = 2 (EXF fields and everything downstream)."""
    # (1) SEAICE_MODEL skipped: its INOUT fields pass through unchanged
    monkeypatch.setattr(sm, "seaice_model", lambda P, g, sg, ex, ins, ad="ecco", expf=None, record=False:
                        ({k: ins[k] for k in sm.INOUT}, {}))
    bad = _planted_step()
    print("\nSEAICE_MODEL skipped ->", len(bad), "fields differ")
    assert {"HEFF", "AREA", "UICE", "Qnet", "theta", "salt"} <= set(bad), sorted(bad)
    monkeypatch.undo()
    # (2) EXTERNAL_FORCING_SURF on the fluxes SEAICE_MODEL received (as if sea ice ran after it)
    stash = {}
    real_si, real_post = sm.seaice_model, ef.oceanic_phys_post_seaice

    def si_stash(P, g, sg, ex, ins, ad="ecco", expf=None, record=False):
        stash.update({k: ins[k] for k in sm.FF_INOUT})
        return real_si(P, g, sg, ex, ins, ad=ad, expf=expf, record=record)

    def post_pre_ice(p, g, ex, ff, theta, salt):
        ff_out, _ = real_post(p, g, ex, ff, theta, salt)
        _, sfo = real_post(p, g, ex, dict(ff, **stash), theta, salt)
        return ff_out, sfo

    monkeypatch.setattr(sm, "seaice_model", si_stash)
    monkeypatch.setattr(ef, "oceanic_phys_post_seaice", post_pre_ice)
    bad = _planted_step()
    print("surface forcing before sea ice ->", len(bad), "fields differ")
    assert {"theta", "salt", "surfaceForcingT", "surfaceForcingU"} <= set(bad), sorted(bad)
    assert not {"HEFF", "AREA", "UICE", "Qnet"} & set(bad), sorted(bad)
    monkeypatch.undo()
    # (3) one stability iteration in EXF_BULKFORMULAE (EXF_CONSTANTS.h:91 niter_bulk = 2)
    monkeypatch.setattr(X, "NITER_BULK", 1)
    bad = _planted_step()
    print("niter_bulk = 1 ->", len(bad), "fields differ")
    assert {"hs", "hl", "evap", "theta", "HEFF"} <= set(bad), sorted(bad)
