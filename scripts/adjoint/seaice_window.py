#!/usr/bin/env python3
"""Multi-step adjoint tests of the sea-ice model alone (plan M2, before ocean coupling): a SEA-ICE-ONLY WINDOW.

HARNESS (a test configuration, not a model deviation: the model code is pkgs/seaice_model.seaice_model unchanged)
  The SEAICE_MODEL driver is stepped N times on its own carried state
      CARRY = ICE_STATE (AREA, HEFF, HSNOW, TICES[7], UICE, VICE) + DYN_CARRY (seaiceMassC/U/V, FORCEX0/Y0, e11, e22,
              e12, DWATN, FORCEX/Y) + sIceLoad
  i.e. exactly what pkgs/seaice_model.py says the model must carry for sea ice (SEAICE_CARRIED + sIceLoad; the gate
  test_seaice_model.test_carried_state_entry_values shows these are the step-to-step entry values). Everything else
  SEAICE_MODEL reads is PRESCRIBED, per step, from the full-V4r4 oracle dumps (oracle.FULL, run dir
  reference/runs/ref_full_serial13_jaxdump_v5_3steps, 1992-01-01 00:00, dt = 3600 s):
      fu, fv, Qnet, Qsw, EmPmR, saltFlux     after CTRL_MAP_FORCING (stage S03): the EXF/CTRL values SEAICE_MODEL
                                             receives before it overwrites them
      saltPlumeFlux = 0                      do_oceanic_phys.F:293 zeroes it before SEAICE_MODEL (full tree)
      uwind, vwind                           as EXF left them (X06; SEAICE_MODEL exchanges them itself)
      wspeed, atemp, aqh, lwdown, swdown, evap, precip, snowprecip, runoff   EXF_FIELDS at I00
      uVel_s, vVel_s (S01), theta_s, salt_s (S00)   the ocean surface level at the start of the step
  schedule "fixed" (default): the iteration-1 inputs at every step (stationary forcing, no artificial jumps);
  schedule "cycle": iterations 1, 2, 3, 1, 2, 3, ... With "cycle" the first three harness steps ARE the Fortran's
  first three SEAICE_MODEL calls: their carried state is bitwise the oracle's P00 at iterations 1-3 (the harness
  gate, test_seaice_adjoint_window.py). The SEAICE_MODEL outputs that would go to the ocean (fu, fv, Qnet, Qsw,
  EmPmR, saltFlux, saltPlumeFlux, uwind, vwind) are discarded: the ocean does not respond to the ice. Why this is a
  valid harness: the reverse pass of a coupled step passes the ocean's cotangent into these same inputs; holding them
  fixed isolates what the tests are about (the sea-ice package's own multi-step linearisation: TL/adjoint
  consistency, FD agreement, cotangent growth, the three adjoint levels) with the step function identical to the one
  the coupled model calls. What it cannot show: ice-ocean feedbacks (theta_s does not cool under growing ice, the
  ocean stress does not change the currents) and EXF's own dependence on the controls (atemp enters here only through
  SEAICE_GROWTH, i.e. the ice-covered fraction; the open-ocean Qnet is prescribed).
  Initial CARRY: ICE_STATE of I00 at iteration 1 (the pickup), DYN_CARRY = seaice_model.dyn_carry_init (as at
  iteration 1 in the Fortran), sIceLoad of S03 at iteration 1.

CONTROLS (dict; all zero = the base window, value-identical):
  HEFF, AREA, UICE, VICE  additive on the initial state: interior points, then the exchange (EXCH_XY_RL,
                          EXCH_UV_XY_RL with signs), as an xx_* initial-condition control would enter
  atemp, fu, fv           time-constant additive shift of the prescribed input at every step (interior, exchanged:
                          EXCH_XY_RL, EXCH_UV_XY_RS with signs as EXF_MAPFIELDS does)
COSTS (after the window; interior points x HEFFM; rA from the grid files):
  J1 = sum HEFF rA over yC > 70N  [km^3]        (Arctic ice volume)
  J2 = sum HEFF rA over yC < 60S  [km^3]        (Southern Ocean ice volume)
  J3 = sum AREA rA over yC > 70N  [1e6 km^2]    (Arctic ice area)
FD DIRECTIONS (relative: d = the base value of the control field on its mask, so x + h d = (1 + h) x and h is h/|x|):
  HEFF (HEFF0), AREA (AREA0), atemp (atemp0, wet points), tau (fu0, fv0 of iteration 1), uvice (UICE0, VICE0).
ADJOINT LEVELS: seaice_model ad = "ecco" | "no_dynamics" | "full" (pkgs/seaice_model.py docstring).

DRIVERS (all take the model pytree M = (P, g, sg, ex) as a jit ARGUMENT, never closed over):
  forward       Python loop over a jitted step (optionally records the switch states per step)
  adjoint       forward keeping every carry on the host, then a reverse loop over one jitted step-VJP (seed
                cotangents looped inside it; per-step remat, memory = one step; per-step per-field cotangent norms)
  tl            Python loop over a jitted step-JVP (tangent-linear model)
  window_scan   lax.scan with jax.checkpoint per step, for jax.grad / jax.jvp of the whole window (cross-check)
TIGHT forward: LSR_ERROR = 1e-12, SEAICElinearIterMax = 20000 (same code path; the "full" derivative is that of the
exactly solved LSR system, the production LSR_ERROR = 2e-4 stops far from it: seaice_lsr.py docstring).

ACTIONS (python scripts/adjoint/seaice_window.py --out DIR --steps N --actions ...; one JSON line per result in
DIR/results.jsonl):
  gate      cycle schedule, 3 steps: carried state == P00 at iterations 1-3 (bitwise)
  time      per-step forward (production and tight LSR) and gradient cost per level
  grad      gradients of --costs (J1,J2,J3) w.r.t. every control, per --levels, at --tol prod|tight, --repeats r;
            saves the fields (grad_<tol>_<level>_<N>s_<J>_r<r>.npz), the directional derivatives along the FD
            directions, and the amplification screen (per-step per-field cotangent norms, terminal seed = the cost)
  fdbase    FD base point for --tol prod|tight: forward (switch states, LSOR counts) and a repeat (noise floor);
            the adjoint values come from the `grad` rows of the same tolerance
  fd        central-FD h-sweep of J1-J3 along --dirs (--hs), --tol prod|tight, switch-flip and LSOR-count changes
            per h (the +-h forwards only: (direction, h) pairs parallelise over processes; compared with fdbase)
  probe     the pass-through fields of each level (one-step reverse self-map == identity)
  dotadj / dottl / tldir: the adjoint half, the tangent half (--amps) of the dot test, TL along --dirs
SUMMARY: python scripts/adjoint/seaice_window.py --summarize DIR [DIR ...]  (markdown tables)
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# gate XLA flags on CPU unless the caller set its own (as conftest.py): IEEE-exact Fortran operation order, so the
# harness gate is bitwise; the adjoint results do not depend on it
if "--xla_cpu_max_isa" not in os.environ.get("XLA_FLAGS", "") and os.environ.get("SEAICE_WINDOW_PROD_FLAGS") != "1":
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

import mitgcm_jax  # noqa: E402,F401  (x64)
from mitgcm_jax.io.dump import DumpSet, read_file  # noqa: E402
from mitgcm_jax.layout import Layout  # noqa: E402
from mitgcm_jax.parallel.exchange import default_exchanger  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.pkgs import seaice_init as si  # noqa: E402
from mitgcm_jax.pkgs import seaice_model as sm  # noqa: E402
from mitgcm_jax.tests import oracle  # noqa: E402

L = Layout()
GRID_DIR = Path("/work/ab0995/a270088/MIT/data/eccov4r4/native_grid_files")  # model.GRID_DIR
ITS = (1, 2, 3)
CARRY = sm.SEAICE_CARRIED + ("sIceLoad",)
PRESCRIBED = tuple(k for k in sm.INPUTS if k not in CARRY)
INIT_CONTROLS = ("HEFF", "AREA", "UICE", "VICE")
SHIFT_CONTROLS = ("atemp", "fu", "fv")
CONTROLS = INIT_CONTROLS + SHIFT_CONTROLS
DIRECTIONS = {"HEFF": ("HEFF",), "AREA": ("AREA",), "atemp": ("atemp",), "tau": ("fu", "fv"),
              "uvice": ("UICE", "VICE")}
COSTS = ("J1", "J2", "J3")
LEVELS = sm.AD_LEVELS
PROGNOSTIC = ("HEFF", "AREA", "HSNOW", "UICE", "VICE")
FF_DUMPED = ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "sIceLoad")
TIGHT = dict(LSR_ERROR=1e-12, SEAICElinearIterMax=20000)
H_SWEEP = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
UICE_CLIP = 0.40  # seaice_dynsolver.F:378


# ---------------------------------------------------------------------------------------------------------------------
# environment: grid, parameters, oracle inputs


def dumpset_parallel(directory):
    """DumpSet with the record headers read in parallel threads (as test_seaice_model._dumpset_parallel)."""
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


class Model(NamedTuple):
    """Everything the step reads besides the carry and the per-step inputs (a pytree: pass it as a jit argument)."""
    P: Any    # seaice_model.SeaiceParams
    g: Any    # seaice_model.seaice_grid(Grid)
    sg: Any   # fixed fields
    ex: Any   # Exchanger


class Env(NamedTuple):
    M: Model           # production LSR tolerance
    M_tight: Model     # LSR_ERROR = 1e-12, SEAICElinearIterMax = 20000
    carry0: dict       # initial CARRY (jnp)
    X3: dict           # prescribed inputs of iterations 1, 2, 3 stacked [3, ...] (jnp)
    W: dict            # cost weights J1, J2, J3 [T, ny, nx]
    dirs: dict         # FD directions {name: {control: array}}
    masks: dict        # wet masks: C (HEFFM, interior), W, S (seaiceMaskU/V, interior)
    yC: Any
    rA: Any
    rundir: Path
    ds: Any            # the oracle DumpSet
    P00: dict          # {it: {field: P00 value}} for the harness gate


def interior_mask():
    m = np.zeros(L.shape2d)
    m[:, L.js(1, L.sNy), L.is_(1, L.sNx)] = 1.0
    return m


def load_env(log=print, rundir=None):
    """Grid (from the files, the production path), SeaiceParams from the full run's namelists, fixed fields, the oracle
    inputs of iterations 1-3, the initial carry, cost weights and FD directions."""
    from mitgcm_jax.grid.load import grid_from_files
    t0 = time.time()
    rundir = Path(rundir or oracle.run_dir(oracle.FULL))
    ds = dumpset_parallel(rundir / "jaxdump")
    nml = RunNamelists(rundir)
    ex = default_exchanger(L)
    gf = grid_from_files(rundir, GRID_DIR, ex, L)
    P = sm.SeaiceParams.from_namelists(nml, gf)
    g = jax.tree.map(jnp.asarray, sm.seaice_grid(gf))
    sg = si.seaice_fixed_fields(gf, ex)
    M = Model(P, g, sg, ex)
    M_tight = Model(P._replace(dyn=dataclasses.replace(P.dyn, **TIGHT)), g, sg, ex)
    F = lambda it, st, k: oracle.field(ds, it, st, k)  # noqa: E731

    def prescribed(it):
        x = {k: F(it, "S03_ctrl_map_forcing", k) for k in FF_DUMPED if k != "sIceLoad"}
        x["saltPlumeFlux"] = np.zeros(L.shape2d)
        x.update({k: F(it, "X06_exf_hflux_sflux", k) for k in sm.EXF_INOUT})
        x.update({k: F(it, "I00_seaice_begin", k) for k in sm.EXF_READ})
        x["uVel_s"] = F(it, "S01_update_rstar_F", "uVel")[:, 0]
        x["vVel_s"] = F(it, "S01_update_rstar_F", "vVel")[:, 0]
        x["theta_s"] = F(it, "S00_begin", "theta")[:, 0]
        x["salt_s"] = F(it, "S00_begin", "salt")[:, 0]
        assert set(x) == set(PRESCRIBED), sorted(set(x) ^ set(PRESCRIBED))
        return x

    xs = [prescribed(it) for it in ITS]
    X3 = {k: jnp.asarray(np.stack([x[k] for x in xs])) for k in PRESCRIBED}
    carry0 = {k: F(1, "I00_seaice_begin", k) for k in sm.ICE_STATE}
    carry0.update({k: np.asarray(v) for k, v in sm.dyn_carry_init(L).items()})
    carry0["sIceLoad"] = F(1, "S03_ctrl_map_forcing", "sIceLoad")
    carry0 = {k: jnp.asarray(carry0[k]) for k in CARRY}
    P00 = {it: {k: F(it, "P00_seaice_model", k) for k in sm.ICE_STATE + ("sIceLoad",)} for it in ITS}
    # costs
    inner = interior_mask()
    HEFFM = np.asarray(sg["HEFFM"])
    yC = np.asarray(gf.f["yC"])
    rA = np.asarray(gf.f["rA"])
    wC = inner * HEFFM
    W = {"J1": jnp.asarray(wC * rA * (yC > 70.0) / 1e9), "J2": jnp.asarray(wC * rA * (yC < -60.0) / 1e9),
         "J3": jnp.asarray(wC * rA * (yC > 70.0) / 1e12)}
    masks = {"C": wC, "W": inner * np.asarray(sg["seaiceMaskU"]), "S": inner * np.asarray(sg["seaiceMaskV"]),
             "wet": inner * np.asarray(gf.f["maskC"])[:, 0]}
    x1 = xs[0]
    c0 = {k: np.asarray(v) for k, v in carry0.items()}
    dirs = {"HEFF": {"HEFF": c0["HEFF"] * masks["C"]},
            "AREA": {"AREA": c0["AREA"] * masks["C"]},
            "atemp": {"atemp": np.asarray(x1["atemp"]) * masks["wet"]},
            "tau": {"fu": np.asarray(x1["fu"]) * masks["W"], "fv": np.asarray(x1["fv"]) * masks["S"]},
            "uvice": {"UICE": c0["UICE"] * masks["W"], "VICE": c0["VICE"] * masks["S"]}}
    log(f"env loaded in {time.time() - t0:.1f} s (rundir {rundir})")
    return Env(M, M_tight, carry0, X3, W, dirs, masks, jnp.asarray(yC), jnp.asarray(rA), rundir, ds, P00)


def zero_controls():
    z = jnp.zeros(L.shape2d)
    return {k: z for k in CONTROLS}


def direction(env, name):
    """Full control pytree of FD direction `name` (zeros on the other controls)."""
    d = zero_controls()
    d.update({k: jnp.asarray(v) for k, v in env.dirs[name].items()})
    return d


def random_controls(env, seed):
    """Random tangent on every control (standard normal on the control's wet interior mask)."""
    rng = np.random.default_rng(seed)
    mk = {"HEFF": "C", "AREA": "C", "atemp": "wet", "UICE": "W", "VICE": "S", "fu": "W", "fv": "S"}
    return {k: jnp.asarray(rng.standard_normal(L.shape2d) * env.masks[mk[k]]) for k in CONTROLS}


def random_carry(env, seed):
    """Random cotangent on the final CARRY: standard normal on the interior wet points of every field (TICES: all 7
    levels; DYN_CARRY included)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in CARRY:
        shp = np.shape(env.carry0[k])
        m = env.masks["W"] if k in ("UICE", "FORCEX", "FORCEX0", "seaiceMassU") else (
            env.masks["S"] if k in ("VICE", "FORCEY", "FORCEY0", "seaiceMassV") else env.masks["C"])
        m = m[:, None] if len(shp) == 4 else m
        out[k] = jnp.asarray(rng.standard_normal(shp) * m)
    return out


def schedule_index(schedule, n):
    if schedule == "fixed":
        return np.zeros(n, np.int32)
    if schedule == "cycle":
        return (np.arange(n) % 3).astype(np.int32)
    raise ValueError(schedule)


# ---------------------------------------------------------------------------------------------------------------------
# the step, the control binding, the costs


def bind(M, carry0, ctl):
    """Controls -> (initial carry, per-step input shifts). Interior values, then the exchange (linear)."""
    ex = M.ex
    inner = jnp.asarray(interior_mask())
    c = dict(carry0)
    c["HEFF"] = carry0["HEFF"] + ex.exch_xy(ctl["HEFF"] * inner)
    c["AREA"] = carry0["AREA"] + ex.exch_xy(ctl["AREA"] * inner)
    du, dv = ex.exch_uv_xy(ctl["UICE"] * inner, ctl["VICE"] * inner, True)
    c["UICE"], c["VICE"] = carry0["UICE"] + du, carry0["VICE"] + dv
    fu, fv = ex.exch_uv_xy(ctl["fu"] * inner, ctl["fv"] * inner, True)
    shift = {"atemp": ex.exch_xy(ctl["atemp"] * inner), "fu": fu, "fv": fv}
    return c, shift


def step_inputs(carry, x, shift):
    ins = dict(x)
    ins.update(carry)
    for k in SHIFT_CONTROLS:
        ins[k] = x[k] + shift[k]
    return ins


def step(M, carry, x, shift, ad, expf=None):
    """One SEAICE_MODEL on the carry with the prescribed inputs x (+ the control shifts); returns the new carry."""
    out, _ = sm.seaice_model(M.P, M.g, M.sg, M.ex, step_inputs(carry, x, shift), ad=ad, expf=expf)
    return {k: out[k] for k in CARRY}


def switch_state(P, sg, rec):
    """Boolean states of the switches that make FD plateaus fragile, interior wet points (int8 [T, ny, nx]):
      clip      |uIce| or |vIce| > 0.40 before the clipping (seaice_dynsolver.F:378-380)
      zeta_max  ZETA = ZMAX (viscous cap of SEAICE_CALC_VISCOSITIES, last Picard pass; ZMAX > 0)
      heff_pos  HEFF > 0 after REG_RIDGE (the ice-covered / ice-free switch of SEAICE_GROWTH, HEFFpreTH > 0)
      hsnow_pos HSNOW > 0 after REG_RIDGE (snow / bare-ice branches of SOLVE4TEMP)
      area_max  AREA = SEAICE_area_max after SEAICE_GROWTH (clipped)
      area_zero AREA = 0 after SEAICE_GROWTH (ice-free clip)
      tsurf_melt TICES(1) = TMELT after SEAICE_GROWTH (surface temperature capped at melting)"""
    m = (sg["HEFFM"] > 0) & jnp.asarray(interior_mask() > 0)
    y5 = rec["dyn"]["Y05"]
    i1, i3, i4 = rec["I01"], rec["I03"], rec["I04"]
    s = {"clip": (jnp.abs(y5["UICE"]) > UICE_CLIP) | (jnp.abs(y5["VICE"]) > UICE_CLIP),
         "zeta_max": (i1["ZMAX"] > 0) & (i1["ZETA"] >= i1["ZMAX"] * sg["HEFFM"]),
         "heff_pos": i3["HEFF"] > 0, "hsnow_pos": i3["HSNOW"] > 0,
         "area_max": i4["AREA"] >= P.growth.area_max, "area_zero": i4["AREA"] <= 0.0,
         "tsurf_melt": i4["TICES"][:, 0] >= P.growth.celsius2K}
    # NEAR a switch (base-trajectory diagnostics only; the near_ keys are left out of the flip counts):
    # ice-covered with exactly zero snow (any perturbation that makes snow flips hsnow_pos), 0 < HSNOW < 1e-10 m,
    # AREA within 1e-6 below the cap, |uIce| or |vIce| within 1e-3 m/s below the clip
    ice = i3["HEFF"] > 0
    spd = jnp.maximum(jnp.abs(y5["UICE"]), jnp.abs(y5["VICE"]))
    s.update(near_hsnow_zero_ice=ice & (i3["HSNOW"] == 0), near_hsnow_tiny=(i3["HSNOW"] > 0) & (i3["HSNOW"] < 1e-10),
             near_area_max=(i4["AREA"] < P.growth.area_max) & (i4["AREA"] > P.growth.area_max - 1e-6),
             near_clip=(spd <= UICE_CLIP) & (spd > UICE_CLIP - 1e-3))
    return {k: (v & m).astype(jnp.int8) for k, v in s.items()}


def step_rec(M, carry, x, shift, expf=None):
    """step (level-independent forward) + the switch states + the LSR sweep counts of both Picard passes."""
    out, rec = sm.seaice_model(M.P, M.g, M.sg, M.ex, step_inputs(carry, x, shift), ad="ecco", expf=expf,
                               record=True)
    counts = jnp.stack([jnp.stack([pr["L04"]["ICOUNT1"], pr["L04"]["ICOUNT2"], pr["L04"]["converged"]])
                        for pr in rec["dyn"]["passes"]])
    return {k: out[k] for k in CARRY}, switch_state(M.P, M.sg, rec), counts


def costs(W, carry):
    return jnp.stack([jnp.sum(W["J1"] * carry["HEFF"]), jnp.sum(W["J2"] * carry["HEFF"]),
                      jnp.sum(W["J3"] * carry["AREA"])])


def seeds_for_costs(env, which=COSTS):
    """Cotangents on the final carry of J1..J3 (stacked on a leading seed axis)."""
    z = {k: jnp.zeros_like(v) for k, v in env.carry0.items()}
    out = []
    for j in which:
        s = dict(z)
        s["AREA" if j == "J3" else "HEFF"] = env.W[j]
        out.append(s)
    return jax.tree.map(lambda *a: jnp.stack(a), *out)


def take(X3, i):
    return {k: v[i] for k, v in X3.items()}


# jitted pieces (M, carry, inputs as arguments)
STEP = jax.jit(step, static_argnames=("ad", "expf"))
STEP_REC = jax.jit(step_rec, static_argnames=("expf",))
BIND = jax.jit(bind)
COSTS_J = jax.jit(costs)


def _per_seed(pull, *cts):
    """pull applied to each seed of seed-batched cotangents (leading axis), stacked again. A static loop, NOT
    jax.vmap: vmap of the sea-ice step's pullback over 3 seeds took 202 s per step on 16 CPU cores against 2.7 s for
    the three pulls looped inside the same jit (no_dynamics level, measured)."""
    S = jax.tree.leaves(cts)[0].shape[0]
    outs = [pull(*jax.tree.map(lambda a: a[i], cts)) for i in range(S)]
    return jax.tree.map(lambda *a: jnp.stack(a), *outs)


@jax.jit
def _bind_vjp(M, carry0, ctl, ct_carry, ct_shift):
    """Pull the (seed-batched) cotangents of the initial carry and of the shifts back to the controls."""
    _, pull = jax.vjp(lambda c: bind(M, carry0, c), ctl)
    return _per_seed(lambda a, b: pull((a, b))[0], ct_carry, ct_shift)


def _step_bwd(M, carry, x, shift, cts, ad, expf=None):
    _, pull = jax.vjp(lambda c, s: step(M, c, x, s, ad, expf), carry, shift)
    return _per_seed(pull, cts)


STEP_BWD = jax.jit(_step_bwd, static_argnames=("ad", "expf"))


def _step_jvp(M, carry, dcarry, x, shift, dshift, ad, expf=None):
    return jax.jvp(lambda c, s: step(M, c, x, s, ad, expf), (carry, shift), (dcarry, dshift))


STEP_JVP = jax.jit(_step_jvp, static_argnames=("ad", "expf"))


@jax.jit
def _bind_jvp(M, carry0, ctl, dctl):
    return jax.jvp(lambda c: bind(M, carry0, c), (ctl,), (dctl,))


# ---------------------------------------------------------------------------------------------------------------------
# drivers


class Fwd(NamedTuple):
    J: np.ndarray            # [3]
    carry: dict              # final carry (jnp)
    carries: list            # per-step carries on the host (keep=True): carries[n] = input of step n
    switches: list           # per-step switch states (record=True): list of {name: int8 array}
    counts: list             # per-step LSOR counts [2 passes, (ICOUNT1, ICOUNT2, converged)]
    seconds: float


def forward(M, env, ctl, n, schedule="fixed", ad="full", keep=False, record=False, expf=None):
    """The window forward (Python loop over a jitted step). The forward values are the same for every level."""
    t0 = time.time()
    idx = schedule_index(schedule, n)
    carry, shift = BIND(M, env.carry0, ctl)
    carries, sw, cnt = [], [], []
    for i in idx:
        if keep:
            carries.append(jax.device_get(carry))
        x = take(env.X3, int(i))
        if record:
            carry, s, c = STEP_REC(M, carry, x, shift, expf=expf)
            sw.append(jax.device_get(s))
            cnt.append(np.asarray(c))
        else:
            carry = STEP(M, carry, x, shift, ad=ad, expf=expf)
    J = np.asarray(COSTS_J(env.W, carry))
    return Fwd(J, carry, carries, sw, cnt, time.time() - t0)


def field_norms(ct):
    """{field: [S] Euclidean norms per seed} of a seed-batched carry cotangent."""
    return {k: np.sqrt(np.sum(np.square(np.asarray(v)).reshape(v.shape[0], -1), axis=1)) for k, v in ct.items()}


class Adj(NamedTuple):
    J: np.ndarray
    grad: dict               # {control: [S, T, ny, nx]} per seed
    ct0: dict                # cotangent of the initial carry [S, ...]
    trace: list              # per step, END OF WINDOW FIRST: {field: [S] norms}; trace[0] = the seed
    forward_seconds: float
    reverse_seconds: float


def adjoint(M, env, ctl, n, seeds, schedule="fixed", ad="full", expf=None, fwd=None):
    """Reverse accumulation over the window: forward keeping every step's carry on the host (or `fwd` from a previous
    forward with keep=True), then one jitted step-VJP per step (seed cotangents looped in the jit), the shift
    cotangents summed over the steps, the initial-carry cotangent pulled back through `bind`. Memory: one step's
    intermediates."""
    if fwd is None:
        fwd = forward(M, env, ctl, n, schedule, ad=ad, keep=True, expf=expf)
    t1 = time.time()
    idx = schedule_index(schedule, n)
    _, shift = BIND(M, env.carry0, ctl)
    ct = seeds
    ct_shift = None
    trace = [field_norms(ct)]
    for i in reversed(range(n)):
        c_ct, s_ct = STEP_BWD(M, jax.device_put(fwd.carries[i]), take(env.X3, int(idx[i])), shift, ct, ad=ad,
                              expf=expf)
        ct = c_ct
        ct_shift = s_ct if ct_shift is None else jax.tree.map(jnp.add, ct_shift, s_ct)
        trace.append(field_norms(ct))
    g = _bind_vjp(M, env.carry0, ctl, ct, ct_shift)
    g = {k: np.asarray(v) for k, v in g.items()}
    return Adj(fwd.J, g, jax.device_get(ct), trace, fwd.seconds, time.time() - t1)


def tl(M, env, ctl, dctl, n, schedule="fixed", ad="full", expf=None):
    """Tangent-linear model over the window: returns (J [3], dJ [3], final dcarry)."""
    idx = schedule_index(schedule, n)
    (carry, shift), (dcarry, dshift) = _bind_jvp(M, env.carry0, ctl, dctl)
    for i in idx:
        carry, dcarry = STEP_JVP(M, carry, dcarry, take(env.X3, int(i)), shift, dshift, ad=ad, expf=expf)
    J, dJ = jax.jvp(lambda c: costs(env.W, c), (carry,), (dcarry,))
    return np.asarray(J), np.asarray(dJ), dcarry


def window_scan(M, W, carry0, X3, idx, ctl, ad, expf=None, remat=True):
    """J(ctl) [3] as ONE function: lax.scan over the steps with jax.checkpoint per step (for jax.grad / jax.jvp of
    the whole window). idx: the per-step iteration index (schedule_index)."""
    c0, shift = bind(M, carry0, ctl)

    def body(c, i):
        return step(M, c, take(X3, i), shift, ad, expf), None

    if remat:
        body = jax.checkpoint(body, prevent_cse=False)
    cN, _ = lax.scan(body, c0, jnp.asarray(idx))
    return costs(W, cN)


# ---------------------------------------------------------------------------------------------------------------------
# statistics


def tree_vdot(a, b):
    return float(sum(np.vdot(np.ravel(np.asarray(a[k])), np.ravel(np.asarray(b[k]))) for k in a))


class Amplification(NamedTuple):
    median: float
    log_spread: float
    worst3: float
    rates: tuple
    passes: bool


def amplification(trace, steps_per_entry=1, bars=(1.010, 0.020, 1.030)):
    """Per-step reverse growth of a cotangent-norm trace (trace[0] = seed at the window end, each further entry one
    step earlier): median, std of log rates, worst 3 consecutive entries; the fesom_jax bars (as
    mitgcm_jax.adjoint.grad.amplification, reimplemented here so the harness does not import the ocean step)."""
    tr = [float(t) for t in trace]
    k = max(1, int(steps_per_entry))
    rates = [(tr[i + 1] / tr[i]) ** (1.0 / k) if tr[i] > 0 else float("nan") for i in range(len(tr) - 1)]
    fin = np.array([r for r in rates if np.isfinite(r) and r > 0])
    if fin.size == 0:
        return Amplification(float("nan"), float("nan"), float("nan"), tuple(rates), False)
    w = max(1, min(3, len(tr) - 1))
    sus = [(tr[i + w] / tr[i]) ** (1.0 / (k * w)) for i in range(len(tr) - w) if tr[i] > 0]
    worst = float(np.max(sus)) if sus else float("nan")
    med, spread = float(np.median(fin)), float(np.std(np.log(fin)))
    return Amplification(med, spread, worst, tuple(rates), bool(med <= bars[0] and spread <= bars[1]
                                                                  and worst <= bars[2]))


def trace_of(trace, fields, seed):
    return [float(np.sqrt(sum(float(t[k][seed]) ** 2 for k in fields))) for t in trace]


# ---------------------------------------------------------------------------------------------------------------------
# experiment actions


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


class Recorder:
    def __init__(self, out, args):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.base = dict(job=os.environ.get("SLURM_JOB_ID"), git=git_head(), host=os.uname().nodename,
                         xla_flags=os.environ.get("XLA_FLAGS", ""), device=jax.devices()[0].platform,
                         steps=args.steps, schedule=args.schedule)

    def row(self, **kw):
        r = dict(self.base, time=time.strftime("%Y-%m-%dT%H:%M:%S"), **kw)
        with open(self.out / "results.jsonl", "a") as f:
            f.write(json.dumps(r, default=_jsonable) + "\n")
        return r

    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(self.out / "log.txt", "a") as f:
            f.write(msg + "\n")


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def act_gate(env, args, R):
    """cycle schedule, 3 steps: the harness carry == the oracle's P00 at iterations 1-3 (every point)."""
    ctl = zero_controls()
    carry, shift = BIND(env.M, env.carry0, ctl)
    bad = {}
    for it in ITS:
        carry = STEP(env.M, carry, take(env.X3, it - 1), shift, ad="ecco")
        for k in sm.ICE_STATE + ("sIceLoad",):
            bad[f"{it}/{k}"] = int(np.sum(np.asarray(carry[k]) != env.P00[it][k]))
    R.log("gate (cycle, 3 steps) mismatches:", {k: v for k, v in bad.items() if v} or "none")
    R.row(action="gate", mismatches=bad, ok=not any(bad.values()))


def act_time(env, args, R):
    """Warm per-step cost: forward (production and tight LSR), gradient per level (per-step VJP, --costs seeds)."""
    ctl = zero_controls()
    n = args.steps
    for name, M in (("prod", env.M), ("tight", env.M_tight)):
        f = forward(M, env, ctl, 1, args.schedule, record=True)       # compile
        f = forward(M, env, ctl, n, args.schedule, record=True)
        cnt = np.array(f.counts)
        R.log(f"forward {name}: {f.seconds / n:.2f} s/step over {n} steps; LSOR counts per step (pass1 u, pass2 u):",
              cnt[:, :, 0].tolist(), "converged:", cnt[:, :, 2].min())
        R.row(action="time", what=f"forward_{name}", s_per_step=f.seconds / n, counts=cnt.tolist(), J=f.J)
    seeds = seeds_for_costs(env, args.costs)
    fwd = forward(env.M, env, ctl, n, args.schedule, keep=True)
    for ad in args.levels:
        t0 = time.time()
        adjoint(env.M, env, ctl, 1, seeds, args.schedule, ad=ad, fwd=Fwd(fwd.J, None, fwd.carries[:1], [], [], 0.0))
        tc = time.time() - t0
        a = adjoint(env.M, env, ctl, n, seeds, args.schedule, ad=ad, fwd=fwd)
        R.log(f"gradient {ad}: first call (compile) {tc:.1f} s; warm reverse {a.reverse_seconds / n:.2f} s/step "
              f"({len(args.costs)} seeds)")
        R.row(action="time", what=f"reverse_{ad}", s_per_step=a.reverse_seconds / n, compile_s=tc,
              seeds=len(args.costs))


def dirderivs(env, grad, costs):
    """<dJ/dctl, d> for every FD direction: {J: {direction: value}}; grad[k] = [S, ...] in the order of `costs`."""
    return {j: {dn: sum(float(np.vdot(grad[k][s], v)) for k, v in env.dirs[dn].items()) for dn in DIRECTIONS}
            for s, j in enumerate(costs)}


def npz_name(tol, ad, n, j, r):
    return f"grad_{tol}_{ad}_{n}s_{j}_r{r}.npz"


def act_grad(env, args, R):
    """Gradients of the --costs per --levels at --tol, --repeats in this process: fields saved per cost
    (npz_name), directional derivatives along the FD directions, the amplification screen (per-step per-field
    cotangent norms, terminal seed = the cost), the pass-through fields of the level, the ecco identity check."""
    M = env.M_tight if args.tol == "tight" else env.M
    ctl = zero_controls()
    n = args.steps
    seeds = seeds_for_costs(env, args.costs)
    fwd = forward(M, env, ctl, n, args.schedule, keep=True)
    R.log(f"forward {n} steps ({args.tol}): J = {fwd.J.tolist()} ({fwd.seconds:.1f} s)")
    for ad in args.levels:
        reps = []
        for r in range(args.repeats):
            a = adjoint(M, env, ctl, n, seeds, args.schedule, ad=ad, fwd=fwd)
            reps.append(a)
            for s, j in enumerate(args.costs):
                np.savez(R.out / npz_name(args.tol, ad, n, j, r), **{k: v[s] for k, v in a.grad.items()})
            finite = all(np.all(np.isfinite(v)) for v in a.grad.values())
            rep = {}
            if r > 0:
                for k in CONTROLS:
                    d = np.abs(a.grad[k] - reps[0].grad[k]).max()
                    m = np.abs(reps[0].grad[k]).max()
                    rep[k] = float(d / m) if m > 0 else float(d)
            dd = dirderivs(env, a.grad, args.costs)
            R.log(f"grad {args.tol} {ad} r{r}: reverse {a.reverse_seconds:.1f} s, finite {finite}, repeat vs r0 {rep}; "
                  f"dirderiv {dd}")
            R.row(action="grad", tol=args.tol, level=ad, costs=args.costs, repeat=r, J=fwd.J, finite=finite,
                  norms={k: [float(np.linalg.norm(a.grad[k][s])) for s in range(len(args.costs))] for k in CONTROLS},
                  repeat_rel=rep, bitwise=(r > 0 and all(v == 0.0 for v in rep.values())), dirderiv=dd,
                  forward_s=a.forward_seconds, reverse_s=a.reverse_seconds)
        a = reps[0]
        passthru = []
        if args.probe:
            passthru = pass_through_fields(M, env, fwd, ad, args.schedule)
            R.log(f"pass-through fields at level {ad} (one-step reverse self-map == identity): {passthru}")
            R.row(action="pass_through", tol=args.tol, level=ad, fields=passthru)
        for s, j in enumerate(args.costs):
            fields = [k for k in CARRY if max(float(t[k][s]) for t in a.trace) > 0]
            dyn = [f for f in fields if f not in passthru]
            stats = {}
            for name, flds in [("prognostic", [f for f in PROGNOSTIC if f in dyn]), ("dynamic", dyn),
                               ("all", fields)] + [(f, [f]) for f in fields]:
                if not flds:
                    continue
                tr = trace_of(a.trace, flds, s)
                am = amplification(tr)
                stats[name] = dict(median=am.median, log_spread=am.log_spread, worst3=am.worst3, passes=am.passes,
                                   end=tr[0], start=tr[-1], max_rate=float(np.nanmax(am.rates)) if am.rates else None)
            R.row(action="screen", tol=args.tol, level=ad, cost=j, stats=stats,
                  trace={k: [float(t[k][s]) for t in a.trace] for k in fields})
            R.log(f"screen {ad} {j}: prognostic {stats.get('prognostic')}; dynamic {stats.get('dynamic')}")
        if ad == "ecco":
            check_ecco_identity(env, a, seeds, args.costs, R)


def pass_through_fields(M, env, fwd, ad, schedule):
    """Carried fields whose one-step reverse self-map is the identity at this level: a random cotangent on field k
    alone at the end of step 1 comes back on k unchanged, bitwise. Such a field's cotangent is passed through every
    step and ACCUMULATES what the step's readers add (Task 21: runoff faked a 1.0105/step growth), so the screen
    reports it separately. One single-seed step-VJP per field."""
    rng = np.random.default_rng(7)
    _, shift = BIND(M, env.carry0, zero_controls())
    x = take(env.X3, int(schedule_index(schedule, 1)[0]))
    z = {k: np.zeros(np.shape(v)) for k, v in env.carry0.items()}
    out = []
    carry = jax.device_put(fwd.carries[0])
    for k in CARRY:
        s = dict(z)
        s[k] = rng.standard_normal(np.shape(z[k]))
        seed = jax.tree.map(lambda a: jnp.asarray(a)[None], s)
        c_ct, _ = STEP_BWD(M, carry, x, shift, seed, ad=ad)
        if np.array_equal(np.asarray(c_ct[k][0]), np.asarray(seed[k][0])):
            out.append(k)
    return out


def act_probe(env, args, R):
    """The pass-through fields of each --levels (pass_through_fields), for the screen."""
    fwd = forward(env.M, env, zero_controls(), 1, args.schedule, keep=True)
    for ad in args.levels:
        t0 = time.time()
        passthru = pass_through_fields(env.M, env, fwd, ad, args.schedule)
        R.log(f"pass-through fields at level {ad} (one-step reverse self-map == identity): {passthru} "
              f"({time.time() - t0:.0f} s)")
        R.row(action="pass_through", tol="prod", level=ad, fields=passthru)


def check_ecco_identity(env, a, seeds, costs, R):
    """ad="ecco": the adjoint of the whole window is the identity on the carried state (ct at the window start ==
    the seed, bitwise) and zero on the input shifts; so dJ1,2/dHEFF0 == W, dJ3/dAREA0 == W, everything else 0."""
    ok = True
    for k in CARRY:
        ok &= bool(np.array_equal(np.asarray(a.ct0[k]), np.asarray(seeds[k])))
    exp = {k: np.zeros((len(costs),) + L.shape2d) for k in CONTROLS}
    for s, j in enumerate(costs):
        exp["AREA" if j == "J3" else "HEFF"][s] = np.asarray(env.W[j])
    gbad = {k: int(np.sum(a.grad[k] != exp[k])) for k in CONTROLS}
    ok &= not any(gbad.values())
    R.log(f"ecco identity over the window: {'EXACT' if ok else 'FAILED'} (grad mismatches {gbad})")
    R.row(action="ecco_identity", ok=ok, grad_mismatches=gbad)


def act_dotadj(env, args, R):
    """Adjoint half of the dot test: <v, J'^T w> for the random v (controls, --seed) and w (final carry, --seed+1),
    per level. The tangent half is `dottl` (may run in another process); the summary pairs them."""
    ctl = zero_controls()
    n = args.steps
    v = random_controls(env, args.seed)
    w = random_carry(env, args.seed + 1)
    fwd = forward(env.M, env, ctl, n, args.schedule, keep=True)
    for ad in args.levels:
        a = adjoint(env.M, env, ctl, n, jax.tree.map(lambda x: x[None], w), args.schedule, ad=ad, fwd=fwd)
        rhs = tree_vdot(v, {k: a.grad[k][0] for k in CONTROLS})
        finite = all(np.all(np.isfinite(g)) for g in a.grad.values())
        R.log(f"dotadj {ad}: <v, J'^T w> = {rhs:.16e} (finite {finite}, reverse {a.reverse_seconds:.1f} s)")
        R.row(action="dotadj", level=ad, seed=args.seed, rhs=rhs, finite=finite, reverse_s=a.reverse_seconds)


def act_dottl(env, args, R):
    """Tangent half of the dot test: <J' (amp v), w> per level and amplitude (--amps)."""
    ctl = zero_controls()
    n = args.steps
    v = random_controls(env, args.seed)
    w = random_carry(env, args.seed + 1)
    for ad in args.levels:
        for amp in args.amps:
            t0 = time.time()
            _, _, dcarry = tl(env.M, env, ctl, jax.tree.map(lambda z: z * amp, v), n, args.schedule, ad=ad)
            lhs = tree_vdot(dcarry, w)
            R.log(f"dottl {ad} amp {amp:g}: <J'v, w> = {lhs:.16e} ({time.time() - t0:.1f} s)")
            R.row(action="dottl", level=ad, seed=args.seed, amp=amp, lhs=lhs, seconds=time.time() - t0)


def act_tldir(env, args, R):
    """TL directional derivatives of J1-J3 along --dirs per level (compared with the adjoint's in the summary)."""
    ctl = zero_controls()
    n = args.steps
    M = env.M_tight if args.tol == "tight" else env.M
    for ad in args.levels:
        for dn in args.dirs:
            t0 = time.time()
            _, dJ, _ = tl(M, env, ctl, direction(env, dn), n, args.schedule, ad=ad)
            R.log(f"tldir {args.tol} {ad} {dn}: dJ = {dJ.tolist()} ({time.time() - t0:.1f} s)")
            R.row(action="tldir", tol=args.tol, level=ad, direction=dn, dJ=dJ, seconds=time.time() - t0)


def act_fdbase(env, args, R):
    """The FD base point for --tol: forward (switch states, LSOR counts) and a repeat without recording (forward
    noise floor; also shows that recording does not change the values). The adjoint values come from `grad` rows of
    the same tolerance."""
    M = env.M_tight if args.tol == "tight" else env.M
    ctl = zero_controls()
    n = args.steps
    base = forward(M, env, ctl, n, args.schedule, record=True)
    base2 = forward(M, env, ctl, n, args.schedule)
    noise = np.abs(base.J - base2.J)
    ice = [np.asarray(s["heff_pos"]) > 0 for s in base.switches]
    at = {k: int(sum(int(s[k].sum()) for s in base.switches)) for k in base.switches[0]}
    at_ice = {k: int(sum(int((np.asarray(s[k]) * m).sum()) for s, m in zip(base.switches, ice)))
              for k in base.switches[0]}
    cnt = np.array(base.counts)
    R.log(f"fd base ({args.tol}): J = {base.J.tolist()}, repeat |dJ| = {noise.tolist()}, {base.seconds:.1f} s; "
          f"switch cells summed over steps {at}; on ice-covered cells {at_ice}; LSOR counts pass1/pass2 "
          f"{cnt[:, :, 0].tolist()} converged min {cnt[:, :, 2].min()}")
    R.row(action="fd_base", tol=args.tol, J=base.J, noise=noise, switch_cells=at, switch_cells_ice=at_ice,
          ice_cells=int(sum(int(m.sum()) for m in ice)), counts=cnt.tolist(), seconds=base.seconds)


def act_fd(env, args, R):
    """Central FD h-sweep of J1-J3 along each direction in --dirs with the --tol forward (compared with the `fd_base`
    adjoint of the same tolerance in the summary); switch flips between the +h and -h forwards and LSOR-count changes
    counted per h. Only the +-h forwards run here, so (direction, h) pairs can go to separate processes."""
    M = env.M_tight if args.tol == "tight" else env.M
    n = args.steps
    hs = args.hs or H_SWEEP
    for dn in args.dirs:
        d = direction(env, dn)
        for h in hs:
            fp = forward(M, env, jax.tree.map(lambda a: h * a, d), n, args.schedule, record=True)
            fm = forward(M, env, jax.tree.map(lambda a: -h * a, d), n, args.schedule, record=True)
            fd = (fp.J - fm.J) / (2.0 * h)
            flips = {k: int(sum(int(np.sum(a[k] != b[k])) for a, b in zip(fp.switches, fm.switches)))
                     for k in fp.switches[0] if not k.startswith("near_")}
            cp, cm = np.array(fp.counts), np.array(fm.counts)
            cdiff = int(np.sum(cp[:, :, :2] != cm[:, :, :2]))
            R.log(f"fd {args.tol} {dn} h={h:g}: fd {fd.tolist()} flips {flips} LSOR-count changes {cdiff} "
                  f"converged min {min(cp[:, :, 2].min(), cm[:, :, 2].min())} ({fp.seconds + fm.seconds:.0f} s)")
            R.row(action="fd", tol=args.tol, direction=dn, h=h, fd=fd, flips=flips, lsor_count_changes=cdiff,
                  converged=float(min(cp[:, :, 2].min(), cm[:, :, 2].min())), Jp=fp.J, Jm=fm.J,
                  seconds=fp.seconds + fm.seconds)


# ---------------------------------------------------------------------------------------------------------------------
# summary tables (no model needed)


def _rows(dirs):
    rows = []
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["_dir"] = str(d)
                    rows.append(r)
    return rows


def _e(x, fmt="{:.1e}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def _rel(a, b):
    den = max(abs(a), abs(b))
    return abs(a - b) / den if den > 0 else 0.0


def plateau_of(rels, hs, bar=1e-3):
    """FD plateau: the longest run of consecutive h (descending) with relative error <= bar; (n, h_hi, h_lo)."""
    best, cur = (0, None, None), []
    for h, r in zip(hs, rels):
        if r is not None and r <= bar:
            cur.append(h)
            if len(cur) > best[0]:
                best = (len(cur), cur[0], cur[-1])
        else:
            cur = []
    return best


def ad_values(rows):
    """{(steps, tol, level, J): {direction: value}} from the first grad row (repeat 0) of each key."""
    out = {}
    for r in rows:
        if r["action"] == "grad" and r["repeat"] == 0:
            for j, dd in r["dirderiv"].items():
                out.setdefault((r["steps"], r["tol"], r["level"], j), dd)
    return out


def summarize(dirs, bar=1e-3):
    """Markdown tables from the results.jsonl (and grad_*.npz) files of the run directories."""
    rows = _rows(dirs)
    AD = ad_values(rows)
    out = []
    P = out.append
    for r in rows:
        if r["action"] == "gate":
            P(f"- harness gate (cycle schedule, 3 steps == P00 it 1-3): {'bitwise' if r['ok'] else 'FAILED'} "
              f"(job {r['job']})")
        if r["action"] == "ecco_identity":
            P(f"- ecco level, {r['steps']} steps: reverse of the window == identity/zero "
              f"{'exactly' if r['ok'] else 'FAILED ' + str(r['grad_mismatches'])} (job {r['job']})")
    # dot tests
    P("\n### TL vs adjoint dot test: random v (all controls), random w (final carry, all fields)\n")
    P("| steps | level | amp | <J'v, w> (TL) | <v, J'^T w> (adjoint) | rel. difference | jobs |")
    P("|---|---|---|---|---|---|---|")
    adj = {(r["steps"], r["level"], r["seed"]): r for r in rows if r["action"] == "dotadj"}
    for r in sorted((r for r in rows if r["action"] == "dottl"), key=lambda r: (r["steps"], r["level"], -r["amp"])):
        a = adj.get((r["steps"], r["level"], r["seed"]))
        if a is None:
            continue
        rhs = r["amp"] * a["rhs"]
        P(f"| {r['steps']} | {r['level']} | {r['amp']:g} | {r['lhs']:.15e} | {rhs:.15e} | {_e(_rel(r['lhs'], rhs))} | "
          f"{r['job']}/{a['job']} |")
    P("\n### TL directional derivatives vs the adjoint (<grad, d>), J1-J3 along the FD directions\n")
    P("| steps | tol | level | direction | rel. difference J1 / J2 / J3 | TL J1, J2, J3 |")
    P("|---|---|---|---|---|---|")
    for r in sorted((r for r in rows if r["action"] == "tldir"),
                    key=lambda r: (r["steps"], r["tol"], r["level"], r["direction"])):
        rels = []
        for s, j in enumerate(COSTS):
            a = AD.get((r["steps"], r["tol"], r["level"], j))
            rels.append(_e(_rel(r["dJ"][s], a[r["direction"]])) if a else "-")
        P(f"| {r['steps']} | {r['tol']} | {r['level']} | {r['direction']} | {' / '.join(rels)} | "
          f"{', '.join('%.6e' % x for x in r['dJ'])} |")
    # FD
    P("\n### FD h-sweeps (central) vs the \"full\" adjoint at the same LSR tolerance\n")
    P(f"Relative error |FD - AD| / |AD| at h = h/|x| = 1e-2, 1e-3, ..., 1e-8; plateau = longest run of consecutive h "
      f"with error <= {bar:g}; flips = cells (summed over steps) whose switch state differs between the +h and -h "
      f"forwards (all switches); LSOR = (step, pass, u/v) sweep counts that differ between +h and -h.\n")
    fds = {}
    for r in rows:
        if r["action"] == "fd":
            fds.setdefault((r["steps"], r["tol"], r["direction"]), {})[r["h"]] = r
    P("| steps | tol | direction | J | AD full | FD rel. error, h = 1e-2 ... 1e-8 | plateau | "
      "flips at h = 1e-2/1e-4/1e-6/1e-8 | LSOR changes |")
    P("|---|---|---|---|---|---|---|---|---|")
    best_fd = {}
    for (n, tol, dn), byh in sorted(fds.items()):
        hs = sorted(byh, reverse=True)
        for s, j in enumerate(COSTS):
            a = AD.get((n, tol, "full", j))
            if a is None:
                continue
            ad = a[dn]
            rels = [abs(byh[h]["fd"][s] - ad) / abs(ad) if ad != 0 else None for h in hs]
            pl = plateau_of(rels, hs, bar)
            fin = [(x, h) for x, h in zip(rels, hs) if x is not None]
            if fin:
                best_fd[(n, tol, dn, j)] = byh[min(fin)[1]]["fd"][s]
            fl = " / ".join(str(sum(byh[h]["flips"].values())) if h in byh else "-" for h in (1e-2, 1e-4, 1e-6, 1e-8))
            lc = " / ".join(str(byh[h]["lsor_count_changes"]) if h in byh else "-" for h in (1e-2, 1e-4, 1e-6, 1e-8))
            P(f"| {n} | {tol} | {dn} | {j} | {ad:.6e} | " + ", ".join(_e(x) for x in rels)
              + f" | {'%d h (%g..%g)' % pl if pl[0] else 'none'} | {fl} | {lc} |")
    P("\n### FD base points\n")
    P("| steps | tol | J1, J2, J3 | repeat noise | switch cells summed over steps (ice-covered cells only) | "
      "LSOR sweeps pass 1, first/last step | forward s |")
    P("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["action"] == "fd_base":
            c = np.array(r["counts"])
            P(f"| {r['steps']} | {r['tol']} | {', '.join('%.12g' % x for x in r['J'])} | "
              f"{', '.join(_e(x) for x in r['noise'])} | {r.get('switch_cells_ice')} of {r.get('ice_cells')} | "
              f"{int(c[0, 0, 0])}/{int(c[-1, 0, 0])} | {r['seconds']:.0f} |")
    # levels
    P("\n### Directional derivatives per level (rel. difference to the full adjoint; FD = best h)\n")
    P("| steps | tol | direction | J | full | FD (rel. to full) | no_dynamics (rel.) | ecco (rel.) |")
    P("|---|---|---|---|---|---|---|---|")
    keys = sorted({(k[0], k[1], k[3]) for k in AD if k[2] == "full"})
    for n, tol, j in keys:
        for dn in DIRECTIONS:
            f = AD[(n, tol, "full", j)][dn]
            cells = []
            fdv = best_fd.get((n, tol, dn, j))
            cells.append("-" if fdv is None else f"{fdv:.4e} ({_e(_rel(fdv, f))})")
            for lv in ("no_dynamics", "ecco"):
                a = AD.get((n, tol, lv, j))
                cells.append("-" if a is None else f"{a[dn]:.4e} ({_e(_rel(a[dn], f)) if f != 0 else '-'})")
            P(f"| {n} | {tol} | {dn} | {j} | {f:.4e} | " + " | ".join(cells) + " |")
    P("\n### Effect of the forward LSR tolerance on the full gradient: AD on the production (LSR_ERROR 2e-4) vs the "
      "tight (1e-12) trajectory, rel. difference\n")
    P("| steps | J | " + " | ".join(DIRECTIONS) + " |")
    P("|---|---|" + "---|" * len(DIRECTIONS))
    for n, j in sorted({(k[0], k[3]) for k in AD if k[1] == "tight" and k[2] == "full"}):
        a, b = AD.get((n, "prod", "full", j)), AD[(n, "tight", "full", j)]
        if a:
            P(f"| {n} | {j} | " + " | ".join(_e(_rel(a[dn], b[dn])) for dn in DIRECTIONS) + " |")
    # gradient fields: level differences and repeats from the npz files
    P("\n### Gradient fields: level differences ||g_a - g_b|| / ||g_full|| per control (prod tol)\n")
    files = {}
    for d in dirs:
        for f in sorted(Path(d).glob("grad_*.npz")):
            files.setdefault(f.name, []).append(f)
    P("| steps | J | control | no_dynamics vs full | ecco vs full |")
    P("|---|---|---|---|---|")
    for n in sorted({r["steps"] for r in rows}):
        for j in COSTS:
            gs = {}
            for lv in LEVELS:
                fl = files.get(npz_name("prod", lv, n, j, 0))
                if fl:
                    gs[lv] = np.load(fl[0])
            if "full" not in gs:
                continue
            for k in CONTROLS:
                gf = gs["full"][k]
                nf = np.linalg.norm(gf)
                cells = []
                for lv in ("no_dynamics", "ecco"):
                    cells.append("-" if lv not in gs else (_e(np.linalg.norm(gs[lv][k] - gf) / nf) if nf > 0 else
                                                           f"|g|={np.linalg.norm(gs[lv][k]):.1e}, full 0"))
                P(f"| {n} | {j} | {k} | {cells[0]} | {cells[1]} |")
    P("\n### Repeats (same inputs): max |g_r - g_0| / max |g_0| over controls\n")
    P("| file | copies | max rel. difference | bitwise |")
    P("|---|---|---|---|")
    groups = {}
    for name, fl in files.items():
        base = name.rsplit("_r", 1)[0]
        groups.setdefault(base, []).extend(fl)
    for base, fl in sorted(groups.items()):
        if len(fl) < 2:
            continue
        g0 = np.load(fl[0])
        worst, bit = 0.0, True
        for f in fl[1:]:
            g = np.load(f)
            for k in CONTROLS:
                m = np.abs(g0[k]).max()
                dd = np.abs(g[k] - g0[k]).max()
                worst = max(worst, dd / m if m > 0 else dd)
                bit &= bool(np.array_equal(g[k], g0[k]))
        P(f"| {base} | {len(fl)} | {_e(worst)} | {bit} |")
    P("\n### Amplification screen (terminal seed = the cost; per-step cotangent-norm growth)\n")
    P("Per-step growth of the carry-cotangent norm going backwards, statistics over the steps AFTER the seed step (the "
      "first reverse step maps the HEFF/AREA seed onto the other fields: a change of field set, not growth). Norms: "
      "prognostic = HEFF, AREA, HSNOW, UICE, VICE; dynamic = every field with a nonzero trace; both WITHOUT the "
      "level's pass-through fields (probe: one-step reverse self-map == identity; their cotangent accumulates); "
      "pass-through = the pass-through fields alone. Bars (fesom_jax): median <= 1.010, log-spread <= 0.020, "
      "worst-3 <= 1.030 per step.\n")
    P("| steps | tol | level | J | norm over | median /step | log-spread | worst-3 /step | max /step | passes | "
      "norm after seed step -> window start |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    passthru = {}
    for r in rows:
        if r["action"] == "pass_through":
            passthru[r["level"]] = r["fields"]
    seen = set()
    for r in rows:
        key = (r["steps"], r.get("tol"), r.get("level"), r.get("cost"))
        if r["action"] != "screen" or key in seen:   # repeats in other processes: the first one
            continue
        seen.add(key)
        tr = r["trace"]
        pt = passthru.get(r["level"])
        nonzero = [k for k in tr if max(tr[k]) > 0]
        sets = [("prognostic", [k for k in PROGNOSTIC if k in nonzero and k not in (pt or [])]),
                ("dynamic", [k for k in nonzero if k not in (pt or [])]),
                ("pass-through", [k for k in nonzero if k in (pt or [])])]
        sets += [(k, [k]) for k in PROGNOSTIC if k in nonzero]
        for name, flds in sets:
            if not flds:
                continue
            norm = [float(np.sqrt(sum(tr[k][i] ** 2 for k in flds))) for i in range(len(tr[flds[0]]))]
            am = amplification(norm[1:])
            tag = "" if pt is not None else " (no probe)"
            P(f"| {r['steps']} | {r['tol']} | {r['level']} | {r['cost']} | {name}{tag} | "
              f"{am.median:.5f} | {am.log_spread:.4f} | {am.worst3:.5f} | "
              f"{_e(float(np.nanmax(am.rates)) if am.rates else None, '{:.4f}')} | {am.passes} | "
              f"{norm[1]:.3e} -> {norm[-1]:.3e} |")
    for lv, f in passthru.items():
        P(f"\n- pass-through fields at level {lv}: {f}")
    P("\n### Cost (CPU, 16 cores, gate XLA flags)\n")
    for r in rows:
        if r["action"] == "time":
            P(f"- {r['steps']} steps: {r['what']} {r['s_per_step']:.2f} s/step (job {r['job']})")
        if r["action"] == "grad":
            P(f"- grad r{r['repeat']} {r['tol']} {r['level']} {r['steps']} steps, {len(r['costs'])} seed(s): forward "
              f"{r['forward_s']:.0f} s, reverse {r['reverse_s']:.0f} s = {r['reverse_s'] / r['steps']:.1f} s/step "
              f"(job {r['job']})")
        if r["action"] == "tldir":
            P(f"- TL {r['tol']} {r['level']} {r['steps']} steps: {r['seconds']:.0f} s = "
              f"{r['seconds'] / r['steps']:.1f} s/step (job {r['job']})")
    return "\n".join(out)


def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        print(summarize(sys.argv[2:]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--schedule", default="fixed", choices=("fixed", "cycle"))
    ap.add_argument("--actions", default="gate,time")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--costs", default=",".join(COSTS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", default="prod", choices=("prod", "tight"))
    ap.add_argument("--dirs", default=",".join(DIRECTIONS))
    ap.add_argument("--hs", default="")
    ap.add_argument("--amps", default="1,1e-6")
    ap.add_argument("--probe", action="store_true", help="grad: probe the pass-through fields of each level")
    args = ap.parse_args(argv)
    args.levels = [x for x in args.levels.split(",") if x]
    args.costs = [x for x in args.costs.split(",") if x]
    args.dirs = [x for x in args.dirs.split(",") if x]
    args.hs = [float(x) for x in args.hs.split(",") if x]
    args.amps = [float(x) for x in args.amps.split(",") if x]
    assert set(args.levels) <= set(LEVELS) and set(args.costs) <= set(COSTS) and set(args.dirs) <= set(DIRECTIONS)
    R = Recorder(args.out, args)
    R.log(f"seaice_window: {' '.join(sys.argv[1:] if argv is None else argv)}; jax {jax.__version__}, "
          f"{jax.devices()}, XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    env = load_env(R.log)
    acts = {"gate": act_gate, "time": act_time, "grad": act_grad, "dotadj": act_dotadj, "dottl": act_dottl,
            "tldir": act_tldir, "fdbase": act_fdbase, "fd": act_fd, "probe": act_probe}
    for a in [x for x in args.actions.split(",") if x]:
        t0 = time.time()
        acts[a](env, args, R)
        R.log(f"action {a} done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
