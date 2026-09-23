#!/usr/bin/env python3
"""Plan Task 21: multi-week gradient on LLC90 (M1 adjoint acceptance), flux-forced V4r4, production configuration.

Model: the production run directory (useCTRL=T with the V4r4 control adjustments, geothermal flux; 1992-01-01,
nIter0 = 1, dt = 3600 s), initial state from init.state_from_pickup (cached once by `cache`, built on CPU with the
gate XLA flags, i.e. bitwise the Fortran start-of-run state), forcing from the EXF record loader.

Cost (ECCO adjoint-sensitivity experiment, namelist_adjsen/data.ecco: gencost 'boxmean' of theta, 3-D mask objmask):
    J = (1/N) sum_{n=1..N} sum_{i,j,k} m(i,j,k) theta_n hFacC_n drF(k) rA(i,j) / sum m h0FacC drF rA
the volume-weighted box-mean theta at the end of every step (ff forward_step.F:1189 ECCO_PHYS -> ecco_phys.F:314-348:
tmpvol = hFacC*drF*rA with the current r* hFacC, divided by the static box volume eccoVol_0, ecco_check.F:67),
averaged over the N steps of the window (the gencost 'month' average with the temporal mask 1 on the one record).
Box m: scripts/prepare_run_adjsen.py, literally: maskC * (XC >= 120) * (YC <= 151) * (YC >= 5) * (YC <= 16), levels
15..20 (1-based). NOTE: the script's comment says 120E-151E but its condition tests YC <= 151 (always true), so the
box the adjsen experiment writes is 120E-180E, 5N-16N, levels 15-20 (~150-320 m); it is used as written. The script
also divides objmask by the box volume and ecco_phys divides again (J_ecco = J / totvol). Default (Nikolay,
2026-09-23: "keep, but print a warning"): --j-scaling literal = J as the Fortran computes it, i.e. the K-normalised
J above divided once more by the box volume (K/m^3), with a WARNING that this double division looks like a bug in
the ECCO adjsen set-up (docs/ECCO_ISSUES.md); --j-scaling kelvin = J in K (the Task 21 results and the tier-2
regression values). The two differ by the constant factor 1/volume only. The box edge (YC <= 151) is kept as
written in the ECCO script too (Nikolay, 2026-09-23), with a NOTE printed.

Controls (theta pytree; zero = the production forward, value-identical):
    theta    [T,Nr,ny,nx] additive theta at the start of iteration 1, interior, then EXCH_XYZ_RL: what xx_theta does in
             CTRL_MAP_INI_GENARR (after CALC_PHI_RLOW_INI; nothing later in INITIALISE_VARIA reads theta)
    kapGM    [T,Nr,ny,nx] additive GM diffusivity (m^2/s), interior, then EXCH_XYZ_RL (xx_kapgm path, model.setup)
    tflux    [T,ny,nx] time-constant adjustment of the TFLUX record buffers (W/m^2, downward: exf_inscal_hflux = -1
             makes hflux = -TFLUX, upward), added to both buffers fld0/fld1 of every step (EXF_SET_FLD interpolates
             linearly, weights summing to 1: a constant shift of the forcing over the window)
    taux, tauy  the same for oceTAUX / oceTAUY (N/m^2, model-grid components)

Actions (one process runs a list of them; one mode per process keeps the chunk executables cached):
    cache      build setup(rundir) + state_from_pickup and save them (CPU, gate flags; run once)
    grad       chunked gradient of J (repeats), optional forward check against the other mode, saves the fields
    screen     chunked gradient of the end-of-window box mean (terminal seed only): the per-chunk cotangent-norm trace
               measures pure reverse propagation (a running cost injects cotangent every step and would read as growth)
    tl         tangent-linear (jax.jvp) directional derivatives along the named directions + the combined dot test at
               amplitudes 1 and 1e-6 (compared with <grad, v> of the saved adjoint gradient)
    fd         central differences along the named directions, h-sweep, forward repeat spread (noise floor)
Every row (OUT/results.jsonl) records action, mode, freezes, window, chunk/stride/schedule, h, memory, times.

    python scripts/adjoint/multiweek_grad.py --out DIR --days 7 --mode exact --actions grad,screen,tl
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import mitgcm_jax  # noqa: E402,F401  (x64)
from mitgcm_jax.adjoint import checkpoint as ck  # noqa: E402
from mitgcm_jax.adjoint import grad as gr  # noqa: E402
from mitgcm_jax.adjoint.modes import EXACT, AdjointConfig  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.state import State  # noqa: E402

RUNDIR = Path("/work/ab0995/a270088/MIT/reference/runs/ref_ff_serial13_1day")
CACHE = Path("/work/ab0995/a270088/MIT/runs/adjoint/init_ref_ff_serial13_1day")
MIXING = ("kapGM", "kapRedi", "diffKr")
# prepare_run_adjsen.py: kkk0 = 15-1, kkk1 = 20-1 (0-based levels 14..19); XC >= 120; 5 <= YC <= 16 (YC <= 151)
BOX = {"xmin": 120.0, "ylo": 5.0, "yhi": 16.0, "yc_max_literal": 151.0, "k0": 14, "k1": 19}
PROGNOSTIC = ("theta", "salt", "uVel", "vVel", "etaN")
FORCING = {"tflux": "hflux", "taux": "ustress", "tauy": "vstress"}   # control name -> EXF buffer name
H_DEFAULT = {"theta": (1e-1, 1e-2, 1e-3, 1e-4), "kapGM": (1e-1, 1e-2, 1e-3, 1e-4), "tflux": (10.0, 1.0, 0.1, 0.01),
             "taux": (1e-2, 1e-3, 1e-4, 1e-5)}


# ---------------------------------------------------------------------------------------------------------------
# model, initial state, forcing


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def build_cache(rundir, cache, log):
    """setup (useCTRL: WC01-adjusted kapGM/kapRedi/diffKr) + state_from_pickup; save to a NEW directory."""
    from mitgcm_jax.init import state_from_pickup
    from mitgcm_jax.model import setup
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    P, g, ex, kLowC = setup(rundir)
    t1 = time.time()
    st = state_from_pickup(P, g, ex, kLowC, rundir)
    t2 = time.time()
    np.savez(cache / "state.npz", it=np.asarray(int(st.it)), **{k: np.asarray(v) for k, v in st.f.items()})
    np.savez(cache / "mixing.npz", **{k: np.asarray(g.f[k]) for k in MIXING})
    meta = dict(rundir=str(rundir), git=git_head(), xla_flags=os.environ.get("XLA_FLAGS", ""),
                platform=jax.devices()[0].platform, setup_s=t1 - t0, init_s=t2 - t1, it=int(st.it),
                nfields=len(st.f), job=os.environ.get("SLURM_JOB_ID"))
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"cache written to {cache}: setup {t1 - t0:.1f} s, state_from_pickup {t2 - t1:.1f} s")


def load_model(rundir, cache, unroll, log):
    """(P, g, ex, kLowC, st0): setup without the ctrl mixing adjustment, then the cached (adjusted) mixing fields and
    the cached initial state."""
    from mitgcm_jax.model import setup
    t0 = time.time()
    P, g, ex, kLowC = setup(rundir, ctrl_mixing=False)
    z = np.load(Path(cache) / "mixing.npz")
    g = g.replace(**{k: jnp.asarray(z[k]) for k in MIXING})
    z = np.load(Path(cache) / "state.npz")
    st0 = State({k: jnp.asarray(z[k]) for k in z.files if k != "it"}, jnp.asarray(int(z["it"])))
    if unroll != 1:
        P = P._replace(cg=dataclasses.replace(P.cg, sum_unroll=int(unroll)))
    log(f"model + cached initial state loaded in {time.time() - t0:.1f} s ({len(st0.f)} fields, it={int(st0.it)})")
    return P, g, ex, kLowC, st0


class FModel(NamedTuple):
    """ck.Model plus the time-constant forcing adjustments {EXF buffer name: [T,ny,nx]} the step adds to the record
    buffers (a pytree: jit argument)."""
    m: Any
    fc: Any


def make_fstep(adj):
    base = ck.make_step(adj)

    def step(fm, st, x):
        if fm.fc:
            bufs = dict(x["bufs"])
            for name, d in fm.fc.items():
                b0, b1 = bufs[name]
                bufs[name] = (b0 + d, b1 + d)
            x = dict(x, bufs=bufs)
        return base(fm.m, st, x)
    return step


def interior_mask(L, three_d):
    m = np.zeros(L.shape3d if three_d else L.shape2d)
    m[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = 1.0
    return m


def box_mask(g):
    """[T,Nr,ny,nx] 0/1: prepare_run_adjsen.py idx_objmask (interior points only)."""
    L = g.layout
    xc, yc = np.asarray(g.xC), np.asarray(g.yC)
    mC = np.asarray(g.maskC)
    col = (xc >= BOX["xmin"]) & (yc <= BOX["yc_max_literal"]) & (yc >= BOX["ylo"]) & (yc <= BOX["yhi"])
    m = (mC > 0) & col[:, None]
    m[:, :BOX["k0"]] = False
    m[:, BOX["k1"] + 1:] = False
    return m & (interior_mask(L, True) > 0)


class Experiment:
    """Everything a process needs: model, initial state, window forcing, cost, controls, named directions."""

    def __init__(self, a, log):
        self.a, self.log = a, log
        self.nml = RunNamelists(a.rundir)
        P, g, ex, kLowC, st0 = load_model(a.rundir, a.cache, a.unroll, log)
        self.P, self.g, self.ex, self.kLowC, self.st0 = P, g, ex, kLowC, st0
        L = self.L = g.layout
        self.N = int(round(a.days * 24 * 3600 / float(self.nml.get("data", "parm03", "deltaTClock"))))
        if self.N % a.chunk:
            raise ValueError(f"{self.N} steps is not a multiple of --chunk {a.chunk}")
        t0 = time.time()
        it0 = int(st0.it)
        self.xs = ck.exf_window(ck.exf_loader_at(P, g, a.rundir, self.nml, it0), self.nml, it0, self.N)
        log(f"EXF window: {self.N} steps from it={it0}, {sum(v.nbytes for v in jax.tree.leaves(self.xs)) / 1e9:.2f}"
            f" GB host, {time.time() - t0:.1f} s")
        self.model = ck.Model(P, g, kLowC, ex)
        self.fm0 = FModel(self.model, {})
        m = box_mask(g)
        self.box = m
        drF, rA, h0 = np.asarray(g.drF), np.asarray(g.rA), np.asarray(g.h0FacC)
        vol0 = float(np.sum(m * h0 * drF[None, :, None, None] * rA[:, None]))       # eccoVol_0 over the box
        self.box_vol = vol0
        self.W = jnp.asarray(m * drF[None, :, None, None] * rA[:, None] / vol0)
        if a.j_scaling == "literal":                   # ecco_phys.F divides the (already /volume) objmask again
            self.W = self.W / vol0
        self.int3 = jnp.asarray(interior_mask(L, True))
        self.int2 = jnp.asarray(interior_mask(L, False))
        log(f"box: {int(m.sum())} wet cells, volume {vol0:.4e} m^3")
        log("NOTE: box as written in ECCO's prepare_run_adjsen.py: its comment says 120E-151E but the condition tests "
            "YC <= 151 (always true), so the box is 120E-180E, 5N-16N, levels 15-20 (docs/ECCO_ISSUES.md)")
        if a.j_scaling == "literal":
            log(f"WARNING: J follows the Fortran adjsen set-up literally: prepare_run_adjsen.py divides objmask by the "
                f"box volume and ecco_phys.F divides by it again, so J = box-mean theta / volume ({vol0:.4e} m^3), "
                f"in K/m^3. We follow the Fortran; this double division is probably a bug in the ECCO set-up "
                f"(docs/ECCO_ISSUES.md). Gradients differ from the K-normalised J by the constant factor 1/volume "
                f"only; use --j-scaling kelvin for J in K.")
        gpath = Path(a.out) / "grid.npz"
        if not gpath.exists():   # for the maps (plot_sensitivity.py runs in the nereus env, without this package)
            I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
            np.savez(gpath, xC=np.asarray(g.xC)[:, J, I], yC=np.asarray(g.yC)[:, J, I],
                     maskC=np.asarray(g.maskC)[..., J, I], box=m[..., J, I], rA=np.asarray(g.rA)[:, J, I],
                     drF=drF, h0FacC=h0[..., J, I], rC=np.asarray(g.rC), kapGM=np.asarray(g.kapGM)[..., J, I])
        N, W, int3, int2 = self.N, self.W, self.int3, self.int2

        def cost(fm, st, x):
            return jnp.sum(W * st.theta * st.hFacC) / N

        def final_cost(fm, st):
            return jnp.sum(W * st.theta * st.hFacC)

        def init_fn(th, st):
            if "theta" in th:
                return st.replace(theta=ex.exch_xy(st.theta + int3 * th["theta"]))   # CTRL_MAP_INI_GENARR + EXCH
            return st

        def params_fn(th, fm):
            m_, gg = fm.m, fm.m.g
            if "kapGM" in th:
                gg = gg.replace(kapGM=m_.ex.exch_xy(gg.kapGM + int3 * th["kapGM"]))
            fc = {FORCING[k]: int2 * th[k] for k in FORCING if k in th}
            return FModel(m_._replace(g=gg), fc)

        self.cost, self.final_cost, self.init_fn, self.params_fn = cost, final_cost, init_fn, params_fn
        self.theta0 = {"theta": jnp.zeros(L.shape3d), "kapGM": jnp.zeros(L.shape3d),
                       "tflux": jnp.zeros(L.shape2d), "taux": jnp.zeros(L.shape2d), "tauy": jnp.zeros(L.shape2d)}
        self.dirs = self.directions()
        self._steps = {}

    # ------------------------------------------------------------------ helpers
    def step(self, mode):
        if mode not in self._steps:
            adj = EXACT if mode == "exact" else AdjointConfig.ecco(self.nml)
            self._steps[mode] = (make_fstep(adj), adj)
        return self._steps[mode]

    def point(self, lon, lat, k):
        """Interior wet point (tile, k, j, i) nearest to lon/lat at level k (0-based)."""
        L, g = self.L, self.g
        xc, yc, mC = np.asarray(g.xC), np.asarray(g.yC), np.asarray(g.maskC)[:, k]
        inner = interior_mask(L, False) > 0
        d = (((xc - lon + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(lat))) ** 2 + (yc - lat) ** 2
        d = np.where(inner & (mC > 0), d, np.inf)
        t, j, i = np.unravel_index(np.argmin(d), d.shape)
        return int(t), int(k), int(j), int(i)

    def directions(self):
        """Named control directions (pytrees like theta0) for FD / TL."""
        L = self.L
        z3, z2 = np.zeros(L.shape3d), np.zeros(L.shape2d)
        out, where = {}, {}
        pA = self.point(150.0, 10.5, 16)            # box centre, level 17
        pB = self.point(150.0, 10.5, 13)            # same column, level 14: just above the box top (outside)
        pC = self.point(150.0, 3.5, 16)             # level 17, ~1.5 deg south of the box (outside)
        for name, p in (("theta_A_centre", pA), ("theta_B_above", pB), ("theta_C_south", pC)):
            d = z3.copy()
            d[p] = 1.0
            out[name] = {"theta": d}
            where[name] = dict(point=p, lon=float(np.asarray(self.g.xC)[p[0], p[2], p[3]]),
                               lat=float(np.asarray(self.g.yC)[p[0], p[2], p[3]]), in_box=bool(self.box[p]))
        kap = np.asarray(self.g.kapGM) * np.asarray(self.int3) * (np.asarray(self.g.maskC) > 0)
        out["kapGM_scale"] = {"kapGM": kap}          # d/d(log kapGM), global
        where["kapGM_scale"] = dict(desc="global relative change of kapGM (direction = kapGM itself)")
        foot = np.any(self.box, axis=1).astype(float) * np.asarray(self.int2)
        out["tflux_box"] = {"tflux": foot}           # 1 W/m^2 downward over the box footprint
        where["tflux_box"] = dict(desc="uniform +1 W/m^2 (downward) TFLUX over the box surface footprint",
                                  ncols=int(foot.sum()))
        out["taux_box"] = {"taux": foot}             # 1 N/m^2 oceTAUX (model-grid x component) over the footprint
        where["taux_box"] = dict(desc="uniform +1 N/m^2 oceTAUX (model-grid x component) over the box footprint",
                                 ncols=int(foot.sum()))
        self.where = where
        full = {}
        for n, d in out.items():
            t = {k: np.zeros_like(np.asarray(v)) for k, v in self.theta0.items()}
            t.update(d)
            full[n] = t
        return full

    def rowbase(self, action, mode):
        adj = self.step(mode)[1]
        return dict(action=action, mode=mode, freezes=dataclasses.asdict(adj), days=self.a.days, nsteps=self.N,
                    chunk=self.a.chunk, stride=self.a.stride, schedule=f"chunked/{self.a.schedule}",
                    unroll=self.a.unroll, job=os.environ.get("SLURM_JOB_ID"), git=git_head(),
                    device=str(jax.devices()[0].device_kind), platform=jax.devices()[0].platform,
                    xla_flags=os.environ.get("XLA_FLAGS", ""), rundir=str(self.a.rundir), box_vol=self.box_vol,
                    j_scaling=self.a.j_scaling)

    def controls(self):
        return {k: self.theta0[k] for k in self.a.controls.split(",")}


# ---------------------------------------------------------------------------------------------------------------
# actions


def emit(out, row, log):
    s = json.dumps(row, default=float)
    with open(Path(out) / "results.jsonl", "a") as fh:
        fh.write(s + "\n")
    log("ROW " + s[:2000])


def save_grad(out, name, g, L):
    """interior fields (float64) of every control of a gradient pytree."""
    I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
    np.savez(Path(out) / f"{name}.npz", **{k: np.asarray(v)[..., J, I] for k, v in g.items()})


def dir_dot(grad, d):
    return float(sum(np.vdot(np.asarray(grad[k]).ravel(), np.asarray(d[k]).ravel()) for k in grad if k in d))


def action_grad(E, mode, out, log):
    a = E.a
    step, adj = E.step(mode)
    theta = E.controls()
    xs_fn, nch = gr.chunks_of(E.xs, a.chunk)
    res = []
    for r in range(a.repeats):
        t = time.time()
        o = gr.chunked_value_and_grad(step, theta, E.fm0, E.st0, n_chunks=nch, chunk_steps=a.chunk, xs_fn=xs_fn,
                                      cost=E.cost, schedule=a.schedule, init_fn=E.init_fn, params_fn=E.params_fn,
                                      boundary_stride=a.stride, log=log)
        wall = time.time() - t
        g = {k: np.asarray(v) for k, v in o.grad.items()}
        save_grad(out, f"grad_{mode}_{a.days}d_r{r}", g, E.L)
        mem = gr.gpu_memory_gb()
        row = E.rowbase("grad", mode)
        row.update(repeat=r, J=o.loss, wall_s=wall, forward_s=o.forward_seconds, reverse_s=o.reverse_seconds,
                   host_gb=o.host_gb, peak_gb=mem["peak"] if mem else None,
                   grad_norm={k: float(np.linalg.norm(v)) for k, v in g.items()},
                   grad_finite=bool(all(np.all(np.isfinite(v)) for v in g.values())),
                   dirderiv={n: dir_dot(g, d) for n, d in E.dirs.items()},
                   samples={n: float(g["theta"][w["point"]]) for n, w in E.where.items() if "point" in w},
                   trace=o.trace)
        res.append((o.loss, g))
        if r > 0:
            J0, g0 = res[0]
            row["repeat_J_equal_r0"] = o.loss == J0
            row["repeat_maxrel_r0"] = {k: float(np.max(np.abs(g[k] - g0[k])) / max(np.max(np.abs(g0[k])), 1e-300))
                                       for k in g}
        emit(out, row, log)
    return res


def action_forward_check(E, out, log):
    """ecco forward == exact forward, bytes: the chunked forward of both modes over the whole window, final State and
    J compared bitwise (the backward switches must not change a forward value)."""
    a = E.a
    xs_fn, nch = gr.chunks_of(E.xs, a.chunk)
    finals = {}
    for mode in ("exact", "ecco"):
        step, _ = E.step(mode)
        fwd, _ = gr._chunk_fns(step, a.schedule, None, E.cost, E.params_fn, ck.SAVE_NAMES)
        th = E.controls()
        st = ck.prepare_state(step, E.params_fn(th, E.fm0), E.st0, jax.tree.map(lambda v: v[0], xs_fn(0)))
        carry = (jax.jit(E.init_fn)(th, st), jnp.zeros((), jnp.float64))
        t = time.time()
        for c in range(nch):
            carry = jax.block_until_ready(fwd(th, E.fm0, carry, jax.device_put(xs_fn(c))))
        finals[mode] = (jax.device_get(carry), time.time() - t)
    (se, Je), (sc, Jc) = finals["exact"][0], finals["ecco"][0]
    ndiff = {k: int(np.count_nonzero(np.asarray(se.f[k]) != np.asarray(sc.f[k]))) for k in se.f}
    row = E.rowbase("forward_check", "ecco")
    row.update(J_exact=float(Je), J_ecco=float(Jc), J_bitwise=bool(float(Je) == float(Jc)),
               fields_differing={k: v for k, v in ndiff.items() if v}, nfields=len(ndiff),
               bitwise=bool(float(Je) == float(Jc) and not any(ndiff.values())),
               forward_s={m: finals[m][1] for m in finals})
    emit(out, row, log)


def carried_constants(step, fm, st, x0):
    """State fields the step passes through unchanged (output variable IS the input variable in the step's jaxpr):
    time-constant inputs carried in the State (runoff, sIceLoad without sea ice). Their cotangent is the gradient with
    respect to a constant field, accumulated over the window (it grows linearly with no amplification at all), so the
    amplification screen leaves them out."""
    st = ck.prepare_state(step, fm, st, x0)          # the carry structure the scan uses (fields in = fields out)
    closed = jax.make_jaxpr(step)(fm, st, x0)
    n_fm = len(jax.tree.leaves(fm))
    keys = st.tree_flatten()[1]
    ins = closed.jaxpr.invars[n_fm:n_fm + len(keys)]
    outs = closed.jaxpr.outvars[:len(keys)]
    return sorted(k for k, i, o in zip(keys, ins, outs) if o is i)


def screen_stats(ft, const, chunk):
    """Amplification of the per-field cotangent-norm trace `ft` (seed first) over the dynamic fields (all but the
    carried constants), theta alone and the prognostic fields. Left out: the seed chunk (theta/hFacC seed -> whole
    State: a change of norm between field sets, not growth) and, for the dynamic norm, the chunk ending at the window
    start (iteration 1 starts AB2, mom_StartAB: gu/gvNm_2 are not read there, so their cotangent is 0 at the start)."""
    out = {}
    for name, sel, lo, hi in (("dynamic", lambda k: k not in const, 1, -1), ("theta", lambda k: k == "theta", 1, None),
                              ("prognostic", lambda k: k in PROGNOSTIC, 1, None)):
        tr = [float(np.sqrt(sum(v ** 2 for k, v in f.items() if sel(k)))) for f in ft]
        if len(tr[lo:hi]) >= 2:
            out[name] = dict(gr.amplification(tr[lo:hi], chunk)._asdict(), trace=tr)
    return out


def action_screen(E, mode, out, log):
    a = E.a
    step, adj = E.step(mode)
    theta = E.controls()
    xs_fn, nch = gr.chunks_of(E.xs, a.chunk)
    t = time.time()
    o = gr.chunked_value_and_grad(step, theta, E.fm0, E.st0, n_chunks=nch, chunk_steps=a.chunk, xs_fn=xs_fn,
                                  final_cost=E.final_cost, cost=None, schedule=a.schedule, init_fn=E.init_fn,
                                  params_fn=E.params_fn, boundary_stride=a.stride, log=log)
    amp = gr.amplification(o.trace, a.chunk)
    # the first chunk maps the theta-only seed onto the whole State (units differ per field): its "rate" is a change
    # of norm, not growth; the screen proper starts at the first chunk boundary before the window end
    amp_prop = gr.amplification(o.trace[1:], a.chunk)
    ft = o.field_trace or []
    const = carried_constants(step, E.params_fn(theta, E.fm0), E.st0, jax.tree.map(lambda v: v[0], xs_fn(0)))
    sub = screen_stats(ft, const, a.chunk)
    log(f"screen {mode}: carried constants {const}; " + "; ".join(
        f"{k}: median {v['median']:.5f} spread {v['log_spread']:.4f} worst3 {v['worst3']:.5f} passes {v['passes']}"
        for k, v in sub.items()))
    top = {}
    if ft:
        last = ft[-1]
        top = dict(sorted(last.items(), key=lambda kv: -kv[1])[:8])
    g = {k: np.asarray(v) for k, v in o.grad.items()}
    save_grad(out, f"screen_{mode}_{a.days}d", g, E.L)
    mem = gr.gpu_memory_gb()
    row = E.rowbase("screen", mode)
    row.update(J_end=o.loss, wall_s=time.time() - t, forward_s=o.forward_seconds, reverse_s=o.reverse_seconds,
               host_gb=o.host_gb, peak_gb=mem["peak"] if mem else None, trace=o.trace,
               amplification=amp._asdict(), amplification_after_seed=amp_prop._asdict(), amplification_fields=sub,
               carried_constants=const,
               field_trace=ft, top_fields_at_start=top, grad_norm={k: float(np.linalg.norm(v)) for k, v in g.items()},
               grad_finite=bool(all(np.all(np.isfinite(v)) for v in g.values())))
    emit(out, row, log)


def objective_fn(E, mode):
    step, _ = E.step(mode)
    return jax.jit(gr._objective(step, schedule="none", segments=None, cost=E.cost, final_cost=None,
                                 init_fn=E.init_fn, params_fn=E.params_fn))


def action_tl(E, mode, out, log):
    """Tangent-linear derivatives (jax.jvp through the whole window, schedule none) along every named direction and a
    combined direction at amplitudes 1 and 1e-6, against <grad, v> of the saved adjoint gradient of the same mode."""
    a = E.a
    step, _ = E.step(mode)
    J = gr._objective(step, schedule="none", segments=None, cost=E.cost, final_cost=None, init_fn=E.init_fn,
                      params_fn=E.params_fn)
    xs_d = jax.device_put(E.xs)
    th = E.controls()
    # model and state as jit ARGUMENTS (closed over they become constants XLA folds: minutes of compile, GB of
    # executable)
    tl = jax.jit(lambda t, v, fm, st, x: jax.jvp(lambda q: J(q, fm, st, x), (t,), (v,)))
    gpath = Path(out) / f"grad_{mode}_{a.days}d_r0.npz"
    gad = None
    if gpath.exists():
        z = np.load(gpath)
        L = E.L
        gad = {}
        for k in z.files:
            full = np.zeros(np.shape(E.theta0[k]))
            full[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = z[k]
            gad[k] = full
    dirs = dict(E.dirs)
    rng = np.random.default_rng(a.seed)
    comb = {k: np.zeros(np.shape(v)) for k, v in th.items()}
    for n, d in E.dirs.items():                      # combined: sum of the named directions with random weights
        w = rng.uniform(0.5, 1.5)
        for k in comb:
            comb[k] = comb[k] + w * np.asarray(d[k]) * (1e-2 if k == "kapGM" else 1.0)
    dirs["combined"] = comb
    for n, d in dirs.items():
        d = {k: jnp.asarray(d[k]) for k in th}
        for amp in ((1.0, 1e-6) if n == "combined" else (1.0,)):
            t = time.time()
            J0, dJ = tl(th, jax.tree.map(lambda v: v * amp, d), E.fm0, E.st0, xs_d)
            dJ = float(dJ) / amp
            row = E.rowbase("tl", mode)
            row.update(direction=n, amp=amp, J=float(J0), tl=dJ, wall_s=time.time() - t, seed=a.seed)
            if gad is not None:
                adv = dir_dot(gad, {k: np.asarray(v) for k, v in d.items()})
                row.update(adjoint=adv, rel=abs(dJ - adv) / max(abs(dJ), abs(adv), 1e-300))
            emit(out, row, log)


def action_fd(E, mode, out, log):
    """Central differences along the named directions (--dirs), h-sweep (--hs or H_DEFAULT per control kind). The
    forward noise floor: spread of --fd-repeats evaluations of J at the base point."""
    a = E.a
    Jf = objective_fn(E, mode)
    xs_d = jax.device_put(E.xs)
    th = E.controls()
    t = time.time()
    j0 = [float(Jf(th, E.fm0, E.st0, xs_d)) for _ in range(max(1, a.fd_repeats))]
    t_first = time.time() - t
    t = time.time()
    float(Jf(th, E.fm0, E.st0, xs_d))
    t_fwd = time.time() - t
    spread = max(j0) - min(j0)
    row = E.rowbase("fd_base", mode)
    row.update(J=j0, spread=spread, forward_s=t_fwd, first_calls_s=t_first)
    emit(out, row, log)
    names = a.dirs.split(",") if a.dirs else list(E.dirs)
    for n in names:
        d = {k: jnp.asarray(E.dirs[n][k]) for k in th}
        kind = next(k for k in H_DEFAULT if n.startswith(k))
        hs = [float(h) for h in a.hs.split(",")] if a.hs else H_DEFAULT[kind]
        for h in hs:
            t = time.time()
            jp = float(Jf(jax.tree.map(lambda x, y: x + h * y, th, d), E.fm0, E.st0, xs_d))
            jm = float(Jf(jax.tree.map(lambda x, y: x - h * y, th, d), E.fm0, E.st0, xs_d))
            row = E.rowbase("fd", mode)
            row.update(direction=n, where=E.where[n], h=h, Jp=jp, Jm=jm, J0=j0[0], fd=(jp - jm) / (2.0 * h),
                       noise_over_h=spread / h, ulp_over_h=float(np.spacing(j0[0])) / h, wall_s=time.time() - t)
            emit(out, row, log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (created; must not exist unless --append)")
    ap.add_argument("--append", action="store_true", help="write into an existing --out (a later stage of one run)")
    ap.add_argument("--rundir", type=Path, default=RUNDIR)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--actions", default="grad", help="comma list: cache,grad,fwdcheck,screen,tl,fd")
    ap.add_argument("--mode", default="ecco", choices=("exact", "ecco"),
                    help="backward semantics (default ecco: Nikolay 2026-09-23; exact available)")
    ap.add_argument("--j-scaling", default="literal", choices=("literal", "kelvin"),
                    help="literal: J as the Fortran adjsen set-up (double division by the box volume, K/m^3, with a "
                         "warning); kelvin: box-mean theta in K")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--chunk", type=int, default=24, help="steps per chunk (chunked reverse accumulation)")
    ap.add_argument("--stride", type=int, default=1, help="boundary stride (chunk boundaries kept on the host)")
    ap.add_argument("--schedule", default="step", help="in-chunk remat schedule: step | sqrt | none")
    ap.add_argument("--unroll", type=int, default=5, help="Cg2dParams.sum_unroll (bitwise; 5 for GPU)")
    ap.add_argument("--controls", default="theta,kapGM,tflux,taux,tauy")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--dirs", default="", help="fd: comma list of named directions (default all)")
    ap.add_argument("--hs", default="", help="fd: comma list of step sizes (default per control kind)")
    ap.add_argument("--fd-repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=a.append)
    logf = open(out / f"log_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}.txt", "a")

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"multiweek_grad: {' '.join(sys.argv)}; devices {jax.devices()}; XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    acts = a.actions.split(",")
    if "cache" in acts:
        build_cache(a.rundir, a.cache, log)
        acts = [x for x in acts if x != "cache"]
        if not acts:
            return 0
    E = Experiment(a, log)
    log(f"named directions: {json.dumps(E.where)}")
    for act in acts:
        t = time.time()
        if act == "grad":
            action_grad(E, a.mode, out, log)
        elif act == "fwdcheck":
            action_forward_check(E, out, log)
        elif act == "screen":
            action_screen(E, a.mode, out, log)
        elif act == "tl":
            action_tl(E, a.mode, out, log)
        elif act == "fd":
            action_fd(E, a.mode, out, log)
        else:
            raise ValueError(act)
        gr.memory_note(f"after {act} ({time.time() - t:.0f} s)", log)
        gr._CHUNK_CACHE.clear()      # one heavy stage per process where possible; at least free its executables
        jax.clear_caches()
    return 0


if __name__ == "__main__":
    sys.exit(main())
