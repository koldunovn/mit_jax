#!/usr/bin/env python3
"""Plan M2 / Task 22: multi-week gradient of the FULL V4r4 model (EXF bulk formulae + sea ice) on LLC90, the M2
adjoint acceptance; the full-tree counterpart of Task 21 (scripts/adjoint/multiweek_grad.py, flux-forced), whose
helpers it imports unchanged (the flux-forced driver is not modified).

Model: the full-tree production run directory (useCTRL=T with the V4r4 initial-state and mixing adjustments, bulk
formulae, SEAICE_MODEL; 1992-01-01, nIter0 = 1, dt = 3600 s), initial state from init.state_from_pickup cached once
(multiweek_grad.py --actions cache --rundir <full run dir> --cache <dir>, CPU with the gate XLA flags: bitwise the
Fortran start-of-run state, sea ice included), EXF inputs from ExfFullRecordLoader (+ zenith_time).

Cost (J, scalar):
    J = J_theta + w_ice * J_ice
    J_theta = (1/N) sum_{n=1..N} sum_box theta_n hFacC_n drF rA / sum_box h0FacC drF rA        [K]
              the Task 21 adjsen box-mean theta (running mean over the window; K-normalised = multiweek_grad
              --j-scaling kelvin; the box is the adjsen box as written, 120E-180E, 5N-16N, levels 15-20)
    J_ice   = sum_{yC > 70N} HEFF_N rA / sum_{yC > 70N} rA                                         [m]
              Arctic ice volume at the END of the window divided by the fixed ocean area north of 70N (surface wet
              interior points), i.e. the Arctic-mean effective ice thickness (the sea-ice-only study, scripts/adjoint/
              seaice_window.py, uses the same region for its J1). w_ice = --ice-weight (default 1).
The two parts live in different regions (tropical Pacific box, Arctic): the directions below test one or the other.

Controls (one pytree; zero = the production forward, value-identical):
    theta   [T,Nr,ny,nx] initial theta (interior) + EXCH_XYZ_RL: the xx_theta path (as Task 21)
    kapGM   [T,Nr,ny,nx] GM diffusivity (interior) + EXCH_XYZ_RL: the xx_kapgm path (as Task 21)
    heff    [T,ny,nx]    initial HEFF (interior) + EXCH_XY_RL (not a V4r4 control: gives the sea-ice sensitivity map)
    atemp, aqh, tauu, tauv, swdown, lwdown, precip   [T,ny,nx]: time-constant adjustments of the EXF atmospheric
            state -- the V4r4 gentim2d control set (data.ctrl.iter0.inclatmctrl: xx_atemp, xx_aqh, xx_tauu, xx_tauv,
            xx_swdown, xx_lwdown, xx_precip) -- in the units and sign of the EXF field after EXF_SET_FLD (EXF_FIELDS.h;
            atemp K, aqh kg/kg, tauu/tauv N/m^2 eastward/northward stress on the ocean before the A-grid rotation,
            swdown/lwdown W/m^2, precip m/s). They are added to BOTH record buffers of every step (as Task 21 did for
            TFLUX), divided by exf_inscal_<field> (data.exf: -1 for ustress, vstress, swdown, lwdown), so that the
            interpolated, scaled field (exf_set_fld.F:287-289, weights summing to 1) moves by the adjustment. They
            enter before EXF_RADIATION / EXF_WIND / EXF_BULKFORMULAE, as ctrl_map_gentim2d does for xx_atemp, xx_aqh,
            xx_swdown, xx_lwdown, xx_precip (exf_getffields.F). DIFFERENCE for tauu/tauv: the Fortran adds xx_tauu /
            xx_tauv in EXF_GETSURFACEFLUXES, after EXF_WIND; the buffer adjustment also enters EXF_WIND (wStress,
            the wind direction cw/sw -> uwind/vwind of the sea-ice air drag).

Adjoint modes (--mode; forward values identical in all, checked by `fwdcheck`):
    ecco         AdjointConfig.ecco(nml): what the TAF adjoint of V4r4 computes (seaice "ecco" = SEAICE_MODEL skipped
                 in the reverse sweep, ggl90 frozen, salt_plume off, gm_sigma stable, cg2d passive, viscFacInAd 1)
    exact_nodyn  AdjointConfig(seaice="no_dynamics"): exact ocean, sea-ice thermodynamics/advection adjoint, the LSR
                 dynamics skipped in reverse (c66g SEAICEuseDYNAMICSswitchInAd)
    exact_full   AdjointConfig(seaice="full"): the exact derivative (LSR by its implicit custom_jvp). NOTE: the implicit
                 derivative is that of the exactly solved LSR system; the production forward stops at LSR_ERROR = 2e-4
                 (seaice_lsr.py), so FD of the production forward differs from it where the ice dynamics matter.
    exact_iceecco AdjointConfig(): exact ocean, sea ice "ecco" (for completeness)

Actions (one process runs a list of them):
    grad      chunked gradient of J (--repeats), saves grad_<mode>_<days>d_r<r>.npz (interior, float64)
    fwdcheck  chunked forward of every --fwd-modes mode over the window: final State and J compared bitwise
    screen    chunked gradient of the terminal cost (end-of-window box mean + w_ice J_ice) alone: per-chunk per-field
              State-cotangent norms; statistics per field group (dynamic, theta, prognostic, seaice), Task 21 recipe
    tl        jax.jvp along --tl-dirs (default all named directions + the combined one at amplitudes 1 and 1e-6)
              against <grad, v> of the saved adjoint gradient of the same mode
    fd        central differences along --dirs, h-sweep (H_DEFAULT per control kind or --hs), forward noise floor
Sharded (--nproc P > 1): the model on P devices of one node (parallel/shard.ShardedModel, shard_map over tiles, the
same kernels); actions grad and fwdcheck; gradients are unpadded and saved like the 1-device ones (grad_<mode>_<days>
d_r<r>.npz in the --out of that run) for comparison with the 1-device gradients (summary script).

    python scripts/adjoint/fullgrad.py --out DIR --days 7 --mode ecco --actions fwdcheck,grad,screen --repeats 3
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multiweek_grad as mw  # noqa: E402  (also puts the repo root on sys.path and enables x64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mitgcm_jax.adjoint import checkpoint as ck  # noqa: E402
from mitgcm_jax.adjoint import grad as gr  # noqa: E402
from mitgcm_jax.adjoint.modes import AdjointConfig  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.state import State  # noqa: E402

RUNDIR = Path("/work/ab0995/a270088/MIT/reference/runs/ref_full_serial13_1day")
CACHE = Path("/work/ab0995/a270088/MIT/runs/adjoint_m2/init_ref_full_serial13_1day")
ARCTIC_LAT = 70.0            # J_ice region: yC > 70N (as J1 of scripts/adjoint/seaice_window.py)
# control name -> EXF record-buffer name (pkgs/exf_full.FIELDS)
FORCING = {"atemp": "atemp", "aqh": "aqh", "tauu": "ustress", "tauv": "vstress", "swdown": "swdown",
           "lwdown": "lwdown", "precip": "precip"}
SEAICE_FIELDS = ("AREA", "HEFF", "HSNOW", "TICES", "UICE", "VICE")
PROGNOSTIC = mw.PROGNOSTIC
MODES = ("ecco", "exact_nodyn", "exact_full", "exact_iceecco")
H_DEFAULT = {"theta": (1e-1, 1e-2, 1e-3, 1e-4), "kapGM": (1e-1, 1e-2, 1e-3, 1e-4), "heff": (1e-1, 1e-2, 1e-3, 1e-4),
             "atemp": (1e-1, 1e-2, 1e-3, 1e-4), "aqh": (1e-3, 1e-4, 1e-5, 1e-6), "tauu": (1e-2, 1e-3, 1e-4, 1e-5)}


def git_head():
    """HEAD of this checkout: git if available (not on the dolpung nodes), else read from the git files."""
    h = mw.git_head()
    if h:
        return h
    try:
        root = mw.ROOT
        gd = root / ".git"
        if gd.is_file():                                      # worktree: "gitdir: <path>"
            gd = Path(gd.read_text().split(":", 1)[1].strip())
        head = (gd / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(" ", 1)[1]
        common = gd / "commondir"
        base = (gd / common.read_text().strip()).resolve() if common.exists() else gd
        for d in (gd, base):
            if (d / ref).exists():
                return (d / ref).read_text().strip()[:7]
        for line in (base / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    except Exception:  # noqa: BLE001
        return None
    return None


def mode_config(mode, nml):
    if mode == "ecco":
        return AdjointConfig.ecco(nml)
    if mode == "exact_nodyn":
        return AdjointConfig(seaice="no_dynamics")
    if mode == "exact_full":
        return AdjointConfig(seaice="full")
    if mode == "exact_iceecco":
        return AdjointConfig()
    raise ValueError(f"mode {mode!r}; one of {MODES}")


class FullExperiment:
    """Model, initial state, window forcing, cost, controls and named directions of the full tree (optionally
    sharded over --nproc devices)."""

    def __init__(self, a, log):
        self.a, self.log = a, log
        self.nml = RunNamelists(a.rundir)
        P, g, ex, kLowC, st0 = mw.load_model(a.rundir, a.cache, a.unroll, log)
        if P.exfb is None or P.seaice is None:
            raise ValueError(f"{a.rundir}: not a full-tree run directory (bulk-formula EXF + sea ice)")
        if a.lsr_impl:
            P = P._replace(seaice=dataclasses.replace(
                P.seaice, dyn=dataclasses.replace(P.seaice.dyn, lsr_impl=a.lsr_impl)))
        # strongly typed State (a weakly typed field would compile a second program; run_jax.py does the same)
        st0 = State({k: jnp.asarray(v, dtype=np.asarray(v).dtype) for k, v in st0.f.items()}, st0.it)
        self.P, self.g, self.ex, self.kLowC = P, g, ex, kLowC
        L = self.L = g.layout
        self.N = int(round(a.days * 24 * 3600 / float(self.nml.get("data", "parm03", "deltaTClock"))))
        if self.N % a.chunk:
            raise ValueError(f"{self.N} steps is not a multiple of --chunk {a.chunk}")
        t0 = time.time()
        it0 = int(st0.it)
        self.xs = ck.exf_window(ck.exf_loader_at(P, g, a.rundir, self.nml, it0), self.nml, it0, self.N)
        log(f"EXF window: {self.N} steps from it={it0}, {sum(v.nbytes for v in jax.tree.leaves(self.xs)) / 1e9:.2f}"
            f" GB host, {time.time() - t0:.1f} s")
        # ---- cost weights (numpy, [nTiles, ...])
        m = mw.box_mask(g)
        self.box = m
        drF, rA, h0 = np.asarray(g.drF), np.asarray(g.rA), np.asarray(g.h0FacC)
        vol0 = float(np.sum(m * h0 * drF[None, :, None, None] * rA[:, None]))
        self.box_vol = vol0
        W = m * drF[None, :, None, None] * rA[:, None] / vol0                  # K-normalised (--j-scaling kelvin)
        inner2 = mw.interior_mask(L, False) > 0
        wet2 = (np.asarray(g.maskC)[:, 0] > 0) & inner2
        arc = wet2 & (np.asarray(g.yC) > ARCTIC_LAT)
        self.arctic = arc
        self.arctic_area = float(np.sum(rA * arc))
        Wice = rA * arc / self.arctic_area * float(a.ice_weight)
        log(f"box: {int(m.sum())} wet cells, volume {vol0:.4e} m^3; Arctic (yC > {ARCTIC_LAT}N): {int(arc.sum())} "
            f"surface cells, {self.arctic_area:.4e} m^2; ice weight {a.ice_weight}")
        # forcing-control scaling: the buffers get d / exf_inscal_<field>
        self.fscale = {}
        for k, buf in FORCING.items():
            s = float(np.asarray(getattr(P.exfb, f"inscal_{buf}")))
            if s == 0.0:
                raise ValueError(f"exf_inscal_{buf} = 0")
            self.fscale[k] = 1.0 / s
        log(f"forcing controls: buffer scale 1/exf_inscal = {self.fscale}")
        gpath = Path(a.out) / "grid.npz"
        if not gpath.exists():   # for the maps (the nereus env has no JAX)
            I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
            np.savez(gpath, xC=np.asarray(g.xC)[:, J, I], yC=np.asarray(g.yC)[:, J, I],
                     maskC=np.asarray(g.maskC)[..., J, I], box=m[..., J, I], rA=rA[:, J, I], drF=drF,
                     h0FacC=h0[..., J, I], rC=np.asarray(g.rC), kapGM=np.asarray(g.kapGM)[..., J, I],
                     arctic=arc[:, J, I], HEFF0=np.asarray(st0.f["HEFF"])[:, J, I],
                     AREA0=np.asarray(st0.f["AREA"])[:, J, I])
        # ---- placement: one device, or P devices (shard_map over tiles)
        int3, int2 = mw.interior_mask(L, True), mw.interior_mask(L, False)
        self.sm = None
        if a.nproc > 1:
            from jax.sharding import NamedSharding, PartitionSpec
            from mitgcm_jax.parallel.shard import ShardedModel
            from mitgcm_jax.parallel.sharded_exchange import AXIS
            sm = self.sm = ShardedModel(g, a.nproc)
            ax = PartitionSpec(AXIS)
            self._bufsh = NamedSharding(sm.mesh, PartitionSpec(None, AXIS))
            self._rep = NamedSharding(sm.mesh, PartitionSpec())
            real = np.zeros(sm.blocks.Tpad, bool)
            real[:L.nTiles] = True

            def put(x, zero_pad=False):      # [nTiles, ...] numpy -> padded, placed (padding tiles zeroed if asked)
                p = sm.blocks.pad(np.asarray(x))
                if zero_pad:
                    p = p * real.reshape((-1,) + (1,) * (p.ndim - 1))
                return jax.device_put(p, sm._tile)

            self.put = put
            self.model = ck.Model(P, sm.g, sm.shard_tiles(kLowC), sm.ex)
            self.st0 = State({k: put(v) for k, v in st0.f.items()}, st0.it)
            sh_exch = jax.jit(jax.shard_map(lambda e, x: e.exch_xy(x), mesh=sm.mesh, in_specs=(ax, ax),
                                            out_specs=ax, check_vma=True))
            ex_d = sm.ex
            exch = lambda x: sh_exch(ex_d, x)           # noqa: E731
            self.W = put(W, True)
            self.Wice = put(Wice, True)
            int3d, int2d = put(int3, True), put(int2, True)
            log(f"sharded: {a.nproc} devices, {sm.blocks.Tpad} padded tiles ({sm.blocks.Tloc} per device)")
        else:
            self.put = lambda x, zero_pad=False: jnp.asarray(x)   # noqa: E731
            self.model = ck.Model(P, g, kLowC, ex)
            self.st0 = st0
            exch = ex.exch_xy
            self.W, self.Wice = jnp.asarray(W), jnp.asarray(Wice)
            int3d, int2d = jnp.asarray(int3), jnp.asarray(int2)
        self.fm0 = mw.FModel(self.model, {})
        N, Wd, Wid = self.N, self.W, self.Wice
        fscale = self.fscale

        def cost(fm, st, x):
            return jnp.sum(Wd * st.theta * st.hFacC) / N

        def ice_cost(st):
            return jnp.sum(Wid * st.HEFF)

        def final_cost(fm, st):
            return ice_cost(st)

        def screen_cost(fm, st):
            return jnp.sum(Wd * st.theta * st.hFacC) + ice_cost(st)

        def init_fn(th, st):
            rep = {}
            if "theta" in th:
                rep["theta"] = exch(st.theta + int3d * th["theta"])     # CTRL_MAP_INI_GENARR + EXCH_XYZ_RL
            if "heff" in th:
                rep["HEFF"] = exch(st.HEFF + int2d * th["heff"])         # (+ EXCH_XY_RL, as SEAICE exchanges HEFF)
            return st.replace(**rep) if rep else st

        def params_fn(th, fm):
            m_, gg = fm.m, fm.m.g
            if "kapGM" in th:
                gg = gg.replace(kapGM=exch(gg.kapGM + int3d * th["kapGM"]))
            fc = {FORCING[k]: (int2d * th[k]) * fscale[k] for k in FORCING if k in th}
            return mw.FModel(m_._replace(g=gg), fc)

        self.cost, self.final_cost, self.screen_cost = cost, final_cost, screen_cost
        self.init_fn, self.params_fn = init_fn, params_fn
        z3, z2 = np.zeros(L.shape3d), np.zeros(L.shape2d)
        self.theta0_np = {"theta": z3, "kapGM": z3, "heff": z2, **{k: z2 for k in FORCING}}
        self.theta0 = {k: self.put(v) for k, v in self.theta0_np.items()}
        self.dirs = self.directions(np.asarray(st0.f["HEFF"]))
        self._steps = {}

    # ------------------------------------------------------------------ helpers
    def step(self, mode):
        if mode not in self._steps:
            adj = mode_config(mode, self.nml)
            if self.sm is None:
                fstep = mw.make_fstep(adj)
            else:
                fstep = make_sharded_fstep(self.sm, adj, tuple(sorted(self.xs)))
            self._steps[mode] = (fstep, adj)
        return self._steps[mode]

    def xs_fn_nch(self):
        """xs_fn (chunk c -> stacked inputs, placed) and the chunk count."""
        xs_fn, nch = gr.chunks_of(self.xs, self.a.chunk)
        if self.sm is None:
            return xs_fn, nch
        src = self.sm.blocks.source_tile

        def placed(c):
            x = xs_fn(c)
            out = {k: jax.tree.map(lambda v: jax.device_put(v, self._rep), v) for k, v in x.items() if k != "bufs"}
            out["bufs"] = {k: tuple(jax.device_put(np.ascontiguousarray(b[:, src]), self._bufsh) for b in v)
                           for k, v in x["bufs"].items()}
            return out
        return placed, nch

    def unpad(self, tree):
        """numpy [nTiles, ...] of a (possibly padded) pytree of arrays."""
        n = self.L.nTiles
        return {k: np.asarray(v)[:n] for k, v in tree.items()}

    def point(self, lon, lat, k):
        L, g = self.L, self.g
        xc, yc, mC = np.asarray(g.xC), np.asarray(g.yC), np.asarray(g.maskC)[:, k]
        inner = mw.interior_mask(L, False) > 0
        d = (((xc - lon + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(lat))) ** 2 + (yc - lat) ** 2
        d = np.where(inner & (mC > 0), d, np.inf)
        t, j, i = np.unravel_index(np.argmin(d), d.shape)
        return int(t), int(k), int(j), int(i)

    def directions(self, heff0):
        """Named control directions (numpy pytrees like theta0_np, [nTiles, ...]) for FD / TL."""
        z3, z2 = np.zeros(self.L.shape3d), np.zeros(self.L.shape2d)
        out, where = {}, {}
        pA = self.point(150.0, 10.5, 16)            # box centre, level 17 (Task 21)
        pB = self.point(150.0, 10.5, 13)            # same column, level 14: just above the box top (outside)
        pC = self.point(150.0, 3.5, 16)             # level 17, south of the box (outside)
        for name, p in (("theta_A_centre", pA), ("theta_B_above", pB), ("theta_C_south", pC)):
            d = z3.copy()
            d[p] = 1.0
            out[name] = {"theta": d}
            where[name] = dict(point=p, lon=float(np.asarray(self.g.xC)[p[0], p[2], p[3]]),
                               lat=float(np.asarray(self.g.yC)[p[0], p[2], p[3]]), in_box=bool(self.box[p]))
        kap = np.asarray(self.g.kapGM) * mw.interior_mask(self.L, True) * (np.asarray(self.g.maskC) > 0)
        out["kapGM_scale"] = {"kapGM": kap}
        where["kapGM_scale"] = dict(desc="global relative change of kapGM (direction = kapGM itself)")
        foot = np.any(self.box, axis=1).astype(float) * mw.interior_mask(self.L, False)
        out["atemp_box"] = {"atemp": foot}
        where["atemp_box"] = dict(desc="uniform +1 K atemp over the box surface footprint", ncols=int(foot.sum()))
        out["tauu_box"] = {"tauu": foot}
        where["tauu_box"] = dict(desc="uniform +1 N/m^2 eastward stress (EXF ustress before the A-grid rotation) over "
                                      "the box surface footprint", ncols=int(foot.sum()))
        arc = self.arctic.astype(float)
        out["atemp_arctic"] = {"atemp": arc}
        where["atemp_arctic"] = dict(desc=f"uniform +1 K atemp over the ocean north of {ARCTIC_LAT}N",
                                     ncols=int(arc.sum()))
        out["heff_arctic"] = {"heff": heff0 * arc}
        where["heff_arctic"] = dict(desc=f"relative change of the initial HEFF north of {ARCTIC_LAT}N (direction = "
                                         f"HEFF0 there)", ncols=int(arc.sum()))
        self.where = where
        full = {}
        for n, d in out.items():
            t = {k: np.zeros_like(v) for k, v in self.theta0_np.items()}
            t.update(d)
            full[n] = t
        return full

    def rowbase(self, action, mode):
        adj = self.step(mode)[1]
        return dict(action=action, tree="full", mode=mode, freezes=dataclasses.asdict(adj), days=self.a.days,
                    nsteps=self.N, chunk=self.a.chunk, stride=self.a.stride, schedule=f"chunked/{self.a.schedule}",
                    unroll=self.a.unroll, nproc=self.a.nproc, ice_weight=self.a.ice_weight, arctic_lat=ARCTIC_LAT,
                    job=os.environ.get("SLURM_JOB_ID"), git=git_head(),
                    device=str(jax.devices()[0].device_kind), platform=jax.devices()[0].platform,
                    ndevices=len(jax.devices()), xla_flags=os.environ.get("XLA_FLAGS", ""), rundir=str(self.a.rundir),
                    box_vol=self.box_vol, arctic_area=self.arctic_area, j_scaling="kelvin",
                    lsr_impl=self.P.seaice.dyn.lsr_impl)

    def controls(self):
        return {k: self.theta0[k] for k in self.a.controls.split(",")}


def make_sharded_fstep(sm, adj, exf_keys):
    """(FModel, State, x) -> State: FORWARD_STEP inside shard_map over the tiles (ShardedModel.step_fn, compiled once
    per mode), with the forcing-control buffer adjustments added in the global (padded) view first."""
    fn = sm.step_fn(exf_keys, adj)

    def step(fm, st, x):
        if fm.fc:
            bufs = dict(x["bufs"])
            for name, d in fm.fc.items():
                b0, b1 = bufs[name]
                bufs[name] = (b0 + d, b1 + d)
            x = dict(x, bufs=bufs)
        m = fm.m
        f1, it1, _ = fn(m.P, m.g, m.ex, m.kLowC, st.f, st.it, x)
        return State(f1, it1)
    return step


# ---------------------------------------------------------------------------------------------------------------
# actions


def action_grad(E, mode, out, log):
    a = E.a
    step, adj = E.step(mode)
    theta = E.controls()
    xs_fn, nch = E.xs_fn_nch()
    res = []
    for r in range(a.repeats):
        t = time.time()
        o = gr.chunked_value_and_grad(step, theta, E.fm0, E.st0, n_chunks=nch, chunk_steps=a.chunk, xs_fn=xs_fn,
                                      cost=E.cost, final_cost=E.final_cost, schedule=a.schedule, init_fn=E.init_fn,
                                      params_fn=E.params_fn, boundary_stride=a.stride, log=log)
        wall = time.time() - t
        g = E.unpad(o.grad)
        mw.save_grad(out, f"grad_{mode}_{a.days:g}d_r{r}", g, E.L)
        mem = gr.gpu_memory_gb()
        row = E.rowbase("grad", mode)
        row.update(repeat=r, J=o.loss, wall_s=wall, forward_s=o.forward_seconds, reverse_s=o.reverse_seconds,
                   host_gb=o.host_gb, peak_gb=mem["peak"] if mem else None,
                   grad_norm={k: float(np.linalg.norm(v)) for k, v in g.items()},
                   grad_finite=bool(all(np.all(np.isfinite(v)) for v in g.values())),
                   dirderiv={n: mw.dir_dot(g, d) for n, d in E.dirs.items()},
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


def emit(out, row, log):
    mw.emit(out, row, log)


def action_fwdcheck(E, out, log):
    """Forward values of every mode in --fwd-modes are the same bytes: chunked forward over the whole window, final
    State (every field) and J (running + terminal parts) compared with the first mode."""
    a = E.a
    xs_fn, nch = E.xs_fn_nch()
    modes = a.fwd_modes.split(",")
    finals = {}
    for mode in modes:
        step, _ = E.step(mode)
        fwd, _ = gr._chunk_fns(step, a.schedule, None, E.cost, E.params_fn, ck.SAVE_NAMES)
        th = E.controls()
        st = ck.prepare_state(step, E.params_fn(th, E.fm0), E.st0, jax.tree.map(lambda v: v[0], xs_fn(0)))
        carry = (jax.jit(E.init_fn)(th, st), jnp.zeros((), jnp.float64))
        t = time.time()
        for c in range(nch):
            carry = jax.block_until_ready(fwd(th, E.fm0, carry, xs_fn(c) if E.sm else jax.device_put(xs_fn(c))))
        Jice = float(jax.jit(lambda s: E.final_cost(None, s))(carry[0]))
        stf = E.unpad(carry[0].f)
        finals[mode] = (stf, float(carry[1]), Jice, time.time() - t)
        log(f"fwdcheck {mode}: J_theta {finals[mode][1]:.16e} J_ice {Jice:.16e} ({finals[mode][3]:.1f} s)")
        del carry
        gr._CHUNK_CACHE.clear()
    ref = modes[0]
    sr, Jr, Jir, _ = finals[ref]
    np.savez(Path(out) / f"fwd_final_{ref}_{a.days:g}d.npz",
             **{k: v for k, v in sr.items() if k in PROGNOSTIC + SEAICE_FIELDS})
    row = E.rowbase("forward", ref)
    row.update(J_theta=Jr, J_ice=Jir, forward_s=finals[ref][3])
    emit(out, row, log)
    for mode in modes[1:]:
        s, Jt, Ji, _ = finals[mode]
        ndiff = {k: int(np.count_nonzero(np.asarray(sr[k]) != np.asarray(s[k]))) for k in sr}
        row = E.rowbase("forward_check", mode)
        row.update(ref_mode=ref, J_theta_ref=Jr, J_theta=Jt, J_ice_ref=Jir, J_ice=Ji,
                   J_bitwise=bool(Jr == Jt and Jir == Ji), fields_differing={k: v for k, v in ndiff.items() if v},
                   nfields=len(ndiff), bitwise=bool(Jr == Jt and Jir == Ji and not any(ndiff.values())),
                   forward_s={m: finals[m][3] for m in modes})
        emit(out, row, log)


def screen_stats(ft, const, chunk):
    """Task 21 statistics (multiweek_grad.screen_stats) plus the sea-ice group."""
    out = mw.screen_stats(ft, const, chunk)
    tr = [float(np.sqrt(sum(v ** 2 for k, v in f.items() if k in SEAICE_FIELDS))) for f in ft]
    if len(tr[1:]) >= 2 and all(t > 0 for t in tr[1:]):
        out["seaice"] = dict(gr.amplification(tr[1:], chunk)._asdict(), trace=tr)
    return out


def action_screen(E, mode, out, log):
    a = E.a
    step, adj = E.step(mode)
    theta = E.controls()
    xs_fn, nch = E.xs_fn_nch()
    t = time.time()
    o = gr.chunked_value_and_grad(step, theta, E.fm0, E.st0, n_chunks=nch, chunk_steps=a.chunk, xs_fn=xs_fn,
                                  final_cost=E.screen_cost, cost=None, schedule=a.schedule, init_fn=E.init_fn,
                                  params_fn=E.params_fn, boundary_stride=a.stride, log=log)
    amp = gr.amplification(o.trace, a.chunk)
    amp_prop = gr.amplification(o.trace[1:], a.chunk)
    ft = o.field_trace or []
    xs0 = jax.tree.map(lambda v: v[0], xs_fn(0))
    const = mw.carried_constants(step, E.params_fn(theta, E.fm0), E.st0, xs0) if E.sm is None else []
    sub = screen_stats(ft, const, a.chunk)
    log(f"screen {mode}: carried constants {const}; " + "; ".join(
        f"{k}: median {v['median']:.5f} spread {v['log_spread']:.4f} worst3 {v['worst3']:.5f} passes {v['passes']}"
        for k, v in sub.items()))
    top = dict(sorted(ft[-1].items(), key=lambda kv: -kv[1])[:10]) if ft else {}
    g = E.unpad(o.grad)
    mw.save_grad(out, f"screen_{mode}_{a.days:g}d", g, E.L)
    mem = gr.gpu_memory_gb()
    row = E.rowbase("screen", mode)
    row.update(J_end=o.loss, wall_s=time.time() - t, forward_s=o.forward_seconds, reverse_s=o.reverse_seconds,
               host_gb=o.host_gb, peak_gb=mem["peak"] if mem else None, trace=o.trace,
               amplification=amp._asdict(), amplification_after_seed=amp_prop._asdict(), amplification_fields=sub,
               carried_constants=const, field_trace=ft, top_fields_at_start=top,
               grad_norm={k: float(np.linalg.norm(v)) for k, v in g.items()},
               grad_finite=bool(all(np.all(np.isfinite(v)) for v in g.values())))
    emit(out, row, log)


def load_grad(E, path):
    z = np.load(path)
    L = E.L
    out = {}
    for k in z.files:
        full = np.zeros(np.shape(E.theta0_np[k]))
        full[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = z[k]
        out[k] = full
    return out


def action_tl(E, mode, out, log):
    """Tangent-linear derivatives (jax.jvp through the whole window, schedule none) along --tl-dirs (default: every
    named direction + "combined" = the named directions with random weights, at amplitudes 1 and 1e-6), against
    <grad, v> of the saved adjoint gradient of the same mode and window (multiweek_grad.action_tl's recipe)."""
    a = E.a
    if E.sm is not None:
        raise NotImplementedError("tl: one device only")
    step, _ = E.step(mode)
    J = gr._objective(step, schedule="none", segments=None, cost=E.cost, final_cost=E.final_cost, init_fn=E.init_fn,
                      params_fn=E.params_fn)
    xs_d = jax.device_put(E.xs)
    th = E.controls()
    tl = jax.jit(lambda t, v, fm, st, x: jax.jvp(lambda q: J(q, fm, st, x), (t,), (v,)))
    gpath = Path(out) / f"grad_{mode}_{a.days:g}d_r0.npz"
    if a.grad_from:
        gpath = Path(a.grad_from)
    gad = load_grad(E, gpath) if gpath.exists() else None
    if gad is None:
        log(f"tl: no adjoint gradient at {gpath} (TL values only)")
    rng = np.random.default_rng(a.seed)
    comb = {k: np.zeros(np.shape(v)) for k, v in E.theta0_np.items()}
    for n, d in E.dirs.items():
        w = rng.uniform(0.5, 1.5)
        for k in comb:
            comb[k] = comb[k] + w * np.asarray(d[k]) * (1e-2 if k == "kapGM" else 1.0)
    dirs = dict(E.dirs)
    dirs["combined"] = comb
    names = a.tl_dirs.split(",") if a.tl_dirs else list(dirs)
    for n in names:
        d = {k: jnp.asarray(dirs[n][k]) for k in th}
        for amp in ((1.0, 1e-6) if n == "combined" else (1.0,)):
            t = time.time()
            J0, dJ = tl(th, jax.tree.map(lambda v: v * amp, d), E.fm0, E.st0, xs_d)
            dJ = float(dJ) / amp
            row = E.rowbase("tl", mode)
            row.update(direction=n, amp=amp, J=float(J0), tl=dJ, wall_s=time.time() - t, seed=a.seed)
            if gad is not None:
                adv = mw.dir_dot(gad, {k: np.asarray(v) for k, v in d.items()})
                row.update(adjoint=adv, rel=abs(dJ - adv) / max(abs(dJ), abs(adv), 1e-300), grad_file=str(gpath))
            emit(out, row, log)


def action_fd(E, mode, out, log):
    """Central differences along --dirs (default all named directions), h-sweep (--hs or H_DEFAULT per control
    kind). Forward noise floor: spread of --fd-repeats evaluations of J at the base point. The FD rows do not depend
    on the mode (forward only); the mode only names the compiled program."""
    a = E.a
    if E.sm is not None:
        raise NotImplementedError("fd: one device only")
    step, _ = E.step(mode)
    Jf = jax.jit(gr._objective(step, schedule="none", segments=None, cost=E.cost, final_cost=E.final_cost,
                               init_fn=E.init_fn, params_fn=E.params_fn))
    Jparts = jax.jit(gr._objective(step, schedule="none", segments=None, cost=None, final_cost=E.final_cost,
                                   init_fn=E.init_fn, params_fn=E.params_fn)) if a.fd_parts else None
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
    if Jparts is not None:
        row["J_ice"] = float(Jparts(th, E.fm0, E.st0, xs_d))
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
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--rundir", type=Path, default=RUNDIR)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--actions", default="grad", help="comma list: grad,fwdcheck,screen,tl,fd")
    ap.add_argument("--mode", default="ecco", choices=MODES)
    ap.add_argument("--fwd-modes", default="ecco,exact_nodyn,exact_full", help="fwdcheck: modes compared (first = ref)")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--schedule", default="step")
    ap.add_argument("--unroll", type=int, default=5, help="Cg2dParams.sum_unroll (bitwise; 5 for GPU)")
    ap.add_argument("--lsr-impl", default="", help="override SeaiceDynParams.lsr_impl (default: the setup's, auto)")
    ap.add_argument("--controls", default="theta,kapGM,heff,atemp,aqh,tauu,tauv,swdown,lwdown,precip")
    ap.add_argument("--ice-weight", type=float, default=1.0, help="w_ice in J = J_theta + w_ice J_ice")
    ap.add_argument("--nproc", type=int, default=1, help="devices (shard_map over tiles) for grad / fwdcheck")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--dirs", default="", help="fd: comma list of named directions (default all)")
    ap.add_argument("--hs", default="", help="fd: comma list of step sizes (default per control kind)")
    ap.add_argument("--fd-repeats", type=int, default=2)
    ap.add_argument("--fd-parts", action="store_true", help="fd: also record J_ice at the base point")
    ap.add_argument("--tl-dirs", default="", help="tl: comma list of directions (default all + combined)")
    ap.add_argument("--grad-from", default="", help="tl: adjoint gradient npz (default grad_<mode>_<days>d_r0.npz)")
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

    log(f"fullgrad: {' '.join(sys.argv)}; devices {jax.devices()}; XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    E = FullExperiment(a, log)
    log(f"named directions: {json.dumps(E.where)}")
    for act in a.actions.split(","):
        t = time.time()
        if act == "grad":
            action_grad(E, a.mode, out, log)
        elif act == "fwdcheck":
            action_fwdcheck(E, out, log)
        elif act == "screen":
            action_screen(E, a.mode, out, log)
        elif act == "tl":
            action_tl(E, a.mode, out, log)
        elif act == "fd":
            action_fd(E, a.mode, out, log)
        else:
            raise ValueError(act)
        gr.memory_note(f"after {act} ({time.time() - t:.0f} s)", log)
        gr._CHUNK_CACHE.clear()
        jax.clear_caches()
    return 0


if __name__ == "__main__":
    sys.exit(main())
