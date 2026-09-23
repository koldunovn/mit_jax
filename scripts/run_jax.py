#!/usr/bin/env python3
"""Run the JAX MITgcm (V4r4 flux-forced) forward model and write movie frames, %MON statistics and snapshots.

    python scripts/run_jax.py OUTDIR --rundir RUNDIR --init-oracle NAME [--init-it 1] --nsteps N
                              [--frame-every 6] [--monitor-every 1] [--snapshot-every 24]

RUNDIR: a Fortran run directory with the namelists and linked inputs (forcing files) of the configuration to run.
Initial state: --init-oracle pickup = built from the run directory's pickup like the Fortran initialisation
(mitgcm_jax/init.py, incl. the useCTRL=T control adjustments); or an oracle name = its S00_begin dump at --init-it. OUTDIR (new) gets:
  frames/frame_<n>.npz   sst (compact 1170x90), eta, iter, date   -> tools/animate_globe.py (nereus env)
  monitor.txt            %MON dynstat lines in the Fortran format (compare with STDOUT.0000)
  snap_<iter>.npz        theta, salt, etaN (compact, float32) every --snapshot-every steps (dumpFreq twin)
  state_final.npz        the full State (restart)
  means_<iter>.npz       with --means-every N: time means of theta, salt, etaN, uVel, vVel, sst (compact, float32)
                         over the steps since the previous mean (accumulated every step; it_first, it_last, nsteps)
  budgets.txt            with --budgets: per-step global volume/SSH/heat/salt budgets (%BUDGET lines,
                         mitgcm_jax/diagnostics/budgets.py) and their cumulative residuals at the end
Outputs are due at absolute iterations (it % every == 0), like Fortran's mod(myTime, freq) == 0 with deltaT = 3600 s
and startTime = nIter0*deltaT, so e.g. --monitor-every 24 matches a Fortran monitorFreq = 86400 run.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax  # noqa: E402

import mitgcm_jax  # noqa: E402,F401
from mitgcm_jax.core.forward_step import forward_step  # noqa: E402
from mitgcm_jax.diagnostics import budgets as budgets_mod  # noqa: E402
from mitgcm_jax.diagnostics import means as means_mod  # noqa: E402
from mitgcm_jax.diagnostics.monitor import dynstat, dynstat_device, format_dynstat  # noqa: E402
from mitgcm_jax.io.llc import tiles_to_compact  # noqa: E402
from mitgcm_jax.model import setup  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod  # noqa: E402
from mitgcm_jax.state import State, state_from_dump  # noqa: E402
from mitgcm_jax.tests import oracle  # noqa: E402


def interior(a, L):
    return np.asarray(a)[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx]


def compact(a, L):
    """Interior of a [tile, (k,) j, i] field as the MITgcm compact global array ((k,) 1170, 90): tiles_to_compact wants
    the tile axis at -3."""
    return tiles_to_compact(np.moveaxis(interior(a, L), 0, -3))


def model_date(nml, myTime):
    """Calendar date of model time myTime: data.cal startDate_1/_2 (yyyymmdd, hhmmss) is model time 0 (baseTime);
    with nIter0=1, deltaT=3600 iteration 1 is 1992-01-01 13:00 and iteration 12 the 1992-01-02 00:00 PO.DAAC
    snapshot (docs/REFERENCE_RUNS.md)."""
    d1 = int(nml.get("data.cal", "cal_nml", "startDate_1"))
    d2 = int(nml.get("data.cal", "cal_nml", "startDate_2", default=0))
    t0 = dt.datetime(d1 // 10000, d1 // 100 % 100, d1 % 100, d2 // 10000, d2 // 100 % 100, d2 % 100)
    return t0 + dt.timedelta(seconds=float(myTime))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir")
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--init-oracle", help="oracle name (start from its S00 dump) or 'pickup' (Fortran-free start)")
    ap.add_argument("--init-it", type=int, default=1)
    ap.add_argument("--nsteps", type=int, required=True)
    ap.add_argument("--frame-every", type=int, default=6)
    ap.add_argument("--monitor-every", type=int, default=1)
    ap.add_argument("--snapshot-every", type=int, default=24)
    ap.add_argument("--restart", help="state_final.npz of a previous run_jax run (continues its iteration count)")
    ap.add_argument("--frame-offset", type=int, default=0, help="number of the first frame file written")
    ap.add_argument("--host-monitor", action="store_true",
                    help="exact host-side (numpy) %%MON statistics incl. del2 (slow: copies the 3-D state to the host)")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="also write state_<iter>.npz every N steps")
    ap.add_argument("--cg2d-unroll", type=int, default=1,
                    help="Cg2dParams.sum_unroll: unroll the Fortran-order tile sums (bitwise identical; 5 on GPU)")
    ap.add_argument("--means-every", type=int, default=0,
                    help="write means_<iter>.npz: time means accumulated every step, every N steps (0: off)")
    ap.add_argument("--means-fields", default=",".join(means_mod.MEAN_FIELDS),
                    help="comma-separated fields for --means-every (State fields, or sst)")
    ap.add_argument("--budgets", action="store_true", help="per-step global budgets to budgets.txt")
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    rundir = Path(a.rundir)
    nml = RunNamelists(rundir)
    log = open(out / "run.log", "w")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"run_jax: {' '.join(sys.argv)}  devices={jax.devices()}")
    t0 = time.time()
    P, g, ex, kLowC = setup(rundir)
    if a.cg2d_unroll != 1:
        import dataclasses
        P = P._replace(cg=dataclasses.replace(P.cg, sum_unroll=a.cg2d_unroll))
    L = g.layout
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    nIter0 = int(nml.get("data", "parm03", "nIter0", default=0))
    if a.restart:
        z = np.load(a.restart)
        a.init_it = int(z["it"])
        st = State({k: jax.numpy.asarray(z[k]) for k in z.files if k != "it"}, jax.numpy.asarray(a.init_it))
        # the EXF record buffers of the Fortran run are those of a run started at nIter0: replay the host-side record
        # logic for the steps before the restart (reads only the records it needs, no model work)
        for it in range(nIter0, a.init_it):
            loader.load(*exf_mod.model_time(nml, it - nIter0 + 1))
        say(f"restart from {a.restart} at it={a.init_it}")
    elif a.init_oracle == "pickup":
        from mitgcm_jax.init import state_from_pickup   # plan Task 8/8b: from the run dir's pickup (+ ctrl)
        st = state_from_pickup(P, g, ex, kLowC, rundir)
        a.init_it = int(st.it)
        st = State({k: jax.numpy.asarray(v) for k, v in st.f.items()}, jax.numpy.asarray(a.init_it))
    else:
        ds = oracle.dumpset(a.init_oracle)
        st = state_from_dump(ds, a.init_it)
        st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, L)["runoff"]))
        st = State({k: jax.numpy.asarray(v) for k, v in st.f.items()}, jax.numpy.asarray(a.init_it))
    say(f"setup + initial state {time.time() - t0:.1f} s")
    step = jax.jit(lambda P, g, kLowC, st, exf_in: forward_step(P, g, ex, kLowC, st, exf_in)[0])
    mon_dev = jax.jit(dynstat_device)

    def monitor(st):
        if a.host_monitor:
            return dynstat(st, g, L)
        return jax.tree_util.tree_map(float, mon_dev(st, g))
    mon = open(out / "monitor.txt", "w")
    if a.means_every:
        mean_fields = tuple(f for f in a.means_fields.split(",") if f)
        means_acc = jax.jit(means_mod.means_accumulate)
        macc, m_first = means_mod.means_init(st, mean_fields), a.init_it + 1
    if a.budgets:
        budgets_mod.check_config(P)
        bud = jax.jit(budgets_mod.step_budget)
        bacc = budgets_mod.budget_acc_init()
        budf = open(out / "budgets.txt", "w")

    def frame(n, st, myTime):
        it = int(st.it)
        sst = tiles_to_compact(interior(st.theta[:, 0], L)).astype(np.float32)
        eta = tiles_to_compact(interior(st.etaN, L)).astype(np.float32)
        date = model_date(nml, myTime)
        np.savez(out / "frames" / f"frame_{n:05d}.npz", sst=sst, eta=eta, iter=it,
                 date=date.strftime("%Y-%m-%d %H:%M"))

    # myTime of the state at the start of iteration it: startTime + (it - nIter0)*deltaT
    t_state = exf_mod.model_time(nml, a.init_it - nIter0 + 1)[0]
    nframe = a.frame_offset
    if not a.restart:
        frame(nframe, st, t_state)
        mon.write(format_dynstat(monitor(st), a.init_it) + "\n")
        nframe += 1
    tstep = []
    for n in range(a.nsteps):
        it = a.init_it + n
        myTime, myIter = exf_mod.model_time(nml, it - nIter0 + 1)
        bufs, facs, _ = loader.load(myTime, myIter)
        t1 = time.time()
        st_prev = st
        st = step(P, g, kLowC, st, {"bufs": bufs, "facs": facs, "myTime": myTime})
        jax.block_until_ready(st.f["theta"])
        tstep.append(time.time() - t1)
        if a.budgets:
            b = bud(P, g, kLowC, st_prev, st)
            bacc = budgets_mod.budget_acc_add(bacc, b)
            budf.write(budgets_mod.format_budget(b, it + 1) + "\n")
            budf.flush()
        del st_prev
        if a.means_every:
            macc = means_acc(macc, st, 1.0)
            if (it + 1) % a.means_every == 0:
                m = means_mod.means_finish(macc)
                np.savez(out / f"means_{it + 1:010d}.npz", it_first=m_first, it_last=it + 1,
                         nsteps=int(float(macc["w"])),
                         **{k: compact(v, L).astype(np.float32) for k, v in m.items()})
                macc, m_first = means_mod.means_init(st, mean_fields), it + 2
        t_state = exf_mod.model_time(nml, it + 1 - nIter0 + 1)[0]
        if (it + 1) % a.monitor_every == 0:
            stats = monitor(st)
            mon.write(format_dynstat(stats, it + 1) + "\n")
            mon.flush()
            th = stats["theta"]
            say(f"it {it + 1:6d} {model_date(nml, t_state):%Y-%m-%d %H:%M} step {tstep[-1]:.2f} s  "
                f"theta mean {th['mean']:.10f} max {th['max']:.4f}  eta [{stats['eta']['min']:.3f}, "
                f"{stats['eta']['max']:.3f}]  |u|max {max(abs(stats['uvel']['min']), stats['uvel']['max']):.3f}")
            if not np.isfinite(th["mean"]):
                say("NaN: stopping")
                break
        if (it + 1) % a.frame_every == 0:
            frame(nframe, st, t_state)
            nframe += 1
        if (it + 1) % a.snapshot_every == 0:
            np.savez(out / f"snap_{it + 1:010d}.npz",
                     theta=compact(st.theta, L).astype(np.float32),       # (Nr, 1170, 90)
                     salt=compact(st.salt, L).astype(np.float32),
                     etaN=compact(st.etaN, L).astype(np.float32))
        if a.checkpoint_every and (it + 1) % a.checkpoint_every == 0:
            np.savez(out / f"state_{it + 1:010d}.npz", it=int(st.it), **{k: np.asarray(v) for k, v in st.f.items()})
    np.savez(out / "state_final.npz", it=int(st.it), **{k: np.asarray(v) for k, v in st.f.items()})
    if a.budgets:
        budf.write("%BUDGET cumulative " + " ".join(f"{k}={float(v): .6e}" for k, v in bacc.items()) + "\n")
        budf.close()
    ts = np.array(tstep[2:]) if len(tstep) > 2 else np.array(tstep)
    say(f"done: {len(tstep)} steps, median step {np.median(ts):.2f} s, total {time.time() - t0:.0f} s, "
        f"{nframe} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
