#!/usr/bin/env python3
"""Run the JAX MITgcm (V4r4 flux-forced) forward model and write movie frames, %MON statistics and snapshots.

    python scripts/run_jax.py OUTDIR --rundir RUNDIR --init-oracle NAME [--init-it 1] --nsteps N
                              [--frame-every 6] [--monitor-every 1] [--snapshot-every 24]

RUNDIR: a Fortran run directory with the namelists and linked inputs (forcing files) of the configuration to run.
Initial state: the Fortran oracle's state at the start of iteration --init-it (S00_begin dump; the port of the
pickup/initialisation, plan Task 8, replaces this). OUTDIR (new) gets:
  frames/frame_<n>.npz   sst (compact 1170x90), eta, iter, date   -> tools/animate_globe.py (nereus env)
  monitor.txt            %MON dynstat lines in the Fortran format (compare with STDOUT.0000)
  snap_<iter>.npz        theta, salt, etaN (compact, float32) every --snapshot-every steps (dumpFreq twin)
  state_final.npz        the full State (restart)
Model date of iteration n: startDate_1 (data.cal) + n*deltaT - startTime ... taken from the calendar of EXF.
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
from mitgcm_jax.diagnostics.monitor import dynstat, dynstat_device, format_dynstat  # noqa: E402
from mitgcm_jax.io.llc import tiles_to_compact  # noqa: E402
from mitgcm_jax.model import setup  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod  # noqa: E402
from mitgcm_jax.state import State, state_from_dump  # noqa: E402
from mitgcm_jax.tests import oracle  # noqa: E402


def interior(a, L):
    return np.asarray(a)[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx]


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
    ap.add_argument("--init-oracle")
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
        st = step(P, g, kLowC, st, {"bufs": bufs, "facs": facs, "myTime": myTime})
        jax.block_until_ready(st.f["theta"])
        tstep.append(time.time() - t1)
        t_state = exf_mod.model_time(nml, it + 1 - nIter0 + 1)[0]
        if (n + 1) % a.monitor_every == 0:
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
        if (n + 1) % a.frame_every == 0:
            frame(nframe, st, t_state)
            nframe += 1
        if (n + 1) % a.snapshot_every == 0:
            np.savez(out / f"snap_{it + 1:010d}.npz",
                     theta=tiles_to_compact(interior(st.theta, L)).astype(np.float32),
                     salt=tiles_to_compact(interior(st.salt, L)).astype(np.float32),
                     etaN=tiles_to_compact(interior(st.etaN, L)).astype(np.float32))
        if a.checkpoint_every and (n + 1) % a.checkpoint_every == 0:
            np.savez(out / f"state_{it + 1:010d}.npz", it=int(st.it), **{k: np.asarray(v) for k, v in st.f.items()})
    np.savez(out / "state_final.npz", it=int(st.it), **{k: np.asarray(v) for k, v in st.f.items()})
    ts = np.array(tstep[2:]) if len(tstep) > 2 else np.array(tstep)
    say(f"done: {len(tstep)} steps, median step {np.median(ts):.2f} s, total {time.time() - t0:.0f} s, "
        f"{nframe} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
