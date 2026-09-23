#!/usr/bin/env python3
"""Forward speed of the full V4r4 model (EXF bulk formulae + sea ice) on 1 or P devices, from the pickup, no output
files (the step loop only; the host-side EXF record loading of each step is included, as in a production run).

    python scripts/runs/full_speed_bench.py [--nsteps 8760] [--nproc 1|4] [--rundir RUN]
                                           (sbatch scripts/runs/full_speed_bench.sbatch ... on A100 / GH200 nodes)

--nproc 1: the jitted forward_step on one device (as scripts/run_jax.py); --nproc P > 1: parallel/shard.ShardedModel
(tile-sharded shard_map, 13 tiles padded to a multiple of P). Prints the median / mean step time, the total wall time
of the loop, the set-up time, and at the end the global mean theta and the mean ice concentration (sanity; P > 1 runs
are bitwise the P = 1 run on the same GPU type, plan Task 20 / LSR lessons).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import mitgcm_jax  # noqa: E402,F401
from mitgcm_jax.core.forward_step import forward_step  # noqa: E402
from mitgcm_jax.init import state_from_pickup  # noqa: E402
from mitgcm_jax.model import setup  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod  # noqa: E402
from mitgcm_jax.pkgs import exf_full as exfb_mod  # noqa: E402
from mitgcm_jax.state import State  # noqa: E402

RUNDIR = "/work/ab0995/a270088/MIT/reference/runs/ref_full_serial13_1month"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=8760)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--rundir", default=RUNDIR)
    ap.add_argument("--cg2d-unroll", type=int, default=5)
    a = ap.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    t0 = time.time()
    nml = RunNamelists(a.rundir)
    P, g, ex, kLowC = setup(a.rundir, tree="full")
    import dataclasses
    P = P._replace(cg=dataclasses.replace(P.cg, sum_unroll=a.cg2d_unroll))
    st = state_from_pickup(P, g, ex, kLowC, a.rundir, tree="full")
    it0 = int(st.it)
    f0 = {k: np.asarray(v) for k, v in st.f.items()}
    if "PmEpR" not in f0:
        f0["PmEpR"] = np.zeros(g.layout.shape2d)
    st = State({k: jnp.asarray(v, dtype=v.dtype) for k, v in f0.items()}, jnp.asarray(it0))
    loader = exfb_mod.ExfFullRecordLoader(P.exfb, g, a.rundir)
    nIter0 = int(nml.get("data", "parm03", "nIter0", default=0))

    def exf_inputs(it):
        myTime, myIter = exf_mod.model_time(nml, it - nIter0 + 1)
        bufs, facs, _ = loader.load(myTime, myIter)
        return {"bufs": bufs, "facs": facs, "myTime": myTime, "zt": exfb_mod.zenith_time(P.exfb, myTime, myIter)}

    if a.nproc == 1:
        step = jax.jit(lambda P, g, kLowC, st, e: forward_step(P, g, ex, kLowC, st, e)[0])
        kL = kLowC
    else:
        from mitgcm_jax.parallel.shard import ShardedModel
        sm = ShardedModel(g, a.nproc)
        f, itd = sm.shard_state(st)
        kL = sm.shard_tiles(kLowC)
    t_setup = time.time() - t0
    print(f"setup + initial state {t_setup:.1f} s; nproc {a.nproc}", flush=True)
    ts = []
    t_loop = time.time()
    for n in range(a.nsteps):
        it = it0 + n
        e = exf_inputs(it)
        t = time.time()
        if a.nproc == 1:
            st = step(P, g, kL, st, e)
            jax.block_until_ready(st.f["theta"])
        else:
            f, itd, _ = sm.step(P, kL, f, itd, sm.shard_exf(e))
            jax.block_until_ready(f["theta"])
        ts.append(time.time() - t)
        if (n + 1) % 720 == 0:
            print(f"step {n + 1}: median {np.median(ts):.3f} s/step, elapsed {time.time() - t_loop:.0f} s", flush=True)
    wall = time.time() - t_loop
    if a.nproc > 1:
        fu = sm.unpad(f)
        theta, area = fu["theta"], fu["AREA"]
    else:
        theta, area = np.asarray(st.f["theta"]), np.asarray(st.f["AREA"])
    L = g.layout
    I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
    vol = np.asarray(g.rA)[:, None, J, I] * np.asarray(g.drF)[None, :, None, None] * np.asarray(g.h0FacC)[..., J, I]
    print(f"done: {a.nsteps} steps on {a.nproc} x {jax.devices()[0].device_kind}: median {np.median(ts[2:]):.3f} s/step,"
          f" mean {np.mean(ts[2:]):.3f} s/step, first step (compile) {ts[0]:.1f} s, loop total {wall:.0f} s "
          f"({wall / 60:.1f} min), setup {t_setup:.0f} s", flush=True)
    print(f"sanity: volume-mean theta {np.sum(theta[..., J, I] * vol) / np.sum(vol):.12f}, "
          f"mean AREA (interior) {np.mean(area[:, J, I]):.12f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
