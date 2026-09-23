#!/usr/bin/env python3
"""Plan Task 20: GPU sharding check. On one node with 4 GPUs:
  1. the GPU floor: two identical 1-GPU runs of N steps (forced oracle, from its iteration-1 state);
  2. 4-GPU tile-sharded run (parallel/shard.ShardedModel, P=4) vs the 1-GPU run, same N steps;
  3. gradient: d(mean SST after 1 step)/d(theta0), 4-GPU sharded vs 1-GPU.
Prints max |a-b| / max|a| per field for each comparison and the step times.

    python scripts/runs/gpu_sharding.py [--nsteps 24]     (sbatch scripts/runs/gpu_sharding.sbatch)
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
from mitgcm_jax.model import setup  # noqa: E402
from mitgcm_jax.parallel.shard import ShardedModel  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod  # noqa: E402
from mitgcm_jax.state import State, state_from_dump  # noqa: E402
from mitgcm_jax.tests import oracle  # noqa: E402

FIELDS = ("theta", "salt", "uVel", "vVel", "wVel", "etaN", "etaH", "GGL90TKE")


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.max(np.abs(a))
    return float(np.max(np.abs(a - b)) / s) if s > 0 else float(np.max(np.abs(a - b)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=24)
    ap.add_argument("--nproc", type=int, default=4)
    a = ap.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    nml = RunNamelists(rundir)
    P, g, ex, kLowC = setup(rundir)
    st0 = state_from_dump(ds, 1)
    st0 = st0.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    st0 = State({k: jnp.asarray(v) for k, v in st0.f.items()}, jnp.asarray(1))
    nIter0 = int(nml.get("data", "parm03", "nIter0", default=0))
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    exf_ins = []
    for n in range(a.nsteps):
        myTime, myIter = exf_mod.model_time(nml, 1 + n - nIter0 + 1)
        bufs, facs, _ = loader.load(myTime, myIter)
        exf_ins.append({"bufs": bufs, "facs": facs, "myTime": myTime})
    step1 = jax.jit(lambda P, g, kLowC, st, e: forward_step(P, g, ex, kLowC, st, e)[0])

    def run1():
        st = st0
        ts = []
        for e in exf_ins:
            t = time.time()
            st = step1(P, g, kLowC, st, e)
            jax.block_until_ready(st.f["theta"])
            ts.append(time.time() - t)
        return st, ts

    A, tA = run1()
    B, tB = run1()
    print(f"1 GPU: median step {np.median(tA[2:]):.3f} s (first {tA[0]:.1f} s)", flush=True)
    print("GPU floor (two identical 1-GPU runs, %d steps): " % a.nsteps
          + ", ".join(f"{k} {rel(A.f[k], B.f[k]):.1e}" for k in FIELDS), flush=True)

    sm = ShardedModel(g, a.nproc)
    f, it = sm.shard_state(st0)
    k4 = sm.shard_tiles(kLowC)
    ts = []
    for e in exf_ins:
        t = time.time()
        f, it, info = sm.step(P, k4, f, it, sm.shard_exf(e))
        jax.block_until_ready(f["theta"])
        ts.append(time.time() - t)
    C = sm.unpad(f)
    print(f"{a.nproc} GPU: median step {np.median(ts[2:]):.3f} s (first {ts[0]:.1f} s)", flush=True)
    print("%d-GPU sharded vs 1-GPU (%d steps): " % (a.nproc, a.nsteps)
          + ", ".join(f"{k} {rel(A.f[k], C[k]):.1e}" for k in FIELDS), flush=True)

    # gradient of the area-weighted mean SST after one step w.r.t. theta0
    L = g.layout
    w = np.zeros(L.shape2d)
    I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
    w[:, J, I] = np.asarray(g.rA)[:, J, I] * np.asarray(g.maskC)[:, 0, J, I]
    w = jnp.asarray(w / w.sum())

    def J1(theta):
        s1 = forward_step(P, g, ex, kLowC, st0.replace(theta=theta), exf_ins[0])[0]
        return jnp.sum(w * s1.theta[:, 0])

    g1 = np.asarray(jax.jit(jax.grad(J1))(st0.theta))
    fw = sm.shard_tiles(w)

    def J4(theta_sharded):
        f0 = dict(f0base)
        f0["theta"] = theta_sharded
        f1, _, _ = sm.step(P, k4, f0, it0, e0)
        return jnp.sum((fw * f1["theta"][:, 0])[: L.nTiles])   # padded order: tiles 1..13 first, then replicas

    f0base, it0 = sm.shard_state(st0)
    e0 = sm.shard_exf(exf_ins[0])
    g4 = sm.unpad({"theta": jax.grad(J4)(f0base["theta"])})["theta"]
    print(f"gradient d(mean SST)/d(theta0): {a.nproc}-GPU vs 1-GPU rel {rel(g1, g4):.2e}; finite "
          f"{np.all(np.isfinite(g4))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
