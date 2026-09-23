#!/usr/bin/env python3
"""M2.4 cost check: SEAICE_DYNSOLVER (LSR, literal Fortran sweep order) on the device this runs on, from the full-tree
oracle's inputs at iterations 1-3. Prints the time per call (after compile), the LSOR sweep counts and the difference
to the Fortran's I01 UICE/VICE (bitwise on CPU with the gate flags; round-off on GPU).

    python scripts/runs/gpu_lsr_bench.py [--repeat 3]     (sbatch -p gpu-devel scripts/runs/gpu_lsr_bench.sbatch)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax  # noqa: E402

import mitgcm_jax  # noqa: E402,F401
from mitgcm_jax.pkgs import seaice_dyn as sd  # noqa: E402
from mitgcm_jax.tests import test_seaice_dyn as T  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    env = T.Env()
    run = jax.jit(lambda p, g, sg, ex, st: sd.dynsolver(p, g, sg, ex, st, record=True))
    for it in (1, 2, 3):
        sg, st = env.sg(it), env.dyn_state(it)
        t = time.time()
        out, rec = run(env.p, env.g, sg, T.EX, st)
        jax.block_until_ready(out["UICE"])
        tc = time.time() - t
        ts = []
        for _ in range(a.repeat):
            t = time.time()
            out, rec = run(env.p, env.g, sg, T.EX, st)
            jax.block_until_ready(out["UICE"])
            ts.append(time.time() - t)
        counts = [(int(pr["L04"]["ICOUNT1"]), int(pr["L04"]["ICOUNT2"])) for pr in rec["passes"]]
        nsweep = sum(c[0] for c in counts)
        d = {k: (T._rel(out[k], env.f(it, "I01_dynsolver", k)), T._ndiff(out[k], env.f(it, "I01_dynsolver", k)))
             for k in ("UICE", "VICE")}
        print(f"it {it}: first call {tc:.1f} s, then median {np.median(ts):.3f} s ({np.median(ts) / nsweep * 1e3:.2f} "
              f"ms/sweep, {nsweep} sweeps {counts}, Fortran {T.COUNTS[it]}); vs Fortran I01: "
              + ", ".join(f"{k} rel {r:.1e} ({n} pts differ)" for k, (r, n) in d.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
