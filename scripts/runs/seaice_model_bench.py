#!/usr/bin/env python3
"""M2.6b-1 cost check: the whole SEAICE_MODEL driver (pkgs/seaice_model.py) jitted on one device, from the full-tree
oracle's inputs at iterations 1-3 (the chained sea-ice state of tests/test_seaice_model.py). Prints the compile time,
the time per step (median of --repeat calls), the LSOR sweep counts and the difference to the Fortran's P00 fields;
with --grad also the time of one jax.grad per adjoint level (ecco, no_dynamics, full).

    python scripts/runs/seaice_model_bench.py [--device cpu|default] [--repeat 3] [--grad] [--gate-flags]

--device cpu places the arguments on jax.devices("cpu")[0] (the host CPU next to a GPU: the M2.6b-2 placement);
--gate-flags sets the bitwise gate XLA flags (no FMA, no algsimp) before jax initialises.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default", choices=("default", "cpu"))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--gate-flags", action="store_true")
    a = ap.parse_args(argv)
    if a.gate_flags:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                                   + " --xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp").strip()
    import jax
    import jax.numpy as jnp

    import mitgcm_jax  # noqa: F401
    from mitgcm_jax.pkgs import seaice_model as sm
    from mitgcm_jax.tests import test_seaice_model as T

    dev = jax.devices("cpu")[0] if a.device == "cpu" else jax.devices()[0]
    print("devices:", jax.devices(), "-> running on", dev, "| XLA_FLAGS:", os.environ.get("XLA_FLAGS", ""),
          flush=True)
    e = T.env()
    args = jax.device_put((e.P, e.g, e.sg, T.EX), dev)
    run = jax.jit(sm.seaice_model, static_argnames=("ad", "expf", "record"))
    carry = None
    for it in T.ITS:
        ins = jax.device_put(T.inputs(it, carry), dev)
        t = time.time()
        out, rec = run(*args, ins, record=True)
        jax.block_until_ready(out)
        tc = time.time() - t
        ts = []
        for _ in range(a.repeat):
            t = time.time()
            out, rec = run(*args, ins, record=True)
            jax.block_until_ready(out)
            ts.append(time.time() - t)
        counts = [(int(pr["L04"]["ICOUNT1"]), int(pr["L04"]["ICOUNT2"])) for pr in rec["dyn"]["passes"]]
        ref = {k: T.F(it, "P00_seaice_model", k) for k in ("HEFF", "AREA", "UICE", "Qnet", "EmPmR")}
        nd = {k: (T.ndiff(out[k], r), f"{T.rel(out[k], r):.1e}") for k, r in ref.items()}
        print(f"it {it}: first call {tc:.1f} s, then median {np.median(ts):.3f} s per step (min {min(ts):.3f}); LSOR "
              f"sweeps {counts} (Fortran {T.COUNTS[it]}); vs P00 (points differing, max rel): {nd}", flush=True)
        carry = {k: out[k] for k in sm.SEAICE_CARRIED + ("sIceLoad",)}
    if a.grad:
        ins = jax.device_put(T.inputs(1), dev)
        w = {k: jnp.ones_like(ins[k]) for k in ("HEFF", "Qnet", "fu", "UICE")}
        for ad in sm.AD_LEVELS:
            def J(i, P, g, sg, ex):
                o, _ = sm.seaice_model(P, g, sg, ex, i, ad=ad)
                return sum(jnp.sum(w[k] * o[k]) for k in w)

            vg = jax.jit(jax.value_and_grad(J))
            t = time.time()
            jax.block_until_ready(vg(ins, *args))
            tc = time.time() - t
            ts = []
            for _ in range(a.repeat):
                t = time.time()
                jax.block_until_ready(vg(ins, *args))
                ts.append(time.time() - t)
            print(f"value_and_grad ad={ad}: first call {tc:.1f} s, then median {np.median(ts):.3f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
