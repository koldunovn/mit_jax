#!/usr/bin/env python3
"""Installation check: JAX version and devices, float64 enabled, a jitted float64 matmul on the default device against
numpy, and the pinned package versions (constraints.txt). Exit code 0 = usable.

    python scripts/check_env.py            (on a GPU node: the GPU must be listed and used)
"""

import re
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import jax  # noqa: E402

import mitgcm_jax  # noqa: E402,F401  (enables x64)
from mitgcm_jax import paths  # noqa: E402

PINNED = ("jax", "jaxlib", "numpy", "scipy")


def main():
    ok = True
    print(f"python {sys.version.split()[0]}, jax {jax.__version__}, backend {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    pins = {}
    for line in (REPO / "constraints.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    for p in PINNED:
        have = metadata.version(p)
        flag = "" if have == pins.get(p) else f"   <- constraints.txt pins {pins.get(p)}"
        ok &= not flag
        print(f"  {p:8s} {have}{flag}")
    x = np.random.default_rng(0).normal(size=(2000, 2000))
    y = jax.jit(lambda a: a @ a)(jax.numpy.asarray(x))
    err = float(np.max(np.abs(np.asarray(y) - x @ x)) / np.max(np.abs(x @ x)))
    good = y.dtype == np.float64 and err < 1e-12
    ok &= good
    print(f"float64 matmul on {y.devices()}: dtype {y.dtype}, max rel. diff vs numpy {err:.1e} "
          f"{'OK' if good else 'FAIL'}")
    for k, v in paths.ALL.items():
        print(f"  {k:22s} {v}{'' if v.exists() else '   (does not exist)'}")
    print("CHECK OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
