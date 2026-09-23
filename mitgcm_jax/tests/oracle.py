"""Access to the Fortran oracle for tests: named reference runs (reference/runs.json) and their jaxdump dumps.

A test that needs a run fails (does not skip) when the run is missing: a gate that silently skips checks nothing.
"""

import functools
import json
from pathlib import Path

from mitgcm_jax.io.dump import DumpSet

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/work/ab0995/a270088/MIT/reference/runs")


def run_dir(name):
    reg = json.loads((REPO / "reference" / "runs.json").read_text())
    if name not in reg:
        raise KeyError(f"reference run {name!r} is not registered in reference/runs.json")
    return RUNS / reg[name]


@functools.lru_cache(maxsize=4)
def dumpset(name):
    return DumpSet(run_dir(name) / "jaxdump")


# Oracles for kernel gates (reference/runs.json names):
#   SMOKE  2 steps, useEXF=F (no surface forcing), dumps at iterations 1, 2
#   FORCED 3 steps, useEXF=T (flux-forced 1992 forcing), useCTRL=F, geothermalFile=' ', dumps at iterations 1, 2, 3
SMOKE = "smoke_ff_jaxdump_v3"
FORCED = "forced_ff_jaxdump_v3"


def field(ds, it, stage, name, layout=None):
    """A dumped field as [T, (nz,) ny, nx] with halos, tiles in W2 order. Per-level records written inside a k loop
    (`<name>_k001` ... , jaxdump K: dumps) are stacked into [T, Nr, ny, nx]."""
    import numpy as np

    from mitgcm_jax.grid.geometry import stack_tiles
    from mitgcm_jax.layout import Layout

    L = layout or Layout()
    if (it, stage, name) in ds.index:
        return stack_tiles(ds, it, stage, name, L)
    levels = sorted(k[2] for k in ds.index if k[0] == it and k[1] == stage and k[2].startswith(name + "_k")
                    and k[2][len(name) + 2:].isdigit())
    if not levels:
        raise KeyError((it, stage, name))
    return np.stack([stack_tiles(ds, it, stage, lv, L) for lv in levels], axis=1)
