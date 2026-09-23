"""Where data, Fortran reference runs and run output live: the one place for machine-specific paths.

Every location has an environment override; unset, it derives from the one above it:

    MITJAX_WORK             work root (large file system)       default /work/ab0995/a270088/MIT (DKRZ Levante)
    MITJAX_DATA             ECCO v4r4 input data                $MITJAX_WORK/data/eccov4r4
    MITJAX_GRID_DIR         LLC90 tileNNN.mitgrid files         $MITJAX_DATA/native_grid_files
    MITJAX_REFERENCE        Fortran reference (bin/, build/, runs/, logs/)   $MITJAX_WORK/reference
    MITJAX_REFERENCE_RUNS   run directories (make_rundir.py)    $MITJAX_REFERENCE/runs
    MITJAX_RUNS             job output (test tiers, benchmarks) $MITJAX_WORK/runs
    MITJAX_RUNS_JAX         JAX model output (scripts/run_jax.py runs)       $MITJAX_WORK/runs_jax

Values are read once, at import. Stdlib only and free of jax: tools that run in other environments (plotting) load
this file by path, and batch scripts evaluate `python mitgcm_jax/paths.py --sh` (shell `export` lines). Run without
arguments it prints the resolved values.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env(name, default):
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else Path(default)


WORK = _env("MITJAX_WORK", "/work/ab0995/a270088/MIT")
DATA = _env("MITJAX_DATA", WORK / "data" / "eccov4r4")
GRID_DIR = _env("MITJAX_GRID_DIR", DATA / "native_grid_files")
REFERENCE = _env("MITJAX_REFERENCE", WORK / "reference")
REFERENCE_RUNS = _env("MITJAX_REFERENCE_RUNS", REFERENCE / "runs")
RUNS = _env("MITJAX_RUNS", WORK / "runs")
RUNS_JAX = _env("MITJAX_RUNS_JAX", WORK / "runs_jax")

ALL = {"MITJAX_WORK": WORK, "MITJAX_DATA": DATA, "MITJAX_GRID_DIR": GRID_DIR, "MITJAX_REFERENCE": REFERENCE,
       "MITJAX_REFERENCE_RUNS": REFERENCE_RUNS, "MITJAX_RUNS": RUNS, "MITJAX_RUNS_JAX": RUNS_JAX}


if __name__ == "__main__":
    import shlex

    sh = "--sh" in sys.argv[1:]
    for k, v in ALL.items():
        print(f"export {k}={shlex.quote(str(v))}" if sh else f"{k:22s} {v}")
