"""mitgcm_jax/paths.py: the one place for machine paths. Defaults derive from the work root, each variable overrides
its own path and the ones derived from it, and no other Python file hard-codes the development machine's work root."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEVANTE = "/work/ab0995/a270088/MIT"


def resolved(**env):
    """paths.py --sh in a clean process with only the given MITJAX_* variables set."""
    e = {k: v for k, v in os.environ.items() if not k.startswith("MITJAX_")}
    e.update(env)
    out = subprocess.run([sys.executable, str(REPO / "mitgcm_jax" / "paths.py"), "--sh"], env=e, check=True,
                         capture_output=True, text=True).stdout
    return dict(line[len("export "):].split("=", 1) for line in out.splitlines())


def test_defaults_derive_from_the_work_root():
    p = resolved()
    assert p["MITJAX_WORK"] == LEVANTE
    assert p["MITJAX_DATA"] == f"{LEVANTE}/data/eccov4r4"
    assert p["MITJAX_GRID_DIR"] == f"{LEVANTE}/data/eccov4r4/native_grid_files"
    assert p["MITJAX_REFERENCE_RUNS"] == f"{LEVANTE}/reference/runs"
    assert p["MITJAX_RUNS_JAX"] == f"{LEVANTE}/runs_jax"
    q = resolved(MITJAX_WORK="/x")
    assert q["MITJAX_DATA"] == "/x/data/eccov4r4" and q["MITJAX_REFERENCE_RUNS"] == "/x/reference/runs"
    assert q["MITJAX_RUNS"] == "/x/runs" and q["MITJAX_RUNS_JAX"] == "/x/runs_jax"


def test_each_variable_overrides_its_subtree_only():
    p = resolved(MITJAX_WORK="/x", MITJAX_DATA="/d", MITJAX_REFERENCE="/r")
    assert p["MITJAX_GRID_DIR"] == "/d/native_grid_files"
    assert p["MITJAX_REFERENCE_RUNS"] == "/r/runs"
    assert p["MITJAX_RUNS_JAX"] == "/x/runs_jax"
    assert resolved(MITJAX_GRID_DIR="~/g")["MITJAX_GRID_DIR"] == str(Path("~/g").expanduser())


def test_no_other_python_file_hard_codes_the_work_root():
    hits = [str(p.relative_to(REPO)) for d in ("mitgcm_jax", "scripts", "tools", "reference")
            for p in (REPO / d).rglob("*.py")
            if p.resolve() != Path(__file__).resolve() and LEVANTE in p.read_text(errors="replace")]
    assert hits == ["mitgcm_jax/paths.py"], hits
