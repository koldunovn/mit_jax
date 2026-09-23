#!/usr/bin/env python3
"""Run the JAX FORWARD_STEP from the Fortran oracle's state at iteration IT and compare, stage by stage, with the
dumps of the same step (first divergence first), then the end state with S00_begin of IT+1.

    XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp" JAX_PLATFORMS=cpu \
        python tools/step_vs_dump.py [--oracle forced_ff_jaxdump_v3] [--it 1] [--nsteps 1]

With --nsteps N > 1 the JAX state is carried (free run) and compared with S00_begin of IT+n after each step.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax  # noqa: E402

import mitgcm_jax  # noqa: E402,F401
from mitgcm_jax.core.forward_step import forward_step  # noqa: E402
from mitgcm_jax.model import setup  # noqa: E402
from mitgcm_jax.params_io import RunNamelists  # noqa: E402
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod  # noqa: E402
from mitgcm_jax.state import state_from_dump  # noqa: E402
from mitgcm_jax.tests import oracle  # noqa: E402

# (aux key, field, dump stage, dump field)
CHECKS = [
    ("S01_update_rstar_F", "hFacC", "S01_update_rstar_F", "hFacC"),
    ("S01_update_rstar_F", "recip_hFacC", "S01_update_rstar_F", "recip_hFacC"),
    ("S02_load_fields", "ustress", "S02_load_fields", "ustress"),
    ("S02_load_fields", "hflux", "S02_load_fields", "hflux"),
    ("S02_load_fields", "Qnet", "S02_load_fields", "Qnet"),
    ("S02_load_fields", "EmPmR", "S02_load_fields", "EmPmR"),
    ("S04_oceanic_phys", "surfaceForcingT", "P01_external_forcing_surf", "surfaceForcingT"),
    ("S04_oceanic_phys", "rhoInSitu", "P02_rho_sigma_ivdc_mxlayer", "rhoInSitu"),
    ("P02", "sigmaR", "P02_rho_sigma_ivdc_mxlayer", "sigmaR"),
    ("S04_oceanic_phys", "saltPlumeDepth", "P03_salt_plume_depth", "saltPlumeDepth"),
    ("S04_oceanic_phys", "GGL90TKE", "P04_ggl90", "GGL90TKE"),
    ("S04_oceanic_phys", "GGL90diffKr", "P04_ggl90", "GGL90diffKr"),
    ("S04_oceanic_phys", "Kwx", "P06_gmredi_exch", "Kwx"),
    ("S04_oceanic_phys", "GM_PsiX", "P06_gmredi_exch", "GM_PsiX"),
    ("S05_dynamics", "totPhiHyd", "S05_dynamics", "totPhiHyd"),
    ("S05_dynamics", "gU", "S05_dynamics", "gU"),
    ("S05_dynamics", "gV", "S05_dynamics", "gV"),
    ("S12_stagger_exchanges", "etaN", "S12_stagger_exchanges", "etaN"),
    ("S12_stagger_exchanges", "uVel", "S12_stagger_exchanges", "uVel"),
    ("S12_stagger_exchanges", "wVel", "S12_stagger_exchanges", "wVel"),
    ("S12_stagger_exchanges", "etaH", "S12_stagger_exchanges", "etaH"),
    ("T01_residual_flow", "uFld", "T01_residual_flow", "uFld"),
    ("T10_T20_adv", "gT", "T10_temp_adv", "gT_loc"),
    ("S13_thermodynamics", "theta", "S13_thermodynamics", "theta"),
    ("S13_thermodynamics", "salt", "S13_thermodynamics", "salt"),
]


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    s = np.max(np.abs(b))
    return float(d.max() / s) if s > 0 else float(d.max())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=oracle.FORCED)
    ap.add_argument("--it", type=int, default=1)
    ap.add_argument("--nsteps", type=int, default=1)
    ap.add_argument("--grid", choices=("files", "dump"), default="files")
    a = ap.parse_args(argv)
    ds = oracle.dumpset(a.oracle)
    rundir = oracle.run_dir(a.oracle)
    t0 = time.time()
    grid = None
    if a.grid == "dump":
        from mitgcm_jax.grid.geometry import grid_from_dump
        grid = grid_from_dump(ds, a.it)
    P, g, ex, kLowC = setup(rundir, grid=grid)
    print(f"setup {time.time() - t0:.1f} s", flush=True)
    st = state_from_dump(ds, a.it)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    nml = RunNamelists(rundir)
    loader = exf_mod.ExfRecordLoader(P.exf, g, rundir)
    nIter0 = int(nml.get("data", "parm03", "nIter0", default=0))
    step = jax.jit(lambda P, g, kLowC, st, exf_in: forward_step(P, g, ex, kLowC, st, exf_in))
    for n in range(a.nsteps):
        it = a.it + n
        myTime, myIter = exf_mod.model_time(nml, it - nIter0 + 1)
        assert myIter == it, (myIter, it)
        bufs, facs, _ = loader.load(myTime, myIter)
        exf_in = {"bufs": bufs, "facs": facs, "myTime": myTime}
        t0 = time.time()
        st1, aux = step(P, g, kLowC, st, exf_in)
        jax.block_until_ready(st1.f["theta"])
        print(f"step it={it}: {time.time() - t0:.1f} s", flush=True)
        if n == 0:
            for key, fld, stage, dfld in CHECKS:
                try:
                    ref = oracle.field(ds, it, stage, dfld)
                except KeyError:
                    print(f"  {stage:28s} {dfld:16s} (not dumped)")
                    continue
                got = aux[key][fld]
                print(f"  {stage:28s} {dfld:16s} rel {rel(got, ref)}", flush=True)
            print("  cg2d:", {k: np.asarray(v).tolist() for k, v in aux["cg2d"].items()
                              if np.asarray(v).size < 5})
        if (it + 1, "S00_begin", "theta") in ds.index:
            worst = []
            for k in ("theta", "salt", "uVel", "vVel", "wVel", "etaN", "etaH", "GGL90TKE", "gU", "gV", "guNm_1"):
                ref = oracle.field(ds, it + 1, "S00_begin", k)
                worst.append((k, rel(st1.f[k], ref)))
            print(f"  end of step vs S00_begin it={it + 1}:", ", ".join(f"{k} {v:.2e}" if isinstance(v, float)
                                                                          else f"{k} {v}" for k, v in worst))
        st = st1
    return 0


if __name__ == "__main__":
    sys.exit(main())
