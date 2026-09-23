#!/usr/bin/env python3
"""SEAICE_LSR with the Pallas sweep (lsr_impl="pallas") inside shard_map(check_vma=True) on 4 GPUs of one node:
uIce, vIce and the LSOR counts must equal the 1-GPU run bitwise, and both the Fortran's Y05 at iteration 1 (the logic
of tests/test_seaice_dyn.py::test_lsr_sharded_p4_bitwise; the Pallas interpreter cannot run under check_vma, so this
is checked on GPUs).   sbatch --gpus=4 ... python scripts/runs/lsr_sharded_gpu_check.py
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax  # noqa: E402

import mitgcm_jax  # noqa: E402,F401
from mitgcm_jax.tests import test_seaice_dyn as T  # noqa: E402


def main():
    print("devices:", jax.devices(), flush=True)
    env = T.Env()
    env.p = dataclasses.replace(env.p, lsr_impl="pallas")
    out, passes = env.lsr(1)
    bad = T._lsr_mismatches(env, 1, out, passes)
    print("1 GPU vs Fortran: counts", [int(pr["L04"]["ICOUNT1"]) for pr in passes], "mismatching stage fields:",
          {k: v for k, v in bad.items() if v}, flush=True)
    T.test_lsr_sharded_p4_bitwise(env)
    print("4 GPUs (shard_map, check_vma=True) == 1 GPU bitwise, counts equal: OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
