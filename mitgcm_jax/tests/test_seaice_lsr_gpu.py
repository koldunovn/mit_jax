"""Tier 2 (one CUDA GPU): the LSOR implementations of SeaiceDynParams.lsr_impl are the same computation on the GPU.
The whole SEAICE_DYNSOLVER (both Picard passes) at iterations 1-3 of full_jaxdump_v5 with "pallas" (the "auto" choice
on CUDA) and "xla_unrolled" (the "auto" choice on other GPUs): every output field equal at every point between the
two, UICE/VICE equal to the Fortran's I01, LSOR sweep counts 178/118, 112/82, 84/58 as in the Fortran.
Recorded 2026-09-23 (scripts/runs/lsr_perf_bench.py --full, same comparison plus the lax.scan path): A100-40 job
27648486, GH200 job 27648487: 0 points differ in all 32 outputs, counts equal; s/step (it 1/2/3) pallas 1.18/0.78/0.58
(A100), 0.78/0.51/0.38 (GH200).
"""

import dataclasses

import jax
import numpy as np
import pytest

from mitgcm_jax.pkgs import seaice_dyn as sd
from mitgcm_jax.tests import test_seaice_dyn as T


@pytest.fixture(scope="module")
def env():
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    assert gpus, f"tier-2 GPU test needs a CUDA GPU, found {jax.devices()}"
    return T.Env()


@pytest.mark.parametrize("it", T.ITS)
def test_dynsolver_pallas_equals_xla_unrolled_gpu(env, it):
    run = jax.jit(lambda p, g, sg, ex, st: sd.dynsolver(p, g, sg, ex, st, record=True))
    res = {}
    for impl in ("pallas", "xla_unrolled"):
        out, rec = run(dataclasses.replace(env.p, lsr_impl=impl), env.g, env.sg(it), T.EX, env.dyn_state(it))
        counts = tuple((int(pr["L04"]["ICOUNT1"]), int(pr["L04"]["ICOUNT2"])) for pr in rec["passes"])
        res[impl] = ({k: np.asarray(v) for k, v in out.items()}, counts)
    (a, ca), (b, cb) = res["pallas"], res["xla_unrolled"]
    assert ca == cb == tuple((n, n) for n in T.COUNTS[it]), (ca, cb)
    bad = {k: int(np.sum(a[k] != b[k])) for k in a}
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    for k in ("UICE", "VICE"):
        assert T._ndiff(a[k], env.f(it, "I01_dynsolver", k)) == 0, k
