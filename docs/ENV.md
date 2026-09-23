# Environment (Levante)

Env `mitgcm-jax` at `/work/ab0995/a270088/mambaforge/envs/mitgcm-jax`, created 2026-09-23, pinned to the
fesom-jax known-good set (`constraints.txt` = `pip freeze` of env `fesom-jax`). **A JAX upgrade is a
deliberate, gated step** (plan M3 canary: tier 1 + gradient gates + a short GPU run, old vs new), never a
drift — `test_env.py` fails if jax/jaxlib/numpy/scipy differ from `constraints.txt`.
Reason: newer JAX broke things in fesom_jax (Nikolay, 2026-09-23: "some adjoint parts did not work" — exact
features not remembered). So the upgrade canary MUST include the gradient gates (test_adjoint_modes*,
test_checkpoint, test_fullfield_grad, the kernel FD gates, the LSR implicit derivative, Pallas vs XLA equivalence)
and the tier-2 adjoint regression, not only the forward.

## Create

Other machines (conda or venv, aarch64/GH200 nodes, NASA NAS Cabeus) and the installation check
(`scripts/check_env.py`): `docs/RUN_ONE_YEAR.md` section 2. On Levante:

```bash
E=/work/ab0995/a270088/mambaforge
$E/bin/mamba create -y -p $E/envs/mitgcm-jax python=3.12.13 pip
cd ~/MIT
$E/envs/mitgcm-jax/bin/pip install --no-cache-dir -e ".[cuda,dev]" -c constraints.txt
```
`--no-cache-dir`: pip's cache would land in `~/.cache/pip`, and home is ~full (56/60 GB).
CUDA 12 + cuDNN come as pip wheels; no system CUDA module. Login nodes are CPU-only (a CUDA plugin
warning on import there is expected).

## Run

- Smoke (seconds; allowed on a login node, restrict cores):
  `JAX_PLATFORMS=cpu taskset -c 0-7 $E/envs/mitgcm-jax/bin/python -m pytest -m smoke -q`
- Tier 1 (compute node): `sbatch scripts/run_tier1.sbatch` → `/work/ab0995/a270088/MIT/runs/tier1/<jobid>/`
  (`provenance.txt`, `pytest.log`, `report.xml`, `verdict.txt`, `git_diff.patch`).
- `conftest.py` sets 4 fake CPU devices (`--xla_force_host_platform_device_count=4`) unless XLA_FLAGS
  already carries a device count. No persistent XLA compilation cache (user decision).
- Fat login node: XLA sizes compile thread pools from the core count and can abort on `ulimit -u`
  (`pthread_create ... failed`) — use `taskset` (fesom_jax ENV.md).

## Recorded versions (installed 2026-09-23)

Python **3.12.13**; jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt **0.10.1**; numpy **2.4.6**;
scipy **1.17.1**; netCDF4 **1.7.4**; pytest **9.0.3**; ml_dtypes 0.5.4; opt_einsum 3.4.0;
CUDA wheels: cuda-runtime 12.9.79, cuBLAS 12.9.2.10, cuDNN 9.23.0.39, nvcc 12.9.86, NCCL 2.30.4.
Every installed package matches `constraints.txt` except conda's own packaging / setuptools 84.0.0 /
wheel 0.48.0. Full freeze:

```
certifi==2026.5.20
cftime==1.6.5
iniconfig==2.3.0
jax==0.10.1
jax-cuda12-pjrt==0.10.1
jax-cuda12-plugin==0.10.1
jaxlib==0.10.1
# Editable Git install with no remote (mitgcm-jax==0.0.1)
ml_dtypes==0.5.4
netCDF4==1.7.4
numpy==2.4.6
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-cccl-cu12==12.9.27
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.23.0.39
nvidia-cufft-cu12==11.4.1.4
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nccl-cu12==2.30.4
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu12==3.6.5
opt_einsum==3.4.0
packaging @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_packaging_1785888127/work
pluggy==1.6.0
Pygments==2.20.0
pytest==9.0.3
scipy==1.17.1
setuptools==84.0.0
wheel==0.48.0
```

- CPU verified (login node, 2026-09-23): x64 float64 under jit, 4 fake devices, shard_map psum.
- GPU: not yet verified on this env (first tier-2 job).

## Open

- Which JAX features broke in fesom_jax on newer versions (not in port_jax docs; its ENV.md says 0.11.0
  ran the suite). Needed to design the upgrade canary.

## dolpung (GH200, aarch64) — 2026-09-23

Venv `/work/ab0995/a270088/MIT/envs/mitgcm-jax-arm`: Python 3.12.13 from the ARM spack tree
(`/sw/spack-levante-0.23.1/linux-rhel9-neoverse_v2/python-3.12.13-hzcoloj`) + the `constraints.txt` set (jax 0.10.1,
CUDA 12.9 wheels), installed offline: `scripts/dolpung/fetch_wheels.sbatch` (shared partition, x86, internet; pip
`--platform` must list every `manylinux_2_17..2_34_aarch64` tag — it does not expand to older glibc tags) ->
`scripts/dolpung/make_env.sbatch` (dolpung node; J0 gate: GPU visible, float64 matmul). Freeze:
`/work/ab0995/a270088/MIT/runs/dolpung/pip_freeze_arm.txt`. Run: `source scripts/dolpung/env_dolpung.sh; $PY ...`
inside a job with `-A mh1571 -p dolpung` (logins are x86; the env file strips the x86 PATH entries srun exports).
Measured: production ff 0.10 s/step on one GH200 (A100-80: 0.17). CPU gates stay on x86 (the bitwise gate flags are
x86 AVX-specific).
