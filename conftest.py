"""Repository-wide pytest setup.

1. Four fake CPU devices, so every sharded code path runs at P=4 on any CPU (plan: kernel gates at
   P=1 and P=4). XLA reads the flag once, when the first backend initialises; importing jax does
   not initialise one, so setting it here, before any test module runs jax code, is early enough.
   test_env.py fails if it was too late.
2. Every collected test gets the marker of its file's cost group from mitgcm_jax/tests/manifest.py;
   a test file missing from the manifest is a collection error, not a silent default.
3. JAX's compilation caches are dropped at each module boundary (fesom_jax lesson: XLA's CPU JIT
   holds several memory mappings per executable and never releases them; a long suite hit ENOMEM at
   86 % of vm.max_map_count, reported by LLVM as "Cannot allocate memory", which reads as a
   numerical failure and is not).
"""

import os
from pathlib import Path

import pytest

FAKE_DEVICES = 4

_flag = f"--xla_force_host_platform_device_count={FAKE_DEVICES}"
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _flag).strip()
# 4. No FMA contraction on CPU: the Fortran oracle is built with -ffp-contract=off, while XLA:CPU at its default ISA
#    (AVX2 includes FMA3) fuses a*b+c. Measured on the JMD95Z EOS (SMOKE it 1): 5896 points differ by up to 4.5e-13
#    with FMA, 0 with --xla_cpu_max_isa=AVX. Oracle gates are therefore bitwise-capable only with this flag.
if "xla_cpu_max_isa" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_max_isa=AVX").strip()

REPO_ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    from mitgcm_jax.tests.manifest import MANIFEST

    unlisted = set()
    for item in items:
        rel = Path(str(item.fspath)).resolve().relative_to(REPO_ROOT).as_posix()
        group = MANIFEST.get(rel)
        if group is None:
            unlisted.add(rel)
            continue
        item.add_marker(getattr(pytest.mark, group))
    if unlisted:
        raise pytest.UsageError(
            "test files missing from mitgcm_jax/tests/manifest.py: " + ", ".join(sorted(unlisted)))


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_per_module():
    yield
    import jax

    jax.clear_caches()
