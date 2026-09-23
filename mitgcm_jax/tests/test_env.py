"""The environment the rest of the suite silently relies on: float64, the pinned package set, four fake
devices, and this checkout (not another one) being the package under test."""

import importlib.metadata
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

import mitgcm_jax

REPO_ROOT = Path(__file__).resolve().parents[2]

# The packages whose version changes numerics or AD; the whole set is in constraints.txt.
PINNED = ("jax", "jaxlib", "numpy", "scipy")


def _constraints():
    pins = {}
    for line in (REPO_ROOT / "constraints.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            pins[name.strip().lower()] = ver.strip()
    return pins


def test_imports_this_checkout():
    assert Path(mitgcm_jax.__file__).resolve().parent == REPO_ROOT / "mitgcm_jax"


def test_pinned_versions_match_constraints():
    pins = _constraints()
    got = {name: importlib.metadata.version(name) for name in PINNED}
    want = {name: pins[name] for name in PINNED}
    assert got == want, "env drifted from constraints.txt; a JAX upgrade goes through the canary (plan M3)"


def test_x64_float64_under_jit():
    assert jax.config.jax_enable_x64
    x = jnp.arange(4.0)
    assert x.dtype == jnp.float64

    @jax.jit
    def f(a):
        return a * (1.0 + 1e-12) - a

    y = f(jnp.ones(3))
    assert y.dtype == jnp.float64
    # 1e-12 is below float32 resolution (6e-8): a silent float32 fallback returns exactly 0.
    np.testing.assert_allclose(np.asarray(y), 1e-12, rtol=1e-3)


def test_four_fake_cpu_devices_and_shard_map_psum():
    devs = jax.devices("cpu")
    assert len(devs) == 4, "conftest's XLA_FLAGS arrived after the CPU backend was initialised"
    mesh = Mesh(np.array(devs), ("tile",))

    @jax.jit
    def total(a):
        return jax.shard_map(lambda b: jax.lax.psum(b.sum(), "tile"),
                             mesh=mesh, in_specs=P("tile"), out_specs=P())(a)

    a = jnp.arange(8.0)
    assert float(total(a)) == 28.0
