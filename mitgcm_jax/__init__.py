"""MITgcm checkpoint66g (ECCO v4r4, LLC90) in JAX: differentiable and parallel.

float64 is mandatory: x64 is enabled on import, before any array is created.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
