"""Implicit vertical diffusion for convection: the convection flag IVDConvCount (plan Task 10).

Literal port of model/src/calc_ivdc.F (CALC_IVDC) as called from flux-forced/code/do_oceanic_phys.F:912-920 with
iMin..iMax, jMin..jMax = the full tile (do_oceanic_phys.F:594-597), for every k > 1 (calcConvect = ivdc_kappa /= 0).
IVDConvCount is zeroed for all k first (do_oceanic_phys.F:686-692, ALLOW_AUTODIFF), so level 1 stays 0.
IVDConvCount is then read by CALC_3D_DIFFUSIVITY (diffusivity ivdc_kappa where it is 1).

The flag is a step function of sigmaR: its derivative is zero almost everywhere (as in TAF), and jnp.where carries no
gradient to sigmaR.
"""

import jax.numpy as jnp


def calc_ivdc(sigmaR, gravitySign):
    """sigmaR [T, Nr, ny, nx] (level axis = Fortran k) -> IVDConvCount [T, Nr, ny, nx].
    calc_ivdc.F:48-56:  IF ( -sigmaR(i,j,k)*gravitySign .GT. 0. ) THEN 1 ELSE 0, for k = 2..Nr."""
    unstable = (-sigmaR[:, 1:]) * gravitySign > 0.0
    return jnp.zeros_like(sigmaR).at[:, 1:].set(jnp.where(unstable, 1.0, 0.0))
