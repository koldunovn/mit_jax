"""Global sums in the Fortran tile order (plan Task 7).

c66g builds with GLOBAL_SUM_ORDER_TILES (eesupp/inc/CPP_EEOPTIONS.h:132). GLOBAL_SUM_TILE_RL
(eesupp/src/global_sum_tile.F) then receives every tile's partial sum (formed by the callers' `DO j; DO i` loops, e.g.
cg2d.F:181-183) and, with MPI (:164-194), zeroes a buffer over ALL tiles, puts its own tiles' partials in it
(:168-181), completes it with MPI_Allreduce(SUM) (:185-186) and adds the tiles in fixed order from 0 (:189-194):
sumPhi = 0 + phi(tile 1) + phi(tile 2) + ... . Without MPI (:198 onwards) the same ordered sum runs on the local
partials. The result does not depend on the decomposition, and neither does ours:

  - single device (`Exchanger.global_sum_tile`): `global_sum_tile` below on the [nTiles] partials;
  - sharded (`ShardedExchanger.global_sum_tile`): `all_tiles` is the zeroed buffer + Allreduce (a `psum` of the
    zero-padded [Tpad] block vector: each entry is one partial plus zeros, exact), then the same `global_sum_tile`
    on the nTiles real entries — the same additions in the same order for every P.
"""

import jax.numpy as jnp
from jax import lax


def global_sum_tile(phiTile):
    """GLOBAL_SUM_TILE_RL (global_sum_tile.F, serial / GLOBAL_SUM_ORDER_TILES): 0 + tile 1 + tile 2 + ... in order.
    phiTile: [nTiles, ...] partial sums (all tiles, tile order)."""
    s = jnp.zeros(phiTile.shape[1:], phiTile.dtype)
    for t in range(phiTile.shape[0]):
        s = s + phiTile[t]
    return s


def all_tiles(per_tile, blocks, axis_name):
    """Inside shard_map: per_tile [Tloc, ...] values of this device's tiles -> [nTiles, ...] values of every real tile
    in tile order, identical (and typed invariant) on every device. Padding tiles are dropped. The vector is formed
    by a psum of zero-padded blocks, so each entry is its tile's value exactly (x + 0 + ... + 0; only a -0.0 becomes
    +0.0, which no ordered sum starting from 0 can see). Differentiable: psum transposes to pvary."""
    d = lax.axis_index(axis_name)
    full = jnp.zeros((blocks.Tpad,) + per_tile.shape[1:], per_tile.dtype)
    full = lax.dynamic_update_slice_in_dim(full, per_tile, d * blocks.Tloc, axis=0)
    return lax.psum(full, axis_name)[:blocks.nTiles]
