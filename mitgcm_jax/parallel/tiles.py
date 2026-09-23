"""Tile identity of the arrays a kernel holds (plan Task 7: one code path for P=1 and P=N).

Kernels work on "the tiles in the arrays they are given": all 13 tiles on one device, one block of tiles per device
under `jax.shard_map` (mitgcm_jax/parallel/shard.py). Almost every kernel is tile-agnostic. A few carry per-tile
STATIC tables built in numpy from the exch2 facet topology and indexed by the global tile number: the corner flags of
FILL_CS_CORNER_TR_RL (grad_sigma.fill_cs_corner_tr, mom_common.fill_cs_corner_tr_rl, mom_calc_relvort3), GGL90's
mskCor, the GAD pass tables. A kernel that holds only some tiles must use the rows of its own tiles.

The sharded driver adds `tile_index` (int32 [T]: the global 0-based tile number of every tile in the arrays; a padding
tile carries its donor's number) to the Grid; `tile_rows(table, tile_index(g))` picks those rows. Without
`tile_index` (the single-device path: all tiles, in order) the table is returned unchanged, so the single-device
program is the same as before sharding existed.
"""

import jax.numpy as jnp

TILE_INDEX = "tile_index"


def tile_index(g):
    """The global tile number of every tile in g's arrays ([T] int), or None for all tiles in order."""
    return g.f.get(TILE_INDEX)


def n_tiles(g):
    """Number of tiles in g's arrays: all tiles on one device, the local block (padding included) under shard_map."""
    return g.f["maskC"].shape[0]


def tile_rows(table, tiles):
    """Rows of a per-tile table [nTiles, ...] (indexed by global tile number) for the tiles held: `tiles` from
    `tile_index(g)`; None returns `table` itself (all tiles, in order)."""
    if tiles is None:
        return table
    return jnp.asarray(table)[tiles]
