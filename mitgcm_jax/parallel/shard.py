"""Tile-sharded model driver (plan Tasks 7-8): host-side padding and placement, and FORWARD_STEP under
`jax.shard_map(..., check_vma=True)` over the tile axis — the same kernels as on one device.

    sm = ShardedModel(g, nproc=4)                     # mesh, sharded exchanger, padded + placed grid
    st1, info = sm.run_step(P, kLowC, st, exf_in)     # one step; st1 unpadded, info = cg2d / r* diagnostics
    # or, for many steps without leaving the devices:
    f, it = sm.shard_state(st); k = sm.shard_tiles(kLowC)
    f, it, info = sm.step(P, k, f, it, sm.shard_exf(exf_in))

Layout. Every [tile, (k,) j, i] array is padded from nTiles (13) to Tpad = P*ceil(13/P) tiles and split into P
contiguous blocks (sharded_exchange.TileBlocks). Padding tiles are replicas of tile 1: the grid, state, forcing and
kLowC of tile 1, `Grid.tile_index` = 0 (so kernels pick tile 1's rows of their per-tile tables, parallel/tiles.py),
and tile 1's exchange sources. They therefore compute exactly what tile 1 computes (finite values, identical bits;
checked by the tests) and are dropped from the output; no real tile reads them (exchange sources are real tiles,
global sums and maxima skip them).

Inside the shard_map the kernels see the local block: `g` holds [Tloc, ...] arrays with the global Layout (index ranges
only; array shapes come from the arrays, `tiles.n_tiles`), `ex` is the local view of the ShardedExchanger (ppermute
exchanges, psum-based fixed-order global sums), everything else is the single-device code. With P=1 the same program
runs on one device (no ppermute rounds; the psum over one device is the identity).

Replicated inputs: the parameters (ModelParams: scalars and vertical profiles only; a leaf with a tile axis is an
error), 1-D vertical grid arrays, EXF interpolation weights and model time, the iteration number.
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from mitgcm_jax.adjoint.modes import EXACT
from mitgcm_jax.core.forward_step import forward_step
from mitgcm_jax.grid.geometry import Grid
from mitgcm_jax.parallel.exchange import MAP_DIR, ExchangeMaps
from mitgcm_jax.parallel.sharded_exchange import AXIS, ShardedExchanger, TileBlocks
from mitgcm_jax.parallel.tiles import TILE_INDEX
from mitgcm_jax.state import State

INFO_CG2D = ("numIters", "firstResidual", "lastResidual", "sumRHS", "rhsMax")


def is_tile_field(a, layout):
    """[nTiles, ..., ny, nx] array (a per-tile model field)."""
    s = np.shape(a)
    return len(s) >= 3 and s[0] == layout.nTiles and tuple(s[-2:]) == (layout.ny, layout.nx)


def tile_mesh(nproc, devices=None, axis_name=AXIS):
    devs = list(jax.devices() if devices is None else devices)
    if len(devs) < nproc:
        raise ValueError(f"{nproc} devices requested, {len(devs)} available")
    return Mesh(np.array(devs[:nproc]), (axis_name,))


class ShardedModel:
    """Host-side set-up of a P-device tile-sharded run and the jitted sharded FORWARD_STEP (compiled once)."""

    def __init__(self, g, nproc, maps=None, devices=None, axis_name=AXIS):
        L = g.layout
        self.L = L
        self.P = nproc
        self.blocks = TileBlocks(L.nTiles, nproc)
        self.mesh = tile_mesh(nproc, devices, axis_name)
        self.axis = axis_name
        self._tile = NamedSharding(self.mesh, PartitionSpec(axis_name))
        self._rep = NamedSharding(self.mesh, PartitionSpec())
        maps = maps or ExchangeMaps.load(MAP_DIR / f"exch_maps_{L.nTiles}x{L.sNx}x{L.sNy}.npz")
        self.ex = ShardedExchanger.build(maps, self.blocks, axis_name=axis_name).device_arrays(self.mesh)
        if TILE_INDEX in g.f:
            raise ValueError("the grid already carries a tile_index (pass the unsharded grid)")
        gf, gspec = {}, {}
        for k, v in g.f.items():
            if is_tile_field(v, L):
                gf[k], gspec[k] = self.put_tiles(v), PartitionSpec(axis_name)
            else:
                gf[k], gspec[k] = (jax.device_put(v, self._rep) if hasattr(v, "shape") else v), PartitionSpec()
        gf[TILE_INDEX] = jax.device_put(self.blocks.source_tile, self._tile)
        gspec[TILE_INDEX] = PartitionSpec(axis_name)
        self.g = Grid(gf, L)
        self._gspec = Grid(gspec, L)
        self._steps = {}

    # ------------------------------------------------------------------ host <-> devices
    def put_tiles(self, a):
        """[nTiles, ...] -> padded [Tpad, ...] placed with the tile axis sharded."""
        return jax.device_put(self.blocks.pad(np.asarray(a)), self._tile)

    def shard_tiles(self, a):
        return self.put_tiles(a)

    def shard_state(self, st):
        """State -> (f: dict of padded sharded fields, it)."""
        bad = [k for k, v in st.f.items() if not is_tile_field(v, self.L)]
        if bad:
            raise ValueError(f"State fields without the tile layout: {bad}")
        return {k: self.put_tiles(v) for k, v in st.f.items()}, st.it

    def shard_exf(self, exf_in):
        """EXF step inputs: record buffers (per tile) sharded, weights and time replicated (unchanged)."""
        out = dict(exf_in)
        out["bufs"] = {k: tuple(self.put_tiles(a) for a in v) for k, v in exf_in["bufs"].items()}
        return out

    def unpad(self, f):
        """dict of padded fields -> dict of [nTiles, ...] numpy arrays."""
        return {k: np.asarray(v)[:self.L.nTiles] for k, v in f.items()}

    # ------------------------------------------------------------------ the sharded step
    def _check_params(self, P):
        bad = [a.shape for a in jax.tree.leaves(P) if is_tile_field(a, self.L)]
        if bad:
            raise ValueError(f"ModelParams leaves with a tile axis (would be replicated, not sharded): {bad}")

    def _build(self, exf_keys, adj):
        ax = PartitionSpec(self.axis)
        rep = PartitionSpec()

        def body(P, g, ex, kLowC, f, it, exf_in):
            st1, aux = forward_step(P, g, ex, kLowC, State(f, it), exf_in, adj)
            info = ex.first_device({k: aux["cg2d"][k] for k in INFO_CG2D})  # equal on all devices, typed varying
            info.update(aux["rstar_checks"])
            return st1.f, st1.it, info

        exf_spec = {k: (ax if k == "bufs" else rep) for k in exf_keys}
        fn = jax.shard_map(body, mesh=self.mesh,
                           in_specs=(rep, self._gspec, ax, ax, ax, rep, exf_spec),
                           out_specs=(ax, rep, rep), check_vma=True)
        return jax.jit(fn)

    def step_fn(self, exf_keys=("bufs", "facs", "myTime"), adj=EXACT):
        """The jitted sharded step fn(P, g, ex, kLowC, f, it, exf_in) -> (f, it, info) (compiled once per mode)."""
        key = (tuple(sorted(exf_keys)), adj)
        if key not in self._steps:
            self._steps[key] = self._build(*key)
        return self._steps[key]

    def step(self, P, kLowC, f, it, exf_in, adj=EXACT):
        """One FORWARD_STEP on sharded inputs (from shard_state / shard_tiles / shard_exf): returns (f, it, info).
        adj: static AdjointConfig passed to forward_step."""
        self._check_params(P)
        return self.step_fn(tuple(exf_in), adj)(P, self.g, self.ex, kLowC, f, it, exf_in)

    def run_step(self, P, kLowC, st, exf_in, adj=EXACT):
        """One step from host data: returns (State with [nTiles, ...] numpy fields, info dict of Python numbers)."""
        f, it = self.shard_state(st)
        f1, it1, info = self.step(P, self.shard_tiles(kLowC), f, it, self.shard_exf(exf_in), adj)
        return State(self.unpad(f1), it1), {k: np.asarray(v).item() for k, v in info.items()}


def run_sharded(P, g, kLowC, st, exf_in, nproc=4, maps=None, devices=None, adj=EXACT):
    """FORWARD_STEP (flux-forced V4r4) sharded over `nproc` devices; same arguments as forward_step without the
    exchanger. Returns (State at the start of the next iteration, [nTiles, ...] numpy fields; info)."""
    return ShardedModel(g, nproc, maps=maps, devices=devices).run_step(P, kLowC, st, exf_in, adj)
