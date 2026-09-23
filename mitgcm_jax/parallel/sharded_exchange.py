"""exch2 halo exchanges and Fortran-order global sums on tiles sharded over devices (plan Task 7, sharded path).

`ShardedExchanger` has the methods of the single-device `Exchanger` (scalar/vector kinds, exch_xy, exch_uv_xy, ...,
global_sum_tile, global_max) and is used INSIDE `jax.shard_map(..., check_vma=True)` over the mesh axis `AXIS`, on the
local block of every [tile, ..., j, i] array. It is a pytree: its per-device tables are leaves with a leading device
axis (shard them with PartitionSpec(AXIS)); the round structure is static.

Tile blocking (`TileBlocks`). The nTiles tiles (13 for LLC90 90x90, W2 order) are split into P contiguous blocks of
Tloc = ceil(nTiles/P); device d holds padded positions d*Tloc .. d*Tloc+Tloc-1. The tile axis is padded to
Tpad = P*Tloc. A padding tile is a REPLICA of the donor tile (tile 1): the driver gives it the donor's data and per-tile
tables (Grid.tile_index = donor) and the exchange gives it the donor's halo sources, so it computes bit for bit what
the donor computes: its values are finite whenever the donor's are, a global max is unchanged, and nothing reads it
(every map source is a real tile; global sums skip it). Its outputs are dropped.

Exchanges. The maps are the single-device ones (the probed Fortran exch2 routines, exchange.ExchangeMaps). For every
output point the host classifies its source as local (same device: a gather from the local arrays) or remote; the
remote requests are collected per (sender, receiver) device pair and the pairs are coloured into rounds in which
every device sends to at most one device and receives from at most one: one `lax.ppermute` per round. A vector kind
(u and v outputs) shares one set of rounds. On device an exchange is: one gather of the send buffer, K ppermutes, one
concatenation, one gather, `where(Fortran does not write the point, own value, gathered)`, times the sign — the
single-device formula with a larger source buffer, i.e. the same copies with the same signs: bitwise equal to the
single-device `Exchanger` for every P. Transpose: gather -> scatter-add, ppermute -> the inverse ppermute; JAX derives
both (no hand-written adjoint; the ragged all-to-all collective is banned: its JAX transpose is wrong).

Global sums: see global_sum.py (the Fortran GLOBAL_SUM_ORDER_TILES algorithm: zero-padded buffer, all-reduce, ordered
sum). `global_max` gathers per-tile maxima the same way (psum of zero-padded blocks, then max over the real tiles), so
it is differentiable (`lax.pmax` has no JVP) and invariant across the mesh axis.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitgcm_jax.parallel.exchange import SCALAR_KINDS, VECTOR_KINDS, Exchanger
from mitgcm_jax.parallel.global_sum import all_tiles, global_sum_tile

AXIS = "tile"


@dataclass(frozen=True)
class TileBlocks:
    """Contiguous tile blocks of P devices (tile t, 0-based, W2 order, lives on device t // Tloc)."""
    nTiles: int
    P: int
    donor: int = 0

    @property
    def Tloc(self):
        return -(-self.nTiles // self.P)

    @property
    def Tpad(self):
        return self.P * self.Tloc

    @property
    def source_tile(self):
        """[Tpad] int32: the real tile whose data, tables and halo sources every padded position carries."""
        return np.array(list(range(self.nTiles)) + [self.donor] * (self.Tpad - self.nTiles), np.int32)

    def pad(self, a):
        """[nTiles, ...] -> [Tpad, ...] (padding tiles = copies of the donor tile)."""
        a = np.asarray(a)
        if a.shape[0] != self.nTiles:
            raise ValueError(f"leading axis {a.shape[0]} is not the tile axis ({self.nTiles})")
        return a[self.source_tile]

    def unpad(self, a):
        return a[:self.nTiles]


# ---------------------------------------------------------------------------------------------------------------
# host-side plan


def colour_pairs(pairs, P):
    """Directed device pairs (sender, receiver) -> rounds, each a partial permutation (every device sends at most
    once and receives at most once per round). Greedy in the order of the cyclic shift (receiver - sender) mod P, so a
    complete neighbour graph takes P-1 rounds (one per shift)."""
    rounds = []
    for e, d in sorted(pairs, key=lambda ed: ((ed[1] - ed[0]) % P, ed[0])):
        for r in rounds:
            if e not in r[1] and d not in r[2]:
                r[0].append((e, d))
                r[1].add(e)
                r[2].add(d)
                break
        else:
            rounds.append(([(e, d)], {e}, {d}))
    return [tuple(r[0]) for r in rounds]


@dataclass(frozen=True)
class KindPlan:
    """Static round structure of one exchange kind (hashable: pytree aux data)."""
    outputs: tuple      # map names of the outputs: ("T",) or ("UVs_u", "UVs_v")
    ncomp: int          # source arrays in the local buffer (1 scalar, 2 vector: [u | v])
    perms: tuple        # per round: ((sender, receiver), ...)
    slots: tuple        # per round: values per message
    offs: tuple         # per round: offset in the send buffer

    @property
    def total(self):
        return int(sum(self.slots))


def build_kind(maps, outputs, blocks, layout):
    """Plan and per-device tables of one kind. maps: ExchangeMaps.maps; outputs: the map names of the outputs.

    Returns (KindPlan, tables) with tables (numpy, leading device axis P):
      loc [P, nout, N] int32   index into the receiver's source vector [own buffer (ncomp*N) | round 0 | round 1 ...]
      keep [P, nout, N] bool   the Fortran exchange does not write the point (keeps its own value)
      sign [P, nout, N] f64    sign of the copy (+1 where kept)
      send_idx [P, S] int32    index into the sender's own buffer of every value it sends (S = max(total, 1))
      send_ok [P, S] bool      slot holds a requested value (else padding of a shorter message, sent as 0)
    """
    L = layout
    n = L.ny * L.nx
    P, Tloc = blocks.P, blocks.Tloc
    N = Tloc * n
    ncomp = len(outputs)
    st = blocks.source_tile
    nout = len(outputs)
    loc = np.zeros((P, nout, N), np.int64)
    keep = np.zeros((P, nout, N), bool)
    sign = np.ones((P, nout, N))
    remote = []                              # (d, o, point mask, sender device per point, buffer index per point)
    req = {}                                 # (sender, receiver) -> unique buffer indices requested
    for o, name in enumerate(outputs):
        src, comp, sgn = maps[name]
        src = np.asarray(src, np.int64).reshape(blocks.nTiles, n)
        comp = np.asarray(comp, np.int64).reshape(blocks.nTiles, n)
        sgn = np.asarray(sgn, np.float64).reshape(blocks.nTiles, n)
        for d in range(P):
            tiles = st[d * Tloc:(d + 1) * Tloc]      # map rows of the local positions (padding: the donor's)
            s, c, g = src[tiles].reshape(-1), comp[tiles].reshape(-1), sgn[tiles].reshape(-1)
            ts, p = s // n, s % n                     # source tile (real) and point
            if np.any(ts >= blocks.nTiles):
                raise ValueError(f"{name}: a map source outside the real tiles")
            e = ts // Tloc                            # device holding the source tile
            bidx = (np.maximum(c, 1) - 1) * N + (ts - e * Tloc) * n + p   # index in e's [u | v] buffer
            w = c > 0
            keep[d, o] = ~w
            sign[d, o] = g
            lo = w & (e == d)
            loc[d, o, lo] = bidx[lo]
            rm = w & (e != d)
            remote.append((d, o, rm, e, bidx))
            for ee in np.unique(e[rm]):
                req.setdefault((int(ee), d), []).append(bidx[rm & (e == ee)])
    req = {k: np.unique(np.concatenate(v)) for k, v in req.items()}
    rounds = colour_pairs(list(req), P)
    slots = tuple(int(max(len(req[pr]) for pr in r)) for r in rounds)
    offs = tuple(int(x) for x in np.concatenate([[0], np.cumsum(slots)])[:-1]) if slots else ()
    round_of = {pr: k for k, r in enumerate(rounds) for pr in r}
    S = max(int(sum(slots)), 1)
    send_idx = np.zeros((P, S), np.int64)
    send_ok = np.zeros((P, S), bool)
    for (e, d), idx in req.items():
        k = round_of[(e, d)]
        send_idx[e, offs[k]:offs[k] + len(idx)] = idx
        send_ok[e, offs[k]:offs[k] + len(idx)] = True
    for d, o, rm, e, bidx in remote:
        for ee in np.unique(e[rm]):
            m = rm & (e == ee)
            k = round_of[(int(ee), d)]
            loc[d, o, m] = ncomp * N + offs[k] + np.searchsorted(req[(int(ee), d)], bidx[m])
    plan = KindPlan(tuple(outputs), ncomp, tuple(tuple(r) for r in rounds), slots, offs)
    tabs = dict(loc=loc.astype(np.int32), keep=keep, sign=sign, send_idx=send_idx.astype(np.int32),
                send_ok=send_ok)
    return plan, tabs


def kind_outputs(kind):
    if kind in SCALAR_KINDS:
        return (kind,)
    if kind in VECTOR_KINDS:
        return (kind + "_u", kind + "_v")
    raise KeyError(kind)


# ---------------------------------------------------------------------------------------------------------------
# device side


@jax.tree_util.register_pytree_node_class
class ShardedExchanger(Exchanger):
    """Halo exchanges and global reductions on the local tile block, inside shard_map over AXIS.

    Build on the host with `ShardedExchanger.build(maps, blocks)`; pass it into the shard_map'd function with
    in_specs PartitionSpec(AXIS) for all its leaves (every table has the device axis first). The Fortran-name methods
    (exch_xy, exch_uv_xy, exch_z, exch_uv_agrid, exch_uv_bgrid) are inherited from Exchanger and call `scalar` /
    `vector` below."""

    def __init__(self, layout, blocks, plans, tabs, axis_name=AXIS):
        self.L = layout                  # the global layout (kernels use it for index ranges only)
        self.blocks = blocks
        self.plans = plans               # kind -> KindPlan
        self.tabs = tabs                 # kind -> dict of arrays [P or 1, ...]
        self.axis_name = axis_name

    @classmethod
    def build(cls, maps, blocks, kinds=None, axis_name=AXIS):
        """maps: exchange.ExchangeMaps; blocks: TileBlocks; kinds: exchange kinds to prepare (default: all probed)."""
        L = maps.layout
        if L.nTiles != blocks.nTiles:
            raise ValueError("maps and tile blocks disagree on the number of tiles")
        if kinds is None:
            kinds = [k for k in list(SCALAR_KINDS) + list(VECTOR_KINDS)
                     if all(o in maps.maps for o in kind_outputs(k))]
        plans, tabs = {}, {}
        for k in kinds:
            plans[k], tabs[k] = build_kind(maps.maps, kind_outputs(k), blocks, L)
        return cls(L, blocks, plans, tabs, axis_name)

    # pytree: tables are leaves, the rest static
    def tree_flatten(self):
        kinds = tuple(sorted(self.tabs))
        names = ("loc", "keep", "sign", "send_idx", "send_ok")
        leaves = tuple(self.tabs[k][nm] for k in kinds for nm in names)
        aux = (self.L, self.blocks, tuple((k, self.plans[k]) for k in kinds), names, self.axis_name)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        L, blocks, plans, names, axis_name = aux
        tabs, i = {}, 0
        for k, _ in plans:
            tabs[k] = dict(zip(names, leaves[i:i + len(names)]))
            i += len(names)
        return cls(L, blocks, dict(plans), tabs, axis_name)

    def device_arrays(self, mesh):
        """The same exchanger with its tables placed on `mesh`, sharded over the device axis."""
        from jax.sharding import NamedSharding, PartitionSpec
        sh = NamedSharding(mesh, PartitionSpec(self.axis_name))
        return jax.tree.map(lambda a: jax.device_put(a, sh), self)

    # ---------------------------------------------------------------- exchanges (inside shard_map)
    def _flat(self, a):
        """[Tloc, ..., ny, nx] -> ([..., Tloc*ny*nx], lead, Tloc)"""
        a = jnp.moveaxis(jnp.asarray(a), 0, -3)
        lead, T = a.shape[:-3], a.shape[-3]
        return a.reshape(*lead, T * self.L.ny * self.L.nx), lead, T

    def _unflat(self, f, lead, T):
        return jnp.moveaxis(f.reshape(*lead, T, self.L.ny, self.L.nx), -3, 0)

    def _exchange(self, kind, arrays):
        plan = self.plans[kind]
        if self.tabs[kind]["loc"].shape[0] != 1:
            raise ValueError("ShardedExchanger works inside shard_map (tables sharded over the device axis)")
        tab = {k: v[0] for k, v in self.tabs[kind].items()}    # local block: [1, ...] -> [...]
        flats = [self._flat(a) for a in arrays]
        lead, T = flats[0][1], flats[0][2]
        if T != self.blocks.Tloc:
            raise ValueError(f"local tile block has {T} tiles, the exchanger expects {self.blocks.Tloc}")
        own = [f for f, _, _ in flats]
        buf = own[0] if len(own) == 1 else jnp.concatenate(own, axis=-1)
        parts = [buf]
        if plan.perms:
            send = jnp.where(tab["send_ok"], buf[..., tab["send_idx"]], jnp.zeros((), buf.dtype))
            for perm, s, o in zip(plan.perms, plan.slots, plan.offs):
                parts.append(lax.ppermute(send[..., o:o + s], self.axis_name, perm=list(perm)))
        src = jnp.concatenate(parts, axis=-1) if len(parts) > 1 else buf
        outs = []
        for o in range(len(plan.outputs)):
            out = jnp.where(tab["keep"][o], own[o], src[..., tab["loc"][o]]) * tab["sign"][o]
            outs.append(self._unflat(out, lead, T))
        return outs

    def scalar(self, a, kind="T"):
        return self._exchange(kind, [a])[0]

    def vector(self, u, v, kind="UVs"):
        uo, vo = self._exchange(kind, [u, v])
        return uo, vo

    # ---------------------------------------------------------------- global reductions (inside shard_map)
    def all_tiles(self, per_tile):
        """[Tloc, ...] values of the local tiles -> [nTiles, ...] of every real tile, tile order, on every device."""
        return all_tiles(per_tile, self.blocks, self.axis_name)

    def global_sum_tile(self, phiTile):
        """GLOBAL_SUM_TILE_RL: local per-tile partials [Tloc] -> 0 + tile 1 + ... + tile nTiles (every P)."""
        return global_sum_tile(self.all_tiles(phiTile))

    def global_max(self, a):
        """_GLOBAL_MAX_RL over every real tile of a local [Tloc, ...] array."""
        a = jnp.asarray(a)
        return jnp.max(self.all_tiles(jnp.max(a.reshape(a.shape[0], -1), axis=1)))

    # ---------------------------------------------------------------- sharding types (inside shard_map)
    def vary(self, x):
        """x (pytree) typed as varying over the tile axis; values unchanged. Needed around custom_linear_solve (cg2d)
        in JAX 0.10.1: the solver's aux outputs (diagnostics, equal on every device) must vary like its right-hand
        side, and invariant values the solver closes over must be varying before the call (else lowering fails with
        "pvary is a invariant->variant collective")."""
        ax = self.axis_name

        def one(a):
            varying = getattr(getattr(jax.typeof(a), "mat", None), "varying", frozenset())
            return a if ax in varying else lax.pcast(a, ax, to="varying")
        return jax.tree.map(one, x)

    def real_tiles(self):
        """[Tloc] bool: the local tiles that are real (not padding)."""
        return lax.axis_index(self.axis_name) * self.blocks.Tloc + jnp.arange(self.blocks.Tloc) < self.blocks.nTiles

    def zero_padding(self, a):
        """a [Tloc, ...] with the padding tiles set to 0."""
        a = jnp.asarray(a)
        return jnp.where(self.real_tiles().reshape((-1,) + (1,) * (a.ndim - 1)), a, jnp.zeros((), a.dtype))

    def first_device(self, x):
        """x (pytree) as held by device 0, typed invariant (psum of x on device 0 and zeros elsewhere: exact). For
        values that are equal on every device but typed varying (`vary`), e.g. to return them replicated."""
        ax = self.axis_name
        on0 = lax.axis_index(ax) == 0
        return jax.tree.map(lambda a: lax.psum(jnp.where(on0, a, jnp.zeros_like(a)), ax), x)

    # ---------------------------------------------------------------- introspection
    def rounds(self, kind):
        """Number of ppermute rounds (= collective-permutes) of one exchange of `kind`."""
        return len(self.plans[kind].perms)
