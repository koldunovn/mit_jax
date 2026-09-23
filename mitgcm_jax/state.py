"""Model state carried from one time step to the next (plan Task 8).

`State` is a pytree: a dict of `[tile, (k,) j, i]` arrays named after the Fortran variables (DYNVARS.h, SURFACE.h,
GGL90.h, ...), plus the iteration counter. It holds every field the Fortran keeps in common blocks between steps and
reads in a step before (re)writing it; scratch fields a step always recomputes before use may be carried too (they
are compared against the dump at the start of the next step, which checks the step end to end).

Constructors:
  - `state_from_dump(ds, it)`: the Fortran oracle's state at the start of iteration `it` (stage S00_begin, plus the
    r* fields of G00_geometry group R, which are dumped at the same point). Used by step gates and as the exact
    initial condition until the pickup/initialisation port (Task 8) is gated against it.
  - `state_from_dump_full(ds, it)`: the same for the full V4r4 tree (plan M2.6b-2): plus the 'b' EXF_FIELDS arrays and
    the sea-ice state of S00i_begin_ice_exf and the partly-written SEAICE_DYNSOLVER arrays (DYN_CARRY).
"""

from dataclasses import dataclass

import jax
import numpy as np

from mitgcm_jax.layout import Layout

# S00_begin fields that are part of the carried state (the rest of S00_begin are diagnostics/scratch that are
# recomputed each step before use; they are still loaded for comparison when present)
S00_FIELDS = ["uVel", "vVel", "wVel", "theta", "salt", "etaN", "etaH", "dEtaHdt", "gU", "gV",
              "guNm_1", "guNm_2", "gvNm_1", "gvNm_2", "gtNm_1", "gtNm_2", "gsNm_1", "gsNm_2",
              "hFacC", "hFacW", "hFacS", "recip_hFacC", "rStarFacC", "rStarFacW", "rStarFacS",
              "GGL90TKE", "GGL90viscArU", "GGL90viscArV", "GGL90diffKr",
              "Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX", "GM_PsiY",
              "rhoInSitu", "IVDConvCount", "hMixLayer", "totPhiHyd", "phiHydLow",
              "surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "fu", "fv", "Qnet", "Qsw",
              "EmPmR", "saltFlux", "pLoad", "phi0surf", "sIceLoad", "saltPlumeDepth", "saltPlumeFlux",
              "aW2d", "aS2d", "aC2d", "pW", "pS", "pC",
              # EXF_FIELDS (group x): carried, their halos keep fldConst / exchanged values between steps
              "ustress", "vstress", "hflux", "sflux", "swflux", "apressure", "saltflx", "spflx"]
# full tree: the EXF_FIELDS arrays of the 'b' dump group (stage S00i_begin_ice_exf); the 'x' group arrays (ustress,
# vstress, hflux, sflux, swflux, apressure, saltflx) are in S00_FIELDS. Together: pkgs/exf_full.EXF_ARRAYS.
S00I_EXF_FIELDS = ["uwind", "vwind", "wspeed", "wStress", "cw", "sw", "sh", "atemp", "aqh", "hs", "hl", "lwflux", "evap",
                   "precip", "snowprecip", "swdown", "lwdown", "zen_albedo", "zen_fsol_diurnal", "zen_fsol_daily",
                   "runoff"]
# G00_geometry group R + recip_hFacW/S (time-dependent under z*)
G00_STATE_FIELDS = ["rStarFacNm1C", "rStarFacNm1W", "rStarFacNm1S", "rStarExpC", "rStarExpW", "rStarExpS",
                    "rStarDhCDt", "rStarDhWDt", "rStarDhSDt", "pStarFacK", "etaHnm1", "hFac_surfC", "hFac_surfW",
                    "hFac_surfS", "recip_hFacW", "recip_hFacS"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State:
    f: dict
    it: object = 0  # myIter at the start of the step (int or traced int)

    def __getattr__(self, name):
        f = object.__getattribute__(self, "f")
        if name in f:
            return f[name]
        raise AttributeError(name)

    def tree_flatten(self):
        keys = tuple(sorted(self.f))
        return tuple(self.f[k] for k in keys) + (self.it,), keys

    @classmethod
    def tree_unflatten(cls, keys, children):
        return cls(dict(zip(keys, children[:-1])), children[-1])

    def replace(self, it=None, **kw):
        f = dict(self.f)
        for k in kw:
            if k not in f:
                raise KeyError(f"State has no field {k!r}")
        f.update(kw)
        return State(f, self.it if it is None else it)

    def add(self, **kw):
        f = dict(self.f)
        f.update(kw)
        return State(f, self.it)


def state_from_dump(ds, it, layout=None):
    from mitgcm_jax.tests.oracle import field

    L = layout or Layout()
    f = {}
    for n in S00_FIELDS:
        try:
            f[n] = field(ds, it, "S00_begin", n, L)
        except KeyError:
            pass
    for n in G00_STATE_FIELDS:
        f[n] = field(ds, it, "G00_geometry", n, L)
    return State({k: np.asarray(v) for k, v in f.items()}, it)


def state_from_dump_full(ds, it, layout=None):
    """Full V4r4 tree: `state_from_dump` + S00I_EXF_FIELDS and the sea-ice state (AREA, HEFF, HSNOW, TICES, UICE,
    VICE) of stage S00i_begin_ice_exf + DYN_CARRY (pkgs/seaice_model): the I01_dynsolver values of iteration it-1 when
    that iteration is dumped (what SEAICE_DYNSOLVER left in the common blocks), else seaice_model.dyn_carry_init (the
    SEAICE_INIT_VARIA values: exact at the model start, and bitwise-equivalent later because every point SEAICE_DYNSOLVER
    reads is rewritten before it is read, tests/test_seaice_model.py::test_dyn_carry_reinit_equivalent)."""
    from mitgcm_jax.pkgs import seaice_model as sm
    from mitgcm_jax.tests.oracle import field

    L = layout or Layout()
    st = state_from_dump(ds, it, L)
    f = dict(st.f)
    for n in S00I_EXF_FIELDS + list(sm.ICE_STATE):
        f[n] = field(ds, it, "S00i_begin_ice_exf", n, L)
    if (it - 1, "I01_dynsolver", "e11") in ds.index:
        f.update({n: field(ds, it - 1, "I01_dynsolver", n, L) for n in sm.DYN_CARRY})
    else:
        f.update({n: np.asarray(v) for n, v in sm.dyn_carry_init(L).items()})
    return State({k: np.asarray(v) for k, v in f.items()}, it)
