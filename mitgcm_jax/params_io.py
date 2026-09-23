"""Run-directory namelists for parameter set-up (plan Task 8 uses this for Config).

Every parameter a kernel uses comes from the run's `data*` files, else from the Fortran default, which the caller
passes explicitly with its citation (never a Python-side guess):

    nml = RunNamelists(rundir)
    viscAh = nml.get("data", "parm01", "viscAh", default=0.0)   # set_defaults.F:NNN viscAh = 0.
Keys are case-insensitive. Scalars come back as scalars, arrays as lists (repeat counts expanded).
"""

from pathlib import Path

from mitgcm_jax.io.namelist import read_namelist


class RunNamelists:
    def __init__(self, rundir):
        self.dir = Path(rundir)
        self._cache = {}

    def file(self, name):
        if name not in self._cache:
            p = self.dir / name
            self._cache[name] = read_namelist(p) if p.exists() else {}
        return self._cache[name]

    def has(self, fname, group, key):
        return key.lower() in self.file(fname).get(group.lower(), {})

    def get(self, fname, group, key, default=None, array=False):
        g = self.file(fname).get(group.lower(), {})
        if key.lower() not in g:
            if default is None:
                raise KeyError(f"{fname}:{group}:{key} not set and no Fortran default given")
            return default
        v = g[key.lower()]
        return v if (array or len(v) != 1) else v[0]


def diagnostics_is_on(nml, diagName):
    """DIAGNOSTICS_IS_ON(diagName) (pkg/diagnostics/diagnostics_is_on.F:47-72) as a static set-up flag: .TRUE. when
    useDiagnostics and diagName is requested in an output list of data.diagnostics (&DIAGNOSTICS_LIST fields(:,n))
    whose frequency(n) > 0, or in a statistics list (&DIAG_STATIS_PARMS stat_fields(:,n)).

    Time dependence: DIAGNOSTICS_SWITCH_ONOFF (diagnostics_switch_onoff.F:81-117) sets ndiag < 0 between snapshots
    only for lists with frequency(n) < 0; a diagnostic requested only in such a list is on at some steps and off at
    others -> NotImplementedError (not a V4r4 case for the diagnostics the model physics asks about). Averaged lists
    (frequency > 0) stay on (ndiag >= 0) for the whole run."""
    if not bool(nml.get("data.pkg", "packages", "useDiagnostics", default=False)):   # packages_boot.F: .FALSE.
        return False
    name = diagName.strip()
    lists = nml.file("data.diagnostics").get("diagnostics_list", {})
    snap_only = False
    for key, vals in lists.items():
        if not key.startswith("fields("):
            continue
        if name not in [str(v).strip() for v in vals]:
            continue
        idx = key[len("fields("):-1].split(",")
        n = idx[1] if len(idx) == 2 else idx[0]          # fields(m,n) / fields(m1:m2,n)
        freq = float(lists.get(f"frequency({n})", [0.0])[0])
        if freq > 0.0:
            return True
        if freq < 0.0:
            snap_only = True
    stats = nml.file("data.diagnostics").get("diag_statis_parms", {})
    for key, vals in stats.items():
        if key.startswith("stat_fields(") and name in [str(v).strip() for v in vals]:
            return True
    if snap_only:
        raise NotImplementedError(f"diagnostic {name!r} is requested only in snapshot lists (frequency < 0): "
                                  "DIAGNOSTICS_IS_ON changes with time (diagnostics_switch_onoff.F:81-117)")
    return False


def params_pytree(cls):
    """Register a frozen parameter dataclass as a pytree: fields annotated `float` become leaves (traced when the
    dataclass is passed as a jit argument), every other field (int, bool, str, tuple, ...) is static metadata.

    Why: XLA's algebraic simplifier folds `(x*c1)*c2` into `x*(c1*c2)` when c1, c2 are compile-time constants
    (closed-over Python floats), changing the rounding by up to 1 ulp versus the Fortran order (measured: 42% of 1e5
    values differ; with c1, c2 traced, 0). Passing parameters as traced leaves keeps the Fortran operation order and
    is also what parameter sensitivities need. Usage:

        @params_pytree
        @dataclass(frozen=True)
        class GGL90Params:
            GGL90alpha: float
            mxlMaxFlag: int
            ...
        jax.jit(ggl90_calc)(params, g, ...)      # params passed as an argument, not closed over
    """
    import dataclasses

    import jax

    fields = dataclasses.fields(cls)
    data = [f.name for f in fields if f.type in (float, "float")]
    meta = [f.name for f in fields if f.name not in data]
    return jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
