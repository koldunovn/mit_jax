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
