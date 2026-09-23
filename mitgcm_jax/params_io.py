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
