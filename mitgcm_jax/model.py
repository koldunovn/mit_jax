"""Model set-up for the V4r4 flux-forced configuration: parameters of every package from the run directory's
namelists, the grid (literal port of the grid initialisation, or the oracle's dumped geometry), the exchanger.

    P, g, ex, kLowC = setup(rundir)                 # grid from the mitgrid/bathymetry files (Task 6 port)
    P, g, ex, kLowC = setup(rundir, grid=grid_from_dump(ds, 1))

Nothing here depends on the Fortran dumps unless a dump grid is passed in.

useCTRL=T (plan Task 8b): the mixing fields kapGM, kapRedi, diffKr of a grid built from files get the ctrl
adjustments of ff CTRL_MAP_INI_GENARR (xx_kapgm, xx_kapredi, xx_diffkr; pkgs/ctrl.py), which the Fortran applies in
INITIALISE_VARIA after INI_MIXING; a grid passed in is taken as it is (a dumped G00 grid already holds them) unless
ctrl_mixing=True. The state controls (etaN, theta, salt, uVel, vVel) are applied by init.state_from_pickup.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mitgcm_jax.core import dynamics as dyn_mod
from mitgcm_jax.core import external_forcing as ef
from mitgcm_jax.core import free_surface as fs
from mitgcm_jax.core import grad_sigma as rs_mod
from mitgcm_jax.core import thermodynamics as th_mod
from mitgcm_jax.core import tracers_correction as tc
from mitgcm_jax.core.cg2d import Cg2dParams, ini_cg2d_norm
from mitgcm_jax.core.forward_step import ModelParams
from mitgcm_jax.core.phi_hyd import kLowC_from_hFac
from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.io.mds import read_bin
from mitgcm_jax.layout import Layout
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.pkgs import ctrl as ctrl_mod
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.pkgs import gad as gad_mod
from mitgcm_jax.pkgs import ggl90 as ggl_mod
from mitgcm_jax.pkgs import gmredi as gm_mod
from mitgcm_jax.pkgs import mom_vecinv as mv_mod
from mitgcm_jax.pkgs import salt_plume as sp_mod

GRID_DIR = Path("/work/ab0995/a270088/MIT/data/eccov4r4/native_grid_files")


def _extra_grid_fields(nml, g, ex, rundir):
    """Fields the kernels need beyond grid_from_files: Bo_surf/recip_Bo (ini_linear_phisurf.F:69-78, usingZCoords:
    Bo_surf = gBaro, gBaro defaults to gravity, set_parms.F) and geothermalFlux (ini_forcing.F:141-150)."""
    L = g.layout
    gravity = float(nml.get("data", "parm01", "gravity", default=9.81))   # set_defaults.F: gravity = 9.81
    gBaro = float(nml.get("data", "parm01", "gBaro", default=gravity))    # set_parms.F: gBaro = gravity if unset
    Bo = np.full(L.shape2d, gBaro)
    rBo = np.full(L.shape2d, 1.0 / gBaro)
    geo = np.zeros(L.shape2d)
    gf = nml.get("data", "parm05", "geothermalFile", default=" ").strip()
    if gf:
        prec = int(nml.get("data", "parm01", "readBinaryPrec", default=32))
        a = compact_to_tiles(read_bin(Path(rundir) / gf, prec=prec)[0])
        geo[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = a
        geo = np.asarray(ex.exch_xy(geo))                                  # ini_forcing.F:145 EXCH_XY_RS
    return {"Bo_surf": Bo, "recip_Bo": rBo, "geothermalFlux": geo}


def _ctrl_mixing(nml, g, ex):
    """useCTRL: ff CTRL_MAP_INI_GENARR (ctrl_map_ini_genarr.F:134-145) on the INI_MIXING fields kapGM, kapRedi,
    diffKr (pkgs/ctrl.py; SMOOTH_IMPLDIFF sees the INI_MASKS_ETC recip_hFacC)."""
    from mitgcm_jax.init import ini_recip_hfac
    gj = type(g)({k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in g.f.items()}, g.layout)
    ci = ctrl_mod.ctrl_init(nml, gj, ex, ctrl_mod.MIXING_TARGETS, ini_recip_hfac(gj.h0FacC))
    out = ctrl_mod.ctrl_map_ini_genarr_jit(ex)(ci, gj, {k: gj.f[k] for k in ctrl_mod.MIXING_TARGETS})
    return g.replace(**{k: np.asarray(out[k]) for k in ctrl_mod.MIXING_TARGETS})


def setup(rundir, grid=None, grid_dir=GRID_DIR, layout=None, ctrl_mixing=None):
    L = layout or Layout()
    nml = RunNamelists(rundir)
    ex = default_exchanger(L)
    if ctrl_mixing is None:
        ctrl_mixing = grid is None
    if grid is None:
        from mitgcm_jax.grid.load import grid_from_files
        grid = grid_from_files(rundir, grid_dir, ex, L)
    g = grid.replace(**_extra_grid_fields(nml, grid, ex, rundir))
    if ctrl_mixing and nml.get("data.pkg", "packages", "useCTRL", default=False):
        g = _ctrl_mixing(nml, g, ex)                     # initialise_varia.F:219 PACKAGES_INIT_VARIABLES -> CTRL
    kLowC = np.asarray(kLowC_from_hFac(np.asarray(g.h0FacC)))
    fsp = fs.FreeSurfParams.from_namelists(nml)
    norm = ini_cg2d_norm(g, g.h0FacW, g.h0FacS, fsp.implicSurfPress, fsp.implicDiv2DFlow)
    P = ModelParams(
        exf=exf_mod.ExfParams.from_namelists(nml),
        sf=ef.SurfForcingParams.from_namelists(nml),
        rs=rs_mod.RhoSigmaParams.from_namelists(nml, g),
        sp=sp_mod.SaltPlumeParams.from_namelists(nml),
        ggl=ggl_mod.GGL90Params.from_namelists(nml, L.Nr),
        gm=gm_mod.GMRediParams.from_namelists(nml),
        dyn=dyn_mod.DynamicsParams.from_namelists(nml, L.Nr),
        mv=mv_mod.MomVecinvParams.from_namelists(nml),
        fs=fsp,
        cg=Cg2dParams.from_namelists(nml, norm),
        gadT=gad_mod.GADParams.from_namelists(nml, "temp"),
        gadS=gad_mod.GADParams.from_namelists(nml, "salt"),
        th=th_mod.ThermoParams.from_namelists(nml, g),
        tc=tc.TracersCorrectionParams.from_namelists(nml),
        ctrl=(ctrl_mod.CtrlConfig.from_namelists(nml) if nml.get("data.pkg", "packages", "useCTRL", default=False)
              else None),
    )
    g = type(g)({k: (jnp.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in g.f.items()}, g.layout)
    return P, g, ex, jnp.asarray(kLowC)
