"""pkg/ctrl of the V4r4 flux-forced build with useCTRL=T, ctrlUseGen=T (plan Task 8b), literally.

Initial-condition and parameter controls (CTRL_INIT_VARIABLES -> CTRL_MAP_INI_GENARR, ctrl_init_variables.F:393-397;
called from PACKAGES_INIT_VARIABLES, packages_init_variables.F:496-503, i.e. after CALC_PHI_RLOW_INI / INI_MIXING and
before the r* / INTEGR_CONTINUITY sequence of INITIALISE_VARIA). ff ctrl_map_ini_genarr.F, for each generic control
whose weight file is set (mult_genarr* only weights the cost, ctrl_cost_gen.F -- it does not switch a control off):

    xx   = read xx_<name>.<optimcycle %010d> (ctrlprec = 32, interior; zero halos)       :261-269 / :418-426
    xx   = SMOOTH_CORREL2D/3D(xx) for preproc 'WC01' (pkg/smooth, operator number preproc_i) :271-276 / :428-433
    xx   = xx / sqrt(weight) where mask .NE. 0 and weight > 0, else 0 (unless 'noscaling') :278-291 / :447-465
    fld  = fld + xx (interior)                                                            :293 / :467
    CTRL_BOUND_2D/3D(fld, mask, xx_genarr*_bounds)                                       :300 / :475-480
    EXCH_XY_RL / EXCH_XYZ_RL(fld) (u, v: one EXCH_UV_XYZ_RL(uVel, vVel, .TRUE.) after both, :146-151)

for etaN (xx_etan; etaH is not adjusted, :85), theta, salt, kapGM, kapRedi (ALLOW_KAPGM/KAPREDI_CONTROL), diffKr
(ALLOW_3D_DIFFKR), uVel/vVel (ALLOW_UVEL0/VVEL0_CONTROL; masks maskW/maskS, :436-445). Each field's adjustment
depends only on its own control, weight, mask and value, so the mixing fields can be adjusted where the grid is
built (model.setup) and the state fields in INITIALISE_VARIA (init.state_from_pickup) without changing a value.

    cc  = CtrlConfig.from_namelists(nml)                           # data.ctrl, data.optim, data.pkg (static)
    ci  = ctrl_init(nml, g, ex, targets, recip_hFacC)              # xx, weights, pkg/smooth operators, WC01(xx)
    out = ctrl_map_ini_genarr(ci, g, ex, fields)                   # jittable; fields: {name: array} to adjust

SMOOTH_CORREL2D/3D reads only the control and the smoothing operator (not the model state), so ctrl_init evaluates
it once per control (`ctrl_smooth`, one compiled smoother vmapped over the controls that share an operator: the same
elementwise operations per control) and ctrl_map_ini_genarr continues with the scaling, add, bound and exchange. With
ctrl_init(..., smooth=False) the smoothing runs inside ctrl_map_ini_genarr instead: one traced map xx -> fields (for
derivatives with respect to the controls). Cost (LLC90, 16 CPU cores): ~0.2-0.26 s per pseudo-time step and 3-D
control, 150 steps; more than half of it is the exch2 gathers (7 exchanges per step).

Time-varying (forcing) controls xx_gentim2d (CTRL_MAP_INI_GENTIM2D, CTRL_MAP_GENTIM2D -> CTRL_GET_GEN every step,
added in ff exf_getffields.F:681-760, exf_getsurfacefluxes.F:106-161 and CTRL_MAP_FORCING, ctrl_map_forcing.F:298-376):
the flux-forced xx_qnet, xx_empmr, xx_qsw, xx_saltflux, xx_pload, xx_tauu, xx_tauv, xx_spflx are identically zero
(every record; the adjustments are folded into the forcing files, atm_flux_forcing_experiments/xx/README). Only that
case is ported: `require_zero_forcing_controls` raises NotImplementedError for a non-zero record (or a bound that
would move 0). With zero controls every add is x + 0 (value-identical), and the one remaining forward effect is the
exchanges of CTRL_MAP_FORCING (`ctrl_map_forcing`): EXCH_XY_RS(saltFlux) rewrites the saltFlux halos, which
EXF_MAPFIELDS does not exchange (oracle ref_ff_jaxdump_v4: S02 vs S03 differ in 1538 saltFlux halo values).
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.io.mds import read_bin
from mitgcm_jax.pkgs import smooth as smooth_mod
from mitgcm_jax.pkgs.smooth import nml_array

CTRLPREC = 32          # ctrl.h:36 ctrlprec = 32 (ff CTRL_OPTIONS.h:33 CTRL_SET_PREC_32)
MAX_CTRL_ARR2D = 1     # ff CTRL_SIZE.h:19
MAX_CTRL_ARR3D = 7     # ff CTRL_SIZE.h:22
MAX_CTRL_TIM2D = 14    # ff CTRL_SIZE.h:25
MAX_CTRL_PROC = 3      # ff CTRL_SIZE.h:28
NML = "ctrl_nml_genarr"

# ff ctrl_map_ini_genarr.F:72-81 / :109-128: (prefix compared with file(1:n), model field). The last matching iarr
# wins, as in the Fortran loops.
TARGETS2D = (("xx_etan", "etaN"),)
TARGETS3D = (("xx_theta", "theta"), ("xx_salt", "salt"), ("xx_kapgm", "kapGM"), ("xx_kapredi", "kapRedi"),
             ("xx_diffkr", "diffKr"), ("xx_uvel", "uVel"), ("xx_vvel", "vVel"))
ORDER = ("etaN", "theta", "salt", "kapGM", "kapRedi", "diffKr", "uVel", "vVel")   # :83-151 call order
MASKS = {"uVel": "maskW", "vVel": "maskS"}                                         # :436-445, else maskC
STATE_TARGETS = ("etaN", "theta", "salt", "uVel", "vVel")                           # DYNVARS.h
MIXING_TARGETS = ("kapGM", "kapRedi", "diffKr")                                     # GMREDI / 3-D diffKr


@dataclass(frozen=True)
class GenArr:
    """One generic 2-D/3-D control (xx_genarr*_(iarr)) with the flags ctrl_map_genarr2d/3d derive (:232-252,
    :389-409)."""
    iarr: int
    file: str
    weight: str
    bounds: tuple
    dowc01: bool
    dosmooth: bool
    doscaling: bool
    numsmo: int

    @classmethod
    def build(cls, iarr, file, weight, bounds, preproc, preproc_i):
        dosmooth, dowc01, doscaling = False, False, True
        numsmo = 1
        for k2 in range(MAX_CTRL_PROC):
            p = str(preproc[k2]).strip()
            if p == "WC01":
                dowc01 = True
                if preproc_i[k2] != 0:
                    numsmo = int(preproc_i[k2])
            if (not dowc01) and p == "smooth":
                dosmooth = True
                if preproc_i[k2] != 0:
                    numsmo = int(preproc_i[k2])
            if p == "noscaling":
                doscaling = False
        return cls(iarr, str(file).strip(), str(weight).strip(), tuple(float(b) for b in bounds), dowc01, dosmooth,
                   doscaling, numsmo)


@dataclass(frozen=True)
class CtrlConfig:
    """Static ctrl configuration (hashable: jit metadata of CtrlInit)."""
    optimcycle: int
    useSMOOTH: bool
    arr2d: tuple          # ((model field, GenArr), ...) in the order of ORDER
    arr3d: tuple
    tim2d: tuple          # ((iarr, file, weight, bounds), ...) with the weight set

    @classmethod
    def from_namelists(cls, nml):
        pkg = lambda k: bool(nml.get("data.pkg", "packages", k, default=False))  # noqa: E731
        if not pkg("useCTRL"):
            raise ValueError("useCTRL=F: no ctrl configuration")
        c = lambda k, d: nml.get("data.ctrl", "ctrl_nml", k, default=d)  # noqa: E731
        bad = []
        if not bool(c("ctrlUseGen", True)):          # ctrl_readparms.F:195 (ALLOW_GEN*_CONTROL defined)
            bad.append("ctrlUseGen=F (CTRL_MAP_INI_ECCO / CTRL_MAP_INI, ctrl_init_variables.F:380-391)")
        optimcycle = int(nml.get("data.optim", "optim", "optimcycle", default=0))    # optim_readparms.F:76
        doinitxx = bool(c("doInitXX", True))                                          # ctrl_readparms.F:183
        if bool(c("doMainUnpack", True)) and (optimcycle != 0 or not doinitxx):      # ctrl_readparms.F:207
            bad.append("doMainUnpack (CTRL_UNPACK, ff the_model_main.F:640-649)")
        if bad:
            raise NotImplementedError("ctrl branch not ported: " + "; ".join(bad))

        def arrays(kind, n):
            f = nml_array(nml, "data.ctrl", NML, f"xx_{kind}_file", n, " ")
            w = nml_array(nml, "data.ctrl", NML, f"xx_{kind}_weight", n, " ")
            b = _bounds(nml, kind, n)
            pp = _proc(nml, kind, "preproc", n, " ")
            pi = _proc(nml, kind, "preproc_i", n, 0)
            return f, w, b, pp, pi

        def pick(kind, n, targets):
            f, w, b, pp, pi = arrays(kind, n)
            found = {}
            for iarr in range(n):                                    # ctrl_map_ini_genarr.F:72-81, 109-128
                if str(w[iarr]).strip() == "":
                    continue
                name = str(f[iarr]).strip()
                for prefix, field in targets:
                    if name.startswith(prefix):
                        found[field] = GenArr.build(iarr + 1, f[iarr], w[iarr], b[iarr], pp[iarr], pi[iarr])
                if kind == "genarr2d" and name.startswith("xx_geothermal"):
                    bad.append("xx_geothermal (ctrl_map_ini_genarr.F:91-94, ALLOW_GEOTHERMAL_FLUX)")
                # xx_bottomdrag: ALLOW_BOTTOMDRAG_CONTROL undefined (ff CTRL_OPTIONS.h) -> no call (:87-90)
            return found

        a2 = pick("genarr2d", MAX_CTRL_ARR2D, TARGETS2D)
        a3 = pick("genarr3d", MAX_CTRL_ARR3D, TARGETS3D)
        if ("uVel" in a3) != ("vVel" in a3):                          # :147 igen_uvel0.GT.0 .and. igen_vvel0.GT.0
            a3.pop("uVel", None)
            a3.pop("vVel", None)
        for spec in list(a2.values()) + list(a3.values()):
            if spec.dosmooth:
                bad.append(f"{spec.file}: preproc 'smooth' (SMOOTH2D/SMOOTH3D, ctrl_map_ini_genarr.F:274, 431)")
        f, w, b, pp, pi = arrays("gentim2d", MAX_CTRL_TIM2D)
        tim = tuple((i + 1, str(f[i]).strip(), str(w[i]).strip(), tuple(float(x) for x in b[i]))
                    for i in range(MAX_CTRL_TIM2D) if str(w[i]).strip() != "")
        if bad:
            raise NotImplementedError("ctrl branch not ported: " + "; ".join(bad))
        return cls(optimcycle=optimcycle, useSMOOTH=pkg("useSMOOTH"),
                   arr2d=tuple((k, a2[k]) for k in ORDER if k in a2),
                   arr3d=tuple((k, a3[k]) for k in ORDER if k in a3), tim2d=tim)

    def spec(self, field):
        for k, s in self.arr2d + self.arr3d:
            if k == field:
                return s
        return None

    @property
    def fields(self):
        return tuple(k for k, _ in self.arr2d + self.arr3d)


# a static pytree node: CtrlConfig can sit in a jit-argument pytree (e.g. a ModelParams field for CTRL_MAP_FORCING)
jax.tree_util.register_static(CtrlConfig)


def _bounds(nml, kind, n):
    """xx_<kind>_bounds(1:5, iarr) (ctrl_readparms.F:236-238 default 0), as written `(1:5,i)` or `(j,i)`."""
    out = [[0.0] * 5 for _ in range(n)]
    g = nml.file("data.ctrl").get(NML, {})
    key = f"xx_{kind}_bounds("
    for k, v in g.items():
        if not k.startswith(key):
            continue
        idx = k[len(key):-1].split(",")
        if len(idx) != 2:
            raise NotImplementedError(f"data.ctrl {k}: bounds index form not supported")
        i = int(idx[1]) - 1
        if ":" in idx[0]:
            lo = int(idx[0].split(":")[0]) - 1
        else:
            lo = int(idx[0]) - 1
        for j, x in enumerate(v):
            out[i][lo + j] = float(x)
    return out


def _proc(nml, kind, what, n, default):
    """xx_<kind>_<what>(k2, iarr), k2 = 1..maxCtrlProc (ctrl_readparms.F:239-244 defaults)."""
    out = [[default] * MAX_CTRL_PROC for _ in range(n)]
    g = nml.file("data.ctrl").get(NML, {})
    key = f"xx_{kind}_{what}("
    for k, v in g.items():
        if not k.startswith(key):
            continue
        k2, i = (int(x) for x in k[len(key):-1].split(","))
        out[i - 1][k2 - 1] = v[0]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# host side: file reads (MDS_READ_FIELD, ctrlprec) and pkg/smooth set-up
# ---------------------------------------------------------------------------------------------------------------------
def _mds_path(rundir, name):
    """MDS_READ_FIELD: the plain file name if it exists, else <name>.data (global file)."""
    p = Path(rundir) / name
    return p if p.exists() else Path(str(p) + ".data")


def _read_interior(path, L, nz):
    """record 1 of a global real*4 file (ctrlprec) -> [T, (nz,) ny, nx] with the interior filled, zero halos."""
    if nz > 1:
        a = read_bin(path, nz=nz, prec=CTRLPREC)[0]                  # (nz, 1170, 90)
        t = np.moveaxis(compact_to_tiles(a), -3, 0)                 # (T, nz, 90, 90)
        out = np.zeros(L.shape3d)
    else:
        a = read_bin(path, prec=CTRLPREC)[0]                         # (1170, 90)
        t = compact_to_tiles(a)                                      # (T, 90, 90)
        out = np.zeros(L.shape2d)
    out[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = t
    return out


def ctrl_inputs(cc: CtrlConfig, rundir, L, fields):
    """xx_<name>.<optimcycle> and the weight file of every control in `fields` (ctrl_map_ini_genarr.F:261-269,
    :418-426): {field: {"xx": [T,(Nr,)ny,nx], "w": same}} (interior from the files, zero halos)."""
    out = {}
    for field in fields:
        spec = cc.spec(field)
        if spec is None:
            continue
        nz = 1 if field in dict(cc.arr2d) else L.Nr
        xx = _read_interior(_mds_path(rundir, f"{spec.file}.{cc.optimcycle:010d}"), L, nz)
        w = _read_interior(_mds_path(rundir, spec.weight), L, nz)
        out[field] = {"xx": jnp.asarray(xx), "w": jnp.asarray(w)}
    return out


@dataclass(frozen=True)
class CtrlInit:
    """Everything ctrl_map_ini_genarr needs besides the grid: static configs and the array inputs (a pytree).
    smoothed: inputs[f]["xx"] already holds SMOOTH_CORREL2D/3D(xx) for the WC01 controls (ctrl_smooth)."""
    cc: CtrlConfig
    scfg: object          # smooth.SmoothConfig
    sp: object            # smooth.SmoothParams
    ops: dict             # smooth.smooth_init_fixed(...)
    inputs: dict          # ctrl_inputs(...)
    recip_hFacC: object   # GRID.h recip_hFacC at CTRL_INIT_VARIABLES (INI_MASKS_ETC value)
    smoothed: bool = False


jax.tree_util.register_dataclass(CtrlInit, data_fields=["sp", "ops", "inputs", "recip_hFacC"],
                                 meta_fields=["cc", "scfg", "smoothed"])


def ctrl_init(nml, g, ex, fields, recip_hFacC, smooth=True):
    """Host-side set-up for the controls of `fields` (subset of ORDER): configuration, control/weight files and the
    pkg/smooth operators (SMOOTH_INIT_FIXED, packages_init_fixed.F; smooth_init_fixed.F); with smooth=True also
    SMOOTH_CORREL2D/3D of the WC01 controls (ctrl_smooth). recip_hFacC: INI_MASKS_ETC recip_hFacC (1/h0FacC where
    h0FacC .NE. 0, else 0)."""
    cc = CtrlConfig.from_namelists(nml)
    scfg = smooth_mod.SmoothConfig.from_namelists(nml) if cc.useSMOOTH else None
    sp = smooth_mod.SmoothParams.from_namelists(nml)
    fields = tuple(f for f in fields if cc.spec(f) is not None)
    need = [f for f in fields if cc.spec(f).dowc01 and cc.useSMOOTH]
    ops = smooth_mod.smooth_init_fixed(scfg, g, ex, nml.dir, recip_hFacC) if need else {"h": {}, "3d": {}, "2d": {}}
    for f in need:                                     # the operator number must be defined in data.smooth
        (scfg.op2d if f in dict(cc.arr2d) else scfg.op3d)(cc.spec(f).numsmo)
    ci = CtrlInit(cc=cc, scfg=scfg, sp=sp, ops=ops, inputs=ctrl_inputs(cc, nml.dir, g.layout, fields),
                  recip_hFacC=jnp.asarray(recip_hFacC))
    return ctrl_smooth_jit(ex)(ci, g) if smooth else ci


def _wc01(ci: CtrlInit, spec, field):
    return ci.cc.useSMOOTH and spec.dowc01 and not ci.smoothed


def ctrl_smooth(ci: CtrlInit, g, ex):
    """SMOOTH_CORREL2D(xx, maskC, numsmo) / SMOOTH_CORREL3D(xx, numsmo) (ctrl_map_ini_genarr.F:271-276, :428-433) of
    every WC01 control in ci.inputs; controls sharing an operator are smoothed together (jax.vmap over a control
    axis after the tile axis: the same operations on each). Returns ci with smoothed=True."""
    if ci.smoothed:
        return ci
    inputs = {k: dict(v) for k, v in ci.inputs.items()}
    groups = {}
    for field, spec in ci.cc.arr2d + ci.cc.arr3d:
        if field in inputs and _wc01(ci, spec, field):
            groups.setdefault((field in dict(ci.cc.arr2d), spec.numsmo), []).append(field)
    for (is2d, nb), fl in groups.items():
        xs = jnp.stack([inputs[f]["xx"] for f in fl], axis=1)
        if is2d:
            fn = lambda x: smooth_mod.smooth_correl2d(ci.sp, ci.scfg.op2d(nb), ci.ops["2d"][nb], g, ex, x,  # noqa
                                                      g.maskC)
        else:
            fn = lambda x: smooth_mod.smooth_correl3d(ci.sp, ci.scfg.op3d(nb), ci.ops["3d"][nb], ci.ops["h"],  # noqa
                                                      g, ex, x, ci.recip_hFacC)
        ys = jax.vmap(fn, in_axes=1, out_axes=1)(xs)
        for i, f in enumerate(fl):
            inputs[f]["xx"] = ys[:, i]
    return dataclasses.replace(ci, inputs=inputs, smoothed=True)


_SJIT = {}


def ctrl_smooth_jit(ex):
    """jax.jit(ctrl_smooth) for this exchanger (cached): f(ci, g)."""
    if id(ex) not in _SJIT:
        _SJIT[id(ex)] = (ex, jax.jit(lambda ci, g: ctrl_smooth(ci, g, ex)))
    return _SJIT[id(ex)][1]


# ---------------------------------------------------------------------------------------------------------------------
# jittable: CTRL_BOUND_2D/3D, CTRL_MAP_GENARR2D/3D, CTRL_MAP_INI_GENARR
# ---------------------------------------------------------------------------------------------------------------------
def ctrl_bound_3d(fld, mask, bounds, L):
    """CTRL_BOUND_3D (ctrl_bound.F:17-70): interior, where mask .NE. 0: > bounds(4) -> bounds(4), then
    < bounds(1) -> bounds(1); only if bounds(1) < bounds(4)."""
    if not bounds[0] < bounds[3]:
        return fld
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    f, m = fld[..., J, I], mask[..., J, I] != 0.0
    f = jnp.where(m & (f > bounds[3]), bounds[3], f)
    f = jnp.where(m & (f < bounds[0]), bounds[0], f)
    return fld.at[..., J, I].set(f)


def ctrl_bound_2d(fld, mask3d, bounds):
    """CTRL_BOUND_2D (ctrl_bound.F:77-128): the FULL range 1-OLy..sNy+OLy, 1-OLx..sNx+OLx, mask level 1."""
    if not bounds[0] < bounds[3]:
        return fld
    m = mask3d[:, 0] != 0.0
    fld = jnp.where(m & (fld > bounds[3]), bounds[3], fld)
    return jnp.where(m & (fld < bounds[0]), bounds[0], fld)


def ctrl_map_genarr2d(ci: CtrlInit, spec: GenArr, field, g, ex, fld):
    """ff CTRL_MAP_GENARR2D (ctrl_map_ini_genarr.F:163-311) on fld [T,ny,nx]."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    inp = ci.inputs[field]
    xx, w = inp["xx"], inp["w"]                                     # :222-230 xx_gen = 0, :265 read (interior)
    if _wc01(ci, spec, field):                                      # :271-276 (unless done in ctrl_smooth)
        xx = smooth_mod.smooth_correl2d(ci.sp, ci.scfg.op2d(spec.numsmo), ci.ops["2d"][spec.numsmo], g, ex, xx,
                                        g.maskC)
    x = xx[:, J, I]
    if spec.doscaling:                                              # :283-291
        ok = (g.maskC[:, 0, J, I] != 0.0) & (w[:, J, I] > 0.0)
        x = jnp.where(ok, x / jnp.sqrt(jnp.where(ok, w[:, J, I], 1.0)), 0.0)
    fld = fld.at[:, J, I].set(fld[:, J, I] + x)                     # :293
    fld = ctrl_bound_2d(fld, g.maskC, spec.bounds)                  # :300
    return ex.exch_xy(fld)                                          # :302 EXCH_XY_RL


def ctrl_map_genarr3d(ci: CtrlInit, spec: GenArr, field, g, ex, fld):
    """ff CTRL_MAP_GENARR3D (ctrl_map_ini_genarr.F:317-495) on fld [T,Nr,ny,nx]; no exchange for uVel/vVel."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    inp = ci.inputs[field]
    xx, w = inp["xx"], inp["w"]                                     # :377-387, :422
    if _wc01(ci, spec, field):                                      # :428-433 (unless done in ctrl_smooth)
        xx = smooth_mod.smooth_correl3d(ci.sp, ci.scfg.op3d(spec.numsmo), ci.ops["3d"][spec.numsmo], ci.ops["h"],
                                        g, ex, xx, ci.recip_hFacC)
    mask = getattr(g, MASKS.get(field, "maskC"))                    # :437-444 localmask
    x = xx[..., J, I]
    if spec.doscaling:                                              # :453-465
        ok = (mask[..., J, I] != 0.0) & (w[..., J, I] > 0.0)
        x = jnp.where(ok, x / jnp.sqrt(jnp.where(ok, w[..., J, I], 1.0)), 0.0)
    fld = fld.at[..., J, I].set(fld[..., J, I] + x)                 # :467
    fld = ctrl_bound_3d(fld, mask, spec.bounds, L)                  # :476-477
    if field not in ("uVel", "vVel"):                               # :485-487
        fld = ex.exch_xy(fld)                                       # EXCH_XYZ_RL
    return fld


def ctrl_map_ini_genarr(ci: CtrlInit, g, ex, fields):
    """ff CTRL_MAP_INI_GENARR (ctrl_map_ini_genarr.F:12-157) for the controls present in `fields` and in
    ci.inputs, in the Fortran order. fields: {name: array}; returns a new dict with the adjusted arrays."""
    out = dict(fields)
    for name, spec in ci.cc.arr2d:                                  # :65-96
        if name in out and name in ci.inputs:
            out[name] = ctrl_map_genarr2d(ci, spec, name, g, ex, out[name])
    arr3 = dict(ci.cc.arr3d)
    for name in ("theta", "salt", "kapGM", "kapRedi", "diffKr"):   # :130-145
        if name in arr3 and name in out and name in ci.inputs:
            out[name] = ctrl_map_genarr3d(ci, arr3[name], name, g, ex, out[name])
    if "uVel" in arr3 and "uVel" in out and "uVel" in ci.inputs:   # :146-151
        u = ctrl_map_genarr3d(ci, arr3["uVel"], "uVel", g, ex, out["uVel"])
        v = ctrl_map_genarr3d(ci, arr3["vVel"], "vVel", g, ex, out["vVel"])
        out["uVel"], out["vVel"] = ex.exch_uv_xy(u, v, True)       # EXCH_UV_XYZ_RL(uvel,vvel,.TRUE.)
    return out


_JIT = {}


def ctrl_map_ini_genarr_jit(ex):
    """jax.jit(ctrl_map_ini_genarr) for this exchanger (cached): f(ci, g, fields)."""
    if id(ex) not in _JIT:
        _JIT[id(ex)] = (ex, jax.jit(lambda ci, g, fields: ctrl_map_ini_genarr(ci, g, ex, fields)))
    return _JIT[id(ex)][1]


# ---------------------------------------------------------------------------------------------------------------------
# time-varying controls (xx_gentim2d): zero check + CTRL_MAP_FORCING
# ---------------------------------------------------------------------------------------------------------------------
_ZERO_CHECKED = {}


def require_zero_forcing_controls(nml):
    """Hard error unless every xx_gentim2d control (weight set, ctrl_map_ini_gentim2d.F:103) is identically zero in
    every record of xx_<name>.<optimcycle> and its bounds keep 0 (CTRL_BOUND_2D, :376-377). With zero controls
    CTRL_MAP_INI_GENTIM2D / CTRL_GET_GEN produce xx_gentim2d = 0 (scaling, WC01 smoothing, docycle/rmcycle and the
    linear time interpolation all map 0 to 0) and every forcing add is x + 0."""
    if not bool(nml.get("data.pkg", "packages", "useCTRL", default=False)):
        return
    cc = CtrlConfig.from_namelists(nml)
    for iarr, name, weight, bounds in cc.tim2d:
        b1, b4 = bounds[0], bounds[3]
        if b1 < b4 and not (b1 <= 0.0 <= b4):
            raise NotImplementedError(f"xx_gentim2d_bounds({iarr}) = {bounds} moves a zero control "
                                      "(ctrl_bound.F:104-125): non-zero forcing controls are not ported")
        path = _mds_path(nml.dir, f"{name}.{cc.optimcycle:010d}")
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in _ZERO_CHECKED:
            a = np.memmap(path, dtype=">f4", mode="r")
            nz = False
            step = 1 << 26
            for s in range(0, a.size, step):
                if np.any(a[s:s + step] != 0):
                    nz = True
                    break
            del a
            _ZERO_CHECKED[key] = not nz
        if not _ZERO_CHECKED[key]:
            raise NotImplementedError(f"{path.name} has non-zero values: time-varying controls (CTRL_GET_GEN "
                                      "interpolation, ctrl_map_forcing.F / exf_getffields.F:681) are not ported")


# CTRL_MAP_FORCING targets (ctrl_map_forcing.F:340-358): prefix of xx_gentim2d_file(iarr), FFIELDS array. SST, SSS
# (FFIELDS relaxation fields) are not carried by the JAX model (no relaxation in V4r4): a control on them is refused.
FORCING_TARGETS = (("xx_qnet", "Qnet"), ("xx_empmr", "EmPmR"), ("xx_qsw", "Qsw"), ("xx_sst", "SST"),
                   ("xx_sss", "SSS"), ("xx_pload", "pLoad"), ("xx_saltflux", "saltFlux"), ("xx_fu", "fu"),
                   ("xx_fv", "fv"))


def ctrl_map_forcing(cc: CtrlConfig, g, ex, ff):
    """CTRL_MAP_FORCING (ctrl_map_forcing.F:298-376, ctrlUseGen) at forward_step.F:524-530, for xx_gentim2d = 0
    (require_zero_forcing_controls). ff: FFIELDS dict (fu, fv, Qnet, EmPmR, Qsw, pLoad, saltFlux, ...); returns the
    new dict. The adds are x + 0 (-0 becomes +0, as in the Fortran); tmpUX/tmpVY (ROTATE_UV2EN_RL of xx_fe/xx_fn,
    :314-332) are zero rotations: fu/fv + tmpUX/tmpVY is value-identical and not carried out (the sign of a zero fu
    can differ). The exchanges (:365-372) are the forward effect."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    out = dict(ff)
    for iarr, name, weight, bounds in cc.tim2d:                     # :334-363 interior, DO iarr inside DO i
        if name.startswith("xx_fe") or name.startswith("xx_fn"):
            raise NotImplementedError(f"{name}: rotated (E/N) forcing controls not ported (ROTATE_UV2EN_RL)")
        for prefix, field in FORCING_TARGETS:
            if name.startswith(prefix):
                if field not in out:
                    raise NotImplementedError(f"{name}: control on {field}, which the model does not carry")
                out[field] = out[field].at[:, J, I].add(jnp.zeros_like(out[field][:, J, I]))
    for n in ("Qnet", "EmPmR", "Qsw"):                              # :365-367 EXCH_XY_RS
        out[n] = ex.exch_xy(out[n])
    # :368-369 EXCH_XY_RS(SST), EXCH_XY_RS(SSS): arrays not carried (no surface relaxation in V4r4)
    for n in ("pLoad", "saltFlux"):                                 # :370-371
        out[n] = ex.exch_xy(out[n])
    out["fu"], out["fv"] = ex.exch_uv_xy(out["fu"], out["fv"], True)   # :372 EXCH_UV_XY_RS(fu, fv, .TRUE.)
    return out
