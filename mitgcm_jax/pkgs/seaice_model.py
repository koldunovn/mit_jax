"""pkg/seaice: SEAICE_MODEL, the sea-ice time step of the full ECCO v4r4 tree (plan M2.6b-1), as one pure function.

Literal port of c66g `pkg/seaice/seaice_model.F` (no V4r4 override; the V4r4 `code/` tree overrides only
seaice_growth.F among the routines it calls). Active branches, read from the preprocessed build
(reference/build/full_serial13_jaxdump_*/bld/seaice_model.f) and V4r4 code/SEAICE_OPTIONS.h, CPP_OPTIONS.h:
ALLOW_EXF, ALLOW_AUTODIFF_TAMC, SEAICE_CGRID, SHORTWAVE_HEATING, ATMOSPHERIC_LOADING defined; ALLOW_THSICE,
ALLOW_OBCS, SEAICE_ITD, SEAICE_VARIABLE_SALINITY, ALLOW_SITRACER, DISABLE_SEAICE_GROWTH undefined. Call sequence:
    :74-77    EXCH_UV_AGRID_3D_RL(uwind, vwind, .TRUE., 1)   (EXF does not update the edges; changes the EXF state)
    :86-101   uIceNm1 = vIceNm1 = 0 (ALLOW_AUTODIFF_TAMC): SEAICE_LSR sets both on every point before reading them
              (seaice_lsr.F:178-195, :270-283), so they carry nothing across steps (pkgs/seaice_lsr.py)
    :146-149  SEAICE_DYNSOLVER incl. SEAICE_OCEAN_STRESS (fu, fv) and the velocity clipping      -> stage I01
    :179-185  SEAICE_ADVDIFF(uIce, vIce) (SEAICEadvHeff/Area/Snow = T)                           -> stage I02
    :194      SEAICE_REG_RIDGE                                                                   -> stage I03
    :200-206  SEAICE_GROWTH (usePW79thermodynamics = T; V4r4 override)                           -> stage I04
    :224-226  _EXCH_XY_RL HEFF, AREA, HSNOW; :240-242 _EXCH_XY_RS EmPmR, saltFlux, Qnet; :244 Qsw (SHORTWAVE_HEATING);
              :247-248 sIceLoad (ATMOSPHERIC_LOADING, useRealFreshWaterFlux = T)                  -> stage P00
Not exchanged by SEAICE_MODEL: TICES, UICE, VICE (UICE/VICE are exchanged inside SEAICE_LSR), fu, fv (exchanged in
SEAICE_OCEAN_STRESS, :292-293 comment), saltPlumeFlux (SALT_PLUME_DO_EXCH, called by DO_OCEANIC_PHYS after
SEAICE_MODEL: core/external_forcing.oceanic_phys_post_seaice).
Not ported (no effect on the model state): DIAGNOSTICS_FILL (:272-286), EXF_ADJOINT_SNAPSHOTS (:295-301, output),
TIMER/DEBUG calls, the CADJ STORE directives; SEAICE_COST_SENSI after the call (do_oceanic_phys.F, cost only).

State (what the model must carry for sea ice; M2.6b-2 puts these into State):
  ICE_STATE   AREA, HEFF, HSNOW, TICES [T, nITD=7, ny, nx], UICE, VICE        (pkgs/seaice_init.py initialises them)
  DYN_CARRY   seaiceMassC/U/V, FORCEX0, FORCEY0, e11, e22, e12, DWATN, FORCEX, FORCEY: arrays SEAICE_DYNSOLVER writes
              only partly, so the unwritten (halo) points carry over from the previous step (initial values:
              `dyn_carry_init`, seaice_init_varia.F). Everything else SEAICE_DYNSOLVER writes (TAUX, TAUY, PRESS0,
              ZMAX, ZMIN, ETA, ZETA, etaZ, zetaZ, PRESS, deltaC, uIceNm1, vIceNm1, uice_fd, vice_fd,
              stressDivergenceX/Y) is set on every point before it is read and need not be carried.
  plus the ocean FFIELDS fu, fv, Qnet, Qsw, EmPmR, saltFlux, saltPlumeFlux, sIceLoad (FF_INOUT) and the EXF_FIELDS
  arrays (pkgs/exf_full.EXF_ARRAYS, 28): SEAICE_MODEL rewrites uwind, vwind (EXF_INOUT) and reads EXF_READ.
Fixed fields `sg` (pkgs/seaice_init.seaice_fixed_fields): HEFFM, k1AtC, k1AtZ, k2AtC, k2AtZ, seaiceMaskU/V,
tensileStrFac.

Adjoint levels (`ad`, static; derivatives only: the forward is byte-identical in all three, tested):
  "ecco"        (default) what TAF computes for V4r4: data.autodiff useSEAICEinAdMode = .FALSE. ->
                autodiff_inadmode_set_ad.F:37 useSEAICE = .FALSE. in the reverse sweep, so the IF (useSEAICE) block
                around SEAICE_MODEL (c66g do_oceanic_phys.F:397-481) is skipped: the adjoint of SEAICE_MODEL is the identity
                on every variable it overwrites (INOUT: ICE_STATE, DYN_CARRY, FF_INOUT, EXF_INOUT) and contributes
                nothing to the variables it only reads (READ: EXF_READ, OCN_READ). Not a stop_gradient.
                (SEAICEadjMODE = SEAICEapproxLevInAd = MIN(0, 0) = 0 in the reverse sweep, autodiff_readparms.F:
                124-125, so do_oceanic_phys.F:381 SEAICE_FAKE (needs -1) is not called either.)
  "no_dynamics" SEAICEuseDYNAMICSswitchInAd = .TRUE. semantics (autodiff_inadmode_set_ad.F:49-51, useSEAICEinAdMode =
                .TRUE.): only the IF (SEAICEuseDYNAMICS) blocks of SEAICE_DYNSOLVER are skipped in reverse
                (seaice_dynsolver.F:270-341 FREEDRIFT + LSR, :365-385 clipping; pkgs/seaice_dyn.dynsolver
                ad_dynamics="skip"); ice masses, dynamic forcing, ice strength, SEAICE_OCEAN_STRESS, SEAICE_ADVDIFF,
                SEAICE_REG_RIDGE and SEAICE_GROWTH keep their exact derivatives (none of them is guarded by
                SEAICEuseDYNAMICS; ADVDIFF is guarded by SEAICEadv*, seaice_model.F:179). SEAICEadjMODE = 0 as in the
                forward (MAX(0, 0), autodiff_readparms.F:126-127): the V4r4 growth / reg_ridge SEAICEadjMODE
                branches (seaice_growth.F:591, 1002, 1364, 1383, 1544, 1559, 1922: >= 1; seaice_reg_ridge.F:122:
                .EQ. 0) take the forward branch.
  "full"        the exact derivative of everything (the LSR by its implicit custom_jvp, pkgs/seaice_lsr.py).
Mechanism: ops/ad_skip.skipped_in_reverse (a linear custom_jvp: identity tangents on the overwritten variables).
`ad_level(nml)` gives the level of a run's data.autodiff.

Gates: mitgcm_jax/tests/test_seaice_model.py (oracle full_jaxdump_v5, iterations 1-3).
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from mitgcm_jax.layout import Layout
from mitgcm_jax.ops.ad_skip import skipped_in_reverse
from mitgcm_jax.params_io import params_pytree
from mitgcm_jax.pkgs import seaice_advdiff as sa
from mitgcm_jax.pkgs import seaice_dyn as sd
from mitgcm_jax.pkgs import seaice_growth as sgr
from mitgcm_jax.pkgs import seaice_init as si

ICE_STATE = si.ICE_STATE  # AREA, HEFF, HSNOW, TICES, UICE, VICE
DYN_CARRY = ("seaiceMassC", "seaiceMassU", "seaiceMassV", "FORCEX0", "FORCEY0", "e11", "e22", "e12", "DWATN",
             "FORCEX", "FORCEY")
FF_INOUT = ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "saltPlumeFlux", "sIceLoad")
EXF_INOUT = ("uwind", "vwind")
EXF_READ = ("wspeed", "atemp", "aqh", "lwdown", "swdown", "evap", "precip", "snowprecip", "runoff")
OCN_READ = ("uVel_s", "vVel_s", "theta_s", "salt_s")  # surface level (k = 1) of uVel, vVel, theta, salt
INOUT = ICE_STATE + DYN_CARRY + FF_INOUT + EXF_INOUT
READ = EXF_READ + OCN_READ
INPUTS = INOUT + READ
SEAICE_CARRIED = ICE_STATE + DYN_CARRY
AD_LEVELS = ("ecco", "no_dynamics", "full")
# the growth kernel's names of the EXF fields it reads (pkgs/seaice_growth.seaice_growth `exf`)
_GROWTH_EXF = (("wspeed", "wspeed"), ("atemp", "atemp"), ("aqh", "aqh"), ("lwdown", "lwdown"), ("swdown", "swdown"),
               ("evap", "evap"), ("precip", "precip"), ("snowPrecip", "snowprecip"), ("runoff", "runoff"))


# the grid fields the four kernels read (pkgs/seaice_dyn.py, seaice_lsr.py, seaice_advdiff.GRID2D, seaice_growth.py)
SEAICE_GRID_2D = ("yC", "fCori", "dxF", "dyF", "dxG", "dyG", "dxV", "dyU", "recip_dxF", "recip_dyF", "recip_dxC",
                  "recip_dyC", "recip_dxV", "recip_dyU", "recip_rA", "recip_rAw", "recip_rAs", "maskInC", "maskInW",
                  "maskInS")
SEAICE_GRID_MASKS = ("maskC", "maskW", "maskS")  # read at k = 1 only
SEAICE_GRID_1D = ("drF",)


def seaice_grid(g):
    """The part of Grid `g` the sea-ice kernels read: SEAICE_GRID_2D, the 3-D masks cut to the surface level (kept as
    [T, 1, ny, nx], the kernels index [:, 0]) and drF (plus the tile index under sharding). Same values, ~50x smaller:
    what a host-CPU copy for SEAICE_MODEL needs (M2.6b-2)."""
    from mitgcm_jax.parallel.tiles import TILE_INDEX

    f = {k: g.f[k] for k in SEAICE_GRID_2D + SEAICE_GRID_1D}
    f.update({k: g.f[k][:, :1] for k in SEAICE_GRID_MASKS})
    if TILE_INDEX in g.f:
        f[TILE_INDEX] = g.f[TILE_INDEX]
    return type(g)(f, g.layout)


def dyn_carry_init(layout=None):
    """Values of DYN_CARRY after SEAICE_INIT_VARIA (pickup start: SEAICE_READ_PICKUP reads none of them):
    seaiceMassC/U/V = 1000 (seaice_init_varia.F:430-432, after the zeroing :282-284), FORCEX0/Y0 = 0 (:307-308),
    DWATN = 0 (:302), FORCEX/Y = 0 (:279-280); e11, e22, e12 are never initialised: SEAICE.h:86 common block, 0 at
    program start. Full tile, [T, ny, nx]."""
    L = layout or Layout()
    z = jnp.zeros(L.shape2d)
    out = {k: z for k in DYN_CARRY}
    for k in ("seaiceMassC", "seaiceMassU", "seaiceMassV"):
        out[k] = jnp.full(L.shape2d, 1000.0)
    return out


@params_pytree
@dataclass(frozen=True)
class SeaiceModelFlags:
    """SEAICE_MODEL's own switches (static)."""
    advdiff: bool       # seaice_model.F:179-180 SEAICEadvHeff .OR. SEAICEadvArea .OR. SEAICEadvSnow .OR. SEAICEadvSalt
    exch_sIceLoad: bool  # seaice_model.F:246-249 ATMOSPHERIC_LOADING (code/CPP_OPTIONS.h) .AND. useRealFreshWaterFlux

    @classmethod
    def from_namelists(cls, nml, adv):
        pkg = lambda k: bool(nml.get("data.pkg", "packages", k, default=False))  # noqa: E731  packages_boot.F
        if not pkg("useSEAICE"):
            raise ValueError("useSEAICE = F: SEAICE_MODEL is not called (do_oceanic_phys.F:397)")
        for k, where in (("useThSIce", "seaice_model.F:79-84, 163-172"), ("useOBCS", "seaice_model.F:157-160, "
                                                                                     "217-221")):
            if pkg(k):
                raise NotImplementedError(f"{k} = T ({where}) not ported")
        s = ("data.seaice", "SEAICE_PARM01")
        if not bool(nml.get(*s, "usePW79thermodynamics", default=True)):          # seaice_readparms.F:234
            raise NotImplementedError("usePW79thermodynamics = F: SEAICE_GROWTH not called (seaice_model.F:200)")
        rfwf = bool(nml.get("data", "parm01", "useRealFreshWaterFlux", default=False))   # set_defaults.F
        return cls(advdiff=adv.called, exch_sIceLoad=rfwf)


class SeaiceParams(NamedTuple):
    """Parameters of the four kernels and of SEAICE_MODEL itself (a pytree: pass it as a jit argument; float fields
    of each dataclass are traced leaves)."""
    dyn: sd.SeaiceDynParams
    adv: sa.SeaiceAdvDiffParams
    ridge: sa.SeaiceRegRidgeParams
    growth: sgr.SeaiceGrowthParams
    flags: SeaiceModelFlags

    @classmethod
    def from_namelists(cls, nml, g, layout=None):
        """From the run's namelists (data, data.seaice, data.exf, data.salt_plume, data.pkg, eedata, data.exch2);
        g: Grid (vertical grid for SWFracB). Unported options raise in the kernels' from_namelists."""
        L = layout or g.layout
        adv = sa.SeaiceAdvDiffParams.from_namelists(nml, L)
        return cls(dyn=sd.SeaiceDynParams.from_namelists(nml),
                   adv=adv,
                   ridge=sa.SeaiceRegRidgeParams.from_namelists(nml, L),
                   growth=sgr.SeaiceGrowthParams.from_namelists(nml, g),
                   flags=SeaiceModelFlags.from_namelists(nml, adv))


def ad_level(nml):
    """The adjoint level TAF uses for this run (data.autodiff, AUTODIFF_PARM01): useSEAICEinAdMode (default .TRUE.,
    autodiff_readparms.F:68) = F -> "ecco"; else SEAICEuseDYNAMICSswitchInAd (default .FALSE., :77) = T ->
    "no_dynamics", F -> "full". SEAICEapproxLevInAd != 0 (default 0, :72) or SEAICEuseFREEDRIFTswitchInAd = T (:76)
    are not ported."""
    a = ("data.autodiff", "AUTODIFF_PARM01")
    if int(nml.get(*a, "SEAICEapproxLevInAd", default=0)) != 0:
        raise NotImplementedError("SEAICEapproxLevInAd != 0 (SEAICEadjMODE branches of seaice_growth.F) not ported")
    if bool(nml.get(*a, "SEAICEuseFREEDRIFTswitchInAd", default=False)):
        raise NotImplementedError("SEAICEuseFREEDRIFTswitchInAd = T not ported")
    if not bool(nml.get(*a, "useSEAICEinAdMode", default=True)):
        return "ecco"
    return "no_dynamics" if bool(nml.get(*a, "SEAICEuseDYNAMICSswitchInAd", default=False)) else "full"


# ---------------------------------------------------------------------------------------------------------------------
# the exchanges of SEAICE_MODEL (module-level functions: the negative controls of the gate replace them)
# ---------------------------------------------------------------------------------------------------------------------
def exch_winds(ex, uwind, vwind):
    """seaice_model.F:76 EXCH_UV_AGRID_3D_RL(uwind, vwind, .TRUE., 1, myThid)."""
    return ex.exch_uv_agrid(uwind, vwind, True)


def exchanges_after_growth(flags, ex, d):
    """seaice_model.F:224-248: _EXCH_XY_RL HEFF, AREA, HSNOW; _EXCH_XY_RS EmPmR, saltFlux, Qnet, Qsw
    (SHORTWAVE_HEATING, code/CPP_OPTIONS.h:23), sIceLoad (IF useRealFreshWaterFlux, ATMOSPHERIC_LOADING). _RS = _RL =
    real*8 in this build; the exchanges are independent (order immaterial)."""
    d = dict(d)
    for k in ("HEFF", "AREA", "HSNOW", "EmPmR", "saltFlux", "Qnet", "Qsw"):
        d[k] = ex.exch_xy(d[k])
    if flags.exch_sIceLoad:
        d["sIceLoad"] = ex.exch_xy(d["sIceLoad"])
    return d


# ---------------------------------------------------------------------------------------------------------------------
# SEAICE_MODEL
# ---------------------------------------------------------------------------------------------------------------------
def _seaice_model(expf, record, ad_dynamics, ins, rest):
    """The literal call sequence (module docstring). ins: INPUTS; rest = (P, g, sg, ex). Returns (out, rec)."""
    P, g, sg, ex = rest
    ins = {k: jnp.asarray(ins[k]) for k in INPUTS}
    rec = {}
    # :74-77
    uwind, vwind = exch_winds(ex, ins["uwind"], ins["vwind"])
    if record:
        rec["I00"] = dict(uwind=uwind, vwind=vwind)
    # :86-101 uIceNm1 = vIceNm1 = 0: see the module docstring (set inside SEAICE_LSR before use)
    # :146-149 SEAICE_DYNSOLVER
    dst = dict(HEFF=ins["HEFF"], AREA=ins["AREA"], uIce=ins["UICE"], vIce=ins["VICE"], uVel=ins["uVel_s"],
               vVel=ins["vVel_s"], fu=ins["fu"], fv=ins["fv"], **{k: ins[k] for k in DYN_CARRY})
    dyn, drec = sd.dynsolver(P.dyn, g, sg, ex, dst, record=record, ad_dynamics=ad_dynamics)
    if record:
        rec["I01"] = dyn
        rec["dyn"] = drec
    # :179-185 SEAICE_ADVDIFF(uIce, vIce)
    if P.flags.advdiff:
        adv, _ = sa.seaice_advdiff(P.adv, g, dyn["UICE"], dyn["VICE"], ins["HEFF"], ins["AREA"], ins["HSNOW"],
                                   sg["HEFFM"])
    else:
        adv = dict(HEFF=ins["HEFF"], AREA=ins["AREA"], HSNOW=ins["HSNOW"])
    if record:
        rec["I02"] = dict(adv, TICES=ins["TICES"], UICE=dyn["UICE"], VICE=dyn["VICE"])
    # :194 SEAICE_REG_RIDGE
    rr = sa.seaice_reg_ridge(P.ridge, adv["HEFF"], adv["AREA"], adv["HSNOW"], ins["TICES"])
    if record:
        rec["I03"] = rr
    # :200-206 SEAICE_GROWTH (usePW79thermodynamics = T, SeaiceModelFlags)
    ice = {k: rr[k] for k in ("AREA", "HEFF", "HSNOW", "TICES", "d_HEFFbyNEG", "d_HSNWbyNEG")}
    ocn = dict(theta_s=ins["theta_s"], salt_s=ins["salt_s"])
    exf = {kg: ins[ke] for kg, ke in _GROWTH_EXF}
    flx = {k: ins[k] for k in ("Qnet", "Qsw", "EmPmR", "saltFlux", "saltPlumeFlux", "sIceLoad")}
    gi, gf, _ = sgr.seaice_growth(P.growth, g, sg["HEFFM"], ice, ocn, exf, flx, expf=expf)
    if record:
        rec["I04"] = dict(gi, **gf, fu=dyn["fu"], fv=dyn["fv"], d_HEFFbyNEG=rr["d_HEFFbyNEG"],
                          d_HSNWbyNEG=rr["d_HSNWbyNEG"])
    # :223-249
    d = exchanges_after_growth(P.flags, ex, dict(gi, **gf))
    out = dict(AREA=d["AREA"], HEFF=d["HEFF"], HSNOW=d["HSNOW"], TICES=d["TICES"], UICE=dyn["UICE"], VICE=dyn["VICE"],
               fu=dyn["fu"], fv=dyn["fv"], Qnet=d["Qnet"], Qsw=d["Qsw"], EmPmR=d["EmPmR"], saltFlux=d["saltFlux"],
               saltPlumeFlux=d["saltPlumeFlux"], sIceLoad=d["sIceLoad"], uwind=uwind, vwind=vwind,
               **{k: dyn[k] for k in DYN_CARRY})
    if record:
        rec["P00"] = dict(out)
    return out, rec


_seaice_model_skipped = skipped_in_reverse(_seaice_model, INOUT, n_static=3)


def seaice_model(P, g, sg, ex, ins, ad="ecco", expf=None, record=False):
    """SEAICE_MODEL (seaice_model.F:13-308) for one step, all tiles; jittable (P, g, sg, ex, ins as arguments; `ad`,
    `expf`, `record` static).

    P: SeaiceParams; g: Grid; sg: fixed fields (seaice_init.seaice_fixed_fields); ex: Exchanger.
    ins: dict of [T, ny, nx] arrays (TICES [T, nITD, ny, nx]) with every key of INPUTS, the values on entry to
      SEAICE_MODEL: ICE_STATE, DYN_CARRY (previous step's outputs; `dyn_carry_init` at the first step), FF_INOUT (fu,
      fv, Qnet, Qsw, EmPmR, saltFlux from EXF_MAPFIELDS / CTRL_MAP_FORCING; saltPlumeFlux after the zeroing of
      core/external_forcing.oceanic_phys_pre_seaice; sIceLoad of the previous step), EXF_INOUT (uwind, vwind as
      EXF_GETFORCING left them), EXF_READ (the EXF_FIELDS arrays after EXF_GETFORCING), OCN_READ (uVel, vVel,
      theta, salt at k = 1: the state at the start of the step).
    ad: adjoint level "ecco" | "no_dynamics" | "full" (module docstring). expf: exp of SEAICE_SOLVE4TEMP (default:
      glibc's, bitwise; jnp.exp for speed). record: also return the dump-stage values.
    Returns (out, rec): out = the INOUT fields after SEAICE_MODEL (stage P00 for the dumped ones); rec (record=True):
    I00 (uwind, vwind), I01 (dynsolver), dyn (its Y*/L* records), I02, I03, I04, P00."""
    if ad not in AD_LEVELS:
        raise ValueError(f"ad={ad!r}: one of {AD_LEVELS}")
    missing = sorted(set(INPUTS) - set(ins))
    if missing:
        raise KeyError(f"seaice_model: inputs missing: {missing}")
    rest = (P, g, sg, ex)
    ins = {k: ins[k] for k in INPUTS}
    if ad == "ecco":
        return _seaice_model_skipped(expf, record, "exact", ins, rest)
    return _seaice_model(expf, record, "skip" if ad == "no_dynamics" else "exact", ins, rest)
