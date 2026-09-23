"""EXTERNAL_FORCING_SURF as called at the start of DO_OCEANIC_PHYS (plan Task 9; full tree M2.6a).

Literal port of c66g `model/src/external_forcing_surf.F` (not overridden) with the calls that precede it in
`DO_OCEANIC_PHYS`, over the whole array (iMin = 1-OLx ... jMax = sNy+OLy, ff/do_oceanic_phys.F:594-597, c66g
:588-591), including SALT_PLUME_FORCING_SURF (pkg/salt_plume/salt_plume_forcing_surf.F, SALT_PLUME_VOLUME undef).
The two V4r4 trees differ before EXTERNAL_FORCING_SURF:
  flux-forced (ff/do_oceanic_phys.F, READIN_SALT_PLUME_FLUX, useSEAICE = F):
      :288-300  saltPlumeDepth = 0 (ALLOW_AUTODIFF); saltPlumeFlux kept (read from spflx by EXF_MAPFIELDS)
      :581      SALT_PLUME_DO_EXCH           :612  EXTERNAL_FORCING_SURF
  full (c66g model/src/do_oceanic_phys.F, useSEAICE = T):
      :286-298  saltPlumeDepth = 0 and saltPlumeFlux = 0 (ALLOW_AUTODIFF, ALLOW_SALT_PLUME)
      :476      SEAICE_MODEL (sets Qnet, Qsw, EmPmR, saltFlux, fu, fv under ice, sIceLoad, saltPlumeFlux: pkgs/seaice_*)
      :575      SALT_PLUME_DO_EXCH           :606  EXTERNAL_FORCING_SURF
`oceanic_phys_pre_seaice` / `oceanic_phys_post_seaice` are the two halves (the full tree calls SEAICE_MODEL between
them); `oceanic_phys_forcing` is their composition for the flux-forced tree (no SEAICE_MODEL). useSEAICE enters
EXTERNAL_FORCING_SURF itself only in the balanceEmPmR / balanceQnet tests (external_forcing_surf.F:87, 91), both off.
The tree is recognised by READIN_SALT_PLUME_FLUX (`readin_salt_plume_flux`).

Branches taken with the V4r4 namelists (anything else raises NotImplementedError in `SurfForcingParams`):
balanceEmPmR = balanceQnet = F (no REMOVE_MEAN), no surface relaxation (doThetaClimRelax = doSaltClimRelax = F since
climsst/sssTauRelax = 0), EXACT_CONSERV with staggerTimeStep (PmEpR = -EmPmR), nonlinFreeSurf > 0 with
useRealFreshWaterFlux (temp_EvPrRn / salt_EvPrRn terms with PmEpR), ATMOSPHERIC_LOADING in z coordinates with
useRealFreshWaterFlux (phi0surf = (pLoad + sIceLoad*g)/rhoConst), SHORTWAVE_HEATING (Qsw subtracted from Qnet).
"""

from dataclasses import dataclass

import jax.numpy as jnp

from mitgcm_jax.params_io import params_pytree

UNSET_RL = 1.234567e5  # EEPARAMS.h:90


@params_pytree
@dataclass(frozen=True)
class SurfForcingParams:
    """`float` fields are pytree leaves (pass the object as a jit argument); bools are static branch switches."""
    HeatCapacity_Cp: float
    rhoConst: float
    recip_rhoConst: float
    mass2rUnit: float
    gravity: float
    temp_EvPrRn: float
    salt_EvPrRn: float
    useSALT_PLUME: bool
    temp_EvPrRn_set: bool     # temp_EvPrRn .NE. UNSET_RL (external_forcing_surf.F:257)
    salt_EvPrRn_set: bool     # salt_EvPrRn .NE. UNSET_RL (external_forcing_surf.F:268)
    useSEAICE: bool = False   # full tree: SEAICE_MODEL between oceanic_phys_pre_seaice and oceanic_phys_post_seaice
    zero_salt_plume_flux: bool = False   # c66g do_oceanic_phys.F:293 (not READIN_SALT_PLUME_FLUX, ff :293-295)

    @classmethod
    def from_namelists(cls, nml):
        g = lambda key, default: nml.get("data", "parm01", key, default=default)  # noqa: E731
        pkg = lambda key, default: nml.get("data.pkg", "packages", key, default=default)  # noqa: E731
        if g("buoyancyRelation", "OCEANIC") != "OCEANIC":                                # set_defaults.F:175
            raise NotImplementedError("only usingZCoords (buoyancyRelation='OCEANIC', ini_parms.F:420) is ported")
        if g("balanceEmPmR", False) or g("balanceQnet", False):                          # set_defaults.F:263-264
            raise NotImplementedError("balanceEmPmR/balanceQnet: REMOVE_MEAN_RS (external_forcing_surf.F:87-94)")
        # doThetaClimRelax/doSaltClimRelax (set_parms.F:213-216) need tau*ClimRelax > 0, which EXF sets from
        # climsst/sssTauRelax (ff/exf_readparms.F:1086,1100); data may not set them (it stops, :1077-1098).
        for key in ("climsstTauRelax", "climsssTauRelax"):
            if nml.get("data.exf", "exf_nml_02", key, default=0.0) != 0.0:
                raise NotImplementedError("surface relaxation: FORCING_SURF_RELAX (external_forcing_surf.F:150)")
        for key in ("tauThetaClimRelax", "tauSaltClimRelax"):
            if nml.get("data", "parm03", key, default=0.0) != 0.0:
                raise NotImplementedError("surface relaxation: FORCING_SURF_RELAX (external_forcing_surf.F:150)")
        if not g("staggerTimeStep", False):                                              # set_defaults.F:181
            raise NotImplementedError("staggerTimeStep=F: PmEpR update (external_forcing_surf.F:124) not ported")
        nonlin = g("nonlinFreeSurf", 0)                                                  # set_defaults.F:252
        if not (nonlin > 0 and g("useRealFreshWaterFlux", False)):     # set_defaults.F:258; external_forcing_surf.F:250
            raise NotImplementedError("only nonlinFreeSurf > 0 with useRealFreshWaterFlux is ported "
                                      "(external_forcing_surf.F:250-343)")
        for key in ("usePTRACERS", "useSHELFICE", "useThSIce", "useCoupler", "useFRAZIL", "useICEFRONT", "useOBCS"):
            if pkg(key, False):
                raise NotImplementedError(f"{key}=T: its surface-forcing path is not ported")
        # useSEAICE (full tree): SEAICE_MODEL modifies the FFIELDS before EXTERNAL_FORCING_SURF (pkgs/seaice_*);
        # inside EXTERNAL_FORCING_SURF it appears only with balanceEmPmR / balanceQnet (:87, :91; both refused above)
        useSEAICE = bool(pkg("useSEAICE", False))
        readin = readin_salt_plume_flux(nml)
        if useSEAICE and readin:
            raise NotImplementedError("useSEAICE with READIN_SALT_PLUME_FLUX (spflxfile): not a V4r4 build")
        if g("allowFreezing", False):                              # set_defaults.F:215; ff/do_oceanic_phys.F:586
            raise NotImplementedError("allowFreezing=T: FREEZE_SURFACE not ported")
        rhoNil = g("rhoNil", 999.8)                                                      # set_defaults.F:106
        rhoConst = float(g("rhoConst", rhoNil))                                          # ini_parms.F:445
        recip_rhoConst = 1.0 / rhoConst                                                  # ini_parms.F:640
        temp_EvPrRn = float(g("temp_EvPrRn", UNSET_RL))                                  # set_defaults.F:259
        salt_EvPrRn = float(g("salt_EvPrRn", 0.0))                                       # set_defaults.F:260
        return cls(HeatCapacity_Cp=float(g("HeatCapacity_Cp", 3994.0)),                 # set_defaults.F:173
                   rhoConst=rhoConst, recip_rhoConst=recip_rhoConst,
                   mass2rUnit=recip_rhoConst,                                            # ini_parms.F:1439
                   gravity=float(g("gravity", 9.81)),                                    # set_defaults.F:104
                   temp_EvPrRn=temp_EvPrRn, salt_EvPrRn=salt_EvPrRn,
                   useSALT_PLUME=bool(pkg("useSALT_PLUME", False)),
                   temp_EvPrRn_set=temp_EvPrRn != UNSET_RL, salt_EvPrRn_set=salt_EvPrRn != UNSET_RL,
                   useSEAICE=useSEAICE, zero_salt_plume_flux=not readin)


def readin_salt_plume_flux(nml):
    """READIN_SALT_PLUME_FLUX (ff EXF_OPTIONS.h:189): defined only in the flux-forced build, whose EXF_NML_02 has the
    extra key spflxfile (ff exf_readparms.F:120) and whose data.exf sets it; the full build has no such namelist
    entry (its EXF_NML_02 read would stop on it). The key's presence in the run's data.exf is therefore the
    run-directory signature of the build option; it is used where the option changes code: DO_OCEANIC_PHYS keeps
    saltPlumeFlux (ff/do_oceanic_phys.F:293-295) instead of zeroing it (c66g do_oceanic_phys.F:293)."""
    return nml.has("data.exf", "exf_nml_02", "spflxfile")


def salt_plume_do_exch(p, ex, saltPlumeFlux):
    """salt_plume_do_exch.F:69-71 (no coupler)."""
    if p.useSALT_PLUME:
        return ex.exch_xy(saltPlumeFlux)
    return saltPlumeFlux


def external_forcing_surf(p, g, fu, fv, Qnet, Qsw, EmPmR, saltFlux, saltPlumeFlux, pLoad, sIceLoad, theta, salt):
    """external_forcing_surf.F:72-396 with iMin..jMax = the whole array (ff/do_oceanic_phys.F:594-597).

    theta, salt: [T, Nr, ny, nx] (only level ks = 1 is read). Returns a dict with surfaceForcingU/V/T/S, PmEpR,
    phi0surf (every point written)."""
    ks = 0                                                                               # external_forcing_surf.F:75
    recip_Cp = 1.0 / p.HeatCapacity_Cp                                                   # external_forcing_surf.F:77
    m2r = p.mass2rUnit
    # EXACT_CONSERV, staggerTimeStep (external_forcing_surf.F:122-132): whole array
    PmEpR = -EmPmR
    # external_forcing_surf.F:136-145
    sFT = jnp.zeros_like(Qnet)
    sFS = jnp.zeros_like(Qnet)
    # Surface fluxes (external_forcing_surf.F:206-225)
    sFU = fu * m2r                                                                       # :210
    sFV = fv * m2r                                                                       # :212
    sFT = sFT - (Qnet - Qsw) * recip_Cp * m2r                                            # :214-219 (SHORTWAVE_HEATING)
    sFS = sFS - saltFlux * m2r                                                           # :221-222
    # SALT_PLUME_FORCING_SURF (external_forcing_surf.F:235-239; salt_plume_forcing_surf.F:66-67)
    if p.useSALT_PLUME:
        sFS = sFS - saltPlumeFlux * m2r
    # Fresh-water flux: nonlinFreeSurf > 0 and useRealFreshWaterFlux (external_forcing_surf.F:250-277)
    if p.temp_EvPrRn_set:                                                                # :257-266
        sFT = sFT + PmEpR * (p.temp_EvPrRn - theta[:, ks]) * m2r
    if p.salt_EvPrRn_set:                                                                # :268-277
        sFS = sFS + PmEpR * (p.salt_EvPrRn - salt[:, ks]) * m2r
    # ATMOSPHERIC_LOADING, usingZCoords, useRealFreshWaterFlux (external_forcing_surf.F:356-366)
    phi0surf = (pLoad + sIceLoad * p.gravity) * p.recip_rhoConst
    return dict(surfaceForcingU=sFU, surfaceForcingV=sFV, surfaceForcingT=sFT, surfaceForcingS=sFS, PmEpR=PmEpR,
                phi0surf=phi0surf)


def oceanic_phys_forcing(p, g, ex, ff, saltPlumeDepth, theta, salt):
    """DO_OCEANIC_PHYS up to and including EXTERNAL_FORCING_SURF of the flux-forced tree (ff/do_oceanic_phys.F:288-614):
    saltPlumeDepth = 0 (:292), SALT_PLUME_DO_EXCH (:581), EXTERNAL_FORCING_SURF (:612).
    ff: FFIELDS dict (fu, fv, Qnet, Qsw, EmPmR, saltFlux, saltPlumeFlux, pLoad, sIceLoad). Returns (ff, out, depth).
    The full tree (useSEAICE) calls SEAICE_MODEL in between: use oceanic_phys_pre_seaice / oceanic_phys_post_seaice."""
    if p.useSEAICE or p.zero_salt_plume_flux:
        raise ValueError("full-tree DO_OCEANIC_PHYS: oceanic_phys_pre_seaice, SEAICE_MODEL, oceanic_phys_post_seaice")
    saltPlumeDepth = jnp.zeros_like(saltPlumeDepth)                                      # ff/do_oceanic_phys.F:292
    ff = dict(ff)
    ff["saltPlumeFlux"] = salt_plume_do_exch(p, ex, ff["saltPlumeFlux"])                # ff/do_oceanic_phys.F:581
    out = external_forcing_surf(p, g, ff["fu"], ff["fv"], ff["Qnet"], ff["Qsw"], ff["EmPmR"], ff["saltFlux"],
                                ff["saltPlumeFlux"], ff["pLoad"], ff["sIceLoad"], theta, salt)
    return ff, out, saltPlumeDepth


def oceanic_phys_pre_seaice(p, ff, saltPlumeDepth):
    """DO_OCEANIC_PHYS before SEAICE_MODEL: the ALLOW_AUTODIFF zeroing (c66g do_oceanic_phys.F:286-298, full range
    1-OLx..sNx+OLx): saltPlumeDepth = 0 (:292) and, unless READIN_SALT_PLUME_FLUX (ff :293-295), saltPlumeFlux = 0
    (:293). Returns (ff, saltPlumeDepth)."""
    saltPlumeDepth = jnp.zeros_like(saltPlumeDepth)                                      # :292
    ff = dict(ff)
    if p.zero_salt_plume_flux:
        ff["saltPlumeFlux"] = jnp.zeros_like(ff["saltPlumeFlux"])                        # :293
    return ff, saltPlumeDepth


def oceanic_phys_post_seaice(p, g, ex, ff, theta, salt):
    """DO_OCEANIC_PHYS after SEAICE_MODEL up to and including EXTERNAL_FORCING_SURF: SALT_PLUME_DO_EXCH (c66g
    do_oceanic_phys.F:573-576, ff :579-582), EXTERNAL_FORCING_SURF (c66g :606-608, ff :612-614; allowFreezing = F,
    no SHELFICE / ICEFRONT / FRAZIL). ff: FFIELDS dict after SEAICE_MODEL. Returns (ff, out)."""
    ff = dict(ff)
    ff["saltPlumeFlux"] = salt_plume_do_exch(p, ex, ff["saltPlumeFlux"])
    out = external_forcing_surf(p, g, ff["fu"], ff["fv"], ff["Qnet"], ff["Qsw"], ff["EmPmR"], ff["saltFlux"],
                                ff["saltPlumeFlux"], ff["pLoad"], ff["sIceLoad"], theta, salt)
    return ff, out
