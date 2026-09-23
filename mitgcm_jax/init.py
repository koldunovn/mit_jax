"""Model initialisation from the run directory (plan Task 8): INITIALISE_VARIA of the V4r4 flux-forced build, from a
pickup (nIter0 > 0), literally, for all tiles.

    st = state_from_pickup(P, g, ex, kLowC, rundir)      # State at the start of iteration nIter0
    info = pickup_info(rundir)                           # READ_PICKUP / CHECK_PICKUP: mom_StartAB, missing fields

`st` holds every field of `state.S00_FIELDS` + `state.G00_STATE_FIELDS` (+ `runoff`), i.e. what the Fortran holds at
the S00_begin dump of the first FORWARD_STEP (forward_step.F:392) and in the r* fields of G00_geometry group R.
Gates: tests/test_init.py (tier1: FORCED, all 88 fields bitwise incl. halos), tests/test_init_step.py (tier1x: SMOKE
bitwise; one FORWARD_STEP from this state == Fortran S00_begin of iteration 2, bitwise).
P, g, ex, kLowC come from `model.setup(rundir)`. P.dyn.ts.mom_StartAB must be the value CHECK_PICKUP derives from the
pickup (`pickup_info(rundir).mom_StartAB`); state_from_pickup raises if it is not.

Call sequence ported (flux-forced code/initialise_varia.F overrides c66g; every routine below is c66g model/src or
pkg/<pkg> unless marked ff = ECCOv4 Release 4/flux-forced/code):
  ff initialise_varia.F:141  INI_NLFS_VARS      ini_nlfs_vars.F:42-103 (etaHnm1, dEtaHdt, PmEpR = 0; hFac_surf* = 0;
                                                rStarFac*, rStarFacNm1*, rStarExp*, pStarFacK = 1; rStarDh*Dt = 0;
                                                hFac = h0Fac). Rmin_surf (:113-145) is only read by CALC_SURF_DR
                                                (select_rStar = 0): not carried.
  :147  INI_DYNVARS        ini_dynvars.F:8-58 (all DYNVARS 3-D/2-D arrays = 0)
  :155  INI_FFIELDS        ini_ffields.F:8-95 (FFIELDS forcing arrays, surfaceForcing* = 0)
  :164  INI_FIELDS         ini_fields.F:30-41 -> READ_PICKUP(nIter0) read_pickup.F:86-116, 259-436 (new way: field
                           list in the .meta), READ_MFLDS_* (pkg/rw/read_mflds.F), CHECK_PICKUP check_pickup.F:60-234,
                           exchanges read_pickup.F:509-535
  :166-186 CALC_PHI_RLOW_INI  CALC_PHI_HYD(k=1..Nr, full tile, myIter=-1): FIND_RHO_2D on theta/salt
                           (calc_phi_hyd.F:156-170), finite-difference integration (:267-282), DIAGS_PHI_RLOW
                           (diags_phi_rlow.F:67-171), DIAGS_PHI_HYD (diags_phi_hyd.F:56-111) -> totPhiHyd, phiHydLow.
                           CALC_GRAD_PHI_HYD writes only locals of INITIALISE_VARIA (dPhiHydX/Y): not computed.
  :192  INI_MIXING         diffKr / kapGM / kapRedi: grid fields (grid/load.py)
  :203  INI_FORCING        ini_forcing.F:47-128 (lambda*ClimRelax = 0 with tau = 0; no forcing file in V4r4 except
                           geothermalFile, which model.setup reads into the grid; exchanges of the zero arrays)
  :207  AUTODIFF_INIT_VARIA  autodiff_init_varia.F: only TsurfCor/SsurfCor (ALLOW_AUTODIFF_INIT_OLD undefined)
  :219  PACKAGES_INIT_VARIABLES  GGL90_INIT_VARIA ggl90_init_varia.F:48-67 + GGL90_READ_PICKUP
                           ggl90_read_pickup.F:61-66; GMREDI_INIT_VARIA gmredi_init_varia.F:44-66; ff EXF_INIT_VARIA
                           (pkgs/exf_fluxforced.exf_init_varia); SALT_PLUME_INIT_VARIA salt_plume_init_varia.F:45-52;
                           GAD_INIT_VARIA empty (GAD_ALLOW_TS_SOM_ADV undefined); SMOOTH_INIT_VARIA empty;
                           useCTRL: CTRL_INIT_VARIABLES (packages_init_variables.F:496-503) -> ff CTRL_MAP_INI_GENARR
                           on etaN, theta, salt, uVel, vVel (pkgs/ctrl.py; kapGM/kapRedi/diffKr: model.setup)
  :236-251 CONVECTIVE_ADJUSTMENT_INI  compiled out (ALLOW_AUTODIFF_WHTAPEIO defined, AUTODIFF_OPTIONS.h:49)
  :259  CALC_R_STAR(etaH, myIter=-1)   core/free_surface.calc_r_star
  :264  UPDATE_R_STAR(.TRUE.)          core/free_surface.update_r_star, with recip_hFac* of INI_MASKS_ETC
                                       (ini_masks_etc.F:478-506)
  :277  UPDATE_CG2D                    core/solve_for_pressure.update_cg2d, on the preconditioner of INI_CG2D
                                       (ini_cg2d.F:5-199, called from initialise_fixed.F:249: its pW/pS/pC halos
                                       beyond sNx+1/sNy+1 are never rewritten)
  :285  INTEGR_CONTINUITY(uVel, vVel, nIter0)  integr_continuity.F:20-82 (myIter.EQ.nIter0 branch), 160-230,
                                       INTEGRATE_FOR_W (integrate_for_w.F:5-88, r* branch), UPDATE_ETAH (update_etah.F)
  :292  CALC_R_STAR(etaH, nIter0)
the_main_loop.F:363-406 AUTODIFF_STORE / AUTODIFF_RESTORE (ff): identity (every stored array is restored, checked
                                      by parsing both files; AUTODIFF_USE_OLDSTORE_2D/3D skip the DYNVARS copies)

Full V4r4 tree (plan M2.6a; initialise_varia.F is the same file in both trees, packages_init_variables.F is c66g):
  - PACKAGES_INIT_VARIABLES also calls SEAICE_INIT_VARIA (packages_init_variables.F:336, after EXF_INIT_VARIA :249,
    before SALT_PLUME_INIT_VARIA :342): pkgs/seaice_init.py -- pickup_seaice (siTICE, siAREA, siHEFF, siHSNOW,
    siUICE, siVICE; TICES copied to all nITD categories), the exchanges, and sIceLoad = HEFF*rhoIce + HSNOW*rhoSnow
    (seaice_init_varia.F:685-692; the only ocean field SEAICE_INIT_VARIA sets). The static HEFFM / k1AtC ... fields
    are grid fields (model.setup). The State then also holds AREA, HEFF, HSNOW, TICES, UICE, VICE.
  - EXF: the full tree's EXF_INIT_VARIA (bulk-formula EXF, pkgs/exf_full.exf_init_varia, P.exfb; exf_init_varia.F:
    44-378 + exf_init_fld.F:88-97) sets the 28 EXF_FIELDS arrays the State carries (pkgs/exf_full.EXF_ARRAYS: every
    field = its fldConst, wStress ... hl, uwind, vwind = 0; evap and the zenith fields keep the 0 of the common block).
  - SEAICE_INIT_VARIA also sets the arrays SEAICE_DYNSOLVER writes only partly (pkgs/seaice_model.DYN_CARRY:
    seaiceMassC/U/V = 1000, seaice_init_varia.F:430-432; FORCEX0/Y0, DWATN, FORCEX/Y = 0, :279-308; e11, e22, e12 = 0,
    common block); the State carries them (pkgs/seaice_model.dyn_carry_init; core/forward_step.py docstring).
  - AUTODIFF_STORE / AUTODIFF_RESTORE (c66g pkg/autodiff/autodiff_store.F / autodiff_restore.F in the full tree)
    restore every array they store (EXF records, sIceLoad, AREA, HEFF, HSNOW, UICE, VICE, ...): identity (gcov).
  - useCTRL: the full data.ctrl has the same genarr controls as ff (xx_etan, xx_theta, xx_salt, xx_kapgm,
    xx_kapredi, xx_diffkr, xx_uvel, xx_vvel; ctrl_map_ini_genarr.F is the same override file) and no gentim2d.
Not ported (hard error at set-up): useECCO, useProfiles, nIter0 = 0 / pickupSuff / old-format pickups, tracer AB
histories; ctrl branches other than ctrlUseGen=T generic controls with WC01 smoothing (pkgs/ctrl.py); sea-ice
branches listed in pkgs/seaice_init.SeaiceInitConfig.

CTRL (useCTRL=T with every mult_* = 0, as in production data.ctrl; plan Task 8b): mult_genarr2d/3d and mult_gentim2d
weight only the cost terms (ctrl_cost_gen.F); ff ctrl_map_ini_genarr.F:83-152 does not read them. With ctrlUseGen=T
it applies every genarr control whose weight file is set: fld = fld + smooth_correl3d(xx)/sqrt(weight) on wet points
(WC01 = 150 pseudo-time steps of pkg/smooth), CTRL_BOUND, EXCH, for xx_etan (2-D, etaN only: etaH is not adjusted),
xx_theta, xx_salt, xx_uvel/xx_vvel (here) and xx_kapgm, xx_kapredi, xx_diffkr (grid fields: model.setup). It runs
in PACKAGES_INIT_VARIABLES, i.e. after CALC_PHI_RLOW_INI (totPhiHyd/phiHydLow keep the pickup theta/salt) and before
the r* / INTEGR_CONTINUITY sequence (which sees the adjusted uVel, vVel; etaH = the adjusted etaN after UPDATE_ETAH).
Gate: tests/test_ctrl.py (oracle ref_ff_jaxdump_v4).
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitgcm_jax.core import eos as eos_mod
from mitgcm_jax.core import free_surface as fs
from mitgcm_jax.core import solve_for_pressure as sfp
from mitgcm_jax.core.implicit import vertical_factors as grav_factors
from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.io.mds import read_mds
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import ctrl as ctrl_mod
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.pkgs import exf_full as exfb_mod
from mitgcm_jax.pkgs import seaice_init as si_mod
from mitgcm_jax.pkgs import seaice_model as sm_mod
from mitgcm_jax.state import G00_STATE_FIELDS, S00_FIELDS, State

PRECFLOAT64 = 64   # EEPARAMS.h precFloat64; read_pickup.F:104 fp = precFloat64, ggl90_read_pickup.F:59 prec
PREC_META = {64: "float64", 32: "float32"}   # .meta dataprec of each precision (mdsio_write_meta.F)
# pkg/generic_advdiff/GAD.h:28,32,36: advection schemes for which GAD_INIT_FIXED keeps Adams-Bashforth on tracers
AB_TRACER_SCHEMES = (2, 3, 4)   # ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH
# EXF_FIELDS.h arrays the step carries (core/forward_step.EXF_STATE)
EXF_FIELDS = ("ustress", "vstress", "hflux", "sflux", "swflux", "apressure", "saltflx", "spflx", "runoff")


# ---------------------------------------------------------------------------------------------------------------------
# configuration (static switches of the initialisation, from the run's namelists)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class InitConfig:
    nIter0: int
    pickup: str              # read_pickup.F:86-96 'pickup.' // I10.10(nIter0)
    pickup_ggl90: str        # ggl90_read_pickup.F:57 'pickup_ggl90.' // suff
    pickupStrictlyMatch: bool
    readGuNm1: bool          # read_pickup.F:282-287 momStepping .AND. (alph_AB.NE.0 .OR. beta_AB.NE.0)
    readGuNm2: bool          # read_pickup.F:288-291 momStepping .AND. beta_AB.NE.0
    m1: int                  # read_pickup.F:280 m1 = 1 + MOD(myIter+1, 2)
    m2: int                  # read_pickup.F:281 m2 = 1 + MOD(myIter, 2)
    useEXF: bool
    useGGL90: bool
    useGMRedi: bool
    useSALT_PLUME: bool
    useCTRL: bool = False    # packages_boot.F; CTRL_INIT_VARIABLES (packages_init_variables.F:496-503)
    useSEAICE: bool = False  # packages_boot.F; SEAICE_INIT_VARIA (packages_init_variables.F:336, pkgs/seaice_init.py)

    @classmethod
    def from_namelists(cls, nml):
        p1 = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
        p3 = lambda k, d: nml.get("data", "parm03", k, default=d)  # noqa: E731
        p4 = lambda k, d: nml.get("data", "parm04", k, default=d)  # noqa: E731
        p5 = lambda k, d: str(nml.get("data", "parm05", k, default=d)).strip()  # noqa: E731
        pkg = lambda k: bool(nml.get("data.pkg", "packages", k, default=False))  # noqa: E731  packages_boot.F
        nIter0 = int(p3("nIter0", 0))                         # ini_parms.F:962-974 (startTime unset in V4r4)
        checks = [
            (nIter0 > 0, "nIter0=0: INI_VEL/INI_THETA/INI_SALT/INI_PSURF path (ini_fields.F:30-35)"),
            (not nml.has("data", "parm03", "startTime"), "startTime set (ini_parms.F:966-974: nIter0 from startTime)"),
            (str(p3("pickupSuff", " ")).strip() == "", "pickupSuff (set_defaults.F:330 ' ')"),
            (int(p3("rwSuffixType", 0)) == 0, "rwSuffixType != 0 (set_defaults.F:354; read_pickup.F:88-92)"),
            (bool(p3("pickup_read_mdsio", True)), "pickup_read_mdsio=F (set_defaults.F:335)"),
            (not pkg("useMNC"), "useMNC (pkg/mnc not compiled in V4r4)"),
            (not pkg("useOffLine"), "useOffLine (ini_fields.F:36)"),
            (int(p4("selectSigmaCoord", 0)) == 0, "selectSigmaCoord (ini_fields.F:48)"),
            (not bool(p1("nonHydrostatic", False)), "nonHydrostatic (ALLOW_NONHYDROSTATIC undefined)"),
            (not bool(p3("usePickupBeforeC54", False)) and not bool(p1("usePickupBeforeC54", False)),
             "usePickupBeforeC54 (set_defaults.F:220; integr_continuity.F:66)"),
            (not bool(p3("startFromPickupAB2", False)), "startFromPickupAB2 (set_defaults.F:314)"),
            (bool(p1("momStepping", True)), "momStepping=F (set_defaults.F:189)"),
            (bool(p1("exactConserv", False)), "exactConserv=F (read_pickup.F:425; integr_continuity.F:10)"),
            (int(p1("nonlinFreeSurf", 0)) > 2 and int(p1("select_rStar", 0)) > 0,
             "nonlinFreeSurf <= 2 or select_rStar <= 0 (ff initialise_varia.F:254-298: r* + UPDATE_CG2D)"),
            (bool(p1("useRealFreshWaterFlux", False)), "useRealFreshWaterFlux=F (integr_continuity.F:83-91)"),
            (str(p1("buoyancyRelation", "OCEANIC")).strip().upper() == "OCEANIC", "buoyancyRelation"),
            (float(p3("tauThetaClimRelax", 0.0)) == 0.0 and float(p3("tauSaltClimRelax", 0.0)) == 0.0,
             "tauTheta/SaltClimRelax > 0 (ini_forcing.F:47-60, lambda*ClimRelax not carried)"),
            (int(p1("tempAdvScheme", 2)) not in AB_TRACER_SCHEMES and int(p1("saltAdvScheme", 2)) not in
             AB_TRACER_SCHEMES, "Adams-Bashforth on T/S (gad_init_fixed.F:147-165: GtNm/GsNm pickup records)"),
            (not pkg("useECCO"), "useECCO=T: ECCO_INIT_VARIA (ff ecco_init_varia.F) is not ported"),
            (not pkg("useProfiles"), "useProfiles=T: PROFILES_INIT_VARIA is not ported"),
        ]
        for key in ("zonalWindFile", "meridWindFile", "surfQFile", "surfQnetFile", "EmPmRfile", "saltFluxFile",
                    "thetaClimFile", "saltClimFile", "lambdaThetaFile", "lambdaSaltFile", "surfQswFile",
                    "pLoadFile", "addMassFile"):
            checks.append((p5(key, " ") == "", f"{key} (ini_forcing.F:68-137; set_defaults.F:372-386 ' ')"))
        bad = [msg for ok, msg in checks if not ok]
        if bad:
            raise NotImplementedError("initialisation branch not ported: " + "; ".join(bad))
        alph_AB = float(p3("alph_AB", 0.5))        # set_defaults.F:312
        beta_AB = float(p3("beta_AB", 5.0 / 12.0))  # set_defaults.F:313
        momStepping = bool(p1("momStepping", True))
        suff = f"{nIter0:010d}"                     # read_pickup.F:89 WRITE(suff,'(I10.10)') myIter
        return cls(nIter0=nIter0, pickup="pickup." + suff, pickup_ggl90="pickup_ggl90." + suff,
                   pickupStrictlyMatch=bool(p3("pickupStrictlyMatch", True)),   # set_defaults.F:331
                   readGuNm1=momStepping and (alph_AB != 0.0 or beta_AB != 0.0),
                   readGuNm2=momStepping and beta_AB != 0.0,
                   m1=1 + (nIter0 + 1) % 2, m2=1 + nIter0 % 2,
                   useEXF=pkg("useEXF"), useGGL90=pkg("useGGL90"), useGMRedi=pkg("useGMRedi"),
                   useSALT_PLUME=pkg("useSALT_PLUME"), useCTRL=pkg("useCTRL"), useSEAICE=pkg("useSEAICE"))


# ---------------------------------------------------------------------------------------------------------------------
# READ_PICKUP / CHECK_PICKUP / GGL90_READ_PICKUP (host side: file reads)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PickupInfo:
    nbFields: int
    fldList: tuple
    missing: tuple           # READ_MFLDS_CHECK list (fields attempted but not in the file)
    mom_StartAB: int         # RESTART.h, set by ini_model_io.F:120-127 then check_pickup.F:60-196
    tempStartAB: int
    saltStartAB: int


def _to_tiles(a, L):
    """compact global (..., 1170, 90) -> [T, ..., sNy, sNx] (tile axis first)."""
    t = compact_to_tiles(a)                   # (..., T, 90, 90)
    return np.moveaxis(t, -3, 0)


def read_pickup(rundir, cfg: InitConfig, L):
    """READ_PICKUP(nIter0) new way (read_pickup.F:102-436) + CHECK_PICKUP. Returns (fields, info): fields maps the
    State names of every array READ_MFLDS_3D_RL fills to its interior values [T, (Nr,) sNy, sNx] (READ_MFLDS ->
    MDS_READ_FIELD writes i=1..sNx, j=1..sNy only); a field missing from the file is absent (it keeps 0)."""
    prefix = Path(rundir) / cfg.pickup
    if not Path(str(prefix) + ".meta").exists():
        # read_mflds.F READ_MFLDS_SET: no meta -> nbFields = -1 -> old-format read (read_pickup.F:164-257)
        raise NotImplementedError(f"{prefix}: pickup without .meta field list (old format) is not ported")
    arr, meta = read_mds(prefix)
    Nr = L.Nr
    fldList = tuple(meta.get("fldList", ()))
    nbFields = len(fldList)                                  # read_mflds.F READ_MFLDS_SET: nbFields = nFlds
    if nbFields >= 0 and meta["dataprec"] != PREC_META[PRECFLOAT64]:   # read_pickup.F:114-122
        raise ValueError(f"READ_PICKUP: pickup-file binary precision {meta['dataprec']} != float64")
    if nbFields <= 0:                                        # read_pickup.F:127-160, 165-257
        raise NotImplementedError("pickup meta without field list (read_pickup.F:165 old way) is not ported")
    nRecords, nDims = meta["nrecords"], meta["nDims"]
    # read_mflds.F READ_MFLDS_SET: nFl3D = (nRecords - nFlds)/(thirdDim - 1) for 2-D records
    if not (nDims == 2 and nbFields < nRecords and (nRecords - nbFields) % (Nr - 1) == 0):
        raise ValueError(f"READ_MFLDS_SET: nRecords={nRecords} does not match nFlds={nbFields} (3rd dim {Nr})")
    nFl3D = (nRecords - nbFields) // (Nr - 1)
    missing = []

    def read(name, nNz):
        """READ_MFLDS_3D_RL(name, nNz): record nj = position in fldList (+ nFl3D*(Nr-1) past the 3-D fields)."""
        if name not in fldList:
            missing.append(name)                              # read_mflds.F: nMissFld = nMissFld + 1
            return None
        nj = fldList.index(name) + 1
        if nj > nFl3D:
            nj = nj + nFl3D * (Nr - 1)
        rec = arr[(nj - 1) * nNz: nj * nNz]                   # MDS_READ_FIELD record nj of nNz levels
        return _to_tiles(rec if nNz > 1 else rec[0], L)

    out = {}
    # read_pickup.F:262-277 state 3-D fields (GM_InMomAsStress: ALLOW_EDDYPSI undefined)
    for name, key in (("Uvel", "uVel"), ("Vvel", "vVel"), ("Theta", "theta"), ("Salt", "salt")):
        out[key] = read(name, Nr)
    # read_pickup.F:280-301 AB-3 momentum histories into slots m1, m2
    if cfg.readGuNm1:
        out[f"guNm_{cfg.m1}"] = read("GuNm1", Nr)
    if cfg.readGuNm2:
        out[f"guNm_{cfg.m2}"] = read("GuNm2", Nr)
    if cfg.readGuNm1:
        out[f"gvNm_{cfg.m1}"] = read("GvNm1", Nr)
    if cfg.readGuNm2:
        out[f"gvNm_{cfg.m2}"] = read("GvNm2", Nr)
    # read_pickup.F:303-341 tracer AB histories: AdamsBashforthGt/Gs/_T/_S all .FALSE. (checked in InitConfig)
    # read_pickup.F:389-392 storePhiHyd4Phys = F (set_parms.F:260, selectP_inEOS_Zc = 0); :399-412 not compiled
    # read_pickup.F:415-432 2-D fields
    out["etaN"] = read("EtaN", 1)
    out["dEtaHdt"] = read("dEtaHdt", 1)                       # exactConserv (InitConfig check)
    out["etaH"] = read("EtaH", 1)                             # nonlinFreeSurf > 0
    out = {k: v for k, v in out.items() if v is not None}
    info = check_pickup(cfg, nbFields, fldList, tuple(missing))
    return out, info


def check_pickup(cfg: InitConfig, nbFields, fldList, missing):
    """CHECK_PICKUP (check_pickup.F:55-234) for the fields READ_PICKUP attempts in V4r4. ini_model_io.F:120-127 sets
    the start-AB levels to nIter0 first (startFromPickupAB2 = F)."""
    nIter0 = cfg.nIter0
    mom = temp = salt = nIter0                                # ini_model_io.F:120-127
    if nbFields >= 1:                                         # check_pickup.F:60-68
        mom = temp = salt = nIter0
    stop = []
    for f in missing:
        if f in ("Uvel", "Vvel", "Theta", "Salt", "EtaN"):    # check_pickup.F:153-161
            stop.append(f"cannot restart without field {f!r}")
        elif f in ("dEtaHdt", "EtaH"):                        # :164-171 (dEtaHdt: useRealFreshWaterFlux = T, :85-86)
            stop.append(f"cannot currently restart without field {f!r}")
        elif f in ("GuNm1", "GvNm1"):                         # :175-177
            mom = 0
        elif f in ("GuNm2", "GvNm2"):                         # :178-180
            mom = min(mom, 1)
        else:                                                 # :198-203
            stop.append(f"missing field {f!r} not recognized")
    if stop:                                                  # :207-208 STOP 'ABNORMAL END: S/R CHECK_PICKUP'
        raise ValueError("CHECK_PICKUP: " + "; ".join(stop))
    if missing and cfg.pickupStrictlyMatch:                   # :209-214
        raise ValueError(f"CHECK_PICKUP: missing {missing} with pickupStrictlyMatch=.TRUE.")
    return PickupInfo(nbFields=nbFields, fldList=fldList, missing=tuple(missing), mom_StartAB=mom,
                      tempStartAB=temp, saltStartAB=salt)


def pickup_info(rundir, layout=None):
    """mom_StartAB etc. of the run's pickup (build DynamicsParams / TimestepParams with `mom_StartAB=`)."""
    from mitgcm_jax.layout import Layout
    nml = RunNamelists(rundir)
    return read_pickup(rundir, InitConfig.from_namelists(nml), layout or Layout())[1]


def read_ggl90_pickup(rundir, cfg: InitConfig, L):
    """GGL90_READ_PICKUP (ggl90_read_pickup.F:44-66): READ_REC_3D_RL('pickup_ggl90.<suff>', float64, Nr, record 1)
    -> interior [T, Nr, sNy, sNx]. useIDEMIX = F (GGL90Params check): no second record."""
    arr, meta = read_mds(Path(rundir) / cfg.pickup_ggl90)
    if meta["dataprec"] != PREC_META[PRECFLOAT64]:
        raise ValueError(f"{cfg.pickup_ggl90}: precision {meta['dataprec']} != float64 (ggl90_read_pickup.F:59)")
    rec = arr[0]
    if rec.ndim == 2:                                          # 2-D records: Nr of them form record 1
        rec = arr[:L.Nr]
    return _to_tiles(rec, L)


# ---------------------------------------------------------------------------------------------------------------------
# jittable pieces
# ---------------------------------------------------------------------------------------------------------------------
def _ksum(init, terms):
    """init + terms[:, 0] + terms[:, 1] + ... (Fortran `DO k=1,Nr: a = a + t(k)`, sequential)."""
    out, _ = lax.scan(lambda a, t: (a + t, None), init, jnp.moveaxis(terms, 1, 0))
    return out


def ini_recip_hfac(h0Fac):
    """ini_masks_etc.F:482-501: recip_hFac = 1/hFac where hFac .NE. 0, else 0 (full range). Under r* this is the
    value UPDATE_R_STAR keeps at dry points."""
    wet = h0Fac != 0.0
    return jnp.where(wet, 1.0 / jnp.where(wet, h0Fac, 1.0), 0.0)


def ini_cg2d(fsp, cgp, g, ex):
    """INI_CG2D (ini_cg2d.F:5-199) at initialise_fixed.F:249 (hFacW/S = h0FacW/S there). Returns (pW, pS, pC, myNorm)
    after the exchanges of :198-199; aW2d/aS2d/aC2d are rebuilt from zero by UPDATE_CG2D (update_cg2d.F:14-24,
    ALLOW_AUTODIFF), only the preconditioner points UPDATE_CG2D does not rewrite keep these values."""
    L = g.layout
    z2 = jnp.zeros_like(g.rA)
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    drF = jnp.asarray(g.drF)[None, :, None, None]
    # :42-67 aW2d, aS2d = sum_k implicSurfPress*implicDiv2DFlow*faceArea*recip_dx(y)C, faceArea = (dyG*drF)*hFacW
    fac = fsp.implicSurfPress * fsp.implicDiv2DFlow
    tW = fac * (g.dyG[:, None, J, I] * drF * g.h0FacW[:, :, J, I]) * g.recip_dxC[:, None, J, I]
    tS = fac * (g.dxG[:, None, J, I] * drF * g.h0FacS[:, :, J, I]) * g.recip_dyC[:, None, J, I]
    aW = _ksum(z2[:, J, I], tW)
    aS = _ksum(z2[:, J, I], tS)
    # :68-87 myNorm = 1/max|aW, aS| (:76-77 per point, :82 _GLOBAL_MAX_RS: order-free); ALLOW_OBCS undefined
    myNorm = ex.global_max(jnp.maximum(jnp.abs(aW), jnp.abs(aS)))
    nz = myNorm != 0.0
    myNorm = jnp.where(nz, 1.0 / jnp.where(nz, myNorm, 1.0), 1.0)
    aW2d = z2.at[:, J, I].set(aW * myNorm)                                     # :88-97
    aS2d = z2.at[:, J, I].set(aS * myNorm)
    aW2d, aS2d = ex.exch_uv_xy(aW2d, aS2d, False)                              # :104 EXCH_UV_XY_RS(.FALSE.)
    # :157-167 main diagonal on j=0..sNy+1, i=0..sNx+1
    J0, I0 = L.js(0, L.sNy + 1), L.is_(0, L.sNx + 1)
    Ip, Jp = L.is_(1, L.sNx + 2), L.js(1, L.sNy + 2)
    ks = fs.k_surf(g.maskC)[:, J0, I0]
    deepFac2F = jnp.asarray(fs.vertical_factors(L.Nr)["deepFac2F"])
    aC = -(aW2d[:, J0, I0] + aW2d[:, J0, Ip] + aS2d[:, J0, I0] + aS2d[:, Jp, I0]
           + fsp.freeSurfFac * myNorm * g.recip_Bo[:, J0, I0] * deepFac2F[ks - 1]
           * g.rA[:, J0, I0] / fsp.deltaTMom / fsp.deltaTFreeSurf)
    aC2d = z2.at[:, J0, I0].set(aC)
    # :168-194 preconditioner on the interior
    Im, Jm = L.is_(0, L.sNx - 1), L.js(0, L.sNy - 1)
    a, aCs, aCw = aC2d[:, J, I], aC2d[:, Jm, I], aC2d[:, J, Im]
    pC = jnp.where(a == 0.0, 1.0, 1.0 / jnp.where(a == 0.0, 1.0, a))           # :173-177
    zw = (a + aCw) == 0.0                                                      # :178-183
    pW = jnp.where(zw, 0.0, -aW2d[:, J, I] / (cgp.cg2dpcOffDFac * jnp.where(zw, 1.0, aCw + a)) ** 2)
    zs = (a + aCs) == 0.0                                                      # :184-189
    pS = jnp.where(zs, 0.0, -aS2d[:, J, I] / (cgp.cg2dpcOffDFac * jnp.where(zs, 1.0, aCs + a)) ** 2)
    pC = ex.exch_xy(z2.at[:, J, I].set(pC))                                    # :198 EXCH_XY_RS
    pW, pS = ex.exch_uv_xy(z2.at[:, J, I].set(pW), z2.at[:, J, I].set(pS), False)  # :199 EXCH_UV_XY_RS(.FALSE.)
    return pW, pS, pC, myNorm


def calc_phi_hyd_ini(pp, eos, g, kLowC, theta, salt, rStarFacC, phi0surf, totPhiHyd):
    """CALC_PHI_RLOW_INI (ff initialise_varia.F:166-186): CALC_PHI_HYD for k=1..Nr on the full tile
    (iMin=1-OLx .. iMax=sNx+OLx, jMin=1-OLy .. jMax=sNy+OLy, :113-116) with myIter=-1, so the density is
    FIND_RHO_2D(theta(k), salt(k), kRef=k) (calc_phi_hyd.F:156-170; find_rho.F zeroes rhoLoc first, then fills the
    same full range). Returns (totPhiHyd, phiHydLow): DIAGS_PHI_HYD and DIAGS_PHI_RLOW outputs (global arrays); the
    CALC_GRAD_PHI_HYD outputs are INITIALISE_VARIA locals and are dropped. At this point rStarFacC = 1 (INI_NLFS_VARS)
    and phi0surf = 0 (ini_linear_phisurf.F:200-209). Operation order as core/phi_hyd.calc_phi_hyd (gated bitwise)."""
    L = g.layout
    Nr = L.Nr
    gravity, recip_rhoConst = pp.gravity, pp.recip_rhoConst
    halfRL, zeroRL = 0.5, 0.0                                                     # EEPARAMS.h
    drC, rC, rF = jnp.asarray(g.drC), jnp.asarray(g.rC), jnp.asarray(g.rF)
    gravFacF = jnp.asarray(grav_factors(Nr)["gravFacF"])
    # calc_phi_hyd.F:267-273 (finite-difference form, integr_GeoPot = 2)
    dRlocM = halfRL * drC[:Nr] * gravFacF[:Nr]
    dRlocM = dRlocM.at[0].set((rF[0] - rC[0]) * gravFacF[0])
    dRlocP = halfRL * drC[1:Nr + 1] * gravFacF[1:Nr + 1]
    dRlocP = dRlocP.at[Nr - 1].set((rC[Nr - 1] - rF[Nr]) * gravFacF[Nr])
    # diags_phi_rlow.F:91-96
    ratioRm = jnp.ones(Nr).at[1:].set(halfRL * drC[1:Nr] / (rF[1:Nr] - rC[1:Nr]))
    ratioRp = jnp.ones(Nr).at[:Nr - 1].set(halfRL * drC[1:Nr] / (rC[:Nr - 1] - rF[1:Nr]))
    ratioRm = ratioRm * gravFacF[:Nr]
    ratioRp = ratioRp * gravFacF[1:Nr + 1]
    # calc_phi_hyd.F:165-170 FIND_RHO_2D(kRef = k) for every level, full tile
    alpha = eos_mod.find_rho_levels(eos, theta, salt, np.arange(1, Nr + 1))
    rlow_dd = rC[None, :, None, None] - g.R_low[:, None]                          # diags_phi_rlow.F:101

    def level(carry, x):
        phiHydF, phiHydLow = carry
        k, a, dM, dP, rM, rP, dd = x
        phiHydC = phiHydF + dM * gravity * a * recip_rhoConst                    # calc_phi_hyd.F:277-278
        phiHydF = phiHydC + dP * gravity * a * recip_rhoConst                    # :279-280
        low = phiHydC + (jnp.minimum(zeroRL, dd) * rM + jnp.maximum(zeroRL, dd) * rP) * gravity * a * recip_rhoConst
        phiHydLow = jnp.where(k == kLowC, low, phiHydLow)                        # diags_phi_rlow.F:100-105
        return (phiHydF, phiHydLow), phiHydC

    z = jnp.zeros_like(theta[:, 0])
    ks = jnp.arange(1, Nr + 1, dtype=jnp.int32)
    # phiHydF = 0 at k=1 (calc_phi_hyd.F:139-145); phiHydLow = 0 at k=1 (diags_phi_rlow.F:67-73), full range
    (_, low), pC = lax.scan(level, (z, z), (ks, jnp.moveaxis(alpha, 1, 0), dRlocM, dRlocP, ratioRm, ratioRp,
                                            jnp.moveaxis(rlow_dd, 1, 0)))
    phiHydC = jnp.moveaxis(pC, 0, 1)
    # diags_phi_rlow.F:162-171 (k = Nr, r*, ocean z-coordinates)
    dPhiRef = (g.Ro_surf - g.R_low) * gravity
    phiHydLow = low * rStarFacC + dPhiRef * (rStarFacC - 1.0) + phi0surf
    # diags_phi_hyd.F:101-111 (r*): overwrites the :56-58 value on the same range
    rSF = rStarFacC[:, None]
    dPhiRefk = (g.Ro_surf[:, None] - rC[None, :, None, None]) * gravity
    totPhiHyd = phiHydC * rSF + jnp.maximum(dPhiRefk, 0.0) * (rSF - 1.0) + phi0surf[:, None]
    return totPhiHyd, phiHydLow


def integr_continuity_ini(p, g, ex, uFld, vFld, hFacW, hFacS, etaN, etaH, dEtaHdt, wVel, PmEpR):
    """INTEGR_CONTINUITY(uVel, vVel, startTime, nIter0) of INITIALISE_VARIA (ff initialise_varia.F:285): the
    myIter.EQ.nIter0 branches (nIter0 != 0, fluidIsWater, useRealFreshWaterFlux, usePickupBeforeC54 = F).
    Returns dict with dEtaHdt (unchanged), PmEpR, etaN (unchanged), wVel, etaHnm1, etaH."""
    L = g.layout
    Nr = L.Nr
    vf = fs.vertical_factors(Nr)
    J1, I1 = L.js(1, L.sNy + 1), L.is_(1, L.sNx + 1)
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    dfc = jnp.asarray(vf["deepFacC"])[None, :, None, None]
    rfc = jnp.asarray(vf["rhoFacC"])[None, :, None, None]
    drF = jnp.asarray(g.drF)
    # integr_continuity.F:32-41 (= integrate_for_w.F:5-14), j,i = 1..sNy+1, 1..sNx+1
    uT = uFld[:, :, J1, I1] * g.dyG[:, None, J1, I1] * dfc * rfc * drF[None, :, None, None] * hFacW[:, :, J1, I1]
    vT = vFld[:, :, J1, I1] * g.dxG[:, None, J1, I1] * dfc * rfc * drF[None, :, None, None] * hFacS[:, :, J1, I1]
    div = uT[:, :, :-1, 1:] - uT[:, :, :-1, :-1] + vT[:, :, 1:, :-1] - vT[:, :, :-1, :-1]
    # :20-56 hDivFlow = 0; DO k: hDivFlow = hDivFlow + maskC*div (ALLOW_ADDFLUID undefined)
    hDivFlow = _ksum(jnp.zeros_like(div[:, 0]), g.maskC[:, :, J, I] * div)
    # :61-82 myIter.EQ.nIter0 .AND. myIter.NE.0 .AND. fluidIsWater .AND. useRealFreshWaterFlux
    PmEpR_i = dEtaHdt[:, J, I] + hDivFlow * g.recip_rA[:, J, I] * jnp.asarray(vf["recip_deepFac2F"])[0]  # :77-79
    PmEpR_i = PmEpR_i * p.rUnit2mass                                                                   # :80
    PmEpR = PmEpR.at[:, J, I].set(PmEpR_i)
    # :127-155 etaN not updated (myIter.EQ.nIter0)
    # :160-173 rStarDhDt
    ks = fs.k_surf(g.maskC)[:, J, I]
    rStarDhDt = (dEtaHdt[:, J, I] * jnp.asarray(vf["deepFac2F"])[ks - 1] * jnp.asarray(vf["rhoFacF"])[ks - 1]
                 * g.recip_Rcol[:, J, I])
    # :180-218 INTEGRATE_FOR_W, k = Nr..1 (integrate_for_w.F:64-88, r* branch); EXACT_CONSERV extra term only for
    # usingPCoords (:190-200)
    conv2d = -div                                                             # integrate_for_w.F:15-20
    rA_i = g.recip_rA[:, J, I]
    rdf, rrf = jnp.asarray(vf["recip_deepFac2F"]), jnp.asarray(vf["recip_rhoFacF"])
    df2, rfF = jnp.asarray(vf["deepFac2F"]), jnp.asarray(vf["rhoFacF"])
    h0 = g.h0FacC[:, :, J, I]
    mC = g.maskC[:, :, J, I]
    k = Nr - 1                                                                # integrate_for_w.F:67-76
    wNr = (conv2d[:, k] * rA_i - rStarDhDt * drF[k] * h0[:, k]) * mC[:, k] * rdf[k] * rrf[k]

    def up(wk1, x):                                                           # integrate_for_w.F:77-87
        cv, h0k, mk, drFk, rdfk, rrfk, df2k1, rfFk1 = x
        w = (wk1 * df2k1 * rfFk1 + cv * rA_i - rStarDhDt * drFk * h0k) * mk * rdfk * rrfk
        return w, w

    xs = (jnp.moveaxis(conv2d[:, :-1], 1, 0), jnp.moveaxis(h0[:, :-1], 1, 0), jnp.moveaxis(mC[:, :-1], 1, 0),
          drF[:Nr - 1], rdf[:Nr - 1], rrf[:Nr - 1], df2[1:Nr], rfF[1:Nr])
    _, w_up = lax.scan(up, wNr, xs, reverse=True)
    w_new = jnp.concatenate([jnp.moveaxis(w_up, 0, 1), wNr[:, None]], axis=1)
    wVel = wVel.at[:, :, J, I].set(w_new)
    # :226-228 no etaN exchange (myIter.EQ.nIter0); :229-230 myIter.EQ.nIter0: _EXCH_XYZ_RL(wVel)
    wVel = ex.exch_xy(wVel)
    # :234-241 UPDATE_ETAH (update_etah.F:9-33, implicDiv2Dflow = 1: full-range copy; no exchange :48-50)
    etaHnm1 = etaH
    etaH = etaN
    return dict(dEtaHdt=dEtaHdt, PmEpR=PmEpR, etaN=etaN, wVel=wVel, etaHnm1=etaHnm1, etaH=etaH)


def initialise_varia(P, g, ex, kLowC, pk, tke, exf, ctrl_in=None, ice=None):
    """INITIALISE_VARIA (ff initialise_varia.F:123-298 = full code/initialise_varia.F) for a pickup start. pk:
    interior arrays of READ_PICKUP (read_pickup); tke: interior of pickup_ggl90, None when useGGL90=F
    (GGL90_INIT_VARIA not called); exf: the EXF_FIELDS arrays of EXF_INIT_VARIA (flux-forced: EXF_FIELDS; full tree:
    pkgs/exf_full.EXF_ARRAYS; zeros when useEXF=F); ctrl_in: pkgs/ctrl.CtrlInit of the state controls (None when
    useCTRL=F); ice:
    (SeaiceInitConfig, pickup_seaice interiors) when useSEAICE (full tree), else None. Returns (fields, aux): fields =
    the State dict, aux = PmEpR, INI_CG2D's myNorm and the calc_r_star.F:182-202 counters of both CALC_R_STAR calls."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    z2, z3 = jnp.zeros_like(g.rA), jnp.zeros_like(g.h0FacC)
    one2 = jnp.ones_like(g.rA)
    f = {}

    def interior(base, a):                  # READ_REC / READ_MFLDS: i=1..sNx, j=1..sNy (mdsio)
        return base if a is None else base.at[..., J, I].set(a)

    # --- :141 INI_NLFS_VARS (ini_nlfs_vars.F:42-103) ---------------------------------------------------------------
    etaHnm1, dEtaHdt, PmEpR = z2, z2, z2                                      # :42-48
    f.update(hFac_surfC=z2, hFac_surfW=z2, hFac_surfS=z2)                     # :53-63
    rStarFacC = rStarFacW = rStarFacS = one2                                  # :66-82 (rStarFacNm1* = 1 too:
    # CALC_R_STAR overwrites them from rStarFac* before any read, calc_r_star.F:27-33)
    f.update(pStarFacK=one2, rStarExpC=one2, rStarExpW=one2, rStarExpS=one2,
             rStarDhCDt=z2, rStarDhWDt=z2, rStarDhSDt=z2)
    f.update(hFacC=g.h0FacC, hFacW=g.h0FacW, hFacS=g.h0FacS)                  # :95-103
    # --- :147 INI_DYNVARS (ini_dynvars.F:8-58) ---------------------------------------------------------------------
    for n in ("uVel", "vVel", "wVel", "theta", "salt", "gU", "gV", "guNm_1", "gvNm_1", "gtNm_1", "gsNm_1",
              "guNm_2", "gvNm_2", "gtNm_2", "gsNm_2", "totPhiHyd", "rhoInSitu", "IVDConvCount"):
        f[n] = z3
    f.update(etaN=z2, etaH=z2, phiHydLow=z2, hMixLayer=z2)
    # --- :155 INI_FFIELDS (ini_ffields.F:8-95) ---------------------------------------------------------------------
    for n in ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad", "sIceLoad", "surfaceForcingU",
              "surfaceForcingV", "surfaceForcingT", "surfaceForcingS"):
        f[n] = z2
    f["phi0surf"] = z2                      # ini_linear_phisurf.F:200-209 (INITIALISE_FIXED)
    # --- :164 INI_FIELDS -> READ_PICKUP (read_pickup.F:262-432), then its exchanges (:509-535) --------------------
    for n in ("uVel", "vVel", "theta", "salt", "guNm_1", "guNm_2", "gvNm_1", "gvNm_2"):
        f[n] = interior(f[n], pk.get(n))
    f["etaN"] = interior(f["etaN"], pk.get("etaN"))
    dEtaHdt = interior(dEtaHdt, pk.get("dEtaHdt"))
    f["etaH"] = interior(f["etaH"], pk.get("etaH"))
    f["uVel"], f["vVel"] = ex.vector(f["uVel"], f["vVel"], "UV3s")            # :509 EXCH_UV_3D_RL(.TRUE.)
    f["theta"] = ex.scalar(f["theta"], "3D")                                  # :515 EXCH_3D_RL
    f["salt"] = ex.scalar(f["salt"], "3D")                                    # :516
    f["guNm_1"], f["gvNm_1"] = ex.vector(f["guNm_1"], f["gvNm_1"], "UV3s")    # :518-519
    f["guNm_2"], f["gvNm_2"] = ex.vector(f["guNm_2"], f["gvNm_2"], "UV3s")    # :520-521
    for n in ("gtNm_1", "gtNm_2", "gsNm_1", "gsNm_2"):                        # :522-525
        f[n] = ex.scalar(f[n], "3D")
    f["etaN"] = ex.exch_xy(f["etaN"])                                         # :531
    f["etaH"] = ex.exch_xy(f["etaH"])                                         # :532
    dEtaHdt = ex.exch_xy(dEtaHdt)                                             # :534
    # --- :166-186 CALC_PHI_RLOW_INI --------------------------------------------------------------------------------
    f["totPhiHyd"], f["phiHydLow"] = calc_phi_hyd_ini(P.dyn.phi, P.rs.eos, g, kLowC, f["theta"], f["salt"],
                                                      rStarFacC, f["phi0surf"], f["totPhiHyd"])
    # --- :192 INI_MIXING: grid fields. :203 INI_FORCING (ini_forcing.F:113-128): exchanges of the zero arrays ------
    f["fu"], f["fv"] = ex.exch_uv_xy(f["fu"], f["fv"], True)                  # :114 EXCH_UV_XY_RS(.TRUE.)
    for n in ("Qnet", "EmPmR", "saltFlux", "Qsw", "pLoad"):                   # :115-117, 123, 126 EXCH_XY_RS
        f[n] = ex.exch_xy(f[n])
    # --- :207 AUTODIFF_INIT_VARIA: no model array. :219 PACKAGES_INIT_VARIABLES -------------------------------------
    # GGL90_INIT_VARIA (ggl90_init_varia.F:48-67 full range; useIDEMIX = F) + GGL90_READ_PICKUP (nIter0 != 0)
    f.update(GGL90viscArU=z3, GGL90viscArV=z3, GGL90diffKr=z3)
    if tke is not None:                     # packages_init_variables.F:202-210 useGGL90
        tke0 = P.ggl.GGL90TKEmin * g.maskC                                    # :60-61
        f["GGL90TKE"] = ex.exch_xy(interior(tke0, tke))                       # ggl90_read_pickup.F:61-63
    else:
        f["GGL90TKE"] = z3
    # GMREDI_INIT_VARIA (gmredi_init_varia.F:44-66: GM_EXTRA_DIAGONAL, GM_NON_UNITY_DIAGONAL, GM_BOLUS_ADVEC)
    for n in ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX", "GM_PsiY"):
        f[n] = z3
    # EXF_INIT_VARIA (packages_init_variables.F:244-251; evaluated by the caller, init_inputs): ff exf_init_varia.F
    # (EXF_FIELDS) or the full tree's (pkgs/exf_full.exf_init_varia, EXF_ARRAYS)
    if exf is not None:
        for n in (exfb_mod.EXF_ARRAYS if P.exfb is not None else EXF_FIELDS):
            f[n] = exf[n]
    # SEAICE_INIT_VARIA (packages_init_variables.F:334-337, useSEAICE; pkgs/seaice_init.py): the sea-ice state and
    # sIceLoad (seaice_init_varia.F:685-692; INI_FFIELDS zeroed it above)
    if ice is not None:
        ice_state, sIceLoad = si_mod.seaice_init_varia(ice[0], g, ex, ice[1])
        f.update(ice_state)
        if sIceLoad is not None:
            f["sIceLoad"] = sIceLoad
        # the partly-written SEAICE_DYNSOLVER arrays (seaice_init_varia.F:279-308, 430-432; pkgs/seaice_model.py)
        f.update(sm_mod.dyn_carry_init(L))
    # SALT_PLUME_INIT_VARIA (salt_plume_init_varia.F:45-52; SALT_PLUME_VOLUME undefined)
    f.update(saltPlumeDepth=z2, saltPlumeFlux=z2)
    # CTRL_INIT_VARIABLES (packages_init_variables.F:496-503, useCTRL) -> ff CTRL_MAP_INI_GENARR
    # (ctrl_init_variables.F:393-397) on the DYNVARS controls; kapGM/kapRedi/diffKr are adjusted in model.setup
    if ctrl_in is not None:
        f.update(ctrl_mod.ctrl_map_ini_genarr(ctrl_in, g, ex, {k: f[k] for k in ctrl_mod.STATE_TARGETS}))
    # --- r* sequence (NONLIN_FRSURF, select_rStar > 0, nonlinFreeSurf > 2) ----------------------------------------
    rs1 = fs.calc_r_star(P.fs, g, ex, f["etaH"], rStarFacC, rStarFacW, rStarFacS)   # :259 CALC_R_STAR(etaH, -1)
    cnt1 = {k: rs1.pop(k) for k in ("icntc1", "icntw", "icnts", "icntc2", "maxhFacC")}
    rStarFacC, rStarFacW, rStarFacS = rs1["rStarFacC"], rs1["rStarFacW"], rs1["rStarFacS"]
    hC, hW, hS, rhC, rhW, rhS = fs.update_r_star(                             # :264 UPDATE_R_STAR(.TRUE.)
        g, rStarFacC, rStarFacW, rStarFacS, ini_recip_hfac(g.h0FacC), ini_recip_hfac(g.h0FacW),
        ini_recip_hfac(g.h0FacS))
    f.update(hFacC=hC, hFacW=hW, hFacS=hS, recip_hFacC=rhC, recip_hFacW=rhW, recip_hFacS=rhS)
    pW0, pS0, pC0, myNorm = ini_cg2d(P.fs, P.cg, g, ex)                       # initialise_fixed.F:249 INI_CG2D
    ops = sfp.update_cg2d(P.fs, P.cg, g, ex, hW, hS, g.recip_Bo, pW0, pS0, pC0)  # :277 UPDATE_CG2D
    f.update(zip(("aW2d", "aS2d", "aC2d", "pW", "pS", "pC"), ops))
    ic = integr_continuity_ini(P.fs, g, ex, f["uVel"], f["vVel"], hW, hS, f["etaN"], f["etaH"], dEtaHdt,
                               f["wVel"], PmEpR)                              # :285 INTEGR_CONTINUITY
    f.update(wVel=ic["wVel"], etaN=ic["etaN"], etaH=ic["etaH"], dEtaHdt=ic["dEtaHdt"])
    etaHnm1 = ic["etaHnm1"]
    rs2 = fs.calc_r_star(P.fs, g, ex, f["etaH"], rStarFacC, rStarFacW, rStarFacS)   # :292 CALC_R_STAR(etaH, nIter0)
    cnt2 = {k: rs2.pop(k) for k in ("icntc1", "icntw", "icnts", "icntc2", "maxhFacC")}
    f.update(rs2)
    f["etaHnm1"] = etaHnm1
    aux = dict(PmEpR=ic["PmEpR"], rstar_checks=(cnt1, cnt2), cg2dNorm=myNorm)
    return f, aux


_JIT = {}


def initialise_varia_jit(ex):
    """jax.jit(initialise_varia) for this exchanger (cached): call as f(P, g, kLowC, pk, tke, exf[, ctrl_in]) with
    the parameters as arguments (traced floats: KERNEL_GUIDE, params_pytree)."""
    if id(ex) not in _JIT:
        _JIT[id(ex)] = (ex, jax.jit(lambda P, g, kLowC, pk, tke, exf, ctrl_in=None, ice=None:
                                    initialise_varia(P, g, ex, kLowC, pk, tke, exf, ctrl_in, ice)))
    return _JIT[id(ex)][1]


def init_inputs(P, rundir, cfg: InitConfig, L):
    """Host-side inputs of initialise_varia: (pk, tke, exf, info) -- READ_PICKUP, GGL90_READ_PICKUP (useGGL90) and
    EXF_INIT_VARIA (useEXF): the flux-forced one (P.exf) or the full tree's (P.exfb, pkgs/exf_full)."""
    pk, info = read_pickup(rundir, cfg, L)
    tke = read_ggl90_pickup(rundir, cfg, L) if cfg.useGGL90 else None
    if cfg.useEXF and P.exfb is not None:   # full tree: bulk-formula EXF (exf_init_varia.F, pkgs/exf_full.py)
        exf = exfb_mod.exf_init_varia(P.exfb, L)
    elif cfg.useEXF:
        exf = exf_mod.exf_init_varia(P.exf, L)
    else:                                   # EXF_INIT_VARIA not called: the EXF_FIELDS common block stays 0
        exf = {n: jnp.zeros(L.shape2d) for n in EXF_FIELDS}
    return pk, tke, exf, info


def seaice_inputs(rundir, cfg: InitConfig):
    """Host-side inputs of SEAICE_INIT_VARIA (useSEAICE): (SeaiceInitConfig, pickup_seaice interiors), else None."""
    if not cfg.useSEAICE:
        return None
    icfg = si_mod.SeaiceInitConfig.from_namelists(RunNamelists(rundir))
    return icfg, {k: jnp.asarray(v) for k, v in si_mod.read_seaice_pickup(rundir, icfg).items()}


def state_from_pickup(P, g, ex, kLowC, rundir, return_aux=False, tree=None):
    """The model State at the start of iteration nIter0, built from the run directory's pickup files (see module
    docstring). Raises NotImplementedError for configurations whose initialisation is not ported, ValueError when
    P.dyn.ts.mom_StartAB does not match CHECK_PICKUP or when CALC_R_STAR would STOP, and when the tree of P (P.exfb set:
    full), the declared `tree` ("ff" | "full", None: detect) and the run directory's namelists (model.resolve_tree)
    disagree."""
    from mitgcm_jax.model import resolve_tree
    L = g.layout
    nml = RunNamelists(rundir)
    tree = resolve_tree(nml, tree)
    if (P.exfb is not None) != (tree == "full"):
        raise ValueError(f"ModelParams of the {'full' if P.exfb is not None else 'ff'} tree with a {tree!r} run "
                         f"directory ({rundir})")
    cfg = InitConfig.from_namelists(nml)
    pk, tke, exf, info = init_inputs(P, rundir, cfg, L)
    if P.dyn.ts.mom_StartAB != info.mom_StartAB or P.dyn.ts.nIter0 != cfg.nIter0:
        raise ValueError(f"DynamicsParams built with mom_StartAB={P.dyn.ts.mom_StartAB}, nIter0={P.dyn.ts.nIter0}; "
                         f"the pickup gives mom_StartAB={info.mom_StartAB} (check_pickup.F:60-180), nIter0="
                         f"{cfg.nIter0}: build DynamicsParams.from_namelists(nml, mom_StartAB=...)")
    pk = {k: jnp.asarray(v) for k, v in pk.items()}
    ctrl_in = None
    if cfg.useCTRL:        # CTRL_INIT_VARIABLES inputs (xx, weights, pkg/smooth operators; recip_hFacC of INI_MASKS_ETC)
        ctrl_in = ctrl_mod.ctrl_init(nml, g, ex, ctrl_mod.STATE_TARGETS, ini_recip_hfac(g.h0FacC))
    ice = seaice_inputs(rundir, cfg)
    f, aux = initialise_varia_jit(ex)(P, g, kLowC, pk, None if tke is None else jnp.asarray(tke), exf, ctrl_in, ice)
    for c in aux["rstar_checks"]:                                             # calc_r_star.F:148-190 STOP
        if int(c["icntc1"]) + int(c["icntw"]) + int(c["icnts"]) > 0:
            raise ValueError("CALC_R_STAR: too SMALL rStarFac[C,W,S] (calc_r_star.F:187-189 STOP)")
    missing = set(S00_FIELDS) | set(G00_STATE_FIELDS)
    if P.exfb is not None:                  # full tree: its EXF_FIELDS (no spflx: READIN_SALT_PLUME_FLUX undefined)
        missing = (missing - set(EXF_FIELDS)) | set(exfb_mod.EXF_ARRAYS)
    if cfg.useSEAICE:
        missing |= set(sm_mod.SEAICE_CARRIED)
    missing -= set(f)
    if missing:
        raise RuntimeError(f"initialise_varia did not produce {sorted(missing)}")
    st = State(dict(f), cfg.nIter0)
    return (st, dict(aux, pickup=info)) if return_aux else st
