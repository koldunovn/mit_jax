"""pkg/exf in the FULL ECCO v4r4 configuration (plan M2.1 + M2.2): LOAD_FIELDS_DRIVER -> EXF_GETFORCING with the
atmospheric-state forcing, EXF_RADIATION (+ EXF_ZENITHANGLE), EXF_WIND, EXF_BULKFORMULAE (Large & Yeager 2004),
EXF_GETSURFACEFLUXES and EXF_MAPFIELDS.

Literal port of what the full V4r4 build executes. Sources: c66g `pkg/exf` and `pkg/cal` (the full tree overrides no
pkg/exf source file; cited `<file>:<line>` = MITgcm_c66g/pkg/exf/<file> unless stated) and the build's options
`ECCOv4 Release 4/code/EXF_OPTIONS.h` (cited `code/EXF_OPTIONS.h`) and `code/CPP_OPTIONS.h`:
  ALLOW_ATM_TEMP (:168), ALLOW_ATM_WIND (:169), ALLOW_DOWNWARD_RADIATION (:170), ALLOW_BULKFORMULAE (:173),
  ALLOW_BULK_LARGEYEAGER04 (:174), EXF_READ_EVAP undef (:175), ALLOW_RUNOFF (:184), ALLOW_RUNOFTEMP undef (:185),
  ALLOW_SALTFLX (:186), EXF_CALC_ATMRHO undef (:188-191, with ATMOSPHERIC_LOADING code/CPP_OPTIONS.h:60),
  ALLOW_ZENITHANGLE (:196), EXF_LWDOWN_WITH_EMISSIVITY (:203), ALLOW_CLIMSST/SSS_RELAXATION (:207-208),
  EXF_SEAICE_FRACTION undef (:211), USE_EXF_INTERPOLATION (:215; every interpMethod = 0 in data.exf, so the
  no-interpolation branches), SHORTWAVE_HEATING (code/CPP_OPTIONS.h:23). code/CTRL_OPTIONS.h: ALLOW_GENTIM2D_CONTROL
  (:51), ALLOW_ROTATE_UV_CONTROLS (:67); ALLOW_ECCO with ECCO_CTRL_DEPRECATED undef (code/ECCO_OPTIONS.h:57).

Split host / device (as the M1 port, pkgs/exf_fluxforced.py, whose calendar and record code is reused):
  - host (numpy, plain Python): pkg/cal, EXF_GetFFieldRec / cal_GetMonthsRec (runoff, period -12), the fld0/fld1
    record buffers of EXF_SET_FLD (`ExfFullRecordLoader`), the zenith-angle albedo table EXF_ZENITHANGLE_TABLE and
    every scalar of EXF_ZENITHANGLE that depends only on the date (`zenith_time`), plus grid-only factors
    (`zenith_static`). These call glibc's libm (math / ctypes) exactly where the gfortran binary calls it (sincos where
    GCC fused a SIN/COS pair, checked in the disassembly of exf_zenithangle*.o), so they are bit-identical.
  - device (JAX, pure, jit-able): `exf_getforcing(p, g, ex, exf, ff, bufs, facs, theta, myTime, zt, zs)`.
    ExfFullParams is a params_pytree: pass it as a jit argument.

Elementary functions on the device come from a `Libm` bundle (mitgcm_jax/ops/libm.py). Measured on this CPU: XLA:CPU's
log, atan, sin, cos equal glibc's bit for bit; its exp differs from glibc's in the last bit for ~14 % of arguments and
arccos for ~7 % (glibc 2.28's exp is the IBM fast path, not correctly rounded). The default `DEVICE_LIBM` therefore
uses `exp_glibc` (= ops.libm.glibc_exp), an exact transcription of glibc's exp (FMA variant; fused multiply-adds
emulated exactly), and jnp for the rest; only zen_fsol_daily (arccos; a diagnostic nothing reads) keeps a round-off
difference. The gates also run every kernel with glibc's own functions (test-only host callbacks) to show the kernel
code itself is literal.

Not ported (dead in this configuration, nothing downstream reads them): EXF_GETCLIM and the SST/SSS maps of
EXF_MAPFIELDS (only FORCING_SURF_RELAX reads them; climsst/sssTauRelax = 0), EXF_CHECK_RANGE (useExfCheckRange = F),
EXF_DIAGNOSTICS_FILL, EXF_MONITOR, EXF_ADJOINT_SNAPSHOTS (output only). The zero forcing-control adds (xx_gentim2d
with no gentim2d control; EXF_GETSURFACEFLUXES's rotated tmpUX/tmpVY = 0) are x + 0: see `exf_getsurfacefluxes`.
Any namelist setting that would activate an unported branch raises NotImplementedError in ExfFullParams.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from mitgcm_jax.ops import libm as _libm
from mitgcm_jax.params_io import params_pytree
from mitgcm_jax.pkgs.exf_fluxforced import (
    SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE, N_MONTH_YEAR, Calendar, FieldRec, UNSET_RL, _fint, _idiv,
    _imod, _r4, cal_addtime, cal_fulldate, cal_getdate, cal_isleap, cal_timeinterval, cal_timepassed,
    cal_toseconds, exf_filter_rl, exf_getffield_start, exf_getffieldrec, exf_getyearlyfieldname, exf_set_fld,
    model_clock, model_time, read_rec_2d)

log = logging.getLogger(__name__)

__all__ = ["ExfFullParams", "ExfFullRecordLoader", "exf_getforcing", "exf_init_varia", "zenith_static",
           "zenith_time", "model_time", "cal_getmonthsrec", "Libm", "DEVICE_LIBM", "JNP_LIBM", "exp_glibc"]

# ---------------------------------------------------------------------------------------------------------- constants
EXF_HALF = 0.5              # EXF_CONSTANTS.h:33 exf_half = 0.5 _d 0
EXF_ONE = 1.0               # EXF_CONSTANTS.h:34 exf_one  = 1.0 _d 0
EXF_TWO = 2.0               # EXF_CONSTANTS.h:35 exf_two  = 2.0 _d 0
STEFAN_BOLTZMANN = 5.670e-8  # EXF_CONSTANTS.h:48 stefanBoltzmann = 5.670 _d -8
KARMAN = 0.4                # EXF_CONSTANTS.h:49 karman = 0.4 _d 0
NITER_BULK = 2              # EXF_CONSTANTS.h:91 niter_bulk = 2
PI = 3.14159265358979323844  # PARAMS.h:20 PI = 3.14159265358979323844D0
DEG2RAD = 2.0 * PI / 360.0   # PARAMS.h:22 deg2rad = 2.D0*PI/360.D0
SOLC = 1368.0                # exf_zenithangle.F:67, exf_zenithangle_table.F:47 SOLC = 1368. _d 0

# Fields read by EXF_GETFFIELDS in the full build (exf_getffields.F; file set in data.exf), in call order, with the
# lines of their defaults in exf_readparms.F: (file, startdate1, period, const, intercept, slope, inscal, interpMethod)
FIELDS = ("ustress", "vstress", "wspeed", "atemp", "aqh", "precip", "swdown", "lwdown", "apressure", "runoff")
_DEFAULT_LINES = {
    "ustress": (611, 453, 455, 456, 457, 458, 673, 887),
    "vstress": (612, 460, 462, 463, 464, 465, 674, 888),
    "wspeed": (615, 481, 483, 484, 485, 486, 677, 903),
    "atemp": (600, 379, 381, 382, 383, 384, 684, 894),
    "aqh": (601, 386, 388, 389, 390, 391, 686, 895),
    "precip": (605, 421, 423, 424, 425, 426, 680, 899),
    "swdown": (618, 502, 504, 505, 506, 507, 694, 905),
    "lwdown": (619, 509, 511, 512, 513, 514, 695, 906),
    "apressure": (620, 516, 518, 519, 520, 521, 690, 907),
    "runoff": (608, 435, 437, 438, 439, 440, 691, 892),
}
# exf_readparms.F default interpMethod (:887-908)
_DEFAULT_INTERP = {"ustress": 12, "vstress": 22}
# EXF_SET_FLD/EXF_SET_UV calls whose file must stay ' ' (the code they switch on is not ported), with the fields'
# default files at exf_readparms.F:599-625
INACTIVE_FILES = ("hfluxfile", "sfluxfile", "hs_file", "hl_file", "evapfile", "snowprecipfile", "runoftempfile",
                  "saltflxfile", "uwindfile", "vwindfile", "swfluxfile", "lwfluxfile", "areamaskfile", "climsstfile",
                  "climsssfile", "climustrfile", "climvstrfile")
# every EXF_FIELDS array the device state carries (EXF_FIELDS.h), plus the zenith-angle fields
EXF_ARRAYS = ("ustress", "vstress", "wspeed", "uwind", "vwind", "hflux", "sflux", "atemp", "aqh", "lwflux", "precip",
              "snowprecip", "swflux", "swdown", "lwdown", "apressure", "runoff", "saltflx", "wStress", "cw", "sw",
              "sh", "hs", "hl", "evap", "zen_albedo", "zen_fsol_diurnal", "zen_fsol_daily")


def _masks(useSEAICE, stressIsOnCgrid):
    """exf_init_fixed.F:53-114: EXF_FILTER_RL mask kind of every field ('c', 'w', 's' or ' ' = none)."""
    m = {"hflux": "c", "sflux": "c", "atemp": "c", "aqh": "c", "precip": "c", "snowprecip": "c", "runoff": "c",
         "saltflx": "c", "ustress": "w" if stressIsOnCgrid else "c", "vstress": "s" if stressIsOnCgrid else "c",
         "uwind": "c", "vwind": "c", "wspeed": "c", "swflux": "c", "lwflux": "c", "swdown": "c", "lwdown": "c",
         "apressure": "c"}                                                        # :53-83
    if useSEAICE:                                                                 # :85-114
        for k in ("hflux", "sflux", "ustress", "vstress", "uwind", "vwind", "wspeed", "swflux", "swdown",
                  "apressure"):
            m[k] = " "
    return m


# =====================================================================================================================
# pkg/cal: cal_GetMonthsRec (host), for the monthly runoff (runoffperiod = -12)
# =====================================================================================================================
def cal_subdates(cal, finaldate, initialdate):
    """cal_subdates.F:63-84: finaldate - initialdate."""
    if (initialdate[3] > 0) == (finaldate[3] > 0):
        if initialdate[3] == -1:                                                         # :66-72
            return cal_addtime(cal, list(finaldate), [-initialdate[0], -initialdate[1], 0, -1])
        return cal_timepassed(cal, list(initialdate), list(finaldate))                  # :75-76
    raise ValueError("cal_SubDates: error 801 (cal_subdates.F:80-82)")


def _secs_in_month(date):
    """The FLOAT(...) seconds-since-start-of-month expression of cal_getmonthsrec.F:106-111 / :119-124 (REAL*4)."""
    return float(_r4((_imod(date[0], 100) - 1) * SECONDS_PER_DAY + _idiv(date[1], 10000) * SECONDS_PER_HOUR
                     + _imod(_idiv(date[1], 100), 100) * SECONDS_PER_MINUTE + _imod(date[1], 100)))


def cal_getmonthsrec(cal, myTime, myIter):
    """cal_getmonthsrec.F:82-207 -> FieldRec(fac, first, changed, count0, count1, 0, 0) for 12 monthly records."""
    shifttime = [1, 0, 0, -1]                                                            # :82-85
    modelsteptime = cal_timeinterval(cal, -cal.modelStep)                               # :87-88
    currentdate = cal_getdate(cal, myIter, myTime)                                      # :91
    present = _imod(_idiv(currentdate[0], 100), 100)                                    # :93
    startofmonth_1 = _idiv(currentdate[0], 100) * 100 + 1                               # :94
    startofmonth = cal_fulldate(cal, startofmonth_1, 0)                                 # :96-97
    endofmonth_1 = _idiv(currentdate[0], 100) * 100 + cal.nDayMonth[present - 1][currentdate[2] - 1]  # :99-100
    endofmonth = cal_fulldate(cal, endofmonth_1, 235959)                                # :101-103
    currentsecs = _secs_in_month(currentdate)                                           # :106-111
    midsecs = float(_r4(_idiv(cal.nDayMonth[present - 1][currentdate[2] - 1] * SECONDS_PER_DAY, 2)))  # :112-113
    midtime = cal_timeinterval(cal, midsecs)                                            # :115
    middate = cal_addtime(cal, startofmonth, midtime)                                   # :116
    prevdate = cal_addtime(cal, currentdate, modelsteptime)                             # :117
    prevsecs = _secs_in_month(prevdate)                                                 # :119-124
    first = (myTime - cal.modelStart) < float(_r4(0.5)) * cal.modelStep                 # :127
    if currentsecs < midsecs:                                                           # :133
        count0 = _imod(present + N_MONTH_YEAR - 2, N_MONTH_YEAR) + 1                   # :135
        prevcount = count0                                                              # :136
        shifttime[0] = -shifttime[0]                                                    # :138
        middate0 = cal_addtime(cal, startofmonth, shifttime)                            # :139
        middate0_1 = _idiv(middate0[0], 100) * 100 + 1                                  # :140
        tempDate = cal_fulldate(cal, middate0_1, 0)                                     # :142-143
        previous = _imod(_idiv(tempDate[0], 100), 100)                                  # :145
        midsecs_np = float(_r4(_idiv(cal.nDayMonth[previous - 1][tempDate[2] - 1] * SECONDS_PER_DAY, 2)))  # :147
        midtime = cal_timeinterval(cal, midsecs_np)                                     # :150
        middate0 = cal_addtime(cal, tempDate, midtime)                                  # :151
        count1 = present                                                                # :153
        middate1 = list(middate)                                                        # :155-158
    else:
        count0 = present                                                                # :162
        if prevsecs < midsecs:                                                          # :164-168
            prevcount = _imod(present + N_MONTH_YEAR - 2, N_MONTH_YEAR) + 1
        else:
            prevcount = present
        middate0 = list(middate)                                                        # :170-173
        count1 = _imod(present + 1, N_MONTH_YEAR)                                       # :175
        if count1 == 0:                                                                 # :176
            count1 = N_MONTH_YEAR
        middate1 = cal_addtime(cal, endofmonth, shifttime)                              # :178
        middate1_1 = _idiv(middate1[0], 100) * 100 + 1                                  # :179
        tempDate = cal_fulldate(cal, middate1_1, 0)                                     # :182-183
        nxt = _imod(_idiv(tempDate[0], 100), 100)                                       # :184
        midsecs_np = float(_r4(_idiv(cal.nDayMonth[nxt - 1][tempDate[2] - 1] * SECONDS_PER_DAY, 2)))  # :185
        midtime = cal_timeinterval(cal, midsecs_np)                                     # :187
        middate1 = cal_addtime(cal, tempDate, midtime)                                  # :188
    difftime = cal_subdates(cal, middate1, middate0)                                    # :192
    diffsecs = cal_toseconds(cal, difftime)                                             # :193
    changed = (not first) and (prevcount != count0)                                     # :196-200
    if currentsecs < midsecs:                                                           # :202-207
        fac = (midsecs - currentsecs) / diffsecs
    else:
        fac = (float(_r4(2.0)) * midsecs + midsecs_np - currentsecs) / diffsecs
    return FieldRec(fac=fac, first=bool(first), changed=bool(changed), count0=count0, count1=count1, year0=0,
                    year1=0)


# =====================================================================================================================
# Parameters
# =====================================================================================================================
@dataclass(frozen=True)
class ExfFullField:
    """Host-side description of one EXF_SET_FLD field (file, record timing, mask, initial value)."""
    name: str
    file: str
    mask: str            # EXF_FILTER_RL kind
    period: float        # > 0: EXF_GetFFieldRec; -12: cal_GetMonthsRec (exf_set_fld.F:128-152)
    startTime: float     # seconds from the start of the year of the first record (useExfYearlyFields)
    repCycle: float
    const: float

    @property
    def monthly(self):
        return self.period == -12.0


_FLOATS = ("exf_offset_atemp", "cen2kel", "ocean_emissivity", "exf_albedo", "atmrho", "atmcp", "flamb", "cvapor_fac",
           "cvapor_exp", "humid_fac", "gamma_blk", "saltsat", "cdrag_1", "cdrag_2", "cdrag_3", "cstanton_1",
           "cstanton_2", "cdalton", "psim_fac", "zref", "hu", "ht", "hq", "umin", "exf_scal_BulkCdn", "gravity_mks",
           "zwln", "ztln", "rhoConstFresh", "startTime", "windstressmax", "outscal_hflux", "outscal_sflux",
           "outscal_ustress", "outscal_vstress", "outscal_swflux", "outscal_apressure", "hflux_remo_intercept",
           "hflux_remo_slope", "sflux_remo_intercept", "sflux_remo_slope")


@params_pytree
@dataclass(frozen=True)
class ExfFullParams:
    """EXF parameters of the full build. `float` fields are pytree leaves (pass the object as a jit argument); the
    rest is static: field descriptions, calendar, switches."""
    fields: tuple                 # ExfFullField per FIELDS entry (static)
    cal: Calendar
    useExfYearlyFields: bool
    twoDigitYear: bool
    exf_iprec: int
    select_ZenAlbedo: int
    useExfZenAlbedo: bool
    useExfZenIncoming: bool
    solve4Stress: bool
    exf_consts: tuple             # ((name, fldConst), ...) of every EXF_ARRAYS field initialised by EXF_INIT_FLD
    # scalars (leaves)
    exf_offset_atemp: float
    cen2kel: float
    ocean_emissivity: float
    exf_albedo: float
    atmrho: float
    atmcp: float
    flamb: float
    cvapor_fac: float
    cvapor_exp: float
    humid_fac: float
    gamma_blk: float
    saltsat: float
    cdrag_1: float
    cdrag_2: float
    cdrag_3: float
    cstanton_1: float
    cstanton_2: float
    cdalton: float
    psim_fac: float
    zref: float
    hu: float
    ht: float
    hq: float
    umin: float
    exf_scal_BulkCdn: float
    gravity_mks: float
    zwln: float
    ztln: float
    rhoConstFresh: float
    startTime: float
    windstressmax: float
    outscal_hflux: float
    outscal_sflux: float
    outscal_ustress: float
    outscal_vstress: float
    outscal_swflux: float
    outscal_apressure: float
    hflux_remo_intercept: float
    hflux_remo_slope: float
    sflux_remo_intercept: float
    sflux_remo_slope: float
    # exf_inscal_<f>, <f>_exfremo_intercept, <f>_exfremo_slope of every FIELDS entry
    inscal_ustress: float
    remo_intercept_ustress: float
    remo_slope_ustress: float
    inscal_vstress: float
    remo_intercept_vstress: float
    remo_slope_vstress: float
    inscal_wspeed: float
    remo_intercept_wspeed: float
    remo_slope_wspeed: float
    inscal_atemp: float
    remo_intercept_atemp: float
    remo_slope_atemp: float
    inscal_aqh: float
    remo_intercept_aqh: float
    remo_slope_aqh: float
    inscal_precip: float
    remo_intercept_precip: float
    remo_slope_precip: float
    inscal_swdown: float
    remo_intercept_swdown: float
    remo_slope_swdown: float
    inscal_lwdown: float
    remo_intercept_lwdown: float
    remo_slope_lwdown: float
    inscal_apressure: float
    remo_intercept_apressure: float
    remo_slope_apressure: float
    inscal_runoff: float
    remo_intercept_runoff: float
    remo_slope_runoff: float

    @property
    def field_map(self):
        return {f.name: f for f in self.fields}

    @classmethod
    def from_namelists(cls, nml):
        g1, g2, g3, g4 = "exf_nml_01", "exf_nml_02", "exf_nml_03", "exf_nml_04"
        get = nml.get
        e1 = lambda k, d: get("data.exf", g1, k, default=d)  # noqa: E731
        e2 = lambda k, d: get("data.exf", g2, k, default=d)  # noqa: E731
        e3 = lambda k, d: get("data.exf", g3, k, default=d)  # noqa: E731
        pkg = lambda k: bool(get("data.pkg", "packages", k, default=False))  # noqa: E731
        if not pkg("useEXF"):
            raise NotImplementedError("useEXF=F: this module ports the EXF path only")
        if not pkg("useCAL"):
            raise NotImplementedError("useCAL=F: only the calendar branches of EXF_GetFFieldRec / EXF_SET_FLD are "
                                      "ported (exf_getffieldrec.F:94, exf_set_fld.F:128)")
        if pkg("useCTRL"):
            # exf_getffields.F xx_gentim2d adds, exf_wind.F:229-239, exf_getsurfacefluxes.F:104-222: x + 0 when every
            # forcing-control record is zero (pkgs/ctrl.py; hard error otherwise)
            from mitgcm_jax.pkgs import ctrl as ctrl_mod
            ctrl_mod.require_zero_forcing_controls(nml)
        useSEAICE = pkg("useSEAICE")
        # exf_readparms.F:320-324: ALLOW_ATM_WIND (code/EXF_OPTIONS.h:169) -> useAtmWind default .TRUE.
        if e1("useAtmWind", True):
            raise NotImplementedError("useAtmWind=T: EXF_SET_UV for winds, wind-stress from bulk formulae not ported "
                                      "(exf_getffields.F, exf_bulkformulae.F:501-510)")
        readStressOnCgrid = bool(e1("readStressOnCgrid", False))                        # exf_readparms.F:319
        if readStressOnCgrid:
            raise NotImplementedError("readStressOnCgrid=T: C-grid stress path not ported (exf_getforcing.F:244)")
        if not e1("rotateStressOnAgrid", False):                                        # exf_readparms.F:318
            raise NotImplementedError("rotateStressOnAgrid=F: only the rotated A-grid stress is ported "
                                      "(exf_set_uv.F:555)")
        if e1("useRelativeWind", False) or e1("noNegativeEvap", False):                 # exf_readparms.F:325-326
            raise NotImplementedError("useRelativeWind / noNegativeEvap not ported")
        if e1("exf_yftype", "RL") != "RL":                                               # exf_readparms.F:666
            raise ValueError("exf_yftype must be 'RL' (exf_readparms.F:1023-1025)")
        if e1("repeatPeriod", 0.0) != 0.0:                                               # exf_readparms.F:590
            raise NotImplementedError("repeatPeriod != 0 not ported")
        if not e1("useExfYearlyFields", False):                                          # exf_readparms.F:667
            raise NotImplementedError("useExfYearlyFields=F: that branch of EXF_GetFFieldRec is not ported")
        if e1("sstExtrapol", 0.0) > 0.0:                                                 # exf_readparms.F:347
            raise NotImplementedError("sstExtrapol > 0 (exf_radiation.F:72-94, exf_bulkformulae.F:291-296)")
        if e2("climsstTauRelax", 0.0) != 0.0 or e2("climsssTauRelax", 0.0) != 0.0:     # exf_readparms.F:534,542
            raise NotImplementedError("climsst/sssTauRelax != 0: surface relaxation not ported")
        for key in INACTIVE_FILES:
            if e2(key, " ").strip():
                raise NotImplementedError(f"data.exf {key} is set: the EXF path it activates is not ported")
        temp_EvPrRn = float(get("data", "parm01", "temp_EvPrRn", default=UNSET_RL))      # set_defaults.F:259
        if temp_EvPrRn != UNSET_RL:
            raise NotImplementedError("temp_EvPrRn set: energy of precip/runoff/evap (exf_mapfields.F:143-205) not "
                                      "ported")
        select_ZenAlbedo = int(e1("select_ZenAlbedo", 0))                                # exf_readparms.F:315
        useExfZenAlbedo = 1 <= select_ZenAlbedo <= 3                                     # exf_readparms.F:1040-1041
        if select_ZenAlbedo != 1:
            raise NotImplementedError(f"select_ZenAlbedo={select_ZenAlbedo}: only the albedo table (1) is ported "
                                      "(exf_zenithangle.F:117-264)")
        useExfZenIncoming = bool(e1("useExfZenIncoming", False))                         # exf_readparms.F:316
        cal = Calendar.from_namelists(nml)
        masks = _masks(useSEAICE, readStressOnCgrid)       # stressIsOnCgrid = readStressOnCgrid, :1030-1037
        fields, scal = [], {}
        startTime, _, _ = model_clock(nml)
        celsius2K = float(get("data", "parm01", "celsius2K", default=273.15))            # set_defaults.F:270
        for name in FIELDS:
            # namelist value, else the default at exf_readparms.F:_DEFAULT_LINES[name]
            fname = e2(f"{name}file", " ").strip()
            period = float(e2(f"{name}period", 0.0))
            const = float(e3(f"{name}const", celsius2K if name == "atemp" else 0.0))
            inscal = float(e3(f"exf_inscal_{name}", 1.0))
            icpt = float(e3(f"{name}_exfremo_intercept", 0.0))
            slope = float(e3(f"{name}_exfremo_slope", 0.0))
            interp = get("data.exf", g4, f"{name}_interpMethod", default=_DEFAULT_INTERP.get(name, 1))
            sd1 = e2(f"{name}startdate1", 0)
            sd2 = e2(f"{name}startdate2", 0)
            if nml.has("data.exf", g2, f"{name}StartTime"):
                raise ValueError(f"{name}StartTime cannot be set with useCAL (exf_getffield_start.F:66-80)")
            if not fname:
                raise NotImplementedError(f"{name}file = ' ': the full-tree port reads every one of {FIELDS}")
            if interp >= 1:
                raise NotImplementedError(f"{name}_interpMethod={interp}: EXF_INTERP not ported "
                                          "(exf_set_fld.F:189, exf_set_uv.F:132)")
            if period == -12.0:
                start = 0.0                                            # exf_getffield_start.F:66-67 (period < 0)
            elif period > 0.0:
                start = exf_getffield_start(cal, True, period, sd1, sd2)   # exf_init_fixed.F:123-367
            else:
                raise NotImplementedError(f"{name}period={period}: only > 0 or -12 is ported (exf_set_fld.F:128-152)")
            fields.append(ExfFullField(name=name, file=fname, mask=masks[name], period=period, startTime=start,
                                       repCycle=0.0, const=const))
            scal.update({f"inscal_{name}": inscal, f"remo_intercept_{name}": icpt, f"remo_slope_{name}": slope})
        # EXF_INIT_FLD constants of the other EXF arrays (exf_init_varia.F; exf_readparms.F:372-561 *const = 0)
        consts = {f.name: f.const for f in fields}
        for name in ("hflux", "sflux", "lwflux", "snowprecip", "swflux", "saltflx"):
            consts[name] = float(e3(f"{name}const", 0.0))
        rhoNil = get("data", "parm01", "rhoNil", default=999.8)                             # set_defaults.F:106
        rhoConst = get("data", "parm01", "rhoConst", default=rhoNil)                        # ini_parms.F:445
        rhoConstFresh = get("data", "parm01", "rhoConstFresh", default=rhoConst)            # ini_parms.F:446
        hu = float(e1("hu", 10.0))                                                          # exf_readparms.F:357
        ht = float(e1("ht", 2.0))                                                           # exf_readparms.F:358
        zref = float(e1("zref", 10.0))                                                      # exf_readparms.F:356
        return cls(fields=tuple(fields), cal=cal, useExfYearlyFields=True,
                   twoDigitYear=bool(e1("twoDigitYear", False)),                           # exf_readparms.F:668
                   exf_iprec=int(e1("exf_iprec", 32)),                                      # exf_readparms.F:664
                   select_ZenAlbedo=select_ZenAlbedo, useExfZenAlbedo=useExfZenAlbedo,
                   useExfZenIncoming=useExfZenIncoming,
                   solve4Stress=bool(e2("wspeedfile", " ").strip()),    # exf_bulkformulae.F:232-240 (LARGEYEAGER04)
                   exf_consts=tuple(sorted(consts.items())),
                   exf_offset_atemp=float(e3("exf_offset_atemp", 0.0)),                     # exf_readparms.F:685
                   cen2kel=float(e1("cen2kel", 273.150)),                                   # exf_readparms.F:334
                   ocean_emissivity=float(e1("ocean_emissivity", 5.50e-8 / 5.670e-8)),     # exf_readparms.F:367
                   exf_albedo=float(e1("exf_albedo", 0.1)),                                 # exf_readparms.F:364
                   atmrho=float(e1("atmrho", 1.200)),                                       # exf_readparms.F:336
                   atmcp=float(e1("atmcp", 1005.000)),                                      # exf_readparms.F:337
                   flamb=float(e1("flamb", 2500000.000)),                                   # exf_readparms.F:338
                   cvapor_fac=float(e1("cvapor_fac", 640380.000)),                          # exf_readparms.F:340
                   cvapor_exp=float(e1("cvapor_exp", 5107.400)),                            # exf_readparms.F:341
                   humid_fac=float(e1("humid_fac", 0.606)),                                 # exf_readparms.F:344
                   gamma_blk=float(e1("gamma_blk", 0.010)),                                 # exf_readparms.F:345
                   saltsat=float(e1("saltsat", 0.980)),                                     # exf_readparms.F:346
                   cdrag_1=float(e1("cdrag_1", 0.0027000)),                                 # exf_readparms.F:348
                   cdrag_2=float(e1("cdrag_2", 0.0001420)),                                 # exf_readparms.F:349
                   cdrag_3=float(e1("cdrag_3", 0.0000764)),                                 # exf_readparms.F:350
                   cstanton_1=float(e1("cstanton_1", 0.0327)),                              # exf_readparms.F:351
                   cstanton_2=float(e1("cstanton_2", 0.0180)),                              # exf_readparms.F:352
                   cdalton=float(e1("cdalton", 0.0346)),                                    # exf_readparms.F:353
                   psim_fac=float(e1("psim_fac", 5.000)),                                   # exf_readparms.F:355
                   zref=zref, hu=hu, ht=ht,
                   hq=ht,                                                                   # exf_readparms.F:1029
                   umin=float(e1("umin", 0.5)),                                             # exf_readparms.F:359
                   exf_scal_BulkCdn=float(e1("exf_scal_BulkCdn", 1.0)),                     # exf_readparms.F:593
                   gravity_mks=float(e1("gravity_mks", 9.81)),                              # exf_readparms.F:335
                   # exf_bulkformulae.F:243-244 zwln = LOG(hu/zref), ztln = LOG(ht/zref): glibc log (math.log), as
                   # the binary computes them (constant inputs; derived once here)
                   zwln=math.log(hu / zref), ztln=math.log(ht / zref),
                   rhoConstFresh=float(rhoConstFresh), startTime=startTime,
                   windstressmax=float(e1("windstressmax", 2.0)),                           # exf_readparms.F:591
                   outscal_hflux=float(e3("exf_outscal_hflux", 1.0)),                       # exf_readparms.F:703
                   outscal_sflux=float(e3("exf_outscal_sflux", 1.0)),                       # exf_readparms.F:704
                   outscal_ustress=float(e3("exf_outscal_ustress", 1.0)),                   # exf_readparms.F:705
                   outscal_vstress=float(e3("exf_outscal_vstress", 1.0)),                   # exf_readparms.F:706
                   outscal_swflux=float(e3("exf_outscal_swflux", 1.0)),                     # exf_readparms.F:707
                   outscal_apressure=float(e3("exf_outscal_apressure", 1.0)),               # exf_readparms.F:710
                   hflux_remo_intercept=float(e3("hflux_exfremo_intercept", 0.0)),          # exf_readparms.F:376
                   hflux_remo_slope=float(e3("hflux_exfremo_slope", 0.0)),                  # exf_readparms.F:377
                   sflux_remo_intercept=float(e3("sflux_exfremo_intercept", 0.0)),          # exf_readparms.F:411
                   sflux_remo_slope=float(e3("sflux_exfremo_slope", 0.0)),                  # exf_readparms.F:412
                   **scal)


# =====================================================================================================================
# Host side: record buffers (EXF_SET_FLD fld0/fld1), zenith-angle table and date scalars
# =====================================================================================================================
class ExfFullRecordLoader:
    """The fld0/fld1 buffers of every EXF_SET_FLD call of the full build (exf_set_fld.F:116-279; EXF_SET_UV's
    no-interpolation branch calls EXF_SET_FLD for each stress component, exf_set_uv.F:532-552).

    `load(myTime, myIter)` must be called once per time step, in order, starting at the model start (first=.TRUE.);
    it returns (bufs, facs, recs) with bufs[name] = (fld0, fld1) numpy [T, ny, nx] (masked, halos = fldConst as
    EXF_INIT_FLD left them), facs[name] = weight of fld0, recs[name] = FieldRec. Every record read is logged."""

    def __init__(self, p, g, rundir):
        self.p = p
        self.g = g
        self.dir = Path(rundir)
        L = g.layout
        # exf_init_fld.F:88-97: fld0 = fld1 = fldConst everywhere
        self.buf = {f.name: [np.full(L.shape2d, f.const), np.full(L.shape2d, f.const)] for f in p.fields}
        self.started = False
        self.loaded = []   # (myIter, name, rec, file) of every record read

    def _read(self, f, fname, rec, myIter):
        log.info('EXF_SET_FLD: field "%s", it=%d, loading rec=%d from file "%s"', f.name, myIter, rec, fname)
        self.loaded.append((myIter, f.name, rec, fname))
        a = read_rec_2d(self.dir / fname, self.p.exf_iprec, rec, self.g.layout)              # exf_set_fld.F:202,261
        return np.asarray(exf_filter_rl(jnp.asarray(a), f.mask, self.g))                    # exf_set_fld.F:219,276

    def record(self, f, myTime, myIter):
        if f.monthly:                                                                         # exf_set_fld.F:128-136
            return cal_getmonthsrec(self.p.cal, myTime, myIter)
        return exf_getffieldrec(self.p.cal, f.startTime, f.period, self.p.useExfYearlyFields, myTime, myIter)

    def load(self, myTime, myIter):
        p, L = self.p, self.g.layout
        J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
        bufs, facs, recs = {}, {}, {}
        for f in p.fields:
            r = self.record(f, myTime, myIter)
            fld0, fld1 = self.buf[f.name]
            if not r.first and not self.started:
                raise RuntimeError("ExfFullRecordLoader.load must start at the model start time (first=.TRUE.)")
            if r.first:                                                                       # exf_set_fld.F:167-222
                fname = exf_getyearlyfieldname(p.useExfYearlyFields, p.twoDigitYear, f.period, r.year0, f.file)
                fld1 = fld1.copy()
                fld1[:, J, I] = self._read(f, fname, r.count0, myIter)[:, J, I]
            if r.first or r.changed:                                                          # exf_set_fld.F:224-279
                fld0 = fld0.copy()                                                            # exf_swapffields.F:58-67
                fld0[:, J, I] = fld1[:, J, I]
                fld1 = fld1.copy()
                fld1[:, J, I] = 0.0
                fname = exf_getyearlyfieldname(p.useExfYearlyFields, p.twoDigitYear, f.period, r.year1, f.file)
                fld1[:, J, I] = self._read(f, fname, r.count1, myIter)[:, J, I]
            self.buf[f.name] = [fld0, fld1]
            bufs[f.name] = (fld0, fld1)
            facs[f.name] = r.fac
            recs[f.name] = r
        self.started = True
        return bufs, facs, recs


_LIBC = None


def _libm_c():
    """glibc's libm through ctypes (sincos is not in Python's math module)."""
    global _LIBC
    if _LIBC is None:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("m"))
        lib.sincos.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        lib.sincos.restype = None
        _LIBC = lib
    return _LIBC


def glibc_sincos(x):
    """glibc sincos(x) -> (sin, cos) for a Python float: what gfortran -O3 calls for a SIN/COS pair of one argument
    (the objects exf_zenithangle.o / exf_zenithangle_table.o call sincos 10 / 5 times, see `zenith_time`)."""
    import ctypes
    s, c = ctypes.c_double(), ctypes.c_double()
    _libm_c().sincos(float(x), ctypes.byref(s), ctypes.byref(c))
    return s.value, c.value


def _vec(fn, x):
    x = np.asarray(x, np.float64)
    return np.fromiter((fn(float(v)) for v in x.ravel()), np.float64, x.size).reshape(x.shape)


def _decli(alpha):
    """DECLI and dD0dDsq (exf_zenithangle.F:283-295 = exf_zenithangle_table.F:58-81); 1.*ALPHA is ALPHA."""
    s1, c1 = glibc_sincos(alpha)
    s2, c2 = glibc_sincos(2.0 * alpha)
    s3, c3 = glibc_sincos(3.0 * alpha)
    decli = (0.006918 - 0.399912 * c1 + 0.070257 * s1 - 0.006758 * c2 + 0.000907 * s2 - 0.002697 * c3
             + 0.001480 * s3)
    dD0dDsq = 1.000110 + 0.034221 * c1 + 0.001280 * s1 + 0.000719 * c2 + 0.000077 * s2
    return decli, dD0dDsq


def zenith_albedo_table():
    """EXF_ZENITHANGLE_TABLE (exf_zenithangle_table.F:49-115): the (366, 181) table of daily-mean ocean albedo,
    numpy float64, [iTyear-1, iLat-1]. Scalar code evaluated with glibc (sincos / cos / pow as the binary calls it),
    vectorised over (iTyear, iLat) with numpy's IEEE +, -, *, / in the Fortran order."""
    iTyear = np.arange(1, 367, dtype=np.float64)
    TYEAR = (iTyear - 1.0) / 365.0                                                      # :53
    ALPHA = (2.0 * PI) * TYEAR                                                          # :58
    dec = [_decli(float(a)) for a in ALPHA]                                             # :59-81
    DECLI = np.array([d[0] for d in dec])
    dD0dDsq = np.array([d[1] for d in dec])
    sc = [glibc_sincos(float(d)) for d in DECLI]                                        # :67-68 ZC, ZS
    ZS = np.array([v[0] for v in sc])[:, None]
    ZC = np.array([v[1] for v in sc])[:, None]
    LLLAT = np.arange(1, 182, dtype=np.float64) - 91.0                                  # :52
    scl = [glibc_sincos(float(v) * DEG2RAD) for v in LLLAT]                             # :69-70 SJ, CJ
    SJ = np.array([v[0] for v in scl])[None, :]
    CJ = np.array([v[1] for v in scl])[None, :]
    TMPA = SJ * ZS                                                                      # :71
    TMPB = CJ * ZC                                                                      # :72
    solc_d = SOLC * dD0dDsq[:, None]                                                    # :94 SOLC*dD0dDsq
    tmpINT1 = np.zeros((366, 181))                                                      # :83
    tmpINT2 = np.zeros((366, 181))                                                      # :84
    pow17 = np.frompyfunc(lambda c: math.pow(c, 1.7), 1, 1)
    for iTday in range(1, 101):                                                         # :85
        TDAY = iTday / 100.0                                                            # :86
        CZEN = TMPA + TMPB * math.cos((2.0 * PI) * TDAY + 0.0 * DEG2RAD)               # :89-90
        CZEN = np.where(CZEN <= 0, 0.0, CZEN)                                           # :91
        FSOL = solc_d * np.maximum(0.0, CZEN)                                           # :94
        ALBSEA1 = ((2.6 / (pow17(CZEN).astype(np.float64) + 0.065))
                   + 15.0 * (CZEN - 0.1) * (CZEN - 0.5) * (CZEN - 1.0)) / 100.0         # :98-100
        tmpINT1 = tmpINT1 + FSOL * ALBSEA1 / 100.0                                      # :103
        tmpINT2 = tmpINT2 + FSOL / 100.0                                                # :104
    return np.where(0.5 * tmpINT2 > tmpINT1, tmpINT1 / np.where(tmpINT2 != 0.0, tmpINT2, 1.0), 0.5)   # :108-112


def zenith_static(p, g, table=None):
    """Grid-only factors of EXF_ZENITHANGLE (device pytree of jnp arrays): the albedo table, zen_albedo_pointer
    (exf_zenithangle_table.F:122-135) split into iLat1/iLat2 (0-based) and weights wLat1/wLat2 (exf_zenithangle.F:
    140-150), and SJ, CJ = glibc sincos(yC*deg2rad) (:301-302, one sincos in the binary), tanLat = TAN(yC*deg2rad)
    (:313), xCrad = xC*deg2rad (:307)."""
    table = zenith_albedo_table() if table is None else table
    yC = np.asarray(g.yC, np.float64)
    xC = np.asarray(g.xC, np.float64)
    LLLAT = yC + 91.0                                                                   # table.F:126
    LLLAT = np.maximum(LLLAT, 1.0)                                                      # table.F:128
    ptr = np.minimum(LLLAT, 181.0)                                                      # table.F:129-131
    at181 = ptr == 181.0                                                                # zenithangle.F:140
    iLat1 = np.where(at181, 181, np.trunc(ptr).astype(np.int64))                        # :141, :146 INT()
    wLat1 = np.where(at181, 0.5, (1.0 + iLat1) - ptr)                                   # :142, :147
    iLat2 = np.where(at181, 181, iLat1 + 1)                                             # :143, :148
    wLat2 = np.where(at181, 0.5, 1.0 - wLat1)                                           # :144, :149
    arg = yC * DEG2RAD
    sc = np.frompyfunc(glibc_sincos, 1, 2)(arg)
    SJ = np.asarray(sc[0], np.float64)
    CJ = np.asarray(sc[1], np.float64)
    return dict(table=jnp.asarray(table), iLat1=jnp.asarray(iLat1 - 1, dtype=jnp.int32),
                iLat2=jnp.asarray(iLat2 - 1, dtype=jnp.int32), wLat1=jnp.asarray(wLat1), wLat2=jnp.asarray(wLat2),
                SJ=jnp.asarray(SJ), CJ=jnp.asarray(CJ), tanLat=jnp.asarray(_vec(math.tan, arg)),
                xCrad=jnp.asarray(xC * DEG2RAD))


def zenith_time(p, myTime, myIter):
    """The date-only scalars of EXF_ZENITHANGLE (exf_zenithangle.F:73-110, 133-136, 283-297), glibc as the binary:
    TYEAR, TDAY, iTyear1/2 (0-based), wTyear1/2, and for the incoming flux 2*PI*TDAY, DECLI, dD0dDsq, ZC, ZS,
    TAN(DECLI), SOLC*dD0dDsq. Returns a dict of Python scalars (pass them to the device as traced arguments)."""
    cal = p.cal
    mydate = cal_getdate(cal, myIter, myTime)                                           # :75
    year0 = _fint(np.float32(mydate[0]) / _r4(10000.0))                                 # :76 (REAL*4 10000.)
    secondsInYear = float(cal.nDaysNoLeap * SECONDS_PER_DAY)                            # :77
    if cal_isleap(cal, year0) == 2:                                                     # :78-79
        secondsInYear = float(cal.nDaysLeap * SECONDS_PER_DAY)
    yearStartDate = [year0 * 10000 + 101, 0, mydate[2], mydate[3]]                      # :81-84
    myDateSeconds = cal_toseconds(cal, cal_timepassed(cal, yearStartDate, mydate))      # :85-86
    TYEAR = myDateSeconds / secondsInYear                                               # :88
    dayStartDate = [mydate[0], 0, mydate[2], mydate[3]]                                 # :90-93
    myDateSeconds = cal_toseconds(cal, cal_timepassed(cal, dayStartDate, mydate))       # :94-95
    TDAY = myDateSeconds / 86400.0                                                      # :97
    iTyear1 = _fint(1 + float(_r4(365.0)) * TYEAR)                                      # :133 (INT of 1+365.*TYEAR)
    wTyear1 = iTyear1 - float(_r4(365.0)) * TYEAR                                       # :134
    iTyear2 = iTyear1 + 1                                                               # :135
    wTyear2 = 1.0 - wTyear1                                                             # :136
    ALPHA = (2.0 * PI) * TYEAR                                                          # :283
    DECLI, dD0dDsq = _decli(ALPHA)                                                      # :284-295
    ZS, ZC = glibc_sincos(DECLI)                                                        # :296-297
    return dict(TYEAR=TYEAR, TDAY=TDAY, iTyear1=iTyear1 - 1, iTyear2=iTyear2 - 1, wTyear1=wTyear1, wTyear2=wTyear2,
                twoPiTday=(2.0 * PI) * TDAY, DECLI=DECLI, dD0dDsq=dD0dDsq, ZC=ZC, ZS=ZS, tanDecli=math.tan(DECLI),
                solc_d=SOLC * dD0dDsq)


# =====================================================================================================================
# Device side (JAX)
# =====================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------
# exp as the gfortran binary computes it (glibc 2.28, FMA variant, every path for |x| <= ~708) and the Libm bundles:
# mitgcm_jax/ops/libm.py (shared with pkgs/seaice_growth.py). Names kept here as aliases. The EXF argument of exp
# (-cvapor_exp/Tsf, EXF_BULKFORMULAE) lies in [-26, -14], the table path of glibc's exp.
# ---------------------------------------------------------------------------------------------------------------------
exp_glibc = _libm.glibc_exp
fma = _libm.fma_emulated
exp_tables = _libm.exp_tables
Libm = _libm.Libm
JNP_LIBM = _libm.JNP_LIBM
DEVICE_LIBM = _libm.DEVICE_LIBM


def exf_init_varia(p, layout):
    """exf_init_varia.F:44-378 + exf_init_fld.F:88-97 (every period > 0 or -12: nothing read at init): every EXF
    array = its fldConst everywhere; wStress, cw, sw, sh, hs, hl = 0 (:45-60), uwind = vwind = 0 (:132-141). evap and
    the zenith fields are not initialised by the Fortran (EXF_FIELDS.h common blocks: 0)."""
    exf = {k: jnp.zeros(layout.shape2d) for k in EXF_ARRAYS}
    for name, c in p.exf_consts:
        exf[name] = jnp.full(layout.shape2d, c)
    return exf


def _interior(layout):
    return layout.js(1, layout.sNy), layout.is_(1, layout.sNx)


def exf_getffields(p, g, exf, bufs, facs, myTime):
    """exf_getffields.F (full build): EXF_SET_UV(ustress, vstress) no-interpolation branch with rotation
    (exf_set_uv.F:532-580), EXF_SET_FLD of wspeed, atemp (+ exf_offset_atemp), aqh, precip, swdown, lwdown,
    apressure, runoff; uwind = vwind = 0 (useAtmWind = F). The calls with file ' ' (hflux, sflux, lwflux, snowprecip,
    swflux, saltflx) do nothing (exf_set_fld.F:116). The useCTRL block adds xx_gentim2d (none: zero) and the
    rotated zero tmpUX/tmpVY to uwind/vwind (x + 0)."""
    L = g.layout
    J, I = _interior(L)
    exf = dict(exf)

    def set_fld(name):
        fld0, fld1 = bufs[name]
        return exf_set_fld(getattr(p, f"inscal_{name}"), getattr(p, f"remo_intercept_{name}"),
                           getattr(p, f"remo_slope_{name}"), exf[name], fld0, fld1, facs[name], myTime,
                           p.startTime, L)
    # EXF_SET_UV (exf_set_uv.F:532-552): two EXF_SET_FLD calls
    u = set_fld("ustress")
    v = set_fld("vstress")
    # vector rotation (exf_set_uv.F:555-580, rotateStressOnAgrid)
    tmp_u, tmp_v = u[:, J, I], v[:, J, I]                                               # :560-561
    cosC, sinC = g.angleCosC[:, J, I], g.angleSinC[:, J, I]
    u = u.at[:, J, I].set(cosC * tmp_u + sinC * tmp_v)                                  # :570-572
    v = v.at[:, J, I].set(-sinC * tmp_u + cosC * tmp_v)                                 # :573-575
    exf["ustress"], exf["vstress"] = u, v
    exf["wspeed"] = set_fld("wspeed")
    # useAtmWind = F: uwind = vwind = 0 on the whole array (exf_getffields.F ELSE of IF(useAtmWind))
    exf["uwind"] = jnp.zeros_like(exf["uwind"])
    exf["vwind"] = jnp.zeros_like(exf["vwind"])
    atemp = set_fld("atemp")
    exf["atemp"] = atemp.at[:, J, I].set(atemp[:, J, I] + p.exf_offset_atemp)           # atemp + exf_offset_atemp
    for name in ("aqh", "precip", "swdown", "lwdown", "apressure", "runoff"):
        exf[name] = set_fld(name)
    return exf


def exf_zenithangle(p, g, exf, zt, zs, libm=DEVICE_LIBM):
    """exf_zenithangle.F:112-337 with useCAL, select_ZenAlbedo = 1 (albedo table) and useExfZenIncoming: returns exf
    with zen_albedo, zen_fsol_diurnal, zen_fsol_daily updated on the interior. zt = zenith_time(...) (scalars),
    zs = zenith_static(...) (grid factors)."""
    L = g.layout
    J, I = _interior(L)
    exf = dict(exf)
    if p.useExfZenAlbedo:                                                               # :112
        # select_ZenAlbedo = 1 (:125-161)
        row1 = jnp.take(zs["table"], zt["iTyear1"], axis=0)
        row2 = jnp.take(zs["table"], zt["iTyear2"], axis=0)
        i1, i2 = zs["iLat1"][:, J, I], zs["iLat2"][:, J, I]
        wL1, wL2 = zs["wLat1"][:, J, I], zs["wLat2"][:, J, I]
        wT1, wT2 = zt["wTyear1"], zt["wTyear2"]
        ALBSEA1 = (wT1 * wL1 * row1[i1] + wT1 * wL2 * row1[i2]
                   + wT2 * wL1 * row2[i1] + wT2 * wL2 * row2[i2])                        # :151-155
        za = 0.5 * p.exf_albedo + 0.5 * ALBSEA1                                          # :158-159
        exf["zen_albedo"] = exf["zen_albedo"].at[:, J, I].set(za)
    if p.useExfZenIncoming:                                                             # :275
        TMPA = zs["SJ"][:, J, I] * zt["ZS"]                                              # :303
        TMPB = zs["CJ"][:, J, I] * zt["ZC"]                                              # :304
        CZEN = TMPA + TMPB * libm.cos(zt["twoPiTday"] + zs["xCrad"][:, J, I])          # :306-307
        CZEN = jnp.where(CZEN <= 0, 0.0, CZEN)                                          # :308
        FSOL = zt["solc_d"] * jnp.maximum(0.0, CZEN)                                     # :309
        exf["zen_fsol_diurnal"] = exf["zen_fsol_diurnal"].at[:, J, I].set(FSOL)          # :310
        H0 = -zs["tanLat"][:, J, I] * zt["tanDecli"]                                     # :313
        H0 = jnp.where(H0 < -1.0, -1.0, H0)                                             # :314
        H0 = jnp.where(H0 > 1.0, 1.0, H0)                                               # :315
        H0 = libm.acos(H0)                                                              # :316
        FSOL = zt["solc_d"] / PI * (H0 * TMPA + libm.sin(H0) * TMPB)                     # :317-318
        exf["zen_fsol_daily"] = exf["zen_fsol_daily"].at[:, J, I].set(FSOL)              # :319
    return exf


def exf_radiation(p, g, exf, theta, zt, zs, libm=DEVICE_LIBM):
    """exf_radiation.F:63-182: lwflux from lwdown and SST (lwfluxfile ' ', lwdownfile set; sstExtrapol = 0 branch
    :96-112, EXF_LWDOWN_WITH_EMISSIVITY), EXF_ZENITHANGLE, swflux = -swdown*(1-zen_albedo) (swfluxfile ' ',
    swdownfile set, useExfZenAlbedo). theta: [T, Nr, ny, nx] (level ks = 1)."""
    L = g.layout
    J, I = _interior(L)
    exf = dict(exf)
    ks = 0                                                                              # :64 ks = 1
    Tsf = theta[:, ks, J, I] + p.cen2kel
    Tsq = Tsf * Tsf                                                                     # (x)**4: gfortran x2*x2
    lw = (p.ocean_emissivity * STEFAN_BOLTZMANN) * (Tsq * Tsq) - exf["lwdown"][:, J, I] * p.ocean_emissivity  # :98-103
    exf["lwflux"] = exf["lwflux"].at[:, J, I].set(lw)
    exf = exf_zenithangle(p, g, exf, zt, zs, libm)                                      # :141-142
    sw = -exf["swdown"][:, J, I] * (1.0 - exf["zen_albedo"][:, J, I])                    # :162-168
    exf["swflux"] = exf["swflux"].at[:, J, I].set(sw)
    return exf


def exf_wind(p, g, exf):
    """exf_wind.F:94-250, useAtmWind = F, stress on the A grid, wspeed read from file: wStress, cw, sw from the
    stress; uwind, vwind = wspeed*(cw, sw); sh = MAX(wspeed, uMin). Interior only."""
    L = g.layout
    J, I = _interior(L)
    exf = dict(exf)
    u, v = exf["ustress"][:, J, I], exf["vstress"][:, J, I]
    usSq = u * u + v * v                                                                # :152-153
    nz = usSq != 0.0                                                                    # :155
    ws = jnp.sqrt(jnp.where(nz, usSq, 1.0))                                              # :156 (guarded for AD)
    wStress = jnp.where(nz, ws, 0.0)                                                    # :156, :161
    cw = jnp.where(nz, u / ws, 0.0)                                                     # :158, :162
    sw = jnp.where(nz, v / ws, 0.0)                                                     # :159, :163
    wspeed = exf["wspeed"][:, J, I]
    uwind = wspeed * cw                                                                 # :220
    vwind = wspeed * sw                                                                 # :221
    sh = jnp.maximum(wspeed, p.umin)                                                    # :248
    for k, a in (("wStress", wStress), ("cw", cw), ("sw", sw), ("uwind", uwind), ("vwind", vwind), ("sh", sh)):
        exf[k] = exf[k].at[:, J, I].set(a)
    return exf


def exf_bulkformulae(p, g, exf, theta, libm=DEVICE_LIBM, return_locals=False):
    """exf_bulkformulae.F:232-527, ALLOW_BULK_LARGEYEAGER04, solve4Stress (wspeedfile set), EXF_CALC_ATMRHO undef,
    sstExtrapol = 0, noNegativeEvap = F: hs, hl, evap (and hflux = 0 where atemp = 0). niter_bulk = 2 fixed
    stability iterations. return_locals=True also returns the interior locals after the first guess (X05a: tstar,
    qstar, ustar, rdn, delq, deltap) and after the iterations (X05b: tstar, qstar, ustar, tau, rdn, rd)."""
    if not p.solve4Stress:
        raise NotImplementedError("solve4Stress = F (wspeedfile ' '): exf_bulkformulae.F:326-337 not ported")
    L = g.layout
    J, I = _interior(L)
    exf = dict(exf)
    zwln, ztln = p.zwln, p.ztln                                                         # :243-244
    czol = p.hu * KARMAN * p.gravity_mks                                                # :245
    recip_rhoConstFresh = 1.0 / p.rhoConstFresh                                         # :247
    ksrf = 0                                                                            # :260 ksrf = 1
    atemp, aqh = exf["atemp"][:, J, I], exf["aqh"][:, J, I]
    sh, wspeed = exf["sh"][:, J, I], exf["wspeed"][:, J, I]
    act = atemp != 0.0                                                                  # :288, :359, :476
    zero = jnp.zeros_like(atemp)
    # first guess (:283-353)
    Tsf = theta[:, ksrf, J, I] + p.cen2kel                                               # :290
    tmpbulk = p.cvapor_fac * libm.exp(-p.cvapor_exp / Tsf)                              # :298
    ssq = p.saltsat * tmpbulk / p.atmrho                                                # :305
    deltap = jnp.where(act, atemp + p.gamma_blk * p.ht - Tsf, 0.0)                      # :283, :307
    delq = jnp.where(act, aqh - ssq, 0.0)                                               # :284, :308
    stable = EXF_HALF + jnp.copysign(EXF_HALF, deltap)                                  # :314
    wsm = sh                                                                            # :321
    tmpbulk = p.exf_scal_BulkCdn * (p.cdrag_1 / wsm + p.cdrag_2 + p.cdrag_3 * wsm)      # :322-323
    rdn = jnp.where(act, jnp.sqrt(tmpbulk), 0.0)                                        # :324, :352
    ustar = jnp.where(act, rdn * wsm, 0.0)                                              # :325, :350
    rhn = (EXF_ONE - stable) * p.cstanton_1 + stable * p.cstanton_2                     # :340
    ren = p.cdalton                                                                     # :341
    tstar = jnp.where(act, rhn * deltap, 0.0)                                           # :343, :348
    qstar = jnp.where(act, ren * delq, 0.0)                                             # :344, :349
    tau = zero                                                                          # :351 (atemp = 0)
    rd = zero
    locals_a = dict(tstar=tstar, qstar=qstar, ustar=ustar, rdn=rdn, delq=delq, deltap=deltap)
    # masked lanes (atemp = 0) keep their values; their divisors are replaced by 1 so every lane stays finite (AD)
    atemp_s = jnp.where(act, atemp, 1.0)
    for _ in range(NITER_BULK):                                                         # :356
        ustar_s = jnp.where(act, ustar, 1.0)
        t0 = atemp_s * (EXF_ONE + p.humid_fac * aqh)                                     # :378-379
        huol = (tstar / t0 + qstar / (EXF_ONE / p.humid_fac + aqh)) * czol / (ustar_s * ustar_s)  # :380-382
        tmpbulk = jnp.minimum(jnp.abs(huol), 10.0)                                       # :384
        huol = jnp.copysign(tmpbulk, huol)                                               # :385
        htol = huol * p.ht / p.hu                                                        # :390
        stable = EXF_HALF + jnp.copysign(EXF_HALF, huol)                                 # :392
        xsq = jnp.sqrt(jnp.abs(EXF_ONE - huol * 16.0))                                   # :398
        x = jnp.sqrt(xsq)                                                                # :403
        psimh = (-p.psim_fac * huol * stable
                 + (EXF_ONE - stable) * (libm.log((EXF_ONE + EXF_TWO * x + xsq) * (EXF_ONE + xsq) * 0.125)
                                         - EXF_TWO * libm.atan(x) + EXF_HALF * PI))       # :404-408
        xsq = jnp.sqrt(jnp.abs(EXF_ONE - htol * 16.0))                                   # :414
        psixh = (-p.psim_fac * htol * stable
                 + (EXF_ONE - stable) * (EXF_TWO * libm.log(EXF_HALF * (EXF_ONE + xsq))))  # :419-420
        dzTmp = (zwln - psimh) / KARMAN                                                  # :427
        usn = wspeed / (EXF_ONE + rdn * dzTmp)                                           # :428
        usm = jnp.maximum(usn, p.umin)                                                   # :437
        tmpbulk = p.exf_scal_BulkCdn * (p.cdrag_1 / usm + p.cdrag_2 + p.cdrag_3 * usm)  # :440-441
        rdn_new = jnp.sqrt(tmpbulk)                                                      # :442
        rd_new = rdn_new / (EXF_ONE + rdn_new * dzTmp)                                   # :444
        ustar_new = rd_new * sh                                                          # :448
        tau_new = p.atmrho * rd_new * wspeed                                             # :453
        rhn = (EXF_ONE - stable) * p.cstanton_1 + stable * p.cstanton_2                  # :458
        ren = p.cdalton                                                                  # :459
        rh = rhn / (EXF_ONE + rhn * (ztln - psixh) / KARMAN)                             # :462
        re = ren / (EXF_ONE + ren * (ztln - psixh) / KARMAN)                             # :463
        qstar = jnp.where(act, re * delq, qstar)                                         # :466
        tstar = jnp.where(act, rh * deltap, tstar)                                       # :467
        rdn = jnp.where(act, rdn_new, rdn)
        rd = jnp.where(act, rd_new, rd)
        ustar = jnp.where(act, ustar_new, ustar)
        tau = jnp.where(act, tau_new, tau)
    hs = jnp.where(act, p.atmcp * tau * tstar, 0.0)                                     # :492, :518
    hl = jnp.where(act, p.flamb * tau * qstar, 0.0)                                     # :493, :519
    evap = jnp.where(act, -recip_rhoConstFresh * tau * qstar, 0.0)                      # :497, :517
    exf["hs"] = exf["hs"].at[:, J, I].set(hs)
    exf["hl"] = exf["hl"].at[:, J, I].set(hl)
    exf["evap"] = exf["evap"].at[:, J, I].set(evap)
    exf["hflux"] = exf["hflux"].at[:, J, I].set(jnp.where(act, exf["hflux"][:, J, I], 0.0))   # :516
    if return_locals:
        return exf, locals_a, dict(tstar=tstar, qstar=qstar, ustar=ustar, tau=tau, rdn=rdn, rd=rd)
    return exf


def exf_hflux_sflux(p, g, exf):
    """exf_getforcing.F:203-237 (ALLOW_ATM_TEMP, SHORTWAVE_HEATING, ALLOW_RUNOFF): hflux = -hs - hl + lwflux,
    sflux = evap - precip - runoff, both masked with maskC(k=1). Interior only."""
    L = g.layout
    J, I = _interior(L)
    exf = dict(exf)
    mC = g.maskC[:, 0, J, I]                                                            # :225 k = 1
    hflux = -exf["hs"][:, J, I] - exf["hl"][:, J, I] + exf["lwflux"][:, J, I]          # :211-214
    sflux = exf["evap"][:, J, I] - exf["precip"][:, J, I]                               # :219
    sflux = sflux - exf["runoff"][:, J, I]                                              # :229
    exf["hflux"] = exf["hflux"].at[:, J, I].set(hflux * mC)                             # :231
    exf["sflux"] = exf["sflux"].at[:, J, I].set(sflux * mC)                             # :232
    return exf


def exf_getsurfacefluxes(p, exf):
    """exf_getsurfacefluxes.F:87-224 (ALLOW_CTRL, ALLOW_GENTIM2D_CONTROL, ALLOW_ROTATE_UV_CONTROLS, ALLOW_ECCO without
    ECCO_CTRL_DEPRECATED): with useCTRL, tmpUE = tmpVN = 0 plus the xx_tauu/xx_tauv gentim2d controls (none are
    configured; ExfFullParams requires every forcing control to be zero), exchanged, rotated to tmpUX/tmpVY
    (ROTATE_UV2EN_RL of zeros) and added to ustress/vstress on the whole array: ustress + (+-0) = ustress by value
    (only the sign of a zero can change), so nothing is computed here."""
    return exf


def exf_mapfields(p, g, ex, exf, ff, myTime):
    """exf_mapfields.F:89-373 (full build: stress on the A grid, temp_EvPrRn unset). ff holds the FFIELDS arrays
    before the call (fu/fv keep their i = 1-OLx / j = 1-OLy lines, which the loops do not write). The SST/SSS maps
    (:307-321, climsst/climsss for FORCING_SURF_RELAX, off) are not ported. Returns (exf, ff)."""
    L = g.layout
    exf, ff = dict(exf), dict(ff)
    ks = 0                                                                              # :90 ks = 1
    Qnet = p.outscal_hflux * exf["hflux"]                                               # :109-113
    Qnet = Qnet - p.outscal_hflux * (p.hflux_remo_intercept + p.hflux_remo_slope * (myTime - p.startTime))  # :114-122
    EmPmR = p.outscal_sflux * exf["sflux"] * p.rhoConstFresh                            # :125-130
    EmPmR = EmPmR - p.rhoConstFresh * p.outscal_sflux * (p.sflux_remo_intercept
                                                         + p.sflux_remo_slope * (myTime - p.startTime))  # :131-139
    u = exf["ustress"]
    u = jnp.where(u > p.windstressmax, p.windstressmax, u)                              # :225-232
    u = jnp.where(u < -p.windstressmax, -p.windstressmax, u)                            # :236-242
    Iu, Ium1 = L.is_(1 - L.OLx + 1, L.sNx + L.OLx), L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    fu = ff["fu"].at[:, :, Iu].set(p.outscal_ustress * (u[:, :, Iu] + u[:, :, Ium1]) * EXF_HALF
                                   * g.maskW[:, ks, :, Iu])                             # :250-257
    v = exf["vstress"]
    v = jnp.where(v > p.windstressmax, p.windstressmax, v)                              # :263-270
    v = jnp.where(v < -p.windstressmax, -p.windstressmax, v)                            # :274-280
    Jv, Jvm1 = L.js(1 - L.OLy + 1, L.sNy + L.OLy), L.js(1 - L.OLy, L.sNy + L.OLy - 1)
    fv = ff["fv"].at[:, Jv, :].set(p.outscal_vstress * (v[:, Jv, :] + v[:, Jvm1, :]) * EXF_HALF
                                   * g.maskS[:, ks, Jv, :])                             # :288-295
    Qsw = p.outscal_swflux * exf["swflux"]                                              # :300-304
    pLoad = p.outscal_apressure * exf["apressure"]                                      # :324-328
    saltFlux = exf["saltflx"]                                                           # :332-336
    Qnet = ex.exch_xy(Qnet)                                                             # :354
    EmPmR = ex.exch_xy(EmPmR)                                                           # :355
    fu, fv = ex.exch_uv_xy(fu, fv, True)                                                # :356
    Qsw = ex.exch_xy(Qsw)                                                               # :360
    pLoad = ex.exch_xy(pLoad)                                                           # :369
    exf["ustress"], exf["vstress"] = u, v
    ff.update(Qnet=Qnet, EmPmR=EmPmR, fu=fu, fv=fv, Qsw=Qsw, pLoad=pLoad, saltFlux=saltFlux)
    return exf, ff


def exf_getforcing(p, g, ex, exf, ff, bufs, facs, theta, myTime, zt, zs, libm=DEVICE_LIBM, return_stages=False):
    """exf_getforcing.F:149-299 (full build), called from LOAD_FIELDS_DRIVER (load_fields_driver.F:144).

    exf: EXF_FIELDS arrays at the start of the step (exf_init_varia at the model start); ff: FFIELDS arrays (fu, fv
    at least); bufs/facs from ExfFullRecordLoader.load; theta: [T, Nr, ny, nx] at the start of the step; zt from
    zenith_time(p, myTime, myIter); zs from zenith_static(p, g). Returns (exf, ff) after EXF_MAPFIELDS; with
    return_stages=True also a dict of the EXF arrays after each dump stage X01, X03, X04, X05, X06, X07 (oracle)."""
    stages = {}
    # EXF_GETCLIM (:150): climsst/climsss files ' ' -> constant fields, read only by FORCING_SURF_RELAX (off)
    exf = exf_getffields(p, g, exf, bufs, facs, myTime)                                # :161
    # :162-166: stressIsOnCgrid = F -> no exchange here
    stages["X01"] = exf
    exf = exf_radiation(p, g, exf, theta, zt, zs, libm)                                 # :176
    stages["X03"] = exf
    exf = exf_wind(p, g, exf)                                                           # :187
    stages["X04"] = exf
    exf = exf_bulkformulae(p, g, exf, theta, libm)                                      # :199
    stages["X05"] = exf
    exf = exf_hflux_sflux(p, g, exf)                                                    # :203-237
    exf = dict(exf)
    exf["ustress"], exf["vstress"] = ex.exch_uv_agrid(exf["ustress"], exf["vstress"], True)   # :244-248
    stages["X06"] = exf
    exf = exf_getsurfacefluxes(p, exf)                                                  # :260
    stages["X07"] = exf
    # useExfCheckRange = F (:262-265); SHORTWAVE_HEATING: hflux = hflux + swflux on the whole array (:280-288)
    exf = dict(exf)
    exf["hflux"] = exf["hflux"] + exf["swflux"]
    # EXF_DIAGNOSTICS_FILL, EXF_MONITOR (:293-296): output only
    exf, ff = exf_mapfields(p, g, ex, exf, ff, myTime)                                 # :299
    if return_stages:
        return exf, ff, stages
    return exf, ff
