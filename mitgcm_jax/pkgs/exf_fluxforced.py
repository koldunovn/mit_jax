"""pkg/exf in the ECCO v4r4 flux-forced configuration (plan Task 9): LOAD_FIELDS_DRIVER -> EXF_GETFORCING.

Literal port of what the flux-forced build executes. Sources: c66g `pkg/exf`, `pkg/cal`, overridden by
`ECCO-v4-Configurations/ECCOv4 Release 4/flux-forced/code/` (cited `ff/<file>`): exf_getffields.F,
exf_mapfields.F, exf_init_fixed.F, exf_init_varia.F, exf_readparms.F, EXF_OPTIONS.h, EXF_PARAM.h, EXF_FIELDS.h.
CPP options of the build (ff/EXF_OPTIONS.h, ff/CPP_OPTIONS.h): ALLOW_ATM_TEMP undef (:168, no bulk formulae: fluxes are
read), ALLOW_ATM_WIND (:169), ALLOW_DOWNWARD_RADIATION (:170), ALLOW_RUNOFF (:184), ALLOW_SALTFLX (:186),
READIN_SALT_PLUME_FLUX (:189), USE_EXF_INTERPOLATION (:218, but every interpMethod = 0 in data.exf), ATMOSPHERIC_LOADING
(ff/CPP_OPTIONS.h:60), SHORTWAVE_HEATING (:23).

Split host / device:
  - host (numpy, plain Python): the calendar (pkg/cal) and the record/weight logic of EXF_GetFFieldRec, and the
    two-record buffers fld0/fld1 of EXF_SET_FLD (`ExfRecordLoader`: reads the 6-hourly yearly files, applies
    EXF_FILTER_RL, swaps at record changes, logs every record it loads);
  - device (JAX, pure, jit-able): `exf_getforcing(p, g, ex, exf, ff, bufs, facs, myTime)` = the time interpolation
    of EXF_SET_FLD, the masks/exchanges of EXF_GETFORCING and EXF_MAPFIELDS. Gradients flow to the record buffers.
    ExfParams is a params_pytree: pass it as a jit argument (its floats are traced, never compile-time constants).

Not ported (dead in this configuration, nothing downstream reads them): EXF_WIND (cw, sw, sh, wStress, wspeed; used
only by bulk formulae / seaice), EXF_GETCLIM and the SST/SSS maps of EXF_MAPFIELDS (only FORCING_SURF_RELAX reads
them and climsst/sssTauRelax = 0), EXF_CHECK_RANGE / EXF_MONITOR / EXF_DIAGNOSTICS_FILL (diagnostics),
EXF_ADJOINT_SNAPSHOTS (empty in forward mode). Every namelist setting that would activate one of them raises
NotImplementedError in `ExfParams.from_namelists`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.params_io import params_pytree

log = logging.getLogger(__name__)

UNSET_RL = 1.234567e5  # EEPARAMS.h:90 UNSET_RL = 1.234567D5
EXF_ONE = 1.0          # EXF_CONSTANTS.h:34 exf_one = 1.0 _d 0

# =====================================================================================================================
# pkg/cal (host side; integer calendar arithmetic, Fortran integer semantics)
# =====================================================================================================================
HOURS_PER_DAY = 24          # cal_set.F:94
MINUTES_PER_HOUR = 60       # cal_set.F:95
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY   # cal_set.F:96
SECONDS_PER_MINUTE = 60     # cal_set.F:97
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR   # cal_set.F:98
SECONDS_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_DAY     # cal_set.F:99
N_MONTH_YEAR = 12           # cal.h:52


def _idiv(a, b):
    """Fortran INTEGER a/b (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _imod(a, b):
    """Fortran MOD(a,b) for INTEGERs (result has the sign of a)."""
    return a - _idiv(a, b) * b


def _fint(x):
    """Fortran INT() of a real: truncation toward zero."""
    return int(x)


def _r4(x):
    """A default-REAL (REAL*4) value, as Fortran holds a literal such as `10000.` or FLOAT(n)."""
    return np.float32(x)


@dataclass(frozen=True)
class Calendar:
    """cal.h common blocks after CAL_SET (Gregorian calendar only, the V4r4 data.cal)."""
    refDate: tuple
    nDaysNoLeap: int
    nDaysLeap: int
    nMaxDayMonth: int
    nDayMonth: tuple          # nDayMonth[month-1][leap-1]
    modelStart: float
    modelStep: float
    modelIter0: int
    modelBaseDate: tuple
    modelStartDate: tuple
    startdate_1: int
    startdate_2: int

    @classmethod
    def from_namelists(cls, nml):
        """CAL_READPARMS + CAL_SET(startTime, endTime, deltaTclock, nIter0, ...) (cal_init_fixed.F:37)."""
        cal_name = nml.get("data.cal", "cal_nml", "TheCalendar", default=" ")        # cal_readparms.F:86
        if cal_name.strip().lower() != "gregorian":
            raise NotImplementedError(f"TheCalendar={cal_name!r}: only 'gregorian' is ported (cal_set.F:102)")
        sd1 = nml.get("data.cal", "cal_nml", "startDate_1", default=0)                # cal_readparms.F:87
        sd2 = nml.get("data.cal", "cal_nml", "startDate_2", default=0)                # cal_readparms.F:88
        startTime, deltaTClock, nIter0 = model_clock(nml)
        # cal_set.F:122-151 (Gregorian): refDate, days per year, days per month (the "2773" bit pattern)
        k = 2773
        ndm = []
        for _ in range(N_MONTH_YEAR):
            j = _imod(k, 2)
            k = _idiv(k - j, 2)
            ndm.append([30 + j, 30 + j])
        ndm[1] = [28, 29]                                                              # cal_set.F:150-151
        base = cls(refDate=(15821015, 0, 1, 1), nDaysNoLeap=365, nDaysLeap=366, nMaxDayMonth=31,
                   nDayMonth=tuple(tuple(r) for r in ndm), modelStart=float(startTime), modelStep=0.0,
                   modelIter0=nIter0, modelBaseDate=(), modelStartDate=(), startdate_1=sd1, startdate_2=sd2)
        # cal_set.F:195-219: modelStep = FLOAT(NINT(modelStep)) (after the integer-step check)
        modelStep = float(deltaTClock)
        if modelStep <= 0.0 or modelStep < 1.0 or abs(modelStep - round(modelStep)) > 0.000001:
            raise ValueError("cal_set.F:204-217: invalid model time step for the calendar")
        modelStep = float(_r4(round(modelStep)))
        base = _replace(base, modelStep=modelStep)
        modelBaseDate = cal_fulldate(base, sd1, sd2)                                   # cal_set.F:227
        iterinitime = cal_timeinterval(base, base.modelStart)                          # cal_set.F:238-239
        modelStartDate = cal_addtime(base, modelBaseDate, iterinitime)                 # cal_set.F:240
        return _replace(base, modelBaseDate=tuple(modelBaseDate), modelStartDate=tuple(modelStartDate))


def _replace(obj, **kw):
    d = dict(obj.__dict__)
    d.update(kw)
    return type(obj)(**d)


def model_clock(nml):
    """(startTime, deltaTClock, nIter0) as INI_PARMS sets them (ini_parms.F:962-975)."""
    for key in ("nIter0", "deltaTClock"):
        if not nml.has("data", "parm03", key):
            raise NotImplementedError(f"data PARM03 {key} unset: the default chain of ini_parms.F:884-975 is not "
                                      "ported")
    if nml.has("data", "parm03", "startTime"):
        raise NotImplementedError("startTime set in data: only startTime = baseTime + deltaTClock*nIter0 is ported "
                                  "(ini_parms.F:966-968)")
    nIter0 = nml.get("data", "parm03", "nIter0")
    deltaTClock = float(nml.get("data", "parm03", "deltaTClock"))
    baseTime = float(nml.get("data", "parm03", "baseTime", default=0.0))                # set_defaults.F:299
    startTime = baseTime + deltaTClock * float(nIter0)                                  # ini_parms.F:968
    return float(startTime), float(deltaTClock), int(nIter0)


def model_time(nml, iloop):
    """(myTime, myIter) at the start of loop step iLoop (1-based), ff/forward_step.F:397-398."""
    startTime, deltaTClock, nIter0 = model_clock(nml)
    return startTime + deltaTClock * float(iloop - 1), nIter0 + (iloop - 1)          # ff/forward_step.F:397-398


def cal_isleap(cal, year):
    """cal_isleap.F:53-61 (Gregorian): 1 = no leap year, 2 = leap year."""
    if _imod(year, 4) != 0:
        return 1
    if _imod(year, 100) == 0 and _imod(year, 400) != 0:
        return 1
    return 2


def cal_convdate(cal, date):
    """cal_convdate.F:68-105 -> (yy, mm, dd, ss, lp, wd)."""
    wrong_sign = (date[0] < 0 and date[1] > 0) or (date[0] > 0 and date[1] < 0)
    if wrong_sign:
        raise ValueError(f"cal_ConvDate: wrong sign in date {date} (cal_convdate.F:69-72)")
    if date[0] < 0 or date[1] < 0:
        date_1, date_2, fac = -date[0], -date[1], -1
    else:
        date_1, date_2, fac = date[0], date[1], 1
    if date[3] != -1:                                                                   # cal_convdate.F:85-93
        yy = _idiv(date_1, 10000)
        mm = _imod(_idiv(date_1, 100), 100)
        dd = _imod(date_1, 100)
    else:
        yy, mm, dd = 0, 0, date_1
    ss = (_imod(date_2, 100) + _imod(_idiv(date_2, 100), 100) * SECONDS_PER_MINUTE
          + _idiv(date_2, 10000) * SECONDS_PER_HOUR)                                   # cal_convdate.F:94-96
    return fac * yy, fac * mm, fac * dd, fac * ss, date[2], date[3]                     # cal_convdate.F:99-105


def cal_timepassed(cal, initialdate, finaldate):
    """cal_timepassed.F:85-189: interval finaldate - initialdate as (days, hhmmss, 0, -1)."""
    numdays = [0, 0, 0, -1]
    if not ((initialdate[3] > 0) == (finaldate[3] > 0)):
        raise ValueError("cal_TimePassed: error 501 (cal_timepassed.F:185-187)")
    caldates = initialdate[3] > 0 and finaldate[3] > 0
    nothingtodo = False
    if initialdate[0] == finaldate[0]:
        if initialdate[1] == finaldate[1]:
            nothingtodo = True
        else:
            swap = initialdate[1] > finaldate[1]
    else:
        swap = initialdate[0] > finaldate[0]
    if nothingtodo:
        return numdays
    if swap:
        yi, mi, di, si, li, wi = cal_convdate(cal, finaldate)
        yf, mf, df, sf, lf, wf = cal_convdate(cal, initialdate)
    else:
        yi, mi, di, si, li, wi = cal_convdate(cal, initialdate)
        yf, mf, df, sf, lf, wf = cal_convdate(cal, finaldate)
    spd = SECONDS_PER_DAY
    if not caldates:                                                                    # cal_timepassed.F:122-130
        ndays = df - di
        nsecs = sf - si
        if nsecs < 0:
            nsecs = nsecs + spd
            ndays = ndays - 1
        ndays = ndays + _idiv(nsecs, spd)
        nsecs = _imod(nsecs, spd)
    else:                                                                               # cal_timepassed.F:131-164
        si = si + (di - 1) * spd
        sf = sf + (df - 1) * spd
        cdi = 0
        for imon in range(1, _imod(mi - 1, 12) + 1):
            cdi = cdi + cal.nDayMonth[imon - 1][li - 1]
        csi = si
        cdf = 0
        for imon in range(1, _imod(mf - 1, 12) + 1):
            cdf = cdf + cal.nDayMonth[imon - 1][lf - 1]
        csf = sf
        if yi == yf:
            ndays = (cdf + _idiv(csf, spd)) - (cdi + _idiv(csi, spd))
            nsecs = (csf - _idiv(csf, spd) * spd) - (csi - _idiv(csi, spd) * spd)
            if nsecs < 0:
                nsecs = nsecs + spd
                ndays = ndays - 1
        else:
            ndays = (cal.nDaysNoLeap - 1) + cal_isleap(cal, yi) - cdi - cal.nDayMonth[mi - 1][li - 1]
            for iyr in range(yi + 1, yf):
                ndays = ndays + (cal.nDaysNoLeap - 1) + cal_isleap(cal, iyr)
            ndays = ndays + cdf
            csi = cal.nDayMonth[mi - 1][li - 1] * spd - csi
            nsecs = csi + csf
    numdays[0] = ndays + _idiv(nsecs, spd)                                              # cal_timepassed.F:168-177
    nsecs = _imod(nsecs, spd)
    hhmmss = _idiv(nsecs, SECONDS_PER_MINUTE)
    numdays[1] = (_idiv(hhmmss, MINUTES_PER_HOUR) * 10000 + _imod(hhmmss, MINUTES_PER_HOUR) * 100
                  + _imod(nsecs, SECONDS_PER_MINUTE))
    if swap:
        numdays[0] = -numdays[0]
        numdays[1] = -numdays[1]
    return numdays


def cal_toseconds(cal, date):
    """cal_toseconds.F:69-99: a time interval (days, hhmmss, 0, -1) in seconds (_RL)."""
    check_sign = 1
    if (date[0] < 0 and date[1] > 0) or (date[0] > 0 and date[1] < 0):
        check_sign = -1
    if not (date[3] == -1 and date[2] == 0 and check_sign >= 0):
        raise ValueError(f"cal_ToSeconds: error 1001 for date {date} (cal_toseconds.F:95-97)")
    if date[0] < 0 or date[1] < 0:
        ndays, hhmmss, fac = float(-date[0]), -date[1], -1.0
    else:
        ndays, hhmmss, fac = float(date[0]), date[1], 1.0
    nsecs = (ndays * SECONDS_PER_DAY + _idiv(hhmmss, 10000) * SECONDS_PER_HOUR
             + _imod(_idiv(hhmmss, 100), 100) * SECONDS_PER_MINUTE + _imod(hhmmss, 100))  # cal_toseconds.F:88-91
    return fac * nsecs


def cal_timeinterval(cal, timeint):
    """cal_timeinterval.F:55-104, timeunit 'secs': seconds (_RL) -> interval (days, hhmmss, 0, -1)."""
    fac = -1 if timeint < 0 else 1
    date = [0, 0, 0, -1]                                                                # cal_timeinterval.F:58-59
    date[0] = _fint(timeint / float(_r4(SECONDS_PER_DAY)))                              # cal_timeinterval.F:71
    tmp1 = float(date[0])
    tmp2 = float(SECONDS_PER_DAY)
    nsecs = _fint(timeint - tmp1 * tmp2)                                                # cal_timeinterval.F:74
    hhmmss = _idiv(nsecs, SECONDS_PER_MINUTE)
    date[1] = (_idiv(hhmmss, MINUTES_PER_HOUR) * 10000
               + (_imod(fac * hhmmss, MINUTES_PER_HOUR) * 100 + _imod(fac * nsecs, SECONDS_PER_MINUTE)) * fac)
    return date


def cal_addtime(cal, date, interval):
    """cal_addtime.F:81-236: date + interval (Gregorian)."""
    if interval[3] != -1:
        raise ValueError("cal_AddTime: error 601 (cal_addtime.F:81-85)")
    spd = SECONDS_PER_DAY
    date_1, date_2, fac = 0, 0, 1
    if date[3] == -1:                                                                   # cal_addtime.F:91-111
        if date[0] >= 0:
            date_1, date_2, intv_1, intv_2 = date[0], date[1], interval[0], interval[1]
        elif interval[0] < 0:
            date_1, date_2, intv_1, intv_2, fac = -date[0], -date[1], -interval[0], -interval[1], -1
        else:
            date_1, date_2, intv_1, intv_2, fac = interval[0], interval[1], date[0], date[1], 1
    else:                                                                               # cal_addtime.F:112-121
        if interval[0] >= 0:
            intv_1, intv_2 = interval[0], interval[1]
        else:
            intv_1, intv_2, fac = -interval[0], -interval[1], -1
    intsecs = fac * (_idiv(intv_2, 10000) * SECONDS_PER_HOUR
                     + (_imod(_idiv(intv_2, 100), 100) * SECONDS_PER_MINUTE + _imod(intv_2, 100)))  # :123-125
    if date[3] == -1:                                                                   # cal_addtime.F:127-143
        datesecs = (_idiv(date_2, 10000) * SECONDS_PER_HOUR + _imod(_idiv(date_2, 100), 100) * SECONDS_PER_MINUTE
                    + _imod(date_2, 100))
        date_1 = date_1 + intv_1
        nsecs = datesecs + intsecs
        if date_1 > 0 and nsecs < 0:
            date_1 = date_1 - 1
            nsecs = nsecs + spd
        nsecs = fac * nsecs
        yi, mi, di, li, wi = 0, 0, fac * date_1, 0, -1
    else:
        yi, mi, di, si, li, wi = cal_convdate(cal, date)                                # cal_addtime.F:145
        if interval[0] >= 0 and interval[1] >= 0:                                       # cal_addtime.F:146-205
            nsecs = si + intsecs
            ndays = interval[0] + _idiv(nsecs, spd)
            nsecs = _imod(nsecs, spd)
            ndays_left = ndays
            if mi == 2 and di == 29 and ndays_left > 1:                                 # :170-176 (Gregorian)
                mi = 3
                di = 1
                ndays_left = ndays_left - 1

            def _diy(yi, mi):                                                           # :179-182, :186-189
                if (mi > 2 and cal_isleap(cal, yi + 1) == 2) or (mi <= 2 and cal_isleap(cal, yi) == 2):
                    return cal.nDaysLeap
                return cal.nDaysNoLeap
            days_in_year = _diy(yi, mi)
            while ndays_left >= days_in_year:                                           # :183-190
                ndays_left = ndays_left - days_in_year
                yi = yi + 1
                days_in_year = _diy(yi, mi)
            li = cal_isleap(cal, yi)                                                    # :191
            for _ in range(ndays_left):                                                 # :194-204
                di = di + 1
                if di > cal.nDayMonth[mi - 1][li - 1]:
                    di = 1
                    mi = mi + 1
                switch = _idiv(mi - 1, N_MONTH_YEAR)
                yi = yi + switch
                mi = _imod(mi - 1, N_MONTH_YEAR) + 1
                if switch == 1:
                    li = cal_isleap(cal, yi)
            wi = _imod(wi + ndays - 1, 7) + 1                                           # :205
        else:                                                                           # cal_addtime.F:207-225
            nsecs = si + intsecs
            if nsecs >= 0:
                ndayssub = intv_1
            else:
                nsecs = nsecs + spd
                ndayssub = intv_1 + 1
            for _ in range(ndayssub):
                di = di - 1
                if di == 0:
                    mi = _imod(mi + 10, N_MONTH_YEAR) + 1
                    switch = _idiv(mi, N_MONTH_YEAR)
                    yi = yi - switch
                    if switch == 1:
                        li = cal_isleap(cal, yi)
                    di = cal.nDayMonth[mi - 1][li - 1]
            wi = _imod(wi + 6 - _imod(ndayssub, 7), 7) + 1
    hhmmss = _idiv(nsecs, SECONDS_PER_MINUTE)                                           # cal_addtime.F:230-236
    added2 = (_idiv(hhmmss, MINUTES_PER_HOUR) * 10000
              + (_imod(fac * hhmmss, MINUTES_PER_HOUR) * 100 + _imod(fac * nsecs, SECONDS_PER_MINUTE)) * fac)
    return [yi * 10000 + mi * 100 + di, added2, li, wi]


def cal_checkdate(cal, date):
    """cal_checkdate.F:55-117, the checks that make CAL_FULLDATE stop (valid = .FALSE.)."""
    if (date[0] < 0 and date[1] > 0) or (date[0] > 0 and date[1] < 0):                 # cal_checkdate.F:59-66
        return False
    if date[3] <= 0:
        return True
    yy, mm, dd, nsecs, lp, wd = cal_convdate(cal, date)                                 # cal_checkdate.F:90
    if mm == 0 or abs(mm) > N_MONTH_YEAR:                                               # :91-96
        return False
    if lp not in (1, 2):                                                                # :101-105
        return False
    return True


def cal_fulldate(cal, yymmdd, hhmmss):
    """cal_fulldate.F:56-98: (yymmdd, hhmmss) -> (yymmdd, hhmmss, leap, weekday)."""
    date = [yymmdd, hhmmss, 1, 1]
    if not cal_checkdate(cal, date):
        raise ValueError(f"CAL_FULLDATE: fatal error from cal_CheckDate for {yymmdd} {hhmmss}")
    theyear = _idiv(yymmdd, 10000)                                                      # cal_fulldate.F:82
    date[2] = cal_isleap(cal, theyear)                                                  # cal_fulldate.F:83
    numberOfDays = cal_timepassed(cal, list(cal.refDate), date)                        # cal_fulldate.F:86
    if numberOfDays[0] < 0:
        raise ValueError("CAL_FULLDATE: date before refDate (cal_fulldate.F:87-96)")
    date[3] = _imod(numberOfDays[0], 7) + 1                                             # cal_fulldate.F:98
    return date


def cal_getdate(cal, myIter, myTime):
    """cal_getdate.F:50-93: model date at (myIter, myTime)."""
    if myIter == -1:
        return [cal.startdate_1, cal.startdate_2, 1, 1]                                 # cal_getdate.F:53-56
    if myTime == cal.modelStart:                                                        # cal_getdate.F:69
        return list(cal.modelStartDate)
    secs = myTime - cal.modelStart                                                      # cal_getdate.F:87
    workdate = cal_timeinterval(cal, secs)                                              # cal_getdate.F:90
    return cal_addtime(cal, list(cal.modelStartDate), workdate)                         # cal_getdate.F:91


# =====================================================================================================================
# EXF parameters (exf_readparms.F, exf_init_fixed.F; flux-forced overrides)
# =====================================================================================================================
# Fields read by EXF_GETFFIELDS in the flux-forced build, in call order (ff/exf_getffields.F:77-473), with the line of
# each field's namelist defaults in ff/exf_readparms.F: (file, period, const, inscal, intercept, slope, StartTime,
# RepCycle, interpMethod, startdate1) and its mask in ff/exf_init_fixed.F.
READ_FIELDS = ("ustress", "vstress", "hflux", "sflux", "swflux", "apressure", "saltflx", "spflx")
_DEFAULT_LINES = {
    "ustress": (626, 469, 470, 689, 471, 472, 655, 969, 910, 467),
    "vstress": (627, 476, 477, 690, 478, 479, 656, 970, 911, 474),
    "hflux": (613, 381, 382, 687, 383, 384, 643, 957, 912, 379),
    "sflux": (621, 416, 417, 688, 418, 419, 651, 965, 913, 414),
    "swflux": (631, 504, 505, 694, 506, 507, 660, 974, 914, 502),
    "apressure": (635, 532, 533, 706, 534, 535, 664, 978, 931, 530),
    "saltflx": (624, 455, 456, 709, 457, 458, 653, 967, 916, 453),
    "spflx": (625, 462, 463, 710, 464, 465, 654, 968, 917, 460),
}
# fields whose EXF_SET_FLD call is compiled but must stay inactive (file = ' '): the paths they would switch on are not
# ported (EXF_WIND/bulk, EXF_RADIATION, runoff energy, sea-ice fraction, climatological relaxation, winds)
INACTIVE_FILES = ("wspeedfile", "swdownfile", "lwdownfile", "runofffile", "runoftempfile", "areamaskfile",
                  "climsstfile", "climsssfile", "climustrfile", "climvstrfile", "uwindfile", "vwindfile",
                  "atempfile", "aqhfile", "precipfile", "snowprecipfile", "evapfile", "lwfluxfile", "hs_file",
                  "hl_file")


@dataclass(frozen=True)
class ExfField:
    """Host-side description of one EXF_SET_FLD field (file, record timing, mask, initial value)."""
    name: str
    file: str
    mask: str            # EXF_FILTER_RL kind: 'c', 'w', 's' or ' '
    period: float
    startTime: float     # seconds from the start of the year of the first record (useExfYearlyFields)
    repCycle: float
    const: float


@params_pytree
@dataclass(frozen=True)
class ExfParams:
    """EXF parameters. `float` fields are pytree leaves (pass the object as a jit argument); the rest is static:
    the host-side field descriptions, the calendar and the switches."""
    fields: tuple                 # ExfField per READ_FIELDS entry, in that order (static)
    cal: Calendar
    useExfYearlyFields: bool
    twoDigitYear: bool
    exf_iprec: int
    stressIsOnCgrid: bool
    windstressmax: float
    outscal_hflux: float
    outscal_sflux: float
    outscal_ustress: float
    outscal_vstress: float
    outscal_swflux: float
    outscal_apressure: float
    rhoConstFresh: float
    startTime: float
    runoffconst: float
    # exf_inscal_<fld>, <fld>_exfremo_intercept, <fld>_exfremo_slope of every READ_FIELDS entry
    inscal_ustress: float
    remo_intercept_ustress: float
    remo_slope_ustress: float
    inscal_vstress: float
    remo_intercept_vstress: float
    remo_slope_vstress: float
    inscal_hflux: float
    remo_intercept_hflux: float
    remo_slope_hflux: float
    inscal_sflux: float
    remo_intercept_sflux: float
    remo_slope_sflux: float
    inscal_swflux: float
    remo_intercept_swflux: float
    remo_slope_swflux: float
    inscal_apressure: float
    remo_intercept_apressure: float
    remo_slope_apressure: float
    inscal_saltflx: float
    remo_intercept_saltflx: float
    remo_slope_saltflx: float
    inscal_spflx: float
    remo_intercept_spflx: float
    remo_slope_spflx: float

    @property
    def field_map(self):
        return {f.name: f for f in self.fields}

    @classmethod
    def from_namelists(cls, nml):
        g1, g2, g3, g4 = "exf_nml_01", "exf_nml_02", "exf_nml_03", "exf_nml_04"
        get = nml.get
        if not get("data.pkg", "packages", "useEXF", default=False):
            raise NotImplementedError("useEXF=F: this module ports the EXF path only")
        if not get("data.pkg", "packages", "useCAL", default=False):
            raise NotImplementedError("useCAL=F: only the calendar branch of EXF_GetFFieldRec is ported "
                                      "(exf_getffieldrec.F:94)")
        if get("data.pkg", "packages", "useCTRL", default=False):
            raise NotImplementedError("useCTRL=T: xx_gentim2d forcing controls (exf_getffields.F:681, "
                                      "exf_getsurfacefluxes.F:106, CTRL_MAP_FORCING) are not ported")
        if get("data.pkg", "packages", "useSEAICE", default=False):
            raise NotImplementedError("useSEAICE=T changes the EXF masks (ff/exf_init_fixed.F:86-115)")
        # ff/exf_readparms.F:328 ALLOW_ATM_WIND => useAtmWind default .TRUE.
        if get("data.exf", g1, "useAtmWind", default=True):
            raise NotImplementedError("useAtmWind=T: EXF_SET_UV for winds / EXF_WIND not ported "
                                      "(ff/exf_getffields.F:124)")
        if get("data.exf", g1, "readStressOnAgrid", default=False) or \
                get("data.exf", g1, "rotateStressOnAgrid", default=False):          # ff/exf_readparms.F:324-325
            raise NotImplementedError("readStressOnAgrid / rotateStressOnAgrid not ported (exf_set_uv.F:555)")
        readStressOnCgrid = get("data.exf", g1, "readStressOnCgrid", default=False)         # ff/exf_readparms.F:326
        if get("data.exf", g1, "exf_yftype", default="RL") != "RL":                          # ff/exf_readparms.F:682
            raise ValueError("exf_yftype must be 'RL' (ff/exf_readparms.F:1049)")
        if get("data.exf", g2, "climsstTauRelax", default=0.0) != 0.0 or \
                get("data.exf", g2, "climsssTauRelax", default=0.0) != 0.0:           # ff/exf_readparms.F:548,556
            raise NotImplementedError("climsst/sssTauRelax != 0: surface relaxation (FORCING_SURF_RELAX) not ported")
        for key in INACTIVE_FILES:
            if get("data.exf", g2, key, default=" ").strip():
                raise NotImplementedError(f"data.exf {key} is set: the EXF path it activates is not ported")
        for key in ("repeatPeriod",):
            if get("data.exf", g1, key, default=0.0) != 0.0:                                # ff/exf_readparms.F:604
                raise NotImplementedError("repeatPeriod != 0 not ported")
        useYearly = get("data.exf", g1, "useExfYearlyFields", default=False)               # ff/exf_readparms.F:683
        if not useYearly:
            raise NotImplementedError("useExfYearlyFields=F: that branch of EXF_GetFFieldRec is not ported "
                                      "(exf_getffieldrec.F:115)")
        cal = Calendar.from_namelists(nml)
        # masks: ff/exf_init_fixed.F:53-79 (useSEAICE=F)
        stressIsOnCgrid = readStressOnCgrid                                                  # ff/exf_readparms.F:1055
        masks = {"hflux": "c", "sflux": "c", "saltflx": "c", "spflx": "c", "swflux": "c", "apressure": "c",
                 "ustress": "w" if stressIsOnCgrid else "c", "vstress": "s" if stressIsOnCgrid else "c"}
        fields, scal = [], {}
        for name in READ_FIELDS:
            # namelist value, else the default of ff/exf_readparms.F at the lines listed in _DEFAULT_LINES[name]:
            # file, period, const, inscal, intercept, slope, (StartTime, RepCycle,) interpMethod, startdate1/2
            fname = get("data.exf", g2, f"{name}file", default=" ").strip()
            period = float(get("data.exf", g2, f"{name}period", default=0.0))
            const = float(get("data.exf", g3, f"{name}const", default=0.0))
            inscal = float(get("data.exf", g3, f"exf_inscal_{name}", default=1.0))
            icpt = float(get("data.exf", g3, f"{name}_exfremo_intercept", default=0.0))
            slope = float(get("data.exf", g3, f"{name}_exfremo_slope", default=0.0))
            interp = get("data.exf", g4, f"{name}_interpMethod",
                         default=12 if name == "ustress" else 22 if name == "vstress" else 1)
            sd1 = get("data.exf", g2, f"{name}startdate1", default=0)
            sd2 = get("data.exf", g2, f"{name}startdate2", default=0)
            if nml.has("data.exf", g2, f"{name}StartTime"):
                raise ValueError(f"{name}StartTime cannot be set with useCAL (exf_getffield_start.F:68-80)")
            if not fname:
                raise NotImplementedError(f"{name}file = ' ': a flux-forced run reads every one of {READ_FIELDS}")
            if interp >= 1:
                raise NotImplementedError(f"{name}_interpMethod={interp}: EXF_INTERP not ported (exf_set_fld.F:189)")
            if period <= 0.0:
                raise NotImplementedError(f"{name}period={period}: only period > 0 is ported "
                                          "(exf_set_fld.F:128-151, exf_getffieldrec.F:100)")
            # ff/exf_init_fixed.F:164-381 -> EXF_GETFFIELD_START (exf_getffield_start.F:66-106)
            start = exf_getffield_start(cal, useYearly, period, sd1, sd2)
            fields.append(ExfField(name=name, file=fname, mask=masks[name], period=period, startTime=start,
                                   repCycle=float(get("data.exf", g1, "repeatPeriod", default=0.0)), const=const))
            scal.update({f"inscal_{name}": inscal, f"remo_intercept_{name}": icpt, f"remo_slope_{name}": slope})
        rhoNil = get("data", "parm01", "rhoNil", default=999.8)                             # set_defaults.F:106
        rhoConst = get("data", "parm01", "rhoConst", default=rhoNil)                        # ini_parms.F:445
        rhoConstFresh = get("data", "parm01", "rhoConstFresh", default=rhoConst)            # ini_parms.F:446
        startTime, _, _ = model_clock(nml)
        return cls(fields=tuple(fields), cal=cal, useExfYearlyFields=bool(useYearly),
                   twoDigitYear=bool(get("data.exf", g1, "twoDigitYear", default=False)),   # ff/exf_readparms.F:684
                   exf_iprec=int(get("data.exf", g1, "exf_iprec", default=32)),             # ff/exf_readparms.F:680
                   windstressmax=float(get("data.exf", g1, "windstressmax", default=2.0)),  # ff/exf_readparms.F:605
                   stressIsOnCgrid=bool(stressIsOnCgrid),
                   outscal_hflux=float(get("data.exf", g3, "exf_outscal_hflux", default=1.0)),     # :720
                   outscal_sflux=float(get("data.exf", g3, "exf_outscal_sflux", default=1.0)),     # :721
                   outscal_ustress=float(get("data.exf", g3, "exf_outscal_ustress", default=1.0)), # :722
                   outscal_vstress=float(get("data.exf", g3, "exf_outscal_vstress", default=1.0)), # :723
                   outscal_swflux=float(get("data.exf", g3, "exf_outscal_swflux", default=1.0)),   # :724
                   outscal_apressure=float(get("data.exf", g3, "exf_outscal_apressure", default=1.0)),  # :727
                   rhoConstFresh=float(rhoConstFresh), startTime=startTime,
                   runoffconst=float(get("data.exf", g3, "runoffconst", default=0.0)),     # ff/exf_readparms.F:445
                   **scal)


def exf_getffield_start(cal, useYearlyFields, fld_period, fld_startdate1, fld_startdate2):
    """exf_getffield_start.F:66-106 (useCAL, period > 0): start time of the first record in seconds."""
    fld_start_time = 0.0                                                                  # exf_getffield_start.F:67
    if fld_period > 0.0:                                                                  # exf_getffield_start.F:83
        date_array = cal_fulldate(cal, fld_startdate1, fld_startdate2)                   # :85-86
        if useYearlyFields:
            yearStartDate = [_fint(np.float32(date_array[0]) / _r4(10000.0)) * 10000 + 101, 0,
                             date_array[2], date_array[3]]                                # :88-91 (REAL*4 10000.)
            difftime = cal_timepassed(cal, yearStartDate, date_array)                    # :92-93
            fld_start_time = cal_toseconds(cal, difftime)                                 # :94
        else:
            raise NotImplementedError("useExfYearlyFields=F start time (exf_getffield_start.F:96-105)")
    return fld_start_time


# =====================================================================================================================
# EXF_GetFFieldRec and file names (host side)
# =====================================================================================================================
@dataclass(frozen=True)
class FieldRec:
    fac: float
    first: bool
    changed: bool
    count0: int
    count1: int
    year0: int
    year1: int


def exf_getffieldrec(cal, fldStartTime, fldPeriod, usefldyearlyfields, myTime, myIter):
    """exf_getffieldrec.F:94-201 (useCAL branch; fldPeriod > 0, yearly fields)."""
    first = (myTime - cal.modelStart) < float(_r4(0.5)) * cal.modelStep                # exf_getffieldrec.F:97
    changed = False                                                                      # exf_getffieldrec.F:98
    if fldPeriod == 0.0 or not usefldyearlyfields:
        raise NotImplementedError("exf_getffieldrec.F:100-153 (period 0 / non-yearly) not ported")
    mydate = cal_getdate(cal, myIter, myTime)                                            # exf_getffieldrec.F:158
    year0 = _fint(np.float32(mydate[0]) / _r4(10000.0))                                  # :159 (REAL*4 10000.)
    yearStartDate = [year0 * 10000 + 101, 0, mydate[2], mydate[3]]                       # :160-163
    difftime = cal_timepassed(cal, yearStartDate, mydate)                                # :164
    myDateSeconds = cal_toseconds(cal, difftime)                                         # :165
    if myDateSeconds < fldStartTime:                                                     # :168
        year0 = year0 - 1
    secondsInYear = float(cal.nDaysNoLeap * SECONDS_PER_DAY)                             # :171
    if cal_isleap(cal, year0) == 2:                                                      # :172-173
        secondsInYear = float(cal.nDaysLeap * SECONDS_PER_DAY)
    if myDateSeconds < fldStartTime:                                                     # :176-177
        myDateSeconds = myDateSeconds + secondsInYear
    fldsectot = myDateSeconds - fldStartTime                                             # :178
    count0 = _fint((fldsectot + 0.5) / fldPeriod) + 1                                   # :179
    year1 = year0                                                                        # :182
    count1 = count0 + 1                                                                  # :183
    if (fldStartTime + count0 * fldPeriod) >= secondsInYear:                            # :184-187
        year1 = year0 + 1
        count1 = 1
    fldsecs = float(np.fmod(fldsectot, fldPeriod))                                       # :190 MOD(fldsectot,fldPeriod)
    fac = float(_r4(1.0)) - fldsecs / fldPeriod                                          # :191
    if year0 != year1:                                                                   # :192-193
        fac = float(_r4(1.0)) - fldsecs / (secondsInYear - (count0 - 1) * fldPeriod)
    if fldsecs - cal.modelStep < 0.0:                                                    # :199
        changed = True
    return FieldRec(fac=fac, first=bool(first), changed=changed, count0=count0, count1=count1,
                    year0=year0, year1=year1)


def exf_getyearlyfieldname(useYearlyFields, twoDigitYear, genperiod, year, genfile):
    """exf_getyearlyfieldname.F:51-63."""
    if useYearlyFields and genperiod > 0:
        if twoDigitYear:
            yearLoc = year - 2000 if year >= 2000 else year - 1900
            return f"{genfile}{yearLoc:02d}"
        return f"{genfile}_{year:04d}"
    return genfile


# =====================================================================================================================
# Host-side record buffers (EXF_SET_FLD fld0/fld1)
# =====================================================================================================================
def read_rec_2d(path, prec, rec, layout):
    """READ_REC_3D_RL(fName, prec, 1, fld, rec, ...) of a global compact LLC file without .meta: record `rec`
    (1-based) as [T, ny, nx] float64 with zero halos (only the interior is used by EXF_SET_FLD)."""
    L = layout
    nx, ny = 90, 1170
    dt = {32: ">f4", 64: ">f8"}[prec]
    n = nx * ny
    a = np.fromfile(path, dtype=dt, count=n, offset=(rec - 1) * n * np.dtype(dt).itemsize)
    if a.size != n:
        raise ValueError(f"{path}: record {rec} is beyond the end of the file")
    t = compact_to_tiles(a.astype(np.float64).reshape(ny, nx))
    out = np.zeros(L.shape2d)
    out[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = t
    return out


def exf_filter_rl(arr, ckind, g):
    """exf_filter_rl.F:41-89: zero the interior points where the level-1 mask of kind ckind is 0."""
    if ckind == " ":
        return arr
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    ks = 0                                                                               # exf_filter_rl.F:43 ks = 1
    m = {"c": g.maskC, "w": g.maskW, "s": g.maskS}[ckind][:, ks]
    return arr.at[:, J, I].set(jnp.where(m[:, J, I] == 0.0, 0.0, arr[:, J, I]))


class ExfRecordLoader:
    """The fld0/fld1 buffers of every EXF_SET_FLD call, with the record logic of exf_set_fld.F:116-279.

    `load(myTime, myIter)` must be called once per time step, in order, starting at the model start (first=.TRUE.);
    it returns (bufs, facs, recs): bufs[name] = (fld0, fld1) numpy [T, ny, nx] (masked, as EXF_SET_FLD keeps them),
    facs[name] = interpolation weight of fld0, recs[name] = FieldRec. Every record read is logged
    ('EXF_SET_FLD: field "hflux", it=1, loading rec=2 from file TFLUX_6hourlyavg_1992').
    """

    def __init__(self, p, g, rundir):
        self.p = p
        self.g = g
        self.dir = Path(rundir)
        L = g.layout
        # exf_init_fld.F: fld0 = fld1 = fldConst everywhere (ff/exf_init_varia.F)
        self.buf = {f.name: [np.full(L.shape2d, f.const), np.full(L.shape2d, f.const)] for f in p.fields}
        self.started = False
        self.loaded = []   # (myIter, name, rec, file) of every record read

    def _read(self, name, fname, rec, myIter):
        f = self.p.field_map[name]
        log.info('EXF_SET_FLD: field "%s", it=%d, loading rec=%d from file "%s"', name, myIter, rec, fname)
        self.loaded.append((myIter, name, rec, fname))
        a = read_rec_2d(self.dir / fname, self.p.exf_iprec, rec, self.g.layout)
        return np.asarray(exf_filter_rl(jnp.asarray(a), f.mask, self.g))                # exf_set_fld.F:219,276

    def load(self, myTime, myIter):
        p, L = self.p, self.g.layout
        J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
        bufs, facs, recs = {}, {}, {}
        for f in p.fields:
            name = f.name
            r = exf_getffieldrec(p.cal, f.startTime, f.period, p.useExfYearlyFields, myTime, myIter)
            fld0, fld1 = self.buf[name]
            if not r.first and not self.started:
                raise RuntimeError("ExfRecordLoader.load must start at the model start time (first=.TRUE.)")
            if r.first:                                                                  # exf_set_fld.F:167-222
                fname = exf_getyearlyfieldname(p.useExfYearlyFields, p.twoDigitYear, f.period, r.year0, f.file)
                fld1 = fld1.copy()
                fld1[:, J, I] = self._read(name, fname, r.count0, myIter)[:, J, I]
            if r.first or r.changed:                                                     # exf_set_fld.F:224-279
                fld0 = fld0.copy()                                                       # exf_swapffields.F:58-67
                fld0[:, J, I] = fld1[:, J, I]
                fld1 = fld1.copy()
                fld1[:, J, I] = 0.0
                fname = exf_getyearlyfieldname(p.useExfYearlyFields, p.twoDigitYear, f.period, r.year1, f.file)
                fld1[:, J, I] = self._read(name, fname, r.count1, myIter)[:, J, I]
            self.buf[name] = [fld0, fld1]
            bufs[name] = (fld0, fld1)
            facs[name] = r.fac
            recs[name] = r
        self.started = True
        return bufs, facs, recs


# =====================================================================================================================
# Device side (JAX): EXF_GETFORCING
# =====================================================================================================================
def exf_init_varia(p, layout):
    """ff/exf_init_varia.F + exf_init_fld.F: every EXF field array = fldConst (halos included; never rewritten)."""
    exf = {f.name: jnp.full(layout.shape2d, f.const) for f in p.fields}
    exf["runoff"] = jnp.full(layout.shape2d, p.runoffconst)                             # ff/exf_init_varia.F:342-352
    return exf


def exf_set_fld(fld_inScale, fldRemove_intercept, fldRemove_slope, fldArr, fld0, fld1, fac, myTime, startTime,
                layout):
    """exf_set_fld.F:281-296: interior time interpolation, scaling and trend removal (fldFile set, period > 0)."""
    L = layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    v = fld_inScale * (fac * fld0[:, J, I] + (EXF_ONE - fac) * fld1[:, J, I])           # exf_set_fld.F:287-289
    v = v - fld_inScale * (fldRemove_intercept + fldRemove_slope * (myTime - startTime))  # exf_set_fld.F:290-292
    return fldArr.at[:, J, I].set(v)


def exf_getffields(p, exf, bufs, facs, myTime, layout):
    """ff/exf_getffields.F:76-473: EXF_SET_UV(ustress, vstress) (-> two EXF_SET_FLD, exf_set_uv.F:532-552) and
    EXF_SET_FLD for hflux, sflux, swflux, apressure, saltflx, spflx. The wspeed/swdown/lwdown/runoff calls have
    file = ' ' (checked in ExfParams) and do nothing (exf_set_fld.F:116)."""
    exf = dict(exf)
    for name in READ_FIELDS:
        fld0, fld1 = bufs[name]
        exf[name] = exf_set_fld(getattr(p, f"inscal_{name}"), getattr(p, f"remo_intercept_{name}"),
                                getattr(p, f"remo_slope_{name}"), exf[name], fld0, fld1, facs[name], myTime,
                                p.startTime, layout)
    return exf


def exf_mapfields(p, g, ex, exf, ff, myTime):
    """ff/exf_mapfields.F:93-388 (ALLOW_ATM_TEMP undef). ff holds the FFIELDS arrays before the call (fu, fv keep
    their values at i = 1-OLx / j = 1-OLy, which the loops do not write). Returns (exf, ff) updated."""
    L = g.layout
    exf, ff = dict(exf), dict(ff)
    # Heat flux (ff/exf_mapfields.F:113-126; hfluxfile set)
    Qnet = p.outscal_hflux * exf["hflux"]
    # Freshwater flux (:129-143; sfluxfile set)
    EmPmR = p.outscal_sflux * exf["sflux"] * p.rhoConstFresh
    # Zonal wind stress clipping (:229-246), whole array
    u = exf["ustress"]
    u = jnp.where(u > p.windstressmax, p.windstressmax, u)
    u = jnp.where(u < -p.windstressmax, -p.windstressmax, u)
    # :247-252 stressIsOnCgrid: DO j = jmin,jmax; DO i = imin+1,imax
    Iu = L.is_(1 - L.OLx + 1, L.sNx + L.OLx)
    fu = ff["fu"].at[:, :, Iu].set(p.outscal_ustress * u[:, :, Iu])
    # Meridional (:267-290): DO j = jmin+1,jmax; DO i = imin,imax
    v = exf["vstress"]
    v = jnp.where(v > p.windstressmax, p.windstressmax, v)
    v = jnp.where(v < -p.windstressmax, -p.windstressmax, v)
    Jv = L.js(1 - L.OLy + 1, L.sNy + L.OLy)
    fv = ff["fv"].at[:, Jv, :].set(p.outscal_vstress * v[:, Jv, :])
    # Short wave (:302-309)
    Qsw = p.outscal_swflux * exf["swflux"]
    # (SST, SSS :311-325 not ported: climsst/sss relaxation off)
    # Atmospheric loading (:327-333)
    pLoad = p.outscal_apressure * exf["apressure"]
    # Salt flux (:335-341), no exchange
    saltFlux = exf["saltflx"]
    # Salt-plume flux (:343-349)
    saltPlumeFlux = exf["spflx"]
    # Tile edges (:366-388)
    Qnet = ex.exch_xy(Qnet)                                                              # :366
    EmPmR = ex.exch_xy(EmPmR)                                                            # :367
    fu, fv = ex.exch_uv_xy(fu, fv, True)                                                 # :368
    Qsw = ex.exch_xy(Qsw)                                                                # :372
    pLoad = ex.exch_xy(pLoad)                                                            # :381
    saltPlumeFlux = ex.exch_xy(saltPlumeFlux)                                            # :387
    exf["ustress"], exf["vstress"] = u, v
    ff.update(Qnet=Qnet, EmPmR=EmPmR, fu=fu, fv=fv, Qsw=Qsw, pLoad=pLoad, saltFlux=saltFlux,
              saltPlumeFlux=saltPlumeFlux)
    return exf, ff


def exf_getforcing(p, g, ex, exf, ff, bufs, facs, myTime):
    """exf_getforcing.F:149-299 (flux-forced build): LOAD_FIELDS_DRIVER's EXF part (load_fields_driver.F:144).

    exf: EXF_FIELDS arrays at the start of the step (exf_init_varia at the model start); ff: FFIELDS arrays (fu, fv
    at least); bufs/facs from ExfRecordLoader.load. Returns (exf, ff) after EXF_MAPFIELDS."""
    L = g.layout
    exf = exf_getffields(p, exf, bufs, facs, myTime, L)                                # exf_getforcing.F:161
    u, v = exf["ustress"], exf["vstress"]
    if p.stressIsOnCgrid:                                                                # exf_getforcing.F:162-166
        u, v = ex.exch_uv_xy(u, v, True)
    # EXF_RADIATION (exf_getforcing.F:176): swfluxfile set, swdownfile ' ' -> nothing (exf_radiation.F:139)
    # EXF_WIND (:187): writes only cw, sw, sh, wStress (unused in this build) -- not ported
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    mC = g.maskC[:, 0]
    sflux = exf["sflux"]
    sflux = sflux.at[:, J, I].set(sflux[:, J, I] - exf["runoff"][:, J, I])             # exf_getforcing.F:229
    hflux = exf["hflux"]
    hflux = hflux.at[:, J, I].set(hflux[:, J, I] * mC[:, J, I])                         # exf_getforcing.F:231
    sflux = sflux.at[:, J, I].set(sflux[:, J, I] * mC[:, J, I])                         # exf_getforcing.F:232
    if p.stressIsOnCgrid:                                                                # exf_getforcing.F:244-248
        u, v = ex.exch_uv_xy(u, v, True)
    else:
        raise NotImplementedError("EXCH_UV_AGRID_3D_RL path (stress on A grid)")
    # EXF_GETSURFACEFLUXES (:260): useCTRL=F -> nothing; EXF_CHECK_RANGE/DIAGNOSTICS/MONITOR: diagnostics only
    exf.update(ustress=u, vstress=v, hflux=hflux, sflux=sflux)
    return exf_mapfields(p, g, ex, exf, ff, myTime)                                     # exf_getforcing.F:299
