"""pkg/smooth (c66g, no V4r4 override): the Weaver & Courtier (2001) correlation operator WC01 that the ctrl package
applies to the initial-condition / parameter controls (ctrl_map_ini_genarr.F:271-276, :428-433), literally.

    cfg = SmoothConfig.from_namelists(nml)                 # data.smooth (smooth_readparms.F)
    sp  = SmoothParams.from_namelists(nml)                 # abEps, rkSign, nIter0 (jit argument)
    ops = smooth_init_fixed(cfg, g, ex, rundir, recip_hFacC)   # SMOOTH_INIT_FIXED + operator re-read
    xx  = smooth_correl3d(sp, cfg.op3d(1), ops["3d"][1], ops["h"], g, ex, xx, recip_hFacC)   # SMOOTH_CORREL3D
    xx  = smooth_correl2d(sp, cfg.op2d(1), ops["2d"][1], g, ex, xx, g.maskC)     # SMOOTH_CORREL2D(xx, maskC, 1)

SMOOTH_CORREL3D = normalise by sqrt(volume), diffuse nbt/2 pseudo-time steps (SMOOTH_DIFF3D: explicit horizontal
diffusion SMOOTH_RHS with Adams-Bashforth-2, implicit vertical diffusion SMOOTH_IMPLDIFF, exchanges), multiply by the
normalisation field smooth3Dnorm (input file). SMOOTH_CORREL2D the same in 2-D (explicit diffusion only).

Operator precision: SMOOTH_INIT3D/2D compute Kux = Lx*Lx/totTime/2 etc. in real*8 and WRITE them to
smooth3Doperator<NNN> / smooth2Doperator<NNN> with smoothprec = 32 (SMOOTH.h:5); SMOOTH_CORREL3D/2D READ them back
(real*4 -> real*8) at every call. smooth_init_fixed reproduces that round trip (float32 rounding of the interior, the
halos of the previous computation outside the exchanged ones).

Ported for the V4r4 data.smooth only (hard error otherwise): smooth3DsizeH = smooth3DsizeZ = 3 (scale files),
smooth2Dsize = 2 (scale file), smooth3Dfilter = smooth2Dfilter = 0 (the normalisation smooth3Dnorm<NNN> /
smooth2Dnorm<NNN> is an input file; SMOOTH_FILTERVAR3D/2D do nothing, smooth_filtervar3d.F:46).
smooth_init3d.F computes the type-1 operator (horizontal along grid axes) whatever smooth3DtypeH is (:14-19): Kwx,
Kwy, Kwz, Kuz, Kvz, Kuy, Kvx are 0; they are still carried and used, as SMOOTH_RHS uses them.

Dead stores not ported (no value read afterwards): smooth_diff3d.F:122-129 (gT_in interior zeroed + exchanged, then
fully overwritten by SMOOTH_RHS, smooth_rhs.F:80) and :152, :176 (gT_in after the step: rebuilt from zero in the next
SMOOTH_RHS; discarded after the last iteration); :60-70 (gT_in, gTm1_in = 0 and the exchanges of these zero arrays).
SMOOTH_IMPLDIFF's tridiagonal coefficients (a, bet, gam; smooth_impldiff.F:48-141) depend only on kappaR and
recip_hFacC: smooth_diff3d computes them once before the loop (the Fortran recomputes the same values each step).

Gate: tests/test_ctrl.py (bitwise vs the Fortran for all eight V4r4 controls, halos included).
"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax

from mitgcm_jax.core.implicit import vertical_factors
from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.io.mds import read_bin
from mitgcm_jax.params_io import params_pytree

SMOOTHPREC = 32               # SMOOTH.h:5 smoothprec = 32
SMOOTH3D_DOIMPLDIFF = True    # SMOOTH.h:8 smooth3DdoImpldiff = .TRUE.
SMOOTH_OP_NB_MAX = 10         # SMOOTH.h:11 smoothOpNbMax = 10
SMOOTH2D_DELTIME = 1.0        # SMOOTH.h:14 smooth2DdelTime = 1. _d 0
SMOOTH3D_DELTIME = 1.0        # SMOOTH.h:15 smooth3DdelTime = 1. _d 0


def nml_array(nml, fname, group, key, n, default):
    """Fortran array `key(1:n)` of a namelist with its default (indexed assignments `key(i)=v` as written)."""
    out = [default] * n
    g = nml.file(fname).get(group.lower(), {})
    for k, v in g.items():
        if k == key.lower():                                     # whole-array assignment key = v1, v2, ...
            for i, x in enumerate(v):
                out[i] = x
        elif k.startswith(key.lower() + "("):
            idx = k[len(key) + 1:-1]
            if ":" in idx or "," in idx:
                raise NotImplementedError(f"{fname}: {k} (array section) not supported for {key}")
            i0 = int(idx) - 1
            for i, x in enumerate(v):
                out[i0 + i] = x
    return out


@dataclass(frozen=True)
class SmoothOp3D:
    """One 3-D operator of data.smooth (static)."""
    nb: int
    nbt: int
    typeH: int
    sizeH: int
    typeZ: int
    sizeZ: int
    filter: int


@dataclass(frozen=True)
class SmoothOp2D:
    nb: int
    nbt: int
    type: int
    size: int
    filter: int


@dataclass(frozen=True)
class SmoothConfig:
    """data.smooth (smooth_readparms.F:113-128), defaults smooth_readparms.F:145-162 (all 0)."""
    ops3d: tuple
    ops2d: tuple

    @classmethod
    def from_namelists(cls, nml):
        if not bool(nml.get("data.pkg", "packages", "useSMOOTH", default=False)):   # packages_boot.F: useSMOOTH=F
            raise NotImplementedError("useSMOOTH=F: SMOOTH_READPARMS returns early (smooth_readparms.F:132-140); "
                                      "the ctrl WC01 path needs pkg/smooth")
        n = SMOOTH_OP_NB_MAX

        def arr(key):
            return nml_array(nml, "data.smooth", "smooth_nml", key, n, 0)

        nbt2, t2, s2, f2 = arr("smooth2Dnbt"), arr("smooth2Dtype"), arr("smooth2Dsize"), arr("smooth2Dfilter")
        nbt3, tH, sH, tZ, sZ, f3 = (arr("smooth3Dnbt"), arr("smooth3DtypeH"), arr("smooth3DsizeH"),
                                    arr("smooth3DtypeZ"), arr("smooth3DsizeZ"), arr("smooth3Dfilter"))
        ops2d, ops3d = [], []
        for i in range(n):
            if t2[i] != 0:                                                         # smooth_init_fixed.F:48
                if s2[i] != 2:
                    raise NotImplementedError(f"smooth2Dsize({i + 1})={s2[i]}: only 2 (scale file "
                                              "smooth2Dscales<NNN>, smooth_init2d.F:230-239) is ported")
                if f2[i] != 0:
                    raise NotImplementedError(f"smooth2Dfilter({i + 1})={f2[i]}: SMOOTH_FILTERVAR2D not ported")
                ops2d.append(SmoothOp2D(i + 1, int(nbt2[i]), int(t2[i]), int(s2[i]), int(f2[i])))
            if tZ[i] != 0 or tH[i] != 0:                                           # smooth_init_fixed.F:60-61
                if sZ[i] != 3 or sH[i] != 3:
                    raise NotImplementedError(f"smooth3DsizeZ/H({i + 1})={sZ[i]}/{sH[i]}: only 3 (scale files "
                                              "smooth3DscalesZ/H<NNN>, smooth_init3d.F:47-52, 105-113) is ported")
                if f3[i] != 0:
                    raise NotImplementedError(f"smooth3Dfilter({i + 1})={f3[i]}: SMOOTH_FILTERVAR3D not ported")
                ops3d.append(SmoothOp3D(i + 1, int(nbt3[i]), int(tH[i]), int(sH[i]), int(tZ[i]), int(sZ[i]),
                                        int(f3[i])))
        return cls(tuple(ops3d), tuple(ops2d))

    def op3d(self, nb):
        for o in self.ops3d:
            if o.nb == nb:
                return o
        raise ValueError(f"smooth 3-D operator {nb} is not defined in data.smooth (smooth3DtypeH/Z = 0)")

    def op2d(self, nb):
        for o in self.ops2d:
            if o.nb == nb:
                return o
        raise ValueError(f"smooth 2-D operator {nb} is not defined in data.smooth (smooth2Dtype = 0)")


@params_pytree
@dataclass(frozen=True)
class SmoothParams:
    abEps: float          # PARAMS.h, data parm03 (set_defaults.F:310 abEps = 0.01 _d 0): ADAMS_BASHFORTH2, diff2d
    rkSign: float         # ini_vertical_grid.F:56 rkSign = -1. _d 0 (smooth_rhs.F:344, 356)
    delTime3d: float      # SMOOTH.h:15
    delTime2d: float      # SMOOTH.h:14
    nIter0: int           # adams_bashforth2.F:64 (myIter.EQ.nIter0 .AND. startAB.EQ.0: abFac = 0)

    @classmethod
    def from_namelists(cls, nml):
        nIter0 = int(nml.get("data", "parm03", "nIter0", default=0))         # set_defaults.F: nIter0 = 0
        if nml.has("data", "parm03", "startTime"):
            raise NotImplementedError("startTime set: nIter0 from startTime (ini_parms.F:962-974) not ported")
        return cls(abEps=float(nml.get("data", "parm03", "abEps", default=0.01)),   # set_defaults.F:310
                   rkSign=-1.0,                                                       # ini_vertical_grid.F:56
                   delTime3d=SMOOTH3D_DELTIME, delTime2d=SMOOTH2D_DELTIME, nIter0=nIter0)


# ---------------------------------------------------------------------------------------------------------------------
# SMOOTH_INIT_FIXED (host side: file reads, operator construction, float32 round trip)
# ---------------------------------------------------------------------------------------------------------------------
def _read_rec(path, L, nz, rec):
    """READ_REC_3D_RL(fname, smoothprec, nz, fld, rec, ...) of a global file -> interior [T, (nz,) sNy, sNx]
    (MDS_READ_FIELD fills i=1..sNx, j=1..sNy only)."""
    a = read_bin(path, nz=nz, prec=SMOOTHPREC)[rec - 1]
    t = np.moveaxis(compact_to_tiles(a), -3, 0)
    return t if nz > 1 else t[:, 0]


def _set_interior(base, vals, L):
    out = np.array(base, dtype=np.float64, copy=True)
    out[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = vals
    return out


def _f32_round_trip(a, L):
    """WRITE_REC_3D_RL(..., smoothprec=32) of a's interior then READ_REC_3D_RL back into the same array: the
    interior becomes real*4-rounded, the halos keep a's values (MDS_WRITE/READ_FIELD work on i=1..sNx, j=1..sNy)."""
    a = np.asarray(a, dtype=np.float64)
    inner = a[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx].astype(np.float32).astype(np.float64)
    return _set_interior(a, inner, L)


def _file(rundir, stem, nb):
    """write(fnamegeneric,'(1a,i3.3)') stem, nb; MDS_READ_FIELD opens the plain name if it exists, else
    <name>.data (mds_read_field.F)."""
    p = Path(rundir) / f"{stem}{nb:03d}"
    return p if p.exists() else Path(str(p) + ".data")


def smooth_init3d(op, g, ex, rundir):
    """SMOOTH_INIT3D (smooth_init3d.F:6-192) for one operator, then the operator re-read + exchanges of
    SMOOTH_CORREL3D (smooth_correl3d.F:40-79). Returns the arrays SMOOTH_DIFF3D uses (float64 numpy)."""
    L = g.layout
    ex_ = lambda a: np.asarray(ex.exch_xy(a))  # noqa: E731  EXCH_XYZ_RL
    totTime = op.nbt * SMOOTH3D_DELTIME                                          # :43
    z3 = np.zeros(L.shape3d)
    # :47-52 sizeZ = 3: smooth3D_Lz from smooth3DscalesZ<NNN> record 1, EXCH (common block: zero halos before)
    Lz = ex_(_set_interior(z3, _read_rec(_file(rundir, "smooth3DscalesZ", op.nb), L, L.Nr, 1), L))
    kappaR = Lz * Lz / totTime / 2                                               # :72-73
    # :81-98 KzMax cap only for sizeZ .NE. 3
    kappaR = ex_(kappaR)                                                         # :100
    # :105-113 sizeH = 3: Lx record 1, Ly record 2 of smooth3DscalesH<NNN>
    fH = _file(rundir, "smooth3DscalesH", op.nb)
    Lx = ex_(_set_interior(z3, _read_rec(fH, L, L.Nr, 1), L))
    Ly = ex_(_set_interior(z3, _read_rec(fH, L, L.Nr, 2), L))
    # :129-149 full range
    K = dict(Kuy=z3.copy(), Kvx=z3.copy(), Kwx=z3.copy(), Kwy=z3.copy(), Kwz=z3.copy(),
             Kux=Lx * Lx / totTime / 2, Kvy=Ly * Ly / totTime / 2, Kuz=z3.copy(), Kvz=z3.copy())
    kappaR = ex_(kappaR)                                                         # :153
    for n in ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "Kuy", "Kvx"):    # :154-162
        K[n] = ex_(K[n])
    K["kappaR"] = kappaR
    # :167-189 WRITE_REC_3D_RL(smooth3Doperator<NNN>, smoothprec) ... smooth_correl3d.F:43-72 READ + EXCH
    out = {n: ex_(_f32_round_trip(K[n], L)) for n in ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "Kuy",
                                                     "Kvx", "kappaR")}
    # smooth_correl3d.F:75-79 smooth3Dnorm<NNN> (input file, smooth3Dfilter = 0), EXCH (common block, zero halos)
    out["norm"] = ex_(_set_interior(z3, _read_rec(_file(rundir, "smooth3Dnorm", op.nb), L, L.Nr, 1), L))
    return out


def smooth_init2d(op, g, ex, rundir):
    """SMOOTH_INIT2D (smooth_init2d.F:6-85) for one operator, then the re-read + exchanges of SMOOTH_CORREL2D
    (smooth_correl2d.F:40-55)."""
    L = g.layout
    ex_ = lambda a: np.asarray(ex.exch_xy(a))  # noqa: E731  EXCH_XY_RL
    totTime = op.nbt * SMOOTH2D_DELTIME                                          # :23
    z2 = np.zeros(L.shape2d)
    f = _file(rundir, "smooth2Dscales", op.nb)                                   # :25-34 (size 2)
    Lx = ex_(_set_interior(z2, _read_rec(f, L, 1, 1), L))
    Ly = ex_(_set_interior(z2, _read_rec(f, L, 1, 2), L))
    Kux = ex_(Lx * Lx / totTime / 2)                                             # :48-61
    Kvy = ex_(Ly * Ly / totTime / 2)
    # :65-70 WRITE smooth2Doperator<NNN> (smoothprec); smooth_correl2d.F:43-48 READ + EXCH
    out = dict(Kux=ex_(_f32_round_trip(Kux, L)), Kvy=ex_(_f32_round_trip(Kvy, L)))
    # smooth_correl2d.F:51-55 smooth2Dnorm<NNN>
    out["norm"] = ex_(_set_interior(z2, _read_rec(_file(rundir, "smooth2Dnorm", op.nb), L, 1, 1), L))
    return out


def smooth_init_fixed(cfg: SmoothConfig, g, ex, rundir, recip_hFacC):
    """SMOOTH_INIT_FIXED (smooth_init_fixed.F:6-75, called from PACKAGES_INIT_FIXED after INI_MASKS_ETC):
    smooth_recip_hFacC/hFacW/hFacS = the INI_MASKS_ETC fields (:33-45), every 2-D and 3-D operator of data.smooth.
    recip_hFacC: the INI_MASKS_ETC value (1/h0FacC where h0FacC .NE. 0, else 0). Returns
    {"h": {recip_hFacC, hFacW, hFacS}, "3d": {nb: arrays}, "2d": {nb: arrays}} (jnp arrays)."""
    h = dict(recip_hFacC=jnp.asarray(recip_hFacC), hFacW=jnp.asarray(g.h0FacW), hFacS=jnp.asarray(g.h0FacS))
    o2 = {op.nb: {k: jnp.asarray(v) for k, v in smooth_init2d(op, g, ex, rundir).items()} for op in cfg.ops2d}
    o3 = {op.nb: {k: jnp.asarray(v) for k, v in smooth_init3d(op, g, ex, rundir).items()} for op in cfg.ops3d}
    return {"h": h, "3d": o3, "2d": o2}


# ---------------------------------------------------------------------------------------------------------------------
# jittable operators
# ---------------------------------------------------------------------------------------------------------------------
def _k_up(a):
    """a(MAX(k-1,1)) for k = 1..Nr (level axis 1)."""
    return jnp.concatenate([a[:, :1], a[:, :-1]], axis=1)


def _k_down(a):
    """a(MIN(k+1,Nr)) for k = 1..Nr."""
    return jnp.concatenate([a[:, 1:], a[:, -1:]], axis=1)


def _pad1(a):
    """an array on j, i = 1-OL+1 .. sN+OL-1 placed in a full array with zero outer ring (the loop range of
    SMOOTH_RHS; the ring keeps the 0 of its initialisation loop, smooth_rhs.F:74-83)."""
    return jnp.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)])


def smooth_rhs(sp, ops, h, g, ex, fld):
    """SMOOTH_RHS (smooth_rhs.F:7-368): gt_in = -div(K grad fld) on iMin..iMax, jMin..jMax = 1-OL+1 .. sN+OL-1
    (every level), 0 elsewhere before the final exchange; smooth3DdoImpldiff = .TRUE.: no GAD_DIFF_R term (:246-260).
    """
    L = g.layout
    Nr = L.Nr
    ny, nx = L.ny, L.nx
    C_, M_, P_ = slice(1, ny - 1), slice(0, ny - 2), slice(2, ny)      # j: jMin..jMax, j-1, j+1
    Ci, Mi, Pi = slice(1, nx - 1), slice(0, nx - 2), slice(2, nx)      # i: iMin..iMax, i-1, i+1
    kp1 = np.minimum(np.arange(Nr) + 1, Nr - 1)                         # MIN(k+1,Nr)
    col = lambda v: jnp.asarray(v)[None, :, None, None]  # noqa: E731
    drF = col(g.drF)
    rdrC = jnp.asarray(g.recip_drC)
    rdrC_k, rdrC_kp1 = col(rdrC[:Nr]), col(rdrC[kp1])
    rdrF = col(g.recip_drF)
    mC, mW, mS = g.maskC, g.maskW, g.maskS
    dyG, dxG = g.dyG[:, None], g.dxG[:, None]
    rdxC, rdyC = g.recip_dxC[:, None], g.recip_dyC[:, None]
    f = fld
    fk1, fkp = _k_up(f), _k_down(f)
    mCkp = _k_down(mC)
    # :93-107 xA, yA, maskUp (full range, every k)
    xA = (dyG * drF) * h["hFacW"]
    yA = (dxG * drF) * h["hFacS"]
    maskUp = _k_up(mC) * mC                                             # used for k > 1 only (:309-316)

    def c(a):
        return a[..., C_, Ci]

    # ---- x (///gmredi_xtr///), :111-160
    df = 0.0 - ((c(xA) * c(ops["Kux"])) * c(rdxC)) * (c(f) - f[..., C_, Mi])                        # :111-119
    dTdz = 0.5 * (+(0.5 * rdrC_k) * (mC[..., C_, Mi] * (fk1[..., C_, Mi] - f[..., C_, Mi])
                                     + c(mC) * (c(fk1) - c(f)))
                  + (0.5 * rdrC_kp1) * (mCkp[..., C_, Mi] * (f[..., C_, Mi] - fkp[..., C_, Mi])
                                        + c(mCkp) * (c(f) - c(fkp))))                                 # :123-135
    df = df - (c(xA) * c(ops["Kuz"])) * dTdz                                                          # :136-137
    dTdy = 0.5 * (+0.5 * ((c(mS) * c(rdyC)) * (c(f) - f[..., M_, Ci])
                          + (mS[..., P_, Ci] * rdyC[..., P_, Ci]) * (f[..., P_, Ci] - c(f)))
                  + 0.5 * ((mS[..., C_, Mi] * c(rdyC)) * (f[..., C_, Mi] - f[..., M_, Mi])
                           + (mS[..., P_, Mi] * rdyC[..., P_, Ci]) * (f[..., P_, Mi] - f[..., C_, Mi])))  # :143-156
    df = df - (c(xA) * c(ops["Kuy"])) * dTdy                                                          # :157-158
    fZon = _pad1(0.0 + df)                                                                          # :77, :165-169
    # ---- y (///gmredi_ytr///), :179-236
    df = 0.0 - ((c(yA) * c(ops["Kvy"])) * c(rdyC)) * (c(f) - f[..., M_, Ci])                        # :179-187
    dTdz = 0.5 * (+(0.5 * rdrC_k) * (mC[..., M_, Ci] * (fk1[..., M_, Ci] - f[..., M_, Ci])
                                     + c(mC) * (c(fk1) - c(f)))
                  + (0.5 * rdrC_kp1) * (mCkp[..., M_, Ci] * (f[..., M_, Ci] - fkp[..., M_, Ci])
                                        + c(mCkp) * (c(f) - c(fkp))))                                 # :191-203
    df = df - (c(yA) * c(ops["Kvz"])) * dTdz                                                          # :204-205
    dTdx = 0.5 * (+0.5 * ((mW[..., C_, Pi] * rdxC[..., C_, Pi]) * (f[..., C_, Pi] - c(f))
                          + (c(mW) * c(rdxC)) * (c(f) - f[..., C_, Mi]))
                  + 0.5 * ((mW[..., M_, Pi] * rdxC[..., C_, Pi]) * (f[..., M_, Pi] - f[..., M_, Ci])
                           + (mW[..., M_, Ci] * c(rdxC)) * (f[..., M_, Ci] - f[..., M_, Mi])))    # :211-224
    df = df - (c(yA) * c(ops["Kvx"])) * dTdx                                                          # :225-226
    fMer = _pad1(0.0 + df)                                                                          # :232-236
    # ---- r (///gmredi rtrans///), k > 1, :264-316; df = 0 before (:238-242, no GAD_DIFF_R)
    fm1 = f[:, :-1]                                                      # level k-1 for k = 2..Nr
    fk = f[:, 1:]
    mWk, mWm = mW[:, 1:], mW[:, :-1]
    mSk, mSm = mS[:, 1:], mS[:, :-1]

    def cr(a):
        return a[..., C_, Ci]

    dTdx = 0.5 * (+0.5 * ((mWk[..., C_, Pi] * rdxC[..., C_, Pi]) * (fk[..., C_, Pi] - cr(fk))
                          + (cr(mWk) * c(rdxC)) * (cr(fk) - fk[..., C_, Mi]))
                  + 0.5 * ((mWm[..., C_, Pi] * rdxC[..., C_, Pi]) * (fm1[..., C_, Pi] - cr(fm1))
                           + (cr(mWm) * c(rdxC)) * (cr(fm1) - fm1[..., C_, Mi])))                    # :267-280
    dTdy = 0.5 * (+0.5 * ((cr(mSk) * c(rdyC)) * (cr(fk) - fk[..., M_, Ci])
                          + (mSk[..., P_, Ci] * rdyC[..., P_, Ci]) * (fk[..., P_, Ci] - cr(fk)))
                  + 0.5 * ((cr(mSm) * c(rdyC)) * (cr(fm1) - fm1[..., M_, Ci])
                           + (mSm[..., P_, Ci] * rdyC[..., P_, Ci]) * (fm1[..., P_, Ci] - cr(fm1))))  # :282-295
    rA = g.rA[:, None][..., C_, Ci]
    df = 0.0 - rA * (cr(ops["Kwx"][:, 1:]) * dTdx + cr(ops["Kwy"][:, 1:]) * dTdy)                    # :297-300
    fV = 0.0 + df * cr(maskUp[:, 1:])                                                                 # :309-316
    fVerT = _pad1(jnp.concatenate([fV, jnp.zeros_like(fV[:, :1])], axis=1))   # fVerT(k-1), k=2..Nr; fVerT(Nr)=0
    # :329-330 exchanges
    fZon, fMer = ex.exch_uv_xy(fZon, fMer, True)                        # EXCH_UV_XYZ_RL(fZon,fMer,.TRUE.)
    fVerT = ex.exch_xy(fVerT)                                           # EXCH_XYZ_RL
    # :335-361 divergence (k = 1: fVerT(k) only)
    dV = jnp.concatenate([fVerT[:, :1], fVerT[:, 1:] - fVerT[:, :-1]], axis=1)   # k=1: fVerT(1); k>1: diff
    sRh = h["recip_hFacC"]
    div = ((fZon[..., C_, Pi] - c(fZon)) + (fMer[..., P_, Ci] - c(fMer))) + c(dV) * sp.rkSign
    gt = _pad1(0.0 - ((c(sRh) * rdrF) * c(g.recip_rA[:, None])) * div)
    return ex.exch_xy(gt)                                               # :366 EXCH_XYZ_RL(gt_in)


def _safe_recip(d):
    """1/d where d .NE. 0 (smooth_impldiff.F:119, :134), bet keeps its initial 1 elsewhere (:106)."""
    nz = d != 0.0
    return jnp.where(nz, 1.0 / jnp.where(nz, d, 1.0), 1.0)


def smooth_impldiff_coeffs(g, deltaTX, kappaR, recip_hFac):
    """The fld-independent part of SMOOTH_IMPLDIFF (smooth_impldiff.F:48-141): a, bet, gam on the interior, level
    axis first ([Nr, T, sNy, sNx]). The Fortran recomputes them at every call from the same inputs; smooth_diff3d
    computes them once (same values)."""
    L = g.layout
    Nr = L.Nr
    vf = vertical_factors(Nr)
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    col = lambda v: jnp.asarray(v)[None, :, None, None]  # noqa: E731
    rdrF = col(g.recip_drF)
    rdrC = jnp.asarray(g.recip_drC)
    rdrC_k, rdrC_kp1 = col(rdrC[:Nr]), col(rdrC[1:Nr + 1])
    rdf2c, rrfc = col(vf["recip_deepFac2C"]), col(vf["recip_rhoFacC"])
    df2F, rfF = col(vf["deepFac2F"][:Nr]), col(vf["rhoFacF"][:Nr])
    df2Fp, rfFp = col(vf["deepFac2F"][1:Nr + 1]), col(vf["rhoFacF"][1:Nr + 1])
    rh = recip_hFac[:, :, J, I]
    K = kappaR[:, :, J, I]
    # :57-73 a(1) = 0; k = 2..Nr (0 where the level above is dry)
    a = -(deltaTX * rh * rdrF * rdf2c * rrfc * K * rdrC_k * df2F * rfF)
    rh_km1 = jnp.concatenate([jnp.zeros_like(rh[:, :1]), rh[:, :-1]], axis=1)
    a = jnp.where(rh_km1 == 0.0, 0.0, a).at[:, 0].set(0.0)
    # :75-91 k = 1..Nr-1 with KappaRX(k+1) (0 where the level below is dry); c(Nr) = 0
    K_kp1 = jnp.concatenate([K[:, 1:], jnp.zeros_like(K[:, :1])], axis=1)
    cc = -(deltaTX * rh * rdrF * rdf2c * rrfc * K_kp1 * rdrC_kp1 * df2Fp * rfFp)
    rh_kp1 = jnp.concatenate([rh[:, 1:], jnp.zeros_like(rh[:, :1])], axis=1)
    cc = jnp.where(rh_kp1 == 0.0, 0.0, cc).at[:, Nr - 1].set(0.0)
    b = 1.0 - cc - a                                                    # :94-100
    km = lambda v: jnp.moveaxis(v, 1, 0)  # noqa: E731
    a_, b_, c_ = km(a), km(b), km(cc)
    # :103-141 bet(1) = 1/b(1) if nonzero (else 1); gam(k) = c(k-1)*bet(k-1); bet(k) = 1/(b(k)-a(k)*gam(k))
    bet1 = _safe_recip(b_[0])

    def fwd(carry, xs):
        bet_m, c_m = carry
        ak, bk, ck = xs
        gam = c_m * bet_m
        bet = _safe_recip(bk - ak * gam)
        return (bet, ck), (gam, bet)

    _, (gam_, bet_) = lax.scan(fwd, (bet1, c_[0]), (a_[1:], b_[1:], c_[1:]))
    bet_ = jnp.concatenate([bet1[None], bet_], axis=0)
    gam_ = jnp.concatenate([jnp.zeros_like(bet1)[None], gam_], axis=0)   # gam(1) unused
    return a_, bet_, gam_


def smooth_impldiff_solve(g, coeffs, fld):
    """The sweeps of SMOOTH_IMPLDIFF (smooth_impldiff.F:144-176) on the interior of fld."""
    L = g.layout
    Nr = L.Nr
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    a_, bet_, gam_ = coeffs
    x_ = jnp.moveaxis(fld[:, :, J, I], 1, 0)
    y1 = x_[0] * bet_[0]                                                # :144-148

    def fwd(y_m, xs):                                                   # :149-156
        ak, betk, xk = xs
        y = betk * (xk - ak * y_m)
        return y, y

    _, y_ = lax.scan(fwd, y1, (a_[1:], bet_[1:], x_[1:]))
    y_ = jnp.concatenate([y1[None], y_], axis=0)

    def bwd(y_p, xs):                                                   # :160-168 backward sweep, k = Nr-1..1
        yk, gk1 = xs
        yk = yk - gk1 * y_p
        return yk, yk

    _, y_up = lax.scan(bwd, y_[Nr - 1], (y_[:Nr - 1], gam_[1:]), reverse=True)
    y_ = jnp.concatenate([y_up, y_[Nr - 1:]], axis=0)
    return fld.at[:, :, J, I].set(jnp.moveaxis(y_, 0, 1))              # :170-176


def smooth_impldiff(g, deltaTX, kappaR, recip_hFac, fld):
    """SMOOTH_IMPLDIFF(bi,bj, 1,sNx, 1,sNy, deltaTX, KappaRX, recip_hFac, gXNm1) (smooth_impldiff.F:6-179): implicit
    vertical diffusion of fld on the interior; operation order of the file (b = 1 - c - a, :97)."""
    return smooth_impldiff_solve(g, smooth_impldiff_coeffs(g, deltaTX, kappaR, recip_hFac), fld)


def smooth_diff3d(sp, ops, h, g, ex, fld, nbt_in, recip_hFacC):
    """SMOOTH_DIFF3D(fld_in, nbt_in) (smooth_diff3d.F:9-190; ALLOW_TAMC_CHECKPOINTING only splits the same loop into
    nested levels): nbt_in pseudo-time steps. recip_hFacC: GRID.h recip_hFacC at the call (SMOOTH_IMPLDIFF, :167);
    SMOOTH_RHS uses the SMOOTH_INIT_FIXED copy h["recip_hFacC"]."""
    mC = g.maskC
    wet = mC != 0.0
    fld = ex.exch_xy(fld)                                               # :68
    gTm1 = jnp.zeros_like(fld)                                          # :60-61, :70 (exchange of zeros)
    # SMOOTH_IMPLDIFF's matrix (smooth_impldiff.F:48-141): the same inputs at every iteration -> computed once
    coeffs = smooth_impldiff_coeffs(g, sp.delTime3d, ops["kappaR"], recip_hFacC)

    def body(i0, carry):
        fld, gTm1 = carry
        gT = smooth_rhs(sp, ops, h, g, ex, fld)                         # :132
        # :136-144 ADAMS_BASHFORTH2(bi,bj,k,Nr, gT_in, gTm1_in, gt_AB, startAB=0, myIter=iloop-1): full range
        myIter = i0                                                     # iloop - 1
        abFac = jnp.where(myIter == sp.nIter0, 0.0, 0.5 + sp.abEps)   # adams_bashforth2.F:64-68 (startAB = 0)
        ab = abFac * (gT - gTm1)                                        # :87
        gTm1 = gT                                                       # :88
        gT = gT + ab                                                    # :89
        # :146-156 time stepping on the full range where maskC .NE. 0
        fld = jnp.where(wet, fld + sp.delTime3d * gT, fld)
        # :163-170 SMOOTH_IMPLDIFF(bi,bj,1,sNx,1,sNy, smooth3DdelTime, smooth3D_kappaR, recip_hFacC, fld_in)
        fld = smooth_impldiff_solve(g, coeffs, fld)
        fld = ex.exch_xy(fld)                                           # :175
        gTm1 = ex.exch_xy(gTm1)                                         # :177
        return fld, gTm1

    fld, _ = lax.fori_loop(0, nbt_in, body, (fld, gTm1))
    return fld


def smooth_correl3d(sp, op: SmoothOp3D, ops, h, g, ex, fld, recip_hFacC):
    """SMOOTH_CORREL3D(fld_in, smoothOpNb) (smooth_correl3d.F:6-115). fld: [T,Nr,ny,nx] (xx_gen: zero halos,
    interior from the control file). ops: smooth_init_fixed(...)["3d"][nb]; h: smooth_init_fixed(...)["h"];
    recip_hFacC: the model's GRID.h recip_hFacC at the call (read by SMOOTH_IMPLDIFF)."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    rdrF = jnp.asarray(g.recip_drF)[None, :, None, None]
    fld = fld.at[..., J, I].set(fld[..., J, I] * jnp.sqrt(g.recip_rA[:, None, J, I] * rdrF))   # :82-93
    fld = ex.exch_xy(fld)                                                                         # :94
    fld = smooth_diff3d(sp, ops, h, g, ex, fld, op.nbt // 2, recip_hFacC)                         # :97-98
    fld = fld.at[..., J, I].set(fld[..., J, I] * ops["norm"][..., J, I])                          # :101-112
    return ex.exch_xy(fld)                                                                        # :113


def smooth_diff2d(sp, ops, g, ex, fld, mask_in, nbt_in):
    """SMOOTH_DIFF2D(fld_in, mask_in, nbt_in) (smooth_diff2d.F:9-205). Local arrays gt_in, gtm1_in, smooth2Dmask
    are large (> -fmax-stack-var-size) and thus static, zero before the first call: their halos are 0 before the
    exchanges of :73-76."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    Jm, Jp, Im, Ip = L.js(0, L.sNy - 1), L.js(2, L.sNy + 1), L.is_(0, L.sNx - 1), L.is_(2, L.sNx + 1)
    z2 = jnp.zeros_like(fld)
    m2 = ex.exch_xy(z2.at[:, J, I].set(mask_in[:, 0, J, I]))           # :61-71, :76
    fld = ex.exch_xy(fld)                                               # :73
    gtm1 = ex.exch_xy(z2)                                               # :66, :75
    Kux, Kvy = ops["Kux"], ops["Kvy"]
    m = m2[:, J, I]
    wet = m != 0.0

    def body(i0, carry):
        fld, gtm1 = carry
        f = fld[:, J, I]
        # :127-161 (gt_in = 0 first; the four fluxes only where smooth2Dmask .NE. 0)
        gt = 0.0
        gt = gt + ((((Kux[:, J, I] * g.dyG[:, J, I]) * m) * m2[:, J, Im]) * (f - fld[:, J, Im])) * g.recip_dxC[:, J, I]
        gt = gt + ((((Kux[:, J, Ip] * g.dyG[:, J, Ip]) * m) * m2[:, J, Ip]) * (f - fld[:, J, Ip])) \
            * g.recip_dxC[:, J, Ip]
        gt = gt + ((((Kvy[:, J, I] * g.dxG[:, J, I]) * m) * m2[:, Jm, I]) * (f - fld[:, Jm, I])) * g.recip_dyC[:, J, I]
        gt = gt + ((((Kvy[:, Jp, I] * g.dxG[:, Jp, I]) * m) * m2[:, Jp, I]) * (f - fld[:, Jp, I])) \
            * g.recip_dyC[:, Jp, I]
        gt = jnp.where(wet, gt, 0.0)
        # :163-188 Adams-Bashforth (myIter = iloop-1)
        first = i0 == 0
        ab15 = jnp.where(first, 1.0, 1.5 + sp.abEps)
        ab05 = jnp.where(first, 0.0, -(0.5 + sp.abEps))
        gm1 = gtm1[:, J, I]
        gt_tmp = ab15 * gt + ab05 * gm1                                 # :177-178
        gtm1 = gtm1.at[:, J, I].set(gt)                                 # :179
        fld = fld.at[:, J, I].set(f - (gt_tmp * g.recip_rA[:, J, I]) * sp.delTime2d)   # :182-183
        # :184 gt_in = 0; :190 EXCH(gt_in) (dead: rebuilt from 0 in the next iteration)
        fld = ex.exch_xy(fld)                                           # :191
        gtm1 = ex.exch_xy(gtm1)                                         # :192
        return fld, gtm1

    fld, _ = lax.fori_loop(0, nbt_in, body, (fld, gtm1))
    return fld


def smooth_correl2d(sp, op: SmoothOp2D, ops, g, ex, fld, mask_in):
    """SMOOTH_CORREL2D(fld_in, mask_in, smoothOpNb) (smooth_correl2d.F:6-87)."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    fld = fld.at[:, J, I].set(fld[:, J, I] * jnp.sqrt(g.recip_rA[:, J, I]))     # :59-68
    fld = ex.exch_xy(fld)                                                         # :69
    fld = smooth_diff2d(sp, ops, g, ex, fld, mask_in, op.nbt // 2)               # :72-73
    fld = fld.at[:, J, I].set(fld[:, J, I] * ops["norm"][:, J, I])               # :76-84
    return ex.exch_xy(fld)                                                        # :86
