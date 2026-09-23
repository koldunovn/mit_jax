"""Sea-ice initialisation of the full V4r4 tree (plan M2.6a): what SEAICE_INIT_VARIA (c66g
pkg/seaice/seaice_init_varia.F; not overridden in V4r4) does for a pickup start, and SEAICE_READ_PICKUP
(pkg/seaice/seaice_read_pickup.F), literally, for the fields the model carries. Only the initialisation: the sea-ice
kernels and their parameters are pkgs/seaice_growth.py, seaice_dyn.py, seaice_lsr.py, seaice_advdiff.py.

Called from PACKAGES_INIT_VARIABLES (packages_init_variables.F:336, after EXF_INIT_VARIA :249 and before
SALT_PLUME_INIT_VARIA :342 and CTRL_INIT_VARIABLES), i.e. in INITIALISE_VARIA after INI_NLFS_VARS (hFacC = h0FacC still)
and CALC_PHI_RLOW_INI, before the r* sequence. Build: SEAICE_CGRID defined, SEAICE_ITD / SEAICE_VARIABLE_SALINITY /
ALLOW_SITRACER undefined, SEAICE_ALLOW_EVP defined (code/SEAICE_OPTIONS.h:46, 59, 62, 92, 101) -> nITD = 7
(SEAICE_SIZE.h:27). gcov (ref_full_serial13_gcov_1day) lines executed: see docs/BRANCHES.md; the branches below.

seaice_init_varia.F, as executed (buoyancyRelation = 'OCEANIC' -> kSurface = 1, :56-60):
  :63-76    HEFFM = 0, then 1 unless hFacC(kSurface) = 0 (full tile)                          -> `seaice_geometry`
  :93-146   k1AtC, k1AtZ, k2AtC, k2AtZ = 0; usingCurvilinearGrid .AND. SEAICEuseMetricTerms: the finite-difference
            metric coefficients (:115-145)                                                    -> `seaice_geometry`
  :253-345  every SEAICE.h array = 0 (full tile), TICES(:, :, 1..nITD) = 0
  :378-391, :442  seaiceMaskU/V from HEFFM + EXCH_UV_XY_RL(.FALSE.)                       -> `seaice_dyn_masks`
            (static: the per-step recomputation in SEAICE_DYNSOLVER, seaice_dynsolver.F:141-171, sits inside
            #ifndef ALLOW_AUTODIFF_TAMC and is compiled out of the V4r4 build)
  :419-437  TICES = 273.0, seaiceMassC/U/V = 1000.0 (full tile, every category)
  :463-466  nIter0 > 0: SEAICE_READ_PICKUP
            seaice_read_pickup.F:68 'pickup_seaice.' // I10.10(nIter0); :76 fp = precFloat64; :85-102 READ_MFLDS_SET,
            precision check; :189-222 (useThSIce = F, SEAICE_multDim = 1): READ_MFLDS_LEV_RL('siTICE', level 1 of
            TICES, doMapTice = .TRUE.), READ_MFLDS_3D_RL siAREA, siHEFF, siHSNOW; :259-262 siUICE, siVICE;
            :263 SEAICEuseBDF2 = F, :270 SEAICEuseEVP = F: nothing more; :285-298 READ_MFLDS_CHECK +
            SEAICE_CHECK_PICKUP; :305-318 doMapTice: TICES(k) = TICES(1) on the interior, k = 2..nITD;
            :321-325 EXCH_UV_XY_RL(uIce, vIce, .TRUE.), EXCH_XY_RL(HEFF), (AREA), EXCH_3D_RL(TICES, nITD),
            EXCH_XY_RL(HSNOW). MDS reads fill the interior only (halos keep the values set above).
  :671-683  ZETA, ETA, PRESS0, ZMAX, ZMIN from HEFF/AREA: overwritten before they are read in every step
            (SEAICE_CALC_ICE_STRENGTH, SEAICE_CALC_VISCOSITIES): not carried
  :685-692  useRealFreshWaterFlux .AND. .NOT.useThSIce: sIceLoad = HEFF*SEAICE_rhoIce + HSNOW*SEAICE_rhoSnow (full tile)
  :697      SEAICE_tensilFac = 0 (default): tensileStrFac keeps 0 from :312                 -> `seaice_dyn_masks`
The carried sea-ice state: AREA, HEFF, HSNOW, TICES [T, nITD, ny, nx], UICE, VICE; plus the ocean field sIceLoad.
The partly-written SEAICE_DYNSOLVER arrays (seaiceMassC/U/V = 1000, FORCEX0/Y0, e11, e22, e12, DWATN, FORCEX/Y = 0 at
this point): pkgs/seaice_model.DYN_CARRY, initial values pkgs/seaice_model.dyn_carry_init.
"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mitgcm_jax.io.llc import compact_to_tiles
from mitgcm_jax.io.mds import read_mds
from mitgcm_jax.params_io import params_pytree

NITD = 7                  # pkg/seaice/SEAICE_SIZE.h:27 nITD = 7 (SEAICE_ITD undefined, code/SEAICE_OPTIONS.h:46)
PRECFLOAT64 = 64          # EEPARAMS.h precFloat64; seaice_read_pickup.F:76 fp = precFloat64
PREC_META = {64: "float64", 32: "float32"}
ICE_STATE = ("AREA", "HEFF", "HSNOW", "TICES", "UICE", "VICE")
ICE_GEOMETRY = ("HEFFM", "k1AtC", "k1AtZ", "k2AtC", "k2AtZ")
ICE_DYN_MASKS = ("seaiceMaskU", "seaiceMaskV", "tensileStrFac")
# every fixed (set once in SEAICE_INIT_VARIA, never written again) field the sea-ice kernels read: the dict `sg` of
# pkgs/seaice_dyn.py / seaice_lsr.py / seaice_model.py (seaice_fixed_fields)
ICE_FIXED = ICE_GEOMETRY + ICE_DYN_MASKS


@params_pytree
@dataclass(frozen=True)
class SeaiceInitConfig:
    """Initialisation switches and the two densities of SEAICE_INIT_VARIA (float fields are traced leaves: pass the
    object as a jit argument). Defaults: pkg/seaice/seaice_readparms.F, cited per field; the run's data.seaice
    (&SEAICE_PARM01) overrides."""
    SEAICE_rhoIce: float
    SEAICE_rhoSnow: float
    nITD: int
    pickup_seaice: str
    pickupStrictlyMatch: bool
    sIceLoad_on: bool           # useRealFreshWaterFlux .AND. .NOT.useThSIce (seaice_init_varia.F:685)

    @classmethod
    def from_namelists(cls, nml):
        s = lambda k, d: nml.get("data.seaice", "seaice_parm01", k, default=d)  # noqa: E731
        pkg = lambda k: bool(nml.get("data.pkg", "packages", k, default=False))  # noqa: E731  packages_boot.F
        p1 = lambda k, d: nml.get("data", "parm01", k, default=d)  # noqa: E731
        p3 = lambda k, d: nml.get("data", "parm03", k, default=d)  # noqa: E731
        bad = []
        if not pkg("useSEAICE"):
            raise ValueError("useSEAICE=F: no sea-ice initialisation")
        if str(p1("buoyancyRelation", "OCEANIC")).strip().upper() != "OCEANIC":
            bad.append("buoyancyRelation != 'OCEANIC' (seaice_init_varia.F:56-58 kSurface = Nr)")
        if int(s("SEAICE_multDim", 1)) != 1:                            # seaice_readparms.F:433
            bad.append("SEAICE_multDim > 1 (seaice_read_pickup.F:190-199 siTICES)")
        if not bool(s("SEAICEuseMetricTerms", True)):                   # seaice_readparms.F:223
            bad.append("SEAICEuseMetricTerms=F (seaice_init_varia.F:102, 115)")
        if not bool(nml.get("data", "parm04", "usingCurvilinearGrid", default=False)):   # set_defaults.F
            bad.append("usingCurvilinearGrid=F (seaice_init_varia.F:102-146)")
        if bool(s("SEAICEuseBDF2", False)):                            # seaice_readparms.F:296
            bad.append("SEAICEuseBDF2 (seaice_read_pickup.F:263-268 siUicNm1/siVicNm1)")
        evp = (nml.has("data.seaice", "seaice_parm01", "SEAICE_deltaTevp") or bool(s("SEAICEuseEVPstar", False))
               or bool(s("SEAICEuseEVPrev", False)))                     # seaice_readparms.F:695-708
        if evp:
            bad.append("SEAICEuseEVP (seaice_read_pickup.F:270-277 siSigm*)")
        if float(s("SEAICE_tensilFac", 0.0)) != 0.0:                    # seaice_readparms.F:384
            bad.append("SEAICE_tensilFac != 0 (seaice_init_varia.F:697-712 tensileStrFac)")
        for key in ("AreaFile", "HeffFile", "HsnowFile", "uIceFile", "vIceFile"):
            if str(s(key, " ")).strip():
                bad.append(f"{key} (read only for a cold start, seaice_init_varia.F:467-560)")
        if pkg("useThSIce"):
            bad.append("useThSIce (seaice_read_pickup.F:189)")
        nIter0 = int(p3("nIter0", 0))                                    # ini_parms.F:962-974
        if nIter0 == 0:
            bad.append("nIter0 = 0: cold start (seaice_init_varia.F:467-560)")
        if str(p3("pickupSuff", " ")).strip() or int(p3("rwSuffixType", 0)) != 0:
            bad.append("pickupSuff / rwSuffixType (seaice_read_pickup.F:66-74)")
        if bad:
            raise NotImplementedError("sea-ice initialisation branch not ported: " + "; ".join(bad))
        return cls(SEAICE_rhoIce=float(s("SEAICE_rhoIce", 0.91e3)),         # seaice_readparms.F:349
                   SEAICE_rhoSnow=float(s("SEAICE_rhoSnow", 330.0)),        # seaice_readparms.F:350
                   nITD=NITD, pickup_seaice=f"pickup_seaice.{nIter0:010d}",  # seaice_read_pickup.F:68
                   pickupStrictlyMatch=bool(p3("pickupStrictlyMatch", True)),   # set_defaults.F:331
                   sIceLoad_on=bool(p1("useRealFreshWaterFlux", False)) and not pkg("useThSIce"))


# ---------------------------------------------------------------------------------------------------------------------
# static fields (grid-only): HEFFM and the metric coefficients
# ---------------------------------------------------------------------------------------------------------------------
def seaice_geometry(g):
    """seaice_init_varia.F:63-76 (HEFFM from the hFacC of INI_NLFS_VARS = h0FacC, kSurface = 1) and :93-146 (k1AtC,
    k1AtZ, k2AtC, k2AtZ; usingCurvilinearGrid .AND. SEAICEuseMetricTerms), full tile. Returns numpy arrays."""
    L = g.layout
    h0 = np.asarray(g.h0FacC)[:, 0]
    HEFFM = np.where(h0 == 0.0, 0.0, 1.0)                                  # :72-74
    f = {n: np.asarray(g.f[n]) for n in ("recip_dyF", "dyG", "recip_dxF", "recip_dyU", "dyC", "recip_dxV", "dxG",
                                          "dxC")}
    z = np.zeros(L.shape2d)
    k1AtC, k1AtZ, k2AtC, k2AtZ = z.copy(), z.copy(), z.copy(), z.copy()   # :96-99
    Jf, If = L.js(1 - L.OLy, L.sNy + L.OLy), L.is_(1 - L.OLx, L.sNx + L.OLx)
    # :118-123  j = 1-OLy..sNy+OLy, i = 1-OLx..sNx+OLx-1
    I, Ip = L.is_(1 - L.OLx, L.sNx + L.OLx - 1), L.is_(2 - L.OLx, L.sNx + L.OLx)
    k1AtC[:, Jf, I] = (f["recip_dyF"][:, Jf, I] * (f["dyG"][:, Jf, Ip] - f["dyG"][:, Jf, I])
                       * f["recip_dxF"][:, Jf, I])
    # :125-130  j = 1-OLy..sNy+OLy, i = 1-OLx+1..sNx+OLx
    I, Im = L.is_(2 - L.OLx, L.sNx + L.OLx), L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    k1AtZ[:, Jf, I] = (f["recip_dyU"][:, Jf, I] * (f["dyC"][:, Jf, I] - f["dyC"][:, Jf, Im])
                       * f["recip_dxV"][:, Jf, I])
    # :132-137  j = 1-OLy..sNy+OLy-1, i = 1-OLx..sNx+OLx
    J, Jp = L.js(1 - L.OLy, L.sNy + L.OLy - 1), L.js(2 - L.OLy, L.sNy + L.OLy)
    k2AtC[:, J, If] = (f["recip_dxF"][:, J, If] * (f["dxG"][:, Jp, If] - f["dxG"][:, J, If])
                       * f["recip_dyF"][:, J, If])
    # :139-144  j = 1-OLy+1..sNy+OLy, i = 1-OLx..sNx+OLx
    J, Jm = L.js(2 - L.OLy, L.sNy + L.OLy), L.js(1 - L.OLy, L.sNy + L.OLy - 1)
    k2AtZ[:, J, If] = (f["recip_dxV"][:, J, If] * (f["dxC"][:, J, If] - f["dxC"][:, Jm, If])
                       * f["recip_dyU"][:, J, If])
    return {"HEFFM": HEFFM, "k1AtC": k1AtC, "k1AtZ": k1AtZ, "k2AtC": k2AtC, "k2AtZ": k2AtZ}


def seaice_dyn_masks(g, ex, HEFFM):
    """seaice_init_varia.F:378-391 (SEAICE_CGRID): seaiceMaskU/V = 0, then 1 where the two HEFFM values around the
    velocity point sum to more than 1.5, on j = 1-OLy+1..sNy+OLy, i = 1-OLx+1..sNx+OLx (the first halo row/column is
    not written: it keeps the 0 of the SEAICE.h common block until the exchange); :442
    EXCH_UV_XY_RL(seaiceMaskU, seaiceMaskV, .FALSE.) (no sign change). tensileStrFac = 0 on the full tile (:312;
    SEAICE_tensilFac = 0, so :700-712 does not run; SeaiceInitConfig rejects anything else). All three are static
    (see the module docstring). Returns jnp arrays [T, ny, nx]."""
    L = g.layout
    HEFFM = jnp.asarray(HEFFM)
    J, I = L.js(1 - L.OLy + 1, L.sNy + L.OLy), L.is_(1 - L.OLx + 1, L.sNx + L.OLx)            # :380-383
    Jm, Im = L.js(1 - L.OLy, L.sNy + L.OLy - 1), L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    z = jnp.zeros(L.shape2d)
    mask_u = HEFFM[:, J, I] + HEFFM[:, J, Im]                                                  # :386
    seaiceMaskU = z.at[:, J, I].set(jnp.where(mask_u > 1.5, 1.0, 0.0))                         # :384, :387
    mask_v = HEFFM[:, J, I] + HEFFM[:, Jm, I]                                                  # :388
    seaiceMaskV = z.at[:, J, I].set(jnp.where(mask_v > 1.5, 1.0, 0.0))                         # :385, :389
    seaiceMaskU, seaiceMaskV = ex.exch_uv_xy(seaiceMaskU, seaiceMaskV, False)                  # :442
    return {"seaiceMaskU": seaiceMaskU, "seaiceMaskV": seaiceMaskV, "tensileStrFac": z}      # :312


def seaice_fixed_fields(g, ex):
    """The fixed sea-ice fields the kernels read (ICE_FIXED, the dict `sg`): `seaice_geometry` (HEFFM, k1AtC, k1AtZ,
    k2AtC, k2AtZ) and `seaice_dyn_masks` (seaiceMaskU/V, tensileStrFac), from the grid alone. jnp arrays."""
    geo = {k: jnp.asarray(v) for k, v in seaice_geometry(g).items()}
    return {**geo, **seaice_dyn_masks(g, ex, geo["HEFFM"])}


# ---------------------------------------------------------------------------------------------------------------------
# SEAICE_READ_PICKUP (host side: file reads) and the jittable remainder of SEAICE_INIT_VARIA
# ---------------------------------------------------------------------------------------------------------------------
def _to_tiles(a):
    return np.moveaxis(compact_to_tiles(a), -3, 0)


def read_seaice_pickup(rundir, cfg: SeaiceInitConfig):
    """seaice_read_pickup.F:66-298 for the V4r4 case (new-style pickup with a field list, SEAICE_multDim = 1): the
    interiors [T, sNy, sNx] of siTICE (level 1 of TICES), siAREA, siHEFF, siHSNOW, siUICE, siVICE; READ_MFLDS_CHECK +
    SEAICE_CHECK_PICKUP on the missing ones. Returns {TICES1, AREA, HEFF, HSNOW, UICE, VICE}."""
    prefix = Path(rundir) / cfg.pickup_seaice
    if not Path(str(prefix) + ".meta").exists():
        raise NotImplementedError(f"{prefix}: pickup without .meta field list (seaice_read_pickup.F:115-186)")
    arr, meta = read_mds(prefix)
    fldList = tuple(meta.get("fldList", ()))
    nbFields = len(fldList)                                   # READ_MFLDS_SET: nbFields = nFlds
    if nbFields >= 0 and meta["dataprec"] != PREC_META[PRECFLOAT64]:          # :91-101
        raise ValueError(f"SEAICE_READ_PICKUP: pickup-file precision {meta['dataprec']} != float64")
    if nbFields <= 0:                                                        # :103-128, 130-186
        raise NotImplementedError("seaice pickup without a field list is not ported")
    nRecords, nITD = meta["nrecords"], cfg.nITD
    # READ_MFLDS_SET with thirdDim = nITD: nFl3D = (nRecords - nFlds)/(nITD - 1)
    if (nRecords - nbFields) % (nITD - 1) != 0:
        raise ValueError(f"READ_MFLDS_SET: nRecords={nRecords} does not match nFlds={nbFields} (3rd dim {nITD})")
    nFl3D = (nRecords - nbFields) // (nITD - 1)
    missing = []

    def read(name):
        """READ_MFLDS_LEV_RL(name, kLo = kHi = 1) / READ_MFLDS_3D_RL(name, nNz = 1): one 2-D record."""
        if name not in fldList:
            missing.append(name)
            return None
        nj = fldList.index(name) + 1
        if nj > nFl3D:
            nj = nj + nFl3D * (nITD - 1)                  # read_mflds.F:457
        else:
            raise NotImplementedError(f"{name}: a 3-D (nITD) record in the seaice pickup is not ported")
        return _to_tiles(arr[nj - 1])

    out = {"TICES1": read("siTICE")}                      # :202-203 (doMapTice = .TRUE., :201)
    if out["TICES1"] is None:
        out["TICES1"] = read("siTICES")                   # :204-206 (level 1 of siTICES)
    for key, name in (("AREA", "siAREA"), ("HEFF", "siHEFF"), ("HSNOW", "siHSNOW"), ("UICE", "siUICE"),
                      ("VICE", "siVICE")):                # :217-222, :259-262
        out[key] = read(name)
    # SEAICE_CHECK_PICKUP (seaice_check_pickup.F:76-200): siTICE missing is tolerated when siTICES was read
    # (tIceFlag <= 1) and vice versa (tIceFlag <= 2); the other listed fields stop the run ("cannot restart without"),
    # unlisted ones too ("not recognized"); with pickupStrictlyMatch any missing field stops it (:191-200)
    if missing:
        tIceFlag = 2 * ("siTICES" in missing) + ("siTICE" in missing)
        fatal = [m for m in missing if not ((m == "siTICE" and tIceFlag <= 1) or (m == "siTICES" and tIceFlag <= 2))]
        if fatal or cfg.pickupStrictlyMatch:
            raise ValueError(f"SEAICE_CHECK_PICKUP: missing {missing} (fatal {fatal}, pickupStrictlyMatch="
                             f"{cfg.pickupStrictlyMatch})")
    return {k: v for k, v in out.items() if v is not None}


def seaice_init_varia(cfg: SeaiceInitConfig, g, ex, pk):
    """SEAICE_INIT_VARIA for a pickup start (see module docstring), jittable. pk: interiors from read_seaice_pickup.
    Returns the carried sea-ice state (AREA, HEFF, HSNOW, TICES [T, nITD, ny, nx], UICE, VICE) and sIceLoad
    (None when useRealFreshWaterFlux is off)."""
    L = g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    z2 = jnp.zeros(L.shape2d)
    # :253-345 zero initialisation; :419-437 TICES = 273.0 for every category (full tile)
    TICES = jnp.full((L.nTiles, cfg.nITD) + L.shape2d[1:], 273.0)
    # SEAICE_READ_PICKUP: interiors (MDS_READ_FIELD writes i = 1..sNx, j = 1..sNy)
    HEFF = z2.at[:, J, I].set(pk["HEFF"])
    AREA = z2.at[:, J, I].set(pk["AREA"])
    HSNOW = z2.at[:, J, I].set(pk["HSNOW"])
    UICE = z2.at[:, J, I].set(pk["UICE"])
    VICE = z2.at[:, J, I].set(pk["VICE"])
    TICES = TICES.at[:, 0, J, I].set(pk["TICES1"])
    # seaice_read_pickup.F:305-318 doMapTice: TICES(i,j,k) = TICES(i,j,1), k = 2..nITD, interior
    TICES = TICES.at[:, 1:, J, I].set(jnp.broadcast_to(TICES[:, :1, J, I], TICES[:, 1:, J, I].shape))
    # :321-325 exchanges
    UICE, VICE = ex.exch_uv_xy(UICE, VICE, True)                  # EXCH_UV_XY_RL(uIce, vIce, .TRUE.)
    HEFF = ex.exch_xy(HEFF)                                       # _EXCH_XY_RL(HEFF)
    AREA = ex.exch_xy(AREA)                                       # _EXCH_XY_RL(AREA)
    TICES = ex.scalar(TICES, "3D")                                # EXCH_3D_RL(TICES, nITD)
    HSNOW = ex.exch_xy(HSNOW)                                     # _EXCH_XY_RL(HSNOW)
    # seaice_init_varia.F:685-692 (full tile)
    sIceLoad = None
    if cfg.sIceLoad_on:
        sIceLoad = HEFF * cfg.SEAICE_rhoIce + HSNOW * cfg.SEAICE_rhoSnow
    return dict(AREA=AREA, HEFF=HEFF, HSNOW=HSNOW, TICES=TICES, UICE=UICE, VICE=VICE), sIceLoad
