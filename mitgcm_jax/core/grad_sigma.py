"""Density, its gradients and convection flags: the part of DO_OCEANIC_PHYS before the vertical-mixing packages
(plan Task 10). Literal port of

    flux-forced/code/do_oceanic_phys.F:640-948   (tile loop: work-array init, FIND_RHO_2D x Nr, the k loop Nr..1 with
                                                  FIND_RHO_2D @ p(k), GRAD_SIGMA, CALC_IVDC, then CALC_OCE_MXLAYER)
    model/src/grad_sigma.F                       GRAD_SIGMA
    eesupp/src/fill_cs_corner_tr_rl.F            FILL_CS_CORNER_TR_RL (called by GRAD_SIGMA, exch2 corners)

V4r4 switches (data, data.pkg, data.gmredi): calcGMRedi = useGMRedi = T, calcConvect = (ivdc_kappa=10 /= 0) = T,
useGGL90 = useSALT_PLUME = T, useDiagnostics = F (doDiagsRho = 0), no DOWN_SLOPE / BBL / OFFLINE. The k-loop condition
(do_oceanic_phys.F:857-860) therefore holds at every k, so FIND_RHO_2D @ p(k) and GRAD_SIGMA run at every level.

The k loop has no recurrence (each level reads only rhoInSitu(k) and FIND_RHO_2D(theta(k-1), salt(k-1), kRef=k)),
so it is vectorised over k. The only state carried between iterations is rhoKm1, which at k = 1 still holds the k = 2
value; GRAD_SIGMA does not read it at k = 1 (sigmaR(k=1) = 0, grad_sigma.F:93-98), so level 1 of sigKm1 is 0 here.

ecco-mode seam (NOT implemented): GMREDI_WITH_STABLE_ADJOINT is defined (flux-forced GMREDI_OPTIONS.h:21), so
do_oceanic_phys.F:900-908 calls ZERO_ADJ_LOC(sigmaX / sigmaY / sigmaR) after every GRAD_SIGMA. Forward: nothing
(pkg/autodiff/zero_adj.F:45-70 is an empty routine). Adjoint (TAF): the adjoint of the whole sigmaX/Y/R arrays is
zeroed in every k iteration; every reader of sigma (CALC_IVDC's step function, GGL90 is not a reader, GMREDI slopes)
runs after the k loop, so in ecco mode the equivalent is jax.lax.stop_gradient on the three sigma arrays returned by
`rho_sigma_ivdc_mxlayer`; rhoInSitu keeps its gradient (phi_hyd and the salt-plume depth read it).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.core import eos as eos_mod
from mitgcm_jax.core.ivdc import calc_ivdc
from mitgcm_jax.core.mxlayer import MxLayerParams, calc_oce_mxlayer

# ini_vertical_grid.F:56  rkSign = -1 (z coordinates: k and r in opposite sense)
RKSIGN = -1.0
# ini_vertical_grid.F:57  gravitySign = -1 (z coordinates)
GRAVITYSIGN = -1.0

# exch2 topology of the 13-tile LLC90 run: reference/data.exch2_13x90x90, dimsFacets(1:10)
FACET_DIMS_LLC90 = ((90, 270), (90, 270), (90, 90), (270, 90), (270, 90))


def exch2_edges(layout, facet_dims=FACET_DIMS_LLC90):
    """exch2_isWedge/isEedge/isSedge/isNedge per tile (0-based tile axis), as bool arrays [nTiles].
    Tile ids: facet by facet, x fastest (w2_set_map_tiles.F:161-184); a tile edge is a facet edge when it has no
    neighbour inside its facet (w2_set_tile2tiles.F:88-110), independent of the facet-to-facet connection."""
    L = layout
    W, E, S, N = [], [], [], []
    for fNx, fNy in facet_dims:
        nbTx, nbTy = fNx // L.sNx, fNy // L.sNy
        for ty in range(1, nbTy + 1):
            for tx in range(1, nbTx + 1):
                iLo, iHi = (tx - 1) * L.sNx + 1, tx * L.sNx
                jLo, jHi = (ty - 1) * L.sNy + 1, ty * L.sNy
                N.append(not jHi < fNy)   # w2_set_tile2tiles.F:91-92
                S.append(not jLo > 1)     # w2_set_tile2tiles.F:97-98
                E.append(not iHi < fNx)   # w2_set_tile2tiles.F:103-104
                W.append(not iLo > 1)     # w2_set_tile2tiles.F:109-110
    if len(W) != L.nTiles:
        raise ValueError(f"facet dims give {len(W)} tiles, layout has {L.nTiles}")
    return {k: np.array(v) for k, v in (("W", W), ("E", E), ("S", S), ("N", N))}


def cs_corners(layout, facet_dims=FACET_DIMS_LLC90):
    """fill_cs_corner_tr_rl.F:77-85: southWest/southEast/northEast/northWest corner flags per tile."""
    e = exch2_edges(layout, facet_dims)
    return {"SW": e["W"] & e["S"], "SE": e["E"] & e["S"], "NE": e["E"] & e["N"], "NW": e["W"] & e["N"]}


def _corner_maps(L, fill4dir):
    """(corner, target (y, x), source (y, x)) index arrays [OLy, OLx] of the Fortran loops j=1..OLy, i=1..OLx.
    Fortran trFld(x, y); Python arrays [..., y, x]."""
    j, i = np.meshgrid(np.arange(1, L.OLy + 1), np.arange(1, L.OLx + 1), indexing="ij")
    sNx, sNy = L.sNx, L.sNy
    if fill4dir == 1:
        spec = (  # fill_cs_corner_tr_rl.F:165-192
            ("SW", (1 - j, 1 - i), (i, 1 - j)),              # trFld( 1-i , 1-j ) = trFld( 1-j , i  )
            ("SE", (1 - j, sNx + i), (i, sNx + j)),          # trFld(sNx+i, 1-j ) = trFld(sNx+j, i  )
            ("NW", (sNy + j, 1 - i), (sNy + 1 - i, 1 - j)),  # trFld( 1-i ,sNy+j) = trFld( 1-j , sNy+1-i )
            ("NE", (sNy + j, sNx + i), (sNy + 1 - i, sNx + j)),  # trFld(sNx+i,sNy+j) = trFld(sNx+j, sNy+1-i )
        )
    elif fill4dir == 2:
        spec = (  # fill_cs_corner_tr_rl.F:235-262
            ("SW", (1 - j, 1 - i), (1 - i, j)),              # trFld( 1-i , 1-j ) = trFld(   j   , 1-i )
            ("SE", (1 - j, sNx + i), (1 - i, sNx + 1 - j)),  # trFld(sNx+i, 1-j ) = trFld(sNx+1-j, 1-i )
            ("NW", (sNy + j, 1 - i), (sNy + i, j)),          # trFld( 1-i ,sNy+j) = trFld(   j   ,sNy+i)
            ("NE", (sNy + j, sNx + i), (sNy + i, sNx + 1 - j)),  # trFld(sNx+i,sNy+j) = trFld(sNx+1-j,sNy+i)
        )
    else:
        raise NotImplementedError(f"FILL_CS_CORNER_TR_RL fill4dir={fill4dir}: only 1 and 2 are used here")
    out = []
    for name, (ty, tx), (sy, sx) in spec:
        out.append((name, (ty - 1 + L.OLy, tx - 1 + L.OLx), (sy - 1 + L.OLy, sx - 1 + L.OLx)))
    return out


def fill_cs_corner_tr(trFld, fill4dir, withSigns, layout, facet_dims=FACET_DIMS_LLC90):
    """FILL_CS_CORNER_TR_RL (fill_cs_corner_tr_rl.F:12-273) on [tile, ..., ny, nx] arrays, all tiles at once.
    useCubedSphereExchange = T (exch2). Corners are processed in the Fortran order SW, SE, NW, NE."""
    L = layout
    negOne = -1.0 if withSigns else 1.0  # fill_cs_corner_tr_rl.F:71-72
    flags = cs_corners(L, facet_dims)
    for name, (ty, tx), (sy, sx) in _corner_maps(L, fill4dir):
        f = flags[name]
        if not f.any():
            continue
        filled = trFld.at[..., ty, tx].set(negOne * trFld[..., sy, sx])
        sel = jnp.asarray(f).reshape((L.nTiles,) + (1,) * (trFld.ndim - 1))
        trFld = jnp.where(sel, filled, trFld)
    return trFld


def grad_sigma(g, rhoK, sigKm1, sigKp1):
    """GRAD_SIGMA (grad_sigma.F:9-110) for all levels at once: rhoK = rhoInSitu(k), sigKm1 = rho(theta(k-1), p(k)),
    sigKp1 = rho(theta(k), p(k)), each [T, Nr, ny, nx] (level axis = Fortran k). Returns sigmaX, sigmaY, sigmaR
    [T, Nr, ny, nx]; points GRAD_SIGMA does not write keep the zero of do_oceanic_phys.F:645-654."""
    L = g.layout
    zero = jnp.zeros_like(rhoK)
    # grad_sigma.F:57-61 local copy; :65-68 corner fill for X
    rhoLoc = fill_cs_corner_tr(rhoK, 1, False, L)
    # grad_sigma.F:70-76  j = 1-OLy..sNy+OLy, i = 1-OLx+1..sNx+OLx
    J, I = L.js(1 - L.OLy, L.sNy + L.OLy), L.is_(1 - L.OLx + 1, L.sNx + L.OLx)
    Im1 = L.is_(1 - L.OLx, L.sNx + L.OLx - 1)
    sigmaX = zero.at[..., J, I].set(g.maskW[..., J, I]
                                   * g.recip_dxC[:, None, J, I]
                                   * (rhoLoc[..., J, I] - rhoLoc[..., J, Im1]))
    # grad_sigma.F:80-83 corner fill for Y (on the X-filled local copy)
    rhoLoc = fill_cs_corner_tr(rhoLoc, 2, False, L)
    # grad_sigma.F:85-91  j = 1-OLy+1..sNy+OLy, i = 1-OLx..sNx+OLx
    J, I = L.js(1 - L.OLy + 1, L.sNy + L.OLy), L.is_(1 - L.OLx, L.sNx + L.OLx)
    Jm1 = L.js(1 - L.OLy, L.sNy + L.OLy - 1)
    sigmaY = zero.at[..., J, I].set(g.maskS[..., J, I]
                                   * g.recip_dyC[:, None, J, I]
                                   * (rhoLoc[..., J, I] - rhoLoc[..., Jm1, I]))
    # grad_sigma.F:93-107  k = 1: 0; k > 1: maskC(k)*maskC(k-1)*recip_drC(k)*rkSign*(sigKp1 - sigKm1)
    recip_drC = jnp.asarray(g.recip_drC)[1:L.Nr]  # recip_drC(k), k = 2..Nr
    sigmaR = zero.at[:, 1:].set(g.maskC[:, 1:] * g.maskC[:, :-1]
                                * recip_drC[None, :, None, None]
                                * RKSIGN
                                * (sigKp1[:, 1:] - sigKm1[:, 1:]))
    return sigmaX, sigmaY, sigmaR


@dataclass(frozen=True)
class RhoSigmaParams:
    """Switches of do_oceanic_phys.F:249-253, 857-860, 912, 941 (static) and the EOS / mixed-layer parameters
    (pytrees: pass the params object as a jit argument so float parameters stay traced)."""
    eos: eos_mod.EOSParams
    calcConvect: bool
    mxl: MxLayerParams

    @classmethod
    def from_namelists(cls, nml, g):
        e = eos_mod.EOSParams.from_namelists(nml, g.rC, g.rF)
        pkg = {k: bool(nml.get("data.pkg", "packages", k, default=False))  # packages_boot.F:106-160: all .FALSE.
               for k in ("useGMRedi", "useKPP", "usePP81", "useKL10", "useMY82", "useGGL90", "useSALT_PLUME",
                         "useDiagnostics", "useDOWN_SLOPE", "useBBL", "useOffLine")}
        for k in ("useDOWN_SLOPE", "useBBL", "useOffLine"):
            if pkg[k]:
                raise NotImplementedError(f"{k}=T: DWNSLP_CALC_RHO / BBL_CALC_RHO / offline paths are not ported")
        if pkg["useDiagnostics"]:
            raise NotImplementedError("useDiagnostics=T: doDiagsRho paths (DIAGS_RHO_L, MXLDEPTH) are not ported")
        # set_defaults.F:216 ivdc_kappa = 0.; do_oceanic_phys.F:253 calcConvect = ivdc_kappa.NE.0.
        calcConvect = float(nml.get("data", "parm01", "ivdc_kappa", default=0.0)) != 0.0
        calcGMRedi = pkg["useGMRedi"]  # do_oceanic_phys.F:251 (no ALLOW_OFFLINE)
        # do_oceanic_phys.F:857-860 without the k-dependent (k > 1 .AND. calcConvect) term: must hold at k = 1 too
        anyk = (calcGMRedi or pkg["usePP81"] or pkg["useKL10"] or pkg["useMY82"] or pkg["useGGL90"]
                or pkg["useSALT_PLUME"])
        if not anyk:
            raise NotImplementedError("do_oceanic_phys.F:857: GRAD_SIGMA skipped at some k: branch not ported")
        if not calcConvect:
            raise NotImplementedError("ivdc_kappa = 0: CALC_IVDC skipped: branch not ported")
        if not calcGMRedi:
            raise NotImplementedError("useGMRedi=F: CALC_OCE_MXLAYER not called (do_oceanic_phys.F:941): not ported")
        return cls(eos=e, calcConvect=calcConvect, mxl=MxLayerParams.from_namelists(nml))


jax.tree_util.register_dataclass(RhoSigmaParams, data_fields=["eos", "mxl"], meta_fields=["calcConvect"])


def rho_sigma_ivdc_mxlayer(p, g, theta, salt, hMixLayer):
    """do_oceanic_phys.F(ff):640-945 for all tiles: inputs theta, salt [T, Nr, ny, nx] (state at the start of the
    step, halos included) and hMixLayer [T, ny, nx] (its value before the call). Returns a dict with rhoInSitu,
    sigmaX, sigmaY, sigmaR, IVDConvCount ([T, Nr, ny, nx]) and hMixLayer."""
    L = g.layout
    Nr = L.Nr
    # do_oceanic_phys.F:797-805  rhoInSitu(k) = FIND_RHO_2D(theta(k), salt(k), kRef = k), k = 1..Nr, full tile
    rhoInSitu = eos_mod.find_rho_levels(p.eos, theta, salt, np.arange(1, Nr + 1))
    # do_oceanic_phys.F:861-876  rhoKm1 = FIND_RHO_2D(theta(k-1), salt(k-1), kRef = k), k = 2..Nr
    rhoKm1 = eos_mod.find_rho_levels(p.eos, theta[:, :-1], salt[:, :-1], np.arange(2, Nr + 1))
    sigKm1 = jnp.concatenate([jnp.zeros_like(rhoInSitu[:, :1]), rhoKm1], axis=1)
    # do_oceanic_phys.F:881-885  rhoKp1 = rhoInSitu(k) (copy "to avoid aliasing"), then GRAD_SIGMA (:886-890)
    rhoKp1 = rhoInSitu
    sigmaX, sigmaY, sigmaR = grad_sigma(g, rhoInSitu, sigKm1, rhoKp1)
    # do_oceanic_phys.F:686-692 IVDConvCount = 0; :912-920 CALC_IVDC for k > 1
    IVDConvCount = calc_ivdc(sigmaR, GRAVITYSIGN)
    # do_oceanic_phys.F:941-945
    hMixLayer = calc_oce_mxlayer(p.mxl, hMixLayer)
    return {"rhoInSitu": rhoInSitu, "sigmaX": sigmaX, "sigmaY": sigmaY, "sigmaR": sigmaR,
            "IVDConvCount": IVDConvCount, "hMixLayer": hMixLayer}


__all__ = ["RhoSigmaParams", "rho_sigma_ivdc_mxlayer", "grad_sigma", "fill_cs_corner_tr", "cs_corners",
           "exch2_edges"]
