"""Ocean mixed-layer depth hMixLayer (plan Task 10): model/src/calc_oce_mxlayer.F (CALC_OCE_MXLAYER).

do_oceanic_phys.F(ff):941-945 calls CALC_OCE_MXLAYER (calcGMRedi = T), but inside it computes nothing in V4r4:
    calcMixLayerDepth = GM_useSubMeso .OR. GM_taper_scheme.EQ.'fm07' .OR. GM_useK3D     (calc_oce_mxlayer.F:66-72)
                        [useGMRedi = T, useKPP = F]; data.gmredi sets GM_taper_scheme = 'stableGmAdjTap' and leaves
                        GM_useSubMeso = GM_useK3D = .FALSE. (gmredi_readparms.F:136, 143)
    useDiagnostics = F                                                                  (calc_oce_mxlayer.F:73-77)
so hMixLayer keeps its previous value (initialised in INI_DYNVARS, never written by the V4r4 forward). The two
depth methods (hMixCriteria < 0 with FIND_ALPHA, > 1 with sigmaR) and hMixSmooth are not ported: selecting them is a
hard error at set-up.
"""

from dataclasses import dataclass

from mitgcm_jax.params_io import params_pytree


@params_pytree
@dataclass(frozen=True)
class MxLayerParams:
    calcMixLayerDepth: bool

    @classmethod
    def from_namelists(cls, nml):
        useGMRedi = bool(nml.get("data.pkg", "packages", "useGMRedi", default=False))  # packages_boot.F
        useKPP = bool(nml.get("data.pkg", "packages", "useKPP", default=False))        # packages_boot.F
        useDiagnostics = bool(nml.get("data.pkg", "packages", "useDiagnostics", default=False))
        calc = False                                                                    # calc_oce_mxlayer.F:66
        if useGMRedi and not useKPP:                                                    # calc_oce_mxlayer.F:68
            grp = "gm_parm01"
            subMeso = bool(nml.get("data.gmredi", grp, "GM_useSubMeso", default=False))  # gmredi_readparms.F:136
            k3d = bool(nml.get("data.gmredi", grp, "GM_useK3D", default=False))          # gmredi_readparms.F:143
            taper = nml.get("data.gmredi", grp, "GM_taper_scheme", default=" ")          # gmredi_readparms.F:103
            calc = subMeso or taper == "fm07" or k3d                                    # calc_oce_mxlayer.F:69-70
        if useDiagnostics:                                                              # calc_oce_mxlayer.F:74-76
            raise NotImplementedError("useDiagnostics=T: MXLDEPTH diagnostic path of CALC_OCE_MXLAYER not ported")
        if calc:
            raise NotImplementedError("calcMixLayerDepth=T (GM_useSubMeso / fm07 / GM_useK3D): not a V4r4 branch")
        return cls(calcMixLayerDepth=calc)


def calc_oce_mxlayer(p, hMixLayer):
    """CALC_OCE_MXLAYER with calcMixLayerDepth = .FALSE. (calc_oce_mxlayer.F:78, 224): hMixLayer unchanged."""
    if p.calcMixLayerDepth:
        raise NotImplementedError("calcMixLayerDepth=T is not ported")
    return hMixLayer
