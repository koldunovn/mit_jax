"""TRACERS_CORRECTION_STEP (model/src/tracers_correction_step.F), plan Task 16b.

In the V4r4 flux-forced build this routine changes nothing:
  - ALLOW_NONHYDROSTATIC is undef (ff code/CPP_OPTIONS.h:50): no TRACERS_IIGW_CORRECTION (tracers_correction_step.F:57);
  - shap_filt, zonal_filt, fizhi, opps, matrix are not in packages.conf: ALLOW_SHAP_FILT / ALLOW_ZONAL_FILT /
    ALLOW_FIZHI / ALLOW_OPPS / ALLOW_MATRIX undefined (tracers_correction_step.F:67-90, :96-112);
  - CONVECTIVE_ADJUSTMENT runs only IF ( cAdjFreq .NE. 0. ) (tracers_correction_step.F:103); V4r4 leaves cAdjFreq at
    its default 0 (set_defaults.F:320; it uses ivdc_kappa instead, and ini_parms.F:948 forbids both).
No exchange is called. `TracersCorrectionParams.from_namelists` raises NotImplementedError for any other setting.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TracersCorrectionParams:
    cAdjFreq: float = 0.0

    @classmethod
    def from_namelists(cls, nml):
        cAdjFreq = float(nml.get("data", "parm03", "cAdjFreq", default=0.0))  # set_defaults.F:320
        if cAdjFreq != 0.0:  # tracers_correction_step.F:103 (ini_parms.F:944-946 turns <0 into deltaTClock)
            raise NotImplementedError("CONVECTIVE_ADJUSTMENT (cAdjFreq != 0) is not ported")
        if nml.get("data", "parm01", "implicitIntGravWave", default=False):  # set_defaults.F:180
            raise NotImplementedError("implicitIntGravWave needs ALLOW_NONHYDROSTATIC (undef in V4r4)")
        for n in ("useSHAP_FILT", "useZONAL_FILT", "useFIZHI", "useOPPS", "useMATRIX"):
            if nml.get("data.pkg", "packages", n, default=False):
                raise NotImplementedError(f"{n}=T: package not compiled in the V4r4 ff build")
        return cls(cAdjFreq=cAdjFreq)


def tracers_correction_step(p, theta, salt):
    """tracers_correction_step.F:44-117 on the V4r4 branches: theta and salt are returned unchanged."""
    return theta, salt
