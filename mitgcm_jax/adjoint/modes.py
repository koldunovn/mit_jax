"""Backward-mode semantics of the V4r4 flux-forced adjoint (plan Task 17; the TAF analysis is docs/ADJOINT_MODES.md).

`AdjointConfig` is a static, hashable configuration that says which derivatives the reverse pass keeps. Every switch
acts on derivatives only: the forward values are byte-identical in every mode (tested on the FORCED oracle step). The
default `AdjointConfig()` is the exact mode: no seam is inserted anywhere, the traced program is the one `jax.grad`
has always differentiated.

    adj = AdjointConfig()                    # exact (default)
    adj = AdjointConfig.ecco(RunNamelists(rundir))   # what the TAF adjoint of this build and data.autodiff computes
    st1, aux = forward_step(P, g, ex, kLowC, st0, exf_in, adj=adj)   # adj is static (close over it or mark static)

Switches (value in the ECCO mode of the V4r4 flux-forced build; each seam sits in core/forward_step.py):

  ggl90       "exact" | "frozen"   ecco: "frozen" (data.autodiff useGGL90inAdMode = .FALSE.). TAF skips GGL90_CALC,
              GGL90_CALC_VISC and GGL90_CALC_DIFF in the reverse sweep, but the total vertical diffusivity kappaRk
              (temp_integrate.F:488/505, salt_integrate.F:480/497) and viscosity kappaRU/kappaRV (dynamics.F:399-402)
              are STOREd on the tape after the GGL90 contributions were added, so the linearised implicit
              diffusion/viscosity uses the forward (GGL90-inclusive) coefficients, and no derivative reaches the TKE or
              the coefficients' state dependence. JAX: lax.stop_gradient on the four GGL90_CALC outputs (GGL90TKE,
              GGL90viscArU/V, GGL90diffKr).
  gm_sigma    "exact" | "stable" | "gm_only"   ecco: "stable" (GMREDI_WITH_STABLE_ADJOINT, flux-forced GMREDI_OPTIONS.h:21: TAF's
              ZERO_ADJ_LOC on sigmaX/Y/R, ff do_oceanic_phys.F:900-907). JAX: lax.stop_gradient on sigmaX/Y/R after
              GRAD_SIGMA, before every reader (GGL90_CALC reads sigmaR too, ggl90_calc.F:218-219); rhoInSitu keeps its
              derivative. CALC_IVDC's step function has no derivative in either mode.
              "gm_only" (NOT a TAF mode; added 2026-09-23 at Nikolay's request, the fesom_jax `freeze_gm_slope`
              analogue): lax.stop_gradient on sigmaX/Y/R only where they enter GMREDI_CALC_TENSOR, i.e. only the
              isopycnal slopes (and their taper) lose their dependence on the density field (the 1/N^2 amplifier;
              Forget et al. 2015: "omitting only the parametric dependency of isopycnal slopes on the ocean density
              field"); GGL90_CALC keeps the derivative of its N^2 (sigmaR) unless ggl90="frozen". The dependence of
              the GM/Redi transports on kapGM/kapRedi stays. `ecco()` never selects it.
  salt_plume  "exact" | "off"      ecco (flux-forced): "exact" (useSALT_PLUMEinAdMode = .TRUE.). "off" is the full-V4r4
              setting (useSALT_PLUMEinAdMode = .FALSE.): every IF (useSALT_PLUME) block is skipped in the reverse
              sweep (SALT_PLUME_DO_EXCH, SALT_PLUME_FORCING_SURF, SALT_PLUME_CALC_DEPTH, SALT_PLUME_TENDENCY_APPLY_S),
              i.e. no derivative through saltPlumeFlux or saltPlumeDepth. JAX: lax.stop_gradient on saltPlumeFlux
              where it enters DO_OCEANIC_PHYS and on saltPlumeDepth.
  cg2d        "exact" | "passive"  ecco: "passive" (pkg/autodiff/cg2d.flow:7-12: only cg2d_b and cg2d_x are active; the
              operator aW2d/aS2d/aC2d is a common block TAF does not differentiate). JAX: Cg2dParams.stop_coeff_grad.
  visc_fac_in_ad  None | float     ecco: data.autodiff viscFacInAd (default 1.0, autodiff_readparms.F:73; the V4r4
              trees do not set it). TAF sets viscFacAdj = viscFacInAd for the whole reverse sweep
              (autodiff_inadmode_set_ad.F:53) and recomputes MOM_CALC_VISC there (no STORE of the viscosities), so the
              adjoint of MOM_VECINV is taken with the 3-D viscosity file fields scaled by viscFacInAd before clipping
              (V4r4 mom_calc_visc.F:406,425,516,535). JAX: `differentiate_at` around MOM_VECINV: value at the forward
              viscosities, derivatives (JVP and VJP) at the viscosities of MOM_CALC_VISC with viscFacAdj=viscFacInAd.
              None = no seam; 1.0 = seam present, derivatives bitwise equal to the exact mode (tested).

  seaice      "ecco" | "no_dynamics" | "full"   (full V4r4 tree only; ignored by the flux-forced step, which has no sea
              ice). DEFAULT "ecco", also in the otherwise exact `AdjointConfig()` (Nikolay, 2026-09-23: the ECCO semantics
              is the sea-ice default everywhere; the exact sea-ice adjoint is an explicit choice, seaice="full").
              "ecco" = data.autodiff useSEAICEinAdMode = .FALSE. (the V4r4 setting): autodiff_inadmode_set_ad.F:37
              useSEAICE = .FALSE. in the reverse sweep, so the IF (useSEAICE) block around SEAICE_MODEL
              (c66g do_oceanic_phys.F:397-481) is skipped: the adjoint of SEAICE_MODEL is the identity on every variable
              it overwrites and adds nothing to those it only reads (NOT a stop_gradient); SEAICEapproxLevInAd =
              MIN(0, 0) = 0 (autodiff_readparms.F:124-125), so SEAICE_FAKE (needs -1) is not called either.
              "no_dynamics" = useSEAICEinAdMode = .TRUE. with SEAICEuseDYNAMICSswitchInAd = .TRUE.
              (autodiff_inadmode_set_ad.F:49-51): only the IF (SEAICEuseDYNAMICS) blocks (FREEDRIFT + LSR, clipping) are
              skipped in the reverse sweep. "full" = the exact derivative (LSR by its implicit derivative).
              Mechanism and tests: pkgs/seaice_model.py (ops/ad_skip.skipped_in_reverse); the seam sits at the
              SEAICE_MODEL call in core/forward_step.do_oceanic_phys.

Full V4r4 tree (`ecco()` of its data.autodiff: useSEAICEinAdMode = useGGL90inAdMode = useSALT_PLUMEinAdMode =
.FALSE.; code/GMREDI_OPTIONS.h:21 GMREDI_WITH_STABLE_ADJOINT as in the flux-forced tree): seaice "ecco", ggl90
"frozen", salt_plume "off", gm_sigma "stable", cg2d "passive", visc_fac_in_ad 1.0 (STDOUT.0000 of full_jaxdump_v5
prints exactly these switch values). The salt-plume seam sits where the full tree's saltPlumeFlux is SET: c66g
do_oceanic_phys.F:293 zeroes it (ALLOW_AUTODIFF, no IF (useSALT_PLUME)), V4r4 SEAICE_GROWTH writes it (seaice_growth.F:
2032, only #ifdef ALLOW_SALT_PLUME: its adjoint runs but receives 0), and every reader is an IF (useSALT_PLUME) block
the reverse sweep skips (SALT_PLUME_DO_EXCH do_oceanic_phys.F:573-576, SALT_PLUME_FORCING_SURF external_forcing_surf.F:
235-239, SALT_PLUME_TENDENCY_APPLY_S apply_forcing.F:931-936), so the stop_gradient goes on SEAICE_MODEL's
saltPlumeFlux output (before SALT_PLUME_DO_EXCH) instead of the DO_OCEANIC_PHYS entry value (which the full tree zeroes),
plus the same one on SALT_PLUME_CALC_DEPTH's output (c66g do_oceanic_phys.F:941-944).

Not switches here: KPP (not used), GMRedi in the adjoint (useGMRediInAdMode = T, default), inAdExact (= T, default:
inAdMode stays .FALSE. in the reverse sweep, autodiff_readparms.F:98-104). Settings that would need an unported
branch raise NotImplementedError in `AdjointConfig.ecco`.

Forward-mode (JVP) note: TAF's tangent-linear model keeps every package (g_autodiff_inadmode_set.F only sets
inAdMode = .FALSE.). A non-exact AdjointConfig also cuts tangents (stop_gradient is a zero tangent), so use the exact
mode for tangent-linear work unless the ECCO approximation is wanted there too.
"""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
from jax import lax

# flux-forced/code/GMREDI_OPTIONS.h:21 = full code/GMREDI_OPTIONS.h:21 (the two files are identical)
#   #define GMREDI_WITH_STABLE_ADJOINT  (ZERO_ADJ_LOC on sigmaX/Y/R in the adjoint)
GMREDI_WITH_STABLE_ADJOINT = True

_CHOICES = {"ggl90": ("exact", "frozen"), "gm_sigma": ("exact", "stable", "gm_only"), "salt_plume": ("exact", "off"),
            "cg2d": ("exact", "passive"), "seaice": ("ecco", "no_dynamics", "full")}


@dataclass(frozen=True)
class AdjointConfig:
    """Static reverse-mode semantics per package (see the module docstring). Hashable; the default is exact."""
    ggl90: str = "exact"
    gm_sigma: str = "exact"
    salt_plume: str = "exact"
    cg2d: str = "exact"
    visc_fac_in_ad: Optional[float] = None
    seaice: str = "ecco"       # the ECCO sea-ice adjoint is the default even here (module docstring)

    def __post_init__(self):
        for name, allowed in _CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"AdjointConfig.{name} = {getattr(self, name)!r}; allowed: {allowed}")
        if self.visc_fac_in_ad is not None:
            object.__setattr__(self, "visc_fac_in_ad", float(self.visc_fac_in_ad))

    @property
    def is_exact(self):
        """True for the default configuration: no ocean seam; the sea-ice level is the default "ecco"."""
        return self == AdjointConfig()

    @classmethod
    def exact(cls):
        return cls()

    @classmethod
    def ecco(cls, nml):
        """The reverse-mode semantics of the TAF adjoint of this build with the run's data.autodiff and data.pkg
        (nml: params_io.RunNamelists of the run directory). Defaults are autodiff_readparms.F:66-73; the
        in-adjoint package switches are ANDed with the forward ones (autodiff_readparms.F:116-120)."""
        if not nml.file("data.autodiff"):
            # autodiff_readparms.F:83-89 OPEN_COPY_DATA_FILE('data.autodiff') stops when the file is missing
            raise FileNotFoundError(f"{nml.dir}/data.autodiff: the TAF adjoint reads it (autodiff_readparms.F:83)")
        ad = lambda key, default: nml.get("data.autodiff", "autodiff_parm01", key, default=default)  # noqa: E731
        pkg = lambda key: bool(nml.get("data.pkg", "packages", key, default=False))  # noqa: E731  packages_boot.F
        if not ad("inAdExact", True):  # autodiff_readparms.F:71
            # inAdMode = .TRUE. in the reverse sweep only changes DST3-flux-limited advection (gad_calc_rhs.F:245,
            # 385, 516, 559), which V4r4 does not use; not ported
            raise NotImplementedError("inAdExact = .FALSE. (inAdMode branches) is not ported")
        if pkg("useKPP"):
            raise NotImplementedError("useKPP: pkg/kpp is not ported")
        useGMRedi, useGGL90, useSALT_PLUME = pkg("useGMRedi"), pkg("useGGL90"), pkg("useSALT_PLUME")
        if useGMRedi and not ad("useGMRediInAdMode", True):  # autodiff_readparms.F:67
            raise NotImplementedError("useGMRediInAdMode = .FALSE. (GM/Redi off in the adjoint) is not ported")
        ggl_in_ad = ad("useGGL90inAdMode", True) and useGGL90  # autodiff_readparms.F:69, 119
        sp_in_ad = ad("useSALT_PLUMEinAdMode", True) and useSALT_PLUME  # autodiff_readparms.F:70, 120
        if pkg("useSEAICE"):
            # useSEAICEinAdMode .AND. useSEAICE (autodiff_readparms.F:68, 118), SEAICEuseDYNAMICSswitchInAd (:77),
            # SEAICEapproxLevInAd (:72, 124-127); unported settings raise there (pkgs/seaice_model.ad_level)
            from mitgcm_jax.pkgs.seaice_model import ad_level
            seaice = ad_level(nml)
        else:
            # no sea ice in this build/run: the switches act only inside IF (useSEAICE) code (the field is unused)
            for key in ("SEAICEuseFREEDRIFTswitchInAd", "SEAICEuseDYNAMICSswitchInAd"):  # autodiff_readparms.F:76-77
                if ad(key, False):
                    raise NotImplementedError(f"{key} set with useSEAICE = .FALSE.")
            seaice = "ecco"
        return cls(
            ggl90="frozen" if (useGGL90 and not ggl_in_ad) else "exact",
            gm_sigma="stable" if (useGMRedi and GMREDI_WITH_STABLE_ADJOINT) else "exact",
            salt_plume="off" if (useSALT_PLUME and not sp_in_ad) else "exact",
            cg2d="passive",  # pkg/autodiff/cg2d.flow:7-12 (every TAF build)
            visc_fac_in_ad=float(ad("viscFacInAd", 1.0)),  # autodiff_readparms.F:73
            seaice=seaice,
        )


def stop_gradient_if(flag, *xs):
    """lax.stop_gradient on every array of `xs` when `flag` (a static Python bool), else `xs` unchanged. The forward
    value is untouched (stop_gradient lowers to nothing)."""
    if not flag:
        return xs if len(xs) != 1 else xs[0]
    out = tuple(jax.tree_util.tree_map(lax.stop_gradient, x) for x in xs)
    return out if len(out) != 1 else out[0]


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def differentiate_at(fn, args, alt):
    """Value `fn(*args)`; derivatives (JVP, and the VJP JAX derives from it by transposition) of `fn` taken at the
    point `alt` (same pytree structure as `args`) instead of `args`. Tangents/cotangents are those of `args`; `alt`
    carries none. `fn` must be a module-level function (no closed-over tracers): pass every array through `args`.

    TAF analogue: a routine the reverse sweep recomputes with different parameters (viscFacAdj = viscFacInAd inside
    MOM_CALC_VISC). With alt == args the derivatives are those of plain autodiff (bitwise, tested)."""
    return fn(*args)


@differentiate_at.defjvp
def _differentiate_at_jvp(fn, primals, tangents):
    args, alt = primals
    t_args, _ = tangents
    out = fn(*args)
    _, t_out = jax.jvp(lambda *a: fn(*a), tuple(alt), tuple(t_args))
    return out, t_out


def visc_params_in_ad(pvisc, factor):
    """MomViscParams as MOM_CALC_VISC sees them in the TAF reverse sweep: viscFacAdj = viscFacInAd
    (autodiff_inadmode_set_ad.F:53; forward value 1, set_defaults.F:131)."""
    return dataclasses.replace(pvisc, viscFacAdj=factor)


EXACT = AdjointConfig()

__all__ = ["AdjointConfig", "EXACT", "GMREDI_WITH_STABLE_ADJOINT", "differentiate_at", "stop_gradient_if",
           "visc_params_in_ad"]
