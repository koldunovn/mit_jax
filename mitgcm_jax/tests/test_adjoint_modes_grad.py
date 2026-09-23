"""Backward-mode semantics (plan Task 17): gradient effect tests on the live FORCED oracle state (tier1x).

Each ECCO switch changes the reverse pass where its physics is active and nowhere else, and never the forward:
  - DO_OCEANIC_PHYS (the step's own function, forward_step.do_oceanic_phys) with gm_sigma="stable",
    ggl90="frozen", salt_plume="off" against the exact mode: the GM/Redi tensor, the GGL90 outputs and the
    saltPlumeFlux path lose their derivatives exactly (structural zeros), rhoInSitu keeps its derivative bitwise,
    the forward outputs are bitwise equal.
  - gm_sigma="gm_only" (not a TAF mode): only the GM/Redi slopes are cut; GGL90's N^2 derivative stays (bitwise the
    exact GGL90 path), which "stable" cuts.
  - MOM_VECINV with the viscFacInAd seam (forward_step.mom_vecinv_adj): factor 1 gives the exact gradient bitwise,
    factor 2 gives a different gradient, equal to plain autodiff of MOM_VECINV at the viscosities of MOM_CALC_VISC
    with viscFacAdj = 2 (the TAF recomputation); the forward is bitwise the exact one.
  - One full step, J = sum of theta at the end of the step over a box: gradient finite everywhere in the exact and
    the ECCO mode; dJ/dGGL90TKE is 0 in the ECCO mode and not in the exact mode; dJ/dtheta differs.
Measured numbers (2026-09-23) are in each test's docstring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.adjoint.modes import EXACT, AdjointConfig, visc_params_in_ad
from mitgcm_jax.core import dynamics as dyn_mod
from mitgcm_jax.core.forward_step import _mom_vecinv_visc, do_oceanic_phys, forward_step, mom_vecinv_adj
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.model import setup
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import exf_fluxforced as exf_mod
from mitgcm_jax.pkgs import mom_common as mc
from mitgcm_jax.state import State, state_from_dump
from mitgcm_jax.tests import oracle

PHYS_ON = AdjointConfig(ggl90="frozen", gm_sigma="stable", salt_plume="off")
PHYS_IN = ("theta", "salt", "uVel", "vVel", "GGL90TKE", "saltPlumeFlux")
CT_SETS = {"gm": ("Kwx", "Kwy", "Kwz", "GM_PsiX", "GM_PsiY"),
           "ggl": ("GGL90TKE", "GGL90viscArU", "GGL90viscArV", "GGL90diffKr"),
           "salt": ("surfaceForcingS",),
           "rho": ("rhoInSitu",)}
PHYS_OUT = sum(CT_SETS.values(), ()) + ("saltPlumeDepth", "IVDConvCount")
MV_IN = ("uVel", "vVel", "wVel", "hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS")


@pytest.fixture(scope="module")
def run():
    ds = oracle.dumpset(oracle.FORCED)
    rundir = oracle.run_dir(oracle.FORCED)
    P, g, ex, kLowC = setup(rundir, grid=grid_from_dump(ds, 1))
    st = state_from_dump(ds, 1)
    st = st.add(runoff=np.asarray(exf_mod.exf_init_varia(P.exf, g.layout)["runoff"]))
    nml = RunNamelists(rundir)
    myTime, myIter = exf_mod.model_time(nml, 1)
    bufs, facs, _ = exf_mod.ExfRecordLoader(P.exf, g, rundir).load(myTime, myIter)
    exf_in = {"bufs": bufs, "facs": facs, "myTime": myTime}
    return ds, nml, P, g, ex, kLowC, st, exf_in


def _phys_vjp(ex, adj):
    def fphys(x, f, P, g, kLowC):
        ff = dict(f)
        ff.update(x)
        op = do_oceanic_phys(P, g, ex, ff, kLowC, adj)
        return {k: op[k] for k in PHYS_OUT}

    @jax.jit
    def vj(x, f, P, g, kLowC, ct):
        out, fn = jax.vjp(lambda x: fphys(x, f, P, g, kLowC), x)
        return out, fn(ct)[0]
    return vj, fphys


def test_oceanic_phys_seams(run):
    """Measured: exact-mode nonzero derivative counts (of 13*50*98*98 = 6.2e6 points) are printed; with the three
    switches on, the switched paths are exactly 0 and the rhoInSitu path is bitwise unchanged."""
    ds, nml, P, g, ex, kLowC, st, exf_in = run
    f = {k: jnp.asarray(v) for k, v in st.f.items()}
    x = {k: f[k] for k in PHYS_IN}
    (vj_exact, fphys), (vj_ecco, _) = _phys_vjp(ex, EXACT), _phys_vjp(ex, PHYS_ON)
    shapes = jax.eval_shape(fphys, x, f, P, g, kLowC)
    out0 = None
    res = {}
    for name, keys in CT_SETS.items():
        ct = {k: (jnp.ones if k in keys else jnp.zeros)(s.shape, s.dtype) for k, s in shapes.items()}
        oe, ge = vj_exact(x, f, P, g, kLowC, ct)
        oc, gc = vj_ecco(x, f, P, g, kLowC, ct)
        if out0 is None:  # forward outputs: bitwise equal in both modes
            for k in PHYS_OUT:
                np.testing.assert_array_equal(np.asarray(oe[k]), np.asarray(oc[k]), err_msg=k)
            out0 = oe
        res[name] = ({k: np.asarray(v) for k, v in ge.items()}, {k: np.asarray(v) for k, v in gc.items()})
    nz = {name: {k: int(np.count_nonzero(v)) for k, v in ge.items()} for name, (ge, gc) in res.items()}
    print("exact-mode nonzero derivative counts:", nz)
    for name, (ge, gc) in res.items():
        for k in PHYS_IN:
            assert np.all(np.isfinite(ge[k])) and np.all(np.isfinite(gc[k])), (name, k)
    # gm_sigma="stable": the GM/Redi tensor depends on theta/salt only through sigmaX/Y/R
    ge, gc = res["gm"]
    assert nz["gm"]["theta"] > 100_000 and nz["gm"]["salt"] > 100_000
    assert not np.any(gc["theta"]) and not np.any(gc["salt"])
    # ggl90="frozen": no derivative through GGL90_CALC (TKE, shear, N^2 all cut)
    ge, gc = res["ggl"]
    assert nz["ggl"]["GGL90TKE"] > 100_000 and nz["ggl"]["uVel"] > 100_000 and nz["ggl"]["theta"] > 100_000
    for k in PHYS_IN:
        assert not np.any(gc[k]), k
    # salt_plume="off": surfaceForcingS no longer depends on saltPlumeFlux (SALT_PLUME_FORCING_SURF skipped)
    ge, gc = res["salt"]
    assert nz["salt"]["saltPlumeFlux"] > 10_000
    assert not np.any(gc["saltPlumeFlux"])
    np.testing.assert_array_equal(ge["salt"], gc["salt"])     # the EmPmR*salt part of surfaceForcingS is untouched
    # rhoInSitu keeps its derivative (ZERO_ADJ_LOC is on sigma only)
    ge, gc = res["rho"]
    assert nz["rho"]["theta"] > 100_000
    for k in PHYS_IN:
        np.testing.assert_array_equal(ge[k], gc[k], err_msg=k)


def test_gm_only_seam(run):
    """gm_sigma="gm_only" (not a TAF mode; the fesom_jax freeze_gm_slope analogue), GGL90 differentiated: the GM/Redi
    tensor loses its theta/salt derivative exactly as with "stable", but the GGL90 outputs keep theirs bitwise (the
    N^2 = sigmaR path into GGL90_CALC stays live), whereas "stable" (TAF's ZERO_ADJ_LOC on sigma for every reader)
    changes the GGL90 path; forward outputs bitwise equal in all three."""
    ds, nml, P, g, ex, kLowC, st, exf_in = run
    f = {k: jnp.asarray(v) for k, v in st.f.items()}
    x = {k: f[k] for k in PHYS_IN}
    (vj_exact, fphys), (vj_gmo, _), (vj_stab, _) = (_phys_vjp(ex, EXACT), _phys_vjp(ex, AdjointConfig(
        gm_sigma="gm_only")), _phys_vjp(ex, AdjointConfig(gm_sigma="stable")))
    shapes = jax.eval_shape(fphys, x, f, P, g, kLowC)

    def ct_for(keys):
        return {k: (jnp.ones if k in keys else jnp.zeros)(s.shape, s.dtype) for k, s in shapes.items()}

    ct = ct_for(CT_SETS["gm"])
    oe, ge = vj_exact(x, f, P, g, kLowC, ct)
    og, gg = vj_gmo(x, f, P, g, kLowC, ct)
    for k in PHYS_OUT:
        np.testing.assert_array_equal(np.asarray(oe[k]), np.asarray(og[k]), err_msg=k)
    assert np.count_nonzero(np.asarray(ge["theta"])) > 100_000
    assert not np.any(np.asarray(gg["theta"])) and not np.any(np.asarray(gg["salt"]))
    ct = ct_for(CT_SETS["ggl"])
    _, ge = vj_exact(x, f, P, g, kLowC, ct)
    _, gg = vj_gmo(x, f, P, g, kLowC, ct)
    _, gs = vj_stab(x, f, P, g, kLowC, ct)
    for k in PHYS_IN:
        assert np.all(np.isfinite(np.asarray(gg[k]))), k
        np.testing.assert_array_equal(np.asarray(ge[k]), np.asarray(gg[k]), err_msg=k)
    d = np.abs(np.asarray(gs["theta"]) - np.asarray(ge["theta"]))
    assert np.count_nonzero(d) > 10_000, np.count_nonzero(d)   # "stable" cuts GGL90's N^2 derivative, "gm_only" not


def test_visc_fac_in_ad(run):
    """Measured: factor 1 == exact bitwise; factor 2: max|g2 - g1| / max|g1| printed; g2 == autodiff at the
    viscFacAdj=2 viscosities (bitwise)."""
    ds, nml, P, g, ex, kLowC, st, exf_in = run
    f = {k: jnp.asarray(st.f[k]) for k in MV_IN}
    kU, kV = dyn_mod.calc_viscosity(P.dyn, g, jnp.asarray(st.f["GGL90viscArU"]), jnp.asarray(st.f["GGL90viscArV"]))
    assert float(jnp.abs(g.f["viscA4Dfld"]).max()) > 0.0 and float(jnp.abs(g.f["viscA4Zfld"]).max()) > 0.0

    def make(adj):
        @jax.jit
        def vj(uv, f, kU, kV, P, g):
            def fn(uv):
                a = dict(f, uVel=uv[0], vVel=uv[1])
                return mom_vecinv_adj(P, g, adj, *(a[k] for k in MV_IN), kU, kV)
            out, b = jax.vjp(fn, uv)
            return out, b(tuple(jnp.ones_like(o) for o in out))[0]
        return vj

    @jax.jit
    def vj_ref(uv, f, kU, kV, P, g):  # plain autodiff of MOM_VECINV at the viscosities TAF recomputes (factor 2)
        visc_ad = mc.mom_calc_visc(visc_params_in_ad(P.mv.visc, 2.0), g)

        def fn(uv):
            a = dict(f, uVel=uv[0], vVel=uv[1])
            return _mom_vecinv_visc(P.mv, g, *(a[k] for k in MV_IN), kU, kV, visc_ad)
        out, b = jax.vjp(fn, uv)
        return b(tuple(jnp.ones_like(o) for o in out))[0]

    uv = (f["uVel"], f["vVel"])
    o0, g0 = make(EXACT)(uv, f, kU, kV, P, g)
    o1, g1 = make(AdjointConfig(visc_fac_in_ad=1.0))(uv, f, kU, kV, P, g)
    o2, g2 = make(AdjointConfig(visc_fac_in_ad=2.0))(uv, f, kU, kV, P, g)
    gr = vj_ref(uv, f, kU, kV, P, g)
    for a, b, c in zip(o0, o1, o2):  # forward: bitwise the exact one
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))
    for a, b in zip(g0, g1):  # factor 1: bitwise the exact gradient
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    rel = max(float(jnp.abs(b - a).max() / jnp.abs(a).max()) for a, b in zip(g0, g2))
    ref = max(float(jnp.abs(b - a).max() / jnp.abs(a).max()) for a, b in zip(gr, g2))
    print(f"viscFacInAd=2: max rel change of d/d(uVel,vVel) {rel:.3e}; vs autodiff at scaled viscosity {ref:.3e}")
    assert rel > 1e-3
    assert ref <= 1e-14


def _step_grad(run, adj):
    ds, nml, P, g, ex, kLowC, st, exf_in = run
    L = g.layout
    box = np.zeros(st.f["theta"].shape)
    box[2, :10, L.OLy + 20:L.OLy + 60, L.OLx + 20:L.OLx + 60] = 1.0   # tile 3 (facet 1), upper 10 levels

    @jax.jit
    def grad(stf, box, P, g, kLowC, exf_in):
        def J(stf):
            st1, _ = forward_step(P, g, ex, kLowC, State(stf, st.it), exf_in, adj=adj)
            return jnp.sum(box * st1.f["theta"])
        return jax.grad(J)(stf)
    stf = {k: jnp.asarray(v) for k, v in st.f.items()}
    return {k: np.asarray(v) for k, v in grad(stf, jnp.asarray(box), P, g, kLowC, exf_in).items()}


def test_step_gradient_ecco_vs_exact(run):
    """One full step, both modes. Measured: all derivatives finite; numbers printed."""
    ge = _step_grad(run, EXACT)
    gc = _step_grad(run, AdjointConfig.ecco(run[1]))
    for k in ge:
        assert np.all(np.isfinite(ge[k])) and np.all(np.isfinite(gc[k])), k
    assert np.count_nonzero(ge["GGL90TKE"]) > 0 and not np.any(gc["GGL90TKE"])
    d = float(np.abs(gc["theta"] - ge["theta"]).max() / np.abs(ge["theta"]).max())
    print(f"step: nonzero dJ/dTKE exact {np.count_nonzero(ge['GGL90TKE'])}, ecco 0; "
          f"dJ/dtheta max rel diff ecco vs exact {d:.3e}; |dJ/dtheta| exact {np.abs(ge['theta']).max():.4e}, "
          f"ecco {np.abs(gc['theta']).max():.4e}")
    assert d > 1e-6
