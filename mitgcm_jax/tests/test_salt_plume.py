"""pkg/salt_plume (plan Task 11): plume depth replay gate (P03_salt_plume_depth) and the salinity tendency.

SALT_PLUME_CALC_DEPTH gate: inputs rhoInSitu(k=1) of P02 (the Fortran's own surface density), theta, salt of
S00_begin, kLowC from the initial hFacC (h0FacC); output saltPlumeDepth of P03 compared at every point (full-tile
loops), both oracles, every dumped iteration. saltPlumeFlux passes through DO_OCEANIC_PHYS unchanged (P01 -> P03);
in the FORCED run it is nonzero (EXF spflx), in SMOKE it is 0.

SALT_PLUME_TENDENCY_APPLY_S has no dump of its own (the orchestrator gates it inside the tracer RHS): here it is
checked (a) against an independent scalar transcription of the Fortran (salt_plume_tendency_apply_s.F:121-156 +
salt_plume_frac.F:84-221) on the FORCED fields at sampled plume columns, bitwise; (b) for conservation: in every column
sum_k term(k)*rA*drF(k)*hFacC(k) = saltPlumeFlux*mass2rUnit*rA (the whole plume flux is redistributed).
Negative controls: SaltPlumeCriterion * (1 + 1e-6) fails the depth gate; recip_drF(k+1) in place of recip_drF(k)
in the tendency fails conservation.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.core import eos as eos_mod
from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import salt_plume as spm
from mitgcm_jax.tests import oracle

CASES = [(oracle.SMOKE, 1), (oracle.SMOKE, 2), (oracle.FORCED, 1), (oracle.FORCED, 2), (oracle.FORCED, 3)]
# rel. to max |saltPlumeDepth|. Achieved 0 (bitwise) for both oracles, all iterations, FMA contraction off; with FMA
# (default AVX2) up to 9.4e-13 (the EOS inside the depth search differs by a few ulp).
TOL_DEPTH = 0.0


def fma_contracted():
    """True if XLA fuses a*b+c into one FMA (default on AVX2 CPUs). The oracle is built with -ffp-contract=off, so
    bitwise gates need XLA_FLAGS=--xla_cpu_max_isa=AVX (no FMA3); with FMA the EOS differs from Fortran by a few ulp
    at ~0.1 % of the points (measured: rhoInSitu rel. error 1.6e-14, sigmaX 2.5e-13)."""
    a = jnp.full((4096,), 1.0 + 2.0 ** -30)
    c = jnp.full((4096,), -(1.0 + 2.0 ** -29))
    r = jax.jit(lambda a, b, c: a * b + c)(a, a, c)
    return bool(np.any(np.asarray(r) != 0.0))


def require_exact_fp():
    if fma_contracted():
        pytest.fail("XLA contracts a*b+c into FMA: the Fortran oracle is -ffp-contract=off; run the gates with "
                    "XLA_FLAGS=--xla_cpu_max_isa=AVX (see fma_contracted)")


def relerr(got, ref):
    d = float(np.max(np.abs(np.asarray(got) - ref)))
    m = float(np.max(np.abs(ref)))
    return d / m if m > 0 else d


@functools.lru_cache(maxsize=1)
def case(name, it):
    ds = oracle.dumpset(name)
    g = grid_from_dump(ds, it)
    nml = RunNamelists(oracle.run_dir(name))
    eos = eos_mod.EOSParams.from_namelists(nml, g.rC, g.rF)
    sp = spm.SaltPlumeParams.from_namelists(nml)
    d = {n: oracle.field(ds, it, "S00_begin", n) for n in ("theta", "salt")}
    d["rhoSurf"] = oracle.field(ds, it, "P02_rho_sigma_ivdc_mxlayer", "rhoInSitu")[:, 0]
    for n in ("saltPlumeDepth", "saltPlumeFlux"):
        d[n + "_P01"] = oracle.field(ds, it, "P01_external_forcing_surf", n)
        d[n] = oracle.field(ds, it, "P03_salt_plume_depth", n)
    for n in ("hFacC", "recip_hFacC"):
        d[n] = oracle.field(ds, it, "S01_update_rstar_F", n)
    return g, eos, sp, d


def depth(sp, eos, g, d):
    kLowC = spm.klowc(g.h0FacC)
    f = jax.jit(spm.salt_plume_calc_depth)  # params and grid as jit arguments (floats traced)
    return np.asarray(f(sp, eos, g, d["rhoSurf"], d["theta"], d["salt"], kLowC))


@pytest.mark.parametrize("name,it", CASES)
def test_p03_depth_replay_gate(name, it):
    require_exact_fp()
    g, eos, sp, d = case(name, it)
    e = relerr(depth(sp, eos, g, d), d["saltPlumeDepth"])
    print(name, it, "saltPlumeDepth rel err", e)
    assert e <= TOL_DEPTH, e
    # zeroed at the top of DO_OCEANIC_PHYS (do_oceanic_phys.F(ff):292), flux untouched (READIN_SALT_PLUME_FLUX)
    assert np.all(d["saltPlumeDepth_P01"] == 0.0)
    np.testing.assert_array_equal(d["saltPlumeFlux"], d["saltPlumeFlux_P01"])
    if name == oracle.FORCED:
        assert np.abs(d["saltPlumeFlux"]).max() > 0
    else:
        assert np.all(d["saltPlumeFlux"] == 0.0)
    # not vacuous: the criterion is met above the bottom somewhere (depth < column depth)
    col = np.asarray(g.rF)[0] - np.asarray(g.R_low)
    assert np.any((d["saltPlumeDepth"] > 0) & (d["saltPlumeDepth"] < col))


def test_klowc_from_masks():
    """kLowC from the initial hFacC equals the deepest level with maskC = 1 (both from ini_masks_etc.F)."""
    g, _, _, _ = case(*CASES[-1])
    np.testing.assert_array_equal(np.asarray(spm.klowc(g.h0FacC)), np.asarray(spm.klowc(g.maskC)))


def test_negative_control_criterion():
    g, eos, sp, d = case(*CASES[-1])
    bad = spm.SaltPlumeParams(**{**sp.__dict__, "SaltPlumeCriterion": sp.SaltPlumeCriterion * (1 + 1e-6)})
    assert relerr(depth(bad, eos, g, d), d["saltPlumeDepth"]) > 1e-12  # well above rounding


# ---------------------------------------------------------------------------------------------- tendency


def tendency_all_levels(sp, g, spd, flux, recip_hFacC):
    """SALT_PLUME_TENDENCY_APPLY_S on gS = 0 for every level at once (full tile), [T, Nr, ny, nx]."""
    Nr = g.layout.Nr
    k = jnp.arange(1, Nr + 1)[:, None, None]
    inside, term = spm.salt_plume_tendency_s(sp, g, k, spd[:, None], flux[:, None], jnp.asarray(g.maskC),
                                            jnp.asarray(recip_hFacC))
    return jnp.where(inside, 0.0 + term, 0.0)


def _frac_scalar(Npower, fact, spdepth, plumek):
    """salt_plume_frac.F:84-221 for one value, PlumeMethod = 1, not NEC (plain Python floats)."""
    facz = abs(fact * plumek)
    if spdepth >= facz and spdepth > 0.0:
        dd20 = abs(spdepth)
        S = 1.0
        So = 1.0
        for _ in range(Npower + 1):
            S = facz * S
            So = dd20 * So
        return max(0.0, S / So)
    return 1.0


def test_tendency_matches_fortran_transcription():
    require_exact_fp()
    g, _, sp, d = case(oracle.FORCED, 1)
    L = g.layout
    spd, flux = d["saltPlumeDepth"], d["saltPlumeFlux"]
    got = np.asarray(jax.jit(tendency_all_levels)(sp, g, spd, flux, d["recip_hFacC"]))
    rF, recip_drF, maskC = np.asarray(g.rF), np.asarray(g.recip_drF), np.asarray(g.maskC)
    cols = np.argwhere((flux != 0) & (spd > 0))
    assert len(cols) > 100
    rng = np.random.default_rng(0)
    for t, j, i in cols[rng.choice(len(cols), 200, replace=False)]:
        for k in range(1, L.Nr + 1):
            gS = 0.0
            if spd[t, j, i] > abs(rF[k - 1]):                       # salt_plume_tendency_apply_s.F:124
                kb1 = _frac_scalar(sp.Npower, -1.0, spd[t, j, i], abs(rF[k - 1]))
                kb2 = _frac_scalar(sp.Npower, -1.0, spd[t, j, i], abs(rF[k]))
                plumefrac = (kb2 - kb1) * maskC[t, k - 1, j, i]     # :144
                plumetend = flux[t, j, i] * plumefrac               # :145
                gS = gS + plumetend * recip_drF[k - 1] * sp.mass2rUnit * d["recip_hFacC"][t, k - 1, j, i]
            assert got[t, k - 1, j, i] == gS, (t, k, j, i, got[t, k - 1, j, i], gS)


def _conservation_err(g, sp, d, tend):
    """max over plume columns of |sum_k tend*rA*drF*hFacC - F_bot*saltPlumeFlux*mass2rUnit*rA| / |...|, where
    F_bot = SALT_PLUME_FRAC(|rF(kLowC+1)|) is the plume fraction above the bottom of the deepest wet cell (1 unless
    R_low lies below that interface, ini_masks_etc.F:170-176: then the rest falls into dry cells, maskC = 0)."""
    L = g.layout
    I3 = (slice(None), slice(None), L.js(1, L.sNy), L.is_(1, L.sNx))
    I2 = (slice(None), L.js(1, L.sNy), L.is_(1, L.sNx))
    rA = np.asarray(g.rA)
    vol = rA[:, None] * np.asarray(g.drF)[None, :, None, None] * d["hFacC"]
    col = np.sum((np.asarray(tend) * vol)[I3], axis=1)
    kLowC = np.asarray(spm.klowc(g.h0FacC))
    rFbot = np.abs(np.asarray(g.rF)[kLowC])                          # |rF(kLowC+1)|
    F_bot = np.asarray(spm.salt_plume_frac(sp, -1.0, jnp.asarray(d["saltPlumeDepth"]), jnp.asarray(rFbot)))
    src = (d["saltPlumeFlux"] * sp.mass2rUnit * rA * F_bot)[I2]
    plume = (d["saltPlumeDepth"][I2] > 0) & (src != 0)
    n_partial = int(((F_bot[I2] < 1.0) & plume).sum())
    return float(np.max(np.abs(col - src)[plume] / np.abs(src[plume]))), int(plume.sum()), n_partial


def test_tendency_conserves_plume_flux():
    g, _, sp, d = case(oracle.FORCED, 1)
    tend = jax.jit(tendency_all_levels)(sp, g, d["saltPlumeDepth"], d["saltPlumeFlux"], d["recip_hFacC"])
    err, n, n_partial = _conservation_err(g, sp, d, tend)
    print("conservation: plume columns", n, "of which plume below the deepest wet cell", n_partial, "max rel err", err)
    assert n > 100 and err < 1e-13, (n, err)


def test_negative_control_tendency_level_shift():
    """Plant: recip_drF(k+1) instead of recip_drF(k) in the tendency (one index shifted): conservation fails."""
    g, _, sp, d = case(oracle.FORCED, 1)
    r = np.asarray(g.recip_drF)
    bad_g = g.replace(recip_drF=np.concatenate([r[1:], r[-1:]]))
    tend_bad = tendency_all_levels(sp, bad_g, d["saltPlumeDepth"], d["saltPlumeFlux"], d["recip_hFacC"])
    err, _, _ = _conservation_err(g, sp, d, tend_bad)
    assert err > 1e-6, err


def test_gradients_finite_and_fd():
    """d saltPlumeDepth / d(theta, salt) and d tendency / d(saltPlumeFlux, saltPlumeDepth): finite everywhere;
    d saltPlumeDepth / d salt(k=1) vs central differences at columns where the depth is interpolated (h-sweep
    1e-4..1e-7, plateau = min over h). Columns are independent, so all test columns share one backward pass and
    each FD evaluation perturbs them all at once."""
    g, eos, sp, d = case(oracle.FORCED, 1)
    kLowC = spm.klowc(g.h0FacC)
    th0, sa0 = jnp.asarray(d["theta"]), jnp.asarray(d["salt"])

    def spd_pg(sp, eos, g, th, sa):
        rhoSurf = eos_mod.find_rho_2d(eos, th[:, 0], sa[:, 0], 1)
        return spm.salt_plume_calc_depth(sp, eos, g, rhoSurf, th, sa, kLowC)

    grad_pg = jax.jit(jax.grad(lambda sp, eos, g, th, sa, w: jnp.sum(w * spd_pg(sp, eos, g, th, sa)), argnums=(3, 4)))
    fwd_pg = jax.jit(spd_pg)

    def gfun(th, sa, w):
        return grad_pg(sp, eos, g, th, sa, w)
    rng = np.random.default_rng(0)
    gth, gsa = gfun(th0, sa0, jnp.asarray(rng.normal(size=d["saltPlumeDepth"].shape)))
    assert np.all(np.isfinite(np.asarray(gth))) and np.all(np.isfinite(np.asarray(gsa)))

    Wt = jnp.asarray(rng.normal(size=d["hFacC"].shape))
    gf, gd = jax.jit(jax.grad(lambda f, s: jnp.sum(Wt * tendency_all_levels(sp, g, s, f, d["recip_hFacC"])),
                              argnums=(0, 1)))(jnp.asarray(d["saltPlumeFlux"]), jnp.asarray(d["saltPlumeDepth"]))
    assert np.all(np.isfinite(np.asarray(gf))) and np.all(np.isfinite(np.asarray(gd)))

    spd = d["saltPlumeDepth"]
    col = np.asarray(g.rF)[0] - np.asarray(g.R_low)
    L = g.layout
    inner = np.zeros(spd.shape, bool)
    inner[:, L.js(1, L.sNy), L.is_(1, L.sNx)] = True
    cand = np.argwhere(inner & (spd > 20.0) & (spd < col - 1.0))
    cand = cand[rng.choice(len(cand), 20, replace=False)]
    sel = np.zeros(spd.shape)
    sel[tuple(cand.T)] = 1.0
    ad = np.asarray(gfun(th0, sa0, jnp.asarray(sel))[1])[cand[:, 0], 0, cand[:, 1], cand[:, 2]]
    use = ad != 0.0  # ad = 0: depth at a level interface (tmpFac = 0 branch), locally constant
    assert use.sum() >= 3, ad
    def fwd(th, sa):
        return fwd_pg(sp, eos, g, th, sa)

    fds = []
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        idx = (cand[:, 0], np.zeros(len(cand), int), cand[:, 1], cand[:, 2])
        p = np.asarray(fwd(th0, sa0.at[idx].add(h)))[tuple(cand.T)]
        m = np.asarray(fwd(th0, sa0.at[idx].add(-h)))[tuple(cand.T)]
        fds.append((p - m) / (2 * h))
    err = np.min(np.abs(np.array(fds) - ad[None]), axis=0)[use] / np.abs(ad[use])
    print("SPD d/dsalt(k=1): AD", ad[use], "plateau rel err", err)
    assert err.max() < 1e-5, err
