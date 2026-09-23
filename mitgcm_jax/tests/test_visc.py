"""MOM_CALC_VISC, V4r4 override (plan Task 14a): viscosity values, Gibraltar x10 region, replay through MOM_VECINV.

The oracle has no dump of the viscosity arrays themselves; they are gated (a) against the literal formula of
flux-forced/code/mom_calc_visc.F:403-430 / :513-540 evaluated in numpy from the dumped geometry (bitwise under the
no-FMA flag), and (b) through the MOM_VECINV replay: guDissip/gvDissip are bitwise equal to the Fortran dump with
the Gibraltar factor (test_mom_vecinv.py), and dropping the factor fails that comparison exactly around the
Gibraltar box and nowhere else (negative control).
"""

import dataclasses

import jax
import numpy as np
import pytest

from mitgcm_jax.pkgs import mom_common as mc
from mitgcm_jax.tests import oracle
from mitgcm_jax.tests.test_mom_vecinv import KERNEL, NO_FMA, RTOL, case, errors, grid, params


def _np(g, n):
    return np.asarray(g.f[n])


def _interior(L):
    """The MOM_CALC_VISC loop range 2-OLx..sNx+OLx-1, 2-OLy..sNy+OLy-1 as (j, i) slices."""
    return L.js(2 - L.OLy, L.sNy + L.OLy - 1), L.is_(2 - L.OLx, L.sNx + L.OLx - 1)


def test_viscosity_matches_literal_formula():
    """viscAh_D/Z = 10^box * MIN(MIN(1e21*L2rdt,1e21), MAX(MAX(0*L2rdt,0), ((viscAhD + viscAhGrid*L2rdt) + 0 + 0)
    + viscFacAdj*viscAhDfld)); viscA4_D/Z likewise with L4rdt and viscA4Dfld; the outermost halo ring keeps
    viscAhD / viscA4D (mom_vecinv.F:359-366). numpy evaluates in the Fortran operation order."""
    p = params(oracle.SMOKE).visc
    g = grid(oracle.SMOKE)
    L = g.layout
    ah_z, ah_d, a4_z, a4_d = (np.asarray(a) for a in jax.jit(mc.mom_calc_visc)(p, g))
    recip_dt = 1.0 / p.deltaTmom
    J, I = _interior(L)
    for pt, rA, rx, ry, xg, yg, ah, a4 in (("D", "rA", "recip_dxF", "recip_dyF", "xC", "yC", ah_d, a4_d),
                                           ("Z", "rAz", "recip_dxV", "recip_dyU", "xG", "yG", ah_z, a4_z)):
        a, b = _np(g, rx), _np(g, ry)
        cond = (a != 0) | (b != 0)
        with np.errstate(divide="ignore"):
            L2 = np.where(cond, 2.0 / (a * a + b * b), _np(g, rA))  # mom_init_fixed.F:80-93
        L4rdt = (0.03125 * recip_dt) * (L2 * L2)
        L2rdt = (0.25 * recip_dt) * L2
        vAh, vA4 = (p.viscAhD, p.viscA4D) if pt == "D" else (p.viscAhZ, p.viscA4Z)
        alin = ((vAh + p.viscAhGrid * L2rdt) + 0.0 + 0.0)[:, None] + p.viscFacAdj * _np(g, "viscAh" + pt + "fld")
        ref = np.minimum(np.minimum(p.viscAhGridMax * L2rdt, p.viscAhMax)[:, None],
                         np.maximum(np.maximum(p.viscAhGridMin * L2rdt, 0.0)[:, None], alin))
        box = mc.gibraltar_mask(np.asarray(_np(g, xg)), np.asarray(_np(g, yg)))[:, None]
        ref = np.where(box, 10.0 * ref, ref)
        alin4 = ((vA4 + p.viscA4Grid * L4rdt) + 0.0 + 0.0)[:, None] + p.viscFacAdj * _np(g, "viscA4" + pt + "fld")
        ref4 = np.minimum(np.minimum(p.viscA4GridMax * L4rdt, p.viscA4Max)[:, None],
                          np.maximum(np.maximum(p.viscA4GridMin * L4rdt, 0.0)[:, None], alin4))
        for got, want, const in ((ah, ref, vAh), (a4, ref4, vA4)):
            d = np.abs(got[..., J, I] - want[..., J, I])
            assert d.max() <= RTOL * np.abs(want[..., J, I]).max(), (pt, d.max())
            if NO_FMA:
                assert np.array_equal(got[..., J, I], want[..., J, I]), pt
            ring = np.ones(got.shape, bool)
            ring[..., J, I] = False
            assert np.all(got[ring] == const)
        # 3-D biharmonic field passes through unchanged (viscA4D = viscA4Grid = 0, viscFacAdj = 1)
        assert np.array_equal(a4[..., J, I], _np(g, "viscA4" + pt + "fld")[..., J, I])


def test_gibraltar_region():
    """The x10 box (33-39N, 7-2W) holds wet D and Z points at the surface, and there viscAh is exactly 10x the
    value without the factor; everywhere else the two agree bitwise."""
    p = params(oracle.SMOKE).visc
    g = grid(oracle.SMOKE)
    with_f = [np.asarray(a) for a in mc.mom_calc_visc(p, g)]
    without = [np.asarray(a) for a in mc.mom_calc_visc(p, g, gibraltar=False)]
    J, I = _interior(g.layout)
    for (xg, yg, mask, a, b) in (("xC", "yC", "maskC", with_f[1], without[1]),
                                 ("xG", "yG", "maskW", with_f[0], without[0])):
        box = mc.gibraltar_mask(_np(g, xg), _np(g, yg))[:, None] & np.ones(a.shape, bool)
        inner = np.zeros(a.shape, bool)
        inner[..., J, I] = True
        sel = box & inner
        wet = sel[:, 0] & (_np(g, mask)[:, 0] > 0)
        assert wet.sum() >= 4, wet.sum()
        np.testing.assert_array_equal(a[sel], 10.0 * b[sel])
        np.testing.assert_array_equal(a[~sel], b[~sel])
    # biharmonic viscosity has no Gibraltar factor
    np.testing.assert_array_equal(with_f[2], without[2])
    np.testing.assert_array_equal(with_f[3], without[3])


@pytest.mark.parametrize("name", [oracle.SMOKE, oracle.FULL])
def test_gibraltar_replay_negative_control(name):
    """MOM_VECINV with the viscosity lacking the Gibraltar factor fails the oracle gate on guDissip and gvDissip,
    and only at points next to the box (within 1.5 degrees); gU/gV are unaffected (bitwise). Flux-forced (SMOKE)
    and full V4r4 tree (the same mom_calc_visc.F override, code/ = flux-forced/code/)."""
    it = 1
    p = params(name)
    g = grid(name)
    inp, ref = case(name, it)
    visc = jax.jit(mc.mom_calc_visc, static_argnames="gibraltar")(p.visc, g, gibraltar=False)
    out = KERNEL(p, g, inp)  # the gate's own run (with the factor)
    bad = KERNEL(p, g, inp, visc)
    err_ok, err_bad = errors(out, ref), errors(bad, ref)
    for n in ("guDissip", "gvDissip"):
        assert err_ok[n][0] <= RTOL
        assert err_bad[n][0] > RTOL, (n, err_bad[n])
        diff = np.asarray(bad[n]) != ref[n]
        x, y = _np(g, "xC")[:, None], _np(g, "yC")[:, None]
        x, y = np.broadcast_to(x, diff.shape)[diff], np.broadcast_to(y, diff.shape)[diff]
        assert np.all((x > -8.5) & (x < -0.5) & (y > 31.5) & (y < 40.5)), (x.min(), x.max(), y.min(), y.max())
    if NO_FMA:
        assert err_bad["gU"][1] == 0 and err_bad["gV"][1] == 0


def test_viscFacAdj_scales_only_3d_fields():
    """viscFacAdj multiplies the four 3-D fields (V4r4 override, unlike c66g's two); the forward value is 1
    (set_defaults.F:131) and reproduces the default; another value changes viscA4 (3-D field) but not viscAh
    (its 3-D fields are zero in V4r4)."""
    p = params(oracle.SMOKE).visc
    g = grid(oracle.SMOKE)
    base = [np.asarray(a) for a in mc.mom_calc_visc(p, g)]
    same = [np.asarray(a) for a in mc.mom_calc_visc(dataclasses.replace(p, viscFacAdj=1.0), g)]
    half = [np.asarray(a) for a in mc.mom_calc_visc(dataclasses.replace(p, viscFacAdj=0.5), g)]
    for a, b in zip(base, same):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(half[0], base[0])
    np.testing.assert_array_equal(half[1], base[1])
    J, I = _interior(g.layout)
    np.testing.assert_array_equal(half[3][..., J, I], 0.5 * base[3][..., J, I])
    assert np.any(half[3] != base[3])
