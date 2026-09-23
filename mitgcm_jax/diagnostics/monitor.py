"""%MON dynstat statistics of a state, host-side numpy (pkg/monitor/monitor.F, mon_calc_stats_rl.F).

For comparing JAX runs with the Fortran STDOUT %MON lines (mitgcm_jax/io/monitor.py parses those). Not in the AD
path. Sums are per tile then over tiles in order (GLOBAL_SUM_TILE_RL); within a tile numpy sums pairwise, not in the
Fortran k/j/i order, so means agree to ~1e-15 relative, not bitwise.
"""

import numpy as np


def mon_calc_stats(arr, hfac, mask, area, dr, layout):
    """mon_calc_stats_rl.F: min, max, mean, sd, del2, vol of arr [T,(k),ny,nx] over mask*hFac > 0 (interior only)."""
    L = layout
    I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
    a = np.asarray(arr, np.float64)
    h = np.asarray(hfac, np.float64)
    if a.ndim == 3:
        a, h = a[:, None], h[:, None]
    dr = np.asarray(dr, np.float64).reshape(1, -1, 1, 1)
    m = np.asarray(mask, np.float64)[:, None]
    ai, hi = a[..., J, I], h[..., J, I]
    tm = m[..., J, I] * hi                                                    # mon_calc_stats_rl.F:98
    on = tm > 0
    if not on.any():
        return dict(min=0.0, max=0.0, mean=0.0, sd=0.0, del2=0.0, vol=0.0)
    # del2 (l.106-116)
    ddx = np.where(h[..., J, 1 + L.OLx:1 + L.OLx + L.sNx] * h[..., J, L.OLx - 1:L.OLx - 1 + L.sNx] > 0,
                   (a[..., J, 1 + L.OLx:1 + L.OLx + L.sNx] - ai) + (a[..., J, L.OLx - 1:L.OLx - 1 + L.sNx] - ai),
                   h[..., J, 1 + L.OLx:1 + L.OLx + L.sNx] * h[..., J, L.OLx - 1:L.OLx - 1 + L.sNx])
    ddy = np.where(h[..., 1 + L.OLy:1 + L.OLy + L.sNy, I] * h[..., L.OLy - 1:L.OLy - 1 + L.sNy, I] > 0,
                   (a[..., 1 + L.OLy:1 + L.OLy + L.sNy, I] - ai) + (a[..., L.OLy - 1:L.OLy - 1 + L.sNy, I] - ai),
                   h[..., 1 + L.OLy:1 + L.OLy + L.sNy, I] * h[..., L.OLy - 1:L.OLy - 1 + L.sNy, I])
    ar = np.asarray(area, np.float64)[:, None][..., J, I]
    vol = np.where(on, ar * dr * tm, 0.0)
    tile = lambda x: np.sum(np.where(on, x, 0.0).reshape(x.shape[0], -1), axis=1)  # noqa: E731
    nb = tile(np.ones_like(ai)).sum()
    del2 = np.sqrt(tile(ddx * ddx + ddy * ddy).sum()) / nb
    tvol = tile(vol).sum()
    mean = tile(vol * ai).sum() / tvol
    sd = np.sqrt(tile(vol * (ai - mean) ** 2).sum() / tvol)
    return dict(min=float(ai[on].min()), max=float(ai[on].max()), mean=float(mean), sd=float(sd), del2=float(del2),
                vol=float(tvol))


def dynstat(state, g, layout):
    """monitor.F:108-121 dynstat block (eta, uvel, vvel, wvel, theta, salt)."""
    thickC = np.asarray(g.drF) * np.asarray(g.rhoFacC)            # monitor.F:97 (deepFac2C = 1)
    thickF = np.asarray(g.drC[:-1]) * np.asarray(g.rhoFacF[:-1])  # monitor.F:98
    out = {}
    out["eta"] = mon_calc_stats(state.etaN, g.maskInC, g.maskInC, g.rA, np.asarray(g.drF)[:1], layout)
    out["uvel"] = mon_calc_stats(state.uVel, state.hFacW, g.maskInW, g.rAw, thickC, layout)
    out["vvel"] = mon_calc_stats(state.vVel, state.hFacS, g.maskInS, g.rAs, thickC, layout)
    out["wvel"] = mon_calc_stats(state.wVel, g.maskC, g.maskInC, g.rA, thickF, layout)
    out["theta"] = mon_calc_stats(state.theta, state.hFacC, g.maskInC, g.rA, thickC, layout)
    out["salt"] = mon_calc_stats(state.salt, state.hFacC, g.maskInC, g.rA, thickC, layout)
    return out


def format_dynstat(stats, it):
    lines = [f"%MON time_tsnumber = {it}"]
    for name, s in stats.items():
        for k in ("max", "min", "mean", "sd", "del2"):
            if k in s:
                lines.append(f"%MON dynstat_{name}_{k} = {s[k]: .13E}")
    return "\n".join(lines)


def dynstat_device(st, g):
    """On-device version of the dynstat block for run monitoring (min, max, volume-weighted mean and sd over wet
    points; no del2). jnp sums (tree order): agrees with `dynstat` to ~1e-14, not bitwise. Returns jnp scalars."""
    import jax.numpy as jnp

    L = g.layout
    I, J = slice(L.OLx, L.OLx + L.sNx), slice(L.OLy, L.OLy + L.sNy)
    thickC = g.drF * g.rhoFacC
    thickF = g.drC[:-1] * g.rhoFacF[:-1]

    def stats(a, h, mask, area, dr):
        if a.ndim == 3:
            a, h = a[:, None], h[:, None]
        a, h = a[..., J, I], h[..., J, I]
        tm = mask[:, None, J, I] * h
        on = tm > 0
        vol = jnp.where(on, area[:, None, J, I] * dr[None, :, None, None] * tm, 0.0)
        tv = vol.sum()
        mean = (vol * a).sum() / tv
        sd = jnp.sqrt((vol * (a - mean) ** 2).sum() / tv)
        return {"min": jnp.min(jnp.where(on, a, jnp.inf)), "max": jnp.max(jnp.where(on, a, -jnp.inf)),
                "mean": mean, "sd": sd}

    return {"eta": stats(st.etaN, g.maskInC, g.maskInC, g.rA, g.drF[:1]),
            "uvel": stats(st.uVel, st.hFacW, g.maskInW, g.rAw, thickC),
            "vvel": stats(st.vVel, st.hFacS, g.maskInS, g.rAs, thickC),
            "wvel": stats(st.wVel, g.maskC, g.maskInC, g.rA, thickF),
            "theta": stats(st.theta, st.hFacC, g.maskInC, g.rA, thickC),
            "salt": stats(st.salt, st.hFacC, g.maskInC, g.rA, thickC)}
