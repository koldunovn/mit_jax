"""SEAICE_MODEL driver (plan M2.6b-1): fixed sea-ice fields from the grid, the whole sea-ice step chained, the three
adjoint levels, jit with traced parameters, host-CPU placement.

Oracle `oracle.FULL` (full_jaxdump_v5, iterations 1-3; stages I00-I04, P00, Y01, G01, S00, S01, S03, X06 in
reference/jaxdump/SUBSTEPS.md). Modules: pkgs/seaice_model.py, pkgs/seaice_init.py (fixed fields), the four kernels.

Gates (all points, halos included; measured values in each test's docstring):
  - fixed fields: seaice_init.seaice_fixed_fields from grid_from_files (the production path) AND from grid_from_dump
    == G01_seaice_geometry (HEFFM, k1AtC, k1AtZ, k2AtC, k2AtZ) and == the y group of Y01/I01 (seaiceMaskU/V,
    tensileStrFac) at iterations 1-3.
  - carried state: dyn_carry_init == the entry values at iteration 1; the previous step's outputs == the entry values
    at iterations 2, 3 (DYN_CARRY, ICE_STATE, sIceLoad).
  - driver chained over iterations 1 -> 2 -> 3 (grid and fixed fields from the files; the sea-ice state, DYN_CARRY
    and sIceLoad carried from the driver's own previous output; EXF, surface ocean and fu..saltFlux from the dumps):
    I00 uwind/vwind, every I01 field (uice_fd/vice_fd: <= 1 ulp at < 200 points, glibc sincos, read by nothing),
    I02, I03, I04 and P00 bitwise; LSOR sweep counts equal.
Negative controls: the post-growth exchanges skipped (P00), reg_ridge before advdiff (I03/I04), clipping off (I01),
the wind exchange skipped (I00), masks without their exchange / from a shifted HEFFM, a planted grid error in k1AtC.
Adjoint levels: forward bitwise identical in "ecco", "no_dynamics", "full"; ecco VJP == identity on INOUT and 0 on
READ exactly; the skipped dynamics blocks' VJPs == identity/zero exactly; effect tests (gradients differ between every
pair); "full": gradient finite on every lane and a 2-point central-difference check (d/d theta_s).
"""

import dataclasses
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitgcm_jax.grid.geometry import grid_from_dump
from mitgcm_jax.io.dump import DumpSet, read_file
from mitgcm_jax.layout import Layout
from mitgcm_jax.parallel.exchange import default_exchanger
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.pkgs import seaice_advdiff as sa
from mitgcm_jax.pkgs import seaice_dyn as sd
from mitgcm_jax.pkgs import seaice_init as si
from mitgcm_jax.pkgs import seaice_model as sm
from mitgcm_jax.tests import oracle

L = Layout()
EX = default_exchanger(L)
ITS = (1, 2, 3)
COUNTS = {1: (178, 118), 2: (112, 82), 3: (84, 58)}  # LSOR sweeps per Picard pass (test_seaice_dyn.py)
GRID_DIR = Path("/work/ab0995/a270088/MIT/data/eccov4r4/native_grid_files")  # model.GRID_DIR
FF_DUMPED = ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "sIceLoad")
PASS_THROUGH = ("surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "pLoad", "phi0surf")
I01_ULP = ("uice_fd", "vice_fd")


def _dumpset_parallel(directory):
    """DumpSet with the record headers read in parallel threads (serial indexing of the full oracle: ~110 s)."""
    files = sorted(Path(directory).glob("jd_*_t*.bin"))
    assert files, directory
    with ThreadPoolExecutor(min(len(files), 64)) as pool:
        per_file = list(pool.map(lambda f: read_file(f, lazy=True), files))
    ds = DumpSet.__new__(DumpSet)
    ds.dir, ds.index = Path(directory), {}
    for recs in per_file:
        for r in recs:
            ds.index.setdefault((r.iter, r.stage, r.field), {})[r.tile] = r
    ds.order = list(ds.index)
    return ds


@functools.lru_cache(maxsize=1)
def env():
    rundir = oracle.run_dir(oracle.FULL)
    ds = _dumpset_parallel(rundir / "jaxdump")
    nml = RunNamelists(rundir)
    from mitgcm_jax.grid.load import grid_from_files
    t = time.time()
    g_files = grid_from_files(rundir, GRID_DIR, EX, L)
    t_grid = time.time() - t
    g_dump = grid_from_dump(ds, 1)
    P = sm.SeaiceParams.from_namelists(nml, g_files)
    g = jax.tree.map(jnp.asarray, sm.seaice_grid(g_files))
    sg = si.seaice_fixed_fields(g_files, EX)
    return SimpleNamespace(rundir=rundir, ds=ds, nml=nml, g_files=g_files, g_dump=g_dump, P=P, g=g, sg=sg,
                           t_grid=t_grid, cache={})


def F(it, stage, name):
    return oracle.field(env().ds, it, stage, name)


def ndiff(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.abs(b).max()
    return float(np.abs(a - b).max() / (m if m > 0 else 1.0))


def inputs(it, carry=None):
    """SEAICE_MODEL's inputs at iteration `it` from the dumps: ICE_STATE (I00), DYN_CARRY (`carry` or
    dyn_carry_init), fu..sIceLoad after CTRL_MAP_FORCING (S03), saltPlumeFlux = 0 (do_oceanic_phys.F:293), uwind/vwind
    as EXF left them (X06: nothing writes them between EXF_BULKFORMULAE and SEAICE_MODEL), the EXF fields SEAICE_MODEL
    reads (I00), surface uVel/vVel (S01), theta/salt (S00). `carry` may override any INOUT key (chained runs)."""
    ins = {k: F(it, "I00_seaice_begin", k) for k in sm.ICE_STATE}
    ins.update({k: np.asarray(v) for k, v in sm.dyn_carry_init(L).items()})
    ins.update({k: F(it, "S03_ctrl_map_forcing", k) for k in FF_DUMPED})
    ins["saltPlumeFlux"] = np.zeros(L.shape2d)
    ins.update({k: F(it, "X06_exf_hflux_sflux", k) for k in sm.EXF_INOUT})
    ins.update({k: F(it, "I00_seaice_begin", k) for k in sm.EXF_READ})
    ins["uVel_s"], ins["vVel_s"] = F(it, "S01_update_rstar_F", "uVel")[:, 0], F(it, "S01_update_rstar_F", "vVel")[:, 0]
    ins["theta_s"], ins["salt_s"] = F(it, "S00_begin", "theta")[:, 0], F(it, "S00_begin", "salt")[:, 0]
    if carry is not None:
        ins.update(carry)
    return {k: jnp.asarray(v) for k, v in ins.items()}


RUN = jax.jit(sm.seaice_model, static_argnames=("ad", "expf", "record"))


def run(ins, ad="ecco", record=True, P=None):
    e = env()
    return RUN(P or e.P, e.g, e.sg, EX, ins, ad=ad, record=record)


def chained():
    """The driver over iterations 1, 2, 3, carrying its own ICE_STATE, DYN_CARRY and sIceLoad (cached)."""
    e = env()
    if "chain" not in e.cache:
        res, carry = {}, None
        for it in ITS:
            ins = inputs(it, carry)
            out, rec = run(ins)
            jax.block_until_ready(out)
            res[it] = (ins, out, rec)
            carry = {k: out[k] for k in sm.SEAICE_CARRIED + ("sIceLoad",)}
        e.cache["chain"] = res
    return e.cache["chain"]


def stage_mismatches(it, rec):
    """{stage/field: number of differing values} over every gated field of the SEAICE_MODEL stages."""
    bad = {}
    for k in ("uwind", "vwind"):
        bad[f"I00/{k}"] = ndiff(rec["I00"][k], F(it, "I00_seaice_begin", k))
    for k, v in rec["I01"].items():
        if k not in I01_ULP:
            bad[f"I01/{k}"] = ndiff(v, F(it, "I01_dynsolver", k))
    for k in ("HEFF", "AREA", "HSNOW", "TICES", "UICE", "VICE"):
        bad[f"I02/{k}"] = ndiff(rec["I02"][k], F(it, "I02_advdiff", k))
    for k in ("HEFF", "AREA", "HSNOW", "TICES", "d_HEFFbyNEG", "d_HSNWbyNEG"):
        bad[f"I03/{k}"] = ndiff(rec["I03"][k], F(it, "I03_reg_ridge", k))
    for k in ("AREA", "HEFF", "HSNOW", "TICES", "d_HEFFbyNEG", "d_HSNWbyNEG", "fu", "fv", "Qnet", "Qsw", "EmPmR",
              "saltFlux", "sIceLoad", "saltPlumeFlux"):
        bad[f"I04/{k}"] = ndiff(rec["I04"][k], F(it, "I04_growth", k))
    for k in ("AREA", "HEFF", "HSNOW", "TICES", "UICE", "VICE", "fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux",
              "sIceLoad", "saltPlumeFlux"):
        bad[f"P00/{k}"] = ndiff(rec["P00"][k], F(it, "P00_seaice_model", k))
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# task 2: fixed fields from the grid


def test_params_and_ad_level():
    """SeaiceParams = the four kernels' params from the full run's namelists; data.autodiff -> "ecco"."""
    e = env()
    for a, b in ((e.P.dyn, sd.SeaiceDynParams.from_namelists(e.nml)),
                 (e.P.adv, sa.SeaiceAdvDiffParams.from_namelists(e.nml, L)),
                 (e.P.ridge, sa.SeaiceRegRidgeParams.from_namelists(e.nml, L))):
        assert jax.tree.leaves(a) == jax.tree.leaves(b) and jax.tree.structure(a) == jax.tree.structure(b)
    assert e.P.flags.advdiff and e.P.flags.exch_sIceLoad
    assert sm.ad_level(e.nml) == "ecco"
    print(f"grid_from_files: {e.t_grid:.1f} s")


@pytest.mark.parametrize("src", ["files", "dump"])
def test_fixed_fields_bitwise(src):
    """seaice_fixed_fields(grid) == G01 (HEFFM, k1AtC, k1AtZ, k2AtC, k2AtZ) and == the y group of Y01 and I01
    (seaiceMaskU/V, tensileStrFac) at iterations 1-3, every point. Measured: 0 differing values, both grids."""
    e = env()
    g = e.g_files if src == "files" else e.g_dump
    sg = si.seaice_fixed_fields(g, EX)
    bad = {}
    for it in ITS:
        for k in si.ICE_GEOMETRY:
            bad[f"G01/{it}/{k}"] = ndiff(sg[k], F(it, "G01_seaice_geometry", k))
        for st in ("Y01_get_dynforcing", "I01_dynsolver"):
            for k in si.ICE_DYN_MASKS:
                bad[f"{st}/{it}/{k}"] = ndiff(sg[k], F(it, st, k))
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    # not vacuous: masks and metric terms non-trivial, halos filled by the exchange
    mU = np.asarray(sg["seaiceMaskU"])
    assert 0.3 < mU[:, L.js(1, L.sNy), L.is_(1, L.sNx)].mean() < 0.9 and mU[:, :L.OLy].sum() > 0
    assert np.abs(np.asarray(sg["k2AtZ"])).max() > 0


def test_fixed_fields_negative_controls():
    """Planted errors make the fixed-field gate fail: masks without their exchange (halo points), masks from HEFFM
    shifted by one column, one recip_dxF value x (1 + 1e-6) (k1AtC, k2AtC)."""
    e = env()
    g = e.g_files
    HEFFM = jnp.asarray(si.seaice_geometry(g)["HEFFM"])
    noex = SimpleNamespace(exch_uv_xy=lambda u, v, s: (u, v))
    m = si.seaice_dyn_masks(g, noex, HEFFM)
    assert ndiff(m["seaiceMaskU"], F(1, "Y01_get_dynforcing", "seaiceMaskU")) > 100
    m = si.seaice_dyn_masks(g, EX, jnp.roll(HEFFM, 1, axis=-1))
    assert ndiff(m["seaiceMaskU"], F(1, "Y01_get_dynforcing", "seaiceMaskU")) > 100
    r = np.array(g.f["recip_dxF"])
    r[3, L.jj(40), L.ii(40)] *= 1 + 1e-6
    geo = si.seaice_geometry(g.replace(recip_dxF=r))
    assert ndiff(geo["k1AtC"], F(1, "G01_seaice_geometry", "k1AtC")) >= 1
    assert ndiff(geo["k2AtC"], F(1, "G01_seaice_geometry", "k2AtC")) >= 1


# ---------------------------------------------------------------------------------------------------------------------
# carried state


def test_carried_state_entry_values():
    """dyn_carry_init == the values SEAICE_DYNSOLVER finds on entry at iteration 1 (FORCEX0/Y0 at Y01, e11 .. FORCEY at
    Y04 -- nothing writes them in between; seaiceMass* on the row/column :112-122 does not write), and the I01 values
    of iteration it-1 == the entry values at it = 2, 3; ICE_STATE at I00(it) == P00(it-1); sIceLoad at S03(it) ==
    P00(it-1). This is what State must carry for sea ice (seaice_model.SEAICE_CARRIED + sIceLoad)."""
    c0 = {k: np.asarray(v) for k, v in sm.dyn_carry_init(L).items()}
    J0, I0 = L.js(1 - L.OLy, 1 - L.OLy), L.is_(1 - L.OLx, 1 - L.OLx)
    bad = {}
    for it in ITS:
        prev = (lambda k: c0[k]) if it == 1 else (lambda k: F(it - 1, "I01_dynsolver", k))
        for k in ("FORCEX0", "FORCEY0"):
            bad[f"{it}/{k}"] = ndiff(prev(k), F(it, "Y01_get_dynforcing", k))
        for k in ("e11", "e22", "e12", "DWATN", "FORCEX", "FORCEY"):
            bad[f"{it}/{k}"] = ndiff(prev(k), F(it, "Y04_before_lsr", k))
        for k in ("seaiceMassC", "seaiceMassU", "seaiceMassV"):
            y = F(it, "Y01_get_dynforcing", k)
            bad[f"{it}/{k}/row"] = ndiff(prev(k)[:, J0, :], y[:, J0, :])
            bad[f"{it}/{k}/col"] = ndiff(prev(k)[:, :, I0], y[:, :, I0])
        if it > 1:
            for k in sm.ICE_STATE:
                bad[f"{it}/{k}"] = ndiff(F(it - 1, "P00_seaice_model", k), F(it, "I00_seaice_begin", k))
            bad[f"{it}/sIceLoad"] = ndiff(F(it - 1, "P00_seaice_model", "sIceLoad"),
                                          F(it, "S03_ctrl_map_forcing", "sIceLoad"))
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    # not vacuous: the unwritten halo row of the ice masses holds the 1000 of seaice_init_varia.F:430
    assert np.all(F(3, "Y01_get_dynforcing", "seaiceMassC")[:, J0, :] == 1000.0)


def test_dyn_carry_reinit_equivalent():
    """The points of DYN_CARRY that SEAICE_DYNSOLVER does not write are never written by anything (they keep the
    seaice_init_varia values), and every written point is overwritten before it is read in the next step: the chain
    with DYN_CARRY reset to dyn_carry_init at every step gives bitwise the same outputs at iterations 2 and 3. (So
    M2.6b-2 may either carry DYN_CARRY in State, the literal choice, or rebuild it from dyn_carry_init every step.)"""
    res = chained()
    for it in (2, 3):
        ins, out, _ = res[it]
        ins0 = dict(ins, **sm.dyn_carry_init(L))
        assert any(ndiff(ins0[k], ins[k]) for k in sm.DYN_CARRY)
        out0, _ = run(ins0, record=False)
        for k in sm.INOUT:
            np.testing.assert_array_equal(np.asarray(out0[k]), np.asarray(out[k]), err_msg=(it, k))


# ---------------------------------------------------------------------------------------------------------------------
# task 3: the driver


@pytest.mark.parametrize("it", ITS)
def test_driver_chain_bitwise(it):
    """SEAICE_MODEL chained over iterations 1 -> it (own sea-ice state carried), grid + fixed fields from the files:
    I00 (uwind, vwind), I01 (all fields), I02, I03, I04, P00 bitwise at every point; uice_fd/vice_fd <= 1 ulp at
    < 200 points; LSOR counts as the Fortran. SEAICE_MODEL does not touch surfaceForcing*, pLoad, phi0surf (P00 ==
    S03). Measured: 0 differing values in every field, iterations 1-3."""
    ins, out, rec = chained()[it]
    bad = stage_mismatches(it, rec)
    for k in sm.EXF_INOUT:  # the inputs were the pre-exchange values
        assert ndiff(ins[k], F(it, "I00_seaice_begin", k)) > 0, k
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    for k in I01_ULP:
        ref = F(it, "I01_dynsolver", k)
        assert rel(rec["I01"][k], ref) < 1e-15 and ndiff(rec["I01"][k], ref) < 200, k
    counts = tuple(int(pr["L04"]["ICOUNT1"]) for pr in rec["dyn"]["passes"])
    assert counts == COUNTS[it]
    for k in PASS_THROUGH:
        assert ndiff(F(it, "P00_seaice_model", k), F(it, "S03_ctrl_map_forcing", k)) == 0, k
    for k in ("saltWtrIce", "frWtrIce"):          # never written in V4r4 (only seaice_init_varia.F:347-348)
        assert not np.any(F(it, "I04_growth", k)), k
    assert not np.any(F(it, "P00_seaice_model", "saltPlumeDepth"))
    for k in sm.INOUT:  # out == the P00 record, and the carried arrays are the I01 ones
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(rec["P00"][k]), err_msg=k)
    # not vacuous: ice moves, grows and melts
    assert np.abs(np.asarray(out["UICE"])).max() > 0.05 and np.abs(np.asarray(out["HEFF"] - ins["HEFF"])).max() > 1e-4


def test_driver_negative_controls(monkeypatch):
    """Each planted error makes the chained gate fail at iteration 1: the post-growth exchanges skipped (P00 halos),
    SEAICE_REG_RIDGE applied before SEAICE_ADVDIFF (I03, I04), the wind exchange skipped (I00), clipping off (I01:
    only if some |uIce| > 0.40 before the clip -- counted and asserted)."""
    it = 1
    ins = inputs(it)
    e = env()

    def run_planted():
        """A new function object per plant: jax caches a trace per function, so re-jitting the same object would
        replay the trace of the previous plant."""
        return jax.jit(lambda *a: sm.seaice_model(*a, record=True))(e.P, e.g, e.sg, EX, ins)

    def bad_stages(rec):
        bad = {k: v for k, v in stage_mismatches(it, rec).items() if v}
        print("  planted error ->", len(bad), "fields differ:", dict(list(bad.items())[:6]))
        return set(bad)

    # (1) no exchanges after SEAICE_GROWTH
    monkeypatch.setattr(sm, "exchanges_after_growth", lambda flags, ex, d: dict(d))
    _, rec = run_planted()
    b = bad_stages(rec)
    assert {"P00/HEFF", "P00/Qnet", "P00/sIceLoad"} <= b and not any(s.startswith(("I01", "I03")) for s in b), b
    monkeypatch.undo()
    # (2) the wind exchange skipped
    monkeypatch.setattr(sm, "exch_winds", lambda ex, u, v: (u, v))
    _, rec = run_planted()
    assert {"I00/uwind", "I00/vwind"} <= bad_stages(rec)
    monkeypatch.undo()
    # (3) reg_ridge before advdiff: the regularisation acts on the advected fields' inputs instead
    real_adv, real_rr = sa.seaice_advdiff, sa.seaice_reg_ridge

    def adv_after_ridge(p, g, u, v, H, A, S, M):
        r = real_rr(e.P.ridge, H, A, S, jnp.zeros((L.nTiles, 7) + L.shape2d[1:]))
        return real_adv(p, g, u, v, r["HEFF"], r["AREA"], r["HSNOW"], M)

    def ridge_noop(p, H, A, S, T):
        z = jnp.zeros_like(H)
        return dict(HEFF=H, AREA=A, HSNOW=S, TICES=T, d_HEFFbyNEG=z, d_HSNWbyNEG=z)

    monkeypatch.setattr(sm, "sa", SimpleNamespace(seaice_advdiff=adv_after_ridge, seaice_reg_ridge=ridge_noop))
    _, rec = run_planted()
    b = bad_stages(rec)
    assert {"I03/HEFF", "I03/AREA", "I04/HEFF"} <= b, b
    monkeypatch.undo()
    # (4) clipping off
    _, rec_ok = chained()[it][1:]
    nclip = int(np.sum(np.abs(np.asarray(rec_ok["dyn"]["Y05"]["UICE"])) > 0.40)
                + np.sum(np.abs(np.asarray(rec_ok["dyn"]["Y05"]["VICE"])) > 0.40))
    print(f"points clipped at it {it}: {nclip}")
    assert nclip > 0
    P = e.P._replace(dyn=dataclasses.replace(e.P.dyn, SEAICE_clipVelocities=False))
    _, rec = run(ins, P=P)
    assert {"I01/UICE", "I01/VICE"} & bad_stages(rec)


def test_jit_params_traced():
    """The driver is one jitted function of arrays: SeaiceParams float fields are traced leaves (a new value reuses
    the compiled executable and changes the result); the static flags and `ad` select the program."""
    e = env()
    ins = inputs(1)
    out0, _ = run(ins)
    n0 = RUN._cache_size()
    P1 = e.P._replace(growth=dataclasses.replace(e.P.growth, SWFracB=e.P.growth.SWFracB * (1 + 1e-6)))
    out1, _ = run(ins, P=P1)
    assert RUN._cache_size() == n0
    assert ndiff(out1["Qnet"], out0["Qnet"]) > 0 and ndiff(out1["UICE"], out0["UICE"]) == 0
    leaves = jax.tree.leaves(e.P)
    assert len(leaves) > 80 and all(np.ndim(x) == 0 for x in leaves)


# ---------------------------------------------------------------------------------------------------------------------
# adjoint levels


def _cotangents(seed, keys, ref):
    rng = np.random.default_rng(seed)
    return {k: jnp.asarray(rng.standard_normal(np.shape(ref[k]))) for k in keys}


def test_ad_levels_forward_identical():
    """The forward of "no_dynamics" and "full" is byte-identical to "ecco" (the default, gated above) at every output
    and dump-stage record."""
    ins, out, rec = chained()[1]
    for ad in ("no_dynamics", "full"):
        o, r = run(ins, ad=ad)
        for a, b in zip(jax.tree.leaves((o, r)), jax.tree.leaves((out, rec))):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=ad)


def test_ecco_vjp_is_identity():
    """ad="ecco": for random cotangents of every output, the VJP returns exactly the same cotangent on every INOUT
    input and exactly 0 on every READ input (useSEAICEinAdMode = .FALSE.: TAF skips SEAICE_MODEL in reverse)."""
    e = env()
    ins = inputs(1)
    f = jax.jit(lambda i: jax.vjp(lambda x: sm.seaice_model(e.P, e.g, e.sg, EX, x, ad="ecco")[0], i))
    out, vjp = f(ins)
    ct = _cotangents(0, sm.INOUT, out)
    (g,) = vjp(ct)
    for k in sm.INOUT:
        np.testing.assert_array_equal(np.asarray(g[k]), np.asarray(ct[k]), err_msg=k)
    for k in sm.READ:
        assert not np.any(np.asarray(g[k])), k


def test_no_dynamics_blocks_vjp_identity():
    """ad="no_dynamics" (SEAICEuseDYNAMICSswitchInAd): the two skipped blocks of SEAICE_DYNSOLVER have VJP == identity
    on the variables they overwrite (seaice_dyn.DYN_BLOCK_INOUT, CLIP_INOUT) and 0 on what they only read, exactly;
    the forward of the wrapped block is bitwise the plain one."""
    e = env()
    it = 1
    blk_in = dict(uIce=F(it, "I00_seaice_begin", "UICE"), vIce=F(it, "I00_seaice_begin", "VICE"),
                  TAUX=F(it, "Y02_ice_strength", "TAUX"), TAUY=F(it, "Y02_ice_strength", "TAUY"),
                  FORCEX0=F(it, "Y02_ice_strength", "FORCEX0"), FORCEY0=F(it, "Y02_ice_strength", "FORCEY0"),
                  HEFF=F(it, "I00_seaice_begin", "HEFF"), uVel=F(it, "S01_update_rstar_F", "uVel")[:, 0],
                  vVel=F(it, "S01_update_rstar_F", "vVel")[:, 0],
                  **{k: F(it, "Y04_before_lsr", k) for k in ("e11", "e22", "e12", "DWATN", "FORCEX", "FORCEY",
                                                             "seaiceMassC", "seaiceMassU", "seaiceMassV", "PRESS0",
                                                             "ZMAX", "ZMIN")})
    blk_in = {k: jnp.asarray(v) for k, v in blk_in.items()}
    rest = (e.P.dyn, e.g, e.sg, EX)
    f = jax.jit(lambda i, r: jax.vjp(lambda x: sd._dynamics_block_skipped(False, None, None, x, r)[0], i))
    out, vjp = f(blk_in, rest)
    plain, _ = jax.jit(lambda i, r: sd.dynamics_block(False, None, None, i, r))(blk_in, rest)
    for k in plain:
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(plain[k]), err_msg=k)
    assert ndiff(out["uIce"], F(it, "Y05_lsr", "UICE")) == 0
    ct = _cotangents(1, out.keys(), out)
    (g,) = vjp(ct)
    for k in blk_in:
        if k in sd.DYN_BLOCK_INOUT:
            np.testing.assert_array_equal(np.asarray(g[k]), np.asarray(ct[k]), err_msg=k)
        else:
            assert not np.any(np.asarray(g[k])), k
    cin = {k: blk_in[k] * 3.0 for k in ("uIce", "vIce")}
    o, vjp = jax.vjp(lambda x: sd._clip_block_skipped(x, None)[0], cin)
    ct = _cotangents(2, o.keys(), o)
    (g,) = vjp(ct)
    for k in sd.CLIP_INOUT:
        np.testing.assert_array_equal(np.asarray(g[k]), np.asarray(ct[k]), err_msg=k)


def _grad_env(ad):
    """d J / d ins for J = sum(wH*HEFF) + sum(wQ*Qnet) + sum(wF*fu) + sum(wU*UICE) of the driver's outputs
    (iteration 1; weights on wet interior points), cached per level. Model arguments passed to jit (not closed
    over)."""
    e = env()
    key = ("grad", ad)
    if key not in e.cache:
        ins = inputs(1)
        rng = np.random.default_rng(5)
        J_, I_ = L.js(1, L.sNy), L.is_(1, L.sNx)
        w = {}
        for k, m in (("HEFF", "HEFFM"), ("Qnet", "HEFFM"), ("fu", "seaiceMaskU"), ("UICE", "seaiceMaskU")):
            a = np.zeros(L.shape2d)
            a[:, J_, I_] = rng.standard_normal((L.nTiles, L.sNy, L.sNx))
            w[k] = jnp.asarray(a) * e.sg[m]

        def J(i, P, g, sg, ex, w):
            out, _ = sm.seaice_model(P, g, sg, ex, i, ad=ad)
            return sum(jnp.sum(w[k] * out[k]) for k in w)

        args = (e.P, e.g, e.sg, EX, w)
        t = time.time()
        g = jax.jit(jax.grad(J))(ins, *args)
        jax.block_until_ready(g)
        Jj = jax.jit(J)
        e.cache[key] = (ins, w, lambda i: Jj(i, *args), {k: np.asarray(v) for k, v in g.items()}, time.time() - t)
    return e.cache[key]


def test_ad_levels_effect():
    """On the live oracle state the three levels give different gradients: "ecco" = weights on HEFF/Qnet/fu/UICE
    exactly and 0 on theta_s; "no_dynamics" adds the thermodynamics (d/d theta_s != 0) but not the LSR; "full" adds
    the LSR (d/d fu and d/d UICE differ from "no_dynamics"). Gradients finite on every lane for every input."""
    ge, gn, gf = (_grad_env(ad) for ad in sm.AD_LEVELS)
    w, g_e, g_n, g_f = ge[1], ge[3], gn[3], gf[3]
    for k in ("HEFF", "Qnet", "fu", "UICE"):
        np.testing.assert_array_equal(g_e[k], np.asarray(w[k]), err_msg=k)
    assert not np.any(g_e["theta_s"]) and not np.any(g_e["atemp"])
    for name, g in (("no_dynamics", g_n), ("full", g_f)):
        for k, v in g.items():
            assert np.all(np.isfinite(v)), (name, k)
    assert np.any(g_n["theta_s"]) and ndiff(g_n["HEFF"], g_e["HEFF"]) > 100
    assert ndiff(g_f["fu"], g_n["fu"]) > 100 and ndiff(g_f["UICE"], g_n["UICE"]) > 100
    # theta_s enters SEAICE_GROWTH only (after the dynamics): same derivative with and without the LSR adjoint
    assert rel(g_f["theta_s"], g_n["theta_s"]) < 1e-12, rel(g_f["theta_s"], g_n["theta_s"])
    print("grad times (s):", {ad: round(x[4], 1) for ad, x in zip(sm.AD_LEVELS, (ge, gn, gf))})


def test_full_gradient_fd_sanity():
    """ad="full": d J / d theta_s at 2 ice-covered interior points vs central differences of the literal forward
    (theta_s enters SEAICE_GROWTH only, so the LSR iterate is unaffected), h = 1e-4 K."""
    ins, w, Jj, g, _ = _grad_env("full")
    e = env()
    heff, th = np.asarray(ins["HEFF"]), np.asarray(ins["theta_s"])
    # away from the freezing-point branch theta >= tempFrz (seaice_growth.F:1042; tempFrz = SEAICE_tempFrz0 = -1.96,
    # SEAICE_dTempFrz_dS = 0 in data.seaice)
    pts = np.argwhere((heff > 0.1) & (np.asarray(e.sg["HEFFM"]) > 0) & (th > -1.9))
    pts = pts[(pts[:, 1] >= L.jj(3)) & (pts[:, 1] <= L.jj(L.sNy - 2)) & (pts[:, 2] >= L.ii(3))
              & (pts[:, 2] <= L.ii(L.sNx - 2))]
    assert len(pts) > 20
    for t, j, i in pts[:: len(pts) // 2][:2]:
        h = 1e-4
        d = np.zeros(L.shape2d)
        d[t, j, i] = h
        fd = (float(Jj(dict(ins, theta_s=ins["theta_s"] + d)))
              - float(Jj(dict(ins, theta_s=ins["theta_s"] - d)))) / (2 * h)
        print(f"d J/d theta_s at {(t, j, i)}: grad {g['theta_s'][t, j, i]:.10e} fd {fd:.10e}")
        assert abs(fd - g["theta_s"][t, j, i]) <= 1e-6 * max(abs(fd), 1e-12), (t, j, i, fd, g["theta_s"][t, j, i])


# ---------------------------------------------------------------------------------------------------------------------
# task 4: placement on the host CPU


def test_host_cpu_placement():
    """The driver runs on an explicitly chosen CPU device with its inputs committed there, while the "ocean" arrays
    live on another device (here fake CPU device 1 stands in for the GPU): device_put in, jitted driver, device_put
    back; outputs bitwise equal to the gate run; jax.grad through the two transfers (ecco level) returns cotangents
    on the ocean device."""
    e = env()
    host = jax.devices("cpu")[0]
    ocean = jax.devices()[1] if len(jax.devices()) > 1 else host
    ins_ocean = jax.device_put(inputs(1), ocean)
    args = jax.device_put((e.P, e.g, e.sg, EX), host)
    out_h, _ = RUN(*args, jax.device_put(ins_ocean, host), ad="ecco", record=False)
    assert all(x.devices() == {host} for x in jax.tree.leaves(out_h))
    out = jax.device_put(out_h, ocean)
    _, ref, _ = chained()[1]
    for k in sm.INOUT:
        assert out[k].devices() == {ocean}
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(ref[k]), err_msg=k)

    def J(i):
        o, _ = RUN(*args, jax.device_put(i, host), ad="ecco", record=False)
        return jnp.sum(jax.device_put(o["HEFF"], ocean))

    g = jax.grad(J)(ins_ocean)
    assert g["HEFF"].devices() == {ocean} and float(jnp.sum(g["HEFF"])) == float(np.asarray(ins_ocean["HEFF"]).size)
