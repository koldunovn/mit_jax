"""Fortran reference runs (plan Tasks 4-5): what the oracle must satisfy before anything is compared against it.

Runs are looked up by name in reference/runs.json (the test FAILS if a registered run is missing: on Levante that
means the reference is broken, not that the test does not apply).
"""

import json
from pathlib import Path

from mitgcm_jax import paths
from mitgcm_jax.io.mds import read_mds
from mitgcm_jax.io.monitor import compare_monitors, read_monitor

REPO = Path(__file__).resolve().parents[2]
RUNS = paths.REFERENCE_RUNS        # $MITJAX_REFERENCE_RUNS
REG = json.loads((REPO / "reference" / "runs.json").read_text())
STATE_FILES = ("T", "S", "Eta", "U", "V", "W", "PH")
SERIAL_ONLY_MISSING = {"dynstat_sst_max", "dynstat_sst_min", "dynstat_sst_mean", "dynstat_sst_sd", "dynstat_sst_del2",
                       "dynstat_sss_max", "dynstat_sss_min", "dynstat_sss_mean", "dynstat_sss_sd", "dynstat_sss_del2"}


def run(name):
    p = RUNS / REG[name]
    assert (p / "STDOUT.0000").exists(), f"reference run {name} -> {p} missing"
    assert "Execution ended Normally" in (p / "STDOUT.0000").read_text(errors="replace"), f"{name} did not end normally"
    return p


def _same_bytes(a, b, it):
    return {f: (a / f"{f}.{it:010d}.data").read_bytes() == (b / f"{f}.{it:010d}.data").read_bytes()
            for f in STATE_FILES}


def test_jaxdump_is_invisible_to_the_model():
    """Instrumented build, dumps off AND on, writes exactly what the plain build writes (plan Task 5). Full tree with
    the M2 stages (EXF bulk + sea ice, plan M2.0): also the ocean, sea-ice and GGL90 pickups, 2 steps."""
    plain = run("smoke_ff_serial13")
    for name in ("smoke_ff_serial13_jaxdump_off", "smoke_ff_serial13_jaxdump_on"):
        other = run(name)
        assert all(_same_bytes(plain, other, 3).values()), name
        a, b = read_monitor(plain / "STDOUT.0000"), read_monitor(other / "STDOUT.0000")
        assert a == b, f"{name}: %MON differs"
    assert any((run("smoke_ff_serial13_jaxdump_on") / "jaxdump").glob("jd_0000000002_t*.bin"))
    plain = run("smoke_full_v5_plain")
    for name in ("smoke_full_v5_jaxdump_off", "smoke_full_v5_jaxdump_on"):
        other = run(name)
        assert all(_same_bytes(plain, other, 3).values()), name
        for pk in ("pickup", "pickup_seaice", "pickup_ggl90"):
            assert (plain / f"{pk}.ckptA.data").read_bytes() == (other / f"{pk}.ckptA.data").read_bytes(), (name, pk)
        assert read_monitor(plain / "STDOUT.0000") == read_monitor(other / "STDOUT.0000"), f"{name}: %MON differs"
    assert any((run("smoke_full_v5_jaxdump_on") / "jaxdump").glob("jd_0000000002_t*.bin"))


def test_tile_layout_spread_is_small_and_recorded():
    """13 x 90x90 (serial) vs 96 x 30x30 (MPI): only summation order differs -> tiny, nonzero differences."""
    a = read_monitor(run("smoke_ff_serial13") / "STDOUT.0000")
    b = read_monitor(run("smoke_ff_mpi96") / "STDOUT.0000")
    diffs, only = compare_monitors(a, b)
    assert set(only) <= SERIAL_ONLY_MISSING
    worst = max(diffs.values())
    assert 0 < worst < 1e-8, worst            # recorded 8e-10 after 2 steps (docs/REFERENCE_RUNS.md)


def test_matched_restart_is_bitwise():
    """Restart from a pickup reproduces the continuous run exactly (both trees; ff needs the approved I6-reader
    deviation) -- the matched-restart mode of plan Task 5 depends on it."""
    for tree in ("ff", "full"):
        a, b = run(f"smoke_{tree}_restart_continuous"), run(f"smoke_{tree}_restart_from2")
        same = _same_bytes(a, b, 5)
        assert all(same.values()), (tree, same)


def test_ff_reader_fix_changes_no_numbers():
    """The ff I6-reader deviation is I/O only: same output as the pre-fix binary on the same run."""
    assert all(_same_bytes(run("smoke_ff_serial13_prefix_fix"), run("smoke_ff_serial13"), 3).values())


def test_ff_stage1_twin_bitwise_and_spread():
    """Flux-forced stage 1 (plan Task 4): two 1-day 96-rank runs of the same binary are bitwise identical; the
    13-tile serial run differs from them only by summation order (recorded 2026-09-23: U 7e-9, V 1.1e-8,
    W 6e-8, PH 1.4e-9 relative after 24 steps; dynstat %MON <= 7e-8, wvel mean)."""
    a, b, s = run("ref_ff_mpi96_1day_a"), run("ref_ff_mpi96_1day_b"), run("ref_ff_serial13_1day")
    assert all(_same_bytes(a, b, 25).values())
    ma, ms = read_monitor(a / "STDOUT.0000"), read_monitor(s / "STDOUT.0000")
    diffs, only = compare_monitors(ma, ms)
    assert set(only) <= SERIAL_ONLY_MISSING
    dyn = {k: v for k, v in diffs.items() if k[1].startswith("dynstat_")}
    assert 0 < max(dyn.values()) < 1e-6, max(dyn.values())


def test_full_stage1_twin_and_podaac_snapshot():
    """Full V4r4 stage 1 (plan Task 4): the 96-rank twin is bitwise; 11 steps from the V4r4 pickup reproduce the
    PO.DAAC native-grid snapshot of 1992-01-02T00 (float32 product of the ifort production run) within float32 +
    compiler floor. Recorded 2026-09-23: T max 4.0e-4 degC (rms 3.8e-7), S max 1.9e-4 (rms 1.3e-7), Eta max 7.8e-5 m
    (rms 3.2e-7); no dry-point values."""
    import importlib.util

    a, b = run("ref_full_mpi96_1day_a"), run("ref_full_mpi96_1day_b")
    assert all(_same_bytes(a, b, 25).values())
    spec = importlib.util.spec_from_file_location("c2p", REPO / "scripts" / "compare_to_podaac.py")
    c2p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2p)
    rows = c2p.compare(run("ref_full_mpi96_11steps"), 12)
    assert set(rows) >= {"T", "S", "Eta"}, rows
    for name, r in rows.items():
        assert r["max_abs"] < 1e-3 and r["rms"] < 1e-6 and r["dry_nonzero_model"] == 0, (name, r)


def test_mpi13_twin_is_bitwise_serial13():
    """13 ranks x one 90x90 tile (mpi13) == one process x 13 tiles (serial13), both trees, 1 day: state and pickups
    byte-identical, every common %MON value identical (mpi13 only adds the SST/SSS stats, printed with one tile per
    process). c66g CPP_EEOPTIONS.h:132 GLOBAL_SUM_ORDER_TILES sums per-tile values in global tile order whatever the
    decomposition, and W2_MAP_PROCS puts exch2 tile p+1 on rank p. The full comparison (every output file, AD tapes
    reassembled per tile) is recorded in docs/REFERENCE_RUNS.md."""
    for tree, pickups in (("ff", ("pickup", "pickup_ggl90")), ("full", ("pickup", "pickup_ggl90", "pickup_seaice"))):
        s, m = run(f"ref_{tree}_serial13_1day"), run(f"ref_{tree}_mpi13_1day")
        same = _same_bytes(s, m, 25)
        assert all(same.values()), (tree, same)
        for pk in pickups:
            assert (s / f"{pk}.ckptA.data").read_bytes() == (m / f"{pk}.ckptA.data").read_bytes(), (tree, pk)
        diffs, only = compare_monitors(read_monitor(s / "STDOUT.0000"), read_monitor(m / "STDOUT.0000"))
        assert set(only) == SERIAL_ONLY_MISSING, (tree, only)
        assert len(diffs) > 2000 and max(diffs.values()) == 0.0, (tree, max(diffs.items(), key=lambda kv: kv[1]))


# ---------------------------------------------------------------------------------------------------------------------
# M2.0: full-tree dump oracle (EXF bulk + sea-ice stages, reference/jaxdump/SUBSTEPS.md). One tier-1 test (budget).

ICE = ("AREA", "HEFF", "HSNOW", "TICES", "UICE", "VICE")
PICKUP_SEAICE = {"siTICE": "TICES", "siAREA": "AREA", "siHEFF": "HEFF", "siHSNOW": "HSNOW", "siUICE": "UICE",
                 "siVICE": "VICE"}
M1_FILES = {"do_oceanic_phys.F", "dynamics.F", "forward_step.F", "salt_integrate.F", "solve_for_pressure.F",
            "temp_integrate.F", "thermodynamics.F"}
M2_FILES = {"exf_getforcing.F", "exf_radiation.F", "exf_bulkformulae.F", "seaice_model.F", "seaice_dynsolver.F",
            "seaice_lsr.F", "seaice_advdiff.F", "seaice_growth.F"}


def _instrument():
    import importlib.util

    path = REPO / "reference" / "jaxdump" / "instrument.py"
    spec = importlib.util.spec_from_file_location("jaxdump_instrument", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dumpset_parallel(directory):
    """mitgcm_jax.io.dump.DumpSet of `directory` with the record headers of all files read in parallel threads: the
    serial index of a full-tree oracle takes ~110 s on cold Lustre (~3 ms per record header, ~50k headers), ~3 s this
    way. Same index as DumpSet(directory) (tile-file order within one iteration does not matter for the lookups).
    Also asserts that no (iteration, stage, field, tile) was written twice (DumpSet would keep only the last one: a
    stage inside a loop needs its '_p<n>' suffix)."""
    from concurrent.futures import ThreadPoolExecutor

    from mitgcm_jax.io.dump import DumpSet, read_file

    files = sorted(Path(directory).glob("jd_*_t*.bin"))
    assert files, directory
    with ThreadPoolExecutor(min(len(files), 64)) as ex:
        per_file = list(ex.map(lambda f: read_file(f, lazy=True), files))
    ds = DumpSet.__new__(DumpSet)
    ds.dir, ds.index = Path(directory), {}
    dup = []
    for recs in per_file:
        for r in recs:
            slot = ds.index.setdefault((r.iter, r.stage, r.field), {})
            if r.tile in slot:
                dup.append((r.iter, r.stage, r.field, r.tile))
            slot[r.tile] = r
    assert not dup, dup[:5]
    ds.order = list(ds.index)
    return ds


def _mon_close(x, ref):
    """%MON prints 14 significant digits (1PE22.13): equal to that precision."""
    return x == ref if ref == 0 else abs(x - ref) <= 1e-13 * abs(ref)


def _check_instrument(ins, tmp):
    """The flux-forced tree gets only the M1 files (M2 stages are full-tree only, so its instrumentation is unchanged),
    the full tree the EXF/sea-ice files too; every anchor resolves (a drifted source fails loudly) and every generated
    line fits fixed-form 72 columns (the first M2 build failed on that)."""
    for tree, files in (("ff", M1_FILES), ("full", M1_FILES | M2_FILES)):
        ins.instrument(tree, tmp / tree)
        assert {q.name for q in (tmp / tree).iterdir()} == files | {"jaxdump.F", "JAXDUMP.h"}, tree
        for f in files:
            lines = (tmp / tree / f).read_text().split("\n")
            for i, ln in enumerate(lines):
                if "CALL JAXDUMP_" not in ln:
                    continue
                j = i
                while True:
                    assert len(lines[j]) <= 72, (tree, f, j + 1, lines[j])
                    j += 1
                    if not lines[j].startswith("     &"):
                        break


def _check_stages(ins, ds):
    """Every full-tree stage (M1 + M2) at iterations 1-3 on all 13 tiles (the packed vertical grid: tile 1), the LSR
    Picard-loop stages once per pass (_p1, _p2; SEAICEnonLinIterMax=2), and nothing else."""
    expect = set()
    for e in ins.entries("full"):
        stage, opts = e[4], e[8]
        expect |= {f"{stage}_p1", f"{stage}_p2"} if "loop" in opts else {stage}
    assert "L02_lsr_coeffs_p2" in expect and "H04_growth_solve4temp" in expect and "X05b_bulk_iter" in expect
    for it in (1, 2, 3):
        have = {st for (i, st, _f) in ds.index if i == it}
        assert have == expect, (it, sorted(expect - have), sorted(have - expect))
    bad = [k for k, recs in ds.index.items()
           if sorted(recs) != ([1] if k[1:] == ("G00_geometry", "vertical") else list(range(1, 14)))]
    assert not bad, bad[:5]  # the packed vertical grid is a tile-1 record by design
    assert sorted({k[0] for k in ds.index}) == [1, 2, 3]


def _check_monitor_vs_plain(p):
    """%MON of the 3 dumped steps = the plain full serial13 build's 1-day run, bitwise (its block 4 also holds step 4's
    EXF and cg2d lines)."""
    a, b = read_monitor(p / "STDOUT.0000"), read_monitor(run("ref_full_serial13_1day") / "STDOUT.0000")
    assert sorted(a) == [1, 2, 3, 4]
    for it in (1, 2, 3, 4):
        extra = set(b[it]) - set(a[it])
        assert set(a[it]) <= set(b[it]), it
        assert not extra if it < 4 else all(k.startswith(("exf_", "cg2d_")) for k in extra), (it, sorted(extra))
        assert {k: a[it][k] for k in a[it]} == {k: b[it][k] for k in a[it]}, it
    assert any(k.startswith("seaice_") for k in a[2]) and any(k.startswith("exf_") for k in a[2])


def _check_ice_state(oracle, ds, p):
    """Sea-ice state vs what the model reads and writes, bitwise: start of step 1 = input pickup_seaice.0000000001,
    end of step n (after SEAICE_MODEL, P00) = start of step n+1 (S00i, halos included), end of step 3 = the
    pickup_seaice.ckptA written at iteration 4. Negative control: step 1 vs step 3 differ."""
    import numpy as np

    for n in (1, 2):
        for f in ICE:
            a, b = oracle.field(ds, n, "P00_seaice_model", f), oracle.field(ds, n + 1, "S00i_begin_ice_exf", f)
            assert np.array_equal(a, b), (n, f)
    assert not np.array_equal(oracle.field(ds, 1, "S00i_begin_ice_exf", "HEFF"),
                              oracle.field(ds, 3, "S00i_begin_ice_exf", "HEFF"))
    for fname, it, stage in (("pickup_seaice.0000000001", 1, "S00i_begin_ice_exf"),
                             ("pickup_seaice.ckptA", 3, "P00_seaice_model")):
        arr, meta = read_mds(p / fname)
        for r, fl in enumerate(meta["fldList"]):
            got = ds.compact(it, stage, PICKUP_SEAICE[fl])[0]
            assert np.array_equal(got, arr[r]), (fname, fl, float(np.max(np.abs(got - arr[r]))))


def _check_monitor_stats(oracle, ds, p):
    """Max/min over the monitor's points (maskIn*, interior) of the dumped fields reproduce the model's %MON:
    seaice_* at tsnumber n = S00i of step n (n=4: P00 of step 3); exf_* of step n = X06 (b group) / X07 (x group) of
    step n, except hflux from X08: EXF_MONITOR runs inside EXF_GETFORCING after hflux += swflux (SHORTWAVE_HEATING) and
    before EXF_MAPFIELDS, which clips ustress/vstress in place at windstressmax (2 N/m2; seen as a max of exactly 2.0).
    Negative control: another step's EXF fields mostly do not match."""
    mon = read_monitor(p / "STDOUT.0000")
    mask = {k: oracle.field(ds, 1, "G00_geometry", f"maskIn{k}")[:, 4:-4, 4:-4] > 0 for k in "CWS"}

    def stats(it, stage, name, m):
        v = oracle.field(ds, it, stage, name)
        v = (v[:, 0] if v.ndim == 4 else v)[:, 4:-4, 4:-4][mask[m]]
        return float(v.max()), float(v.min())

    for n in (1, 2, 3, 4):
        it, stage = (n, "S00i_begin_ice_exf") if n < 4 else (3, "P00_seaice_model")
        for f, mn, m in (("AREA", "area", "C"), ("HEFF", "heff", "C"), ("HSNOW", "hsnow", "C"), ("UICE", "uice", "W"),
                         ("VICE", "vice", "S")):
            mx_, mi_ = stats(it, stage, f, m)
            assert _mon_close(mx_, mon[n][f"seaice_{mn}_max"]) and _mon_close(mi_, mon[n][f"seaice_{mn}_min"]), (n, f)
    exf = (("X07_exf_getsurfacefluxes", ("ustress", "vstress", "sflux", "swflux", "apressure")),
           ("X08_exf_mapfields", ("hflux",)),
           ("X06_exf_hflux_sflux", ("wspeed", "atemp", "aqh", "lwflux", "evap", "precip", "swdown", "lwdown",
                                    "runoff")))
    wrong = 0
    for n in (1, 2, 3):
        for stage, names in exf:
            for f in names:
                mx_, mi_ = stats(n, stage, f, "C")
                assert _mon_close(mx_, mon[n][f"exf_{f}_max"]) and _mon_close(mi_, mon[n][f"exf_{f}_min"]), (n, f)
                o = stats(3 if n < 3 else 1, stage, f, "C")
                wrong += not (_mon_close(o[0], mon[n][f"exf_{f}_max"]) and _mon_close(o[1], mon[n][f"exf_{f}_min"]))
    return wrong


def _check_exf(oracle, ds):
    """X08 (after EXF_MAPFIELDS) = the M1 stage S02 (after LOAD_FIELDS_DRIVER) bitwise; hflux/sflux at X06 = the
    EXF_GETFORCING formula (-hs - hl + lwflux; evap - precip - runoff; times maskC k=1; SHORTWAVE_HEATING) from the
    dumped terms, exactly. Negative control: a flipped sign of hs does not match."""
    import numpy as np

    interior = (slice(None), slice(4, -4), slice(4, -4))
    mC = oracle.field(ds, 1, "G00_geometry", "maskC")[:, 0][interior]
    for it in (1, 2, 3):
        for f in ("ustress", "vstress", "hflux", "sflux", "fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad"):
            assert np.array_equal(oracle.field(ds, it, "X08_exf_mapfields", f),
                                  oracle.field(ds, it, "S02_load_fields", f)), (it, f)
        g = {f: oracle.field(ds, it, "X06_exf_hflux_sflux", f)[interior]
             for f in ("hs", "hl", "lwflux", "evap", "precip", "runoff", "hflux", "sflux")}
        assert np.array_equal((-g["hs"] - g["hl"] + g["lwflux"]) * mC, g["hflux"]), it
        assert np.array_equal(((g["evap"] - g["precip"]) - g["runoff"]) * mC, g["sflux"]), it
        assert not np.array_equal((g["hs"] - g["hl"] + g["lwflux"]) * mC, g["hflux"]), it


def test_full_oracle_m2_stages_and_consistency(tmp_path):
    """Full-tree dump oracle full_jaxdump_v5 (plan M2.0): the instrumentation is tree-scoped and fixed-form clean; the
    oracle has every stage at iterations 1-3; its %MON equals the plain build's; its sea-ice dumps equal the pickups the
    model reads/writes and chain bitwise across steps; dumped ice/EXF fields reproduce the model's own %MON max/min;
    EXF stages agree with each other and with the hflux/sflux formula. Recorded 2026-09-23: 60648 records (4668 keys),
    12.5 GB per iteration (M2 stages 0.78 GB); LSOR sweeps (ICOUNT1 = ICOUNT2) per Picard pass 178/118, 112/82, 84/58 at
    iterations 1/2/3; 16 s on a compute node."""
    from mitgcm_jax.tests import oracle

    p = run(oracle.FULL)
    ins = _instrument()
    _check_instrument(ins, tmp_path)
    ds = _dumpset_parallel(p / "jaxdump")
    _check_stages(ins, ds)
    _check_monitor_vs_plain(p)
    _check_ice_state(oracle, ds, p)
    wrong = _check_monitor_stats(oracle, ds, p)
    assert wrong >= 40, wrong  # negative control: of 45 EXF max/min checks, 45 fail on another step (2026-09-23)
    _check_exf(oracle, ds)
