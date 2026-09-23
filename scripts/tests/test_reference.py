"""Fortran reference runs (plan Tasks 4-5): what the oracle must satisfy before anything is compared against it.

Runs are looked up by name in reference/runs.json (the test FAILS if a registered run is missing: on Levante that
means the reference is broken, not that the test does not apply).
"""

import json
from pathlib import Path

from mitgcm_jax.io.monitor import compare_monitors, read_monitor

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/work/ab0995/a270088/MIT/reference/runs")
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
    """Instrumented build, dumps off AND on, writes exactly what the plain build writes (plan Task 5)."""
    plain = run("smoke_ff_serial13")
    for name in ("smoke_ff_serial13_jaxdump_off", "smoke_ff_serial13_jaxdump_on"):
        other = run(name)
        assert all(_same_bytes(plain, other, 3).values()), name
        a, b = read_monitor(plain / "STDOUT.0000"), read_monitor(other / "STDOUT.0000")
        assert a == b, f"{name}: %MON differs"
    assert any((run("smoke_ff_serial13_jaxdump_on") / "jaxdump").glob("jd_0000000002_t*.bin"))


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
