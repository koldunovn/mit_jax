"""Cost group of every test file — the one place a test file's tier is decided.

conftest.py applies the group as a pytest marker to every test collected from the file and refuses to
collect a test file that is not listed here, so a new file cannot silently land in no tier (or in the
10-minute tier 1 by accident). Paths are relative to the repository root.

Groups:
    smoke  seconds; safe on a login node (`pytest -m smoke`); also part of tier 1
    tier1  fast suite: < 10 min and < 100 tests on one CPU compute node, every commit
    tier2  GPU suite, ~1 h, nightly / milestone
    tier3  milestone climate twins
"""

GROUPS = ("smoke", "tier1", "tier2", "tier3")

# Tier 1 as run by scripts/run_tier1.sbatch.
TIER1_EXPRESSION = "smoke or tier1"
TIER1_MAX_TESTS = 100

MANIFEST = {
    "mitgcm_jax/tests/test_env.py": "smoke",
    "mitgcm_jax/tests/test_manifest.py": "smoke",
    "scripts/tests/test_check_pytest_report.py": "smoke",
    "scripts/tests/test_overrides.py": "tier1",
    "scripts/tests/test_data_manifest.py": "tier1",
    "scripts/tests/test_llc_layout.py": "tier1",
    "mitgcm_jax/tests/test_io_readers.py": "smoke",
    "mitgcm_jax/tests/test_monitor.py": "smoke",
}

# Directories searched for test files (must match pyproject testpaths).
TEST_DIRS = ("mitgcm_jax/tests", "scripts/tests")


def audit(root, manifest=None):
    """Return (unlisted, missing, bad_group): test files on disk not in the manifest, manifest entries
    with no file, and entries whose group is not one of GROUPS."""
    from pathlib import Path

    manifest = MANIFEST if manifest is None else manifest
    root = Path(root)
    on_disk = {p.relative_to(root).as_posix()
               for d in TEST_DIRS if (root / d).is_dir()
               for p in (root / d).rglob("test_*.py")}
    unlisted = sorted(on_disk - set(manifest))
    missing = sorted(set(manifest) - on_disk)
    bad_group = sorted(f for f, g in manifest.items() if g not in GROUPS)
    return unlisted, missing, bad_group
