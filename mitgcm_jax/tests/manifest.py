"""Cost group of every test file — the one place a test file's tier is decided.

conftest.py applies the group as a pytest marker to every test collected from the file and refuses to
collect a test file that is not listed here, so a new file cannot silently land in no tier (or in the
10-minute tier 1 by accident). Paths are relative to the repository root.

Groups:
    smoke  seconds; safe on a login node (`pytest -m smoke`); also part of tier 1
    tier1  fast suite: < 10 min and < 100 tests on one CPU compute node, every commit; includes the integrated
           one-step gate (test_step_fluxforced.py), which is bitwise at every stage and so covers every kernel
    tier1x extended CPU suite (~15 min): per-kernel replay gates, negative controls and gradient-vs-FD checks, data
           re-hash; nightly and whenever a kernel changes (scripts/run_tier1x.sbatch)
    nightly long CPU suite (~1 h): the full-V4r4 step gates (every stage, free run, 24 steps), the full-tree adjoint
           modes and the sea-ice multi-step adjoint window (scripts/run_nightly.sbatch; Nikolay 2026-09-23)
    tier2  GPU suite, ~1 h, nightly / milestone
    tier3  milestone climate twins
"""

GROUPS = ("smoke", "tier1", "tier1x", "nightly", "tier2", "tier3")

# Tier 1 as run by scripts/run_tier1.sbatch.
TIER1_EXPRESSION = "smoke or tier1"
TIER1_MAX_TESTS = 100

MANIFEST = {
    "mitgcm_jax/tests/test_env.py": "smoke",
    "mitgcm_jax/tests/test_manifest.py": "smoke",
    "mitgcm_jax/tests/test_paths.py": "smoke",
    "scripts/tests/test_check_pytest_report.py": "smoke",
    "scripts/tests/test_overrides.py": "tier1",
    "scripts/tests/test_data_manifest.py": "tier1x",
    "scripts/tests/test_llc_layout.py": "tier1",
    "scripts/tests/test_reference.py": "tier1",
    "mitgcm_jax/tests/test_io_readers.py": "smoke",
    "mitgcm_jax/tests/test_monitor.py": "smoke",
    "mitgcm_jax/tests/test_dump_io.py": "smoke",
    "mitgcm_jax/tests/test_exchange.py": "tier1",
    "mitgcm_jax/tests/test_fullfield_grad.py": "tier1x",
    "mitgcm_jax/tests/test_gpu_sharding.py": "tier2",
    "mitgcm_jax/tests/test_adjoint_regression.py": "tier2",
    "mitgcm_jax/tests/test_adjoint_regression_full.py": "tier2",
    "mitgcm_jax/tests/test_sharded.py": "tier1",
    "mitgcm_jax/tests/test_sharded_step.py": "tier1x",
    "mitgcm_jax/tests/test_step_fluxforced.py": "tier1",
    "mitgcm_jax/tests/test_grid.py": "tier1",
    "mitgcm_jax/tests/test_monitor_stats.py": "tier1x",
    "mitgcm_jax/tests/test_exf_fluxforced.py": "tier1",
    "mitgcm_jax/tests/test_exf_full.py": "tier1x",
    "mitgcm_jax/tests/test_grid_load.py": "tier1",
    "mitgcm_jax/tests/test_phi_hyd.py": "tier1x",
    "mitgcm_jax/tests/test_dynamics.py": "tier1x",
    "mitgcm_jax/tests/test_ggl90.py": "tier1x",
    "mitgcm_jax/tests/test_mom_vecinv.py": "tier1x",
    "mitgcm_jax/tests/test_visc.py": "tier1x",
    "mitgcm_jax/tests/test_eos_sigma.py": "tier1x",
    "mitgcm_jax/tests/test_salt_plume.py": "tier1x",
    "mitgcm_jax/tests/test_gmredi.py": "tier1x",
    "mitgcm_jax/tests/test_thermo.py": "tier1x",
    "mitgcm_jax/tests/test_gad.py": "tier1x",
    "mitgcm_jax/tests/test_seaice_advdiff.py": "tier1x",
    "mitgcm_jax/tests/test_cg2d.py": "tier1x",
    "mitgcm_jax/tests/test_free_surface.py": "tier1x",
    "mitgcm_jax/tests/test_init.py": "tier1",
    "mitgcm_jax/tests/test_init_step.py": "tier1x",
    "mitgcm_jax/tests/test_adjoint_modes.py": "tier1",
    "mitgcm_jax/tests/test_adjoint_modes_grad.py": "tier1x",
    "mitgcm_jax/tests/test_checkpoint.py": "tier1x",
    "mitgcm_jax/tests/test_checkpoint_drivers.py": "tier1",
    "mitgcm_jax/tests/test_ctrl.py": "tier1x",
    "mitgcm_jax/tests/test_budgets.py": "tier1x",
    "mitgcm_jax/tests/test_means.py": "tier1",
    "mitgcm_jax/tests/test_seaice_growth.py": "tier1x",
    "mitgcm_jax/tests/test_seaice_dyn.py": "tier1x",
    "mitgcm_jax/tests/test_external_forcing_full.py": "tier1x",
    "mitgcm_jax/tests/test_init_full.py": "tier1x",
    "mitgcm_jax/tests/test_step_full.py": "nightly",
    "mitgcm_jax/tests/test_adjoint_modes_full.py": "nightly",
    "mitgcm_jax/tests/test_seaice_model.py": "tier1x",
    "mitgcm_jax/tests/test_seaice_adjoint_window.py": "nightly",
    "mitgcm_jax/tests/test_seaice_lsr_pallas.py": "tier1x",
    "mitgcm_jax/tests/test_seaice_lsr_gpu.py": "tier2",
    "mitgcm_jax/tests/test_seaice_gmres_bitwise.py": "tier1x",
    "mitgcm_jax/tests/test_seaice_gmres_bitwise_full.py": "nightly",
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
