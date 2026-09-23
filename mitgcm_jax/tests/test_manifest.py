"""Every test file sits in exactly one cost group, and the audit that says so can fail."""

from pathlib import Path

from mitgcm_jax.tests.manifest import GROUPS, MANIFEST, audit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_matches_files_on_disk():
    unlisted, missing, bad_group = audit(REPO_ROOT)
    assert unlisted == [], f"test files not in manifest.py: {unlisted}"
    assert missing == [], f"manifest.py lists files that do not exist: {missing}"
    assert bad_group == [], f"unknown cost group (allowed {GROUPS}): {bad_group}"


def test_audit_negative_controls(tmp_path):
    (tmp_path / "mitgcm_jax/tests").mkdir(parents=True)
    (tmp_path / "scripts/tests").mkdir(parents=True)
    (tmp_path / "mitgcm_jax/tests/test_listed.py").touch()
    (tmp_path / "scripts/tests/test_planted.py").touch()
    manifest = {
        "mitgcm_jax/tests/test_listed.py": "tier1",
        "mitgcm_jax/tests/test_gone.py": "smoke",
        "mitgcm_jax/tests/test_typo.py": "tier9",
    }
    unlisted, missing, bad_group = audit(tmp_path, manifest)
    assert unlisted == ["scripts/tests/test_planted.py"]
    assert missing == ["mitgcm_jax/tests/test_gone.py", "mitgcm_jax/tests/test_typo.py"]
    assert bad_group == ["mitgcm_jax/tests/test_typo.py"]


def test_every_collected_test_carries_its_group_marker(request):
    group = MANIFEST["mitgcm_jax/tests/test_manifest.py"]
    assert request.node.get_closest_marker(group) is not None
