# Porting lessons

One entry per task, written in the same commit as the task. Cite `file:line`; state what was measured.

## Task 1 — repository skeleton and environment (2026-09-23)

- Env `mitgcm-jax` reproduces the fesom-jax set exactly: `constraints.txt` is the full `pip freeze` of env
  `fesom-jax` (minus its editable package and conda-owned build tools); after install every package matched it.
  `test_env.py::test_pinned_versions_match_constraints` turns silent drift into a failure.
- Home is at 56/60 GB: pip must run with `--no-cache-dir` (its default cache is `~/.cache/pip`).
- The fake-device count must reach XLA before the first backend initialises. Setting `XLA_FLAGS` in the root
  `conftest.py` works because importing jax does not initialise a backend. Negative control: with
  `XLA_FLAGS=--xla_force_host_platform_device_count=2` preset, `test_four_fake_cpu_devices_and_shard_map_psum` fails.
- Cost groups live in one file (`mitgcm_jax/tests/manifest.py`); `conftest.py` applies them as markers and aborts
  collection on an unlisted test file (checked by removing an entry: "test files missing from manifest.py").
- The tier runner judges a run from the JUnit report (tests > 0, no failures/errors, testcases consistent with
  totals, ≤ 100 tests) plus pytest's exit code; each failure mode has a negative control in
  `scripts/tests/test_check_pytest_report.py`.
