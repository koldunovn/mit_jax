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

## Task 2 — audit of the two override trees (2026-09-23)

- Read the diff before trusting a file list. The plan named five flux-forced overrides (`apply_forcing`, `impldiff`,
  `mom_vecinv`, `mom_vi_hdissip`, `momentum_correction_step`) as physics; all five only add budget diagnostics, and
  the flux-forced run does not even switch diagnostics on. Porting them "because they are overrides" would have added
  code with no reference to test it against. Likewise CATALOG.md's "3 forward changes" were 1: `phiHydLow` at init
  feeds only the ECCO cost, and the u/v control fix changes nothing when `data.ctrl` lists both controls.
- The forward differences between c66g and V4r4 live in CPP options and namelists, not in overridden source. The
  audit therefore generates a table of every macro whose state differs (`scripts/audit_overrides.py`); Task 8's
  `config_cpp.py` must read the build's headers, never assume c66g defaults.
- A compiled package changes forward code even when it only serves the adjoint: `pkg/autodiff` in `packages.conf`
  defines `ALLOW_AUTODIFF`, and c66g then resets the r* variables every step (`forward_step.F:418–450`), zeroes
  `saltPlumeFlux` each step (`do_oceanic_phys.F:286–298`), etc. The oracle must be built with the full package list.
- The forward `AUTODIFF_INADMODE_SET/UNSET` are empty; only their TAF adjoints flip `viscFacAdj` and the package
  switches. V4r4's `mom_calc_visc.F` scales all four 3-D viscosity fields by `viscFacAdj` (c66g: two) — the JAX
  `viscFacInAd` seam must match the override, not c66g.
- The audit is a test: the inventory and CPP tables in `docs/OVERRIDES.md` are generated (`--update`) and compared
  byte-for-byte (`--check`); every changed file needs a hand-written classification row. Negative controls: planted
  new file, missing row, extra row, edited count, missing markers.
