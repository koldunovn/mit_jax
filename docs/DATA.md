# ECCO v4r4 input data and reference products

Everything lives under `$MITJAX_DATA` (`mitgcm_jax/paths.py`; on Levante `/work/ab0995/a270088/MIT/data/eccov4r4/`, home
quota is full). Staged 2026-09-23 from PO.DAAC with `scripts/fetch_eccov4r4.py` (stdlib; Earthdata login from
`~/.netrc`; system CA bundle, because the mambaforge one is broken). Checked by `scripts/tests/test_data_manifest.py`
(tier 1).

## Sources

| group (`fetch_eccov4r4.py`) | PO.DAAC collection / granule | size | status |
|---|---|---|---|
| `ancillary_small` | `ECCO_L4_ANCILLARY_DATA_V4R4`: `native_grid_files`, `input_init`, `doc`, `misc` | 0.28 GiB | fetched, unpacked |
| `products_fixed` | `ECCO_L4_GEOMETRY_LLC0090GRID_V4R4`, `ECCO_L4_OCEAN_3D_MIX_COEFFS_LLC0090GRID_V4R4` | 0.03 GiB | fetched |
| `products_snap_19920102` | T/S, SSH, OBP, sea-ice conc/thickness, sea-ice velocity snapshots at 1992-01-02T00 (11 steps in) | 0.04 GiB | fetched |
| `products_monthly_1992` | monthly means 1992: T/S, SSH, velocity, MLD, sea ice (5 × 12 files) | 0.72 GiB | fetched |
| `flux_forcing` | `ancillary_data_atm_flux_forcing_experiments_ECCO_V4r4.tar.gz` | 92.1 GiB | downloading (job 27632235) |
| `input_forcing` | `ancillary_data_input_forcing_ECCO_V4r4.tar.gz` (adjusted + unadjusted forcing, `other/`, control weights) | 191.5 GiB | downloading (job 27632236) |
| — | `data_constraints` (9 GB, cost-function obs), `output_insitu` (6 GB) | | not fetched (needed only for the cost function) |

The forcing is published **only as whole archives** (no per-year granules on PO.DAAC). ECCO Drive
(`ecco.jpl.nasa.gov/drive`) serves single files but needs its own WebDAV password, which we do not have. So the full
archives are downloaded and kept as our own copies; only what is needed is unpacked (1992 and all non-yearly files),
and later years are unpacked from the kept archive when needed.

Measured download rate: ~25 MiB/s per connection from Levante (login and `shared` nodes alike); two jobs in parallel
get ~25 MiB/s each.

## Layout

```
eccov4r4/
  ancillary_small/  products_*/  flux_forcing/  input_forcing/   downloaded files as published (+ MANIFEST.sha256)
  <group>/*.tar.gz.index.txt                                     member list of each archive (size, type, name)
  native_grid_files/ input_init/ doc/ misc/ ...                  unpacked archive contents (paths as in the archive)
  MANIFEST.extracted.sha256                                      sha256 of every unpacked file
  logs/                                                          fetch/extract logs (fetch-<jobid>.out, extract-<jobid>.out)
```

## Integrity
- Download: every file verified against PO.DAAC's published `.sha512` sidecar (all ancillary archives and the netCDF
  products have one) before its `.part` is renamed; interrupted downloads resume via HTTP Range.
- Own record: `<group>/MANIFEST.sha256` (sha256 + the published checksum) and `MANIFEST.extracted.sha256`
  (sha256 of each unpacked member, hashed while unpacking). Tier 1 re-hashes every recorded file under 2 GiB.
- Nothing is ever deleted or overwritten by the scripts: an existing file is accepted only if byte-identical.

## Jobs
- `sbatch scripts/fetch_job.sbatch GROUP ...` — download on the `shared` partition (has internet; survives logout).
  One job per group at a time (two jobs on the same group would append to the same `.part`).
- `sbatch [--dependency=afterok:<job>] scripts/extract_job.sbatch ARCHIVE --years 1992 [--exclude GLOB]` — one
  streaming pass: index + unpack. Queued: 27632252 (flux-forced, after 27632235), 27632253 (full forcing, excluding
  `*unadjusted*`, after 27632236).

## What a run needs (`scripts/audit_run_inputs.py RUNDIR --search DIR ...`)
The audit derives every input file from the run's namelists with rules citing the c66g code that opens them:
PARM05 files (initial T/S atlases only when `nIter0=0`, `ini_fields.F:30`), pickups (`pickup`, `pickup_ggl90`,
`pickup_seaice` when `nIter0>0`; `pickup_ecco` optional), EXF yearly files `<name>_YYYY` for the years the run window
touches (next year when the run passes the last record, Dec 31 21:00), GM/Redi 3-D files, ctrl `xx_*.0000000129`
+ weights, smooth operator files (`smooth2Dscales001`, `smooth3DscalesH/Z001`, norms), and cost inputs (prefix check).
Keys that look like file names but have no rule are printed as UNRESOLVED (none for V4r4).

State on 2026-09-23 with `input_init` + `native_grid_files` staged: every missing input of both trees is expected in
the two forcing archives — forcing files, control weights (`*_weights_*`, `r2.w*`), `geothermalFlux.bin`,
`runoff-2d-Fekete-1deg-mon-V4-SMOOTH.bin`, and (flux-forced) the all-zero `xx_*` forcing controls and
`weights_ones.data`. The full V4r4 cost inputs (`data_constraints`) are not staged; Task 4 decides whether the
reference runs drop `useECCO`/`useProfiles` (then `useCAL=.TRUE.` must be set explicitly).

## Facts learned from the files
- `input_init/pickup.0000000001` is float64, 403 records of 90×1170: `Uvel, Vvel, Theta, Salt, GuNm1, GuNm2, GvNm1,
  GvNm2` (50 levels each) + `EtaN, dEtaHdt, EtaH`. Its `.meta` says `timeStepNumber = 192840`
  (`timeInterval = 6.94224e8 s`): it was written by an earlier run and renamed to iteration 1. No tracer AB
  histories (`GtNm1`, `GsNm1`): the model starts tracer AB from the pickup without them (`pickupStrictlyMatch=F`;
  Task 8 ports the start-up logic).
- Bathymetry has **60,646 wet columns**, depth to 5998 m. Raw `.bin` inputs are big-endian float32 (reading them
  little-endian gives ±3.4e38 — the byte-order negative control in the test).
- `xx_*.0000000129.data` are float32, non-dimensional (`ALLOW_NONDIMENSIONAL_CONTROL_IO`), 50 levels (2-D for `etan`).
- `doc/STDOUT.0000` is the production run's standard output: `%MON` blocks every 1752 steps (73 days) for the whole
  1992–2017 run — a free long-run yardstick for the tier-3 twins.
