# Running the one-year experiments (1992)

This guide reproduces the two one-year runs of [VALIDATION.md](VALIDATION.md) with the JAX model, starting from an
empty machine: software, input data, run directories, the runs, checks against known numbers, plots and movies.
Commands are written for any Linux system with an NVIDIA GPU; the last two sections give the details for DKRZ Levante
(where the runs were made) and NASA NAS Cabeus.

| | flux-forced ocean (`ff`) | full model (`full`) |
|---|---|---|
| surface forcing | 6-hourly air-sea fluxes of the V4r4 flux-forced experiment | 6-hourly V4r4 adjusted atmospheric state, bulk formulae |
| sea ice | none | dynamic-thermodynamic sea ice |
| period | 1992-01-01 13:00 to 1992-12-31 13:00: 8760 one-hour steps | same |
| initial state | V4r4 pickup files + V4r4 control adjustments | same + sea-ice pickup |
| one GPU (s/step) | 0.17 A100, 0.10 GH200 | 0.43 A100-80, 0.29 GH200 |
| one year on one GPU | about 30 min (A100) | 46-48 min (GH200), about 70 min (A100-80) |
| input data used | 9.7 GiB (from 284 GiB of downloads) | 6.0 GiB (from 192 GiB of downloads) |
| output of the year run | about 27 GB | about 27 GB |

Tested hardware: NVIDIA A100-40 and A100-80 (x86_64) and GH200 (aarch64) GPUs; CPUs work for short runs (a full-model
step takes about 5 s on 22 cores).

The commands below are run from the repository root. `python` is the interpreter of the environment of section 2.

## 1. Code

```bash
git clone https://github.com/koldunovn/mit_jax.git
cd mit_jax
# the V4r4 namelists (and, for Fortran builds only, the V4r4 code overrides); the scripts expect this directory name
git clone https://github.com/ECCO-GROUP/ECCO-v4-Configurations.git
git -C ECCO-v4-Configurations checkout 66282489b641bfc31ff63caaeb11209ea19242d7
```

Only for Fortran reference runs (section 11): `git clone https://github.com/MITgcm/MITgcm.git MITgcm_c66g` and
`git -C MITgcm_c66g checkout checkpoint66g`. Both clones are ignored by git.

## 2. Python environment

Python 3.12 and the package versions pinned in `constraints.txt` (JAX 0.10.1; [ENV.md](ENV.md)). The CUDA 12
libraries come as pip wheels, so no CUDA installation is needed, only an NVIDIA driver that supports CUDA 12.

```bash
ENV=$HOME/envs/mitgcm-jax                 # any location
conda create -y -p $ENV python=3.12.13 pip  # or mamba / micromamba; a venv from any Python 3.12 works as well
$ENV/bin/pip install --no-cache-dir -e ".[cuda,dev]" -c constraints.txt     # CPU only: ".[dev]"
```

The same command works on aarch64 (GH200) nodes: every pinned package exists as an aarch64 wheel. For nodes without
internet access, `scripts/dolpung/fetch_wheels.sbatch` downloads the aarch64 wheels on a machine with internet and
`scripts/dolpung/make_env.sbatch` installs them offline (both written for DKRZ; the pip commands in them are generic).

Check the installation on a GPU node:

```bash
python scripts/check_env.py                         # GPU listed, float64 matmul OK, pinned versions: "CHECK OK"
JAX_PLATFORMS=cpu python -m pytest -m smoke -q      # 17 passed, a few seconds
```

Plots and movies (section 9) use a second environment with [nereus](https://github.com/koldunovn/nereus), cartopy
and ffmpeg:

```bash
NENV=$HOME/envs/nereus
conda create -y -p $NENV -c conda-forge python=3.11 numpy scipy xarray netcdf4 matplotlib cartopy cmocean dask healpy ffmpeg
$NENV/bin/pip install --no-cache-dir git+https://github.com/koldunovn/nereus.git
```

## 3. Paths

All machine-specific locations are environment variables, defined in [mitgcm_jax/paths.py](../mitgcm_jax/paths.py).
Set the work root (a file system with room for about 350 GB) before everything else:

```bash
export MITJAX_WORK=/big/disk/mitjax            # default: /work/ab0995/a270088/MIT (DKRZ Levante)
eval "$(python mitgcm_jax/paths.py --sh)"      # exports the derived variables below for the shell commands
python mitgcm_jax/paths.py                     # prints the resolved paths
```

| variable | default | contents |
|---|---|---|
| `MITJAX_WORK` | `/work/ab0995/a270088/MIT` | root of everything below |
| `MITJAX_DATA` | `$MITJAX_WORK/data/eccov4r4` | ECCO v4r4 downloads and unpacked input files |
| `MITJAX_GRID_DIR` | `$MITJAX_DATA/native_grid_files` | `tile00[1-5].mitgrid` |
| `MITJAX_REFERENCE` | `$MITJAX_WORK/reference` | Fortran builds (`bin/`, `build/`, `logs/`) |
| `MITJAX_REFERENCE_RUNS` | `$MITJAX_REFERENCE/runs` | run directories (JAX and Fortran) |
| `MITJAX_RUNS_JAX` | `$MITJAX_WORK/runs_jax` | output of JAX runs (by convention) |
| `MITJAX_RUNS` | `$MITJAX_WORK/runs` | output of test and benchmark jobs |

Each variable can be set on its own; unset ones derive from the one above (repeat the `eval` line after changing
one). Batch scripts also read `MITJAX_PYTHON`
(interpreter of the model environment), `MITJAX_NEREUS_PYTHON` (plotting environment) and `MITJAX_TREE` (repository
to run; default: the checkout that holds the script).

## 4. Input data

The ECCO v4r4 inputs are published by NASA PO.DAAC. `scripts/fetch_eccov4r4.py` (standard library only) queries the
NASA CMR catalogue, downloads with an Earthdata login, verifies each file against the checksum PO.DAAC publishes,
resumes interrupted downloads, and never deletes or overwrites anything.

**Earthdata login.** Create an account at <https://urs.earthdata.nasa.gov> and put it into `~/.netrc`:

```bash
echo "machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD" >> ~/.netrc
chmod 600 ~/.netrc
```

**Download.** The forcing is published only as whole archives, so both runs need large downloads:

| group | contents | download | needed by |
|---|---|---|---|
| `ancillary_small` | grid (`native_grid_files`), initial state and fixed fields (`input_init`), ECCO's own run output (`doc`), `misc` | 0.28 GiB | both |
| `input_forcing` | full-model forcing (all years), control weights, geothermal flux, runoff | 191.5 GiB | both |
| `flux_forcing` | flux-forced experiment forcing (all years) and its zero forcing controls | 92.1 GiB | ff |
| `products_fixed` | V4r4 grid geometry and mixing coefficients (netCDF) | 0.03 GiB | tests only |

```bash
python scripts/fetch_eccov4r4.py groups                          # the groups and their PO.DAAC collections
python scripts/fetch_eccov4r4.py list ancillary_small            # what would be downloaded (no download)
python scripts/fetch_eccov4r4.py fetch ancillary_small
python scripts/fetch_eccov4r4.py fetch input_forcing             # ~2.2 h at 25 MiB/s
python scripts/fetch_eccov4r4.py fetch flux_forcing              # ~1.1 h (flux-forced run only)
```

Files land in `$MITJAX_DATA/<group>/`. Run long downloads in a batch job or a terminal multiplexer; a new `fetch` of
the same group continues from the partial `.part` file (run only one fetch per group at a time). If `list` or `fetch`
reports that CMR returned no granules or times out, the NASA CMR search service is having problems; try again later.

**Unpack** only what the 1992 runs need (all non-yearly files and the 1992 files; the archives are kept, so other
years can be unpacked later):

```bash
D=$MITJAX_DATA
python scripts/fetch_eccov4r4.py extract $D/ancillary_small/ancillary_data_{native_grid_files,input_init,doc,misc}_ECCO_V4r4.tar.gz
python scripts/fetch_eccov4r4.py extract $D/input_forcing/ancillary_data_input_forcing_ECCO_V4r4.tar.gz \
    --years 1992 --exclude '*unadjusted*'                                                   # ~25 min
python scripts/fetch_eccov4r4.py extract $D/flux_forcing/ancillary_data_atm_flux_forcing_experiments_ECCO_V4r4.tar.gz \
    --years 1992                                                                            # ~15 min, ff only
```

Each archive gets a member list (`<archive>.index.txt`); unpacked files are recorded with their sha256 in
`$MITJAX_DATA/MANIFEST.extracted.sha256`. Afterwards `$MITJAX_DATA` holds `native_grid_files/`, `input_init/`, `doc/`,
`misc/`, `input_forcing/{adjusted,control_weights,other,...}/` and `atm_flux_forcing_experiments/`.
[DATA.md](DATA.md) has more on the data and the integrity checks.

**Files the runs read** (`reference/make_rundir.py` finds them under `$MITJAX_DATA`, whatever the sub-directory):

| directory | files | ff | full |
|---|---|---|---|
| `native_grid_files/` | `tile001.mitgrid` ... `tile005.mitgrid` | yes | yes |
| `input_init/` | `bathy_eccollc_90x50_min2pts.bin`, `fenty_biharmonic_visc_v11.bin`, `total_{diffkr,kapgm,kapredi}_r009bit11.bin`, `pickup.0000000001.{data,meta}`, `pickup_ggl90.0000000001.{data,meta}`, `smooth2Dnorm001.{data,meta}`, `smooth2Dscales001`, `smooth3Dnorm001.{data,meta}`, `smooth3DscalesH001`, `smooth3DscalesZ001`, `xx_{diffkr,etan,kapgm,kapredi,salt,theta,uvel,vvel}.0000000129.{data,meta}` | yes | yes |
| `input_init/` | `pickup_seaice.0000000001.{data,meta}` | | yes |
| `input_forcing/control_weights/` | `r2.w{diffkr,kapgm,kapredi}Fldv2.data`, `{Salt,Theta,Uvel,Vvel}_weights_nonseasonal_rmsv2_areascaled.bin`, `SSH_weights_nonseasonal_rms_areascaled.bin` | yes | yes |
| `input_forcing/other/` | `geothermalFlux.bin` | yes | yes |
| `input_forcing/other/` | `runoff-2d-Fekete-1deg-mon-V4-SMOOTH.bin` | | yes |
| `input_forcing/adjusted/` | `eccov4r4_{dlw,dsw,pres,rain,spfh2m,tmp2m_degC,ustr,vstr,wspeed}_1992` (5.6 GB) | | yes |
| `atm_flux_forcing_experiments/atm_flux_forcing/` | `{oceFWflx,oceQsw,oceSflux,oceSPflx,oceTAUX,oceTAUY,sIceLoadPatmPload_nopabar,TFLUX}_6hourlyavg_1992` (4.9 GB) | yes | |
| `atm_flux_forcing_experiments/xx/` | `weights_ones.data`, `xx_{empmr,pload,qnet,qsw,saltflux,spflx,tauu,tauv}.0000000129.{data,meta}` (4.6 GB) | yes | |

In total 71 files (10.4 GB) for `ff` and 58 files (6.4 GB) for `full`. To avoid the large downloads on a second
machine, copy these directories from a machine that has them (`rsync -a`, or NASA's `shiftc`) into `$MITJAX_DATA`,
keeping the directory names.

## 5. Run directories

A run directory holds the V4r4 namelists and links to the input files. The JAX model reads its parameters from the
namelists, exactly as the Fortran does. `--no-binary` makes a run directory without a Fortran executable:

```bash
python reference/make_rundir.py ff   serial13 jax_ff_1992   --nsteps 8760 --no-binary
python reference/make_rundir.py full serial13 jax_full_1992 --nsteps 8760 --no-binary
```

This creates `$MITJAX_REFERENCE_RUNS/jax_ff_1992` and `jax_full_1992` (10-30 s each; the script hashes the inputs
into `INPUTS.sha256`). It copies the namelists of the tree from `ECCO-v4-Configurations`, adds `data.exch2` for the
13 tiles of 90 × 90 points the JAX model uses (layout `serial13`), links every input file the namelists name, and
refuses, creating nothing, if one is missing (it prints which). `OVERRIDES.txt` lists the changes to the V4r4
namelists: the cost-function packages are switched off (`useECCO=F`, `useProfiles=F`, `useCAL=T`; this does not
change the forward run), and `nTimeSteps`/`monitorFreq` are set (the JAX model ignores both: the number of steps is a
`run_jax.py` option). An existing directory is never overwritten; choose a new name.

The validated runs used run directories that differ from these only in `nTimeSteps`, `dumpFreq` and the Fortran
executable; two-step runs from both are bitwise identical (checked for both trees).

## 6. A one-day test

Before a year, run one day (24 steps; a few minutes on a GPU including compilation):

```bash
R=$MITJAX_REFERENCE_RUNS
python scripts/run_jax.py $MITJAX_RUNS_JAX/test_full_1day --rundir $R/jax_full_1992 --tree full \
    --init-oracle pickup --nsteps 24 --cg2d-unroll 5 --monitor-every 24
```

The line for iteration 24 in the output (and in `run.log`) should read

```
full:  it     24 1992-01-02 12:00 step ... s  theta mean 3.5853140960 max 32.3954  eta [-5.953, 1.559]  |u|max 1.041
ff:    it     24 1992-01-02 12:00 step ... s  theta mean 3.5853148988 max 32.3862  eta [-5.955, 1.497]  |u|max 1.013
```

(`ff`: `--rundir $R/jax_ff_1992 --tree ff`). The digits shown agree between GPUs and with the Fortran; in the full
model the last digit of the mean temperature may differ after one day (sea-ice round-off).

## 7. The one-year runs

The validated runs used these options:

```bash
R=$MITJAX_REFERENCE_RUNS; O=$MITJAX_RUNS_JAX
python scripts/run_jax.py $O/ff_1992 --rundir $R/jax_ff_1992 --tree ff --init-oracle pickup --nsteps 8760 \
    --cg2d-unroll 5 --monitor-every 24 --frame-every 6 --snapshot-every 240 --checkpoint-every 720
python scripts/run_jax.py $O/full_1992 --rundir $R/jax_full_1992 --tree full --init-oracle pickup --nsteps 8760 \
    --cg2d-unroll 5 --monitor-every 24 --frame-every 6 --snapshot-every 720 --checkpoint-every 720
```

Before the run set `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` (share of GPU memory JAX may take, as in the
validated runs).

| option | meaning |
|---|---|
| `OUTDIR` (first argument) | new directory for the output; an existing one is an error |
| `--rundir`, `--tree` | run directory; `--tree` declares the configuration and is checked against the namelists |
| `--init-oracle pickup` | initial state from the run directory's pickup files, with the V4r4 control adjustments, as the Fortran initialises |
| `--nsteps 8760` | number of one-hour steps |
| `--cg2d-unroll 5` | faster global sums in the pressure solver (same summation order, bitwise identical results) |
| `--monitor-every 24` | global statistics (`%MON`, as in MITgcm's STDOUT) once per model day |
| `--frame-every 6` | movie frame every 6 h: SST and SSH (full model: also sea-ice concentration and thickness) |
| `--snapshot-every N` | theta, salinity and SSH, float32, every N steps |
| `--checkpoint-every 720` | full model state every 30 days (restart points) |

Other options: `--means-every N` (time means), `--budgets` (volume, heat and salt budgets per step), `--host-monitor`
(statistics computed on the host, slower); `python scripts/run_jax.py --help`.

The output directory gets `run.log` (one line per model day), `monitor.txt` (`%MON` blocks in the Fortran format;
full model also the sea-ice and forcing statistics), `frames/frame_NNNNN.npz` (1461 files), `snap_<iter>.npz`,
`state_<iter>.npz` (1.9 GB each) and `state_final.npz` (the state at iteration 8761, 1992-12-31 13:00).

**Batch job.** `scripts/runs/one_year.sbatch` runs exactly these commands on one GPU (Slurm; the PBS scripts for NASA
NAS in `scripts/cabeus/` call it):

```bash
sbatch -A ACCOUNT -p GPU_PARTITION --output=LOGFILE scripts/runs/one_year.sbatch full $O/full_1992 $R/jax_full_1992
sbatch -A ACCOUNT -p GPU_PARTITION --output=LOGFILE scripts/runs/one_year.sbatch ff   $O/ff_1992   $R/jax_ff_1992
```

Its `#SBATCH` header is set for DKRZ Levante (log directory, 2 h, 64 GB of host memory); options on the command line
override it. It takes the interpreter from `MITJAX_PYTHON`.

**Runs in parts** (queues with short time limits): `--restart` continues from a `state_<iter>.npz` or
`state_final.npz`, and `--frame-offset` continues the frame numbering (the first part writes 1 + NSTEPS/6 frames):

```bash
J=$(sbatch --parsable ... scripts/runs/one_year.sbatch ff $O/ff_1992_part1 $R/jax_ff_1992 4380)
sbatch --dependency=afterok:$J ... scripts/runs/one_year.sbatch ff $O/ff_1992_part2 $R/jax_ff_1992 4380 \
    $O/ff_1992_part1/state_final.npz 731
```

The validated flux-forced year was made in two such parts of 4380 steps (30-minute queue). A restart reproduces the
continuous run: on A100-80 GPUs, 24 + 24 steps with a restart and 48 continuous steps of the full model give bitwise
identical states, `%MON` blocks and frames.

Expected wall times on one GPU, including start-up (15-60 s) and compilation (1-2 min): flux-forced ocean about
0.17 s per step on an A100 (the validated year: two parts of 16 min each on an A100-40) and 0.10 s on a GH200; full
model 0.29 s on a GH200 (the validated year: 48 min) and 0.43 s on an A100-80 (68 min without output). Several GPUs
are not faster at this resolution ([VALIDATION.md](VALIDATION.md), speed).

## 8. Checking the result

**The run finished:** the last line of `run.log` is `done: 8760 steps, median step ... s, ...`. A run that produces
NaNs stops with `NaN: stopping`.

**Global statistics at the end of the year** (iteration 8760, 1992-12-31 12:00; `%MON` blocks of `monitor.txt`):

| statistic | ff: JAX (1 A100) | ff: Fortran 13 ranks | ff: Fortran 96 ranks |
|---|---|---|---|
| `dynstat_theta_mean` | 3.5809912628248 | 3.5809912628248 | 3.5809912628248 |
| `dynstat_theta_sd` | 4.4212721098144 | 4.4212721098179 | 4.4212721098162 |
| `dynstat_salt_mean` | 34.727009803595 | 34.727009803595 | 34.727009803595 |
| `dynstat_eta_sd` | 0.68806011488177 | 0.68806011485040 | 0.68806011487148 |
| `dynstat_uvel_max` | 1.0486632340785 | 1.0486632340796 | 1.0486632340862 |

| statistic | full: JAX (1 GH200) | full: Fortran 13 ranks | full: Fortran 96 ranks | ECCO production run |
|---|---|---|---|---|
| `dynstat_theta_mean` | 3.5809890820852 | 3.5809890820486 | 3.5809891829249 | 3.5809891832136 |
| `dynstat_salt_mean` | 34.727009566866 | 34.727009566910 | 34.727009669893 | 34.727009669821 |
| `dynstat_eta_sd` | 0.68789222761769 | 0.68789223553465 | 0.68794127853571 | 0.68794126621121 |
| `seaice_area_mean` | 0.053011274580456 | 0.053011277375446 | 0.053003630667563 | 0.053003621497052 |
| `seaice_heff_mean` | 0.061076088860225 | 0.061076092885224 | 0.061088805432748 | 0.061088799709048 |

A run on another GPU will not reproduce these values to the last digit. For the flux-forced ocean it should agree
with the three columns to about 10 significant digits. The full model is sensitive to round-off through the sea ice:
runs on the 13 tiles of the JAX model (JAX and Fortran 13 ranks) agree with each other to 7-11 significant digits,
and differ from runs on the 96 tiles of the production layout (Fortran 96 ranks and ECCO's own run) from the 4th to
the 8th digit. A JAX run should be close to the "13 ranks" column; differences of the size of the 13-rank/96-rank
difference are the tolerance ([VALIDATION.md](VALIDATION.md), acceptance criteria).

**Full model against ECCO's own production run.** `$MITJAX_DATA/doc/STDOUT.0000` (from `ancillary_small`) is the
standard output of the V4r4 production run (96 ranks), with `%MON` blocks every 1752 steps. Compare:

```bash
python tools/compare_monitor.py $O/full_1992/monitor.txt $MITJAX_DATA/doc/STDOUT.0000 --every 1752 \
    --groups dynstat,seaice --per-stat
```

It compares iterations 1752, 3504, 5256, 7008 and 8760. The validated JAX year gives (excerpt; the Fortran 13-rank
year gives the same numbers to the digits shown, so these are the 13-tile/96-tile spread, not errors):

```
compared 5 iterations (1752..8760); overall worst 2.03e-02 (dynstat_wvel_mean at it 1752)
statistic                           n   max |diff|    max rel
dynstat_eta_sd                      5    5.438e-05   7.47e-05
dynstat_wvel_mean                   5    2.032e-11   2.03e-02
dynstat_theta_mean                  5    2.289e-07   6.40e-08
dynstat_salt_mean                   5    1.030e-07   2.96e-09
seaice_area_mean                    5    1.423e-05   2.30e-04
seaice_heff_mean                    5    1.271e-05   2.08e-04
```

`dynstat_wvel_mean` is a near-zero quantity (judged relative to 1e-9); the `del2` statistics are missing because
`run_jax.py` computes them only with `--host-monitor`. A run whose differences are far above these (for example 1e-6
relative in the mean temperature) has a problem.

**Against a Fortran run** (section 11): `python tools/compare_monitor.py JAX/monitor.txt FORTRAN/STDOUT.0000
--every 24 [--groups dynstat,seaice,exf] [--per-stat]` prints the worst relative difference per compared day.

## 9. Plots and movies

`scripts/runs/plot_one_year.sh` makes the standard figures of a run: the grid files (below), daily global statistics
(`monitor_ts.png`), an SST movie (Robinson projection) and, for the full model, Arctic and Antarctic sea-ice
concentration movies (`.mp4` and `.gif`). Run it in a CPU job (about 2 min with 32 cores):

```bash
export MITJAX_PYTHON=$ENV/bin/python MITJAX_NEREUS_PYTHON=$NENV/bin/python JOBS=16
bash scripts/runs/plot_one_year.sh full $R/jax_full_1992 $MITJAX_WORK/figs/full_1992 $O/full_1992
bash scripts/runs/plot_one_year.sh ff $R/jax_ff_1992 $MITJAX_WORK/figs/ff_1992 $O/ff_1992_part1 $O/ff_1992_part2
```

The tools can also be used one by one (nereus environment unless noted):

- `tools/write_grid_mds.py RUNDIR OUTDIR` (model environment): writes the grid files `XC`, `YC`, `RAC`, `Depth`,
  `hFacC`, `RC`, `DRF` in MITgcm's format, identical to the Fortran's, for the plotting tools.
- `tools/animate_globe.py FRAMES_DIR GRID_DIR OUT --projection robinson|arctic|antarctic|orthographic --var sst|eta|area|heff`:
  movies from the frames (see `--help` for colour ranges, stride, frame rate).
- `tools/plot_monitor_ts.py OUT.png --ref "LABEL=monitor.txt" [--run "LABEL=STDOUT.0000" ...] --every 24`: time series
  of `%MON` statistics of several runs and their differences from the first.
- `tools/plot_diff_maps.py ff_year|full_month|full_year OUT.png --jax STATE.npz --mesh GRID_DIR`: maps of the
  differences between the JAX state and the Fortran reference runs (needs the Fortran runs of section 11 under their
  names in `$MITJAX_REFERENCE_RUNS`).

## 10. Tests

`JAX_PLATFORMS=cpu python -m pytest -m smoke -q` runs anywhere in seconds. The other tiers (`tier1`, `tier1x`,
`nightly`, `tier2`; [mitgcm_jax/tests/manifest.py](../mitgcm_jax/tests/manifest.py)) compare against the Fortran
oracle: reference runs with per-substep dumps, registered by name in `reference/runs.json` and looked up in
`$MITJAX_REFERENCE_RUNS`, and the input data in `$MITJAX_DATA`. They fail (they do not skip) when these are missing.
The oracle runs are not distributed; they are made with the instrumented Fortran build (`JAXDUMP=1`, section 11,
[REFERENCE_RUNS.md](REFERENCE_RUNS.md)). The job scripts run each tier on one CPU node and decide pass or fail from
the JUnit report: `scripts/run_tier1.sbatch` (< 10 min), `scripts/run_tier1x.sbatch`, `scripts/run_nightly.sbatch`.

## 11. Optional: Fortran reference runs

The Fortran model is needed only for comparisons and for the test oracle. `reference/build.sh TREE LAYOUT` builds
MITgcm checkpoint66g with the V4r4 code of the tree (`ff` or `full`) and a tile layout: `mpi96` (production, 96 ranks
of 30 × 30), `serial13` (one process, the 13 tiles of the JAX model) or `mpi13` (13 ranks of one tile: the same
tiles and global sums as `serial13`, bitwise identical to it, several times faster). It needs gfortran (>= 10), MPI and
netCDF-Fortran; the `module load` lines and `reference/optfile_levante_gfortran` are those of DKRZ Levante and have
to be adapted elsewhere. The executable is frozen as `$MITJAX_REFERENCE/bin/mitgcmuv_<tree>_<layout>_<hash>`.

```bash
sbatch reference/jobs/build.sbatch full mpi13          # or: bash reference/build.sh full mpi13
python reference/make_rundir.py full mpi13 ref_full_mpi13_1year --nsteps 8760 --monitor 86400 \
    --set data:parm03:dumpFreq=2592000.0
sbatch -p PARTITION --ntasks=13 --mem=32G --time=03:00:00 reference/jobs/run.sbatch $R/ref_full_mpi13_1year
```

`reference/jobs/run.sbatch` runs `mpi96` with `srun -n 96`, `mpi13` with `srun -n 13` and `serial13` as one process,
and checks for "Execution ended Normally". Measured on Levante for one year: flux-forced 24 min on 96 ranks and 8.1 h
serial; full model 29 min on 96 ranks and 87 min on 13 ranks. [REFERENCE_RUNS.md](REFERENCE_RUNS.md) has the details.

## DKRZ Levante

The defaults of every path and interpreter are Levante's, so nothing needs to be set there:

- work root `/work/ab0995/a270088/MIT`; model environment `/work/ab0995/a270088/mambaforge/envs/mitgcm-jax`; nereus
  environment `/work/ab0995/a270088/mambaforge/envs/nereus`; GH200 (aarch64) environment
  `$MITJAX_WORK/envs/mitgcm-jax-arm` (`scripts/dolpung/env_dolpung.sh`).
- The data of section 4 and the run directories of the validated runs (`ref_ff_serial13_1day`,
  `ref_full_serial13_1month`) exist; outputs of the validated runs: `runs_jax/ff_prod_1992_gpu_v2`,
  `runs_jax/full_1992_gh200`.
- Downloads and unpacking as jobs on the `shared` partition (internet access):
  `sbatch scripts/fetch_job.sbatch GROUP`, `sbatch [--dependency=afterok:JOB] scripts/extract_job.sbatch ARCHIVE --years 1992`.
- One year, A100-80: `sbatch -A ab0995_gpu -p gpu --constraint=a100_80 scripts/runs/one_year.sbatch TREE OUTDIR RUNDIR`.
  GH200 (`dolpung` partition, aarch64 environment):
  `MITJAX_ENV=dolpung sbatch -A mh1571 -p dolpung scripts/runs/one_year.sbatch TREE OUTDIR RUNDIR`.
- Plots: `sbatch -p shared -A ab0995 -c 32 --mem=64G --time=00:30:00 --wrap "bash scripts/runs/plot_one_year.sh ..."`.
- Tests: smoke on a login node with `JAX_PLATFORMS=cpu taskset -c 0-7 python -m pytest -m smoke -q`; tier 1 with
  `sbatch scripts/run_tier1.sbatch` (verdict in `$MITJAX_RUNS/tier1/<job id>/verdict.txt`).

## NASA NAS Cabeus

Cabeus has two GPU node types; both run the model. **These instructions and the scripts in `scripts/cabeus/` have not
been run on Cabeus (no access); they repeat the commands tested on Levante with PBS syntax.** Check module names,
queue names and limits in the NAS knowledge base:
[Cabeus](https://www.nas.nasa.gov/hecc/resources/cabeus.html),
[preparing to run on the Grace Hopper nodes](https://www.nas.nasa.gov/hecc/support/kb/preparing-to-run-on-cabeus-grace-hopper-nodes_702.html),
[preparing to run on the A100 nodes](https://www.nas.nasa.gov/hecc/support/kb/preparing-to-run-on-cabeus-a100-gpu-nodes_686.html),
[requesting GPU resources](https://www.nas.nasa.gov/hecc/support/kb/requesting-cabeus-gpu-resources_646.html).

| | GH200 nodes | A100 nodes |
|---|---|---|
| node | 72-core Grace (aarch64) + one H100 GPU (96 GB), 480 GB; whole node per job | AMD EPYC 7763 (x86_64, 64 cores) + four A100-80, 512 GB; four vnodes of 1 GPU and 16 cores, shareable |
| front ends (build, install) | `cghfe01`, `cghfe02` (aarch64) | `cfe01`, `cfe02` (x86_64) |
| PBS server | `pbs05a` | `pbspl4` |
| one GPU | `-l select=1:ncpus=72:ngpus=1:model=gh200` | `-l select=1:ncpus=16:ngpus=1:mem=60GB:model=mil_a100` and `-l place=scatter:shared` |
| queues | `gpu_devel@pbs05a` (2 h), `gpu_normal@pbs05a` (24 h), `gpu_long@pbs05a` (120 h), `gpu_wide@pbs05a` (12 h) | `gpu_devel@pbspl4` (2 h), `gpu_normal@pbspl4` (24 h) |
| expected speed | full 0.29 s/step (year ~50 min), ff ~0.10 s/step | full ~0.43 s/step (year ~75 min), ff ~0.17 s/step |

Use one GPU. The model's multi-device mode runs in one process, so on Cabeus it could only use the four GPUs of one
A100 node, and at LLC90 four GPUs are slower than one (0.47 vs 0.43 s/step for the full model on A100-80). A year
fits into `gpu_normal` (24 h) on either node type, so no restart chain is needed.

**File systems.** `$HOME` is small: keep the repository there if you like, but put environments, data, run
directories and output on `/nobackup`:

```bash
export MITJAX_WORK=/nobackup/$USER/mitjax        # scripts/cabeus/env_cabeus.sh uses this default
```

**Install** (once per node type, on its front end; the GH200 and A100 nodes need separate environments because the
architectures differ). The default locations are those of `scripts/cabeus/env_cabeus.sh`:

```bash
cd mit_jax                                           # the clone of section 1 (with ECCO-v4-Configurations)
# GH200, on cghfe01:
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/gh2
conda create -y -p /nobackup/$USER/envs/mitgcm-jax-gh200 python=3.12.13 pip
/nobackup/$USER/envs/mitgcm-jax-gh200/bin/pip install --no-cache-dir -e ".[cuda,dev]" -c constraints.txt
# A100, on cfe01: any conda (for example a Miniforge installed under /nobackup/$USER), then
conda create -y -p /nobackup/$USER/envs/mitgcm-jax-a100 python=3.12.13 pip
/nobackup/$USER/envs/mitgcm-jax-a100/bin/pip install --no-cache-dir -e ".[cuda,dev]" -c constraints.txt
```

The JAX CUDA wheels bring their own CUDA 12 libraries (no `nvhpc` or CUDA module is needed at run time); only the
node's NVIDIA driver matters. If the front ends cannot reach PyPI or conda-forge directly, see the NAS knowledge base
for the proxy settings, or build the environment offline from downloaded wheels as in
`scripts/dolpung/fetch_wheels.sbatch` and `scripts/dolpung/make_env.sbatch`. For plots, create the nereus environment
of section 2 on an x86_64 front end at `/nobackup/$USER/envs/nereus`.

**Check** (from the repository root; output in `mitjax_check.o<job id>`):

```bash
qsub scripts/cabeus/check_gh200.pbs          # or check_a100.pbs: GPU listed, "CHECK OK", 17 smoke tests passed
```

**Data.** `scripts/fetch_eccov4r4.py` needs outbound HTTPS to `cmr.earthdata.nasa.gov`, `urs.earthdata.nasa.gov`
and the PO.DAAC download servers, and an Earthdata login in `~/.netrc` (section 4). Whether and how the NAS front ends
reach external servers (proxies) is described in the NAS knowledge base. Downloading 284 GiB takes several hours; run
it where long-running transfers are allowed. Alternatively, copy the unpacked input directories of section 4 (16 GB for
both runs) from another machine with `shiftc` or `rsync` into `$MITJAX_DATA`, keeping the directory names. Then make
the run directories as in section 5 (on a front end; about 30 s each).

**One year:**

```bash
export MITJAX_WORK=/nobackup/$USER/mitjax
R=$MITJAX_WORK/reference/runs; O=$MITJAX_WORK/runs_jax
qsub -v TREE=full,OUT=$O/full_1992,RUNDIR=$R/jax_full_1992 scripts/cabeus/one_year_gh200.pbs
qsub -v TREE=ff,OUT=$O/ff_1992,RUNDIR=$R/jax_ff_1992 scripts/cabeus/one_year_a100.pbs
```

The PBS scripts source `scripts/cabeus/env_cabeus.sh` (work root and interpreters; edit it or pass the variables with
`qsub -v`) and then run `scripts/runs/one_year.sbatch` with bash, so the run is the same as on Levante. The output
directory must not exist. To split a run (for example to use `gpu_devel`), pass `NSTEPS`, and for the second part
`RESTART` and `FOFF`, and chain the jobs:

```bash
J=$(qsub -q gpu_devel@pbs05a -l walltime=02:00:00 -v TREE=ff,OUT=$O/ff_part1,RUNDIR=$R/jax_ff_1992,NSTEPS=4380 \
      scripts/cabeus/one_year_gh200.pbs)
qsub -W depend=afterok:$J -v TREE=ff,OUT=$O/ff_part2,RUNDIR=$R/jax_ff_1992,NSTEPS=4380,RESTART=$O/ff_part1/state_final.npz,FOFF=731 \
    scripts/cabeus/one_year_gh200.pbs
```

**Checks and plots:** section 8 on a front end (`compare_monitor.py` reads text files only), and
`qsub -v TREE=full,RUNDIR=$R/jax_full_1992,FIG=$MITJAX_WORK/figs/full_1992,RUNOUT=$O/full_1992 scripts/cabeus/plots.pbs`
for the figures and movies (section 9; for a run in two parts add `RUNOUT2`).
