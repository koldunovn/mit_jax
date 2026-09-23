#!/usr/bin/env python3
"""Create a run directory for one ECCO v4r4 tree: Fortran reference runs, or JAX-only runs with --no-binary.

    make_rundir.py TREE LAYOUT NAME --nsteps N [--monitor SECONDS] [--binary PATH | --no-binary]
                   [--set FILE:GROUP:KEY=VALUE ...]
                   [--pickup-from RUNDIR --niter0 N]    matched restart: start from that run's pickup*.<N> files
                   [--cost-pkgs production]             keep the production useECCO/useProfiles (no data.pkg override)

TREE = full | ff, LAYOUT = mpi96 | serial13 | mpi13. The directory $MITJAX_REFERENCE_RUNS/<NAME> (mitgcm_jax/paths.py)
must not exist. It gets: the tree's production namelists (from the ECCO-v4-Configurations clone in the repository
root), the overrides below (every one printed and written to OVERRIDES.txt), `data.exch2` for 13x90x90 when
LAYOUT=serial13 or mpi13 (one file for both: same tiles, same tile numbering), symlinks to every input file
`scripts/audit_run_inputs.py` derives from $MITJAX_DATA (the script refuses to create a run with a missing required
input), diagnostics sub-directories, and the executable `mitgcmuv` (newest frozen binary for TREE/LAYOUT in
$MITJAX_REFERENCE/bin unless --binary).

--no-binary: no executable (a run directory for the JAX model only, scripts/run_jax.py --rundir; needs no Fortran
build). The JAX model runs 13 tiles of 90x90, so its run directories use LAYOUT serial13 (or mpi13, same data.exch2).

Standing overrides for reference runs (deviations from the production namelists, each documented):
  data.pkg  useECCO=F, useProfiles=F, useCAL=T   cost-function packages off (--cost-pkgs off, the default). useCAL
                                                   is implied by useECCO/useProfiles (packages_boot.F:206-207) and
                                                   must then be set.
            --cost-pkgs production: no data.pkg override (the tree's production useECCO/useProfiles; useCAL implied);
            the pkg/ecco and pkg/profiles inputs (data_constraints archive) are linked and a missing one that the
            model reads unconditionally refuses the run; missing inquire-only files (a gencost term switched off)
            are listed and recorded in OVERRIDES.txt.
  data      nTimeSteps=N, monitorFreq=S          run length and monitor cadence (%MON every S seconds).
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from mitgcm_jax.paths import DATA, REFERENCE, REFERENCE_RUNS  # noqa: E402  ($MITJAX_*, mitgcm_jax/paths.py)
from mitgcm_jax.io.namelist import read_namelist  # noqa: E402
import audit_run_inputs as air  # noqa: E402

V4 = REPO / "ECCO-v4-Configurations" / "ECCOv4 Release 4"
BIN = REFERENCE / "bin"               # frozen Fortran executables (reference/build.sh)
SEARCH = [DATA / "input_init", DATA / "native_grid_files"]  # + unpacked forcing dirs, found below


def set_value(text, group, key, value):
    """Set `key = value,` inside namelist group `group` (case-insensitive). Replaces a one-line assignment or inserts
    one before the group terminator. Refuses multi-line assignments rather than guessing."""
    gm = re.search(rf"^\s*&{group}\b.*?(?=^\s*(?:/|&|&end)\s*$)", text, re.S | re.M | re.I)
    if not gm:
        raise SystemExit(f"group &{group} not found")
    block = gm.group(0)
    pat = re.compile(rf"^[ \t]*{re.escape(key)}[ \t]*=.*$", re.M | re.I)
    hits = pat.findall(block)
    new_line = f" {key}={value},"
    if len(hits) > 1:
        raise SystemExit(f"{group}:{key} assigned {len(hits)} times")
    if hits:
        after = block[block.index(hits[0]) + len(hits[0]):]
        nxt = after.lstrip("\n").split("\n", 1)[0]
        if nxt.strip() and not re.match(r"\s*(#|\w+\s*(\([^)]*\))?\s*=|/|&)", nxt):
            raise SystemExit(f"{group}:{key} continues on the next line; edit by hand")
        block2 = pat.sub(new_line, block, count=1)
    else:
        block2 = block.rstrip("\n") + "\n" + new_line + "\n"
    return text[:gm.start()] + block2 + text[gm.end():]


def newest_binary(tree, layout, variant=""):
    pat = re.compile(rf"mitgcmuv_{tree}_{layout}{variant}_[0-9a-f]{{12}}")
    bins = sorted((p for p in (BIN.iterdir() if BIN.is_dir() else ()) if pat.fullmatch(p.name)),
                  key=lambda p: p.stat().st_mtime)
    if not bins:
        raise SystemExit(f"no binary for {tree}/{layout} in {BIN}; build with reference/jobs/build.sbatch "
                         f"(or --no-binary for a JAX-only run directory)")
    return bins[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", choices=("full", "ff"))
    ap.add_argument("layout", choices=("mpi96", "serial13", "mpi13"))
    ap.add_argument("name")
    ap.add_argument("--nsteps", type=int, required=True)
    ap.add_argument("--monitor", type=float, default=3600.0)
    ap.add_argument("--binary", help="executable to link as mitgcmuv (default: newest frozen binary)")
    ap.add_argument("--no-binary", action="store_true",
                    help="no executable: a run directory for the JAX model only (no Fortran build needed)")
    ap.add_argument("--variant", default="", help="binary variant suffix, e.g. _jaxdump or _gcov")
    ap.add_argument("--set", nargs="*", default=[], help="extra FILE:GROUP:KEY=VALUE overrides")
    ap.add_argument("--pickup-from", help="run directory whose pickup*.<niter0> files start this run")
    ap.add_argument("--niter0", type=int)
    ap.add_argument("--cost-pkgs", choices=("off", "production"), default="off",
                    help="off: standing override useECCO=F, useProfiles=F, useCAL=T; production: keep data.pkg")
    a = ap.parse_args(argv)

    run = REFERENCE_RUNS / a.name
    if run.exists():
        raise SystemExit(f"{run} exists; choose a new name (nothing is overwritten)")
    if a.binary and a.no_binary:
        raise SystemExit("--binary and --no-binary exclude each other")
    src = V4 / ("namelist" if a.tree == "full" else "flux-forced/namelist")
    if not src.is_dir():
        raise SystemExit(f"{src} not found: clone https://github.com/ECCO-GROUP/ECCO-v4-Configurations into {REPO} "
                         f"(docs/RUN_ONE_YEAR.md)")
    overrides = [] if a.cost_pkgs == "production" else [
        ("data.pkg", "packages", "useECCO", ".FALSE."), ("data.pkg", "packages", "useProfiles", ".FALSE."),
        ("data.pkg", "packages", "useCAL", ".TRUE.")]
    overrides += [("data", "parm03", "nTimeSteps", str(a.nsteps)), ("data", "parm03", "monitorFreq", f"{a.monitor:.1f}")]
    if (a.pickup_from is None) != (a.niter0 is None):
        raise SystemExit("--pickup-from and --niter0 go together")
    if a.pickup_from:
        overrides.append(("data", "parm03", "nIter0", str(a.niter0)))
    for s in a.set:
        f, g, kv = s.split(":", 2)
        k, v = kv.split("=", 1)
        overrides.append((f, g, k, v))

    files = {p.name: p.read_text() for p in src.iterdir() if p.is_file() and (p.name.startswith("data") or p.name == "eedata")}
    if a.layout in ("serial13", "mpi13"):
        files["data.exch2"] = (REPO / "reference" / "data.exch2_13x90x90").read_text()
    for f, g, k, v in overrides:
        files[f] = set_value(files[f], g, k, v)

    # resolve every input first, in a throw-away copy of the namelists: a refused run leaves nothing behind
    search = SEARCH + sorted(p for p in DATA.glob("**/") if p.is_dir() and "logs" not in p.parts)
    if a.pickup_from:  # the source run's pickups take precedence over input_init's pickup.0000000001
        search = [Path(a.pickup_from)] + search
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in files.items():
            (Path(tmp) / name).write_text(text)
        for f, g, k, v in overrides:  # every override reads back as intended
            got = read_namelist(Path(tmp) / f)[g.lower()][k.lower()]
            print(f"override {f}:{g}:{k} = {v}  (reads back {got})")
        items, unresolved = air.needs(tmp)
        shipped = {p.name for p in Path(tmp).iterdir()}  # namelist-directory files (e.g. data.err) are copied
    links, missing, switched_off = {}, [], []
    for n in items:
        if n.klass == "cost" and a.cost_pkgs != "production":
            continue
        if n.name in shipped:
            continue
        paths, absent = air.resolve(n, search)
        for p in paths:
            links[p.name] = p
        if absent and n.required:
            missing += absent
        elif absent and n.klass == "cost":
            switched_off += [f"{x}  <- {n.source}  [{n.why}]" for x in absent]
    if unresolved:
        print("UNRESOLVED:", *unresolved, sep="\n  ")
    if switched_off:
        print(f"ABSENT optional cost inputs (inquired; a missing one switches its cost term off): {len(switched_off)}",
              *switched_off, sep="\n  ")
    if missing:
        print(f"REFUSED: {len(missing)} required inputs missing, e.g. {missing[:6]}")
        return 1
    binary = None if a.no_binary else (Path(a.binary) if a.binary else newest_binary(a.tree, a.layout, a.variant))

    run.mkdir(parents=True)
    for name, text in files.items():
        (run / name).write_text(text)
    with open(run / "OVERRIDES.txt", "w") as o:
        o.write(f"tree {a.tree} layout {a.layout} namelists from {src}\n")
        for f, g, k, v in overrides:
            o.write(f"{f}:{g}:{k}={v}\n")
        if a.cost_pkgs == "production":
            o.write("cost packages as in the production data.pkg (useCAL implied by packages_boot.F:206-207)\n")
            for x in switched_off:
                o.write(f"absent optional cost input: {x}\n")
        if a.layout in ("serial13", "mpi13"):
            o.write("data.exch2 <- reference/data.exch2_13x90x90 (no blankList)\n")
        if binary is None:
            o.write("no executable (--no-binary): run directory for the JAX model only\n")
    for name, p in links.items():
        (run / name).symlink_to(p)
    # diagnostics output sub-directories (ECCO's misc/tools/mkdir_subdir_diags.py does the same)
    dg = read_namelist(run / "data.diagnostics").get("diagnostics_list", {}) if (run / "data.diagnostics").exists() else {}
    for k, v in dg.items():
        if k.startswith("filename(") and v and "/" in v[0]:
            (run / v[0]).parent.mkdir(parents=True, exist_ok=True)
    if binary is not None:
        (run / "mitgcmuv").symlink_to(binary)
    air.audit(run, sha_out=run / "INPUTS.sha256")
    print(f"RUNDIR {run}  binary {binary.name if binary else 'none (--no-binary)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
