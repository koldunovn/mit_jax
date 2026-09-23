#!/usr/bin/env python3
"""Create a Fortran reference run directory for one ECCO v4r4 tree (plan Task 4).

    make_rundir.py TREE LAYOUT NAME --nsteps N [--monitor SECONDS] [--binary PATH] [--set FILE:GROUP:KEY=VALUE ...]

TREE = full | ff, LAYOUT = mpi96 | serial13. The directory /work/.../MIT/reference/runs/<NAME> must not exist.
It gets: the tree's production namelists, the overrides below (every one printed and written to OVERRIDES.txt),
`data.exch2` for 13x90x90 when LAYOUT=serial13, symlinks to every input file `scripts/audit_run_inputs.py`
derives (the script refuses to create a run with a missing required input), diagnostics sub-directories, and the
executable (newest frozen binary for TREE/LAYOUT unless --binary).

Standing overrides for reference runs (deviations from the production namelists, each documented):
  data.pkg  useECCO=F, useProfiles=F, useCAL=T   cost-function packages; their observation inputs (data_constraints)
                                                   are not staged. They do not feed back on the model state (to be
                                                   confirmed by a same-binary run with them on). useCAL was implied
                                                   by useECCO/useProfiles (packages_boot.F) and must be set.
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
from mitgcm_jax.io.namelist import read_namelist  # noqa: E402
import audit_run_inputs as air  # noqa: E402

V4 = REPO / "ECCO-v4-Configurations" / "ECCOv4 Release 4"
WORK = Path("/work/ab0995/a270088/MIT")
DATA = WORK / "data" / "eccov4r4"
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


def newest_binary(tree, layout):
    bins = sorted((WORK / "reference" / "bin").glob(f"mitgcmuv_{tree}_{layout}_*"), key=lambda p: p.stat().st_mtime)
    bins = [b for b in bins if not b.name.endswith(".txt")]
    if not bins:
        raise SystemExit(f"no binary for {tree}/{layout}; build with reference/jobs/build.sbatch")
    return bins[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", choices=("full", "ff"))
    ap.add_argument("layout", choices=("mpi96", "serial13"))
    ap.add_argument("name")
    ap.add_argument("--nsteps", type=int, required=True)
    ap.add_argument("--monitor", type=float, default=3600.0)
    ap.add_argument("--binary")
    ap.add_argument("--set", nargs="*", default=[], help="extra FILE:GROUP:KEY=VALUE overrides")
    a = ap.parse_args(argv)

    run = WORK / "reference" / "runs" / a.name
    if run.exists():
        raise SystemExit(f"{run} exists; choose a new name (nothing is overwritten)")
    src = V4 / ("namelist" if a.tree == "full" else "flux-forced/namelist")
    overrides = [("data.pkg", "packages", "useECCO", ".FALSE."), ("data.pkg", "packages", "useProfiles", ".FALSE."),
                 ("data.pkg", "packages", "useCAL", ".TRUE."), ("data", "parm03", "nTimeSteps", str(a.nsteps)),
                 ("data", "parm03", "monitorFreq", f"{a.monitor:.1f}")]
    for s in a.set:
        f, g, kv = s.split(":", 2)
        k, v = kv.split("=", 1)
        overrides.append((f, g, k, v))

    files = {p.name: p.read_text() for p in src.iterdir() if p.is_file() and (p.name.startswith("data") or p.name == "eedata")}
    if a.layout == "serial13":
        files["data.exch2"] = (REPO / "reference" / "data.exch2_13x90x90").read_text()
    for f, g, k, v in overrides:
        files[f] = set_value(files[f], g, k, v)

    # resolve every input first, in a throw-away copy of the namelists: a refused run leaves nothing behind
    search = SEARCH + sorted(p for p in DATA.glob("**/") if p.is_dir() and "logs" not in p.parts)
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in files.items():
            (Path(tmp) / name).write_text(text)
        for f, g, k, v in overrides:  # every override reads back as intended
            got = read_namelist(Path(tmp) / f)[g.lower()][k.lower()]
            print(f"override {f}:{g}:{k} = {v}  (reads back {got})")
        items, unresolved = air.needs(tmp)
    links, missing = {}, []
    for n in items:
        p = air.locate(n.name, n.kind, search)
        if p is None:
            if n.required and n.klass == "input":
                missing.append(n.name)
        else:
            links[p.name] = p
    if unresolved:
        print("UNRESOLVED:", *unresolved, sep="\n  ")
    if missing:
        print(f"REFUSED: {len(missing)} required inputs missing, e.g. {missing[:6]}")
        return 1

    run.mkdir(parents=True)
    for name, text in files.items():
        (run / name).write_text(text)
    with open(run / "OVERRIDES.txt", "w") as o:
        o.write(f"tree {a.tree} layout {a.layout} namelists from {src}\n")
        for f, g, k, v in overrides:
            o.write(f"{f}:{g}:{k}={v}\n")
        if a.layout == "serial13":
            o.write("data.exch2 <- reference/data.exch2_13x90x90 (no blankList)\n")
    for name, p in links.items():
        (run / name).symlink_to(p)
    # diagnostics output sub-directories (ECCO's misc/tools/mkdir_subdir_diags.py does the same)
    dg = read_namelist(run / "data.diagnostics").get("diagnostics_list", {}) if (run / "data.diagnostics").exists() else {}
    for k, v in dg.items():
        if k.startswith("filename(") and v and "/" in v[0]:
            (run / v[0]).parent.mkdir(parents=True, exist_ok=True)
    binary = Path(a.binary) if a.binary else newest_binary(a.tree, a.layout)
    (run / "mitgcmuv").symlink_to(binary)
    air.audit(run, sha_out=run / "INPUTS.sha256")
    print(f"RUNDIR {run}  binary {binary.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
