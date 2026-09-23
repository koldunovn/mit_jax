#!/usr/bin/env python3
"""Summarise a gcov-instrumented MITgcm run: which routines executed, per source package (plan Task 5 -> BRANCHES).

    branch_coverage.py BUILD_DIR RUN_DIR [--json OUT] [--markdown OUT]

BUILD_DIR is a `GCOV=1` build (reference/build.sh). reference/jobs/run.sbatch sets GCOV_PREFIX=RUN_DIR/gcov, so the
run's .gcda files sit under RUN_DIR/gcov/<BUILD_DIR/bld>; this script links them with the build's .gcno and .f into
RUN_DIR/gcov_obj (links only) and runs gcov there. For every compiled
file with a .gcda, `gcov -f -n` gives per-function line coverage; a routine counts as EXECUTED if any line ran.
Line numbers refer to the preprocessed .f files (MITgcm's cpp -P strips line markers), so this is a routine-level
scope list; branch detail inside a routine is read from `gcov -b` output on demand. Origin of each routine = where
its .F symlink points (c66g package, V4r4 override tree, or the jaxdump shim).
Run on a login node only for small builds: gcov over ~1100 files takes about a minute.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GCOV_MODULE = "gcc/11.2.0-gcc-11.2.0"
_FUNC = re.compile(r"Function '([^']+)'\s*\nLines executed:([\d.]+)% of (\d+)")


def origin(fsrc):
    p = Path(os.path.realpath(fsrc))
    s = str(p)
    if "/MITgcm_c66g/" in s:
        rel = s.split("/MITgcm_c66g/", 1)[1]
        parts = rel.split("/")
        return "c66g:" + ("/".join(parts[:2]) if parts[0] == "pkg" else parts[0] + "/" + parts[1])
    if p.parent.name == "code":  # copied into the build's code dir: a tree override, the jaxdump shim, or SIZE.h
        if p.name in ("jaxdump.F", "JAXDUMP.h"):
            return "jaxdump shim"
        tree = "flux-forced" if p.parent.parent.name.startswith("ff_") else "full"
        return f"V4r4 override ({tree})"
    return "other (" + p.name + ")"


def run_gcov(bld):
    out = {}
    files = sorted(f for f in Path(bld).glob("*.f") if f.with_suffix(".gcda").exists())
    for i in range(0, len(files), 100):
        chunk = [f.name for f in files[i:i + 100]]
        cmd = f"module load {GCOV_MODULE} >/dev/null 2>&1; cd {bld} && gcov -f -n {' '.join(chunk)}"
        txt = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout
        for m in _FUNC.finditer(txt):
            name, pct, n = m.group(1), float(m.group(2)), int(m.group(3))
            out[name] = (pct, n)
    return out, files


def object_dir(build_dir, run_dir):
    bld = (Path(build_dir) / "bld").resolve()
    gcda = Path(run_dir) / "gcov" / str(bld).lstrip("/")
    if not gcda.is_dir():
        raise SystemExit(f"no .gcda under {gcda}: was the run made with reference/jobs/run.sbatch?")
    obj = Path(run_dir) / "gcov_obj"
    obj.mkdir(exist_ok=True)
    for f in list(bld.glob("*.gcno")) + list(bld.glob("*.f")) + list(gcda.glob("*.gcda")):
        link = obj / f.name
        if not link.exists():
            link.symlink_to(f)
    return obj


def summarise(build_dir, run_dir):
    bld = object_dir(build_dir, run_dir)
    funcs, files = run_gcov(bld)
    # map function -> source file (by the .f that defines it)
    defn = {}
    for f in files:
        for m in re.finditer(r"^\s+(?:[A-Za-z*0-9 ]+\s+)?(?:SUBROUTINE|FUNCTION)\s+(\w+)", f.read_text(errors="replace"),
                             re.M | re.I):
            defn[m.group(1).lower() + "_"] = Path(build_dir) / "bld" / f.with_suffix(".F").name
    rows = []
    for name, (pct, n) in funcs.items():
        src = defn.get(name)
        rows.append({"routine": name.rstrip("_"), "executed": pct > 0, "lines_pct": pct, "lines": n,
                     "file": src.name if src else "?", "origin": origin(src) if src else "unmapped (main/C code)"})
    return rows, len(files)


def markdown(rows, nfiles, build_dir):
    by = {}
    for r in rows:
        by.setdefault(r["origin"], []).append(r)
    ex = [r for r in rows if r["executed"]]
    lines = [f"Build `{build_dir}`: {nfiles} compiled files with coverage data, {len(rows)} routines, "
             f"**{len(ex)} executed** ({sum(r['lines'] for r in ex)} source lines in executed routines).", "",
             "| origin | executed routines | not executed |", "|---|---|---|"]
    for o in sorted(by):
        e = sorted(r["routine"] for r in by[o] if r["executed"])
        lines.append(f"| {o} | {len(e)} | {len(by[o]) - len(e)} |")
    lines += ["", "Executed routines (lines % = share of the routine's lines that ran):", ""]
    for o in sorted(by):
        e = sorted((r for r in by[o] if r["executed"]), key=lambda r: r["routine"])
        if e:
            lines.append(f"- **{o}**: " + ", ".join(f"`{r['routine']}` ({r['lines_pct']:.0f}%)" for r in e))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("build_dir")
    ap.add_argument("run_dir")
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    a = ap.parse_args(argv)
    rows, nfiles = summarise(a.build_dir, a.run_dir)
    if not rows:
        raise SystemExit("no coverage data (.gcda) found: has the gcov binary been run?")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    md = markdown(rows, nfiles, a.build_dir)
    if a.markdown:
        Path(a.markdown).write_text(md + "\n")
    print(md[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
