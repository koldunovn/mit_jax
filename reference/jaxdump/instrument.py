#!/usr/bin/env python3
"""Insert jaxdump calls into copies of the MITgcm sources of one V4r4 tree (plan Task 5).

    instrument.py TREE OUTDIR          TREE = full | ff; writes instrumented copies + jaxdump.F + JAXDUMP.h to OUTDIR
    instrument.py --markdown           print the SUBSTEPS table (reference/jaxdump/SUBSTEPS.md is generated from it)

Each STAGES entry names a source file, an anchor (regex matched against non-comment lines), which occurrence to use,
how many occurrences the file must contain (a drifted source fails loudly), and what to dump right after the anchor
statement (after its continuation lines). The source is the tree's override if it has one, else c66g.
Sources are never modified in place.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
C66G = REPO / "MITgcm_c66g"
V4 = REPO / "ECCO-v4-Configurations" / "ECCOv4 Release 4"
TREES = {"full": V4 / "code", "ff": V4 / "flux-forced" / "code"}
C66G_PATH = {"forward_step.F": "model/src/forward_step.F", "do_oceanic_phys.F": "model/src/do_oceanic_phys.F"}

# (file, anchor, occurrence, expected total, stage, dump statements, scope, what the substep does)
# dump statements: 'S:<groups>' -> JAXDUMP_STATE(stage, groups, bi0, bj0);  'T:<name>:<kind>:<nz>' -> JAXDUMP_TILE of a
# routine-local array. scope 'all' dumps every tile (bi0=0), 'tile' only the current bi,bj (inside a tile loop).
FS, OP = "forward_step.F", "do_oceanic_phys.F"
STAGES = [
    (FS, r"CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "S00_begin", ["S:dtarfmkgpxc"], "all", "state at the start of the step"),
    (FS, r"CALL UPDATE_R_STAR\(\s*\.FALSE\.", 1, 1, "S01_update_rstar_F", ["S:rd"], "all",
     "RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) (every step: ALLOW_AUTODIFF)"),
    (FS, r"CALL LOAD_FIELDS_DRIVER\(", 1, 1, "S02_load_fields", ["S:xfp"], "all", "EXF read, time interpolation, map"),
    (FS, r"CALL CTRL_MAP_FORCING\(", 1, 1, "S03_ctrl_map_forcing", ["S:xf"], "all", "time-varying controls (zero)"),
    (OP, r"CALL EXTERNAL_FORCING_SURF\(", 1, 1, "P01_external_forcing_surf", ["S:fp"], "all",
     "surface forcing arrays (before the tile loop)"),
    (OP, r"CALL CALC_OCE_MXLAYER\(", 1, 1, "P02_rho_sigma_ivdc_mxlayer",
     ["S:m", "T:sigmaX:W:Nr", "T:sigmaY:S:Nr", "T:sigmaR:C:Nr"], "tile",
     "FIND_RHO_2D, GRAD_SIGMA, CALC_IVDC (k loop), CALC_OCE_MXLAYER"),
    (OP, r"CALL SALT_PLUME_CALC_DEPTH\(", 1, 1, "P03_salt_plume_depth", ["S:p"], "tile", "salt-plume depth"),
    (OP, r"CALL GGL90_CALC\(", 1, 1, "P04_ggl90", ["S:k"], "tile", "GGL90 TKE, viscosity, diffusivity"),
    (OP, r"CALL GMREDI_CALC_TENSOR\(", 1, 1, "P05_gmredi_tensor", ["S:g"], "tile", "GM/Redi slopes, taper, tensor"),
    (OP, r"CALL GMREDI_DO_EXCH\(", 1, 1, "P06_gmredi_exch", ["S:g"], "all", "GM/Redi tensor halo exchange"),
    (FS, r"CALL DO_OCEANIC_PHYS\(", 1, 1, "S04_oceanic_phys", ["S:fmkgprt"], "all", "all of DO_OCEANIC_PHYS"),
    (FS, r"CALL DYNAMICS\(", 1, 1, "S05_dynamics", ["S:adm"], "all",
     "phi_hyd, momentum tendencies, AB3, implicit viscosity -> gU, gV"),
    (FS, r"CALL UPDATE_R_STAR\(\s*\.TRUE\.", 1, 1, "S06_update_rstar_T", ["S:r"], "all", "r* at the new time"),
    (FS, r"CALL UPDATE_CG2D\(", 1, 1, "S07_update_cg2d", ["S:c"], "all", "cg2d operator + preconditioner"),
    (FS, r"CALL SOLVE_FOR_PRESSURE\(", 1, 1, "S08_solve_for_pressure", ["S:d"], "all", "cg2d solve -> etaN"),
    (FS, r"CALL MOMENTUM_CORRECTION_STEP\(", 1, 1, "S09_momentum_correction", ["S:d"], "all", "u, v corrected"),
    (FS, r"CALL INTEGR_CONTINUITY\(", 1, 1, "S10_integr_continuity", ["S:d"], "all", "w, etaH"),
    (FS, r"CALL CALC_R_STAR\(", 1, 1, "S11_calc_rstar", ["S:r"], "all", "rStarFac from etaH"),
    (FS, r"CALL DO_STAGGER_FIELDS_EXCHANGES\(", 2, 2, "S12_stagger_exchanges", ["S:d"], "all",
     "exchanges before the staggered tracer step"),
    (FS, r"CALL THERMODYNAMICS\(", 2, 2, "S13_thermodynamics", ["S:ta"], "all",
     "GM residual flow, DST3 advection, diffusion, AB3 on theta/salt, implicit vertical"),
    (FS, r"CALL TRACERS_CORRECTION_STEP\(", 1, 1, "S14_tracers_correction", ["S:t"], "all", "end of step"),
]

_COMMENT = re.compile(r"^[cC*!]")
_CONT = re.compile(r"^     [^ 0]")


# forward_step.F advances the counter right after DYNAMICS (myIter = nIter0 + iLoop, forward_step.F:823 in c66g
# and both overrides): stages after that point pass myIter-1 so every record of one step carries the step's START iteration.
AFTER_ITER_UPDATE = {"S06_update_rstar_T", "S07_update_cg2d", "S08_solve_for_pressure", "S09_momentum_correction",
                     "S10_integr_continuity", "S11_calc_rstar", "S12_stagger_exchanges", "S13_thermodynamics",
                     "S14_tracers_correction"}


def _calls(stage, dumps, scope):
    bi, bj = ("bi", "bj") if scope == "tile" else ("0", "0")
    it = "myIter-1" if stage in AFTER_ITER_UPDATE else "myIter"
    out = []
    for d in dumps:
        kind, rest = d.split(":", 1)
        if kind == "S":
            out += [f"      CALL JAXDUMP_STATE( '{stage}', '{rest}',",
                    f"     &                    {bi}, {bj}, {it}, myThid )"]
        else:
            name, pk, nz = rest.split(":")
            out += [f"      CALL JAXDUMP_TILE( '{stage}', '{name}', '{pk}',",
                    f"     &                   {name}, {nz}, bi, bj, {it}, myThid )"]
    return out


def source_for(tree, fname):
    p = TREES[tree] / fname
    return p if p.exists() else C66G / C66G_PATH[fname]


def instrument(tree, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = []
    for fname in sorted({s[0] for s in STAGES}):
        src = source_for(tree, fname)
        lines = src.read_text().split("\n")
        inserts = {}
        for f, anchor, occ, total, stage, dumps, scope, _ in STAGES:
            if f != fname:
                continue
            hits = [i for i, ln in enumerate(lines) if not _COMMENT.match(ln) and re.search(anchor, ln)]
            if len(hits) != total:
                raise SystemExit(f"{src}: anchor {anchor!r} found {len(hits)} times, expected {total}")
            i = hits[occ - 1]
            j = i + 1
            while j < len(lines) and _CONT.match(lines[j]):
                j += 1
            inserts.setdefault(j, []).extend(_calls(stage, dumps, scope))
            report.append((stage, fname, str(src.relative_to(REPO)), i + 1, scope, dumps))
        if fname == FS:
            upd = [i for i, ln in enumerate(lines) if re.match(r"\s+myIter = nIter0 \+ iLoop\s*$", ln)]
            dyn = [i for i, ln in enumerate(lines) if re.search(r"CALL DYNAMICS\(", ln) and not _COMMENT.match(ln)]
            rst = [i for i, ln in enumerate(lines) if re.search(r"CALL UPDATE_R_STAR\(\s*\.TRUE\.", ln)]
            if not (len(upd) == 1 and dyn[0] < upd[0] < rst[0]):
                raise SystemExit(f"{src}: iteration-counter update not where AFTER_ITER_UPDATE assumes")
        new = []
        for k, ln in enumerate(lines):
            if k in inserts:
                new += ["C--   jaxdump (mitgcm-jax Task 5) -- no effect unless JAXDUMP_DIR is set"] + inserts[k]
            new.append(ln)
        if len(lines) in inserts:
            new += inserts[len(lines)]
        (outdir / fname).write_text("\n".join(new))
    for f in ("jaxdump.F", "JAXDUMP.h"):
        shutil.copy(Path(__file__).with_name(f), outdir / f)
    return sorted(report, key=lambda r: r[0])


def markdown():
    rows = ["| stage | file (full / ff source) : line of anchor | scope | dumps | what |", "|---|---|---|---|---|"]
    for f, anchor, occ, total, stage, dumps, scope, what in STAGES:
        locs = []
        for tree in ("full", "ff"):
            src = source_for(tree, f)
            hits = [i for i, ln in enumerate(src.read_text().split("\n"))
                    if not _COMMENT.match(ln) and re.search(anchor, ln)]
            locs.append(f"{src.relative_to(REPO).as_posix().replace('ECCO-v4-Configurations/ECCOv4 Release 4/', '')}"
                        f":{hits[occ - 1] + 1}")
        rows.append(f"| `{stage}` | {' / '.join(locs)} | {scope} | {', '.join(dumps)} | {what} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", nargs="?", choices=("full", "ff"))
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    if a.markdown:
        print(markdown())
        return 0
    for stage, fname, src, line, scope, dumps in instrument(a.tree, a.outdir):
        print(f"{stage:28s} {src}:{line}  {scope}  {dumps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
