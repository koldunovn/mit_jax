#!/usr/bin/env python3
"""Pre-flight audit of an MITgcm (c66g, ECCO v4r4) run's input files.

Parses the run's namelists, derives every input file the model will read, and checks that each exists in the run
directory (or in `--search` directories, first hit wins). Rules follow the c66g code that opens the files (cited
per rule below); namelist keys that look like file names but have no rule are listed as UNRESOLVED, never ignored.

    audit_run_inputs.py RUNDIR [--search DIR ...] [--sha256 OUT.txt] [--years]

Exit status 1 if a required file is missing. `--sha256` writes `sha256  name  size` for every found file (our own
record of exactly which bytes a run used). Cost-function inputs (pkg/ecco, pkg/profiles) are reported as class
`cost`; gencost data files are found as the file itself or as one `<name>_<yyyy>` per model year (ECCO_CHECK_FILES).
Most gencost files are only inquired by the model (a missing one switches its cost term off): they are optional.
"""

import argparse
import datetime as dt
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitgcm_jax.io.namelist import read_namelist  # noqa: E402


@dataclass
class Need:
    name: str        # file name, or prefix when kind == "prefix"
    source: str      # namelist file:group:key
    why: str         # code citation / condition
    klass: str = "input"   # input | cost
    kind: str = "exact"    # exact | prefix | obs (the file itself, else <name>_<yyyy> for each of `years`)
    required: bool = True  # False: optional (the code inquires; absence switches a cost term off or is harmless)
    years: tuple = ()


def _nml(rundir, fname):
    p = Path(rundir) / fname
    return read_namelist(p) if p.exists() else {}


def _get(nml, group, key, default=None):
    v = nml.get(group, {}).get(key)
    return v[0] if v else default


def _nonblank(v):
    return isinstance(v, str) and v.strip() != ""


def run_window(data, cal):
    """(start, end) datetimes of the integration from data.cal startDate and data PARM03 (c66g cal_init)."""
    d1 = int(_get(cal, "cal_nml", "startdate_1", 19920101))
    d2 = int(_get(cal, "cal_nml", "startdate_2", 0))
    base = dt.datetime.strptime(f"{d1:08d}{d2:06d}", "%Y%m%d%H%M%S")
    dtc = float(_get(data, "parm03", "deltatclock", _get(data, "parm03", "deltat", 0.0)))
    n0 = int(_get(data, "parm03", "niter0", 0))
    nt = _get(data, "parm03", "ntimesteps")
    if nt is None:
        end_time = float(_get(data, "parm03", "endtime"))
        nt = int(round(end_time / dtc)) - n0
    return base + dt.timedelta(seconds=n0 * dtc), base + dt.timedelta(seconds=(n0 + int(nt)) * dtc)


def exf_years(start, end, d1, d2, period):
    """Years whose yearly EXF file is read. Records sit at startdate + k*period within each year; the model
    interpolates between the records bracketing the current time (pkg/exf/exf_set_fld.F with
    useExfYearlyFields): the year before is needed if `start` precedes that year's first record, the year after if
    `end` is past its last record."""
    first = dt.datetime.strptime(f"{int(d1):08d}{int(d2):06d}", "%Y%m%d%H%M%S")
    off = first - dt.datetime(first.year, 1, 1)
    years = set(range(start.year, end.year + 1))
    if start < dt.datetime(start.year, 1, 1) + off:
        years.add(start.year - 1)
    nrec = int((dt.datetime(end.year + 1, 1, 1) - (dt.datetime(end.year, 1, 1) + off)).total_seconds() // period)
    last = dt.datetime(end.year, 1, 1) + off + dt.timedelta(seconds=(nrec - 1) * period)
    if end > last:
        years.add(end.year + 1)
    return sorted(years)


def needs(rundir):
    rundir = Path(rundir)
    data, pkg, cal = _nml(rundir, "data"), _nml(rundir, "data.pkg"), _nml(rundir, "data.cal")
    use = {k[3:].lower(): bool(v[0]) for k, v in pkg.get("packages", {}).items() if k.startswith("use")}
    out, unresolved, handled = [], [], set()

    def add(n):
        out.append(n)

    # --- core: data PARM05 (model/src/ini_parms.F reads; each file opened as noted)
    n0 = int(_get(data, "parm03", "niter0", 0))
    ic_keys = {"hydrogthetafile", "hydrogsaltfile", "uvelinitfile", "vvelinitfile", "psurfinitfile"}
    for k, v in data.get("parm05", {}).items():
        handled.add(("data", k))
        if not v or not _nonblank(v[0]) or k == "adtapedir":
            continue
        if k in ic_keys:
            add(Need(v[0], f"data:parm05:{k}", "read only if nIter0=0 (model/src/ini_fields.F:30)",
                     required=(n0 == 0)))
        else:
            add(Need(v[0], f"data:parm05:{k}", "model/src/ini_* (always read when set)"))

    # --- curvilinear grid: model/src/ini_curvilinear_grid.F:275-281 reads 'tile'//I3.3(face)//'.mitgrid' per facet
    #     (exch2 face number) when horizGridFile is blank; facet count from data.exch2 dimsFacets
    if _get(data, "parm04", "usingcurvilineargrid", False):
        hgf = _get(data, "parm05", "horizgridfile", "")
        dims = _nml(rundir, "data.exch2").get("w2_exch2_parm01", {})
        dims = next((v for k, v in dims.items() if k.startswith("dimsfacets")), [])
        for face in range(1, len(dims) // 2 + 1):
            name = f"tile{face:03d}.mitgrid" if not _nonblank(hgf) else f"{hgf}.face{face:03d}.bin"
            add(Need(name, "data.exch2:dimsfacets", "model/src/ini_curvilinear_grid.F:275-281"))

    # --- pickups (model/src/ini_fields.F:41 READ_PICKUP; pkg read_pickup routines), nIter0 > 0
    if n0 > 0:
        suf = f"{n0:010d}"
        for stem, on, why in (("pickup", True, "model/src/read_pickup.F"),
                              ("pickup_ggl90", use.get("ggl90"), "pkg/ggl90/ggl90_read_pickup.F"),
                              ("pickup_seaice", use.get("seaice"), "pkg/seaice/seaice_read_pickup.F")):
            if on:
                for ext in (".data", ".meta"):
                    add(Need(f"{stem}.{suf}{ext}", "data:parm03:niter0", why))
        if use.get("ecco"):
            add(Need(f"pickup_ecco.{suf}", "data:parm03:niter0",
                     "flux-forced ecco_read_pickup.F: optional, prints a message if absent", required=False))

    # --- EXF (pkg/exf/exf_readparms.F; yearly suffix in exf_set_fld.F / exf_getffield_start.F)
    if use.get("exf"):
        exf = _nml(rundir, "data.exf")
        yearly = bool(_get(exf, "exf_nml_01", "useexfyearlyfields", False))
        start, end = run_window(data, cal)
        g2 = exf.get("exf_nml_02", {})
        for k, v in g2.items():
            if not k.endswith("file"):
                continue
            handled.add(("data.exf", k))
            if not v or not _nonblank(v[0]):
                continue
            var = k[:-4]
            period = float(g2.get(var + "period", [0.0])[0])
            if yearly and period > 0:
                d1 = g2.get(var + "startdate1", [19920101])[0]
                d2 = g2.get(var + "startdate2", [0])[0]
                for y in exf_years(start, end, d1, d2, period):
                    add(Need(f"{v[0]}_{y}", f"data.exf:exf_nml_02:{k}",
                             f"useExfYearlyFields, period {period:g} s, run {start:%Y-%m-%d %H:%M}–{end:%Y-%m-%d %H:%M}"))
            else:
                add(Need(v[0], f"data.exf:exf_nml_02:{k}", f"period {period:g} (not yearly-suffixed)"))

    # --- GM/Redi 3-D coefficient files (pkg/gmredi/gmredi_init_fixed.F, ALLOW_KAPGM/KAPREDI_3DFILE)
    if use.get("gmredi"):
        for k, v in _nml(rundir, "data.gmredi").get("gm_parm01", {}).items():
            if k.endswith("file"):
                handled.add(("data.gmredi", k))
                if v and _nonblank(v[0]):
                    add(Need(v[0], f"data.gmredi:gm_parm01:{k}", "pkg/gmredi (3-D K file)"))

    # --- ctrl (pkg/ctrl: xx files <name>.<optimcycle %010d>.data/.meta, weights read as given)
    if use.get("ctrl"):
        oc = int(_get(_nml(rundir, "data.optim"), "optim", "optimcycle", 0))
        for k, v in _nml(rundir, "data.ctrl").get("ctrl_nml_genarr", {}).items():
            m = re.fullmatch(r"xx_gen(arr2d|arr3d|tim2d)_(file|weight)\(\d+\)", k)
            if not m:
                continue
            handled.add(("data.ctrl", k))
            if not v or not _nonblank(v[0]):
                continue
            if m.group(2) == "file":
                for ext in (".data", ".meta"):
                    add(Need(f"{v[0]}.{oc:010d}{ext}", f"data.ctrl:ctrl_nml_genarr:{k}",
                             f"ctrl_map_ini_gen*.F reads the xx file of optimcycle {oc}"))
            else:
                add(Need(v[0], f"data.ctrl:ctrl_nml_genarr:{k}", "ctrl weight file"))

    # --- smooth operators (pkg/smooth/smooth_init2d.F:41, smooth_init3d.F:107, smooth_correl2d.F:52,
    #     smooth_correl3d.F:76; names 'smooth2Dscales'//I3.3 etc.; norms read when smooth*filter = 0)
    if use.get("smooth"):
        sm = _nml(rundir, "data.smooth").get("smooth_nml", {})
        ops = sorted({int(m.group(2)) for k in sm if (m := re.fullmatch(r"smooth([23])dnbt\((\d+)\)", k))})
        for op in ops:
            if f"smooth2dnbt({op})" in sm:
                add(Need(f"smooth2Dscales{op:03d}", "data.smooth", "smooth_init2d.F:41"))
                if sm.get(f"smooth2dfilter({op})", [0])[0] == 0:
                    for ext in (".data", ".meta"):
                        add(Need(f"smooth2Dnorm{op:03d}{ext}", "data.smooth", "smooth_correl2d.F:52 (filter=0)"))
            if f"smooth3dnbt({op})" in sm:
                for s in ("H", "Z"):
                    add(Need(f"smooth3Dscales{s}{op:03d}", "data.smooth", "smooth_init3d.F:107"))
                if sm.get(f"smooth3dfilter({op})", [0])[0] == 0:
                    for ext in (".data", ".meta"):
                        add(Need(f"smooth3Dnorm{op:03d}{ext}", "data.smooth", "smooth_correl3d.F:76 (filter=0)"))

    # --- cost function inputs (pkg/ecco, pkg/profiles). Most gencost files are only inquired: a missing one switches
    #     its cost term off with a WARNING (required=False, "switch-off"); files read unconditionally are required.
    if use.get("ecco"):
        start, end = run_window(data, cal)
        ecco = _nml(rundir, "data.ecco")
        cn = ecco.get("ecco_cost_nml", {})
        for k, v in cn.items():
            if not re.search(r"file$", k):
                continue
            handled.add(("data.ecco", k))
            if not v or not _nonblank(v[0]):
                continue
            if k in ("data_errfile", "temp0errfile", "salt0errfile"):
                pass  # NOT read: they are read only in pkg/ecco/ecco_cost_weights.F, whose body is compiled only
                #       with ECCO_CTRL_DEPRECATED (:56), undefined in V4r4 (confirmed: strace of a V4r4 run opens none)
            else:
                unresolved.append(f"data.ecco:ecco_cost_nml:{k} = {v[0]!r}")
        gn = ecco.get("ecco_gencost_nml", {})
        ks = sorted({int(m.group(1)) for key in gn if (m := re.search(r"\((?:\d+,)?(\d+)\)$", key))})
        for i in ks:
            g = lambda key, d=None: gn.get(f"{key}({i})", [d])[0]  # noqa: E731
            for key in gn:
                if key.endswith(f"({i})") or key.endswith(f",{i})"):
                    handled.add(("data.ecco", key))
            name, dfile, efile, mask = g("gencost_name", ""), g("gencost_datafile"), g("gencost_errfile"), g("gencost_mask")
            src = f"data.ecco:ecco_gencost_nml:({i}) {name or '(no name)'}"
            preprocs = [gn[key][0] for key in gn if re.fullmatch(rf"gencost_preproc\(\d+,{i}\)", key)]
            if _nonblank(dfile):  # ECCO_CHECK_FILES (ecco_check.F:737-770): the file itself, else <name>_<yyyy> for
                #                   every model year >= the year of gencost_startdate1 (default 0, ecco_readparms.F:582)
                y0 = int(g("gencost_startdate1", 0) or 0) // 10000
                years = tuple(y for y in range(start.year, end.year + 1) if y >= y0)
                add(Need(dfile, src + " datafile", "ecco_check.F:737-770 (missing => term switched off)",
                         klass="cost", kind="obs", required=False, years=years))
            if _nonblank(efile):
                if "variaweight" in preprocs:
                    add(Need(efile, src + " errfile (variaweight)", "ecco_check.F:348-350 (yearly weights)",
                             klass="cost", kind="obs", required=False,
                             years=tuple(range(start.year, end.year + 1))))
                else:
                    add(Need(efile, src + " errfile", "ecco_check.F:352-365/408-420 (inquire; missing => term "
                             "switched off)", klass="cost", required=False))
            if _nonblank(mask):   # boxmean needs <mask>C, horflux <mask>W and <mask>S (ecco_check.F:449-506);
                #                   <mask>K is optional (ecco_readparms.F:775-783)
                sfx = "WS" if str(g("gencost_barfile", "")).startswith("m_horflux") else "C"
                for s in sfx:
                    add(Need(mask + s, src + " mask", "ecco_check.F:460-506 (inquire; missing => term switched off)",
                             klass="cost", required=False))
            for key in gn:
                if re.fullmatch(rf"gencost_posproc_c\(\d+,{i}\)", key) and _nonblank(gn[key][0]):
                    add(Need(gn[key][0], src + " posproc_c", "smooth_hetero2d.F:41 READ_REC_3D_RL (fatal if "
                             "missing)", klass="cost"))
        # sshv4-mdt: the along-track SLA files of tp/ers/gfo are read for EVERY year of the MDT period
        # (gencost_startdate1..enddate1 of the mdt term), whatever the run window (V4r4 cost_gencost_sshv4.F:293-305,
        # 383-442); a missing one is a bare STOP (pkg/ecco/cost_sla_read_yd.F:105-109)
        names = {gn.get(f"gencost_name({i})", [""])[0]: i for i in ks}
        if "sshv4-mdt" in names:
            m = names["sshv4-mdt"]
            y0 = int(gn.get(f"gencost_startdate1({m})", [0])[0]) // 10000
            y1 = int(gn.get(f"gencost_enddate1({m})", [0])[0]) // 10000
            for sat in ("sshv4-tp", "sshv4-ers", "sshv4-gfo"):
                d = gn.get(f"gencost_datafile({names.get(sat, 0)})", [None])[0]
                if _nonblank(d):
                    for y in range(y0, y1 + 1):
                        add(Need(f"{d}_{y}", f"data.ecco:ecco_gencost_nml:({m}) sshv4-mdt period {y0}-{y1} ({sat})",
                                 "cost_gencost_sshv4.F:383-442; cost_sla_read_yd.F:105-109 STOP if missing",
                                 klass="cost"))
    if use.get("profiles"):
        pr = _nml(rundir, "data.profiles").get("profiles_nml", {})
        for k, v in pr.items():
            if k.startswith("profilesfiles("):
                handled.add(("data.profiles", k))
                if v and _nonblank(v[0]):  # opened as <name>.nc in the run directory, NOT under profilesDir (which
                    #                        only receives the .equi output, profiles_ini_io.F:44-61); NF_OPEN's error
                    #                        code is not checked (profiles_init_fixed.F:187-188): required
                    add(Need(v[0] + ".nc", f"data.profiles:profiles_nml:{k}", "profiles_init_fixed.F:187-188",
                             klass="cost"))

    # --- anything else that looks like a file name in an active package's namelist: surface it
    active = {"data", "data.exf", "data.gmredi", "data.ggl90", "data.salt_plume", "data.seaice", "data.ctrl",
              "data.smooth", "data.ecco", "data.profiles", "data.diagnostics", "data.cal", "data.exch2", "eedata"}
    for fname in sorted(active):
        pkgname = fname.split(".", 1)[1] if "." in fname else None
        if pkgname and pkgname in use and not use[pkgname]:
            continue
        for grp, kv in _nml(rundir, fname).items():
            for k, v in kv.items():
                if (fname, k) in handled or not v or not _nonblank(v[0]):
                    continue
                if re.search(r"file(\(\d+\))?$|filename|maskfile", k) and "diag" not in fname:
                    unresolved.append(f"{fname}:{grp}:{k} = {v[0]!r}")
    seen, uniq = set(), []
    for n in out:  # one entry per file (e.g. weights_ones.data serves 8 controls); first source kept
        if n.name not in seen:
            seen.add(n.name)
            uniq.append(n)
    return uniq, unresolved


def locate(name, kind, dirs):
    for d in dirs:
        if kind == "exact":
            p = Path(d) / name
            if p.exists():
                return p
        else:
            base = Path(d) / name
            hits = sorted(base.parent.glob(base.name + "*")) if base.parent.exists() else []
            if hits:
                return hits[0]
    return None


def resolve(n, dirs):
    """(paths satisfying Need n, names still missing). kind "obs": the file itself, else one <name>_<yyyy> per year."""
    if n.kind == "obs":
        p = locate(n.name, "exact", dirs)
        if p is not None:
            return [p], []
        hits = {y: locate(f"{n.name}_{y}", "exact", dirs) for y in n.years}
        return [q for q in hits.values() if q], [f"{n.name}_{y}" for y, q in hits.items() if q is None]
    p = locate(n.name, n.kind, dirs)
    return ([p], []) if p is not None else ([], [n.name])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(rundir, search=(), sha_out=None, show_all=False):
    dirs = [Path(rundir), *map(Path, search)]
    items, unresolved = needs(rundir)
    missing = []
    found = []
    lines = {}  # (status, klass, source, why, stem) -> [years or names]; collapses yearly series
    for n in items:
        paths, absent = resolve(n, dirs)
        found += [(n, p) for p in paths]
        if absent and n.required:
            missing.append(n)
        for name, status in [(p.name, "ok") for p in paths] + [
                (a, "MISSING" if n.required else "absent (optional)") for a in absent]:
            if show_all or status != "ok":
                m = re.fullmatch(r"(.*)_(\d{4})", name)
                stem, tag = (m.group(1) + "_{YYYY}", m.group(2)) if m else (name, None)
                lines.setdefault((status, n.klass, n.source, n.why, stem), []).append(tag)
    for (status, klass, source, why, stem), tags in lines.items():
        years = [t for t in tags if t]
        what = stem.replace("{YYYY}", f"{{{years[0]}..{years[-1]}}} ({len(years)} files)") if years else stem
        print(f"{status:18s} {klass:5s} {what}  <- {source}  [{why}]")
    print(f"{len(items)} referenced ({sum(n.klass == 'cost' for n in items)} cost), {len(found)} found, "
          f"{len(missing)} required missing ({sum(n.klass == 'cost' for n in missing)} cost)")
    for u in unresolved:
        print(f"UNRESOLVED (no rule, not checked): {u}")
    if sha_out:
        with open(sha_out, "w") as f:
            for n, p in found:
                f.write(f"{sha256(p)}  {p.name}  {p.stat().st_size}\n")
    return missing, unresolved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir")
    ap.add_argument("--search", nargs="*", default=[])
    ap.add_argument("--sha256", metavar="OUT")
    ap.add_argument("--all", action="store_true", help="also list files that were found")
    a = ap.parse_args(argv)
    missing, _ = audit(a.rundir, a.search, a.sha256, a.all)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
