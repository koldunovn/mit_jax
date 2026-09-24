#!/usr/bin/env python3
"""Tables of the M2 adjoint acceptance (plan Task 22) from the results.jsonl rows and gradient files that
scripts/adjoint/fullgrad.py writes (no JAX needed):

    python3 scripts/adjoint/fullgrad_summary.py DIR [DIR ...] [--ref-mode exact_full] [--shard SHARDED_DIR:ONE_GPU_DIR]

FD plateau: >= 2 of the 4 h with relative error <= 1e-3 against the reference mode's adjoint (Task 21 bar);
reported against every mode that has a gradient in the same window.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BAR_FD = 1e-3
MODE_ORDER = ("exact_full", "exact_nodyn", "exact_iceecco", "ecco")


def rows(dirs):
    out = []
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["_dir"] = str(d)
                    out.append(r)
    return out


def e(x, f="{:.1e}"):
    return "-" if x is None else f.format(x)


def mode_sort(m):
    return MODE_ORDER.index(m) if m in MODE_ORDER else 99


def grads_by(rs):
    """{(days, mode, nproc, ice_weight): {repeat: row}}"""
    g = defaultdict(dict)
    for r in rs:
        if r["action"] == "grad":
            g[(r["days"], r["mode"], r.get("nproc", 1), r.get("ice_weight", 1.0))][r.get("repeat", 0)] = r
    return g


OL = 4          # halo width of the saved interior fields (fullgrad saves [..., OLy:OLy+sNy, OLx:OLx+sNx])
POINT_CONTROL = {"atemp_pt": "atemp", "atemp_arctic_pt": "atemp", "heff_pt": "heff"}


def dirderiv(G, key, n, fdrow=None):
    """Directional derivative of direction n from the r0 gradient row of `key`, else (single-column directions,
    gradient computed before the direction existed) read from the saved gradient file at the FD row's point."""
    g = G.get(key, {}).get(0)
    if g is None:
        return None
    if n in g["dirderiv"]:
        return g["dirderiv"][n]
    pt = (fdrow or {}).get("where", {}).get("point")
    if n in POINT_CONTROL and pt is not None and len(pt) == 3:
        days, mode = key[0], key[1]
        f = Path(g["_dir"]) / f"grad_{mode}_{days:g}d_r0.npz"
        if f.exists():
            t, j, i = pt
            return float(np.load(f)[POINT_CONTROL[n]][t, j - OL, i - OL])
    return None


def fd_tables(rs, ref_mode, P):
    fd = [r for r in rs if r["action"] == "fd"]
    base = {r["days"]: r for r in rs if r["action"] == "fd_base"}
    G = grads_by(rs)
    tl = {(r["days"], r["mode"], r["direction"]): r for r in rs if r["action"] == "tl" and r.get("amp", 1.0) == 1.0}
    if not fd:
        return
    P(f"### FD h-sweep vs the {ref_mode} adjoint, per named direction\n")
    P(f"Relative error |FD - AD| / |AD| at each h, AD = the {ref_mode} adjoint (r0, J = J_theta + J_ice); plateau = "
      f"number of h with rel. error <= {BAR_FD:g} (bar: >= 2 of 4); J_theta / J_ice parts: the same against the "
      f"ice-weight-0 adjoint (J_theta) and the difference (J_ice), where both parts were recorded and the part carries "
      f">= 0.1 % of the derivative; the other modes' adjoints relative to the best FD value.\n")
    P(f"| window | direction | AD {ref_mode} | TL {ref_mode} | FD rel. error at h = ... | plateau | J_theta part: "
      f"errors, plateau | J_ice part: errors, plateau | other modes rel. to best FD | FD job |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for days in sorted({r["days"] for r in fd}):
        modes = sorted({k[1] for k in G if k[0] == days and k[2] == 1 and k[3] == 1.0}, key=mode_sort)
        for n in sorted({r["direction"] for r in fd if r["days"] == days}):
            byh = {}
            for r in fd:
                if r["days"] == days and r["direction"] == n and ("fd_theta" in r or r["h"] not in byh):
                    byh[r["h"]] = r
            sel = [byh[h] for h in sorted(byh, reverse=True)]
            ad = dirderiv(G, (days, ref_mode, 1, 1.0), n, sel[0])
            if ad is None:
                continue
            errs = [abs(r["fd"] - ad) / max(abs(ad), 1e-300) for r in sel]
            best = sel[int(np.argmin(errs))]["fd"]
            npl = sum(x <= BAR_FD for x in errs)
            parts = ["-", "-"]
            ad0 = dirderiv(G, (days, ref_mode, 1, 0.0), n, sel[0])
            if ad0 is not None and all("fd_theta" in r for r in sel):
                for ip, (lab, adp) in enumerate((("fd_theta", ad0), ("fd_ice", ad - ad0))):
                    if abs(adp) <= 1e-3 * abs(ad):     # < 0.1 % of J's derivative: FD of that part is noise
                        parts[ip] = f"not tested ({adp / ad:.0e} of AD)"
                        continue
                    pe = [abs(r[lab] - adp) / abs(adp) for r in sel]
                    parts[ip] = "; ".join(f"{x:.1e}" for x in pe) + f" ({sum(x <= BAR_FD for x in pe)}/{len(pe)})"
            others = []
            for m in modes:
                if m == ref_mode:
                    continue
                v = dirderiv(G, (days, m, 1, 1.0), n, sel[0])
                if v is not None:
                    others.append(f"{m} {abs(v - best) / max(abs(best), 1e-300):.1e}")
            t = tl.get((days, ref_mode, n))
            P(f"| {days:g} d | {n} | {ad:.6e} | {e(t['tl'], '{:.6e}') if t else '-'} | "
              + "; ".join(f"{r['h']:g}: {x:.1e}" for r, x in zip(sel, errs))
              + f" | {npl}/{len(sel)} | {parts[0]} | {parts[1]} | {'; '.join(others) or '-'} | "
                f"{','.join(sorted({str(r['job']) for r in sel}))} |")
    P("")
    for days, b in sorted(base.items()):
        P(f"Forward noise floor {days:g} d: spread {b['spread']:.1e} (J = {b['J'][0]:.16g}, {b['forward_s']:.1f} s per "
          f"plain forward, job {b['job']})")
    P("")


def tl_tables(rs, P):
    tl = [r for r in rs if r["action"] == "tl" and "rel" in r]
    if not tl:
        return
    P("### TL (jax.jvp) vs adjoint (chunked jax.vjp)\n")
    P("| window | mode | combined amp 1 | combined amp 1e-6 | named directions: max rel. difference | job |")
    P("|---|---|---|---|---|---|")
    for days in sorted({r["days"] for r in tl}):
        for mode in sorted({r["mode"] for r in tl if r["days"] == days}, key=mode_sort):
            s = [r for r in tl if r["days"] == days and r["mode"] == mode]
            c1 = [r["rel"] for r in s if r["direction"] == "combined" and r["amp"] == 1.0]
            c6 = [r["rel"] for r in s if r["direction"] == "combined" and r["amp"] != 1.0]
            nd = [r["rel"] for r in s if r["direction"] != "combined"]
            P(f"| {days:g} d | {mode} | {e(c1[0] if c1 else None)} | {e(c6[0] if c6 else None)} | "
              f"{e(max(nd) if nd else None)} ({len(nd)} dirs) | {s[0]['job']} |")
    P("")


def amplification(trace, k, bars=(1.010, 0.020, 1.030)):
    """grad.amplification (numpy copy): per-step rates of a chunk-boundary norm trace (window end first)."""
    tr = [float(t) for t in trace]
    rates = [(tr[i + 1] / tr[i]) ** (1.0 / k) if tr[i] > 0 else float("nan") for i in range(len(tr) - 1)]
    fin = np.array([x for x in rates if np.isfinite(x) and x > 0])
    if fin.size == 0:
        return None
    w = max(1, min(3, len(tr) - 1))
    sus = [(tr[i + w] / tr[i]) ** (1.0 / (k * w)) for i in range(len(tr) - w) if tr[i] > 0]
    med, spread, worst = float(np.median(fin)), float(np.std(np.log(fin))), float(np.max(sus))
    return dict(median=med, log_spread=spread, worst3=worst, trace=tr,
                passes=bool(med <= bars[0] and spread <= bars[1] and worst <= bars[2]))


# sea-ice thermodynamic state (without UICE, VICE: in exact_nodyn their reverse is the identity of the skipped LSR, so
# their cotangent accumulates like a carried constant; sea-ice-only study, docs/PORTING_LESSONS.md)
SEAICE_THERMO = ("AREA", "HEFF", "HSNOW", "TICES")


def screen_tables(rs, P):
    sc = [r for r in rs if r["action"] == "screen"]
    if not sc:
        return
    P("### Amplification screen (terminal seed; per-chunk State-cotangent norm per field group)\n")
    P("Groups: dynamic = every State field except the carried constants; prognostic = theta, salt, uVel, vVel, etaN; "
      "seaice = AREA, HEFF, HSNOW, TICES, UICE, VICE; seaice_thermo = without UICE, VICE. Statistics after the seed "
      "chunk (dynamic: also without the chunk ending at iteration 1).\n")
    P("| window | mode | nproc | group | median /step | log-spread | worst-3 /step | passes | norm end -> start | job |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(sc, key=lambda r: (r["days"], mode_sort(r["mode"]))):
        af = dict(r["amplification_fields"])
        ft = r.get("field_trace") or []
        th = [float(np.sqrt(sum(v ** 2 for k, v in f.items() if k in SEAICE_THERMO))) for f in ft]
        if len(th) > 2 and all(t > 0 for t in th[1:]):
            a = amplification(th[1:], r["chunk"])
            if a is not None:
                a["trace"] = th
                af["seaice_thermo"] = a
        for grp, v in af.items():
            tr = v.get("trace") or []
            P(f"| {r['days']:g} d | {r['mode']} | {r.get('nproc', 1)} | {grp} | {v['median']:.5f} | "
              f"{v['log_spread']:.4f} | {v['worst3']:.5f} | {'yes' if v['passes'] else 'NO'} | "
              f"{e(tr[1] if len(tr) > 1 else None, '{:.3e}')} -> {e(tr[-1] if tr else None, '{:.3e}')} | {r['job']} |")
    P("")


def grad_tables(rs, P):
    G = grads_by(rs)
    if not G:
        return
    base = {r["days"]: r for r in rs if r["action"] == "fd_base"}
    P("### Gradient runs: repeats, cost, memory\n")
    P("| window | mode | nproc | repeat | J | max rel. diff to r0 (theta, kapGM, heff, atemp, tauu) | forward s | "
      "reverse s | rev/fwd | wall s | wall / plain forward | device peak GB | host GB | job |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(G, key=lambda k: (k[0], mode_sort(k[1]), k[2], -k[3])):
        days, mode, nproc, iw = key
        for rep, r in sorted(G[key].items()):
            mr = r.get("repeat_maxrel_r0")
            mrs = "-" if mr is None else ", ".join(e(mr.get(k)) for k in ("theta", "kapGM", "heff", "atemp", "tauu"))
            if mr is not None:
                mrs += " (J bitwise)" if r.get("repeat_J_equal_r0") else " (J differs)"
            pf = base.get(days, {}).get("forward_s")
            ratio = r["wall_s"] / pf if pf else None
            P(f"| {days:g} d | {mode}{'' if iw == 1.0 else f' (ice weight {iw:g})'} | {nproc} | {rep} | {r['J']:.16g} | {mrs} | {r['forward_s']:.0f} | "
              f"{r['reverse_s']:.0f} | {r['reverse_s'] / r['forward_s']:.2f} | {r['wall_s']:.0f} | {e(ratio, '{:.1f}')} | "
              f"{e(r.get('peak_gb'), '{:.1f}')} | {r['host_gb']:.1f} | {r['job']} |")
    P("")
    P("Directional derivatives per mode (r0) and relative difference to the first listed mode:\n")
    for days in sorted({k[0] for k in G}):
        modes = sorted({k[1] for k in G if k[0] == days and k[2] == 1 and k[3] == 1.0}, key=mode_sort)
        if len(modes) < 2:
            continue
        ref = G[(days, modes[0], 1, 1.0)][0]["dirderiv"]
        P(f"| {days:g} d direction | " + " | ".join(modes) + " |")
        P("|---|" + "---|" * len(modes))
        for n in ref:
            vals = []
            for m in modes:
                v = G[(days, m, 1, 1.0)][0]["dirderiv"].get(n, float("nan"))
                vals.append(f"{v:.6e}" + ("" if m == modes[0] else
                                           f" ({abs(v - ref[n]) / max(abs(ref[n]), 1e-300):.1e})"))
            P(f"| {n} | " + " | ".join(vals) + " |")
        P("")


def fwd_tables(rs, P):
    fc = [r for r in rs if r["action"] == "forward_check"]
    if not fc:
        return
    P("### Forward values per mode (chunked forward, whole window, every State field + both cost parts, bytes)\n")
    P("| window | nproc | mode vs ref | bitwise | fields differing | J_theta | J_ice | job |")
    P("|---|---|---|---|---|---|---|---|")
    for r in sorted(fc, key=lambda r: (r["days"], mode_sort(r["mode"]))):
        P(f"| {r['days']:g} d | {r.get('nproc', 1)} | {r['mode']} vs {r['ref_mode']} | {'yes' if r['bitwise'] else 'NO'} "
          f"| {r['fields_differing'] or '-'} ({r['nfields']} fields) | {r['J_theta']:.16g} | {r['J_ice']:.16g} | "
          f"{r['job']} |")
    P("")


def shard_compare(pairs, P):
    if not pairs:
        return
    P("### Sharded (P GPUs, shard_map) vs 1 GPU\n")
    P("| sharded run | 1-GPU run | file | J sharded | J 1 GPU | rel. diff per control (max |a-b| / max |b|) | 1-GPU "
      "repeat floor (r1 vs r0) |")
    P("|---|---|---|---|---|---|---|")
    for pair in pairs:
        sd, od = pair.split(":")
        rs_s, rs_o = rows([sd]), rows([od])
        for fs in sorted(Path(sd).glob("grad_*_r0.npz")):
            fo = Path(od) / fs.name
            if not fo.exists():
                continue
            a, b = dict(np.load(fs)), dict(np.load(fo))
            rel = {k: float(np.max(np.abs(a[k] - b[k])) / max(np.max(np.abs(b[k])), 1e-300)) for k in a if k in b}
            fl = Path(od) / fs.name.replace("_r0", "_r1")
            floor = "-"
            if fl.exists():
                c = dict(np.load(fl))
                floor = ", ".join(f"{k} {np.max(np.abs(c[k] - b[k])) / max(np.max(np.abs(b[k])), 1e-300):.1e}"
                                  for k in b)
            Js = [r["J"] for r in rs_s if r["action"] == "grad" and r.get("repeat", 0) == 0]
            Jo = [r["J"] for r in rs_o if r["action"] == "grad" and r.get("repeat", 0) == 0]
            P(f"| {sd} | {od} | {fs.name} | {e(Js[0] if Js else None, '{:.16g}')} | {e(Jo[0] if Jo else None, '{:.16g}')}"
              f" | " + ", ".join(f"{k} {v:.1e}" for k, v in rel.items()) + f" | {floor} |")
    P("")
    P("Forward (final State of the chunked forward, whole window): sharded vs 1 GPU\n")
    P("| sharded run | 1-GPU run | file | J_theta, J_ice sharded | 1 GPU | fields differing (points) | max rel. diff |")
    P("|---|---|---|---|---|---|---|")
    for pair in pairs:
        sd, od = pair.split(":")
        rs_s, rs_o = rows([sd]), rows([od])
        for fs in sorted(Path(sd).glob("fwd_final_*.npz")):
            fo = Path(od) / fs.name
            if not fo.exists():
                continue
            a, b = dict(np.load(fs)), dict(np.load(fo))
            nd = {k: int(np.count_nonzero(a[k] != b[k])) for k in a if k in b}
            rel = max(float(np.max(np.abs(a[k] - b[k])) / max(np.max(np.abs(b[k])), 1e-300)) for k in a if k in b)
            js = [(r["J_theta"], r["J_ice"]) for r in rs_s if r["action"] == "forward"]
            jo = [(r["J_theta"], r["J_ice"]) for r in rs_o if r["action"] == "forward"]
            P(f"| {sd} | {od} | {fs.name} | {js[0] if js else '-'} | {jo[0] if jo else '-'} | "
              f"{ {k: v for k, v in nd.items() if v} or 'none (bitwise)'} | {rel:.1e} |")
    P("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref-mode", default="exact_full")
    ap.add_argument("--shard", action="append", default=[], help="SHARDED_DIR:ONE_GPU_DIR")
    a = ap.parse_args()
    rs = rows(a.dirs)
    P = print
    fd_tables(rs, a.ref_mode, P)
    if a.ref_mode != "exact_nodyn":
        fd_tables(rs, "exact_nodyn", P)
    tl_tables(rs, P)
    screen_tables(rs, P)
    grad_tables(rs, P)
    fwd_tables(rs, P)
    shard_compare(a.shard, P)


if __name__ == "__main__":
    main()
