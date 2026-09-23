#!/usr/bin/env python3
"""Plan Task 21: the result tables of docs/ADJOINT_RESULTS.md from the results.jsonl rows multiweek_grad.py wrote.

    python3 scripts/adjoint/summarize_results.py /work/ab0995/a270088/MIT/runs/adjoint/w07_* ...  (stdlib only)
"""

import json
import math
import sys
from pathlib import Path

PLATEAU_RTOL = 1e-3   # an FD row counts towards the plateau when |FD - AD| / |AD| <= this
# State fields FORWARD_STEP passes through unchanged (multiweek_grad.carried_constants, measured on the production
# step): their cotangent accumulates the gradient w.r.t. a constant field and is left out of the screen
CARRIED_CONSTANTS = ("gsNm_1", "gsNm_2", "gtNm_1", "gtNm_2", "hFac_surfC", "hFac_surfS", "hFac_surfW", "hMixLayer",
                     "runoff", "sIceLoad")
PROGNOSTIC = ("theta", "salt", "uVel", "vVel", "etaN")
BARS = (1.010, 0.020, 1.030)


def amplification(tr, k):
    """grad.amplification (median, log-spread, worst 3 consecutive chunks per step), stdlib."""
    rates = [(tr[i + 1] / tr[i]) ** (1.0 / k) for i in range(len(tr) - 1) if tr[i] > 0 and tr[i + 1] > 0]
    if not rates:
        return None
    srt = sorted(rates)
    n = len(srt)
    med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    lg = [math.log(r) for r in rates]
    mu = sum(lg) / n
    spread = math.sqrt(sum((x - mu) ** 2 for x in lg) / n)
    w = max(1, min(3, len(tr) - 1))
    sus = [(tr[i + w] / tr[i]) ** (1.0 / (k * w)) for i in range(len(tr) - w) if tr[i] > 0]
    worst = max(sus) if sus else float("nan")
    ok = med <= BARS[0] and spread <= BARS[1] and worst <= BARS[2]
    return dict(median=med, log_spread=spread, worst3=worst, passes=ok, n=n, rates=rates)


def screen_traces(r):
    const = set(r.get("carried_constants") or CARRIED_CONSTANTS)
    ft = r.get("field_trace") or []
    sel = {"dynamic": lambda k: k not in const, "theta": lambda k: k == "theta",
           "prognostic": lambda k: k in PROGNOSTIC}
    return {n: [math.sqrt(sum(v * v for k, v in f.items() if s(k))) for f in ft] for n, s in sel.items()}


def load(dirs):
    rows = []
    for d in dirs:
        p = Path(d) / "results.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                r["_dir"] = Path(d).name
                rows.append(r)
    return rows


def f(x, n=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    if x != x:
        return "nan"
    return f"{x:.{n}e}" if (abs(x) < 1e-2 or abs(x) >= 1e4) and x != 0 else f"{x:.{n + 1}g}"


def fz(fr):
    return ", ".join(f"{k}={v}" for k, v in fr.items() if v not in (None, "exact"))


def main(dirs):
    rows = load(dirs)
    grads = [r for r in rows if r["action"] == "grad"]
    g0 = {(r["days"], r["mode"]): r for r in grads if r.get("repeat", 0) == 0}
    tl = [r for r in rows if r["action"] == "tl"]
    fd = [r for r in rows if r["action"] == "fd"]
    base = {r["days"]: r for r in rows if r["action"] == "fd_base"}
    out = []
    # ---------------------------------------------------------------- FD plateaus
    out.append("### Exact mode: FD h-sweep vs adjoint (and TL) per named control\n")
    out.append("| window | direction | adjoint (exact) | TL (exact) | FD rel. error at h = ... | plateau "
               f"(rel <= {PLATEAU_RTOL:g}) | ecco adjoint rel. to best FD | job (grad / FD) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for days in sorted({r["days"] for r in fd}):
        for n in sorted({r["direction"] for r in fd if r["days"] == days}):
            sel = sorted([r for r in fd if r["days"] == days and r["direction"] == n], key=lambda r: -r["h"])
            ge, gc = g0.get((days, "exact")), g0.get((days, "ecco"))
            if ge is None:
                continue
            ad = ge["dirderiv"][n]
            t = [r for r in tl if r["days"] == days and r["direction"] == n and r["mode"] == "exact"]
            errs = [(r["h"], abs(r["fd"] - ad) / max(abs(ad), 1e-300)) for r in sel]
            cells = "; ".join(f"{h:g}: {e:.1e}" for h, e in errs)
            pl = [h for h, e in errs if e <= PLATEAU_RTOL]
            best = min(sel, key=lambda r: abs(r["fd"] - ad))
            ecco = "-"
            if gc is not None:
                ecco = f"{abs(gc['dirderiv'][n] - best['fd']) / max(abs(best['fd']), 1e-300):.1e}"
            plat = (f"{len(pl)}/{len(errs)} h ({min(pl):g}..{max(pl):g})" if pl else f"0/{len(errs)}")
            out.append(f"| {days:g} d | {n} | {f(ad, 6)} | {f(t[0]['tl'], 6) if t else '-'} | {cells} | {plat} | "
                       f"{ecco} | {ge['job']} / {sel[0]['job']} |")
    if base:
        out.append("")
        out.append("Forward noise floor (spread of repeated J evaluations at the base point, same process): " +
                   "; ".join(f"{d:g} d: {r['spread']:.1e} (J = {r['J'][0]:.15f}, {r['forward_s']:.1f} s/forward)"
                             for d, r in sorted(base.items())))
    # ---------------------------------------------------------------- dot tests
    comb = [r for r in tl if r["direction"] == "combined"]
    if comb:
        out.append("\n### TL (jax.jvp) vs adjoint (chunked jax.vjp): combined direction\n")
        out.append("| window | mode | amplitude | TL | adjoint <g, v> | rel. difference | job |")
        out.append("|---|---|---|---|---|---|---|")
        for r in sorted(comb, key=lambda r: (r["days"], r["mode"], -r["amp"])):
            out.append(f"| {r['days']:g} d | {r['mode']} | {r['amp']:g} | {r['tl']:.15e} | "
                       f"{r.get('adjoint', float('nan')):.15e} | {r.get('rel', float('nan')):.1e} | {r['job']} |")
        per = [r for r in tl if r["direction"] != "combined" and "rel" in r]
        if per:
            out.append("")
            out.append("Per named direction, TL vs adjoint (max rel. difference over directions): " + "; ".join(
                f"{d:g} d: {max(r['rel'] for r in per if r['days'] == d):.1e}"
                for d in sorted({r['days'] for r in per})))
    # ---------------------------------------------------------------- screens
    sc = [r for r in rows if r["action"] == "screen"]
    if sc:
        out.append("\n### Amplification screen (terminal seed: end-of-window box mean; per-chunk State-cotangent norm)\n")
        out.append("Statistics over the chunks after the seed chunk (the first chunk maps the theta/hFacC seed onto the "
                   "whole State: a change of norm between field sets, not growth); carried constants left out; the "
                   "dynamic norm also without the chunk ending at the window start (AB2 start: gu/gvNm_2 unread).\n")
        out.append("| window | mode | freezes | chunk | norm over | median /step | log-spread | worst-3 /step | passes | "
                   "norm at window end -> start | job |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sc, key=lambda r: (r["days"], r["mode"])):
            for name, tr in screen_traces(r).items():
                # dynamic: also without the chunk ending at the window start (AB2 start: gu/gvNm_2 unread there)
                a = amplification(tr[1:-1] if name == "dynamic" else tr[1:], r["chunk"])
                if a is None:
                    continue
                out.append(f"| {r['days']:g} d | {r['mode']} | {fz(r['freezes']) or 'none'} | {r['chunk']} | {name} | "
                           f"{a['median']:.5f} | {a['log_spread']:.4f} | {a['worst3']:.5f} | {f(a['passes'])} | "
                           f"{tr[1]:.3e} -> {tr[-1]:.3e} | {r['job']} |")
    # ---------------------------------------------------------------- repeats and ecco vs exact
    if grads:
        out.append("\n### Gradient runs: repeats, ecco vs exact, cost\n")
        out.append("| window | mode | repeat | J | norm g_theta0 | norm g_kapGM | norm g_tflux | norm g_taux | max rel. diff to r0 "
                   "(theta0, kapGM, tflux) | forward s | reverse s | rev/fwd | device peak GB | host GB | job |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(grads, key=lambda r: (r["days"], r["mode"], r.get("repeat", 0))):
            gn = r["grad_norm"]
            rep = r.get("repeat_maxrel_r0")
            reps = ", ".join(f"{rep[k]:.1e}" for k in ("theta", "kapGM", "tflux")) if rep else "-"
            if rep is not None:
                reps += " (J bitwise)" if r.get("repeat_J_equal_r0") else " (J differs)"
            out.append(f"| {r['days']:g} d | {r['mode']} | {r.get('repeat', 0)} | {r['J']:.15f} | {f(gn['theta'])} | "
                       f"{f(gn['kapGM'])} | {f(gn['tflux'])} | {f(gn['taux'])} | {reps} | {r['forward_s']:.0f} | "
                       f"{r['reverse_s']:.0f} | {r['reverse_s'] / r['forward_s']:.2f} | {f(r['peak_gb'])} | "
                       f"{f(r['host_gb'])} | {r['job']} |")
        warm = []
        for days in sorted({r["days"] for r in grads}):
            pf = base.get(days)
            w = [r for r in grads if r["days"] == days and r.get("repeat", 0) > 0]
            if pf and w:
                tot = min(r["forward_s"] + r["reverse_s"] for r in w)
                warm.append(f"{days:g} d: {tot:.0f} s = {tot / pf['forward_s']:.1f} x one plain forward "
                            f"({pf['forward_s']:.1f} s, {pf['forward_s'] / w[0]['nsteps']:.3f} s/step)")
        if warm:
            out.append("\nWarm gradient (chunked forward + reverse, ecco repeats) against one plain jitted forward of the "
                       "same window: " + "; ".join(warm))
        dev = []
        for (days, mode), r in sorted(g0.items()):
            if mode != "ecco" or (days, "exact") not in g0:
                continue
            e = g0[(days, "exact")]
            parts = []
            for n in r["dirderiv"]:
                a, b = e["dirderiv"][n], r["dirderiv"][n]
                parts.append(f"{n} {abs(b - a) / max(abs(a), 1e-300):.1e}")
            dev.append(f"| {days:g} d | " + "; ".join(parts) + " |")
        if dev:
            out.append("\nECCO mode vs exact mode, relative difference of the directional derivatives:\n")
            out.append("| window | per direction |")
            out.append("|---|---|")
            out.extend(dev)
    fc = [r for r in rows if r["action"] == "forward_check"]
    if fc:
        out.append("\nECCO-mode forward vs exact-mode forward (whole window, all State fields, bytes): " + "; ".join(
            f"{r['days']:g} d: {'bitwise' if r['bitwise'] else 'DIFFERS ' + str(r['fields_differing'])} "
            f"({r['nfields']} fields, J {r['J_ecco']:.15f}, job {r['job']})" for r in sorted(fc, key=lambda r: r['days'])))
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
