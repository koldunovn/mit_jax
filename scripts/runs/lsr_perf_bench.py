#!/usr/bin/env python3
"""Cost of the LSOR sweep implementations (SeaiceDynParams.lsr_impl, seaice_lsr.py "Implementations") on the device
this runs on.

For each variant (`auto`, `pallas`, `xla_unrolled`, `xla`, `xla:u<k>` = "xla" with seaice_lsr.XLA_UNROLL = k,
`pallas_interpret`; variant_params):
  1. compile time and ms/sweep of a fixed number of sweeps (LSOR of iteration 1, Picard pass 1, with LSR_ERROR < 0 so
     that no convergence stops it): (t(N) - t(1)) / (N - 1);
  2. the first sweep (L03) and the whole LSOR of pass 1 (L04: sweep counts ICOUNT1/2) against the Fortran dumps;
  3. with --full: the whole SEAICE_DYNSOLVER at iterations 1-3 (as gpu_lsr_bench.py): s/step, sweep counts, the
     difference to the Fortran's I01 UICE/VICE, and every output field and count against the first variant;
  4. with --deriv: tangent and gradient of one LSOR solve (implicit derivative, GMRES).
Bitwise on CPU needs the gate flags (XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_disable_hlo_passes=algsimp").

    python scripts/runs/lsr_perf_bench.py --variants xla,pallas [--nsweep 20] [--full]
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

import mitgcm_jax  # noqa: E402,F401
from mitgcm_jax.pkgs import seaice_dyn as sd  # noqa: E402
from mitgcm_jax.pkgs import seaice_lsr as sl  # noqa: E402
from mitgcm_jax.pkgs import seaice_lsr_pallas as slp  # noqa: E402
from mitgcm_jax.tests import test_seaice_dyn as T  # noqa: E402


_DEFAULTS = dict(CHUNK=slp.CHUNK, CHUNK_B=slp.CHUNK_B, ROWS=slp.ROWS, NUM_WARPS=slp.NUM_WARPS)
_XLA_UNROLL = sl.XLA_UNROLL


def variant_params(p, name):
    """name = impl[:u<XLA_UNROLL>][:k<CHUNK>][:b<CHUNK_B>][:r<ROWS>][:w<NUM_WARPS>] (module constants of seaice_lsr /
    seaice_lsr_pallas, read at trace time; default the modules' values)."""
    impl, *opt = name.split(":")
    cfg = dict(_DEFAULTS)
    xu = _XLA_UNROLL
    for o in opt:
        if o.startswith("u"):
            xu = int(o[1:])
        elif o[0] in "kbrw":
            cfg[{"k": "CHUNK", "b": "CHUNK_B", "r": "ROWS", "w": "NUM_WARPS"}[o[0]]] = int(o[1:])
    for k, v in cfg.items():
        setattr(slp, k, v)
    sl.XLA_UNROLL = xu
    return dataclasses.replace(p, lsr_impl=impl)


def timed(f, *a, repeat=3):
    ts = []
    out = None
    for _ in range(repeat):
        t = time.time()
        out = f(*a)
        jax.block_until_ready(out)
        ts.append(time.time() - t)
    return out, float(np.median(ts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="xla,pallas")
    ap.add_argument("--nsweep", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--deriv", action="store_true", help="also time the tangent and the gradient of one LSOR solve")
    a = ap.parse_args(argv)
    print("devices:", jax.devices(), flush=True)
    env = T.Env()
    it = 1
    sg, ls = env.sg(it), env.lsr_state(it)
    # pass-1 LSOR inputs: SEAICE_LSR with one sweep per pass (pass 1's coefficients do not depend on the LSOR result)
    run1 = jax.jit(lambda p, g, sg, ex, ls: sl.seaice_lsr(p, g, sg, ex, ls, record=True, max_iter=1))
    _, passes = run1(env.p, env.g, sg, T.EX, ls)
    co, u0, v0 = passes[0]["co"], passes[0]["L02_inputs"]["uIce"], passes[0]["L02_inputs"]["vIce"]
    f03 = {k: env.f(it, "L03_lsor_sweep1_p1", k) for k in ("UICE", "VICE")}
    f04 = {k: env.f(it, "L04_lsor_end_p1", k) for k in ("UICE", "VICE", "ICOUNT1", "ICOUNT2")}
    results = {}
    for name in a.variants.split(","):
        try:
            solve = jax.jit(lambda p, co, u, v, n, e: sl.lsor_solve(p, T.EX, co, u, v, max_iter=n, lsr_error=e))
            results[name] = micro(a, env, name, co, u0, v0, f03, f04, solve)
            if a.deriv:
                deriv(a, env, name, co, *results[name])
        except Exception as e:  # noqa: BLE001  (report and go on with the next variant)
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:3000]}", flush=True)
    names = list(results)
    for n in names[1:]:
        print(f"[{n}] vs [{names[0]}] pass-1 LSOR: u {T._ndiff(results[n][0], results[names[0]][0])}, "
              f"v {T._ndiff(results[n][1], results[names[0]][1])} pts differ", flush=True)
    if not a.full:
        return 0
    full(a, env, names)
    return 0


def micro(a, env, name, co, u0, v0, f03, f04, solve):
    p = variant_params(env.p, name)
    args1 = (p, co, u0, v0, np.int32(1), np.float64(-1.0))
    t = time.time()
    comp = solve.lower(*args1).compile()
    tcomp = time.time() - t
    _, t1 = timed(comp, *args1, repeat=a.repeat)
    _, tn = timed(comp, p, co, u0, v0, np.int32(a.nsweep), np.float64(-1.0), repeat=a.repeat)
    ms = (tn - t1) / (a.nsweep - 1) * 1e3
    u1, v1, _ = comp(*args1)
    d03 = (T._ndiff(u1, f03["UICE"]), T._ndiff(v1, f03["VICE"]))
    uL, vL, info = comp(p, co, u0, v0, np.int32(p.SEAICElinearIterMax), np.float64(p.LSR_ERROR))
    cnt = (int(info["ICOUNT1"]), int(info["ICOUNT2"]), int(info["sweeps"]))
    d04 = {k: (T._ndiff(x, f04[k]), T._rel(x, f04[k])) for k, x in (("UICE", uL), ("VICE", vL))}
    fc = (int(f04["ICOUNT1"][0, 0, 0]), int(f04["ICOUNT2"][0, 0, 0]))
    ksw = kernel_only(a, p, co)
    print(f"[{name}] compile {tcomp:.1f} s; {ms:.3f} ms/sweep ({ksw:.3f} ms in the sweep alone; 1 sweep "
          f"{t1 * 1e3:.2f} ms, {a.nsweep} sweeps {tn * 1e3:.1f} ms); L03 ndiff u/v {d03}; pass-1 LSOR ICOUNT1/2, sweeps {cnt} (Fortran {fc}); L04 "
          + ", ".join(f"{k} {n} pts differ (rel {r:.1e})" for k, (n, r) in d04.items()), flush=True)
    return uL, vL


def deriv(a, env, name, co, u, v):
    """Tangent (jvp: GMRES on A, preconditioner = the XLA line-SOR sweep) and gradient (GMRES on A^T, transposed
    preconditioner) of J = sum(w*u) + sum(w*v) through one LSOR solve (pass 1, it 1) w.r.t. the coefficients, rhs
    direction; time per call and the dot-test agreement."""
    p = variant_params(env.p, name)
    rng = np.random.default_rng(0)
    wu, wv = (jnp.asarray(rng.standard_normal(u.shape)) for _ in range(2))
    d = {k: jnp.zeros_like(x) for k, x in co.items()}
    for k in ("rhsU", "rhsV"):
        d[k] = jnp.asarray(rng.standard_normal(u.shape)) * co[k] * 1e-2

    def J(c):
        uu, vv, _ = sl.lsor_solve(p, T.EX, c, u, v)
        return jnp.sum(wu * uu) + jnp.sum(wv * vv)

    tan_f = jax.jit(lambda c, dc: jax.jvp(J, (c,), (dc,))[1])
    grad_f = jax.jit(jax.grad(J))
    t = time.time()
    tan = float(tan_f(co, d))
    tct = time.time() - t
    t = time.time()
    g = grad_f(co)
    jax.block_until_ready(g)
    tcg = time.time() - t
    _, tt = timed(tan_f, co, d, repeat=a.repeat)
    g, tg = timed(grad_f, co, repeat=a.repeat)
    adj = float(sum(jnp.vdot(g[k], d[k]) for k in co))
    print(f"[{name}] derivative of one LSOR solve: tangent {tt:.2f} s (first call {tct:.1f} s), gradient {tg:.2f} s "
          f"(first call {tcg:.1f} s); adjoint vs tangent rel {abs(adj - tan) / abs(tan):.1e}", flush=True)


def kernel_only(a, p, co):
    """ms per call of the sweep alone (no halo/exchange/convergence work), a.nsweep calls in one fori_loop."""
    L = T.EX.L
    ln = sl._lines(co, L)
    lanes = ln["A"].shape[-1]
    u = co["rhsU"]
    tmp = jnp.concatenate([sl._to_lines_u(u, L), sl._to_lines_v(u, L)], axis=-1) * 0.0
    lo, hi, prev0, nxt = sl._halo_parts(u * 0.0, u * 0.0, L)
    w = jnp.full((lanes,), 0.95)

    def many(ln, tmp, lo, hi, prev0, nxt, w):
        sweep = sl._sweep_impl(p, ln)
        return lax.fori_loop(0, a.nsweep, lambda k, t: sweep(t, lo, hi, prev0, nxt, w), tmp)

    f = jax.jit(many)
    jax.block_until_ready(f(ln, tmp, lo, hi, prev0, nxt, w))
    _, t = timed(f, ln, tmp, lo, hi, prev0, nxt, w, repeat=a.repeat)
    return t / a.nsweep * 1e3


def full(a, env, names):
    first = {}
    for name in names:
        run = jax.jit(lambda p, g, sg, ex, st: sd.dynsolver(p, g, sg, ex, st, record=True))
        p = variant_params(env.p, name)
        for it in (1, 2, 3):
            sg, st = env.sg(it), env.dyn_state(it)
            t = time.time()
            out, rec = run(p, env.g, sg, T.EX, st)
            jax.block_until_ready(out["UICE"])
            tc = time.time() - t
            (out, rec), ts = timed(run, p, env.g, sg, T.EX, st, repeat=a.repeat)
            counts = [(int(pr["L04"]["ICOUNT1"]), int(pr["L04"]["ICOUNT2"])) for pr in rec["passes"]]
            nsweep = sum(c[0] for c in counts)
            d = {k: (T._rel(out[k], env.f(it, "I01_dynsolver", k)), T._ndiff(out[k], env.f(it, "I01_dynsolver", k)))
                 for k in ("UICE", "VICE")}
            print(f"[{name}] dynsolver it {it}: first call {tc:.1f} s, then {ts:.3f} s/step ({ts / nsweep * 1e3:.2f} "
                  f"ms/sweep incl. the rest, {nsweep} sweeps {counts}, Fortran {T.COUNTS[it]}); vs Fortran I01: "
                  + ", ".join(f"{k} rel {r:.1e} ({n} pts differ)" for k, (r, n) in d.items()), flush=True)
            outs = {k: np.asarray(v) for k, v in out.items()}
            if it not in first:
                first[it] = (name, outs, counts)
            else:
                n0, o0, c0 = first[it]
                bad = {k: int(np.sum(outs[k] != o0[k])) for k in o0}
                print(f"[{name}] vs [{n0}] dynsolver it {it}: counts {counts} vs {c0} "
                      f"({'equal' if counts == c0 else 'DIFFERENT'}); points differing in the {len(bad)} output "
                      f"fields: {sum(bad.values())} (UICE {bad['UICE']}, VICE {bad['VICE']})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
