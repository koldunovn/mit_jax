# Optimisation candidates (for the optimisation campaign)

Ideas that would make the port faster but are NOT done now. Classes (Nikolay, 2026-09-23):
- **bitwise** — same operations, same order, same result: can be done any time once gated (e.g. cg2d sum unroll 5,
  the Pallas LSR sweep — both already in).
- **round-off** — same algorithm, different operation order: results differ at round-off, not bitwise with the
  Fortran; allowed in the optimisation campaign as a documented, gated change ("if we can do result the same and
  things are faster").
- **algorithm** — a different method: different results; a DEVIATION needing Nikolay's approval.

| candidate | class | expected gain | notes |
|---|---|---|---|
| cg2d global sums as a tree / pairwise reduction instead of the literal tile-ordered sequential sum | round-off | cg2d is ~160 iterations x 2 global sums per step; a tree sum is parallel on GPU | GLOBAL_SUM_ORDER_TILES order kept now (bitwise gates); gate a tree sum by the round-off-floor tests + %MON drift |
| skip CALC_OCE_MXLAYER + FIND_ALPHA when hMixLayer only feeds diagnostics | algorithm (diagnostics-only effect) | one Nr-level scan + EOS derivatives per step | Nikolay 2026-09-23: keep it for now |
| sea-ice LSR -> zebra (red-black) LSR / JFNK / mEVP | algorithm | GPU-parallel sea-ice dynamics (LSR is sequential chains: 2.6-4 ms/sweep with Pallas) | V4r4 uses plain LSR; zebra exists in MITgcm (SEAICE_LSR_ZEBRA); mEVP is FESOM's choice |
| host-offloaded / disk checkpointing for long adjoint windows | bitwise | memory beyond 28 days | plan M3 |
| GMRES of the implicit LSR derivative: DONE bitwise on the CPU (branch gmres-speed: `_gmres_windowed`, basis stored window by window, active basis rows only; bitwise vs ef3ea4e) | bitwise | see PORTING_LESSONS "GMRES ... bitwise CPU speed-up" | remaining ideas below |
| GMRES basis kernels on more CPU threads (XLA:CPU runs the small-output window-sum fusions on one thread) | bitwise if the per-window orders are kept (e.g. one fusion per tile block) | the basis work is ~2/3 of a solve after the windowed rewrite | XLA splits fusions by output size only; independent fusions/loops were measured to run one after the other |
| GMRES inner products and updates as einsum/dot_general (library GEMV order) | round-off | measured NOT faster than the windowed form (tangent+transpose 16.6 s vs 15.1 s, loaded node); 8% of the solution values differ, max 1.6e-16 of max | not adopted |
| GMRES basis restricted to wet points inside each window | bitwise for the sums (zero terms), but the signs of zero at dry points / exact-zero wet points must be reproduced | ~1/(wet fraction) less basis traffic | not tried |
