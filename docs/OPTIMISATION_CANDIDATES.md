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
