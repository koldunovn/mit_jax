# Brainstorm 2026-09-23 — MITgcm (c66g, ECCO v4r4) → JAX

Decisions (user-confirmed, one question at a time):
1. First big success = forward + adjoint together: 1 yr full V4r4 forward within run-to-run spread of Fortran,
   AND a validated JAX gradient over a multi-week window on real LLC90 (flux-forced ocean first).
2. Oracle = instrumented MITgcm c66g Fortran (gfortran), env-gated per-substep dumps keyed by (facet,i,j,k)
   + input dumps for replay; direct Fortran→JAX, no C stage; harness built before any JAX code.
3. Layout = tiles + halos, one code path: arrays [tile,k,j,i]; halo fill from one exch2-derived map
   (gather on 1 device, ppermute rounds when sharded); tile size a parameter.
4. Staging = real LLC90 grid + V4r4 namelists from day one; V4r4 flux-forced first, physics switched on stage by
   stage vs Fortran of the same config; then EXF bulk + sea ice (offline_exf_seaice/lab_sea to isolate LSR).
   Port only branches V4r4 uses.
5. Adjoint semantics = both, static switches: 'ecco' (V4r4 data.autodiff: seaice/GGL90/salt plume off in
   reverse, GM slope freeze, viscFacInAd) and 'exact'; forward byte-identical in both.
6. Tests = ~10 min tier 1 (CPU node, <100 tests), ~1 h tier 2 (GPU nightly/milestone), tier 3 milestone
   climate twins.
7. Env = new env on /work, but REVISED after user note ("some things did not work in fesom-jax in newer
   versions of jax"): start pinned to the fesom-jax known-good set (jax 0.10.1, constraints.txt); a JAX upgrade
   is a deliberate, gated step (canary = tier-1 suite + gradient gates + a short GPU run on old vs new), never a
   drift. The breakage is not recorded in port_jax/docs (ENV.md says 0.11.0 ran the suite) — ask Nikolay for details.
8. Section 1 (architecture/components: mitgcm_jax/{grid,parallel,core,<pkgs>,adjoint,io}, reference/, tools/)
   approved.

Context: ~/MIT/CATALOG.md, /work/ab0995/a270088/MIT/notes/*.md
9. Section 2 (reference harness: gfortran c66g+V4r4 builds (flux-forced, full), jaxdump shim per substep keyed
   (facet,i,j,k) + routine input dumps for replay, matched pickup start, tools/diffdump.py with tolerance classes,
   PO.DAAC data staged, always-on range checks + SSH/heat/salt budgets) approved.
10. Section 3 (differentiability: finite masked lanes + safe ops + full-IC-field grad gate; implicit solver
    differentiation (cg2d custom_linear_solve, vertical tri/penta-diag scans, LSR fixed iterations, custom_root
    later for exact); fixed iteration counts; static AdjointConfig per package ecco/exact/off with forward
    byte-identical; fesom_jax checkpoint stack + named halo policy + disk level later; Params controls, WC01 via
    linear_transpose later; gradient trust protocol incl. TAF output_adm comparison in ecco mode) approved.
11. Section 4 (parallel: shard_map over tile axis, check_vma, 90x90 default / 30x30 option, blank tiles dropped;
    one exch2 map as gather or coloured ppermute rounds, vector flags, corner code literal; fixed tile-order global
    sums; invariants + negative controls; ragged_all_to_all banned; perf targets 1 GPU + 1 node vs 96-core Fortran,
    vmap ensembles; multi-node by design but not milestone 1; sharded multi-week gradient on one node IS in M1) approved.
12. Section 5 (tests tier1 <10 min CPU / tier2 ~1 h GPU / tier3 milestone twins, <100 tests; milestones
    M0 foundations, M1 flux-forced ocean staged + sharded multi-week gradient, M2 full V4r4 (EXF, LSR ice, 1-yr twin),
    M3 26-yr twin, perf, long adjoints/disk level, TAF comparison, exact ice adjoint, ctrl/cost; process rules)
    approved. Next: write plan via planning:make.
13. Plan review (NEEDS REVISION) addressed: flux-forced override tree audited as its own task, Fortran step order,
    data staging (1992 adjusted forcing, control_weights, smooth), achievable reference gates, ecco seams (sigmaX/Y/R,
    TAF inAdMode semantics, viscFacInAd custom_vjp), early safe ops/full-field grad gate/sharded loader, task splits.
14. TAF gradient comparison dropped from M1/M2 acceptance; moves to M3 (Nikolay).
