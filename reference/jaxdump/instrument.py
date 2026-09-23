#!/usr/bin/env python3
"""Insert jaxdump calls into copies of the MITgcm sources of one V4r4 tree (plan Task 5).

    instrument.py TREE OUTDIR          TREE = full | ff; writes instrumented copies + jaxdump.F + JAXDUMP.h to OUTDIR
    instrument.py --markdown           print the SUBSTEPS table (reference/jaxdump/SUBSTEPS.md is generated from it)

Each STAGES entry names a source file, an anchor (regex matched against non-comment lines), which occurrence to use,
how many occurrences the file must contain (a drifted source fails loudly), and what to dump right after the anchor
statement (after its continuation lines). The source is the tree's override if it has one, else c66g.
Sources are never modified in place. A stage may be restricted to some trees (M2 stages exist only in the full tree:
the flux-forced build does not run EXF bulk formulae or sea ice); a tree's build gets instrumented copies only of the
files that carry one of its stages, so adding full-tree stages leaves the ff instrumentation byte-identical.
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
C66G_DIRS = ("model/src", "pkg/exf", "pkg/seaice")

# (file, anchor, occurrence, expected total, stage, dump statements, scope, what the substep does[, options])
# dump statements: 'S:<groups>' -> JAXDUMP_STATE(stage, groups, bi0, bj0);  'T:<name>:<kind>:<nz>' -> JAXDUMP_TILE of a
# routine-local array of the current tile; 'G:<name>:<kind>:<nz>' -> JAXDUMP_LOCAL of a routine-local all-tile array;
# 'K:<name>:<kind>[:<expr>]' -> JAXDUMP_TILEK of the 2-D level k (loop variable k) of the current tile inside a k loop,
# field <name>_k<kkk>; <expr> (default <name>) is the array element the 2-D slice starts at;
# 'U:<name>:<kind>:<nz>' -> JAXDUMP_TILEI of a routine-local INTERIOR-only array (1:sNx,1:sNy,nz) of the current tile
# (halos written as 0); 'N:<name>[:<expr>]' -> JAXDUMP_SCALAR of a scalar (integer or real) as a constant field on
# every tile (kind 'N').
# scope 'all' dumps every tile (bi0=0), 'tile' only the current bi,bj (inside a tile loop).
# Anchors: 'BEFORE:<re>' inserts before the statement (to catch a routine's inputs); 'ENDDO<n>:<re>' inserts after the
# ENDDO that closes the n-th DO loop enclosing the matched line (the matched line itself counts as level 1 when it is a
# DO statement); a plain '<re>' inserts after the statement.
# options (dict): 'trees' tuple of trees that get the stage (default both); 'loop' an integer Fortran expression whose
# value is appended to the stage name as '_p<n>' (stages inside an iteration loop, e.g. the LSR Picard passes);
# 'cond' a Fortran logical expression guarding the dumps (e.g. only the first LSOR sweep).
FS, OP, DY, TH, SP = "forward_step.F", "do_oceanic_phys.F", "dynamics.F", "thermodynamics.F", "solve_for_pressure.F"
TI, SI = "temp_integrate.F", "salt_integrate.F"
# M2 (full tree): EXF bulk forcing and sea ice
XG, XR, XB = "exf_getforcing.F", "exf_radiation.F", "exf_bulkformulae.F"
IM, IY, IL, IA, IG = "seaice_model.F", "seaice_dynsolver.F", "seaice_lsr.F", "seaice_advdiff.F", "seaice_growth.F"
FULL = {"trees": ("full",)}
STAGES = [
    (FS, r"CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "S00_begin", ["S:dtarfmkgpxc"], "all", "state at the start of the step"),
    (FS, r"CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "G00_geometry", ["S:GVRX"], "all",
     "grid, masks, 3-D mixing parameters, packed vertical grid, extra r* fields, exch2 exchange probe"),
    (FS, r"CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "G01_seaice_geometry", ["S:I"], "all",
     "sea-ice static fields: HEFFM, k1/k2 metric terms (full tree)", FULL),
    (FS, r"CALL AUTODIFF_INADMODE_UNSET\(", 1, 1, "S00i_begin_ice_exf", ["S:iube"], "all",
     "sea-ice state and EXF fields (interpolated + bracketing records) at the start of the step (full tree)", FULL),
    (FS, r"CALL UPDATE_R_STAR\(\s*\.FALSE\.", 1, 1, "S01_update_rstar_F", ["S:rd"], "all",
     "RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) (every step: ALLOW_AUTODIFF)"),
    # --- M2: EXF_GETFORCING (called from LOAD_FIELDS_DRIVER), full tree: c66g pkg/exf ---
    (XG, r"CALL EXF_GETFFIELDS\(", 1, 1, "X01_exf_getffields", ["S:bex"], "all",
     "EXF_GETFFIELDS: read, time-interpolate, rotate (A-grid stress) the forcing records", FULL),
    (XR, r"CALL EXF_ZENITHANGLE\(", 1, 1, "X02_exf_zenithangle", ["S:b"], "all",
     "EXF_ZENITHANGLE inside EXF_RADIATION: zen_fsol_*, swdown rescaled (useExfZenIncoming)", FULL),
    (XG, r"CALL EXF_RADIATION\(", 1, 1, "X03_exf_radiation", ["S:bx"], "all",
     "EXF_RADIATION: lwflux (emissivity, SST^4), swflux", FULL),
    (XG, r"CALL EXF_WIND\(", 1, 1, "X04_exf_wind", ["S:b"], "all",
     "EXF_WIND: wStress, cw, sw, sh, uwind/vwind from the stress (useAtmWind=F)", FULL),
    (XB, r"BEFORE:DO iter=1,niter_bulk", 1, 1, "X05a_bulk_init",
     ["U:tstar:C:1", "U:qstar:C:1", "U:ustar:C:1", "U:rdn:C:1", "U:delq:C:1", "U:deltap:C:1"], "tile",
     "EXF_BULKFORMULAE: neutral first guess before the stability iterations", FULL),
    (XB, r"ENDDO1:DO iter=1,niter_bulk", 1, 1, "X05b_bulk_iter",
     ["U:tstar:C:1", "U:qstar:C:1", "U:ustar:C:1", "U:tau:C:1", "U:rdn:C:1", "U:rd:C:1"], "tile",
     "EXF_BULKFORMULAE: turbulent scales after the niter_bulk stability iterations (where atemp=0 the locals are "
     "never set: values there are undefined)", FULL),
    (XG, r"CALL EXF_BULKFORMULAE\(", 1, 1, "X05_exf_bulkformulae", ["S:b"], "all",
     "EXF_BULKFORMULAE (Large-Yeager 2004): hs, hl, evap", FULL),
    (XG, r"BEFORE:CALL EXF_GETSURFACEFLUXES\(", 1, 1, "X06_exf_hflux_sflux", ["S:bx"], "all",
     "hflux, sflux (net, runoff, masked) and the stress exchange (EXCH_UV_AGRID_3D_RL)", FULL),
    (XG, r"CALL EXF_GETSURFACEFLUXES\(", 1, 1, "X07_exf_getsurfacefluxes", ["S:x"], "all",
     "EXF_GETSURFACEFLUXES (control adjustments)", FULL),
    (XG, r"CALL EXF_MAPFIELDS\(", 1, 1, "X08_exf_mapfields", ["S:xf"], "all",
     "EXF_MAPFIELDS: fu, fv, Qnet, Qsw, EmPmR, saltFlux, pLoad", FULL),
    (FS, r"CALL LOAD_FIELDS_DRIVER\(", 1, 1, "S02_load_fields", ["S:xfp"], "all", "EXF read, time interpolation, map"),
    (FS, r"CALL CTRL_MAP_FORCING\(", 1, 1, "S03_ctrl_map_forcing", ["S:xf"], "all", "time-varying controls (zero)"),
    # --- M2: SEAICE_MODEL (called from DO_OCEANIC_PHYS), full tree: c66g pkg/seaice + V4r4 seaice_growth.F ---
    (IM, r"BEFORE:CALL SEAICE_DYNSOLVER\s*\(", 1, 1, "I00_seaice_begin", ["S:iub"], "all",
     "SEAICE_MODEL inputs (uwind/vwind after EXCH_UV_AGRID_3D_RL)", FULL),
    (IY, r"CALL SEAICE_GET_DYNFORCING\s*\(", 1, 1, "Y01_get_dynforcing", ["S:y"], "all",
     "ice masses, TAUX/TAUY from fu/fv (SEAICE_EXTERNAL_FLUXES, useAtmWind=F)", FULL),
    (IY, r"CALL SEAICE_CALC_ICE_STRENGTH\(", 1, 1, "Y02_ice_strength", ["S:y"], "tile",
     "FORCEX0/Y0 and SEAICE_CALC_ICE_STRENGTH: PRESS0, ZMAX, ZMIN", FULL),
    (IY, r"CALL SEAICE_FREEDRIFT\(", 1, 1, "Y03_freedrift", ["S:y"], "all",
     "SEAICE_FREEDRIFT (LSR_mixIniGuess=0): uice_fd, vice_fd", FULL),
    (IY, r"BEFORE:CALL SEAICE_LSR\(", 1, 1, "Y04_before_lsr", ["S:iuyv"], "all", "all inputs of SEAICE_LSR", FULL),
    (IL, r"CALL SEAICE_OCEANDRAG_COEFFS\(", 1, 1, "L01_lsr_visc_drag",
     ["S:uv", "G:uIceC:W:1", "G:vIceC:S:1"], "all",
     "Picard pass p: uIce/uIceNm1 update, strain rates, viscosities, pressure, ocean drag DWATN",
     {**FULL, "loop": "ipass"}),
    (IL, r"CALL SEAICE_LSR_CALC_COEFFS\(", 1, 1, "L02_lsr_coeffs",
     ["S:v", "G:etaPlusZeta:C:1", "G:zetaMinusEta:C:1", "G:dragSym:C:1", "G:rhsU:W:1", "G:rhsV:S:1",
      "G:AU:W:1", "G:BU:W:1", "G:CU:W:1", "G:uRt1:W:1", "G:uRt2:W:1",
      "G:AV:S:1", "G:BV:S:1", "G:CV:S:1", "G:vRt1:S:1", "G:vRt2:S:1"], "all",
     "Picard pass p: FORCEX/Y, rhsU/V (SEAICE_LSR_RHSU/V), tridiagonal coefficients (SEAICE_LSR_CALC_COEFFS)",
     {**FULL, "loop": "ipass"}),
    (IL, r"CALL EXCH_UV_XY_RL\( uIce, vIce,\.TRUE\.,myThid\)", 1, 1, "L03_lsor_sweep1", ["S:u"], "all",
     "Picard pass p: uIce, vIce after the FIRST LSOR sweep (TRIDIAGU/V, relaxation) and its exchange",
     {**FULL, "loop": "ipass", "cond": "m.EQ.1"}),
    (IL, r"ENDDO1:DO m = 1, SOLV_MAX_TMP", 1, 1, "L04_lsor_end",
     ["S:u", "N:ICOUNT1", "N:ICOUNT2", "N:S1", "N:S2", "N:WFAU", "N:WFAV"], "all",
     "Picard pass p: uIce, vIce after the LSOR loop (before masking); iteration counts ICOUNT1/2, last dU/dV "
     "S1/S2, relaxation WFAU/V", {**FULL, "loop": "ipass"}),
    (IY, r"CALL SEAICE_LSR\(", 1, 1, "Y05_lsr", ["S:uv"], "all", "SEAICE_LSR result (masked uIce, vIce)", FULL),
    (IY, r"CALL SEAICE_OCEAN_STRESS\s*\(", 1, 1, "Y06_ocean_stress", ["S:f"], "all",
     "SEAICE_OCEAN_STRESS: fu, fv under ice (+ EXCH_UV_XY_RS)", FULL),
    (IM, r"CALL SEAICE_DYNSOLVER\s*\(", 1, 1, "I01_dynsolver", ["S:uyvf"], "all",
     "SEAICE_DYNSOLVER incl. velocity clipping (SEAICE_clipVelocities)", FULL),
    (IA, r"CALL SEAICE_ADVECTION\(", 1, 6, "A01_heff_adv",
     ["T:uTrans:W:1", "T:vTrans:S:1", "T:gFld:C:1", "T:afx:W:1", "T:afy:S:1"], "tile",
     "HEFF: DST3-FL advective tendency gFld and fluxes afx/afy (SEAICE_ADVECTION, scheme 33)", FULL),
    (IA, r"CALL SEAICE_DIFFUSION\(", 1, 11, "A02_heff_diff", ["T:gFld:C:1"], "tile",
     "HEFF: tendency after SEAICE_DIFFUSION (Laplacian, SEAICEdiffKhHeff)", FULL),
    (IA, r"CALL SEAICE_ADVECTION\(", 2, 6, "A03_area_adv", ["T:gFld:C:1", "T:afx:W:1", "T:afy:S:1"], "tile",
     "AREA: advective tendency (after the HEFF update)", FULL),
    (IA, r"CALL SEAICE_DIFFUSION\(", 2, 11, "A04_area_diff", ["T:gFld:C:1"], "tile", "AREA: + diffusion", FULL),
    (IA, r"CALL SEAICE_ADVECTION\(", 4, 6, "A05_snow_adv", ["T:gFld:C:1", "T:afx:W:1", "T:afy:S:1"], "tile",
     "HSNOW: advective tendency", FULL),
    (IA, r"CALL SEAICE_DIFFUSION\(", 4, 11, "A06_snow_diff", ["T:gFld:C:1"], "tile", "HSNOW: + diffusion", FULL),
    (IM, r"CALL SEAICE_ADVDIFF\(", 1, 1, "I02_advdiff", ["S:iu"], "all", "SEAICE_ADVDIFF: HEFF, AREA, HSNOW", FULL),
    (IM, r"CALL SEAICE_REG_RIDGE\(", 1, 1, "I03_reg_ridge", ["S:ih"], "all",
     "SEAICE_REG_RIDGE: negative-value and area regularisation, d_HEFFbyNEG, d_HSNWbyNEG", FULL),
    (IG, r"BEFORE:CALL SEAICE_BUDGET_OCEAN\(", 1, 1, "H01_growth_pre_budget",
     ["U:HEFFpreTH:C:1", "U:HSNWpreTH:C:1", "U:AREApreTH:C:1", "U:heffActual:C:1", "U:hsnowActual:C:1",
      "U:recip_heffActual:C:1", "U:UG:C:1", "U:TmixLoc:C:1"], "tile",
     "SEAICE_GROWTH (V4r4): regularised thicknesses, UG, TmixLoc = inputs of SEAICE_BUDGET_OCEAN", FULL),
    (IG, r"CALL SEAICE_BUDGET_OCEAN\(", 1, 1, "H02_growth_budget_ocean",
     ["U:a_QbyATM_open:C:1", "U:a_QSWbyATM_open:C:1"], "tile", "open-water heat budget (W/m2)", FULL),
    (IG, r"BEFORE:CALL SEAICE_SOLVE4TEMP\(", 1, 1, "H03_growth_pre_solve4temp",
     ["U:UG:C:1", "U:heffActualMult:C:nITD", "U:hsnowActualMult:C:nITD", "U:ticeInMult:C:nITD"], "tile",
     "inputs of SEAICE_SOLVE4TEMP (all categories)", {**FULL, "cond": "IT.EQ.1"}),
    (IG, r"CALL SEAICE_SOLVE4TEMP\(", 1, 1, "H04_growth_solve4temp",
     ["U:ticeInMult:C:nITD", "U:ticeOutMult:C:nITD", "U:a_QbyATMmult_cover:C:nITD",
      "U:a_QSWbyATMmult_cover:C:nITD", "U:a_FWbySublimMult:C:nITD"], "tile",
     "SEAICE_SOLVE4TEMP: ice surface temperature, ice-covered heat and sublimation fluxes (W/m2)",
     {**FULL, "cond": "IT.EQ.SEAICE_multDim"}),
    (IG, r"ENDDO2:r_QbyOCN\(i,j\) = a_QbyOCN\(i,j\)", 1, 1, "H05_growth_heat_stocks",
     ["U:a_QbyATM_cover:C:1", "U:a_QSWbyATM_cover:C:1", "U:a_QbyATM_open:C:1", "U:a_QSWbyATM_open:C:1",
      "U:r_QbyATM_cover:C:1", "U:r_QbyATM_open:C:1", "U:a_FWbySublim:C:1", "U:r_FWbySublim:C:1",
      "U:a_QbyOCN:C:1", "U:r_QbyOCN:C:1"], "tile",
     "end of PART 2: heat stocks in effective ice metres (atmosphere cover/open, ocean)", FULL),
    (IG, r"ENDDO2:QSW\(I,J,bi,bj\)  = QSW\(I,J,bi,bj\)\*convertHI2Q", 1, 1, "H06_growth_ocean_forcing",
     ["S:ih", "U:d_HEFFbyOCNonICE:C:1", "U:d_HEFFbyATMonOCN:C:1", "U:d_HEFFbyFLOODING:C:1",
      "U:d_HEFFbyATMonOCN_open:C:1", "U:d_HEFFbyATMonOCN_cover:C:1", "U:d_HSNWbyATMonSNW:C:1",
      "U:d_HSNWbyOCNonSNW:C:1", "U:d_HSNWbyRAIN:C:1", "U:d_HFRWbyRAIN:C:1", "U:d_HEFFbySublim:C:1",
      "U:d_HSNWbySublim:C:1", "U:r_QbyATM_cover:C:1", "U:r_QbyATM_open:C:1", "U:r_FWbySublim:C:1"], "tile",
     "PARTS 3-7 up to the Qnet/Qsw conversion to W/m2: thickness increments by process, updated HEFF/AREA/HSNOW, "
     "r_Q* residuals (Qnet/Qsw themselves: I04)", FULL),
    (IM, r"CALL SEAICE_GROWTH\(", 1, 1, "I04_growth", ["S:ihfp"], "all",
     "SEAICE_GROWTH: thermodynamics, ocean forcing Qnet/Qsw/EmPmR/saltFlux, sIceLoad, salt-plume flux", FULL),
    (OP, r"CALL SEAICE_MODEL\(", 1, 1, "P00_seaice_model", ["S:iufp"], "all",
     "all of SEAICE_MODEL (after its HEFF/AREA/HSNOW/forcing exchanges)", FULL),
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
    (DY, r"CALL CALC_PHI_HYD\(", 1, 1, "D00a_phi_hyd",
     ["K:dPhiHydX:W", "K:dPhiHydY:S", "K:phiHydC:C", "K:phiHydF:C"], "tile",
     "hydrostatic pressure (per level k): gradient terms dPhiHydX/Y, phiHydC, phiHydF (next interface)"),
    (DY, r"CALL MOM_VECINV\(", 1, 1, "D00b_mom_vecinv",
     ["K:gU:W:gU(1-OLx,1-OLy,k,bi,bj)", "K:gV:S:gV(1-OLx,1-OLy,k,bi,bj)", "K:guDissip:W", "K:gvDissip:S"], "tile",
     "vector-invariant momentum tendency of level k (gU, gV) and dissipation kept out of AB (guDissip, gvDissip)"),
    (DY, r"BEFORE:CALL IMPLDIFF\(", 1, 4, "D01_before_impl_visc", ["S:a", "T:kappaRU:W:Nr+1", "T:kappaRV:S:Nr+1"],
     "tile", "explicit gU, gV (after TIMESTEP) and vertical viscosities, input of IMPLDIFF (ALLOW_AUTODIFF path)"),
    (DY, r"CALL IMPLDIFF\(", 2, 4, "D02_after_impl_visc", ["S:a"], "tile", "gU, gV after implicit viscosity"),
    (FS, r"CALL DYNAMICS\(", 1, 1, "S05_dynamics", ["S:adm"], "all",
     "phi_hyd, momentum tendencies, AB3, implicit viscosity -> gU, gV"),
    (FS, r"CALL UPDATE_R_STAR\(\s*\.TRUE\.", 1, 1, "S06_update_rstar_T", ["S:r"], "all", "r* at the new time"),
    (FS, r"CALL UPDATE_CG2D\(", 1, 1, "S07_update_cg2d", ["S:c"], "all", "cg2d operator + preconditioner"),
    (SP, r"BEFORE:CALL CG2D\(", 1, 1, "C01_cg2d_inputs", ["G:cg2d_b:C:1", "G:cg2d_x:C:1", "S:c"], "all",
     "cg2d right-hand side, first guess and operator"),
    (SP, r"CALL CG2D\(", 1, 1, "C02_cg2d_solution", ["G:cg2d_x:C:1"], "all", "cg2d solution (before exchange)"),
    (FS, r"CALL SOLVE_FOR_PRESSURE\(", 1, 1, "S08_solve_for_pressure", ["S:d"], "all", "cg2d solve -> etaN"),
    (FS, r"CALL MOMENTUM_CORRECTION_STEP\(", 1, 1, "S09_momentum_correction", ["S:d"], "all", "u, v corrected"),
    (FS, r"CALL INTEGR_CONTINUITY\(", 1, 1, "S10_integr_continuity", ["S:d"], "all", "w, etaH"),
    (FS, r"CALL CALC_R_STAR\(", 1, 1, "S11_calc_rstar", ["S:r"], "all", "rStarFac from etaH"),
    (FS, r"CALL DO_STAGGER_FIELDS_EXCHANGES\(", 2, 2, "S12_stagger_exchanges", ["S:d"], "all",
     "exchanges before the staggered tracer step"),
    (TH, r"CALL GMREDI_RESIDUAL_FLOW\(", 1, 1, "T01_residual_flow", ["T:uFld:W:Nr", "T:vFld:S:Nr", "T:wFld:C:Nr"],
     "tile", "Eulerian + bolus velocity used by tracer advection"),
    (TI, r"CALL GAD_ADVECTION\(", 1, 1, "T10_temp_adv", ["T:gT_loc:C:Nr"], "tile",
     "theta: multi-dimensional DST3 advective tendency"),
    (TI, r"BEFORE:CALL TIMESTEP_TRACER\(", 1, 1, "T11_temp_gT", ["T:gT_loc:C:Nr"], "tile",
     "theta: total explicit tendency after forcing, diffusion, AB3 and r* rescale"),
    (TI, r"CALL TIMESTEP_TRACER\(", 1, 1, "T12_temp_step", ["T:gT_loc:C:Nr"], "tile", "theta: T + dt*gT"),
    (TI, r"CALL GAD_IMPLICIT_R\(", 1, 1, "T13_temp_impl", ["T:gT_loc:C:Nr", "T:kappaRk:C:Nr", "T:recip_hFac:C:Nr"],
     "tile", "theta after implicit vertical advection + diffusion (and its inputs kappaRk, recip_hFac)"),
    (TH, r"CALL TEMP_INTEGRATE\(", 1, 1, "T02_temp_integrate", ["S:ta"], "tile", "theta advanced (AB3, implicit)"),
    (SI, r"CALL GAD_ADVECTION\(", 1, 1, "T20_salt_adv", ["T:gS_loc:C:Nr"], "tile",
     "salt: multi-dimensional DST3 advective tendency"),
    (SI, r"BEFORE:CALL TIMESTEP_TRACER\(", 1, 1, "T21_salt_gS", ["T:gS_loc:C:Nr"], "tile",
     "salt: total explicit tendency after forcing, diffusion, AB3 and r* rescale"),
    (SI, r"CALL TIMESTEP_TRACER\(", 1, 1, "T22_salt_step", ["T:gS_loc:C:Nr"], "tile", "salt: S + dt*gS"),
    (SI, r"CALL GAD_IMPLICIT_R\(", 1, 1, "T23_salt_impl", ["T:gS_loc:C:Nr", "T:kappaRk:C:Nr", "T:recip_hFac:C:Nr"],
     "tile", "salt after implicit vertical advection + diffusion (and its inputs)"),
    (TH, r"CALL SALT_INTEGRATE\(", 1, 1, "T03_salt_integrate", ["S:ta"], "tile", "salt advanced (AB3, implicit)"),
    (FS, r"CALL THERMODYNAMICS\(", 2, 2, "S13_thermodynamics", ["S:ta"], "all",
     "GM residual flow, DST3 advection, diffusion, AB3 on theta/salt, implicit vertical"),
    (FS, r"CALL TRACERS_CORRECTION_STEP\(", 1, 1, "S14_tracers_correction", ["S:t"], "all", "end of step"),
]

_COMMENT = re.compile(r"^[cC*!]")
_CPP = re.compile(r"^#")
_CONT = re.compile(r"^     [^ 0]")
_DO = re.compile(r"^\s+(?:\d+\s+)?DO\s+(?:[A-Za-z]\w*\s*=|WHILE\b)", re.I)
_LABELLED_DO = re.compile(r"^\s+(?:\d+\s+)?DO\s+\d+", re.I)
_ENDDO = re.compile(r"^\s+(?:\d+\s+)?END\s*DO\b", re.I)


# forward_step.F advances the counter right after DYNAMICS (myIter = nIter0 + iLoop, forward_step.F:823 in c66g
# and both overrides): stages after that point pass myIter-1 so every record of one step carries the step's START iteration.
AFTER_ITER_UPDATE = {"T10_temp_adv", "T11_temp_gT", "T12_temp_step", "T13_temp_impl", "T20_salt_adv", "T21_salt_gS",
                     "T22_salt_step", "T23_salt_impl", "C01_cg2d_inputs", "C02_cg2d_solution", "T01_residual_flow", "T02_temp_integrate",
                     "T03_salt_integrate", "S06_update_rstar_T", "S07_update_cg2d", "S08_solve_for_pressure", "S09_momentum_correction",
                     "S10_integr_continuity", "S11_calc_rstar", "S12_stagger_exchanges", "S13_thermodynamics",
                     "S14_tracers_correction"}


def entries(tree=None):
    """STAGES as (file, anchor, occ, total, stage, dumps, scope, what, opts); only those of `tree` when given."""
    out = []
    for e in STAGES:
        opts = e[8] if len(e) > 8 else {}
        if tree is None or tree in opts.get("trees", tuple(TREES)):
            out.append((*e[:8], opts))
    return out


def _wrap(lines):
    """Re-wrap one generated CALL to fixed-form 72 columns when the default two-line layout is too long."""
    if all(len(ln) <= 72 for ln in lines):
        return lines
    text = " ".join(ln[6:].strip() if ln.startswith("     &") else ln.strip() for ln in lines)
    head, inner = text.split("(", 1)
    inner = inner.rsplit(")", 1)[0]
    args, depth, cur_arg, quoted = [], 0, "", False  # split at top-level commas only (expressions keep theirs)
    for ch in inner:
        if ch == "'":
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            args.append(cur_arg.strip())
            cur_arg = ""
        else:
            cur_arg += ch
    args.append(cur_arg.strip())
    out, cur = [], f"      {head.strip()}( "
    for n, a in enumerate(args):
        tok = a + (", " if n < len(args) - 1 else " )")
        if len(cur) + len(tok.rstrip()) > 72:
            out.append(cur.rstrip())
            cur = "     &   "
        cur += tok
    out.append(cur.rstrip())
    assert all(len(ln) <= 72 for ln in out), out
    return out


def _calls(stage, dumps, scope, opts=None):
    out = []
    for d in dumps:
        out += _wrap(_call(stage, d, scope))
    if "loop" in (opts or {}):  # stage name gets '_p<value>' (JAXDUMP_PASS; 0 switches the suffix off again)
        out = [f"      CALL JAXDUMP_PASS( {opts['loop']} )"] + out + ["      CALL JAXDUMP_PASS( 0 )"]
    if "cond" in (opts or {}):
        out = [f"      IF ( {opts['cond']} ) THEN"] + out + ["      ENDIF"]
    return out


def _call(stage, d, scope):
    bi, bj = ("bi", "bj") if scope == "tile" else ("0", "0")
    it = "myIter-1" if stage in AFTER_ITER_UPDATE else "myIter"
    out = []
    kind, rest = d.split(":", 1)
    if kind == "S":
        out += [f"      CALL JAXDUMP_STATE( '{stage}', '{rest}',",
                f"     &                    {bi}, {bj}, {it}, myThid )"]
    elif kind == "T":
        name, pk, nz = rest.split(":")
        out += [f"      CALL JAXDUMP_TILE( '{stage}', '{name}', '{pk}',",
                f"     &                   {name}, {nz}, bi, bj, {it}, myThid )"]
    elif kind == "K":
        parts = rest.split(":", 2)
        name, pk = parts[0], parts[1]
        expr = parts[2] if len(parts) > 2 else name
        out += [f"      CALL JAXDUMP_TILEK( '{stage}', '{name}', '{pk}',",
                f"     &   {expr}, k, bi, bj, {it}, myThid )"]
    elif kind == "U":
        name, pk, nz = rest.split(":")
        out += [f"      CALL JAXDUMP_TILEI( '{stage}', '{name}', '{pk}',",
                f"     &                    {name}, {nz}, bi, bj, {it}, myThid )"]
    elif kind == "N":
        parts = rest.split(":", 1)
        name = parts[0]
        expr = parts[1] if len(parts) > 1 else name
        out += [f"      CALL JAXDUMP_SCALAR( '{stage}', '{name}',",
                f"     &                     DBLE({expr}), {it}, myThid )"]
    elif kind == "G":
        name, pk, nz = rest.split(":")
        out += [f"      CALL JAXDUMP_LOCAL( '{stage}', '{name}', '{pk}',",
                f"     &                    {name}, {nz}, {it}, myThid )"]
    else:
        raise SystemExit(f"{stage}: unknown dump kind {d!r}")
    return out


def source_for(tree, fname):
    p = TREES[tree] / fname
    if p.exists():
        return p
    hits = [C66G / d / fname for d in C66G_DIRS if (C66G / d / fname).exists()]
    if len(hits) != 1:
        raise SystemExit(f"{fname}: found {len(hits)} times in c66g {C66G_DIRS}")
    return hits[0]


def _is_code(ln):
    return not _COMMENT.match(ln) and not _CPP.match(ln)


def _statement_end(lines, i):
    """Index of the last line of the statement starting at line i: continuation lines, including those that follow
    comment or preprocessor lines inside the statement (e.g. an #ifdef'd argument)."""
    last, j = i, i + 1
    while j < len(lines):
        if _CONT.match(lines[j]):
            last, j = j, j + 1
            continue
        k = j
        while k < len(lines) and not _is_code(lines[k]):
            k += 1
        if k > j and k < len(lines) and _CONT.match(lines[k]):
            j = k
            continue
        break
    return last


def _enddo_after(lines, i, level, src):
    """Line index of the ENDDO closing the `level`-th DO loop enclosing line i (line i counts when it is a DO)."""
    def check(ln, k):
        if _LABELLED_DO.match(ln):
            raise SystemExit(f"{src}:{k + 1}: labelled DO loop, ENDDO anchors do not support it")
    enclosing = []
    if _DO.match(lines[i]):
        enclosing.append(i)
    depth, k = 0, i - 1
    while len(enclosing) < level and k >= 0:
        ln = lines[k]
        if _is_code(ln):
            check(ln, k)
            if _ENDDO.match(ln):
                depth += 1
            elif _DO.match(ln):
                if depth:
                    depth -= 1
                else:
                    enclosing.append(k)
        k -= 1
    if len(enclosing) < level:
        raise SystemExit(f"{src}:{i + 1}: fewer than {level} enclosing DO loops")
    d0 = enclosing[level - 1]
    depth = 0
    for k in range(d0, len(lines)):
        ln = lines[k]
        if not _is_code(ln):
            continue
        check(ln, k)
        if _DO.match(ln):
            depth += 1
        elif _ENDDO.match(ln):
            depth -= 1
            if depth == 0:
                indent = lambda s: len(s) - len(s.lstrip())  # noqa: E731
                if indent(ln) != indent(lines[d0]):
                    raise SystemExit(f"{src}:{k + 1}: ENDDO indentation differs from its DO at line {d0 + 1}")
                return k
    raise SystemExit(f"{src}:{d0 + 1}: no matching ENDDO")


def _parse_anchor(anchor):
    """-> (mode, level, regex): mode 'after' | 'before' | 'enddo'."""
    if anchor.startswith("BEFORE:"):
        return "before", 0, anchor[len("BEFORE:"):]
    m = re.match(r"ENDDO(\d+):(.*)$", anchor)
    if m:
        return "enddo", int(m.group(1)), m.group(2)
    return "after", 0, anchor


def _hits(lines, pat):
    return [i for i, ln in enumerate(lines) if not _COMMENT.match(ln) and re.search(pat, ln)]


def instrument(tree, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = []
    stages = entries(tree)
    for fname in sorted({s[0] for s in stages}):
        src = source_for(tree, fname)
        lines = src.read_text().split("\n")
        inserts = {}
        for f, anchor, occ, total, stage, dumps, scope, _, opts in stages:
            if f != fname:
                continue
            mode, level, pat = _parse_anchor(anchor)
            hits = _hits(lines, pat)
            if len(hits) != total:
                raise SystemExit(f"{src}: anchor {pat!r} found {len(hits)} times, expected {total}")
            i = hits[occ - 1]
            if mode == "before":
                j = i
            elif mode == "enddo":
                j = _enddo_after(lines, i, level, src) + 1
            else:
                j = _statement_end(lines, i) + 1
            inserts.setdefault(j, []).extend(_calls(stage, dumps, scope, opts))
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
    for f, anchor, occ, total, stage, dumps, scope, what, opts in entries():
        locs = []
        for tree in ("full", "ff"):
            if tree not in opts.get("trees", tuple(TREES)):
                locs.append("—")
                continue
            src = source_for(tree, f)
            hits = _hits(src.read_text().split("\n"), _parse_anchor(anchor)[2])
            locs.append(f"{src.relative_to(REPO).as_posix().replace('ECCO-v4-Configurations/ECCOv4 Release 4/', '')}"
                        f":{hits[occ - 1] + 1}")
        sc = scope
        if "loop" in opts:
            sc += f", per `{opts['loop']}` (stage `_p<n>`)"
        if "cond" in opts:
            sc += f", if `{opts['cond']}`"
        mode, level, _ = _parse_anchor(anchor)
        where = {"before": "before ", "enddo": f"after the ENDDO of loop level {level} around "}.get(mode, "")
        rows.append(f"| `{stage}` | {where}{' / '.join(locs)} | {sc} | {', '.join(dumps)} | {what} |")
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
