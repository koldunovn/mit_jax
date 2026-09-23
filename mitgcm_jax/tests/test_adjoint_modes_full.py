"""Backward-mode semantics on the full V4r4 tree (plan M2.6b-2, tier1x): AdjointConfig.ecco of the full data.autodiff,
the seam census of the full-tree FORWARD_STEP, the forward byte-identical in every mode, and effect tests of the two
seams whose placement is new in the full tree (the sea-ice level at the SEAICE_MODEL call, the salt-plume flux seam
after SEAICE_MODEL) on the live oracle state. docs/ADJOINT_MODES.md, adjoint/modes.py.

  - ecco(): the full run's data.autodiff (useSEAICEinAdMode = useGGL90inAdMode = useSALT_PLUMEinAdMode = .FALSE.) and
    code/GMREDI_OPTIONS.h:21 give seaice "ecco", ggl90 "frozen", salt_plume "off", gm_sigma "stable", cg2d "passive",
    visc_fac_in_ad 1.0; the oracle's STDOUT.0000 prints these switch values; the published V4r4 namelist directory
    gives the same (it was refused before M2.6b-2). AdjointConfig() keeps seaice "ecco" (the default everywhere).
  - Census: relative to AdjointConfig(seaice="full") (no seam), each switch adds exactly its own equations to the
    traced full step: stop_gradient sigma 3, GGL90 4, cg2d 3, salt plume 2 (SEAICE_MODEL's saltPlumeFlux output and
    the salt-plume depth); custom_jvp_call: viscFacInAd 1, seaice "ecco" 1 (the whole SEAICE_MODEL), "no_dynamics" 2
    (the FREEDRIFT+LSR block and the clipping block).
  - Gradient-driver default (Nikolay, 2026-09-23): adjoint/checkpoint.make_step(nml=...) is the run's ECCO config
    (make_step() without a config or namelists is an error); the scan integrator of the gradient drivers
    (checkpoint.integrate, schedule "step", EXF window of the full tree incl. the zenith scalars) run over iterations
    1-2 is bitwise the Fortran at the start of iteration 3.
  - Forward: step 1 with every switch on, in two combinations covering all sea-ice levels (no_dynamics / full, with
    stable / gm_only sigma), is bitwise the Fortran at the start of iteration 2 (the "ecco" default is test_step_full).
  - Effects at the DO_OCEANIC_PHYS level (forward_step.do_oceanic_phys, entry state = the Fortran state at the start of
    iteration 2), J = sum(w * surfaceForcingT) etc. with random weights on the wet interior:
      * seaice "ecco": d(surfaceForcingT)/d(Qnet on entry) = -w/(Cp*rhoConst) exactly (SEAICE_MODEL's adjoint is the
        identity on the Qnet it overwrites), d/d theta = 0 exactly (it adds nothing to what it only reads);
        "no_dynamics": the sea-ice thermodynamics reads theta (d/d theta non-zero) and overwrites Qnet.
      * salt plume "off" with seaice "no_dynamics": d(surfaceForcingS)/d salt loses the plume path
        (saltPlumeFlux = f(salt at k=1) in SEAICE_GROWTH); d(saltPlumeDepth)/d(theta, salt) = 0 exactly. With seaice
        "ecco" the plume-flux seam is redundant: d(surfaceForcingS)/d salt is bitwise the same with and without it
        (the sea-ice skip already passes nothing from SEAICE_MODEL's saltPlumeFlux output to the ocean state, and its
        entry value is zeroed at c66g do_oceanic_phys.F:293); only the depth seam acts.
      * forward outputs bitwise equal in all four configurations; every gradient finite.
Measured 2026-09-23 (16 CPU cores): census base (seaice "full", no seam) 5 stop_gradient / 39 custom_jvp_call, every
delta as listed; non-zero d(surfaceForcingT)/d theta: ecco 0, no_dynamics 12294 points; d(saltPlumeDepth)/d salt:
167710 points (exact), 0 (salt_plume "off"); d(surfaceForcingS)/d salt ecco == ecco+off bitwise.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from mitgcm_jax.adjoint import checkpoint as ck
from mitgcm_jax.adjoint.modes import AdjointConfig
from mitgcm_jax.core import forward_step as fs_mod
from mitgcm_jax.params_io import RunNamelists
from mitgcm_jax.tests import oracle
from mitgcm_jax.tests.test_step_full import CG2D_ITERS, dumped_state, end_state_mismatches, env, step_fn

FULL_V4R4_NAMELISTS = oracle.REPO / "ECCO-v4-Configurations" / "ECCOv4 Release 4" / "namelist"
FULL_ECCO = AdjointConfig(ggl90="frozen", gm_sigma="stable", salt_plume="off", cg2d="passive", visc_fac_in_ad=1.0,
                          seaice="ecco")
NO_SEAM = AdjointConfig(seaice="full")
ALL_ON = (AdjointConfig(ggl90="frozen", gm_sigma="stable", salt_plume="off", cg2d="passive", visc_fac_in_ad=2.0,
                        seaice="no_dynamics"),
          AdjointConfig(ggl90="frozen", gm_sigma="gm_only", salt_plume="off", cg2d="passive", visc_fac_in_ad=2.0,
                        seaice="full"))


def test_ecco_config_full():
    """AdjointConfig.ecco on the full tree == FULL_ECCO; STDOUT.0000 prints the same data.autodiff switch values."""
    e = env()
    assert AdjointConfig.ecco(e.nml) == FULL_ECCO
    assert AdjointConfig.ecco(RunNamelists(FULL_V4R4_NAMELISTS)) == FULL_ECCO
    assert AdjointConfig().seaice == "ecco" and AdjointConfig().is_exact
    lines = (e.rundir / "STDOUT.0000").read_text().splitlines()
    printed = {}
    for i, s in enumerate(lines):
        for key in ("useSEAICEinAdMode", "useGGL90inAdMode", "useSALT_PLUMEinAdMode", "useGMRediInAdMode",
                    "SEAICEuseDYNAMICSswitchInAd", "inAdExact"):
            if f") {key} =" in s or f") {key}=" in s:
                printed[key] = lines[i + 1].split(")")[-1].strip()
    assert printed == {"useSEAICEinAdMode": "F", "useGGL90inAdMode": "F", "useSALT_PLUMEinAdMode": "F",
                       "useGMRediInAdMode": "T", "SEAICEuseDYNAMICSswitchInAd": "F", "inAdExact": "T"}, printed


def _census(adj):
    e = env()
    st = dumped_state(1)
    jp = jax.make_jaxpr(lambda P, g, kLowC, st, x: fs_mod.forward_step(P, g, e.ex, kLowC, st, x, adj=adj)[0])(
        e.P, e.g, e.kLowC, st, e.exf_in[1])
    s = str(jp)
    return s.count(" stop_gradient "), s.count("custom_jvp_call")


def test_seam_census_full():
    """Each switch adds exactly its own seam equations to the traced full step (module docstring)."""
    t = time.time()
    sg0, cj0 = _census(NO_SEAM)
    expect = {AdjointConfig(seaice="ecco"): (0, 1), AdjointConfig(seaice="no_dynamics"): (0, 2),
              AdjointConfig(gm_sigma="stable", seaice="full"): (3, 0),
              AdjointConfig(gm_sigma="gm_only", seaice="full"): (3, 0),
              AdjointConfig(ggl90="frozen", seaice="full"): (4, 0),
              AdjointConfig(cg2d="passive", seaice="full"): (3, 0),
              AdjointConfig(salt_plume="off", seaice="full"): (2, 0),
              AdjointConfig(visc_fac_in_ad=1.0, seaice="full"): (0, 1),
              FULL_ECCO: (12, 2)}
    got = {}
    for adj, want in expect.items():
        sg, cj = _census(adj)
        got[adj] = (sg - sg0, cj - cj0)
    print(f"\nbase (seaice='full', no seam): {sg0} stop_gradient, {cj0} custom_jvp_call; {time.time() - t:.0f} s")
    assert got == expect, {str(k): v for k, v in got.items() if v != expect[k]}


def test_forward_bitwise_all_switches_full():
    """Step 1 with every switch on (two combinations, all sea-ice levels covered) == Fortran at iteration 2."""
    e = env()
    for adj in ALL_ON:
        st1, aux = step_fn(adj)(e.P, e.g, e.kLowC, dumped_state(1), e.exf_in[1])
        assert int(aux["cg2d"]["numIters"]) == CG2D_ITERS[1], adj
        bad = end_state_mismatches(st1, 2)
        assert not bad, (adj, bad)


def test_gradient_driver_default_and_scan_full():
    """make_step defaults to the run's ECCO config; the scan driver (integrate, schedule "step") over two full-tree
    steps == the Fortran at iteration 3 (the forward of the gradient drivers is the model)."""
    e = env()
    try:
        ck.make_step()
    except ValueError as err:
        assert "AdjointConfig.ecco" in str(err)
    else:
        raise AssertionError("make_step() without config or namelists must raise")
    step = ck.make_step(nml=e.nml)
    model = ck.Model(e.P, e.g, e.kLowC, e.ex)
    xs = ck.exf_window(ck.exf_loader_at(e.P, e.g, e.rundir, e.nml, 1), e.nml, 1, 2)
    assert set(xs) == {"bufs", "facs", "myTime", "zt"} and np.shape(xs["zt"]["TDAY"]) == (2,)
    t = time.time()
    st2, _ = jax.jit(lambda m, s, x: ck.integrate(step, m, s, x, schedule="step"))(model, dumped_state(1), xs)
    jax.block_until_ready(st2.f["theta"])
    print(f"\nscan driver, 2 steps incl. compile: {time.time() - t:.0f} s")
    bad = end_state_mismatches(st2, 3)
    assert not bad, bad


# ---------------------------------------------------------------------------------------------------------------------
# effect tests at the DO_OCEANIC_PHYS level
PHYS_IN = ("theta", "salt", "Qnet", "saltPlumeFlux", "HEFF")
PHYS_OUT = ("surfaceForcingT", "surfaceForcingS", "saltPlumeDepth")
CONFIGS = {"ecco": AdjointConfig(seaice="ecco"), "nodyn": AdjointConfig(seaice="no_dynamics"),
           "nodyn_off": AdjointConfig(seaice="no_dynamics", salt_plume="off"),
           "ecco_off": AdjointConfig(seaice="ecco", salt_plume="off")}


def _phys_vjp(adj):
    ex = env().ex

    def fphys(x, f, P, g, kLowC):
        op = fs_mod.do_oceanic_phys(P, g, ex, dict(f, **x), kLowC, adj)
        return {k: op[k] for k in PHYS_OUT}

    @jax.jit
    def vj(x, f, P, g, kLowC, cts):
        out, fn = jax.vjp(lambda x: fphys(x, f, P, g, kLowC), x)
        return out, [fn(ct)[0] for ct in cts]
    return vj


def test_oceanic_phys_seaice_and_salt_plume_seams():
    """Effect tests of the sea-ice level and the full-tree salt-plume seams (module docstring)."""
    e = env()
    L = e.g.layout
    J, I = L.js(1, L.sNy), L.is_(1, L.sNx)
    f = dict(dumped_state(2).f)
    x = {k: f[k] for k in PHYS_IN}
    rng = np.random.default_rng(3)
    wet = np.asarray(e.g.maskC)[:, 0]

    def weight():
        w = np.zeros(L.shape2d)
        w[:, J, I] = rng.standard_normal((L.nTiles, L.sNy, L.sNx))
        return jnp.asarray(w * wet)

    wT, wS, wD = weight(), weight(), weight()
    z = jnp.zeros(L.shape2d)
    cts = [dict(surfaceForcingT=wT, surfaceForcingS=z, saltPlumeDepth=z),
           dict(surfaceForcingT=z, surfaceForcingS=wS, saltPlumeDepth=z),
           dict(surfaceForcingT=z, surfaceForcingS=z, saltPlumeDepth=wD)]
    res, times = {}, {}
    for name, adj in CONFIGS.items():
        t = time.time()
        out, gs = _phys_vjp(adj)(x, f, e.P, e.g, e.kLowC, cts)
        jax.block_until_ready(gs)
        times[name] = round(time.time() - t, 1)
        res[name] = ({k: np.asarray(v) for k, v in out.items()},
                     [{k: np.asarray(v) for k, v in g.items()} for g in gs])
    print("\nVJP times incl. compile (s):", times)
    out0 = res["ecco"][0]
    for name, (out, gs) in res.items():
        for k in PHYS_OUT:
            np.testing.assert_array_equal(out[k], out0[k], err_msg=(name, k))
        for g in gs:
            for k, v in g.items():
                assert np.all(np.isfinite(v)), (name, k)
    gT = {n: r[1][0] for n, r in res.items()}
    gS = {n: r[1][1] for n, r in res.items()}
    gD = {n: r[1][2] for n, r in res.items()}
    nz = lambda a: int(np.count_nonzero(a))  # noqa: E731
    print("nonzero d(sFT)/d theta:", {n: nz(g["theta"]) for n, g in gT.items()},
          " d(sFS)/d salt:", {n: nz(g["salt"]) for n, g in gS.items()},
          " d(depth)/d salt:", {n: nz(g["salt"]) for n, g in gD.items()})
    # sea ice "ecco": identity on Qnet, nothing to theta; "no_dynamics": the thermodynamics is differentiated
    sf = e.P.sf
    np.testing.assert_allclose(gT["ecco"]["Qnet"], -np.asarray(wT) / sf.HeatCapacity_Cp * sf.mass2rUnit, rtol=1e-15,
                               atol=0.0)
    assert nz(gT["ecco"]["Qnet"]) == nz(wT)
    assert not np.any(gT["ecco"]["theta"]) and not np.any(gT["ecco"]["HEFF"])
    assert nz(gT["nodyn"]["theta"]) > 1000 and nz(gT["nodyn"]["HEFF"]) > 1000
    assert nz(gT["nodyn"]["Qnet"][:, J, I] - gT["ecco"]["Qnet"][:, J, I]) > 1000
    # the entry saltPlumeFlux is zeroed (c66g do_oceanic_phys.F:293): no derivative in any configuration
    for g in (gT, gS, gD):
        for n in CONFIGS:
            assert not np.any(g[n]["saltPlumeFlux"]), n
    # salt plume "off": the plume-flux path (SEAICE_GROWTH saltPlumeFlux(salt)) is cut when the sea ice is
    # differentiated; with the "ecco" sea ice it is already dead (bitwise equal); the depth seam cuts theta and salt
    assert nz(gS["nodyn"]["salt"] - gS["nodyn_off"]["salt"]) > 100
    np.testing.assert_array_equal(gS["ecco"]["salt"], gS["ecco_off"]["salt"])
    for n in ("nodyn_off", "ecco_off"):
        assert not np.any(gD[n]["salt"]) and not np.any(gD[n]["theta"]), n
    for n in ("nodyn", "ecco"):
        assert nz(gD[n]["salt"]) > 100 and nz(gD[n]["theta"]) > 100, n
