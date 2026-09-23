"""Elementary functions of the Fortran oracle's math libraries, bit for bit, for the device kernels (plan M2.6b-1).

The gfortran oracle (RHEL 8, glibc 2.28, AMD EPYC 7763, -O3 -ffp-contract=off) calls two different exp functions:
  - `glibc_exp`: libm's scalar `exp` (EXF_BULKFORMULAE saturation humidity, SEAICE_SOLVE4TEMP 22 calls per ice point,
    ...): the ifunc-selected FMA variant of the IBM Accurate Mathematical Library exp, NOT correctly rounded. XLA's exp
    differs from it by 1-2 ulp at ~14 % of the arguments. Full-range transcription (every glibc 2.28 path for
    |x| <= ~708.0039; jnp.exp beyond), fused multiply-adds emulated exactly (`fma_emulated`).
  - `exp_libmvec`: libmvec's `_ZGVbN2v_exp` (SSE4.1 kernel, SVML-derived, ~1-4 ulp), which gfortran 11 calls where it
    VECTORISES an EXP loop (seaice_calc_ice_strength.o: PRESS0). A different function from libm's exp; kept distinct.
    Check `nm *.o | grep _ZGV` before deciding which exp a Fortran EXP is.
Both carry a custom_jvp (derivative = exp(x) * x_dot): AD never enters the bit arithmetic.

`Libm` bundles the elementary functions a kernel calls (pkgs/exf_full.py takes one). Measured on this CPU (M2.1-2):
XLA:CPU's log, atan, sin, cos equal glibc 2.28's bit for bit; exp (~14 %) and arccos (~7 %) do not. `DEVICE_LIBM` =
XLA's functions with glibc's exp (the default of the gated kernels); `JNP_LIBM` = XLA's functions only (faster,
ulp-level differences; production option).

History: M2.1-2 (pkgs/exf_full.exp_glibc, only the 1.04 < |x| <= 708 table path, jnp.exp elsewhere), M2.3
(pkgs/seaice_growth.glibc_exp, full range) and M2.4 (pkgs/seaice_dyn.exp_libmvec) each carried a copy; they now import
from here and keep their names as aliases. The full-range glibc_exp replaces exf_full's partial one: its table path is
the same transcription (identical results there; the EXF argument -cvapor_exp/Tsf lies in [-26, -14]), and it is
glibc-exact for small |x| too, where exf's version fell back to jnp.exp.

Gates: tests/test_seaice_growth.py (tables and 13 constants == /lib64/libm-2.28.so, bitwise vs glibc on 1.4e7
arguments), tests/test_exf_full.py::test_exp_glibc, tests/test_seaice_dyn.py (PRESS0 bitwise with exp_libmvec, fails
with jnp.exp). Every result needs the gate XLA flags (no FMA contraction, no algsimp; conftest.py).
"""

from decimal import Decimal, localcontext
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = ["glibc_exp", "fma_emulated", "exp_tables", "glibc_exp_tables_vs_libm", "exp_libmvec", "Libm",
           "JNP_LIBM", "DEVICE_LIBM"]


# ------------------------------------------------------------------------------------------------ glibc 2.28 exp
#
# glibc_exp: bit-for-bit emulation of the exp() the gfortran oracle calls. RHEL8 glibc 2.28, /lib64/libm-2.28.so:
# `exp` -> `__exp_finite` (ifunc at 0x28380) -> on AMD EPYC 7763 (FMA + AVX2) the FMA-compiled IBM Accurate
# Mathematical Library exp at 0x771e0 (sysdeps/ieee754/dbl-64/e_exp.c after the 2.28 slow-path removal; NOT correctly
# rounded: 11 of 1e5 random results differ from the correctly rounded exp). XLA's exp differs from it by 1-2 ulp at
# ~14 % of arguments. Paths (hx = high word of x, n = hx & 0x7fffffff), transcribed from the disassembly, every
# vfmadd/vfnmadd as an (emulated, exact) fused multiply-add:
#     n <= 0x3e2fffff                 (|x| < 2^-28)       1 + x                                  (0x77253)
#     0x3e2fffff < n <= 0x3ecfffff    (|x| < ~3.8e-6)    fma(fma(x, 0.5, 1), x, 1)              (0x77490)
#     0x3ecfffff < n <= 0x3f862e41    (|x| <= ~0.01085)  1 + (x + poly(x))                      (0x775a0)
#     0x3f862e41 < n <= 0x3ff0a2b1    (|x| <= ~1.0397)   accurate-table path                    (0x772b0)
#     0x3ff0a2b1 < n <= 0x40862001    (|x| <= ~708.0039) coar/fine table path (IBM __exp)       (0x774b0)
#     otherwise (near overflow/underflow, inf, nan)      jnp.exp FALLBACK, not bitwise          (0x77368, specials)
# The rounding-mode save/restore (vstmxcsr) is a no-op in round-to-nearest. Tables (.rodata, VA == file offset): coar
# (0xfa4a0) = exp(k*2^-9), k = -178..177, fine (0xf84a0) = exp(m*2^-18), m = 0..511, as (hi, lo), hi = exp rounded to
# 27 significant bits, lo = RN(exp - hi): regenerated below (decimal, 80 digits), bit-identical to the binary; the
# accurate table (0xf7c40) holds 67 + 67 Gal-type nodes x_k ~ +-(k+1)/64 (listed: they cannot be derived) and
# RN(exp(x_k)) (regenerated). `glibc_exp_tables_vs_libm()` compares all of them and the 13 constants with the binary.
# FMA without hardware FMA: TwoProduct (Veltkamp split) a*b = uh + ul, TwoSum c + uh = th + tl, then
# fma = RN(th + RO(tl + ul)) (Boldo & Melquiond 2008; round-to-odd from TwoSum + one integer step of the bit pattern).
# Needs IEEE arithmetic as written (the gate flags: no FMA contraction, no algsimp); bexp and base are read from the
# integer words of y and yy (exact identities) because algsimp folds (c + t) - c -> t. Measured on a compute node with
# 2e6 uniform arguments per range vs Python math.exp (same glibc, same CPU model): 0 mismatches in [-15,-0.075],
# [+-0.0108,+-1.0397], [1,8], [20,30], log-uniform |x| in [0.0108,708] and [1e-12,0.0108] under the gate flags; default
# XLA flags: <= 7 per 2e6 for |x| >= 0.0108, 1.9 % below (all <= 1 ulp). Cost ~28 ms per 1e6 values (jnp.exp 1.2 ms).

_E_THREE51 = float.fromhex("0x1.8p+52")               # 0xd3320
_E_LOG2E = float.fromhex("0x1.71547652b82fep+0")      # 0xde4e0
_E_THREE33 = float.fromhex("0x1.8p+34")               # 0xfbae0
_E_LN_TWO1 = float.fromhex("0x1.62e42fefa3800p-1")    # 0xfbb00
_E_LN_TWO2 = float.fromhex("0x1.ef35793c76730p-45")   # 0xfbaf8
_E_P3 = float.fromhex("0x1.5555555555a0fp-3")         # 0xfbae8
_E_P2 = float.fromhex("0x1.00000000004dcp-1")         # 0xfbaf0
_E_C120 = float.fromhex("0x1.1111111111111p-7")       # 0xc55c8
_E_C24 = float.fromhex("0x1.5555555555555p-5")        # 0xc55d0
_E_C720 = float.fromhex("0x1.6c16c16c16c17p-10")      # 0xc55d8
_E_C6 = float.fromhex("0x1.5555555555555p-3")         # 0xc55e0
_E_HALF = 0.5                                         # 0xad0a8
_E_ONE = 1.0                                          # 0xac8b0
_E_CONST_ADDR = {0xd3320: _E_THREE51, 0xde4e0: _E_LOG2E, 0xfbae0: _E_THREE33, 0xfbb00: _E_LN_TWO1,
                 0xfbaf8: _E_LN_TWO2, 0xfbae8: _E_P3, 0xfbaf0: _E_P2, 0xc55c8: _E_C120, 0xc55d0: _E_C24,
                 0xc55d8: _E_C720, 0xc55e0: _E_C6, 0xad0a8: _E_HALF, 0xac8b0: _E_ONE}
_E_N_TINY, _E_N_SMALL, _E_N_POLY, _E_N_MED, _E_N_MAIN = 0x3e2fffff, 0x3ecfffff, 0x3f862e41, 0x3ff0a2b1, 0x40862001
_E_SPLIT = 134217729.0  # 2^27 + 1
# accurate-table nodes (0xf7c40, even entries): 67 positive, then 67 negative
_E_NODES_HEX = (
    "0x1.ffffffffffc82p-7", "0x1.fffffffffffdbp-6", "0x1.80000000000a0p-5", "0x1.fffffffffff79p-5",
    "0x1.3fffffffffffcp-4", "0x1.8000000000060p-4", "0x1.c000000000061p-4", "0x1.fffffffffffd6p-4",
    "0x1.1ffffffffff58p-3", "0x1.3ffffffffff75p-3", "0x1.5ffffffffff00p-3", "0x1.8000000000020p-3",
    "0x1.9ffffffffa629p-3", "0x1.c00000000000fp-3", "0x1.e00000000007fp-3", "0x1.0000000000072p-2",
    "0x1.0fffffffffecap-2", "0x1.1ffffffffff8fp-2", "0x1.300000000003bp-2", "0x1.4000000000034p-2",
    "0x1.4ffffffffff89p-2", "0x1.5ffffffffffe7p-2", "0x1.6ffffffffff78p-2", "0x1.7ffffffffff65p-2",
    "0x1.8ffffffffffd5p-2", "0x1.9ffffffffff6ep-2", "0x1.affffffffffc3p-2", "0x1.c000000000053p-2",
    "0x1.d00000000004dp-2", "0x1.e000000000096p-2", "0x1.efffffffffefap-2", "0x1.fffffffffffd0p-2",
    "0x1.0800000000002p-1", "0x1.100000000001fp-1", "0x1.17ffffffffff8p-1", "0x1.1fffffffffffap-1",
    "0x1.27fffffffffc4p-1", "0x1.2fffffffffffdp-1", "0x1.380000000001fp-1", "0x1.3ffffffffffd8p-1",
    "0x1.4800000000052p-1", "0x1.4ffffffffffc8p-1", "0x1.5800000000013p-1", "0x1.5ffffffffffbcp-1",
    "0x1.680000000002dp-1", "0x1.7000000000040p-1", "0x1.780000000004fp-1", "0x1.7ffffffffff6fp-1",
    "0x1.87fffffffffe5p-1", "0x1.9000000000035p-1", "0x1.97fffffffffb3p-1", "0x1.a000000000000p-1",
    "0x1.a80000000004ap-1", "0x1.affffffffffedp-1", "0x1.b7ffffffffffbp-1", "0x1.c00000000001dp-1",
    "0x1.c800000000079p-1", "0x1.cffffffffff51p-1", "0x1.d7fffffffff74p-1", "0x1.e000000000011p-1",
    "0x1.e80000000001ep-1", "0x1.effffffffff9ep-1", "0x1.f7fffffffffedp-1", "0x1.0000000000034p+0",
    "0x1.03fffffffffe2p+0", "0x1.07fffffffff4bp+0", "0x1.0bffffffffffdp+0", "-0x1.fffffffffffe4p-7",
    "-0x1.ffffffffffb0bp-6", "-0x1.7ffffffffffa7p-5", "-0x1.ffffffffffea8p-5", "-0x1.3ffffffffffb3p-4",
    "-0x1.7ffffffffffe3p-4", "-0x1.bffffffffff9ap-4", "-0x1.fffffffffff98p-4", "-0x1.1ffffffffffe9p-3",
    "-0x1.3ffffffffffe0p-3", "-0x1.5fffffffff553p-3", "-0x1.7ffffffffff8bp-3", "-0x1.9fffffffffe51p-3",
    "-0x1.bffffffffff6ep-3", "-0x1.dffffffffff7fp-3", "-0x1.fffffffffff7ap-3", "-0x1.0fffffffffffep-2",
    "-0x1.1ffffffffff41p-2", "-0x1.2ffffffffffbap-2", "-0x1.3fffffffffff8p-2", "-0x1.4ffffffffff90p-2",
    "-0x1.5ffffffffffdbp-2", "-0x1.6ffffffffff9ap-2", "-0x1.7ffffffffff9fp-2", "-0x1.8ffffffffffeep-2",
    "-0x1.9fffffffffc4ap-2", "-0x1.affffffffff30p-2", "-0x1.bfffffffffff0p-2", "-0x1.cfffffffffff3p-2",
    "-0x1.dfffffffffff3p-2", "-0x1.effffffffff80p-2", "-0x1.fffffffffffdfp-2", "-0x1.0800000000000p-1",
    "-0x1.0ffffffffffa4p-1", "-0x1.17fffffffff0ap-1", "-0x1.2000000000000p-1", "-0x1.27fffffffffbbp-1",
    "-0x1.2fffffffffe32p-1", "-0x1.37ffffffff042p-1", "-0x1.3ffffffffff77p-1", "-0x1.47fffffffff6bp-1",
    "-0x1.4fffffffffff1p-1", "-0x1.57ffffffffe02p-1", "-0x1.5ffffffffffe5p-1", "-0x1.67fffffffffb0p-1",
    "-0x1.6ffffffffffb2p-1", "-0x1.77fffffffff7fp-1", "-0x1.7ffffffffffe8p-1", "-0x1.87fffffffffc8p-1",
    "-0x1.8fffffffffb30p-1", "-0x1.97fffffffffefp-1", "-0x1.9ffffffffffa7p-1", "-0x1.a7fffffffffdcp-1",
    "-0x1.affffffffff95p-1", "-0x1.b7fffffffffcbp-1", "-0x1.bffffffffff32p-1", "-0x1.c7fffffffff6ap-1",
    "-0x1.cffffffffffb6p-1", "-0x1.d7fffffffffcap-1", "-0x1.dffffffffffcdp-1", "-0x1.e7ffffffffffbp-1",
    "-0x1.effffffffff88p-1", "-0x1.f7fffffffffbbp-1", "-0x1.fffffffffffdbp-1", "-0x1.03fffffffff00p+0",
    "-0x1.07ffffffffe6fp+0", "-0x1.0bfffffffffd6p+0",
)


def _glibc_exp_tables():
    """(coar[712], fine[1024], acc[268]) float64, regenerated from first principles (see the section comment)."""
    import decimal
    from fractions import Fraction

    def dec_exp(q):
        with decimal.localcontext() as ctx:
            ctx.prec = 80
            return Fraction((decimal.Decimal(q.numerator) / decimal.Decimal(q.denominator)).exp())

    def rn_bits(v, nb):
        e = v.numerator.bit_length() - v.denominator.bit_length()
        while Fraction(2) ** e > v:
            e -= 1
        while Fraction(2) ** (e + 1) <= v:
            e += 1
        scale = Fraction(2) ** (nb - 1 - e)
        return Fraction(round(v * scale)) / scale

    def hi_lo(q):
        ex = dec_exp(q)
        hi = rn_bits(ex, 27)
        return float(hi), float(ex - hi)

    coar = np.empty(712)
    for n, k in enumerate(range(-178, 178)):
        coar[2 * n], coar[2 * n + 1] = hi_lo(Fraction(k, 2 ** 9))
    fine = np.empty(1024)
    for m in range(512):
        fine[2 * m], fine[2 * m + 1] = hi_lo(Fraction(m, 2 ** 18))
    acc = np.empty(268)
    for n, h in enumerate(_E_NODES_HEX):
        x = float.fromhex(h)
        acc[2 * n], acc[2 * n + 1] = x, float(dec_exp(Fraction(x)))
    return coar, fine, acc


_E_TABLES = _glibc_exp_tables()


def glibc_exp_tables_vs_libm(path="/lib64/libm-2.28.so"):
    """List of differences between the regenerated tables/constants and the bytes of the libm binary ([] = none)."""
    import struct

    b = open(path, "rb").read()
    bad = [hex(a) for a, v in _E_CONST_ADDR.items() if struct.unpack("<d", b[a:a + 8])[0] != v]
    coar, fine, acc = _E_TABLES
    for name, a, t in (("acc", 0xf7c40, acc), ("fine", 0xf84a0, fine), ("coar", 0xfa4a0, coar)):
        nd = int(np.sum(np.frombuffer(b[a:a + 8 * t.size], "<u8") != t.view(np.uint64)))
        if nd:
            bad.append(f"{name}: {nd} entries differ")
    return bad


def _f2i(v):
    return jax.lax.bitcast_convert_type(v, jnp.int64)


def _i2f(v):
    return jax.lax.bitcast_convert_type(v, jnp.float64)


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_prod(a, b):
    def split(v):
        c = _E_SPLIT * v
        vh = c - (c - v)
        return vh, v - vh

    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


def fma_emulated(a, b, c):
    """RN(a*b + c) in float64 without a hardware FMA (Boldo-Melquiond; exact barring overflow/underflow)."""
    a, b, c = (jnp.asarray(v, jnp.float64) for v in (a, b, c))
    uh, ul = _two_prod(a, b)
    th, tl = _two_sum(c, uh)
    s, e = _two_sum(tl, ul)                                  # round-to-odd of tl + ul
    sb = _f2i(s)
    step = jnp.where(jnp.signbit(e) == jnp.signbit(s), jnp.int64(1), jnp.int64(-1))
    sb = jnp.where((e != 0) & ((sb & 1) == 0), sb + step, sb)
    return th + _i2f(sb)


def _exp_poly(d):
    """d + d^2*(1/2 + d/6) + d^4*(1/24 + d/120 + d^2/720) in the binary's FMA order (0x772b0 / 0x775a0)."""
    a = fma_emulated(d, _E_C120, _E_C24)
    b = fma_emulated(d, _E_C6, _E_HALF)
    d2 = d * d
    a = fma_emulated(d2, _E_C720, a)
    a = a * (d2 * d2)
    b = fma_emulated(b, d2, a)
    return b + d


def _signed_low(v):
    k = _f2i(v) & 0xffffffff
    return (k ^ 0x80000000) - 0x80000000


def _glibc_exp_impl(x):
    coar, fine, acc = (jnp.asarray(t, jnp.float64) for t in _E_TABLES)
    x = jnp.asarray(x, jnp.float64)
    hx = _f2i(x) >> 32
    n = hx & 0x7fffffff
    # main path (0x774b0)
    y = fma_emulated(x, _E_LOG2E, _E_THREE51)
    ky = _signed_low(y)
    bexp = ky.astype(jnp.float64)                            # == y - THREE51 exactly (0x774d4)
    t = fma_emulated(-bexp, _E_LN_TWO1, x)
    yy = _E_THREE33 + t
    kb = _signed_low(yy)
    base = kb.astype(jnp.float64) * 3.814697265625e-06       # == yy - THREE33 exactly (0x774fb), 2^-18
    i = jnp.clip(((kb >> 8) & ~1) + 356, 0, 710)
    j = jnp.clip((kb * 2) & 0x3fe, 0, 1022)
    dl = fma_emulated(-bexp, _E_LN_TWO2, t - base)
    pp = fma_emulated(dl, _E_P3, _E_P2)
    eps = fma_emulated(dl * dl, pp, dl)
    ch, cl = jnp.take(coar, i), jnp.take(coar, i + 1)
    fh, fl = jnp.take(fine, j), jnp.take(fine, j + 1)
    al = ch * fh
    bet = fma_emulated(ch, fl, fh * cl)
    bet = fma_emulated(cl, fl, bet)
    r = fma_emulated(bet, eps, bet)
    r = fma_emulated(eps, al, r)
    res_main = (r + al) * _i2f(((ky + 1023) & 0xfff) << 52)
    # accurate-table path (0x772b0)
    M = (hx & 0xfffff) | 0x100000
    m = M >> jnp.clip(1036 - (n >> 20), 0, 63)
    idx = (m - 1) & ~1
    idx = jnp.clip(jnp.where(hx < 0, idx + 134, idx), 0, 266)
    a0, a1 = jnp.take(acc, idx), jnp.take(acc, idx + 1)
    res_med = fma_emulated(a1, _exp_poly(x - a0), a1)
    # small |x| paths
    res_poly = _exp_poly(x) + _E_ONE
    res_quad = fma_emulated(fma_emulated(x, _E_HALF, _E_ONE), x, _E_ONE)
    res_tiny = _E_ONE + x
    return jnp.where(n <= _E_N_TINY, res_tiny,
                     jnp.where(n <= _E_N_SMALL, res_quad,
                               jnp.where(n <= _E_N_POLY, res_poly,
                                         jnp.where(n <= _E_N_MED, res_med,
                                                   jnp.where(n <= _E_N_MAIN, res_main, jnp.exp(x))))))


@jax.custom_jvp
def glibc_exp(x):
    """exp(x), float64, bit for bit the glibc 2.28 exp of the oracle for |x| <= ~708 (jnp.exp beyond). The derivative is
    glibc_exp(x) * x_dot (AD never enters the bit arithmetic)."""
    return _glibc_exp_impl(x)


@glibc_exp.defjvp
def _glibc_exp_jvp(primals, tangents):
    (x,), (xd,) = primals, tangents
    y = glibc_exp(x)
    return y, y * xd


def exp_tables():
    """(coar[712], fine[1024]) as glibc stores them (libm 0xfa4a0, 0xf84a0): hi, lo pairs of e^((k-178)/512) and
    e^(m/2^18)."""
    return _E_TABLES[0], _E_TABLES[1]


# ------------------------------------------------------------------------------------------------ libmvec exp
#
# glibc libmvec _ZGVbN2v_exp (SSE4.1 kernel), the EXP gfortran calls in the vectorised SEAICE_CALC_ICE_STRENGTH loop.
# Constants read from libmvec-2.28's data block (lea 0xb0c3(%rip) -> 0x12c40: table, +0x2000 InvLn2, +0x2040 Shifter,
# +0x2080 Ln2hi, +0x20c0 Ln2lo, +0x2100 PC1, +0x2140 PC2, +0x2180 PC3, +0x21c0 index mask 0x3ff, +0x2200 abs mask,
# +0x2240 domain range 0x4086232a).
_VEXP_InvLn2 = float.fromhex("0x1.71547652b82fep+10")
_VEXP_Shifter = float.fromhex("0x1.8p+52")
_VEXP_Ln2hi = float.fromhex("0x1.62e42fec00000p-11")
_VEXP_Ln2lo = float.fromhex("0x1.d1cf79abc9e3bp-42")
_VEXP_PC1 = 1.0
_VEXP_PC2 = float.fromhex("0x1.0000001ebfbe0p-1")
_VEXP_PC3 = float.fromhex("0x1.5555555555556p-3")
_VEXP_DOMAIN = 0x4086232A  # |x| above ~708.39: the kernel calls scalar exp for that lane


def _vexp_table():
    with localcontext() as ctx:
        ctx.prec = 40
        ln2 = Decimal(2).ln()
        return np.array([float((ln2 * j / 1024).exp()) for j in range(1024)])  # correctly rounded 2^(j/1024)


_VEXP_TABLE = _vexp_table()


@jax.custom_jvp
def exp_libmvec(x):
    """exp(x) exactly as glibc-2.28 libmvec _ZGVbN2v_exp (SSE4.1 path) computes it, for |x| < 708.39 (outside that
    range the library calls scalar exp: here XLA's exp, not bitwise; not reached by the ice strength, |x| <= cStar).
    Operation order of the kernel: dK = x*InvLn2; dN = roundpd(dK) (nearest-even); dM = Shifter + dK;
    r = (x - dN*Ln2hi) - dN*Ln2lo; p = (PC3*r + PC2)*r + PC1; q = PC1 + r*p; j = bits(dM) & 0x3ff;
    result = bits(T[j]*q) + ((bits(dM) & ~0x3ff) << 42) (integer add = scaling by 2^M). Needs no FMA and no algsimp
    rewriting (conftest XLA flags) to be bitwise."""
    x = jnp.asarray(x, jnp.float64)
    dK = x * _VEXP_InvLn2
    dN = jnp.round(dK)  # roundpd $0: round to nearest even
    dM = _VEXP_Shifter + dK
    r = (x - dN * _VEXP_Ln2hi) - dN * _VEXP_Ln2lo
    p = (_VEXP_PC3 * r + _VEXP_PC2) * r + _VEXP_PC1
    q = _VEXP_PC1 + r * p
    bits = lax.bitcast_convert_type(dM, jnp.int64)
    j = bits & 0x3FF
    Mbits = lax.shift_left(bits & ~jnp.int64(0x3FF), jnp.int64(42))
    Tq = jnp.asarray(_VEXP_TABLE)[j] * q
    res = lax.bitcast_convert_type(lax.bitcast_convert_type(Tq, jnp.int64) + Mbits, jnp.float64)
    hi = lax.shift_right_logical(lax.bitcast_convert_type(x, jnp.int64), jnp.int64(32)) & 0x7FFFFFFF
    return jnp.where(hi > _VEXP_DOMAIN, jnp.exp(x), res)


@exp_libmvec.defjvp
def _exp_libmvec_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = exp_libmvec(x)
    return y, y * dx


# ------------------------------------------------------------------------------------------------ bundles


class Libm(NamedTuple):
    """Elementary functions the device kernels call (the gfortran binary calls glibc for each of them)."""
    exp: Callable
    log: Callable
    atan: Callable
    sin: Callable
    cos: Callable
    acos: Callable


# XLA's functions (log, atan, sin, cos equal glibc's here; exp and arccos do not)
JNP_LIBM = Libm(exp=jnp.exp, log=jnp.log, atan=jnp.arctan, sin=jnp.sin, cos=jnp.cos, acos=jnp.arccos)
# the kernels' default: XLA's functions with glibc's exp (emulated); arccos (zen_fsol_daily, diagnostic) stays XLA's
DEVICE_LIBM = JNP_LIBM._replace(exp=glibc_exp)
