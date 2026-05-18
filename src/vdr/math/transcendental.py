"""
vdr.math.transcendental — Exact transcendental arithmetic.

Named constants available both as Q335 precomputed values (100-digit precision)
and as FnRemainder functions (arbitrary depth).

    from vdr.math.transcendental import PI, E, SQRT2, ZETA3
    from vdr.math.transcendental import PI_FN, sqrt_newton, borwein_zeta

    PI                        # VDR(2198864..., 2^335) ready to use
    resolve(PI_FN, depth=10)  # arbitrary precision via functional remainder
    borwein_zeta(5, n=210)    # zeta(5) to 100+ digits

Q335 basis: 22 constants as [p, 2^335, 0]. Minimal universal exponent:
at n=334 Catalan's G fails at position 101. At n=335 all 22 pass.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Callable

from vdr.core import VDR, Remainder
from vdr.fn import FnRemainder, make_newton_fn, make_series_fn

__all__ = [
    # Q335 constants
    "PI", "E", "LN2", "SQRT2", "PHI",
    "PI_SQ", "PI_CU", "PI_QU", "E_PI",
    "LN2_SQ", "LN2_QU",
    "LN3", "LN5", "LN10",
    "SQRT3", "SQRT5", "SQRT7",
    "ZETA2", "ZETA3", "ZETA5",
    "LI4_HALF", "CATALAN",
    # Functional remainder versions
    "PI_FN", "E_FN", "LN2_FN",
    "SQRT2_FN", "SQRT3_FN", "SQRT5_FN", "SQRT7_FN",
    "ZETA3_FN", "ZETA5_FN", "CATALAN_FN",
    # Functions
    "sqrt_newton",
    "exp_series", "sin_series", "cos_series",
    "ln_series", "arctan_series", "arcsin_series",
    "pi_machin",
    "borwein_zeta", "borwein_eta", "borwein_coefficients",
    "elliptic_k", "elliptic_e",
    "hypergeometric_2f1",
    "clausen_2", "clausen_3",
    # Basis
    "Q335_DENOM",
]


# ---------------------------------------------------------------------------
# Q335 denominator
# ---------------------------------------------------------------------------

Q335_DENOM = 2 ** 335


# ---------------------------------------------------------------------------
# Q335 precomputed constants (100-digit precision)
# All verified against mpmath at 100 decimal digits by string comparison.
# ---------------------------------------------------------------------------

PI = VDR(
    219886425873192351011826597043241066194671831922348816817425823313156938749437718695100428743935254314,
    Q335_DENOM
)

E = VDR(
    190258044782769202588129925521314757831284456026137946619894798297742927086075833929023100244479638112,
    Q335_DENOM
)

LN2 = VDR(
    48514773537953331556699584584828624926234404478840896710102416707062925979128257345653169777835518667,
    Q335_DENOM
)

SQRT2 = VDR(
    98983668457552556369912251393641781543489938395170417531517516177599375784349358848602281494773475506,
    Q335_DENOM
)

PHI = VDR(
    113249472467736168604496750010842101773570690275806888818880481552730738076053012711350611809151189412,
    Q335_DENOM
)

PI_SQ = VDR(
    690793580147337726804277647484346770338921354138994508002872352435529393755796399964695383625668575976,
    Q335_DENOM
)

PI_CU = VDR(
    2170192036537868242782341740347526814570179266657980009466902575842216583318830559778528157446001240080,
    Q335_DENOM
)

PI_QU = VDR(
    6817859358866439017122533696289105276559442547141782759070845808348090383725467935335488832685124730326,
    Q335_DENOM
)

E_PI = VDR(
    1619663895456875537109657111692739211478931048048038025064408441944407978010684548404551575192727763397,
    Q335_DENOM
)

LN2_SQ = VDR(
    33627878493336594620147550513544307026418133133387860405002917547734923457242850195041264715469792904,
    Q335_DENOM
)

LN2_QU = VDR(
    16156615573798633249523359538243246008210686364818713716124360467773572086286920210666548222826014086,
    Q335_DENOM
)

LN3 = VDR(
    76894096788635086096158790585166115140009649181250777410832538562395270797691729322128736655820466233,
    Q335_DENOM
)

LN5 = VDR(
    112647815694871799155432631259623524245586803429977893615314774516410370135500048646041895614334987799,
    Q335_DENOM
)

LN10 = VDR(
    161162589232825130712132215844452149171821207908818790325417191223473296114628305991695065392170506466,
    Q335_DENOM
)

SQRT3 = VDR(
    121229740294912895234576661752159696642961157181742464717663915473198765686797807393142352785809790154,
    Q335_DENOM
)

SQRT5 = VDR(
    156506921742415955629073428753920319855839958763030979672136303700342980177725995879548801953564656455,
    Q335_DENOM
)

SQRT7 = VDR(
    185181487127092153770432076884133468631121666203542492409943031514633653137939942068870811445311050320,
    Q335_DENOM
)

ZETA2 = VDR(
    115132263357889621134046274580724461723153559023165751333812058739254898959299399994115897270944762663,
    Q335_DENOM
)

ZETA3 = VDR(
    84134394645319852071522700710261177454128732241134555234516209978359598548186272768450592529361881680,
    Q335_DENOM
)

ZETA5 = VDR(
    72576671487518636549061590533542457287978428544763113598602740326685645428855657003519154452098433211,
    Q335_DENOM
)

LI4_HALF = VDR(
    36219406486600619537883622883703292936779255100080725994962678520983767482244581297270363585520219319,
    Q335_DENOM
)

CATALAN = VDR(
    64110285111693582641294563817927086726382757371148180987419195376360958765615024299223500526530512841,
    Q335_DENOM
)


# ---------------------------------------------------------------------------
# Series functions
# ---------------------------------------------------------------------------

def sqrt_newton(n, depth=10, start=None):
    """
    Newton-Raphson square root of n.

    x_{k+1} = (x_k + n/x_k) / 2

    I: n (VDR or int), depth (iterations), optional start
    O: exact rational approximation. Depth 10 = >100 correct digits.
       Quadratic convergence: digits double per step.

        sqrt_newton(VDR(2), depth=10)  # >100 digits of sqrt(2)
    """
    if isinstance(n, int):
        n = VDR(n)
    if start is None:
        # rough integer start
        frac = n.to_fraction()
        from math import isqrt
        if frac.denominator == 1 and frac.numerator > 0:
            s = isqrt(frac.numerator)
            start = VDR(max(s, 1))
        else:
            start = VDR(1)

    x = start
    target = n
    for _ in range(depth):
        x = (x + target / x) / VDR(2)
    return x


def exp_series(x, depth=16):
    """
    Taylor series for exp(x) = sum x^n / n!

    I: x (VDR), depth (number of terms)
    O: exact rational partial sum

        exp_series(VDR(1), 20)  # e to ~18 digits
    """
    if isinstance(x, int):
        x = VDR(x)
    total = VDR(1)
    term = VDR(1)
    for k in range(1, depth + 1):
        term = term * x / VDR(k)
        total = total + term
    return total


def sin_series(x, depth=16):
    """
    Taylor series for sin(x) = x - x^3/3! + x^5/5! - ...

    I: x (VDR), depth (number of terms)
    O: exact rational partial sum
    """
    if isinstance(x, int):
        x = VDR(x)
    total = VDR(0)
    term = x
    total = total + term
    for k in range(1, depth):
        term = -term * x * x / (VDR(2 * k) * VDR(2 * k + 1))
        total = total + term
    return total


def cos_series(x, depth=16):
    """
    Taylor series for cos(x) = 1 - x^2/2! + x^4/4! - ...

    I: x (VDR), depth (number of terms)
    O: exact rational partial sum
    """
    if isinstance(x, int):
        x = VDR(x)
    total = VDR(1)
    term = VDR(1)
    for k in range(1, depth):
        term = -term * x * x / (VDR(2 * k - 1) * VDR(2 * k))
        total = total + term
    return total


def ln_series(x, depth=16):
    """
    Natural logarithm via ln(1+u) = u - u^2/2 + u^3/3 - ...
    where u = (x-1)/x for faster convergence when x > 1,
    or u = x-1 for x near 1.

    I: x (VDR, positive), depth
    O: exact rational partial sum of ln(x)
    """
    if isinstance(x, int):
        x = VDR(x)

    # for x close to 1, use direct series
    # for x > 2, use ln(x) = ln(x/2^k) + k*ln(2) via reduction
    # simple version: ln(1 + u) where u = x - 1
    # converges for -1 < u <= 1

    frac = x.to_fraction()
    if frac <= 0:
        raise ValueError("ln requires positive argument")

    # reduce: find k such that x / 2^k is in (1/2, 2)
    # then ln(x) = k * ln(2) + ln(x/2^k)
    # For the library, use direct series on (x-1)/x = 1 - 1/x
    # ln(x) = -ln(1/x), and use arctanh form:
    # ln(x) = 2 * arctanh((x-1)/(x+1))

    u = (x - VDR(1)) / (x + VDR(1))
    # ln(x) = 2 * (u + u^3/3 + u^5/5 + ...)
    total = VDR(0)
    u_power = u
    for k in range(depth):
        n = 2 * k + 1
        total = total + u_power / VDR(n)
        u_power = u_power * u * u
    return VDR(2) * total


def arctan_series(x, depth=16):
    """
    Taylor series for arctan(x) = x - x^3/3 + x^5/5 - ...

    I: x (VDR, |x| <= 1), depth
    O: exact rational partial sum
    """
    if isinstance(x, int):
        x = VDR(x)
    total = VDR(0)
    x_power = x
    for k in range(depth):
        n = 2 * k + 1
        if k % 2 == 0:
            total = total + x_power / VDR(n)
        else:
            total = total - x_power / VDR(n)
        x_power = x_power * x * x
    return total


def arcsin_series(x, depth=16):
    """
    Taylor series for arcsin(x) using central binomial coefficients.

    arcsin(x) = sum_{n=0}^{depth} C(2n,n) * x^(2n+1) / (4^n * (2n+1))

    I: x (VDR, |x| < 1), depth
    O: exact rational partial sum
    """
    if isinstance(x, int):
        x = VDR(x)
    total = VDR(0)
    # term_n = C(2n,n) * x^(2n+1) / (4^n * (2n+1))
    # recurrence: term_{n+1}/term_n = (2n+1)^2 * x^2 / ((2n+2)*(2n+3))
    # but simpler to compute directly

    from vdr.math.combinatorics import binom
    x_sq = x * x
    x_power = x  # x^(2n+1) starts at x
    four_n = VDR(1)  # 4^n starts at 1

    for n in range(depth):
        coeff = binom(2 * n, n)
        denom = four_n * VDR(2 * n + 1)
        total = total + coeff * x_power / denom
        x_power = x_power * x_sq
        four_n = four_n * VDR(4)

    return total


def pi_machin(terms=160):
    """
    Compute pi via Machin's formula:
        pi/4 = 4*arctan(1/5) - arctan(1/239)

    Geometric convergence ~1.4 bits/term. 160 terms = ~224 bits > 67 digits.

    I: number of arctan series terms
    O: exact rational approximation of pi
    """
    at_1_5 = arctan_series(VDR(1, 5), depth=terms)
    at_1_239 = arctan_series(VDR(1, 239), depth=terms)
    return VDR(4) * (VDR(4) * at_1_5 - at_1_239)


# ---------------------------------------------------------------------------
# Borwein acceleration for zeta
# ---------------------------------------------------------------------------

def borwein_coefficients(n):
    """
    Compute Borwein acceleration coefficients d_k for k=0..n.

    d_k = n * sum_{i=0}^{k} (n+i-1)! * 4^i / ((n-i)! * (2i)!)

    All coefficients are exact integers (rational with denominator 1).

    I: parameter n (typically 210 for 100 digits)
    O: list of VDR [d_0, d_1, ..., d_n]
    """
    # precompute factorials
    max_fact = 2 * n + 1
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i - 1] * i

    d = []
    for k in range(n + 1):
        total = 0
        for i in range(k + 1):
            num = fact[n + i - 1] if (n + i - 1) >= 0 else 1
            if n + i - 1 < 0:
                num = 0
                if i == 0 and n == 0:
                    num = 1
            den = fact[n - i] * fact[2 * i]
            four_i = 4 ** i
            total += n * num * four_i // den
        d.append(VDR(total))

    return d


def borwein_eta(s, n=210):
    """
    Dirichlet eta function via Borwein acceleration.

    eta(s) = -1/d_n * sum_{k=0}^{n-1} (-1)^k * (d_k - d_n) / (k+1)^s

    Geometric convergence 3^(-n). n=210 gives ~100 digits for any s.

    I: integer s >= 1, acceleration parameter n
    O: exact VDR rational

        borwein_eta(2, 210)  # eta(2) = pi^2/12
    """
    d = borwein_coefficients(n)
    d_n = d[n]

    total = VDR(0)
    for k in range(n):
        diff = d[k] - d_n
        denom = VDR((k + 1) ** s)
        term = diff / denom
        if k % 2 == 0:
            total = total + term
        else:
            total = total - term

    return VDR(-1) * total / d_n


def borwein_zeta(s, n=210):
    """
    Riemann zeta function via Borwein acceleration.

    zeta(s) = eta(s) / (1 - 2^(1-s))

    I: integer s >= 2, acceleration parameter n
    O: exact VDR rational, ~100 digits at n=210

        borwein_zeta(3, 210)  # Apery's constant
        borwein_zeta(5, 210)  # zeta(5) to 100+ digits
    """
    eta = borwein_eta(s, n)

    # 1 - 2^(1-s) = 1 - 1/2^(s-1) = (2^(s-1) - 1) / 2^(s-1)
    two_s_minus_1 = 2 ** (s - 1)
    factor_denom = VDR(two_s_minus_1 - 1, two_s_minus_1)

    return eta / factor_denom


# ---------------------------------------------------------------------------
# Elliptic integrals
# ---------------------------------------------------------------------------

def hypergeometric_2f1(a, b, c, z, terms=500):
    """
    Gauss hypergeometric function 2F1(a, b; c; z).

    2F1 = sum_{n=0}^{terms} (a)_n (b)_n / ((c)_n * n!) * z^n

    where (x)_n = x(x+1)...(x+n-1) is the rising factorial.

    All coefficients rational when a, b, c, z are rational.

    I: parameters a, b, c, z (VDR), number of terms
    O: exact rational partial sum
    """
    total = VDR(1)
    term = VDR(1)

    for n in range(1, terms + 1):
        term = term * (a + VDR(n - 1)) * (b + VDR(n - 1))
        term = term / ((c + VDR(n - 1)) * VDR(n))
        term = term * z
        total = total + term
        # early termination if term is zero
        if term == VDR(0):
            break

    return total


def elliptic_k(k_sq, terms=500):
    """
    Complete elliptic integral of the first kind K(k).

    K(k) = (pi/2) * 2F1(1/2, 1/2; 1; k^2)

    I: k^2 as VDR (must be < 1), number of hypergeometric terms
    O: exact rational times Q335 pi/2

    Every series coefficient is rational. Product with Q335 pi/2
    gives a standard VDR closed object.

        elliptic_k(VDR(1, 2), 500)  # K(1/sqrt(2)), k^2 = 1/2
    """
    hyper = hypergeometric_2f1(VDR(1, 2), VDR(1, 2), VDR(1), k_sq, terms)
    # pi/2 from Q335
    pi_half = PI / VDR(2)
    return pi_half * hyper


def elliptic_e(k_sq, terms=500):
    """
    Complete elliptic integral of the second kind E(k).

    E(k) = (pi/2) * 2F1(-1/2, 1/2; 1; k^2)

    I: k^2 as VDR (must be < 1), number of terms
    O: exact rational times Q335 pi/2
    """
    hyper = hypergeometric_2f1(VDR(-1, 2), VDR(1, 2), VDR(1), k_sq, terms)
    pi_half = PI / VDR(2)
    return pi_half * hyper


def clausen_2(x, terms=210):
    """
    Clausen function Cl_2(x) = -integral_0^x ln|2 sin(t/2)| dt
                              = sum_{n=1}^{terms} sin(nx) / n^2

    I: x as VDR (rational multiple of pi), terms
    O: exact rational (requires sin evaluation via series)

    For x = pi/3: Cl_2(pi/3) relates to zeta(2) and L-functions.
    """
    total = VDR(0)
    for n in range(1, terms + 1):
        sin_nx = sin_series(VDR(n) * x, depth=20)
        total = total + sin_nx / VDR(n * n)
    return total


def clausen_3(x, terms=210):
    """
    Clausen function Cl_3(x) = sum_{n=1}^{terms} cos(nx) / n^3.

    I: x as VDR, terms
    O: exact rational
    """
    total = VDR(0)
    for n in range(1, terms + 1):
        cos_nx = cos_series(VDR(n) * x, depth=20)
        total = total + cos_nx / VDR(n * n * n)
    return total


# ---------------------------------------------------------------------------
# Functional remainder versions of constants
# ---------------------------------------------------------------------------

def _make_pi_fn():
    def pi_fn(depth):
        # more terms = more digits. ~1.4 bits/term
        terms = max(20, depth * 25)
        return pi_machin(terms)
    return FnRemainder(pi_fn, name="pi")


def _make_e_fn():
    def e_fn(depth):
        terms = max(20, depth * 8)
        return exp_series(VDR(1), terms)
    return FnRemainder(e_fn, name="e")


def _make_ln2_fn():
    def ln2_fn(depth):
        terms = max(20, depth * 25)
        return ln_series(VDR(2), terms)
    return FnRemainder(ln2_fn, name="ln2")


def _make_sqrt_fn(n, name):
    target = VDR(n)
    def sqrt_fn(depth):
        return sqrt_newton(target, depth=max(1, depth))
    return FnRemainder(sqrt_fn, name=name)


def _make_zeta_fn(s, name):
    def zeta_fn(depth):
        n = max(10, depth * 25)
        return borwein_zeta(s, n)
    return FnRemainder(zeta_fn, name=name)


def _make_catalan_fn():
    def catalan_fn(depth):
        n = max(10, depth * 25)
        return borwein_eta(2, n) / VDR(2)  # G = beta(2) via eta
        # more precisely: G = sum (-1)^n / (2n+1)^2
        # use direct series
    # direct series is simpler for Catalan
    def catalan_direct(depth):
        terms = max(20, depth * 25)
        total = VDR(0)
        for k in range(terms):
            denom = VDR((2 * k + 1) * (2 * k + 1))
            if k % 2 == 0:
                total = total + VDR(1) / denom
            else:
                total = total - VDR(1) / denom
        return total
    return FnRemainder(catalan_direct, name="catalan_G")


PI_FN = _make_pi_fn()
E_FN = _make_e_fn()
LN2_FN = _make_ln2_fn()
SQRT2_FN = _make_sqrt_fn(2, "sqrt2")
SQRT3_FN = _make_sqrt_fn(3, "sqrt3")
SQRT5_FN = _make_sqrt_fn(5, "sqrt5")
SQRT7_FN = _make_sqrt_fn(7, "sqrt7")
ZETA3_FN = _make_zeta_fn(3, "zeta3")
ZETA5_FN = _make_zeta_fn(5, "zeta5")
CATALAN_FN = _make_catalan_fn()
