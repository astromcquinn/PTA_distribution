"""Finite-source distributions for a Poisson gravitational-wave background.

All models use rho(u) = u**(-5/2) F(u)/(4 sqrt(pi)), with
integral u rho(u) du = 1 and F(infinity) = 1.  The argument ``N`` is the
effective count N_k in arXiv:2604.19701v1, not a literal source count.
The returned densities are for X = N**(1/3) (h_c,k**2 / mean(h_c,k**2) - 1).

``order=None`` denotes the exact model characteristic function.  Orders
0, 1, 2, and 3 retain the universal stable limit and successive terms in
epsilon=N**(-1/3), including logarithms and exponential cross terms.
These asymptotic densities are not clipped or renormalized and can become
negative outside their useful range.

The sphere, gaussian, and mixture models also have a zero-source atom at
X=-N**(1/3).  Numerical inversions return only their continuous density;
use ``zero_source_probability`` for the atom's mass.  At N>=10 its mass
is at most 6.44e-19.  Fourier cutoff/grid defaults target N>=10 and core
values of X; callers exploring other regimes should check convergence.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math
from typing import Iterable

import numpy as np
from scipy.integrate import quad
from scipy.special import airye, erfc, erfi, expi


model_names = ("paper", "sphere", "gaussian", "mixture")
model_labels = {
    "paper": "Paper population (Section V)",
    "sphere": "Uniform sphere",
    "gaussian": "Gaussian radial profile",
    "mixture": "Two-scale radial mixture",
}
PI = np.pi
EULER_GAMMA = np.euler_gamma
TAIL_AMPLITUDE = 1.0 / (4.0 * np.sqrt(PI))
SPHERE_U0 = 1.0 / (4.0 * PI)
SOURCE_COUNTS = {"paper": np.inf, "sphere": 4.0 * PI / 3.0,
                 "gaussian": 8.0, "mixture": 22.0}


def _validate_model(model: str) -> None:
    if model not in model_names:
        raise ValueError(f"Unknown model {model!r}; choose from {model_names}")


def _validate_N(N: float) -> None:
    if not np.isfinite(N) or N <= 0:
        raise ValueError("N must be finite and positive")


def _restore_scalar(value: np.ndarray, original: object):
    return value.item() if np.ndim(original) == 0 else value


def source_shape(u, model: str = "paper"):
    """Return F(u); its common bright-source limit is unity.

    Source fluxes u must be nonnegative.  F(0)=0 is defined by continuity.
    These are controlled comparison models, not calibrated PTA predictions.
    """
    _validate_model(model)
    arr = np.asarray(u, dtype=float)
    if np.any(arr < 0) or np.any(np.isnan(arr)):
        raise ValueError("Source flux u must be nonnegative")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if model == "paper":
            result = (1.0 + 1.0 / np.sqrt(4.0 * PI * arr)) ** -2
        elif model == "sphere":
            result = (arr >= SPHERE_U0).astype(float)
        elif model == "gaussian":
            result = np.exp(-1.0 / (16.0 * arr))
        else:
            result = (np.exp(-1.0 / (64.0 * arr)) / 3.0
                      + 2.0 * np.exp(-1.0 / (4.0 * arr)) / 3.0)
    result = np.where(arr == 0, 0.0, result)
    return _restore_scalar(result, u)


def source_density(u, model: str = "paper"):
    """Return rho(u) for strictly positive dimensionless source flux u."""
    arr = np.asarray(u, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("rho is defined here only for strictly positive u")
    result = TAIL_AMPLITUDE * arr ** -2.5 * source_shape(arr, model)
    return _restore_scalar(result, u)


# Taylor coefficients obtained exactly before conversion to floating point.
# They stabilize the cancellation-prone paper expression at small q.
_PAPER_TERMS = 20
_k_fraction = [Fraction((-1)**n * (1-n), math.factorial(n))
               for n in range(_PAPER_TERMS + 1)]
_h_fraction = [sum((_k_fraction[j]
                    / (math.factorial(n-j) * (2*(n-j)+1))
                    for j in range(n+1)), Fraction(0))
               for n in range(_PAPER_TERMS + 1)]
_g_fraction = [sum((_k_fraction[j]
                    / ((n-j) * math.factorial(n-j))
                    for j in range(n)), Fraction(0))
               for n in range(_PAPER_TERMS + 1)]
_PAPER_K = np.array([float(v) for v in _k_fraction])
_PAPER_H = np.array([float(v) for v in _h_fraction])
_PAPER_G = np.array([float(v) for v in _g_fraction])


def _psi_paper(q: np.ndarray) -> np.ndarray:
    b = q / (4.0 * PI)
    out = np.empty_like(b)
    small = np.abs(b) < 0.5
    s = b[small]
    if s.size:
        half = np.zeros_like(s)
        integer = np.zeros_like(s)
        log_s = np.log(s)
        for n in range(1, _PAPER_TERMS+1):
            half += 2*np.sqrt(PI) * _PAPER_H[n] * s**(n+0.5)
        for n in range(2, _PAPER_TERMS+1):
            integer += (-_PAPER_K[n] * (EULER_GAMMA+log_s)
                        - _PAPER_G[n]) * s**n
        out[small] = 4*PI * (half+integer)
    b = b[~small]
    if b.size:
        out[~small] = 4*PI * (
            (1+b)*np.exp(-b)*(PI*erfi(np.sqrt(b))-expi(b))
            - 2*np.sqrt(PI*b) + b + np.log(b) + EULER_GAMMA)
    return out


def _psi_sphere(q: np.ndarray) -> np.ndarray:
    out = np.empty_like(q)
    small = np.abs(q * SPHERE_U0) < 0.5
    s = q[small]
    if s.size:
        result = s**1.5 / 3.0
        # Subtract the stable measure's absent low-flux interval exactly.
        for m in range(2, 22):
            result -= (TAIL_AMPLITUDE * (-s)**m
                       * SPHERE_U0**(m-1.5)
                       / (math.factorial(m)*(m-1.5)))
        out[small] = result
    s = q[~small]
    if s.size:
        lam = SOURCE_COUNTS["sphere"]
        out[~small] = (s+s**1.5*erfc(np.sqrt(s*SPHERE_U0))/3.0
                       +(lam-2*s/3.0)*np.exp(-s*SPHERE_U0)-lam)
    return out


def _psi_gaussian_component(q: np.ndarray, c: float) -> np.ndarray:
    s = 2*np.sqrt(c*q)
    lam = 1.0 / (8.0*c**1.5)
    out = np.empty_like(q)
    small = np.abs(s) < 0.75
    v = s[small]
    if v.size:
        result = np.zeros_like(v)
        # The m=2 term cancels the compensating mean term exactly.
        for m in range(3, 26):
            result += ((-1)**m * (1-m) / math.factorial(m)) * v**m
        out[small] = lam*result
    v = s[~small]
    if v.size:
        out[~small] = lam*((1+v)*np.exp(-v)-1) + q[~small]/(4*np.sqrt(c))
    return out


def psi(q, model: str = "paper"):
    """Compensated Laplace exponent integral rho(u)(e^-qu-1+qu) du.

    Accept real nonnegative q or complex q in the closed right half-plane.
    Powers and logarithms have their principal branches.  Small-argument
    convergent series avoid loss of precision in the exact closed forms.
    """
    _validate_model(model)
    arr = np.asarray(q, dtype=complex)
    if np.any(~np.isfinite(arr)) or np.any(arr.real < 0):
        raise ValueError("q must be finite and have nonnegative real part")
    flat = arr.reshape(-1)
    out = np.zeros_like(flat)
    nonzero = flat != 0
    values = flat[nonzero]
    if model == "paper":
        out[nonzero] = _psi_paper(values)
    elif model == "sphere":
        out[nonzero] = _psi_sphere(values)
    elif model == "gaussian":
        out[nonzero] = _psi_gaussian_component(values, 1.0/16.0)
    else:
        out[nonzero] = (_psi_gaussian_component(values, 1.0/64.0)/3.0
                        + 2.0*_psi_gaussian_component(values, 1.0/4.0)/3.0)
    return _restore_scalar(out.reshape(arr.shape), q)


def log_characteristic(t, N: float, model: str = "paper"):
    """Exact log E[exp(i t X)] for the centered, rescaled variable X."""
    _validate_N(N)
    arr = np.asarray(t, dtype=float)
    result = N * psi(-1j*arr*N**(-2.0/3.0), model)
    return _restore_scalar(np.asarray(result), t)


def characteristic(t, N: float, model: str = "paper"):
    """Exact characteristic function, including any zero-source atom."""
    result = np.exp(log_characteristic(t, N, model))
    return _restore_scalar(np.asarray(result), t)


def zero_source_probability(N: float, model: str = "paper") -> float:
    """Mass at X=-N**(1/3); zero for the paper's infinite-count model."""
    _validate_N(N)
    _validate_model(model)
    return float(np.exp(-SOURCE_COUNTS[model]*N))


def correction_symbols(t, N: float, model: str = "paper"):
    """Return A, B, C in log(phi)=z^1.5/3+eps*A+eps²*B+eps³*C.

    For the paper model A and C include log(N), with eps=N^(-1/3).
    """
    _validate_N(N)
    _validate_model(model)
    arr = np.asarray(t, dtype=float)
    z = -1j*arr
    if model == "paper":
        log_z = np.zeros_like(z)
        np.log(z/(4*PI*N**(2.0/3.0)), out=log_z, where=(z != 0))
        A = z*z*(log_z+EULER_GAMMA-0.5)/(8*PI)
        B = -z**2.5/(10*PI)
        C = -z**3*(log_z+EULER_GAMMA-4.0/3.0)/(48*PI*PI)
    elif model == "sphere":
        A, B, C = -z*z/(8*PI), np.zeros_like(z), z**3/(288*PI*PI)
    elif model == "gaussian":
        A, B, C = -z*z/16, z**2.5/120, -z**3/1152
    else:
        A, B, C = -3*z*z/32, 11*z**2.5/480, -43*z**3/9216
    return tuple(_restore_scalar(np.asarray(item), t) for item in (A, B, C))


def asymptotic_characteristic(t, N: float, model: str = "paper", order: int = 3):
    """Characteristic-function expansion through order eps**order.

    This expands exp(log(phi)); all square and cubic cross terms are kept.
    It is not obtained by exponentiating a truncated cumulant at large t.
    """
    if order not in (0, 1, 2, 3):
        raise ValueError("order must be one of 0, 1, 2, 3")
    _validate_N(N)
    _validate_model(model)
    arr = np.asarray(t, dtype=float)
    z = -1j*arr
    eps = N**(-1.0/3.0)
    A, B, C = correction_symbols(arr, N, model)
    factor = np.ones_like(z)
    if order >= 1:
        factor += eps*A
    if order >= 2:
        factor += eps**2*(B+A*A/2)
    if order >= 3:
        factor += eps**3*(C+A*B+A**3/6)
    result = np.exp(z**1.5/3.0)*factor
    return _restore_scalar(result, t)


def stable_density(x):
    """Universal reflected map-Airy density, using scaled Airy functions."""
    arr = np.asarray(x, dtype=float)
    ai, aip, _, _ = airye(arr*arr)
    result = -2*np.exp(2*(arr**3-np.abs(arr)**3)/3)*(arr*ai+aip)
    return _restore_scalar(result, x)


@lru_cache(maxsize=8)
def _fourier_grid(tmax: float, dt: float):
    if not (np.isfinite(tmax) and np.isfinite(dt) and tmax > 0 and dt > 0):
        raise ValueError("tmax and dt must be finite and positive")
    steps = max(2, int(np.ceil(tmax/dt)))
    steps += steps % 2
    t = np.linspace(0.0, tmax, steps+1)
    weights = np.full(steps+1, 2.0)
    weights[1:-1:2] = 4.0
    weights[[0, -1]] = 1.0
    weights *= (tmax/steps)/(3*PI)
    t.flags.writeable = False
    weights.flags.writeable = False
    return t, weights


@lru_cache(maxsize=2)
def _phase_grid(x_bytes: bytes, count: int, tmax: float, dt: float):
    xs = np.frombuffer(x_bytes, dtype=np.float64, count=count)
    t, _ = _fourier_grid(tmax, dt)
    phase = np.exp(-1j*np.outer(xs, t))
    phase.flags.writeable = False
    return phase


def clear_fourier_cache() -> None:
    """Release the two cached phase matrices after large plotting batches."""
    _phase_grid.cache_clear()
    _fourier_grid.cache_clear()


def density_curves(xs, N: float, model: str = "paper",
                   orders: Iterable[int | None] = (None, 0, 1, 2, 3),
                   tmax: float = 160.0, dt: float = 0.02):
    """Return a dictionary of rescaled PDF curves, reusing Fourier grids.

    Dictionary keys are the requested orders; None denotes the exact model's
    continuous density.  The underlying Simpson integral is applied to the
    difference from the analytically known stable density.  This removes
    its common nonanalytic t**(3/2) endpoint term, improving grid accuracy.
    At most two phase matrices are cached.  Smaller dt and larger tmax can
    be used for an independent convergence check.  No values are clipped.
    """
    _validate_N(N)
    _validate_model(model)
    requested = tuple(dict.fromkeys(orders))
    if any(order not in (None, 0, 1, 2, 3) for order in requested):
        raise ValueError("orders must contain only None, 0, 1, 2, 3")
    arr = np.asarray(xs, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError("xs must be finite")
    flat = np.ascontiguousarray(arr.reshape(-1))
    base = np.asarray(stable_density(flat))
    result = {}
    if 0 in requested:
        result[0] = base.reshape(arr.shape).copy()
    other = [order for order in requested if order != 0]
    if other:
        t, weights = _fourier_grid(float(tmax), float(dt))
        phase = _phase_grid(flat.tobytes(), flat.size, float(tmax), float(dt))
        stable_cf = np.exp((-1j*t)**1.5/3)
        columns = []
        for order in other:
            if order is None:
                cf = characteristic(t, N, model)
                atom = zero_source_probability(N, model)
                if atom:
                    cf = cf-atom*np.exp(-1j*t*N**(1.0/3.0))
            else:
                cf = asymptotic_characteristic(t, N, model, order)
            columns.append(weights*(cf-stable_cf))
        integrals = (phase @ np.column_stack(columns)).real
        for j, order in enumerate(other):
            result[order] = (base+integrals[:, j]).reshape(arr.shape)
    return {order: _restore_scalar(result[order], xs) for order in requested}


def density_grid(xs, N: float, model: str = "paper", order: int | None = None,
                 tmax: float = 160.0, dt: float = 0.02):
    """Return one exact or asymptotic PDF curve; see density_curves."""
    return density_curves(xs, N, model, (order,), tmax, dt)[order]


def density_quad(x: float, N: float, model: str = "paper",
                 order: int | None = None, tmax: float = 200.0,
                 epsabs: float = 2e-10) -> float:
    """Reference adaptive Fourier inversion at one x.

    This independently integrates the full Fourier integrand, rather than
    the stable-subtracted Simpson rule used for plotting.  Increase tmax
    when exploring very small N or far-tail x.
    """
    _validate_N(N)
    _validate_model(model)
    if order not in (None, 0, 1, 2, 3):
        raise ValueError("order must be None, 0, 1, 2, or 3")
    atom = zero_source_probability(N, model) if order is None else 0.0

    def integrand(t):
        if order is None:
            cf = characteristic(t, N, model)
            if atom:
                cf -= atom*np.exp(-1j*t*N**(1.0/3.0))
        else:
            cf = asymptotic_characteristic(t, N, model, order)
        return float((np.exp(-1j*t*x)*cf).real)

    edges = sorted(set([0.0] + [v for v in (1, 5, 15, 40, 100) if v < tmax]
                       + [float(tmax)]))
    value = sum(quad(integrand, a, b, epsabs=epsabs, epsrel=epsabs,
                     limit=250)[0] for a, b in zip(edges[:-1], edges[1:]))
    return float(value/PI)


__all__ = ["model_names", "model_labels", "source_shape", "source_density",
           "psi", "log_characteristic", "characteristic", "correction_symbols",
           "asymptotic_characteristic", "stable_density", "density_grid",
           "density_curves", "density_quad", "zero_source_probability",
           "clear_fourier_cache"]
