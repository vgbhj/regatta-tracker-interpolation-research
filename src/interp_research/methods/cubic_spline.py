"""Натуральный кубический сплайн (реализация с нуля + scipy cross-check).

Натуральный кубический сплайн — кусочный полином третьей степени,
непрерывный вместе с первой и второй производными во всех внутренних узлах.
Граничные условия «свободные концы»: $M_0 = M_{n-1} = 0$,
где $M_i = S''(t_i)$ — вторые производные в узлах.

На каждом отрезке $[t_i, t_{i+1}]$ сплайн имеет вид:

$$
S_i(t) = \\frac{M_i (t_{i+1} - t)^3 + M_{i+1} (t - t_i)^3}{6 h_i}
       + \\left(\\frac{x_i}{h_i} - \\frac{M_i h_i}{6}\\right)(t_{i+1} - t)
       + \\left(\\frac{x_{i+1}}{h_i} - \\frac{M_{i+1} h_i}{6}\\right)(t - t_i)
$$

где $h_i = t_{i+1} - t_i$.

Моменты $M_i$ находятся из трёхдиагональной системы,
решаемой методом прогонки (модуль `interp_research.numeric`).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

from interp_research.numeric import tridiagonal_solve


def _compute_moments(
    t: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Вычисление моментов M_i натурального сплайна."""
    n = len(t)
    h = np.diff(t)

    # Трёхдиагональная система для внутренних моментов M[1..n-2]
    n_inner = n - 2
    diag_a = np.empty(n_inner)
    diag_b = np.empty(n_inner)
    diag_c = np.empty(n_inner)
    rhs = np.empty(n_inner)

    for i in range(n_inner):
        k = i + 1  # индекс в исходном массиве
        diag_a[i] = h[k - 1]
        diag_b[i] = 2.0 * (h[k - 1] + h[k])
        diag_c[i] = h[k]
        rhs[i] = 6.0 * ((x[k + 1] - x[k]) / h[k] - (x[k] - x[k - 1]) / h[k - 1])

    m_inner = tridiagonal_solve(diag_a, diag_b, diag_c, rhs)

    M = np.zeros(n, dtype=np.float64)
    M[1:-1] = m_inner
    return M


def interpolate(
    t_known: ArrayLike,
    x_known: ArrayLike,
    t_query: ArrayLike,
) -> NDArray[np.float64]:
    """Натуральный кубический сплайн (собственная реализация)."""
    t_known = np.asarray(t_known, dtype=np.float64)
    x_known = np.asarray(x_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)

    M = _compute_moments(t_known, x_known)
    h = np.diff(t_known)

    idx = np.searchsorted(t_known, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_known) - 2)

    hi = h[idx]
    dt_right = t_known[idx + 1] - t_query  # (t_{i+1} - t)
    dt_left = t_query - t_known[idx]        # (t - t_i)

    result = (
        (M[idx] * dt_right**3 + M[idx + 1] * dt_left**3) / (6.0 * hi)
        + (x_known[idx] / hi - M[idx] * hi / 6.0) * dt_right
        + (x_known[idx + 1] / hi - M[idx + 1] * hi / 6.0) * dt_left
    )
    return result


def interpolate_scipy(
    t_known: ArrayLike,
    x_known: ArrayLike,
    t_query: ArrayLike,
) -> NDArray[np.float64]:
    """Натуральный кубический сплайн (scipy, для cross-check)."""
    t_known = np.asarray(t_known, dtype=np.float64)
    x_known = np.asarray(x_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)

    cs = CubicSpline(t_known, x_known, bc_type="natural")
    return cs(t_query)
