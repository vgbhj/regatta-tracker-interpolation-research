"""Кубическая эрмитова интерполяция с учётом динамики движения.

Метод учитывает динамику движения через оценку касательных в узлах.
Применительно к парусной траектории это означает, что между узлами
интерполяция следует не просто гладкой кривой, а кривой, согласованной
с направлением мгновенного движения.

Касательные $dx/dt$ в узлах оцениваются конечными разностями:

- Центральные разности для внутренних узлов:
  $$\\dot{x}_i = \\frac{x_{i+1} - x_{i-1}}{t_{i+1} - t_{i-1}}$$

- Односторонние разности на концах:
  $$\\dot{x}_0 = \\frac{x_1 - x_0}{t_1 - t_0}, \\quad
    \\dot{x}_{n-1} = \\frac{x_{n-1} - x_{n-2}}{t_{n-1} - t_{n-2}}$$

Затем строится кубический эрмитов сплайн $S(t)$ такой, что
$S(t_i) = x_i$ и $S'(t_i) = \\dot{x}_i$.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicHermiteSpline


def _estimate_derivatives(
    t: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Оценка производных dx/dt через конечные разности."""
    n = len(t)
    dxdt = np.empty(n, dtype=np.float64)

    # Односторонняя на левом конце
    dxdt[0] = (x[1] - x[0]) / (t[1] - t[0])

    # Центральные разности для внутренних узлов
    dxdt[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])

    # Односторонняя на правом конце
    dxdt[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])

    return dxdt


def interpolate(
    t_known: ArrayLike,
    x_known: ArrayLike,
    t_query: ArrayLike,
) -> NDArray[np.float64]:
    """Кубическая эрмитова интерполяция с касательными из конечных разностей."""
    t_known = np.asarray(t_known, dtype=np.float64)
    x_known = np.asarray(x_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)

    dxdt = _estimate_derivatives(t_known, x_known)
    hermite = CubicHermiteSpline(t_known, x_known, dxdt)
    return np.asarray(hermite(t_query), dtype=np.float64)
