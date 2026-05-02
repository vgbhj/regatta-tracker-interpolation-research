"""Кусочно-линейная интерполяция (реализация с нуля).

Для каждой точки запроса $t$ находим отрезок $[t_i, t_{i+1}]$,
содержащий $t$, и вычисляем значение по формуле:

$$
x(t) = x_i + \\frac{x_{i+1} - x_i}{t_{i+1} - t_i} \\cdot (t - t_i)
$$

Крайние точки: если $t$ выходит за пределы $[t_0, t_n]$,
используется ближайший крайний отрезок (линейная экстраполяция).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def interpolate(
    t_known: ArrayLike,
    x_known: ArrayLike,
    t_query: ArrayLike,
) -> NDArray[np.float64]:
    """Кусочно-линейная интерполяция по узловым точкам."""
    t_known = np.asarray(t_known, dtype=np.float64)
    x_known = np.asarray(x_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)

    idx = np.searchsorted(t_known, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_known) - 2)

    t_lo = t_known[idx]
    t_hi = t_known[idx + 1]
    x_lo = x_known[idx]
    x_hi = x_known[idx + 1]

    alpha = (t_query - t_lo) / (t_hi - t_lo)
    return x_lo + alpha * (x_hi - x_lo)
