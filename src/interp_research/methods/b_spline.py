"""Кубический B-сплайн (обёртка над scipy).

Реализация на scipy используется в исследовательской части как эталонная;
в продуктовом коде на TypeScript та же логика реализована вручную
с использованием базисных функций B-сплайна.

B-сплайн степени $k$ представляется в виде:

$$
S(t) = \\sum_{i=0}^{n-1} c_i B_{i,k}(t)
$$

где $B_{i,k}$ — базисные B-сплайн-функции, $c_i$ — коэффициенты,
найденные из условий интерполяции $S(t_j) = x_j$.

`scipy.interpolate.make_interp_spline` строит интерполяционный
B-сплайн с автоматическим выбором узлового вектора (not-a-knot).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import make_interp_spline


def interpolate(
    t_known: ArrayLike,
    x_known: ArrayLike,
    t_query: ArrayLike,
) -> NDArray[np.float64]:
    """Кубический B-сплайн (k=3) через scipy."""
    t_known = np.asarray(t_known, dtype=np.float64)
    x_known = np.asarray(x_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)

    spline = make_interp_spline(t_known, x_known, k=3)
    return np.asarray(spline(t_query), dtype=np.float64)
