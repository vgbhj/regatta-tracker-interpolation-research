"""Интерполяция полиномом Лагранжа со скользящим окном (реализация с нуля).

Для каждой точки запроса $t$ выбираются $k = \\text{degree} + 1$
ближайших узловых точек, и значение вычисляется по формуле Лагранжа:

$$
L(t) = \\sum_{j=0}^{k-1} x_j \\prod_{\\substack{m=0 \\\\ m \\neq j}}^{k-1}
        \\frac{t - t_m}{t_j - t_m}
$$

Скользящее окно позволяет избежать осцилляций Рунге,
свойственных глобальному полиному высокой степени.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _lagrange_single(t: float, t_nodes: NDArray, x_nodes: NDArray) -> float:
    """Значение полинома Лагранжа в одной точке."""
    k = len(t_nodes)
    result = 0.0
    for j in range(k):
        basis = 1.0
        for m in range(k):
            if m != j:
                basis *= (t - t_nodes[m]) / (t_nodes[j] - t_nodes[m])
        result += x_nodes[j] * basis
    return result


def _select_window(
    t: float, t_known: NDArray, window_size: int
) -> tuple[int, int]:
    """Индексы начала и конца окна из window_size ближайших узлов."""
    n = len(t_known)
    pos = np.searchsorted(t_known, t)
    start = max(0, pos - window_size // 2)
    end = start + window_size
    if end > n:
        end = n
        start = max(0, end - window_size)
    return start, end


def interpolate(
    t_known: ArrayLike,
    x_known: ArrayLike,
    t_query: ArrayLike,
    degree: int = 8,
) -> NDArray[np.float64]:
    """Интерполяция полиномом Лагранжа степени degree со скользящим окном."""
    t_known = np.asarray(t_known, dtype=np.float64)
    x_known = np.asarray(x_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)

    window_size = degree + 1
    scalar_input = t_query.ndim == 0
    t_query = np.atleast_1d(t_query)

    result = np.empty_like(t_query)
    for i, t in enumerate(t_query):
        start, end = _select_window(t, t_known, window_size)
        result[i] = _lagrange_single(t, t_known[start:end], x_known[start:end])

    if scalar_input:
        return result[0]  # type: ignore[return-value]
    return result
