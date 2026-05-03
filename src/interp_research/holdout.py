"""Разбиение данных на узлы интерполяции и контрольные точки."""

import numpy as np
from numpy.typing import NDArray


def holdout_split(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    step: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Разбивает временной ряд на узлы интерполяции и контрольные точки.

    Каждая ``step``-я точка (начиная с индекса 0) **остаётся** как узел
    интерполяции, все промежуточные становятся контрольными.
    Контрольные точки после последнего узла отбрасываются (нельзя
    экстраполировать).

    Parameters
    ----------
    t : массив временных меток
    x : массив значений координаты
    step : шаг прореживания (каждая step-я точка — узел)

    Returns
    -------
    (t_known, x_known, t_held, x_held)
    """
    n = len(t)
    mask_known = np.zeros(n, dtype=bool)
    mask_known[::step] = True
    last_known = np.max(np.nonzero(mask_known))
    mask_held = ~mask_known
    mask_held[last_known:] = False

    return t[mask_known], x[mask_known], t[mask_held], x[mask_held]


def holdout_split_2d(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    step: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Разбивает двумерный временной ряд на узлы интерполяции и контрольные точки.

    Аналог :func:`holdout_split` для пары координат (x, y).

    Returns
    -------
    (t_known, x_known, y_known, t_held, x_held, y_held)
    """
    n = len(t)
    mask_known = np.zeros(n, dtype=bool)
    mask_known[::step] = True
    last_known = np.max(np.nonzero(mask_known))
    mask_held = ~mask_known
    mask_held[last_known:] = False

    return (
        t[mask_known],
        x[mask_known],
        y[mask_known],
        t[mask_held],
        x[mask_held],
        y[mask_held],
    )
