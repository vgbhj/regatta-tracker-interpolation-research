"""Разбиение данных на обучающую и удержанную выборки."""

import numpy as np
from numpy.typing import NDArray


def holdout_split(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    step: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Разбивает временной ряд на обучающую и тестовую выборки.

    Каждая ``step``-я точка (начиная с индекса ``step - 1``) попадает
    в тестовую выборку, остальные — в обучающую.

    Parameters
    ----------
    t : массив временных меток
    x : массив значений координаты
    step : шаг прореживания (каждая step-я точка — тестовая)

    Returns
    -------
    (t_train, x_train, t_test, x_test)
    """
    mask_test = np.zeros(len(t), dtype=bool)
    mask_test[step - 1 :: step] = True

    t_train = t[~mask_test]
    x_train = x[~mask_test]
    t_test = t[mask_test]
    x_test = x[mask_test]

    return t_train, x_train, t_test, x_test


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
    """Разбивает двумерный временной ряд на обучающую и тестовую выборки.

    Аналог :func:`holdout_split` для пары координат (x, y).

    Returns
    -------
    (t_train, x_train, y_train, t_test, x_test, y_test)
    """
    mask_test = np.zeros(len(t), dtype=bool)
    mask_test[step - 1 :: step] = True

    return (
        t[~mask_test],
        x[~mask_test],
        y[~mask_test],
        t[mask_test],
        x[mask_test],
        y[mask_test],
    )
