"""Метрики ошибок интерполяции."""

import numpy as np
from numpy.typing import NDArray


def rmse(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    """Корень из среднеквадратичной ошибки (RMSE)."""
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    """Средняя абсолютная ошибка (MAE)."""
    return float(np.mean(np.abs(true - pred)))


def max_error(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    """Максимальная абсолютная ошибка."""
    return float(np.max(np.abs(true - pred)))


def rmse_2d(
    true_x: NDArray[np.float64],
    true_y: NDArray[np.float64],
    pred_x: NDArray[np.float64],
    pred_y: NDArray[np.float64],
) -> float:
    """Комбинированная RMSE по двум координатам.

    Вычисляется как корень из среднего суммы квадратов покомпонентных
    отклонений:

    .. math::

        \\text{RMSE}_{2D} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N}
        \\bigl[(x_i - \\hat x_i)^2 + (y_i - \\hat y_i)^2\\bigr]}

    Предполагается, что координаты уже в метрах локальной проекции.
    """
    sq = (true_x - pred_x) ** 2 + (true_y - pred_y) ** 2
    return float(np.sqrt(np.mean(sq)))
