"""Численные методы общего назначения.

Модуль содержит базовые алгоритмы линейной алгебры,
используемые другими модулями пакета.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def tridiagonal_solve(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
) -> NDArray[np.float64]:
    """Решение трёхдиагональной системы методом прогонки (алгоритм Томаса).

    Система имеет вид:

    $$
    b_0 x_0 + c_0 x_1 = d_0
    a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i, \\quad i = 1, \\dots, n-2
    a_{n-1} x_{n-2} + b_{n-1} x_{n-1} = d_{n-1}
    $$

    Параметры:
        a — поддиагональ длины n (a[0] игнорируется).
        b — главная диагональ длины n.
        c — наддиагональ длины n (c[n-1] игнорируется).
        d — правая часть длины n.

    Возвращает вектор решения x длины n.
    """
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    c = np.asarray(c, dtype=np.float64).copy()
    d = np.asarray(d, dtype=np.float64).copy()

    n = len(b)

    # Прямой ход
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # Обратный ход
    x = np.empty(n, dtype=np.float64)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x
