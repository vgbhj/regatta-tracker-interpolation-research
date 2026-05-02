"""Бенчмаркинг методов интерполяции."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def benchmark_method(
    method_fn: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ],
    t_known: NDArray[np.float64],
    x_known: NDArray[np.float64],
    t_query: NDArray[np.float64],
    *,
    n_runs: int = 1000,
    n_warmup: int = 10,
) -> dict[str, float]:
    """Замеряет время выполнения метода интерполяции.

    Выполняет ``n_warmup`` прогревочных вызовов (результат отбрасывается),
    затем ``n_runs`` замеров через ``time.perf_counter``.

    Returns
    -------
    dict с ключами ``mean_us``, ``median_us``, ``std_us``,
    ``min_us``, ``max_us`` — статистики в микросекундах.
    """
    for _ in range(n_warmup):
        method_fn(t_known, x_known, t_query)

    timings_us: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        method_fn(t_known, x_known, t_query)
        elapsed = time.perf_counter() - start
        timings_us.append(elapsed * 1e6)

    arr = np.array(timings_us)
    return {
        "mean_us": float(np.mean(arr)),
        "median_us": float(np.median(arr)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
    }
