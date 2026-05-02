"""Тесты модуля benchmark — замер производительности."""

import numpy as np

from interp_research.benchmark import benchmark_method


def _dummy_interpolate(
    t_known: np.ndarray,
    x_known: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """Тривиальная «интерполяция» — возвращает нули."""
    return np.zeros_like(t_query)


class TestBenchmarkMethod:
    def test_returns_expected_keys(self):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 0.0])
        t_q = np.array([0.5, 1.5])

        result = benchmark_method(_dummy_interpolate, t, x, t_q, n_runs=50, n_warmup=5)

        expected_keys = {"mean_us", "median_us", "std_us", "min_us", "max_us"}
        assert set(result.keys()) == expected_keys

    def test_values_are_positive(self):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 0.0])
        t_q = np.array([0.5])

        result = benchmark_method(_dummy_interpolate, t, x, t_q, n_runs=20, n_warmup=2)

        assert result["mean_us"] > 0
        assert result["median_us"] > 0
        assert result["min_us"] > 0
        assert result["max_us"] > 0
        assert result["std_us"] >= 0

    def test_min_le_median_le_max(self):
        t = np.linspace(0, 10, 50)
        x = np.sin(t)
        t_q = np.linspace(0, 10, 200)

        result = benchmark_method(_dummy_interpolate, t, x, t_q, n_runs=100, n_warmup=5)

        assert result["min_us"] <= result["median_us"]
        assert result["median_us"] <= result["max_us"]

    def test_calls_method_correct_number_of_times(self):
        call_count = 0

        def _counting_fn(t_known, x_known, t_query):
            nonlocal call_count
            call_count += 1
            return np.zeros_like(t_query)

        t = np.array([0.0, 1.0])
        x = np.array([0.0, 1.0])
        t_q = np.array([0.5])

        n_runs = 30
        n_warmup = 5
        benchmark_method(_counting_fn, t, x, t_q, n_runs=n_runs, n_warmup=n_warmup)

        assert call_count == n_runs + n_warmup
