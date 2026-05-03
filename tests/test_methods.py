"""Тесты методов интерполяции."""

import numpy as np
import pytest

from interp_research.methods import linear, lagrange, cubic_spline, b_spline, hermite
from interp_research.numeric import tridiagonal_solve


# ── Линейная интерполяция ────────────────────────────────────────────


class TestLinear:
    """Известные значения на простом примере."""

    def test_midpoints(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        x = np.array([0.0, 2.0, 1.0, 3.0])
        t_q = np.array([0.5, 1.5, 2.5])
        result = linear.interpolate(t, x, t_q)
        np.testing.assert_allclose(result, [1.0, 1.5, 2.0])

    def test_at_nodes(self):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([10.0, 20.0, 30.0])
        result = linear.interpolate(t, x, t)
        np.testing.assert_allclose(result, x)

    def test_quarter_points(self):
        t = np.array([0.0, 4.0])
        x = np.array([0.0, 8.0])
        t_q = np.array([1.0, 2.0, 3.0])
        result = linear.interpolate(t, x, t_q)
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0])


# ── Лагранжева интерполяция ──────────────────────────────────────────


class TestLagrange:
    """Полином 8-й степени: 9 узлов → точная реконструкция."""

    @pytest.fixture()
    def poly8(self):
        """Полином p(t) = t^8 - 3t^5 + 2t^2 - 7 на 9 равномерных узлах."""
        rng = np.random.default_rng(42)
        coeffs = np.array([1.0, 0, 0, -3, 0, 0, 2, 0, -7])
        t_nodes = np.linspace(-1, 1, 9)
        x_nodes = np.polyval(coeffs, t_nodes)
        t_query = np.sort(rng.uniform(-1, 1, 50))
        x_exact = np.polyval(coeffs, t_query)
        return t_nodes, x_nodes, t_query, x_exact

    def test_exact_on_poly8(self, poly8):
        t_nodes, x_nodes, t_query, x_exact = poly8
        result = lagrange.interpolate(t_nodes, x_nodes, t_query, degree=8)
        np.testing.assert_allclose(result, x_exact, atol=1e-10)

    def test_at_nodes(self, poly8):
        t_nodes, x_nodes, _, _ = poly8
        result = lagrange.interpolate(t_nodes, x_nodes, t_nodes, degree=8)
        np.testing.assert_allclose(result, x_nodes, atol=1e-14)

    def test_lower_degree_on_linear(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = 2.0 * t + 1.0
        t_q = np.array([0.5, 1.5, 3.5])
        result = lagrange.interpolate(t, x, t_q, degree=1)
        np.testing.assert_allclose(result, 2.0 * t_q + 1.0, atol=1e-14)


# ── Метод прогонки ───────────────────────────────────────────────────


class TestTridiagonalSolve:
    """Тест на известном решении трёхдиагональной системы."""

    def test_known_system(self):
        # Система 4×4:
        # [2 1 0 0] [x0]   [1]
        # [1 3 1 0] [x1] = [2]
        # [0 1 3 1] [x2]   [3]
        # [0 0 1 2] [x3]   [4]
        a = np.array([0.0, 1.0, 1.0, 1.0])
        b = np.array([2.0, 3.0, 3.0, 2.0])
        c = np.array([1.0, 1.0, 1.0, 0.0])
        d = np.array([1.0, 2.0, 3.0, 4.0])

        x = tridiagonal_solve(a, b, c, d)

        A = np.array([
            [2, 1, 0, 0],
            [1, 3, 1, 0],
            [0, 1, 3, 1],
            [0, 0, 1, 2],
        ], dtype=float)
        x_ref = np.linalg.solve(A, d)
        np.testing.assert_allclose(x, x_ref, atol=1e-14)

    def test_identity_like(self):
        n = 10
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.arange(n, dtype=float)
        x = tridiagonal_solve(a, b, c, d)
        np.testing.assert_allclose(x, d, atol=1e-15)


# ── Кубический сплайн ───────────────────────────────────────────────


class TestCubicSpline:
    """Cross-check собственной реализации с scipy + тест на sin."""

    def test_cross_check_with_scipy(self):
        rng = np.random.default_rng(123)
        t = np.sort(rng.uniform(0, 10, 20))
        x = np.sin(t)
        t_q = np.linspace(t[0], t[-1], 200)

        own = cubic_spline.interpolate(t, x, t_q)
        ref = cubic_spline.interpolate_scipy(t, x, t_q)
        np.testing.assert_allclose(own, ref, atol=1e-9)

    def test_sin_rmse(self):
        t = np.linspace(0, 2 * np.pi, 30)
        x = np.sin(t)
        t_q = np.linspace(0, 2 * np.pi, 500)
        x_exact = np.sin(t_q)

        result = cubic_spline.interpolate(t, x, t_q)
        rmse = np.sqrt(np.mean((result - x_exact) ** 2))
        assert rmse < 1e-3, f"RMSE слишком большой: {rmse}"

    def test_at_nodes(self):
        t = np.linspace(0, 5, 15)
        x = np.cos(t)
        result = cubic_spline.interpolate(t, x, t)
        np.testing.assert_allclose(result, x, atol=1e-12)


# ── B-сплайн ────────────────────────────────────────────────────────


class TestBSpline:
    """Тест B-сплайна на гладкой функции."""

    def test_sin_rmse(self):
        t = np.linspace(0, 2 * np.pi, 30)
        x = np.sin(t)
        t_q = np.linspace(0, 2 * np.pi, 500)
        x_exact = np.sin(t_q)

        result = b_spline.interpolate(t, x, t_q)
        rmse = np.sqrt(np.mean((result - x_exact) ** 2))
        assert rmse < 1e-3, f"RMSE слишком большой: {rmse}"

    def test_at_nodes(self):
        t = np.linspace(0, 5, 15)
        x = np.cos(t)
        result = b_spline.interpolate(t, x, t)
        np.testing.assert_allclose(result, x, atol=1e-12)


# ── Эрмитова интерполяция (эрмитова интерполяция) ─────────────────────────────


class TestHermite:
    """Тест на параметрической кривой (cos t, sin t)."""

    def test_circle_rmse_accurate(self):
        """Оба метода точны на (cos t, sin t); hermite — в том же порядке."""
        t = np.linspace(0, 2 * np.pi, 40)
        t_q = np.linspace(0, 2 * np.pi, 500)

        for coord_fn in [np.cos, np.sin]:
            x = coord_fn(t)
            x_exact = coord_fn(t_q)

            rmse_ma = np.sqrt(np.mean((hermite.interpolate(t, x, t_q) - x_exact) ** 2))
            rmse_cs = np.sqrt(np.mean((cubic_spline.interpolate(t, x, t_q) - x_exact) ** 2))
            assert rmse_ma < 1e-3, f"hermite RMSE слишком большой: {rmse_ma:.2e}"
            assert rmse_cs < 1e-3, f"cubic_spline RMSE слишком большой: {rmse_cs:.2e}"
            # На гладких данных кубический сплайн оптимизирует вторые производные
            # глобально, поэтому может быть точнее. Проверяем, что hermite
            # остаётся в пределах двух порядков — разумная близость.
            assert rmse_ma < rmse_cs * 100, (
                f"hermite RMSE ({rmse_ma:.2e}) на порядки хуже "
                f"cubic_spline ({rmse_cs:.2e})"
            )

    def test_at_nodes(self):
        t = np.linspace(0, 5, 15)
        x = np.cos(t)
        result = hermite.interpolate(t, x, t)
        np.testing.assert_allclose(result, x, atol=1e-12)
