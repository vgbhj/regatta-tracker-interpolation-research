"""Тесты методов интерполяции."""

import numpy as np
import pytest

from interp_research.methods import linear, lagrange


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
