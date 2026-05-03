"""Тесты модуля holdout — разбиение на узлы интерполяции и контрольные точки."""

import numpy as np

from interp_research.holdout import holdout_split, holdout_split_2d


class TestHoldoutSplit:
    """Базовые сценарии 1D-разбиения."""

    def test_step2(self):
        t = np.arange(10, dtype=float)
        x = t * 2

        t_known, x_known, t_held, x_held = holdout_split(t, x, step=2)

        np.testing.assert_array_equal(t_known, [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(t_held, [1, 3, 5, 7])
        np.testing.assert_array_equal(x_known, t_known * 2)
        np.testing.assert_array_equal(x_held, t_held * 2)

    def test_step3(self):
        t = np.arange(9, dtype=float)
        x = np.ones(9)

        t_known, x_known, t_held, x_held = holdout_split(t, x, step=3)

        np.testing.assert_array_equal(t_known, [0, 3, 6])
        np.testing.assert_array_equal(t_held, [1, 2, 4, 5])

    def test_no_extrapolation(self):
        """Все контрольные точки лежат строго между первым и последним узлом."""
        t = np.arange(20, dtype=float)
        x = np.sin(t)

        t_known, _, t_held, _ = holdout_split(t, x, step=4)

        assert t_held[-1] < t_known[-1]

    def test_sizes(self):
        n = 100
        step = 5
        t = np.arange(n, dtype=float)
        x = np.zeros(n)

        t_known, x_known, t_held, x_held = holdout_split(t, x, step=step)

        assert len(t_known) == 20  # 0, 5, 10, ..., 95
        assert len(t_held) == 76   # indices 1..94 без кратных 5
        assert len(x_known) == len(t_known)
        assert len(x_held) == len(t_held)


class TestHoldoutSplit2D:
    """Разбиение для двумерных данных."""

    def test_basic(self):
        t = np.arange(6, dtype=float)
        x = t * 10
        y = t * 100

        t_known, x_known, y_known, t_held, x_held, y_held = holdout_split_2d(
            t, x, y, step=3
        )

        np.testing.assert_array_equal(t_known, [0, 3])
        np.testing.assert_array_equal(x_known, [0, 30])
        np.testing.assert_array_equal(y_known, [0, 300])
        np.testing.assert_array_equal(t_held, [1, 2])
        assert t_held[-1] < t_known[-1]

    def test_consistency_with_1d(self):
        """2D-разбиение согласовано с 1D для каждой координаты."""
        t = np.arange(12, dtype=float)
        x = np.sin(t)
        y = np.cos(t)
        step = 4

        t_known_1d, x_known_1d, t_held_1d, x_held_1d = holdout_split(t, x, step)
        t_kn, x_kn, y_kn, t_he, x_he, y_he = holdout_split_2d(t, x, y, step)

        np.testing.assert_array_equal(t_kn, t_known_1d)
        np.testing.assert_array_equal(x_kn, x_known_1d)
        np.testing.assert_array_equal(t_he, t_held_1d)
        np.testing.assert_array_equal(x_he, x_held_1d)
