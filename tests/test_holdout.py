"""Тесты модуля holdout — разбиение на обучающую и тестовую выборки."""

import numpy as np

from interp_research.holdout import holdout_split, holdout_split_2d


class TestHoldoutSplit:
    """Базовые сценарии 1D-разбиения."""

    def test_step2(self):
        t = np.arange(10, dtype=float)
        x = t * 2

        t_train, x_train, t_test, x_test = holdout_split(t, x, step=2)

        np.testing.assert_array_equal(t_test, [1, 3, 5, 7, 9])
        np.testing.assert_array_equal(t_train, [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(x_test, t_test * 2)
        np.testing.assert_array_equal(x_train, t_train * 2)

    def test_step3(self):
        t = np.arange(9, dtype=float)
        x = np.ones(9)

        t_train, x_train, t_test, x_test = holdout_split(t, x, step=3)

        np.testing.assert_array_equal(t_test, [2, 5, 8])
        assert len(t_train) == 6

    def test_no_overlap(self):
        """Обучающая и тестовая выборки вместе дают полный набор."""
        t = np.arange(20, dtype=float)
        x = np.sin(t)

        t_train, _, t_test, _ = holdout_split(t, x, step=4)

        combined = np.sort(np.concatenate([t_train, t_test]))
        np.testing.assert_array_equal(combined, t)

    def test_sizes(self):
        n = 100
        step = 5
        t = np.arange(n, dtype=float)
        x = np.zeros(n)

        t_train, x_train, t_test, x_test = holdout_split(t, x, step=step)

        assert len(t_test) == n // step
        assert len(t_train) == n - n // step
        assert len(x_train) == len(t_train)
        assert len(x_test) == len(t_test)


class TestHoldoutSplit2D:
    """Разбиение для двумерных данных."""

    def test_basic(self):
        t = np.arange(6, dtype=float)
        x = t * 10
        y = t * 100

        t_train, x_train, y_train, t_test, x_test, y_test = holdout_split_2d(
            t, x, y, step=3
        )

        np.testing.assert_array_equal(t_test, [2, 5])
        np.testing.assert_array_equal(x_test, [20, 50])
        np.testing.assert_array_equal(y_test, [200, 500])
        assert len(t_train) == 4

    def test_consistency_with_1d(self):
        """2D-разбиение согласовано с 1D для каждой координаты."""
        t = np.arange(12, dtype=float)
        x = np.sin(t)
        y = np.cos(t)
        step = 4

        t_train_1d, x_train_1d, t_test_1d, x_test_1d = holdout_split(t, x, step)
        t_tr, x_tr, y_tr, t_te, x_te, y_te = holdout_split_2d(t, x, y, step)

        np.testing.assert_array_equal(t_tr, t_train_1d)
        np.testing.assert_array_equal(x_tr, x_train_1d)
        np.testing.assert_array_equal(t_te, t_test_1d)
        np.testing.assert_array_equal(x_te, x_test_1d)
