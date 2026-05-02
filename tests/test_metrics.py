"""Тесты модуля metrics — метрики ошибок интерполяции."""

import numpy as np

from interp_research.metrics import mae, max_error, rmse, rmse_2d


class TestRMSE:
    def test_zero_error(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == 0.0

    def test_known_value(self):
        true = np.array([0.0, 0.0])
        pred = np.array([3.0, 4.0])
        # sqrt(mean([9, 16])) = sqrt(12.5)
        np.testing.assert_allclose(rmse(true, pred), np.sqrt(12.5))

    def test_symmetric(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5])
        assert rmse(a, b) == rmse(b, a)


class TestMAE:
    def test_zero_error(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == 0.0

    def test_known_value(self):
        true = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert mae(true, pred) == 2.0

    def test_symmetric(self):
        a = np.array([1.0, 5.0])
        b = np.array([3.0, 3.0])
        assert mae(a, b) == mae(b, a)


class TestMaxError:
    def test_zero_error(self):
        a = np.array([1.0, 2.0])
        assert max_error(a, a) == 0.0

    def test_known_value(self):
        true = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, -5.0, 3.0])
        assert max_error(true, pred) == 5.0


class TestRMSE2D:
    def test_zero_error(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert rmse_2d(x, y, x, y) == 0.0

    def test_known_value(self):
        true_x = np.array([0.0])
        true_y = np.array([0.0])
        pred_x = np.array([3.0])
        pred_y = np.array([4.0])
        # sqrt(mean([9+16])) = sqrt(25) = 5
        assert rmse_2d(true_x, true_y, pred_x, pred_y) == 5.0

    def test_multiple_points(self):
        true_x = np.array([0.0, 0.0])
        true_y = np.array([0.0, 0.0])
        pred_x = np.array([1.0, 2.0])
        pred_y = np.array([0.0, 0.0])
        # sqrt(mean([1, 4])) = sqrt(2.5)
        np.testing.assert_allclose(
            rmse_2d(true_x, true_y, pred_x, pred_y), np.sqrt(2.5)
        )
