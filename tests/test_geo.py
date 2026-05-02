"""Тесты для геодезических утилит."""

import numpy as np
import pytest

from interp_research.geo import LocalProjection, bearing, haversine


class TestHaversine:
    def test_moscow_to_spb(self) -> None:
        """Москва (55.7558, 37.6173) → Санкт-Петербург (59.9343, 30.3351) ≈ 634 км."""
        d = haversine(55.7558, 37.6173, 59.9343, 30.3351)
        assert 630_000 < float(d) < 638_000

    def test_zero_distance(self) -> None:
        d = haversine(0.0, 0.0, 0.0, 0.0)
        assert float(d) == pytest.approx(0.0, abs=1e-6)

    def test_vectorized(self) -> None:
        lat1 = np.array([0.0, 55.7558])
        lon1 = np.array([0.0, 37.6173])
        lat2 = np.array([0.0, 59.9343])
        lon2 = np.array([0.0, 30.3351])
        d = haversine(lat1, lon1, lat2, lon2)
        assert d.shape == (2,)
        assert float(d[0]) == pytest.approx(0.0, abs=1e-6)
        assert 630_000 < float(d[1]) < 638_000

    def test_antipodal(self) -> None:
        """Расстояние между антиподами ≈ π·R ≈ 20 015 км."""
        d = haversine(0.0, 0.0, 0.0, 180.0)
        assert float(d) == pytest.approx(np.pi * 6_371_000, rel=1e-6)


class TestBearing:
    def test_due_north(self) -> None:
        """Из (0, 0) строго на север → 0°."""
        b = bearing(0.0, 0.0, 1.0, 0.0)
        assert float(b) == pytest.approx(0.0, abs=0.01)

    def test_due_east(self) -> None:
        """Из (0, 0) строго на восток → 90°."""
        b = bearing(0.0, 0.0, 0.0, 1.0)
        assert float(b) == pytest.approx(90.0, abs=0.01)

    def test_due_south(self) -> None:
        b = bearing(1.0, 0.0, 0.0, 0.0)
        assert float(b) == pytest.approx(180.0, abs=0.01)

    def test_due_west(self) -> None:
        b = bearing(0.0, 0.0, 0.0, -1.0)
        assert float(b) == pytest.approx(270.0, abs=0.01)

    def test_moscow_to_spb(self) -> None:
        """Азимут Москва → Питер — примерно северо-запад (~320°)."""
        b = bearing(55.7558, 37.6173, 59.9343, 30.3351)
        assert 315 < float(b) < 325

    def test_result_in_range(self) -> None:
        b = bearing(10.0, 20.0, -30.0, -40.0)
        assert 0 <= float(b) < 360


class TestLocalProjection:
    def test_roundtrip_scalar(self) -> None:
        proj = LocalProjection(55.75, 37.62)
        lat, lon = 55.76, 37.63
        x, y = proj.to_local(lat, lon)
        lat2, lon2 = proj.to_geo(x, y)
        assert float(lat2) == pytest.approx(lat, abs=1e-6)
        assert float(lon2) == pytest.approx(lon, abs=1e-6)

    def test_roundtrip_vectorized(self) -> None:
        proj = LocalProjection(55.75, 37.62)
        lats = np.array([55.76, 55.77, 55.78])
        lons = np.array([37.63, 37.64, 37.65])
        x, y = proj.to_local(lats, lons)
        assert x.shape == (3,)
        lat2, lon2 = proj.to_geo(x, y)
        np.testing.assert_allclose(lat2, lats, atol=1e-6)
        np.testing.assert_allclose(lon2, lons, atol=1e-6)

    def test_origin_is_zero(self) -> None:
        proj = LocalProjection(55.75, 37.62)
        x, y = proj.to_local(55.75, 37.62)
        assert float(x) == pytest.approx(0.0, abs=1e-6)
        assert float(y) == pytest.approx(0.0, abs=1e-6)

    def test_displacement_magnitude(self) -> None:
        """Сдвиг на ~1° широты ≈ ~111 км на север."""
        proj = LocalProjection(0.0, 0.0)
        x, y = proj.to_local(1.0, 0.0)
        assert float(x) == pytest.approx(0.0, abs=1.0)
        assert 110_000 < float(y) < 112_000
