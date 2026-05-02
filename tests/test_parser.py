"""Тесты для парсера GPX-файлов.

Использует фикстурный файл tests/fixtures/sample.gpx.
"""

from pathlib import Path

import numpy as np
import pytest

from interp_research.parser import load_gpx

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_GPX = FIXTURES_DIR / "sample.gpx"


@pytest.fixture(autouse=True)
def _check_fixture() -> None:
    if not SAMPLE_GPX.exists():
        pytest.skip(
            f"Фикстура {SAMPLE_GPX} не найдена. "
            "Скопируйте один из реальных GPX из data/raw/ в tests/fixtures/sample.gpx"
        )


class TestLoadGpx:
    def test_returns_dict(self) -> None:
        tracks = load_gpx(SAMPLE_GPX)
        assert isinstance(tracks, dict)
        assert len(tracks) > 0

    def test_array_shape(self) -> None:
        tracks = load_gpx(SAMPLE_GPX)
        for name, arr in tracks.items():
            assert arr.ndim == 2, f"Track '{name}': expected 2D array"
            assert arr.shape[1] == 3, f"Track '{name}': expected 3 columns"
            assert arr.shape[0] >= 2, f"Track '{name}': expected at least 2 points"

    def test_time_starts_at_zero(self) -> None:
        tracks = load_gpx(SAMPLE_GPX)
        for name, arr in tracks.items():
            assert arr[0, 0] == pytest.approx(0.0), f"Track '{name}': time must start at 0"

    def test_time_strictly_monotonic(self) -> None:
        tracks = load_gpx(SAMPLE_GPX)
        for name, arr in tracks.items():
            dt = np.diff(arr[:, 0])
            assert np.all(dt > 0), f"Track '{name}': time is not strictly monotonic"

    def test_lat_lon_reasonable(self) -> None:
        tracks = load_gpx(SAMPLE_GPX)
        for name, arr in tracks.items():
            assert np.all(np.abs(arr[:, 1]) <= 90), f"Track '{name}': latitude out of range"
            assert np.all(np.abs(arr[:, 2]) <= 180), f"Track '{name}': longitude out of range"

    def test_no_speed_outliers(self) -> None:
        """После очистки не должно быть скоростей > 30 м/с."""
        from interp_research.geo import haversine

        tracks = load_gpx(SAMPLE_GPX)
        for name, arr in tracks.items():
            if len(arr) < 2:
                continue
            dists = haversine(arr[:-1, 1], arr[:-1, 2], arr[1:, 1], arr[1:, 2])
            dts = np.diff(arr[:, 0])
            speeds = dists / dts
            assert np.all(speeds <= 30.0), (
                f"Track '{name}': max speed {float(speeds.max()):.1f} m/s exceeds threshold"
            )
