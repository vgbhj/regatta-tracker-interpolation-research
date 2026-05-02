"""Парсинг GPX-файлов и очистка GPS-треков для исследования интерполяции."""

from __future__ import annotations

from pathlib import Path

import gpxpy
import numpy as np
import numpy.typing as npt

from interp_research.geo import haversine

MAX_SPEED_MS: float = 30.0  # порог скорости для фильтрации выбросов, м/с


def load_gpx(path: Path) -> dict[str, npt.NDArray[np.float64]]:
    """Загружает GPX-файл и возвращает очищенные треки яхт.

    Возвращает словарь {yacht_name: ndarray (N, 3)}, где колонки —
    [t_seconds_from_start, lat, lon].

    Очистка данных:
    1. Удаление точек без временной метки.
    2. Удаление дубликатов по времени (оставляем первую).
    3. Удаление нарушений монотонности времени.
    4. Удаление выбросов по скорости (> 30 м/с между соседними точками).
    """
    with open(path, encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    tracks: dict[str, npt.NDArray[np.float64]] = {}

    for track in gpx.tracks:
        name = track.name or Path(path).stem

        raw_points: list[tuple[float, float, float]] = []
        for segment in track.segments:
            for pt in segment.points:
                if pt.time is None:
                    continue
                raw_points.append((pt.time.timestamp(), pt.latitude, pt.longitude))

        if not raw_points:
            continue

        data = np.array(raw_points, dtype=np.float64)
        data = data[np.argsort(data[:, 0])]

        data = _remove_time_duplicates(data)
        data = _enforce_monotonicity(data)
        data = _remove_speed_outliers(data)

        if len(data) < 2:
            continue

        data[:, 0] -= data[0, 0]

        tracks[name] = data

    return tracks


def _remove_time_duplicates(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Удаляет строки с повторяющимися временными метками (оставляет первую)."""
    _, idx = np.unique(data[:, 0], return_index=True)
    return data[idx]


def _enforce_monotonicity(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Удаляет точки, нарушающие строгую монотонность времени."""
    mask = np.ones(len(data), dtype=bool)
    last_t = data[0, 0]
    for i in range(1, len(data)):
        if data[i, 0] <= last_t:
            mask[i] = False
        else:
            last_t = data[i, 0]
    return data[mask]


def _remove_speed_outliers(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Удаляет точки, дающие скорость > MAX_SPEED_MS м/с относительно соседей."""
    if len(data) < 2:
        return data

    mask = np.ones(len(data), dtype=bool)
    for i in range(1, len(data)):
        dt = data[i, 0] - data[i - 1, 0]
        if dt <= 0:
            mask[i] = False
            continue
        dist = float(haversine(data[i - 1, 1], data[i - 1, 2], data[i, 1], data[i, 2]))
        speed = dist / dt
        if speed > MAX_SPEED_MS:
            mask[i] = False

    return data[mask]
