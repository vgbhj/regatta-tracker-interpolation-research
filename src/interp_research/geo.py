"""Геодезические утилиты для работы с GPS-координатами.

Все функции векторизованы через numpy — принимают как скаляры, так и массивы.
Используется сферическая модель Земли с радиусом R = 6 371 000 м.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

R_EARTH: float = 6_371_000.0  # средний радиус Земли, м


def haversine(
    lat1: npt.ArrayLike,
    lon1: npt.ArrayLike,
    lat2: npt.ArrayLike,
    lon2: npt.ArrayLike,
) -> np.floating | npt.NDArray[np.floating]:
    """Расстояние между двумя точками на сфере по формуле гаверсинусов (м).

    Формула:
        a = sin²(Δφ/2) + cos(φ₁) · cos(φ₂) · sin²(Δλ/2)
        c = 2 · atan2(√a, √(1−a))
        d = R · c

    где φ — широта, λ — долгота (в радианах), R — радиус Земли.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R_EARTH * c


def bearing(
    lat1: npt.ArrayLike,
    lon1: npt.ArrayLike,
    lat2: npt.ArrayLike,
    lon2: npt.ArrayLike,
) -> np.floating | npt.NDArray[np.floating]:
    """Начальный азимут (forward azimuth) от точки 1 к точке 2 в градусах [0, 360).

    Формула:
        θ = atan2(sin(Δλ) · cos(φ₂),
                  cos(φ₁) · sin(φ₂) − sin(φ₁) · cos(φ₂) · cos(Δλ))

    Результат приводится к диапазону [0, 360) через (θ° + 360) mod 360.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2)
    )
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    theta = np.degrees(np.arctan2(x, y))
    return theta % 360


class LocalProjection:
    """Локальная плоская проекция относительно опорной точки (эквидистантная).

    Переводит (lat, lon) в метрические координаты (x — восток, y — север)
    вблизи опорной точки. Точность высокая для расстояний до ~50 км.

    Формулы прямого преобразования:
        x = R · (λ − λ₀) · cos(φ₀)
        y = R · (φ − φ₀)

    Обратное преобразование:
        φ = φ₀ + y / R
        λ = λ₀ + x / (R · cos(φ₀))

    где φ₀, λ₀ — координаты опорной точки в радианах.
    """

    def __init__(self, ref_lat: float, ref_lon: float) -> None:
        self.ref_lat_rad = np.radians(ref_lat)
        self.ref_lon_rad = np.radians(ref_lon)
        self._cos_ref = np.cos(self.ref_lat_rad)

    def to_local(
        self,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
    ) -> tuple[np.floating | npt.NDArray[np.floating], np.floating | npt.NDArray[np.floating]]:
        """Преобразует (lat, lon) в градусах → (x, y) в метрах от опорной точки."""
        lat_rad = np.radians(np.asarray(lat, dtype=np.float64))
        lon_rad = np.radians(np.asarray(lon, dtype=np.float64))
        x = R_EARTH * (lon_rad - self.ref_lon_rad) * self._cos_ref
        y = R_EARTH * (lat_rad - self.ref_lat_rad)
        return x, y

    def to_geo(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
    ) -> tuple[np.floating | npt.NDArray[np.floating], np.floating | npt.NDArray[np.floating]]:
        """Преобразует (x, y) в метрах → (lat, lon) в градусах."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lat = np.degrees(self.ref_lat_rad + y / R_EARTH)
        lon = np.degrees(self.ref_lon_rad + x / (R_EARTH * self._cos_ref))
        return lat, lon
