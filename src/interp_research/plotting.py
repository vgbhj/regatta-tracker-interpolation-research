"""Единый стиль matplotlib для всех ноутбуков и фигур ВКР.

Применяется один раз во второй ячейке ноутбука вызовом ``set_thesis_style()``.
Палитра ``THESIS_COLORS`` — пять цветов под пять методов интерполяции, упорядочены
по возрастанию светлоты, чтобы оставаться различимыми при ч/б печати.
"""

from __future__ import annotations

import matplotlib as mpl

# Cantimetric: 1 inch = 2.54 cm; целевой размер 16×10 см.
_CM_PER_INCH: float = 2.54
_FIGSIZE_CM: tuple[float, float] = (16.0, 10.0)
_FIGSIZE_INCHES: tuple[float, float] = (
    _FIGSIZE_CM[0] / _CM_PER_INCH,
    _FIGSIZE_CM[1] / _CM_PER_INCH,
)

# Палитра для пяти методов интерполяции.
# Подобрана так, чтобы:
#   - быть нейтральной (без кричащих насыщенных цветов),
#   - монотонно возрастать по светлоте → различима в ч/б,
#   - различаться по тону → различима в цвете.
THESIS_COLORS: list[str] = [
    "#1f1f1f",  # near-black
    "#3b5b80",  # muted slate blue
    "#7d9b6a",  # sage green
    "#c08552",  # terracotta
    "#b8b8b8",  # light gray
]


def set_thesis_style() -> None:
    """Настроить ``matplotlib.rcParams`` под стиль бакалаврской работы.

    Параметры:
        - шрифт: DejaVu Serif (приближение к Times New Roman, поддерживает кириллицу),
          размер 11 pt;
        - размер фигуры по умолчанию: 16×10 см;
        - dpi: 150 для интерактивного отображения, 300 для сохранения;
        - фон: белый, без декоративной заливки;
        - сетка: тонкая (0.5 pt), полупрозрачная.
    """
    mpl.rcParams.update(
        {
            # Шрифты.
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif", "serif"],
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 12,
            # Размер и разрешение.
            "figure.figsize": _FIGSIZE_INCHES,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Фон.
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # Линии и сетка.
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.5,
            "grid.linestyle": "-",
            "grid.color": "#999999",
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            # Чище — без верхней и правой рамок.
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Цикл цветов — палитра ВКР.
            "axes.prop_cycle": mpl.cycler(color=THESIS_COLORS),
            # Легенда без рамки выглядит опрятнее.
            "legend.frameon": False,
        }
    )


__all__ = ["THESIS_COLORS", "set_thesis_style"]
