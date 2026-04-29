# Источники GPX-треков

Все треки — реальные GPS-записи парусных гонок и тренировок, скачанные из открытых репозиториев под лицензией MIT. Атрибуция авторов — обязательна при использовании в работе.

## Сводка

| Файл | Точек | Регион / Регата | Класс / Лодка | Год | Источник |
|------|-------|-----------------|---------------|-----|----------|
| `gin-sul-rund-hanskalbsand-2024-yury.gpx` | 1694 | Эльба, Германия / **Gin Sul Rund Hanskalbsand 2024** (официальная регата) | Sail (Strava) | 2024 | kirienko/gpx-player |
| `hamburg-elbe-alex.gpx` | 1909 | Эльба, Гамбург / клубная гонка | sailing_v2 (Garmin) | — | kirienko/gpx-player |
| `hamburg-elbe-richard.gpx` | 1567 | Эльба, Гамбург / клубная гонка | sailing_v2 (Garmin) | — | kirienko/gpx-player |
| `hamburg-elbe-track2.gpx` | 337 | Эльба, Гамбург / "Hamburg Sailing" | sailing_v2 (Garmin) | — | kirienko/gpx-player |
| `hamburg-elbe-track3.gpx` | 535 | Эльба, Гамбург / "Hamburg Segeln" | sailing_v2 (Garmin) | — | kirienko/gpx-player |
| `ottawa-2016-05-25.gpx` | 1263 | Река Оттава, Канада / Nepean Sailing Club, тренировка | парусный швертбот | 2016 | mkobetic/gpx |
| `ottawa-2016-06-05.gpx` | 2931 | Река Оттава, Канада / Nepean Sailing Club, тренировка | парусный швертбот | 2016 | mkobetic/gpx |
| `ottawa-2016-08-24.gpx` | 4052 | Река Оттава, Канада / Nepean Sailing Club, тренировка | парусный швертбот | 2016 | mkobetic/gpx |

Итого: 8 треков, ~14.3k точек, 2 географических региона (Канада / Германия), от ~30 минут до ~3 часов записи.

## Происхождение

### kirienko/gpx-player (5 файлов, Гамбург / Эльба)
- Репозиторий: https://github.com/kirienko/gpx-player
- Лицензия: MIT, © 2023 Yury Kirienko
- Папка с примерами: `example-data/`
- Файлы переименованы:
  - `osm-demo-Yury.gpx` → `gin-sul-rund-hanskalbsand-2024-yury.gpx`
  - `osm-demo-Alex.gpx` → `hamburg-elbe-alex.gpx`
  - `osm-demo-Richard.gpx` → `hamburg-elbe-richard.gpx`
  - `track2.gpx` → `hamburg-elbe-track2.gpx`
  - `track3.gpx` → `hamburg-elbe-track3.gpx`
- `track1.gpx` (Hamburg Cycling) сознательно не включён — это велозаезд, не парус.

### mkobetic/gpx (3 файла, Оттава)
- Репозиторий: https://github.com/mkobetic/gpx
- Лицензия: MIT, © 2016 Martin Kobetic
- Папка с примерами: `samples/in/`
- Файлы переименованы по дате записи:
  - `160525.gpx` → `ottawa-2016-05-25.gpx`
  - `160605.gpx` → `ottawa-2016-06-05.gpx`
  - `160824.gpx` → `ottawa-2016-08-24.gpx`

## Как воспроизвести скачивание

```bash
git clone --depth 1 https://github.com/kirienko/gpx-player.git /tmp/kp
git clone --depth 1 https://github.com/mkobetic/gpx.git /tmp/mk
cp /tmp/kp/example-data/{osm-demo-Yury,osm-demo-Alex,osm-demo-Richard,track2,track3}.gpx data/raw/
cp /tmp/mk/samples/in/{160525,160605,160824}.gpx data/raw/
# затем переименовать по таблице выше
```

## Замечания по разнообразию данных

- **Регионы**: 2 (река Оттава, Канада + река Эльба, Германия) — разные гидрологические условия.
- **Поколения GPS**: записи 2016 (GPSBabel) и 2023–2024 (Garmin Connect, Strava).
- **Длительность**: от ~10 мин (track2) до ~3 ч (ottawa-2016-06-05) — разные масштабы.
- **Тип активности**: 1 явная регата (Gin Sul Rund Hanskalbsand 2024), остальное — клубные гонки и тренировки. Для задачи **сравнения интерполяционных методов** это не недостаток: важна реалистичная динамика парусной лодки (повороты, лавировка, разная скорость), а не статус соревнования.

## Что ещё можно добавить (если нужно больше)

- **chartedsails (chacal/chartedsails)** — содержит 4 CSV-лога формата Expedition с реальных регат (Auckland, San Francisco, Solent, Valencia) с полями `Lat`, `Lon`, `Cog`, `Sog`, `Utc` и десятками других. Требует CSV→GPX конвертера или прямого парсера CSV.
- **raceqs.com** — публичные 3D-реплеи гонок, GPX можно выгружать поштучно после регистрации.
- **Strava (тег "Sailing")** — публичные активности других пользователей доступны на экспорт GPX.
- **OpenSeaMap GPS traces** — публичные морские треки в OSM (не все парус, но фильтруются по тегу).
