"""Видимость из окон отеля: закат, море, достопримечательности.

Сцена — растр в локальных метрах (x на восток, y на север, массив [iy, ix],
iy растёт на север). Поверхность = рельеф (DEM) + высота зданий.
Из точки наблюдения (окно, этаж f) пускаем лучи по азимутам и считаем
угол возвышения горизонта: max_d atan((H(d) - z) / d), с поправкой на
кривизну Земли и рефракцию. Цель видна, если она выше этого горизонта.

Только numpy. Азимут — по часовой от севера, в градусах.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

EARTH_R = 6_371_000.0
REFRACTION_K = 0.13  # стандартный коэффициент рефракции
MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


# --------------------------------------------------------------------------
# Сцена и координаты
# --------------------------------------------------------------------------
@dataclass
class Scene:
    dem: np.ndarray                 # высота рельефа над уровнем моря, м  [iy, ix]
    buildings: np.ndarray           # высота зданий над землёй, м (0 — нет здания)
    res: float                      # размер пикселя, м
    x0: float = 0.0                 # координаты центра пикселя [0, 0]
    y0: float = 0.0
    water: np.ndarray | None = None  # bool-маска воды (море/озеро)

    def __post_init__(self):
        if self.dem.shape != self.buildings.shape:
            raise ValueError("dem и buildings должны быть одного размера")
        self.surface = self.dem + self.buildings

    def sample(self, arr: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Значения растра в точках (nearest). Возвращает (values, inside_mask)."""
        ix = np.rint((x - self.x0) / self.res).astype(int)
        iy = np.rint((y - self.y0) / self.res).astype(int)
        ny, nx = arr.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return arr[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)], inside

    def scaled(self, k: float) -> "Scene":
        """Копия сцены с высотами зданий, умноженными на k (анализ чувствительности)."""
        return Scene(self.dem, self.buildings * k, self.res, self.x0, self.y0, self.water)


def latlon_to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Эквидистантная проекция вокруг (lat0, lon0) -> (x_east, y_north) в метрах."""
    x = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R
    return x, y


@dataclass
class Facade:
    name: str       # "запад", "юг", ...
    azimuth: float  # направление нормали фасада (куда смотрят окна)
    x: float        # положение окна (на стене), локальные метры
    y: float


# --------------------------------------------------------------------------
# Лучи и горизонт
# --------------------------------------------------------------------------
def eye_z(scene: Scene, x: float, y: float, floor: int, floor_h: float = 3.0, eye: float = 1.5) -> float:
    """Высота глаз наблюдателя над уровнем моря. floor=1 — первый этаж."""
    ground, _ = scene.sample(scene.dem, np.array([x]), np.array([y]))
    return float(ground[0]) + (floor - 1) * floor_h + eye


def _rays(scene: Scene, x: float, y: float, az_deg: np.ndarray, max_dist: float,
          exclude: np.ndarray | None):
    """Эффективные высоты поверхности вдоль лучей: форма (A, D)."""
    d = np.arange(scene.res, max_dist + scene.res, scene.res)
    az = np.radians(az_deg)[:, None]
    xs = x + d[None, :] * np.sin(az)
    ys = y + d[None, :] * np.cos(az)

    H, inside = scene.sample(scene.surface, xs, ys)
    if exclude is not None:  # собственное здание отеля не заслоняет вид
        ex, _ = scene.sample(exclude.astype(float), xs, ys)
        ground, _ = scene.sample(scene.dem, xs, ys)
        H = np.where(ex > 0.5, ground, H)

    drop = (1 - REFRACTION_K) * d ** 2 / (2 * EARTH_R)  # кривизна Земли
    H_eff = np.where(inside, H - drop[None, :], -1e6)   # вне растра — препятствий нет
    water = None
    if scene.water is not None:
        w, _ = scene.sample(scene.water.astype(float), xs, ys)
        water = (w > 0.5) & inside
    return d, H_eff, water


def horizon_angles(scene: Scene, x: float, y: float, floors: list[int], az_deg: np.ndarray,
                   max_dist: float = 5000.0, exclude: np.ndarray | None = None) -> np.ndarray:
    """Угол горизонта (градусы над горизонталью), форма (len(floors), len(az))."""
    d, H, _ = _rays(scene, x, y, az_deg, max_dist, exclude)[:3]
    z = np.array([eye_z(scene, x, y, f) for f in floors])
    ang = np.arctan2(H[None, :, :] - z[:, None, None], d[None, None, :])
    return np.degrees(ang.max(axis=2))


# --------------------------------------------------------------------------
# Солнце
# --------------------------------------------------------------------------
def sunset_azimuth(lat: float, day: date) -> float | None:
    """Азимут захода солнца (приближение по склонению). None — полярный день/ночь."""
    n = day.timetuple().tm_yday
    decl = 23.44 * math.sin(2 * math.pi * (284 + n) / 365)
    c = math.sin(math.radians(decl)) / math.cos(math.radians(lat))
    if abs(c) > 1:
        return None
    return 360.0 - math.degrees(math.acos(c))


# --------------------------------------------------------------------------
# Видимость цели
# --------------------------------------------------------------------------
def sunset_visibility(scene: Scene, facade: Facade, floors: list[int], lat: float,
                      dates: list[date], exclude: np.ndarray | None = None,
                      fov: float = 120.0, max_dist: float = 5000.0,
                      sun_clear_deg: float = 0.25) -> dict[date, int | None]:
    """Для каждой даты — самый низкий этаж, с которого из окон фасада виден закат (или None)."""
    half = fov / 2
    azs = facade.azimuth + np.arange(-half, half + 1e-9, 1.0)
    ang = horizon_angles(scene, facade.x, facade.y, floors, azs, max_dist, exclude)
    out: dict[date, int | None] = {}
    for dt in dates:
        sa = sunset_azimuth(lat, dt)
        if sa is None:
            out[dt] = None
            continue
        diff = (sa - facade.azimuth + 180) % 360 - 180
        if abs(diff) > half:  # солнце садится вне поля зрения окна
            out[dt] = None
            continue
        i = int(np.argmin(np.abs(azs - (facade.azimuth + diff))))
        h = ang[:, max(0, i - 1): i + 2].max(axis=1)  # консервативно: ±1°
        ok = np.where(h <= sun_clear_deg)[0]
        out[dt] = floors[int(ok[0])] if len(ok) else None
    return out


def landmark_visible(scene: Scene, x: float, y: float, floor: int, tx: float, ty: float,
                     target_z: float, exclude: np.ndarray | None = None) -> dict:
    """Видна ли точка (tx, ty) на высоте target_z (абс.) из окна на этаже floor."""
    dist = math.hypot(tx - x, ty - y)
    az = math.degrees(math.atan2(tx - x, ty - y)) % 360
    d, H, _ = _rays(scene, x, y, np.array([az]), dist, exclude)[:3]
    z = eye_z(scene, x, y, floor)
    keep = d < dist - 2 * scene.res  # не считаем здание самой цели
    block = float(np.degrees(np.arctan2(H[0, keep] - z, d[keep])).max()) if keep.any() else -90.0
    to_target = math.degrees(math.atan2(target_z - z, dist))
    return {"visible": to_target > block, "target_angle": to_target, "blocking_angle": block, "dist_m": dist}


def sea_view_fraction(scene: Scene, facade: Facade, floor: int, fov: float = 120.0,
                      max_dist: float = 10_000.0, exclude: np.ndarray | None = None) -> float:
    """Доля сектора обзора окна, в которой видна вода (0..1)."""
    if scene.water is None:
        raise ValueError("в сцене нет маски воды")
    azs = facade.azimuth + np.arange(-fov / 2, fov / 2 + 1e-9, 1.0)
    d, H, water = _rays(scene, facade.x, facade.y, azs, max_dist, exclude)
    z = eye_z(scene, facade.x, facade.y, floor)
    ang = np.arctan2(H - z, d[None, :])
    prev = np.concatenate([np.full((len(azs), 1), -np.inf), np.maximum.accumulate(ang, axis=1)[:, :-1]], axis=1)
    return float(((ang > prev) & water).any(axis=1).mean())


# --------------------------------------------------------------------------
# Характеристика «закат из окон» в формате контракта {value, confidence, meta}
# --------------------------------------------------------------------------
def _month_ranges(months: list[int]) -> str:
    months = sorted(months)
    parts, start, prev = [], months[0], months[0]
    for m in months[1:] + [None]:
        if m is not None and m == prev + 1:
            prev = m
            continue
        parts.append(MONTHS_RU[start - 1] if start == prev else f"{MONTHS_RU[start - 1]}–{MONTHS_RU[prev - 1]}")
        if m is not None:
            start = prev = m
    return ", ".join(parts)


def sunset_feature(scene: Scene, facades: list[Facade], floors: list[int], lat: float,
                   exclude: np.ndarray | None = None, year: int = 2026,
                   height_scales: tuple[float, ...] = (0.8, 1.2)) -> dict:
    """Закат по фасадам и месяцам + оценка уверенности.

    Уверенность = доля (месяц, фасад), где результат меняется не более чем на
    1 этаж при ошибке высот зданий на ±20% (в OSM высоты часто оценочные:
    levels*3 м).
    """
    dates = [date(year, m, 15) for m in range(1, 13)]
    floors = sorted(floors)

    def run(sc: Scene):
        return {f.name: sunset_visibility(sc, f, floors, lat, dates, exclude) for f in facades}

    base = run(scene)
    scenarios = [run(scene.scaled(k)) for k in height_scales]

    value, conf_by_facade, texts = {}, {}, []
    for f in facades:
        res = base[f.name]
        by_month = {dt.month: fl for dt, fl in res.items()}
        value[f.name] = by_month
        cells = [(dt, res[dt]) for dt in dates]
        def same(a, b):  # совпадение с допуском в один этаж
            return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1)
        agree = [sum(same(sc[f.name][dt], v) for dt, v in cells) / len(cells) for sc in scenarios]
        conf_by_facade[f.name] = round(float(np.mean(agree)) if agree else 1.0, 2)

        groups = defaultdict(list)
        for m, fl in by_month.items():
            if fl is not None:
                groups[fl].append(m)
        if groups:
            desc = "; ".join(f"с {fl}-го этажа ({_month_ranges(ms)})" for fl, ms in sorted(groups.items()))
            texts.append(f"{f.name}: {desc}")

    return {
        "value": value,
        "confidence": round(float(np.mean(list(conf_by_facade.values()))), 2),
        "meta": {
            "summary": "; ".join(texts) if texts else "закат из окон не виден",
            "confidence_by_facade": conf_by_facade,
            "method": "DEM+buildings ray casting, refraction k=0.13, sun_clear=0.25°",
            "height_scales_tested": list(height_scales),
        },
    }


def sea_view_feature(scene: Scene, facades: list[Facade], floors: list[int], exclude: np.ndarray | None = None,
                     min_fraction: float = 0.2, height_scales: tuple[float, ...] = (0.8, 1.2),
                     max_dist: float = 10_000.0, max_probe_floors: int = 6) -> dict:
    """Вид на море по фасадам и этажам: доля сектора обзора, где видна вода.

    Считается на нескольких пробных этажах (первый, последний и между ними).
    Уверенность — доля (фасад, этаж), где доля обзора при ошибке высот зданий
    на ±20% меняется не больше чем на 0.1.
    """
    floors = sorted(floors)
    idx = np.unique(np.linspace(0, len(floors) - 1, min(len(floors), max_probe_floors)).round().astype(int))
    probe = [floors[i] for i in idx]

    def run(sc: Scene):
        return {f.name: {fl: sea_view_fraction(sc, f, fl, max_dist=max_dist, exclude=exclude) for fl in probe}
                for f in facades}

    base = run(scene)
    scenarios = [run(scene.scaled(k)) for k in height_scales]
    value, texts, conf_cells = {}, [], []
    for f in facades:
        fr = base[f.name]
        value[f.name] = {str(fl): round(v, 2) for fl, v in fr.items()}
        conf_cells += [abs(sc[f.name][fl] - v) <= 0.1 for sc in scenarios for fl, v in fr.items()]
        first = next((fl for fl in probe if fr[fl] >= min_fraction), None)
        if first is not None:
            texts.append(f"{f.name}: с {first}-го этажа (до {max(fr.values()):.0%} обзора)")
    
    return {
        "value": value,
        "confidence": round(float(np.mean(conf_cells)) if conf_cells else 1.0, 2),
        "meta": {"summary": "; ".join(texts) if texts else "вид на море не выявлен",
                 "probe_floors": probe, "min_fraction": min_fraction,
                 "method": "ray casting to water mask, DEM+buildings, refraction k=0.13"},
    }
