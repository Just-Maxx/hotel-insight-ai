"""Оценка шума у окон отеля по дорогам, железной дороге, ВПП и ночным заведениям.

Модель инженерная, не сертифицированная и НЕ откалиброванная по замерам:
- каждый источник = энергия, распределённая по точкам вдоль линии (шаг ~10 м);
  суммирование точек даёт для линии затухание 1/r (−3 дБ на удвоение расстояния),
  для точечного источника — 1/r² (−6 дБ);
- экранирование зданиями: по лучу источник→окно ищем самое «заслоняющее» препятствие
  в нашей сцене (рельеф+здания), потери по Маекаве: IL = 10·lg(3+20N), N = 2δ/λ, не более 20 дБ;
- собственная стена: источники «за спиной» фасада (угол к нормали > 90°) получают −12 дБ;
- опорные уровни L10 (Leq на 10 м от оси) — типовые значения по классу дороги, взятые как
  эвристика. Поэтому уверенность ограничена сверху 0.7, а выводы — это ранжирование, а не измерение.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .visibility import Facade, Scene, eye_z

D0 = 10.0                       # опорное расстояние, м
LAMBDA = 343.0 / 500.0          # длина волны на 500 Гц, м
WALL_PENALTY_DB = 12.0
BACKGROUND_DB = 30.0            # фоновый уровень тихого района
MAX_IL_DB = 20.0
CONFIDENCE_CEILING = 0.7        # потолок для некалиброванной модели
SHIFT_DB = 4.0                  # проверка устойчивости класса шума к ошибке опорных уровней

# дБ(А), Leq днём на 10 м от оси — ЭВРИСТИЧЕСКИЕ типовые значения
L10 = {
    "highway:motorway": 78, "highway:trunk": 74, "highway:primary": 70, "highway:secondary": 66,
    "highway:tertiary": 62, "highway:unclassified": 56, "highway:residential": 54,
    "highway:living_street": 46, "highway:service": 48,
    "railway:rail": 70, "railway:light_rail": 64, "railway:tram": 62,
    "aeroway:runway": 75,
    "amenity:nightclub": 78, "amenity:bar": 68, "amenity:pub": 66,
}
# ночью поток транспорта обычно ниже на 8-10 дБ; заведения, наоборот, на пике
NIGHT_OFFSET = {"highway": -8.0, "railway": -4.0, "aeroway": -10.0, "amenity": 0.0}
CLASSES = [(40.0, "тихо"), (50.0, "умеренно"), (55.0, "шумно")]  # по ночному уровню; 40 дБ — рекомендация ВОЗ


def noise_class(db: float) -> str:
    for limit, name in CLASSES:
        if db < limit:
            return name
    return "очень шумно"


class NoiseModel:
    def __init__(self, sources: dict, step: float = 10.0):
        X, Y, E0, KIND, ELEM = [], [], [], [], []
        self.elems: list[tuple[str, str]] = []
        kinds = ["highway", "railway", "aeroway", "amenity"]

        for ln in sources["lines"]:
            if ln["cls"] not in L10 or len(ln["pts"]) < 2:
                continue
            p = ln["pts"]
            a, b = p[:-1], p[1:]
            seg = np.hypot(*(b - a).T)
            ok = seg > 1e-6
            a, b, seg = a[ok], b[ok], seg[ok]
            if not len(seg):
                continue
            n = np.maximum(1, np.ceil(seg / step)).astype(int)
            rep = np.repeat(np.arange(len(seg)), n)
            j = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            t = (j + 0.5) / n[rep]
            pts = a[rep] + (b - a)[rep] * t[:, None]
            delta = (seg / n)[rep]
            X.append(pts[:, 0]); Y.append(pts[:, 1])
            E0.append(10 ** (L10[ln["cls"]] / 10) * D0 / math.pi * delta)  # линия: ∑ даёт 1/r
            KIND += [kinds.index(ln["cls"].split(":")[0])] * len(pts)
            self.elems.append((ln["cls"], ln["name"]))
            ELEM += [len(self.elems) - 1] * len(pts)

        for pt in sources["points"]:
            if pt["cls"] not in L10:
                continue
            X.append(np.array([pt["xy"][0]])); Y.append(np.array([pt["xy"][1]]))
            E0.append(np.array([10 ** (L10[pt["cls"]] / 10) * D0 ** 2]))  # точка: 1/r²
            KIND.append(kinds.index("amenity"))
            self.elems.append((pt["cls"], pt["name"]))
            ELEM.append(len(self.elems) - 1)

        self.empty = not X
        if self.empty:
            return
        self.X, self.Y = np.concatenate(X), np.concatenate(Y)
        self.E0 = np.concatenate(E0)
        self.kind = np.array(KIND)
        self.elem = np.array(ELEM)
        self.night_gain = np.array([10 ** (NIGHT_OFFSET[k] / 10) for k in kinds])[self.kind]
        self.tree = cKDTree(np.column_stack([self.X, self.Y]))

    def levels(self, scene: Scene, x: float, y: float, normal_az: float, z_list: list[float],
               exclude: np.ndarray | None = None, max_dist: float = 600.0) -> list[dict]:
        """Для каждой высоты окна — день/ночь (дБ(А)) и главный источник."""
        idx = [] if self.empty else self.tree.query_ball_point([x, y], max_dist)
        if not idx:
            bg = round(BACKGROUND_DB, 1)
            return [{"day": bg, "night": bg, "top": None} for _ in z_list]
        idx = np.array(idx)
        dx, dy = self.X[idx] - x, self.Y[idx] - y
        dist = np.maximum(np.hypot(dx, dy), 1.0)
        r = np.maximum(dist, 3.0)
        az = np.degrees(np.arctan2(dx, dy))
        behind = np.abs((az - normal_az + 180) % 360 - 180) > 90

        K = int(math.ceil(max_dist / scene.res))
        t = (np.arange(1, K) / K)[None, :]
        rx, ry = x + dx[:, None] * t, y + dy[:, None] * t
        d_k = dist[:, None] * t
        H, inside = scene.sample(scene.surface, rx, ry)
        if exclude is not None:  # собственное здание не экранирует (учтено штрафом стены)
            ex, _ = scene.sample(exclude.astype(float), rx, ry)
            g, _ = scene.sample(scene.dem, rx, ry)
            H = np.where(ex > 0.5, g, H)
        # клетки у самого источника не считаем препятствием: источник стоит на дороге, а не внутри дома
        H = np.where(inside & (d_k < dist[:, None] - 0.75 * scene.res), H, -1e6)
        zs = scene.sample(scene.dem, self.X[idx], self.Y[idx])[0] + 0.5
        rows = np.arange(len(idx))

        out = []
        for zr in z_list:
            ang = (H - zr) / np.maximum(d_k, 1e-3)
            k = ang.argmax(axis=1)
            blocked = ang[rows, k] > (zs - zr) / dist
            he, d1 = H[rows, k], d_k[rows, k]
            d2 = np.maximum(dist - d1, 0.0)
            delta = np.sqrt(d1 ** 2 + (he - zr) ** 2) + np.sqrt(d2 ** 2 + (he - zs) ** 2) \
                - np.sqrt(dist ** 2 + (zr - zs) ** 2)
            il = np.where(blocked, 10 * np.log10(3 + 20 * np.maximum(2 * delta / LAMBDA, 0)), 0.0)
            il = np.minimum(il, MAX_IL_DB) + np.where(behind, WALL_PENALTY_DB, 0.0)
            E = self.E0[idx] / r ** 2 * 10 ** (-il / 10)
            En = E * self.night_gain[idx]
            bg = 10 ** (BACKGROUND_DB / 10)
            by_elem = np.bincount(self.elem[idx], weights=En, minlength=len(self.elems))
            top = int(by_elem.argmax())
            out.append({
                "day": round(10 * math.log10(E.sum() + bg), 1),
                "night": round(10 * math.log10(En.sum() + bg), 1),
                "top": {"cls": self.elems[top][0], "name": self.elems[top][1]} if by_elem[top] > 0 else None,
            })
        return out


def noise_feature(model: NoiseModel, scene: Scene, facades: list[Facade], floors: list[int],
                  exclude: np.ndarray | None = None, max_probe_floors: int = 3) -> dict:
    """Характеристика «тишина» в формате {value, confidence, meta}: ночной/дневной уровень по фасадам и этажам."""
    floors = sorted(floors)
    idx = np.unique(np.linspace(0, len(floors) - 1, min(len(floors), max_probe_floors)).round().astype(int))
    probe = [floors[i] for i in idx]
    value, cells = {}, []
    for f in facades:
        zs = [eye_z(scene, f.x, f.y, fl) for fl in probe]
        res = model.levels(scene, f.x, f.y, f.azimuth, zs, exclude)
        value[f.name] = {str(fl): {"day": r["day"], "night": r["night"], "class": noise_class(r["night"]),
                                   "top_source": r["top"]} for fl, r in zip(probe, res)}
        cells += [r["night"] for r in res]

    stable = [noise_class(n - SHIFT_DB) == noise_class(n) == noise_class(n + SHIFT_DB) for n in cells]
    best = min(((v["night"], name, fl) for name, d in value.items() for fl, v in d.items()))
    worst = max(((v["night"], name, fl) for name, d in value.items() for fl, v in d.items()))
    src = value[worst[1]][worst[2]]["top_source"]
    src_txt = f", главный источник: {src['name'] or src['cls']}" if src else ""
    return {
        "value": value,
        "confidence": round(CONFIDENCE_CEILING * float(np.mean(stable)), 2),
        "meta": {
            "summary": (f"тише всего: {best[1]}, {best[2]}-й этаж ({best[0]} дБ ночью, {noise_class(best[0])}); "
                        f"шумнее всего: {worst[1]}, {worst[2]}-й этаж ({worst[0]} дБ{src_txt})"),
            "probe_floors": probe,
            "method": "line/point sources, Maekawa barrier loss on DEM+buildings, wall penalty 12 dB",
            "calibration": "none: типовые опорные уровни по классу дороги (эвристика)",
        },
    }
