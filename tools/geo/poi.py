"""Характеристики по точкам интереса: пешая доступность, что работает ночью, детская инфраструктура."""
from __future__ import annotations

import re

import numpy as np

from .walking import Field, WalkGraph

NIGHT_MINUTE = 120  # 02:00
_INTERVAL = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


def is_open_night(oh: str | None, minute: int = NIGHT_MINUTE) -> bool | None:
    """Открыто ли в `minute` (минуты от полуночи) каждый день. None — не удалось определить.

    Понимаем только однозначное: `24/7`, `Mo-Su ...`, интервалы без дней, в том числе через полночь.
    Любые ограничения по дням, праздникам, `off`, `sunrise` — «неизвестно», а не догадка.
    """
    if not oh or not oh.strip():
        return None
    s = oh.strip()
    if s == "24/7":
        return True
    verdict = None
    for rule in (r.strip() for r in s.split(";") if r.strip()):
        m = re.fullmatch(r"(?:Mo-Su\s+)?((?:\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})(?:\s*,\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})*)", rule)
        if not m:
            return None
        hit = False
        for h1, m1, h2, m2 in _INTERVAL.findall(m.group(1)):
            a, b = int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
            hit |= (a <= minute < b) if a <= b else (minute >= a or minute < b)
        verdict = hit if verdict is None else (verdict or hit)
    return verdict


def build_target_sets(pois: list[dict], beach_pts: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Наборы целей для полей достижимости + статистика по неизвестным часам работы."""
    def xy(items):
        return np.array([[p["x"], p["y"]] for p in items], dtype=float).reshape(-1, 2)

    sets = {"beach": np.asarray(beach_pts, dtype=float).reshape(-1, 2)}
    for kind in ("bus_stop", "station", "playground", "kids_medical"):
        sets[kind] = xy([p for p in pois if p["kind"] == kind])
    stats = {}
    for kind, name in (("pharmacy", "pharmacy_night"), ("grocery", "grocery_night"), ("food", "food_night")):
        items = [p for p in pois if p["kind"] == kind]
        verdicts = [is_open_night(p.get("opening_hours")) for p in items]
        sets[name] = xy([p for p, v in zip(items, verdicts) if v])
        stats[kind] = {"total": len(items), "open_night": sum(bool(v) for v in verdicts),
                       "unknown": sum(v is None for v in verdicts)}
    return sets, stats


def _conf(found: list[dict | None]) -> float:
    got = [d for d in found if d]
    if not got:
        return 0.3  # ничего не найдено: чаще всего неполнота OSM, а не реальное отсутствие
    return round(float(np.mean([0.9 if d["snap_m"] <= 60 else 0.7 for d in got])), 2)


def _fmt(d: dict | None, label: str) -> str | None:
    if not d:
        return None
    extra = f", подъём {d['ascent_m']:.0f} м" if d["ascent_m"] >= 5 else ""
    cross = f", пересечений крупных дорог: {d['busy_crossings']}" if d["busy_crossings"] else ""
    return f"{label} — {d['minutes']:.0f} мин ({d['distance_m']} м{extra}{cross})"


def walk_access_feature(wg: WalkGraph, fields: dict[str, Field | None], x: float, y: float) -> dict:
    value = {k: wg.describe(fields.get(k), x, y) for k in ("beach", "bus_stop", "station")}
    parts = [p for p in (_fmt(value["beach"], "пляж"), _fmt(value["bus_stop"], "остановка"),
                         _fmt(value["station"], "ж/д станция")) if p]
    return {"value": value, "confidence": _conf(list(value.values())),
            "meta": {"summary": "; ".join(parts) if parts else "цели в пешей доступности не найдены",
                     "method": "Dijkstra по пешей сети OSM, функция Тоблера (уклон из DEM)"}}


def _within(pois, x, y, r):
    return [p for p in pois if np.hypot(p["x"] - x, p["y"] - y) <= r]


def night_amenities_feature(wg: WalkGraph, fields: dict[str, Field | None], pois: list[dict],
                            stats: dict, x: float, y: float) -> dict:
    value = {"pharmacy": wg.describe(fields.get("pharmacy_night"), x, y),
             "grocery": wg.describe(fields.get("grocery_night"), x, y),
             "food": wg.describe(fields.get("food_night"), x, y)}
    near = [p for p in _within(pois, x, y, 1000) if p["kind"] in ("pharmacy", "grocery", "food")]
    unknown = sum(is_open_night(p.get("opening_hours")) is None for p in near)
    value["night_open_within_500m"] = sum(
        is_open_night(p.get("opening_hours")) is True for p in _within(pois, x, y, 500)
        if p["kind"] in ("pharmacy", "grocery", "food"))
    value["unknown_hours_share_1km"] = round(unknown / len(near), 2) if near else None
    parts = [p for p in (_fmt(value["pharmacy"], "круглосуточная аптека"),
                         _fmt(value["grocery"], "ночной магазин"), _fmt(value["food"], "кафе ночью")) if p]
    conf = _conf([value["pharmacy"], value["grocery"], value["food"]])
    if value["unknown_hours_share_1km"] and value["unknown_hours_share_1km"] > 0.5:
        conf = round(min(conf, 0.5), 2)  # у большинства точек в OSM нет часов работы
    return {"value": value, "confidence": conf,
            "meta": {"summary": "; ".join(parts) if parts else "ночных точек рядом не найдено",
                     "method": "opening_hours из OSM (только однозначные записи) + пеший маршрут",
                     "note": "точки без часов работы в OSM не учитываются; см. unknown_hours_share_1km"}}


def kids_feature(wg: WalkGraph, fields: dict[str, Field | None], pois: list[dict], x: float, y: float) -> dict:
    value = {"playground": wg.describe(fields.get("playground"), x, y),
             "playgrounds_within_300m": len(_within([p for p in pois if p["kind"] == "playground"], x, y, 300)),
             "pedestrian_streets_m_300": round(wg.pedestrian_length_near(x, y, 300.0)),
             "kids_medical": wg.describe(fields.get("kids_medical"), x, y)}
    parts = [p for p in (_fmt(value["playground"], "детская площадка"),
                         _fmt(value["kids_medical"], "детская медицина")) if p]
    if value["pedestrian_streets_m_300"]:
        parts.append(f"пешеходных улиц в 300 м: {value['pedestrian_streets_m_300']} м")
    return {"value": value, "confidence": _conf([value["playground"], value["kids_medical"]]),
            "meta": {"summary": "; ".join(parts) if parts else "детской инфраструктуры рядом не найдено",
                     "method": "OSM: leisure=playground, детские медучреждения по тегам и названию, highway=pedestrian"}}
