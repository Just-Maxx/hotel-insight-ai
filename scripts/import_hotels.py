"""Мост между геоконвейером и нашей БД: data/demo/hotels.json -> data/demo/facts.db.

Формат записи (его выдаёт scripts/build_demo.py и pipeline.compute_hotel_features):
  {"hotel_id", "name", "lat", "lon", "levels", "features": {имя: {value, confidence, meta{summary[,score]},
                                                                   source, fetched_at, ttl_days}}}
score (0..1, «насколько подходит») нужен для ранжирования. Если конвейер не отдал meta.score,
считаем сами по SCORERS (формулы для всех шести характеристик ниже). Неизвестная характеристика
сохраняется без score, и поиск честно покажет её как «данных нет».

    py scripts/import_hotels.py data/demo/hotels.json --city Сочи
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import db


def sunset_score(value: dict, levels: int) -> float | None:
    """Доля месяцев, когда закат виден с нижней половины этажей (лучший фасад)."""
    half = max(1, -(-levels // 2))
    try:
        return max(sum(v is not None and v <= half for v in months.values()) for months in value.values()) / 12
    except (ValueError, AttributeError, TypeError):
        return None


def sea_score(value, levels: int) -> float | None:
    """Ожидаем {фасад: {этаж: доля}}: берём лучшую долю на верхнем этаже."""
    try:
        return max(max(float(f) for f in floors.values()) for floors in value.values())
    except (ValueError, AttributeError, TypeError):
        return None


def _ramp(x, best: float, worst: float) -> float:
    """1.0 при x <= best, 0.0 при x >= worst, между — линейно. None (не найдено) -> 0."""
    if x is None:
        return 0.0
    return max(0.0, min(1.0, (worst - float(x)) / (worst - best)))


def _minutes(d) -> float | None:
    return d.get("minutes") if isinstance(d, dict) else None


def quiet_score(value: dict, levels: int) -> float | None:
    """Самая тихая сторона/этаж (ночью): <=35 дБ -> 1, >=55 дБ -> 0. Худшее место — в summary."""
    try:
        return _ramp(min(v["night"] for d in value.values() for v in d.values()), 35.0, 55.0)
    except (ValueError, KeyError, AttributeError, TypeError):
        return None


def walk_score(value: dict, levels: int) -> float | None:
    """Пляж важнее (вес 2), затем остановка и станция: <=5 мин -> 1, >=30 мин -> 0."""
    try:
        parts = [(2.0, value["beach"]), (1.0, value["bus_stop"]), (1.0, value["station"])]
        return sum(w * _ramp(_minutes(d), 5, 30) for w, d in parts) / sum(w for w, _ in parts)
    except (KeyError, TypeError):
        return None


def night_score(value: dict, levels: int) -> float | None:
    """Аптека, магазин, кафе с подтверждённой ночной работой: <=10 мин -> 1, >=40 -> 0."""
    try:
        return sum(_ramp(_minutes(value[k]), 10, 40) for k in ("pharmacy", "grocery", "food")) / 3
    except (KeyError, TypeError):
        return None


def kids_score(value: dict, levels: int) -> float | None:
    try:
        s = 0.5 * _ramp(_minutes(value["playground"]), 5, 25)
        s += 0.2 * (1.0 if value.get("playgrounds_within_300m") else 0.0)
        s += 0.15 * (1.0 if value.get("pedestrian_streets_m_300") else 0.0)
        s += 0.15 * _ramp(_minutes(value["kids_medical"]), 15, 45)
        return s
    except (KeyError, TypeError):
        return None


SCORERS = {"sunset_view": sunset_score, "sea_view": sea_score, "quiet": quiet_score,
           "walk_access": walk_score, "night_amenities": night_score, "kids_infrastructure": kids_score}


def score_of(name: str, feat: dict, levels: int) -> float | None:
    if (feat.get("meta") or {}).get("incomplete"):
        return None
    s = (feat.get("meta") or {}).get("score")
    if s is not None:
        return max(0.0, min(1.0, float(s)))
    fn = SCORERS.get(name)
    return fn(feat.get("value"), levels) if fn and feat.get("value") is not None else None


def import_records(records: list[dict], city: str, con=None) -> dict:
    con = con or db.connect()
    stats = {"hotels": 0, "facts": 0, "no_score": []}
    for r in records:
        levels = r.get("levels") or 1
        db.upsert_hotel(con, r["hotel_id"], r["name"], r.get("city") or city, r["lat"], r["lon"], levels)
        stats["hotels"] += 1
        for name, f in r["features"].items():
            sc = score_of(name, f, levels)
            if sc is None and name not in stats["no_score"]:
                stats["no_score"].append(name)
            db.add_fact(con, r["hotel_id"], name, value=f.get("value"), score=sc,
                        confidence=float(f.get("confidence", 0.5)),
                        summary=(f.get("meta") or {}).get("summary", ""), source=f.get("source", "unknown"),
                        ttl_days=int(f.get("ttl_days", 180)), fetched_at=f.get("fetched_at"))
            stats["facts"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--city", default="Сочи")
    ap.add_argument("--keep", action="store_true", help="не очищать БД перед импортом")
    a = ap.parse_args()
    if not a.keep:
        db.reset()
    st = import_records(json.loads(Path(a.json_path).read_text(encoding="utf-8")), a.city)
    print(f"Импортировано: отелей {st['hotels']}, фактов {st['facts']} -> {db.DB_PATH}")
    if st["no_score"]:
        print("Без score (нужен meta.score от конвейера):", ", ".join(st["no_score"]))


if __name__ == "__main__":
    main()
