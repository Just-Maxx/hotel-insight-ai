"""Поиск, ранжирование и сравнение отелей по сохранённым характеристикам (без вызова LLM)."""
import json

from agents.orchestrator import Plan, Task, register
from core.labels import (DEFAULT_WALK_MINUTES, WALK_TARGET_GENITIVE, feature_label, requirement_label)
from core.registry import load_registry
from storage import db
from storage.resolver import resolve, source_class, source_label

WEIGHT = {"must": 3.0, "nice": 1.0}
MISS_THRESHOLD = 0.2  # score ниже — требование считаем невыполненным


def _item(feature: str, importance: str, res: dict) -> dict:
    fact = res["fact"]
    return {"feature": feature, "target": None, "label": feature_label(feature), "importance": importance,
            "score": round(fact["score"], 2), "ok": fact["score"] >= MISS_THRESHOLD,
            "confidence": round(res["confidence"], 2), "summary": fact["summary"], "source": fact["source"],
            "source_label": source_label(fact["source"]), "fetched_at": fact["fetched_at"][:10],
            "stale": res["stale"]}


def walk_minutes_score(minutes: float | None, max_minutes: float) -> float:
    """Чем ближе, тем лучше: до половины порога — 1.0, на пороге — 0.6, на двойном пороге — 0."""
    if minutes is None:
        return 0.0
    half = max_minutes / 2
    if minutes <= half:
        return 1.0
    if minutes <= max_minutes:
        return round(1.0 - 0.4 * (minutes - half) / half, 3)
    return round(max(0.0, 0.6 * (1 - (minutes - max_minutes) / max_minutes)), 3)


def geo_walk_value(facts: list[dict]) -> dict | None:
    """Время пешком до пляжа/остановки/станции из расчёта по карте (последний геофакт с данными)."""
    geo = [f for f in facts if f["feature"] == "walk_access" and source_class(f["source"]) == "geo"]
    for f in sorted(geo, key=lambda f: f["fetched_at"], reverse=True):
        try:
            v = json.loads(f["value_json"]) if f.get("value_json") else None
        except (TypeError, ValueError):
            v = None
        if isinstance(v, dict) and any(k in v for k in ("beach", "bus_stop", "station")):
            return v
    return None


def _walk_summary(d: dict | None, target: str, limit: float) -> str:
    if not d:
        return f"{WALK_TARGET_GENITIVE[target]} в пешей доступности не найдено"
    climb = f", подъём {d['ascent_m']:.0f} м" if d.get("ascent_m", 0) >= 5 else ""
    return f"{d['minutes']:.0f} мин пешком ({d['distance_m']} м{climb}) — допустимо до {limit:.0f} мин"


def _walk_item(r, res: dict, walk_value: dict | None) -> dict | None:
    """Пожелание «пляж/остановка/станция в N минутах пешком»: считаем по нужной цели, а не по сумме всех."""
    if not walk_value or r.target not in walk_value:
        return None
    limit = float(r.max_minutes or DEFAULT_WALK_MINUTES)
    d = walk_value.get(r.target)
    minutes = d["minutes"] if d else None
    fact = res["fact"]
    return {"feature": "walk_access", "target": r.target, "label": requirement_label("walk_access", r.target),
            "importance": r.importance, "score": walk_minutes_score(minutes, limit),
            "ok": minutes is not None and minutes <= limit, "confidence": round(res["confidence"], 2),
            "summary": _walk_summary(d, r.target, limit), "source": fact["source"],
            "source_label": "расчёт по карте", "fetched_at": fact["fetched_at"][:10], "stale": res["stale"],
            "minutes": minutes, "max_minutes": int(limit)}


def rank_hotels(plan: Plan, top_k: int = 3, con=None) -> dict:
    con = con or db.connect()
    hotels = db.list_hotels(con)
    if plan.location:
        local = [h for h in hotels if plan.location.lower() in (h["city"] or "").lower()]
        if not local:
            return {"ranked": [], "note": f"В демо-данных нет отелей в регионе «{plan.location}»."}
        hotels = local

    reqs = [r for r in plan.requirements if r.feature != "other"]
    unsupported = [r.detail for r in plan.requirements if r.feature == "other"]
    total_w = sum(WEIGHT[r.importance] for r in reqs) or 1.0
    registry = load_registry()

    ranked = []
    for h in hotels:
        facts = db.get_facts(con, h["id"])
        resolved = resolve(facts)
        walk_value = geo_walk_value(facts)
        points, matched, misses, gaps, warnings = 0.0, [], [], [], []
        for r in reqs:
            res = resolved.get(r.feature)
            if not res or res["fact"]["score"] is None:
                gaps.append({"feature": r.feature, "importance": r.importance,
                             "title": requirement_label(r.feature, r.target)})  # -> вопрос отельеру
                continue
            if r.target:
                item = _walk_item(r, res, walk_value)
                if item is None:
                    gaps.append({'feature': r.feature, 'importance': r.importance,
                                 'title': requirement_label(r.feature, r.target, r.max_minutes)})
                    continue  # Общая оценка доступности не подтверждает конкретные «10 минут до пляжа».
            else:
                item = _item(r.feature, r.importance, res)
            points += WEIGHT[r.importance] * item["score"] * item["confidence"]
            (matched if item["ok"] else misses).append(item)
            if res["conflict"]:
                warnings.append(f"«{item['label']}»: источники данных расходятся — стоит проверить")
            if res["stale"]:
                warnings.append(f"«{item['label']}»: данные давно не обновлялись")
        # must без данных или с провалом — жёсткий штраф
        must_fail = sum(1 for m in misses if m["importance"] == "must") + \
                    sum(1 for g in gaps if g["importance"] == "must")
        ranked.append({"hotel": h["name"], "hotel_id": h["id"], "lat": h["lat"], "lon": h["lon"],
                       "floors": h["floors"], "score": round(points / total_w, 3), "must_fail": must_fail,
                       "matched": matched, "misses": misses, "gaps": gaps, "warnings": warnings})

    ranked.sort(key=lambda x: (x["must_fail"], -x["score"]))
    return {"ranked": ranked[:top_k], "unsupported": unsupported, "total_hotels": len(hotels)}


def compare_hotels(hotel_ids: list[str], features: list[str] | None = None, con=None) -> dict:
    """Таблица «характеристика × отель» с источником, уверенностью и пометками. Победитель по строке — max score."""
    con = con or db.connect()
    registry = load_registry()
    features = features or list(registry)
    rows, names = [], {}
    resolved = {}
    for hid in hotel_ids:
        h = db.get_hotel(con, hid)
        if not h:
            raise ValueError(f"Неизвестный отель: {hid}")
        names[hid] = h["name"]
        resolved[hid] = resolve(db.get_facts(con, hid))
    for f in features:
        cells, best = {}, None
        for hid in hotel_ids:
            res = resolved[hid].get(f)
            if not res:
                cells[hid] = None
                continue
            fact = res["fact"]
            cells[hid] = {"score": fact["score"], "confidence": round(res["confidence"], 2),
                          "summary": fact["summary"], "source": source_label(fact["source"]),
                          "conflict": res["conflict"], "stale": res["stale"]}
            if fact["score"] is not None and (best is None or fact["score"] > cells[best]["score"]):
                best = hid
        top = [hid for hid, c in cells.items() if c and c["score"] is not None and best
               and abs(c["score"] - cells[best]["score"]) < 0.05]
        rows.append({"feature": f, "title": registry.get(f, {}).get("title", f), "cells": cells, "best": top})
    return {"hotels": names, "rows": rows}


# Строки простого сравнения: (подпись, характеристика, цель дороги или None)
COMPARE_ROWS = [("Тишина", "quiet", None), ("Вид на закат", "sunset_view", None), ("Вид на море", "sea_view", None),
                ("Пляж в 15 минутах пешком", "walk_access", "beach"), ("Можно с питомцем", "pets_allowed", None),
                ("Работает ночью", "night_amenities", None), ("Для детей", "kids_infrastructure", None)]


def level_of(score: float | None) -> str:
    return "unknown" if score is None else "good" if score >= 0.6 else "mid" if score >= 0.3 else "bad"


def compare_simple(hotel_ids: list[str], con=None) -> dict:
    """Сравнение для человека: на каждую пару «параметр × отель» — только уровень good/mid/bad/unknown
    и короткая подсказка. Все остальные детали (уверенность, источники) остаются в расчёте, но не показываются."""
    con = con or db.connect()
    registry = load_registry()
    hotels, resolved, walk = {}, {}, {}
    for hid in hotel_ids:
        h = db.get_hotel(con, hid)
        if not h:
            raise ValueError(f"Неизвестный отель: {hid}")
        hotels[hid] = h["name"]
        facts = db.get_facts(con, hid)
        resolved[hid], walk[hid] = resolve(facts), geo_walk_value(facts)
    rows = [(label, feat, target) for label, feat, target in COMPARE_ROWS if feat in registry]
    known = {feat for _, feat, _ in rows}  # характеристики, добавленные в кабинете, появляются внизу сами
    rows += [(spec["title"], feat, None) for feat, spec in registry.items() if feat not in known and feat != "walk_access"]
    out_rows = []
    for label, feat, target in rows:
        cells = {}
        for hid in hotel_ids:
            res = resolved[hid].get(feat)
            if target:  # пляж: время пешком из расчёта, порог 15 минут
                d = (walk[hid] or {}).get(target)
                if walk[hid] is None:
                    cells[hid] = {"level": "unknown", "hint": "нет данных"}
                else:
                    score = walk_minutes_score(d["minutes"] if d else None, DEFAULT_WALK_MINUTES)
                    cells[hid] = {"level": level_of(score),
                                  "hint": f"{d['minutes']:.0f} мин пешком" if d else "в пешей доступности не найден"}
                continue
            if not res or res["fact"]["score"] is None:
                cells[hid] = {"level": "unknown", "hint": "нет данных"}
                continue
            lvl = level_of(res["fact"]["score"])
            hint = res["fact"]["summary"]
            cells[hid] = {"level": lvl, "hint": hint if len(hint) <= 110 else hint[:109].rstrip() + "…"}
        out_rows.append({"label": label, "feature": feat, "cells": cells})
    totals = {hid: sum(r["cells"][hid]["level"] == "good" for r in out_rows) for hid in hotel_ids}
    known_counts = {hid: sum(r["cells"][hid]["level"] != "unknown" for r in out_rows) for hid in hotel_ids}
    best = max(totals.values()) if totals else 0
    return {"hotels": hotels, "rows": out_rows, "totals": totals, "known": known_counts,
            "best": [hid for hid, t in totals.items() if t == best and best > 0], "rows_total": len(out_rows)}


@register("search_agent")
def search_agent(task: Task, plan: Plan) -> dict:
    return rank_hotels(plan)
