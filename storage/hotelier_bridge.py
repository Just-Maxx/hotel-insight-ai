"""Мост между агентом отельера и БД: ответы, подтверждения и споры становятся фактами с источником.

Принципы:
- слова отельера не затирают расчёт по геоданным: это отдельный факт с другим source,
  а разбираться с расхождением будет storage/resolver.py (и пометит needs_review);
- confidence — условная вера в источник, а не вероятность: человек 0.6, сайт 0.5,
  имитация (демо) 0.5 с отдельным source, подтверждение расчёта человеком 0.85;
- повторный ответ того же источника заменяет прежний, а не копится в БД.
"""
from __future__ import annotations

from agents.hotelier_agent import QUESTIONS
from schemas.hotelier import HotelierSession, PetPolicy
from storage import db
from storage.resolver import resolve

CONFIDENCE = {"hotelier_human": 0.6, "hotel_website": 0.5, "hotelier_simulator": 0.5,
              "hotelier_confirmed": 0.85, "hotelier_dispute": 0.6}
TTL_DAYS = {"hotelier_human": 180, "hotel_website": 90, "hotelier_simulator": 30,
            "hotelier_confirmed": 365, "hotelier_dispute": 180}
DISPUTE_LEVELS = {"нет": 0.0, "частично": 0.5, "да": 1.0}


def pet_summary(p: PetPolicy) -> str:
    if p.allowed is None:
        return "правила проживания с животными не определены"
    if p.allowed is False:
        return "с животными не принимают"
    animals = {"dog": "собаки", "cat": "кошки", "other": "другие животные"}
    parts = [", ".join(animals[a] for a in p.animals) or "животные"]
    parts.append("вес не ограничен" if p.weight_unlimited else
                 f"до {p.max_weight_kg:g} кг" if p.max_weight_kg is not None else "предел веса не назван")
    if p.fee_rub_per_day is not None:
        parts.append("без доплаты" if p.fee_rub_per_day == 0 else f"доплата {p.fee_rub_per_day:g} ₽/сутки")
    if p.needs_approval is not None:
        parts.append("нужно согласование" if p.needs_approval else "согласование не нужно")
    return ", ".join(parts)


def pet_score(p: PetPolicy) -> float | None:
    """0 — нельзя; 1 — можно без ограничений; ограничения по весу/согласованию/доплате снижают оценку."""
    if p.allowed is None:
        return None
    if not p.allowed:
        return 0.0
    s = 1.0
    if not p.weight_unlimited and (p.max_weight_kg is None or p.max_weight_kg < 15):
        s -= 0.25 if p.max_weight_kg is not None else 0.35  # малый предел веса / вес не назван
    if p.needs_approval:
        s -= 0.1
    if p.fee_rub_per_day and p.fee_rub_per_day >= 1000:
        s -= 0.1
    return round(max(s, 0.1), 2)


def store_session(con, session: HotelierSession) -> list[dict]:
    """Сохраняет итоги диалога. Возвращает [{feature, saved, summary}] для показа в интерфейсе."""
    report = []
    for fact in session.facts:
        if fact.status != "reported":
            report.append({"feature": fact.feature, "saved": False,
                           "summary": {"conflict": "ответ противоречив", "error": "ошибка разбора"}.get(
                               fact.status, "сведений нет")})
            continue
        if fact.feature == "pets_allowed" and fact.pet_policy is not None:
            score, summary = pet_score(fact.pet_policy), pet_summary(fact.pet_policy)
            value = fact.pet_policy.model_dump()
        else:
            score = fact.suitability
            summary = "; ".join(c.detail for c in fact.claims) or "со слов источника"
            value = {"claims": [c.model_dump() for c in fact.claims]}
        value["verified"] = False
        value["source_url"] = session.source_url
        db.delete_facts(con, session.hotel_id, fact.feature, session.source)
        db.add_fact(con, session.hotel_id, fact.feature, value=value, score=score,
                    confidence=CONFIDENCE[session.source], summary=summary, source=session.source,
                    ttl_days=TTL_DAYS[session.source], fetched_at=session.fetched_at)
        report.append({"feature": fact.feature, "saved": True, "summary": summary, "score": score})
    return report


def _best_geo(con, hotel_id: str, feature: str) -> dict | None:
    res = resolve([f for f in db.get_facts(con, hotel_id) if f["feature"] == feature])
    return res[feature]["fact"] if feature in res else None


def confirm_fact(con, hotel_id: str, feature: str) -> dict:
    """Отельер подтверждает текущее значение: независимый источник, согласие поднимет уверенность."""
    best = _best_geo(con, hotel_id, feature)
    if best is None or best["score"] is None:
        raise ValueError("Нечего подтверждать: у характеристики нет значения с оценкой")
    db.delete_facts(con, hotel_id, feature, "hotelier_confirmed")
    db.delete_facts(con, hotel_id, feature, "hotelier_dispute")
    db.add_fact(con, hotel_id, feature, value={"confirmed_summary": best["summary"], "verified": False},
                score=best["score"], confidence=CONFIDENCE["hotelier_confirmed"],
                summary=f"Отельер подтвердил: {best['summary']}", source="hotelier_confirmed",
                ttl_days=TTL_DAYS["hotelier_confirmed"])
    return {"feature": feature, "score": best["score"]}


def dispute_fact(con, hotel_id: str, feature: str, actual: str, comment: str) -> dict:
    """Отельер не согласен: actual — «нет» | «частично» | «да» (как обстоит дело на самом деле)."""
    if actual not in DISPUTE_LEVELS:
        raise ValueError("actual: " + " | ".join(DISPUTE_LEVELS))
    if not comment or not comment.strip():
        raise ValueError("Опишите, как обстоит дело на самом деле")
    db.delete_facts(con, hotel_id, feature, "hotelier_confirmed")
    db.delete_facts(con, hotel_id, feature, "hotelier_dispute")
    db.add_fact(con, hotel_id, feature, value={"actual": actual, "comment": comment.strip(), "verified": False},
                score=DISPUTE_LEVELS[actual], confidence=CONFIDENCE["hotelier_dispute"],
                summary=f"Отельер: {actual} — {comment.strip()}", source="hotelier_dispute",
                ttl_days=TTL_DAYS["hotelier_dispute"])
    return {"feature": feature, "score": DISPUTE_LEVELS[actual]}


def queue_gaps(con, hotel_id: str, features: list[str], reason: str = "нет данных") -> list[int]:
    """Недостающие характеристики -> вопросы в очередь отельера (без дублей)."""
    ids = []
    for f in features:
        if f in QUESTIONS:
            qid = db.add_question(con, hotel_id, f, QUESTIONS[f], reason)
            if qid:
                ids.append(qid)
    return ids
