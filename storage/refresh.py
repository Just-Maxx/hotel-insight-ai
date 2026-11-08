"""Актуализация: находим устаревшие факты и решаем, что с ними делать.

- факты от отельера/сайта  -> ставим вопрос в очередь кабинета («данные устарели»);
- расчёты по геоданным     -> пересчёт требует слои OSM/DEM: перезапустите scripts/build_demo.py
  (ответы Overpass кэшируются) и scripts/import_hotels.py --keep. Новый расчёт автоматически вытеснит
  старый: у устаревшего факта уверенность вдвое ниже (storage/resolver.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

from agents.hotelier_agent import QUESTIONS, sync_questions
from storage import db
from storage.resolver import is_stale, source_class


def stale_facts(con, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    names = {h["id"]: h["name"] for h in db.list_hotels(con)}
    out = []
    for f in con.execute("SELECT * FROM facts"):
        f = dict(f)
        if not is_stale(f, now):
            continue
        fetched = datetime.fromisoformat(f["fetched_at"])
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        out.append({"hotel_id": f["hotel_id"], "hotel": names.get(f["hotel_id"], f["hotel_id"]),
                    "feature": f["feature"], "source": f["source"], "age_days": (now - fetched).days,
                    "ttl_days": f["ttl_days"],
                    "action": "ask_hotelier" if source_class(f["source"]) == "hotelier" else "recompute"})
    return sorted(out, key=lambda x: -x["age_days"])


def queue_refresh(con, now: datetime | None = None) -> dict:
    """Ставит вопросы по устаревшим фактам отельеров; геофакты только перечисляет."""
    sync_questions()
    items = stale_facts(con, now)
    queued = 0
    for it in items:
        if it["action"] == "ask_hotelier" and it["feature"] in QUESTIONS:
            queued += bool(db.add_question(con, it["hotel_id"], it["feature"], QUESTIONS[it["feature"]],
                                           f"данные устарели ({it['age_days']} дн., срок {it['ttl_days']})"))
    return {"stale": items, "questions_queued": queued,
            "need_recompute": [i for i in items if i["action"] == "recompute"]}
