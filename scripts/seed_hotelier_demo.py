"""Демо-сценарии для кабинета отельера на синтетическом городке (все ответы — имитация).

  1. часть отелей «ответила» про животных (через настоящий hotelier_agent.clarify + simulator);
  2. отельер «Морского бриза» подтвердил расчёт вида на море  -> согласие источников, уверенность выше;
  3. отельер «Тихого дворика» оспорил расчёт: море с террасы виден -> противоречие, система его замечает;
  4. у остальных отелей вопросы про животных ждут в очереди;
  5. факт «У вокзала» искусственно состарен                   -> попадает в список устаревших.
Скрипт идемпотентен: запускается после import_hotels.py.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.hotelier_agent import QUESTIONS, clarify
from simulator.hotelier_sim import HotelierSimulator
from storage import db
from storage.hotelier_bridge import confirm_fact, dispute_fact, queue_gaps, store_session

ANSWERED = ["w1", "w2", "w3", "w5"]


def seed(con=None) -> dict:
    con = con or db.connect()
    stats = {"answered": 0, "queued": 0}
    for hid in ANSWERED:
        if db.get_hotel(con, hid) is None:
            continue
        session = clarify(hid, ["pets_allowed"], HotelierSimulator(hid), source="hotelier_simulator", use_llm=False)
        stats["answered"] += sum(r["saved"] for r in store_session(con, session))
    if db.get_hotel(con, "w1"):
        confirm_fact(con, "w1", "sea_view")
    if db.get_hotel(con, "w3"):
        dispute_fact(con, "w3", "sea_view", "да",
                     "[демо-сценарий] С террасы на крыше море видно: дом напротив ниже, чем указано в OSM (30 м).")
    for h in db.list_hotels(con):
        if h["id"] not in ANSWERED:
            stats["queued"] += len(queue_gaps(con, h["id"], ["pets_allowed"], "нет данных"))
    old = (datetime.now(timezone.utc) - timedelta(days=95)).isoformat()
    con.execute("UPDATE facts SET fetched_at=? WHERE hotel_id='w7' AND feature='night_amenities'", (old,))
    con.commit()
    return stats


if __name__ == "__main__":
    print("Демо-сценарии отельеров:", seed())
