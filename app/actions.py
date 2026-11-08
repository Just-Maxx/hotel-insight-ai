"""Действия интерфейса, не относящиеся к отрисовке: диалог с отельером и добавление своего критерия."""
from __future__ import annotations

import re

from agents.hotelier_agent import clarify, sync_questions
from agents.orchestrator import llm_available
from core.registry import ID_RE, load_registry
from simulator.hotelier_sim import HotelierSimulator
from storage.hotelier_bridge import store_session

_TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                      "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def ask_hotelier(con, hotel_id: str, feature: str, answer: str | None, human: bool):
    """Диалог с отельером: ответ человека или имитация вымышленной карточки. Результат — в БД."""
    sync_questions()
    use_llm = llm_available()
    if human:
        session = clarify(hotel_id, [feature], lambda q: answer, source="hotelier_human", use_llm=use_llm, max_rounds=1)
    else:
        session = clarify(hotel_id, [feature], HotelierSimulator(hotel_id), source="hotelier_simulator",
                          use_llm=use_llm, max_rounds=2)
    return store_session(con, session), session


def make_feature_id(title: str) -> str:
    """Название по-русски -> технический id (пользователь его не видит): «Шумоизоляция окон» -> shumoizolyatsiya_okon."""
    raw = "".join(_TRANSLIT.get(ch, ch) for ch in title.lower())
    base = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")[:36]
    if not base or not base[0].isalpha():
        base = "custom_" + base
    base = base if len(base) >= 3 else base + "_x"
    taken, fid, n = set(load_registry()), base, 2
    while fid in taken:
        fid, n = f"{base}_{n}", n + 1
    assert ID_RE.match(fid), fid
    return fid


def friendly_reason(reason: str | None) -> str:
    r = (reason or "").lower()
    if r.startswith("данные устарели"):
        return "Данные давно не обновлялись"
    if "противореч" in r:
        return "Данные из разных источников расходятся"
    if "новая" in r:
        return "Новый вопрос от системы"
    return "Системе не хватает данных"
