"""Режим без LLM: разбор запроса по ключевым словам и шаблонное объяснение.

Нужен, чтобы демо запускалось без ключа Gemini и чтобы система не падала при сбое API.
Это намеренно простой разбор: он понимает типовые формулировки, но не заменяет модель
(в интерфейсе режим всегда показан). Расчёты и ранжирование от режима не зависят.
"""
from __future__ import annotations

import re

from core.labels import CLOSE_WALK_MINUTES, DEFAULT_WALK_MINUTES, requirement_label
from core.registry import load_registry
from storage import db

# характеристика -> регулярки (по нижнему регистру, «ё» заменена на «е»).
# Дорогу пешком (walk_access) разбирает отдельная логика ниже: ей нужны «до чего» и «сколько минут».
KEYWORDS = {
    "sunset_view": [r"закат", r"заход\w* солнца"],
    "sea_view": [r"вид\w* на мор", r"мор\w* (?:из|в) окн", r"видом на мор", r"панорам\w* мор"],
    "quiet": [r"тих", r"шум", r"спокойн", r"выспат", r"тишин"],
    "pets_allowed": [r"питомц", r"собак", r"кошк", r"кот\b", r"животн", r"с псом"],
    "night_amenities": [r"ночью", r"круглосуточ", r"24/7", r"аптек\w* ночью"],
    "kids_infrastructure": [r"с детьм", r"ребен", r"ребён", r"детск", r"дет(?:и|ей|ям)\b", r"семь", r"площадк"],
}
NICE = re.compile(r"желательно|лучше|было бы|по возможности|хорошо если|не обязательно|бонус|а еще|а ещё|в идеале")
HOTEL_WORDS = re.compile(r"отел|номер|гостиниц|жиль|остановиться|поселиться|ночевк|проживан")
WANT_WORDS = re.compile(r"хочу|хотим|ищу|ищем|нужен|нужна|нужно|нужны|подбер|посоветуй|найди|отдых|отпуск|поехать|поездк")

# «вид на море» — это другая характеристика (что видно из окна), её нельзя путать с «море рядом»
SEA_VIEW = re.compile(r"(?:вид\w*|панорам\w*)\s+(?:\S+\s+){0,2}?на\s+(?:\S+\s+)?мор\w*|мор\w*\s+(?:из|в)\s+окн\w*|панорам\w*\s+мор\w*")
VIEW_PHRASE = re.compile(r"(?:вид\w*|панорам\w*)\s+(?:\S+\s+){0,2}?на\s+(?:\S+\s+)?\w+|\w+\s+(?:из|в)\s+окн\w*")

# до чего считаем дорогу пешком
WALK_TARGETS = {
    "beach": re.compile(r"\bмор(?:е|я|ю|ем|и|ей)\b|пляж\w*|берег\w*|набережн\w*"),
    "bus_stop": re.compile(r"остановк\w*|автобус\w*|маршрутк\w*"),
    "station": re.compile(r"вокзал\w*|ж/д|железнодорожн\w*|станци\w*|электричк\w*"),
}
# слова, которые делают пожелание про дорогу: «рядом», «пешком», «в N минутах»
PROXIMITY = re.compile(r"пешк\w*|пеш\w*\s+доступн\w*|пешеходн\w*\s+доступн\w*|шагов\w*\s+доступн\w*|"
                       r"рядом|недалеко|близко|близост\w*|поблизости|возле|около|в\s+шаге|ходьб\w*|"
                       r"\d+\s*мин|первая\s+линия|на\s+берегу|у\s+(?:самого\s+)?(?:моря|воды|пляжа)")
# «транспорт», «пешком до всего» — пожелание про дорогу без конкретной цели
GENERIC_WALK = re.compile(r"пешк\w*|пеш\w*\s+доступн\w*|пешеходн\w*\s+доступн\w*|шагов\w*\s+доступн\w*|"
                          r"транспорт|близост\w*|недалеко от|близко к")
VERY_CLOSE = re.compile(r"у\s+самого\s+(?:моря|воды|пляжа)|первая\s+линия|на\s+берегу|в\s+шаге|шаговая")
MINUTES = re.compile(r"(\d{1,3})\s*-?\s*(?:мин|минут)")
MINUTE_WORDS = {r"пят(?:ь|и)": 5, r"десят(?:ь|и)": 10, r"пятнадцат(?:ь|и)": 15, r"двадцат(?:ь|и)": 20,
                r"тридцат(?:ь|и)": 30, r"пол-?часа|получас\w*": 30}


def _norm(s: str) -> str:
    return s.lower().replace("ё", "е")


def _patterns() -> dict[str, list[str]]:
    pats = {k: list(v) for k, v in KEYWORDS.items()}
    for fid, spec in load_registry().items():
        if fid in pats or fid == "walk_access":
            continue
        words = re.findall(r"[а-яa-z]{4,}", _norm(f"{spec['title']} {spec.get('description', '')[:60]}"))
        pats[fid] = [re.escape(w[:5]) for w in dict.fromkeys(words)][:4]  # основа слова (первые 5 букв)
    return pats


def _minutes_in(clause: str) -> int | None:
    """Сколько минут пешком назвал человек: «до 10 минут», «в пяти минутах», «полчаса»."""
    m = MINUTES.search(clause)
    if m:
        return max(1, min(int(m.group(1)), 120))
    for word, val in MINUTE_WORDS.items():
        if re.search(rf"\b(?:{word})\b", clause):
            return val
    return None


def _walk_requirements(clause: str) -> list[tuple[str | None, int | None, str]]:
    """Пожелания про дорогу пешком: [(цель или None, минуты или None, фраза)].

    «Хочу море в пешей доступности» -> пляж, 15 минут. «Вид на море» сюда не попадает.
    """
    rest = SEA_VIEW.sub(" ", clause)  # убираем «вид на море», остальное разбираем отдельно
    rest = VIEW_PHRASE.sub(" ", rest) if not WALK_TARGETS["beach"].search(rest) else rest
    minutes = _minutes_in(clause)
    if minutes is None:
        minutes = CLOSE_WALK_MINUTES if VERY_CLOSE.search(rest) else DEFAULT_WALK_MINUTES
    found = []
    for target, rx in WALK_TARGETS.items():
        if not rx.search(rest):
            continue
        # море и пляж достаточно назвать; остановка и станция — только вместе со словом «рядом/пешком»
        if target == "beach" or (target == "bus_stop") or PROXIMITY.search(rest):
            found.append((target, minutes, clause.strip()))
    if found:
        return found
    if GENERIC_WALK.search(rest):
        return [(None, None, clause.strip())]
    return []


def plan_offline(text: str):
    """Текст -> Plan без LLM. Импорт Plan внутри функции — избегаем цикла с orchestrator."""
    from agents.orchestrator import Plan, Requirement, Task

    low = _norm(text)
    pats = _patterns()
    registry = load_registry()

    reqs: dict[tuple[str, str | None], Requirement] = {}

    def put(fid: str, clause: str, importance: str, target=None, minutes=None):
        key = (fid, target)
        if key in reqs and reqs[key].importance == "must":
            return
        reqs[key] = Requirement(feature=fid, detail=clause.strip(), importance=importance,
                                target=target, max_minutes=minutes if target else None)

    for clause in re.split(r"[,;.!?\n]|\sа также\s|\sно\s", low):
        if not clause.strip():
            continue
        importance = "nice" if NICE.search(clause) else "must"
        for fid, rx in pats.items():
            if any(re.search(p, clause) for p in rx):
                put(fid, clause, importance)
        for target, minutes, phrase in _walk_requirements(clause):
            put("walk_access", phrase, importance, target, minutes)

    # «Нужно море недалеко, чтобы дойти пешком»: запятая оторвала «пешком» от «моря» — общее пожелание лишнее
    if any(k[0] == "walk_access" and k[1] for k in reqs):
        for k in [k for k, r in reqs.items() if k[0] == "walk_access" and not k[1] and "транспорт" not in r.detail]:
            del reqs[k]

    con = db.connect()
    try:
        cities = sorted({h["city"] for h in db.list_hotels(con) if h["city"]})
    finally:
        con.close()
    location = next((c for c in cities if _norm(c)[: max(4, len(c) - 2)] in low), None)

    if not reqs and not (HOTEL_WORDS.search(low) or WANT_WORDS.search(low)):
        return Plan(intent="chitchat")
    if not reqs:
        return Plan(intent="search_hotels", location=location,
                    clarifying_question="Что для вас важно в отеле? Например: тишина, вид на закат или море, "
                                        "пляж в 15 минутах пешком, можно с питомцем, площадка для детей.")
    if location is None and len(cities) > 1:
        return Plan(intent="search_hotels", requirements=list(reqs.values()),
                    clarifying_question="В каком городе или районе ищем? Сейчас в данных: " + ", ".join(cities))
    tasks = [Task(agent="search_agent", instruction=f"Найди и ранжируй отели ({location or 'все'}) по: " +
                  "; ".join(f"{requirement_label(r.feature, r.target, r.max_minutes)} [{r.importance}]"
                            for r in reqs.values()))]
    return Plan(intent="search_hotels", location=location, requirements=list(reqs.values()), tasks=tasks)


def _short(text: str, n: int = 140) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def explain_offline(search_output: dict) -> str:
    """Шаблонное объяснение: только факты из расчёта, как и у LLM-версии."""
    registry = load_registry()
    blocks = []
    for i, r in enumerate(search_output.get("ranked", []), 1):
        lines = [f"{i}. {r['hotel']} — подходит на {r['score']:.0%}"]
        for m in r["matched"]:
            warn = " (данные приблизительные)" if m["confidence"] < 0.7 else ""
            lines.append(f"   + {m.get('label') or m['feature']}: {_short(m['summary'])}{warn}")
        for m in r["misses"]:
            lines.append(f"   − {m.get('label') or m['feature']}: не дотягивает — {_short(m['summary'])}")
        for g in r["gaps"]:
            lines.append(f"   ? {g['title']}: данных нет — можно спросить отельера")
        for w in r["warnings"]:
            lines.append(f"   ! {_short(w, 220)}")
        blocks.append("\n".join(lines))
    if search_output.get("unsupported"):
        blocks.append("Пока не умею проверять: " + "; ".join(search_output["unsupported"]))
    return "\n\n".join(blocks)
