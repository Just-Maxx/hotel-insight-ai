"""Достоверность: выбираем лучшее значение, замечаем противоречия и устаревание.

Правила (все числа — эвристики, не статистика):
  1. Эффективная уверенность факта = confidence, вдвое меньше, если истёк ttl.
  2. Если два НЕЗАВИСИМЫХ класса источников (геоданные / отельер) согласны (score расходится <= AGREE_GAP),
     уверенность итогового значения растёт: 1-(1-c1)(1-c2).
  3. Если расходятся больше чем на CONFLICT_GAP — это противоречие. Берём значение с большей
     эффективной уверенностью, но уверенность режем в CONFLICT_PENALTY раз и помечаем needs_review:
     система не выбирает «победителя» молча, а просит человека проверить (кабинет отельера).
"""
from collections import defaultdict
from datetime import datetime, timezone

CONFLICT_GAP = 0.4    # источники расходятся по score больше чем на это — противоречие
AGREE_GAP = 0.2       # расхождение не больше этого — источники согласны
STALE_PENALTY = 0.5   # устаревшее значение теряет половину уверенности
CONFLICT_PENALTY = 0.7
AGREE_CAP = 0.99

SOURCE_LABEL = {"hotelier_human": "отельер", "hotelier_confirmed": "отельер (подтвердил расчёт)",
                "hotelier_dispute": "отельер (оспорил расчёт)", "hotelier_simulator": "имитация отельера (демо)",
                "hotel_website": "сайт отеля"}


def source_class(source: str) -> str:
    """geo — расчёт по открытым данным; hotelier — слова отельера/сайта. Это независимые источники."""
    return "hotelier" if source.startswith(("hotelier", "hotel_website")) else "geo"


def source_label(source: str) -> str:
    return SOURCE_LABEL.get(source, "расчёт по карте" if source_class(source) == "geo" else source)


def is_stale(fact: dict, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    fetched = datetime.fromisoformat(fact["fetched_at"])
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (now - fetched).days > (fact.get("ttl_days") or 180)


def effective_confidence(fact: dict) -> float:
    return fact["confidence"] * (STALE_PENALTY if is_stale(fact) else 1.0)


def resolve(facts: list[dict]) -> dict[str, dict]:
    """feature -> {fact, conflict, stale, alternatives, confidence, agreement, needs_review, resolution}."""
    by_feature = defaultdict(list)
    for f in facts:
        by_feature[f["feature"]].append(f)

    out = {}
    for feature, items in by_feature.items():
        items.sort(key=lambda f: (effective_confidence(f), f["fetched_at"]), reverse=True)
        best, rest = items[0], items[1:]
        conf = effective_confidence(best)
        scored = [i for i in items if i["score"] is not None]
        scores = [i["score"] for i in scored]
        conflict = len(scores) > 1 and (max(scores) - min(scores)) > CONFLICT_GAP

        agreement, resolution = False, None
        others = [i for i in rest if i["score"] is not None and best["score"] is not None
                  and source_class(i["source"]) != source_class(best["source"])]
        if conflict:
            conf *= CONFLICT_PENALTY
            loser = next((i for i in rest if i["score"] is not None and best["score"] is not None
                          and abs(i["score"] - best["score"]) > CONFLICT_GAP), None)
            if loser:
                resolution = (f"Источники расходятся: {source_label(best['source'])} — {best['summary']!r} "
                              f"(уверенность {effective_confidence(best):.2f}) против "
                              f"{source_label(loser['source'])} — {loser['summary']!r} "
                              f"({effective_confidence(loser):.2f}). Принято значение с большей уверенностью, "
                              f"уверенность снижена; нужна проверка.")
        else:
            agree = [i for i in others if abs(i["score"] - best["score"]) <= AGREE_GAP]
            if agree:
                agreement = True
                conf = min(AGREE_CAP, 1 - (1 - conf) * (1 - effective_confidence(agree[0])))
                resolution = (f"{source_label(best['source'])} и {source_label(agree[0]['source'])} согласны — "
                              f"уверенность повышена до {conf:.2f}.")
        out[feature] = {"fact": best, "conflict": conflict, "stale": is_stale(best),
                        "confidence": round(conf, 3), "agreement": agreement,
                        "needs_review": conflict or is_stale(best), "resolution": resolution,
                        "alternatives": [{"source": a["source"], "score": a["score"], "summary": a["summary"],
                                          "confidence": a["confidence"], "fetched_at": a["fetched_at"]}
                                         for a in rest]}
    return out
