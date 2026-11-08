"""Метрики качества на размеченной выборке (eval/labels.json). Запуск без сети и без ключа LLM:

    python eval/run_eval.py          # печатает таблицу и пишет eval/report.json + eval/REPORT.md
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agents.hotelier_agent  # noqa: F401
import agents.search_agent  # noqa: F401
from agents.orchestrator import analyze
from agents.search_agent import rank_hotels
from core.registry import load_registry
from storage import db
from storage.resolver import resolve, source_class


def predict(feature: str, geo_fact: dict) -> bool | None:
    """Решающее правило «характеристика есть» по сохранённому факту геоданных."""
    try:
        v = json.loads(geo_fact["value_json"]) if geo_fact.get("value_json") else None
        if feature == "walk_access":
            return v["beach"]["minutes"] <= 20 if v and v.get("beach") else False
        if feature == "night_amenities":
            return v["pharmacy"]["minutes"] <= 10 if v and v.get("pharmacy") else False
        if feature == "kids_infrastructure":
            return (v.get("playgrounds_within_300m") or 0) >= 1
    except (KeyError, TypeError):
        return None
    return None if geo_fact["score"] is None else geo_fact["score"] >= 0.5


def evaluate(con=None) -> dict:
    con = con or db.connect()
    labels = json.loads((ROOT / "eval" / "labels.json").read_text(encoding="utf-8"))
    registry = load_registry()
    hotels = db.list_hotels(con)
    facts = {h["id"]: db.get_facts(con, h["id"]) for h in hotels}

    # --- точность характеристик (только расчёт по геоданным)
    acc, errors = {}, []
    for feat, spec in labels["features"].items():
        ok = n = 0
        for hid, truth in spec["labels"].items():
            if truth is None:
                continue
            geo = [f for f in facts.get(hid, []) if f["feature"] == feat and source_class(f["source"]) == "geo"]
            if not geo:
                continue
            pred = predict(feat, max(geo, key=lambda f: f["fetched_at"]))
            if pred is None:
                continue
            n += 1
            ok += pred == truth
            if pred != truth:
                errors.append({"hotel": hid, "feature": feat, "truth": truth, "pred": pred})
        acc[feat] = {"correct": ok, "total": n, "accuracy": round(ok / n, 2) if n else None, "basis": spec["basis"]}
    tot_ok = sum(a["correct"] for a in acc.values())
    tot_n = sum(a["total"] for a in acc.values())

    # --- покрытие: доля пар (отель, характеристика реестра) со значением, пригодным для ранжирования
    pairs = [(h["id"], f) for h in hotels for f in registry]
    have = sum(1 for hid, f in pairs if (r := resolve(facts[hid]).get(f)) and r["fact"]["score"] is not None)
    by_feature = {f: sum(1 for h in hotels if (r := resolve(facts[h["id"]]).get(f)) and r["fact"]["score"] is not None)
                  for f in registry}

    # --- поиск: тот же офлайн-планировщик, воспроизводимо
    q_rows, asked, asked_total = [], 0, 0
    for item in labels["queries"]:
        plan = analyze(item["q"], use_llm=False)
        out = rank_hotels(plan, top_k=len(hotels), con=con)
        order = [r["hotel_id"] for r in out["ranked"]]
        rel = set(item["relevant"])
        k = min(3, len(rel))
        p_at_k = sum(h in rel for h in order[:k]) / k if k else 0
        rr = next((1 / (i + 1) for i, h in enumerate(order) if h in rel), 0)
        gaps = sum(len(r["gaps"]) for r in out["ranked"])
        asked += gaps
        asked_total += len([r for r in plan.requirements if r.feature != "other"]) * len(hotels)
        q_rows.append({"query": item["q"], "top3": order[:3], "relevant": sorted(rel),
                       f"precision@{k}": round(p_at_k, 2), "hit@3": any(h in rel for h in order[:3]),
                       "mrr": round(rr, 2), "gaps": gaps, "features": [r.feature for r in plan.requirements]})

    # --- противоречия: система должна заметить ожидаемые и не выдумать лишние
    flagged = {(h["id"], f) for h in hotels for f, r in resolve(facts[h["id"]]).items() if r["conflict"]}
    exp = {tuple(x) for x in labels["expected_conflicts"]}
    tp = len(flagged & exp)

    return {
        "feature_accuracy": acc, "overall_accuracy": round(tot_ok / tot_n, 2) if tot_n else None,
        "errors": errors,
        "coverage": {"pairs_with_value": have, "pairs_total": len(pairs), "share": round(have / len(pairs), 2),
                     "by_feature": by_feature, "hotels": len(hotels)},
        "search": {"queries": q_rows,
                   "hit@3": round(sum(r["hit@3"] for r in q_rows) / len(q_rows), 2),
                   "mrr": round(sum(r["mrr"] for r in q_rows) / len(q_rows), 2)},
        "hotelier_question_share": {"gaps": asked, "requested_pairs": asked_total,
                                    "share": round(asked / asked_total, 2) if asked_total else None},
        "conflicts": {"expected": sorted(exp), "flagged": sorted(flagged), "recall": tp / len(exp) if exp else None,
                      "precision": tp / len(flagged) if flagged else None},
    }


def to_markdown(r: dict) -> str:
    L = ["# Метрики качества (синтетический демо-городок)", "",
         "Разметка — `eval/labels.json`. **Это проверка на синтетике: для реального региона замените разметку.**", "",
         "## Точность характеристик (расчёт по геоданным)", "",
         "| Характеристика | Основа разметки | Верно | Точность |", "|---|---|---|---|"]
    for f, a in r["feature_accuracy"].items():
        L.append(f"| {f} | {a['basis']} | {a['correct']}/{a['total']} | {a['accuracy']} |")
    L += ["", f"Итого: **{r['overall_accuracy']}**. Ошибки: " +
          (", ".join(f"{e['hotel']}/{e['feature']} (эталон {e['truth']}, система {e['pred']})" for e in r["errors"]) or "нет"),
          "", "## Покрытие", "",
          f"Пар «отель × характеристика» со значением: {r['coverage']['pairs_with_value']}/{r['coverage']['pairs_total']} "
          f"= **{r['coverage']['share']}**. По характеристикам: " +
          ", ".join(f"{k}: {v}/{r['coverage']['hotels']}" for k, v in r["coverage"]["by_feature"].items()), "",
          "## Поиск", "", "| Запрос | Топ-3 | Релевантные | Precision@k | MRR |", "|---|---|---|---|---|"]
    for q in r["search"]["queries"]:
        pk = next(v for k, v in q.items() if k.startswith("precision@"))
        L.append(f"| {q['query']} | {', '.join(q['top3'])} | {', '.join(q['relevant'])} | {pk} | {q['mrr']} |")
    L += ["", f"hit@3 = **{r['search']['hit@3']}**, MRR = **{r['search']['mrr']}**", "",
          "## Вопросы к отельеру", "",
          f"Недостающих значений по запросам: {r['hotelier_question_share']['gaps']} из "
          f"{r['hotelier_question_share']['requested_pairs']} запрошенных пар = **{r['hotelier_question_share']['share']}**.", "",
          "## Противоречия", "",
          f"Ожидалось: {r['conflicts']['expected']}; найдено: {r['conflicts']['flagged']}; "
          f"recall={r['conflicts']['recall']}, precision={r['conflicts']['precision']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    res = evaluate()
    (ROOT / "eval" / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "eval" / "REPORT.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
