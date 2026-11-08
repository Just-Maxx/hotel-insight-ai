"""Страница «О системе»: насколько свежие данные и насколько точно работает поиск. Таблицы — по-русски."""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.labels import feature_label
from storage import db
from storage.refresh import queue_refresh, stale_facts
from storage.resolver import source_label

ROOT = Path(__file__).resolve().parents[2]
BASIS = {"analytic": "по координатам на карте", "design": "по замыслу демо-сцены"}
ACTION = {"ask_hotelier": "Спросить отельера", "recompute": "Пересчитать по карте"}


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def _freshness(con) -> None:
    st.header("Свежесть данных")
    st.write("У каждого сведения есть срок годности: расположение пляжа меняется редко, а режим работы аптек — чаще. "
             "Когда срок вышел, система меньше доверяет такому сведению и просит отельера его подтвердить.")
    items = stale_facts(con)
    if not items:
        st.success("Устаревших данных нет.")
        return
    st.dataframe(pd.DataFrame([{
        "Отель": i["hotel"], "Что устарело": feature_label(i["feature"]),
        "Откуда данные": source_label(i["source"]), "Возраст, дней": i["age_days"],
        "Срок годности, дней": i["ttl_days"], "Что делать": ACTION[i["action"]]} for i in items]),
        hide_index=True, width="stretch")
    st.caption("«Спросить отельера» — вопрос появится в его кабинете. «Пересчитать по карте» делается при "
               "обновлении данных о городе (скриптом, один раз в несколько месяцев).")
    if st.button("Задать отельерам вопросы по устаревшим данным"):
        st.success(f"Вопросов поставлено: {queue_refresh(con)['questions_queued']}")


def _quality(con, hotels: list[dict]) -> None:
    sys.path.insert(0, str(ROOT / "eval"))
    from run_eval import evaluate
    names = {h["id"]: h["name"] for h in hotels}
    st.header("Качество работы")
    labels = json.loads((ROOT / 'eval' / 'labels.json').read_text(encoding='utf-8'))
    labelled = {hid for spec in labels['features'].values() for hid in spec['labels']}
    if not hotels or not set(names).issubset(labelled):
        st.info("Эталонная разметка относится к другим отелям. Для оценки текущей БД подготовьте eval/labels.json.")
        return
    st.info("Проверка сделана на демо-городке: заранее известно, какие отели чем хороши, и мы сверяем с этим ответы "
            "системы. Для реального региона эталон нужно размечать вручную.")
    if st.button("Проверить заново") or "eval_report" not in st.session_state:
        with st.spinner("Проверяю…"):
            st.session_state["eval_report"] = evaluate(con)
    rep = st.session_state["eval_report"]

    c = st.columns(4)
    c[0].metric("Верно определено характеристик", _pct(rep["overall_accuracy"]),
                help="Доля случаев, когда расчёт по карте совпал с эталоном (например, «из окна видно море»).")
    c[1].metric("Данных хватает", _pct(rep["coverage"]["share"]),
                help="Для какой доли пар «отель — характеристика» у системы есть значение.")
    c[2].metric("Нужный отель в первой тройке", _pct(rep["search"]["hit@3"]),
                help="В скольких тестовых запросах среди трёх лучших нашёлся подходящий отель.")
    rc = rep["conflicts"]["recall"]
    c[3].metric("Замечает противоречия", _pct(rc),
                help="Нашла ли система случаи, где расчёт по карте и слова отельера не сходятся.")

    st.subheader("Насколько точно определяются характеристики")
    st.dataframe(pd.DataFrame([{
        "Характеристика": feature_label(f), "Проверено отелей": a["total"], "Определено верно": a["correct"],
        "Точность": _pct(a["accuracy"]), "Эталон составлен": BASIS.get(a["basis"], a["basis"])}
        for f, a in rep["feature_accuracy"].items()]), hide_index=True, width="stretch")
    if rep["errors"]:
        st.caption("Где система ошиблась:")
        yn = lambda v: "Да" if v else "Нет"
        st.dataframe(pd.DataFrame([{
            "Отель": names.get(e["hotel"], e["hotel"]), "Характеристика": feature_label(e["feature"]),
            "Как на самом деле": yn(e["truth"]), "Что решила система": yn(e["pred"])} for e in rep["errors"]]),
            hide_index=True, width="stretch")

    st.subheader("Находит ли поиск подходящие отели")
    rows = []
    for q in rep["search"]["queries"]:
        rel = set(q["relevant"])
        pos = next((i + 1 for i, h in enumerate(q["top3"]) if h in rel), None)
        rows.append({"Запрос": q["query"], "Система предложила": ", ".join(names.get(h, h) for h in q["top3"]),
                     "Подходящие по эталону": ", ".join(names.get(h, h) for h in q["relevant"]),
                     "Подходящий в тройке": "Да" if q["hit@3"] else "Нет",
                     "Первый подходящий на месте": pos or "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    q = rep["hotelier_question_share"]
    st.subheader("Чего системе не хватает")
    st.write(f"В тестовых запросах системе не хватило данных в {q['gaps']} случаях из {q['requested_pairs']} "
             f"({_pct(q['share'])}) — в таких случаях она просит отельера уточнить.")
    cf = rep["conflicts"]
    flagged = ", ".join(f"{names.get(h, h)} — {feature_label(f)}" for h, f in cf["flagged"]) or "нет"
    st.write(f"Противоречия между картой и словами отельера: найдено — {flagged}.")


def render(con, hotels: list[dict]) -> None:
    st.title("О системе")
    st.write("Система находит отели по тому, чего нет в обычных карточках: вид из окна, шум по сторонам здания, "
             "реальная дорога пешком до пляжа, ночные магазины, удобства для детей и правила для питомцев. "
             "Часть сведений считается по карте, часть сообщают отельеры.")
    _freshness(con)
    st.divider()
    _quality(con, hotels)
