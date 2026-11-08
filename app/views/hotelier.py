"""Кабинет отельера: ответить на вопросы системы и проверить, что гости узнают о вашем отеле."""
from __future__ import annotations

import streamlit as st

from agents.hotelier_agent import QUESTIONS, sync_questions
from agents.search_agent import level_of
from app.actions import ask_hotelier, friendly_reason, make_feature_id
from app.ui import demo_mode, dot, short
from core.labels import feature_label
from core.registry import add_feature
from simulator.hotelier_sim import CARDS
from storage import db
from storage.hotelier_bridge import DISPUTE_LEVELS, confirm_fact, dispute_fact, queue_gaps
from storage.resolver import resolve, source_label

ANSWER_TYPES = {"Да / нет": "boolean", "Свободный текст": "text"}


def _questions_block(con, hid: str, pending: list[dict]) -> None:
    st.subheader("Вопросы к вам")
    if not pending:
        st.success("Всё хорошо: у нас больше нет вопросов к вашему отелю.")
        return
    for q in pending:
        with st.container(border=True):
            st.markdown(f"**{feature_label(q['feature'])}**")
            st.caption(friendly_reason(q["reason"]))
            st.write(q["question"])
            ans = st.text_area("Ваш ответ своими словами", key=f"ans_{q['id']}",
                               placeholder="Например: да, принимаем собак до 10 кг, доплата 500 ₽ в сутки")
            c1, c2, _ = st.columns([1.2, 1.8, 3])
            if c1.button("Отправить", key=f"send_{q['id']}", type="primary") and ans.strip():
                try:
                    report, _s = ask_hotelier(con, hid, q["feature"], ans, human=True)
                    if report and report[0]["saved"]:
                        db.close_question(con, q["id"], ans)
                        st.toast("Спасибо! Записали: " + report[0]["summary"])
                        st.rerun()
                    else:
                        st.warning("Не получилось однозначно понять ответ. Попробуйте ответить короче и конкретнее: "
                                   + QUESTIONS.get(q["feature"], ""))
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))
            if demo_mode() and hid in CARDS and c2.button("Ответить за отельера (демо)", key=f"sim_{q['id']}"):
                report, _s = ask_hotelier(con, hid, q["feature"], None, human=False)
                if report and report[0]["saved"]:
                    db.close_question(con, q["id"], "[демо] " + report[0]["summary"])
                    st.rerun()
                st.warning("В демо-режиме нет ответа на этот вопрос.")


def _facts_block(con, hid: str) -> None:
    st.subheader("Что узнают гости о вашем отеле")
    st.caption("Если всё верно — нажмите «Верно». Если нет — «Неверно» и скажите, как на самом деле.")
    resolved = resolve(db.get_facts(con, hid))
    if not resolved:
        st.info("Пока нет сведений об этом отеле.")
    for feat, r in resolved.items():
        f = r["fact"]
        badges = [source_label(f["source"])]
        if r["stale"]:
            badges.append("⏳ давно не проверяли — подтвердите, что это всё ещё так")
        if r["conflict"]:
            badges.append("⚠️ данные расходятся — проверьте")
        with st.container(border=True):
            c1, c2, c3 = st.columns([7, 1.4, 1.8])
            c1.markdown(f"{dot(level_of(f['score']))}&nbsp;&nbsp;**{feature_label(feat)}** — {short(f['summary'], 160)}  \n"
                        f"<small>{' · '.join(badges)}</small>", unsafe_allow_html=True)
            if f["score"] is None and not r["alternatives"]:
                continue
            if c2.button("Верно", key=f"ok_{hid}_{feat}"):
                try:
                    confirm_fact(con, hid, feat)
                    st.toast(f"«{feature_label(feat)}» подтверждено")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
            with c3.popover("Неверно"):
                actual = st.radio(f"Как на самом деле с «{feature_label(feat)}»?", list(DISPUTE_LEVELS),
                                  format_func=str.capitalize, horizontal=True, key=f"act_{hid}_{feat}")
                comment = st.text_input("Что именно не так?", key=f"cmt_{hid}_{feat}")
                if st.button("Отправить", key=f"dis_{hid}_{feat}"):
                    try:
                        dispute_fact(con, hid, feat, actual, comment)
                        st.toast("Спасибо, исправили")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))


def _custom_feature_block(con, hotels: list[dict]) -> None:
    with st.expander("➕ Добавить свой критерий для гостей"):
        st.caption("Например, «Шумоизоляция окон» или «Есть парковка». Система сама спросит об этом у отельеров "
                   "и будет учитывать в поиске.")
        with st.form("new_feature", clear_on_submit=True):
            title = st.text_input("Название (так увидят гости)", placeholder="Шумоизоляция окон")
            question = st.text_input("Вопрос отельеру", placeholder="Есть ли в номерах шумоизоляция окон?")
            kind = st.radio("Какой ответ ждём", list(ANSWER_TYPES), horizontal=True,
                            help="Без подключённой нейросети надёжно разбирается только «Да / нет».")
            ask_all = st.checkbox("Сразу задать вопрос всем отелям", value=True)
            if st.form_submit_button("Добавить"):
                try:
                    if not title.strip() or not question.strip():
                        raise ValueError("Заполните название и вопрос")
                    fid = make_feature_id(title)
                    add_feature(fid, title.strip(), title.strip(), question.strip(), answer_type=ANSWER_TYPES[kind])
                    sync_questions()
                    n = sum(len(queue_gaps(con, h["id"], [fid], "новая характеристика")) for h in hotels) if ask_all else 0
                    st.success(f"Критерий добавлен. Вопросов отельерам поставлено: {n}.")
                except ValueError as e:
                    st.error(str(e))


def render(con, hotels: list[dict]) -> None:
    names = {h["id"]: h["name"] for h in hotels}
    st.markdown("**Зачем это нужно.** Часть сведений нельзя узнать по карте: принимают ли с животными, "
                "какие номера тише. Ответьте на несколько вопросов и проверьте, что система пишет о вашем отеле, — "
                "тогда гости найдут вас по правильным запросам.")
    hid = st.selectbox("Ваш отель", list(names), format_func=names.get, key="cab_hotel")
    sync_questions()
    pending = db.list_questions(con, "pending", hid)
    review = sum(1 for r in resolve(db.get_facts(con, hid)).values() if r["needs_review"])
    m1, m2 = st.columns(2)
    m1.metric("Вопросов к вам", len(pending))
    m2.metric("Стоит проверить", review)
    _questions_block(con, hid, pending)
    st.divider()
    _facts_block(con, hid)
    st.divider()
    _custom_feature_block(con, hotels)
