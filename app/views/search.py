"""Поиск: свободный запрос -> понятный разбор -> карточки отелей."""
from __future__ import annotations

import streamlit as st

from agents.orchestrator import run, llm_available
from agents.offline import explain_offline
from agents.search_agent import rank_hotels
from app.ui import demo_mode, emoji_mark
from core.labels import IMPORTANCE_LABEL, requirement_label
from simulator.hotelier_sim import CARDS

EXAMPLES = ["Хочу море в пешей доступности",
            "Тихий номер с видом на закат недалеко от моря",
            "Едем с детьми, нужна детская площадка рядом и тишина",
            "Нужна круглосуточная аптека рядом, ночью",
            "Еду с собакой, желательно вид на море",
            "Пляж в 10 минутах пешком и остановка рядом"]


def _understood(plan) -> str:
    parts = []
    for r in plan.requirements:
        label = r.detail if r.feature == "other" else requirement_label(r.feature, r.target, r.max_minutes)
        label = f"**{label}**" if r.importance == "must" else label
        tail = "пока не умею проверять" if r.feature == "other" else IMPORTANCE_LABEL[r.importance]
        parts.append(f"{label} ({tail})")
    return " · ".join(parts)


def render(con, hotels: list[dict], ask_hotelier) -> None:
    for i in range(0, len(EXAMPLES), 3):
        cols = st.columns(3)
        for c, ex in zip(cols, EXAMPLES[i:i + 3]):
            if c.button(ex, key=f"ex_{i}_{ex[:12]}", width="stretch"):
                st.session_state["q"] = ex
    query = st.text_input("Опишите своими словами, что важно", key="q",
                          placeholder="например: тихий номер недалеко от моря, можно с собакой")
    top_k = st.slider("Сколько отелей показать", 1, len(hotels), min(3, len(hotels))) if len(hotels) > 1 else 1
    if st.button("Найти", type="primary"):
        if not query.strip():
            st.warning("Введите пожелания к отелю — например, «море в пешей доступности».")
            st.session_state.pop('out', None)
        else:
            with st.spinner("Gemini разбирает запрос и подбирает критерии…" if llm_available() else "Резервный подбор по данным БД…"):
                st.session_state["out"] = run(query.strip(), top_k=top_k)
                st.session_state["out_q"] = query.strip()

    out = st.session_state.get("out")
    if not out:
        return
    plan = out["plan"]
    st.caption("Результаты по запросу: " + st.session_state.get('out_q', ''))
    st.caption("Разбор запроса: " + ('Gemini (онлайн)' if plan.planner == 'llm' else 'резервный режим без LLM'))
    if plan.intent != 'chitchat':
        st.caption('Город: ' + (plan.location or 'не ограничен; показаны отели доступных городов'))
    for notice in plan.notices:
        st.warning(notice)
    for diagnostic in out.get('diagnostics', []):
        if diagnostic.get('message'):
            (st.error if diagnostic['status'] == 'error' else st.warning)(diagnostic['message'])
    if plan.intent == "chitchat":
        st.info("Я подбираю отели. Расскажите, что для вас важно: тишина, вид, пляж пешком, дети, питомец…")
        return
    if out["ask_user"]:
        st.info(out["ask_user"])
        if not st.session_state.get('out_q') or len(st.session_state['out_q']) > 4000:
            return

    understood = _understood(plan)
    if understood:
        st.markdown("Я понял так: " + understood)
    else:
        st.info("Показан общий список отелей. Уточните критерии для персонального подбора.")
    try:
        res = rank_hotels(plan, top_k=top_k, con=con)
    except Exception:
        st.error("Не удалось получить отели из базы. Проверьте данные и повторите поиск.")
        return  # заново: ответы отельеров сразу меняют картину
    if res.get("note"):
        st.warning(res["note"])
    for i, r in enumerate(res.get("ranked", []), 1):
        with st.container(border=True):
            head, sc = st.columns([4, 1])
            head.subheader(f"{i}. {r['hotel']}")
            if out.get('default_listing'):
                sc.metric("Соответствие", "Не оценено")
            else:
                sc.metric("Подходит", f"{r['score']:.0%}")
            for m in r["matched"]:
                notes = [m["source_label"]]
                if m["confidence"] < 0.7:
                    notes.append("данные приблизительные")
                if m.get("stale"):
                    notes.append("давно не обновлялись")
                st.markdown(f"{emoji_mark(m['score'])} **{m['label']}** — {m['summary']}")
                st.caption(' · '.join(notes))
            for m in r["misses"]:
                st.markdown(f"🔴 **{m['label']}** — не дотягивает: {m['summary']}")
            for g in r["gaps"]:
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"⚪ **{g['title']}** — данных пока нет")
                if demo_mode() and r["hotel_id"] in CARDS and c2.button(
                        "Спросить отельера (демо)", key=f"ask_{r['hotel_id']}_{g['feature']}"):
                    report, _ = ask_hotelier(con, r["hotel_id"], g["feature"], None, human=False)
                    st.toast(f"{r['hotel']}: " + "; ".join(x["summary"] for x in report))
                    st.rerun()
            for w in r["warnings"]:
                st.caption("⚠️ " + w)
    if res.get("unsupported"):
        st.caption("Пока не умею проверять: " + "; ".join(res["unsupported"]))
    original = next((item.get('output', {}).get('ranked', []) for item in out['results']
                     if item['agent'] == 'search_agent'), [])
    if not res.get('ranked'):
        st.info(res.get('note') or "В базе нет отелей для этого запроса. Попробуйте другой город.")
    elif not out.get('default_listing'):
        answer = out.get('answer') if original == res['ranked'] else explain_offline(res)
        if answer:
            with st.expander("Пояснение к выбору", expanded=True):
                if original != res['ranked']:
                    st.caption('Выдача обновлена. Пояснение по текущим данным, без повторного вызова Gemini.')
                st.markdown(answer)
