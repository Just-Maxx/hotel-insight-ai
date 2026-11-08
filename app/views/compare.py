"""Сравнение отелей: сетка из кружков, без чисел и служебных пометок."""
from __future__ import annotations

import html

import streamlit as st

from agents.search_agent import compare_simple
from app.ui import dot, legend_html


def render(con, hotels: list[dict]) -> None:
    names = {h["id"]: h["name"] for h in hotels}
    st.markdown("Выберите 2–4 отеля — в таблице сразу видно, кто в чём сильнее.")
    pick = st.multiselect("Отели для сравнения", list(names), format_func=names.get,
                          default=list(names)[:3], max_selections=4, key="cmp_pick",
                          label_visibility="collapsed", placeholder="Выберите отели")
    if len(pick) < 2:
        st.info("Выберите хотя бы два отеля.")
        return

    cmp = compare_simple(pick, con=con)
    best = set(cmp["best"])
    head = "".join(f'<th class="{"best" if hid in best else ""}">{html.escape(cmp["hotels"][hid])}</th>' for hid in pick)
    body = ""
    for row in cmp["rows"]:
        cells = "".join(f'<td>{dot(row["cells"][hid]["level"], row["cells"][hid]["hint"])}</td>' for hid in pick)
        body += f'<tr><td>{html.escape(row["label"])}</td>{cells}</tr>'
    total = "".join(
        f'<td class="{"best" if hid in best else ""}">{cmp["totals"][hid]} из {cmp["rows_total"]}</td>' for hid in pick)
    st.markdown(f'<div class="cmp-wrap"><table class="cmp"><thead><tr><th></th>{head}</tr></thead>'
                f'<tbody>{body}<tr class="total"><td>Подходит по</td>{total}</tr></tbody></table></div>',
                unsafe_allow_html=True)
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption("Наведите курсор на кружок, чтобы увидеть пояснение.")

    if best and len(best) < len(pick):
        st.success("Больше всего совпадений: " + ", ".join(cmp["hotels"][h] for h in cmp["best"]) + ".")
    thin = [cmp["hotels"][h] for h in pick if cmp["known"][h] <= cmp["rows_total"] // 2]
    if thin:
        st.caption("Мало данных: " + ", ".join(thin) + " — итог может измениться, когда отельер ответит на вопросы.")
