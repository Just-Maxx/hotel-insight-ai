"""Hotel Insight AI: отели по скрытым характеристикам.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

import agents.hotelier_agent  # noqa: F401  (регистрирует агента)
import agents.search_agent  # noqa: F401
from agents.orchestrator import llm_available
from app import ui
from app.actions import ask_hotelier
from app.views import about, compare, hotelier, map_view, search
from core.registry import load_registry
from simulator.hotelier_sim import CARDS

st.set_page_config(page_title="Hotel Insight AI: отели по скрытым характеристикам", page_icon="🏨", layout="wide")
ui.inject_css()

try:
    con = ui.get_con()
    hotels = ui.db.list_hotels(con)
except Exception:
    st.error("Не удалось открыть базу отелей. Проверьте данные и перезапустите приложение.")
    st.stop()
if not hotels:
    st.error("В базе нет отелей. Соберите демо-данные командой `python scripts/setup_demo.py` и перезапустите приложение.")
    st.stop()


def main_page() -> None:
    st.title("🏨 Отели, о которых не написано в карточках")
    tab_search, tab_cmp, tab_hotelier, tab_map = st.tabs(["🔎 Поиск", "⚖️ Сравнение", "🧑‍💼 Для отельера", "🗺️ Карта"])
    with tab_search:
        search.render(con, hotels, ask_hotelier)
    with tab_cmp:
        compare.render(con, hotels)
    with tab_hotelier:
        hotelier.render(con, hotels)
    with tab_map:
        map_view.render(hotels)


def about_page() -> None:
    about.render(con, hotels)


with st.sidebar:
    st.toggle("Демо: имитировать ответы отельеров", value=True, key="demo_mode",
              help="Показывает кнопки «Ответить за отельера (демо)» — удобно, чтобы показать, как данные уточняются.")
    st.caption(f"Отелей в базе: {len(hotels)} · характеристик: {len(load_registry())}")
    st.caption("Поиск: " + ("Gemini — основной разбор; при сбое явный резервный режим" if llm_available()
                            else "API-ключ Gemini не задан; доступен резервный режим"))
    if all(h['id'] in CARDS for h in hotels):
        st.warning("Загружены вымышленные демонстрационные отели.")
    else:
        st.info("Данные отелей из текущей БД. Геооценки приблизительные; проверяйте источники.")

pages = [st.Page(main_page, title="Подбор отелей", icon="🏨", url_path="search", default=True),
         st.Page(about_page, title="О системе", icon="ℹ️", url_path="about")]
st.navigation(pages).run()
