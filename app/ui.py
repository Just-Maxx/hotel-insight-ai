"""Общие мелочи интерфейса: кружки, стили, подключение к базе."""
from __future__ import annotations

import html

import streamlit as st

from core.labels import feature_label
from storage import db

LEVEL_SYMBOL = {"good": "✓", "mid": "~", "bad": "✕", "unknown": "?"}
LEVEL_WORD = {"good": "подходит", "mid": "частично", "bad": "не подходит", "unknown": "нет данных"}

CSS = """
<style>
.dot{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;
     color:#fff;font-weight:700;font-size:14px;line-height:1;cursor:default}
.dot.good{background:#2e9e5b}.dot.mid{background:#e0a800}.dot.bad{background:#d6453d}
.dot.unknown{background:transparent;color:#9aa0a6;border:2px dashed #9aa0a6;width:22px;height:22px;font-size:12px}
.cmp-wrap{overflow-x:auto}
table.cmp{border-collapse:collapse;width:100%;min-width:420px}
table.cmp th,table.cmp td{padding:10px 12px;text-align:center;border-bottom:1px solid rgba(128,128,128,.25)}
table.cmp th:first-child,table.cmp td:first-child{text-align:left;font-weight:500}
table.cmp thead th{font-weight:600;vertical-align:bottom}
table.cmp tr.total td{font-weight:600;border-top:2px solid rgba(128,128,128,.5);border-bottom:none}
table.cmp .best{background:rgba(46,158,91,.14);border-radius:6px}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 2px 0;font-size:.9rem;opacity:.85}
.legend span{display:inline-flex;align-items:center;gap:6px}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def dot(level: str, hint: str = "") -> str:
    """Кружок с символом внутри (не только цвет — удобно и при дальтонизме). hint — подсказка при наведении."""
    tip = html.escape(hint or LEVEL_WORD[level], quote=True)
    return f'<span class="dot {level}" title="{tip}">{LEVEL_SYMBOL[level]}</span>'


def legend_html() -> str:
    return '<div class="legend">' + "".join(
        f'<span>{dot(l)} {LEVEL_WORD[l]}</span>' for l in ("good", "mid", "bad", "unknown")) + "</div>"


def emoji_mark(score) -> str:
    """Кружок-эмодзи для строк результата поиска."""
    return "⚪" if score is None else "🟢" if score >= 0.6 else "🟡" if score >= 0.3 else "🔴"


def short(text: str, n: int = 140) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def get_con():
    return db.connect()


def demo_mode() -> bool:
    return bool(st.session_state.get("demo_mode", True))


__all__ = ["dot", "legend_html", "emoji_mark", "short", "get_con", "demo_mode", "inject_css", "feature_label"]
