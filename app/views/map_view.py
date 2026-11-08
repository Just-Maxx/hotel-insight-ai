"""Карта отелей (без изменений относительно прежней версии)."""
import pandas as pd
import streamlit as st


def render(hotels: list[dict]) -> None:
    st.map(pd.DataFrame(hotels).rename(columns={"lon": "longitude", "lat": "latitude"})[["latitude", "longitude"]])
    st.caption(", ".join(h["name"] for h in hotels))
