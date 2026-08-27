from __future__ import annotations

import streamlit as st

from .pages import (
    backtest_page, dashboard, data_page, experiments_page, live_paper,
    models_page, predictions_page, settings_page, trades_page,
)


def main() -> None:
    st.set_page_config(page_title="XAUUSD TradingLab", page_icon=":material/query_stats:", layout="wide")
    pages = {
        "Dashboard": dashboard,
        "Live Paper": live_paper,
        "Backtest": backtest_page,
        "Experiments": experiments_page,
        "Models": models_page,
        "Trades": trades_page,
        "Predictions": predictions_page,
        "Data": data_page,
        "Settings": settings_page,
    }
    selected = st.sidebar.radio("Navigation", list(pages))
    st.sidebar.caption("XAUUSD M1 quantitative research")
    pages[selected]()
