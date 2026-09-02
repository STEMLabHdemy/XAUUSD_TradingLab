from __future__ import annotations

from pathlib import Path

import streamlit as st

from .live_dashboard import live_market_panel


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    st.set_page_config(page_title="XAUUSD Live Paper", page_icon=":material/candlestick_chart:", layout="wide")
    # This is intentionally the only rendered page.  Research/training tools
    # remain available from their dedicated scripts, rather than loading their
    # dataframes, charts and modules into the realtime process.
    st.title("Live paper")
    st.caption("Feed reale MT5, capitale virtuale; nessun ordine viene inviato al broker.")
    live_market_panel(str(ROOT), show_paper_controls=True)
