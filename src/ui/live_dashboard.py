from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.live import LiveMarketService
from src.live.timeframes import chart_bars


def _price(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def live_candlestick_figure(
    m1_bars: pd.DataFrame,
    timeframe: str,
    display_timezone: str,
    limit: int,
    auto_follow: bool,
) -> go.Figure:
    market = chart_bars(m1_bars, timeframe, limit)
    if market.empty:
        return go.Figure()
    local_time = pd.to_datetime(market.datetime_utc, utc=True).dt.tz_convert(display_timezone)
    direction_up = market.mid_close.ge(market.mid_open)
    volume_colors = direction_up.map({True: "#34D399", False: "#F87171"})
    state = market.is_complete.map({True: "completa", False: "in formazione"})
    hover = [
        (
            f"<b>{stamp:%d/%m/%Y %H:%M} {stamp.tzname()}</b><br>"
            f"Stato: {bar_state}<br>Open: {open_:,.2f}<br>High: {high:,.2f}<br>"
            f"Low: {low:,.2f}<br>Close: {close:,.2f}<br>Spread: {spread:,.2f}"
        )
        for stamp, bar_state, open_, high, low, close, spread in zip(
            local_time, state, market.mid_open, market.mid_high, market.mid_low,
            market.mid_close, market.spread_close, strict=True,
        )
    ]
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=.025,
        row_heights=[.82, .18],
    )
    figure.add_trace(go.Candlestick(
        x=local_time,
        open=market.mid_open,
        high=market.mid_high,
        low=market.mid_low,
        close=market.mid_close,
        increasing={"line": {"color": "#34D399", "width": 1.2}, "fillcolor": "#163F3B"},
        decreasing={"line": {"color": "#F87171", "width": 1.2}, "fillcolor": "#4A2530"},
        text=hover,
        hoverinfo="text",
        name=f"XAUUSD {timeframe}",
    ), row=1, col=1)
    figure.add_trace(go.Bar(
        x=local_time, y=market.tick_volume, marker_color=volume_colors,
        opacity=.42, name="Tick volume", hovertemplate="Volume: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)
    latest = market.iloc[-1]
    figure.add_hline(
        y=float(latest.mid_close), line_color="#60A5FA", line_width=1,
        line_dash="dot", annotation_text=f" {_price(float(latest.mid_close))}",
        annotation_position="right", row=1, col=1,
    )
    uirevision = f"{timeframe}-{int(latest.timestamp)}" if auto_follow else f"{timeframe}-manual"
    figure.update_layout(
        height=680,
        margin={"l": 12, "r": 58, "t": 15, "b": 12},
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font={"color": "#CBD5E1", "family": "Inter"},
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,
        uirevision=uirevision,
        xaxis_rangeslider_visible=False,
    )
    figure.update_xaxes(
        showgrid=True, gridcolor="#23324A", zeroline=False, fixedrange=False,
        showspikes=True, spikecolor="#94A3B8", spikethickness=1, spikemode="across",
        rangeslider_visible=False,
    )
    figure.update_yaxes(
        showgrid=True, gridcolor="#23324A", zeroline=False, fixedrange=False,
        side="right", showspikes=True, spikecolor="#94A3B8", spikethickness=1,
    )
    figure.update_yaxes(title_text="Prezzo", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    return figure


@st.cache_resource
def get_live_service(project_root: str) -> LiveMarketService:
    return LiveMarketService(Path(project_root))


@st.fragment(run_every=1)
def live_market_panel(project_root: str, show_paper_controls: bool = False) -> None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        timeframe = st.segmented_control(
            "Timeframe", ["M1", "M5"], default="M1", required=True,
            key=f"live_timeframe_{show_paper_controls}", persist_state="session",
        )
        visible = st.segmented_control(
            "Candele", [100, 250, 500], default=500, required=True,
            key=f"live_visible_{show_paper_controls}", persist_state="session",
        )
        auto_follow = st.toggle(
            "Segui il prezzo", value=True, key=f"live_follow_{show_paper_controls}",
            persist_state="session",
        )
    try:
        service = get_live_service(project_root)
        snapshot = service.poll()
    except Exception as exc:
        st.error(f"Feed MT5 non disponibile: {exc}", icon=":material/error:")
        st.caption("Apri MetaTrader 5, accedi al conto demo e lascia visibile XAUUSD.")
        return

    tick = snapshot.tick
    local_tick = tick.datetime_utc.tz_convert(service.display_timezone)
    now = pd.Timestamp(datetime.now(timezone.utc))
    age_seconds = max(0.0, (now - tick.datetime_utc).total_seconds())
    offset_hours = (snapshot.status.server_utc_offset_seconds or 0) / 3600
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge("MT5 connesso", icon=":material/check_circle:", color="green")
        st.badge("Conto demo" if snapshot.status.account_demo else "Conto non-demo", color="blue" if snapshot.status.account_demo else "red")
        st.badge("Ordini Python disabilitati", icon=":material/lock:", color="gray")
        validation = snapshot.m5_validation
        st.badge(
            f"M5 verificato ({validation['matched_bars']} barre)" if validation.get("valid") else "M5 da verificare",
            icon=":material/verified:" if validation.get("valid") else ":material/warning:",
            color="green" if validation.get("valid") else "orange",
        )
        st.caption(f"{snapshot.status.server} · {snapshot.status.symbol} · server UTC{offset_hours:+g}")

    with st.container(horizontal=True):
        st.metric("BID", _price(tick.bid), border=True)
        st.metric("ASK", _price(tick.ask), border=True)
        st.metric("Spread", _price(tick.spread), border=True)
        st.metric("Ultimo tick", f"{local_tick:%H:%M:%S}", f"{age_seconds:.1f}s", delta_color="off", border=True)

    st.caption(
        f"Ora grafico: {local_tick:%d/%m/%Y %H:%M:%S} Europe/Rome · "
        f"UTC: {tick.datetime_utc:%d/%m/%Y %H:%M:%S} · feed MT5 del broker"
    )
    figure = live_candlestick_figure(
        snapshot.m1_bars, str(timeframe), service.display_timezone, int(visible), bool(auto_follow),
    )
    st.plotly_chart(
        figure,
        width="stretch",
        height=680,
        key=f"live_chart_{timeframe}_{show_paper_controls}",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "doubleClick": "reset+autosize",
        },
    )

    inference = snapshot.inference
    with st.container(border=True):
        st.subheader("Inference realtime", help="Si aggiorna una volta per ogni candela M1 completata.")
        with st.container(horizontal=True):
            st.metric("P_up 1m", "N/D", border=True)
            st.metric("P_up 3m", "N/D", border=True)
            st.metric("P_up 5m", f"{inference.probability_up:.1%}" if inference.probability_up is not None else "N/D", border=True)
            st.metric("P_up 10m", "N/D", border=True)
            st.metric("Segnale finale", inference.final_signal, border=True)
        if inference.available:
            inference_local = inference.inference_time_utc.tz_convert(service.display_timezone) if inference.inference_time_utc is not None else None
            st.caption(
                f"Modello: {inference.model} · candidato: {inference.candidate} · "
                f"ultima candela elaborata: {inference_local:%d/%m/%Y %H:%M} Europe/Rome"
            )
        st.warning(inference.reason, icon=":material/science:")

    if show_paper_controls:
        with st.container(border=True):
            st.subheader("Paper account")
            st.caption("Phase 7: saldo virtuale, posizioni, PnL e marker sul grafico.")
            with st.container(horizontal=True):
                st.button("Avvia paper", icon=":material/play_arrow:", disabled=True)
                st.button("Ferma paper", icon=":material/stop:", disabled=True)
                st.button("Reset account", icon=":material/restart_alt:", disabled=True)
