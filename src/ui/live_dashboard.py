from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
from xauusd_realtime_chart import xauusd_realtime_chart
import yaml

from src.live import LiveMarketService
from src.live.timeframes import chart_bars
from src.paper import PaperConfig, PaperRuntime


def _price(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _pnl_color(value: float) -> str:
    if value > 0:
        return "color: #34D399; font-weight: 600"
    if value < 0:
        return "color: #F87171; font-weight: 600"
    return "color: #94A3B8"


def _open_positions_frame(
    runtime: PaperRuntime,
    tick: object,
    display_timezone: str,
) -> pd.DataFrame:
    """Build one live, human-readable row for every open paper position."""
    rows: list[dict[str, object]] = []
    now = pd.Timestamp(tick.datetime_utc)
    for model, account in runtime.accounts.items():
        state = account.snapshot()
        position = state["position"]
        if not position:
            continue
        side = str(position["side"])
        entry_time = pd.Timestamp(position["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")
        held_minutes = max(0, int((now - entry_time).total_seconds() // 60))
        mark_price = float(tick.bid if side == "LONG" else tick.ask)
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        stop_distance = (
            mark_price - float(stop_loss) if side == "LONG" else float(stop_loss) - mark_price
        ) if stop_loss is not None else None
        take_distance = (
            float(take_profit) - mark_price if side == "LONG" else mark_price - float(take_profit)
        ) if take_profit is not None else None
        quantity = float(position["quantity"])
        notional = mark_price * quantity
        margin = notional / float(account.config.leverage)
        risk_to_stop = max(0.0, float(stop_distance) * quantity) if stop_distance is not None else None
        rows.append({
            "Modello": model,
            "Direzione": f"▲ {side}" if side == "LONG" else f"▼ {side}",
            "Aperta alle": entry_time.tz_convert(display_timezone).strftime("%d/%m %H:%M:%S"),
            "Durata": f"{held_minutes // 60}h {held_minutes % 60:02d}m" if held_minutes >= 60 else f"{held_minutes} min",
            "Quantità": quantity,
            "Nozionale": notional,
            "Margine": margin,
            "Rischio a SL": risk_to_stop,
            "Ingresso": float(position["entry_price"]),
            "Prezzo ora": mark_price,
            "PnL aperto": float(state["unrealized_pnl"]),
            "SL": float(stop_loss) if stop_loss is not None else None,
            "Distanza SL": stop_distance,
            "TP": float(take_profit) if take_profit is not None else None,
            "Distanza TP": take_distance,
        })
    return pd.DataFrame(rows)


def _portfolio_totals(runtime: PaperRuntime) -> dict[str, float | int]:
    states = [(account, account.snapshot()) for account in runtime.accounts.values()]
    starting_capital = sum(account.config.starting_capital for account, _ in states)
    realized = sum(float(state["realized_pnl"]) for _, state in states)
    unrealized = sum(float(state["unrealized_pnl"]) for _, state in states)
    return {
        "starting_capital": starting_capital,
        "equity": sum(float(state["equity"]) for _, state in states),
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "open_positions": sum(state["position"] is not None for _, state in states),
        "exposure": sum(float(state["exposure"]) for _, state in states),
        "used_margin": sum(float(state["used_margin"]) for _, state in states),
    }


def _all_trades_frame(runtime: PaperRuntime, tick: object, display_timezone: str) -> pd.DataFrame:
    """One ledger containing every open and closed trade from every paper account."""
    rows: list[dict[str, object]] = []
    for model, account in runtime.accounts.items():
        state = account.snapshot()
        position = state.get("position")
        if position:
            side = str(position["side"])
            mark = float(tick.bid if side == "LONG" else tick.ask)
            quantity = float(position["quantity"])
            rows.append({
                "Stato": "APERTO", "Modello": model, "Trade": position["trade_id"],
                "Direzione": side, "Apertura": position["entry_time"], "Chiusura": None,
                "Ingresso": float(position["entry_price"]), "Prezzo/Uscita": mark,
                "Quantità": quantity, "Nozionale": mark * quantity,
                "Margine": mark * quantity / account.config.leverage,
                "SL": position.get("stop_loss"), "TP": position.get("take_profit"),
                "PnL": float(state["unrealized_pnl"]), "Costi": None, "Motivo uscita": "—",
            })
        for trade in state.get("trades", []):
            quantity = float(trade["quantity"])
            rows.append({
                "Stato": "CHIUSO", "Modello": model, "Trade": trade["trade_id"],
                "Direzione": trade["side"], "Apertura": trade["entry_time"],
                "Chiusura": trade["exit_time"], "Ingresso": float(trade["entry_price"]),
                "Prezzo/Uscita": float(trade["exit_price"]), "Quantità": quantity,
                "Nozionale": float(trade["raw_entry_price"]) * quantity,
                "Margine": float(trade["raw_entry_price"]) * quantity / account.config.leverage,
                "SL": trade.get("stop_loss"), "TP": trade.get("take_profit"),
                "PnL": float(trade["net_pnl"]), "Costi": float(trade["costs"]),
                "Motivo uscita": trade["exit_reason"],
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["Apertura"] = pd.to_datetime(frame["Apertura"], utc=True).dt.tz_convert(display_timezone)
    frame["Chiusura"] = pd.to_datetime(frame["Chiusura"], utc=True, errors="coerce").dt.tz_convert(display_timezone)
    return frame.sort_values("Apertura", ascending=False).reset_index(drop=True)


def _render_portfolio_overview(runtime: PaperRuntime, tick: object, display_timezone: str) -> None:
    with st.container(border=True):
        st.subheader("Portafoglio paper in tempo reale")
        st.caption("Somma dei tre modelli sullo stesso feed MT5. Capitale virtuale; nessun ordine viene inviato al broker.")
        totals = _portfolio_totals(runtime)
        total_return = totals["total_pnl"] / totals["starting_capital"] if totals["starting_capital"] else 0.0
        with st.container(horizontal=True):
            st.metric(
                "PnL totale",
                f"{totals['total_pnl']:+,.2f} USD",
                f"{total_return:+.3%} sul capitale",
                border=True,
            )
            st.metric("PnL realizzato", f"{totals['realized_pnl']:+,.2f} USD", border=True)
            st.metric("PnL posizioni aperte", f"{totals['unrealized_pnl']:+,.2f} USD", border=True)
            st.metric("Posizioni aperte", f"{totals['open_positions']} / {len(runtime.accounts)}", border=True)
            st.metric("Esposizione", f"{totals['exposure']:,.2f} USD", border=True)
            st.metric("Margine usato", f"{totals['used_margin']:,.2f} USD", border=True)

        positions = _open_positions_frame(runtime, tick, display_timezone)
        st.markdown("**Posizioni aperte adesso**")
        st.caption(
            "Quantità = unità di XAUUSD. Nozionale = prezzo × quantità; margine = nozionale ÷ leva. "
            "Il margine non è la perdita massima: il rischio a SL è la distanza dallo stop × quantità, prima dei costi."
        )
        if positions.empty:
            st.info("Nessun modello ha una posizione aperta in questo momento.", icon=":material/hourglass_empty:")
            return
        styled_positions = positions.style.map(_pnl_color, subset=["PnL aperto"])
        st.dataframe(
            styled_positions,
            width="stretch",
            hide_index=True,
            column_config={
                "Modello": st.column_config.TextColumn(pinned=True),
                "Quantità": st.column_config.NumberColumn(format="%.2f"),
                "Nozionale": st.column_config.NumberColumn(format="%.2f USD"),
                "Margine": st.column_config.NumberColumn(format="%.2f USD"),
                "Rischio a SL": st.column_config.NumberColumn(format="%.2f USD"),
                "Ingresso": st.column_config.NumberColumn(format="%.2f"),
                "Prezzo ora": st.column_config.NumberColumn(format="%.2f"),
                "PnL aperto": st.column_config.NumberColumn(format="%+.2f USD"),
                "SL": st.column_config.NumberColumn(format="%.2f"),
                "Distanza SL": st.column_config.NumberColumn(format="%+.2f"),
                "TP": st.column_config.NumberColumn(format="%.2f"),
                "Distanza TP": st.column_config.NumberColumn(format="%+.2f"),
            },
        )


def live_candlestick_figure(
    m1_bars: pd.DataFrame,
    timeframe: str,
    display_timezone: str,
    limit: int,
    auto_follow: bool,
    events: pd.DataFrame | None = None,
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
    has_markers = False
    if events is not None and not events.empty:
        event_frame = events.copy()
        event_frame["local_time"] = pd.to_datetime(event_frame.timestamp, utc=True).dt.tz_convert(display_timezone)
        event_frame = event_frame[event_frame.local_time.between(local_time.min(), local_time.max())]
        has_markers = not event_frame.empty
        for event, symbol, color in (("BUY", "triangle-up", "#22C55E"), ("SELL", "triangle-down", "#EF4444"), ("EXIT", "x", "#F59E0B")):
            selected = event_frame[event_frame.event.eq(event)]
            if selected.empty:
                continue
            hover = []
            for row in selected.to_dict("records"):
                pnl = row.get("net_pnl")
                hover.append(
                    f"<b>{event}</b><br>{row['local_time']:%d/%m/%Y %H:%M:%S}<br>"
                    f"Prezzo: {float(row['price']):,.2f}<br>Modello: {row.get('model', 'N/D')}<br>"
                    f"Confidenza P_up: {float(row.get('confidence')):.1%}<br>Spread: {float(row.get('spread', 0)):,.2f}<br>"
                    f"Rendimento atteso: {row.get('expected_return') or 'N/D'}<br>Regime: {row.get('regime', 'N/D')}<br>"
                    f"SL: {row.get('stop_loss')}<br>TP: {row.get('take_profit')}"
                    + (f"<br>PnL netto: {float(pnl):,.2f}" if pnl is not None else "")
                )
            figure.add_trace(go.Scatter(
                x=selected.local_time, y=selected.price, mode="markers", name=event,
                marker={"symbol": symbol, "color": color, "size": 14, "line": {"color": "white", "width": 1}},
                text=hover, hoverinfo="text",
            ), row=1, col=1)
    latest = market.iloc[-1]
    figure.add_hline(
        y=float(latest.mid_close), line_color="#60A5FA", line_width=1,
        line_dash="dot", annotation_text=f" {_price(float(latest.mid_close))}",
        annotation_position="right", row=1, col=1,
    )
    # Stable across market updates so Plotly preserves client-side zoom and pan.
    # Switching timeframe/follow mode deliberately creates a fresh viewport.
    uirevision = f"xauusd-{timeframe}-{'follow' if auto_follow else 'manual'}-v5"
    figure.update_layout(
        height=680,
        margin={"l": 12, "r": 58, "t": 15, "b": 12},
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font={"color": "#CBD5E1", "family": "Inter"},
        hovermode="x unified",
        dragmode="pan",
        showlegend=has_markers,
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


@st.cache_resource
def get_paper_runtime(project_root: str) -> PaperRuntime:
    root = Path(project_root)
    values = yaml.safe_load((root / "configs/paper.yaml").read_text(encoding="utf-8")) or {}
    return PaperRuntime(root, PaperConfig(**values))


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
            help="Segue il mercato finché non fai zoom o pan. Cambia questo interruttore per ripristinare la vista.",
        )
    # Reserve stable UI locations before the slower MT5/model work. This keeps
    # the previous chart mounted while the next realtime update is computed.
    market_header_slot = st.container()
    chart_slot = st.container()
    try:
        service = get_live_service(project_root)
        snapshot = service.poll()
        runtime = get_paper_runtime(project_root)
        completed = snapshot.m1_bars[snapshot.m1_bars.is_complete.astype(bool)].reset_index(drop=True)
        runtime.process(snapshot.tick, completed)
    except Exception as exc:
        st.error(f"Feed MT5 non disponibile: {exc}", icon=":material/error:")
        st.caption("Apri MetaTrader 5, accedi al conto demo e lascia visibile XAUUSD.")
        return

    tick = snapshot.tick
    local_tick = tick.datetime_utc.tz_convert(service.display_timezone)
    now = pd.Timestamp(datetime.now(timezone.utc))
    age_seconds = max(0.0, (now - tick.datetime_utc).total_seconds())
    offset_hours = (snapshot.status.server_utc_offset_seconds or 0) / 3600
    with market_header_slot:
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
        selected_model = next(iter(runtime.accounts))
        if show_paper_controls:
            selected_model = st.selectbox("Paper account / modello", list(runtime.accounts), key="paper_selected_model")
            _render_portfolio_overview(runtime, tick, service.display_timezone)
    selected_account = runtime.accounts[selected_model]
    selected_state = selected_account.snapshot()
    figure = live_candlestick_figure(
        snapshot.m1_bars, str(timeframe), service.display_timezone, int(visible), bool(auto_follow),
        selected_account.events_frame() if show_paper_controls else None,
    )
    figure_payload = json.loads(pio.to_json(figure, validate=False, remove_uids=True))
    figure_payload.get("layout", {}).pop("template", None)
    with chart_slot:
        xauusd_realtime_chart(
            figure_payload,
            key=f"live_chart_{show_paper_controls}",
            viewport_revision=str(figure.layout.uirevision),
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
                "doubleClick": "reset+autosize",
            },
        )

    inference = snapshot.inference
    paper_inference = runtime.inference_for(selected_model) if show_paper_controls else None
    displayed_inference = paper_inference or inference
    with st.container(border=True):
        st.subheader("Inference realtime", help="Si aggiorna una volta per ogni candela M1 completata.")
        if show_paper_controls:
            with st.container(horizontal=True):
                st.metric("Modello selezionato", selected_model, border=True)
                st.metric(
                    "Orizzonte previsto",
                    f"{displayed_inference.horizon_minutes} minuti" if displayed_inference.horizon_minutes else "N/D",
                    border=True,
                )
                st.metric(
                    f"P_up {displayed_inference.horizon_minutes or '?'}m",
                    f"{displayed_inference.probability_up:.1%}" if displayed_inference.probability_up is not None else "N/D",
                    border=True,
                )
                st.metric("Segnale paper", selected_state["last_signal"], border=True)
            st.caption(
                "La previsione viene aggiornata a ogni candela M1 completa; l'orizzonte indica quanto avanti "
                "nel tempo il modello cerca di prevedere, non ogni quanto viene eseguito."
            )
        else:
            with st.container(horizontal=True):
                st.metric(
                    f"P_up {displayed_inference.horizon_minutes or '?'}m",
                    f"{displayed_inference.probability_up:.1%}" if displayed_inference.probability_up is not None else "N/D",
                    border=True,
                )
                st.metric("Segnale", displayed_inference.final_signal, border=True)
        if displayed_inference.available:
            inference_local = displayed_inference.inference_time_utc.tz_convert(service.display_timezone) if displayed_inference.inference_time_utc is not None else None
            st.caption(
                f"Modello: {displayed_inference.model} · candidato: {displayed_inference.candidate} · "
                f"ultima candela elaborata: {inference_local:%d/%m/%Y %H:%M} Europe/Rome"
            )
        st.warning(displayed_inference.reason, icon=":material/science:")

    if show_paper_controls:
        with st.container(border=True):
            st.subheader(f"Dettaglio account · {selected_model}")
            st.caption("Controlli e risultati del modello selezionato.")
            with st.container(horizontal=True):
                if st.button("Avvia paper", icon=":material/play_arrow:", disabled=bool(selected_state["running"])):
                    selected_account.start(); st.rerun(scope="fragment")
                if st.button("Ferma paper", icon=":material/stop:", disabled=not bool(selected_state["running"])):
                    selected_account.stop(); st.rerun(scope="fragment")
                confirm_reset = st.checkbox("Conferma reset", key=f"confirm_reset_{selected_model}")
                if st.button("Reset account", icon=":material/restart_alt:", disabled=not confirm_reset):
                    selected_account.reset(); st.rerun(scope="fragment")
            selected_state = selected_account.snapshot()
            selected_total_pnl = float(selected_state["realized_pnl"]) + float(selected_state["unrealized_pnl"])
            with st.container(horizontal=True):
                st.metric("Saldo", f"{selected_state['balance']:,.2f} {selected_account.config.currency}", border=True)
                st.metric("Equity", f"{selected_state['equity']:,.2f}", f"{selected_state['equity'] - selected_account.config.starting_capital:+,.2f}", border=True)
                st.metric("PnL totale", f"{selected_total_pnl:+,.2f}", border=True)
                st.metric("PnL aperto", f"{selected_state['unrealized_pnl']:+,.2f}", border=True)
            with st.container(horizontal=True):
                st.metric("PnL realizzato", f"{selected_state['realized_pnl']:+,.2f}", border=True)
                st.metric("Margine libero", f"{selected_state['free_margin']:,.2f}", border=True)
                st.metric("Max drawdown", f"{selected_state['max_drawdown']:.2%}", border=True)
                st.metric("Trade chiusi", f"{len(selected_state['trades'])}", border=True)
            position = selected_state["position"]
            st.info(
                f"{'RUNNING' if selected_state['running'] else 'STOPPED'} · Posizione: "
                f"{position['side'] if position else 'FLAT'} · {selected_state['last_reason']}",
                icon=":material/play_circle:" if selected_state["running"] else ":material/pause_circle:",
            )

            open_models = [name for name, account in runtime.accounts.items() if account.snapshot()["position"]]
            with st.expander("Gestisci una posizione aperta", icon=":material/edit:", expanded=bool(open_models)):
                if not open_models:
                    st.caption("Nessuna posizione aperta da modificare.")
                else:
                    managed_model = st.selectbox("Posizione", open_models, key="managed_paper_position")
                    managed_account = runtime.accounts[managed_model]
                    managed_position = managed_account.snapshot()["position"]
                    side = managed_position["side"]
                    executable = float(tick.bid if side == "LONG" else tick.ask)
                    quantity = float(managed_position["quantity"])
                    st.caption(
                        f"{side} · {quantity:.2f} unità · prezzo eseguibile {executable:.2f} · "
                        f"nozionale {executable * quantity:,.2f} USD · "
                        f"margine {executable * quantity / managed_account.config.leverage:,.2f} USD"
                    )
                    with st.form(f"protection_{managed_account.account_id}", border=False):
                        with st.container(horizontal=True):
                            use_sl = st.checkbox("Stop loss attivo", value=managed_position.get("stop_loss") is not None)
                            new_sl = st.number_input(
                                "Nuovo SL", min_value=0.01,
                                value=float(managed_position.get("stop_loss") or executable), step=0.10,
                            )
                            use_tp = st.checkbox("Take profit attivo", value=managed_position.get("take_profit") is not None)
                            new_tp = st.number_input(
                                "Nuovo TP", min_value=0.01,
                                value=float(managed_position.get("take_profit") or executable), step=0.10,
                            )
                        if st.form_submit_button("Salva SL e TP", icon=":material/save:", type="primary"):
                            try:
                                managed_account.update_protection(
                                    tick, float(new_sl) if use_sl else None, float(new_tp) if use_tp else None
                                )
                                st.success("Protezione aggiornata e registrata nello storico eventi.")
                                st.rerun(scope="fragment")
                            except ValueError as exc:
                                st.error(str(exc))
                    confirm_close = st.checkbox(
                        f"Confermo la chiusura virtuale di {managed_model}",
                        key=f"confirm_manual_close_{managed_account.account_id}",
                    )
                    if st.button(
                        "Chiudi posizione a mercato", icon=":material/close:",
                        disabled=not confirm_close, key=f"manual_close_{managed_account.account_id}",
                    ):
                        managed_account.close_manually(tick)
                        st.rerun(scope="fragment")

            st.subheader("Confronto tra modelli", help="Stesso feed e stessa configurazione, stato e PnL indipendenti.")
            comparison = runtime.comparison().rename(columns={
                "model": "Modello", "status": "Stato", "balance": "Saldo", "equity": "Equity",
                "realized_pnl": "PnL realizzato", "unrealized_pnl": "PnL aperto", "total_pnl": "PnL totale",
                "return_pct": "Rendimento", "free_margin": "Margine libero", "trades": "Trade",
                "position": "Posizione", "signal": "Segnale", "max_drawdown": "Max drawdown",
            })
            styled_comparison = comparison.style.map(
                _pnl_color, subset=["PnL realizzato", "PnL aperto", "PnL totale"]
            )
            st.dataframe(
                styled_comparison,
                width="stretch",
                hide_index=True,
                column_config={
                    "Modello": st.column_config.TextColumn(pinned=True),
                    "Saldo": st.column_config.NumberColumn(format="%.2f"),
                    "Equity": st.column_config.NumberColumn(format="%.2f"),
                    "PnL realizzato": st.column_config.NumberColumn(format="%+.2f"),
                    "PnL aperto": st.column_config.NumberColumn(format="%+.2f"),
                    "PnL totale": st.column_config.NumberColumn(format="%+.2f"),
                    "Rendimento": st.column_config.NumberColumn(format="percent"),
                    "Margine libero": st.column_config.NumberColumn(format="%.2f"),
                    "Max drawdown": st.column_config.NumberColumn(format="percent"),
                },
            )

            with st.expander("Configurazione esperimento"):
                cfg = selected_account.config
                st.caption(
                    "La size è fissa in unità. “Rischio/trade %” è un limite di sicurezza: blocca l'ingresso se "
                    "distanza SL × unità supera quella percentuale dell'equity; non calcola automaticamente la size."
                )
                with st.container(horizontal=True):
                    capital = st.number_input("Capitale", min_value=1.0, value=float(cfg.starting_capital), step=1000.0)
                    size = st.number_input("Unità", min_value=.01, value=float(cfg.position_size_units), step=.1)
                    leverage = st.number_input("Leva", min_value=1.0, value=float(cfg.leverage), step=1.0)
                    risk = st.number_input("Rischio/trade %", min_value=.01, max_value=100.0, value=float(cfg.risk_per_trade_pct), step=.1)
                    sl = st.number_input("Stop loss (prezzo)", min_value=0.01, value=float(cfg.stop_loss_price or 5.0), step=.1)
                    tp = st.number_input("Take profit (prezzo)", min_value=0.01, value=float(cfg.take_profit_price or 10.0), step=.1)
                with st.container(horizontal=True):
                    buy = st.number_input("Soglia BUY", min_value=.51, max_value=.99, value=float(cfg.buy_threshold), step=.01)
                    sell = st.number_input("Soglia SELL", min_value=.01, max_value=.49, value=float(cfg.sell_threshold), step=.01)
                    persistence = st.number_input("Conferme", min_value=1, max_value=5, value=int(cfg.persistence), step=1)
                    cooldown = st.number_input("Cooldown minuti", min_value=0, value=int(cfg.cooldown_minutes), step=1)
                    max_spread = st.number_input("Spread massimo", min_value=0.0, value=float(cfg.max_allowed_spread), step=.1)
                    commission = st.number_input("Commissione/unità/lato", min_value=0.0, value=float(cfg.commission_per_unit_per_side), step=.01)
                    slippage = st.number_input("Slippage/lato", min_value=0.0, value=float(cfg.slippage_price_per_side), step=.01)
                    max_trades = st.number_input("Trade max/giorno", min_value=1, value=int(cfg.max_daily_trades), step=1)
                    max_loss = st.number_input("Perdita max/giorno", min_value=0.0, value=float(cfg.max_daily_loss), step=100.0)
                apply_confirm = st.checkbox("Confermo: applica e resetta tutti e tre gli account", key="paper_config_confirm")
                if st.button("Applica nuova configurazione", disabled=not apply_confirm, icon=":material/tune:"):
                    runtime.reconfigure_and_reset(PaperConfig(
                        **{**asdict(cfg), "starting_capital": capital, "position_size_units": size,
                           "leverage": leverage, "risk_per_trade_pct": risk, "stop_loss_price": sl, "take_profit_price": tp,
                           "buy_threshold": buy, "sell_threshold": sell, "persistence": int(persistence),
                           "cooldown_minutes": int(cooldown), "max_allowed_spread": max_spread,
                           "commission_per_unit_per_side": commission, "slippage_price_per_side": slippage,
                           "max_daily_trades": int(max_trades), "max_daily_loss": max_loss}
                    ))
                    st.rerun(scope="fragment")

            st.subheader("Registro completo dei trade")
            st.caption("Tutte le posizioni aperte e tutti i trade chiusi, riuniti per ogni modello paper.")
            ledger = _all_trades_frame(runtime, tick, service.display_timezone)
            if ledger.empty:
                st.caption("Nessun trade aperto o chiuso per ora.")
            else:
                with st.container(horizontal=True):
                    status_filter = st.segmented_control(
                        "Stato", ["TUTTI", "APERTI", "CHIUSI"], default="TUTTI",
                        key="paper_ledger_status", required=True,
                    )
                    model_filter = st.multiselect(
                        "Modelli", list(runtime.accounts), default=list(runtime.accounts),
                        key="paper_ledger_models",
                    )
                visible_ledger = ledger[ledger["Modello"].isin(model_filter)]
                if status_filter == "APERTI":
                    visible_ledger = visible_ledger[visible_ledger["Stato"].eq("APERTO")]
                elif status_filter == "CHIUSI":
                    visible_ledger = visible_ledger[visible_ledger["Stato"].eq("CHIUSO")]
                styled_ledger = visible_ledger.style.map(_pnl_color, subset=["PnL"])
                st.dataframe(
                    styled_ledger, width="stretch", hide_index=True, height=420,
                    column_config={
                        "Modello": st.column_config.TextColumn(pinned=True),
                        "Apertura": st.column_config.DatetimeColumn(format="DD/MM/YY HH:mm:ss"),
                        "Chiusura": st.column_config.DatetimeColumn(format="DD/MM/YY HH:mm:ss"),
                        "Ingresso": st.column_config.NumberColumn(format="%.2f"),
                        "Prezzo/Uscita": st.column_config.NumberColumn(format="%.2f"),
                        "Quantità": st.column_config.NumberColumn(format="%.2f"),
                        "Nozionale": st.column_config.NumberColumn(format="%.2f USD"),
                        "Margine": st.column_config.NumberColumn(format="%.2f USD"),
                        "SL": st.column_config.NumberColumn(format="%.2f"),
                        "TP": st.column_config.NumberColumn(format="%.2f"),
                        "PnL": st.column_config.NumberColumn(format="%+.2f USD"),
                        "Costi": st.column_config.NumberColumn(format="%.2f USD"),
                    },
                )
                st.download_button(
                    "Esporta tutto in CSV", ledger.to_csv(index=False), "paper_trades_completo.csv", "text/csv",
                    icon=":material/download:",
                )
