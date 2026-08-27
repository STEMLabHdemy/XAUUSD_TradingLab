from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .services import (
    ROOT, download_progress, load_experiments, load_market_range, load_model_metrics,
    load_quality_monthly, load_quality_summary, load_recent_market, load_run,
)


def _candlestick(market: pd.DataFrame, title: str) -> go.Figure:
    figure = go.Figure(go.Candlestick(
        x=market.datetime_utc, open=market.mid_open, high=market.mid_high,
        low=market.mid_low, close=market.mid_close, name="MID (diagnostic chart)",
    ))
    figure.update_layout(title=title, xaxis_rangeslider_visible=False, height=520)
    return figure


def dashboard() -> None:
    st.title("XAUUSD TradingLab")
    st.warning("RESEARCH / HISTORICAL OOS — this screen is not realtime LIVE PAPER.")
    market = load_recent_market(500)
    progress = download_progress()
    quality = load_quality_summary()
    experiments = load_experiments()
    columns = st.columns(4)
    if not market.empty:
        latest = market.iloc[-1]
        columns[0].metric("Latest MID", f"{latest.mid_close:,.3f}")
        columns[1].metric("Latest BID / ASK", f"{latest.close_bid:,.3f} / {latest.close_ask:,.3f}")
        columns[2].metric("Spread", f"{latest.spread_close:.3f}")
        columns[3].metric("Latest historical candle", str(latest.datetime_utc))
        st.plotly_chart(_candlestick(market, "Last 500 historical candles"), width="stretch")
    st.subheader("Historical download")
    st.write(f"Paired months: **{progress['paired_months']}**, range: **{progress['earliest_month']} → {progress['latest_month']}**, unpaired months: **{progress['unpaired_months']}**")
    if not quality.empty:
        st.caption(f"Master snapshot: {quality.iloc[0].earliest_candle} → {quality.iloc[0].latest_candle}; {int(quality.iloc[0].number_of_candles):,} candles")
    if not experiments.empty:
        best = experiments.sort_values(["sharpe", "profit_factor"], ascending=False).iloc[0]
        st.subheader("Best provisional experiment (not production-approved)")
        st.write(f"{best.model}, threshold {best.buy_threshold:.2f}, persistence {int(best.persistence)} — PnL {best.net_pnl:.2f}, Sharpe {best.sharpe:.2f}, robustness {best.parameter_robustness_sharpe_median:.2f}")


def live_paper() -> None:
    st.title("Live Paper")
    st.error("MT5 realtime feed is not enabled yet (Phase 6). Historical replay is not presented as LIVE.")
    st.button("START PAPER", disabled=True)
    st.button("STOP PAPER", disabled=True)
    st.button("RESET ACCOUNT", disabled=True)
    st.info("Real broker orders are disabled: ENABLE_LIVE_TRADING = false")


def experiments_page() -> None:
    st.title("Experiment Lab")
    experiments = load_experiments()
    if experiments.empty:
        st.info("No experiments available.")
        return
    sort_by = st.selectbox("Sort by", ["sharpe", "profit_factor", "expectancy", "net_pnl", "max_drawdown"])
    ascending = sort_by == "max_drawdown"
    st.dataframe(experiments.sort_values(sort_by, ascending=ascending), use_container_width=True, hide_index=True)
    st.caption("Ranking should consider robustness, drawdown, trade count and stability—not PnL alone.")


def models_page() -> None:
    st.title("Models")
    metrics = load_model_metrics()
    if metrics.empty:
        st.info("No model metrics available.")
        return
    evaluation = st.selectbox("Evaluation", sorted(metrics.evaluation.unique()))
    selected = metrics[metrics.evaluation.eq(evaluation)]
    st.dataframe(selected.sort_values("brier_score"), use_container_width=True, hide_index=True)
    if "roc_auc" in selected:
        st.bar_chart(selected.set_index("model")["roc_auc"])
    st.warning("All current models and metrics are provisional until final historical retraining.")


def _select_experiment() -> str | None:
    experiments = load_experiments()
    if experiments.empty:
        st.info("No experiment runs available.")
        return None
    labels = {
        row.experiment_id: f"{row.model} | threshold {row.buy_threshold:.2f} | persistence {int(row.persistence)} | PnL {row.net_pnl:.2f}"
        for row in experiments.itertuples()
    }
    return st.selectbox("Experiment", list(labels), format_func=labels.get)


def backtest_page() -> None:
    st.title("Historical Backtest")
    st.warning("Historical OOS backtest — not LIVE.")
    experiment_id = _select_experiment()
    if not experiment_id:
        return
    predictions, trades, equity = load_run(experiment_id)
    if predictions.empty:
        return
    market = load_market_range(int(predictions.timestamp.min()), int(predictions.timestamp.max()))
    figure = _candlestick(market.tail(1500), "Historical BID/ASK-aware backtest")
    visible_start = market.tail(1500).datetime_utc.min()
    visible_trades = trades[pd.to_datetime(trades.entry_time, utc=True) >= visible_start] if not trades.empty else trades
    if not visible_trades.empty:
        longs = visible_trades[visible_trades.side.eq("LONG")]
        shorts = visible_trades[visible_trades.side.eq("SHORT")]
        figure.add_trace(go.Scatter(x=longs.entry_time, y=longs.entry_price, mode="markers", marker_symbol="triangle-up", marker_color="green", marker_size=12, name="BUY"))
        figure.add_trace(go.Scatter(x=shorts.entry_time, y=shorts.entry_price, mode="markers", marker_symbol="triangle-down", marker_color="red", marker_size=12, name="SELL"))
        figure.add_trace(go.Scatter(x=visible_trades.exit_time, y=visible_trades.exit_price, mode="markers", marker_symbol="x", marker_color="orange", marker_size=10, name="EXIT"))
    st.plotly_chart(figure, width="stretch")
    if not equity.empty:
        st.plotly_chart(go.Figure(go.Scatter(x=equity.datetime_utc, y=equity.equity, name="Equity")), width="stretch")


def predictions_page() -> None:
    st.title("Prediction History")
    experiment_id = _select_experiment()
    if not experiment_id:
        return
    predictions, _, _ = load_run(experiment_id)
    probability_columns = [column for column in predictions if column.startswith("p_up_")]
    figure = go.Figure()
    for column in probability_columns:
        figure.add_trace(go.Scatter(x=predictions.datetime_utc, y=predictions[column], name=column))
    figure.add_trace(go.Scatter(x=predictions.datetime_utc, y=predictions.temporal_score, name="temporal_score", line={"width": 2}))
    figure.add_hline(y=.68, line_dash="dash", line_color="green")
    figure.add_hline(y=.32, line_dash="dash", line_color="red")
    st.plotly_chart(figure, width="stretch")
    st.dataframe(predictions.tail(1000), use_container_width=True, hide_index=True)


def trades_page() -> None:
    st.title("Trade History")
    experiment_id = _select_experiment()
    if not experiment_id:
        return
    _, trades, _ = load_run(experiment_id)
    st.dataframe(trades, use_container_width=True, hide_index=True)
    st.download_button("Export trades CSV", trades.to_csv(index=False), f"trades_{experiment_id}.csv", "text/csv")


def data_page() -> None:
    st.title("Data Quality")
    summary, monthly = load_quality_summary(), load_quality_monthly()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    if not monthly.empty:
        st.subheader("Monthly audit")
        st.dataframe(monthly, use_container_width=True, hide_index=True)


def settings_page() -> None:
    st.title("Settings")
    st.checkbox("Auto update recent historical data on startup", value=True, disabled=True, help="Will update only recent months; full history is never launched by the GUI.")
    st.toggle("ENABLE_LIVE_TRADING", value=False, disabled=True)
    st.info("Real-money order routing is intentionally unavailable.")
    for name in ("data", "features", "models", "training", "strategy", "backtest", "live"):
        with st.expander(f"configs/{name}.yaml"):
            st.code((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"), language="yaml")
