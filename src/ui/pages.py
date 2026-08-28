from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .live_dashboard import live_market_panel
from .live_dashboard import get_paper_runtime
from src.modeling.training_jobs import compatible_models, list_training_jobs, start_training_job

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
    st.caption("Mercato realtime MT5 · ricerca quantitativa e paper trading")
    live_market_panel(str(ROOT))
    progress = download_progress()
    quality = load_quality_summary()
    experiments = load_experiments()
    st.subheader("Archivio storico")
    st.write(f"Mesi BID/ASK completi: **{progress['paired_months']}**, intervallo: **{progress['earliest_month']} → {progress['latest_month']}**, mesi spaiati: **{progress['unpaired_months']}**")
    if not quality.empty:
        st.caption(f"Master snapshot: {quality.iloc[0].earliest_candle} → {quality.iloc[0].latest_candle}; {int(quality.iloc[0].number_of_candles):,} candles")
    if not experiments.empty:
        best = experiments.sort_values(["sharpe", "profit_factor"], ascending=False).iloc[0]
        st.subheader("Best provisional experiment (not production-approved)")
        st.write(f"{best.model}, threshold {best.buy_threshold:.2f}, persistence {int(best.persistence)} — PnL {best.net_pnl:.2f}, Sharpe {best.sharpe:.2f}, robustness {best.parameter_robustness_sharpe_median:.2f}")


def live_paper() -> None:
    st.title("Live paper")
    st.caption("Feed reale MT5, capitale virtuale; nessun ordine viene inviato al broker.")
    live_market_panel(str(ROOT), show_paper_controls=True)


def experiments_page() -> None:
    st.title("Experiment Lab")
    experiments = load_experiments()
    if experiments.empty:
        st.info("No experiments available.")
        return
    sort_by = st.selectbox("Sort by", ["sharpe", "profit_factor", "expectancy", "net_pnl", "max_drawdown"])
    ascending = sort_by == "max_drawdown"
    st.dataframe(experiments.sort_values(sort_by, ascending=ascending), width="stretch", hide_index=True)
    st.caption("Ranking should consider robustness, drawdown, trade count and stability—not PnL alone.")


def models_page() -> None:
    st.title("Laboratorio modelli")
    st.caption("Avvia training temporali, confronta le metriche OOS e scegli i modelli compatibili con il paper trading.")
    catalog_tab, train_tab, results_tab = st.tabs(["Catalogo e selezione", "Nuovo training", "Risultati"])

    with catalog_tab:
        catalog = compatible_models(ROOT)
        if catalog.empty:
            st.info("Non ci sono artefatti binari compatibili con il paper trading.")
        else:
            display = catalog.copy()
            display.insert(0, "id", display.index.astype(str))
            st.dataframe(
                display.drop(columns=["artifact", "manifest"]), width="stretch", hide_index=True,
                column_config={
                    "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
                    "brier_score": st.column_config.NumberColumn("Brier (meno è meglio)", format="%.4f"),
                    "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
                    "rows": st.column_config.NumberColumn("Righe", format="%d"),
                },
            )
            choices = {
                str(index): (
                    f"{row['model']} · {row['calibration']} · H{row['horizon']} · "
                    f"{row['run']} · AUC {row.get('roc_auc', float('nan')):.3f}"
                )
                for index, row in catalog.iterrows()
            }
            selection_path = ROOT / "data/live/paper/model_selection.json"
            selected_artifacts: set[str] = set()
            if selection_path.exists():
                try:
                    selected_artifacts = {
                        str(item["artifact"])
                        for item in json.loads(selection_path.read_text(encoding="utf-8")).get("models", [])
                    }
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    selected_artifacts = set()
            default_ids = [
                str(index) for index, row in catalog.iterrows()
                if str(row["artifact"]) in selected_artifacts
            ]
            if not default_ids:
                sigmoid = catalog[catalog["calibration"].eq("sigmoid") & catalog["run"].eq("Attivo")]
                default_ids = [str(index) for index in sigmoid.index[:3]]
            selected_ids = st.multiselect(
                "Modelli da confrontare nel Live Paper (massimo 4)", list(choices),
                default=default_ids, format_func=choices.get,
            )
            st.caption("La selezione crea account paper separati. Non invia ordini reali e non sovrascrive i modelli esistenti.")
            if st.button("Usa questa selezione nel paper", type="primary", icon=":material/check:"):
                if not selected_ids or len(selected_ids) > 4:
                    st.error("Seleziona da 1 a 4 modelli.")
                else:
                    models = []
                    used_labels: set[str] = set()
                    for selected_id in selected_ids:
                        row = catalog.loc[int(selected_id)]
                        base_label = f"{row['model']} · H{int(row['horizon'])} · {row['calibration']}"
                        label = base_label if base_label not in used_labels else f"{base_label} · {row['run']}"
                        used_labels.add(label)
                        models.append({
                            "label": label, "artifact": row["artifact"], "manifest": row["manifest"],
                        })
                    selection_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = selection_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"models": models}, ensure_ascii=False, indent=2), encoding="utf-8")
                    temporary.replace(selection_path)
                    get_paper_runtime.clear()
                    st.success("Selezione salvata. Il Live Paper userà questi modelli al prossimo accesso.")

    with train_tab:
        st.info(
            "Il training parte in background e usa separazione temporale, walk-forward, gap e un blocco OOS intatto. "
            "Gli artefatti finiscono in una nuova cartella: nulla viene promosso automaticamente.",
            icon=":material/info:",
        )
        with st.form("new_model_training"):
            rows = st.select_slider(
                "Righe storiche", options=[50_000, 100_000, 250_000, 500_000, 1_000_000], value=250_000,
                help="Più righe richiedono più tempo e memoria.",
            )
            horizon = st.segmented_control("Orizzonte target (minuti)", [1, 3, 5, 10, 15], default=5, required=True)
            model_names = st.multiselect(
                "Famiglie", ["logistic_regression", "lightgbm", "xgboost"],
                default=["logistic_regression", "lightgbm", "xgboost"],
            )
            submitted = st.form_submit_button("Avvia training", type="primary", icon=":material/model_training:")
        if submitted:
            try:
                job = start_training_job(ROOT, int(rows), int(horizon), list(model_names))
                st.success(f"Training {job['run_id']} avviato in background (PID {job['pid']}).")
            except (ValueError, OSError) as exc:
                st.error(f"Impossibile avviare il training: {exc}")

    with results_tab:
        jobs = list_training_jobs(ROOT)
        if not jobs:
            st.info("Non ci sono ancora training avviati dall'interfaccia.")
        else:
            summary = pd.DataFrame([{key: job.get(key) for key in (
                "run_id", "status", "started_at", "finished_at", "rows", "horizon", "models", "pid", "error"
            )} for job in jobs])
            st.dataframe(summary, width="stretch", hide_index=True)
            run_id = st.selectbox("Dettaglio esecuzione", [job["run_id"] for job in jobs])
            job = next(item for item in jobs if item["run_id"] == run_id)
            metrics_path = Path(job["directory"]) / "results/baseline_metrics_provisional.csv"
            if metrics_path.exists():
                run_metrics = pd.read_csv(metrics_path)
                evaluations = [value for value in run_metrics.evaluation.unique() if "untouched_oos" in value]
                evaluation = st.selectbox("Valutazione", evaluations, key="training_result_evaluation")
                selected = run_metrics[run_metrics.evaluation.eq(evaluation)].sort_values("brier_score")
                st.dataframe(selected, width="stretch", hide_index=True)
                if "roc_auc" in selected:
                    st.bar_chart(selected.set_index("model")["roc_auc"])
            with st.expander("Log del training"):
                st.code(job.get("log") or "Log non ancora disponibile.")

    st.warning("Metriche buone non garantiscono profitti: prima della promozione serve sempre una verifica economica OOS con spread e costi.")


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
    st.dataframe(predictions.tail(1000), width="stretch", hide_index=True)


def trades_page() -> None:
    st.title("Trade History")
    experiment_id = _select_experiment()
    if not experiment_id:
        return
    _, trades, _ = load_run(experiment_id)
    st.dataframe(trades, width="stretch", hide_index=True)
    st.download_button("Export trades CSV", trades.to_csv(index=False), f"trades_{experiment_id}.csv", "text/csv")


def data_page() -> None:
    st.title("Data Quality")
    summary, monthly = load_quality_summary(), load_quality_monthly()
    st.dataframe(summary, width="stretch", hide_index=True)
    if not monthly.empty:
        st.subheader("Monthly audit")
        st.dataframe(monthly, width="stretch", hide_index=True)


def settings_page() -> None:
    st.title("Settings")
    st.checkbox("Auto update recent historical data on startup", value=True, disabled=True, help="Will update only recent months; full history is never launched by the GUI.")
    st.toggle("ENABLE_LIVE_TRADING", value=False, disabled=True)
    st.info("Real-money order routing is intentionally unavailable.")
    for name in ("data", "features", "models", "training", "strategy", "backtest", "paper", "live"):
        with st.expander(f"configs/{name}.yaml"):
            st.code((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"), language="yaml")
