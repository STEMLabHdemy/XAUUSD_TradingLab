"""Causal historical replay of an adaptive meta-model for I42 signals.

The learner never sees a label until 30 completed minutes after its decision.
It is refit once per UTC day from a trailing 120-day event window, then gates
only that day's structure signals.  This is a walk-forward simulation, not a
model trained on all 90 evaluation days at once.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.backtest.engine import BacktestConfig, Backtester
from src.backtest.lab import load_history
from src.backtest.metrics import performance_metrics
from src.features import FeatureEngine
from src.signals.structure import structure_signals

PARAMS = {"reversal_atr": 1.7, "ema_span": 20, "touch_atr": .30,
          "touch_lookback": 3, "breakout_lookback": 1, "swing_buffer_atr": .30}
FEATURES = ["return_1m", "return_5m", "return_15m", "return_30m", "atr_15_pct",
            "rolling_volatility_15m", "rsi_14", "macd_histogram", "roc_10",
            "ema_distance_20", "ema_distance_50", "bollinger_position", "bollinger_width",
            "spread_to_atr", "hour_sin", "hour_cos", "session_london", "session_new_york"]


def _records(frame: pd.DataFrame) -> list[dict]:
    data = frame.replace({np.nan: None}).copy()
    for column in data.columns:
        if "time" in str(column):
            values = pd.to_datetime(data[column], utc=True, errors="coerce")
            if values.notna().any(): data[column] = values.map(lambda value: value.isoformat() if pd.notna(value) else None)
    return data.to_dict("records")


def run(
    root: Path, days: int = 90, bootstrap_days: int = 180, threshold: float = .55,
    rolling_training_days: int = 120, params: dict | None = None, persist: bool = True,
) -> dict:
    root = root.resolve(); end = pd.Timestamp.now(tz="UTC").floor("min")
    evaluation_start = end - pd.Timedelta(days=days)
    history_start = evaluation_start - pd.Timedelta(days=bootstrap_days)
    bars = load_history(root, history_start, end)
    features = FeatureEngine().transform(bars)
    active_params = dict(PARAMS if params is None else params)
    signals = structure_signals(bars, **active_params)
    frame = features.copy().reset_index(drop=True)
    frame["signal"] = signals.signal.to_numpy(); frame["structure_power"] = signals.power.to_numpy()
    frame["side"] = np.where(frame.signal.eq("BUY"), 1, np.where(frame.signal.eq("SELL"), -1, 0))
    # Executable fixed-horizon outcome: bid/ask already includes spread;
    # slippage applies on both sides at entry and exit.
    horizon = 30; entry = np.where(frame.side.eq(1), frame.open_ask.shift(-1), frame.open_bid.shift(-1))
    exit_price = np.where(frame.side.eq(1), frame.close_bid.shift(-horizon), frame.close_ask.shift(-horizon))
    frame["future_net"] = frame.side * (exit_price - entry) - .10
    frame["label"] = (frame.future_net > 0).astype(int)
    valid = frame.side.ne(0) & frame.future_net.notna() & frame[FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    frame["probability"] = np.nan
    day_series = pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("D")
    eval_days = pd.date_range(evaluation_start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    for day in eval_days:
        train = valid & (day_series < day - pd.Timedelta(minutes=horizon)) & (day_series >= day - pd.Timedelta(days=rolling_training_days))
        today = valid & (day_series == day)
        if train.sum() < 60 or today.sum() == 0 or frame.loc[train, "label"].nunique() < 2:
            continue
        x_train = frame.loc[train, FEATURES].to_numpy(float); y_train = frame.loc[train, "label"].to_numpy(int)
        scaler = StandardScaler().fit(x_train)
        # Interpretable, regularised challenger. class_weight prevents the
        # frequent non-profitable class from making a useless all-NO_TRADE gate.
        model = LogisticRegression(C=.15, max_iter=500, class_weight="balanced", random_state=42)
        model.fit(scaler.transform(x_train), y_train)
        frame.loc[today, "probability"] = model.predict_proba(scaler.transform(frame.loc[today, FEATURES].to_numpy(float)))[:, 1]
    chosen = pd.to_datetime(frame.datetime_utc, utc=True).between(evaluation_start, end)
    base = frame.loc[chosen].copy().reset_index(drop=True)
    base_signal = base.signal.copy()
    gated_signal = base_signal.where(base.probability.ge(threshold), "HOLD")
    cfg = BacktestConfig(stop_loss_price=None, take_profit_price=None, max_holding_minutes=None,
        max_daily_trades=2, cooldown_minutes=10, atr_stop_multiple=2.0,
        atr_break_even_r=1.25, atr_trailing_multiple=2.5)
    baseline = base.copy(); baseline["signal"] = base_signal
    adaptive = base.copy(); adaptive["signal"] = gated_signal
    plain = Backtester(cfg).run(baseline); gated = Backtester(cfg).run(adaptive)
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {"name": "Daily rolling logistic gate", "params": active_params, "features": FEATURES,
                   "label": "30-minute executable net outcome", "threshold": threshold,
                   "bootstrap_days": bootstrap_days, "rolling_training_days": rolling_training_days,
                   "warning": "Historical walk-forward experiment only; not approved for paper/live execution."},
        "period": {"start": evaluation_start.isoformat(), "end": end.isoformat(), "days": days},
        "coverage": {"base_events": int((base_signal != "HOLD").sum()), "predicted_events": int(base.probability.notna().sum()),
                     "gated_events": int((gated_signal != "HOLD").sum())},
        "baseline": {"metrics": performance_metrics(plain.trades, plain.equity_curve, cfg.starting_capital), "trades": _records(plain.trades), "equity": _records(plain.equity_curve[["datetime_utc", "equity"]])},
        "adaptive": {"metrics": performance_metrics(gated.trades, gated.equity_curve, cfg.starting_capital), "trades": _records(gated.trades), "equity": _records(gated.equity_curve[["datetime_utc", "equity"]])},
        "candles": [{"time": int(pd.Timestamp(row.datetime_utc).timestamp()), "open": float(row.mid_open), "high": float(row.mid_high), "low": float(row.mid_low), "close": float(row.mid_close), "volume": float(getattr(row, "tick_volume", 0.0))} for row in base.itertuples(index=False)]}
    if persist:
        target = root / "results" / "online_adaptation" / "result.json"; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--days", type=int, default=90); args = parser.parse_args()
    result = run(args.project_root, args.days); print(json.dumps({"baseline": result["baseline"]["metrics"], "adaptive": result["adaptive"]["metrics"]}, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
