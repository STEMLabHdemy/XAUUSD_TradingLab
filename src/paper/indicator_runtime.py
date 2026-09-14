"""One independent paper portfolio: causal market-structure pullbacks."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.backtest.lab import load_history
from src.features import FeatureEngine
from src.live.inference import LiveInference
from src.live.mt5_client import MarketTick
from src.signals.structure import structure_signals
from .engine import PaperAccount, PaperConfig


class IndicatorPaperRuntime:
    SPECS = {"I42": "I42 - M15 structure + M1 pullback", "I43": "I43 - A000913 adaptive structure"}
    A913_PARAMS = {"reversal_atr": 1.15, "ema_span": 20, "touch_atr": .15,
                   "touch_lookback": 3, "breakout_lookback": 2, "swing_buffer_atr": .0}
    A913_FEATURES = ["return_1m", "return_5m", "return_15m", "return_30m", "atr_15_pct",
                     "rolling_volatility_15m", "rsi_14", "macd_histogram", "roc_10",
                     "ema_distance_20", "ema_distance_50", "bollinger_position", "bollinger_width",
                     "spread_to_atr", "hour_sin", "hour_cos", "session_london", "session_new_york"]

    def __init__(self, root: Path, base: PaperConfig):
        self.directory = root / "data/live/paper/structure_v1"; self.directory.mkdir(parents=True, exist_ok=True)
        meta = self.directory / "run.json"
        if meta.exists(): self.run_id = json.loads(meta.read_text(encoding="utf-8"))["run_id"]
        else:
            self.run_id = f"structure_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            meta.write_text(json.dumps({"run_id": self.run_id, "kind": "causal_market_structure"}, indent=2), encoding="utf-8")
        config = replace(base, strategy_id="I42", entry_mode="controlled", max_open_positions_override=1,
                         persistence=1, cooldown_minutes=10, max_daily_trades=3,
                         stop_loss_price=None, take_profit_price=None, atr_stop_multiple=1.5,
                         atr_break_even_r=1.0, atr_trailing_multiple=2.5)
        account = PaperAccount("structure_v1_i42", self.SPECS["I42"], config, self.directory)
        if account.state.get("run_id") != self.run_id: account.set_run_id(self.run_id, datetime.now(timezone.utc).isoformat())
        if not account.state.get("running"): account.start()
        adaptive_config = replace(base, strategy_id="I43", entry_mode="controlled", max_open_positions_override=1,
                                  persistence=1, cooldown_minutes=10, max_daily_trades=2,
                                  stop_loss_price=None, take_profit_price=None, atr_stop_multiple=2.0,
                                  atr_break_even_r=1.25, atr_trailing_multiple=2.5)
        adaptive = PaperAccount("structure_v1_i43", self.SPECS["I43"], adaptive_config, self.directory)
        if adaptive.state.get("run_id") != self.run_id: adaptive.set_run_id(self.run_id, datetime.now(timezone.utc).isoformat())
        if not adaptive.state.get("running"): adaptive.start()
        self.accounts = {self.SPECS["I42"]: account, self.SPECS["I43"]: adaptive}
        self.root = root; self._gate_day: pd.Timestamp | None = None; self._gate_scaler = None; self._gate_model = None

    def _refresh_gate(self, day: pd.Timestamp) -> None:
        """Fit only on labels available before this UTC day; never on live-day outcomes."""
        history = load_history(self.root, day - pd.Timedelta(days=310), day)
        features = FeatureEngine().transform(history).reset_index(drop=True)
        signals = structure_signals(history, **self.A913_PARAMS)
        frame = features.copy(); frame["signal"] = signals.signal.to_numpy()
        side = np.where(frame.signal.eq("BUY"), 1, np.where(frame.signal.eq("SELL"), -1, 0))
        entry = np.where(side == 1, frame.open_ask.shift(-1), frame.open_bid.shift(-1))
        exit_price = np.where(side == 1, frame.close_bid.shift(-30), frame.close_ask.shift(-30))
        future_net = side * (exit_price - entry) - .10
        times = pd.to_datetime(frame.datetime_utc, utc=True)
        valid = (side != 0) & pd.Series(future_net).notna() & frame[self.A913_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        train = valid & (times < day - pd.Timedelta(minutes=30)) & (times >= day - pd.Timedelta(days=120))
        self._gate_day = day; self._gate_model = self._gate_scaler = None
        if train.sum() < 60 or pd.Series(future_net[train] > 0).nunique() < 2: return
        self._gate_scaler = StandardScaler().fit(frame.loc[train, self.A913_FEATURES].to_numpy(float))
        self._gate_model = LogisticRegression(C=.15, max_iter=500, class_weight="balanced", random_state=42)
        self._gate_model.fit(self._gate_scaler.transform(frame.loc[train, self.A913_FEATURES].to_numpy(float)), (future_net[train] > 0).astype(int))

    def _adaptive_inference(self, bars: pd.DataFrame) -> LiveInference:
        signals = structure_signals(bars, **self.A913_PARAMS); row = signals.iloc[-1]
        timestamp = pd.Timestamp(row.datetime_utc); day = timestamp.floor("D")
        if self._gate_day != day: self._refresh_gate(day)
        candidate = str(row.signal); probability = .0
        if candidate in {"BUY", "SELL"} and self._gate_model is not None:
            current = FeatureEngine().transform(bars).iloc[[-1]][self.A913_FEATURES].replace([np.inf, -np.inf], np.nan)
            if current.notna().all(axis=None): probability = float(self._gate_model.predict_proba(self._gate_scaler.transform(current.to_numpy(float)))[0, 1])
        signal = candidate if probability >= .55 else "HOLD"
        return LiveInference(True, "A000913-adaptive-paper", None, probability, timestamp, signal, candidate,
                             f"{row.reason}; gate={probability:.1%}", signal_id=int(row.timestamp),
                             power=None if pd.isna(row.power) else float(row.power))

    def process(self, tick: MarketTick, bars: pd.DataFrame) -> None:
        if len(bars) < 500: return
        signals = structure_signals(bars)
        row = signals.iloc[-1]; candidate = str(row.signal)
        score = {"BUY": .60, "SELL": .40, "HOLD": .50}[candidate]
        context = {"atr_15": float(row.atr_15), "regime": f"structure_{str(row.structure_trend).lower()}",
                   "prior_return_15m_pct": None, "range_15m_pct": None}
        inference = LiveInference(True, "causal-market-structure", None, score, pd.Timestamp(row.datetime_utc), candidate,
                                  candidate, str(row.reason), signal_id=int(row.timestamp), power=None if pd.isna(row.power) else float(row.power))
        self.accounts[self.SPECS["I42"]].process(tick, inference, context)
        adaptive = self._adaptive_inference(bars)
        adaptive_context = {**context, "regime": "a000913_adaptive", "atr_15": float(row.atr_15)}
        self.accounts[self.SPECS["I43"]].process(tick, adaptive, adaptive_context)
