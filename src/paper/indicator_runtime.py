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

from src.features import FeatureEngine
from src.live.inference import LiveInference
from src.live.mt5_client import MarketTick
from src.signals.structure import structure_signals
from .engine import PaperAccount, PaperConfig


class IndicatorPaperRuntime:
    SPECS = {
        "I42": "I42 - M15 structure + M1 pullback",
        "I43": "I43 - A000913 adaptive structure",
        "I44": "I44 - A000913 fast (paper experiment)",
        "I45": "I45 - A000913 faster (paper experiment)",
    }
    A913_PARAMS = {"reversal_atr": 1.15, "ema_span": 20, "touch_atr": .15,
                   "touch_lookback": 3, "breakout_lookback": 2, "swing_buffer_atr": .0}
    A913_FEATURES = ["return_1m", "return_5m", "return_15m", "return_30m", "atr_15_pct",
                     "rolling_volatility_15m", "rsi_14", "macd_histogram", "roc_10",
                     "ema_distance_20", "ema_distance_50", "bollinger_position", "bollinger_width",
                     "spread_to_atr", "hour_sin", "hour_cos", "session_london", "session_new_york"]
    # I44/I45 intentionally relax the signal filters.  They are independent
    # paper experiments: more observations, not a claim of better performance.
    ADAPTIVE_SPECS = {
        "I43": {"params": A913_PARAMS, "gate": .55, "train_days": 120, "daily_limit": 2, "cooldown": 10},
        "I44": {"params": {"reversal_atr": .90, "ema_span": 20, "touch_atr": .30,
                            "touch_lookback": 4, "breakout_lookback": 1, "swing_buffer_atr": .0},
                "gate": .50, "train_days": 90, "daily_limit": 4, "cooldown": 5},
        "I45": {"params": {"reversal_atr": .75, "ema_span": 15, "touch_atr": .30,
                            "touch_lookback": 4, "breakout_lookback": 1, "swing_buffer_atr": .0},
                "gate": .50, "train_days": 60, "daily_limit": 5, "cooldown": 3},
    }

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
        self.accounts = {self.SPECS["I42"]: account}
        for strategy_id, spec in self.ADAPTIVE_SPECS.items():
            adaptive_config = replace(base, strategy_id=strategy_id, entry_mode="controlled", max_open_positions_override=1,
                                      persistence=1, cooldown_minutes=spec["cooldown"], max_daily_trades=spec["daily_limit"],
                                      stop_loss_price=None, take_profit_price=None, atr_stop_multiple=2.0,
                                      atr_break_even_r=1.25, atr_trailing_multiple=2.5)
            adaptive = PaperAccount(f"structure_v1_{strategy_id.lower()}", self.SPECS[strategy_id], adaptive_config, self.directory)
            if adaptive.state.get("run_id") != self.run_id: adaptive.set_run_id(self.run_id, datetime.now(timezone.utc).isoformat())
            if not adaptive.state.get("running"): adaptive.start()
            self.accounts[self.SPECS[strategy_id]] = adaptive
        self.root = root
        self._gate_states = {strategy_id: {"day": None, "scaler": None, "model": None}
                             for strategy_id in self.ADAPTIVE_SPECS}

    def _refresh_gate(self, strategy_id: str, day: pd.Timestamp) -> None:
        """Fit only on labels available before this UTC day; never on live-day outcomes."""
        # Imported lazily to avoid the historical lab's optional paper catalog
        # importing this runtime while Python is still initialising it.
        from src.backtest.lab import load_history
        spec = self.ADAPTIVE_SPECS[strategy_id]
        history = load_history(self.root, day - pd.Timedelta(days=max(310, spec["train_days"] + 190)), day)
        features = FeatureEngine().transform(history).reset_index(drop=True)
        signals = structure_signals(history, **spec["params"])
        frame = features.copy(); frame["signal"] = signals.signal.to_numpy()
        side = np.where(frame.signal.eq("BUY"), 1, np.where(frame.signal.eq("SELL"), -1, 0))
        entry = np.where(side == 1, frame.open_ask.shift(-1), frame.open_bid.shift(-1))
        exit_price = np.where(side == 1, frame.close_bid.shift(-30), frame.close_ask.shift(-30))
        future_net = side * (exit_price - entry) - .10
        times = pd.to_datetime(frame.datetime_utc, utc=True)
        valid = (side != 0) & pd.Series(future_net).notna() & frame[self.A913_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        train = valid & (times < day - pd.Timedelta(minutes=30)) & (times >= day - pd.Timedelta(days=spec["train_days"]))
        state = self._gate_states[strategy_id]
        state["day"] = day; state["model"] = state["scaler"] = None
        if train.sum() < 60 or pd.Series(future_net[train] > 0).nunique() < 2: return
        state["scaler"] = StandardScaler().fit(frame.loc[train, self.A913_FEATURES].to_numpy(float))
        state["model"] = LogisticRegression(C=.15, max_iter=500, class_weight="balanced", random_state=42)
        state["model"].fit(state["scaler"].transform(frame.loc[train, self.A913_FEATURES].to_numpy(float)), (future_net[train] > 0).astype(int))

    def _adaptive_inference(self, bars: pd.DataFrame, strategy_id: str) -> LiveInference:
        spec = self.ADAPTIVE_SPECS[strategy_id]
        signals = structure_signals(bars, **spec["params"]); row = signals.iloc[-1]
        timestamp = pd.Timestamp(row.datetime_utc); day = timestamp.floor("D")
        state = self._gate_states[strategy_id]
        if state["day"] != day: self._refresh_gate(strategy_id, day)
        candidate = str(row.signal); probability = .0
        if candidate in {"BUY", "SELL"} and state["model"] is not None:
            current = FeatureEngine().transform(bars).iloc[[-1]][self.A913_FEATURES].replace([np.inf, -np.inf], np.nan)
            if current.notna().all(axis=None): probability = float(state["model"].predict_proba(state["scaler"].transform(current.to_numpy(float)))[0, 1])
        signal = candidate if probability >= spec["gate"] else "HOLD"
        # PaperAccount's controlled mode derives direction from this score. Map
        # the approved direction explicitly; an unapproved gate must sit in the
        # neutral band and therefore cannot open a position.
        engine_score = .60 if signal == "BUY" else .40 if signal == "SELL" else .50
        return LiveInference(True, f"A000913-{strategy_id.lower()}-paper", None, engine_score, timestamp, signal, signal,
                             f"{row.reason}; gate={probability:.1%}/{spec['gate']:.0%}", signal_id=int(row.timestamp),
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
        for strategy_id in self.ADAPTIVE_SPECS:
            adaptive = self._adaptive_inference(bars, strategy_id)
            adaptive_context = {**context, "regime": f"a000913_{strategy_id.lower()}", "atr_15": float(row.atr_15)}
            self.accounts[self.SPECS[strategy_id]].process(tick, adaptive, adaptive_context)
