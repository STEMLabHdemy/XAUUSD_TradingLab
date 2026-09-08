"""Independent, rule-based paper portfolios: no ML inference is used here."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from src.features import FeatureEngine
from src.live.inference import LiveInference
from src.live.mt5_client import MarketTick
from .engine import PaperAccount, PaperConfig


class IndicatorPaperRuntime:
    SPECS = {
        "T": "T - EMA20/50 + MACD trend",
        "U": "U - RSI14 + Bollinger mean reversion",
        "V": "V - Donchian 20 breakout",
    }

    def __init__(self, root: Path, base: PaperConfig):
        self.root = root
        self.directory = root / "data/live/paper/indicator_v1"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = self.directory / "run.json"
        if self.metadata.exists():
            self.run_id = json.loads(self.metadata.read_text(encoding="utf-8"))["run_id"]
        else:
            self.run_id = f"ind_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            self.metadata.write_text(json.dumps({"run_id": self.run_id, "kind": "rule_based_indicators"}, indent=2), encoding="utf-8")
        self.accounts = {}
        for strategy_id, label in self.SPECS.items():
            config = replace(base, strategy_id=strategy_id, entry_mode="controlled", max_open_positions_override=1)
            account = PaperAccount(f"indicator_v1_{strategy_id.lower()}", label, config, self.directory)
            if account.state.get("run_id") != self.run_id:
                account.set_run_id(self.run_id, datetime.now(timezone.utc).isoformat())
            if not account.state.get("running"):
                account.start()
            self.accounts[label] = account

    @staticmethod
    def _candidate(strategy_id: str, frame: pd.DataFrame) -> tuple[str, str]:
        row, prior = frame.iloc[-1], frame.iloc[-2]
        if strategy_id == "T":
            if row.ema_distance_20 < row.ema_distance_50 and row.macd_histogram > 0: return "BUY", "EMA20>EMA50 and MACD positive"
            if row.ema_distance_20 > row.ema_distance_50 and row.macd_histogram < 0: return "SELL", "EMA20<EMA50 and MACD negative"
        elif strategy_id == "U":
            if row.rsi_14 < 30 and row.bollinger_position < .15: return "BUY", "RSI oversold below Bollinger band"
            if row.rsi_14 > 70 and row.bollinger_position > .85: return "SELL", "RSI overbought above Bollinger band"
        elif strategy_id == "V":
            highs, lows = frame.mid_high.rolling(20).max().shift(1), frame.mid_low.rolling(20).min().shift(1)
            if row.mid_close > highs.iloc[-1]: return "BUY", "20-bar Donchian upside breakout"
            if row.mid_close < lows.iloc[-1]: return "SELL", "20-bar Donchian downside breakout"
        return "HOLD", "indicator conditions not met"

    def process(self, tick: MarketTick, completed_m1: pd.DataFrame) -> None:
        if len(completed_m1) < 202: return
        featured = FeatureEngine().transform(completed_m1)
        now = pd.Timestamp(featured.datetime_utc.iloc[-1])
        context = {"prior_return_15m_pct": float(featured.return_15m.iloc[-1]), "range_15m_pct": None}
        for _, account in self.accounts.items():
            candidate, reason = self._candidate(account.config.strategy_id, featured)
            score = {"BUY": .60, "SELL": .40, "HOLD": .50}[candidate]
            inference = LiveInference(True, "technical-indicators", None, score, now, candidate, candidate, reason, signal_id=int(featured.timestamp.iloc[-1]))
            account.process(tick, inference, context)
