"""One independent paper portfolio: causal market-structure pullbacks."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

from src.live.inference import LiveInference
from src.live.mt5_client import MarketTick
from src.signals.structure import structure_signals
from .engine import PaperAccount, PaperConfig


class IndicatorPaperRuntime:
    SPECS = {"I42": "I42 - M15 structure + M1 pullback"}

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
