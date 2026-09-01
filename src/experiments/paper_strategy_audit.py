"""Historical audit of the exact paper policies 0 and A-L on one shared signal stream."""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import yaml

from src.backtest import performance_metrics
from src.live.inference import LiveInference
from src.live.mt5_client import MarketTick
from src.paper.engine import PaperAccount, PaperConfig, PaperRuntime


class _MemoryAccount(PaperAccount):
    """PaperAccount policy with persistence disabled for a fast, isolated audit."""
    def _save(self) -> None:
        self._sync_position_alias(self.state)


def _inference(row: Any, prior_highs: list[float]) -> LiveInference:
    down, neutral, up = float(row.p_down), float(row.p_neutral), float(row.p_up)
    candidate = {0: "SELL", 1: "HOLD", 2: "BUY"}[max(range(3), key=(down, neutral, up).__getitem__)]
    return LiveInference(
        True, "audit cost-aware", int(row.horizon), {"SELL": .4, "HOLD": .5, "BUY": .6}[candidate],
        pd.Timestamp(row.datetime_utc), candidate, "NO_TRADE", "historical shared signal",
        probability_down=down, probability_neutral=neutral, signal_id=int(row.timestamp),
        short_reversal_price_confirmed=len(prior_highs) >= 3 and float(row.high_bid) > max(prior_highs[-3:]),
    )


def audit_paper_strategies(frame: pd.DataFrame, project_root: Path | str, horizon: int, model: str) -> pd.DataFrame:
    """Run fixed live policies. A signal at a M1 close executes from the next M1 open."""
    root = Path(project_root)
    values = yaml.safe_load((root / "configs/paper.yaml").read_text(encoding="utf-8")) or {}
    base = PaperConfig(**values)
    rows = frame.sort_values("timestamp").reset_index(drop=True)
    accounts: dict[str, PaperAccount] = {}
    with tempfile.TemporaryDirectory() as directory:
        for spec in PaperRuntime.STRATEGIES:
            account = _MemoryAccount(
                f"audit_{model}_{spec.strategy_id}", spec.label,
                PaperRuntime._strategy_config(base, spec), Path(directory),
            )
            account.set_run_id("historical_audit")
            account.start()
            accounts[spec.strategy_id] = account

        prior: LiveInference | None = None
        highs: list[float] = []
        for row in rows.itertuples(index=False):
            timestamp = pd.Timestamp(row.datetime_utc)
            spread = float(row.open_ask - row.open_bid)
            # Prior close's signal is actionable at this open: no look-ahead.
            if prior is not None:
                opening = MarketTick(timestamp, timestamp, float(row.open_bid), float(row.open_ask), spread, "XAUUSD")
                for account in accounts.values():
                    account.process(opening, prior)
                # Conservative intrabar order, matching the existing backtester:
                # a stop gets priority if the same M1 touches both stop and target.
                for account in accounts.values():
                    positions = list(account.snapshot().get("positions", []))
                    for position in positions:
                        if position["side"] == "LONG" and position.get("stop_loss") is not None and float(row.low_bid) <= float(position["stop_loss"]):
                            low = MarketTick(timestamp, timestamp, float(row.low_bid), float(row.low_bid) + spread, spread, "XAUUSD")
                            account.process(low, prior)
                        elif position["side"] == "LONG" and position.get("take_profit") is not None and float(row.high_bid) >= float(position["take_profit"]):
                            high = MarketTick(timestamp, timestamp, float(row.high_bid), float(row.high_bid) + spread, spread, "XAUUSD")
                            account.process(high, prior)
                        elif position["side"] == "SHORT" and position.get("stop_loss") is not None and float(row.high_ask) >= float(position["stop_loss"]):
                            high = MarketTick(timestamp, timestamp, float(row.high_ask) - spread, float(row.high_ask), spread, "XAUUSD")
                            account.process(high, prior)
                        elif position["side"] == "SHORT" and position.get("take_profit") is not None and float(row.low_ask) <= float(position["take_profit"]):
                            low = MarketTick(timestamp, timestamp, float(row.low_ask) - spread, float(row.low_ask), spread, "XAUUSD")
                            account.process(low, prior)
            prior = _inference(row, highs)
            highs.append(float(row.high_bid))

        output: list[dict[str, object]] = []
        final_tick = MarketTick(timestamp, timestamp, float(row.close_bid), float(row.close_ask), float(row.close_ask - row.close_bid), "XAUUSD")
        for strategy_id, account in accounts.items():
            account.close_all_for_session(final_tick)
            state = account.snapshot()
            trades = pd.DataFrame(state["trades"])
            if not trades.empty:
                trades["holding_minutes"] = (
                    pd.to_datetime(trades["exit_time"], utc=True) - pd.to_datetime(trades["entry_time"], utc=True)
                ).dt.total_seconds().div(60).clip(lower=0)
            curve = pd.DataFrame(state["portfolio_history"])
            if not curve.empty:
                curve = curve.rename(columns={"timestamp": "datetime_utc", "open_positions": "in_position"})
            metrics = performance_metrics(trades, curve, base.starting_capital)
            output.append({
                "model": model, "horizon": horizon, "strategy_id": strategy_id, "strategy": account.model,
                "audit_net_pnl": metrics["net_pnl"], "audit_profit_factor": metrics["profit_factor"],
                "audit_max_drawdown": metrics["max_drawdown"], "audit_trades": metrics["trades"],
                "audit_win_rate": metrics["win_rate"], "audit_expectancy": metrics["expectancy"],
            })
    return pd.DataFrame(output)
