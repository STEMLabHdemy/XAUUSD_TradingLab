from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    starting_capital: float = 100_000.0
    position_size_units: float = 1.0
    leverage: float = 20.0
    commission_per_unit_per_side: float = 0.0
    slippage_price_per_side: float = 0.05
    stop_loss_price: float | None = 5.0
    take_profit_price: float | None = 10.0
    max_holding_minutes: int | None = 30


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


@dataclass
class _Position:
    trade_id: int
    side: str
    entry_time: pd.Timestamp
    raw_entry_price: float
    entry_price: float
    confidence: float
    spread: float
    stop_loss: float | None
    take_profit: float | None
    entry_index: int
    session: str | None
    trend_regime: str | None
    volatility_regime: str | None


class Backtester:
    """Event-driven one-position engine; signals at t execute no earlier than t+1 open."""

    REQUIRED = {
        "timestamp", "datetime_utc", "open_bid", "high_bid", "low_bid", "close_bid",
        "open_ask", "high_ask", "low_ask", "close_ask", "signal",
    }

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        if self.config.starting_capital <= 0 or self.config.position_size_units <= 0:
            raise ValueError("Capital and position size must be positive")

    def run(self, data: pd.DataFrame) -> BacktestResult:
        missing = self.REQUIRED.difference(data.columns)
        if missing:
            raise ValueError(f"Backtester missing columns: {sorted(missing)}")
        bars = data.sort_values("timestamp").reset_index(drop=True)
        position: _Position | None = None
        pending_signal: str | None = None
        pending_reason = "signal"
        pending_confidence = float("nan")
        balance = self.config.starting_capital
        trade_rows: list[dict[str, object]] = []
        equity_rows: list[dict[str, object]] = []
        next_trade_id = 1
        quantity = self.config.position_size_units
        slip = self.config.slippage_price_per_side
        commission_round_trip = 2 * self.config.commission_per_unit_per_side * quantity

        def close_position(row: pd.Series, raw_exit: float, exit_reason: str, exit_time: pd.Timestamp) -> None:
            nonlocal position, balance
            assert position is not None
            direction = 1 if position.side == "LONG" else -1
            exit_price = raw_exit - slip if position.side == "LONG" else raw_exit + slip
            gross = direction * (raw_exit - position.raw_entry_price) * quantity
            slippage_cost = 2 * slip * quantity
            costs = slippage_cost + commission_round_trip
            net = gross - costs
            balance += net
            trade_rows.append({
                "trade_id": position.trade_id, "side": position.side,
                "entry_time": position.entry_time, "entry_price": position.entry_price,
                "raw_entry_price": position.raw_entry_price, "exit_time": exit_time,
                "exit_price": exit_price, "raw_exit_price": raw_exit,
                "confidence": position.confidence, "spread": position.spread,
                "stop_loss": position.stop_loss, "take_profit": position.take_profit,
                "exit_reason": exit_reason, "gross_pnl": gross,
                "costs": costs, "net_pnl": net,
                "holding_minutes": max(0, int((exit_time - position.entry_time).total_seconds() // 60)),
                "session": position.session, "trend_regime": position.trend_regime,
                "volatility_regime": position.volatility_regime,
            })
            position = None

        for index, row in bars.iterrows():
            timestamp = pd.Timestamp(row.datetime_utc)
            executable = pd.notna(row.open_bid) and pd.notna(row.open_ask)
            if pending_signal == "EXIT" and position is not None and executable:
                raw_exit = float(row.open_bid if position.side == "LONG" else row.open_ask)
                close_position(row, raw_exit, pending_reason, timestamp)
            elif pending_signal in {"BUY", "SELL"} and position is None and executable:
                side = "LONG" if pending_signal == "BUY" else "SHORT"
                raw_entry = float(row.open_ask if side == "LONG" else row.open_bid)
                entry_price = raw_entry + slip if side == "LONG" else raw_entry - slip
                stop = entry_price - self.config.stop_loss_price if side == "LONG" and self.config.stop_loss_price else None
                stop = entry_price + self.config.stop_loss_price if side == "SHORT" and self.config.stop_loss_price else stop
                take = entry_price + self.config.take_profit_price if side == "LONG" and self.config.take_profit_price else None
                take = entry_price - self.config.take_profit_price if side == "SHORT" and self.config.take_profit_price else take
                position = _Position(
                    next_trade_id, side, timestamp, raw_entry, entry_price,
                    pending_confidence, float(row.open_ask - row.open_bid),
                    stop, take, index, row.get("session"), row.get("trend_regime"), row.get("volatility_regime"),
                )
                next_trade_id += 1
            pending_signal = None

            if position is not None:
                if position.side == "LONG" and pd.notna(row.low_bid) and pd.notna(row.high_bid):
                    stop_hit = position.stop_loss is not None and float(row.low_bid) <= position.stop_loss
                    take_hit = position.take_profit is not None and float(row.high_bid) >= position.take_profit
                    if stop_hit:
                        raw = min(float(row.open_bid), position.stop_loss) if float(row.open_bid) < position.stop_loss else position.stop_loss
                        close_position(row, raw, "stop_loss", timestamp)
                    elif take_hit:
                        raw = max(float(row.open_bid), position.take_profit) if float(row.open_bid) > position.take_profit else position.take_profit
                        close_position(row, raw, "take_profit", timestamp)
                elif position.side == "SHORT" and pd.notna(row.high_ask) and pd.notna(row.low_ask):
                    stop_hit = position.stop_loss is not None and float(row.high_ask) >= position.stop_loss
                    take_hit = position.take_profit is not None and float(row.low_ask) <= position.take_profit
                    if stop_hit:
                        raw = max(float(row.open_ask), position.stop_loss) if float(row.open_ask) > position.stop_loss else position.stop_loss
                        close_position(row, raw, "stop_loss", timestamp)
                    elif take_hit:
                        raw = min(float(row.open_ask), position.take_profit) if float(row.open_ask) < position.take_profit else position.take_profit
                        close_position(row, raw, "take_profit", timestamp)

            signal = str(row.signal)
            if position is not None and self.config.max_holding_minutes is not None:
                held = int((timestamp - position.entry_time).total_seconds() // 60)
                if held >= self.config.max_holding_minutes:
                    pending_signal, pending_reason = "EXIT", "time_exit"
                elif signal == "EXIT":
                    pending_signal, pending_reason = "EXIT", "signal_exit"
            elif position is None and signal in {"BUY", "SELL"}:
                pending_signal, pending_reason = signal, "signal_entry"
                pending_confidence = float(row.get("score", np.nan))

            unrealized = 0.0
            if position is not None:
                if position.side == "LONG" and pd.notna(row.close_bid):
                    unrealized = (float(row.close_bid) - position.raw_entry_price) * quantity - 2 * slip * quantity - commission_round_trip
                elif position.side == "SHORT" and pd.notna(row.close_ask):
                    unrealized = (position.raw_entry_price - float(row.close_ask)) * quantity - 2 * slip * quantity - commission_round_trip
            equity_rows.append({
                "timestamp": row.timestamp, "datetime_utc": timestamp,
                "balance": balance, "equity": balance + unrealized,
                "unrealized_pnl": unrealized, "in_position": int(position is not None),
            })

        if position is not None and len(bars):
            row = bars.iloc[-1]
            raw_exit = float(row.close_bid if position.side == "LONG" else row.close_ask)
            close_position(row, raw_exit, "end_of_test", pd.Timestamp(row.datetime_utc))
            equity_rows[-1]["balance"] = balance
            equity_rows[-1]["equity"] = balance
            equity_rows[-1]["unrealized_pnl"] = 0.0
            equity_rows[-1]["in_position"] = 0

        return BacktestResult(pd.DataFrame(trade_rows), pd.DataFrame(equity_rows))
