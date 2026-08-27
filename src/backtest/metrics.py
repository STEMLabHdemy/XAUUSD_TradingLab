from __future__ import annotations

import numpy as np
import pandas as pd


def _longest_losing_streak(pnl: pd.Series) -> int:
    longest = current = 0
    for value in pnl:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def performance_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    starting_capital: float,
    annualization_periods: int = 252 * 24 * 60,
) -> dict[str, float | int]:
    if equity_curve.empty:
        return {"net_pnl": 0.0, "total_return": 0.0, "trades": 0}
    equity = equity_curve.equity.astype(float)
    returns = equity.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    net_pnl = float(equity.iloc[-1] - starting_capital)
    total_return = net_pnl / starting_capital
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(abs(drawdown.min()))
    years = max((pd.to_datetime(equity_curve.datetime_utc).iloc[-1] - pd.to_datetime(equity_curve.datetime_utc).iloc[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    annual_return = (equity.iloc[-1] / starting_capital) ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    sharpe = float(np.sqrt(annualization_periods) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
    downside = returns[returns < 0].std()
    sortino = float(np.sqrt(annualization_periods) * returns.mean() / downside) if pd.notna(downside) and downside > 0 else 0.0
    if trades.empty:
        trade_pnl = pd.Series(dtype=float)
    else:
        trade_pnl = trades.net_pnl.astype(float)
    winners = trade_pnl[trade_pnl > 0]
    losers = trade_pnl[trade_pnl < 0]
    gross_profit, gross_loss = winners.sum(), abs(losers.sum())
    days = max((pd.to_datetime(equity_curve.datetime_utc).max() - pd.to_datetime(equity_curve.datetime_utc).min()).total_seconds() / 86400, 1)
    average_winner = float(winners.mean()) if len(winners) else 0.0
    average_loser = float(losers.mean()) if len(losers) else 0.0
    return {
        "net_pnl": net_pnl, "total_return": total_return,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_drawdown,
        "calmar": float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0,
        "win_rate": float((trade_pnl > 0).mean()) if len(trade_pnl) else 0.0,
        "average_winner": average_winner, "average_loser": average_loser,
        "payoff_ratio": float(average_winner / abs(average_loser)) if average_loser < 0 else 0.0,
        "expectancy": float(trade_pnl.mean()) if len(trade_pnl) else 0.0,
        "trades": int(len(trade_pnl)), "trades_per_day": float(len(trade_pnl) / days),
        "exposure": float(equity_curve.in_position.mean()),
        "average_holding_minutes": float(trades.holding_minutes.mean()) if len(trades) else 0.0,
        "longest_losing_streak": _longest_losing_streak(trade_pnl),
    }


def performance_breakdowns(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {}
    enriched = trades.copy()
    entry = pd.to_datetime(enriched.entry_time, utc=True)
    enriched["year"] = entry.dt.year
    enriched["month"] = entry.dt.strftime("%Y-%m")
    enriched["hour"] = entry.dt.hour
    outputs: dict[str, pd.DataFrame] = {}
    for column in ("year", "month", "hour", "session", "volatility_regime", "trend_regime", "side"):
        if column in enriched:
            outputs[column] = enriched.groupby(column, dropna=False).agg(
                net_pnl=("net_pnl", "sum"), trades=("trade_id", "count"),
                win_rate=("net_pnl", lambda values: float((values > 0).mean())),
                expectancy=("net_pnl", "mean"),
            ).reset_index()
    return outputs
