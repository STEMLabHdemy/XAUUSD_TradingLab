"""Continuously explore causal market-structure configurations.

This is deliberately a research process: it reads local OHLC history and
writes a compact leaderboard.  It neither connects to the broker nor alters
the paper ledger.  Candidates are deterministic and diverse, rather than
randomly brute-forcing a single lucky parameter set.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time

import pandas as pd

from src.backtest.engine import BacktestConfig, Backtester
from src.backtest.lab import load_history
from src.backtest.metrics import performance_metrics
from src.features import FeatureEngine
from src.signals.structure import structure_signals


def candidates():
    """A shuffled full-factorial grid of interpretable structural rules."""
    grid = list(itertools.product(
        (0.9, 1.15, 1.4, 1.7, 2.0),       # swing reversal size, ATR
        (10, 20, 30, 40),                  # M15 trend EMA
        (0.05, 0.15, 0.30),                # acceptable pullback proximity
        (2, 3, 4),                         # pullback lookback candles
        (1, 2, 3),                         # recovery breakout confirmation
        (0.0, 0.30, 0.60),                 # room above/below swing
        (1.0, 1.25, 1.5, 1.75, 2.0),       # initial stop ATR
        (0.75, 1.0, 1.25),                 # break-even activation R
        (1.5, 2.0, 2.5, 3.0, 3.5),         # trailing ATR
        (1, 2, 3),                         # max entries per Rome day
    ))
    # A deterministic permutation distributes every dimension early in the run.
    grid.sort(key=lambda row: hash(tuple(str(x) for x in row)) & 0xffffffff)
    for row in itertools.cycle(grid):
        yield {
            "reversal_atr": row[0], "ema_span": row[1], "touch_atr": row[2],
            "touch_lookback": row[3], "breakout_lookback": row[4],
            "swing_buffer_atr": row[5], "atr_stop_multiple": row[6],
            "atr_break_even_r": row[7], "atr_trailing_multiple": row[8],
            "max_daily_trades": row[9],
        }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _record(index: int, params: dict[str, float | int], metrics: dict[str, float], elapsed: float) -> dict:
    # Filter very sparse or obviously over-traded variants before ranking.
    trades = int(metrics.get("trades", 0))
    score = float(metrics.get("net_pnl", 0.0))
    if trades < 20 or trades > 450:
        score -= 10_000
    return {"id": f"S{index:06d}", "tested_at_utc": datetime.now(timezone.utc).isoformat(),
            "params": params, "metrics": metrics, "ranking_score": score,
            "elapsed_seconds": round(elapsed, 2)}


def run(root: Path, days: int = 90, pause_seconds: float = .1) -> None:
    root = root.resolve()
    output = root / "results" / "continuous_structure_search" / "leaderboard.json"
    status = root / "results" / "continuous_structure_search" / "status.json"
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=days)
    # Loaded once.  The selected 90-day sample remains frozen during this run,
    # making rankings directly comparable; restarting tomorrow creates a fresh
    # sample and keeps the existing leaderboard as historical evidence.
    bars = load_history(root, start, end)
    base = FeatureEngine().transform(bars)
    selected = pd.to_datetime(base.datetime_utc, utc=True).between(start, end)
    base = base.loc[selected].reset_index(drop=True)
    leaderboard: list[dict] = []
    tested = 0
    for params in candidates():
        began = time.perf_counter()
        try:
            signals = structure_signals(bars, **{key: params[key] for key in (
                "reversal_atr", "ema_span", "touch_atr", "touch_lookback", "breakout_lookback", "swing_buffer_atr")})
            data = base.copy()
            data["signal"] = signals.loc[selected, "signal"].to_numpy()
            config = BacktestConfig(
                stop_loss_price=None, take_profit_price=None, max_holding_minutes=None,
                max_daily_trades=int(params["max_daily_trades"]), cooldown_minutes=10,
                atr_stop_multiple=float(params["atr_stop_multiple"]),
                atr_break_even_r=float(params["atr_break_even_r"]),
                atr_trailing_multiple=float(params["atr_trailing_multiple"]),
            )
            outcome = Backtester(config).run(data)
            metrics = performance_metrics(outcome.trades, outcome.equity_curve, config.starting_capital)
            tested += 1
            leaderboard.append(_record(tested, params, metrics, time.perf_counter() - began))
            leaderboard.sort(key=lambda item: item["ranking_score"], reverse=True)
            leaderboard = leaderboard[:100]
            _write(output, {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(), "period": {"start": start.isoformat(), "end": end.isoformat(), "days": days},
                "tested": tested, "leaderboard": leaderboard,
                "note": "Classifica di ricerca in corso. Non Ã¨ validazione fuori campione e non genera ordini.",
            })
            _write(status, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "tested": tested, "last_id": f"S{tested:06d}", "last_seconds": round(time.perf_counter()-began, 2),
                            "message": "ricerca attiva"})
        except Exception as exc:
            _write(status, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "tested": tested, "message": f"errore candidato: {exc}"})
        time.sleep(pause_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous causal structure backtest search")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--pause-seconds", type=float, default=.1)
    args = parser.parse_args()
    run(args.project_root, args.days, args.pause_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
