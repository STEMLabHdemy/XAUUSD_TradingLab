"""Continuously validate discovered candidates on an untouched historical block."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def evaluate(root: Path, start: pd.Timestamp, end: pd.Timestamp, params: dict) -> dict:
    bars = load_history(root, start, end)
    features = FeatureEngine().transform(bars)
    selected = pd.to_datetime(features.datetime_utc, utc=True).between(start, end)
    data = features.loc[selected].copy().reset_index(drop=True)
    signals = structure_signals(bars, **{key: params[key] for key in (
        "reversal_atr", "ema_span", "touch_atr", "touch_lookback", "breakout_lookback", "swing_buffer_atr")})
    data["signal"] = signals.loc[selected, "signal"].to_numpy()
    cfg = BacktestConfig(stop_loss_price=None, take_profit_price=None, max_holding_minutes=None,
        max_daily_trades=int(params["max_daily_trades"]), cooldown_minutes=10,
        atr_stop_multiple=float(params["atr_stop_multiple"]), atr_break_even_r=float(params["atr_break_even_r"]),
        atr_trailing_multiple=float(params["atr_trailing_multiple"]))
    outcome = Backtester(cfg).run(data)
    return performance_metrics(outcome.trades, outcome.equity_curve, cfg.starting_capital)


def run(root: Path, days: int = 90, interval_seconds: int = 300) -> None:
    root = root.resolve(); base = root / "results" / "continuous_structure_search"
    output, status = base / "validation.json", base / "validation_status.json"
    done: dict[str, dict] = {}
    while True:
        try:
            source = json.loads((base / "leaderboard.json").read_text(encoding="utf-8"))
            train_start = pd.Timestamp(source["period"]["start"])
            validation_end, validation_start = train_start, train_start - pd.Timedelta(days=days)
            for candidate in source.get("leaderboard", [])[:25]:
                ident = candidate["id"]
                if ident in done:
                    continue
                began = time.perf_counter()
                metrics = evaluate(root, validation_start, validation_end, candidate["params"])
                done[ident] = {"id": ident, "params": candidate["params"],
                    "in_sample": candidate["metrics"], "out_of_sample": metrics,
                    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "seconds": round(time.perf_counter()-began, 2)}
                ranking = sorted(done.values(), key=lambda row: (
                    min(float(row["in_sample"]["net_pnl"]), float(row["out_of_sample"]["net_pnl"])),
                    float(row["out_of_sample"]["profit_factor"])), reverse=True)
                write(output, {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "train_period": source["period"], "validation_period": {"start": validation_start.isoformat(), "end": validation_end.isoformat(), "days": days},
                    "validated": len(done), "results": ranking,
                    "note": "Periodo precedente non usato dalla ricerca. Un risultato positivo qui riduce l'overfitting, ma non prova un edge live."})
                write(status, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "validated": len(done), "message": f"validato {ident}"})
            write(status, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "validated": len(done), "message": "in attesa di nuovi candidati top"})
        except Exception as exc:
            write(status, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "validated": len(done), "message": f"errore validazione: {exc}"})
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args(); run(args.project_root, args.days, args.interval_seconds); return 0

if __name__ == "__main__":
    raise SystemExit(main())
