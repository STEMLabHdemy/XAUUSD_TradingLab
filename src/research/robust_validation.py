"""Stress-test the two adaptive research challengers on fixed historical folds.

This is deliberately a *finite*, pre-declared study.  It does not search new
parameters: it checks whether the two already-selected challengers retain a
positive profile across independent chronological windows and harsher costs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

import pandas as pd

from src.research.online_adaptation import run


CHALLENGERS: dict[str, dict] = {
    "R214": {
        "label": "A000214 core",
        "source_candidate": "A000214",
        "params": {"reversal_atr": 1.7, "ema_span": 30, "touch_atr": 0.30,
                   "touch_lookback": 3, "breakout_lookback": 1, "swing_buffer_atr": 0.0},
        "threshold": 0.50,
        "rolling_training_days": 90,
    },
    "R913": {
        "label": "A000913 core",
        "source_candidate": "A000913",
        "params": {"reversal_atr": 1.15, "ema_span": 20, "touch_atr": 0.15,
                   "touch_lookback": 3, "breakout_lookback": 2, "swing_buffer_atr": 0.0},
        "threshold": 0.55,
        "rolling_training_days": 120,
    },
}
COST_MULTIPLIERS = (1.0, 2.0, 3.0)
FOLDS = 4
VERSION = 1


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _latest_timestamp(root: Path) -> pd.Timestamp:
    values: list[pd.Timestamp] = []
    for path in (root / "data/processed/XAUUSD_M1_MASTER.parquet", root / "data/live/MT5_XAUUSD_M1.parquet"):
        if path.exists():
            frame = pd.read_parquet(path, columns=["datetime_utc"])
            values.append(pd.to_datetime(frame["datetime_utc"], utc=True).max())
    if not values:
        raise ValueError("Storico XAUUSD non disponibile")
    return max(values).floor("min")


def _summary(rows: list[dict], challenger: dict) -> dict:
    normal = [row for row in rows if row["cost_multiplier"] == 1.0]
    stress = [row for row in rows if row["cost_multiplier"] > 1.0]
    metrics = [row["adaptive"] for row in normal]
    pnls = [float(item["net_pnl"]) for item in metrics]
    pfs = [float(item["profit_factor"]) for item in metrics]
    stress_pnls = [float(row["adaptive"]["net_pnl"]) for row in stress]
    profitable = sum(pnl > 0 for pnl in pnls)
    # This is a research triage label, never an approval for a live account.
    median_stress = statistics.median(stress_pnls) if stress_pnls else None
    grade = "DA APPROFONDIRE" if len(normal) == FOLDS and len(stress) == FOLDS * 2 and profitable >= 3 and statistics.median(pfs) > 1 and median_stress > 0 else "IN CORSO"
    return {
        "id": challenger["id"], "label": challenger["label"], "source_candidate": challenger["source_candidate"],
        "config": {key: challenger[key] for key in ("params", "threshold", "rolling_training_days")},
        "folds_complete": len(normal), "profitable_folds": profitable,
        "median_pnl": statistics.median(pnls) if pnls else None,
        "worst_pnl": min(pnls) if pnls else None,
        "median_pf": statistics.median(pfs) if pfs else None,
        "median_stress_pnl": median_stress,
        "grade": grade,
    }


def run_study(root: Path, days: int = 90, pause_seconds: float = 0.2) -> None:
    root = root.resolve()
    directory = root / "results" / "adaptive_robustness"
    result_path, status_path = directory / "result.json", directory / "status.json"
    latest = _latest_timestamp(root)
    folds = [{"id": f"F{index + 1}", "end": (latest - pd.Timedelta(days=days * index)).isoformat()}
             for index in range(FOLDS)]
    existing: dict[str, dict] = {}
    if result_path.exists():
        try:
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("version") == VERSION:
                existing = {row["key"]: row for row in prior.get("runs", [])}
        except (OSError, json.JSONDecodeError, KeyError):
            existing = {}
    total = len(CHALLENGERS) * len(folds) * len(COST_MULTIPLIERS)
    for challenger_id, raw in CHALLENGERS.items():
        challenger = {"id": challenger_id, **raw}
        for fold in folds:
            for multiplier in COST_MULTIPLIERS:
                key = f"{challenger_id}:{fold['id']}:{multiplier:g}x"
                if key not in existing:
                    began = time.perf_counter()
                    replay = run(root, days=days, end=pd.Timestamp(fold["end"]), params=challenger["params"],
                                 threshold=float(challenger["threshold"]), rolling_training_days=int(challenger["rolling_training_days"]),
                                 cost_multiplier=multiplier, persist=False)
                    existing[key] = {
                        "key": key, "challenger_id": challenger_id, "fold": fold,
                        "cost_multiplier": multiplier, "adaptive": replay["adaptive"]["metrics"],
                        "baseline": replay["baseline"]["metrics"], "seconds": round(time.perf_counter() - began, 2),
                    }
                ordered = list(existing.values())
                by_id = {ident: [row for row in ordered if row["challenger_id"] == ident] for ident in CHALLENGERS}
                ranking = sorted((_summary(by_id[ident], {"id": ident, **CHALLENGERS[ident]}) for ident in CHALLENGERS),
                                 key=lambda item: (item["grade"] == "DA APPROFONDIRE", item["median_pnl"] or float("-inf")), reverse=True)
                payload = {"version": VERSION, "updated_at_utc": datetime.now(timezone.utc).isoformat(), "days_per_fold": days,
                           "latest_data_utc": latest.isoformat(), "folds": folds, "cost_multipliers": COST_MULTIPLIERS,
                           "runs": ordered, "ranking": ranking,
                           "note": "Configurazioni fissate prima del test; i costi sono stress di slippage. Ricerca storica, non approvazione live."}
                _write(result_path, payload)
                _write(status_path, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "completed": len(existing), "total": total, "message": f"stress test {key}"})
                time.sleep(pause_seconds)
    _write(status_path, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                         "completed": len(existing), "total": total, "message": "studio robustezza completato"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run_study(args.project_root, args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
