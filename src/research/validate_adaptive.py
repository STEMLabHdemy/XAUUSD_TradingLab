"""Continuously validate the best adaptive walk-forward candidates out of sample."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pandas as pd

from src.research.online_adaptation import run


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_forever(root: Path, days: int = 90, interval_seconds: int = 300) -> None:
    root = root.resolve(); directory = root / "results" / "adaptive_search"
    result_path, status_path = directory / "validation.json", directory / "validation_status.json"
    done: dict[str, dict] = {}
    while True:
        try:
            board = json.loads((directory / "leaderboard.json").read_text(encoding="utf-8"))
            # The candidate search uses [train_start, train_end]. Validation
            # uses the immediately preceding non-overlapping block, and each
            # replay again has its own earlier bootstrap data.
            validation_end = pd.Timestamp(board["period"]["start"])
            for candidate in board.get("leaderboard", [])[:25]:
                ident = candidate["id"]
                if ident in done:
                    continue
                began = time.perf_counter(); config = candidate["config"]
                replay = run(root, days=days, end=validation_end,
                             threshold=float(config["threshold"]), rolling_training_days=int(config["rolling_training_days"]),
                             params=config["params"], persist=False)
                adaptive, baseline = replay["adaptive"]["metrics"], replay["baseline"]["metrics"]
                done[ident] = {"id": ident, "config": config, "in_sample": candidate["adaptive"],
                               "out_of_sample": adaptive, "out_of_sample_base": baseline,
                               "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                               "seconds": round(time.perf_counter()-began, 2)}
                ranked = sorted(done.values(), key=lambda item: (
                    min(float(item["in_sample"]["net_pnl"]), float(item["out_of_sample"]["net_pnl"])),
                    float(item["out_of_sample"]["profit_factor"])), reverse=True)
                _write(result_path, {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "validated": len(done),
                    "search_period": board["period"], "validation_period": replay["period"], "results": ranked,
                    "note": "Ogni candidato è ri-eseguito causalmente su un blocco storico non usato nella ricerca. Solo OOS positivo non è una prova di edge live."})
                _write(status_path, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "validated": len(done), "message": f"validato {ident}"})
            _write(status_path, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "validated": len(done), "message": "in attesa di nuovi candidati top"})
        except Exception as exc:
            _write(status_path, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "validated": len(done), "message": f"errore validazione: {exc}"})
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path.cwd()); parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args(); run_forever(args.project_root, args.days); return 0


if __name__ == "__main__": raise SystemExit(main())
