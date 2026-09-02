"""Lightweight PaperRuntime launcher: MT5 feed + paper engine, no Streamlit UI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import yaml

from src.live import LiveMarketService
from src.paper.engine import PaperConfig, PaperRuntime


def _write_status(path: Path, runtime: PaperRuntime, *, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": __import__("os").getpid(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": runtime.run_id,
        "mode": "headless",
        "message": message,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(project_root: Path, interval_seconds: float) -> None:
    values = yaml.safe_load((project_root / "configs/paper.yaml").read_text(encoding="utf-8")) or {}
    service = LiveMarketService(project_root)
    runtime = PaperRuntime(project_root, PaperConfig(**values))
    status_path = project_root / "data/live/paper/comparison_v1/headless_status.json"
    print(f"Paper headless attivo. Run: {runtime.run_id}. Premi Ctrl+C per fermarlo.", flush=True)
    try:
        while True:
            try:
                snapshot = service.poll()
                completed = snapshot.m1_bars[snapshot.m1_bars.is_complete.astype(bool)].reset_index(drop=True)
                runtime.process(snapshot.tick, completed)
                _write_status(status_path, runtime, message="MT5 connesso; paper engine attivo")
            except Exception as exc:  # The loop survives MT5 temporary outages.
                _write_status(status_path, runtime, message=f"in attesa MT5: {exc}")
                print(f"MT5/paper temporaneamente non disponibile: {exc}", flush=True)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("Paper headless arrestato dall'utente.", flush=True)
    finally:
        _write_status(status_path, runtime, message="paper engine fermo")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MT5 paper engine without Streamlit")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    run(args.project_root.resolve(), args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
