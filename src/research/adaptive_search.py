"""Continuous search of causal online-adaptation configurations.

Each candidate is a full chronological replay: daily models learn only from
earlier labelled events, then gate later structural signals.  It is therefore
fundamentally different from fitting parameters on all 90 days at once.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time

from src.research.online_adaptation import run


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def candidates():
    # Narrow, interpretable region suggested by the prior research; this is
    # exploration of adaptive policy, not an infinite random parameter dump.
    grid = list(itertools.product((1.15, 1.4, 1.7), (20, 30, 40), (.15, .30),
                                  (2, 3), (1, 2), (0., .3, .6),
                                  (.50, .55, .60, .65), (60, 90, 120)))
    grid.sort(key=lambda row: hash(tuple(map(str, row))) & 0xffffffff)
    for values in itertools.cycle(grid):
        yield {"params": {"reversal_atr": values[0], "ema_span": values[1], "touch_atr": values[2],
                              "touch_lookback": values[3], "breakout_lookback": values[4], "swing_buffer_atr": values[5]},
               "threshold": values[6], "rolling_training_days": values[7]}


def run_forever(root: Path, days: int = 90, pause_seconds: float = .2) -> None:
    root = root.resolve(); directory = root / "results" / "adaptive_search"
    board_path, status_path = directory / "leaderboard.json", directory / "status.json"
    leaderboard: list[dict] = []; tested = 0
    for config in candidates():
        began = time.perf_counter()
        try:
            result = run(root, days=days, threshold=config["threshold"], rolling_training_days=config["rolling_training_days"], params=config["params"], persist=False)
            adaptive, baseline = result["adaptive"]["metrics"], result["baseline"]["metrics"]
            tested += 1
            score = float(adaptive["net_pnl"])
            if adaptive["trades"] < 30: score -= 10_000
            row = {"id": f"A{tested:06d}", "tested_at_utc": datetime.now(timezone.utc).isoformat(),
                   "config": config, "adaptive": adaptive, "baseline": baseline,
                   "ranking_score": score, "seconds": round(time.perf_counter()-began, 2)}
            leaderboard.append(row); leaderboard.sort(key=lambda item: item["ranking_score"], reverse=True); leaderboard = leaderboard[:100]
            _write(board_path, {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "tested": tested,
                "period": result["period"], "leaderboard": leaderboard,
                "note": "Ogni riga è un walk-forward causale: modello aggiornato ogni giorno solo con esiti precedenti. Non è approvazione live."})
            _write(status_path, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "tested": tested, "last_id": row["id"], "last_seconds": row["seconds"], "message": "ricerca adattiva attiva"})
        except Exception as exc:
            _write(status_path, {"pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "tested": tested, "message": f"errore candidato: {exc}"})
        time.sleep(pause_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path.cwd()); parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args(); run_forever(args.project_root, args.days); return 0


if __name__ == "__main__": raise SystemExit(main())
