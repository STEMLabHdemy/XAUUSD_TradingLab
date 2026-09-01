"""Create immutable, source-audited training snapshots from historical and MT5 M1 data."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


MASTER_COLUMNS = [
    "timestamp", "datetime_utc", "open_bid", "high_bid", "low_bid", "close_bid",
    "open_ask", "high_ask", "low_ask", "close_ask", "spread_open", "spread_close",
    "mid_open", "mid_high", "mid_low", "mid_close",
]


def create_training_snapshot(project_root: Path | str, destination: Path | str | None = None) -> Path:
    root = Path(project_root).resolve()
    master_path = root / "data/processed/XAUUSD_M1_MASTER.parquet"
    live_path = root / "data/live/MT5_XAUUSD_M1.parquet"
    if not master_path.exists():
        raise FileNotFoundError("Master storico non trovato")
    historical = pd.read_parquet(master_path)[MASTER_COLUMNS].copy()
    historical["source"] = "dukascopy"
    frames = [historical]
    if live_path.exists():
        live = pd.read_parquet(live_path)
        if "is_complete" in live:
            live = live[live.is_complete.astype(bool)].copy()
        if not live.empty:
            live["spread_open"] = live.get("spread_open", live["spread_close"])
            live = live[[column for column in MASTER_COLUMNS if column in live]].copy()
            live["source"] = "mt5"
            frames.append(live)
    # Prefer historical BID+ASK where it has caught up; use MT5 only as the
    # recent bridge.  This keeps old training windows reproducible.
    combined = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "source"])
    combined = combined.drop_duplicates("timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(destination) if destination else root / "data/processed/training_snapshots" / f"XAUUSD_M1_{stamp}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    combined.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "rows": len(combined),
        "start_utc": str(combined.datetime_utc.iloc[0]), "end_utc": str(combined.datetime_utc.iloc[-1]),
        "historical_rows": int((combined.source == "dukascopy").sum()), "mt5_bridge_rows": int((combined.source == "mt5").sum()),
        "master": str(master_path), "live_bridge": str(live_path),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output
