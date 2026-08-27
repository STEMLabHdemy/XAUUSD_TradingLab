from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd


class LiveBarStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(".metadata.json")

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame()

    def append_completed(self, bars: pd.DataFrame, metadata: dict[str, object] | None = None) -> int:
        incoming = bars[bars.is_complete.astype(bool)].copy() if "is_complete" in bars else bars.copy()
        if incoming.empty:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.load()
        previous_timestamps = set(existing.timestamp.astype("int64")) if not existing.empty else set()
        combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
        combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        added = len(set(incoming.timestamp.astype("int64")) - previous_timestamps)
        if added or not self.path.exists():
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp.parquet")
            combined.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, self.path)
        if metadata is not None:
            temporary_metadata = self.metadata_path.with_name(f".{self.metadata_path.name}.{uuid4().hex}.tmp")
            temporary_metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            os.replace(temporary_metadata, self.metadata_path)
        return added
