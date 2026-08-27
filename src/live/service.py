from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import pandas as pd
import yaml

from .inference import LiveInference, LiveInferenceEngine
from .mt5_client import ConnectionStatus, MT5Client, MarketTick
from .storage import LiveBarStore
from .timeframes import aggregate_m1, compare_native_bars


@dataclass(frozen=True)
class LiveSnapshot:
    status: ConnectionStatus
    tick: MarketTick
    m1_bars: pd.DataFrame
    inference: LiveInference
    stored_new_bars: int
    m5_validation: dict[str, object]


class LiveMarketService:
    def __init__(self, project_root: Path | str):
        self.root = Path(project_root)
        config = yaml.safe_load((self.root / "configs/live.yaml").read_text(encoding="utf-8")) or {}
        if bool(config.get("enable_live_trading", False)):
            raise RuntimeError("Live broker trading must remain disabled")
        terminal_path = Path(config.get("terminal_path", r"C:\Program Files\MetaTrader 5\terminal64.exe"))
        candidates = tuple(config.get("symbol_candidates", ["XAUUSD", "GOLD"]))
        fallback_offset = config.get("fallback_server_utc_offset_seconds")
        self.client = MT5Client(
            terminal_path, candidates,
            int(fallback_offset) if fallback_offset is not None else None,
        )
        self.store = LiveBarStore(self.root / config.get("storage_path", "data/live/MT5_XAUUSD_M1.parquet"))
        self.inference_engine = LiveInferenceEngine(
            self.root / config.get("model_path", "models/lightgbm_up_5m_sigmoid_provisional.joblib"),
            self.root / config.get("model_manifest_path", "models/baseline_manifest_provisional.json"),
            float(config.get("buy_threshold", .68)), float(config.get("sell_threshold", .32)),
        )
        self.history_bars = max(500, int(config.get("history_m1_bars", 2600)))
        self.display_timezone = str(config.get("display_timezone", "Europe/Rome"))
        self._bars = pd.DataFrame()
        self._inference: LiveInference | None = None
        self._last_inference_timestamp: int | None = None
        self._m5_validation: dict[str, object] = {"matched_bars": 0, "max_absolute_error": float("nan"), "valid": False}
        self._last_validation_bucket: int | None = None
        self._lock = RLock()

    def _refresh_bars(self) -> pd.DataFrame:
        bootstrapping = len(self._bars) < 500
        requested = self.history_bars if bootstrapping else 10
        if bootstrapping:
            fresh = pd.DataFrame()
            for batch_size in dict.fromkeys((10, 100, 500, requested)):
                candidate = self.client.bars("M1", batch_size)
                if len(candidate) > len(fresh):
                    fresh = candidate
        else:
            fresh = self.client.bars("M1", requested)
        if not self._bars.empty:
            latest_known = int(self._bars.timestamp.max())
            latest_fresh = int(fresh.timestamp.max())
            gap_minutes = max(0, (latest_fresh - latest_known) // 60_000)
            if gap_minutes >= requested - 2:
                fresh = self.client.bars("M1", min(self.history_bars, int(gap_minutes + 10)))
            fresh = pd.concat([self._bars, fresh], ignore_index=True)
        self._bars = fresh.sort_values("timestamp").drop_duplicates("timestamp", keep="last").tail(self.history_bars).reset_index(drop=True)
        return self._bars

    def poll(self) -> LiveSnapshot:
        with self._lock:
            status = self.client.connect()
            tick = self.client.latest_tick()
            bars = self._refresh_bars()
            completed = bars[bars.is_complete.astype(bool)].reset_index(drop=True)
            metadata = {
                "source": "MetaTrader 5", "server": status.server, "symbol": status.symbol,
                "server_utc_offset_seconds": status.server_utc_offset_seconds,
                "timestamps_normalized_to": "UTC", "ask_ohlc": "approximated from BID OHLC plus MT5 bar spread",
            }
            added = self.store.append_completed(completed, metadata)
            latest_completed = int(completed.timestamp.iloc[-1]) if not completed.empty else None
            if self._inference is None or latest_completed != self._last_inference_timestamp:
                self._inference = self.inference_engine.predict(completed)
                self._last_inference_timestamp = latest_completed
            validation_bucket = latest_completed // 300_000 if latest_completed is not None else None
            if validation_bucket != self._last_validation_bucket:
                self.client.bars("M5", 10)
                native_m5 = self.client.bars("M5", 100)
                self._m5_validation = compare_native_bars(
                    aggregate_m1(bars, 5), native_m5, self.client.point() * 1.1,
                )
                self._last_validation_bucket = validation_bucket
            assert self._inference is not None
            return LiveSnapshot(status, tick, bars.copy(), self._inference, added, self._m5_validation.copy())
