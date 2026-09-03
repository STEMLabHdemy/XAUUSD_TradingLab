from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live.clock import infer_server_utc_offset, normalize_server_epoch
from src.live.mt5_client import MT5Client
from src.live.storage import LiveBarStore
from src.live.timeframes import aggregate_m1, chart_bars, compare_native_bars
from src.data.market_hours import live_session_open_mask


def m1_frame(rows: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2026-08-27 20:00", periods=rows, freq="min", tz="UTC")
    frame = pd.DataFrame({"datetime_utc": dates})
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    frame["timestamp"] = (frame.datetime_utc - epoch) // pd.Timedelta(milliseconds=1)
    values = pd.Series(range(rows), dtype=float) + 100
    for base in ("bid", "ask"):
        adjustment = 1 if base == "ask" else 0
        frame[f"open_{base}"] = values + adjustment
        frame[f"high_{base}"] = values + adjustment + 2
        frame[f"low_{base}"] = values + adjustment - 1
        frame[f"close_{base}"] = values + adjustment + .5
    frame["mid_open"] = values + .5
    frame["mid_high"] = values + 2.5
    frame["mid_low"] = values - .5
    frame["mid_close"] = values + 1
    frame["spread_close"] = 1.0
    frame["spread_points"] = 10
    frame["tick_volume"] = 3
    frame["real_volume"] = 0
    frame["source"] = "MT5"
    frame["source_symbol"] = "XAUUSD"
    frame["server_utc_offset_seconds"] = 10800
    frame["raw_server_datetime"] = frame.datetime_utc + pd.Timedelta(hours=3)
    frame["is_complete"] = True
    return frame


class LiveClockTests(unittest.TestCase):
    def test_broker_utc_plus_three_is_detected_and_normalized(self) -> None:
        observed = datetime(2026, 8, 27, 23, 30, tzinfo=timezone.utc)
        raw = (observed + timedelta(hours=3)).timestamp()
        offset = infer_server_utc_offset(raw, observed)
        self.assertEqual(offset, 10800)
        normalized = normalize_server_epoch(pd.Series([raw]), offset)
        self.assertEqual(normalized.iloc[0], pd.Timestamp(observed))


class TimeframeTests(unittest.TestCase):
    def test_m5_ohlc_aggregation_uses_exact_five_minute_buckets(self) -> None:
        result = aggregate_m1(m1_frame(), 5)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.is_complete.all())
        self.assertEqual(result.open_bid.iloc[0], 100)
        self.assertEqual(result.high_bid.iloc[0], 106)
        self.assertEqual(result.low_bid.iloc[0], 99)
        self.assertEqual(result.close_bid.iloc[0], 104.5)
        self.assertEqual(result.tick_volume.iloc[0], 15)

    def test_incomplete_or_gapped_bucket_is_not_marked_complete(self) -> None:
        frame = m1_frame().drop(index=3).reset_index(drop=True)
        result = aggregate_m1(frame, 5)
        self.assertFalse(bool(result.is_complete.iloc[0]))
        visible = chart_bars(frame, "M5")
        self.assertNotEqual(int(visible.timestamp.iloc[0]), int(result.timestamp.iloc[0]))

    def test_native_comparison_reports_exact_match(self) -> None:
        aggregated = aggregate_m1(m1_frame(), 5)
        native = aggregated.copy()
        result = compare_native_bars(aggregated, native, tolerance=.01)
        self.assertTrue(result["valid"])
        self.assertEqual(result["max_absolute_error"], 0)


class LiveStorageTests(unittest.TestCase):
    def test_store_is_deduplicated_and_only_persists_completed_bars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LiveBarStore(Path(directory) / "live.parquet")
            frame = m1_frame(3)
            frame.loc[2, "is_complete"] = False
            self.assertEqual(store.append_completed(frame), 2)
            self.assertEqual(store.append_completed(frame), 0)
            saved = store.load()
            self.assertEqual(len(saved), 2)
            self.assertTrue(saved.timestamp.is_unique)

    def test_client_has_no_order_routing_surface(self) -> None:
        self.assertFalse(hasattr(MT5Client, "order_send"))

    def test_retain_removes_closed_session_bars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LiveBarStore(Path(directory) / "live.parquet")
            frame = m1_frame(3)
            frame["datetime_utc"] = pd.to_datetime([
                "2026-09-02T20:59:00Z",  # 22:59 Europe/Rome, still open
                "2026-09-02T21:00:00Z",  # 23:00 Europe/Rome, daily break
                "2026-09-05T10:00:00Z",  # Saturday
            ], utc=True)
            self.assertEqual(store.append_completed(frame), 3)
            self.assertEqual(store.retain(live_session_open_mask(store.load())), 2)
            saved = store.load()
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.datetime_utc.iloc[0], pd.Timestamp("2026-09-02T20:59:00Z"))


class LiveMarketCalendarTests(unittest.TestCase):
    def test_daily_break_and_weekend_are_excluded(self) -> None:
        frame = pd.DataFrame({"datetime_utc": pd.to_datetime([
            "2026-09-02T20:59:00Z",  # Wed 22:59 Europe/Rome
            "2026-09-02T21:00:00Z",  # Wed 23:00 Europe/Rome
            "2026-09-02T22:00:00Z",  # Thu 00:00 Europe/Rome
            "2026-09-05T10:00:00Z",  # Saturday
            "2026-09-06T10:00:00Z",  # Sunday
        ], utc=True)})
        self.assertEqual(live_session_open_mask(frame).tolist(), [True, False, True, False, False])


if __name__ == "__main__":
    unittest.main()
