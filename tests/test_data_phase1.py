from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.io import atomic_write_parquet, read_side_csv
from src.data.pipeline import deduplicate, merge_bid_ask, update_history
from src.data.validation import data_quality_summary, validate_side_frame


ROOT = Path(__file__).resolve().parents[1]
BID_SAMPLE = ROOT / "data/raw/bid/xauusd-m1-bid-2026-08-20-2026-08-21.csv"
ASK_SAMPLE = ROOT / "data/raw/ask/xauusd-m1-ask-2026-08-20-2026-08-21.csv"


class DataPhase1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bid = read_side_csv(BID_SAMPLE, "bid")
        cls.ask = read_side_csv(ASK_SAMPLE, "ask")

    def test_samples_validate(self) -> None:
        self.assertTrue(validate_side_frame(self.bid, "bid").valid)
        self.assertTrue(validate_side_frame(self.ask, "ask").valid)

    def test_timestamp_is_utc_and_aligned(self) -> None:
        self.assertEqual(str(self.bid.datetime_utc.dt.tz), "UTC")
        self.assertEqual(set(self.bid.timestamp), set(self.ask.timestamp))

    def test_duplicate_handling_is_deterministic(self) -> None:
        duplicated = pd.concat([self.bid.assign(_source_file="a"), self.bid.iloc[:1].assign(_source_file="z")])
        clean, removed = deduplicate(duplicated)
        self.assertEqual(removed, 1)
        self.assertEqual(len(clean), len(self.bid))

    def test_outer_merge_and_bid_ask_spread(self) -> None:
        merged = merge_bid_ask(self.bid.assign(_source_file="bid"), self.ask.assign(_source_file="ask"))
        self.assertEqual(len(merged), len(self.bid))
        self.assertTrue(merged._merge.eq("both").all())
        self.assertTrue(merged.spread_close.ge(0).all())
        self.assertAlmostEqual(merged.iloc[0].mid_close, (4513.125 + 4513.955) / 2)

    def test_unmatched_rows_are_retained_and_reported(self) -> None:
        merged = merge_bid_ask(self.bid.assign(_source_file="bid"), self.ask.iloc[:-1].assign(_source_file="ask"))
        bid_report = validate_side_frame(self.bid, "bid")
        ask_report = validate_side_frame(self.ask.iloc[:-1], "ask")
        summary = data_quality_summary(merged, bid_report, ask_report)
        self.assertEqual(int(summary.iloc[0].missing_ask_timestamps), 1)
        self.assertEqual(len(merged), len(self.bid))

    def test_invalid_ohlc_is_detected(self) -> None:
        broken = self.bid.copy()
        broken.loc[0, "high_bid"] = broken.loc[0, "low_bid"] - 1
        self.assertEqual(validate_side_frame(broken, "bid").invalid_ohlc, 1)

    def test_csv_parser_accepts_epoch_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seconds.csv"
            path.write_text("timestamp,open,high,low,close\n1700000000,1,2,0.5,1.5\n", encoding="utf-8")
            parsed = read_side_csv(path, "bid")
            self.assertEqual(str(parsed.datetime_utc.dt.tz), "UTC")
            self.assertEqual(int(parsed.timestamp.iloc[0]), 1_700_000_000_000)

    def test_incremental_update_refuses_to_bootstrap_full_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "full historical download"):
                update_history(directory)

    def test_atomic_parquet_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sample.parquet"
            atomic_write_parquet(self.bid.head(3), destination)
            restored = pd.read_parquet(destination)
            self.assertEqual(len(restored), 3)

    def test_weekday_gap_is_reported(self) -> None:
        gapped = self.bid.drop(index=10).reset_index(drop=True)
        report = validate_side_frame(gapped, "bid")
        self.assertEqual(report.unexpected_gaps, 1)
        self.assertEqual(report.weekday_gaps, 1)


if __name__ == "__main__":
    unittest.main()
