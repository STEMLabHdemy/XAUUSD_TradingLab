from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.io import read_side_csv
from src.data.pipeline import merge_bid_ask
from src.data.snapshot import build_master_snapshot
from src.data.market_hours import filter_market_closed
from src.features import FeatureEngine
from src.modeling.dataset import aggregate_training_bars
from src.targets import TargetConfig, TargetEngine


ROOT = Path(__file__).resolve().parents[1]
BID_SAMPLE = ROOT / "data/raw/bid/xauusd-m1-bid-2026-08-20-2026-08-21.csv"
ASK_SAMPLE = ROOT / "data/raw/ask/xauusd-m1-ask-2026-08-20-2026-08-21.csv"


def sample_merged() -> pd.DataFrame:
    bid = read_side_csv(BID_SAMPLE, "bid").assign(_source_file=BID_SAMPLE.name)
    ask = read_side_csv(ASK_SAMPLE, "ask").assign(_source_file=ASK_SAMPLE.name)
    return merge_bid_ask(bid, ask).drop(columns=["_merge", "_source_file_x", "_source_file_y"])


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.market = sample_merged()

    def test_feature_engine_creates_required_families(self) -> None:
        result = FeatureEngine().transform(self.market)
        required = {
            "return_60m", "log_return_60m", "atr_60", "rolling_volatility_60m",
            "rsi_14", "macd_histogram", "ema_distance_200", "bollinger_position",
            "spread_to_atr", "spread_percentile", "session_london", "trend_regime",
            "volatility_regime", "range_regime",
        }
        self.assertTrue(required.issubset(result.columns))

    def test_future_price_changes_do_not_change_past_features(self) -> None:
        engine = FeatureEngine()
        cutoff = 900
        baseline = engine.transform(self.market)
        changed = self.market.copy()
        price_columns = ["mid_open", "mid_high", "mid_low", "mid_close"]
        changed.loc[cutoff + 1:, price_columns] *= 10
        modified = engine.transform(changed)
        feature_columns = [column for column in baseline.columns if column not in self.market.columns]
        assert_frame_equal(
            baseline.loc[:cutoff, feature_columns], modified.loc[:cutoff, feature_columns],
            check_dtype=True, check_exact=True,
        )

    def test_targets_are_forward_aligned(self) -> None:
        result = TargetEngine(TargetConfig(horizons=(1, 3))).transform(self.market)
        expected = self.market.mid_close.iloc[3] / self.market.mid_close.iloc[0] - 1
        self.assertAlmostEqual(result.future_return_3m.iloc[0], expected)
        self.assertTrue(result.future_return_3m.tail(3).isna().all())
        self.assertTrue(result.up_3m.tail(3).isna().all())
        self.assertTrue(result.direction_3m.tail(3).isna().all())

    def test_targets_do_not_cross_market_gaps(self) -> None:
        gapped = self.market.head(10).copy().reset_index(drop=True)
        gapped.loc[5:, "timestamp"] += 3_600_000
        gapped["datetime_utc"] = pd.to_datetime(gapped.timestamp, unit="ms", utc=True)
        result = TargetEngine(TargetConfig(horizons=(1,))).transform(gapped)
        self.assertTrue(pd.isna(result.future_return_1m.iloc[4]))
        features = FeatureEngine().transform(gapped)
        self.assertTrue(pd.isna(features.return_1m.iloc[5]))

    def test_neutral_target_accounts_for_spread(self) -> None:
        flat = self.market.head(2).copy()
        flat.loc[flat.index[1], "mid_close"] = flat.loc[flat.index[0], "mid_close"]
        result = TargetEngine(TargetConfig(horizons=(1,), neutral_cost_multiplier=1)).transform(flat)
        self.assertEqual(result.direction_1m.iloc[0], "NEUTRAL")

    def test_executable_target_uses_next_open_and_future_opposite_side(self) -> None:
        market = self.market.head(8).copy().reset_index(drop=True)
        market[["open_bid", "open_ask", "close_bid", "close_ask"]] = [100.0, 100.4, 100.0, 100.4]
        market.loc[3, "close_bid"] = 101.2
        result = TargetEngine(TargetConfig(
            horizons=(3,), executable_minimum_net_move=.5,
            slippage_price_per_side=.05,
        )).transform(market)

        self.assertAlmostEqual(result.loc[0, "future_long_net_3m"], .7)
        self.assertEqual(result.loc[0, "executable_direction_3m"], "UP")
        self.assertTrue(pd.isna(result.loc[5, "executable_direction_3m"]))

    def test_m5_aggregation_and_h30_target_use_real_minutes(self) -> None:
        bars = aggregate_training_bars(self.market.head(120), 5)
        self.assertTrue(len(bars) >= 20)
        self.assertTrue((bars.timestamp.diff().dropna() == 5 * 60_000).all())
        labelled = TargetEngine(TargetConfig(horizons=(30,), bar_minutes=5)).transform(bars)
        self.assertAlmostEqual(
            labelled.future_return_30m.iloc[0],
            bars.mid_close.iloc[6] / bars.mid_close.iloc[0] - 1,
        )
        self.assertTrue(labelled.future_return_30m.tail(6).isna().all())

    def test_streaming_snapshot_writes_canonical_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/raw/bid").mkdir(parents=True)
            (root / "data/raw/ask").mkdir(parents=True)
            shutil.copy2(BID_SAMPLE, root / "data/raw/bid/xauusd_bid_m1_2026_08.csv")
            shutil.copy2(ASK_SAMPLE, root / "data/raw/ask/xauusd_ask_m1_2026_08.csv")
            summary = build_master_snapshot(root, date(2026, 8, 20), date(2026, 8, 21))
            master = pd.read_parquet(root / "data/processed/XAUUSD_M1_MASTER.parquet")
            self.assertEqual(len(master), 1439)
            self.assertEqual(int(summary.number_of_candles.iloc[0]), 1439)
            self.assertIn("spread_close", master.columns)
            self.assertTrue((root / "reports/data_quality_monthly.csv").exists())

    def test_sunday_fill_is_removed_but_real_open_is_retained(self) -> None:
        times = pd.date_range("2026-08-02 21:57", periods=6, freq="min", tz="UTC")
        frame = pd.DataFrame({"datetime_utc": times})
        for side in ("bid", "ask"):
            frame[f"open_{side}"] = 100.0
            frame[f"high_{side}"] = 100.0
            frame[f"low_{side}"] = 100.0
            frame[f"close_{side}"] = 100.0
        frame.loc[3:, "high_bid"] = 101.0
        frame.loc[3:, "high_ask"] = 102.0
        filtered, excluded = filter_market_closed(frame)
        self.assertEqual(excluded, 3)
        self.assertEqual(filtered.datetime_utc.iloc[0], pd.Timestamp("2026-08-02 22:00", tz="UTC"))


if __name__ == "__main__":
    unittest.main()
