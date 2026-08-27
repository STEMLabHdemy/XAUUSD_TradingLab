from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, Backtester, performance_metrics
from src.signals import AggregationConfig, SignalAggregator, SignalConfig, SignalEngine


class Phase4Tests(unittest.TestCase):
    def test_weighted_temporal_probability(self) -> None:
        probabilities = pd.Series([.1, .2, .3, .4, .5])
        score = SignalAggregator().temporal(probabilities)
        expected = .40 * .5 + .25 * .4 + .15 * .3 + .12 * .2 + .08 * .1
        self.assertAlmostEqual(score.iloc[4], expected)
        self.assertTrue(score.iloc[:4].isna().all())

    def test_multi_horizon_weighting_and_agreement(self) -> None:
        config = AggregationConfig(horizon_weights={1: .25, 5: .75}, minimum_horizon_agreement=.7)
        values = pd.DataFrame({"p_up_1m": [.4], "p_up_5m": [.8]})
        result = SignalAggregator(config).multi_horizon(values)
        self.assertAlmostEqual(result.multi_horizon_score.iloc[0], .7)
        self.assertTrue(bool(result.horizon_agreement_passed.iloc[0]))

    def test_no_trade_persistence_and_position_state(self) -> None:
        engine = SignalEngine(SignalConfig(persistence=2, cooldown_minutes=3))
        t = pd.Timestamp("2026-01-01T00:00:00Z")
        self.assertEqual(engine.decide(t, .5, 2000, .5).signal, "NO_TRADE")
        self.assertEqual(engine.decide(t + pd.Timedelta(minutes=1), .7, 2000, .5).signal, "NO_TRADE")
        self.assertEqual(engine.decide(t + pd.Timedelta(minutes=2), .7, 2000, .5).signal, "BUY")
        self.assertEqual(engine.decide(t + pd.Timedelta(minutes=3), .7, 2000, .5).signal, "HOLD")
        self.assertEqual(engine.decide(t + pd.Timedelta(minutes=4), .3, 2000, .5).signal, "EXIT")
        self.assertEqual(engine.decide(t + pd.Timedelta(minutes=5), .3, 2000, .5).signal, "NO_TRADE")

    def test_expected_return_and_spread_filters(self) -> None:
        config = SignalConfig(persistence=1, require_expected_return_filter=True, max_spread=2)
        engine = SignalEngine(config)
        t = pd.Timestamp("2026-01-01T00:00:00Z")
        blocked = engine.decide(t, .8, 2000, 1, expected_return=.0001)
        self.assertEqual(blocked.signal, "NO_TRADE")
        self.assertIn("expected move does not exceed costs", blocked.reasons)
        passed = engine.decide(t + pd.Timedelta(minutes=1), .8, 2000, 1, expected_return=.002)
        self.assertEqual(passed.signal, "BUY")

    def _bars(self) -> pd.DataFrame:
        times = pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC")
        return pd.DataFrame({
            "timestamp": times.astype("int64") // 1_000_000, "datetime_utc": times,
            "open_bid": [100., 100., 102.], "high_bid": [101., 101., 103.],
            "low_bid": [99., 99., 101.], "close_bid": [100., 100., 102.],
            "open_ask": [101., 101., 103.], "high_ask": [102., 102., 104.],
            "low_ask": [100., 100., 102.], "close_ask": [101., 101., 103.],
            "signal": ["BUY", "EXIT", "NO_TRADE"], "score": [.8, .4, .4],
        })

    def test_long_uses_ask_entry_bid_exit_and_slippage(self) -> None:
        config = BacktestConfig(position_size_units=1, slippage_price_per_side=.1, stop_loss_price=None, take_profit_price=None, max_holding_minutes=None)
        result = Backtester(config).run(self._bars())
        trade = result.trades.iloc[0]
        self.assertEqual(trade.raw_entry_price, 101.)
        self.assertEqual(trade.raw_exit_price, 102.)
        self.assertAlmostEqual(trade.entry_price, 101.1)
        self.assertAlmostEqual(trade.exit_price, 101.9)
        self.assertAlmostEqual(trade.gross_pnl, 1.)
        self.assertAlmostEqual(trade.costs, .2)
        self.assertAlmostEqual(trade.net_pnl, .8)

    def test_short_uses_bid_entry_ask_exit(self) -> None:
        bars = self._bars()
        bars.signal = ["SELL", "EXIT", "NO_TRADE"]
        config = BacktestConfig(position_size_units=1, slippage_price_per_side=0, stop_loss_price=None, take_profit_price=None, max_holding_minutes=None)
        trade = Backtester(config).run(bars).trades.iloc[0]
        self.assertEqual(trade.raw_entry_price, 100.)
        self.assertEqual(trade.raw_exit_price, 103.)
        self.assertEqual(trade.net_pnl, -3.)

    def test_performance_metrics(self) -> None:
        result = Backtester(BacktestConfig(stop_loss_price=None, take_profit_price=None, max_holding_minutes=None)).run(self._bars())
        metrics = performance_metrics(result.trades, result.equity_curve, 100_000)
        self.assertEqual(metrics["trades"], 1)
        self.assertIn("profit_factor", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()
