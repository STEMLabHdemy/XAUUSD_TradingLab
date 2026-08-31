from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from threading import RLock

import pandas as pd

from src.live.inference import LiveInference
from src.live.mt5_client import MarketTick
from src.paper import PaperAccount, PaperConfig, PaperRuntime


def tick(minute: int, bid: float, ask: float) -> MarketTick:
    stamp = pd.Timestamp("2026-08-28 10:00", tz="UTC") + pd.Timedelta(minutes=minute)
    return MarketTick(stamp, stamp, bid, ask, ask - bid, "XAUUSD")


def inference(minute: int, probability: float) -> LiveInference:
    stamp = pd.Timestamp("2026-08-28 09:59", tz="UTC") + pd.Timedelta(minutes=minute)
    return LiveInference(True, "test", 5, probability, stamp, "BUY", "NO_TRADE", "test")


class PaperAccountTests(unittest.TestCase):
    def account(self, directory: str, **overrides) -> PaperAccount:
        values = {
            "persistence": 1, "cooldown_minutes": 0, "slippage_price_per_side": .1,
            "commission_per_unit_per_side": .2, "position_size_units": 2,
            "stop_loss_price": 5, "take_profit_price": 10,
        }
        values.update(overrides)
        return PaperAccount("test", "TestModel", PaperConfig(**values), Path(directory))

    def test_long_uses_ask_entry_bid_exit_and_costs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            state = account.snapshot()
            self.assertEqual(state["position"]["raw_entry_price"], 101)
            self.assertEqual(state["position"]["entry_price"], 101.1)
            account.process(tick(1, 104, 105), inference(1, .1))
            trade = account.trades_frame().iloc[0]
            self.assertEqual(trade.raw_exit_price, 104)
            self.assertEqual(trade.exit_price, 103.9)
            self.assertAlmostEqual(trade.gross_pnl, 6.0)
            self.assertAlmostEqual(trade.costs, 1.2)
            self.assertAlmostEqual(trade.net_pnl, 4.8)
            self.assertAlmostEqual(account.snapshot()["balance"], 100004.8)

    def test_short_uses_bid_entry_ask_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .1))
            self.assertEqual(account.snapshot()["position"]["entry_price"], 99.9)
            account.process(tick(1, 96, 97), inference(1, .9))
            trade = account.trades_frame().iloc[0]
            self.assertEqual(trade.raw_exit_price, 97)
            self.assertEqual(trade.exit_price, 97.1)
            self.assertAlmostEqual(trade.net_pnl, 4.8)

    def test_same_completed_bar_cannot_open_twice_and_state_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            account.process(tick(0, 100, 101), inference(0, .9))
            self.assertEqual(account.snapshot()["next_trade_id"], 2)
            reloaded = self.account(directory)
            self.assertEqual(reloaded.snapshot()["position"]["trade_id"], 1)
            self.assertTrue(reloaded.snapshot()["running"])

    def test_mark_to_market_margin_and_stop_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, leverage=10)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            account.process(tick(0, 102, 103), inference(0, .9))
            state = account.snapshot()
            self.assertGreater(state["unrealized_pnl"], 0)
            self.assertAlmostEqual(state["used_margin"], 20.4)
            account.process(tick(1, 95, 96), inference(1, .9))
            self.assertIsNone(account.snapshot()["position"])
            self.assertEqual(account.trades_frame().iloc[0].exit_reason, "stop_loss")

    def test_stopped_account_never_opens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory)
            account.process(tick(0, 100, 101), inference(0, .9))
            self.assertIsNone(account.snapshot()["position"])

    def test_manual_protection_update_is_validated_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory)
            account.start()
            current = tick(0, 100, 101)
            account.process(current, inference(0, .9))
            with self.assertRaisesRegex(ValueError, "sotto il BID"):
                account.update_protection(current, 100, 110)
            account.update_protection(current, 95, 110)
            reloaded = self.account(directory)
            position = reloaded.snapshot()["position"]
            self.assertEqual(position["stop_loss"], 95)
            self.assertEqual(position["take_profit"], 110)
            self.assertEqual(reloaded.snapshot()["events"][-1]["event"], "MODIFY")

    def test_manual_close_records_virtual_trade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .1))
            account.close_manually(tick(1, 98, 99))
            state = account.snapshot()
            self.assertIsNone(state["position"])
            self.assertEqual(state["trades"][-1]["exit_reason"], "manual_close")
            self.assertEqual(state["last_reason"], "chiusura manuale")

    def test_session_close_flattens_all_legs_and_blocks_late_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, entry_mode="burst")
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            account.process(tick(1, 100, 101), inference(1, .9))
            session_tick = MarketTick(
                pd.Timestamp("2026-08-28 20:55", tz="UTC"),
                pd.Timestamp("2026-08-28 20:55", tz="UTC"), 100, 101, 1, "XAUUSD",
            )
            account.process(session_tick, inference(2, .9))
            state = account.snapshot()
            self.assertEqual(state["positions"], [])
            self.assertEqual(len(state["trades"]), 2)
            self.assertTrue(all(row["exit_reason"] == "session_close" for row in state["trades"]))
            self.assertEqual(state["last_reason"], "session close: 2 position(s) closed at last available tick")

    def test_runtime_comparison_separates_realized_open_and_total_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            account.process(tick(0, 103, 104), inference(0, .9))
            runtime = PaperRuntime.__new__(PaperRuntime)
            runtime.accounts = {"TestModel": account}

            row = runtime.comparison().iloc[0]

            self.assertAlmostEqual(row.total_pnl, row.realized_pnl + row.unrealized_pnl)
            self.assertAlmostEqual(row.return_pct, row.total_pnl / account.config.starting_capital)
            self.assertEqual(row.position, "1 LONG")

    def test_controlled_mode_keeps_one_open_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, entry_mode="controlled")
            account.start()
            for minute in range(5):
                account.process(tick(minute, 100, 101), inference(minute, .9))
            self.assertEqual(len(account.snapshot()["positions"]), 1)

    def test_intermediate_mode_opens_three_legs_with_three_minute_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, entry_mode="intermediate")
            account.start()
            for minute in range(7):
                account.process(tick(minute, 100, 101), inference(minute, .9))
            state = account.snapshot()
            self.assertEqual(len(state["positions"]), 3)
            self.assertEqual([row["trade_id"] for row in state["positions"]], [1, 2, 3])

    def test_burst_mode_opens_one_leg_per_completed_bar_and_closes_all_on_reversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, entry_mode="burst")
            account.start()
            for minute in range(12):
                account.process(tick(minute, 100, 101), inference(minute, .9))
            self.assertEqual(len(account.snapshot()["positions"]), 10)
            account.process(tick(12, 100, 101), inference(12, .1))
            state = account.snapshot()
            self.assertEqual(state["positions"], [])
            self.assertEqual(len(state["trades"]), 10)
            self.assertTrue(all(row["exit_reason"] == "probability reversal" for row in state["trades"]))
            account.process(tick(13, 100, 101), inference(13, .9))
            self.assertEqual(len(account.snapshot()["positions"]), 1)

    def test_manual_actions_target_one_leg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, entry_mode="burst")
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            account.process(tick(1, 100, 101), inference(1, .9))
            account.update_protection(tick(2, 100, 101), 94, 112, trade_id=2)
            positions = account.snapshot()["positions"]
            self.assertEqual(positions[0]["stop_loss"], 96.1)
            self.assertEqual(positions[1]["stop_loss"], 94)
            account.close_manually(tick(2, 100, 101), trade_id=2)
            self.assertEqual([row["trade_id"] for row in account.snapshot()["positions"]], [1])

    def test_mode_change_preserves_open_legs_and_aggregates_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, entry_mode="burst", leverage=10)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            account.process(tick(1, 100, 101), inference(1, .9))
            self.assertEqual(len(account.snapshot()["positions"]), 2)
            self.assertAlmostEqual(account.snapshot()["used_margin"], 40.0)
            account.set_entry_mode("controlled")
            state = account.snapshot()
            self.assertEqual(len(state["positions"]), 2)
            self.assertEqual(account.config.entry_mode, "controlled")
            account.process(tick(2, 100, 101), inference(2, .9))
            self.assertEqual(len(account.snapshot()["positions"]), 2)

    def test_config_update_preserves_open_legs_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            account = self.account(directory, entry_mode="burst", max_daily_trades=2)
            account.start()
            account.process(tick(0, 100, 101), inference(0, .9))
            original_position = account.snapshot()["positions"][0]["trade_id"]
            account.update_config_preserving_history(PaperConfig(
                **{**asdict(account.config), "max_daily_trades": None}
            ))
            state = account.snapshot()
            self.assertEqual(state["positions"][0]["trade_id"], original_position)
            self.assertIsNone(account.config.max_daily_trades)
            account.process(tick(1, 100, 101), inference(1, .9))
            self.assertEqual(len(account.snapshot()["positions"]), 2)

    def test_runtime_exposes_selected_model_inference(self) -> None:
        runtime = PaperRuntime.__new__(PaperRuntime)
        expected = inference(0, .7)
        runtime._inferences = {"XGBoost H15": expected}
        runtime._lock = RLock()
        self.assertIs(runtime.inference_for("XGBoost H15"), expected)
        self.assertIsNone(runtime.inference_for("missing"))


if __name__ == "__main__":
    unittest.main()
