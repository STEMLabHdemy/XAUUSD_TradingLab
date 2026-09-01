from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from streamlit.testing.v1 import AppTest

from tests.test_live import m1_frame


class UISmokeTests(unittest.TestCase):
    def test_dashboard_renders_without_exception(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=20)
        self.assertEqual(app.exception, [])
        self.assertTrue(any("XAUUSD TradingLab" in title.value for title in app.title))

    def test_live_chart_viewport_revision_is_stable_across_ticks(self) -> None:
        from src.ui.live_dashboard import live_candlestick_figure

        first = m1_frame(10)
        second = first.copy()
        second.loc[second.index[-1], "mid_close"] += 1
        figure_a = live_candlestick_figure(first, "M1", "Europe/Rome", 10, True)
        figure_b = live_candlestick_figure(second, "M1", "Europe/Rome", 10, True)
        manual = live_candlestick_figure(second, "M1", "Europe/Rome", 10, False)
        self.assertEqual(figure_a.layout.uirevision, figure_b.layout.uirevision)
        self.assertNotEqual(figure_b.layout.uirevision, manual.layout.uirevision)
        self.assertGreater(figure_a.layout.yaxis.domain[0], figure_a.layout.yaxis2.domain[1])
        self.assertFalse(figure_a.layout.xaxis.rangeslider.visible)

    def test_live_chart_accepts_mixed_iso8601_event_timestamps(self) -> None:
        from src.ui.live_dashboard import live_candlestick_figure

        events = pd.DataFrame([
            {"event": "BUY", "timestamp": "2026-09-01T00:11:59+00:00", "price": 4500.0,
             "model": "Test", "confidence": .6, "spread": .2},
            {"event": "EXIT", "timestamp": "2026-09-01T00:12:00.125000+00:00", "price": 4501.0,
             "model": "Test", "confidence": .6, "spread": .2, "net_pnl": 1.0},
        ])
        figure = live_candlestick_figure(m1_frame(10), "M1", "Europe/Rome", 10, True, events)
        self.assertGreaterEqual(len(figure.data), 2)

    def test_model_laboratory_renders_without_exception(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=20)
        app.sidebar.radio[0].set_value("Models").run(timeout=20)
        self.assertEqual(app.exception, [])
        self.assertTrue(any("Laboratorio modelli" in title.value for title in app.title))


if __name__ == "__main__":
    unittest.main()
