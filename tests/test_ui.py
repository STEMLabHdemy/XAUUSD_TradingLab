from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.ui.live_dashboard import live_candlestick_figure
from tests.test_live import m1_frame


class UISmokeTests(unittest.TestCase):
    def test_dashboard_renders_without_exception(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=20)
        self.assertEqual(app.exception, [])
        self.assertTrue(any("XAUUSD TradingLab" in title.value for title in app.title))

    def test_live_chart_viewport_revision_is_stable_across_ticks(self) -> None:
        first = m1_frame(10)
        second = first.copy()
        second.loc[second.index[-1], "mid_close"] += 1
        figure_a = live_candlestick_figure(first, "M1", "Europe/Rome", 10, True)
        figure_b = live_candlestick_figure(second, "M1", "Europe/Rome", 10, True)
        manual = live_candlestick_figure(second, "M1", "Europe/Rome", 10, False)
        self.assertEqual(figure_a.layout.uirevision, figure_b.layout.uirevision)
        self.assertNotEqual(figure_b.layout.uirevision, manual.layout.uirevision)


if __name__ == "__main__":
    unittest.main()
