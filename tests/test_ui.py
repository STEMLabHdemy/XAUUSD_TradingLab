from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class UISmokeTests(unittest.TestCase):
    def test_dashboard_renders_without_exception(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=20)
        self.assertEqual(app.exception, [])
        self.assertTrue(any("XAUUSD TradingLab" in title.value for title in app.title))


if __name__ == "__main__":
    unittest.main()
