"""Lightweight local MT5 Paper dashboard. No Streamlit, no broker orders."""
from __future__ import annotations

from pathlib import Path
import sys
import traceback
from typing import Any
import json

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.mt5_client import MT5Client


STATIC = ROOT / "web" / "static"
app = FastAPI(title="XAUUSD Live Paper")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_paper_directory = ROOT / "data" / "live" / "paper" / "comparison_v1"


def _load_ledger_states() -> tuple[str, dict[str, dict[str, Any]]]:
    """Read the states written by the headless engine without ever modifying them."""
    metadata = json.loads((_paper_directory / "run.json").read_text(encoding="utf-8"))
    run_id = str(metadata["run_id"])
    states: dict[str, dict[str, Any]] = {}
    for path in _paper_directory.glob("comparison_v1_*_*/state.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("run_id") == run_id:
                states[str(state["model"])] = state
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    if not states:
        raise RuntimeError("Nessun ledger della sessione corrente disponibile")
    return run_id, states


def _load_indicator_states() -> tuple[str, dict[str, dict[str, Any]]]:
    directory = ROOT / "data" / "live" / "paper" / "indicator_v1"
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    run_id = str(metadata["run_id"])
    states = {}
    for path in directory.glob("indicator_v1_*/state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("run_id") == run_id: states[str(state["model"])] = state
    return run_id, states


def _market_payload() -> tuple[dict[str, Any], pd.DataFrame]:
    """Read-only MT5 request. The headless engine remains the sole ledger writer."""
    client = MT5Client(Path(r"C:\Program Files\MetaTrader 5\terminal64.exe"), ("XAUUSD", "GOLD"))
    try:
        client.connect()
        tick = client.latest_tick()
        bars = client.bars("M1", 300)
        return ({"bid": tick.bid, "ask": tick.ask, "spread": tick.spread,
                 "timestamp": tick.datetime_utc.isoformat()}, bars)
    finally:
        client.shutdown()


def _latest_inference() -> dict[str, Any] | None:
    """The sole inference is produced by headless.py and exposed through its status file."""
    try:
        payload = json.loads((_paper_directory / "headless_status.json").read_text(encoding="utf-8"))
        return payload.get("inference")
    except (OSError, json.JSONDecodeError):
        return None


def _number(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/indicators")
def indicators_page() -> FileResponse:
    return FileResponse(STATIC / "indicators.html")


@app.get("/api/indicators")
def indicators() -> dict[str, Any]:
    try:
        run_id, accounts = _load_indicator_states()
        cards = []
        for name, state in accounts.items():
            cards.append({"name": name, "strategy": state.get("config", {}).get("strategy_id", "?"),
                          "total_pnl": _number(state["realized_pnl"]) + _number(state["unrealized_pnl"]),
                          "trades": len(state.get("trades") or []), "open_positions": len(state.get("positions") or []),
                          "last_signal": state.get("last_signal"), "last_reason": state.get("last_reason")})
        return {"run_id": run_id, "cards": cards}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc) or repr(exc)) from exc


@app.get("/api/snapshot")
def snapshot(strategy: str | None = None) -> dict[str, Any]:
    """Return a compact read-only view of the active headless paper session."""
    try:
        run_id, accounts = _load_ledger_states()
        tick, bars = _market_payload()
        account_names = list(accounts)
        selected = strategy if strategy in accounts else account_names[0]
        account = accounts[selected]
        bars = bars.tail(300).copy()
        bars["datetime_utc"] = pd.to_datetime(bars["datetime_utc"], utc=True)
        candles = [
            {"time": int(row.datetime_utc.timestamp()), "open": float(row.mid_open), "high": float(row.mid_high),
             "low": float(row.mid_low), "close": float(row.mid_close), "volume": float(row.tick_volume)}
            for row in bars.itertuples(index=False)
        ]
        chart_start, chart_end = candles[0]["time"], candles[-1]["time"]
        events = pd.DataFrame(account.get("events", [])).tail(200)
        markers = []
        if not events.empty:
            for row in events.itertuples(index=False):
                timestamp = pd.to_datetime(row.timestamp, utc=True, errors="coerce")
                if pd.isna(timestamp):
                    continue
                event_time = int(timestamp.timestamp())
                if not chart_start <= event_time <= chart_end:
                    continue
                event = str(row.event)
                if event not in {"BUY", "SELL", "EXIT"}:
                    continue
                markers.append({
                    "time": event_time, "position": "belowBar" if event == "BUY" else "aboveBar",
                    "color": "#22c55e" if event == "BUY" else "#ef4444" if event == "SELL" else "#f59e0b",
                    "shape": "arrowUp" if event == "BUY" else "arrowDown" if event == "SELL" else "circle",
                    "text": event,
                })
        levels = []
        for position in account.get("positions", []):
            side = str(position["side"])
            entry_color = "#38bdf8" if side == "LONG" else "#f97316"
            levels.append({"price": float(position["entry_price"]), "color": entry_color,
                           "line_style": 0, "title": f"{side} entry #{position['trade_id']}"})
            if position.get("stop_loss") is not None:
                levels.append({"price": float(position["stop_loss"]), "color": "#ef4444",
                               "line_style": 2, "title": f"SL #{position['trade_id']}"})
            if position.get("take_profit") is not None:
                levels.append({"price": float(position["take_profit"]), "color": "#22c55e",
                               "line_style": 2, "title": f"TP #{position['trade_id']}"})
        cards = []
        for name, state in accounts.items():
            cards.append({
                "name": name, "strategy": state.get("config", {}).get("strategy_id", "?"),
                "total_pnl": _number(state["realized_pnl"]) + _number(state["unrealized_pnl"]),
                "realized_pnl": _number(state["realized_pnl"]), "unrealized_pnl": _number(state["unrealized_pnl"]),
                "open_positions": len(state.get("positions") or []), "trades": len(state.get("trades") or []),
                "max_drawdown": _number(state["max_drawdown"]), "last_signal": state.get("last_signal"),
            })
        return {
            "tick": tick,
            "candles": candles, "markers": markers[-80:], "levels": levels,
            "selected": selected, "strategies": account_names,
            "cards": cards, "run_id": run_id, "inference": _latest_inference(),
        }
    except Exception as exc:  # Browser gets the actionable failure, server stays alive.
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=str(exc) or repr(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
