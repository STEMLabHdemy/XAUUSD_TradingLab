"""Lightweight local MT5 Paper dashboard. No Streamlit, no broker orders."""
from __future__ import annotations

from pathlib import Path
import sys
import traceback
from typing import Any
import json
from threading import RLock, Thread
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.mt5_client import MT5Client
from src.backtest import BacktestConfig
from src.backtest.lab import run_lab, strategy_catalog, load_history
from src.backtest.engine import Backtester
from src.features import FeatureEngine
from src.signals.structure import structure_signals


STATIC = ROOT / "web" / "static"
app = FastAPI(title="XAUUSD Live Paper")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_paper_directory = ROOT / "data" / "live" / "paper" / "comparison_v1"
_market_cache_lock = RLock()
_last_market_payload: tuple[dict[str, Any], pd.DataFrame] | None = None
_backtest_lock = RLock()
_backtest_jobs: dict[str, dict[str, Any]] = {}
_backtest_results = ROOT / "results" / "backtest_lab"


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
    directory = ROOT / "data" / "live" / "paper" / "structure_v1"
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    run_id = str(metadata["run_id"])
    states = {}
    for path in directory.glob("structure_v1_*/state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        strategy_id = str(state.get("config", {}).get("strategy_id", ""))
        if state.get("run_id") == run_id and strategy_id == "I42":
            states[str(state["model"])] = state
    return run_id, states


def _market_payload(bar_count: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Read-only MT5 request with a last-good snapshot fallback for the UI."""
    global _last_market_payload
    # Same explicit fallback used by the headless service.  It matters when a
    # broker gives us a slightly stale tick, from which timezone cannot be
    # inferred safely on a brand-new client instance.
    client = MT5Client(
        Path(r"C:\Program Files\MetaTrader 5\terminal64.exe"),
        ("XAUUSD", "XAUUSD.ecn", "XAUUSD.cnt", "GOLD"),
        fallback_offset_seconds=10_800,
    )
    try:
        client.connect()
        tick = client.latest_tick()
        bars = client.bars("M1", bar_count)
        payload = {"bid": tick.bid, "ask": tick.ask, "spread": tick.spread,
                   "timestamp": tick.datetime_utc.isoformat(), "stale": False}
        with _market_cache_lock:
            _last_market_payload = (payload, bars.copy())
        return payload, bars
    except Exception:
        with _market_cache_lock:
            if _last_market_payload is not None:
                payload, bars = _last_market_payload
                cached = {**payload, "stale": True}
                return cached, bars.tail(bar_count).copy()
        raise
    finally:
        client.shutdown()


def _latest_inference() -> dict[str, Any] | None:
    """The sole inference is produced by headless.py and exposed through its status file."""
    try:
        payload = json.loads((_paper_directory / "headless_status.json").read_text(encoding="utf-8"))
        return payload.get("inference")
    except (OSError, json.JSONDecodeError):
        return None


def _paper_status() -> dict[str, Any] | None:
    try:
        return json.loads((ROOT / "data/live/paper/structure_v1/headless_status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _number(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "indicators.html")


@app.get("/indicators")
def indicators_page() -> FileResponse:
    return FileResponse(STATIC / "indicators.html")


@app.get("/backtest")
def backtest_page() -> FileResponse:
    return FileResponse(STATIC / "backtest.html")


@app.get("/research")
def research_page() -> FileResponse:
    return FileResponse(STATIC / "research.html")


@app.get("/adaptation")
def adaptation_page() -> FileResponse:
    return FileResponse(STATIC / "adaptation.html")


@app.get("/adaptive-research")
def adaptive_research_page() -> FileResponse:
    return FileResponse(STATIC / "adaptive_research.html")


@app.get("/api/research/leaderboard")
def research_leaderboard() -> dict[str, Any]:
    """Read-only snapshot written by the continuous local search process."""
    target = ROOT / "results" / "continuous_structure_search" / "leaderboard.json"
    status = ROOT / "results" / "continuous_structure_search" / "status.json"
    try:
        payload = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"leaderboard": [], "tested": 0}
        payload["status"] = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {"message": "ricerca non ancora avviata"}
        return payload
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Snapshot ricerca non leggibile: {exc}") from exc


@app.get("/api/research/validation")
def research_validation() -> dict[str, Any]:
    base = ROOT / "results" / "continuous_structure_search"
    try:
        payload = json.loads((base / "validation.json").read_text(encoding="utf-8")) if (base / "validation.json").exists() else {"results": [], "validated": 0}
        payload["status"] = json.loads((base / "validation_status.json").read_text(encoding="utf-8")) if (base / "validation_status.json").exists() else {"message": "validazione non ancora avviata"}
        return payload
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Snapshot validazione non leggibile: {exc}") from exc


@app.get("/api/research/adaptation")
def research_adaptation() -> dict[str, Any]:
    target = ROOT / "results" / "online_adaptation" / "result.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Replay adattivo non ancora eseguito")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Risultato adattivo non leggibile: {exc}") from exc


@app.get("/api/research/adaptive-leaderboard")
def adaptive_leaderboard() -> dict[str, Any]:
    directory = ROOT / "results" / "adaptive_search"
    try:
        result = json.loads((directory / "leaderboard.json").read_text(encoding="utf-8")) if (directory / "leaderboard.json").exists() else {"tested": 0, "leaderboard": []}
        result["status"] = json.loads((directory / "status.json").read_text(encoding="utf-8")) if (directory / "status.json").exists() else {"message": "ricerca non avviata"}
        return result
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"Classifica adattiva non leggibile: {exc}") from exc


@app.get("/api/research/adaptive-details/{candidate_id}")
def adaptive_candidate_details(candidate_id: str) -> dict[str, Any]:
    """Replay and cache the exact adaptive configuration selected in the UI."""
    directory = ROOT / "results" / "adaptive_search"
    cached = directory / "details" / f"{candidate_id}.json"
    try:
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        board = json.loads((directory / "leaderboard.json").read_text(encoding="utf-8"))
        candidate = next(row for row in board.get("leaderboard", []) if row["id"] == candidate_id)
        from src.research.online_adaptation import run as run_adaptive
        result = run_adaptive(ROOT, days=int(board["period"]["days"]),
                              threshold=float(candidate["config"]["threshold"]),
                              rolling_training_days=int(candidate["config"]["rolling_training_days"]),
                              params=candidate["config"]["params"], persist=False)
        payload = {"candidate": candidate, **result}
        cached.parent.mkdir(parents=True, exist_ok=True)
        temp = cached.with_name(f"{cached.stem}.{uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(cached)
        return payload
    except StopIteration as exc:
        raise HTTPException(status_code=404, detail="Candidato non più nella top 100") from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Replay adattivo fallito: {exc}") from exc


@app.get("/api/research/details/{candidate_id}")
def research_candidate_details(candidate_id: str) -> dict[str, Any]:
    """Replay one ranked configuration and cache its auditable chart payload."""
    base_dir = ROOT / "results" / "continuous_structure_search"
    cached = base_dir / "details" / f"{candidate_id}.json"
    try:
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        board = json.loads((base_dir / "leaderboard.json").read_text(encoding="utf-8"))
        candidate = next(item for item in board.get("leaderboard", []) if item["id"] == candidate_id)
        params = candidate["params"]
        period = board["period"]
        start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
        bars = load_history(ROOT, start, end)
        features = FeatureEngine().transform(bars)
        selected = pd.to_datetime(features.datetime_utc, utc=True).between(start, end)
        data = features.loc[selected].copy().reset_index(drop=True)
        signals = structure_signals(bars, **{key: params[key] for key in (
            "reversal_atr", "ema_span", "touch_atr", "touch_lookback", "breakout_lookback", "swing_buffer_atr")})
        data["signal"] = signals.loc[selected, "signal"].to_numpy()
        config = BacktestConfig(stop_loss_price=None, take_profit_price=None, max_holding_minutes=None,
            max_daily_trades=int(params["max_daily_trades"]), cooldown_minutes=10,
            atr_stop_multiple=float(params["atr_stop_multiple"]), atr_break_even_r=float(params["atr_break_even_r"]),
            atr_trailing_multiple=float(params["atr_trailing_multiple"]))
        outcome = Backtester(config).run(data)
        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            result = frame.copy()
            for column in result.columns:
                if "time" in str(column):
                    values = pd.to_datetime(result[column], utc=True, errors="coerce")
                    if values.notna().any(): result[column] = values.map(lambda value: value.isoformat() if pd.notna(value) else None)
            return _json_safe(result.to_dict("records"))
        candles = [{"time": int(pd.Timestamp(row.datetime_utc).timestamp()), "open": float(row.mid_open),
                    "high": float(row.mid_high), "low": float(row.mid_low), "close": float(row.mid_close),
                    "volume": float(getattr(row, "tick_volume", 0.0))} for row in data.itertuples(index=False)]
        payload = {"candidate": candidate, "candles": candles, "trades": records(outcome.trades),
                   "equity": records(outcome.equity_curve[["datetime_utc", "equity"]])}
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_name(f"{cached.stem}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cached)
        return payload
    except StopIteration as exc:
        raise HTTPException(status_code=404, detail="Candidato non più nella top 100") from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Replay candidato fallito: {exc}") from exc


@app.get("/api/backtests/catalog")
def backtest_catalog() -> dict[str, Any]:
    return {"strategies": [{"id": key, "name": value} for key, value in strategy_catalog().items()],
            "defaults": {"starting_capital": 100000, "position_size_units": 1.0, "leverage": 20,
                         "commission_per_unit_per_side": 0.0, "slippage_price_per_side": 0.05,
                         "stop_loss_price": 5.0, "take_profit_price": 10.0, "max_holding_minutes": 30,
                         "buy_threshold": .55, "sell_threshold": .45}}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list): return [_json_safe(item) for item in value]
    return value


def _run_backtest_job(job_id: str, request: dict[str, Any]) -> None:
    def progress(value: float) -> None:
        with _backtest_lock:
            _backtest_jobs[job_id]["progress"] = round(float(value) * 100, 1)
    try:
        settings = dict(request.get("settings") or {})
        allowed = set(BacktestConfig.__dataclass_fields__)
        config = BacktestConfig(**{key: settings[key] for key in settings if key in allowed})
        result = run_lab(ROOT, start=str(request["start"]), end=str(request["end"]),
                         strategies=list(request.get("strategies") or []), config=config,
                         buy_threshold=float(settings.get("buy_threshold", .55)),
                         sell_threshold=float(settings.get("sell_threshold", .45)), progress=progress)
        result = _json_safe(result)
        target = _backtest_results / job_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        with _backtest_lock:
            _backtest_jobs[job_id].update({"status": "completed", "progress": 100.0, "result": result,
                                           "finished_at": datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        with _backtest_lock:
            _backtest_jobs[job_id].update({"status": "failed", "error": str(exc) or repr(exc),
                                           "finished_at": datetime.now(timezone.utc).isoformat()})


@app.post("/api/backtests")
def start_backtest(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
    with _backtest_lock:
        if any(job.get("status") == "running" for job in _backtest_jobs.values()):
            raise HTTPException(status_code=409, detail="C'è già un backtest in corso: attendi che termini.")
        job_id = f"bt_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:6]}"
        _backtest_jobs[job_id] = {"id": job_id, "status": "running", "progress": 0.0,
                                  "started_at": datetime.now(timezone.utc).isoformat()}
    Thread(target=_run_backtest_job, args=(job_id, request), daemon=True).start()
    return {"id": job_id, "status": "running"}


@app.get("/api/backtests/{job_id}")
def backtest_status(job_id: str) -> dict[str, Any]:
    with _backtest_lock:
        job = _backtest_jobs.get(job_id)
        if job is None:
            persisted = _backtest_results / job_id / "result.json"
            if persisted.exists():
                return {"id": job_id, "status": "completed", "progress": 100.0,
                        "result": json.loads(persisted.read_text(encoding="utf-8"))}
            raise HTTPException(status_code=404, detail="Backtest non trovato")
        return _json_safe(dict(job))


@app.get("/api/indicators")
def indicators(strategy: str | None = None, window: str = "50") -> dict[str, Any]:
    return snapshot(strategy=strategy, source="indicators", window=window)


@app.get("/api/snapshot")
def snapshot(strategy: str | None = None, source: str = "model", window: str = "50") -> dict[str, Any]:
    """Return a compact read-only view of the active headless paper session."""
    try:
        run_id, accounts = _load_indicator_states() if source == "indicators" else _load_ledger_states()
        account_names = list(accounts)
        selected = strategy if strategy in accounts else account_names[0]
        account = accounts[selected]
        allowed_windows = {"50": 50, "250": 250, "1000": 1000, "5000": 5000}
        if window == "all":
            started = pd.to_datetime(account.get("experiment_started_at"), utc=True, errors="coerce")
            # Avoid making a local browser unusable after a run left open for weeks.
            minutes = int((pd.Timestamp.now(tz="UTC") - started).total_seconds() // 60) + 20 if pd.notna(started) else 5000
            bar_count = max(50, min(minutes, 10_000))
        else:
            bar_count = allowed_windows.get(window, 50)
        # Fetch enough warm-up history for a faithful EMA100 even when the user
        # chooses a compact 50-candle display window.
        tick, history_bars = _market_payload(max(bar_count, 250))
        history_bars["ema20"] = history_bars["mid_close"].ewm(span=20, adjust=False).mean()
        history_bars["ema100"] = history_bars["mid_close"].ewm(span=100, adjust=False).mean()
        bars = history_bars.tail(bar_count).copy()
        bars["datetime_utc"] = pd.to_datetime(bars["datetime_utc"], utc=True)
        candles = [
            {"time": int(row.datetime_utc.timestamp()), "open": float(row.mid_open), "high": float(row.mid_high),
             "low": float(row.mid_low), "close": float(row.mid_close), "volume": float(row.tick_volume)}
            for row in bars.itertuples(index=False)
        ]
        selected_id = str(account.get("config", {}).get("strategy_id", ""))
        overlays = []
        if selected_id in {"I04", "I21", "I31", "I32", "I33"}:
            overlays = [
                {"name": "EMA 20", "color": "#fb923c", "data": [
                    {"time": int(row.datetime_utc.timestamp()), "value": float(row.ema20)} for row in bars.itertuples(index=False)
                ]},
                {"name": "EMA 100", "color": "#4ade80", "data": [
                    {"time": int(row.datetime_utc.timestamp()), "value": float(row.ema100)} for row in bars.itertuples(index=False)
                ]},
            ]
        chart_start, chart_end = candles[0]["time"], candles[-1]["time"]
        events = pd.DataFrame(account.get("events", []))
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
            quantity = _number(position.get("quantity")) or 0.0
            raw_exit = _number(tick["bid"] if side == "LONG" else tick["ask"]) or 0.0
            raw_entry = _number(position.get("raw_entry_price")) or 0.0
            direction = 1.0 if side == "LONG" else -1.0
            config = account.get("config", {})
            exit_cost = quantity * (
                _number(config.get("slippage_price_per_side")) or 0.0
                + (_number(config.get("commission_per_unit_per_side")) or 0.0)
            )
            open_pnl = direction * (raw_exit - raw_entry) * quantity - (_number(position.get("entry_costs")) or 0.0) - exit_cost
            pnl_text = f"{open_pnl:+.2f} USD"
            levels.append({"price": float(position["entry_price"]), "color": entry_color,
                           "line_style": 0, "title": f"{side} #{position['trade_id']} | {pnl_text}"})
            if position.get("stop_loss") is not None:
                levels.append({"price": float(position["stop_loss"]), "color": "#ef4444",
                               "line_style": 2, "title": f"SL #{position['trade_id']}"})
            if position.get("take_profit") is not None:
                levels.append({"price": float(position["take_profit"]), "color": "#22c55e",
                               "line_style": 2, "title": f"TP #{position['trade_id']}"})
        cards = []
        for name, state in accounts.items():
            # Older records and some non-triggering setups can carry JSON NaN;
            # normalize before aggregating so a single missing power never
            # takes the whole dashboard offline.
            trade_powers = []
            for row in [*(state.get("trades") or []), *(state.get("positions") or [])]:
                power = _number(row.get("power"))
                if power is not None:
                    trade_powers.append(power)
            cards.append({
                "name": name, "strategy": state.get("config", {}).get("strategy_id", "?"),
                "total_pnl": _number(state["realized_pnl"]) + _number(state["unrealized_pnl"]),
                "realized_pnl": _number(state["realized_pnl"]), "unrealized_pnl": _number(state["unrealized_pnl"]),
                "open_positions": len(state.get("positions") or []), "trades": len(state.get("trades") or []),
                "max_drawdown": _number(state["max_drawdown"]), "last_signal": state.get("last_signal"),
                "average_power": (sum(trade_powers) / len(trade_powers)) if trade_powers else None,
                "powered_trades": len(trade_powers),
            })
        return {
            "tick": tick,
            "candles": candles, "markers": markers, "levels": levels, "overlays": overlays,
            "selected": selected, "strategies": account_names,
            "cards": cards, "run_id": run_id, "inference": _latest_inference() if source == "model" else None,
            "paper_status": _paper_status(), "chart_window": window,
        }
    except Exception as exc:  # Browser gets the actionable failure, server stays alive.
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=str(exc) or repr(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
