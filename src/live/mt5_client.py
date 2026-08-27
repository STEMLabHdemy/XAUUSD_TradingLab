from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from .clock import infer_server_utc_offset, normalize_server_epoch


class MT5ConnectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConnectionStatus:
    connected: bool
    terminal_path: str
    server: str | None
    account_demo: bool
    trade_allowed: bool
    symbol: str | None
    server_utc_offset_seconds: int | None


@dataclass(frozen=True)
class MarketTick:
    datetime_utc: pd.Timestamp
    raw_server_datetime: pd.Timestamp
    bid: float
    ask: float
    spread: float
    source_symbol: str


class MT5Client:
    """Minimal read-only MT5 adapter; intentionally exposes no order methods."""

    TIMEFRAME_NAMES = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30"}

    def __init__(
        self,
        terminal_path: Path | str,
        symbol_candidates: tuple[str, ...],
        fallback_offset_seconds: int | None = None,
    ):
        self.terminal_path = str(Path(terminal_path))
        self.symbol_candidates = symbol_candidates
        self._mt5: Any | None = None
        self._symbol: str | None = None
        self._offset_seconds: int | None = fallback_offset_seconds
        self._lock = RLock()

    def _module(self) -> Any:
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
            except ImportError as exc:
                raise MT5ConnectionError("MetaTrader5 is not installed; run python -m pip install -r requirements.txt") from exc
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> ConnectionStatus:
        with self._lock:
            mt5 = self._module()
            terminal = mt5.terminal_info()
            if terminal is None or not terminal.connected:
                if not mt5.initialize(path=self.terminal_path):
                    raise MT5ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")
            self._symbol = self._resolve_symbol()
            tick = mt5.symbol_info_tick(self._symbol)
            if tick is None:
                raise MT5ConnectionError(f"No tick available for {self._symbol}: {mt5.last_error()}")
            self._update_offset(float(tick.time_msc) / 1000)
            return self.status()

    def _resolve_symbol(self) -> str:
        mt5 = self._module()
        for candidate in self.symbol_candidates:
            if mt5.symbol_info(candidate) is not None and mt5.symbol_select(candidate, True):
                return candidate
        symbols = mt5.symbols_get() or ()
        matches = [symbol.name for symbol in symbols if "XAUUSD" in symbol.name.upper() or symbol.name.upper() == "GOLD"]
        for candidate in matches:
            if mt5.symbol_select(candidate, True):
                return candidate
        raise MT5ConnectionError("No broker XAUUSD/GOLD symbol was found")

    def _update_offset(self, raw_epoch_seconds: float) -> None:
        try:
            self._offset_seconds = infer_server_utc_offset(raw_epoch_seconds)
        except ValueError:
            if self._offset_seconds is None:
                raise MT5ConnectionError("Cannot infer broker server timezone from a fresh tick")

    @property
    def symbol(self) -> str:
        if self._symbol is None:
            self.connect()
        assert self._symbol is not None
        return self._symbol

    @property
    def server_utc_offset_seconds(self) -> int:
        if self._offset_seconds is None:
            self.connect()
        assert self._offset_seconds is not None
        return self._offset_seconds

    def status(self) -> ConnectionStatus:
        mt5 = self._module()
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        connected = bool(terminal and terminal.connected)
        demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
        return ConnectionStatus(
            connected=connected,
            terminal_path=self.terminal_path,
            server=getattr(account, "server", None),
            account_demo=bool(account and account.trade_mode == demo_mode),
            trade_allowed=bool(terminal and terminal.trade_allowed),
            symbol=self._symbol,
            server_utc_offset_seconds=self._offset_seconds,
        )

    def latest_tick(self) -> MarketTick:
        with self._lock:
            self.connect()
            mt5 = self._module()
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                raise MT5ConnectionError(f"No tick available for {self.symbol}: {mt5.last_error()}")
            raw_seconds = float(tick.time_msc) / 1000
            self._update_offset(raw_seconds)
            normalized = pd.Timestamp(datetime.fromtimestamp(raw_seconds - self.server_utc_offset_seconds, tz=timezone.utc))
            raw = pd.Timestamp(datetime.fromtimestamp(raw_seconds, tz=timezone.utc))
            return MarketTick(normalized, raw, float(tick.bid), float(tick.ask), float(tick.ask - tick.bid), self.symbol)

    def bars(self, timeframe: str = "M1", count: int = 500) -> pd.DataFrame:
        with self._lock:
            self.connect()
            mt5 = self._module()
            normalized_timeframe = timeframe.upper()
            attribute = self.TIMEFRAME_NAMES.get(normalized_timeframe)
            if attribute is None:
                raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
            rates = mt5.copy_rates_from_pos(self.symbol, getattr(mt5, attribute), 0, int(count))
            if rates is None or len(rates) == 0:
                raise MT5ConnectionError(f"No {normalized_timeframe} bars for {self.symbol}: {mt5.last_error()}")
            info = mt5.symbol_info(self.symbol)
            if info is None:
                raise MT5ConnectionError(f"No symbol metadata for {self.symbol}")
            frame = pd.DataFrame(rates).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
            frame["raw_server_datetime"] = pd.to_datetime(frame.time, unit="s", utc=True)
            frame["datetime_utc"] = normalize_server_epoch(frame.time, self.server_utc_offset_seconds)
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            frame["timestamp"] = (frame.datetime_utc - epoch) // pd.Timedelta(milliseconds=1)
            spread_price = frame.spread.astype(float) * float(info.point)
            for name in ("open", "high", "low", "close"):
                bid = frame[name].astype(float)
                frame[f"{name}_bid"] = bid
                frame[f"{name}_ask"] = bid + spread_price
                frame[f"mid_{name}"] = bid + spread_price / 2
            frame["spread_close"] = spread_price
            frame["spread_points"] = frame.spread.astype(int)
            frame["source"] = "MT5"
            frame["source_symbol"] = self.symbol
            frame["server_utc_offset_seconds"] = self.server_utc_offset_seconds
            frame["is_complete"] = True
            frame.loc[frame.index[-1], "is_complete"] = False
            columns = [
                "timestamp", "datetime_utc", "raw_server_datetime",
                "open_bid", "high_bid", "low_bid", "close_bid",
                "open_ask", "high_ask", "low_ask", "close_ask",
                "mid_open", "mid_high", "mid_low", "mid_close",
                "spread_close", "spread_points", "tick_volume", "real_volume",
                "source", "source_symbol", "server_utc_offset_seconds", "is_complete",
            ]
            return frame[columns]

    def point(self) -> float:
        with self._lock:
            self.connect()
            info = self._module().symbol_info(self.symbol)
            if info is None:
                raise MT5ConnectionError(f"No symbol metadata for {self.symbol}")
            return float(info.point)

    def shutdown(self) -> None:
        with self._lock:
            if self._mt5 is not None:
                self._mt5.shutdown()
