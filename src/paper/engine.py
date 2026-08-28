from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from src.live.inference import LiveInference, LiveInferenceEngine
from src.live.mt5_client import MarketTick


@dataclass(frozen=True)
class PaperConfig:
    starting_capital: float = 100_000.0
    currency: str = "USD"
    leverage: float = 20.0
    risk_per_trade_pct: float = 1.0
    position_size_units: float = 1.0
    commission_per_unit_per_side: float = 0.0
    slippage_price_per_side: float = 0.05
    max_daily_trades: int = 20
    max_allowed_spread: float = 5.0
    stop_loss_price: float | None = 5.0
    take_profit_price: float | None = 10.0
    max_daily_loss: float = 2_000.0
    buy_threshold: float = .68
    sell_threshold: float = .32
    persistence: int = 2
    cooldown_minutes: int = 3
    probability_exit_threshold: float = .50

    def validate(self) -> None:
        if self.starting_capital <= 0 or self.position_size_units <= 0 or self.leverage <= 0:
            raise ValueError("Capital, position size and leverage must be positive")
        if not 0 <= self.sell_threshold < self.buy_threshold <= 1:
            raise ValueError("Require 0 <= sell threshold < buy threshold <= 1")
        if self.persistence < 1 or self.max_daily_trades < 1:
            raise ValueError("Persistence and max daily trades must be at least one")
        if not 0 < self.risk_per_trade_pct <= 100:
            raise ValueError("Risk per trade must be in (0, 100]")
        if min(self.commission_per_unit_per_side, self.slippage_price_per_side, self.max_allowed_spread, self.max_daily_loss) < 0:
            raise ValueError("Costs and limits cannot be negative")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class PaperAccount:
    """Persistent, one-position virtual account. It has no broker-order interface."""

    def __init__(self, account_id: str, model: str, config: PaperConfig, directory: Path | str):
        config.validate()
        self.account_id, self.model, self.config = account_id, model, config
        self.directory = Path(directory) / account_id
        self.path = self.directory / "state.json"
        self._lock = RLock()
        self.state = self._load_or_new()

    def _new_state(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id, "model": self.model, "config": asdict(self.config),
            "config_fingerprint": self.config.fingerprint, "running": False,
            "balance": self.config.starting_capital, "realized_pnl": 0.0,
            "equity": self.config.starting_capital, "unrealized_pnl": 0.0,
            "used_margin": 0.0, "free_margin": self.config.starting_capital,
            "exposure": 0.0, "peak_equity": self.config.starting_capital, "max_drawdown": 0.0,
            "position": None, "trades": [], "events": [], "equity_history": [],
            "pending_direction": None, "persistence_count": 0, "last_exit_time": None,
            "last_processed_bar": None, "last_signal": "NO_TRADE", "last_reason": "paper stopped",
            "next_trade_id": 1,
        }

    def _load_or_new(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new_state()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("config_fingerprint") != self.config.fingerprint:
            # Existing experiments keep their original immutable configuration.
            self.config = PaperConfig(**state["config"])
        return state

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def start(self) -> None:
        with self._lock:
            self.state["running"] = True
            self.state["last_reason"] = "waiting for next completed M1 candle"
            self._save()

    def stop(self) -> None:
        with self._lock:
            self.state["running"] = False
            self.state["last_reason"] = "paper stopped by user"
            self._save()

    def reset(self) -> None:
        with self._lock:
            self.state = self._new_state()
            self._save()

    def _today_stats(self, timestamp: pd.Timestamp) -> tuple[int, float]:
        day = timestamp.strftime("%Y-%m-%d")
        trades = [row for row in self.state["trades"] if str(row["exit_time"]).startswith(day)]
        entries = [row for row in self.state["events"] if row["event"] in {"BUY", "SELL"} and str(row["timestamp"]).startswith(day)]
        return len(entries), sum(float(row["net_pnl"]) for row in trades)

    def _mark(self, tick: MarketTick, save: bool = True) -> None:
        position = self.state["position"]
        unrealized = used = exposure = 0.0
        if position:
            quantity = float(position["quantity"])
            raw_exit = tick.bid if position["side"] == "LONG" else tick.ask
            direction = 1 if position["side"] == "LONG" else -1
            gross = direction * (raw_exit - float(position["raw_entry_price"])) * quantity
            future_cost = self.config.slippage_price_per_side * quantity + self.config.commission_per_unit_per_side * quantity
            unrealized = gross - float(position["entry_costs"]) - future_cost
            exposure = raw_exit * quantity
            used = exposure / self.config.leverage
        equity = float(self.state["balance"]) + unrealized
        self.state.update(unrealized_pnl=unrealized, equity=equity, used_margin=used,
                          free_margin=equity - used, exposure=exposure)
        self.state["peak_equity"] = max(float(self.state["peak_equity"]), equity)
        peak = float(self.state["peak_equity"])
        drawdown = (peak - equity) / peak if peak else 0.0
        self.state["max_drawdown"] = max(float(self.state["max_drawdown"]), drawdown)
        if save:
            self._save()

    def _open(self, side: str, tick: MarketTick, inference: LiveInference, reason: str) -> None:
        quantity, slip = self.config.position_size_units, self.config.slippage_price_per_side
        raw = tick.ask if side == "LONG" else tick.bid
        price = raw + slip if side == "LONG" else raw - slip
        stop = price - self.config.stop_loss_price if side == "LONG" and self.config.stop_loss_price else None
        stop = price + self.config.stop_loss_price if side == "SHORT" and self.config.stop_loss_price else stop
        take = price + self.config.take_profit_price if side == "LONG" and self.config.take_profit_price else None
        take = price - self.config.take_profit_price if side == "SHORT" and self.config.take_profit_price else take
        trade_id = int(self.state["next_trade_id"])
        self.state["next_trade_id"] = trade_id + 1
        position = {
            "trade_id": trade_id, "side": side, "entry_time": tick.datetime_utc.isoformat(),
            "raw_entry_price": raw, "entry_price": price, "quantity": quantity,
            "confidence": inference.probability_up, "spread": tick.spread, "stop_loss": stop,
            "take_profit": take, "entry_costs": (slip + self.config.commission_per_unit_per_side) * quantity,
            "model": self.model, "regime": "not_available", "expected_return": None, "reason": reason,
        }
        self.state["position"] = position
        self.state["events"].append({**position, "event": "BUY" if side == "LONG" else "SELL", "timestamp": position["entry_time"], "price": price})

    def _close(self, tick: MarketTick, reason: str) -> None:
        position = self.state["position"]
        if not position:
            return
        quantity, slip = float(position["quantity"]), self.config.slippage_price_per_side
        raw = tick.bid if position["side"] == "LONG" else tick.ask
        exit_price = raw - slip if position["side"] == "LONG" else raw + slip
        direction = 1 if position["side"] == "LONG" else -1
        gross = direction * (raw - float(position["raw_entry_price"])) * quantity
        costs = float(position["entry_costs"]) + (slip + self.config.commission_per_unit_per_side) * quantity
        net = gross - costs
        self.state["balance"] = float(self.state["balance"]) + net
        self.state["realized_pnl"] = float(self.state["realized_pnl"]) + net
        row = {
            "trade_id": position["trade_id"], "side": position["side"], "entry_time": position["entry_time"],
            "entry_price": position["entry_price"], "raw_entry_price": position["raw_entry_price"],
            "exit_time": tick.datetime_utc.isoformat(), "exit_price": exit_price, "raw_exit_price": raw,
            "quantity": quantity, "confidence": position["confidence"], "spread": position["spread"],
            "stop_loss": position["stop_loss"], "take_profit": position["take_profit"], "exit_reason": reason,
            "gross_pnl": gross, "costs": costs, "net_pnl": net, "model": self.model,
        }
        self.state["trades"].append(row)
        self.state["events"].append({**row, "event": "EXIT", "timestamp": row["exit_time"], "price": exit_price})
        self.state["position"] = None
        self.state["last_exit_time"] = tick.datetime_utc.isoformat()
        self.state["pending_direction"] = None
        self.state["persistence_count"] = 0

    def process(self, tick: MarketTick, inference: LiveInference) -> None:
        with self._lock:
            self._mark(tick, save=False)
            if not self.state["running"]:
                self._save()
                return
            position = self.state["position"]
            protective_exit = False
            if position:
                if position["side"] == "LONG" and position["stop_loss"] is not None and tick.bid <= position["stop_loss"]:
                    self._close(tick, "stop_loss")
                    protective_exit = True
                elif position["side"] == "LONG" and position["take_profit"] is not None and tick.bid >= position["take_profit"]:
                    self._close(tick, "take_profit")
                    protective_exit = True
                elif position["side"] == "SHORT" and position["stop_loss"] is not None and tick.ask >= position["stop_loss"]:
                    self._close(tick, "stop_loss")
                    protective_exit = True
                elif position["side"] == "SHORT" and position["take_profit"] is not None and tick.ask <= position["take_profit"]:
                    self._close(tick, "take_profit")
                    protective_exit = True

            bar_time = inference.inference_time_utc
            if not inference.available or bar_time is None or self.state["last_processed_bar"] == bar_time.isoformat():
                self._mark(tick)
                return
            self.state["last_processed_bar"] = bar_time.isoformat()
            if protective_exit:
                self.state["last_signal"], self.state["last_reason"] = "EXIT", "protective exit"
                self._mark(tick)
                return
            score = float(inference.probability_up)
            position = self.state["position"]
            signal, reason = "NO_TRADE", "inside no-trade zone"
            if position:
                exit_long = position["side"] == "LONG" and score < self.config.probability_exit_threshold
                exit_short = position["side"] == "SHORT" and score > self.config.probability_exit_threshold
                if exit_long or exit_short:
                    signal, reason = "EXIT", "probability reversal"
                    self._close(tick, reason)
                else:
                    signal, reason = "HOLD", "open position remains supported"
            else:
                candidate = "BUY" if score >= self.config.buy_threshold else "SELL" if score <= self.config.sell_threshold else None
                count, daily_pnl = self._today_stats(tick.datetime_utc)
                blockers = []
                if tick.spread > self.config.max_allowed_spread: blockers.append("spread exceeds maximum")
                if count >= self.config.max_daily_trades: blockers.append("daily trade limit")
                if self.config.max_daily_loss > 0 and daily_pnl <= -self.config.max_daily_loss: blockers.append("daily loss limit")
                risk_amount = float(self.state["equity"]) * self.config.risk_per_trade_pct / 100
                if self.config.stop_loss_price and self.config.stop_loss_price * self.config.position_size_units > risk_amount:
                    blockers.append("risk per trade limit")
                required_margin = tick.ask * self.config.position_size_units / self.config.leverage
                if required_margin > float(self.state["free_margin"]): blockers.append("insufficient virtual margin")
                last_exit = pd.Timestamp(self.state["last_exit_time"]) if self.state["last_exit_time"] else None
                if last_exit is not None and tick.datetime_utc < last_exit + pd.Timedelta(minutes=self.config.cooldown_minutes): blockers.append("cooldown active")
                if candidate and not blockers:
                    if self.state["pending_direction"] == candidate:
                        self.state["persistence_count"] += 1
                    else:
                        self.state["pending_direction"], self.state["persistence_count"] = candidate, 1
                    if self.state["persistence_count"] >= self.config.persistence:
                        signal, reason = candidate, f"threshold and {self.config.persistence} confirmations passed"
                        self._open("LONG" if candidate == "BUY" else "SHORT", tick, inference, reason)
                        self.state["pending_direction"], self.state["persistence_count"] = None, 0
                    else:
                        reason = f"confirmation {self.state['persistence_count']}/{self.config.persistence}"
                else:
                    self.state["pending_direction"], self.state["persistence_count"] = None, 0
                    if blockers: reason = "; ".join(blockers)
            self.state["last_signal"], self.state["last_reason"] = signal, reason
            self._mark(tick, save=False)
            self.state["equity_history"].append({"timestamp": tick.datetime_utc.isoformat(), "equity": self.state["equity"], "balance": self.state["balance"]})
            self.state["equity_history"] = self.state["equity_history"][-10_000:]
            self._save()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.state))

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshot()["trades"])

    def events_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshot()["events"])


class PaperRuntime:
    def __init__(self, root: Path | str, config: PaperConfig):
        self.root, self.config = Path(root), config
        manifest = self.root / "models/baseline_manifest_provisional.json"
        definitions = {
            "LightGBM": "lightgbm_up_5m_sigmoid_provisional.joblib",
            "Logistic Regression": "logistic_regression_up_5m_sigmoid_provisional.joblib",
            "XGBoost": "xgboost_up_5m_sigmoid_provisional.joblib",
        }
        self.engines, self.accounts = {}, {}
        for label, filename in definitions.items():
            path = self.root / "models" / filename
            if path.exists():
                self.engines[label] = LiveInferenceEngine(path, manifest, config.buy_threshold, config.sell_threshold)
                key = label.lower().replace(" ", "_")
                self.accounts[label] = PaperAccount(key, label, config, self.root / "data/live/paper")
        self._last_bar: int | None = None
        self._inferences: dict[str, LiveInference] = {}
        self._lock = RLock()

    def process(self, tick: MarketTick, completed_m1: pd.DataFrame) -> None:
        with self._lock:
            latest = int(completed_m1.timestamp.iloc[-1]) if not completed_m1.empty else None
            if latest != self._last_bar:
                self._inferences = {name: engine.predict(completed_m1) for name, engine in self.engines.items()}
                self._last_bar = latest
            for name, account in self.accounts.items():
                inference = self._inferences.get(name)
                if inference is not None:
                    account.process(tick, inference)

    def comparison(self) -> pd.DataFrame:
        rows = []
        for name, account in self.accounts.items():
            state = account.snapshot()
            total_pnl = float(state["realized_pnl"]) + float(state["unrealized_pnl"])
            rows.append({
                "model": name,
                "status": "RUNNING" if state["running"] else "STOPPED",
                "balance": state["balance"],
                "equity": state["equity"],
                "realized_pnl": state["realized_pnl"],
                "unrealized_pnl": state["unrealized_pnl"],
                "total_pnl": total_pnl,
                "return_pct": total_pnl / account.config.starting_capital,
                "free_margin": state["free_margin"],
                "trades": len(state["trades"]),
                "position": state["position"]["side"] if state["position"] else "FLAT",
                "signal": state["last_signal"],
                "max_drawdown": state["max_drawdown"],
            })
        return pd.DataFrame(rows)

    def reconfigure_and_reset(self, config: PaperConfig) -> None:
        """Apply one fair configuration to every model and start fresh experiments."""
        config.validate()
        with self._lock:
            self.config = config
            for name, old in list(self.accounts.items()):
                account = PaperAccount(old.account_id, name, config, self.root / "data/live/paper")
                account.config = config
                account.state = account._new_state()
                account._save()
                self.accounts[name] = account
            for engine in self.engines.values():
                engine.buy_threshold, engine.sell_threshold = config.buy_threshold, config.sell_threshold
