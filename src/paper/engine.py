from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import time
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from src.live.inference import CostAwareLiveInferenceEngine, LiveInference, LiveInferenceEngine
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
    max_daily_trades: int | None = 20
    max_allowed_spread: float = 5.0
    stop_loss_price: float | None = 5.0
    take_profit_price: float | None = 10.0
    max_daily_loss: float = 2_000.0
    buy_threshold: float = .68
    sell_threshold: float = .32
    persistence: int = 2
    cooldown_minutes: int = 3
    probability_exit_threshold: float = .50
    entry_mode: str = "controlled"
    session_flatten_enabled: bool = True
    session_flatten_local_time: str = "22:55"
    session_timezone: str = "Europe/Rome"
    strategy_id: str = "A"
    anti_burst_enabled: bool = False
    smart_short_enabled: bool = False
    max_open_positions_override: int | None = None
    short_protect_break_even_pnl: float = 2.0
    short_protect_lock_trigger_pnl: float = 4.0
    short_protect_lock_pnl: float = 2.0
    short_protect_trailing_trigger_pnl: float = 6.0
    short_protect_trailing_distance: float = 2.0
    short_reversal_confirmations: int = 2

    def validate(self) -> None:
        if self.starting_capital <= 0 or self.position_size_units <= 0 or self.leverage <= 0:
            raise ValueError("Capital, position size and leverage must be positive")
        if not 0 <= self.sell_threshold < self.buy_threshold <= 1:
            raise ValueError("Require 0 <= sell threshold < buy threshold <= 1")
        if self.persistence < 1 or (self.max_daily_trades is not None and self.max_daily_trades < 1):
            raise ValueError("Persistence and max daily trades must be at least one when enabled")
        if not 0 < self.risk_per_trade_pct <= 100:
            raise ValueError("Risk per trade must be in (0, 100]")
        if min(self.commission_per_unit_per_side, self.slippage_price_per_side, self.max_allowed_spread, self.max_daily_loss) < 0:
            raise ValueError("Costs and limits cannot be negative")
        if self.entry_mode not in {"controlled", "intermediate", "burst"}:
            raise ValueError("Entry mode must be controlled, intermediate or burst")
        if self.strategy_id not in {"A", "B", "C", "D"}:
            raise ValueError("Strategy id must be A, B, C or D")
        if self.max_open_positions_override is not None and self.max_open_positions_override < 1:
            raise ValueError("Maximum open positions override must be positive")
        if self.short_reversal_confirmations < 2:
            raise ValueError("Smart short requires at least two reversal confirmations")
        if min(
            self.short_protect_break_even_pnl, self.short_protect_lock_trigger_pnl,
            self.short_protect_lock_pnl, self.short_protect_trailing_trigger_pnl,
            self.short_protect_trailing_distance,
        ) < 0:
            raise ValueError("Smart-short protection values cannot be negative")
        try:
            time.fromisoformat(self.session_flatten_local_time)
        except (TypeError, ValueError) as exc:
            raise ValueError("Session flatten time must be HH:MM or HH:MM:SS") from exc

    @property
    def entry_rules(self) -> tuple[int, int, int]:
        """Return max open legs, confirmations and minimum minutes between entries."""
        if self.entry_mode == "intermediate":
            default = 3, self.persistence, 3
        elif self.entry_mode == "burst":
            default = 10, 1, 0
        else:
            default = 1, self.persistence, self.cooldown_minutes
        return self.max_open_positions_override or default[0], default[1], default[2]

    @property
    def exit_cooldown_minutes(self) -> int:
        if self.entry_mode == "burst":
            return 0
        return self.cooldown_minutes

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class PaperAccount:
    """Persistent virtual account with no broker-order interface."""

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
            "position": None, "positions": [], "trades": [], "events": [], "equity_history": [],
            "pending_direction": None, "persistence_count": 0, "last_exit_time": None,
            "last_entry_time": None,
            "last_processed_bar": None, "last_signal": "NO_TRADE", "last_reason": "paper stopped",
            "last_signal_id": None, "short_reversal_count": 0, "short_protect_mode": False,
            "reentry_gates": {"LONG": {"blocked": False, "non_directional_bars": 0, "rearmed": False},
                              "SHORT": {"blocked": False, "non_directional_bars": 0, "rearmed": False}},
            "next_trade_id": 1,
        }

    def _load_or_new(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new_state()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if "positions" not in state:
            state["positions"] = [state["position"]] if state.get("position") else []
        state.setdefault("last_entry_time", None)
        state.setdefault("last_signal_id", None)
        state.setdefault("short_reversal_count", 0)
        state.setdefault("short_protect_mode", False)
        state.setdefault("reentry_gates", {"LONG": {"blocked": False, "non_directional_bars": 0, "rearmed": False},
                                             "SHORT": {"blocked": False, "non_directional_bars": 0, "rearmed": False}})
        self._sync_position_alias(state)
        if state.get("config_fingerprint") != self.config.fingerprint:
            # Existing experiments keep their original immutable configuration.
            self.config = PaperConfig(**state["config"])
        return state

    @staticmethod
    def _sync_position_alias(state: dict[str, Any]) -> None:
        """Keep the legacy singular field readable while positions is canonical."""
        positions = state.get("positions", [])
        state["position"] = positions[0] if positions else None

    def _positions(self) -> list[dict[str, Any]]:
        positions = self.state.setdefault("positions", [])
        self._sync_position_alias(self.state)
        return positions

    def set_entry_mode(self, mode: str) -> None:
        with self._lock:
            updated = replace(self.config, entry_mode=mode)
            updated.validate()
            self.config = updated
            self.state["config"] = asdict(updated)
            self.state["config_fingerprint"] = updated.fingerprint
            self.state["last_reason"] = f"entry mode changed to {mode}"
            self._save()

    def update_config_preserving_history(self, config: PaperConfig) -> None:
        """Apply risk/execution settings without closing positions or erasing the ledger."""
        config.validate()
        with self._lock:
            self.config = config
            self.state["config"] = asdict(config)
            self.state["config_fingerprint"] = config.fingerprint
            self.state["last_reason"] = "configurazione aggiornata; storico e posizioni preservati"
            self._save()

    def _save(self) -> None:
        self._sync_position_alias(self.state)
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

    def update_protection(
        self,
        tick: MarketTick,
        stop_loss: float | None,
        take_profit: float | None,
        trade_id: int | None = None,
    ) -> None:
        """Update virtual SL/TP after validating them against the executable price."""
        with self._lock:
            positions = self._positions()
            position = next(
                (row for row in positions if trade_id is None or int(row["trade_id"]) == int(trade_id)),
                None,
            )
            if not position:
                raise ValueError("Non c'è una posizione aperta da modificare")
            for label, value in (("Stop loss", stop_loss), ("Take profit", take_profit)):
                if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                    raise ValueError(f"{label} deve essere un prezzo positivo")
            side = str(position["side"])
            executable = float(tick.bid if side == "LONG" else tick.ask)
            if side == "LONG":
                if stop_loss is not None and float(stop_loss) >= executable:
                    raise ValueError(f"Per un LONG lo stop deve essere sotto il BID corrente ({executable:.2f})")
                if take_profit is not None and float(take_profit) <= executable:
                    raise ValueError(f"Per un LONG il target deve essere sopra il BID corrente ({executable:.2f})")
            else:
                if stop_loss is not None and float(stop_loss) <= executable:
                    raise ValueError(f"Per uno SHORT lo stop deve essere sopra l'ASK corrente ({executable:.2f})")
                if take_profit is not None and float(take_profit) >= executable:
                    raise ValueError(f"Per uno SHORT il target deve essere sotto l'ASK corrente ({executable:.2f})")
            old_stop, old_take = position.get("stop_loss"), position.get("take_profit")
            position["stop_loss"] = float(stop_loss) if stop_loss is not None else None
            position["take_profit"] = float(take_profit) if take_profit is not None else None
            self.state["events"].append({
                "event": "MODIFY", "timestamp": tick.datetime_utc.isoformat(),
                "trade_id": position["trade_id"], "model": self.model,
                "side": side, "price": executable, "old_stop_loss": old_stop,
                "old_take_profit": old_take, "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"],
            })
            self.state["last_reason"] = "SL/TP modificati manualmente"
            self._mark(tick, save=False)
            self._save()

    def close_manually(self, tick: MarketTick, trade_id: int | None = None) -> None:
        """Close the current virtual position at the executable market side."""
        with self._lock:
            positions = self._positions()
            position = next(
                (row for row in positions if trade_id is None or int(row["trade_id"]) == int(trade_id)),
                None,
            )
            if not position:
                raise ValueError("Non c'è una posizione aperta da chiudere")
            self._close_position(position, tick, "manual_close")
            self.state["last_signal"] = "EXIT"
            self.state["last_reason"] = "chiusura manuale"
            self._mark(tick, save=False)
            self._save()

    def close_all_for_session(self, tick: MarketTick) -> int:
        """Flatten every virtual leg, retaining each one in the trade ledger."""
        with self._lock:
            closed = self._close_all(tick, "session_close")
            self.state["last_signal"] = "EXIT" if closed else "NO_TRADE"
            self.state["last_reason"] = (
                f"session close: {closed} position(s) closed at last available tick"
                if closed else "session close: no open positions"
            )
            self._mark(tick, save=False)
            self._save()
            return closed

    def _session_flatten_active(self, tick: MarketTick) -> bool:
        if not self.config.session_flatten_enabled:
            return False
        local_time = tick.datetime_utc.tz_convert(self.config.session_timezone).time()
        return local_time >= time.fromisoformat(self.config.session_flatten_local_time)

    def _today_stats(self, timestamp: pd.Timestamp) -> tuple[int, float]:
        day = timestamp.strftime("%Y-%m-%d")
        trades = [row for row in self.state["trades"] if str(row["exit_time"]).startswith(day)]
        entries = [row for row in self.state["events"] if row["event"] in {"BUY", "SELL"} and str(row["timestamp"]).startswith(day)]
        return len(entries), sum(float(row["net_pnl"]) for row in trades)

    def _mark(self, tick: MarketTick, save: bool = True) -> None:
        unrealized = used = exposure = 0.0
        for position in self._positions():
            quantity = float(position["quantity"])
            raw_exit = tick.bid if position["side"] == "LONG" else tick.ask
            direction = 1 if position["side"] == "LONG" else -1
            gross = direction * (raw_exit - float(position["raw_entry_price"])) * quantity
            future_cost = self.config.slippage_price_per_side * quantity + self.config.commission_per_unit_per_side * quantity
            unrealized += gross - float(position["entry_costs"]) - future_cost
            leg_exposure = raw_exit * quantity
            exposure += leg_exposure
            used += leg_exposure / self.config.leverage
        equity = float(self.state["balance"]) + unrealized
        self.state.update(unrealized_pnl=unrealized, equity=equity, used_margin=used,
                          free_margin=equity - used, exposure=exposure)
        self.state["peak_equity"] = max(float(self.state["peak_equity"]), equity)
        peak = float(self.state["peak_equity"])
        drawdown = (peak - equity) / peak if peak else 0.0
        self.state["max_drawdown"] = max(float(self.state["max_drawdown"]), drawdown)
        if save:
            self._save()

    def _position_unrealized_pnl(self, position: dict[str, Any], tick: MarketTick) -> float:
        quantity = float(position["quantity"])
        raw_exit = tick.bid if position["side"] == "LONG" else tick.ask
        direction = 1 if position["side"] == "LONG" else -1
        gross = direction * (raw_exit - float(position["raw_entry_price"])) * quantity
        future_cost = self.config.slippage_price_per_side * quantity + self.config.commission_per_unit_per_side * quantity
        return gross - float(position["entry_costs"]) - future_cost

    def _block_direction_after_stop(self, side: str) -> None:
        if not self.config.anti_burst_enabled:
            return
        gate = self.state["reentry_gates"].setdefault(
            side, {"blocked": False, "non_directional_bars": 0, "rearmed": False},
        )
        gate.update(blocked=True, non_directional_bars=0, rearmed=False)

    def _update_reentry_gates(self, candidate: str | None) -> None:
        """Require two non-directional bars and a later return after a stopped-out side."""
        if not self.config.anti_burst_enabled:
            return
        candidate_side = {"BUY": "LONG", "SELL": "SHORT"}.get(candidate)
        for side, gate in self.state["reentry_gates"].items():
            if not gate.get("blocked"):
                continue
            if candidate_side != side:
                gate["non_directional_bars"] = int(gate.get("non_directional_bars", 0)) + 1
                if gate["non_directional_bars"] >= 2:
                    gate["rearmed"] = True
            elif gate.get("rearmed"):
                gate.update(blocked=False, non_directional_bars=0, rearmed=False)

    def _side_is_blocked(self, side: str) -> bool:
        return bool(self.config.anti_burst_enabled and self.state["reentry_gates"].get(side, {}).get("blocked"))

    def _apply_smart_short_protection(self, tick: MarketTick) -> int:
        """Tighten only SHORT stops after the first protected reversal; never widen a stop."""
        if not (self.config.smart_short_enabled and self.state.get("short_protect_mode")):
            return 0
        changed = 0
        for position in self._positions():
            if position["side"] != "SHORT":
                continue
            pnl = self._position_unrealized_pnl(position, tick)
            if pnl < self.config.short_protect_break_even_pnl:
                continue
            quantity = float(position["quantity"])
            total_costs = float(position["entry_costs"]) + (
                self.config.slippage_price_per_side + self.config.commission_per_unit_per_side
            ) * quantity
            protected_pnl = 0.0
            if pnl >= self.config.short_protect_lock_trigger_pnl:
                protected_pnl = self.config.short_protect_lock_pnl
            desired_stop = float(position["raw_entry_price"]) - (protected_pnl + total_costs) / quantity
            if pnl >= self.config.short_protect_trailing_trigger_pnl:
                desired_stop = min(desired_stop, tick.ask + self.config.short_protect_trailing_distance)
            old_stop = position.get("stop_loss")
            if old_stop is None or desired_stop < float(old_stop):
                position["stop_loss"] = desired_stop
                self.state["events"].append({
                    "event": "SMART_PROTECT", "timestamp": tick.datetime_utc.isoformat(),
                    "trade_id": position["trade_id"], "model": self.model, "side": "SHORT",
                    "price": tick.ask, "old_stop_loss": old_stop, "stop_loss": desired_stop,
                    "net_pnl_at_update": pnl, "signal_id": self.state.get("last_signal_id"),
                })
                changed += 1
        return changed

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
            "signal_id": inference.signal_id,
        }
        self._positions().append(position)
        self.state["last_entry_time"] = position["entry_time"]
        self._sync_position_alias(self.state)
        self.state["events"].append({**position, "event": "BUY" if side == "LONG" else "SELL", "timestamp": position["entry_time"], "price": price})

    def _close_position(self, position: dict[str, Any], tick: MarketTick, reason: str) -> None:
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
            "signal_id": self.state.get("last_signal_id"),
        }
        self.state["trades"].append(row)
        self.state["events"].append({**row, "event": "EXIT", "timestamp": row["exit_time"], "price": exit_price})
        self.state["positions"] = [
            row for row in self._positions() if int(row["trade_id"]) != int(position["trade_id"])
        ]
        self._sync_position_alias(self.state)
        self.state["last_exit_time"] = tick.datetime_utc.isoformat()
        self.state["pending_direction"] = None
        self.state["persistence_count"] = 0
        if reason == "stop_loss":
            self._block_direction_after_stop(str(position["side"]))

    def _close_all(self, tick: MarketTick, reason: str) -> int:
        positions = list(self._positions())
        for position in positions:
            self._close_position(position, tick, reason)
        return len(positions)

    def process(self, tick: MarketTick, inference: LiveInference) -> None:
        with self._lock:
            self._mark(tick, save=False)
            if inference.signal_id is not None:
                self.state["last_signal_id"] = inference.signal_id
            if not self.state["running"]:
                self._save()
                return
            if self._session_flatten_active(tick):
                self.close_all_for_session(tick)
                return
            smart_updates = self._apply_smart_short_protection(tick)
            protective_exits = 0
            for position in list(self._positions()):
                if position["side"] == "LONG" and position["stop_loss"] is not None and tick.bid <= position["stop_loss"]:
                    self._close_position(position, tick, "stop_loss")
                    protective_exits += 1
                elif position["side"] == "LONG" and position["take_profit"] is not None and tick.bid >= position["take_profit"]:
                    self._close_position(position, tick, "take_profit")
                    protective_exits += 1
                elif position["side"] == "SHORT" and position["stop_loss"] is not None and tick.ask >= position["stop_loss"]:
                    self._close_position(position, tick, "stop_loss")
                    protective_exits += 1
                elif position["side"] == "SHORT" and position["take_profit"] is not None and tick.ask <= position["take_profit"]:
                    self._close_position(position, tick, "take_profit")
                    protective_exits += 1

            bar_time = inference.inference_time_utc
            if not inference.available or bar_time is None or self.state["last_processed_bar"] == bar_time.isoformat():
                self._mark(tick, save=False)
                if smart_updates:
                    self.state["last_reason"] = f"smart short protection tightened on {smart_updates} position(s)"
                self._save()
                return
            self.state["last_processed_bar"] = bar_time.isoformat()
            if protective_exits:
                self.state["last_signal"] = "EXIT"
                self.state["last_reason"] = f"{protective_exits} protective exit(s)"
                self._mark(tick)
                return
            score = float(inference.probability_up)
            signal, reason = "NO_TRADE", "inside no-trade zone"
            positions = self._positions()
            candidate = "BUY" if score >= self.config.buy_threshold else "SELL" if score <= self.config.sell_threshold else None
            self._update_reentry_gates(candidate)
            if positions:
                side = str(positions[0]["side"])
                exit_long = side == "LONG" and score < self.config.probability_exit_threshold
                exit_short = side == "SHORT" and score > self.config.probability_exit_threshold
                if exit_long or exit_short:
                    if exit_short and self.config.smart_short_enabled:
                        reversals = int(self.state.get("short_reversal_count", 0)) + 1
                        self.state["short_reversal_count"] = reversals
                        if reversals < self.config.short_reversal_confirmations:
                            self.state["short_protect_mode"] = True
                            signal, reason = "HOLD", f"smart short protect mode: reversal {reversals}/{self.config.short_reversal_confirmations}"
                            smart_updates += self._apply_smart_short_protection(tick)
                        else:
                            signal, reason = "EXIT", "smart short confirmed reversal"
                            closed = self._close_all(tick, reason)
                            self.state["short_reversal_count"] = 0
                            self.state["short_protect_mode"] = False
                            self.state["last_signal"], self.state["last_reason"] = signal, f"{reason}: {closed} position(s)"
                            self._mark(tick)
                            return
                    else:
                        signal, reason = "EXIT", "probability reversal"
                        closed = self._close_all(tick, reason)
                        self.state["short_reversal_count"] = 0
                        self.state["short_protect_mode"] = False
                        self.state["last_signal"], self.state["last_reason"] = signal, f"{reason}: {closed} position(s)"
                        self._mark(tick)
                        return
                elif side == "SHORT":
                    self.state["short_reversal_count"] = 0
                    self.state["short_protect_mode"] = False

            if candidate:
                count, daily_pnl = self._today_stats(tick.datetime_utc)
                blockers = []
                max_positions, confirmations, minimum_entry_gap = self.config.entry_rules
                if tick.spread > self.config.max_allowed_spread: blockers.append("spread exceeds maximum")
                if self.config.max_daily_trades is not None and count >= self.config.max_daily_trades:
                    blockers.append("daily trade limit")
                if self.config.max_daily_loss > 0 and daily_pnl <= -self.config.max_daily_loss: blockers.append("daily loss limit")
                if len(positions) >= max_positions: blockers.append(f"position limit {len(positions)}/{max_positions}")
                candidate_side = "LONG" if candidate == "BUY" else "SHORT"
                if self._side_is_blocked(candidate_side): blockers.append(f"anti-raffica: {candidate_side} blocked after stop")
                if positions and any(position["side"] != candidate_side for position in positions):
                    blockers.append("opposite position already open")
                risk_amount = float(self.state["equity"]) * self.config.risk_per_trade_pct / 100
                if self.config.stop_loss_price and self.config.stop_loss_price * self.config.position_size_units > risk_amount:
                    blockers.append("risk per trade limit")
                required_margin = tick.ask * self.config.position_size_units / self.config.leverage
                if required_margin > float(self.state["free_margin"]): blockers.append("insufficient virtual margin")
                last_exit = pd.Timestamp(self.state["last_exit_time"]) if self.state["last_exit_time"] else None
                if not positions and last_exit is not None and tick.datetime_utc < last_exit + pd.Timedelta(minutes=self.config.exit_cooldown_minutes):
                    blockers.append("cooldown active")
                last_entry = pd.Timestamp(self.state["last_entry_time"]) if self.state.get("last_entry_time") else None
                if last_entry is not None and tick.datetime_utc < last_entry + pd.Timedelta(minutes=minimum_entry_gap):
                    blockers.append("minimum interval between entries")
                if not blockers:
                    if self.state["pending_direction"] == candidate:
                        self.state["persistence_count"] += 1
                    else:
                        self.state["pending_direction"], self.state["persistence_count"] = candidate, 1
                    if self.state["persistence_count"] >= confirmations:
                        signal = candidate
                        reason = f"{self.config.entry_mode}: threshold and {confirmations} confirmation(s) passed"
                        self._open(candidate_side, tick, inference, reason)
                        self.state["pending_direction"], self.state["persistence_count"] = None, 0
                    else:
                        reason = f"confirmation {self.state['persistence_count']}/{confirmations}"
                else:
                    self.state["pending_direction"], self.state["persistence_count"] = None, 0
                    reason = "; ".join(blockers)
                    if positions:
                        signal = "HOLD"
            elif positions:
                signal, reason = "HOLD", f"{len(positions)} open position(s) remain supported"
            else:
                self.state["pending_direction"], self.state["persistence_count"] = None, 0
            if smart_updates:
                reason = f"{reason}; smart short protection tightened on {smart_updates} position(s)"
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
    STRATEGIES = (
        ("A", "A · Baseline", False, False, None),
        ("B", "B · Anti-raffica", True, False, 1),
        ("C", "C · Smart SHORT", False, True, None),
        ("D", "D · Combinata", True, True, 1),
    )

    def __init__(self, root: Path | str, config: PaperConfig):
        self.root, self.config = Path(root), config
        manifest = self.root / "models/baseline_manifest_provisional.json"
        definitions: list[tuple[str, Path, Path, str, str]] = [
            ("LightGBM", self.root / "models/lightgbm_up_5m_sigmoid_provisional.joblib", manifest, "lightgbm", "binary"),
            ("Logistic Regression", self.root / "models/logistic_regression_up_5m_sigmoid_provisional.joblib", manifest, "logistic_regression", "binary"),
            ("XGBoost", self.root / "models/xgboost_up_5m_sigmoid_provisional.joblib", manifest, "xgboost", "binary"),
        ]
        selection_path = self.root / "data/live/paper/model_selection.json"
        if selection_path.exists():
            try:
                selected = json.loads(selection_path.read_text(encoding="utf-8"))
                custom = []
                for item in selected.get("models", []):
                    model_path, manifest_path = Path(item["artifact"]), Path(item["manifest"])
                    if model_path.exists() and manifest_path.exists():
                        account_key = hashlib.sha256(str(model_path.resolve()).encode()).hexdigest()[:12]
                    custom.append((str(item["label"]), model_path, manifest_path, account_key, str(item.get("kind", "binary"))))
                if custom:
                    definitions = custom
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
        available = next((definition for definition in definitions if definition[1].exists() and definition[2].exists()), None)
        if available is None:
            raise RuntimeError("Nessun modello compatibile disponibile per il Paper")
        source_label, path, model_manifest, key, kind = available
        self.source_model_label = source_label
        self.engine = CostAwareLiveInferenceEngine(path, model_manifest) if kind == "cost_aware" else LiveInferenceEngine(
            path, model_manifest, config.buy_threshold, config.sell_threshold,
        )
        # Exactly one model invocation creates one signal event, fan-out to A/B/C/D.
        self.engines = {source_label: self.engine}
        self.accounts: dict[str, PaperAccount] = {}
        for strategy_id, label, anti_burst, smart_short, maximum in self.STRATEGIES:
            account_key = f"comparison_v1_{key}_{strategy_id.lower()}"
            strategy_config = self._strategy_config(config, strategy_id, anti_burst, smart_short, maximum)
            self.accounts[label] = PaperAccount(
                account_key, label, strategy_config, self.root / "data/live/paper/comparison_v1",
            )
        self._last_bar: int | None = None
        self._inference: LiveInference | None = None
        self._inferences: dict[str, LiveInference] = {}
        self._lock = RLock()

    @staticmethod
    def _strategy_config(
        config: PaperConfig, strategy_id: str, anti_burst: bool, smart_short: bool, maximum: int | None,
    ) -> PaperConfig:
        return replace(
            config, strategy_id=strategy_id, anti_burst_enabled=anti_burst,
            smart_short_enabled=smart_short, max_open_positions_override=maximum,
        )

    def _config_for_account(self, base: PaperConfig, account: PaperAccount) -> PaperConfig:
        return self._strategy_config(
            base, account.config.strategy_id, account.config.anti_burst_enabled,
            account.config.smart_short_enabled, account.config.max_open_positions_override,
        )

    def process(self, tick: MarketTick, completed_m1: pd.DataFrame) -> None:
        with self._lock:
            latest = int(completed_m1.timestamp.iloc[-1]) if not completed_m1.empty else None
            if latest != self._last_bar:
                inference = self.engine.predict(completed_m1)
                self._inference = replace(inference, signal_id=latest) if inference.available else inference
                self._inferences = {name: self._inference for name in self.accounts}
                self._last_bar = latest
            if self._inference is not None:
                for account in self.accounts.values():
                    account.process(tick, self._inference)

    def comparison(self) -> pd.DataFrame:
        rows = []
        for name, account in self.accounts.items():
            state = account.snapshot()
            positions = state.get("positions") or ([state["position"]] if state.get("position") else [])
            total_pnl = float(state["realized_pnl"]) + float(state["unrealized_pnl"])
            rows.append({
                "model": name,
                "strategy_id": account.config.strategy_id,
                "source_model": getattr(self, "source_model_label", "N/D"),
                "status": "RUNNING" if state["running"] else "STOPPED",
                "balance": state["balance"],
                "equity": state["equity"],
                "realized_pnl": state["realized_pnl"],
                "unrealized_pnl": state["unrealized_pnl"],
                "total_pnl": total_pnl,
                "return_pct": total_pnl / account.config.starting_capital,
                "free_margin": state["free_margin"],
                "trades": len(state["trades"]),
                "entry_mode": account.config.entry_mode,
                "open_positions": len(positions),
                "position_limit": account.config.entry_rules[0],
                "position": f"{len(positions)} {positions[0]['side']}" if positions else "FLAT",
                "signal": state["last_signal"],
                "max_drawdown": state["max_drawdown"],
            })
        return pd.DataFrame(rows)

    def inference_for(self, model: str) -> LiveInference | None:
        """Return the one shared inference consumed by every strategy."""
        with self._lock:
            if hasattr(self, "_inference") and model in self.accounts:
                return self._inference
            return self._inferences.get(model)

    def set_entry_mode(self, mode: str) -> None:
        """Apply one entry-intensity preset to all accounts without resetting them."""
        with self._lock:
            updated = replace(self.config, entry_mode=mode)
            updated.validate()
            self.config = updated
            for account in self.accounts.values():
                account.set_entry_mode(mode)

    def update_config_preserving_history(self, config: PaperConfig) -> None:
        """Update every paper account in place, preserving open legs and ledger."""
        config.validate()
        with self._lock:
            self.config = config
            for account in self.accounts.values():
                account.update_config_preserving_history(self._config_for_account(config, account))
            if isinstance(self.engine, LiveInferenceEngine):
                self.engine.buy_threshold, self.engine.sell_threshold = config.buy_threshold, config.sell_threshold

    def close_all_for_session(self, tick: MarketTick) -> dict[str, int]:
        """Flatten all paper accounts at one observed virtual market tick."""
        with self._lock:
            return {name: account.close_all_for_session(tick) for name, account in self.accounts.items()}

    def reconfigure_and_reset(self, config: PaperConfig) -> None:
        """Apply one fair configuration to every model and start fresh experiments."""
        config.validate()
        with self._lock:
            self.config = config
            for name, old in list(self.accounts.items()):
                account_config = self._config_for_account(config, old)
                account = PaperAccount(old.account_id, name, account_config, old.directory.parent)
                account.config = account_config
                account.state = account._new_state()
                account._save()
                self.accounts[name] = account
            if isinstance(self.engine, LiveInferenceEngine):
                self.engine.buy_threshold, self.engine.sell_threshold = config.buy_threshold, config.sell_threshold
