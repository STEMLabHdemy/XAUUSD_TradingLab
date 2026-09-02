from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

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
    short_reversal_mode: str | None = None
    short_reversal_price_confirm_bars: int = 0
    # Optional entry filters used only by explicitly-labelled research portfolios.
    # Values are decimal returns, e.g. .001 = +0.10%.
    short_entry_max_prior_return_15m: float | None = None
    short_entry_max_range_15m: float | None = None
    direction_lock_rearm_bars: int = 2
    probability_reversal_enabled: bool = True

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
        if self.strategy_id not in {"0", *set("ABCDEFGHIJKLMNO")}:
            raise ValueError("Strategy id must be 0 or between A and O")
        if self.max_open_positions_override is not None and self.max_open_positions_override < 1:
            raise ValueError("Maximum open positions override must be positive")
        if self.short_reversal_confirmations < 2:
            raise ValueError("Smart short requires at least two reversal confirmations")
        if self.short_reversal_mode not in {None, "immediate", "smart", "disabled"}:
            raise ValueError("Short reversal mode must be immediate, smart or disabled")
        if self.short_reversal_price_confirm_bars < 0 or self.direction_lock_rearm_bars < 2:
            raise ValueError("Confirmation bars must be non-negative; direction lock needs at least two bars")
        for value in (self.short_entry_max_prior_return_15m, self.short_entry_max_range_15m):
            if value is not None and value < 0:
                raise ValueError("SHORT entry filters cannot be negative")
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
            "portfolio_history": [], "run_id": None, "experiment_started_at": None,
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
        state.setdefault("portfolio_history", [])
        state.setdefault("run_id", None)
        state.setdefault("experiment_started_at", None)
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

    def set_run_id(self, run_id: str, started_at: str | None = None) -> None:
        """Attach an experiment identifier without touching positions or ledger."""
        with self._lock:
            self.state["run_id"] = run_id
            if started_at is not None:
                self.state["experiment_started_at"] = started_at
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

    def _record_event(self, event: str, tick: MarketTick, **values: Any) -> None:
        self.state["events"].append({
            "event": event, "timestamp": tick.datetime_utc.isoformat(),
            "run_id": self.state.get("run_id"), "strategy_id": self.config.strategy_id,
            "model": self.model, "signal_id": self.state.get("last_signal_id"),
            **values,
        })

    def _append_portfolio_snapshot(self, tick: MarketTick) -> None:
        self.state["portfolio_history"].append({
            "timestamp": tick.datetime_utc.isoformat(), "run_id": self.state.get("run_id"),
            "strategy_id": self.config.strategy_id, "signal_id": self.state.get("last_signal_id"),
            "balance": self.state["balance"], "equity": self.state["equity"],
            "realized_pnl": self.state["realized_pnl"], "unrealized_pnl": self.state["unrealized_pnl"],
            "max_drawdown": self.state["max_drawdown"], "exposure": self.state["exposure"],
            "used_margin": self.state["used_margin"], "free_margin": self.state["free_margin"],
            "open_positions": len(self._positions()),
        })
        self.state["portfolio_history"] = self.state["portfolio_history"][-20_000:]

    def _record_decision(
        self, tick: MarketTick, inference: LiveInference, signal: str, reason: str,
        market_context: dict[str, float | None] | None = None,
        allocation_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist every completed-candle decision, including explicit no-trade outcomes."""
        self._record_event(
            "SIGNAL_DECISION", tick,
            inference_timestamp=inference.inference_time_utc.isoformat() if inference.inference_time_utc is not None else None,
            bid=tick.bid, ask=tick.ask, spread=tick.spread,
            p_up=inference.probability_up, p_down=inference.probability_down,
            p_neutral=inference.probability_neutral, predicted_class=inference.candidate,
            decision=signal, decision_reason=reason, open_positions=len(self._positions()),
            direction_locks=json.loads(json.dumps(self.state["reentry_gates"])),
            short_reversal_count=self.state.get("short_reversal_count", 0),
            short_protect_mode=self.state.get("short_protect_mode", False),
            **(market_context or {}),
            **(allocation_metadata or {}),
        )
        self._append_portfolio_snapshot(tick)

    def _block_direction_after_stop(self, side: str, tick: MarketTick) -> None:
        if not self.config.anti_burst_enabled:
            return
        gate = self.state["reentry_gates"].setdefault(
            side, {"blocked": False, "non_directional_bars": 0, "rearmed": False},
        )
        gate.update(blocked=True, non_directional_bars=0, rearmed=False)
        self._record_event("DIRECTION_LOCK_ON", tick, side=side, reason="stop_loss", lock_state=gate.copy())

    def _update_reentry_gates(self, tick: MarketTick, candidate: str | None) -> None:
        """Require two non-directional bars and a later return after a stopped-out side."""
        if not self.config.anti_burst_enabled:
            return
        candidate_side = {"BUY": "LONG", "SELL": "SHORT"}.get(candidate)
        for side, gate in self.state["reentry_gates"].items():
            if not gate.get("blocked"):
                continue
            if candidate_side != side:
                gate["non_directional_bars"] = int(gate.get("non_directional_bars", 0)) + 1
                if gate["non_directional_bars"] >= self.config.direction_lock_rearm_bars and not gate.get("rearmed"):
                    gate["rearmed"] = True
                    self._record_event("DIRECTION_LOCK_ARMED", tick, side=side, lock_state=gate.copy())
            elif gate.get("rearmed"):
                gate.update(blocked=False, non_directional_bars=0, rearmed=False)
                self._record_event("DIRECTION_LOCK_OFF", tick, side=side, reason="direction_returned", lock_state=gate.copy())

    def _side_is_blocked(self, side: str) -> bool:
        return bool(self.config.anti_burst_enabled and self.state["reentry_gates"].get(side, {}).get("blocked"))

    def _exit_policy_value(self, position: dict[str, Any], field: str) -> Any:
        """Use an O trade's frozen source policy, otherwise this account's config."""
        policy = position.get("exit_policy")
        if isinstance(policy, dict) and field in policy:
            return policy[field]
        return getattr(self.config, field)

    def _apply_smart_short_protection(self, tick: MarketTick) -> int:
        """Tighten only SHORT stops after the first protected reversal; never widen a stop."""
        if not self.state.get("short_protect_mode"):
            return 0
        changed = 0
        for position in self._positions():
            if position["side"] != "SHORT":
                continue
            if not bool(self._exit_policy_value(position, "smart_short_enabled")):
                continue
            pnl = self._position_unrealized_pnl(position, tick)
            if pnl < float(self._exit_policy_value(position, "short_protect_break_even_pnl")):
                continue
            quantity = float(position["quantity"])
            total_costs = float(position["entry_costs"]) + (
                self.config.slippage_price_per_side + self.config.commission_per_unit_per_side
            ) * quantity
            protected_pnl = 0.0
            if pnl >= float(self._exit_policy_value(position, "short_protect_lock_trigger_pnl")):
                protected_pnl = float(self._exit_policy_value(position, "short_protect_lock_pnl"))
            desired_stop = float(position["raw_entry_price"]) - (protected_pnl + total_costs) / quantity
            if pnl >= float(self._exit_policy_value(position, "short_protect_trailing_trigger_pnl")):
                desired_stop = min(desired_stop, tick.ask + float(self._exit_policy_value(position, "short_protect_trailing_distance")))
            old_stop = position.get("stop_loss")
            if old_stop is None or desired_stop < float(old_stop):
                position["stop_loss"] = desired_stop
                protection = "break_even" if protected_pnl == 0 else "lock"
                if pnl >= float(self._exit_policy_value(position, "short_protect_trailing_trigger_pnl")):
                    protection = "trailing"
                self._record_event(
                    "SMART_PROTECT", tick, trade_id=position["trade_id"], side="SHORT", price=tick.ask,
                    old_stop_loss=old_stop, stop_loss=desired_stop, net_pnl_at_update=pnl,
                    protection=protection,
                )
                changed += 1
        return changed

    def _open(
        self, side: str, tick: MarketTick, inference: LiveInference, reason: str,
        market_context: dict[str, float | None] | None = None, allocation_weight: float = 1.0,
        source_exit_policy: dict[str, Any] | None = None,
    ) -> None:
        quantity = self.config.position_size_units * allocation_weight
        slip = self.config.slippage_price_per_side
        raw = tick.ask if side == "LONG" else tick.bid
        price = raw + slip if side == "LONG" else raw - slip
        policy = source_exit_policy or {}
        stop_distance = policy.get("stop_loss_price", self.config.stop_loss_price)
        take_distance = policy.get("take_profit_price", self.config.take_profit_price)
        stop = price - stop_distance if side == "LONG" and stop_distance else None
        stop = price + stop_distance if side == "SHORT" and stop_distance else stop
        take = price + take_distance if side == "LONG" and take_distance else None
        take = price - take_distance if side == "SHORT" and take_distance else take
        trade_id = int(self.state["next_trade_id"])
        self.state["next_trade_id"] = trade_id + 1
        position = {
            "trade_id": trade_id, "side": side, "entry_time": tick.datetime_utc.isoformat(),
            "raw_entry_price": raw, "entry_price": price, "quantity": quantity,
            "confidence": inference.probability_up, "spread": tick.spread, "stop_loss": stop,
            "take_profit": take, "entry_costs": (slip + self.config.commission_per_unit_per_side) * quantity,
            "model": self.model, "strategy_id": self.config.strategy_id, "run_id": self.state.get("run_id"),
            "regime": "not_available", "expected_return": None, "reason": reason,
            "signal_id": inference.signal_id,
            "allocation_weight": allocation_weight,
            "source_strategy_id": policy.get("strategy_id"),
            "exit_policy": policy or None,
            "entry_prior_return_15m_pct": (market_context or {}).get("prior_return_15m_pct"),
            "entry_range_15m_pct": (market_context or {}).get("range_15m_pct"),
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
            "strategy_id": self.config.strategy_id, "run_id": self.state.get("run_id"),
            "signal_id": self.state.get("last_signal_id"),
            "entry_prior_return_15m_pct": position.get("entry_prior_return_15m_pct"),
            "entry_range_15m_pct": position.get("entry_range_15m_pct"),
            "allocation_weight": position.get("allocation_weight", 1.0),
            "source_strategy_id": position.get("source_strategy_id"),
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
            self._block_direction_after_stop(str(position["side"]), tick)

    def _close_all(self, tick: MarketTick, reason: str) -> int:
        positions = list(self._positions())
        for position in positions:
            self._close_position(position, tick, reason)
        return len(positions)

    def process(
        self, tick: MarketTick, inference: LiveInference,
        market_context: dict[str, float | None] | None = None,
        entry_blocker: str | None = None, entry_size_multiplier: float = 1.0,
        allocation_metadata: dict[str, Any] | None = None,
        source_exit_policy: dict[str, Any] | None = None,
    ) -> None:
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
                self._mark(tick, save=False)
                self._record_decision(tick, inference, "EXIT", self.state["last_reason"], market_context, allocation_metadata)
                self._save()
                return
            score = float(inference.probability_up)
            signal, reason = "NO_TRADE", "inside no-trade zone"
            positions = self._positions()
            candidate = "BUY" if score >= self.config.buy_threshold else "SELL" if score <= self.config.sell_threshold else None
            self._update_reentry_gates(tick, candidate)
            if positions:
                managed_position = positions[0]
                side = str(managed_position["side"])
                exit_threshold = float(self._exit_policy_value(managed_position, "probability_exit_threshold"))
                exit_long = side == "LONG" and score < exit_threshold
                exit_short = side == "SHORT" and score > exit_threshold
                if (exit_long or exit_short) and bool(self._exit_policy_value(managed_position, "probability_reversal_enabled")):
                    short_mode = self._exit_policy_value(managed_position, "short_reversal_mode") or (
                        "smart" if self._exit_policy_value(managed_position, "smart_short_enabled") else "immediate"
                    )
                    if exit_short and short_mode == "disabled":
                        self.state["short_reversal_count"] = 0
                        self.state["short_protect_mode"] = False
                        signal, reason = "HOLD", "short probability reversal disabled by strategy"
                    elif exit_short and short_mode == "smart":
                        price_confirmed = (
                            int(self._exit_policy_value(managed_position, "short_reversal_price_confirm_bars")) == 0
                            or bool(inference.short_reversal_price_confirmed)
                        )
                        if not price_confirmed:
                            self.state["short_reversal_count"] = 0
                            self.state["short_protect_mode"] = False
                            signal, reason = "HOLD", "smart short reversal rejected: price confirmation missing"
                            self._record_event(
                                "SMART_SHORT_REVERSAL", tick, stage="rejected", action="hold",
                                reason="price_confirmation_missing",
                                required_bars=int(self._exit_policy_value(managed_position, "short_reversal_price_confirm_bars")),
                            )
                        else:
                            reversals = int(self.state.get("short_reversal_count", 0)) + 1
                            self.state["short_reversal_count"] = reversals
                            confirmations_required = int(self._exit_policy_value(managed_position, "short_reversal_confirmations"))
                            if reversals < confirmations_required:
                                self.state["short_protect_mode"] = True
                                signal, reason = "HOLD", f"smart short protect mode: reversal {reversals}/{confirmations_required}"
                                self._record_event("SMART_SHORT_REVERSAL", tick, stage="first", action="protect", reversals=reversals)
                                smart_updates += self._apply_smart_short_protection(tick)
                            else:
                                signal, reason = "EXIT", "smart short confirmed reversal"
                                self._record_event("SMART_SHORT_REVERSAL", tick, stage="second", action="close", reversals=reversals)
                                closed = self._close_all(tick, reason)
                                self.state["short_reversal_count"] = 0
                                self.state["short_protect_mode"] = False
                                self.state["last_signal"], self.state["last_reason"] = signal, f"{reason}: {closed} position(s)"
                                self._mark(tick, save=False)
                                self._record_decision(tick, inference, signal, self.state["last_reason"], market_context, allocation_metadata)
                                self._save()
                                return
                    else:
                        signal, reason = "EXIT", "probability reversal"
                        closed = self._close_all(tick, reason)
                        self.state["short_reversal_count"] = 0
                        self.state["short_protect_mode"] = False
                        self.state["last_signal"], self.state["last_reason"] = signal, f"{reason}: {closed} position(s)"
                        self._mark(tick, save=False)
                        self._record_decision(tick, inference, signal, self.state["last_reason"], market_context, allocation_metadata)
                        self._save()
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
                if entry_blocker:
                    blockers.append(entry_blocker)
                prior_return = (market_context or {}).get("prior_return_15m_pct")
                range_15m = (market_context or {}).get("range_15m_pct")
                if (candidate_side == "SHORT" and self.config.short_entry_max_prior_return_15m is not None
                        and prior_return is not None
                        and prior_return > self.config.short_entry_max_prior_return_15m):
                    blockers.append(
                        "SHORT trend filter: M15 prior return "
                        f"{prior_return:+.3%} exceeds {self.config.short_entry_max_prior_return_15m:+.3%}"
                    )
                if (candidate_side == "SHORT" and self.config.short_entry_max_range_15m is not None
                        and range_15m is not None
                        and range_15m > self.config.short_entry_max_range_15m):
                    blockers.append(
                        "SHORT volatility filter: M15 range "
                        f"{range_15m:.3%} exceeds {self.config.short_entry_max_range_15m:.3%}"
                    )
                if self._side_is_blocked(candidate_side): blockers.append(f"anti-raffica: {candidate_side} blocked after stop")
                if positions and any(position["side"] != candidate_side for position in positions):
                    blockers.append("opposite position already open")
                risk_amount = float(self.state["equity"]) * self.config.risk_per_trade_pct / 100
                entry_units = self.config.position_size_units * entry_size_multiplier
                entry_stop_distance = (source_exit_policy or {}).get("stop_loss_price", self.config.stop_loss_price)
                if entry_stop_distance and float(entry_stop_distance) * entry_units > risk_amount:
                    blockers.append("risk per trade limit")
                required_margin = tick.ask * entry_units / self.config.leverage
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
                        self._open(
                            candidate_side, tick, inference, reason, market_context, entry_size_multiplier,
                            source_exit_policy,
                        )
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
            self._record_decision(tick, inference, signal, reason, market_context, allocation_metadata)
            if not self.state.get("_skip_equity_history", False):
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


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    label: str
    anti_burst: bool = False
    max_open_positions: int | None = None
    short_reversal_mode: str = "immediate"
    short_reversal_confirmations: int = 2
    short_reversal_price_confirm_bars: int = 0
    direction_lock_rearm_bars: int = 2
    probability_reversal_enabled: bool = True
    protection_overrides: tuple[tuple[str, float], ...] = ()
    short_entry_max_prior_return_15m: float | None = None
    short_entry_max_range_15m: float | None = None

    def __iter__(self):
        """Compatibility with the original five-value A-D test fixture."""
        yield self.strategy_id
        yield self.label
        yield self.anti_burst
        yield self.short_reversal_mode == "smart"
        yield self.max_open_positions


class PaperRuntime:
    """One inference stream fanned out to independent, comparable ledgers."""

    STRATEGIES = (
        # A-D are frozen controls: their operational rules are unchanged.
        StrategySpec("A", "A · Baseline"),
        StrategySpec("B", "B · Anti-raffica", anti_burst=True, max_open_positions=1),
        StrategySpec("C", "C · Smart SHORT", short_reversal_mode="smart"),
        StrategySpec("D", "D · Combinata", anti_burst=True, max_open_positions=1, short_reversal_mode="smart"),
        # New, explicit hypotheses. They all consume the exact same M1 event.
        StrategySpec("E", "E · Smart SHORT 3", short_reversal_mode="smart", short_reversal_confirmations=3),
        StrategySpec("F", "F · No reversal SHORT", short_reversal_mode="disabled"),
        StrategySpec("G", "G · Smart SHORT + prezzo", short_reversal_mode="smart", short_reversal_price_confirm_bars=3),
        StrategySpec("H", "H · 3 LONG + prezzo", short_reversal_mode="smart", short_reversal_confirmations=3, short_reversal_price_confirm_bars=3),
        StrategySpec("I", "I · Smart SHORT protetto", short_reversal_mode="smart", protection_overrides=(
            ("short_protect_break_even_pnl", 0.0), ("short_protect_lock_trigger_pnl", 2.0), ("short_protect_lock_pnl", 0.5),
        )),
        StrategySpec("J", "J · Smart SHORT trailing largo", short_reversal_mode="smart", protection_overrides=(
            ("short_protect_trailing_distance", 4.0),
        )),
        StrategySpec("K", "K · Anti-raffica 2", anti_burst=True, max_open_positions=2),
        StrategySpec("L", "L · Anti-raffica severa", anti_burst=True, max_open_positions=1, direction_lock_rearm_bars=3),
        # Pure control: no model-driven exit. Only SL, TP or session close can flatten it.
        StrategySpec("0", "0 · Solo SL/TP", probability_reversal_enabled=False),
        # E is retained unchanged as the direct benchmark. M/N test only the
        # independently-observed conditions that harmed historical SHORTs.
        StrategySpec("M", "M · E + blocco SHORT contro-trend M15", short_reversal_mode="smart",
                     short_reversal_confirmations=3, short_entry_max_prior_return_15m=.001),
        StrategySpec("N", "N · E + blocco SHORT trend + volatilita", short_reversal_mode="smart",
                     short_reversal_confirmations=3, short_entry_max_prior_return_15m=.001,
                     short_entry_max_range_15m=.003),
        # A new, independent ledger.  It has E's exit policy but admits new
        # positions only when the causal meta allocator approves them.
        StrategySpec("O", "O · Meta momentum (policy della fonte)", short_reversal_mode="smart",
                     short_reversal_confirmations=3, max_open_positions=1),
    )

    def __init__(self, root: Path | str, config: PaperConfig):
        self.root, self.config = Path(root), config
        self.comparison_directory = self.root / "data/live/paper/comparison_v1"
        self.run_metadata_path = self.comparison_directory / "run.json"
        self.run_id = self._load_or_create_run()
        self._register_strategy_catalog()
        self._run_started_at = self._run_start_timestamp()
        self.signal_log_path = self.comparison_directory / "signals.jsonl"
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
        # Exactly one model invocation creates one signal event, fan-out to A-L.
        self.engines = {source_label: self.engine}
        self.accounts: dict[str, PaperAccount] = {}
        for spec in self.STRATEGIES:
            account_key = f"comparison_v1_{key}_{spec.strategy_id.lower()}"
            strategy_config = self._strategy_config(config, spec)
            is_new_account = not (self.comparison_directory / account_key / "state.json").exists()
            self.accounts[spec.label] = PaperAccount(
                account_key, spec.label, strategy_config, self.comparison_directory,
            )
            self.accounts[spec.label].set_run_id(self.run_id)
            if is_new_account and any(account.snapshot().get("running") for account in self.accounts.values()):
                self.accounts[spec.label].set_run_id(self.run_id, datetime.now(timezone.utc).isoformat())
                self.accounts[spec.label].start()
        self._last_bar: int | None = None
        self._inference: LiveInference | None = None
        self._inferences: dict[str, LiveInference] = {}
        self._lock = RLock()

    def _load_or_create_run(self) -> str:
        self.comparison_directory.mkdir(parents=True, exist_ok=True)
        if self.run_metadata_path.exists():
            try:
                metadata = json.loads(self.run_metadata_path.read_text(encoding="utf-8"))
                if isinstance(metadata.get("run_id"), str):
                    return metadata["run_id"]
            except (OSError, json.JSONDecodeError):
                pass
        run_id = f"al_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        self.run_metadata_path.write_text(json.dumps({
            "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "strategy_lab_0_to_o", "strategies": [item.strategy_id for item in self.STRATEGIES],
        }, indent=2), encoding="utf-8")
        return run_id

    def _run_start_timestamp(self) -> pd.Timestamp | None:
        try:
            metadata = json.loads(self.run_metadata_path.read_text(encoding="utf-8"))
            value = pd.to_datetime(metadata.get("created_at"), utc=True, errors="coerce")
            return None if pd.isna(value) else value
        except (OSError, json.JSONDecodeError):
            return None

    def _register_strategy_catalog(self) -> None:
        """Keep metadata descriptive when a new non-resetting strategy is added."""
        try:
            metadata = json.loads(self.run_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        strategy_ids = [item.strategy_id for item in self.STRATEGIES]
        if metadata.get("strategies") != strategy_ids or metadata.get("experiment") != "strategy_lab_0_to_o":
            metadata["experiment"] = "strategy_lab_0_to_o"
            metadata["strategies"] = strategy_ids
            metadata["strategy_catalog_updated_at"] = datetime.now(timezone.utc).isoformat()
            self.run_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_shared_signal(self, tick: MarketTick, inference: LiveInference) -> None:
        if inference.signal_id is None:
            return
        metadata = json.loads(self.run_metadata_path.read_text(encoding="utf-8"))
        if metadata.get("last_signal_id") == inference.signal_id:
            return
        row = {
            "run_id": self.run_id, "signal_id": inference.signal_id,
            "timestamp": inference.inference_time_utc.isoformat() if inference.inference_time_utc is not None else None,
            "tick_timestamp": tick.datetime_utc.isoformat(), "bid": tick.bid, "ask": tick.ask, "spread": tick.spread,
            "p_up": inference.probability_up, "p_down": inference.probability_down,
            "p_neutral": inference.probability_neutral, "predicted_class": inference.candidate,
            "model": inference.model, "horizon_minutes": inference.horizon_minutes,
            "short_reversal_price_confirmed": inference.short_reversal_price_confirmed,
        }
        with self.signal_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata["last_signal_id"] = inference.signal_id
        metadata["last_signal_timestamp"] = row["timestamp"]
        self.run_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _strategy_config(config: PaperConfig, spec: StrategySpec) -> PaperConfig:
        values: dict[str, Any] = {
            "strategy_id": spec.strategy_id,
            "anti_burst_enabled": spec.anti_burst,
            "smart_short_enabled": spec.short_reversal_mode == "smart",
            "max_open_positions_override": spec.max_open_positions,
            "short_reversal_mode": spec.short_reversal_mode,
            "short_reversal_confirmations": spec.short_reversal_confirmations,
            "short_reversal_price_confirm_bars": spec.short_reversal_price_confirm_bars,
            "direction_lock_rearm_bars": spec.direction_lock_rearm_bars,
            "probability_reversal_enabled": spec.probability_reversal_enabled,
            "short_entry_max_prior_return_15m": spec.short_entry_max_prior_return_15m,
            "short_entry_max_range_15m": spec.short_entry_max_range_15m,
        }
        values.update(dict(spec.protection_overrides))
        return replace(config, **values)

    def _config_for_account(self, base: PaperConfig, account: PaperAccount) -> PaperConfig:
        spec = next(item for item in self.STRATEGIES if item.strategy_id == account.config.strategy_id)
        return self._strategy_config(base, spec)

    def process(self, tick: MarketTick, completed_m1: pd.DataFrame) -> None:
        with self._lock:
            latest = int(completed_m1.timestamp.iloc[-1]) if not completed_m1.empty else None
            new_bar = latest != self._last_bar
            latest_time = (
                pd.Timestamp(completed_m1.datetime_utc.iloc[-1])
                if not completed_m1.empty and "datetime_utc" in completed_m1 else None
            )
            # A restarted dashboard can still see the final candle before a
            # market pause.  Do not inject that old event into a fresh cohort.
            if (latest_time is not None and self._run_started_at is not None
                    and latest_time < self._run_started_at):
                self._last_bar = latest
                return
            if new_bar:
                inference = self.engine.predict(completed_m1)
                price_confirmed = False
                lookback = 3
                if len(completed_m1) > lookback and "high" in completed_m1:
                    recent_high = float(completed_m1.high.iloc[-1])
                    prior_high = float(completed_m1.high.iloc[-(lookback + 1):-1].max())
                    price_confirmed = recent_high > prior_high
                self._inference = replace(
                    inference, signal_id=latest, short_reversal_price_confirmed=price_confirmed,
                ) if inference.available else inference
                self._inferences = {name: self._inference for name in self.accounts}
                self._market_context = self._m15_market_context(completed_m1)
                self._last_bar = latest
                if self._inference.available and hasattr(self, "run_metadata_path"):
                    self._record_shared_signal(tick, self._inference)
            if self._inference is not None:
                allocation = self._meta_allocation(self._inference, getattr(self, "_market_context", None))
                for account in self.accounts.values():
                    if account.config.strategy_id != "O":
                        account.process(tick, self._inference, getattr(self, "_market_context", None))
                meta_account = next((account for account in self.accounts.values() if account.config.strategy_id == "O"), None)
                if meta_account is not None:
                    source_id = allocation["metadata"].get("meta_source_strategy")
                    source = next((account for account in self.accounts.values() if account.config.strategy_id == source_id), None)
                    source_decision = source.snapshot().get("last_signal") if source is not None else None
                    allocation["metadata"]["meta_source_decision"] = source_decision
                    if allocation["entry_blocker"] is None and source_decision not in {"BUY", "SELL"}:
                        allocation["entry_blocker"] = (
                            f"meta allocator: source {source_id} did not open a new trade ({source_decision or 'N/D'})"
                        )
                    if new_bar and self._inference.available:
                        meta_account._record_event("META_ALLOCATION", tick, **allocation["metadata"])
                    meta_account.process(
                        tick, self._inference, getattr(self, "_market_context", None),
                        entry_blocker=allocation["entry_blocker"],
                        entry_size_multiplier=allocation["weight"],
                        allocation_metadata=allocation["metadata"],
                        source_exit_policy=self._exit_policy_snapshot(source.config) if source is not None else None,
                    )
                if self._inference.available and hasattr(self, "comparison_directory"):
                    # Export after a completed-M1 decision, not on every tick.
                    # Rewriting all CSVs on every UI refresh was avoidable disk IO.
                    if new_bar:
                        self.write_strategy_exports()

    @staticmethod
    def _exit_policy_snapshot(config: PaperConfig) -> dict[str, Any]:
        """Freeze exactly the exit rules of O's chosen source inside the trade."""
        fields = (
            "strategy_id", "stop_loss_price", "take_profit_price", "probability_exit_threshold",
            "probability_reversal_enabled", "short_reversal_mode", "smart_short_enabled",
            "short_reversal_confirmations", "short_reversal_price_confirm_bars",
            "short_protect_break_even_pnl", "short_protect_lock_trigger_pnl", "short_protect_lock_pnl",
            "short_protect_trailing_trigger_pnl", "short_protect_trailing_distance",
        )
        return {field: getattr(config, field) for field in fields}

    @staticmethod
    def _m15_market_context(completed_m1: pd.DataFrame) -> dict[str, float | None]:
        """Calculate one causal M15 context shared by every strategy."""
        required = {"mid_close", "mid_high", "mid_low"}
        if len(completed_m1) < 16 or not required.issubset(completed_m1.columns):
            return {"prior_return_15m_pct": None, "range_15m_pct": None}
        window = completed_m1.iloc[-16:]
        start, end = float(window.mid_close.iloc[0]), float(window.mid_close.iloc[-1])
        if start <= 0 or end <= 0:
            return {"prior_return_15m_pct": None, "range_15m_pct": None}
        return {
            "prior_return_15m_pct": end / start - 1.0,
            "range_15m_pct": (float(window.mid_high.max()) - float(window.mid_low.min())) / end,
        }

    @staticmethod
    def _rolling_pnl_change(state: dict[str, Any], now: pd.Timestamp, minutes: int) -> float | None:
        """PnL delta that would have been known before the current M1 decision."""
        history = pd.DataFrame(state.get("portfolio_history", []))
        if history.empty or "timestamp" not in history:
            return None
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
        baseline = history[history["timestamp"] <= now - pd.Timedelta(minutes=minutes)]
        if baseline.empty:
            return None
        row = baseline.iloc[-1]
        prior = float(row["realized_pnl"]) + float(row["unrealized_pnl"])
        current = float(state["realized_pnl"]) + float(state["unrealized_pnl"])
        return current - prior

    @staticmethod
    def _recent_drawdown_amount(state: dict[str, Any], now: pd.Timestamp, minutes: int = 15) -> float:
        """Loss from the local peak only; this deliberately is not a 120m DD."""
        history = pd.DataFrame(state.get("portfolio_history", []))
        if history.empty or "timestamp" not in history:
            return 0.0
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.dropna(subset=["timestamp"])
        window = history[history["timestamp"] >= now - pd.Timedelta(minutes=minutes)]
        current = float(state["realized_pnl"]) + float(state["unrealized_pnl"])
        if window.empty:
            return 0.0
        peak = max((float(row.realized_pnl) + float(row.unrealized_pnl) for row in window.itertuples()), default=current)
        return min(0.0, current - peak)

    def _meta_allocation(
        self, inference: LiveInference, market_context: dict[str, float | None] | None,
    ) -> dict[str, Any]:
        """Select one paper source using only fast, already-observed PnL.

        This is intentionally a small, transparent first meta strategy. It
        ranks every existing paper portfolio, but never sums their correlated
        exposure: O has one position limit and selects one source only.
        """
        metadata: dict[str, Any] = {
            "meta_source_strategy": None, "meta_score": None,
            "meta_pnl_5m": None, "meta_pnl_15m": None, "meta_pnl_30m": None,
            "meta_recent_drawdown_15m": None, "meta_allocation_weight": 0.0,
            "meta_source_decision": None,
        }
        if not inference.available or inference.inference_time_utc is None:
            return {"entry_blocker": "meta allocator: inference unavailable", "weight": 1.0, "metadata": metadata}
        now = pd.Timestamp(inference.inference_time_utc)
        candidates: list[tuple[float, PaperAccount, dict[str, float | None]]] = []
        for account in self.accounts.values():
            if account.config.strategy_id == "O":
                continue
            state = account.snapshot()
            if not state.get("running"):
                continue
            pnl_5 = self._rolling_pnl_change(state, now, 5)
            if pnl_5 is None:
                continue
            pnl_15 = self._rolling_pnl_change(state, now, 15)
            pnl_30 = self._rolling_pnl_change(state, now, 30)
            recent_dd = self._recent_drawdown_amount(state, now)
            # Saturation keeps one lucky burst from dictating an enormous size.
            score = .60 * math.tanh(pnl_5 / 5.0)
            if pnl_15 is not None:
                score += .30 * math.tanh(pnl_15 / 10.0)
            if pnl_30 is not None:
                score += .10 * math.tanh(pnl_30 / 15.0)
            score += .25 * math.tanh(recent_dd / 5.0)
            candidates.append((score, account, {
                "pnl_5m": pnl_5, "pnl_15m": pnl_15, "pnl_30m": pnl_30, "recent_dd": recent_dd,
            }))
        if not candidates:
            return {"entry_blocker": "meta allocator: warming up (serve storico rolling di almeno 5 minuti)", "weight": 1.0, "metadata": metadata}
        score, source, metrics = max(candidates, key=lambda item: item[0])
        metadata.update({
            "meta_source_strategy": source.config.strategy_id, "meta_score": score,
            "meta_pnl_5m": metrics["pnl_5m"], "meta_pnl_15m": metrics["pnl_15m"],
            "meta_pnl_30m": metrics["pnl_30m"], "meta_recent_drawdown_15m": metrics["recent_dd"],
        })
        # This is a context safety gate derived from the observed losing SHORT
        # cluster.  It applies only to a new entry, never forces an exit.
        if inference.candidate == "SELL":
            prior_return = (market_context or {}).get("prior_return_15m_pct")
            range_15m = (market_context or {}).get("range_15m_pct")
            if prior_return is not None and prior_return > .001:
                return {"entry_blocker": f"meta allocator: SHORT blocked, M15 trend {prior_return:+.3%}", "weight": 1.0, "metadata": metadata}
            if range_15m is not None and range_15m > .003:
                return {"entry_blocker": f"meta allocator: SHORT blocked, M15 range {range_15m:.3%}", "weight": 1.0, "metadata": metadata}
        if score <= 0:
            return {"entry_blocker": f"meta allocator: fast momentum not positive ({score:+.3f})", "weight": 1.0, "metadata": metadata}
        weight = 1.5 if score >= .50 else 1.0
        metadata["meta_allocation_weight"] = weight
        return {"entry_blocker": None, "weight": weight, "metadata": metadata}

    def signals_frame(self) -> pd.DataFrame:
        if not self.signal_log_path.exists():
            return pd.DataFrame()
        rows = [json.loads(line) for line in self.signal_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)

    def events_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for account in self.accounts.values():
            for event in account.snapshot().get("events", []):
                rows.append({
                    "run_id": self.run_id, "strategy_id": account.config.strategy_id,
                    "strategy": account.model, **event,
                })
        return pd.DataFrame(rows)

    def portfolio_history_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for account in self.accounts.values():
            for snapshot in account.snapshot().get("portfolio_history", []):
                rows.append({"run_id": self.run_id, "strategy": account.model, **snapshot})
        return pd.DataFrame(rows)

    @staticmethod
    def _performance_metrics(state: dict[str, Any], total_pnl: float) -> dict[str, Any]:
        """Return comparable speed metrics without pretending short samples are daily evidence."""
        started = pd.to_datetime(state.get("experiment_started_at"), utc=True, errors="coerce")
        history = pd.DataFrame(state.get("portfolio_history", []))
        if started is pd.NaT or pd.isna(started) or history.empty:
            return {"started_at": state.get("experiment_started_at"), "elapsed_hours": None,
                    "pnl_per_day": None, "pnl_last_hour": None}
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
        if history.empty:
            return {"started_at": state.get("experiment_started_at"), "elapsed_hours": None,
                    "pnl_per_day": None, "pnl_last_hour": None}
        now = history["timestamp"].iloc[-1]
        elapsed_hours = max(0.0, (now - started).total_seconds() / 3600)
        pnl_per_day = total_pnl / (elapsed_hours / 24) if elapsed_hours >= 1 else None
        one_hour_ago = now - pd.Timedelta(hours=1)
        prior = history[history["timestamp"] <= one_hour_ago]
        pnl_last_hour = None
        if not prior.empty:
            row = prior.iloc[-1]
            previous_total = float(row["realized_pnl"]) + float(row["unrealized_pnl"])
            pnl_last_hour = total_pnl - previous_total
        return {
            "started_at": started.isoformat(), "elapsed_hours": elapsed_hours,
            "pnl_per_day": pnl_per_day, "pnl_last_hour": pnl_last_hour,
        }

    def comparison(self) -> pd.DataFrame:
        rows = []
        for name, account in self.accounts.items():
            state = account.snapshot()
            positions = state.get("positions") or ([state["position"]] if state.get("position") else [])
            total_pnl = float(state["realized_pnl"]) + float(state["unrealized_pnl"])
            performance = self._performance_metrics(state, total_pnl)
            rows.append({
                "model": name,
                "run_id": getattr(self, "run_id", "N/D"),
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
                "short_trend_filter_m15_pct": (
                    None if account.config.short_entry_max_prior_return_15m is None
                    else account.config.short_entry_max_prior_return_15m * 100
                ),
                "short_volatility_filter_m15_pct": (
                    None if account.config.short_entry_max_range_15m is None
                    else account.config.short_entry_max_range_15m * 100
                ),
                "position": f"{len(positions)} {positions[0]['side']}" if positions else "FLAT",
                "signal": state["last_signal"],
                "max_drawdown": state["max_drawdown"],
                **performance,
            })
        return pd.DataFrame(rows)

    def write_strategy_exports(self) -> Path:
        """Materialize small, analysis-ready CSVs; the dashboard need not render large ledgers."""
        destination = self.comparison_directory / "exports" / self.run_id
        destination.mkdir(parents=True, exist_ok=True)
        for account in self.accounts.values():
            state = account.snapshot()
            folder = destination / f"{account.config.strategy_id}_{account.model.split('·', 1)[-1].strip().replace(' ', '_')}"
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(state.get("trades", [])).to_csv(folder / "trades.csv", index=False)
            pd.DataFrame(state.get("positions", [])).to_csv(folder / "open_positions.csv", index=False)
            pd.DataFrame(state.get("events", [])).to_csv(folder / "events.csv", index=False)
            pd.DataFrame(state.get("portfolio_history", [])).to_csv(folder / "portfolio_history.csv", index=False)
            pd.DataFrame([{
                **self.comparison().set_index("strategy_id").loc[account.config.strategy_id].to_dict(),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }]).to_csv(folder / "summary.csv", index=False)
        self.signals_frame().to_csv(destination / "signals_m1_shared.csv", index=False)
        self.comparison().to_csv(destination / "comparison_summary.csv", index=False)
        return destination

    def reset_experiment(self) -> str:
        """Start an explicitly new A-L cohort after the caller archived the prior one."""
        with self._lock:
            started_at = datetime.now(timezone.utc).isoformat()
            self.run_id = f"al_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
            self.run_metadata_path.write_text(json.dumps({
                "run_id": self.run_id, "created_at": started_at,
                "experiment": "strategy_lab_0_to_o", "strategies": [item.strategy_id for item in self.STRATEGIES],
                "reset_reason": "user_requested_clean_simultaneous_cohort",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self._run_started_at = pd.Timestamp(started_at)
            self.signal_log_path.write_text("", encoding="utf-8")
            by_id = {item.strategy_id: item for item in self.STRATEGIES}
            for account in self.accounts.values():
                account.config = self._strategy_config(self.config, by_id[account.config.strategy_id])
                account.state = account._new_state()
                account.state["run_id"] = self.run_id
                account.state["experiment_started_at"] = started_at
                account._save()
            self._last_bar = None
            self._inference, self._inferences = None, {}
            self.write_strategy_exports()
            return self.run_id

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
