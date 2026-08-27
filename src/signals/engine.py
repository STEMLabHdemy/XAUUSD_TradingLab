from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class PositionState(str, Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class SignalConfig:
    buy_threshold: float = .68
    sell_threshold: float = .32
    persistence: int = 2
    cooldown_minutes: int = 3
    probability_exit_threshold: float = .50
    require_expected_return_filter: bool = False
    expected_return_safety_margin: float = 1.25
    commission_price_equivalent: float = 0.0
    slippage_price_per_side: float = .05
    max_spread: float = 5.0
    allowed_sessions: tuple[str, ...] = ()
    allowed_trend_regimes: tuple[str, ...] = ()
    allowed_volatility_regimes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    timestamp: pd.Timestamp
    signal: str
    position_before: str
    position_after: str
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class SignalEngine:
    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        if not 0 <= self.config.sell_threshold < self.config.buy_threshold <= 1:
            raise ValueError("Require 0 <= sell_threshold < buy_threshold <= 1")
        if self.config.persistence < 1:
            raise ValueError("persistence must be at least 1")
        self.position = PositionState.FLAT
        self.pending_direction: str | None = None
        self.persistence_count = 0
        self.last_exit_time: pd.Timestamp | None = None

    def reset(self) -> None:
        self.position = PositionState.FLAT
        self.pending_direction = None
        self.persistence_count = 0
        self.last_exit_time = None

    def _exit(self, timestamp: pd.Timestamp, score: float, reason: str) -> Decision:
        before = self.position.value
        self.position = PositionState.FLAT
        self.last_exit_time = timestamp
        self.pending_direction = None
        self.persistence_count = 0
        return Decision(timestamp, "EXIT", before, "FLAT", score, (reason,))

    def decide(
        self,
        timestamp: pd.Timestamp,
        score: float,
        price: float,
        spread: float,
        expected_return: float | None = None,
        session: str | None = None,
        trend_regime: str | None = None,
        volatility_regime: str | None = None,
        agreement_passed: bool = True,
    ) -> Decision:
        timestamp = pd.Timestamp(timestamp)
        before = self.position.value
        if pd.isna(score):
            return Decision(timestamp, "HOLD" if self.position != PositionState.FLAT else "NO_TRADE", before, before, score, ("score unavailable",))

        if self.position == PositionState.LONG:
            if score <= self.config.sell_threshold:
                return self._exit(timestamp, score, "strong reversal from LONG")
            if score < self.config.probability_exit_threshold:
                return self._exit(timestamp, score, "probability exit from LONG")
            return Decision(timestamp, "HOLD", before, before, score, ("LONG remains supported",))
        if self.position == PositionState.SHORT:
            if score >= self.config.buy_threshold:
                return self._exit(timestamp, score, "strong reversal from SHORT")
            if score > self.config.probability_exit_threshold:
                return self._exit(timestamp, score, "probability exit from SHORT")
            return Decision(timestamp, "HOLD", before, before, score, ("SHORT remains supported",))

        candidate = "BUY" if score >= self.config.buy_threshold else "SELL" if score <= self.config.sell_threshold else None
        if candidate is None:
            self.pending_direction = None
            self.persistence_count = 0
            return Decision(timestamp, "NO_TRADE", before, before, score, ("inside no-trade zone",))

        blockers: list[str] = []
        if spread > self.config.max_spread:
            blockers.append("spread exceeds maximum")
        if not agreement_passed:
            blockers.append("horizon agreement failed")
        if self.config.allowed_sessions and session not in self.config.allowed_sessions:
            blockers.append("session filtered")
        if self.config.allowed_trend_regimes and trend_regime not in self.config.allowed_trend_regimes:
            blockers.append("trend regime filtered")
        if self.config.allowed_volatility_regimes and volatility_regime not in self.config.allowed_volatility_regimes:
            blockers.append("volatility regime filtered")
        if self.last_exit_time is not None and timestamp < self.last_exit_time + pd.Timedelta(minutes=self.config.cooldown_minutes):
            blockers.append("cooldown active")
        if self.config.require_expected_return_filter:
            if expected_return is None or pd.isna(expected_return):
                blockers.append("expected return unavailable")
            else:
                expected_move = abs(expected_return) * price
                cost = (spread + 2 * self.config.slippage_price_per_side + self.config.commission_price_equivalent)
                if expected_move <= cost * self.config.expected_return_safety_margin:
                    blockers.append("expected move does not exceed costs")
        if blockers:
            self.pending_direction = None
            self.persistence_count = 0
            return Decision(timestamp, "NO_TRADE", before, before, score, tuple(blockers))

        if self.pending_direction == candidate:
            self.persistence_count += 1
        else:
            self.pending_direction = candidate
            self.persistence_count = 1
        if self.persistence_count < self.config.persistence:
            return Decision(timestamp, "NO_TRADE", before, before, score, (f"confirmation {self.persistence_count}/{self.config.persistence}",))

        self.position = PositionState.LONG if candidate == "BUY" else PositionState.SHORT
        self.pending_direction = None
        self.persistence_count = 0
        reasons = ("confidence threshold passed", f"{self.config.persistence} confirmations", "spread acceptable", "cooldown satisfied")
        return Decision(timestamp, candidate, before, self.position.value, score, reasons)
