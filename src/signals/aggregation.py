from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggregationConfig:
    temporal_method: str = "weighted"
    temporal_weights: tuple[float, ...] = (0.40, 0.25, 0.15, 0.12, 0.08)
    temporal_window: int = 5
    ema_alpha: float = 0.40
    voting_threshold: float = 0.50
    horizon_weights: dict[int, float] = field(default_factory=lambda: {
        1: .10, 3: .15, 5: .25, 10: .20, 15: .20, 30: .10,
    })
    minimum_horizon_agreement: float = .60


class SignalAggregator:
    def __init__(self, config: AggregationConfig | None = None):
        self.config = config or AggregationConfig()

    def temporal(self, probabilities: pd.Series) -> pd.Series:
        values = pd.to_numeric(probabilities, errors="coerce")
        method = self.config.temporal_method
        window = self.config.temporal_window
        if method == "weighted":
            weights = np.asarray(self.config.temporal_weights, dtype=float)
            if weights.size == 0 or weights.sum() <= 0:
                raise ValueError("temporal_weights must contain positive weight")
            weights = weights / weights.sum()
            score = sum(values.shift(lag) * weight for lag, weight in enumerate(weights))
            valid = pd.concat([values.shift(lag) for lag in range(len(weights))], axis=1).notna().all(axis=1)
            return score.where(valid)
        if method == "mean":
            return values.rolling(window, min_periods=window).mean()
        if method == "median":
            return values.rolling(window, min_periods=window).median()
        if method == "ema":
            return values.ewm(alpha=self.config.ema_alpha, adjust=False, min_periods=window).mean()
        if method == "voting":
            votes = values.ge(self.config.voting_threshold).astype(float)
            return votes.rolling(window, min_periods=window).mean()
        raise ValueError(f"Unknown temporal method: {method}")

    def multi_horizon(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        available: list[tuple[int, str, float]] = []
        for horizon, weight in self.config.horizon_weights.items():
            candidates = (f"P_up_{horizon}m", f"p_up_{horizon}m", f"p_up_{horizon}m_smoothed")
            column = next((candidate for candidate in candidates if candidate in probabilities), None)
            if column is not None and weight > 0:
                available.append((horizon, column, weight))
        if not available:
            raise ValueError("No configured probability horizon columns are available")
        weights = np.asarray([entry[2] for entry in available], dtype=float)
        weights /= weights.sum()
        matrix = probabilities[[entry[1] for entry in available]].astype(float)
        score = matrix.mul(weights, axis=1).sum(axis=1, min_count=len(available))
        agreement_up = matrix.ge(.5).mul(weights, axis=1).sum(axis=1)
        agreement_down = matrix.le(.5).mul(weights, axis=1).sum(axis=1)
        return pd.DataFrame({
            "multi_horizon_score": score,
            "horizon_agreement_up": agreement_up,
            "horizon_agreement_down": agreement_down,
            "horizon_agreement_passed": np.maximum(agreement_up, agreement_down).ge(self.config.minimum_horizon_agreement),
        }, index=probabilities.index)
