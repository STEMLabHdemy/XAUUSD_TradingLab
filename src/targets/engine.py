from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetConfig:
    horizons: tuple[int, ...] = (1, 3, 5, 10, 15, 30, 60)
    neutral_cost_multiplier: float = 1.25
    executable_minimum_net_move: float = 0.50
    slippage_price_per_side: float = 0.05


class TargetEngine:
    """Create forward labels; these columns must never be model features."""

    def __init__(self, config: TargetConfig | None = None):
        self.config = config or TargetConfig()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "mid_close", "spread_close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"TargetEngine missing columns: {sorted(missing)}")
        result = data.copy()
        close = result.mid_close.astype(float)
        timestamp = pd.to_numeric(result.timestamp, errors="coerce")
        neutral_threshold = (result.spread_close.astype(float) / close) * self.config.neutral_cost_multiplier
        for horizon in self.config.horizons:
            future_close = close.shift(-horizon)
            future_return = future_close / close - 1
            future_timestamp = timestamp.shift(-horizon)
            known = future_close.notna() & future_timestamp.sub(timestamp).eq(horizon * 60_000)
            result[f"future_return_{horizon}m"] = future_return.where(known)
            binary = pd.Series(pd.NA, index=result.index, dtype="Int8")
            binary.loc[known] = future_return.loc[known].gt(0).astype("int8")
            result[f"up_{horizon}m"] = binary
            labels = pd.Series(pd.NA, index=result.index, dtype="string")
            labels.loc[known & future_return.gt(neutral_threshold)] = "UP"
            labels.loc[known & future_return.lt(-neutral_threshold)] = "DOWN"
            labels.loc[known & future_return.le(neutral_threshold) & future_return.ge(-neutral_threshold)] = "NEUTRAL"
            result[f"direction_{horizon}m"] = labels
            execution_columns = {"open_bid", "open_ask", "close_bid", "close_ask"}
            if execution_columns.issubset(result.columns):
                next_timestamp = timestamp.shift(-1)
                executable = known & next_timestamp.sub(timestamp).eq(60_000)
                round_trip_slippage = 2 * self.config.slippage_price_per_side
                long_net = (
                    result.close_bid.astype(float).shift(-horizon)
                    - result.open_ask.astype(float).shift(-1)
                    - round_trip_slippage
                )
                short_net = (
                    result.open_bid.astype(float).shift(-1)
                    - result.close_ask.astype(float).shift(-horizon)
                    - round_trip_slippage
                )
                result[f"future_long_net_{horizon}m"] = long_net.where(executable)
                result[f"future_short_net_{horizon}m"] = short_net.where(executable)
                trade_labels = pd.Series(pd.NA, index=result.index, dtype="string")
                trade_labels.loc[executable] = "NEUTRAL"
                trade_labels.loc[
                    executable
                    & long_net.gt(self.config.executable_minimum_net_move)
                    & long_net.ge(short_net)
                ] = "UP"
                trade_labels.loc[
                    executable
                    & short_net.gt(self.config.executable_minimum_net_move)
                    & short_net.gt(long_net)
                ] = "DOWN"
                result[f"executable_direction_{horizon}m"] = trade_labels
        return result
