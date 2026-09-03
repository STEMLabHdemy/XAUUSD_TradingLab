from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    bar_minutes: int = 1
    return_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 60)
    atr_windows: tuple[int, ...] = (5, 15, 30, 60)
    volatility_windows: tuple[int, ...] = (5, 15, 30, 60)
    ema_windows: tuple[int, ...] = (5, 10, 20, 50, 100, 200)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    spread_window: int = 60
    warmup_rows: int = 200


class FeatureEngine:
    """Single causal feature implementation for training, backtests, and live bars."""

    INPUT_COLUMNS = {
        "timestamp", "datetime_utc", "mid_open", "mid_high", "mid_low", "mid_close", "spread_close"
    }

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        if self.config.bar_minutes < 1:
            raise ValueError("bar_minutes must be positive")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = self.INPUT_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"FeatureEngine missing columns: {sorted(missing)}")
        if data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
            raise ValueError("Input must have unique, increasing timestamps")

        frame = data.copy()
        close, open_, high, low = (frame[f"mid_{name}"].astype(float) for name in ("close", "open", "high", "low"))
        spread = frame.spread_close.astype(float)
        previous_close = close.shift(1)
        timestamps = pd.to_numeric(frame.timestamp, errors="coerce")

        for horizon in self.config.return_horizons:
            steps = max(1, int(round(horizon / self.config.bar_minutes)))
            actual_minutes = steps * self.config.bar_minutes
            contiguous = timestamps.sub(timestamps.shift(steps)).eq(actual_minutes * 60_000)
            frame[f"return_{horizon}m"] = (close / close.shift(steps) - 1).where(contiguous)
            frame[f"log_return_{horizon}m"] = np.log(close / close.shift(steps)).where(contiguous)

        candle_range = high - low
        signed_body = close - open_
        frame["candle_range"] = candle_range
        frame["candle_range_pct"] = candle_range / close
        frame["candle_body"] = signed_body.abs()
        frame["candle_signed_body"] = signed_body
        frame["candle_upper_wick"] = high - pd.concat([open_, close], axis=1).max(axis=1)
        frame["candle_lower_wick"] = pd.concat([open_, close], axis=1).min(axis=1) - low
        frame["candle_body_to_range"] = signed_body.abs() / candle_range.replace(0, np.nan)
        frame["candle_close_position"] = (close - low) / candle_range.replace(0, np.nan)

        true_range = pd.concat([
            high - low, (high - previous_close).abs(), (low - previous_close).abs()
        ], axis=1).max(axis=1)
        for window in self.config.atr_windows:
            atr = true_range.rolling(window, min_periods=window).mean()
            frame[f"atr_{window}"] = atr
            frame[f"atr_{window}_pct"] = atr / close
        one_minute_log_return = np.log(close / previous_close).where(
            timestamps.diff().eq(self.config.bar_minutes * 60_000)
        )
        for window in self.config.volatility_windows:
            frame[f"rolling_volatility_{window}m"] = one_minute_log_return.rolling(window, min_periods=window).std()

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.config.rsi_window, adjust=False, min_periods=self.config.rsi_window).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.config.rsi_window, adjust=False, min_periods=self.config.rsi_window).mean()
        relative_strength = gain / loss.replace(0, np.nan)
        frame["rsi_14"] = 100 - 100 / (1 + relative_strength)
        ema_fast = close.ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.config.macd_slow, adjust=False).mean()
        frame["macd"] = ema_fast - ema_slow
        frame["macd_signal"] = frame.macd.ewm(span=self.config.macd_signal, adjust=False).mean()
        frame["macd_histogram"] = frame.macd - frame.macd_signal
        contiguous_10m = timestamps.sub(timestamps.shift(10)).eq(10 * 60_000)
        frame["roc_10"] = (close / close.shift(10) - 1).where(contiguous_10m)
        frame["momentum_10"] = (close / close.shift(10) - 1).where(contiguous_10m)

        emas: dict[int, pd.Series] = {}
        for window in self.config.ema_windows:
            emas[window] = close.ewm(span=window, adjust=False, min_periods=window).mean()
            frame[f"ema_distance_{window}"] = close / emas[window] - 1

        bb_middle = close.rolling(self.config.bollinger_window, min_periods=self.config.bollinger_window).mean()
        bb_std = close.rolling(self.config.bollinger_window, min_periods=self.config.bollinger_window).std()
        frame["bollinger_middle"] = bb_middle
        frame["bollinger_upper"] = bb_middle + self.config.bollinger_std * bb_std
        frame["bollinger_lower"] = bb_middle - self.config.bollinger_std * bb_std
        frame["bollinger_width"] = (frame.bollinger_upper - frame.bollinger_lower) / bb_middle
        frame["bollinger_position"] = (close - frame.bollinger_lower) / (frame.bollinger_upper - frame.bollinger_lower).replace(0, np.nan)

        spread_window = self.config.spread_window
        spread_mean = spread.rolling(spread_window, min_periods=spread_window).mean()
        spread_std = spread.rolling(spread_window, min_periods=spread_window).std()
        frame["spread"] = spread
        frame["spread_pct"] = spread / close
        frame["spread_to_atr"] = spread / frame["atr_15"].replace(0, np.nan)
        frame["spread_zscore"] = (spread - spread_mean) / spread_std.replace(0, np.nan)
        frame["spread_percentile"] = spread.rolling(spread_window, min_periods=spread_window).rank(pct=True)

        utc = pd.to_datetime(frame.datetime_utc, utc=True)
        hour = utc.dt.hour
        minute = utc.dt.minute
        frame["hour"] = hour
        frame["minute"] = minute
        frame["weekday"] = utc.dt.weekday
        frame["hour_sin"] = np.sin(2 * np.pi * (hour + minute / 60) / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * (hour + minute / 60) / 24)
        frame["minute_sin"] = np.sin(2 * np.pi * minute / 60)
        frame["minute_cos"] = np.cos(2 * np.pi * minute / 60)
        frame["session_asia"] = ((hour >= 0) & (hour < 8)).astype("int8")
        frame["session_london"] = ((hour >= 7) & (hour < 16)).astype("int8")
        frame["session_new_york"] = ((hour >= 13) & (hour < 22)).astype("int8")
        frame["session_london_new_york_overlap"] = ((hour >= 13) & (hour < 16)).astype("int8")

        trend_strength = emas[20] / emas[50] - 1
        frame["trend_regime"] = np.select(
            [trend_strength > 0.0005, trend_strength < -0.0005], ["bullish", "bearish"], default="neutral"
        )
        volatility = frame["rolling_volatility_60m"]
        trailing_vol_median = volatility.rolling(1440, min_periods=60).median()
        frame["volatility_regime"] = np.where(volatility > trailing_vol_median, "high", "normal")
        frame["range_regime"] = np.where(trend_strength.abs() < 0.0005, "range", "directional")
        return frame

    def valid_rows(self, features: pd.DataFrame) -> pd.Series:
        positions = pd.Series(np.arange(len(features)), index=features.index)
        return positions.ge(self.config.warmup_rows)
