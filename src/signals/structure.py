"""Causal market-structure trend/pullback signals.

This module deliberately does not predict a future candle.  It confirms M15
swings only after an ATR-sized reversal, maps the confirmed structure to M1,
then seeks a pullback recovery in that existing direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FeatureEngine


def _m15_structure(bars: pd.DataFrame, reversal_atr: float = 1.25) -> pd.DataFrame:
    raw = bars.copy()
    raw["datetime_utc"] = pd.to_datetime(raw.datetime_utc, utc=True)
    frame = raw.set_index("datetime_utc")
    m15 = frame.resample("15min", label="right", closed="right").agg(
        high=("mid_high", "max"), low=("mid_low", "min"), close=("mid_close", "last")
    ).dropna()
    previous = m15.close.shift(1)
    true_range = pd.concat([m15.high-m15.low, (m15.high-previous).abs(), (m15.low-previous).abs()], axis=1).max(axis=1)
    m15["atr"] = true_range.rolling(14, min_periods=14).mean()
    highs: list[float] = []; lows: list[float] = []
    direction = 0; peak = trough = np.nan
    trend: list[str] = []; last_high: list[float] = []; last_low: list[float] = []
    for row in m15.itertuples():
        atr = float(row.atr) if pd.notna(row.atr) else np.nan
        if not np.isfinite(atr) or atr <= 0:
            trend.append("UNKNOWN"); last_high.append(np.nan); last_low.append(np.nan); continue
        if direction == 0:
            peak, trough, direction = float(row.high), float(row.low), 1
        elif direction > 0:
            peak = max(float(peak), float(row.high))
            if float(row.close) <= peak - reversal_atr * atr:
                highs.append(float(peak)); trough = float(row.low); direction = -1
        else:
            trough = min(float(trough), float(row.low))
            if float(row.close) >= trough + reversal_atr * atr:
                lows.append(float(trough)); peak = float(row.high); direction = 1
        up = len(highs) >= 2 and len(lows) >= 2 and highs[-1] > highs[-2] and lows[-1] > lows[-2]
        down = len(highs) >= 2 and len(lows) >= 2 and highs[-1] < highs[-2] and lows[-1] < lows[-2]
        trend.append("UP" if up else "DOWN" if down else "RANGE")
        last_high.append(highs[-1] if highs else np.nan); last_low.append(lows[-1] if lows else np.nan)
    m15["structure_trend"] = trend; m15["swing_high"] = last_high; m15["swing_low"] = last_low
    # Keep the completed M15 OHLC here too: the entry rule below must be
    # evaluated on a finished M15 candle, never on an evolving M1 fragment.
    return m15[["high", "low", "close", "structure_trend", "swing_high", "swing_low", "atr"]]


def structure_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """Return one causal I42 decision per M1 row, plus diagnostics for audit."""
    features = FeatureEngine().transform(bars).copy()
    structural = _m15_structure(features)
    timed = pd.to_datetime(features.datetime_utc, utc=True)
    # The prior M1 implementation was too permissive: every tiny 3-minute
    # bounce could become a trade.  Structure and pullback confirmation now
    # both happen on completed M15 candles; M1 is only the execution clock.
    m15 = structural
    e20 = m15.close.ewm(span=20, adjust=False).mean()
    recent_touch_long = m15.low.le(e20 + .15*m15.atr).rolling(3).max().shift(1).fillna(0).astype(bool)
    recent_touch_short = m15.high.ge(e20 - .15*m15.atr).rolling(3).max().shift(1).fillna(0).astype(bool)
    recover_long = m15.close.gt(m15.high.shift(1).rolling(2).max()) & m15.close.gt(e20)
    recover_short = m15.close.lt(m15.low.shift(1).rolling(2).min()) & m15.close.lt(e20)
    safe_long = m15.swing_low.notna() & m15.close.gt(m15.swing_low + .30*m15.atr)
    safe_short = m15.swing_high.notna() & m15.close.lt(m15.swing_high - .30*m15.atr)
    long_m15 = m15.structure_trend.eq("UP") & recent_touch_long & recover_long & safe_long
    short_m15 = m15.structure_trend.eq("DOWN") & recent_touch_short & recover_short & safe_short
    # An entry is an event, not a persistent state.  One M15 recovery can
    # generate at most one candidate trade.
    long_m15 &= ~long_m15.shift(1, fill_value=False)
    short_m15 &= ~short_m15.shift(1, fill_value=False)
    event = pd.DataFrame({"long": long_m15, "short": short_m15, "trend": m15.structure_trend,
                          "swing_high": m15.swing_high, "swing_low": m15.swing_low, "atr": m15.atr}).reindex(timed)
    lookup = structural.reindex(timed, method="ffill").reset_index(drop=True)
    close = features.mid_close; atr = features.atr_15
    long = event["long"].fillna(False).to_numpy(dtype=bool)
    short = event["short"].fillna(False).to_numpy(dtype=bool)
    signal = pd.Series("HOLD", index=features.index)
    signal.loc[long] = "BUY"; signal.loc[short] = "SELL"
    distance = pd.Series(np.nan, index=features.index)
    distance.loc[long] = (close.loc[long] - event.swing_low.loc[long].to_numpy()) / atr.loc[long]
    distance.loc[short] = (event.swing_high.loc[short].to_numpy() - close.loc[short]) / atr.loc[short]
    power = (distance.clip(0, 2.5) / 2.5 * 100).where(signal.ne("HOLD"))
    output = features[["datetime_utc", "timestamp", "mid_open", "mid_high", "mid_low", "mid_close", "spread_close"]].copy()
    output["signal"] = signal; output["power"] = power; output["structure_trend"] = lookup.structure_trend
    output["swing_high"] = lookup.swing_high; output["swing_low"] = lookup.swing_low; output["atr_15"] = atr
    output["reason"] = np.where(long, "M15 higher-high/higher-low + pullback recovery", np.where(short, "M15 lower-high/lower-low + pullback recovery", "structure/pullback not confirmed"))
    return output
