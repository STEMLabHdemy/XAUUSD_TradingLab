from __future__ import annotations

import numpy as np
import pandas as pd


TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30}
OHLC_BASES = ("bid", "ask", "mid")


def aggregate_m1(m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes < 1:
        raise ValueError("minutes must be positive")
    if minutes == 1:
        return m1.copy().reset_index(drop=True)
    if m1.empty:
        return m1.copy()
    frame = m1.sort_values("datetime_utc").drop_duplicates("timestamp", keep="last").copy()
    frame["datetime_utc"] = pd.to_datetime(frame.datetime_utc, utc=True)
    frame["bucket"] = frame.datetime_utc.dt.floor(f"{minutes}min")
    aggregations: dict[str, str] = {
        "spread_close": "last", "spread_points": "last", "tick_volume": "sum",
        "real_volume": "sum", "source": "last", "source_symbol": "last",
        "server_utc_offset_seconds": "last", "raw_server_datetime": "last",
    }
    for base in OHLC_BASES:
        aggregations.update({
            f"open_{base}" if base != "mid" else "mid_open": "first",
            f"high_{base}" if base != "mid" else "mid_high": "max",
            f"low_{base}" if base != "mid" else "mid_low": "min",
            f"close_{base}" if base != "mid" else "mid_close": "last",
        })
    grouped = frame.groupby("bucket", sort=True, observed=True)
    result = grouped.agg(aggregations)
    result["source_rows"] = grouped.size()
    result["all_source_complete"] = grouped.is_complete.all()
    first = grouped.datetime_utc.min()
    last = grouped.datetime_utc.max()
    expected_last = result.index + pd.to_timedelta(minutes - 1, unit="min")
    result["is_complete"] = (
        result.source_rows.eq(minutes)
        & result.all_source_complete
        & first.eq(result.index)
        & last.eq(expected_last)
    )
    result["datetime_utc"] = result.index
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    result["timestamp"] = (result.datetime_utc - epoch) // pd.Timedelta(milliseconds=1)
    return result.reset_index(drop=True).drop(columns=["all_source_complete"])


def chart_bars(m1: pd.DataFrame, timeframe: str, limit: int = 500) -> pd.DataFrame:
    normalized = timeframe.upper()
    if normalized not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported chart timeframe: {timeframe}")
    result = aggregate_m1(m1, TIMEFRAME_MINUTES[normalized])
    if result.empty:
        return result
    keep = result.is_complete.astype(bool)
    keep.iloc[-1] = True
    return result.loc[keep].tail(limit).reset_index(drop=True)


def compare_native_bars(aggregated: pd.DataFrame, native: pd.DataFrame, tolerance: float) -> dict[str, object]:
    complete = aggregated[aggregated.is_complete].copy()
    native_complete = native[native.is_complete].copy()
    joined = complete.merge(native_complete, on="timestamp", suffixes=("_agg", "_native"))
    columns = ("open_bid", "high_bid", "low_bid", "close_bid")
    if joined.empty:
        return {"matched_bars": 0, "max_absolute_error": np.nan, "valid": False}
    errors = [np.abs(joined[f"{column}_agg"] - joined[f"{column}_native"]) for column in columns]
    maximum = float(pd.concat(errors, ignore_index=True).max())
    return {"matched_bars": len(joined), "max_absolute_error": maximum, "valid": maximum <= tolerance}
