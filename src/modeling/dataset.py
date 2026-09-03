from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.features import FeatureEngine
from src.targets import TargetConfig, TargetEngine


def load_recent_rows(master_path: Path | str, max_rows: int = 100_000) -> pd.DataFrame:
    parquet = pq.ParquetFile(master_path)
    groups: list[int] = []
    rows = 0
    for index in range(parquet.num_row_groups - 1, -1, -1):
        groups.append(index)
        rows += parquet.metadata.row_group(index).num_rows
        if rows >= max_rows:
            break
    table = parquet.read_row_groups(sorted(groups))
    return table.to_pandas().tail(max_rows).reset_index(drop=True)


def aggregate_training_bars(m1: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Create complete, gap-safe M5/M15 research bars from canonical M1 data."""
    if timeframe_minutes < 1:
        raise ValueError("timeframe_minutes must be positive")
    frame = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
    if timeframe_minutes == 1 or frame.empty:
        return frame.reset_index(drop=True)
    frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True)
    frame["_bucket"] = frame.datetime_utc.dt.floor(f"{timeframe_minutes}min")
    aggregations: dict[str, str] = {"spread_close": "last"}
    for side in ("bid", "ask"):
        aggregations.update({
            f"open_{side}": "first", f"high_{side}": "max",
            f"low_{side}": "min", f"close_{side}": "last",
        })
    aggregations.update({"mid_open": "first", "mid_high": "max", "mid_low": "min", "mid_close": "last"})
    if "spread_open" in frame:
        aggregations["spread_open"] = "first"
    if "source" in frame:
        aggregations["source"] = "last"
    grouped = frame.groupby("_bucket", sort=True, observed=True)
    result = grouped.agg(aggregations)
    first, last = grouped.datetime_utc.min(), grouped.datetime_utc.max()
    complete = (
        grouped.size().eq(timeframe_minutes)
        & first.eq(result.index)
        & last.eq(result.index + pd.to_timedelta(timeframe_minutes - 1, unit="min"))
    )
    result = result.loc[complete].copy()
    result["datetime_utc"] = result.index
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    result["timestamp"] = ((result.datetime_utc - epoch) // pd.Timedelta(milliseconds=1)).astype("int64")
    return result.reset_index(drop=True)


def prepare_binary_dataset(
    market: pd.DataFrame, horizon: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    featured = FeatureEngine().transform(market)
    labelled = TargetEngine(TargetConfig(horizons=(horizon,))).transform(featured)
    target_name = f"up_{horizon}m"
    original_columns = set(market.columns)
    feature_columns = [
        column for column in featured.columns
        if column not in original_columns and pd.api.types.is_numeric_dtype(featured[column])
    ]
    usable = labelled[target_name].notna() & labelled[feature_columns].notna().any(axis=1)
    return labelled.loc[usable, feature_columns].reset_index(drop=True), labelled.loc[usable, target_name].reset_index(drop=True), labelled.loc[usable].reset_index(drop=True)
