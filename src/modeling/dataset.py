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
