from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(ttl=30)
def load_quality_summary() -> pd.DataFrame:
    path = ROOT / "reports/data_quality_report.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def load_quality_monthly() -> pd.DataFrame:
    path = ROOT / "reports/data_quality_monthly.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def load_experiments() -> pd.DataFrame:
    path = ROOT / "results/experiments.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def load_model_metrics() -> pd.DataFrame:
    path = ROOT / "results/baseline_metrics_provisional.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def load_recent_market(rows: int = 500) -> pd.DataFrame:
    path = ROOT / "data/processed/XAUUSD_M1_MASTER.parquet"
    if not path.exists():
        return pd.DataFrame()
    parquet = pq.ParquetFile(path)
    groups: list[int] = []
    count = 0
    for index in range(parquet.num_row_groups - 1, -1, -1):
        groups.append(index)
        count += parquet.metadata.row_group(index).num_rows
        if count >= rows:
            break
    return parquet.read_row_groups(sorted(groups)).to_pandas().tail(rows).reset_index(drop=True)


@st.cache_data(ttl=30)
def load_market_range(minimum_timestamp: int, maximum_timestamp: int) -> pd.DataFrame:
    path = ROOT / "data/processed/XAUUSD_M1_MASTER.parquet"
    return pd.read_parquet(path, filters=[("timestamp", ">=", minimum_timestamp), ("timestamp", "<=", maximum_timestamp)])


@st.cache_data(ttl=30)
def load_run(experiment_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = ROOT / "results" / "runs" / experiment_id
    predictions = pd.read_parquet(directory / "predictions.parquet") if (directory / "predictions.parquet").exists() else pd.DataFrame()
    trades = pd.read_parquet(directory / "trades.parquet") if (directory / "trades.parquet").exists() else pd.DataFrame()
    equity = pd.read_parquet(directory / "equity.parquet") if (directory / "equity.parquet").exists() else pd.DataFrame()
    return predictions, trades, equity


def download_progress() -> dict[str, object]:
    bid = {path.stem[-7:] for path in (ROOT / "data/raw/bid").glob("xauusd_bid_m1_????_??.csv")}
    ask = {path.stem[-7:] for path in (ROOT / "data/raw/ask").glob("xauusd_ask_m1_????_??.csv")}
    paired = sorted(bid & ask)
    return {
        "paired_months": len(paired),
        "earliest_month": paired[0].replace("_", "-") if paired else None,
        "latest_month": paired[-1].replace("_", "-") if paired else None,
        "unpaired_months": len(bid ^ ask),
    }
