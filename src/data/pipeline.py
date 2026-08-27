from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig, load_config
from .io import atomic_write_parquet, read_side_directory, read_side_files
from .validation import data_quality_summary, validate_side_frame, write_quality_report
from .market_hours import filter_market_closed


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    result = frame.sort_values(["timestamp", "_source_file"], kind="stable").drop_duplicates("timestamp", keep="last")
    return result.sort_values("timestamp").reset_index(drop=True), before - len(result)


def merge_bid_ask(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    bid_clean, _ = deduplicate(bid)
    ask_clean, _ = deduplicate(ask)
    bid_clean = bid_clean.drop(columns=["datetime_utc"], errors="ignore")
    ask_clean = ask_clean.drop(columns=["datetime_utc"], errors="ignore")
    merged = bid_clean.merge(ask_clean, on="timestamp", how="outer", validate="one_to_one", indicator=True)
    merged["datetime_utc"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True, errors="coerce")
    for name in ("open", "high", "low", "close"):
        merged[f"mid_{name}"] = (merged[f"{name}_bid"] + merged[f"{name}_ask"]) / 2.0
    merged["spread_open"] = merged["open_ask"] - merged["open_bid"]
    merged["spread_close"] = merged["close_ask"] - merged["close_bid"]
    ordered = [
        "timestamp", "datetime_utc", "open_bid", "high_bid", "low_bid", "close_bid",
        "open_ask", "high_ask", "low_ask", "close_ask", "spread_open", "spread_close",
        "mid_open", "mid_high", "mid_low", "mid_close",
    ]
    volume_columns = [column for column in ("volume_bid", "volume_ask") if column in merged]
    audit_columns = [column for column in ("_source_file_x", "_source_file_y", "_merge") if column in merged]
    return merged[[*ordered, *volume_columns, *audit_columns]].sort_values("timestamp").reset_index(drop=True)


def load_validate_merge(config: DataConfig) -> tuple[pd.DataFrame, object, object, list[dict[str, str]]]:
    bid, bid_errors = read_side_directory(config.path(config.raw_bid_dir), "bid")
    ask, ask_errors = read_side_directory(config.path(config.raw_ask_dir), "ask")
    if bid.empty or ask.empty:
        raise RuntimeError("Both BID and ASK CSV data are required")
    bid_report = validate_side_frame(bid, "bid")
    ask_report = validate_side_frame(ask, "ask")
    merged = merge_bid_ask(bid, ask)
    return merged, bid_report, ask_report, [*bid_errors, *ask_errors]


def build_master(project_root: Path | str | None = None) -> pd.DataFrame:
    config = load_config(project_root)
    merged, bid_report, ask_report, errors = load_validate_merge(config)
    merged, excluded_closed = filter_market_closed(merged)
    summary = data_quality_summary(merged, bid_report, ask_report, errors)
    summary["excluded_market_closed_rows"] = excluded_closed
    write_quality_report(summary, config.path(config.quality_report_path))
    atomic_write_parquet(merged.drop(columns=["_merge", "_source_file_x", "_source_file_y"], errors="ignore"), config.path(config.master_path))
    return summary


def update_history(project_root: Path | str | None = None) -> pd.DataFrame:
    """Download only missing/recent months, then atomically refresh that master slice."""
    from .download import DownloadManager

    config = load_config(project_root)
    master_path = config.path(config.master_path)
    if not master_path.exists():
        raise FileNotFoundError("Historical database has not been created. Run full historical download first.")
    master = pd.read_parquet(master_path)
    if master.empty:
        raise RuntimeError("Master Parquet is empty; a full historical build is required")
    latest = pd.to_datetime(master["datetime_utc"].max(), utc=True)
    latest_month = pd.Timestamp(year=latest.year, month=latest.month, day=1, tz="UTC")
    start = latest_month - pd.DateOffset(months=config.incremental_overlap_months)
    manager = DownloadManager(config)
    try:
        manager.download_range(start.date(), pd.Timestamp.now(tz="UTC").date(), allow_skip=False)
    finally:
        manager.close()

    months = pd.date_range(start.normalize(), pd.Timestamp.now(tz="UTC").normalize(), freq="MS")
    bid_paths = [config.path(config.raw_bid_dir) / f"xauusd_bid_m1_{month:%Y_%m}.csv" for month in months]
    ask_paths = [config.path(config.raw_ask_dir) / f"xauusd_ask_m1_{month:%Y_%m}.csv" for month in months]
    bid, bid_errors = read_side_files([path for path in bid_paths if path.exists()], "bid")
    ask, ask_errors = read_side_files([path for path in ask_paths if path.exists()], "ask")
    if bid.empty or ask.empty:
        raise RuntimeError("Incremental download produced no usable paired BID/ASK data")
    recent = merge_bid_ask(bid, ask)
    recent = recent[recent["datetime_utc"] >= start]
    recent, excluded_closed = filter_market_closed(recent)
    preserved = master[pd.to_datetime(master["datetime_utc"], utc=True) < start]
    clean_recent = recent.drop(columns=["_merge", "_source_file_x", "_source_file_y"], errors="ignore")
    updated = pd.concat([preserved, clean_recent], ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    atomic_write_parquet(updated, master_path)
    bid_report = validate_side_frame(updated, "bid")
    ask_report = validate_side_frame(updated, "ask")
    summary = data_quality_summary(updated, bid_report, ask_report, [*bid_errors, *ask_errors])
    summary["excluded_market_closed_rows"] = excluded_closed
    write_quality_report(summary, config.path(config.quality_report_path))
    return summary
