from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import DataConfig, load_config
from .io import read_side_csv
from .pipeline import merge_bid_ask
from .market_hours import filter_market_closed
from .validation import _gap_counts, validate_side_frame, write_quality_report


def _month_keys(start: date, end: date) -> list[str]:
    month_start = pd.Timestamp(start).replace(day=1)
    return [stamp.strftime("%Y_%m") for stamp in pd.date_range(month_start, end, freq="MS", inclusive="left")]


def build_master_snapshot(
    project_root: Path | str | None,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Build one canonical Parquet incrementally from immutable paired months."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required. Run: python -m pip install -r requirements.txt") from exc

    config: DataConfig = load_config(project_root)
    destination = config.path(config.master_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    keys = _month_keys(start, end)
    bid_files = {p.stem[-7:]: p for p in config.path(config.raw_bid_dir).glob("xauusd_bid_m1_????_??.csv")}
    ask_files = {p.stem[-7:]: p for p in config.path(config.raw_ask_dir).glob("xauusd_ask_m1_????_??.csv")}
    missing_bid_months = [key for key in keys if key not in bid_files]
    missing_ask_months = [key for key in keys if key not in ask_files]
    if missing_bid_months or missing_ask_months:
        raise RuntimeError(
            f"Snapshot range is not fully paired; missing BID={missing_bid_months}, ASK={missing_ask_months}"
        )

    writer = None
    spreads: list[np.ndarray] = []
    total_rows = duplicates = missing_bid = missing_ask = 0
    invalid_ohlc = nan_ohlc = nonpositive = negative_spreads = 0
    unexpected_gaps = weekend_gaps = weekday_gaps = 0
    min_price = float("inf")
    max_price = float("-inf")
    earliest = latest = previous_last = None
    excluded_market_closed_rows = 0
    monthly_rows: list[dict[str, object]] = []
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    try:
        for key in keys:
            bid = read_side_csv(bid_files[key], "bid").assign(_source_file=bid_files[key].name)
            ask = read_side_csv(ask_files[key], "ask").assign(_source_file=ask_files[key].name)
            bid_report = validate_side_frame(bid, "bid")
            ask_report = validate_side_frame(ask, "ask")
            if not bid_report.valid or not ask_report.valid:
                raise RuntimeError(f"Invalid raw month {key}: BID={bid_report}, ASK={ask_report}")
            merged = merge_bid_ask(bid, ask)
            merged = merged[(merged.datetime_utc >= start_ts) & (merged.datetime_utc < end_ts)]
            merged, excluded_closed = filter_market_closed(merged)
            excluded_market_closed_rows += excluded_closed
            if merged.empty:
                continue

            duplicates += bid_report.duplicates + ask_report.duplicates
            invalid_ohlc += bid_report.invalid_ohlc + ask_report.invalid_ohlc
            nan_ohlc += bid_report.nan_ohlc + ask_report.nan_ohlc
            nonpositive += bid_report.nonpositive_prices + ask_report.nonpositive_prices
            missing_bid += int(merged._merge.eq("right_only").sum())
            missing_ask += int(merged._merge.eq("left_only").sum())
            monthly_rows.append({
                "month": key.replace("_", "-"),
                "bid_rows": len(bid), "ask_rows": len(ask), "merged_rows": len(merged),
                "missing_bid_timestamps": int(merged._merge.eq("right_only").sum()),
                "missing_ask_timestamps": int(merged._merge.eq("left_only").sum()),
                "bid_duplicates": bid_report.duplicates, "ask_duplicates": ask_report.duplicates,
                "bid_invalid_ohlc": bid_report.invalid_ohlc, "ask_invalid_ohlc": ask_report.invalid_ohlc,
                "bid_weekday_gaps": bid_report.weekday_gaps, "ask_weekday_gaps": ask_report.weekday_gaps,
                "earliest_candle": merged.datetime_utc.min(), "latest_candle": merged.datetime_utc.max(),
                "excluded_market_closed_rows": excluded_closed,
            })
            spread = merged.spread_close.dropna()
            spreads.append(spread.to_numpy(dtype="float64"))
            negative_spreads += int(spread.lt(0).sum())
            prices = merged.filter(regex=r"^(open|high|low|close)_(bid|ask)$").to_numpy(dtype="float64")
            min_price = min(min_price, float(np.nanmin(prices)))
            max_price = max(max_price, float(np.nanmax(prices)))

            gap_input = merged.datetime_utc
            if previous_last is not None:
                gap_input = pd.concat([pd.Series([previous_last]), gap_input], ignore_index=True)
            gaps = _gap_counts(gap_input)
            unexpected_gaps += gaps[0]
            weekend_gaps += gaps[1]
            weekday_gaps += gaps[2]
            previous_last = merged.datetime_utc.iloc[-1]
            earliest = merged.datetime_utc.iloc[0] if earliest is None else min(earliest, merged.datetime_utc.iloc[0])
            latest = merged.datetime_utc.iloc[-1] if latest is None else max(latest, merged.datetime_utc.iloc[-1])

            clean = merged.drop(columns=["_merge", "_source_file_x", "_source_file_y"], errors="ignore")
            table = pa.Table.from_pandas(clean, preserve_index=False)
            metadata = {
                b"instrument": b"XAUUSD", b"timeframe": b"M1", b"source": b"Dukascopy",
                b"price_sides": b"BID,ASK", b"snapshot_start_utc": start.isoformat().encode(),
                b"snapshot_end_utc_exclusive": end.isoformat().encode(),
            }
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(clean)
        if writer is None:
            raise RuntimeError("No rows were available for the requested snapshot")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)

    spread_all = np.concatenate(spreads) if spreads else np.array([], dtype="float64")
    median_spread = float(np.nanmedian(spread_all)) if spread_all.size else float("nan")
    mad = float(np.nanmedian(np.abs(spread_all - median_spread))) if spread_all.size else float("nan")
    extreme_limit = median_spread + 10 * mad
    years = (latest - earliest).total_seconds() / (365.25 * 86400) if earliest is not None and latest is not None else 0.0
    summary = pd.DataFrame([{
        "earliest_candle": earliest, "latest_candle": latest, "number_of_candles": total_rows,
        "number_of_years": years, "duplicates": duplicates,
        "missing_bid_timestamps": missing_bid, "missing_ask_timestamps": missing_ask,
        "invalid_ohlc_rows": invalid_ohlc, "nan_ohlc_rows": nan_ohlc,
        "negative_or_zero_price_rows": nonpositive, "negative_spreads": negative_spreads,
        "extreme_spreads": int(np.sum(spread_all > extreme_limit)) if spread_all.size else 0,
        "median_spread": median_spread,
        "spread_p95": float(np.nanquantile(spread_all, .95)) if spread_all.size else float("nan"),
        "spread_p99": float(np.nanquantile(spread_all, .99)) if spread_all.size else float("nan"),
        "max_spread": float(np.nanmax(spread_all)) if spread_all.size else float("nan"),
        "min_price": min_price, "max_price": max_price,
        "unexpected_gaps": unexpected_gaps, "weekend_gaps": weekend_gaps,
        "weekday_gaps": weekday_gaps, "corrupt_files": 0,
        "missing_bid_months": 0, "missing_ask_months": 0,
        "excluded_market_closed_rows": excluded_market_closed_rows,
        "snapshot_start_utc": start.isoformat(), "snapshot_end_utc_exclusive": end.isoformat(),
    }])
    write_quality_report(summary, config.path(config.quality_report_path))
    monthly_path = config.path("reports/data_quality_monthly.csv")
    monthly_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    return summary
