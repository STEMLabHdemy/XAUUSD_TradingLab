from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class SideValidation:
    side: str
    rows: int
    duplicates: int
    invalid_timestamps: int
    nan_ohlc: int
    invalid_ohlc: int
    nonpositive_prices: int
    unexpected_gaps: int
    weekend_gaps: int
    weekday_gaps: int

    @property
    def valid(self) -> bool:
        return not any((self.invalid_timestamps, self.nan_ohlc, self.invalid_ohlc, self.nonpositive_prices))


def _gap_counts(datetimes: pd.Series) -> tuple[int, int, int]:
    stamps = datetimes.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    deltas = stamps.diff()
    gap_positions = list(deltas[deltas > pd.Timedelta(minutes=1)].index)
    unexpected = len(gap_positions)
    weekend = 0
    for position in gap_positions:
        previous, end = stamps.iloc[position - 1], stamps.iloc[position]
        days = pd.date_range(previous.normalize(), end.normalize(), freq="D")
        weekend += int(any(day.weekday() >= 5 for day in days))
    return unexpected, weekend, unexpected - weekend


def validate_side_frame(frame: pd.DataFrame, side: str) -> SideValidation:
    prices = [f"{name}_{side}" for name in ("open", "high", "low", "close")]
    missing = set(["timestamp", "datetime_utc", *prices]).difference(frame.columns)
    if missing:
        raise ValueError(f"missing {side} columns: {sorted(missing)}")
    high, low, open_, close = (frame[f"{name}_{side}"] for name in ("high", "low", "open", "close"))
    invalid_ohlc = (high.lt(low) | high.lt(open_) | high.lt(close) | low.gt(open_) | low.gt(close))
    gaps = _gap_counts(frame["datetime_utc"])
    return SideValidation(
        side=side,
        rows=len(frame),
        duplicates=int(frame["timestamp"].duplicated(keep=False).sum()),
        invalid_timestamps=int(frame["datetime_utc"].isna().sum()),
        nan_ohlc=int(frame[prices].isna().any(axis=1).sum()),
        invalid_ohlc=int(invalid_ohlc.sum()),
        nonpositive_prices=int(frame[prices].le(0).any(axis=1).sum()),
        unexpected_gaps=gaps[0],
        weekend_gaps=gaps[1],
        weekday_gaps=gaps[2],
    )


def data_quality_summary(
    merged: pd.DataFrame,
    bid_report: SideValidation,
    ask_report: SideValidation,
    file_errors: Iterable[dict[str, str]] = (),
) -> pd.DataFrame:
    dates = merged["datetime_utc"].dropna()
    spread = merged.get("spread_close", pd.Series(dtype=float)).dropna()
    all_prices = merged.filter(regex=r"^(open|high|low|close)_(bid|ask)$").stack()
    negative_spreads = int(spread.lt(0).sum())
    median_spread = spread.median() if not spread.empty else float("nan")
    mad = (spread - median_spread).abs().median() if not spread.empty else float("nan")
    extreme_limit = median_spread + 10 * mad if pd.notna(mad) else float("nan")
    years = (dates.max() - dates.min()).total_seconds() / (365.25 * 86400) if len(dates) > 1 else 0.0
    row = {
        "earliest_candle": dates.min(),
        "latest_candle": dates.max(),
        "number_of_candles": len(merged),
        "number_of_years": years,
        "duplicates": bid_report.duplicates + ask_report.duplicates,
        "missing_bid_timestamps": int(
            merged["_merge"].eq("right_only").sum() if "_merge" in merged
            else merged.get("open_bid", pd.Series(dtype=float)).isna().sum()
        ),
        "missing_ask_timestamps": int(
            merged["_merge"].eq("left_only").sum() if "_merge" in merged
            else merged.get("open_ask", pd.Series(dtype=float)).isna().sum()
        ),
        "invalid_ohlc_rows": bid_report.invalid_ohlc + ask_report.invalid_ohlc,
        "nan_ohlc_rows": bid_report.nan_ohlc + ask_report.nan_ohlc,
        "negative_or_zero_price_rows": bid_report.nonpositive_prices + ask_report.nonpositive_prices,
        "negative_spreads": negative_spreads,
        "extreme_spreads": int(spread.gt(extreme_limit).sum()) if pd.notna(extreme_limit) else 0,
        "median_spread": median_spread,
        "spread_p95": spread.quantile(0.95) if not spread.empty else float("nan"),
        "spread_p99": spread.quantile(0.99) if not spread.empty else float("nan"),
        "max_spread": spread.max() if not spread.empty else float("nan"),
        "min_price": all_prices.min() if not all_prices.empty else float("nan"),
        "max_price": all_prices.max() if not all_prices.empty else float("nan"),
        "unexpected_gaps": max(bid_report.unexpected_gaps, ask_report.unexpected_gaps),
        "weekend_gaps": max(bid_report.weekend_gaps, ask_report.weekend_gaps),
        "weekday_gaps": max(bid_report.weekday_gaps, ask_report.weekday_gaps),
        "corrupt_files": len(list(file_errors)),
    }
    return pd.DataFrame([row])


def write_quality_report(summary: pd.DataFrame, destination: Path | str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


def report_dict(report: SideValidation) -> dict[str, object]:
    return {**asdict(report), "valid": report.valid}
