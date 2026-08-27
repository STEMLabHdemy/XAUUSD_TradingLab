from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

OHLC = ("open", "high", "low", "close")
VOLUME_ALIASES = ("volume", "tick_volume", "vol")


def parse_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    populated = numeric.dropna()
    if len(populated) == len(series) and not populated.empty:
        magnitude = float(populated.abs().median())
        unit = "ms" if magnitude >= 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def read_side_csv(path: Path | str, side: str) -> pd.DataFrame:
    side = side.lower()
    if side not in {"bid", "ask"}:
        raise ValueError("side must be 'bid' or 'ask'")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "timestamp" not in frame.columns:
        for candidate in ("datetime", "date", "time", "gmt time"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "timestamp"})
                break
    required = {"timestamp", *OHLC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    frame["datetime_utc"] = parse_timestamp(frame["timestamp"])
    # pandas 3 may preserve second/millisecond resolution, so astype(int64)
    # cannot safely be assumed to mean nanoseconds. Convert explicitly to ms.
    frame["timestamp"] = frame["datetime_utc"].map(
        lambda value: int(value.timestamp() * 1_000) if pd.notna(value) else pd.NA
    ).astype("Int64")
    keep = ["timestamp", "datetime_utc"]
    for column in OHLC:
        renamed = f"{column}_{side}"
        frame[renamed] = pd.to_numeric(frame[column], errors="coerce")
        keep.append(renamed)
    for candidate in VOLUME_ALIASES:
        if candidate in frame.columns:
            renamed = f"volume_{side}"
            frame[renamed] = pd.to_numeric(frame[candidate], errors="coerce")
            keep.append(renamed)
            break
    return frame[keep]


def read_side_directory(directory: Path | str, side: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    return read_side_files(sorted(Path(directory).glob("*.csv")), side)


def read_side_files(paths: list[Path], side: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    file_errors: list[dict[str, str]] = []
    for path in paths:
        try:
            part = read_side_csv(path, side)
            part["_source_file"] = path.name
            frames.append(part)
        except Exception as exc:
            file_errors.append({"side": side, "file": str(path), "error": str(exc)})
    if not frames:
        columns = ["timestamp", "datetime_utc", *(f"{c}_{side}" for c in OHLC), "_source_file"]
        return pd.DataFrame(columns=columns), file_errors
    return pd.concat(frames, ignore_index=True), file_errors


def atomic_write_parquet(frame: pd.DataFrame, destination: Path | str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        os.replace(temporary, path)
    except ImportError as exc:
        raise RuntimeError("PyArrow is required for Parquet. Run: python -m pip install -r requirements.txt") from exc
    finally:
        temporary.unlink(missing_ok=True)
