from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


OFFSET_STEP_SECONDS = 15 * 60
MAX_OFFSET_SECONDS = 14 * 60 * 60


def infer_server_utc_offset(
    raw_epoch_seconds: float,
    observed_utc: datetime | None = None,
    tolerance_seconds: float = 180,
) -> int:
    """Infer a broker clock offset when its wall time is encoded as Unix UTC.

    Some MT5 brokers expose server-local wall time in the Python epoch fields.
    The offset is rounded to a 15-minute timezone increment and accepted only
    for a fresh tick close to the observing machine's UTC clock.
    """
    observed = observed_utc or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    difference = float(raw_epoch_seconds) - observed.timestamp()
    candidate = int(round(difference / OFFSET_STEP_SECONDS) * OFFSET_STEP_SECONDS)
    if abs(candidate) > MAX_OFFSET_SECONDS:
        raise ValueError("Broker server offset is outside the supported timezone range")
    if abs(difference - candidate) > tolerance_seconds:
        raise ValueError("Latest MT5 tick is too stale to infer the broker server offset")
    return candidate


def normalize_server_epoch(values: pd.Series, offset_seconds: int, unit: str = "s") -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    divisor = 1000 if unit == "ms" else 1
    adjusted = numeric - offset_seconds * divisor
    return pd.to_datetime(adjusted, unit=unit, utc=True)
