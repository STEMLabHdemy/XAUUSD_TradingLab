from __future__ import annotations

import pandas as pd


def market_open_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify tradable XAUUSD rows without modifying the raw source.

    Dukascopy monthly M1 exports can forward-fill Sunday prices from 00:00 UTC
    until the weekly market opens. Monday-Friday rows are retained, Saturdays
    are excluded, and each Sunday is retained only from its first matched,
    non-flat BID/ASK candle onward.
    """
    utc = pd.to_datetime(frame["datetime_utc"], utc=True)
    weekday = utc.dt.weekday
    keep = weekday.lt(5)
    sunday = weekday.eq(6)
    if not sunday.any():
        return keep.astype(bool)

    required = [
        "open_bid", "high_bid", "low_bid", "close_bid",
        "open_ask", "high_ask", "low_ask", "close_ask",
    ]
    matched = frame[required].notna().all(axis=1)
    bid_has_range = frame.high_bid.ne(frame.low_bid) | frame.open_bid.ne(frame.close_bid)
    ask_has_range = frame.high_ask.ne(frame.low_ask) | frame.open_ask.ne(frame.close_ask)
    active = sunday & matched & (bid_has_range | ask_has_range)
    sunday_date = utc.dt.date
    for session_date in sunday_date[sunday].unique():
        day_rows = sunday & sunday_date.eq(session_date)
        active_rows = frame.index[day_rows & active]
        if len(active_rows):
            first_active = active_rows[0]
            keep.loc[day_rows & frame.index.to_series().ge(first_active)] = True
    return keep.astype(bool)


def filter_market_closed(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = market_open_mask(frame)
    return frame.loc[mask].reset_index(drop=True), int((~mask).sum())
