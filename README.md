# XAUUSD TradingLab

Phase 1 provides restartable Dukascopy XAUUSD M1 BID/ASK downloads, incremental
updates, validation, timestamp deduplication, outer BID/ASK merge auditing, and
atomic Parquet storage. Phase 2 adds a streaming snapshot builder, a causal
FeatureEngine shared by future training/backtest/live paths, and aligned
multi-horizon classification/regression targets. ML training, backtesting, GUI,
MT5, and paper trading remain deferred to later phases.

Phase 3 provides common probabilistic wrappers for Logistic Regression, Random
Forest, LightGBM, and XGBoost; expanding/rolling walk-forward splitters with a
purge gap; untouched temporal OOS evaluation; and sigmoid/isotonic calibration.
Current model artifacts are marked `provisional` until the historical download
and final master are complete.

Phase 4 adds configurable temporal/multi-horizon aggregation, `NO_TRADE`, signal
persistence/cooldown/regime/cost filters, an event-driven next-bar BID/ASK
backtester, SL/TP/time exits, complete trade/equity ledgers, performance
breakdowns, and Parquet-backed experiment sweeps. Historical playback is never
labelled as live.

## Setup

```powershell
python -m pip install -r .\requirements.txt
```

## Full history download (manual only)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_full_history.ps1
```

The command starts with the current month and works backwards toward May 2003,
one month and one price side at a time. Valid normalized
files are skipped when it is restarted. Existing valid files with Dukascopy's
original filename are adopted without deleting the original. HTTP 429 rate-limit
responses use a longer progressive wait and are written to the log. The
application never invokes this full download automatically.

## Phase 1 data commands

```powershell
python -m src.data.cli validate
python -m src.data.cli build-master
powershell -ExecutionPolicy Bypass -File .\scripts\update_history.ps1
```

To build a fixed snapshot safely while older months are still downloading:

```powershell
python -m src.data.cli build-master --start 2020-01-01 --end 2026-08-27
```

Do not build the canonical master from a small sample and mistake it for the
complete database. The incremental updater deliberately refuses to run when the
master Parquet does not exist.
