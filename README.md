# XAUUSD TradingLab

Phase 1 provides restartable Dukascopy XAUUSD M1 BID/ASK downloads, incremental
updates, validation, timestamp deduplication, outer BID/ASK merge auditing, and
atomic Parquet storage. ML, backtesting, GUI, MT5, and paper trading are
intentionally deferred to later phases.

## Setup

```powershell
python -m pip install -r .\requirements.txt
```

## Full history download (manual only)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_full_history.ps1
```

The command downloads one month and one price side at a time. Valid normalized
files are skipped when it is restarted. The application never invokes this full
download automatically.

## Phase 1 data commands

```powershell
python -m src.data.cli validate
python -m src.data.cli build-master
powershell -ExecutionPolicy Bypass -File .\scripts\update_history.ps1
```

Do not build the canonical master from a small sample and mistake it for the
complete database. The incremental updater deliberately refuses to run when the
master Parquet does not exist.

