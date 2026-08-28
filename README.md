# XAUUSD TradingLab

Phase 1 provides restartable Dukascopy XAUUSD M1 BID/ASK downloads, incremental
updates, validation, timestamp deduplication, outer BID/ASK merge auditing, and
atomic Parquet storage. Phase 2 adds a streaming snapshot builder, a causal
FeatureEngine shared by training/backtest/live paths, and aligned multi-horizon
classification/regression targets.

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

Phase 5 provides the Streamlit research interface. Phase 6 adds a strictly
read-only MetaTrader 5 connection for realtime XAUUSD BID/ASK ticks and completed
M1 candles, automatic broker-clock normalization to UTC, Europe/Rome display,
atomic local live storage, M1/M5 charts, and provisional 5-minute inference.
M1-to-M5 aggregation is checked against native MT5 M5 candles. Real broker order
routing is absent.

Phase 7 adds persistent realtime paper accounts for LightGBM, Logistic Regression,
and XGBoost. All three receive the same MT5 feed while maintaining independent
positions, balance, equity, margin, drawdown, trades, and risk controls. Virtual
execution is BID/ASK-aware and includes configurable slippage and commission;
BUY/SELL/EXIT markers and CSV trade export are available in Live Paper. The
current models remain provisional single-horizon 5-minute models, so their paper
results are evidence to collect, not a production trading recommendation.

## Setup

```powershell
python -m pip install -r .\requirements.txt
```

## Realtime dashboard

Open MetaTrader 5, log into a demo account, select the broker's XAUUSD symbol,
and leave the terminal running. Then start the app:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_app.ps1
```

The dashboard reads the already-open MT5 session; account passwords are never
stored in the project. Completed broker candles are written under `data/live/`,
which is intentionally excluded from Git.

Open **Live Paper**, choose a model, then click **Avvia paper**. Account state is
saved under `data/live/paper/` and survives browser refreshes and application
restarts. Configuration changes deliberately require confirmation and reset all
three accounts so their comparison remains fair.

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
