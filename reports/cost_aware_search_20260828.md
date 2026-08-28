# Cost-aware model search - 2026-08-28

## Decision

No new candidate is approved for realtime paper promotion. The execution-aware
target produces a substantially stronger classification signal than the prior
MID-price binary target, but none of the tested strategies remains profitable
and sufficiently sampled in the untouched audit period. Active model artifacts
were left unchanged.

## Target and execution alignment

Each signal is generated at the close of minute `t`. A simulated trade enters
at the next available candle open and uses the executable side of the market:

- long: enter at ASK, exit at future BID;
- short: enter at BID, exit at future ASK;
- 0.05 price units of slippage are charged on each side;
- DOWN/NEUTRAL/UP labels require the winning direction to exceed a net move of
  0.50 price units after spread and slippage;
- exact timestamp continuity is required, and all splits preserve chronology
  with a 60-row purge gap.

## Training search

The research used the latest 500,000 rows of the canonical M1 master and tested
six multiclass candidates at 3, 5 and 10 minute horizons: two Logistic
Regression variants, two LightGBM variants and two XGBoost variants. Each model
used expanding walk-forward folds followed by a final 20% temporal OOS block.

| Horizon | Best classifier | OOS macro AUC | Minimum walk-forward AUC |
|---|---|---:|---:|
| 3m | XGBoost medium | 0.6463 | 0.6300 |
| 5m | XGBoost medium | 0.6325 | 0.6160 |
| 10m | XGBoost medium | 0.6149 | 0.6035 |

These AUC values show a real ranking signal, especially at three minutes, but
classification quality alone does not demonstrate tradability.

## Strategy audit

Model finalists were selected exclusively from the earlier walk-forward folds;
no metric from the final OOS block was used for selection. For each horizon, 72
configurations varied the finalist model, weighted/EMA smoothing, thresholds
0.55/0.60/0.65, one or two confirmations, and stop/take pairs 2/4, 3/6 and 5/10.
Maximum holding time matched the prediction horizon.

The model OOS block was then split chronologically: the first 60% formed two
strategy-development folds, followed by a purge gap, while the final 40% was
reserved for audit. Reliability required positive PnL and profit factor above
one in both development folds and the audit, plus at least 30 audit trades.

| Horizon | Configurations | Development-eligible | Best eligible audit PnL | Audit PF | Audit trades | Passes |
|---|---:|---:|---:|---:|---:|---:|
| 3m | 72 | 7 | -9.74 | 0.53 | 13 | 0 |
| 5m | 72 | 2 | -20.30 | 0.52 | 24 | 0 |
| 10m | 72 | 8 | -0.80 | 0.95 | 7 | 0 |

The 10-minute candidate is closest to break-even but has only seven audit
trades, far below the minimum sample, and is still negative after costs. It is
not suitable for promotion.

## Reproducibility and next experiment

Research artifacts are isolated under `results/cost_aware_search_20260828/`.
The next useful experiment is not another broad parameter sweep over the same
target. It should add regime-aware validation and expected-net-return modeling,
then freeze the design before opening a completely new quarantine period.
