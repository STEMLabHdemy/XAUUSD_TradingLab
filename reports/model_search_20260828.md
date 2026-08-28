# Model candidate search — 2026-08-28

## Decision

No newly trained candidate is approved for realtime paper promotion.

The strongest classifiers show a small directional signal, but every tested
strategy configuration loses money after BID/ASK execution and slippage in both
development folds and in the separate audit segment. Existing live artifacts
were therefore left unchanged.

## Data and isolation

- Canonical master used: 2020-01-01 through 2026-08-26.
- Master size: 2,461,076 M1 candles.
- Training artifacts were written under `results/model_search_20260828/`.
- Active files under `models/` were not overwritten.
- Every run preserved chronological order and used a 60-minute purge gap.
- Each model run reserved its final 20% as untouched model OOS data.

## Model search

The search compared:

- training samples: 250,000, 500,000 and 1,000,000 recent M1 rows;
- horizons: 3, 5, 10 and 15 minutes;
- models: Logistic Regression, LightGBM and XGBoost;
- probability calibration: sigmoid and isotonic;
- expanding walk-forward folds plus final temporal OOS.

Best classification candidates:

| Candidate | OOS ROC AUC | OOS Brier | Walk-forward minimum AUC |
|---|---:|---:|---:|
| XGBoost, 500k rows, 3m, sigmoid | 0.5477 | 0.2419 | 0.5594 |
| XGBoost, 500k rows, 3m, isotonic | 0.5473 | 0.2414 | 0.5594 |
| LightGBM, 500k rows, 3m, sigmoid | 0.5472 | 0.2420 | 0.5589 |
| XGBoost, 500k rows, 5m, sigmoid | 0.5469 | 0.2424 | 0.5611 |

The 3-minute/500k family is the most stable classifier in this batch. The
1-million-row 5-minute run did not improve OOS discrimination, confirming that
more history is not automatically better.

## Strategy search

The two strongest horizons (3m and 5m) received a separate strategy search.
For each horizon, 128 configurations varied:

- LightGBM versus XGBoost;
- weighted versus EMA temporal smoothing;
- probability thresholds 0.55, 0.60, 0.65 and 0.68;
- one or two confirmations;
- stop/take pairs 3/6 and 5/10 price units;
- maximum holding time 10 or 30 minutes.

Execution used next-bar timing, ASK entry/BID exit for longs, BID entry/ASK exit
for shorts, historical spread and 0.05 price units of slippage per side. The
model OOS block was subdivided into two strategy-development folds and a final
40% audit segment separated by a 60-row gap.

| Horizon | Configurations | Positive in both development folds | Best minimum fold PnL | Best audit PnL | Reliability passes |
|---|---:|---:|---:|---:|---:|
| 3m | 128 | 0 | -27.04 | -42.82 | 0 |
| 5m | 128 | 0 | -23.99 | -34.97 | 0 |

## Interpretation

The present binary target asks whether the future MID close is merely above the
current MID close. It does not require the move to cover spread and slippage,
and it is not aligned with the paper strategy's SL/TP/probability exits. The
classifiers can therefore rank direction slightly better than chance while
remaining untradeable.

The next training generation should use cost-aware DOWN/NEUTRAL/UP labels or
expected-return regression, align the training horizon with the exit policy,
and reserve a new final quarantine period after all design choices are frozen.
