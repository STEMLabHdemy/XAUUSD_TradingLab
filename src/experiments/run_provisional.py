from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest import BacktestConfig
from src.experiments import ExperimentEngine, ExperimentSpec
from src.signals import AggregationConfig, SignalConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run provisional OOS strategy sweep")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    predictions = pd.read_parquet(root / "results/baseline_oos_predictions_provisional.parquet")
    minimum, maximum = int(predictions.timestamp.min()), int(predictions.timestamp.max())
    market = pd.read_parquet(
        root / "data/processed/XAUUSD_M1_MASTER.parquet",
        filters=[("timestamp", ">=", minimum), ("timestamp", "<=", maximum)],
    )
    columns = {
        "logistic_regression": "p_up_5m_logistic_regression_isotonic",
        "lightgbm": "p_up_5m_lightgbm_isotonic",
        "xgboost": "p_up_5m_xgboost_isotonic",
    }
    engine = ExperimentEngine(root)
    rows = []
    total = len(columns) * 3 * 3
    completed = 0
    for model, probability_column in columns.items():
        for buy_threshold in (.60, .65, .68):
            for persistence in (1, 2, 3):
                completed += 1
                print(f"Experiment {completed}/{total}: {model}, threshold={buy_threshold}, persistence={persistence}")
                spec = ExperimentSpec(
                    model=model,
                    probability_column=probability_column,
                    aggregation=AggregationConfig(temporal_method="weighted"),
                    signal=SignalConfig(
                        buy_threshold=buy_threshold, sell_threshold=1 - buy_threshold,
                        persistence=persistence, cooldown_minutes=3, max_spread=5.0,
                    ),
                    backtest=BacktestConfig(
                        starting_capital=100_000, position_size_units=1,
                        commission_per_unit_per_side=0, slippage_price_per_side=.05,
                        stop_loss_price=5, take_profit_price=10, max_holding_minutes=30,
                    ),
                )
                rows.append(engine.run(market, predictions, spec))
    results = engine.save_results(rows)
    provisional_ids = {row["experiment_id"] for row in rows}
    current = results[results.experiment_id.isin(provisional_ids)].sort_values(
        ["sharpe", "profit_factor", "expectancy"], ascending=False
    )
    print(current.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
