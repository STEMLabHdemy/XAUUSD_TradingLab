from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from src.backtest import BacktestConfig, Backtester, performance_metrics
from src.signals import AggregationConfig, SignalAggregator, SignalConfig, SignalEngine


@dataclass(frozen=True)
class CandidateConfig:
    model: str
    calibration: str
    smoothing: str
    threshold: float
    persistence: int
    stop_loss: float
    take_profit: float
    max_holding_minutes: int

    @property
    def probability_column(self) -> str:
        return f"p_up_{{horizon}}m_{self.model}_{self.calibration}"


def _evaluate(frame: pd.DataFrame, score: pd.Series, config: CandidateConfig) -> dict[str, float | int]:
    working = frame.copy()
    working["score"] = score.loc[working.index]
    signal_engine = SignalEngine(SignalConfig(
        buy_threshold=config.threshold,
        sell_threshold=1 - config.threshold,
        persistence=config.persistence,
        cooldown_minutes=3,
        probability_exit_threshold=.50,
        slippage_price_per_side=.05,
        max_spread=5.0,
    ))
    working["signal"] = [
        signal_engine.decide(
            pd.Timestamp(row.datetime_utc), float(row.score), float(row.mid_close),
            float(row.spread_close), session=getattr(row, "session", None),
        ).signal
        for row in working.itertuples(index=False)
    ]
    result = Backtester(BacktestConfig(
        starting_capital=100_000,
        position_size_units=1,
        commission_per_unit_per_side=0,
        slippage_price_per_side=.05,
        stop_loss_price=config.stop_loss,
        take_profit_price=config.take_profit,
        max_holding_minutes=config.max_holding_minutes,
    )).run(working)
    return performance_metrics(result.trades, result.equity_curve, 100_000)


def _load_oos(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(run_root / "results/baseline_oos_predictions_provisional.parquet")
    predictions = predictions.sort_values("timestamp").reset_index(drop=True)
    market = pd.read_parquet(
        run_root / "data/processed/XAUUSD_M1_MASTER.parquet",
        filters=[
            ("timestamp", ">=", int(predictions.timestamp.min())),
            ("timestamp", "<=", int(predictions.timestamp.max())),
        ],
    )
    joined = market.merge(
        predictions.drop(columns=["datetime_utc", "mid_close"], errors="ignore"),
        on="timestamp", how="inner", validate="one_to_one",
    ).sort_values("timestamp").reset_index(drop=True)
    hour = pd.to_datetime(joined.datetime_utc, utc=True).dt.hour
    joined["session"] = "Other"
    joined.loc[(hour >= 0) & (hour < 8), "session"] = "Asia"
    joined.loc[(hour >= 7) & (hour < 16), "session"] = "London"
    joined.loc[(hour >= 13) & (hour < 22), "session"] = "New York"
    joined.loc[(hour >= 13) & (hour < 16), "session"] = "London/New York overlap"
    return predictions, joined


def search(run_root: Path, horizon: int, top_audit: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, joined = _load_oos(run_root)
    selection_end = int(len(joined) * .60)
    audit_start = min(len(joined), selection_end + 60)
    selection = joined.iloc[:selection_end].copy()
    audit = joined.iloc[audit_start:].copy()
    fold_boundary = len(selection) // 2
    folds = (selection.iloc[:fold_boundary].copy(), selection.iloc[fold_boundary:].copy())

    configs = [
        CandidateConfig(model, "isotonic", smoothing, threshold, persistence, stop, take, holding)
        for model in ("lightgbm", "xgboost")
        for smoothing in ("weighted", "ema")
        for threshold in (.55, .60, .65, .68)
        for persistence in (1, 2)
        for stop, take in ((3.0, 6.0), (5.0, 10.0))
        for holding in (10, 30)
    ]
    scores: dict[tuple[str, str], pd.Series] = {}
    for model in ("lightgbm", "xgboost"):
        column = f"p_up_{horizon}m_{model}_isotonic"
        for smoothing in ("weighted", "ema"):
            scores[(model, smoothing)] = SignalAggregator(
                AggregationConfig(temporal_method=smoothing)
            ).temporal(joined[column])

    development_rows: list[dict[str, object]] = []
    for index, config in enumerate(configs, start=1):
        score = scores[(config.model, config.smoothing)]
        fold_metrics = [_evaluate(fold, score, config) for fold in folds]
        row: dict[str, object] = {**asdict(config), "horizon": horizon}
        for fold_number, metrics in enumerate(fold_metrics, start=1):
            for name in ("net_pnl", "profit_factor", "sharpe", "max_drawdown", "win_rate", "trades", "expectancy"):
                row[f"fold{fold_number}_{name}"] = metrics[name]
        row["positive_folds"] = sum(float(metrics["net_pnl"]) > 0 for metrics in fold_metrics)
        row["min_fold_pnl"] = min(float(metrics["net_pnl"]) for metrics in fold_metrics)
        row["mean_fold_pnl"] = sum(float(metrics["net_pnl"]) for metrics in fold_metrics) / len(fold_metrics)
        row["min_fold_profit_factor"] = min(float(metrics["profit_factor"]) for metrics in fold_metrics)
        development_rows.append(row)
        if index % 16 == 0 or index == len(configs):
            print(f"strategy development {index}/{len(configs)}", flush=True)

    development = pd.DataFrame(development_rows).sort_values(
        ["positive_folds", "min_fold_pnl", "min_fold_profit_factor", "mean_fold_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    audit_rows: list[dict[str, object]] = []
    for row in development.head(top_audit).to_dict("records"):
        config = CandidateConfig(**{key: row[key] for key in CandidateConfig.__dataclass_fields__})
        metrics = _evaluate(audit, scores[(config.model, config.smoothing)], config)
        audit_row = {**row, **{f"audit_{name}": value for name, value in metrics.items()}}
        audit_row["reliability_pass"] = bool(
            row["positive_folds"] == 2
            and float(row["min_fold_profit_factor"]) > 1
            and float(metrics["net_pnl"]) > 0
            and float(metrics["profit_factor"]) > 1
            and int(metrics["trades"]) >= 30
        )
        audit_rows.append(audit_row)
    audited = pd.DataFrame(audit_rows).sort_values(
        ["reliability_pass", "audit_net_pnl", "audit_profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    development.to_csv(run_root / "results/strategy_development.csv", index=False)
    audited.to_csv(run_root / "results/strategy_audit.csv", index=False)
    summary = {
        "horizon": horizon,
        "selection_rows": len(selection),
        "audit_rows": len(audit),
        "configurations_tested": len(configs),
        "reliability_passes": int(audited.reliability_pass.sum()) if not audited.empty else 0,
    }
    (run_root / "results/strategy_search_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return development, audited


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporally separated strategy search for trained candidates")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--top-audit", type=int, default=12)
    args = parser.parse_args()
    development, audited = search(args.run_root.resolve(), args.horizon, args.top_audit)
    print("Top development candidates")
    print(development.head(8).to_string(index=False))
    print("Audit results")
    print(audited.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
