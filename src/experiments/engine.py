from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.backtest import BacktestConfig, Backtester, performance_breakdowns, performance_metrics
from src.signals import AggregationConfig, SignalAggregator, SignalConfig, SignalEngine


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    probability_column: str
    forecast_horizon: int = 5
    training_window: str = "provisional_recent"
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


class ExperimentEngine:
    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root)

    @staticmethod
    def experiment_id(spec: ExperimentSpec) -> str:
        payload = json.dumps(asdict(spec), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def run(self, market: pd.DataFrame, predictions: pd.DataFrame, spec: ExperimentSpec) -> dict[str, object]:
        experiment_id = self.experiment_id(spec)
        if spec.probability_column not in predictions:
            raise ValueError(f"Missing probability column {spec.probability_column}")
        inference = predictions.copy()
        aggregator = SignalAggregator(spec.aggregation)
        inference["temporal_score"] = aggregator.temporal(inference[spec.probability_column])
        inference["multi_horizon_score"] = inference.temporal_score
        inference["horizon_agreement_passed"] = True
        joined = market.merge(inference, on=["timestamp", "datetime_utc"], how="inner", suffixes=("", "_prediction"))
        if joined.empty:
            raise RuntimeError("Predictions do not overlap market bars")
        hour = pd.to_datetime(joined.datetime_utc, utc=True).dt.hour
        joined["session"] = pd.Series("Other", index=joined.index)
        joined.loc[(hour >= 0) & (hour < 8), "session"] = "Asia"
        joined.loc[(hour >= 7) & (hour < 16), "session"] = "London"
        joined.loc[(hour >= 13) & (hour < 22), "session"] = "New York"
        joined.loc[(hour >= 13) & (hour < 16), "session"] = "London/New York overlap"

        strategy = SignalEngine(spec.signal)
        decisions = []
        for row in joined.itertuples(index=False):
            decision = strategy.decide(
                pd.Timestamp(row.datetime_utc), float(row.multi_horizon_score), float(row.mid_close),
                float(row.spread_close), getattr(row, "expected_return", None),
                getattr(row, "session", None), getattr(row, "trend_regime", None),
                getattr(row, "volatility_regime", None), bool(row.horizon_agreement_passed),
            )
            decisions.append(decision)
        joined["signal"] = [decision.signal for decision in decisions]
        joined["signal_reasons"] = ["; ".join(decision.reasons) for decision in decisions]
        joined["score"] = joined.multi_horizon_score

        backtest = Backtester(spec.backtest).run(joined)
        metrics = performance_metrics(backtest.trades, backtest.equity_curve, spec.backtest.starting_capital)
        breakdowns = performance_breakdowns(backtest.trades)
        yearly = breakdowns.get("year", pd.DataFrame())
        metrics["year_to_year_positive_fraction"] = float((yearly.net_pnl > 0).mean()) if not yearly.empty else float("nan")
        metrics["year_to_year_pnl_std"] = float(yearly.net_pnl.std()) if len(yearly) > 1 else float("nan")
        model_metrics_path = self.project_root / "results" / "baseline_metrics_provisional.csv"
        walk_forward_auc_std = float("nan")
        if model_metrics_path.exists():
            model_metrics = pd.read_csv(model_metrics_path)
            folds = model_metrics[(model_metrics.model == spec.model) & (model_metrics.evaluation == "walk_forward")]
            if len(folds) > 1:
                walk_forward_auc_std = float(folds.roc_auc.std())
        row = {
            "experiment_id": experiment_id, "model": spec.model,
            "training_window": spec.training_window, "forecast_horizon": spec.forecast_horizon,
            "buy_threshold": spec.signal.buy_threshold, "sell_threshold": spec.signal.sell_threshold,
            "smoothing": spec.aggregation.temporal_method, "persistence": spec.signal.persistence,
            "cooldown": spec.signal.cooldown_minutes, "stop_loss": spec.backtest.stop_loss_price,
            "take_profit": spec.backtest.take_profit_price,
            "holding_time": spec.backtest.max_holding_minutes,
            "slippage": spec.backtest.slippage_price_per_side,
            "commission": spec.backtest.commission_per_unit_per_side,
            "walk_forward_auc_std": walk_forward_auc_std,
            **metrics,
        }

        run_dir = self.project_root / "results" / "runs" / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_columns = [
            column for column in ("timestamp", "datetime_utc", "mid_close", spec.probability_column,
                                  "temporal_score", "multi_horizon_score", "spread_close",
                                  "trend_regime", "volatility_regime", "signal", "signal_reasons")
            if column in joined
        ]
        joined[prediction_columns].to_parquet(run_dir / "predictions.parquet", index=False)
        backtest.trades.to_parquet(run_dir / "trades.parquet", index=False)
        backtest.trades.to_csv(run_dir / "trades.csv", index=False)
        backtest.equity_curve.to_parquet(run_dir / "equity.parquet", index=False)
        (run_dir / "config.json").write_text(json.dumps(asdict(spec), indent=2, default=str), encoding="utf-8")
        for name, breakdown in breakdowns.items():
            breakdown.to_csv(run_dir / f"breakdown_{name}.csv", index=False)
        return row

    def save_results(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        path = self.project_root / "results" / "experiments.parquet"
        new = pd.DataFrame(rows)
        if path.exists():
            existing = pd.read_parquet(path)
            new = pd.concat([existing, new], ignore_index=True)
        new = new.drop_duplicates("experiment_id", keep="last")
        if "walk_forward_auc_std" not in new:
            new["walk_forward_auc_std"] = float("nan")
        model_metrics_path = self.project_root / "results" / "baseline_metrics_provisional.csv"
        if model_metrics_path.exists():
            model_metrics = pd.read_csv(model_metrics_path)
            walk = model_metrics[model_metrics.evaluation.eq("walk_forward")]
            stability = walk.groupby("model").roc_auc.std()
            new["walk_forward_auc_std"] = new.walk_forward_auc_std.fillna(new.model.map(stability))
        robustness = []
        for row in new.itertuples():
            neighbours = new[
                new.model.eq(row.model)
                & new.buy_threshold.sub(row.buy_threshold).abs().le(.05 + 1e-9)
                & new.persistence.sub(row.persistence).abs().le(1)
            ]
            robustness.append(float(neighbours.sharpe.median()))
        new["parameter_robustness_sharpe_median"] = robustness
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.parquet")
        new.to_parquet(temporary, index=False)
        temporary.replace(path)
        return new
