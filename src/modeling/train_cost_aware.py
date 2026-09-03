from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.features import FeatureConfig, FeatureEngine
from src.modeling.dataset import aggregate_training_bars, load_recent_rows
from src.modeling.walk_forward import WalkForwardConfig, WalkForwardSplitter, temporal_development_oos_split
from src.targets import TargetConfig, TargetEngine


LABELS = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    parameters: dict[str, object]
    scale: bool = False


def candidates(names: tuple[str, ...] | None = None) -> tuple[Candidate, ...]:
    available = (
        Candidate("logistic_c0p1", "logistic", {"C": .1}, True),
        Candidate("logistic_c1", "logistic", {"C": 1.0}, True),
        Candidate("lightgbm_shallow", "lightgbm", {"n_estimators": 150, "learning_rate": .05, "max_depth": 3, "num_leaves": 15}),
        Candidate("lightgbm_medium", "lightgbm", {"n_estimators": 250, "learning_rate": .04, "max_depth": 5, "num_leaves": 31}),
        Candidate("xgboost_shallow", "xgboost", {"n_estimators": 150, "learning_rate": .05, "max_depth": 3}),
        Candidate("xgboost_medium", "xgboost", {"n_estimators": 250, "learning_rate": .04, "max_depth": 5}),
    )
    if names is None:
        return available
    unknown = set(names).difference(candidate.name for candidate in available)
    if unknown:
        raise ValueError(f"Unknown candidates: {sorted(unknown)}")
    return tuple(candidate for candidate in available if candidate.name in names)


def _estimator(candidate: Candidate) -> object:
    if candidate.family == "logistic":
        return LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced",
            **candidate.parameters,
        )
    if candidate.family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="multiclass", random_state=42, n_jobs=2, verbosity=-1,
            class_weight="balanced", **candidate.parameters,
        )
    if candidate.family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", random_state=42,
            n_jobs=2, subsample=.8, colsample_bytree=.8, **candidate.parameters,
        )
    raise ValueError(candidate.family)


def _pipeline(candidate: Candidate, columns: list[str]) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if candidate.scale:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), columns),
    ], remainder="drop")
    return Pipeline([("preprocessor", preprocessing), ("classifier", _estimator(candidate))])


def _fit(candidate: Candidate, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    pipeline = _pipeline(candidate, list(features.columns))
    if candidate.family == "xgboost":
        weights = compute_sample_weight(class_weight="balanced", y=target)
        pipeline.fit(features, target, classifier__sample_weight=weights)
    else:
        pipeline.fit(features, target)
    return pipeline


def _metrics(target: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(3, dtype=float)[target.to_numpy(dtype=int)]
    return {
        "multiclass_brier": float(np.mean(np.square(probabilities - one_hot).sum(axis=1))),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1, 2])),
        "macro_roc_auc": float(roc_auc_score(target, probabilities, labels=[0, 1, 2], multi_class="ovr", average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "recall_down": float(recall_score(target, predicted, labels=[0], average="macro", zero_division=0)),
        "recall_neutral": float(recall_score(target, predicted, labels=[1], average="macro", zero_division=0)),
        "recall_up": float(recall_score(target, predicted, labels=[2], average="macro", zero_division=0)),
        "predicted_trade_fraction": float(np.mean(predicted != 1)),
    }


def _dataset(master: Path, rows: int, horizon: int, minimum_move: float, timeframe_minutes: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    market = load_recent_rows(master, rows)
    market = aggregate_training_bars(market, timeframe_minutes)
    original_columns = set(market.columns)
    # Keep feature labels truthful: on M5, ``return_15m`` is three complete
    # M5 bars, never a misleading one-bar return named "1m".
    default_features = FeatureConfig()
    real_minute_horizons = tuple(
        value for value in default_features.return_horizons
        if value >= timeframe_minutes and value % timeframe_minutes == 0
    )
    featured = FeatureEngine(FeatureConfig(
        bar_minutes=timeframe_minutes,
        return_horizons=real_minute_horizons,
    )).transform(market)
    labelled = TargetEngine(TargetConfig(
        horizons=(horizon,), executable_minimum_net_move=minimum_move,
        slippage_price_per_side=.05,
        bar_minutes=timeframe_minutes,
    )).transform(featured)
    target_column = f"executable_direction_{horizon}m"
    feature_columns = [
        column for column in featured.columns
        if column not in original_columns and pd.api.types.is_numeric_dtype(featured[column])
    ]
    usable = labelled[target_column].notna() & labelled[feature_columns].notna().any(axis=1)
    selected = labelled.loc[usable].reset_index(drop=True)
    return (
        selected[feature_columns],
        selected[target_column].map(LABELS).astype("int8"),
        selected,
    )


def train_cost_aware(
    project_root: Path,
    output_root: Path,
    rows: int,
    horizon: int,
    minimum_move: float,
    data_path: Path | None = None,
    candidate_names: tuple[str, ...] | None = None,
    timeframe_minutes: int = 1,
) -> pd.DataFrame:
    if horizon % timeframe_minutes:
        raise ValueError("horizon must be a multiple of timeframe_minutes")
    features, target, labelled = _dataset(
        data_path or project_root / "data/processed/XAUUSD_M1_MASTER.parquet",
        rows, horizon, minimum_move, timeframe_minutes,
    )
    purge_gap_rows = max(horizon // timeframe_minutes, 60 // timeframe_minutes)
    development, oos = temporal_development_oos_split(len(features), oos_fraction=.20, gap=purge_gap_rows)
    walk = WalkForwardSplitter(WalkForwardConfig(
        initial_train_size=max(1000, len(development) // 2),
        test_size=max(500, len(development) // 4), gap=purge_gap_rows, mode="expanding",
    ))
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / "models"
    result_dir = output_root / "results"
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    prediction_output = labelled.loc[oos, [
        "timestamp", "datetime_utc", "mid_close", "spread_close",
        "open_bid", "high_bid", "low_bid", "close_bid",
        "open_ask", "high_ask", "low_ask", "close_ask",
        f"future_long_net_{horizon}m", f"future_short_net_{horizon}m",
        f"executable_direction_{horizon}m",
    ]].reset_index(drop=True)
    result_rows: list[dict[str, object]] = []

    selected_candidates = candidates(candidate_names)
    for candidate_number, candidate in enumerate(selected_candidates, start=1):
        fold_values: list[dict[str, float]] = []
        for fold, (train_index, test_index) in enumerate(walk.split(len(development)), start=1):
            model = _fit(candidate, features.iloc[train_index], target.iloc[train_index])
            metrics = _metrics(target.iloc[test_index], model.predict_proba(features.iloc[test_index]))
            fold_values.append(metrics)
            result_rows.append({
                "candidate": candidate.name, "family": candidate.family,
                "evaluation": "walk_forward", "fold": fold,
                "train_rows": len(train_index), "test_rows": len(test_index),
                "train_end": labelled.datetime_utc.iloc[train_index[-1]],
                "test_start": labelled.datetime_utc.iloc[test_index[0]], **metrics,
            })
        print(f"candidate {candidate_number}/{len(selected_candidates)} final fit: {candidate.name}", flush=True)
        final_model = _fit(candidate, features.iloc[development], target.iloc[development])
        probabilities = final_model.predict_proba(features.iloc[oos])
        metrics = _metrics(target.iloc[oos], probabilities)
        result_rows.append({
            "candidate": candidate.name, "family": candidate.family,
            "evaluation": "untouched_oos", "fold": pd.NA,
            "train_rows": len(development), "test_rows": len(oos),
            "train_end": labelled.datetime_utc.iloc[development[-1]],
            "test_start": labelled.datetime_utc.iloc[oos[0]],
            "walk_auc_mean": float(np.mean([value["macro_roc_auc"] for value in fold_values])),
            "walk_auc_min": float(np.min([value["macro_roc_auc"] for value in fold_values])),
            "walk_auc_std": float(np.std([value["macro_roc_auc"] for value in fold_values])),
            **metrics,
        })
        prediction_output[f"p_down_{candidate.name}"] = probabilities[:, 0]
        prediction_output[f"p_neutral_{candidate.name}"] = probabilities[:, 1]
        prediction_output[f"p_up_{candidate.name}"] = probabilities[:, 2]
        prediction_output[f"directional_score_{candidate.name}"] = probabilities[:, 2] + .5 * probabilities[:, 1]
        joblib.dump(final_model, model_dir / f"{candidate.name}_h{horizon}_move{minimum_move:g}.joblib")

    metrics_frame = pd.DataFrame(result_rows)
    metrics_frame.to_csv(result_dir / "metrics.csv", index=False)
    prediction_output.to_parquet(result_dir / "oos_predictions.parquet", index=False)
    manifest = {
        "status": "research_only", "rows_requested": rows, "usable_rows": len(features),
        "timeframe_minutes": timeframe_minutes, "horizon_minutes": horizon, "minimum_net_move": minimum_move,
        "slippage_price_per_side": .05, "labels": LABELS,
        "feature_columns": list(features.columns), "candidates": [candidate.name for candidate in selected_candidates],
        "data_path": str(data_path or project_root / "data/processed/XAUUSD_M1_MASTER.parquet"),
        "random_shuffle": False, "purge_gap_rows": purge_gap_rows, "oos_rows": len(oos),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Train execution-aligned cost-aware multiclass candidates")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--timeframe-minutes", type=int, default=1)
    parser.add_argument("--minimum-move", type=float, default=.50)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--candidates", nargs="+")
    args = parser.parse_args()
    metrics = train_cost_aware(
        args.project_root.resolve(), args.output_root.resolve(), args.rows,
        args.horizon, args.minimum_move, args.data_path.resolve() if args.data_path else None,
        tuple(args.candidates) if args.candidates else None, args.timeframe_minutes,
    )
    print(metrics[metrics.evaluation.eq("untouched_oos")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
