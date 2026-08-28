from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .calibration import TemporallyCalibratedModel, calibration_table, probability_metrics
from .dataset import load_recent_rows, prepare_binary_dataset
from .models import ModelConfig, create_model
from .walk_forward import WalkForwardConfig, WalkForwardSplitter, temporal_development_oos_split


def train_baselines(
    project_root: Path,
    rows: int = 100_000,
    horizon: int = 5,
    model_names: tuple[str, ...] = ("logistic_regression", "lightgbm", "xgboost"),
    output_root: Path | None = None,
) -> pd.DataFrame:
    master_path = project_root / "data/processed/XAUUSD_M1_MASTER.parquet"
    market = load_recent_rows(master_path, rows)
    features, target, labelled = prepare_binary_dataset(market, horizon)
    development, oos = temporal_development_oos_split(len(features), oos_fraction=.20, gap=60)
    if len(development) < 1000 or len(oos) < 100:
        raise RuntimeError("Not enough usable rows for temporal training")

    results: list[dict[str, object]] = []
    prediction_output = labelled.loc[oos, ["timestamp", "datetime_utc", "mid_close"]].reset_index(drop=True)
    calibration_outputs: list[pd.DataFrame] = []
    artifact_root = output_root.resolve() if output_root is not None else project_root
    model_dir = artifact_root / "models"
    result_dir = artifact_root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(n_estimators=100, max_depth=5, n_jobs=2)

    naive_probabilities = np.full(len(oos), 0.5)
    results.append({
        "model": "naive_0.5", "evaluation": "untouched_oos", "fold": pd.NA,
        "train_rows": 0, "test_rows": len(oos), "train_end": pd.NaT,
        "test_start": labelled.datetime_utc.iloc[oos[0]],
        **probability_metrics(target.iloc[oos], naive_probabilities),
    })
    development_rate = float(target.iloc[development].mean())
    results.append({
        "model": "naive_development_rate", "evaluation": "untouched_oos", "fold": pd.NA,
        "train_rows": len(development), "test_rows": len(oos),
        "train_end": labelled.datetime_utc.iloc[development[-1]],
        "test_start": labelled.datetime_utc.iloc[oos[0]],
        **probability_metrics(target.iloc[oos], np.full(len(oos), development_rate)),
    })

    walk_config = WalkForwardConfig(
        initial_train_size=max(1000, len(development) // 2),
        test_size=max(500, len(development) // 4),
        gap=60,
        mode="expanding",
    )
    for model_name in model_names:
        for fold, (train_index, test_index) in enumerate(WalkForwardSplitter(walk_config).split(len(development)), start=1):
            model = create_model(model_name, config).fit(features.iloc[train_index], target.iloc[train_index])
            probabilities = model.predict_proba(features.iloc[test_index])
            results.append({
                "model": model.name, "evaluation": "walk_forward", "fold": fold,
                "train_rows": len(train_index), "test_rows": len(test_index),
                "train_end": labelled.datetime_utc.iloc[train_index[-1]],
                "test_start": labelled.datetime_utc.iloc[test_index[0]],
                **probability_metrics(target.iloc[test_index], probabilities),
            })

        calibration_size = max(500, int(len(development) * .15))
        calibration_start = len(development) - calibration_size
        fit_end = calibration_start - 60
        fit_index = development[:fit_end]
        calibration_index = development[calibration_start:]
        base = create_model(model_name, config).fit(features.iloc[fit_index], target.iloc[fit_index])
        base_probabilities = base.predict_proba(features.iloc[oos])
        results.append({
            "model": base.name, "evaluation": "untouched_oos_uncalibrated", "fold": pd.NA,
            "train_rows": len(fit_index), "test_rows": len(oos),
            "train_end": labelled.datetime_utc.iloc[fit_index[-1]],
            "test_start": labelled.datetime_utc.iloc[oos[0]],
            **probability_metrics(target.iloc[oos], base_probabilities),
        })
        prediction_output[f"p_up_{horizon}m_{base.name}_uncalibrated"] = base_probabilities

        for method in ("sigmoid", "isotonic"):
            calibrated = TemporallyCalibratedModel(base, method).fit_calibration(
                features.iloc[calibration_index], target.iloc[calibration_index]
            )
            probabilities = calibrated.predict_proba(features.iloc[oos])
            metrics = probability_metrics(target.iloc[oos], probabilities)
            results.append({
                "model": base.name, "evaluation": f"untouched_oos_{method}", "fold": pd.NA,
                "train_rows": len(fit_index), "calibration_rows": len(calibration_index),
                "test_rows": len(oos), "train_end": labelled.datetime_utc.iloc[fit_index[-1]],
                "calibration_start": labelled.datetime_utc.iloc[calibration_index[0]],
                "test_start": labelled.datetime_utc.iloc[oos[0]], **metrics,
            })
            prediction_output[f"p_up_{horizon}m_{base.name}_{method}"] = probabilities
            curve = calibration_table(target.iloc[oos], probabilities)
            curve.insert(0, "method", method)
            curve.insert(0, "model", base.name)
            calibration_outputs.append(curve)
            joblib.dump(calibrated, model_dir / f"{base.name}_up_{horizon}m_{method}_provisional.joblib")

    metrics_frame = pd.DataFrame(results)
    metrics_frame.to_csv(result_dir / "baseline_metrics_provisional.csv", index=False)
    pd.concat(calibration_outputs, ignore_index=True).to_csv(result_dir / "calibration_curves_provisional.csv", index=False)
    prediction_output.insert(3, f"actual_up_{horizon}m", target.iloc[oos].astype(int).to_numpy())
    prediction_output.to_parquet(result_dir / "baseline_oos_predictions_provisional.parquet", index=False)
    manifest = {
        "status": "provisional", "master": str(master_path), "rows_requested": rows,
        "usable_rows": len(features), "horizon_minutes": horizon,
        "feature_columns": list(features.columns), "models": list(model_names),
        "random_shuffle": False, "untouched_oos_rows": len(oos), "purge_gap_rows": 60,
    }
    (model_dir / "baseline_manifest_provisional.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Train provisional time-series baselines")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "lightgbm", "xgboost"])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = train_baselines(
        args.project_root.resolve(), args.rows, args.horizon, tuple(args.models), args.output_root
    )
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
