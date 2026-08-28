from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

import pandas as pd


RUNS_DIRECTORY = Path("results/training_runs")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def start_training_job(
    project_root: Path,
    rows: int,
    horizon: int,
    model_names: list[str],
) -> dict[str, Any]:
    allowed_rows = {50_000, 100_000, 250_000, 500_000, 1_000_000}
    allowed_horizons = {1, 3, 5, 10, 15}
    allowed_models = {"logistic_regression", "lightgbm", "xgboost"}
    if rows not in allowed_rows or horizon not in allowed_horizons:
        raise ValueError("Configurazione training non consentita")
    if not model_names or not set(model_names).issubset(allowed_models):
        raise ValueError("Seleziona almeno un modello valido")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    run_dir = project_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    job = {
        "run_id": run_id, "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None, "rows": rows, "horizon": horizon, "models": model_names,
        "pid": None, "error": None,
    }
    _write_json(run_dir / "job.json", job)
    command = [
        sys.executable, "-m", "src.modeling.training_worker", "--project-root", str(project_root),
        "--run-dir", str(run_dir), "--rows", str(rows), "--horizon", str(horizon),
        "--models", *model_names,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=project_root, stdout=log, stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    job["pid"] = process.pid
    _write_json(run_dir / "job.json", job)
    return job


def list_training_jobs(project_root: Path) -> list[dict[str, Any]]:
    root = project_root / RUNS_DIRECTORY
    jobs: list[dict[str, Any]] = []
    if not root.exists():
        return jobs
    for path in sorted(root.glob("*/job.json"), reverse=True):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job["directory"] = str(path.parent)
            log_path = path.parent / "worker.log"
            job["log"] = log_path.read_text(encoding="utf-8", errors="replace")[-5000:] if log_path.exists() else ""
            jobs.append(job)
        except (OSError, json.JSONDecodeError):
            continue
    return jobs


def compatible_models(project_root: Path) -> pd.DataFrame:
    """Discover binary calibrated artifacts that the current paper engine can load."""
    rows: list[dict[str, Any]] = []
    roots = [(project_root, "Attivo")]
    roots.extend((path, path.name) for path in (project_root / RUNS_DIRECTORY).glob("*") if path.is_dir())
    for root, run_label in roots:
        model_dir = root / "models"
        manifest = model_dir / "baseline_manifest_provisional.json"
        if not manifest.exists():
            continue
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics_path = root / "results/baseline_metrics_provisional.csv"
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        for artifact in sorted(model_dir.glob("*_provisional.joblib")):
            stem = artifact.stem
            method = "isotonic" if "_isotonic_" in stem else "sigmoid" if "_sigmoid_" in stem else "unknown"
            model = stem.split("_up_")[0]
            if {"model", "evaluation"}.issubset(metrics.columns):
                metric = metrics[
                    metrics["model"].eq(model)
                    & metrics["evaluation"].eq(f"untouched_oos_{method}")
                ]
            else:
                metric = pd.DataFrame()
            record: dict[str, Any] = {
                "run": run_label, "model": model, "horizon": metadata.get("horizon_minutes"),
                "calibration": method, "rows": metadata.get("rows_requested"),
                "artifact": str(artifact), "manifest": str(manifest), "compatible": True,
            }
            if not metric.empty:
                for key in ("roc_auc", "brier_score", "log_loss"):
                    if key in metric:
                        record[key] = float(metric.iloc[-1][key])
            rows.append(record)
    return pd.DataFrame(rows)
