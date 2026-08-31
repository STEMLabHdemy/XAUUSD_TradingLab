from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FeatureEngine


@dataclass(frozen=True)
class LiveInference:
    available: bool
    model: str | None
    horizon_minutes: int | None
    probability_up: float | None
    inference_time_utc: pd.Timestamp | None
    candidate: str
    final_signal: str
    reason: str


class LiveInferenceEngine:
    def __init__(
        self,
        model_path: Path | str,
        manifest_path: Path | str,
        buy_threshold: float = .68,
        sell_threshold: float = .32,
    ):
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._model = None
        self._manifest: dict[str, object] | None = None

    def _load(self) -> bool:
        if self._model is not None and self._manifest is not None:
            return True
        if not self.model_path.exists() or not self.manifest_path.exists():
            return False
        self._model = joblib.load(self.model_path)
        self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return True

    def predict(self, completed_m1: pd.DataFrame) -> LiveInference:
        if not self._load():
            return LiveInference(False, None, None, None, None, "N/D", "NO_TRADE", "Modello live non disponibile")
        if len(completed_m1) < FeatureEngine().config.warmup_rows + 1:
            return LiveInference(False, None, None, None, None, "N/D", "NO_TRADE", "Warm-up M1 insufficiente")
        assert self._manifest is not None and self._model is not None
        featured = FeatureEngine().transform(completed_m1)
        feature_columns = list(self._manifest["feature_columns"])
        missing = set(feature_columns).difference(featured.columns)
        if missing:
            return LiveInference(False, None, None, None, None, "N/D", "NO_TRADE", f"Feature mancanti: {sorted(missing)}")
        row = featured.iloc[[-1]][feature_columns].replace([np.inf, -np.inf], np.nan)
        probability = float(self._model.predict_proba(row)[0])
        candidate = "BUY" if probability >= self.buy_threshold else "SELL" if probability <= self.sell_threshold else "HOLD"
        horizon = int(self._manifest.get("horizon_minutes", 5))
        model_name = getattr(self._model, "name", self.model_path.stem)
        return LiveInference(
            True, str(model_name), horizon, probability,
            pd.Timestamp(completed_m1.datetime_utc.iloc[-1]), candidate, "NO_TRADE",
            f"Modello singolo H{horizon}; conferma multi-orizzonte non ancora disponibile",
        )


class CostAwareLiveInferenceEngine:
    """Adapter for the research-only UP/DOWN/NEUTRAL classifiers.

    It intentionally exposes a binary-compatible score to PaperAccount only at
    the execution boundary: 0.60 for UP, 0.40 for DOWN and 0.50 for NEUTRAL.
    The underlying three-class probabilities remain visible in the reason.
    """

    def __init__(self, model_path: Path | str, manifest_path: Path | str):
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self._model = None
        self._manifest: dict[str, object] | None = None

    def _load(self) -> bool:
        if self._model is not None and self._manifest is not None:
            return True
        if not self.model_path.exists() or not self.manifest_path.exists():
            return False
        self._model = joblib.load(self.model_path)
        self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return True

    def predict(self, completed_m1: pd.DataFrame) -> LiveInference:
        if not self._load():
            return LiveInference(False, None, None, None, None, "N/D", "NO_TRADE", "Modello cost-aware non disponibile")
        if len(completed_m1) < FeatureEngine().config.warmup_rows + 1:
            return LiveInference(False, None, None, None, None, "N/D", "NO_TRADE", "Warm-up M1 insufficiente")
        assert self._manifest is not None and self._model is not None
        featured = FeatureEngine().transform(completed_m1)
        feature_columns = list(self._manifest["feature_columns"])
        missing = set(feature_columns).difference(featured.columns)
        if missing:
            return LiveInference(False, None, None, None, None, "N/D", "NO_TRADE", f"Feature mancanti: {sorted(missing)}")
        row = featured.iloc[[-1]][feature_columns].replace([np.inf, -np.inf], np.nan)
        probabilities = np.asarray(self._model.predict_proba(row)[0], dtype=float)
        classes = [int(value) for value in getattr(self._model, "classes_", (0, 1, 2))]
        by_class = dict(zip(classes, probabilities, strict=True))
        down, neutral, up = (float(by_class.get(index, 0.0)) for index in (0, 1, 2))
        winner = int(max(by_class, key=by_class.get))
        candidate = {0: "SELL", 1: "HOLD", 2: "BUY"}[winner]
        score = {"SELL": .40, "HOLD": .50, "BUY": .60}[candidate]
        horizon = int(self._manifest.get("horizon_minutes", 15))
        return LiveInference(
            True, f"{self.model_path.stem} (cost-aware)", horizon, score,
            pd.Timestamp(completed_m1.datetime_utc.iloc[-1]), candidate, "NO_TRADE",
            f"Research-only cost-aware: DOWN {down:.1%}, NEUTRAL {neutral:.1%}, UP {up:.1%}",
        )
