from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .models import SklearnProbabilisticModel


class TemporallyCalibratedModel:
    def __init__(self, fitted_model: SklearnProbabilisticModel, method: str = "sigmoid"):
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("method must be sigmoid or isotonic")
        if fitted_model.pipeline is None:
            raise RuntimeError("Base model must be fitted before calibration")
        self.name = f"{fitted_model.name}_{method}"
        self.model = fitted_model
        self.calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_model.pipeline), method=method)

    def fit_calibration(self, features: pd.DataFrame, target: pd.Series) -> "TemporallyCalibratedModel":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
            self.calibrator.fit(features, target.astype(int))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
            return self.calibrator.predict_proba(features)[:, 1]


def probability_metrics(target: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
    }


def calibration_table(target: pd.Series | np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(target, probabilities, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": predicted, "observed_up_frequency": observed})
