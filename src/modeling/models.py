from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class ProbabilisticModel(Protocol):
    name: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ProbabilisticModel": ...
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    n_estimators: int = 150
    learning_rate: float = 0.05
    max_depth: int = 6
    n_jobs: int = 2


class SklearnProbabilisticModel:
    def __init__(self, name: str, estimator: object, scale: bool):
        self.name = name
        self.estimator = estimator
        self.scale = scale
        self.pipeline: Pipeline | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SklearnProbabilisticModel":
        numeric_columns = list(features.columns)
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        preprocessor = ColumnTransformer([("numeric", Pipeline(steps), numeric_columns)], remainder="drop")
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", self.estimator)])
        self.pipeline.fit(features, target.astype(int))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
            return self.pipeline.predict_proba(features)[:, 1]


def create_model(name: str, config: ModelConfig | None = None) -> SklearnProbabilisticModel:
    config = config or ModelConfig()
    normalized = name.lower().replace("-", "_")
    if normalized in {"logistic", "logistic_regression"}:
        estimator = LogisticRegression(max_iter=1000, C=1.0, random_state=config.random_state)
        return SklearnProbabilisticModel("logistic_regression", estimator, scale=True)
    if normalized in {"random_forest", "rf"}:
        estimator = RandomForestClassifier(
            n_estimators=config.n_estimators, max_depth=config.max_depth,
            random_state=config.random_state, n_jobs=config.n_jobs,
        )
        return SklearnProbabilisticModel("random_forest", estimator, scale=False)
    if normalized in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is not installed") from exc
        estimator = LGBMClassifier(
            n_estimators=config.n_estimators, learning_rate=config.learning_rate,
            num_leaves=31, max_depth=config.max_depth, random_state=config.random_state,
            n_jobs=config.n_jobs, verbosity=-1,
        )
        return SklearnProbabilisticModel("lightgbm", estimator, scale=False)
    if normalized in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost is not installed") from exc
        estimator = XGBClassifier(
            n_estimators=config.n_estimators, learning_rate=config.learning_rate,
            max_depth=config.max_depth, subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            random_state=config.random_state, n_jobs=config.n_jobs,
        )
        return SklearnProbabilisticModel("xgboost", estimator, scale=False)
    raise ValueError(f"Unknown model: {name}")
