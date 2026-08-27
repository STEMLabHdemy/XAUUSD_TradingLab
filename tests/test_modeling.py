from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.modeling.calibration import TemporallyCalibratedModel, probability_metrics
from src.modeling.models import ModelConfig, create_model
from src.modeling.walk_forward import WalkForwardConfig, WalkForwardSplitter, temporal_development_oos_split


class ModelingTests(unittest.TestCase):
    def test_expanding_walk_forward_respects_gap_and_order(self) -> None:
        splitter = WalkForwardSplitter(WalkForwardConfig(50, 10, gap=5, mode="expanding"))
        folds = list(splitter.split(80))
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            self.assertLess(train.max(), test.min())
            self.assertEqual(test.min() - train.max() - 1, 5)
            self.assertEqual(train.min(), 0)

    def test_rolling_walk_forward_has_bounded_train_window(self) -> None:
        splitter = WalkForwardSplitter(WalkForwardConfig(50, 10, gap=5, mode="rolling", rolling_train_size=20))
        for train, _ in splitter.split(80):
            self.assertLessEqual(len(train), 20)

    def test_untouched_oos_is_after_development_with_gap(self) -> None:
        development, oos = temporal_development_oos_split(100, .2, gap=5)
        self.assertEqual(len(oos), 20)
        self.assertEqual(oos.min() - development.max() - 1, 5)

    def test_logistic_probabilities_and_temporal_calibration(self) -> None:
        rng = np.random.default_rng(42)
        features = pd.DataFrame({"x": np.linspace(-3, 3, 500), "noise": rng.normal(size=500)})
        target = (features.x + rng.normal(scale=.7, size=500) > 0).astype(int)
        base = create_model("logistic_regression", ModelConfig()).fit(features.iloc[:300], target.iloc[:300])
        calibrated = TemporallyCalibratedModel(base, "sigmoid").fit_calibration(features.iloc[300:400], target.iloc[300:400])
        probabilities = calibrated.predict_proba(features.iloc[400:])
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        metrics = probability_metrics(target.iloc[400:], probabilities)
        self.assertIn("brier_score", metrics)
        self.assertIn("log_loss", metrics)

    def test_optional_gradient_boosting_models_construct(self) -> None:
        self.assertEqual(create_model("lightgbm").name, "lightgbm")
        self.assertEqual(create_model("xgboost").name, "xgboost")


if __name__ == "__main__":
    unittest.main()
