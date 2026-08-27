from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_size: int
    test_size: int
    step_size: int | None = None
    gap: int = 60
    mode: str = "expanding"
    rolling_train_size: int | None = None


class WalkForwardSplitter:
    def __init__(self, config: WalkForwardConfig):
        if config.mode not in {"expanding", "rolling"}:
            raise ValueError("mode must be expanding or rolling")
        if config.initial_train_size <= config.gap:
            raise ValueError("initial_train_size must be larger than gap")
        if config.mode == "rolling" and not config.rolling_train_size:
            raise ValueError("rolling_train_size is required for rolling mode")
        self.config = config

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        step = self.config.step_size or self.config.test_size
        test_start = self.config.initial_train_size
        while test_start + self.config.test_size <= n_samples:
            train_end = test_start - self.config.gap
            train_start = 0
            if self.config.mode == "rolling":
                train_start = max(0, train_end - int(self.config.rolling_train_size or 0))
            yield np.arange(train_start, train_end), np.arange(test_start, test_start + self.config.test_size)
            test_start += step


def temporal_development_oos_split(n_samples: int, oos_fraction: float = 0.20, gap: int = 60) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < oos_fraction < 1:
        raise ValueError("oos_fraction must be between 0 and 1")
    oos_start = int(n_samples * (1 - oos_fraction))
    development_end = max(0, oos_start - gap)
    return np.arange(0, development_end), np.arange(oos_start, n_samples)
