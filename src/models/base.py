"""Interfaces abstractas."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAnomalyModel(ABC):
    name: str = "base_anomaly"

    def __init__(self, **params):
        self.params = params
        self.model = None
        self.threshold_: float | None = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseAnomalyModel": ...

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score por muestra; mayor = más anómalo."""

    def set_threshold(self, train_scores: np.ndarray, strategy: str = "quantile", value: float = 0.95):
        if strategy == "quantile":
            self.threshold_ = float(np.quantile(train_scores, value))
        elif strategy == "std":
            self.threshold_ = float(train_scores.mean() + value * train_scores.std())
        else:
            raise ValueError(strategy)
        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold no fijado. Llama set_threshold().")
        return (self.score_samples(X) > self.threshold_).astype(int)

    def save(self, dir_path: Path) -> None:
        raise NotImplementedError

    def load(self, dir_path: Path) -> "BaseAnomalyModel":
        raise NotImplementedError


class BaseForecaster(ABC):
    name: str = "base_forecast"

    def __init__(self, **params):
        self.params = params
        self.horizon: int = params.get("horizon", 1)
        self.targets: list[str] = params.get("targets", [])

    @abstractmethod
    def fit(self, df, feature_cols: list[str]) -> "BaseForecaster": ...

    @abstractmethod
    def predict(self, df_last: "pd.DataFrame", horizon: int) -> "np.ndarray": ...

    def save(self, dir_path: Path) -> None:
        raise NotImplementedError

    def load(self, dir_path: Path) -> "BaseForecaster":
        raise NotImplementedError