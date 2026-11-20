"""Wrappers sklearn para detección no supervisada."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.models.base import BaseAnomalyModel


class _SkAnomaly(BaseAnomalyModel):
    def fit(self, X):
        self.model.fit(X)
        return self

    def score_samples(self, X):
        # score = -score_samples(normal)  -> mayor = más anómalo
        return -self.model.score_samples(X)

    def save(self, dir_path):
        joblib.dump(
            {"model": self.model, "params": self.params, "threshold_": self.threshold_},
            dir_path / "model.joblib",
        )

    def load(self, dir_path):
        bundle = joblib.load(dir_path / "model.joblib")
        self.model = bundle["model"]
        self.params = bundle["params"]
        self.threshold_ = bundle["threshold_"]
        return self


class IsolationForestModel(_SkAnomaly):
    name = "isolation_forest"

    def __init__(self, **params):
        super().__init__(**params)
        self.model = IsolationForest(**params)


class LOFModel(_SkAnomaly):
    name = "local_outlier_factor"

    def __init__(self, **params):
        super().__init__(**params)
        p = {**params, "novelty": True}     # novelty=True permite score en nuevos datos
        self.model = LocalOutlierFactor(**p)


class OneClassSVMModel(_SkAnomaly):
    name = "one_class_svm"

    def __init__(self, **params):
        super().__init__(**params)
        self.model = OneClassSVM(**params)