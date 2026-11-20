"""Wrappers de clustering de regímenes operativos."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.models.base import BaseAnomalyModel  # reutilizamos contrato fit/predict/save/load

try:
    import hdbscan as _hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False


class BaseRegimeModel(BaseAnomalyModel):
    """
    Interfaz común para clustering de regímenes.
    - fit(X)   → entrena
    - score_samples(X)  → -distancia mínima al centro (mayor = más lejos de todo centro)
                           o -probabilidad GMM; sirve como anomalía vs regímenes conocidos.
    - predict(X)        → etiqueta de régimen (o -1 si ruido)
    """

    name = "base_regime"

    def __init__(self, **params):
        super().__init__(**params)
        self.n_regimes_: int | None = None

    def regime_stats(self, X: np.ndarray, labels: np.ndarray) -> dict:
        """Distribución de tamaño por régimen + centroides (si existen)."""
        out = {"regime_counts": {}}
        for k in np.unique(labels):
            out["regime_counts"][int(k)] = int((labels == k).sum())
        if hasattr(self.model, "cluster_centers_"):
            out["centroids"] = self.model.cluster_centers_.tolist()
        return out


class KMeansRegime(BaseRegimeModel):
    name = "kmeans"

    def __init__(self, n_clusters: int = 4, random_state: int = 42, **params):
        super().__init__(n_clusters=n_clusters, random_state=random_state, **params)
        from sklearn.cluster import KMeans
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, **params)

    def fit(self, X):
        self.model.fit(X)
        self.n_regimes_ = self.model.n_clusters
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        # distancia mínima al centroide → mayor = más lejos de cualquier régimen
        d = self.model.transform(X)                    # [N, K]
        return -d.min(axis=1)

    def save(self, d):
        joblib.dump({"model": self.model, "params": self.params,
                     "n_regimes_": self.n_regimes_}, d / "model.joblib")

    def load(self, d):
        b = joblib.load(d / "model.joblib")
        self.model, self.params, self.n_regimes_ = b["model"], b["params"], b["n_regimes_"]
        return self


class GMMRegime(BaseRegimeModel):
    name = "gmm"

    def __init__(self, n_components: int = 4, covariance_type: str = "full",
                 random_state: int = 42, **params):
        super().__init__(n_components=n_components, random_state=random_state, **params)
        from sklearn.mixture import GaussianMixture
        self.model = GaussianMixture(n_components=n_components,
                                     covariance_type=covariance_type,
                                     random_state=random_state, **params)

    def fit(self, X):
        self.model.fit(X)
        self.n_regimes_ = self.model.n_components
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        # log-likelihood → negativa = más anómalo (más improbable)
        return -self.model.score_samples(X)

    def save(self, d):
        joblib.dump({"model": self.model, "params": self.params,
                     "n_regimes_": self.n_regimes_}, d / "model.joblib")

    def load(self, d):
        b = joblib.load(d / "model.joblib")
        self.model, self.params, self.n_regimes_ = b["model"], b["params"], b["n_regimes_"]
        return self


class HDBSCANRegime(BaseRegimeModel):
    """
    HDBSCAN detecta densidad y descarta ruido (label -1).
    Nota: no tiene .predict; usamos approximate_predict del paquete hdbscan.
    """

    name = "hdbscan"

    def __init__(self, min_cluster_size: int = 50, min_samples: int | None = None,
                 metric: str = "euclidean", **params):
        if not _HAS_HDBSCAN:
            raise ImportError("Instala `hdbscan`: pip install hdbscan")
        super().__init__(min_cluster_size=min_cluster_size,
                         min_samples=min_samples, metric=metric, **params)
        self.model = None

    def fit(self, X):
        self.model = _hdbscan.HDBSCAN(
            min_cluster_size=self.params["min_cluster_size"],
            min_samples=self.params["min_samples"],
            metric=self.params["metric"],
            prediction_data=True,
        ).fit(X)
        labels = self.model.labels_
        self.n_regimes_ = int(len(set(labels)) - (1 if -1 in labels else 0))
        return self

    def predict(self, X):
        labels, _ = _hdbscan.approximate_predict(self.model, X)
        return labels

    def score_samples(self, X):
        _, strength = _hdbscan.approximate_predict(self.model, X)
        return -np.asarray(strength)                    # menor pertenencia = más anómalo

    def save(self, d):
        joblib.dump({"model": self.model, "params": self.params,
                     "n_regimes_": self.n_regimes_}, d / "model.joblib")

    def load(self, d):
        b = joblib.load(d / "model.joblib")
        self.model, self.params, self.n_regimes_ = b["model"], b["params"], b["n_regimes_"]
        return self