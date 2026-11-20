"""Forecasters: LightGBM (estrategia directa) y SARIMAX."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster


class LGBMForecaster(BaseForecaster):
    """
    Un modelo por (target, horizonte) — estrategia 'directa'.
    Evita acumular error del esquema recursivo.
    """
    name = "lightgbm"

    def __init__(self, targets: list[str], horizon: int, **params):
        super().__init__(targets=targets, horizon=horizon, **params)
        self.models_: dict[str, dict[int, object]] = {}
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]):
        import lightgbm as lgb
        self.feature_names_ = feature_cols
        for tgt in self.targets:
            self.models_[tgt] = {}
            for h in range(1, self.horizon + 1):
                y = df[tgt].shift(-h)
                mask = y.notna()
                X = df.loc[mask, feature_cols]
                m = lgb.LGBMRegressor(**self.params)
                m.fit(X, y[mask])
                self.models_[tgt][h] = m
        return self

    def predict(self, df_last: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
        """df_last: DataFrame con la última fila de features ya construida."""
        preds: dict[str, np.ndarray] = {}
        x = df_last[self.feature_names_].iloc[[0]]
        for tgt in self.targets:
            arr = [self.models_[tgt][h].predict(x)[0]
                   for h in range(1, horizon + 1)]
            preds[tgt] = np.array(arr)
        return preds

    def save(self, dir_path):
        joblib.dump(
            {"models_": self.models_, "params": self.params,
             "targets": self.targets, "horizon": self.horizon,
             "feature_names_": self.feature_names_},
            dir_path / "model.joblib",
        )
        (dir_path / "meta.json").write_text(json.dumps(
            {"type": self.name, "params": self.params}, indent=2))

    def load(self, dir_path):
        b = joblib.load(dir_path / "model.joblib")
        self.models_ = b["models_"]; self.params = b["params"]
        self.targets = b["targets"]; self.horizon = b["horizon"]
        self.feature_names_ = b["feature_names_"]
        return self


class SARIMAXForecaster(BaseForecaster):
    name = "sarimax"

    def __init__(self, targets: list[str], horizon: int,
                 order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                 exog: list[str] | None = None, **params):
        super().__init__(targets=targets, horizon=horizon, **params)
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.exog_cols = exog or []
        self.results_: dict[str, object] = {}

    def fit(self, df: pd.DataFrame, feature_cols: list[str]):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        for tgt in self.targets:
            exog = df[self.exog_cols].values if self.exog_cols else None
            mod = SARIMAX(df[tgt].values, exog=exog,
                          order=self.order, seasonal_order=self.seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            self.results_[tgt] = mod.fit(disp=False)
        return self

    def predict(self, df_last: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
        exog = None
        if self.exog_cols:
            # persistencia de exógenas: usar última fila repetida (simple, documentado)
            last = df_last[self.exog_cols].iloc[-1].values
            exog = np.tile(last, (horizon, 1))
        out: dict[str, np.ndarray] = {}
        for tgt, res in self.results_.items():
            out[tgt] = np.asarray(res.forecast(steps=horizon, exog=exog))
        return out

    def save(self, dir_path):
        joblib.dump({"results_": self.results_, "order": self.order,
                     "seasonal_order": self.seasonal_order,
                     "targets": self.targets, "exog_cols": self.exog_cols,
                     "horizon": self.horizon},
                    dir_path / "model.joblib")
        (dir_path / "meta.json").write_text(json.dumps(
            {"type": self.name, "params": {}}, indent=2))

    def load(self, dir_path):
        b = joblib.load(dir_path / "model.joblib")
        self.results_ = b["results_"]; self.order = b["order"]
        self.seasonal_order = b["seasonal_order"]; self.targets = b["targets"]
        self.exog_cols = b["exog_cols"]; self.horizon = b["horizon"]
        return self