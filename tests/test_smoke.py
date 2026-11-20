"""Smoke test end-to-end con CSV sintético."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.pipelines import anomaly, forecast
from src.utils import setup_logger


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 1000
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    base = np.sin(np.arange(n) / 20) * 5 + 20
    df = pd.DataFrame({
        "datetime": ts,
        "temperatura_M1": base + rng.normal(0, 0.3, n),
        "temperatura_M2": base * 0.8 + rng.normal(0, 0.3, n),
        "temperatura_M3": base * 1.1 + rng.normal(0, 0.3, n),
    })
    df.loc[500:505, "temperatura_M1"] += 15  # anomalía inyectada
    p = tmp_path / "sensor_data_1.csv"
    df.to_csv(p, index=False)
    (tmp_path / "sensor_data_1.csv").rename(tmp_path / "sensor_data_1.csv")
    return tmp_path


def test_anomaly_smoke(synthetic_csv: Path, tmp_path: Path):
    cfg_dict = {
        "task": "anomaly_detection", "csv": "sensor_data_1.csv",
        "data": {"datetime_column": "datetime"},
        "features": {"sensors": ["temperatura_M1", "temperatura_M2"],
                     "rolling_windows": [5], "diff_lags": [1]},
        "split": {"train_frac": 0.6},
        "model": {"type": "isolation_forest",
                  "params": {"n_estimators": 50, "random_state": 0}},
        "threshold": {"strategy": "quantile", "value": 0.99},
    }
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg_dict))
    cfg = Config.load(cpath)
    out = anomaly.run(cfg, cpath, synthetic_csv, tmp_path / "out",
                      save=True, logger=setup_logger())
    assert "metrics" in out and out["metrics"]["n_samples"] > 0
    assert any((tmp_path / "out").rglob("metrics.json"))


def test_forecast_smoke(synthetic_csv: Path, tmp_path: Path):
    cfg_dict = {
        "task": "forecasting", "csv": "sensor_data_1.csv",
        "data": {"datetime_column": "datetime"},
        "target": {"columns": ["temperatura_M1"], "horizon": 5},
        "features": {"lags": [1, 2, 3], "calendar": ["hour"]},
        "split": {"train_frac": 0.6, "val_frac": 0.1, "eval_stride": 20},
        "model": {"type": "lightgbm",
                  "params": {"n_estimators": 50, "verbose": -1}},
    }
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg_dict))
    cfg = Config.load(cpath)
    out = forecast.run(cfg, cpath, synthetic_csv, tmp_path / "out",
                       save=True, logger=setup_logger())
    assert "global" in out["metrics"]
    assert out["metrics"]["global"]["rmse"] > 0

def test_clustering_smoke(synthetic_csv: Path, tmp_path: Path):
    cfg_dict = {
        "task": "regime_clustering", "csv": "sensor_data_1.csv",
        "data": {"datetime_column": "datetime"},
        "features": {"sensors": ["temperatura_M1", "temperatura_M2"],
                     "window_size": 3, "standardize": True},
        "regime": {"reduction": {"method": "pca", "n_components": 2}},
        "model": {"type": "kmeans", "params": {"n_clusters": 3, "random_state": 0}},
    }
    cpath = tmp_path / "cfg_clu.json"
    cpath.write_text(json.dumps(cfg_dict))
    cfg = Config.load(cpath)
    from src.pipelines import clustering
    out = clustering.run(cfg, cpath, synthetic_csv, tmp_path / "out",
                         save=True, logger=setup_logger())
    assert out["metrics"]["n_regimes"] == 3


def test_gridsearch_smoke(synthetic_csv: Path, tmp_path: Path):
    from src.gridsearch import run_search
    cfg_dict = {
        "task": "anomaly_detection", "csv": "sensor_data_1.csv",
        "data": {"datetime_column": "datetime"},
        "features": {"sensors": ["temperatura_M1"], "standardize": True},
        "split": {"train_frac": 0.6},
        "model": {"type": "isolation_forest",
                  "params": {"n_estimators": 50, "random_state": 0}},
        "threshold": {"strategy": "quantile", "value": 0.99},
        "output": {"save_model": True, "save_plots": False, "save_predictions": False},
        "search": {"param_grid": {"model.params.n_estimators": [30, 50]},
                   "metric": "anomaly_rate", "higher_is_better": False},
    }
    cpath = tmp_path / "cfg_search.json"
    cpath.write_text(json.dumps(cfg_dict))
    summary = run_search(cpath, synthetic_csv, tmp_path / "out",
                         setup_logger(), save=True)
    assert summary["n_combos"] == 2 and summary["best"] is not None