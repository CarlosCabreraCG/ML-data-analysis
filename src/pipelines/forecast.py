"""Pipeline de forecasting con evaluación rolling-origin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data import build_forecast_frame, load_csv
from src.models.registry import build_forecaster
from src.plotting import plot_forecast
from src.utils import (dump_json, next_run_id, run_dir, save_model_bundle,
                       timestamp)


def _metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae":  float(np.mean(np.abs(y_true - y_pred))),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100),
    }
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    m["smape"] = float(np.mean(np.abs(y_true - y_pred) / np.clip(denom, 1e-8, None)) * 100)
    return m


def _rolling_eval(model, df_feats, target, horizon, stride):
    """Rolling-origin: en cada ancla predice horizonte H, compara con reales."""
    y_true_all, y_pred_all = [], []
    n = len(df_feats)
    start = 0
    for anchor in range(start, n - horizon, stride):
        window = df_feats.iloc[: anchor + 1]
        # Reconstruir la última fila con datos disponibles hasta el ancla
        try:
            preds = model.predict(window, horizon)[target]
        except Exception:
            continue
        actual = df_feats[target].iloc[anchor + 1: anchor + 1 + horizon].values
        if len(actual) < horizon:
            break
        y_true_all.extend(actual.tolist())
        y_pred_all.extend(preds.tolist())
    return np.array(y_true_all), np.array(y_pred_all)


def run(cfg: Config, config_path: Path, data_dir: Path,
        output_dir: Path, save: bool, logger) -> dict:
    cfg.validate()
    logger.info(f"[forecast] CSV={cfg.csv} config={config_path.name}")

    df = load_csv(data_dir / cfg.csv, cfg.data)

    # Construimos features con lags del target + exógenas
    feats_df = build_forecast_frame(df, cfg.target, cfg.features, cfg.target.horizon)
    feature_cols = [c for c in feats_df.columns if c not in cfg.target.columns]

    # Split temporal
    n = len(feats_df)
    n_train = int(n * cfg.split.train_frac)
    n_val = int(n * cfg.split.val_frac)
    train_df = feats_df.iloc[:n_train]
    test_df = feats_df.iloc[n_train + n_val:]      # test puro

    params = {**cfg.model.params,
              "targets": cfg.target.columns,
              "horizon": cfg.target.horizon}
    model = build_forecaster(cfg.model.type, params).fit(train_df, feature_cols)

    # Rolling eval sobre el bloque de test
    stride = cfg.split.eval_stride or cfg.target.horizon
    all_true, all_pred, per_target = [], [], {}
    for tgt in cfg.target.columns:
        yt, yp = _rolling_eval(model, test_df, tgt, cfg.target.horizon, stride)
        if len(yt) == 0:
            logger.warning(f"[forecast] sin datos suficientes para evaluar {tgt}")
            continue
        per_target[tgt] = _metrics(yt, yp)
        per_target[tgt]["n_eval"] = int(len(yt))
        all_true.extend(yt.tolist()); all_pred.extend(yp.tolist())

    global_metrics = _metrics(all_true, all_pred) if all_true else {}
    metrics = {"global": global_metrics, "per_target": per_target}
    result = {"metrics": metrics}

    if not save:
        logger.info(f"[forecast] (sin --save) métricas: {metrics}")
        return result

    base = output_dir / Path(cfg.csv).stem / config_path.stem
    rid = next_run_id(base)
    rdir = run_dir(output_dir, cfg.csv, config_path, rid)
    logger.info(f"[forecast] guardando en {rdir}")

    dump_json(rdir / "run_metadata.json", {
        "task": cfg.task, "run_id": rid, "timestamp": timestamp(),
        "csv": cfg.csv, "config_file": str(config_path),
        "config": json.loads(Path(config_path).read_text()),
        "feature_cols": feature_cols,
    })

    if cfg.output.save_model:
        mdir = rdir / "model"
        save_model_bundle(model, mdir)
        dump_json(mdir / "meta.json", {
            "type": cfg.model.type, "params": cfg.model.params,
            "task": "forecasting",
            "feature_cols": feature_cols,
            "targets": cfg.target.columns,
            "horizon": cfg.target.horizon,
        })

    dump_json(rdir / "metrics.json", metrics)

    if cfg.output.save_plots and all_true:
        # Recomputar por target para plots
        for tgt in cfg.target.columns:
            yt, yp = _rolling_eval(model, test_df, tgt, cfg.target.horizon, stride)
            if len(yt):
                plot_forecast(yt, yp, tgt, cfg.target.horizon, rdir / "plots")

    logger.info(f"[forecast] OK → {rdir}")
    return result