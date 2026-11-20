"""Pipeline de detección de anomalías no supervisada."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data import build_anomaly_matrix, load_csv
from src.models.registry import build_anomaly_model
from src.plotting import plot_anomaly_overview
from src.utils import (dump_json, next_run_id, run_dir, save_model_bundle,
                       timestamp)


def _evaluate(scores, preds, labels):
    out = {
        "n_samples": int(len(scores)),
        "anomaly_rate": float(preds.mean()),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_p95": float(np.quantile(scores, 0.95)),
    }
    if labels is not None:
        y = labels.astype(int)
        out["pr_auc"] = float(average_precision_score(y, scores))
        out["f1"] = float(f1_score(y, preds, zero_division=0))
        out["fpr"] = float(((preds == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
        out["recall"] = float(((preds == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    return out


def run(cfg: Config, config_path: Path, data_dir: Path,
        output_dir: Path, save: bool, logger) -> dict:
    cfg.validate()
    logger.info(f"[anomaly] CSV={cfg.csv} config={config_path.name}")

    df = load_csv(data_dir / cfg.csv, cfg.data)
    labels = df[cfg.evaluation.label_column].values if cfg.evaluation.label_column in df.columns else None

    X, feat_names, timestamps = build_anomaly_matrix(df, cfg.features)
    logger.info(f"Matriz de features: {X.shape}")

    if cfg.features.standardize:
        scaler = StandardScaler().fit(X[: int(len(X) * cfg.split.train_frac)])
        X = scaler.transform(X)
    else:
        scaler = None

    # Split temporal — se entrena SOLO sobre la porción "normal"
    n_train = int(len(X) * cfg.split.train_frac)
    X_train, X_all = X[:n_train], X

    model = build_anomaly_model(cfg.model.type, cfg.model.params).fit(X_train)
    train_scores = model.score_samples(X_train)
    model.set_threshold(train_scores, cfg.threshold.strategy, cfg.threshold.value)

    scores = model.score_samples(X_all)
    preds = model.predict(X_all)

    metrics = _evaluate(scores, preds, labels)
    metrics["threshold"] = model.threshold_
    metrics["feature_names"] = feat_names
    metrics["n_features"] = int(X.shape[1])

    result = {"metrics": metrics, "n_samples": len(X)}

    if not save:
        logger.info(f"[anomaly] (sin --save) métricas: {metrics}")
        return result

    base = output_dir / Path(cfg.csv).stem / config_path.stem
    rid = next_run_id(base)
    rdir = run_dir(output_dir, cfg.csv, config_path, rid)
    logger.info(f"[anomaly] guardando en {rdir}")

    # 1) metadata
    dump_json(rdir / "run_metadata.json", {
        "task": cfg.task, "run_id": rid, "timestamp": timestamp(),
        "csv": cfg.csv, "config_file": str(config_path),
        "config": json.loads(Path(config_path).read_text()),
        "feature_names": feat_names,
    })

    # 2) modelo
    if cfg.output.save_model:
        mdir = rdir / "model"
        save_model_bundle(model, mdir)
        # guardar scaler + meta para predict
        import joblib
        if scaler is not None:
            joblib.dump(scaler, mdir / "scaler.joblib")
        dump_json(mdir / "meta.json", {
            "type": cfg.model.type, "params": cfg.model.params,
            "task": "anomaly_detection",
            "threshold_": model.threshold_,
            "feature_names": feat_names,
        })

    # 3) métricas
    dump_json(rdir / "metrics.json", metrics)

    # 4) predicciones
    if cfg.output.save_predictions:
        pdf = df.loc[timestamps].copy()
        pdf["anomaly_score"] = scores
        pdf["is_anomaly"] = preds
        pdf.to_csv(rdir / "predictions.csv")

    # 5) plots
    if cfg.output.save_plots:
        plot_anomaly_overview(df.loc[timestamps], scores, preds,
                              model.threshold_, cfg.features.sensors,
                              rdir / "plots")

    logger.info(f"[anomaly] OK → {rdir}")
    return result