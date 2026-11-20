"""Pipeline de clustering de regímenes operativos."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data import build_anomaly_matrix, load_csv
from src.models.registry import build_regime_model
from src.plotting import plot_regime_timeline, plot_regimes_2d
from src.utils import (dump_json, next_run_id, run_dir, save_model_bundle,
                       timestamp)


def _fit_reducer(X: np.ndarray, red_cfg, seed: int = 42):
    if red_cfg.method == "pca":
        n_comp = red_cfg.n_components or min(2, X.shape[1])
        reducer = PCA(n_components=n_comp, whiten=red_cfg.whiten, random_state=seed)
        return reducer, reducer.fit_transform(X)
    return None, X


def _valid_for_score(labels: np.ndarray) -> bool:
    uniq = set(np.unique(labels))
    uniq.discard(-1)
    return len(uniq) >= 2


def run(cfg: Config, config_path: Path, data_dir: Path,
        output_dir: Path, save: bool, logger) -> dict:
    cfg.validate()
    logger.info(f"[clustering] CSV={cfg.csv} config={config_path.name}")

    df = load_csv(data_dir / cfg.csv, cfg.data)
    X, feat_names, timestamps = build_anomaly_matrix(df, cfg.features)
    logger.info(f"[clustering] matriz: {X.shape}")

    if cfg.features.standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    else:
        scaler = None

    reducer, X_red = _fit_reducer(X, cfg.regime.reduction)

    model = build_regime_model(cfg.model.type, cfg.model.params).fit(X_red)
    labels = model.predict(X_red)
    scores = model.score_samples(X_red)

    # ---- métricas ----
    metrics = {
        "n_samples": int(len(X)),
        "n_regimes": int(model.n_regimes_),
        "noise_frac": float((labels == -1).mean()) if -1 in labels else 0.0,
        "regime_counts": {int(k): int(v) for k, v in
                          zip(*np.unique(labels, return_counts=True))},
    }
    if _valid_for_score(labels):
        mask = labels != -1
        metrics["silhouette"] = float(silhouette_score(X_red[mask], labels[mask]))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_red[mask], labels[mask]))

    result = {"metrics": metrics}

    if not save:
        logger.info(f"[clustering] (sin --save) métricas: {metrics}")
        return result

    base = output_dir / Path(cfg.csv).stem / config_path.stem
    rid = next_run_id(base)
    rdir = run_dir(output_dir, cfg.csv, config_path, rid)
    logger.info(f"[clustering] guardando en {rdir}")

    dump_json(rdir / "run_metadata.json", {
        "task": cfg.task, "run_id": rid, "timestamp": timestamp(),
        "csv": cfg.csv, "config_file": str(config_path),
        "config": json.loads(Path(config_path).read_text()),
        "feature_names": feat_names,
    })

    if cfg.output.save_model:
        mdir = rdir / "model"
        save_model_bundle(model, mdir)
        if scaler is not None:
            joblib.dump(scaler, mdir / "scaler.joblib")
        if reducer is not None:
            joblib.dump(reducer, mdir / "reducer.joblib")
        dump_json(mdir / "meta.json", {
            "type": cfg.model.type, "params": cfg.model.params,
            "task": "regime_clustering",
            "reduction": cfg.regime.reduction.method,
            "feature_names": feat_names,
        })

    dump_json(rdir / "metrics.json", metrics)

    if cfg.output.save_predictions:
        import pandas as pd
        pdf = df.loc[timestamps].copy()
        pdf["regime"] = labels
        pdf["regime_score"] = scores
        pdf.to_csv(rdir / "predictions.csv")

    if cfg.output.save_plots:
        pdir = rdir / "plots"
        # Proyectar a 2D para visualizar (si no es PCA, hacemos PCA auxiliar)
        if X_red.shape[1] >= 2:
            viz = X_red[:, :2]
        else:
            viz = PCA(n_components=2).fit_transform(X_red)
        plot_regimes_2d(viz, labels, pdir)
        signal = df.loc[timestamps, cfg.features.sensors[0]].values
        plot_regime_timeline(timestamps, labels, pdir, target_col=signal)

    logger.info(f"[clustering] OK → {rdir}")
    return result