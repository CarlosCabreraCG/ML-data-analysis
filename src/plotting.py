"""Plots por tarea. Se guardan bajo <run_dir>/plots/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_anomaly_overview(
    df: pd.DataFrame, scores: np.ndarray, preds: np.ndarray,
    threshold: float, sensors: list[str], out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) histograma de scores con threshold
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_title("Distribución de anomaly score"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "score_hist.png", dpi=130); plt.close(fig)

    # 2) serie temporal con anomalías marcadas
    for s in sensors:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df[s], color="black", lw=0.8, label=s)
        anom = df.index[preds.astype(bool)]
        if len(anom):
            ax.scatter(anom, df.loc[anom, s], color="crimson", s=10, zorder=3, label="anomalía")
        ax.set_title(f"{s} — detección de anomalías"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / f"timeseries_{s}.png", dpi=130); plt.close(fig)


def plot_forecast(
    y_true: np.ndarray, y_pred: np.ndarray, target: str,
    horizon: int, out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    idx = np.arange(len(y_true))
    axes[0].plot(idx, y_true, label="real", color="black", lw=1)
    axes[0].plot(idx, y_pred, label="pred", color="tab:orange", lw=1)
    axes[0].set_title(f"{target} — real vs pred (H={horizon})"); axes[0].legend()

    err = y_true - y_pred
    axes[1].hist(err, bins=60, color="teal", alpha=0.8)
    axes[1].set_title(f"Residuales — {target}"); axes[1].set_xlabel("error")
    fig.tight_layout(); fig.savefig(out_dir / f"forecast_{target}.png", dpi=130); plt.close(fig)