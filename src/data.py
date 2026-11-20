"""Carga de CSV y preprocesamiento / feature engineering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DataCfg, FeatureCfg, TargetCfg


# ---------------- carga ----------------

def load_csv(path: Path, cfg: DataCfg) -> pd.DataFrame:
    df = pd.read_csv(path)
    if cfg.datetime_column not in df.columns:
        raise ValueError(f"'{cfg.datetime_column}' no está en {path}.")
    df[cfg.datetime_column] = pd.to_datetime(
        df[cfg.datetime_column], format=cfg.datetime_format, errors="coerce"
    )
    df = df.dropna(subset=[cfg.datetime_column]).sort_values(cfg.datetime_column)
    df = df.set_index(cfg.datetime_column).asfreq(cfg.resample) if cfg.resample else df.set_index(cfg.datetime_column)
    # Interpolación limitada (respeta huecos largos)
    df = df.interpolate(method="linear", limit=cfg.interpolation_limit).ffill().bfill()
    return df


# ---------------- features ----------------

def build_anomaly_matrix(df: pd.DataFrame, fc: FeatureCfg) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Construye matriz [N, F] (ventaneada si window_size>1) y devuelve timestamps."""
    cols = list(fc.sensors)
    base = df[cols].copy()
    feats = [base]

    for w in fc.rolling_windows:
        roll = base.rolling(w, min_periods=1)
        for stat in fc.rolling_stats:
            feats.append(getattr(roll, stat)().add_suffix(f"_{stat}_{w}"))

    for lag in fc.diff_lags:
        feats.append(base.diff(lag).add_suffix(f"_diff_{lag}"))

    feat_df = pd.concat(feats, axis=1)
    timestamps = feat_df.index

    arr = feat_df.to_numpy(dtype=float)
    # Drop filas iniciales con NaN (por rolling/diff)
    valid = ~np.isnan(arr).any(axis=1)
    arr, timestamps = arr[valid], timestamps[valid]

    if fc.window_size > 1:
        w = fc.window_size
        windows = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)  # [N-w+1, F, w]
        arr = windows.reshape(windows.shape[0], -1)                          # [N-w+1, F*w]
        timestamps = timestamps[w - 1:]                                      # causal: fin de ventana

    return arr, feat_df.columns.tolist(), timestamps


def build_forecast_frame(
    df: pd.DataFrame,
    target: TargetCfg,
    fc: FeatureCfg,
    horizon: int,
) -> pd.DataFrame:
    """Devuelve un DataFrame con y, lags, rolling, exógenas y calendario."""
    out = pd.DataFrame(index=df.index)
    # target
    for c in target.columns:
        out[c] = df[c].astype(float)

    # lags + rolling del target
    for c in target.columns:
        for lag in fc.lags:
            out[f"{c}_lag_{lag}"] = out[c].shift(lag)
        for w in fc.rolling_windows:
            for stat in fc.rolling_stats:
                out[f"{c}_{stat}_{w}"] = getattr(out[c].rolling(w), stat)()

    # exógenas (shift 1 para evitar leakage)
    for c in fc.exogenous:
        if c in df.columns:
            out[f"{c}_lag_1"] = df[c].shift(1).astype(float)

    # calendario
    for cal in fc.calendar:
        if cal == "hour":
            out["hour"] = df.index.hour
        elif cal == "weekday":
            out["weekday"] = df.index.dayofweek
        elif cal == "month":
            out["month"] = df.index.month

    out = out.dropna()
    return out