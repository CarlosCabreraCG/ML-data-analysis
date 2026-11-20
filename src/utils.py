"""Utilidades de rutas, run-id y persistencia."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


def setup_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("sensor_analytics")


def config_stem(config_path: Path) -> str:
    return config_path.stem


def csv_stem(csv_name: str) -> str:
    return Path(csv_name).stem


def run_base_dir(output_dir: Path, csv_name: str, config_path: Path) -> Path:
    """outputs/{csv_stem}/{config_stem}/"""
    return output_dir / csv_stem(csv_name) / config_stem(config_path)


def next_run_id(base_dir: Path) -> str:
    """Devuelve 'run_0001', 'run_0002', ... (por par csv+config)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit():
            nums.append(int(d.name[4:]))
    return f"run_{(max(nums) + 1) if nums else 1:04d}"


def run_dir(output_dir: Path, csv_name: str, config_path: Path, run_id: str) -> Path:
    d = run_base_dir(output_dir, csv_name, config_path) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model_bundle(model, dir_path: Path) -> None:
    """Cada wrapper define .save(dir). Fallback joblib si no lo implementa."""
    dir_path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save"):
        model.save(dir_path)
    else:
        joblib.dump(model, dir_path / "model.joblib")


def load_model_bundle(kind: str, dir_path: Path):
    from src.models.registry import build_anomaly_model, build_forecaster
    meta = load_json(dir_path / "meta.json")
    if kind == "anomaly":
        return build_anomaly_model(meta["type"], meta["params"]).load(dir_path)
    return build_forecaster(meta["type"], meta["params"]).load(dir_path)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")