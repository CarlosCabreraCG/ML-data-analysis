"""
Grid-search declarativo. Itera combinaciones de un JSON con bloque 'search',
materializa configs temporales y las ejecuta con el pipeline correspondiente.

Estrategia:
  - Cada JSON que declare 'search' produce N ejecuciones (una por combo).
  - Los resultados quedan bajo outputs/{csv}/{config_stem}/run_XXXX (estándar).
  - El ranking global se guarda en
        outputs/{csv}/_gridsearch/{config_stem}/summary.json
    para no contaminar la carpeta estándar de runs.
"""
from __future__ import annotations

import itertools
import json
import random
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.config import Config
from src.pipelines import anomaly, clustering, forecast
from src.utils import dump_json, setup_logger, timestamp


# ---- utilidades de mutación anidada ----

def _set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _materialize_combos(base: dict, grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        cfg = deepcopy(base)
        for k, v in zip(keys, combo):
            _set_path(cfg, k, v)
        combos.append(cfg)
    return combos


def _sample_combos(combos: list[dict], n_iter: int | None, seed: int) -> list[dict]:
    if n_iter is None or n_iter >= len(combos):
        return combos
    rng = random.Random(seed)
    return rng.sample(combos, n_iter)


def _metric_of(metrics: dict, task: str, metric: str) -> float | None:
    """Extrae la métrica de interés del dict de métricas del pipeline."""
    if task == "forecasting":
        return metrics.get("global", {}).get(metric)
    return metrics.get(metric)


def _higher_is_better(task: str, metric: str) -> bool:
    return metric in {"pr_auc", "f1", "recall", "silhouette", "score_mean"}


def _dispatch_pipeline(cfg: Config, cfg_path: Path, data_dir: Path,
                       output_dir: Path, logger):
    if cfg.task == "anomaly_detection":
        return anomaly.run(cfg, cfg_path, data_dir, output_dir, save=True, logger=logger)
    if cfg.task == "forecasting":
        return forecast.run(cfg, cfg_path, data_dir, output_dir, save=True, logger=logger)
    if cfg.task == "regime_clustering":
        return clustering.run(cfg, cfg_path, data_dir, output_dir, save=True, logger=logger)
    raise ValueError(cfg.task)


# ---- ejecución de un JSON con búsqueda ----

def run_search(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    logger,
    save: bool = True,
) -> dict:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "search" not in raw:
        raise ValueError(f"{config_path} no tiene bloque 'search'.")

    search = raw["search"]
    grid = search.get("param_grid", {})
    n_iter = search.get("n_iter")
    seed = search.get("random_state", 42)
    metric = search.get("metric", "rmse")

    all_combos = _materialize_combos(raw, grid)
    combos = _sample_combos(all_combos, n_iter, seed)
    logger.info(f"[gridsearch] {config_path.name}: {len(combos)} combinaciones")

    # clonar la config SIN el bloque search para evitar recursión
    base_wo_search = {k: v for k, v in raw.items() if k != "search"}
    cfg_template = Config.load(_write_tmp(base_wo_search))
    task = cfg_template.task
    higher = search.get("higher_is_better")
    if higher is None:
        higher = _higher_is_better(task, metric)

    records = []
    for i, combo in enumerate(combos, 1):
        combo_full = {**combo, "output": combo.get("output", {})}
        # materializar a disco para respetar la semántica de Config.load
        cfg_tmp = _write_tmp(combo_full)
        cfg = Config.load(cfg_tmp)
        logger.info(f"[gridsearch] ({i}/{len(combos)}) params={_extract_params(combo)}")

        try:
            result = _dispatch_pipeline(cfg, cfg_tmp, data_dir, output_dir, logger)
            score = _metric_of(result.get("metrics", {}), task, metric)
        except Exception as e:                     # una combo no debe tumbar el batch
            logger.warning(f"[gridsearch] combo falló: {e}")
            result, score = {"error": str(e)}, None

        records.append({
            "combo_index": i,
            "params": _extract_params(combo),
            "metric": metric,
            "score": score,
            "metrics": result.get("metrics", {}),
            "config": combo_full,
        })
        cfg_tmp.unlink(missing_ok=True)

    # ranking
    valid = [r for r in records if r["score"] is not None]
    valid.sort(key=lambda r: r["score"], reverse=higher)

    summary = {
        "config_file": str(config_path),
        "task": task,
        "metric": metric,
        "higher_is_better": higher,
        "n_combos": len(records),
        "n_valid": len(valid),
        "timestamp": timestamp(),
        "best": valid[0] if valid else None,
        "ranking": valid,
        "failures": [r for r in records if r["score"] is None],
    }

    if save:
        sdir = output_dir / Path(cfg_template.csv).stem / "_gridsearch" / config_path.stem
        dump_json(sdir / "summary.json", summary)
        # también YAML por legibilidad
        (sdir / "summary.yaml").write_text(
            yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info(f"[gridsearch] ranking → {sdir / 'summary.json'}")

    return summary


# ---- helpers ----

def _write_tmp(cfg_dict: dict) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(cfg_dict, f); f.close()
    return Path(f.name)


def _extract_params(combo: dict) -> dict:
    """Solo los overrides que introdujo el grid (no la config completa)."""
    return {k: v for k, v in combo.items()
            if k.startswith("model.") or k.startswith("features.")
            or k.startswith("threshold.")}