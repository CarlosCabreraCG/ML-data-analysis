"""
Configuración declarativa por (CSV, tarea, modelo).
Cada JSON contiene toda la información necesaria para entrenar y evaluar.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Task = Literal["anomaly_detection", "forecasting", "regime_clustering"]


# ---------- sub-bloques ----------

@dataclass
class DataCfg:
    datetime_column: str = "datetime"
    datetime_format: str | None = None
    resample: str | None = None            # ej. "5min", None = sin resample
    interpolation_limit: int = 3


@dataclass
class FeatureCfg:
    # Anomalías
    sensors: list[str] = field(default_factory=list)
    window_size: int = 1                    # 1 = point; >1 = contextual
    rolling_windows: list[int] = field(default_factory=list)
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std"])
    diff_lags: list[int] = field(default_factory=list)
    standardize: bool = True
    # Forecasting
    exogenous: list[str] = field(default_factory=list)
    lags: list[int] = field(default_factory=list)
    calendar: list[str] = field(default_factory=list)   # hour, weekday, ...


@dataclass
class SplitCfg:
    strategy: str = "temporal"
    train_frac: float = 0.7
    val_frac: float = 0.15                  # solo forecasting
    eval_stride: int | None = None          # rolling-origin, default = horizon


@dataclass
class TargetCfg:
    columns: list[str] = field(default_factory=list)
    horizon: int = 12


@dataclass
class ModelCfg:
    type: str = ""
    strategy: str = "direct"                # direct | recursive (forecasting)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdCfg:
    strategy: Literal["quantile", "std"] = "quantile"
    value: float = 0.95


@dataclass
class EvaluationCfg:
    label_column: str | None = None
    metrics: list[str] = field(default_factory=lambda: ["anomaly_rate", "fpr"])


@dataclass
class OutputCfg:
    save_model: bool = True
    save_plots: bool = True
    save_predictions: bool = True

@dataclass
class ReductionCfg:
    """Reducción de dimensionalidad opcional antes del clustering."""
    method: Literal["none", "pca"] = "none"
    n_components: int | None = None
    whiten: bool = False


@dataclass
class RegimeCfg:
    reduction: ReductionCfg = field(default_factory=ReductionCfg)
    # asignación de régimen a datos nuevos: 'nearest_centroid' | 'predict'
    assign_strategy: Literal["predict", "nearest_centroid"] = "predict"


@dataclass
class SearchCfg:
    """
    Declaración opcional de búsqueda de hiperparámetros.
    El runner materializa una config por combinación y ejecuta train --save.
    """
    param_grid: dict[str, list] = field(default_factory=dict)   # {"model.params.X": [..]}
    n_iter: int | None = None                                    # None = full grid
    random_state: int = 42
    metric: str = "rmse"          # clave del dict de métricas (anomaly: 'pr_auc' | 'f1'; forecast: 'rmse')
    higher_is_better: bool | None = None   # auto según task/métrica

# ---------- raíz ----------

@dataclass
class Config:
    task: Task
    csv: str
    description: str = ""
    data: DataCfg = field(default_factory=DataCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    target: TargetCfg | None = None
    model: ModelCfg = field(default_factory=ModelCfg)
    threshold: ThresholdCfg = field(default_factory=ThresholdCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    regime: RegimeCfg | None = None
    search: SearchCfg | None = None
    
    # ---------- carga ----------
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            task=raw["task"],
            csv=raw["csv"],
            description=raw.get("description", ""),
            data=DataCfg(**raw.get("data", {})),
            features=FeatureCfg(**raw.get("features", {})),
            split=SplitCfg(**raw.get("split", {})),
            target=TargetCfg(**raw["target"]) if raw.get("target") else None,
            model=ModelCfg(**raw["model"]),
            threshold=ThresholdCfg(**raw.get("threshold", {})),
            evaluation=EvaluationCfg(**raw.get("evaluation", {})),
            output=OutputCfg(**raw.get("output", {})),
            regime=RegimeCfg(
                reduction=ReductionCfg(**(raw.get("regime", {}).get("reduction", {}))),
                assign_strategy=raw.get("regime", {}).get("assign_strategy", "predict"),
            ) if raw.get("regime") is not None else None,
            search=SearchCfg(**raw["search"]) if raw.get("search") else None,
        )

    def validate(self) -> None:
        if self.task == "anomaly_detection":
            if not self.features.sensors:
                raise ValueError("anomaly_detection requiere 'features.sensors'.")
        elif self.task == "forecasting":
            if not self.target or not self.target.columns:
                raise ValueError("forecasting requiere 'target.columns'.")
            if not self.features.lags:
                raise ValueError("forecasting requiere 'features.lags'.")
        elif self.task == "regime_clustering":
            if not self.features.sensors:
                raise ValueError("regime_clustering requiere 'features.sensors'.")
            if self.regime is None:
                raise ValueError("regime_clustering requiere bloque 'regime'.")
        else:
            raise ValueError(f"task desconocida: {self.task}")