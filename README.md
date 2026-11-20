## `README.md`

```markdown
# Sensor Analytics

Pipeline profesional para **detección de anomalías no supervisada** y **forecasting** sobre series temporales multivariables de sensores.

## Instalación
```bash
pip install -e ".[dev]"        # base + tests
pip install -e ".[dev,torch]"  # + Autoencoder
```

## Uso

```bash
# Entrenar (sin guardar nada)
sensor-analytics train --config configs/anomaly_isoforest.json

# Entrenar y guardar modelo + métricas + plots
sensor-analytics train --config configs/anomaly_isoforest.json --save

# Segunda corrida del mismo CSV + config  → genera run_0002 automáticamente
sensor-analytics train --config configs/anomaly_isoforest.json --save

# Forecast
sensor-analytics train --config configs/forecast_lgbm.json --save

# Reutilizar modelo guardado
sensor-analytics predict \
  --run-dir outputs/sensor_data_1/forecast_lgbm/run_0001 \
  --save
```

## Layout de resultados

```
outputs/
└── sensor_data_1/                    # <- por CSV de entrenamiento
    └── anomaly_isoforest/            # <- por configuración JSON
        └── run_0001/                 # <- por corrida
            ├── run_metadata.json
            ├── metrics.json
            ├── predictions.csv
            ├── model/
            │   ├── model.joblib
            │   ├── meta.json
            │   └── scaler.joblib
            └── plots/
                ├── score_hist.png
                └── timeseries_temperatura_M1.png
```

## Añadir un modelo nuevo

1. Crear clase que herede de `BaseAnomalyModel` o `BaseForecaster` con `fit / score / predict / save / load`.
2. Registrarlo en `src/models/registry.py`.
3. Declararlo en el JSON: `"model": {"type": "<nombre>", "params": {...}}`.

## Notas de diseño

- **Guardado opcional**: solo con `--save`. Sin él, se entrena y evalúa en memoria.
- **Sin leakage**: anomalías entrenan solo con el tramo inicial "normal"; forecasting construye lags con `shift` y respeta split temporal estricto.
- **Multi-config por CSV**: cada JSON define una receta independiente; conviven sin colisión gracias al triple `(csv, config, run)`.
- **Ventaneo unificado**: `features.window_size > 1` convierte cualquier modelo tabular en detector contextual (útil para Autoencoder temporal).
```

---

## `tests/test_smoke.py`

```python
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
```

---

## Puntos clave a destacar para el equipo

1. **Identificación de corrida**: `outputs/{csv}/{config}/run_XXXX/`. Al lanzar el mismo JSON dos veces se crea `run_0002`, etc. Nunca se pisa. `run_metadata.json` guarda config, CSV, timestamp, features.
2. **`--save` es explícito**. Por defecto no se escribe nada (útil para grid search rápido en memoria).
3. **Anomalías (no supervisado)**: split temporal → `fit` solo con la fracción inicial (asumida "normal") → threshold por **cuantil de scores de entrenamiento** (configurable a `std`). Métricas con y sin labels.
4. **Forecasting**: estrategia **directa** (un modelo por target×horizonte) porque evita acumulación de error del esquema recursivo, lo cual es crítico con LGBM. Evaluación **rolling-origin** con `eval_stride` (por defecto = horizonte).
5. **Ventana unificada** (`window_size`) para anomalías: permite desde point-anomaly (1) hasta contextual (AE sobre ventanas).
6. **Extensión trivial**:
   - Nuevo modelo: 1 clase + 1 línea en `registry.py` + `"type"` en el JSON.
   - Nuevo CSV: solo se agrega el `.csv` a `data/` y se referencia en el JSON.
   - Varios modelos sobre el mismo CSV: duplicar JSON cambiando `model.type` y parámetros.
7. **Clustering de regímenes** (mencionado como adicional): encaja como tercer pipeline (`src/pipelines/clustering.py`) con `KMeans / HDBSCAN / GMM` y métricas `Silhouette / Davies-Bouldin`, reutilizando `data.py` y `registry.py` sin tocar nada de lo anterior.

Si quieres, en el siguiente paso puedo añadir el **tercer pipeline de clustering de regímenes** y un **grid-search runner** que itere todos los JSON de una carpeta.