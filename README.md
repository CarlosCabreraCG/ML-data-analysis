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

# Clustering de regímenes (KMeans)
sensor-analytics train --config configs/clustering_regimes.json --save

# HDBSCAN
sensor-analytics train --config configs/clustering_hdbscan.json --save

# Grid-search puntual
sensor-analytics search --config configs/anomaly_isoforest_search.json

# Usando --search con el subcomando train
sensor-analytics train --config configs/forecast_lgbm_search.json --search

# Batch: ejecuta TODOS los JSON de una carpeta
sensor-analytics batch --configs-dir configs/ --save


# Reutilizar modelo guardado
sensor-analytics predict \
  --run-dir outputs/sensor_data_1/forecast_lgbm/run_0001 \
  --save
```

## Tests

```bash
python -m pytest tests/ -v -W error::RuntimeWarning
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