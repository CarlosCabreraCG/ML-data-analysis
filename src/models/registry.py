from src.models.anomaly import IsolationForestModel, LOFModel, OneClassSVMModel
from src.models.forecasting import LGBMForecaster, SARIMAXForecaster
from src.models.clustering import KMeansRegime, GMMRegime, HDBSCANRegime

_ANOMALY = {
    "isolation_forest": IsolationForestModel,
    "local_outlier_factor": LOFModel,
    "one_class_svm": OneClassSVMModel,
}

_FORECAST = {
    "lightgbm": LGBMForecaster,
    "sarimax": SARIMAXForecaster,
}

# Clustering:
_REGIME = {
    "kmeans": KMeansRegime,
    "gmm": GMMRegime,
    "hdbscan": HDBSCANRegime,
}


def build_anomaly_model(type_: str, params: dict):
    if type_ == "autoencoder":
        from src.models.autoencoder import AutoencoderModel
        return AutoencoderModel(**params)
    if type_ not in _ANOMALY:
        raise KeyError(f"Anomaly type desconocido: {type_}. Opciones: {list(_ANOMALY) + ['autoencoder']}")
    return _ANOMALY[type_](**params)


def build_forecaster(type_: str, params: dict):
    if type_ not in _FORECAST:
        raise KeyError(f"Forecast type desconocido: {type_}. Opciones: {list(_FORECAST)}")
    return _FORECAST[type_](**params)

def build_regime_model(type_: str, params: dict):
    if type_ not in _REGIME:
        raise KeyError(f"Regime type desconocido: {type_}. Opciones: {list(_REGIME)}")
    return _REGIME[type_](**params)