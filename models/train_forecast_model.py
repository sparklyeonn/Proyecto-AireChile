"""
models/train_forecast_model.py
==============================
AireChile Analytics — Entrenamiento del modelo de pronóstico MP2.5.

Entrena un RandomForestRegressor para estimar el valor numérico de
MP2.5 del día siguiente (no solo la categoría).

¿Por qué un regresor para el pronóstico de 7 días?
----------------------------------------------------
El clasificador existente (train_model.py) predice directamente la
clase buena/regular/mala, lo cual es perfecto para la predicción del
día siguiente. Pero para el horizonte de 7 días necesitamos hacer
predicción recursiva:

    Día 1: predecir mp25[t+1] usando mp25[t] real
    Día 2: predecir mp25[t+2] usando mp25[t+1] ESTIMADO
    Día 3: predecir mp25[t+3] usando mp25[t+2] ESTIMADO
    ...

Para esto necesitamos un número (no una categoría) que pueda usarse
como "mp25_dia_anterior" en la siguiente iteración.

El regresor predice el valor numérico de MP2.5 → luego se clasifica
con los mismos umbrales DS59 de siempre.

Features usadas (las mismas que el clasificador):
    mp25, mp25_dia_anterior, mp25_promedio_7d,
    mes, dia_semana,
    temperatura_max, temperatura_min, temperatura_promedio,
    humedad_relativa, velocidad_viento, precipitacion

Variable objetivo:
    mp25 del día siguiente (shift -1 del dataset)

Uso:
    python models/train_forecast_model.py

Salidas:
    models/model_mp25_regressor.pkl
    models/metrics/forecast_metrics.json
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH  = "data/processed/dataset_modelo_base.csv"
DEFAULT_MODEL_PATH    = "models/model_mp25_regressor.pkl"
DEFAULT_METRICS_DIR   = "models/metrics"

# Features — idénticas al clasificador para coherencia
FEATURES = [
    "mp25",
    "mp25_dia_anterior",
    "mp25_promedio_7d",
    "mes",
    "dia_semana",
    "temperatura_max",
    "temperatura_min",
    "temperatura_promedio",
    "humedad_relativa",
    "velocidad_viento",
    "precipitacion",
]

# El target es el MP2.5 del día siguiente (valor numérico)
# Se construye con shift(-1) en el dataset
TARGET_REGRESION = "mp25_dia_siguiente"

RF_PARAMS = {
    "n_estimators":     200,
    "max_depth":        10,
    "min_samples_leaf": 5,
    "random_state":     42,
    "n_jobs":           -1,
}

TEST_SIZE = 0.20


def cargar_y_preparar(ruta: str | Path) -> pd.DataFrame:
    """
    Carga el dataset y construye la variable objetivo mp25_dia_siguiente.

    El shift(-1) toma el mp25 de la fila siguiente como target de la
    fila actual. La última fila queda con NaN y se elimina.

    Args:
        ruta: Path al dataset_modelo_base.csv

    Returns:
        DataFrame con la columna mp25_dia_siguiente agregada

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si faltan columnas o quedan muy pocos datos
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: '{ruta}'\n"
            "Ejecuta primero: python etl/etl_meteo_main.py"
        )

    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Verificar features
    faltantes = [f for f in FEATURES if f not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")

    # Construir target numérico: mp25 del día siguiente
    df[TARGET_REGRESION] = df["mp25"].shift(-1)

    # Eliminar filas con nulos en features O en target
    df_modelo = df[FEATURES + [TARGET_REGRESION]].dropna().reset_index(drop=True)

    if len(df_modelo) < 100:
        raise ValueError(
            f"Solo quedan {len(df_modelo)} filas tras limpiar nulos. "
            "Se necesitan al menos 100 para entrenar."
        )

    logger.info(
        f"Dataset preparado: {len(df_modelo):,} filas | "
        f"mp25_siguiente — min={df_modelo[TARGET_REGRESION].min():.1f}, "
        f"max={df_modelo[TARGET_REGRESION].max():.1f}, "
        f"media={df_modelo[TARGET_REGRESION].mean():.1f}"
    )
    return df_modelo


def split_temporal(df: pd.DataFrame) -> tuple:
    """
    Split temporal 80/20 sin shuffle.
    Ver explicación en train_model.py — misma razón: evitar data leakage.
    """
    n = len(df)
    corte = int(n * (1 - TEST_SIZE))
    X = df[FEATURES]
    y = df[TARGET_REGRESION]
    return X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]


def entrenar(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    logger.info(
        f"Entrenando RandomForestRegressor — "
        f"{RF_PARAMS['n_estimators']} árboles, "
        f"max_depth={RF_PARAMS['max_depth']}..."
    )
    modelo = RandomForestRegressor(**RF_PARAMS)
    modelo.fit(X_train, y_train)
    logger.info("Entrenamiento completado")
    return modelo


def evaluar(
    modelo: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evalúa el regresor con métricas estándar para series de contaminantes.

    MAE  → error medio absoluto en µg/m³ (más interpretable)
    RMSE → error cuadrático medio (penaliza picos)
    R²   → varianza explicada (1.0 = perfecto, 0 = no mejor que la media)
    """
    y_pred = modelo.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    # Accuracy aproximada: % de predicciones dentro de ±10 µg/m³
    within_10 = float(np.mean(np.abs(y_pred - y_test) <= 10) * 100)

    metricas = {
        "mae":             round(mae, 3),
        "rmse":            round(rmse, 3),
        "r2":              round(r2, 4),
        "within_10_ugm3":  round(within_10, 2),
        "n_test":          int(len(y_test)),
        "features":        FEATURES,
    }

    logger.info(f"MAE:  {mae:.2f} µg/m³")
    logger.info(f"RMSE: {rmse:.2f} µg/m³")
    logger.info(f"R²:   {r2:.4f}")
    logger.info(f"Predicciones dentro de ±10 µg/m³: {within_10:.1f}%")

    return metricas


def guardar_artefactos(
    modelo: RandomForestRegressor,
    metricas: dict,
    model_path: str | Path,
    metrics_dir: str | Path,
) -> None:
    model_path  = Path(model_path)
    metrics_dir = Path(metrics_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, model_path)
    logger.info(f"Modelo guardado: {model_path}")

    ruta_m = metrics_dir / "forecast_metrics.json"
    with open(ruta_m, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)
    logger.info(f"Métricas guardadas: {ruta_m}")

    # Feature importance
    fi = pd.DataFrame({
        "feature":     FEATURES,
        "importancia": modelo.feature_importances_.round(4),
    }).sort_values("importancia", ascending=False)
    fi.to_csv(metrics_dir / "forecast_feature_importance.csv", index=False)

    logger.info("Top 5 features (regresor):")
    for _, row in fi.head(5).iterrows():
        bar = "█" * int(row["importancia"] * 40)
        logger.info(f"  {row['feature']:<25} {row['importancia']:.4f}  {bar}")


def train_forecast_model(
    dataset_path: str | Path | None = None,
    model_path:   str | Path | None = None,
    metrics_dir:  str | Path | None = None,
) -> tuple[RandomForestRegressor, dict]:
    """
    Pipeline completo de entrenamiento del modelo de pronóstico.

    Returns:
        (modelo_entrenado, dict_metricas)
    """
    ds   = dataset_path or os.getenv("MODEL_DATASET_PATH",      DEFAULT_DATASET_PATH)
    mp   = model_path   or os.getenv("FORECAST_MODEL_OUTPUT_PATH", DEFAULT_MODEL_PATH)
    md   = metrics_dir  or os.getenv("FORECAST_METRICS_DIR",    DEFAULT_METRICS_DIR)

    logger.info("=" * 55)
    logger.info("AireChile Analytics — Entrenamiento Regresor MP2.5")
    logger.info("=" * 55)

    df       = cargar_y_preparar(ds)
    X_train, X_test, y_train, y_test = split_temporal(df)
    modelo   = entrenar(X_train, y_train)
    metricas = evaluar(modelo, X_test, y_test)
    guardar_artefactos(modelo, metricas, mp, md)

    return modelo, metricas


if __name__ == "__main__":
    try:
        modelo, metricas = train_forecast_model()
        print("\n" + "=" * 55)
        print("  MODELO DE PRONÓSTICO ENTRENADO")
        print("=" * 55)
        print(f"  MAE   : {metricas['mae']:.2f} µg/m³")
        print(f"  RMSE  : {metricas['rmse']:.2f} µg/m³")
        print(f"  R²    : {metricas['r2']:.4f}")
        print(f"  ±10   : {metricas['within_10_ugm3']:.1f}% de predicciones")
        print(f"\n  Guardado en: {os.getenv('FORECAST_MODEL_OUTPUT_PATH', DEFAULT_MODEL_PATH)}")
        print("=" * 55 + "\n")
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)