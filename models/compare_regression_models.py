"""
models/compare_regression_models.py
===================================
AireChile Analytics — Comparación formal de modelos de regresión.

Este script compara múltiples algoritmos para estimar el valor numérico
de MP2.5 del día siguiente.

Variable objetivo:
    mp25_dia_siguiente

Si la variable objetivo no existe, se construye usando:
    df["mp25"].shift(-1)

Modelos comparados:
    1. LinearRegression
    2. DecisionTreeRegressor
    3. RandomForestRegressor
    4. GradientBoostingRegressor

El script usa split temporal:
    - Datos más antiguos para entrenamiento.
    - Datos más recientes para prueba.

También usa TimeSeriesSplit en el tuning para mantener la coherencia temporal
durante la validación cruzada.

Archivos generados:
    models/metrics/regression_model_comparison.csv
    models/metrics/regression_model_comparison_summary.json
    models/metrics/regression_tuning_results.json

Uso:
    python models/compare_regression_models.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATASET = ROOT / os.getenv(
    "DATASET_MODELO_BASE_PATH",
    "data/processed/dataset_modelo_base.csv",
)
METRICS_DIR = ROOT / os.getenv("MODEL_METRICS_DIR", "models/metrics")

TARGET_REG = "mp25_dia_siguiente"
RANDOM_STATE = 42
TEST_SIZE = 0.20

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

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}


# ---------------------------------------------------------------------------
# Carga y preparación del dataset
# ---------------------------------------------------------------------------

def cargar_dataset(ruta: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga el dataset base y prepara el target de regresión.

    Si el dataset no contiene mp25_dia_siguiente, el script la genera con
    shift(-1), usando el MP2.5 del día siguiente como objetivo.

    El orden por fecha se aplica antes de construir el target para que el
    día siguiente sea realmente el registro temporal posterior.

    Args:
        ruta: Ruta al archivo dataset_modelo_base.csv.

    Returns:
        X: DataFrame con variables predictoras.
        y: Serie con MP2.5 del día siguiente.

    Raises:
        FileNotFoundError: Si el dataset no existe.
        ValueError: Si falta la columna mp25.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: '{ruta}'. "
            "Ejecuta primero: python etl/etl_meteo_main.py"
        )

    df = pd.read_csv(ruta)

    # Orden temporal del dataset.
    # En regresión es especialmente importante porque mp25_dia_siguiente
    # puede construirse con shift(-1). Si el dataset estuviera desordenado,
    # el target quedaría incorrecto.
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.sort_values("fecha").reset_index(drop=True)

    if "mp25" not in df.columns:
        raise ValueError("Columna 'mp25' no encontrada en el dataset.")

    if TARGET_REG not in df.columns:
        logger.info(
            f"Columna '{TARGET_REG}' no encontrada. "
            "Se construirá usando shift(-1) sobre mp25."
        )
        df[TARGET_REG] = df["mp25"].shift(-1)
        df = df.iloc[:-1].copy()

    # Solo se usan variables disponibles.
    # Esto mantiene el script flexible si el dataset cambia o si falta alguna
    # variable meteorológica.
    features_disponibles = [feature for feature in FEATURES if feature in df.columns]
    features_faltantes = [feature for feature in FEATURES if feature not in df.columns]

    if features_faltantes:
        logger.warning(f"Features no disponibles e ignoradas: {features_faltantes}")

    df_modelo = df[features_disponibles + [TARGET_REG]].dropna().reset_index(drop=True)

    X = df_modelo[features_disponibles]
    y = df_modelo[TARGET_REG]

    logger.info(
        f"Dataset cargado: {len(X):,} filas | "
        f"features usadas: {len(features_disponibles)} | "
        f"rango target MP2.5: [{y.min():.2f}, {y.max():.2f}] µg/m³"
    )

    return X, y


def split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos respetando el orden temporal.

    Esto simula un escenario real: entrenar con datos históricos y probar
    con datos más recientes.

    Args:
        X: Variables predictoras.
        y: Variable objetivo.
        test_size: Proporción final reservada para test.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    n = len(X)
    corte = int(n * (1 - test_size))

    return (
        X.iloc[:corte],
        X.iloc[corte:],
        y.iloc[:corte],
        y.iloc[corte:],
    )


# ---------------------------------------------------------------------------
# Definición de modelos
# ---------------------------------------------------------------------------

def definir_modelos() -> dict:
    """
    Define los modelos de regresión a comparar.

    LinearRegression usa StandardScaler porque depende de la escala.
    Los modelos basados en árboles no requieren escalamiento.
    """
    return {
        "LinearRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        ),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Comparación de modelos
# ---------------------------------------------------------------------------

def comparar_modelos(
    modelos: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Entrena y evalúa cada modelo de regresión.

    Métricas:
        MAE: error absoluto promedio, fácil de interpretar.
        RMSE: penaliza más los errores grandes.
        R2: proporción de varianza explicada por el modelo.

    Args:
        modelos: Diccionario nombre_modelo -> estimador.
        X_train: Variables de entrenamiento.
        X_test: Variables de prueba.
        y_train: Target de entrenamiento.
        y_test: Target de prueba.

    Returns:
        DataFrame ordenado por RMSE ascendente.
    """
    resultados = []

    for nombre, modelo in modelos.items():
        logger.info(f"Entrenando {nombre}...")

        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            # MP2.5 no puede ser negativo. Si un modelo lineal predice negativo,
            # se corrige a cero para mantener sentido físico.
            y_pred = np.maximum(y_pred, 0)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            resultados.append(
                {
                    "modelo": nombre,
                    "mae": round(float(mae), 3),
                    "rmse": round(float(rmse), 3),
                    "r2": round(float(r2), 4),
                }
            )

            logger.info(
                f"{nombre:<35} "
                f"MAE={mae:.3f} | "
                f"RMSE={rmse:.3f} | "
                f"R2={r2:.4f}"
            )

        except Exception as error:
            logger.error(f"Error entrenando {nombre}: {error}")

    df_resultados = pd.DataFrame(resultados)

    if df_resultados.empty:
        raise ValueError("No se pudo entrenar ningún modelo de regresión.")

    df_resultados = df_resultados.sort_values(
        "rmse",
        ascending=True,
    ).reset_index(drop=True)

    return df_resultados


# ---------------------------------------------------------------------------
# Tuning RandomForest
# ---------------------------------------------------------------------------

def tuning_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Aplica GridSearchCV sobre RandomForestRegressor.

    Se usa TimeSeriesSplit para mantener la lógica temporal del problema.
    La métrica de tuning es RMSE mediante neg_root_mean_squared_error, porque
    GridSearchCV maximiza los scores y por eso Scikit-learn usa el negativo.

    Args:
        X_train: Variables de entrenamiento.
        y_train: Target de entrenamiento.

    Returns:
        Diccionario con mejores parámetros y RMSE de validación cruzada.
    """
    logger.info("Iniciando tuning con TimeSeriesSplit para RandomForestRegressor...")

    base_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    time_series_cv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=RF_PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=time_series_cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    rmse_cv = abs(float(grid_search.best_score_))

    resultado = {
        "mejores_parametros": grid_search.best_params_,
        "mejor_score_cv": round(rmse_cv, 3),
        "metrica_cv": "rmse (neg_root_mean_squared_error)",
        "cv_tipo": "TimeSeriesSplit",
        "n_splits_cv": 3,
        "nota": (
            "El RMSE se interpreta en µg/m³. Se usa TimeSeriesSplit para "
            "evitar fuga de información temporal durante el tuning."
        ),
    }

    logger.info(f"Mejores parámetros: {grid_search.best_params_}")
    logger.info(f"Mejor RMSE en CV: {rmse_cv:.3f} µg/m³")

    return resultado


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------

def guardar_resultados(
    df_resultados: pd.DataFrame,
    tuning: dict,
    metrics_dir: Path,
) -> None:
    """
    Guarda los resultados de comparación, resumen y tuning.

    El CSV permite comparar modelos en tabla.
    Los JSON guardan la interpretación y son útiles para documentación,
    dashboard o API.
    """
    if df_resultados.empty:
        raise ValueError("No hay resultados para guardar.")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "regression_model_comparison.csv"
    df_resultados.to_csv(csv_path, index=False, encoding="utf-8")

    mejor = df_resultados.iloc[0]

    justificaciones = {
        "RandomForestRegressor": (
            "RandomForestRegressor obtuvo el menor RMSE. Es robusto ante "
            "outliers de MP2.5, captura relaciones no lineales y funciona bien "
            "con variables de distintas escalas."
        ),
        "GradientBoostingRegressor": (
            "GradientBoostingRegressor obtuvo el menor RMSE. Puede capturar "
            "patrones complejos, aunque RandomForest suele ser más simple de "
            "explicar e interpretar en un contexto académico."
        ),
        "DecisionTreeRegressor": (
            "DecisionTreeRegressor obtuvo el menor RMSE. Es interpretable, "
            "pero puede ser menos estable que un ensamble de árboles."
        ),
        "LinearRegression": (
            "LinearRegression obtuvo el menor RMSE. Esto indica una relación "
            "aproximadamente lineal en este conjunto, aunque puede no capturar "
            "interacciones no lineales asociadas a meteorología y contaminación."
        ),
    }

    summary = {
        "mejor_modelo": mejor["modelo"],
        "metrica_seleccion": "rmse",
        "mejor_rmse": float(mejor["rmse"]),
        "mejor_mae": float(mejor["mae"]),
        "mejor_r2": float(mejor["r2"]),
        "justificacion": justificaciones.get(
            mejor["modelo"],
            f"{mejor['modelo']} obtuvo el menor RMSE.",
        ),
        "cantidad_modelos_comparados": len(df_resultados),
        "split_tipo": "temporal",
        "cv_tipo_tuning": "TimeSeriesSplit",
        "modelos_comparados": df_resultados["modelo"].tolist(),
        "interpretacion_rmse": (
            "El RMSE indica el error promedio penalizando más los errores grandes. "
            "En este proyecto se interpreta directamente en µg/m³ de MP2.5."
        ),
    }

    summary_path = metrics_dir / "regression_model_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    tuning_path = metrics_dir / "regression_tuning_results.json"
    with open(tuning_path, "w", encoding="utf-8") as file:
        json.dump(tuning, file, indent=2, ensure_ascii=False)

    logger.info(f"Comparación guardada en: {csv_path}")
    logger.info(f"Resumen guardado en: {summary_path}")
    logger.info(f"Tuning guardado en: {tuning_path}")


# ---------------------------------------------------------------------------
# Salida por consola
# ---------------------------------------------------------------------------

def imprimir_tabla(df_resultados: pd.DataFrame) -> None:
    """
    Imprime una tabla simple para revisar los resultados en consola.
    """
    print("\n" + "=" * 72)
    print("  COMPARACIÓN DE MODELOS DE REGRESIÓN — AireChile Analytics")
    print("=" * 72)
    print(
        f"  {'Modelo':<35} "
        f"{'MAE (µg/m³)':>12} "
        f"{'RMSE (µg/m³)':>13} "
        f"{'R²':>8}"
    )
    print("  " + "-" * 68)

    for idx, row in df_resultados.iterrows():
        marca = " ← mejor RMSE" if idx == 0 else ""
        print(
            f"  {row['modelo']:<35} "
            f"{row['mae']:>12.3f} "
            f"{row['rmse']:>13.3f} "
            f"{row['r2']:>8.4f}"
            f"{marca}"
        )

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def compare_regression_models(
    dataset_path: Path | None = None,
    metrics_dir: Path | None = None,
    run_tuning: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta el flujo completo de comparación de modelos de regresión.

    Args:
        dataset_path: Ruta alternativa al dataset.
        metrics_dir: Carpeta alternativa para métricas.
        run_tuning: Activa o desactiva el tuning con GridSearchCV.

    Returns:
        DataFrame con métricas de los modelos comparados.
    """
    ruta_dataset = dataset_path or DATASET
    carpeta_metricas = metrics_dir or METRICS_DIR

    logger.info("=" * 60)
    logger.info("AireChile Analytics — Comparación Modelos Regresión")
    logger.info("=" * 60)

    X, y = cargar_dataset(ruta_dataset)

    X_train, X_test, y_train, y_test = split_temporal(X, y)

    logger.info(
        f"Split temporal: train={len(X_train):,} | test={len(X_test):,}"
    )

    modelos = definir_modelos()
    df_resultados = comparar_modelos(
        modelos,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    tuning = {}
    if run_tuning:
        tuning = tuning_random_forest_regressor(X_train, y_train)

    guardar_resultados(df_resultados, tuning, carpeta_metricas)
    imprimir_tabla(df_resultados)

    return df_resultados


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        compare_regression_models()
        print(
            f"Archivos generados en {METRICS_DIR}:\n"
            "  regression_model_comparison.csv\n"
            "  regression_model_comparison_summary.json\n"
            "  regression_tuning_results.json\n"
        )
    except (FileNotFoundError, ValueError) as error:
        logger.error(str(error))
        sys.exit(1)