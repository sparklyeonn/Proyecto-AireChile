"""
models/compare_classification_models.py
=======================================
AireChile Analytics — Comparación formal de modelos de clasificación.

Este script compara múltiples algoritmos para predecir si la calidad del aire
del día siguiente será buena, regular o mala.

Variable objetivo:
    nivel_calidad_aire_dia_siguiente

Modelos comparados:
    1. LogisticRegression
    2. DecisionTreeClassifier
    3. RandomForestClassifier
    4. GradientBoostingClassifier

El script usa split temporal:
    - Datos más antiguos para entrenamiento.
    - Datos más recientes para prueba.

También usa TimeSeriesSplit en el tuning para evitar que el modelo aprenda
con información futura durante la validación cruzada.

Archivos generados:
    models/metrics/classification_model_comparison.csv
    models/metrics/classification_model_comparison_summary.json
    models/metrics/classification_tuning_results.json

Uso:
    python models/compare_classification_models.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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

TARGET = "nivel_calidad_aire_dia_siguiente"
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
    Carga el dataset base del proyecto y separa variables predictoras y target.

    El orden por fecha es importante porque la calidad del aire tiene
    comportamiento temporal. El modelo debe entrenar con datos antiguos y
    evaluarse con datos más recientes, evitando fuga de información futura.

    Args:
        ruta: Ruta al archivo dataset_modelo_base.csv.

    Returns:
        X: DataFrame con variables predictoras.
        y: Serie con la categoría del día siguiente.

    Raises:
        FileNotFoundError: Si el dataset no existe.
        ValueError: Si falta la variable objetivo.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: '{ruta}'. "
            "Ejecuta primero: python etl/etl_meteo_main.py"
        )

    df = pd.read_csv(ruta)

    # Orden temporal del dataset.
    # Si existe la columna fecha, se convierte a datetime y se ordena.
    # Esto asegura que el split temporal use pasado para entrenar
    # y datos recientes para test.
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.sort_values("fecha").reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(
            f"Columna objetivo '{TARGET}' no encontrada en el dataset. "
            "Verifica que el pipeline ETL haya generado la variable objetivo."
        )

    # Solo se usan las variables que existen realmente en el dataset.
    # Esto evita que el script se rompa si una columna meteorológica no está disponible.
    features_disponibles = [feature for feature in FEATURES if feature in df.columns]
    features_faltantes = [feature for feature in FEATURES if feature not in df.columns]

    if features_faltantes:
        logger.warning(f"Features no disponibles e ignoradas: {features_faltantes}")

    df_modelo = df[features_disponibles + [TARGET]].dropna().reset_index(drop=True)

    X = df_modelo[features_disponibles]
    y = df_modelo[TARGET]

    logger.info(
        f"Dataset cargado: {len(X):,} filas | "
        f"features usadas: {len(features_disponibles)} | "
        f"clases: {sorted(y.unique())}"
    )

    return X, y


def split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset respetando el orden temporal.

    No usa shuffle porque en datos de calidad del aire hay dependencia temporal.
    El objetivo es simular un escenario real: entrenar con pasado y evaluar
    con registros más recientes.

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
    Define los modelos de clasificación a comparar.

    LogisticRegression usa StandardScaler porque depende de la escala
    de las variables. Los modelos basados en árboles no requieren escalamiento,
    ya que separan los datos mediante umbrales.
    """
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
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
    Entrena y evalúa cada modelo de clasificación.

    Las métricas ponderadas son útiles cuando las clases están desbalanceadas,
    por ejemplo cuando existen más días buenos o regulares que días malos.

    Args:
        modelos: Diccionario nombre_modelo -> estimador.
        X_train: Variables de entrenamiento.
        X_test: Variables de prueba.
        y_train: Target de entrenamiento.
        y_test: Target de prueba.

    Returns:
        DataFrame ordenado por f1_weighted descendente.
    """
    resultados = []

    for nombre, modelo in modelos.items():
        logger.info(f"Entrenando {nombre}...")

        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            resultados.append(
                {
                    "modelo": nombre,
                    "accuracy": round(float(accuracy), 4),
                    "precision_weighted": round(float(precision), 4),
                    "recall_weighted": round(float(recall), 4),
                    "f1_weighted": round(float(f1), 4),
                }
            )

            logger.info(
                f"{nombre:<35} "
                f"accuracy={accuracy:.4f} | "
                f"precision={precision:.4f} | "
                f"recall={recall:.4f} | "
                f"f1={f1:.4f}"
            )

        except Exception as error:
            logger.error(f"Error entrenando {nombre}: {error}")

    df_resultados = pd.DataFrame(resultados)

    if df_resultados.empty:
        raise ValueError("No se pudo entrenar ningún modelo de clasificación.")

    df_resultados = df_resultados.sort_values(
        "f1_weighted",
        ascending=False,
    ).reset_index(drop=True)

    return df_resultados


# ---------------------------------------------------------------------------
# Tuning RandomForest
# ---------------------------------------------------------------------------

def tuning_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Aplica GridSearchCV sobre RandomForestClassifier.

    Se usa TimeSeriesSplit para respetar la estructura temporal de los datos.
    Esto evita validar con datos del pasado después de haber entrenado con datos
    del futuro.

    Args:
        X_train: Variables de entrenamiento.
        y_train: Target de entrenamiento.

    Returns:
        Diccionario con mejores parámetros y score de validación cruzada.
    """
    logger.info("Iniciando tuning con TimeSeriesSplit para RandomForestClassifier...")

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    time_series_cv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=RF_PARAM_GRID,
        scoring="f1_weighted",
        cv=time_series_cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    resultado = {
        "mejores_parametros": grid_search.best_params_,
        "mejor_score_cv": round(float(grid_search.best_score_), 4),
        "metrica_cv": "f1_weighted",
        "cv_tipo": "TimeSeriesSplit",
        "n_splits_cv": 3,
        "justificacion_cv": (
            "Se usa TimeSeriesSplit porque los datos tienen orden temporal. "
            "Esto reduce el riesgo de data leakage durante el tuning."
        ),
    }

    logger.info(f"Mejores parámetros: {grid_search.best_params_}")
    logger.info(f"Mejor f1_weighted en CV: {grid_search.best_score_:.4f}")

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

    Los CSV sirven para revisar resultados en tabla.
    Los JSON sirven para documentación, dashboard o API.
    """
    if df_resultados.empty:
        raise ValueError("No hay resultados para guardar.")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "classification_model_comparison.csv"
    df_resultados.to_csv(csv_path, index=False, encoding="utf-8")

    mejor = df_resultados.iloc[0]
    n_modelos = len(df_resultados)

    justificaciones = {
        "RandomForestClassifier": (
            "RandomForest obtuvo el mejor f1_weighted entre los modelos comparados. "
            "Es robusto ante outliers de MP2.5, no requiere escalamiento y permite "
            "interpretar importancia de variables."
        ),
        "GradientBoostingClassifier": (
            "GradientBoosting obtuvo el mejor f1_weighted. Es capaz de capturar "
            "patrones complejos, aunque RandomForest sigue siendo una alternativa "
            "más simple de explicar académicamente."
        ),
        "DecisionTreeClassifier": (
            "DecisionTree obtuvo el mejor f1_weighted. Es altamente interpretable, "
            "aunque puede ser menos estable que un ensamble como RandomForest."
        ),
        "LogisticRegression": (
            "LogisticRegression obtuvo el mejor f1_weighted. Funciona como baseline "
            "lineal y permite una interpretación simple, aunque puede capturar menos "
            "relaciones no lineales."
        ),
    }

    summary = {
        "mejor_modelo": mejor["modelo"],
        "metrica_seleccion": "f1_weighted",
        "mejor_f1_weighted": float(mejor["f1_weighted"]),
        "mejor_accuracy": float(mejor["accuracy"]),
        "justificacion": justificaciones.get(
            mejor["modelo"],
            f"{mejor['modelo']} obtuvo el mayor f1_weighted.",
        ),
        "cantidad_modelos_comparados": n_modelos,
        "split_tipo": "temporal",
        "cv_tipo_tuning": "TimeSeriesSplit",
        "modelos_comparados": df_resultados["modelo"].tolist(),
        "interpretacion": (
            "f1_weighted se usa porque considera precision y recall ponderados "
            "por la cantidad de observaciones de cada clase."
        ),
    }

    summary_path = metrics_dir / "classification_model_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    tuning_path = metrics_dir / "classification_tuning_results.json"
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
    print("\n" + "=" * 76)
    print("  COMPARACIÓN DE MODELOS DE CLASIFICACIÓN — AireChile Analytics")
    print("=" * 76)
    print(
        f"  {'Modelo':<35} "
        f"{'Accuracy':>9} "
        f"{'Precision':>10} "
        f"{'Recall':>8} "
        f"{'F1':>8}"
    )
    print("  " + "-" * 72)

    for idx, row in df_resultados.iterrows():
        marca = " ← mejor" if idx == 0 else ""
        print(
            f"  {row['modelo']:<35} "
            f"{row['accuracy']:>9.4f} "
            f"{row['precision_weighted']:>10.4f} "
            f"{row['recall_weighted']:>8.4f} "
            f"{row['f1_weighted']:>8.4f}"
            f"{marca}"
        )

    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def compare_classification_models(
    dataset_path: Path | None = None,
    metrics_dir: Path | None = None,
    run_tuning: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta el flujo completo de comparación de modelos de clasificación.

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
    logger.info("AireChile Analytics — Comparación Modelos Clasificación")
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
        tuning = tuning_random_forest(X_train, y_train)

    guardar_resultados(df_resultados, tuning, carpeta_metricas)
    imprimir_tabla(df_resultados)

    return df_resultados


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        compare_classification_models()
        print(
            f"Archivos generados en {METRICS_DIR}:\n"
            "  classification_model_comparison.csv\n"
            "  classification_model_comparison_summary.json\n"
            "  classification_tuning_results.json\n"
        )
    except (FileNotFoundError, ValueError) as error:
        logger.error(str(error))
        sys.exit(1)