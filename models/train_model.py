"""
models/train_model.py
=====================
AireChile Analytics — Entrenamiento del modelo predictivo.

Entrena un RandomForestClassifier para predecir si la calidad del
aire del día siguiente será buena, regular o mala, usando las
condiciones meteorológicas y de contaminación del día actual.

¿Por qué RandomForestClassifier?
---------------------------------
1. Robusto ante outliers: los datos de MP2.5 tienen picos extremos
   (episodios de preemergencia). RF maneja esto mejor que modelos lineales
   porque cada árbol vota independientemente y los outliers solo afectan
   algunos árboles.

2. No requiere escalado: las features tienen unidades muy distintas
   (µg/m³, °C, %, km/h). RF no es sensible a la escala, eliminando
   la necesidad de StandardScaler o MinMaxScaler.

3. class_weight="balanced": Santiago tiene muchos más días con calidad
   "buena" en verano que días "mala". Sin balanceo el modelo aprendería
   a predecir siempre "buena" y tendría accuracy alta pero inútil.
   El parámetro balanced ajusta los pesos inversamente proporcional a
   la frecuencia de cada clase.

4. Feature importance nativa: RF entrega directamente la importancia
   de cada variable, lo que permite explicar el modelo al docente o
   cliente sin herramientas adicionales.

¿Por qué split temporal y no aleatorio?
-----------------------------------------
Los datos de calidad del aire son una serie temporal con correlación
entre días consecutivos (el MP2.5 de hoy predice el de mañana).
Un split aleatorio mezclaría fechas futuras en el entrenamiento,
lo que constituye "data leakage": el modelo vería el futuro durante
el entrenamiento y sus métricas en test serían irrealmente altas.

El split temporal correcto:
    - Entrenamiento: 80% de datos más ANTIGUOS (primeros 4 años)
    - Test:          20% de datos más RECIENTES (últimos ~10 meses)
Este esquema replica el escenario real: el modelo nunca ve el futuro.

Uso:
    python models/train_model.py

Salidas:
    models/model.pkl                      → modelo serializado
    models/metrics/model_metrics.json     → accuracy, f1, recall, precision
    models/metrics/feature_importance.csv → importancia de cada variable
    models/metrics/confusion_matrix.csv   → matriz de confusión

Variables de entorno (.env):
    MODEL_DATASET_PATH  → dataset_modelo_base.csv
    MODEL_OUTPUT_PATH   → models/model.pkl
    MODEL_METRICS_DIR   → models/metrics/
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rutas con valores por defecto (sobreescribibles desde .env)
DEFAULT_DATASET_PATH = "data/processed/dataset_modelo_base.csv"
DEFAULT_MODEL_PATH   = "models/model.pkl"
DEFAULT_METRICS_DIR  = "models/metrics"

# Features que usa el modelo — en este orden exacto
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

TARGET = "nivel_calidad_aire_dia_siguiente"

# Hiperparámetros del modelo
# n_estimators=200: suficientes árboles para estabilidad sin ser lento
# max_depth=10: limita la profundidad para evitar sobreajuste (árboles muy
#               profundos memorizan el training set en lugar de generalizar)
# min_samples_leaf=5: cada hoja necesita al menos 5 muestras (regularización)
# class_weight="balanced": ajusta pesos inversamente a frecuencia de clase
# random_state=42: reproducibilidad del experimento
RF_PARAMS = {
    "n_estimators":   200,
    "max_depth":      10,
    "min_samples_leaf": 5,
    "class_weight":   "balanced",
    "random_state":   42,
    "n_jobs":         -1,   # usar todos los cores disponibles
}

# Proporción de datos reservados para test (temporal: los más recientes)
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_dataset(ruta: str | Path) -> pd.DataFrame:
    """
    Carga el dataset base del modelo y valida su estructura.

    Args:
        ruta: Path al CSV del dataset

    Returns:
        pd.DataFrame con los datos

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si faltan columnas obligatorias o el dataset está vacío
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: '{ruta}'\n"
            "Ejecuta primero: python etl/etl_meteo_main.py"
        )

    df = pd.read_csv(ruta, parse_dates=["fecha"])

    if df.empty:
        raise ValueError(f"El dataset '{ruta.name}' está vacío.")

    # Verificar columnas obligatorias
    cols_requeridas = set(FEATURES) | {TARGET, "fecha"}
    faltantes = cols_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Columnas faltantes en el dataset: {sorted(faltantes)}\n"
            "Verifica que etl_meteo_main.py generó el dataset completo."
        )

    logger.info(
        f"Dataset cargado: {len(df):,} filas | "
        f"{df['fecha'].min().date()} → {df['fecha'].max().date()}"
    )
    return df


def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X e y eliminando filas con nulos en features o target.

    No imputa valores faltantes: para datos de contaminación, imputar
    puede introducir sesgo (un día sin datos meteo no es un día promedio).
    Se eliminan directamente las filas incompletas.

    Args:
        df: DataFrame completo

    Returns:
        Tupla (X, y) donde X son las features e y el target

    Raises:
        ValueError: Si quedan menos de 100 filas tras eliminar nulos
    """
    # Columnas relevantes: features + target
    df_modelo = df[FEATURES + [TARGET]].copy()

    n_antes = len(df_modelo)
    df_modelo = df_modelo.dropna().reset_index(drop=True)
    n_despues = len(df_modelo)
    n_eliminadas = n_antes - n_despues

    if n_eliminadas > 0:
        logger.info(
            f"Filas eliminadas por nulos: {n_eliminadas:,} "
            f"({n_eliminadas/n_antes*100:.1f}%)"
        )

    if n_despues < 100:
        raise ValueError(
            f"Solo quedan {n_despues} filas tras eliminar nulos. "
            "El dataset tiene muy pocos datos para entrenar un modelo confiable."
        )

    X = df_modelo[FEATURES]
    y = df_modelo[TARGET]

    logger.info(f"Filas para entrenamiento: {len(X):,}")
    logger.info(f"Distribución del target:\n{y.value_counts().to_string()}")

    return X, y


def split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple:
    """
    Divide X e y en train/test manteniendo el orden temporal.

    IMPORTANTE: No se usa train_test_split con shuffle=True porque los
    datos tienen correlación temporal (hoy predice mañana). Mezclar
    aleatoriamente introduciría data leakage: el modelo vería datos
    futuros durante el entrenamiento.

    El split correcto para series temporales:
        - Entrenamiento: primeros (1 - test_size)% de los datos
        - Test:          últimos test_size% de los datos

    Args:
        X:         DataFrame de features (debe estar ordenado por fecha)
        y:         Serie con el target
        test_size: Fracción para test (0.2 = 20% más reciente)

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    corte = int(n * (1 - test_size))

    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    logger.info(
        f"Split temporal — train: {len(X_train):,} filas "
        f"({len(X_train)/n*100:.0f}%) | "
        f"test: {len(X_test):,} filas ({len(X_test)/n*100:.0f}%)"
    )

    # Verificar que el test tiene las 3 clases para métricas completas
    clases_test = set(y_test.unique())
    clases_train = set(y_train.unique())
    if clases_test != clases_train:
        logger.warning(
            f"Las clases en test {clases_test} difieren de train {clases_train}. "
            "Las métricas por clase podrían ser incompletas."
        )

    return X_train, X_test, y_train, y_test


def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Entrena el RandomForestClassifier con los hiperparámetros definidos.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento

    Returns:
        Modelo entrenado
    """
    logger.info(
        f"Entrenando RandomForestClassifier con {RF_PARAMS['n_estimators']} "
        f"árboles, max_depth={RF_PARAMS['max_depth']}, "
        f"class_weight={RF_PARAMS['class_weight']}..."
    )

    modelo = RandomForestClassifier(**RF_PARAMS)
    modelo.fit(X_train, y_train)

    logger.info("Entrenamiento completado")
    return modelo


def evaluar_modelo(
    modelo: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evalúa el modelo en el set de test y retorna todas las métricas.

    Métricas calculadas:
        - accuracy:  proporción de predicciones correctas (general)
        - precision: cuando predice "mala", ¿cuántas veces acierta?
        - recall:    de todos los días malos, ¿cuántos detecta?
        - f1_score:  media armónica de precision y recall
        - classification_report: desglose por clase
        - confusion_matrix: tabla de verdaderos/falsos positivos por clase

    La métrica más importante para este proyecto es el recall de la
    clase "mala": es preferible emitir una falsa alerta (falso positivo)
    a no detectar un día de preemergencia real (falso negativo).

    Args:
        modelo:  Modelo entrenado
        X_test:  Features de test
        y_test:  Target real de test

    Returns:
        Dict con todas las métricas
    """
    y_pred = modelo.predict(X_test)
    clases = sorted(modelo.classes_)

    # Métricas generales (weighted average para multi-clase)
    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec       = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Métricas por clase (para análisis detallado)
    prec_por_clase = precision_score(
        y_test, y_pred, average=None, labels=clases, zero_division=0
    )
    rec_por_clase = recall_score(
        y_test, y_pred, average=None, labels=clases, zero_division=0
    )
    f1_por_clase = f1_score(
        y_test, y_pred, average=None, labels=clases, zero_division=0
    )

    # Classification report completo como string
    reporte = classification_report(
        y_test, y_pred, labels=clases, zero_division=0
    )

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=clases)

    metricas = {
        "accuracy":         round(float(acc), 4),
        "precision_weighted": round(float(prec), 4),
        "recall_weighted":    round(float(rec), 4),
        "f1_weighted":        round(float(f1), 4),
        "n_train":            int(len(X_test)),
        "n_test":             int(len(X_test)),
        "clases":             clases,
        "metricas_por_clase": {
            clase: {
                "precision": round(float(prec_por_clase[i]), 4),
                "recall":    round(float(rec_por_clase[i]), 4),
                "f1_score":  round(float(f1_por_clase[i]), 4),
            }
            for i, clase in enumerate(clases)
        },
        "classification_report": reporte,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": clases,
    }

    logger.info(f"Accuracy en test: {acc:.4f} ({acc*100:.1f}%)")
    logger.info(f"F1 weighted:      {f1:.4f}")
    logger.info(
        f"Recall clase 'mala': "
        f"{metricas['metricas_por_clase'].get('mala', {}).get('recall', 'N/A')}"
    )
    logger.info(f"\n{reporte}")

    return metricas


def calcular_feature_importance(
    modelo: RandomForestClassifier,
    features: list[str],
) -> pd.DataFrame:
    """
    Extrae la importancia de cada feature del modelo.

    La importancia en RF mide cuánto reduce cada feature la impureza
    de Gini en promedio a lo largo de todos los árboles. Una importancia
    alta indica que esa variable es muy discriminativa para clasificar.

    Args:
        modelo:   Modelo RandomForest entrenado
        features: Lista de nombres de features en el orden usado al entrenar

    Returns:
        DataFrame con columnas 'feature' e 'importancia', ordenado
        de mayor a menor importancia
    """
    importancias = pd.DataFrame({
        "feature":     features,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)

    importancias["importancia"] = importancias["importancia"].round(4)

    logger.info("Top 5 features más importantes:")
    for _, row in importancias.head(5).iterrows():
        bar = "█" * int(row["importancia"] * 40)
        logger.info(f"  {row['feature']:<25} {row['importancia']:.4f}  {bar}")

    return importancias


def guardar_artefactos(
    modelo: RandomForestClassifier,
    metricas: dict,
    importancias: pd.DataFrame,
    model_path: str | Path,
    metrics_dir: str | Path,
) -> None:
    """
    Guarda todos los artefactos del modelo en disco.

    Artefactos guardados:
        model.pkl              → modelo serializado con joblib
        model_metrics.json     → métricas en formato JSON
        feature_importance.csv → importancia de variables
        confusion_matrix.csv   → matriz de confusión

    Args:
        modelo:       Modelo entrenado
        metricas:     Dict de métricas de evaluar_modelo()
        importancias: DataFrame de feature importance
        model_path:   Ruta donde guardar el .pkl
        metrics_dir:  Directorio donde guardar las métricas
    """
    model_path  = Path(model_path)
    metrics_dir = Path(metrics_dir)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Modelo serializado
    joblib.dump(modelo, model_path)
    logger.info(f"Modelo guardado: {model_path}")

    # 2. Métricas en JSON
    # Eliminar el classification_report (string largo) del JSON principal
    metricas_json = {
        k: v for k, v in metricas.items()
        if k != "classification_report"
    }
    ruta_metrics = metrics_dir / "model_metrics.json"
    with open(ruta_metrics, "w", encoding="utf-8") as f:
        json.dump(metricas_json, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas guardadas: {ruta_metrics}")

    # 3. Feature importance
    ruta_fi = metrics_dir / "feature_importance.csv"
    importancias.to_csv(ruta_fi, index=False, encoding="utf-8")
    logger.info(f"Feature importance guardada: {ruta_fi}")

    # 4. Matriz de confusión
    clases = metricas["confusion_matrix_labels"]
    cm_df  = pd.DataFrame(
        metricas["confusion_matrix"],
        index=[f"real_{c}"      for c in clases],
        columns=[f"pred_{c}" for c in clases],
    )
    ruta_cm = metrics_dir / "confusion_matrix.csv"
    cm_df.to_csv(ruta_cm, encoding="utf-8")
    logger.info(f"Matriz de confusión guardada: {ruta_cm}")

    # 5. Classification report como texto
    ruta_cr = metrics_dir / "classification_report.txt"
    with open(ruta_cr, "w", encoding="utf-8") as f:
        f.write(metricas["classification_report"])
    logger.info(f"Classification report guardado: {ruta_cr}")


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def train_model(
    dataset_path: str | Path | None = None,
    model_path:   str | Path | None = None,
    metrics_dir:  str | Path | None = None,
) -> tuple[RandomForestClassifier, dict]:
    """
    Ejecuta el pipeline completo de entrenamiento del modelo.

    Args:
        dataset_path: Ruta al dataset base (None = usar .env/default)
        model_path:   Ruta donde guardar el modelo (None = usar .env/default)
        metrics_dir:  Directorio de métricas (None = usar .env/default)

    Returns:
        Tupla (modelo_entrenado, dict_metricas)

    Raises:
        FileNotFoundError: Si el dataset no existe
        ValueError: Si los datos son inválidos o insuficientes
    """
    ds_path = dataset_path or os.getenv("MODEL_DATASET_PATH", DEFAULT_DATASET_PATH)
    m_path  = model_path   or os.getenv("MODEL_OUTPUT_PATH",  DEFAULT_MODEL_PATH)
    m_dir   = metrics_dir  or os.getenv("MODEL_METRICS_DIR",  DEFAULT_METRICS_DIR)

    logger.info("=" * 60)
    logger.info("AireChile Analytics — Entrenamiento del modelo")
    logger.info("=" * 60)

    # Paso 1: Cargar datos
    df = cargar_dataset(ds_path)

    # Paso 2: Preparar X e y
    X, y = preparar_datos(df)

    # Paso 3: Split temporal
    X_train, X_test, y_train, y_test = split_temporal(X, y)

    # Paso 4: Entrenar
    modelo = entrenar_modelo(X_train, y_train)

    # Paso 5: Evaluar
    metricas = evaluar_modelo(modelo, X_test, y_test)

    # Paso 6: Feature importance
    importancias = calcular_feature_importance(modelo, FEATURES)

    # Paso 7: Guardar artefactos
    guardar_artefactos(modelo, metricas, importancias, m_path, m_dir)

    logger.info("=" * 60)
    logger.info("Entrenamiento completado exitosamente")
    logger.info("=" * 60)

    return modelo, metricas


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        modelo, metricas = train_model()

        print("\n" + "=" * 60)
        print("  MODELO ENTRENADO — RESUMEN")
        print("=" * 60)
        print(f"  Accuracy   : {metricas['accuracy']:.4f}  ({metricas['accuracy']*100:.1f}%)")
        print(f"  F1 weighted: {metricas['f1_weighted']:.4f}")
        print(f"  Precision  : {metricas['precision_weighted']:.4f}")
        print(f"  Recall     : {metricas['recall_weighted']:.4f}")
        print()
        print("  Métricas por clase:")
        for clase in ["buena", "regular", "mala"]:
            m = metricas["metricas_por_clase"].get(clase, {})
            print(
                f"    {clase:<10}  "
                f"precision={m.get('precision','N/A'):.4f}  "
                f"recall={m.get('recall','N/A'):.4f}  "
                f"f1={m.get('f1_score','N/A'):.4f}"
            )
        print()
        print("  Archivos generados:")
        print(f"    {os.getenv('MODEL_OUTPUT_PATH', DEFAULT_MODEL_PATH)}")
        print(f"    {os.getenv('MODEL_METRICS_DIR', DEFAULT_METRICS_DIR)}/model_metrics.json")
        print(f"    {os.getenv('MODEL_METRICS_DIR', DEFAULT_METRICS_DIR)}/feature_importance.csv")
        print(f"    {os.getenv('MODEL_METRICS_DIR', DEFAULT_METRICS_DIR)}/confusion_matrix.csv")
        print("=" * 60 + "\n")

    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)