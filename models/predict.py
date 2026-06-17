"""
models/predict.py
=================
AireChile Analytics — Generación de predicciones con el modelo entrenado.

Carga el modelo serializado (model.pkl) y genera la predicción de calidad
del aire para el día siguiente a partir del último registro disponible
en el dataset.

Lógica de predicción:
    1. Cargar model.pkl entrenado
    2. Leer el dataset y ordenar por fecha
    3. Tomar la última fila con todas las features disponibles
    4. Construir el vector de features X_pred
    5. Predecir la clase y las probabilidades por clase
    6. Calcular la fecha predicha (fecha_base + 1 día)
    7. Guardar resultado en prediccion_actual.csv

Formato de salida (prediccion_actual.csv):
    fecha_base            → fecha del registro usado como input
    fecha_predicha        → fecha_base + 1 día
    nivel_predicho        → buena | regular | mala
    probabilidad_predicho → probabilidad de la clase predicha (0-1)
    prob_buena            → probabilidad clase "buena"
    prob_regular          → probabilidad clase "regular"
    prob_mala             → probabilidad clase "mala"
    mp25_base             → valor de mp25 usado como input

Uso:
    python models/predict.py

Variables de entorno (.env):
    MODEL_OUTPUT_PATH      → models/model.pkl
    MODEL_DATASET_PATH     → dataset_modelo_base.csv
    PREDICTION_OUTPUT_PATH → prediccion_actual.csv
"""

import logging
import os
import sys
from pathlib import Path
from datetime import timedelta

import joblib
import pandas as pd
from dotenv import load_dotenv

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

DEFAULT_MODEL_PATH      = "models/model.pkl"
DEFAULT_DATASET_PATH    = "data/processed/dataset_modelo_base.csv"
DEFAULT_PREDICTION_PATH = "data/processed/prediccion_actual.csv"

# Features en el MISMO ORDEN que se usó en entrenamiento
# (si el orden cambia, las predicciones serán incorrectas)
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

# Emojis para visualización en consola
EMOJIS = {"buena": "🟢", "regular": "🟡", "mala": "🔴"}


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_modelo(ruta: str | Path):
    """
    Carga el modelo serializado desde disk.

    Args:
        ruta: Path al archivo model.pkl

    Returns:
        Modelo RandomForestClassifier cargado

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no es un modelo válido
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: '{ruta}'\n"
            "Ejecuta primero: python models/train_model.py"
        )

    try:
        modelo = joblib.load(ruta)
        logger.info(f"Modelo cargado desde: {ruta}")

        # Verificar que tiene el método predict_proba (RF siempre lo tiene)
        if not hasattr(modelo, "predict_proba"):
            raise ValueError(
                f"El archivo '{ruta.name}' no es un clasificador válido."
            )

        clases = list(modelo.classes_)
        logger.info(f"Clases del modelo: {clases}")
        return modelo

    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"Error al cargar el modelo: {e}") from e


def obtener_ultimo_registro(
    dataset_path: str | Path,
) -> tuple[pd.Series, pd.Timestamp]:
    """
    Lee el dataset y devuelve el último registro con todas las features
    disponibles (sin nulos en ninguna feature del modelo).

    Se busca el último registro válido y no simplemente la última fila,
    porque la última fila del dataset podría tener NaN en mp25_promedio_7d
    u otras features calculadas.

    Args:
        dataset_path: Ruta al CSV del dataset base

    Returns:
        Tupla (serie_con_features, fecha_base)

    Raises:
        FileNotFoundError: Si el dataset no existe
        ValueError: Si no hay ningún registro con todas las features
    """
    ruta = Path(dataset_path)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: '{ruta}'\n"
            "Ejecuta primero: python etl/etl_meteo_main.py"
        )

    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    logger.info(
        f"Dataset cargado: {len(df):,} filas | "
        f"último día: {df['fecha'].max().date()}"
    )

    # Verificar que existen las columnas de features
    faltantes = [f for f in FEATURES if f not in df.columns]
    if faltantes:
        raise ValueError(
            f"Columnas faltantes en el dataset: {faltantes}\n"
            "Verifica que el dataset fue generado por etl_meteo_main.py"
        )

    # Buscar el último registro sin nulos en ninguna feature
    df_completo = df[FEATURES + ["fecha"]].dropna()

    if df_completo.empty:
        raise ValueError(
            "No hay ningún registro con todas las features disponibles. "
            "Verifica la calidad del dataset."
        )

    ultimo = df_completo.iloc[-1]
    fecha_base = pd.Timestamp(ultimo["fecha"])

    logger.info(
        f"Último registro válido para predicción: {fecha_base.date()} | "
        f"mp25={ultimo['mp25']:.2f} µg/m³"
    )
    return ultimo, fecha_base


def generar_prediccion(
    modelo,
    registro: pd.Series,
    fecha_base: pd.Timestamp,
) -> dict:
    """
    Genera la predicción para el día siguiente usando el último registro.

    Args:
        modelo:     Modelo entrenado
        registro:   Serie con los valores de features del día actual
        fecha_base: Fecha del registro usado como input

    Returns:
        Dict con la predicción completa:
            fecha_base            → str fecha del input
            fecha_predicha        → str fecha predicha (base + 1 día)
            nivel_predicho        → str clase predicha
            probabilidad_predicho → float probabilidad de la clase predicha
            prob_buena            → float
            prob_regular          → float
            prob_mala             → float
            mp25_base             → float valor mp25 del día base
    """
    # Construir vector de features en el orden correcto
    X_pred = pd.DataFrame([registro[FEATURES]])

    # Predicción de clase
    clase_predicha = modelo.predict(X_pred)[0]

    # Probabilidades por clase
    probs = modelo.predict_proba(X_pred)[0]
    clases = list(modelo.classes_)
    prob_dict = {f"prob_{c}": round(float(p), 4) for c, p in zip(clases, probs)}

    # Probabilidad de la clase predicha
    idx_predicha = clases.index(clase_predicha)
    prob_predicha = round(float(probs[idx_predicha]), 4)

    fecha_predicha = fecha_base + timedelta(days=1)

    resultado = {
        "fecha_base":            str(fecha_base.date()),
        "fecha_predicha":        str(fecha_predicha.date()),
        "nivel_predicho":        clase_predicha,
        "probabilidad_predicho": prob_predicha,
        **{f"prob_{c}": prob_dict.get(f"prob_{c}", 0.0) for c in ["buena", "regular", "mala"]},
        "mp25_base":             round(float(registro["mp25"]), 2),
    }

    logger.info(
        f"Predicción: {fecha_predicha.date()} → "
        f"{clase_predicha.upper()} "
        f"(probabilidad: {prob_predicha:.1%})"
    )

    return resultado


def guardar_prediccion(
    resultado: dict,
    ruta: str | Path,
) -> Path:
    """
    Guarda el resultado de la predicción como CSV.

    Args:
        resultado: Dict con la predicción
        ruta:      Ruta de salida

    Returns:
        Path al archivo guardado
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df_pred = pd.DataFrame([resultado])
    df_pred.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"Predicción guardada en: {ruta}")
    return ruta


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def predict(
    model_path:      str | Path | None = None,
    dataset_path:    str | Path | None = None,
    prediction_path: str | Path | None = None,
) -> dict:
    """
    Genera y guarda la predicción de calidad del aire para el día siguiente.

    Args:
        model_path:      Ruta al model.pkl (None = usar .env/default)
        dataset_path:    Ruta al dataset (None = usar .env/default)
        prediction_path: Ruta de salida (None = usar .env/default)

    Returns:
        Dict con la predicción completa

    Raises:
        FileNotFoundError: Si el modelo o dataset no existen
        ValueError: Si los datos son inválidos
    """
    m_path = model_path      or os.getenv("MODEL_OUTPUT_PATH",      DEFAULT_MODEL_PATH)
    d_path = dataset_path    or os.getenv("MODEL_DATASET_PATH",     DEFAULT_DATASET_PATH)
    p_path = prediction_path or os.getenv("PREDICTION_OUTPUT_PATH", DEFAULT_PREDICTION_PATH)

    logger.info("=" * 60)
    logger.info("AireChile Analytics — Generando predicción")
    logger.info("=" * 60)

    # 1. Cargar modelo
    modelo = cargar_modelo(m_path)

    # 2. Obtener último registro
    registro, fecha_base = obtener_ultimo_registro(d_path)

    # 3. Generar predicción
    resultado = generar_prediccion(modelo, registro, fecha_base)

    # 4. Guardar
    guardar_prediccion(resultado, p_path)

    return resultado


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        resultado = predict()

        emoji  = EMOJIS.get(resultado["nivel_predicho"], "❓")
        nivel  = resultado["nivel_predicho"].upper()
        prob   = resultado["probabilidad_predicho"]

        print("\n" + "=" * 60)
        print("  PREDICCIÓN AIRECHILE ANALYTICS")
        print("=" * 60)
        print(f"  Fecha base (hoy)     : {resultado['fecha_base']}")
        print(f"  Fecha predicha       : {resultado['fecha_predicha']}")
        print(f"  MP2.5 hoy            : {resultado['mp25_base']} µg/m³")
        print()
        print(f"  Calidad del aire mañana:")
        print(f"  {emoji}  {nivel}  (confianza: {prob:.1%})")
        print()
        print("  Probabilidades por clase:")
        print(f"    🟢 Buena   : {resultado.get('prob_buena', 0):.1%}")
        print(f"    🟡 Regular : {resultado.get('prob_regular', 0):.1%}")
        print(f"    🔴 Mala    : {resultado.get('prob_mala', 0):.1%}")
        print()
        print(f"  Guardado en: {os.getenv('PREDICTION_OUTPUT_PATH', DEFAULT_PREDICTION_PATH)}")
        print("=" * 60 + "\n")

    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)