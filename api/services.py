"""
api/services.py
===============
AireChile Analytics — Lógica de negocio de la API REST.

Cada función de este módulo se encarga de leer un archivo procesado,
validar su existencia y estructura, y transformarlo en el schema
de respuesta correspondiente.

Separar la lógica de los endpoints (main.py) facilita el testing
y permite cambiar la fuente de datos (CSV → PostgreSQL) sin tocar
los endpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from api.schemas import (
    PrediccionActualResponse,
    DiaPronostico,
    Forecast7DiasResponse,
    MetricasResponse,
    MetricasPorClase,
    EstacionResponse,
)

# ---------------------------------------------------------------------------
# Rutas de archivos (relativas a la raíz del proyecto)
# Pueden sobreescribirse con variables de entorno para mayor flexibilidad
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent

RUTA_PREDICCION   = ROOT / os.getenv("PREDICTION_OUTPUT_PATH",         "data/processed/prediccion_actual.csv")
RUTA_FORECAST     = ROOT / os.getenv("PREDICTION_7_DAYS_OUTPUT_PATH",  "data/processed/prediccion_7_dias.csv")
RUTA_METRICS      = ROOT / os.getenv("MODEL_METRICS_DIR",              "models/metrics") / "model_metrics.json"
RUTA_FEAT_IMP     = ROOT / os.getenv("MODEL_METRICS_DIR",              "models/metrics") / "feature_importance.csv"
RUTA_DATASET      = ROOT / os.getenv("DATASET_MODELO_BASE_PATH",       "data/processed/dataset_modelo_base.csv")


# ---------------------------------------------------------------------------
# Helper: recomendaciones por nivel
# ---------------------------------------------------------------------------

RECOMENDACIONES: dict[str, str] = {
    "buena":   "Condiciones favorables. Se pueden realizar actividades normales al aire libre.",
    "regular": "Personas sensibles deben reducir actividad física intensa al aire libre. "
               "Se recomienda monitorear la evolución del aire.",
    "mala":    "Evitar actividad física al aire libre. Niños, adultos mayores y personas con "
               "enfermedades respiratorias deben reducir la exposición al máximo.",
}


def get_recomendacion(nivel: str) -> str:
    """
    Retorna la recomendación operativa según el nivel de calidad del aire.

    Args:
        nivel: 'buena', 'regular' o 'mala'

    Returns:
        Texto con la recomendación
    """
    return RECOMENDACIONES.get(
        nivel.lower() if nivel else "",
        "Sin recomendación disponible para este nivel."
    )


# ---------------------------------------------------------------------------
# Servicio: predicción actual
# ---------------------------------------------------------------------------

def get_prediccion_actual() -> PrediccionActualResponse:
    """
    Lee prediccion_actual.csv y retorna la predicción del día siguiente.

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo está vacío o mal formado
    """
    if not RUTA_PREDICCION.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_PREDICCION.relative_to(ROOT)}. "
            "Ejecute primero: python models/predict.py"
        )

    df = pd.read_csv(RUTA_PREDICCION)
    if df.empty:
        raise ValueError("prediccion_actual.csv está vacío.")

    row = df.iloc[0]

    def safe_float(col: str) -> float | None:
        try:
            return float(row[col]) if col in row and pd.notna(row[col]) else None
        except (TypeError, ValueError):
            return None

    nivel = str(row.get("nivel_predicho", "")).lower()

    return PrediccionActualResponse(
        fecha_base          = str(row.get("fecha_base", "")),
        fecha_prediccion    = str(row.get("fecha_predicha", "")),
        nivel_calidad_aire_predicho = nivel,
        probabilidad        = safe_float("probabilidad_predicho"),
        prob_buena          = safe_float("prob_buena"),
        prob_regular        = safe_float("prob_regular"),
        prob_mala           = safe_float("prob_mala"),
        mp25_base           = safe_float("mp25_base"),
        recomendacion       = get_recomendacion(nivel),
        fuente              = "modelo_clasificador",
    )


# ---------------------------------------------------------------------------
# Servicio: pronóstico 7 días
# ---------------------------------------------------------------------------

def get_forecast_7_dias() -> Forecast7DiasResponse:
    """
    Lee prediccion_7_dias.csv y retorna la lista de pronósticos.

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo está vacío
    """
    if not RUTA_FORECAST.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_FORECAST.relative_to(ROOT)}. "
            "Ejecute primero: python models/predict_7_days.py"
        )

    df = pd.read_csv(RUTA_FORECAST, parse_dates=["fecha"])
    if df.empty:
        raise ValueError("prediccion_7_dias.csv está vacío.")

    def safe_float(row, col: str) -> float | None:
        try:
            return float(row[col]) if col in row.index and pd.notna(row[col]) else None
        except (TypeError, ValueError):
            return None

    dias = []
    for _, row in df.iterrows():
        nivel = str(row.get("nivel_calidad_aire_predicho", "")).lower()
        dias.append(DiaPronostico(
            horizonte_dia               = int(row.get("horizonte_dia", 0)),
            fecha                       = str(row["fecha"].date()) if hasattr(row["fecha"], "date") else str(row["fecha"])[:10],
            mp25_estimado               = round(float(row.get("mp25_estimado", 0)), 2),
            nivel_calidad_aire_predicho = nivel,
            temperatura_max             = safe_float(row, "temperatura_max"),
            temperatura_min             = safe_float(row, "temperatura_min"),
            velocidad_viento            = safe_float(row, "velocidad_viento"),
            precipitacion               = safe_float(row, "precipitacion"),
            recomendacion               = get_recomendacion(nivel),
        ))

    # Metadatos generales del pronóstico
    estacion = str(df["estacion"].iloc[0]) if "estacion" in df.columns else "Puente Alto"
    fecha_gen = str(df["fecha_generacion"].iloc[0]) if "fecha_generacion" in df.columns else None

    return Forecast7DiasResponse(
        estacion         = estacion,
        fecha_generacion = fecha_gen,
        dias             = dias,
    )


# ---------------------------------------------------------------------------
# Servicio: métricas del modelo
# ---------------------------------------------------------------------------

def get_metricas() -> MetricasResponse:
    """
    Lee model_metrics.json y feature_importance.csv.

    Raises:
        FileNotFoundError: Si el archivo de métricas no existe
    """
    if not RUTA_METRICS.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_METRICS.relative_to(ROOT)}. "
            "Ejecute primero: python models/train_model.py"
        )

    with open(RUTA_METRICS, encoding="utf-8") as f:
        data = json.load(f)

    # Construir métricas por clase
    mpc_raw = data.get("metricas_por_clase", {})
    mpc = {
        clase: MetricasPorClase(
            precision = m.get("precision", 0),
            recall    = m.get("recall", 0),
            f1_score  = m.get("f1_score", 0),
        )
        for clase, m in mpc_raw.items()
    } if mpc_raw else None

    # Feature importance (top 5)
    top_features = None
    if RUTA_FEAT_IMP.exists():
        fi = pd.read_csv(RUTA_FEAT_IMP)
        top_features = fi.head(5).rename(columns={"importancia": "importancia"})[
            ["feature", "importancia"]
        ].to_dict(orient="records")

    return MetricasResponse(
        accuracy             = data.get("accuracy", 0),
        f1_weighted          = data.get("f1_weighted", 0),
        precision_weighted   = data.get("precision_weighted", 0),
        recall_weighted      = data.get("recall_weighted", 0),
        n_test               = data.get("n_test"),
        clases               = data.get("clases"),
        metricas_por_clase   = mpc,
        top_features         = top_features,
    )


# ---------------------------------------------------------------------------
# Servicio: estaciones
# ---------------------------------------------------------------------------

def get_estaciones() -> list[EstacionResponse]:
    """
    Retorna la lista de estaciones de monitoreo del proyecto.
    Actualmente fija en Puente Alto; extensible a más estaciones.
    """
    # Intentar leer el rango de fechas del dataset si existe
    periodo = None
    if RUTA_DATASET.exists():
        try:
            df = pd.read_csv(RUTA_DATASET, usecols=["fecha"], nrows=1)
            # Leer última fila también
            df_full = pd.read_csv(RUTA_DATASET, usecols=["fecha"])
            fecha_min = df_full["fecha"].min()[:10]
            fecha_max = df_full["fecha"].max()[:10]
            periodo = f"{fecha_min} / {fecha_max}"
        except Exception:
            pass

    return [
        EstacionResponse(
            estacion            = "Puente Alto",
            comuna              = "Puente Alto",
            latitud             = -33.6117,
            longitud            = -70.5758,
            fuente_calidad_aire = "SINCA",
            fuente_meteorologia = "Open-Meteo",
            periodo_datos       = periodo,
        )
    ]