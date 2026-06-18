"""
models/predict_7_days.py
========================
AireChile Analytics — Pronóstico de calidad del aire a 7 días.

Genera una estimación de MP2.5 y nivel de calidad del aire para
los próximos 7 días usando predicción recursiva:

    Iteración 1: features reales del último día disponible
                 + meteorología futura día 1
                 → predice mp25[día 1]

    Iteración 2: mp25[día 1] estimado como mp25_dia_anterior
                 promedio_7d actualizado
                 + meteorología futura día 2
                 → predice mp25[día 2]

    ... y así hasta el día 7.

Archivos requeridos:
    models/model_mp25_regressor.pkl      → modelo entrenado
    data/processed/dataset_modelo_base.csv → datos históricos
    data/raw/open_meteo_forecast_7dias.csv → pronóstico meteorológico

Salida:
    data/processed/prediccion_7_dias.csv

Columnas del output:
    fecha, estacion, comuna, mp25_estimado,
    nivel_calidad_aire_predicho, temperatura_max, temperatura_min,
    temperatura_promedio, humedad_relativa, velocidad_viento,
    precipitacion, horizonte_dia, fecha_generacion

Uso:
    python models/predict_7_days.py

Variables de entorno (.env):
    FORECAST_MODEL_OUTPUT_PATH  → model_mp25_regressor.pkl
    MODEL_DATASET_PATH          → dataset_modelo_base.csv
    METEO_FORECAST_RAW_PATH    → open_meteo_forecast_7dias.csv
    PREDICTION_7_DAYS_OUTPUT_PATH → prediccion_7_dias.csv
    FORECAST_DAYS               → número de días (default: 7)
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH      = "models/model_mp25_regressor.pkl"
DEFAULT_DATASET_PATH    = "data/processed/dataset_modelo_base.csv"
DEFAULT_FORECAST_PATH   = "data/raw/open_meteo_forecast_7dias.csv"
DEFAULT_OUTPUT_PATH     = "data/processed/prediccion_7_dias.csv"
DEFAULT_FORECAST_DAYS   = 7

# Mismas features que train_forecast_model.py — orden exacto
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

# Umbrales norma DS59
UMBRAL_BUENA   = 25.0
UMBRAL_REGULAR = 50.0

EMOJIS = {"buena": "🟢", "regular": "🟡", "mala": "🔴"}


def clasificar_mp25(valor: float) -> str:
    """Clasifica MP2.5 con umbrales DS59."""
    if pd.isna(valor) or valor < 0:
        return "sin_dato"
    if valor <= UMBRAL_BUENA:
        return "buena"
    if valor <= UMBRAL_REGULAR:
        return "regular"
    return "mala"


def cargar_modelo(ruta: str | Path):
    """Carga el regresor serializado."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: '{ruta}'\n"
            "Ejecuta primero: python models/train_forecast_model.py"
        )
    modelo = joblib.load(ruta)
    logger.info(f"Modelo cargado: {ruta.name}")
    return modelo


def cargar_estado_inicial(dataset_path: str | Path) -> dict:
    """
    Obtiene el estado del último día disponible del dataset histórico.

    Retorna un dict con los valores reales del último día que serán
    usados como punto de partida para la predicción recursiva.

    Returns:
        Dict con:
            fecha_base      → fecha del último registro real
            mp25_actual     → mp25 del último día
            mp25_anterior   → mp25 del día anterior al último
            mp25_7d_reales  → lista de los últimos 7 valores de mp25
                              (para calcular promedio móvil actualizado)
    """
    ruta = Path(dataset_path)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: '{ruta}'\n"
            "Ejecuta primero: python etl/etl_meteo_main.py"
        )

    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Tomar los últimos 8 días (necesitamos 7 para el promedio móvil
    # más el día actual como punto de partida)
    df_reciente = df[["fecha", "mp25", "estacion", "comuna"]].dropna().tail(8)

    if df_reciente.empty:
        raise ValueError("No hay datos suficientes en el dataset.")

    ultimo = df_reciente.iloc[-1]
    mp25_7d = df_reciente["mp25"].tolist()  # hasta 8 valores

    estado = {
        "fecha_base":    pd.Timestamp(ultimo["fecha"]),
        "mp25_actual":   float(ultimo["mp25"]),
        "mp25_anterior": float(df_reciente.iloc[-2]["mp25"]) if len(df_reciente) >= 2 else float(ultimo["mp25"]),
        "mp25_7d_reales": mp25_7d,
        "estacion":      str(ultimo["estacion"]),
        "comuna":        str(ultimo["comuna"]),
    }

    logger.info(
        f"Estado inicial: {estado['fecha_base'].date()} | "
        f"MP2.5={estado['mp25_actual']:.1f} | "
        f"estación={estado['estacion']}"
    )
    return estado


def cargar_forecast_meteo(ruta: str | Path) -> pd.DataFrame:
    """
    Carga el CSV de pronóstico meteorológico.
    Si no existe, lo descarga automáticamente.
    """
    ruta = Path(ruta)

    if not ruta.exists():
        logger.warning(
            f"CSV de pronóstico no encontrado: '{ruta}'. "
            "Descargando automáticamente..."
        )
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from etl.extract_meteo_forecast import extract_meteo_forecast
            df = extract_meteo_forecast(raw_path=ruta)
            return df
        except Exception as e:
            raise FileNotFoundError(
                f"No se pudo obtener el pronóstico meteorológico: {e}\n"
                "Ejecuta primero: python etl/extract_meteo_forecast.py"
            ) from e

    df = pd.read_csv(ruta, parse_dates=["fecha"])
    logger.info(f"Pronóstico meteorológico cargado: {len(df)} días")
    return df


def _calcular_promedio_7d(historial_mp25: list) -> float:
    """
    Calcula el promedio móvil de los últimos 7 valores de mp25.
    Si hay menos de 3 valores, retorna la media de los disponibles.
    """
    valores = [v for v in historial_mp25 if not np.isnan(v)]
    ultimos_7 = valores[-7:] if len(valores) >= 7 else valores
    if len(ultimos_7) < 3:
        return float(np.mean(ultimos_7)) if ultimos_7 else 0.0
    return float(np.mean(ultimos_7))


def predecir_7_dias(
    modelo,
    estado_inicial: dict,
    df_forecast_meteo: pd.DataFrame,
    forecast_days: int = DEFAULT_FORECAST_DAYS,
) -> pd.DataFrame:
    """
    Ejecuta la predicción recursiva de MP2.5 para los próximos N días.

    En cada iteración:
        1. Construye el vector de features con el estado actual
        2. Predice mp25 del día siguiente con el regresor
        3. Actualiza el estado: mp25_actual → mp25_dia_anterior,
           agrega el estimado al historial para recalcular promedio_7d
        4. Repite

    Args:
        modelo:            RandomForestRegressor entrenado
        estado_inicial:    Dict con estado del último día real
        df_forecast_meteo: DataFrame con pronóstico meteorológico
        forecast_days:     Número de días a pronosticar

    Returns:
        DataFrame con una fila por día pronosticado
    """
    fecha_base      = estado_inicial["fecha_base"]
    mp25_actual     = estado_inicial["mp25_actual"]
    mp25_anterior   = estado_inicial["mp25_anterior"]
    historial_mp25  = list(estado_inicial["mp25_7d_reales"])
    estacion        = estado_inicial["estacion"]
    comuna          = estado_inicial["comuna"]
    fecha_generacion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convertir df_forecast_meteo a dict indexado por fecha para acceso rápido
    meteo_por_fecha = {}
    for _, row in df_forecast_meteo.iterrows():
        meteo_por_fecha[pd.Timestamp(row["fecha"]).date()] = row

    filas = []

    for horizonte in range(1, forecast_days + 1):
        fecha_pred = fecha_base + timedelta(days=horizonte)
        fecha_date = fecha_pred.date()

        # Obtener meteorología del día pronosticado
        meteo = meteo_por_fecha.get(fecha_date)
        if meteo is None:
            # Si no hay dato meteorológico para este día, usar el último disponible
            logger.warning(
                f"Sin datos meteorológicos para {fecha_date}. "
                "Usando último día disponible."
            )
            if meteo_por_fecha:
                meteo = list(meteo_por_fecha.values())[-1]
            else:
                raise ValueError(
                    "El DataFrame de pronóstico meteorológico está vacío."
                )

        # Calcular promedio móvil 7d con el historial actual
        mp25_promedio_7d = _calcular_promedio_7d(historial_mp25)

        # Construir vector de features
        features_dict = {
            "mp25":               mp25_actual,
            "mp25_dia_anterior":  mp25_anterior,
            "mp25_promedio_7d":   mp25_promedio_7d,
            "mes":                fecha_pred.month,
            "dia_semana":         fecha_pred.dayofweek,
            "temperatura_max":    float(meteo.get("temperatura_max", 15.0)),
            "temperatura_min":    float(meteo.get("temperatura_min", 5.0)),
            "temperatura_promedio": float(meteo.get("temperatura_promedio", 10.0)),
            "humedad_relativa":   float(meteo.get("humedad_relativa", 65.0)),
            "velocidad_viento":   float(meteo.get("velocidad_viento", 10.0)),
            "precipitacion":      float(meteo.get("precipitacion", 0.0)),
        }

        X_pred = pd.DataFrame([features_dict])[FEATURES]

        # Predecir MP2.5
        mp25_estimado = float(modelo.predict(X_pred)[0])
        mp25_estimado = max(0.0, round(mp25_estimado, 2))  # no puede ser negativo

        nivel = clasificar_mp25(mp25_estimado)

        filas.append({
            "fecha":                    fecha_pred.strftime("%Y-%m-%d"),
            "estacion":                 estacion,
            "comuna":                   comuna,
            "mp25_estimado":            mp25_estimado,
            "nivel_calidad_aire_predicho": nivel,
            "temperatura_max":          features_dict["temperatura_max"],
            "temperatura_min":          features_dict["temperatura_min"],
            "temperatura_promedio":     features_dict["temperatura_promedio"],
            "humedad_relativa":         features_dict["humedad_relativa"],
            "velocidad_viento":         features_dict["velocidad_viento"],
            "precipitacion":            features_dict["precipitacion"],
            "horizonte_dia":            horizonte,
            "fecha_generacion":         fecha_generacion,
        })

        logger.info(
            f"  Día {horizonte} ({fecha_date}): "
            f"MP2.5={mp25_estimado:.1f} µg/m³ → "
            f"{EMOJIS.get(nivel, '?')} {nivel.upper()}"
        )

        # Actualizar estado para la próxima iteración
        mp25_anterior  = mp25_actual
        mp25_actual    = mp25_estimado
        historial_mp25.append(mp25_estimado)

    return pd.DataFrame(filas)


def guardar_prediccion(df: pd.DataFrame, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"Predicción 7 días guardada en: {ruta}")
    return ruta


def predict_7_days(
    model_path:      str | Path | None = None,
    dataset_path:    str | Path | None = None,
    forecast_path:   str | Path | None = None,
    output_path:     str | Path | None = None,
    forecast_days:   int | None = None,
) -> pd.DataFrame:
    """
    Orquesta el pronóstico completo de 7 días.

    Returns:
        DataFrame con el pronóstico

    Raises:
        FileNotFoundError: Si falta el modelo o el dataset
        ValueError: Si los datos son inválidos
    """
    m_path = model_path    or os.getenv("FORECAST_MODEL_OUTPUT_PATH", DEFAULT_MODEL_PATH)
    d_path = dataset_path  or os.getenv("MODEL_DATASET_PATH",          DEFAULT_DATASET_PATH)
    f_path = forecast_path or os.getenv("METEO_FORECAST_RAW_PATH",     DEFAULT_FORECAST_PATH)
    o_path = output_path   or os.getenv("PREDICTION_7_DAYS_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)
    n_dias = forecast_days or int(os.getenv("FORECAST_DAYS",           str(DEFAULT_FORECAST_DAYS)))

    logger.info("=" * 55)
    logger.info(f"AireChile Analytics — Pronóstico {n_dias} días")
    logger.info("=" * 55)

    modelo         = cargar_modelo(m_path)
    estado_inicial = cargar_estado_inicial(d_path)
    df_meteo       = cargar_forecast_meteo(f_path)

    logger.info(f"Generando predicción recursiva para {n_dias} días...")
    df_pred = predecir_7_dias(modelo, estado_inicial, df_meteo, n_dias)

    guardar_prediccion(df_pred, o_path)
    return df_pred


if __name__ == "__main__":
    try:
        df = predict_7_days()

        print("\n" + "=" * 60)
        print("  PRONÓSTICO AIRECHILE — 7 DÍAS")
        print("=" * 60)
        for _, row in df.iterrows():
            emoji = EMOJIS.get(row["nivel_calidad_aire_predicho"], "?")
            print(
                f"  Día {int(row['horizonte_dia'])} "
                f"({row['fecha']})  "
                f"MP2.5={row['mp25_estimado']:>6.1f} µg/m³  "
                f"{emoji} {row['nivel_calidad_aire_predicho'].upper()}"
            )
        print("=" * 60 + "\n")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)