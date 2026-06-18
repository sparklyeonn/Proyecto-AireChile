"""
etl/extract_meteo_forecast.py
=============================
AireChile Analytics — Extractor de meteorología futura (pronóstico).

Consulta el endpoint de pronóstico de Open-Meteo para obtener las
condiciones meteorológicas de los próximos N días (por defecto 7).

A diferencia de extract_meteo.py que usa el endpoint histórico
(archive-api), este módulo usa el endpoint de forecast estándar:
    https://api.open-meteo.com/v1/forecast

Las variables solicitadas son las mismas que usa el modelo:
    temperature_2m_max    → temperatura_max
    temperature_2m_min    → temperatura_min
    temperature_2m_mean   → temperatura_promedio
    wind_speed_10m_max    → velocidad_viento
    precipitation_sum     → precipitacion

NOTA sobre humedad relativa:
    La variable relative_humidity_2m_mean no siempre está disponible
    en el endpoint de forecast gratuito. Si no está disponible, se usa
    relative_humidity_2m_max como proxy o se imputa con el promedio
    histórico. El modelo es robusto ante este caso.

Uso:
    python etl/extract_meteo_forecast.py

Variables de entorno (.env):
    FORECAST_DAYS              → días a pronosticar (default: 7)
    METEO_FORECAST_RAW_PATH   → ruta de salida del CSV crudo
    METEO_LATITUDE             → latitud (-33.6117 para Puente Alto)
    METEO_LONGITUDE            → longitud (-70.5758 para Puente Alto)
"""

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
DEFAULT_LATITUDE      = -33.6117
DEFAULT_LONGITUDE     = -70.5758
DEFAULT_FORECAST_DAYS = 7
DEFAULT_TIMEZONE      = "America/Santiago"
DEFAULT_RAW_PATH      = "data/raw/open_meteo_forecast_7dias.csv"
REQUEST_TIMEOUT_S     = 30
MAX_RETRIES           = 3
RETRY_WAIT_S          = 5

# Variables diarias del pronóstico
# Se solicitan en este orden; si alguna no está disponible, se maneja abajo
DAILY_VARS_PRIMARIAS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
]

# Fallback si relative_humidity_2m_mean no está disponible
DAILY_VARS_FALLBACK = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_max",
    "wind_speed_10m_max",
    "precipitation_sum",
]

# Mapeo de nombres API → nombres del proyecto
MAPA_COLUMNAS = {
    "time":                        "fecha",
    "temperature_2m_max":          "temperatura_max",
    "temperature_2m_min":          "temperatura_min",
    "temperature_2m_mean":         "temperatura_promedio",
    "relative_humidity_2m_mean":   "humedad_relativa",
    "relative_humidity_2m_max":    "humedad_relativa",   # fallback
    "wind_speed_10m_max":          "velocidad_viento",
    "precipitation_sum":           "precipitacion",
}


def _leer_config() -> dict:
    return {
        "url":           os.getenv("OPENMETEO_FORECAST_URL", DEFAULT_FORECAST_URL),
        "latitude":      float(os.getenv("METEO_LATITUDE",   str(DEFAULT_LATITUDE))),
        "longitude":     float(os.getenv("METEO_LONGITUDE",  str(DEFAULT_LONGITUDE))),
        "forecast_days": int(os.getenv("FORECAST_DAYS",      str(DEFAULT_FORECAST_DAYS))),
        "timezone":      os.getenv("METEO_TIMEZONE",         DEFAULT_TIMEZONE),
        "raw_path":      os.getenv("METEO_FORECAST_RAW_PATH", DEFAULT_RAW_PATH),
    }


def _hacer_peticion(config: dict, vars_daily: list) -> dict:
    """
    Hace la petición GET a la API de forecast con reintentos.
    Intenta primero con las variables primarias; si falla por variable
    no disponible, reintenta con las de fallback.

    Returns:
        Dict con la respuesta JSON de la API
    """
    params = {
        "latitude":     config["latitude"],
        "longitude":    config["longitude"],
        "daily":        ",".join(vars_daily),
        "timezone":     config["timezone"],
        "forecast_days": config["forecast_days"],
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }

    for intento in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"Consultando Open-Meteo Forecast API "
                f"(intento {intento}/{MAX_RETRIES})..."
            )
            response = requests.get(
                config["url"],
                params=params,
                timeout=REQUEST_TIMEOUT_S,
            )

            if response.status_code == 400:
                detalle = response.json().get("reason", response.text)
                raise ValueError(f"Error 400 de Open-Meteo: {detalle}")

            response.raise_for_status()
            datos = response.json()
            logger.info(
                f"Respuesta recibida: {config['forecast_days']} días de pronóstico"
            )
            return datos

        except requests.exceptions.Timeout:
            if intento < MAX_RETRIES:
                logger.warning(f"Timeout. Reintentando en {RETRY_WAIT_S}s...")
                time.sleep(RETRY_WAIT_S)
            else:
                raise RuntimeError(
                    "Timeout al conectar con Open-Meteo. "
                    "Verifica tu conexión a internet."
                )
        except requests.exceptions.ConnectionError:
            if intento < MAX_RETRIES:
                logger.warning(f"Error de conexión. Reintentando en {RETRY_WAIT_S}s...")
                time.sleep(RETRY_WAIT_S)
            else:
                raise RuntimeError(
                    "No se pudo conectar a Open-Meteo. "
                    "Verifica tu conexión a internet."
                )


def _json_a_dataframe(datos: dict) -> pd.DataFrame:
    """
    Convierte la respuesta JSON en DataFrame y renombra columnas.
    """
    if "daily" not in datos:
        raise KeyError(
            f"La respuesta no contiene 'daily'. "
            f"Claves: {list(datos.keys())}"
        )

    bloque = datos["daily"]
    if not bloque.get("time"):
        raise ValueError("El bloque daily está vacío.")

    df = pd.DataFrame(bloque)

    # Renombrar columnas
    df = df.rename(columns={
        k: v for k, v in MAPA_COLUMNAS.items() if k in df.columns
    })

    # Convertir fecha
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")

    # Si no hay temperatura_promedio, calcularla como promedio de max y min
    if "temperatura_promedio" not in df.columns:
        if "temperatura_max" in df.columns and "temperatura_min" in df.columns:
            df["temperatura_promedio"] = (
                (df["temperatura_max"] + df["temperatura_min"]) / 2
            ).round(2)
            logger.info("temperatura_promedio calculada como (max+min)/2")

    # Si no hay humedad_relativa, imputar con valor histórico típico de Santiago
    if "humedad_relativa" not in df.columns:
        logger.warning(
            "humedad_relativa no disponible en la API. "
            "Se imputará con valor promedio histórico de Puente Alto (65%)."
        )
        df["humedad_relativa"] = 65.0

    # Asegurar columnas finales en orden consistente
    columnas_finales = [
        "fecha", "temperatura_max", "temperatura_min", "temperatura_promedio",
        "humedad_relativa", "velocidad_viento", "precipitacion",
    ]
    df = df[[c for c in columnas_finales if c in df.columns]]

    logger.info(
        f"DataFrame de pronóstico: {len(df)} filas × {df.shape[1]} columnas"
    )
    return df


def extract_meteo_forecast(
    forecast_days: int | None = None,
    raw_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Descarga el pronóstico meteorológico de los próximos N días.

    Args:
        forecast_days: Número de días a pronosticar (None = usar .env)
        raw_path:      Ruta donde guardar el CSV crudo (None = usar .env)

    Returns:
        pd.DataFrame con columnas meteorológicas para cada día futuro

    Raises:
        RuntimeError: Si falla la conexión después de los reintentos
        ValueError:   Si la respuesta de la API es inválida
    """
    config = _leer_config()
    if forecast_days:
        config["forecast_days"] = forecast_days
    if raw_path:
        config["raw_path"] = str(raw_path)

    logger.info(
        f"Extrayendo pronóstico meteorológico: "
        f"{config['forecast_days']} días | "
        f"lat={config['latitude']}, lon={config['longitude']}"
    )

    # Intentar primero con variables primarias, luego con fallback
    datos = None
    for vars_daily in [DAILY_VARS_PRIMARIAS, DAILY_VARS_FALLBACK]:
        try:
            datos = _hacer_peticion(config, vars_daily)
            break
        except ValueError as e:
            if "relative_humidity_2m_mean" in str(e):
                logger.warning(
                    "relative_humidity_2m_mean no disponible. "
                    "Usando relative_humidity_2m_max como fallback."
                )
                continue
            raise

    if datos is None:
        raise RuntimeError("No se pudo obtener datos de Open-Meteo.")

    df = _json_a_dataframe(datos)

    # Guardar CSV crudo
    ruta = Path(config["raw_path"])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"CSV de pronóstico guardado en: {ruta}")

    return df


if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("AireChile Analytics — Pronóstico Open-Meteo")
    logger.info("=" * 55)

    try:
        df = extract_meteo_forecast()
        print("\n" + "=" * 55)
        print("  PRONÓSTICO METEOROLÓGICO — 7 DÍAS")
        print("=" * 55)
        print(df.to_string(index=False))
        print("=" * 55 + "\n")
    except (RuntimeError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)