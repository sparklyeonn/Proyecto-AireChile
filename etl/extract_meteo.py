"""
etl/extract_meteo.py
====================
AireChile Analytics — Extractor de datos meteorológicos Open-Meteo.

Consulta el endpoint histórico de Open-Meteo (archive-api) para obtener
datos diarios de temperatura, humedad, viento y precipitación para
Santiago / Puente Alto.

Endpoint usado:
    https://archive-api.open-meteo.com/v1/archive

Variables diarias solicitadas:
    temperature_2m_max        → temperatura máxima del día (°C)
    temperature_2m_min        → temperatura mínima del día (°C)
    temperature_2m_mean       → temperatura promedio del día (°C)
    relative_humidity_2m_mean → humedad relativa promedio (%)
    wind_speed_10m_max        → velocidad máxima del viento (km/h)
    precipitation_sum         → precipitación acumulada diaria (mm)

Salida:
    CSV crudo en data/raw/open_meteo_puente_alto_2022_2026.csv
    con columnas exactamente como las devuelve la API.

Uso:
    python etl/extract_meteo.py

Variables de entorno (.env):
    OPENMETEO_BASE_URL      → URL base del endpoint histórico
    METEO_LATITUDE          → latitud (ej: -33.6117)
    METEO_LONGITUDE         → longitud (ej: -70.5758)
    METEO_START_DATE        → fecha inicio (ej: 2022-01-01)
    METEO_END_DATE          → fecha fin   (ej: 2026-06-13)
    METEO_RAW_PATH          → ruta de salida del CSV crudo
"""

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
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

# Valores por defecto (pueden sobreescribirse con .env)
DEFAULT_BASE_URL    = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_LATITUDE    = "-33.6117"
DEFAULT_LONGITUDE   = "-70.5758"
DEFAULT_START_DATE  = "2022-01-01"
DEFAULT_END_DATE    = "2026-06-13"
DEFAULT_RAW_PATH    = "data/raw/open_meteo_puente_alto_2022_2026.csv"
DEFAULT_TIMEZONE    = "America/Santiago"
REQUEST_TIMEOUT_S   = 60    # segundos antes de abortar la petición
MAX_RETRIES         = 3     # reintentos ante error de red
RETRY_WAIT_S        = 5     # segundos entre reintentos

# Variables diarias a solicitar a la API
# Nombres exactos según documentación Open-Meteo Historical API
DAILY_VARIABLES = [
    "temperature_2m_max",       # temperatura máxima
    "temperature_2m_min",       # temperatura mínima
    "temperature_2m_mean",      # temperatura promedio
    "relative_humidity_2m_mean",# humedad relativa media (variable adicional)
    "wind_speed_10m_max",       # velocidad viento máxima
    "precipitation_sum",        # precipitación acumulada
]


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def _leer_config() -> dict:
    """
    Lee la configuración desde variables de entorno con valores por defecto.

    Returns:
        dict con todas las variables de configuración
    """
    config = {
        "base_url":   os.getenv("OPENMETEO_BASE_URL",  DEFAULT_BASE_URL),
        "latitude":   float(os.getenv("METEO_LATITUDE",  DEFAULT_LATITUDE)),
        "longitude":  float(os.getenv("METEO_LONGITUDE", DEFAULT_LONGITUDE)),
        "start_date": os.getenv("METEO_START_DATE", DEFAULT_START_DATE),
        "end_date":   os.getenv("METEO_END_DATE",   DEFAULT_END_DATE),
        "raw_path":   os.getenv("METEO_RAW_PATH",   DEFAULT_RAW_PATH),
        "timezone":   os.getenv("METEO_TIMEZONE",   DEFAULT_TIMEZONE),
    }

    logger.info(
        f"Configuración: lat={config['latitude']}, lon={config['longitude']}, "
        f"{config['start_date']} → {config['end_date']}"
    )
    return config


def _construir_params(config: dict) -> dict:
    """
    Construye el dict de parámetros para la petición GET a Open-Meteo.

    Args:
        config: Dict de configuración de _leer_config()

    Returns:
        Dict de parámetros listos para requests.get(params=...)
    """
    return {
        "latitude":   config["latitude"],
        "longitude":  config["longitude"],
        "start_date": config["start_date"],
        "end_date":   config["end_date"],
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   config["timezone"],
        "wind_speed_unit": "kmh",         # km/h (consistente con SINCA)
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }


def _hacer_peticion(url: str, params: dict) -> dict:
    """
    Realiza la petición GET a Open-Meteo con reintentos ante errores de red.

    Args:
        url:    URL base del endpoint
        params: Parámetros de la petición

    Returns:
        Dict con la respuesta JSON de la API

    Raises:
        requests.exceptions.Timeout: Si la petición supera REQUEST_TIMEOUT_S
        requests.exceptions.HTTPError: Si la API devuelve un error HTTP
        RuntimeError: Si se agotan los reintentos
    """
    for intento in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"Consultando Open-Meteo (intento {intento}/{MAX_RETRIES})..."
            )
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_S,
            )

            # Verificar código HTTP
            if response.status_code == 400:
                # Error de parámetros — no tiene sentido reintentar
                try:
                    detalle = response.json().get("reason", response.text)
                except Exception:
                    detalle = response.text
                raise ValueError(
                    f"Error 400 de Open-Meteo: {detalle}\n"
                    "Revisa METEO_START_DATE, METEO_END_DATE, "
                    "METEO_LATITUDE y METEO_LONGITUDE en tu .env"
                )

            response.raise_for_status()

            datos = response.json()
            logger.info("Respuesta recibida correctamente de Open-Meteo")
            return datos

        except requests.exceptions.Timeout:
            logger.warning(
                f"Timeout en intento {intento}. "
                f"Esperando {RETRY_WAIT_S}s antes de reintentar..."
            )
            if intento < MAX_RETRIES:
                time.sleep(RETRY_WAIT_S)
            else:
                raise RuntimeError(
                    f"Se agotaron {MAX_RETRIES} intentos por timeout. "
                    "Verifica tu conexión a internet."
                )

        except requests.exceptions.ConnectionError:
            logger.warning(f"Error de conexión en intento {intento}.")
            if intento < MAX_RETRIES:
                time.sleep(RETRY_WAIT_S)
            else:
                raise RuntimeError(
                    "No se pudo conectar a Open-Meteo después de "
                    f"{MAX_RETRIES} intentos. Verifica tu conexión."
                )


def _json_a_dataframe(datos: dict) -> pd.DataFrame:
    """
    Convierte la respuesta JSON de Open-Meteo en un DataFrame.

    La API devuelve un objeto con clave "daily" que contiene:
        {
            "time":                   ["2022-01-01", "2022-01-02", ...],
            "temperature_2m_max":     [25.3, 24.1, ...],
            "temperature_2m_min":     [12.1, 11.8, ...],
            ...
        }

    Args:
        datos: Dict de la respuesta JSON de la API

    Returns:
        pd.DataFrame con columna "time" y una columna por variable

    Raises:
        KeyError: Si la respuesta no contiene la clave "daily"
        ValueError: Si el bloque "daily" está vacío
    """
    if "daily" not in datos:
        raise KeyError(
            "La respuesta de Open-Meteo no contiene la clave 'daily'. "
            f"Claves presentes: {list(datos.keys())}"
        )

    bloque_daily = datos["daily"]

    if not bloque_daily.get("time"):
        raise ValueError(
            "El bloque 'daily' de Open-Meteo llegó vacío. "
            "Verifica el rango de fechas en METEO_START_DATE / METEO_END_DATE."
        )

    df = pd.DataFrame(bloque_daily)

    # Verificar que están todas las variables solicitadas
    variables_ausentes = [v for v in DAILY_VARIABLES if v not in df.columns]
    if variables_ausentes:
        logger.warning(
            f"Variables no presentes en la respuesta: {variables_ausentes}. "
            "Pueden no estar disponibles para este modelo/período."
        )

    logger.info(
        f"DataFrame construido: {len(df):,} filas × {df.shape[1]} columnas"
    )
    return df


def _guardar_crudo(df: pd.DataFrame, ruta: str | Path) -> Path:
    """
    Guarda el DataFrame crudo como CSV en data/raw/.

    Args:
        df:   DataFrame con los datos crudos de la API
        ruta: Ruta de salida

    Returns:
        Path al archivo guardado
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"CSV crudo guardado en: {ruta} ({len(df):,} filas)")
    return ruta


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def extract_meteo(
    start_date: str | None = None,
    end_date:   str | None = None,
    latitude:   float | None = None,
    longitude:  float | None = None,
    raw_path:   str | Path | None = None,
) -> pd.DataFrame:
    """
    Descarga datos meteorológicos históricos diarios desde Open-Meteo
    y los devuelve como DataFrame.

    Los parámetros pueden pasarse explícitamente o leerse desde .env.
    Los parámetros explícitos tienen prioridad sobre .env.

    Args:
        start_date: Fecha inicio en formato YYYY-MM-DD
        end_date:   Fecha fin en formato YYYY-MM-DD
        latitude:   Latitud de la ubicación
        longitude:  Longitud de la ubicación
        raw_path:   Ruta donde guardar el CSV crudo (None = no guardar)

    Returns:
        pd.DataFrame con columnas crudas de la API:
            time                      (str) — "YYYY-MM-DD"
            temperature_2m_max        (float64)
            temperature_2m_min        (float64)
            temperature_2m_mean       (float64)
            relative_humidity_2m_mean (float64)
            wind_speed_10m_max        (float64)
            precipitation_sum         (float64)

    Raises:
        ValueError: Si los parámetros de la API son incorrectos
        RuntimeError: Si la conexión falla después de los reintentos
    """
    # Leer config base y sobreescribir con parámetros explícitos si se pasaron
    config = _leer_config()
    if start_date:  config["start_date"] = start_date
    if end_date:    config["end_date"]   = end_date
    if latitude:    config["latitude"]   = latitude
    if longitude:   config["longitude"]  = longitude
    if raw_path:    config["raw_path"]   = str(raw_path)

    logger.info("Iniciando extracción Open-Meteo Historical API")

    # Construir parámetros y hacer petición
    params  = _construir_params(config)
    datos   = _hacer_peticion(config["base_url"], params)

    # Convertir JSON → DataFrame
    df = _json_a_dataframe(datos)

    # Guardar CSV crudo si se especificó ruta
    if config.get("raw_path"):
        _guardar_crudo(df, config["raw_path"])

    return df


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AireChile Analytics — Extractor Open-Meteo")
    logger.info("=" * 60)

    try:
        df = extract_meteo()

        print("\n" + "=" * 55)
        print("  EXTRACCIÓN OPEN-METEO EXITOSA")
        print(f"  Registros  : {len(df):,}")
        print(f"  Fecha min  : {df['time'].min()}")
        print(f"  Fecha max  : {df['time'].max()}")
        print("=" * 55)
        print("\nColumnas y primeros valores:")
        print(df.head(3).to_string())
        print("\nNulos por columna:")
        print(df.isnull().sum())

    except (ValueError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)