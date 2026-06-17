"""
etl/transform_meteo.py
======================
AireChile Analytics — Transformación de datos meteorológicos Open-Meteo.

Responsabilidad: leer el CSV crudo de Open-Meteo y producir un DataFrame
limpio con columnas renombradas, tipos correctos y sin duplicados.

Transformaciones aplicadas:
    1. Renombrar columnas a nombres del proyecto (snake_case en español)
    2. Convertir columna de fecha a datetime64
    3. Validar columnas obligatorias
    4. Eliminar duplicados por fecha
    5. Ordenar cronológicamente
    6. Reportar nulos por columna

Mapeo de nombres de columnas:
    time                      → fecha
    temperature_2m_max        → temperatura_max
    temperature_2m_min        → temperatura_min
    temperature_2m_mean       → temperatura_promedio
    relative_humidity_2m_mean → humedad_relativa
    wind_speed_10m_max        → velocidad_viento
    precipitation_sum         → precipitacion

Uso:
    from etl.transform_meteo import transform_meteo

    df = transform_meteo("data/raw/open_meteo_puente_alto_2022_2026.csv")

Variables de entorno (.env):
    METEO_RAW_PATH       → ruta del CSV crudo
    METEO_PROCESSED_PATH → ruta de salida del CSV procesado
"""

import logging
import os
import sys
from pathlib import Path

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

DEFAULT_RAW_PATH       = "data/raw/open_meteo_puente_alto_2022_2026.csv"
DEFAULT_PROCESSED_PATH = "data/processed/open_meteo_transformado.csv"

# Mapeo: nombre API → nombre proyecto
MAPA_COLUMNAS = {
    "time":                       "fecha",
    "temperature_2m_max":         "temperatura_max",
    "temperature_2m_min":         "temperatura_min",
    "temperature_2m_mean":        "temperatura_promedio",
    "relative_humidity_2m_mean":  "humedad_relativa",
    "wind_speed_10m_max":         "velocidad_viento",
    "precipitation_sum":          "precipitacion",
}

# Columnas que deben existir después del renombrado (para validación)
COLUMNAS_OBLIGATORIAS = [
    "fecha",
    "temperatura_max",
    "temperatura_min",
    "temperatura_promedio",
    "humedad_relativa",
    "velocidad_viento",
    "precipitacion",
]


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def _cargar_csv_meteo(ruta: Path) -> pd.DataFrame:
    """
    Carga el CSV crudo de Open-Meteo.

    Args:
        ruta: Path al CSV crudo

    Returns:
        pd.DataFrame crudo

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el CSV está vacío
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo crudo no encontrado: '{ruta}'\n"
            "Ejecuta primero: python etl/extract_meteo.py"
        )

    df = pd.read_csv(ruta, encoding="utf-8")

    if df.empty:
        raise ValueError(f"El archivo '{ruta.name}' está vacío.")

    logger.info(f"CSV cargado: {len(df):,} filas × {df.shape[1]} columnas")
    return df


def _renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra las columnas de la API a los nombres del proyecto.
    Solo renombra las columnas que existen — no falla si falta alguna.

    Args:
        df: DataFrame con nombres de columna de Open-Meteo

    Returns:
        df con columnas renombradas
    """
    mapa_aplicable = {k: v for k, v in MAPA_COLUMNAS.items() if k in df.columns}
    df = df.rename(columns=mapa_aplicable)

    cols_no_mapeadas = [c for c in df.columns if c not in MAPA_COLUMNAS.values()]
    if cols_no_mapeadas:
        logger.info(f"Columnas adicionales no mapeadas (se conservan): {cols_no_mapeadas}")

    return df


def _validar_columnas(df: pd.DataFrame) -> None:
    """
    Verifica que el DataFrame tenga todas las columnas obligatorias
    después del renombrado.

    Args:
        df: DataFrame con columnas renombradas

    Raises:
        ValueError: Si falta alguna columna obligatoria
    """
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Columnas faltantes tras el renombrado: {faltantes}\n"
            "Verifica que Open-Meteo devolvió todas las variables "
            "en DAILY_VARIABLES de extract_meteo.py"
        )
    logger.info("✓ Todas las columnas obligatorias presentes")


def _convertir_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna 'fecha' de string a datetime64.
    Open-Meteo devuelve fechas en formato ISO "YYYY-MM-DD".

    Args:
        df: DataFrame con columna 'fecha' como string

    Returns:
        df con 'fecha' como datetime64[ns]
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d", errors="coerce")

    n_nat = df["fecha"].isna().sum()
    if n_nat > 0:
        logger.warning(f"{n_nat:,} fechas no parseables. Se eliminarán.")
        df = df[df["fecha"].notna()].reset_index(drop=True)

    return df


def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados por fecha y ordena cronológicamente.

    Args:
        df: DataFrame con fecha ya como datetime64

    Returns:
        df limpio y ordenado
    """
    df = df.copy()
    n_antes = len(df)

    df = (
        df.drop_duplicates(subset=["fecha"], keep="first")
          .sort_values("fecha")
          .reset_index(drop=True)
    )

    n_dup = n_antes - len(df)
    if n_dup > 0:
        logger.info(f"Duplicados eliminados: {n_dup:,}")

    return df


def _reportar_nulos(df: pd.DataFrame) -> None:
    """
    Registra en el log el porcentaje de nulos por columna meteorológica.
    Los nulos en temperatura, viento, etc. son normales para algunos
    días en el modelo ERA5.

    Args:
        df: DataFrame transformado
    """
    logger.info("Nulos por columna meteorológica:")
    for col in COLUMNAS_OBLIGATORIAS[1:]:  # excluir 'fecha'
        if col in df.columns:
            n = int(df[col].isna().sum())
            pct = n / len(df) * 100 if len(df) > 0 else 0
            nivel = "⚠️ " if pct > 5 else "  "
            logger.info(f"  {nivel}{col:<25} {n:>4} nulos ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def transform_meteo(
    ruta_crudo: str | Path | None = None,
    ruta_salida: str | Path | None = None,
) -> pd.DataFrame:
    """
    Transforma el CSV crudo de Open-Meteo en un DataFrame limpio y
    estandarizado, listo para hacer merge con el dataset SINCA.

    Args:
        ruta_crudo:  Ruta al CSV crudo de Open-Meteo.
                     Si es None, usa METEO_RAW_PATH del .env.
        ruta_salida: Ruta donde guardar el CSV procesado.
                     Si es None, usa METEO_PROCESSED_PATH del .env.
                     Si es False, no guarda.

    Returns:
        pd.DataFrame con columnas:
            fecha              (datetime64[ns])
            temperatura_max    (float64) — °C
            temperatura_min    (float64) — °C
            temperatura_promedio (float64) — °C
            humedad_relativa   (float64) — %
            velocidad_viento   (float64) — km/h
            precipitacion      (float64) — mm

    Raises:
        FileNotFoundError: Si el CSV crudo no existe
        ValueError: Si faltan columnas o el DataFrame queda vacío
    """
    # Resolver rutas
    ruta_csv = Path(
        ruta_crudo or os.getenv("METEO_RAW_PATH", DEFAULT_RAW_PATH)
    )
    ruta_out = ruta_salida or os.getenv(
        "METEO_PROCESSED_PATH", DEFAULT_PROCESSED_PATH
    )

    logger.info("Iniciando transformación datos meteorológicos")

    # Pipeline de transformación
    df = (
        _cargar_csv_meteo(ruta_csv)
        .pipe(_renombrar_columnas)
    )

    _validar_columnas(df)

    df = (
        df
        .pipe(_convertir_fecha)
        .pipe(_limpiar)
    )

    if df.empty:
        raise ValueError(
            "El DataFrame meteorológico quedó vacío tras la limpieza."
        )

    _reportar_nulos(df)

    # Seleccionar solo columnas del proyecto (descartar columnas extra de la API)
    cols_finales = [c for c in COLUMNAS_OBLIGATORIAS if c in df.columns]
    df = df[cols_finales].reset_index(drop=True)

    # Guardar CSV procesado
    if ruta_out is not False:
        ruta_out = Path(ruta_out)
        ruta_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta_out, index=False, encoding="utf-8")
        logger.info(
            f"CSV procesado guardado en: {ruta_out} ({len(df):,} filas)"
        )

    logger.info(
        f"Transformación completada — {len(df):,} registros | "
        f"{df['fecha'].min().date()} → {df['fecha'].max().date()}"
    )
    return df


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AireChile Analytics — Transformación Open-Meteo")
    logger.info("=" * 60)

    try:
        df = transform_meteo()

        print("\n" + "=" * 55)
        print("  TRANSFORMACIÓN METEOROLÓGICA EXITOSA")
        print(f"  Registros  : {len(df):,}")
        print(f"  Fecha min  : {df['fecha'].min().date()}")
        print(f"  Fecha max  : {df['fecha'].max().date()}")
        print("=" * 55)
        print("\nColumnas:")
        print(df.dtypes)
        print("\nPrimeras 3 filas:")
        print(df.head(3).to_string())
        print("\nNulos:")
        print(df.isnull().sum())

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)