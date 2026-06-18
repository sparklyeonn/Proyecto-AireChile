"""
etl/load_postgres.py
====================
AireChile Analytics — Carga de datos procesados a PostgreSQL.

Lee los CSV generados por el pipeline ETL y los carga en las
tablas correspondientes de PostgreSQL.

Estrategia de carga (if_exists="replace"):
    Se usa "replace" en lugar de "append" por simplicidad y porque
    en esta etapa del proyecto no hay producción activa. Cada vez
    que se ejecuta el pipeline, se reemplaza la tabla completa con
    los datos actualizados. Esta estrategia es segura para un proyecto
    académico y fácil de defender: los datos fuente son los CSV
    procesados, y la base de datos es un espejo reproducible de ellos.

    Para producción real se reemplazaría por upsert (INSERT ... ON CONFLICT).

Mapeo CSV → tabla:
    sinca_transformado.csv      → mediciones_sinca
    open_meteo_transformado.csv → meteorologia
    dataset_modelo_base.csv     → dataset_modelo
    prediccion_actual.csv       → predicciones_modelo

Uso:
    python etl/load_postgres.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# Mapeo: nombre descriptivo → (ruta CSV, nombre tabla PostgreSQL)
CARGAS = {
    "SINCA": (
        ROOT / os.getenv("SINCA_PROCESSED_PATH", "data/processed/sinca_transformado.csv"),
        "mediciones_sinca",
    ),
    "Open-Meteo": (
        ROOT / os.getenv("METEO_PROCESSED_PATH", "data/processed/open_meteo_transformado.csv"),
        "meteorologia",
    ),
    "Dataset modelo": (
        ROOT / os.getenv("DATASET_MODELO_BASE_PATH", "data/processed/dataset_modelo_base.csv"),
        "dataset_modelo",
    ),
    "Predicción": (
        ROOT / os.getenv("PREDICTION_OUTPUT_PATH", "data/processed/prediccion_actual.csv"),
        "predicciones_modelo",
    ),
}

# Columnas de fecha que deben convertirse a datetime antes de cargar
COLS_FECHA = {
    "mediciones_sinca":    ["fecha"],
    "meteorologia":        ["fecha"],
    "dataset_modelo":      ["fecha"],
    "predicciones_modelo": ["fecha_base", "fecha_predicha"],
}


def _registrar_log_etl(
    engine,
    proceso: str,
    estado: str,
    registros: int = 0,
    mensaje: str = "",
) -> None:
    """
    Inserta un registro en la tabla log_etl.
    Si falla (ej: tabla no existe aún), solo muestra advertencia.

    Args:
        engine:    SQLAlchemy Engine
        proceso:   Nombre del proceso (ej: "carga_sinca")
        estado:    'OK', 'ERROR' o 'ADVERTENCIA'
        registros: Cantidad de filas procesadas
        mensaje:   Mensaje descriptivo o detalle del error
    """
    try:
        df_log = pd.DataFrame([{
            "proceso":              proceso,
            "fecha_ejecucion":      datetime.now(),
            "estado":               estado,
            "registros_procesados": registros,
            "mensaje":              mensaje[:500] if mensaje else "",
        }])
        df_log.to_sql(
            "log_etl", engine,
            if_exists="append",
            index=False,
            method="multi",
        )
    except Exception as e:
        logger.warning(f"No se pudo registrar en log_etl: {e}")


def _preparar_dataframe(df: pd.DataFrame, tabla: str) -> pd.DataFrame:
    """
    Prepara el DataFrame para la carga: convierte fechas, limpia
    nombres de columnas y elimina columnas que no existen en el schema.

    Args:
        df:    DataFrame leído del CSV
        tabla: Nombre de la tabla destino

    Returns:
        DataFrame listo para to_sql()
    """
    df = df.copy()

    # Limpiar nombres de columnas (quitar espacios)
    df.columns = [c.strip() for c in df.columns]

    # Convertir columnas de fecha a datetime
    for col in COLS_FECHA.get(tabla, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Para predicciones_modelo: quitar columna 'id' si viene del CSV
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


def cargar_csv_a_tabla(
    engine,
    nombre: str,
    ruta_csv: Path,
    tabla: str,
) -> int:
    """
    Carga un CSV a una tabla PostgreSQL.

    Estrategia: if_exists="replace" — reemplaza la tabla completa.
    Esto garantiza que la base de datos siempre refleja el estado
    actual de los CSVs procesados.

    Args:
        engine:   SQLAlchemy Engine
        nombre:   Nombre descriptivo del dataset (para logs)
        ruta_csv: Path al archivo CSV
        tabla:    Nombre de la tabla destino en PostgreSQL

    Returns:
        Número de filas cargadas

    Raises:
        FileNotFoundError: Si el CSV no existe
        SQLAlchemyError:   Si falla la carga
    """
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"CSV no encontrado: '{ruta_csv}'\n"
            "Verifica que el pipeline ETL fue ejecutado correctamente."
        )

    logger.info(f"Cargando {nombre} → tabla '{tabla}'...")

    # Leer CSV
    df = pd.read_csv(ruta_csv)
    if df.empty:
        logger.warning(f"El CSV '{ruta_csv.name}' está vacío. Se omite.")
        return 0

    # Preparar
    df = _preparar_dataframe(df, tabla)
    n_filas = len(df)

    # Cargar a PostgreSQL
    df.to_sql(
        tabla,
        engine,
        if_exists="replace",   # reemplaza la tabla completa
        index=False,
        method="multi",        # inserta en lotes (más rápido)
        chunksize=500,         # lotes de 500 filas
    )

    logger.info(
        f"  ✓ {nombre}: {n_filas:,} filas cargadas en '{tabla}'"
    )
    return n_filas


def load_all(engine) -> dict[str, int]:
    """
    Carga todos los CSV procesados a sus tablas correspondientes.

    Args:
        engine: SQLAlchemy Engine

    Returns:
        Dict con nombre del dataset → filas cargadas
        Los datasets que fallaron tienen valor -1.
    """
    resultados = {}

    for nombre, (ruta_csv, tabla) in CARGAS.items():
        try:
            n = cargar_csv_a_tabla(engine, nombre, ruta_csv, tabla)
            resultados[nombre] = n
            _registrar_log_etl(
                engine,
                proceso=f"carga_{tabla}",
                estado="OK",
                registros=n,
                mensaje=f"Carga completada desde {ruta_csv.name}",
            )
        except FileNotFoundError as e:
            logger.warning(str(e))
            resultados[nombre] = -1
            _registrar_log_etl(
                engine,
                proceso=f"carga_{tabla}",
                estado="ADVERTENCIA",
                registros=0,
                mensaje=str(e),
            )
        except SQLAlchemyError as e:
            logger.error(f"Error al cargar '{nombre}' → '{tabla}': {e}")
            resultados[nombre] = -1
            _registrar_log_etl(
                engine,
                proceso=f"carga_{tabla}",
                estado="ERROR",
                registros=0,
                mensaje=str(e),
            )

    return resultados


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from database.db import get_engine

    logger.info("=" * 55)
    logger.info("AireChile Analytics — Carga a PostgreSQL")
    logger.info("=" * 55)

    try:
        engine = get_engine()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    resultados = load_all(engine)

    print("\n" + "=" * 55)
    print("  CARGA A POSTGRESQL — RESUMEN")
    print("=" * 55)
    total_filas = 0
    for nombre, n in resultados.items():
        if n >= 0:
            print(f"  ✓ {nombre:<22} {n:>6,} filas")
            total_filas += n
        else:
            print(f"  ⚠ {nombre:<22} (no cargado — ver logs)")
    print(f"  {'─'*40}")
    print(f"  Total cargado:         {total_filas:>6,} filas")
    print("=" * 55 + "\n")