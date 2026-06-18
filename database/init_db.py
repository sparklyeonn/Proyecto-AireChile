"""
database/init_db.py
===================
AireChile Analytics — Inicialización del esquema PostgreSQL.

Crea todas las tablas definidas en database/schema.sql en la base
de datos PostgreSQL configurada en .env.

Es seguro ejecutar múltiples veces: usa CREATE TABLE IF NOT EXISTS,
por lo que no sobreescribe datos existentes.

Uso:
    python database/init_db.py

Resultado esperado:
    ✓ Tablas creadas: estaciones, mediciones_sinca, meteorologia,
                      dataset_modelo, predicciones_modelo, log_etl
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import inspect, text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

TABLAS_ESPERADAS = [
    "estaciones",
    "mediciones_sinca",
    "meteorologia",
    "dataset_modelo",
    "predicciones_modelo",
    "log_etl",
]


def leer_schema() -> str:
    """
    Lee el contenido del archivo schema.sql.

    Returns:
        String con el SQL del schema

    Raises:
        FileNotFoundError: Si schema.sql no existe
    """
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el schema en: {SCHEMA_PATH}\n"
            "Asegúrate de que database/schema.sql existe en el proyecto."
        )
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    logger.info(f"Schema leído desde: {SCHEMA_PATH}")
    return sql


def crear_tablas(engine, sql: str) -> None:
    """
    Ejecuta el SQL del schema para crear las tablas.

    Usa CREATE TABLE IF NOT EXISTS, por lo que es idempotente:
    no falla si las tablas ya existen y no borra datos existentes.

    Args:
        engine: SQLAlchemy Engine
        sql:    Contenido del schema.sql

    Raises:
        Exception: Si falla alguna sentencia SQL
    """
    with engine.connect() as conn:
        # Ejecutar el schema completo como un bloque de transacción
        conn.execute(text(sql))
        conn.commit()
    logger.info("Schema ejecutado correctamente")


def verificar_tablas(engine) -> list[str]:
    """
    Verifica qué tablas del schema fueron creadas efectivamente.

    Args:
        engine: SQLAlchemy Engine

    Returns:
        Lista de nombres de tablas encontradas en la base de datos
    """
    inspector = inspect(engine)
    tablas_existentes = inspector.get_table_names()

    encontradas = []
    for tabla in TABLAS_ESPERADAS:
        if tabla in tablas_existentes:
            encontradas.append(tabla)
            logger.info(f"  ✓ {tabla}")
        else:
            logger.warning(f"  ✗ {tabla} — no encontrada")

    return encontradas


def init_db() -> bool:
    """
    Inicializa la base de datos ejecutando el schema.

    Returns:
        True si todas las tablas fueron creadas correctamente
    """
    # Importar aquí para que el error de conexión sea claro
    try:
        from database.db import get_engine
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from database.db import get_engine

    logger.info("=" * 55)
    logger.info("AireChile Analytics — Inicialización PostgreSQL")
    logger.info("=" * 55)

    # 1. Leer schema
    try:
        sql = leer_schema()
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    # 2. Conectar
    try:
        engine = get_engine()
    except RuntimeError as e:
        logger.error(str(e))
        return False

    # 3. Crear tablas
    try:
        crear_tablas(engine, sql)
    except Exception as e:
        logger.error(f"Error al ejecutar el schema: {e}")
        return False

    # 4. Verificar
    logger.info("Verificando tablas creadas:")
    encontradas = verificar_tablas(engine)

    todas_ok = len(encontradas) == len(TABLAS_ESPERADAS)

    if todas_ok:
        logger.info(
            f"Base de datos inicializada correctamente "
            f"({len(encontradas)}/{len(TABLAS_ESPERADAS)} tablas)"
        )
    else:
        faltantes = set(TABLAS_ESPERADAS) - set(encontradas)
        logger.warning(f"Tablas no creadas: {faltantes}")

    return todas_ok


if __name__ == "__main__":
    exito = init_db()
    if exito:
        print("\n✓ Base de datos lista. Puedes cargar datos con:")
        print("  python etl/load_postgres.py")
    else:
        print("\n✗ La inicialización falló. Revisa los logs.")
        sys.exit(1)