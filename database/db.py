"""
database/db.py
==============
AireChile Analytics — Módulo de conexión a PostgreSQL.

Expone get_engine() como punto de entrada único para obtener
un engine SQLAlchemy. Todas las operaciones de base de datos
del proyecto deben usar esta función para garantizar que las
credenciales se lean siempre desde .env y que los errores de
conexión se reporten de forma clara.

Uso:
    from database.db import get_engine

    engine = get_engine()
    df.to_sql("mediciones_sinca", engine, if_exists="replace", index=False)

Variables de entorno requeridas (.env):
    DATABASE_URL       → URL completa (prioridad si está definida)
    POSTGRES_HOST      → localhost
    POSTGRES_PORT      → 5432
    POSTGRES_DB        → airechile
    POSTGRES_USER      → postgres
    POSTGRES_PASSWORD  → postgres
"""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Valores por defecto (sobreescribibles desde .env)
DEFAULT_HOST     = "localhost"
DEFAULT_PORT     = "5432"
DEFAULT_DB       = "airechile"
DEFAULT_USER     = "postgres"
DEFAULT_PASSWORD = "postgres"


def _construir_url() -> str:
    """
    Construye la URL de conexión a PostgreSQL.

    Si DATABASE_URL está definida en .env, la usa directamente.
    Si no, construye la URL a partir de las variables individuales.

    Returns:
        String con la URL de conexión SQLAlchemy

    Raises:
        ValueError: Si faltan variables obligatorias
    """
    # Prioridad 1: DATABASE_URL completa
    url = os.getenv("DATABASE_URL")
    if url:
        # Asegurar que usa el driver psycopg2
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        logger.debug("Usando DATABASE_URL desde .env")
        return url

    # Prioridad 2: variables individuales
    host     = os.getenv("POSTGRES_HOST",     DEFAULT_HOST)
    port     = os.getenv("POSTGRES_PORT",     DEFAULT_PORT)
    db       = os.getenv("POSTGRES_DB",       DEFAULT_DB)
    user     = os.getenv("POSTGRES_USER",     DEFAULT_USER)
    password = os.getenv("POSTGRES_PASSWORD", DEFAULT_PASSWORD)

    if not all([host, port, db, user, password]):
        raise ValueError(
            "Faltan variables de entorno para PostgreSQL.\n"
            "Define en tu .env: POSTGRES_HOST, POSTGRES_PORT, "
            "POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD\n"
            "O define DATABASE_URL directamente."
        )

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    logger.debug(f"URL construida: postgresql+psycopg2://{user}:***@{host}:{port}/{db}")
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Crea y retorna un engine SQLAlchemy conectado a PostgreSQL.

    El engine se cachea con lru_cache para reutilizarlo en toda la
    sesión sin abrir conexiones innecesarias.

    Returns:
        sqlalchemy.engine.Engine listo para usar

    Raises:
        ValueError: Si faltan variables de entorno
        RuntimeError: Si no se puede establecer la conexión
    """
    try:
        url    = _construir_url()
        engine = create_engine(
            url,
            pool_pre_ping=True,    # verifica la conexión antes de usarla
            pool_recycle=3600,     # renueva conexiones cada hora
            echo=False,            # True para ver SQL generado (debug)
        )

        # Verificar que la conexión funciona
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("Conexión a PostgreSQL establecida correctamente")
        return engine

    except OperationalError as e:
        raise RuntimeError(
            f"No se pudo conectar a PostgreSQL: {e}\n\n"
            "Verifica:\n"
            "  1. PostgreSQL está corriendo (pg_ctl status / services.msc)\n"
            "  2. Las credenciales en .env son correctas\n"
            "  3. La base de datos existe (CREATE DATABASE airechile;)\n"
            "  4. psycopg2-binary está instalado (pip install psycopg2-binary)"
        ) from e

    except SQLAlchemyError as e:
        raise RuntimeError(f"Error de SQLAlchemy: {e}") from e


def probar_conexion() -> bool:
    """
    Prueba la conexión sin lanzar excepciones.
    Útil para verificar el estado antes de ejecutar el ETL.

    Returns:
        True si la conexión es exitosa, False si falla
    """
    try:
        get_engine()
        return True
    except (RuntimeError, ValueError) as e:
        logger.error(f"Prueba de conexión falló: {e}")
        return False


if __name__ == "__main__":
    logger.info("Probando conexión a PostgreSQL...")
    if probar_conexion():
        print("✓ Conexión exitosa a PostgreSQL")
    else:
        print("✗ No se pudo conectar. Revisa los logs.")