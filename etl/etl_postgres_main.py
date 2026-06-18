"""
etl/etl_postgres_main.py
========================
AireChile Analytics — Orquestador del pipeline PostgreSQL.

Ejecuta el flujo completo de inicialización y carga de datos:

    PASO 1 — Verificar conexión a PostgreSQL
    PASO 2 — Inicializar schema (crear tablas si no existen)
    PASO 3 — Cargar todos los CSV procesados
    PASO 4 — Mostrar resumen

Este archivo es el punto de entrada de la etapa de persistencia.
Puede ejecutarse de forma independiente o llamarse desde un
orquestador general del pipeline.

Prerrequisitos:
    1. PostgreSQL corriendo y accesible
    2. Base de datos creada: CREATE DATABASE airechile;
    3. Variables de entorno configuradas en .env
    4. Pipeline ETL ejecutado (CSVs en data/processed/)

Uso:
    python etl/etl_postgres_main.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_etl_postgres() -> bool:
    """
    Ejecuta el pipeline completo de persistencia en PostgreSQL.

    Returns:
        True si todo se completó correctamente, False si hubo errores
    """
    tiempo_inicio = datetime.now()

    logger.info("=" * 60)
    logger.info("AireChile Analytics — ETL PostgreSQL")
    logger.info(f"Inicio: {tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # PASO 1 — Verificar conexión
    # ------------------------------------------------------------------
    logger.info("PASO 1/3 — Verificando conexión a PostgreSQL...")
    try:
        from database.db import get_engine, probar_conexion
    except ImportError as e:
        logger.error(f"No se pudo importar database.db: {e}")
        return False

    if not probar_conexion():
        logger.error(
            "No se pudo conectar a PostgreSQL. Verifica:\n"
            "  · PostgreSQL está corriendo\n"
            "  · Las credenciales en .env son correctas\n"
            "  · La base de datos 'airechile' existe\n"
            "  · psycopg2-binary está instalado"
        )
        return False

    logger.info("  ✓ Conexión establecida")

    # ------------------------------------------------------------------
    # PASO 2 — Inicializar schema
    # ------------------------------------------------------------------
    logger.info("PASO 2/3 — Inicializando schema (CREATE TABLE IF NOT EXISTS)...")
    try:
        from database.init_db import init_db
        exito_schema = init_db()
    except Exception as e:
        logger.error(f"Error al inicializar el schema: {e}")
        return False

    if not exito_schema:
        logger.error("El schema no se inicializó correctamente.")
        return False

    logger.info("  ✓ Schema verificado")

    # ------------------------------------------------------------------
    # PASO 3 — Cargar datos
    # ------------------------------------------------------------------
    logger.info("PASO 3/3 — Cargando datos a PostgreSQL...")
    try:
        from etl.load_postgres import load_all
        engine = get_engine()
        resultados = load_all(engine)
    except Exception as e:
        logger.error(f"Error durante la carga: {e}")
        return False

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    duracion = (datetime.now() - tiempo_inicio).total_seconds()

    exitos    = {k: v for k, v in resultados.items() if v >= 0}
    fallidos  = {k: v for k, v in resultados.items() if v < 0}
    total     = sum(v for v in exitos.values())

    print("\n" + "=" * 60)
    print("  PIPELINE POSTGRESQL — COMPLETADO")
    print("=" * 60)
    print(f"  Duración             : {duracion:.1f} segundos")
    print()
    print("  Tablas cargadas:")
    for nombre, n in exitos.items():
        print(f"    ✓ {nombre:<25} {n:>6,} filas")
    if fallidos:
        print()
        print("  No cargados (CSV faltante):")
        for nombre in fallidos:
            print(f"    ⚠ {nombre}")
    print(f"  {'─' * 45}")
    print(f"  Total filas insertadas: {total:>6,}")
    print()
    print("  Puedes consultar los datos con:")
    print("    psql -U postgres -d airechile")
    print("    SELECT COUNT(*) FROM dataset_modelo;")
    print("=" * 60 + "\n")

    return len(fallidos) == 0


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exito = run_etl_postgres()
    sys.exit(0 if exito else 1)