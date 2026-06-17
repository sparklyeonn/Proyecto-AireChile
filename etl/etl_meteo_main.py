"""
etl/etl_meteo_main.py
=====================
AireChile Analytics — Orquestador del pipeline meteorológico.

Ejecuta el flujo completo de integración meteorológica:

    PASO 1 — Extracción: consulta Open-Meteo Historical API
    PASO 2 — Transformación: limpia y renombra columnas
    PASO 3 — Merge: une con dataset SINCA por fecha
    PASO 4 — Guardado: dataset_modelo_base.csv listo para el modelo

Este archivo es el punto de entrada de la etapa meteorológica.
Cuando se integre PostgreSQL, etl_main.py lo llamará como paso 2
del pipeline general (después del ETL SINCA).

Uso:
    python etl/etl_meteo_main.py

Variables de entorno necesarias (.env):
    SINCA_PROCESSED_PATH      → ruta del dataset SINCA limpio
    METEO_RAW_PATH            → donde guardar el CSV crudo de Open-Meteo
    METEO_PROCESSED_PATH      → donde guardar el CSV meteorológico limpio
    DATASET_MODELO_BASE_PATH  → ruta del dataset final del modelo
    METEO_LATITUDE            → coordenada latitud
    METEO_LONGITUDE           → coordenada longitud
    METEO_START_DATE          → fecha inicio (debe cubrir el rango SINCA)
    METEO_END_DATE            → fecha fin
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.extract_meteo import extract_meteo
from etl.transform_meteo import transform_meteo
from etl.merge_sinca_meteo import merge_sinca_meteo

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def run_etl_meteo() -> Path:
    """
    Ejecuta el pipeline completo de integración meteorológica.

    Returns:
        Path al archivo dataset_modelo_base.csv generado

    Raises:
        SystemExit: Si algún paso falla de forma crítica
    """
    load_dotenv()
    tiempo_inicio = datetime.now()

    logger.info("=" * 62)
    logger.info("AireChile Analytics — ETL Meteorológico + Merge")
    logger.info(f"Inicio: {tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 62)

    # Rutas desde .env
    meteo_raw  = os.getenv("METEO_RAW_PATH",           "data/raw/open_meteo_puente_alto_2022_2026.csv")
    meteo_proc = os.getenv("METEO_PROCESSED_PATH",     "data/processed/open_meteo_transformado.csv")
    sinca_proc = os.getenv("SINCA_PROCESSED_PATH",     "data/processed/sinca_transformado.csv")
    output     = os.getenv("DATASET_MODELO_BASE_PATH", "data/processed/dataset_modelo_base.csv")

    # -----------------------------------------------------------------------
    # PASO 1 — Extracción Open-Meteo
    # -----------------------------------------------------------------------
    logger.info("PASO 1/3 — Extracción desde Open-Meteo Historical API")
    try:
        df_raw = extract_meteo(raw_path=meteo_raw)
        logger.info(f"  ✓ {len(df_raw):,} registros extraídos")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error en extracción Open-Meteo: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # PASO 2 — Transformación meteorológica
    # -----------------------------------------------------------------------
    logger.info("PASO 2/3 — Transformación y normalización")
    try:
        df_meteo = transform_meteo(
            ruta_crudo=meteo_raw,
            ruta_salida=meteo_proc,
        )
        logger.info(f"  ✓ {len(df_meteo):,} registros transformados")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error en transformación meteorológica: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # PASO 3 — Merge SINCA + Open-Meteo
    # -----------------------------------------------------------------------
    logger.info("PASO 3/3 — Merge SINCA + meteorología")
    try:
        df_final = merge_sinca_meteo(
            ruta_sinca=sinca_proc,
            ruta_meteo=meteo_proc,
            ruta_salida=output,
        )
        logger.info(f"  ✓ {len(df_final):,} registros en el dataset final")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error en merge: {e}")
        logger.error(
            "Verifica que SINCA_PROCESSED_PATH existe. "
            "Si aún no ejecutaste etl_sinca_main.py, hazlo primero."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Resumen final
    # -----------------------------------------------------------------------
    duracion = (datetime.now() - tiempo_inicio).total_seconds()
    ruta_final = Path(output)

    print("\n" + "=" * 62)
    print("  PIPELINE METEOROLÓGICO — COMPLETADO")
    print("=" * 62)
    print(f"  Duración             : {duracion:.1f} segundos")
    print(f"  Registros en dataset : {len(df_final):,}")
    print(f"  Rango de fechas      : "
          f"{df_final['fecha'].min().date()} → {df_final['fecha'].max().date()}")
    print(f"  Columnas totales     : {df_final.shape[1]}")
    print()

    # Columnas meteorológicas con cobertura
    cols_meteo = [
        "temperatura_max", "temperatura_min", "temperatura_promedio",
        "humedad_relativa", "velocidad_viento", "precipitacion"
    ]
    print("  Cobertura columnas meteorológicas:")
    for col in cols_meteo:
        if col in df_final.columns:
            n_ok  = df_final[col].notna().sum()
            pct   = n_ok / len(df_final) * 100
            print(f"    {col:<25} {n_ok:>5,} / {len(df_final):,}  ({pct:.1f}%)")

    print()
    print("  Distribución TARGET (nivel_calidad_aire_dia_siguiente):")
    if "nivel_calidad_aire_dia_siguiente" in df_final.columns:
        dist = df_final["nivel_calidad_aire_dia_siguiente"].value_counts()
        for nivel, n in dist.items():
            pct = n / len(df_final) * 100
            bar = "█" * int(pct / 3)
            print(f"    {nivel:<10} {n:>5,} días  ({pct:.1f}%)  {bar}")
    print()
    print(f"  Archivo guardado en: {ruta_final}")
    print("=" * 62 + "\n")

    return ruta_final


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_etl_meteo()