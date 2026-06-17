"""
etl/etl_sinca_main.py
=====================
AireChile Analytics — Orquestador del pipeline ETL SINCA.

Ejecuta el flujo completo de extracción y transformación de datos SINCA:

    1. Leer ruta del CSV desde variables de entorno (.env)
    2. Extraer datos crudos con extract_sinca()
    3. Transformar y enriquecer con transform_sinca()
    4. Validar el dataset resultante
    5. Guardar CSV limpio en data/processed/
    6. Imprimir resumen del proceso con estadísticas

Este archivo es el punto de entrada para la etapa SINCA del ETL.
Cuando se integren Open-Meteo y PostgreSQL, este módulo será llamado
desde etl_main.py como parte del pipeline general.

Uso:
    # Desde la raíz del proyecto:
    python etl/etl_sinca_main.py

    # Con ruta explícita:
    SINCA_RAW_PATH=data/raw/sinca_puente_alto_mp25_2022_2026.csv python etl/etl_sinca_main.py

Variables de entorno (.env):
    SINCA_RAW_PATH   → ruta al archivo CSV de SINCA
    SINCA_ESTACION   → nombre de la estación (opcional, se infiere del archivo)
    SINCA_COMUNA     → nombre de la comuna  (opcional, igual que estación)
    SINCA_OUTPUT     → ruta de salida del CSV procesado (opcional)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Agregar raíz del proyecto al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.extract_sinca import extract_sinca
from etl.transform_sinca import transform_sinca, guardar_procesado

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
# Constantes de configuración por defecto
# ---------------------------------------------------------------------------
DEFAULT_SINCA_PATH  = "data/raw/sinca_puente_alto_mp25_2022_2026.csv"
DEFAULT_OUTPUT_PATH = "data/processed/sinca_puente_alto_mp25_limpio.csv"
DEFAULT_ESTACION    = "Puente Alto"
DEFAULT_COMUNA      = "Puente Alto"


# ---------------------------------------------------------------------------
# Validaciones post-transformación
# ---------------------------------------------------------------------------

def validar_dataset(df, nombre_etapa: str = "post-transformación") -> bool:
    """
    Aplica un conjunto de validaciones al DataFrame transformado.
    Registra advertencias en lugar de lanzar excepciones para no
    detener el pipeline por problemas menores.

    Validaciones:
        - Columnas obligatorias presentes
        - Columna mp25 es numérica
        - No hay filas sin fecha
        - El target tiene al menos dos clases distintas (necesario para el modelo)
        - Porcentaje de nulos en mp25 es razonable (< 20%)

    Args:
        df:           DataFrame a validar
        nombre_etapa: Etiqueta para los mensajes de log

    Returns:
        True si todas las validaciones críticas pasan, False si alguna falla.
    """
    ok = True

    COLUMNAS_OBLIGATORIAS = [
        "fecha", "estacion", "comuna", "mp25", "estado_registro",
        "nivel_calidad_aire", "mes", "dia_semana",
        "mp25_dia_anterior", "mp25_promedio_7d",
        "nivel_calidad_aire_dia_siguiente",
    ]

    logger.info(f"--- Validando dataset ({nombre_etapa}) ---")

    # 1. Columnas obligatorias
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        logger.error(f"FALLO — Columnas faltantes: {faltantes}")
        ok = False
    else:
        logger.info("✓ Todas las columnas obligatorias presentes")

    # 2. mp25 es numérico
    import pandas as pd
    if not pd.api.types.is_numeric_dtype(df["mp25"]):
        logger.error("FALLO — La columna mp25 no es numérica")
        ok = False
    else:
        logger.info("✓ Columna mp25 es numérica")

    # 3. Sin filas sin fecha
    n_sin_fecha = df["fecha"].isna().sum()
    if n_sin_fecha > 0:
        logger.error(f"FALLO — {n_sin_fecha:,} filas sin fecha válida")
        ok = False
    else:
        logger.info("✓ Todas las filas tienen fecha válida")

    # 4. Target tiene al menos dos clases
    clases_target = df["nivel_calidad_aire_dia_siguiente"].dropna().unique()
    if len(clases_target) < 2:
        logger.warning(
            f"ADVERTENCIA — El target solo tiene {len(clases_target)} clase(s): "
            f"{clases_target}. El modelo necesita al menos 2 clases."
        )
    else:
        logger.info(f"✓ Target tiene {len(clases_target)} clases: {sorted(clases_target)}")

    # 5. Porcentaje de nulos en mp25
    pct_nulos_mp25 = df["mp25"].isna().mean() * 100
    if pct_nulos_mp25 > 20:
        logger.warning(
            f"ADVERTENCIA — {pct_nulos_mp25:.1f}% de nulos en mp25. "
            "Considera revisar la calidad del archivo fuente."
        )
    else:
        logger.info(f"✓ Nulos en mp25: {pct_nulos_mp25:.1f}%")

    return ok


def imprimir_resumen(df, ruta_salida: Path, tiempo_inicio: datetime) -> None:
    """
    Imprime en consola un resumen completo del resultado del pipeline:
    dimensiones, rango de fechas, distribución del target y ruta de salida.

    Args:
        df:           DataFrame transformado
        ruta_salida:  Ruta donde se guardó el CSV procesado
        tiempo_inicio: datetime de inicio del proceso (para calcular duración)
    """
    import pandas as pd

    duracion = (datetime.now() - tiempo_inicio).total_seconds()

    # Distribución del target (solo filas con valor)
    df_con_target = df["nivel_calidad_aire_dia_siguiente"].dropna()
    dist_target   = df_con_target.value_counts().to_dict()
    total_target  = len(df_con_target)

    # Distribución nivel actual
    dist_actual = df["nivel_calidad_aire"].value_counts().to_dict()

    print("\n" + "=" * 62)
    print("  PIPELINE ETL SINCA — COMPLETADO")
    print("=" * 62)
    print(f"  Duración           : {duracion:.1f} segundos")
    print(f"  Registros totales  : {len(df):,}")
    print(f"  Rango de fechas    : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  Estación           : {df['estacion'].iloc[0]}")
    print(f"  Comuna             : {df['comuna'].iloc[0]}")
    print()
    print("  Distribución nivel_calidad_aire (día actual):")
    for nivel in ["buena", "regular", "mala"]:
        n   = dist_actual.get(nivel, 0)
        pct = n / len(df) * 100 if len(df) > 0 else 0
        bar = "█" * int(pct / 3)
        print(f"    {nivel:<10} {n:>5,} días  ({pct:5.1f}%)  {bar}")
    print()
    print(f"  TARGET (nivel_calidad_aire_dia_siguiente):")
    print(f"  Filas con target   : {total_target:,}")
    print(f"  Filas sin target   : {len(df) - total_target:,} (última fila por diseño)")
    for nivel in ["buena", "regular", "mala"]:
        n   = dist_target.get(nivel, 0)
        pct = n / total_target * 100 if total_target > 0 else 0
        bar = "█" * int(pct / 3)
        print(f"    {nivel:<10} {n:>5,} días  ({pct:5.1f}%)  {bar}")
    print()

    # Nulos por columna de features
    print("  Nulos por columna:")
    cols_mostrar = ["mp25", "mp25_dia_anterior", "mp25_promedio_7d",
                    "nivel_calidad_aire_dia_siguiente"]
    for col in cols_mostrar:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"    {col:<45} {n:>4,}")
    print()
    print(f"  Archivo guardado en: {ruta_salida}")
    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def run_etl_sinca() -> Path:
    """
    Ejecuta el pipeline ETL completo para los datos SINCA.

    Lee la configuración desde variables de entorno, extrae los datos
    del CSV, los transforma, valida el resultado y lo guarda en
    data/processed/.

    Returns:
        Path al archivo CSV procesado guardado en data/processed/

    Raises:
        FileNotFoundError: Si el archivo SINCA no existe
        ValueError: Si el dataset resultante está vacío o es inválido
        SystemExit: Si ocurre un error crítico no recuperable
    """
    load_dotenv()
    tiempo_inicio = datetime.now()

    logger.info("=" * 62)
    logger.info("AireChile Analytics — ETL SINCA")
    logger.info(f"Inicio: {tiempo_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 62)

    # --- Leer configuración desde .env ---
    ruta_csv  = os.getenv("SINCA_RAW_PATH",  DEFAULT_SINCA_PATH)
    estacion  = os.getenv("SINCA_ESTACION",  DEFAULT_ESTACION)
    comuna    = os.getenv("SINCA_COMUNA",    DEFAULT_COMUNA)
    ruta_out  = os.getenv("SINCA_OUTPUT",    DEFAULT_OUTPUT_PATH)

    logger.info(f"Archivo fuente : {ruta_csv}")
    logger.info(f"Estación       : {estacion}")
    logger.info(f"Comuna         : {comuna}")
    logger.info(f"Archivo salida : {ruta_out}")

    # --- PASO 1: Extracción ---
    logger.info("PASO 1/3 — Extracción de datos SINCA")
    try:
        df_raw = extract_sinca(
            ruta_csv,
            estacion=estacion,
            comuna=comuna,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "Solución: descarga el archivo histórico desde "
            "https://sinca.mma.gob.cl/ → Export histórico por parámetro "
            "(Formato B) y guárdalo en data/raw/"
        )
        sys.exit(1)
    except ValueError as e:
        logger.error(f"El archivo no tiene el formato esperado: {e}")
        sys.exit(1)

    # --- PASO 2: Transformación ---
    logger.info("PASO 2/3 — Transformación y enriquecimiento")
    try:
        df_limpio = transform_sinca(df_raw)
    except ValueError as e:
        logger.error(f"Error en transformación: {e}")
        sys.exit(1)

    # --- PASO 3: Validación y guardado ---
    logger.info("PASO 3/3 — Validación y guardado")
    es_valido = validar_dataset(df_limpio)

    if not es_valido:
        logger.error(
            "El dataset no pasó las validaciones críticas. "
            "Revisa los errores anteriores antes de continuar."
        )
        sys.exit(1)

    ruta_guardada = guardar_procesado(df_limpio, ruta_out)

    # --- Resumen final ---
    imprimir_resumen(df_limpio, ruta_guardada, tiempo_inicio)

    return ruta_guardada


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_etl_sinca()