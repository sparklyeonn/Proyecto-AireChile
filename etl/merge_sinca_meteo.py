"""
etl/merge_sinca_meteo.py
========================
AireChile Analytics — Unión de datos SINCA con meteorología Open-Meteo.

Une el dataset limpio de SINCA con los datos meteorológicos por la
columna 'fecha' usando un left join (el dataset SINCA es la base).

Estrategia del merge:
    - Base: sinca_transformado.csv   (eje temporal de referencia)
    - Derecha: open_meteo_transformado.csv
    - Tipo de join: LEFT — se conservan todas las filas de SINCA aunque
      no haya datos meteorológicos para esa fecha
    - Clave de unión: 'fecha' (datetime64 en ambos datasets)

Validaciones post-merge:
    - Reporta filas de SINCA sin datos meteorológicos correspondientes
    - Alerta si se pierde más del 10% de las filas SINCA
    - Verifica que la variable objetivo no se pierda
    - Verifica que el dataset final no quede vacío

Columnas del dataset final (dataset_modelo_base.csv):
    fecha | estacion | comuna | mp25 | estado_registro
    nivel_calidad_aire | mes | dia_semana | mp25_dia_anterior
    mp25_promedio_7d | temperatura_max | temperatura_min
    temperatura_promedio | humedad_relativa | velocidad_viento
    precipitacion | nivel_calidad_aire_dia_siguiente

Uso:
    from etl.merge_sinca_meteo import merge_sinca_meteo

    df = merge_sinca_meteo()

Variables de entorno (.env):
    SINCA_PROCESSED_PATH      → sinca_transformado.csv
    METEO_PROCESSED_PATH      → open_meteo_transformado.csv
    DATASET_MODELO_BASE_PATH  → dataset_modelo_base.csv
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

DEFAULT_SINCA_PATH   = "data/processed/sinca_transformado.csv"
DEFAULT_METEO_PATH   = "data/processed/open_meteo_transformado.csv"
DEFAULT_OUTPUT_PATH  = "data/processed/dataset_modelo_base.csv"

# Porcentaje máximo de filas SINCA sin datos meteorológicos antes de alertar
UMBRAL_ALERTA_NULOS_PCT = 10.0

# Orden final de columnas del dataset de modelo
COLUMNAS_FINALES = [
    "fecha",
    "estacion",
    "comuna",
    "mp25",
    "estado_registro",
    "nivel_calidad_aire",
    "mes",
    "dia_semana",
    "mp25_dia_anterior",
    "mp25_promedio_7d",
    "temperatura_max",
    "temperatura_min",
    "temperatura_promedio",
    "humedad_relativa",
    "velocidad_viento",
    "precipitacion",
    "nivel_calidad_aire_dia_siguiente",
]


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def _cargar_csv(ruta: Path, nombre: str) -> pd.DataFrame:
    """
    Carga un CSV procesado y convierte la columna 'fecha' a datetime64.

    Args:
        ruta:   Path al CSV
        nombre: Nombre descriptivo para mensajes de error

    Returns:
        pd.DataFrame con 'fecha' como datetime64

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el CSV está vacío
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo '{nombre}' no encontrado: '{ruta}'\n"
            "Asegúrate de haber ejecutado el ETL previo."
        )

    df = pd.read_csv(ruta, encoding="utf-8", parse_dates=["fecha"])

    if df.empty:
        raise ValueError(f"El archivo '{ruta.name}' está vacío.")

    logger.info(
        f"{nombre}: {len(df):,} filas | "
        f"{df['fecha'].min().date()} → {df['fecha'].max().date()}"
    )
    return df


def _validar_compatibilidad(df_sinca: pd.DataFrame, df_meteo: pd.DataFrame) -> None:
    """
    Verifica que los rangos de fechas de ambos datasets se solapan
    suficientemente para un merge útil.

    Args:
        df_sinca: Dataset SINCA
        df_meteo: Dataset meteorológico

    Raises:
        ValueError: Si los datasets no tienen solapamiento de fechas
    """
    sinca_min = df_sinca["fecha"].min()
    sinca_max = df_sinca["fecha"].max()
    meteo_min = df_meteo["fecha"].min()
    meteo_max = df_meteo["fecha"].max()

    # Solapamiento: el inicio del más tardío debe ser antes del fin del más temprano
    inicio_solape = max(sinca_min, meteo_min)
    fin_solape    = min(sinca_max, meteo_max)

    if inicio_solape > fin_solape:
        raise ValueError(
            f"Los datasets no tienen solapamiento de fechas:\n"
            f"  SINCA:    {sinca_min.date()} → {sinca_max.date()}\n"
            f"  Open-Meteo: {meteo_min.date()} → {meteo_max.date()}\n"
            "Verifica METEO_START_DATE y METEO_END_DATE en tu .env"
        )

    dias_solape = (fin_solape - inicio_solape).days
    logger.info(
        f"Solapamiento de fechas: {inicio_solape.date()} → "
        f"{fin_solape.date()} ({dias_solape:,} días)"
    )


def _ejecutar_merge(
    df_sinca: pd.DataFrame,
    df_meteo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Realiza el left join de SINCA con meteorología por la columna 'fecha'.

    Se usa left join para preservar todas las filas de SINCA: si un día
    no tiene datos meteorológicos, las columnas meteo quedarán NaN pero
    la fila SINCA se conserva.

    Args:
        df_sinca: Dataset SINCA (base del join)
        df_meteo: Dataset meteorológico (derecha del join)

    Returns:
        pd.DataFrame merged
    """
    n_sinca = len(df_sinca)

    # Asegurar que 'fecha' tiene el mismo tipo en ambos datasets
    df_sinca = df_sinca.copy()
    df_meteo = df_meteo.copy()
    df_sinca["fecha"] = pd.to_datetime(df_sinca["fecha"])
    df_meteo["fecha"] = pd.to_datetime(df_meteo["fecha"])

    # Eliminar de df_meteo columnas que ya existen en df_sinca (excepto 'fecha')
    # para evitar columnas duplicadas con sufijos _x/_y
    cols_sinca_sin_fecha = set(df_sinca.columns) - {"fecha"}
    cols_meteo_limpias   = [
        c for c in df_meteo.columns
        if c not in cols_sinca_sin_fecha
    ]
    df_meteo_filtrado = df_meteo[cols_meteo_limpias]

    df_merged = pd.merge(
        df_sinca,
        df_meteo_filtrado,
        on="fecha",
        how="left",
    )

    logger.info(
        f"Merge completado: {n_sinca:,} filas SINCA → "
        f"{len(df_merged):,} filas resultado"
    )

    if len(df_merged) != n_sinca:
        logger.warning(
            f"El merge cambió el número de filas: "
            f"{n_sinca:,} → {len(df_merged):,}. "
            "Puede haber fechas duplicadas en el dataset meteorológico."
        )
    return df_merged


def _validar_resultado(df: pd.DataFrame, n_sinca_original: int) -> bool:
    """
    Valida el DataFrame resultante del merge.

    Validaciones:
        1. No está vacío
        2. Tiene la columna 'fecha'
        3. Tiene la variable objetivo
        4. No se perdieron demasiadas filas por falta de datos meteo

    Args:
        df: DataFrame resultante del merge
        n_sinca_original: Número de filas del dataset SINCA original

    Returns:
        True si pasa todas las validaciones críticas
    """
    ok = True

    # 1. No vacío
    if df.empty:
        logger.error("FALLO — El dataset resultante está vacío")
        return False
    logger.info(f"✓ Dataset no vacío: {len(df):,} filas")

    # 2. Columna fecha presente
    if "fecha" not in df.columns:
        logger.error("FALLO — Columna 'fecha' no encontrada en el resultado")
        ok = False
    else:
        logger.info("✓ Columna 'fecha' presente")

    # 3. Variable objetivo presente
    if "nivel_calidad_aire_dia_siguiente" not in df.columns:
        logger.error("FALLO — Variable objetivo 'nivel_calidad_aire_dia_siguiente' no encontrada")
        ok = False
    else:
        n_con_target = df["nivel_calidad_aire_dia_siguiente"].notna().sum()
        logger.info(f"✓ Variable objetivo presente — {n_con_target:,} filas con valor")

    # 4. Filas sin datos meteorológicos
    cols_meteo = ["temperatura_max", "temperatura_min", "precipitacion"]
    cols_presentes = [c for c in cols_meteo if c in df.columns]
    if cols_presentes:
        n_sin_meteo = df[cols_presentes[0]].isna().sum()
        pct_sin_meteo = n_sin_meteo / len(df) * 100 if len(df) > 0 else 0

        if pct_sin_meteo > UMBRAL_ALERTA_NULOS_PCT:
            logger.warning(
                f"⚠️ {pct_sin_meteo:.1f}% de filas ({n_sin_meteo:,}) sin datos "
                f"meteorológicos. Considera ampliar el rango de METEO_START_DATE "
                f"/ METEO_END_DATE para cubrir todo el rango SINCA."
            )
        else:
            logger.info(
                f"✓ Cobertura meteorológica: "
                f"{100 - pct_sin_meteo:.1f}% de las filas con datos meteo"
            )

    return ok


def _seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona y ordena las columnas finales del dataset de modelo.
    Las columnas en COLUMNAS_FINALES que no existen en df se ignoran.

    Args:
        df: DataFrame merged

    Returns:
        df con solo las columnas finales en el orden correcto
    """
    cols_disponibles = [c for c in COLUMNAS_FINALES if c in df.columns]
    cols_extra = [c for c in df.columns if c not in COLUMNAS_FINALES]

    if cols_extra:
        logger.info(f"Columnas extra no incluidas en el output: {cols_extra}")

    cols_faltantes = [c for c in COLUMNAS_FINALES if c not in df.columns]
    if cols_faltantes:
        logger.warning(f"Columnas esperadas no encontradas: {cols_faltantes}")

    return df[cols_disponibles].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def merge_sinca_meteo(
    ruta_sinca:  str | Path | None = None,
    ruta_meteo:  str | Path | None = None,
    ruta_salida: str | Path | None = None,
) -> pd.DataFrame:
    """
    Une el dataset SINCA con los datos meteorológicos de Open-Meteo.

    Args:
        ruta_sinca:  Ruta al CSV de SINCA procesado.
                     Si es None, usa SINCA_PROCESSED_PATH del .env.
        ruta_meteo:  Ruta al CSV de Open-Meteo procesado.
                     Si es None, usa METEO_PROCESSED_PATH del .env.
        ruta_salida: Ruta donde guardar el dataset final.
                     Si es None, usa DATASET_MODELO_BASE_PATH del .env.

    Returns:
        pd.DataFrame con todas las columnas SINCA + meteorología,
        listo para entrenar el modelo RandomForest.

    Raises:
        FileNotFoundError: Si algún archivo fuente no existe
        ValueError: Si los datasets no tienen solapamiento temporal
                    o si el resultado queda vacío
    """
    # Resolver rutas
    ruta_s = Path(ruta_sinca or os.getenv("SINCA_PROCESSED_PATH", DEFAULT_SINCA_PATH))
    ruta_m = Path(ruta_meteo or os.getenv("METEO_PROCESSED_PATH", DEFAULT_METEO_PATH))
    ruta_o = Path(ruta_salida or os.getenv("DATASET_MODELO_BASE_PATH", DEFAULT_OUTPUT_PATH))

    logger.info("Iniciando merge SINCA + Open-Meteo")

    # Cargar datasets
    df_sinca = _cargar_csv(ruta_s, "SINCA procesado")
    df_meteo = _cargar_csv(ruta_m, "Open-Meteo procesado")
    n_sinca  = len(df_sinca)

    # Verificar compatibilidad de fechas
    _validar_compatibilidad(df_sinca, df_meteo)

    # Ejecutar merge
    df_merged = _ejecutar_merge(df_sinca, df_meteo)

    # Validar resultado
    es_valido = _validar_resultado(df_merged, n_sinca)
    if not es_valido:
        raise ValueError(
            "El dataset resultante no pasó las validaciones. "
            "Revisa los errores anteriores."
        )

    # Seleccionar y ordenar columnas finales
    df_final = _seleccionar_columnas(df_merged)

    # Guardar
    ruta_o.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(ruta_o, index=False, encoding="utf-8")
    logger.info(f"Dataset final guardado en: {ruta_o} ({len(df_final):,} filas)")

    return df_final


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AireChile Analytics — Merge SINCA + Open-Meteo")
    logger.info("=" * 60)

    try:
        df = merge_sinca_meteo()

        print("\n" + "=" * 60)
        print("  MERGE COMPLETADO — dataset_modelo_base.csv")
        print("=" * 60)
        print(f"  Registros totales    : {len(df):,}")
        print(f"  Rango de fechas      : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
        print(f"  Columnas totales     : {df.shape[1]}")
        print()
        print("  Columnas del dataset:")
        for col in df.columns:
            n_nulos = df[col].isna().sum()
            print(f"    {col:<45} nulos: {n_nulos:>4}")
        print()
        print("  Primeras 3 filas:")
        print(df.head(3).to_string())
        print()
        print("  Distribución del TARGET:")
        print(df["nivel_calidad_aire_dia_siguiente"].value_counts())

    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)