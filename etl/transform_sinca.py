"""
etl/transform_sinca.py
======================
AireChile Analytics — Transformación de datos SINCA.

Responsabilidad: recibir el DataFrame crudo de extract_sinca.py y
producir el dataset limpio y enriquecido listo para cargar a PostgreSQL
y entrenar el modelo predictivo.

Transformaciones aplicadas (en orden):
    1. Construir mp25 definitivo (validado → preliminar como respaldo)
    2. Eliminar filas sin fecha válida o sin ningún valor de mp25
    3. Eliminar duplicados por fecha+estación
    4. Ordenar cronológicamente
    5. Calcular nivel_calidad_aire con umbrales norma chilena DS59
    6. Agregar columnas temporales: mes, dia_semana
    7. Calcular mp25_dia_anterior (lag de 1 día)
    8. Calcular mp25_promedio_7d (media móvil 7 días)
    9. Calcular nivel_calidad_aire_dia_siguiente (variable objetivo del modelo)
   10. Seleccionar y ordenar columnas finales

Umbrales de calidad del aire (MP2.5, µg/m³) — Decreto DS59/DS12:
    buena   : 0 – 25
    regular : 25.1 – 50
    mala    : > 50

Salida de transform_sinca():
    DataFrame con columnas:
        fecha                          (datetime64[ns])
        estacion                       (str)
        comuna                         (str)
        mp25                           (float64) — µg/m³
        estado_registro                (str)      — 'validado' | 'preliminar'
        nivel_calidad_aire             (str)      — buena | regular | mala
        mes                            (int)      — 1–12
        dia_semana                     (int)      — 0=lunes, 6=domingo
        mp25_dia_anterior              (float64)
        mp25_promedio_7d               (float64)
        nivel_calidad_aire_dia_siguiente (str)    — TARGET del modelo

Uso:
    from etl.extract_sinca import extract_sinca
    from etl.transform_sinca import transform_sinca

    df_raw    = extract_sinca("data/raw/sinca_puente_alto_mp25_2022_2026.csv")
    df_limpio = transform_sinca(df_raw)

También puede ejecutarse directamente:
    python etl/transform_sinca.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
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

# ---------------------------------------------------------------------------
# Umbrales de calidad del aire (Norma chilena DS59 para MP2.5)
# ---------------------------------------------------------------------------
# Estos valores son la única fuente de verdad para la clasificación.
# Si el docente o los datos requieren ajuste, modificar solo aquí.
UMBRAL_BUENA   = 25.0   # <= este valor → buena
UMBRAL_REGULAR = 50.0   # <= este valor → regular  (> 25)
                         # > UMBRAL_REGULAR           → mala

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
    "nivel_calidad_aire_dia_siguiente",
]


# ---------------------------------------------------------------------------
# Funciones de transformación (cada una tiene una responsabilidad única)
# ---------------------------------------------------------------------------

def _construir_mp25(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la columna mp25 definitiva y el estado_registro que
    indica de qué columna proviene el valor.

    Prioridad:
        1. mp25_validado    → estado 'validado'
        2. mp25_preliminar  → estado 'preliminar' (respaldo)
        3. NaN              → estado 'sin_dato' (fila se eliminará después)

    mp25_no_validado se ignora completamente porque en los datos reales
    de Puente Alto 2022-2026 está 100% nulo, y los datos no validados
    no son confiables para un modelo predictivo.

    Args:
        df: DataFrame con columnas mp25_validado y mp25_preliminar

    Returns:
        df con columnas mp25 y estado_registro agregadas
    """
    df = df.copy()

    # Inicializar con NaN y estado desconocido
    df["mp25"]            = np.nan
    df["estado_registro"] = "sin_dato"

    # Usar validado donde esté disponible
    mask_validado = df["mp25_validado"].notna()
    df.loc[mask_validado, "mp25"]            = df.loc[mask_validado, "mp25_validado"]
    df.loc[mask_validado, "estado_registro"] = "validado"

    # Completar con preliminar donde validado es nulo
    mask_prelim = df["mp25_validado"].isna() & df["mp25_preliminar"].notna()
    df.loc[mask_prelim, "mp25"]            = df.loc[mask_prelim, "mp25_preliminar"]
    df.loc[mask_prelim, "estado_registro"] = "preliminar"

    n_validado  = mask_validado.sum()
    n_prelim    = mask_prelim.sum()
    n_sin_dato  = (df["estado_registro"] == "sin_dato").sum()

    logger.info(
        f"mp25 construido — validado: {n_validado:,} | "
        f"preliminar (respaldo): {n_prelim:,} | "
        f"sin dato: {n_sin_dato:,}"
    )
    return df


def _limpiar_filas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas que no pueden usarse en el modelo:
        - Filas con fecha inválida (NaT)
        - Filas sin ningún valor de mp25
        - Filas con mp25 negativo (error de sensor)
        - Filas duplicadas por (fecha, estacion)

    Args:
        df: DataFrame con fecha y mp25 ya construidos

    Returns:
        df limpio, ordenado por fecha
    """
    df = df.copy()
    n_inicial = len(df)

    # Fechas inválidas
    mask_fecha_nula = df["fecha"].isna()
    if mask_fecha_nula.sum() > 0:
        logger.warning(f"Eliminando {mask_fecha_nula.sum():,} filas con fecha inválida")
        df = df[~mask_fecha_nula]

    # Sin valor de mp25
    mask_sin_mp25 = df["mp25"].isna()
    if mask_sin_mp25.sum() > 0:
        logger.warning(f"Eliminando {mask_sin_mp25.sum():,} filas sin valor de mp25")
        df = df[~mask_sin_mp25]

    # mp25 negativo (imposible físicamente)
    mask_negativo = df["mp25"] < 0
    if mask_negativo.sum() > 0:
        logger.warning(
            f"Eliminando {mask_negativo.sum():,} filas con mp25 negativo "
            f"(valores: {df.loc[mask_negativo, 'mp25'].values})"
        )
        df = df[~mask_negativo]

    # Duplicados por fecha+estacion (puede haber solapamiento entre descargas)
    n_antes_dup = len(df)
    df = df.drop_duplicates(subset=["fecha", "estacion"], keep="first")
    n_dup = n_antes_dup - len(df)
    if n_dup > 0:
        logger.info(f"Duplicados eliminados: {n_dup:,}")

    # Ordenar cronológicamente
    df = df.sort_values("fecha").reset_index(drop=True)

    n_final = len(df)
    logger.info(
        f"Limpieza completada — de {n_inicial:,} a {n_final:,} filas "
        f"({n_inicial - n_final:,} eliminadas)"
    )
    return df


def _clasificar_calidad(mp25: float | None) -> str:
    """
    Clasifica un valor de MP2.5 en tres categorías según la norma
    chilena DS59.

    Args:
        mp25: Valor de MP2.5 en µg/m³ (puede ser NaN)

    Returns:
        'buena' | 'regular' | 'mala' | 'sin_dato'
    """
    if pd.isna(mp25):
        return "sin_dato"
    if mp25 <= UMBRAL_BUENA:
        return "buena"
    if mp25 <= UMBRAL_REGULAR:
        return "regular"
    return "mala"


def _agregar_nivel_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica _clasificar_calidad() a la columna mp25 para crear
    nivel_calidad_aire.

    Args:
        df: DataFrame con columna mp25

    Returns:
        df con columna nivel_calidad_aire agregada
    """
    df = df.copy()
    df["nivel_calidad_aire"] = df["mp25"].apply(_clasificar_calidad)

    # Distribución de clases para verificar balance del dataset
    dist = df["nivel_calidad_aire"].value_counts()
    logger.info(f"Distribución nivel_calidad_aire: {dist.to_dict()}")

    return df


def _agregar_columnas_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas derivadas de la fecha que el modelo usará
    para capturar patrones estacionales y semanales.

    Columnas agregadas:
        mes        → entero 1-12
        dia_semana → entero 0 (lunes) a 6 (domingo)

    Args:
        df: DataFrame con columna fecha de tipo datetime64

    Returns:
        df con columnas mes y dia_semana agregadas
    """
    df = df.copy()
    df["mes"]        = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=lunes, 6=domingo
    return df


def _calcular_mp25_dia_anterior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula mp25_dia_anterior usando un shift de 1 posición.

    Asume que el DataFrame está ordenado cronológicamente y que
    las fechas son consecutivas (un registro por día por estación).
    Si hay huecos en las fechas, el shift tomará el valor del último
    día disponible, no necesariamente el día anterior real. Esto es
    aceptable para el modelo dado el volumen de datos.

    Para datasets multi-estación se aplica el shift dentro de cada
    grupo por estación.

    Args:
        df: DataFrame ordenado por fecha con columna mp25

    Returns:
        df con columna mp25_dia_anterior
    """
    df = df.copy()
    df["mp25_dia_anterior"] = (
        df.groupby("estacion")["mp25"]
        .shift(1)
    )
    n_nulos = df["mp25_dia_anterior"].isna().sum()
    logger.info(
        f"mp25_dia_anterior calculado — NaN esperados: 1 por estación "
        f"(primera fila sin día anterior). Total NaN: {n_nulos}"
    )
    return df


def _calcular_promedio_7d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula mp25_promedio_7d como la media móvil de los últimos 7 días
    de mp25 (ventana deslizante hacia atrás, sin incluir el día actual).

    El parámetro min_periods=3 permite calcular el promedio aunque
    haya menos de 7 días con datos (útil para el inicio de la serie
    o semanas con muchos nulos). Se puede ajustar a 7 si se prefiere
    ser más estricto.

    Args:
        df: DataFrame ordenado por fecha con columna mp25

    Returns:
        df con columna mp25_promedio_7d
    """
    df = df.copy()
    df["mp25_promedio_7d"] = (
        df.groupby("estacion")["mp25"]
        .transform(
            lambda x: x.shift(1)           # excluir el día actual
                       .rolling(window=7, min_periods=3)
                       .mean()
                       .round(2)
        )
    )
    n_nulos = df["mp25_promedio_7d"].isna().sum()
    logger.info(
        f"mp25_promedio_7d calculado — NaN en primeras semanas: {n_nulos:,}"
    )
    return df


def _calcular_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la variable objetivo del modelo:
    nivel_calidad_aire_dia_siguiente.

    Para cada fila del día t, el target es la clasificación del mp25
    del día t+1. Esto permite que el modelo aprenda:
    "dado lo que pasó hoy, ¿qué calidad tendrá el aire mañana?"

    La última fila de cada estación tendrá NaN en el target porque
    no existe el día siguiente en el dataset. Esas filas NO se eliminan
    aquí — se eliminan al momento de entrenar el modelo en train_model.py,
    para conservar los datos para predicción en tiempo real.

    Args:
        df: DataFrame con columnas mp25 y nivel_calidad_aire,
            ordenado por fecha

    Returns:
        df con columna nivel_calidad_aire_dia_siguiente
    """
    df = df.copy()

    # mp25 del día siguiente (shift -1)
    mp25_siguiente = df.groupby("estacion")["mp25"].shift(-1)

    # Clasificar según umbrales
    df["nivel_calidad_aire_dia_siguiente"] = mp25_siguiente.apply(
        _clasificar_calidad
    )

    # La última fila por estación quedará con "sin_dato" (shift da NaN)
    # La reemplazamos por NaN explícito para que train_model.py la filtre
    df.loc[
        df["nivel_calidad_aire_dia_siguiente"] == "sin_dato",
        "nivel_calidad_aire_dia_siguiente"
    ] = None

    n_con_target  = df["nivel_calidad_aire_dia_siguiente"].notna().sum()
    n_sin_target  = df["nivel_calidad_aire_dia_siguiente"].isna().sum()
    dist_target   = df["nivel_calidad_aire_dia_siguiente"].value_counts()

    logger.info(
        f"Target calculado — con valor: {n_con_target:,} | "
        f"sin valor (última fila): {n_sin_target:,}"
    )
    logger.info(f"Distribución del target: {dist_target.to_dict()}")

    return df


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def transform_sinca(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el DataFrame crudo de extract_sinca.py en el dataset
    limpio y enriquecido listo para PostgreSQL y el modelo predictivo.

    Aplica el pipeline completo de transformación en el orden definido
    en el módulo. Cada paso es una función independiente para facilitar
    el testing unitario con pytest.

    Args:
        df_raw: DataFrame producido por extract_sinca(). Debe tener las
                columnas: fecha, estacion, comuna, mp25_validado,
                mp25_preliminar, mp25_no_validado.

    Returns:
        pd.DataFrame con las columnas COLUMNAS_FINALES.
        Nunca retorna un DataFrame vacío: lanza ValueError si el resultado
        tiene 0 filas.

    Raises:
        ValueError: Si df_raw está vacío o si el DataFrame resultante
                    tiene 0 filas después de la limpieza.
        KeyError: Si faltan columnas requeridas en df_raw.
    """
    # Verificar que df_raw no esté vacío
    if df_raw.empty:
        raise ValueError("df_raw está vacío. Verifica que extract_sinca() retornó datos.")

    # Verificar columnas mínimas requeridas
    cols_requeridas = {"fecha", "estacion", "comuna", "mp25_validado", "mp25_preliminar"}
    faltantes = cols_requeridas - set(df_raw.columns)
    if faltantes:
        raise KeyError(
            f"Columnas requeridas faltantes en df_raw: {faltantes}\n"
            "Asegúrate de usar el DataFrame producido por extract_sinca()."
        )

    logger.info("=" * 55)
    logger.info("Iniciando pipeline de transformación SINCA")
    logger.info(f"Registros de entrada: {len(df_raw):,}")
    logger.info("=" * 55)

    # Pipeline de transformación
    df = (
        df_raw
        .pipe(_construir_mp25)
        .pipe(_limpiar_filas)
        .pipe(_agregar_nivel_calidad)
        .pipe(_agregar_columnas_temporales)
        .pipe(_calcular_mp25_dia_anterior)
        .pipe(_calcular_promedio_7d)
        .pipe(_calcular_target)
    )

    # Verificar que quedaron datos después de limpiar
    if df.empty:
        raise ValueError(
            "El DataFrame quedó vacío después de la limpieza. "
            "Revisa la calidad del archivo fuente."
        )

    # Seleccionar y ordenar columnas finales
    # (elimina mp25_validado, mp25_preliminar, mp25_no_validado del output)
    cols_disponibles = [c for c in COLUMNAS_FINALES if c in df.columns]
    df = df[cols_disponibles].reset_index(drop=True)

    # Resumen final
    logger.info("=" * 55)
    logger.info(f"Transformación completada")
    logger.info(f"Registros de salida  : {len(df):,}")
    logger.info(
        f"Rango de fechas      : "
        f"{df['fecha'].min().date()} → {df['fecha'].max().date()}"
    )
    logger.info(
        f"Filas con target     : "
        f"{df['nivel_calidad_aire_dia_siguiente'].notna().sum():,}"
    )
    logger.info("=" * 55)

    return df


def guardar_procesado(
    df: pd.DataFrame,
    ruta_salida: str | Path = "data/processed/sinca_transformado.csv",
) -> Path:
    """
    Guarda el DataFrame transformado como CSV en data/processed/.

    El archivo guardado sirve como checkpoint: si PostgreSQL no está
    disponible (ej: en etapas tempranas del desarrollo), el dashboard
    puede leer directamente desde aquí.

    Args:
        df: DataFrame transformado por transform_sinca()
        ruta_salida: Ruta donde guardar el CSV

    Returns:
        Path al archivo guardado
    """
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"Dataset procesado guardado en: {ruta} ({len(df):,} filas)")
    return ruta


# ---------------------------------------------------------------------------
# Ejecución directa (para pruebas rápidas)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # Importación relativa que funciona tanto al ejecutar directamente
    # como al importar desde otro módulo
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from etl.extract_sinca import extract_sinca

    ruta_default = os.getenv(
        "SINCA_RAW_PATH",
        "data/raw/sinca_puente_alto_mp25_2022_2026.csv"
    )

    logger.info("Ejecutando transform_sinca.py directamente (modo prueba)")

    try:
        # Extracción
        df_raw = extract_sinca(ruta_default)

        # Transformación
        df_limpio = transform_sinca(df_raw)

        # Guardar checkpoint
        ruta_guardado = guardar_procesado(df_limpio)

        # Mostrar resultado
        print("\n" + "=" * 60)
        print("  TRANSFORMACIÓN EXITOSA")
        print(f"  Registros     : {len(df_limpio):,}")
        print(f"  Fecha min     : {df_limpio['fecha'].min().date()}")
        print(f"  Fecha max     : {df_limpio['fecha'].max().date()}")
        print(f"  Con target    : {df_limpio['nivel_calidad_aire_dia_siguiente'].notna().sum():,}")
        print(f"  Guardado en   : {ruta_guardado}")
        print("=" * 60)

        print("\nColumnas del dataset final:")
        for col in df_limpio.columns:
            n_nulos = df_limpio[col].isna().sum()
            print(f"  {col:<45} {str(df_limpio[col].dtype):<12} nulos: {n_nulos}")

        print("\nPrimeras 5 filas:")
        print(df_limpio.head().to_string())

        print("\nDistribución del TARGET (nivel_calidad_aire_dia_siguiente):")
        print(df_limpio["nivel_calidad_aire_dia_siguiente"].value_counts())

        print("\nDistribución por MES (nivel_calidad_aire):")
        print(
            df_limpio.groupby("mes")["nivel_calidad_aire"]
            .value_counts()
            .unstack(fill_value=0)
        )

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise