"""
etl/extract_sinca.py
====================
AireChile Analytics — Extractor de datos SINCA (Formato B).

Responsabilidad única: leer uno o varios archivos CSV del SINCA en
Formato B y devolver un DataFrame crudo pero tipado, sin ninguna
transformación de negocio.

Formato B (verificado con datos reales de Puente Alto 2022-2026):
    - Sin metadatos iniciales; el encabezado está en la línea 1.
    - Separador: punto y coma (;)
    - Decimales: coma (,)
    - Columnas: FECHA (YYMMDD) | HORA (HHMM) | Registros validados |
                Registros preliminares | Registros no validados
    - Fecha en formato YYMMDD (ej: 220101 → 2022-01-01)
    - Datos diarios: HORA siempre 0000 (no se requiere agrupación horaria)
    - El nombre de la estación y la comuna NO vienen en el archivo;
      se infieren del nombre del archivo o se pasan como parámetros.

Salida de extract_sinca():
    DataFrame con columnas estandarizadas:
        fecha                 → datetime64[ns]
        estacion              → str
        comuna                → str
        mp25_validado         → float64  (Registros validados)
        mp25_preliminar       → float64  (Registros preliminares)
        mp25_no_validado      → float64  (Registros no validados — para trazabilidad)

    La columna mp25 definitiva se construye en transform_sinca.py,
    no aquí. La extracción no toma decisiones de negocio.

Uso:
    from etl.extract_sinca import extract_sinca

    df = extract_sinca("data/raw/sinca_puente_alto_mp25_2022_2026.csv")
    # o con parámetros explícitos:
    df = extract_sinca(
        "data/raw/sinca_puente_alto_mp25_2022_2026.csv",
        estacion="Puente Alto",
        comuna="Puente Alto",
    )

También puede ejecutarse directamente:
    python etl/extract_sinca.py
"""

import logging
import re
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

# Nombres exactos de las columnas en el CSV del SINCA Formato B
# (verificados con archivos reales de Puente Alto 2022-2026)
COL_FECHA       = "FECHA (YYMMDD)"
COL_HORA        = "HORA (HHMM)"
COL_VALIDADO    = "Registros validados"
COL_PRELIMINAR  = "Registros preliminares"
COL_NO_VALID    = "Registros no validados"

COLUMNAS_ESPERADAS = {COL_FECHA, COL_HORA, COL_VALIDADO, COL_PRELIMINAR}


# ---------------------------------------------------------------------------
# Funciones internas
# ---------------------------------------------------------------------------

def _detectar_encoding(ruta: Path) -> str:
    """
    Detecta el encoding del archivo probando los más comunes.

    Args:
        ruta: Path al archivo

    Returns:
        Nombre del encoding detectado
    """
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(ruta, encoding=enc) as f:
                f.read(4096)
            return enc
        except UnicodeDecodeError:
            continue
    logger.warning("Encoding no detectado, usando latin-1")
    return "latin-1"


def _inferir_estacion_comuna(ruta: Path) -> tuple[str, str]:
    """
    Infiere el nombre de la estación y la comuna a partir del nombre
    del archivo. Esta inferencia es necesaria porque el Formato B del
    SINCA no incluye la estación como columna.

    Convención de nombres esperada (flexible):
        sinca_<estacion>_mp25_<años>.csv
        Ejemplo: sinca_puente_alto_mp25_2022_2026.csv → "Puente Alto"

    Si el nombre del archivo no sigue la convención, retorna valores
    genéricos que el usuario puede sobreescribir con los parámetros
    explícitos de extract_sinca().

    Args:
        ruta: Path al archivo CSV

    Returns:
        Tupla (estacion, comuna) como strings con formato título
    """
    nombre = ruta.stem.lower()  # sin extensión, ej: "sinca_puente_alto_mp25_2022_2026"

    # Intentar extraer el nombre entre "sinca_" y "_mp"
    match = re.search(r"sinca_(.+?)_mp", nombre)
    if match:
        nombre_raw = match.group(1).replace("_", " ").title()
        logger.info(f"Estación inferida del nombre de archivo: '{nombre_raw}'")
        return nombre_raw, nombre_raw

    # Si no matchea la convención, usar el nombre completo del archivo
    logger.warning(
        f"No se pudo inferir la estación de '{ruta.name}'. "
        "Usando 'Desconocida'. Pasa estacion= y comuna= como parámetros."
    )
    return "Desconocida", "Desconocida"


def _validar_columnas(df: pd.DataFrame, ruta: Path) -> None:
    """
    Verifica que el DataFrame tenga las columnas mínimas esperadas
    del Formato B. Lanza un error descriptivo si falta alguna.

    Args:
        df: DataFrame recién cargado
        ruta: Path al archivo (para mensajes de error)

    Raises:
        ValueError: Si faltan columnas requeridas
    """
    presentes   = set(df.columns)
    faltantes   = COLUMNAS_ESPERADAS - presentes

    if faltantes:
        raise ValueError(
            f"El archivo '{ruta.name}' no tiene las columnas esperadas del "
            f"Formato B.\nColumnas faltantes: {sorted(faltantes)}\n"
            f"Columnas presentes: {sorted(presentes)}\n"
            "Verifica que sea un export histórico por parámetro (Formato B) "
            "y no un export rápido por estación (Formato A)."
        )

    # Advertir si falta la columna de no-validados (no es crítica)
    if COL_NO_VALID not in presentes:
        logger.warning(
            f"Columna '{COL_NO_VALID}' no encontrada en '{ruta.name}'. "
            "Se continuará sin ella."
        )


def _parsear_fecha(serie: pd.Series) -> pd.Series:
    """
    Convierte la columna FECHA del Formato B (YYMMDD como entero)
    a datetime64.

    Ejemplos de conversión:
        220101  → 2022-01-01
        260613  → 2026-06-13
        251231  → 2025-12-31

    Args:
        serie: Serie con valores enteros en formato YYMMDD

    Returns:
        Serie de tipo datetime64[ns]
    """
    # Convertir a string con cero a la izquierda si tiene 5 dígitos
    # (puede pasar con fechas como 220101 → "220101", pero también
    #  podría venir como 22101 si enero tiene día 1 sin cero)
    fechas_str = serie.astype(str).str.zfill(6)
    fechas_dt  = pd.to_datetime(fechas_str, format="%y%m%d", errors="coerce")

    n_nulos = fechas_dt.isna().sum()
    if n_nulos > 0:
        logger.warning(
            f"Se encontraron {n_nulos} fechas no parseables en FECHA (YYMMDD). "
            "Esas filas quedarán con NaT y serán eliminadas en transform_sinca.py."
        )
    return fechas_dt


def _cargar_csv_formato_b(ruta: Path, encoding: str) -> pd.DataFrame:
    """
    Carga el CSV del SINCA Formato B con la configuración exacta
    necesaria para este formato.

    No usa decimal="," porque el nombre de columna "MP 2,5" (presente
    en Formato A) podría confundir al parser si se mezclan archivos.
    Las comas decimales se convierten a puntos después de cargar.

    Args:
        ruta: Path al CSV
        encoding: Encoding detectado

    Returns:
        pd.DataFrame crudo con columnas renombradas
    """
    df = pd.read_csv(
        ruta,
        sep=";",
        encoding=encoding,
        quotechar='"',
        skipinitialspace=True,
        engine="python",   # más tolerante a irregularidades del CSV
    )

    # Limpiar nombres de columnas: quitar espacios y comillas
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]

    # Eliminar columnas sin nombre (artefacto del export SINCA:
    # el CSV termina con ";" lo que crea una columna "Unnamed")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Eliminar filas completamente vacías
    df = df.dropna(how="all").reset_index(drop=True)

    # Convertir comas decimales a puntos en columnas numéricas
    cols_numericas = [
        c for c in df.columns
        if c not in (COL_FECHA, COL_HORA)
    ]
    for col in cols_numericas:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def extract_sinca(
    ruta_archivo: str | Path,
    estacion: str | None = None,
    comuna: str | None   = None,
) -> pd.DataFrame:
    """
    Lee un archivo CSV del SINCA en Formato B y devuelve un DataFrame
    crudo pero con tipos correctos, listo para ser procesado por
    transform_sinca.py.

    No aplica ninguna lógica de negocio: no clasifica calidad del aire,
    no calcula promedios móviles, no elimina outliers.

    Args:
        ruta_archivo: Ruta al archivo CSV del SINCA (str o Path).
        estacion: Nombre de la estación de monitoreo. Si no se pasa,
                  se infiere del nombre del archivo.
        comuna: Nombre de la comuna. Si no se pasa, se usa el mismo
                valor que estacion.

    Returns:
        pd.DataFrame con columnas:
            fecha            (datetime64[ns]) — fecha del registro
            estacion         (str)            — nombre de la estación
            comuna           (str)            — nombre de la comuna
            mp25_validado    (float64)        — µg/m³, puede tener NaN
            mp25_preliminar  (float64)        — µg/m³, puede tener NaN
            mp25_no_validado (float64)        — µg/m³, casi siempre NaN

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no tiene las columnas del Formato B.
        Exception: Para cualquier otro error de lectura.
    """
    ruta = Path(ruta_archivo)

    # 1. Verificar que el archivo existe
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: '{ruta}'\n"
            "Descarga los datos desde https://sinca.mma.gob.cl/ "
            "y guárdalos en data/raw/."
        )

    logger.info(f"Extrayendo datos SINCA desde: {ruta.name}")

    # 2. Detectar encoding
    encoding = _detectar_encoding(ruta)

    # 3. Cargar el CSV
    try:
        df_raw = _cargar_csv_formato_b(ruta, encoding)
    except Exception as e:
        logger.error(f"Error al leer '{ruta.name}': {e}")
        raise

    logger.info(f"Archivo cargado: {len(df_raw):,} filas × {df_raw.shape[1]} columnas")

    # 4. Validar que tenga las columnas esperadas
    _validar_columnas(df_raw, ruta)

    # 5. Inferir estación y comuna si no se pasaron explícitamente
    est_inferida, com_inferida = _inferir_estacion_comuna(ruta)
    estacion_final = estacion if estacion else est_inferida
    comuna_final   = comuna   if comuna   else com_inferida

    # 6. Parsear fecha YYMMDD → datetime
    fechas = _parsear_fecha(df_raw[COL_FECHA])

    # 7. Construir DataFrame de salida con nombres estandarizados
    df_out = pd.DataFrame({
        "fecha":            fechas,
        "estacion":         estacion_final,
        "comuna":           comuna_final,
        "mp25_validado":    df_raw[COL_VALIDADO],
        "mp25_preliminar":  df_raw[COL_PRELIMINAR],
        "mp25_no_validado": df_raw.get(COL_NO_VALID, pd.Series([None] * len(df_raw))),
    })

    # 8. Resumen de extracción
    n_total      = len(df_out)
    n_validados  = df_out["mp25_validado"].notna().sum()
    n_prelim     = df_out["mp25_preliminar"].notna().sum()
    n_sin_valor  = df_out[["mp25_validado", "mp25_preliminar"]].isna().all(axis=1).sum()

    logger.info(
        f"Extracción completada — {n_total:,} registros | "
        f"validados: {n_validados:,} | "
        f"preliminares: {n_prelim:,} | "
        f"sin valor en ambas columnas: {n_sin_valor:,}"
    )
    logger.info(
        f"Rango de fechas: {df_out['fecha'].min().date()} "
        f"→ {df_out['fecha'].max().date()}"
    )
    logger.info(f"Estación: {estacion_final} | Comuna: {comuna_final}")

    return df_out


def extract_multiples_archivos(
    rutas: list[str | Path],
    estacion: str | None = None,
    comuna: str | None   = None,
) -> pd.DataFrame:
    """
    Extrae y concatena datos de múltiples archivos SINCA del mismo
    parámetro y estación.

    Útil cuando los datos históricos están divididos en varios archivos
    anuales (ej: un CSV por año).

    Args:
        rutas: Lista de rutas a archivos CSV del SINCA.
        estacion: Nombre de la estación (se aplica a todos los archivos).
        comuna: Nombre de la comuna (se aplica a todos los archivos).

    Returns:
        pd.DataFrame concatenado, sin duplicados, ordenado por fecha.

    Raises:
        ValueError: Si ningún archivo pudo cargarse correctamente.
    """
    dfs = []
    errores = []

    for ruta in rutas:
        try:
            df = extract_sinca(ruta, estacion=estacion, comuna=comuna)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Saltando '{ruta}': {e}")
            errores.append(str(ruta))

    if not dfs:
        raise ValueError(
            f"No se pudo cargar ningún archivo. Errores en: {errores}"
        )

    if errores:
        logger.warning(f"Archivos con error (ignorados): {errores}")

    df_concat = pd.concat(dfs, ignore_index=True)

    # Eliminar duplicados por fecha+estacion (puede haber solapamiento entre archivos)
    antes = len(df_concat)
    df_concat = df_concat.drop_duplicates(
        subset=["fecha", "estacion"]
    ).sort_values("fecha").reset_index(drop=True)
    eliminados = antes - len(df_concat)

    if eliminados > 0:
        logger.info(f"Duplicados eliminados al concatenar archivos: {eliminados:,}")

    logger.info(
        f"Dataset combinado: {len(df_concat):,} registros | "
        f"{df_concat['fecha'].min().date()} → {df_concat['fecha'].max().date()}"
    )
    return df_concat


# ---------------------------------------------------------------------------
# Ejecución directa (para pruebas rápidas)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    ruta_default = os.getenv(
        "SINCA_RAW_PATH",
        "data/raw/sinca_puente_alto_mp25_2022_2026.csv"
    )

    logger.info("Ejecutando extract_sinca.py directamente (modo prueba)")

    try:
        df = extract_sinca(ruta_default)
        print("\n" + "=" * 55)
        print(f"  EXTRACCIÓN EXITOSA")
        print(f"  Registros  : {len(df):,}")
        print(f"  Fecha min  : {df['fecha'].min().date()}")
        print(f"  Fecha max  : {df['fecha'].max().date()}")
        print(f"  Estación   : {df['estacion'].iloc[0]}")
        print("=" * 55)
        print("\nPrimeras 5 filas:")
        print(df.head().to_string())
        print("\nInfo de columnas:")
        print(df.dtypes)
        print("\nValores nulos por columna:")
        print(df.isnull().sum())
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)