"""
etl/profile_sinca.py
====================
AireChile Analytics — Explorador de estructura del archivo SINCA.

Este script NO transforma ni carga datos.
Su único propósito es leer el archivo descargado desde SINCA y generar
un reporte de su estructura real: columnas, tipos, nulos, fechas y
variables relevantes para el proyecto.

FORMATOS SINCA SOPORTADOS
--------------------------
El SINCA exporta datos en (al menos) dos formatos distintos según
la pantalla desde la que se descarga:

Formato A — "Export rápido por estación" (múltiples parámetros):
    Línea 1: "Estacion";CODIGO;NOMBRE
    Línea 2: "Fecha de generacion";YYYY-MM-DD HH:MM
    Línea 3: (vacía)
    Línea 4: "Fecha y hora";"MP 10";"MP 2,5";"SO2";"NO2";"CO";"O3"
    Líneas 5+: datos horarios
    Separador: ; | Decimales: , | Fecha y hora en una sola columna

Formato B — "Export histórico por parámetro" (un parámetro a la vez):
    Línea 1: FECHA (YYMMDD);HORA (HHMM);Registros validados;...
    Líneas 2+: datos horarios con fecha YYMMDD y hora HHMM separadas
    Separador: ; | Decimales: , | Sin metadatos iniciales

Este script detecta automáticamente cuál formato tiene el archivo.

Ejecutar desde la raíz del proyecto:
    python etl/profile_sinca.py

El reporte se guarda en: docs/reporte_columnas_sinca.md
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

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
# Variables de entorno
# ---------------------------------------------------------------------------
load_dotenv()

SINCA_RAW_PATH = os.getenv("SINCA_RAW_PATH", "data/raw/sinca_santiago.csv")
DOCS_PATH      = Path("docs")
REPORT_PATH    = DOCS_PATH / "reporte_columnas_sinca.md"

# Palabras clave que identifican líneas de metadatos del Formato A
_META_KEYWORDS = ("estacion", "estación", "fecha de generacion", "fecha de generación")


# ---------------------------------------------------------------------------
# Detección de formato
# ---------------------------------------------------------------------------

def detectar_encoding(ruta: Path) -> str:
    """
    Detecta el encoding del archivo probando los más comunes.

    Args:
        ruta: Path al archivo

    Returns:
        Nombre del encoding ('utf-8', 'latin-1', etc.)
    """
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(ruta, encoding=enc) as f:
                f.read(4096)
            logger.info(f"Encoding detectado: {enc}")
            return enc
        except UnicodeDecodeError:
            continue
    logger.warning("Encoding no detectado, usando latin-1 por defecto")
    return "latin-1"


def detectar_formato_sinca(ruta: Path, encoding: str) -> dict:
    """
    Lee las primeras líneas del archivo para determinar cuál de los
    dos formatos SINCA tiene.

    Formato A: la primera línea contiene metadatos de estación
               (empieza con "Estacion" o "estacion").
    Formato B: la primera línea ES el encabezado de datos
               (contiene "FECHA", "HORA" o "fecha y hora").

    Args:
        ruta: Path al archivo CSV
        encoding: Encoding detectado

    Returns:
        dict con claves:
            'formato'   → 'A' o 'B'
            'skiprows'  → cuántas líneas saltar antes del encabezado
            'metadatos' → dict con info de estación/fecha (solo Formato A)
    """
    with open(ruta, encoding=encoding, errors="replace") as f:
        primeras = [f.readline() for _ in range(6)]

    resultado = {"formato": None, "skiprows": 0, "metadatos": {}}

    # Revisar si la primera línea es un metadato (Formato A)
    primera_lower = primeras[0].lower().strip().strip('"')
    if any(primera_lower.startswith(kw) for kw in _META_KEYWORDS):
        resultado["formato"] = "A"

        # Extraer metadatos de las primeras líneas
        meta = {}
        skiprows = 0
        for linea in primeras:
            linea_s = linea.strip()
            if not linea_s:
                skiprows += 1
                continue
            partes = linea_s.split(";")
            clave  = partes[0].strip().strip('"').lower()
            valor  = ";".join(partes[1:]).strip().strip('"') if len(partes) > 1 else ""

            if any(clave.startswith(kw) for kw in _META_KEYWORDS):
                if "estacion" in clave or "estación" in clave:
                    meta["estacion"] = valor
                elif "fecha" in clave:
                    meta["fecha_generacion"] = valor
                skiprows += 1
            else:
                # Ya llegamos al encabezado real
                break

        resultado["skiprows"]  = skiprows
        resultado["metadatos"] = meta
        logger.info(
            f"Formato A detectado — {skiprows} líneas de metadatos. "
            f"Estación: {meta.get('estacion', 'no detectada')}"
        )

    else:
        # La primera línea ya ES el encabezado de datos (Formato B)
        resultado["formato"]  = "B"
        resultado["skiprows"] = 0
        logger.info("Formato B detectado — sin metadatos iniciales, encabezado en línea 1")

    return resultado


# ---------------------------------------------------------------------------
# Carga del archivo
# ---------------------------------------------------------------------------

def cargar_csv_sinca(ruta: Path, encoding: str, skiprows: int) -> pd.DataFrame:
    """
    Carga el CSV del SINCA.

    No usa decimal="," porque el nombre de columna "MP 2,5" contiene
    una coma y confundiría al parser. Las comas decimales se convierten
    a puntos después de leer, solo en columnas numéricas.

    Args:
        ruta: Path al CSV
        encoding: Encoding del archivo
        skiprows: Líneas a saltar antes del encabezado real

    Returns:
        pd.DataFrame limpio
    """
    df = pd.read_csv(
        ruta,
        sep=";",
        encoding=encoding,
        skiprows=skiprows,
        quotechar='"',
        skipinitialspace=True,
        engine="python",
    )

    # Limpiar nombres de columnas
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]

    # Eliminar columnas sin nombre (artefactos del export SINCA)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Eliminar filas completamente vacías
    df = df.dropna(how="all").reset_index(drop=True)

    # Detectar columna de fecha para no convertirla
    cols_excluir = {
        c for c in df.columns
        if any(kw in c.lower() for kw in ["fecha", "date", "hora", "time"])
    }

    # Convertir comas decimales a puntos en columnas numéricas
    for col in df.columns:
        if col in cols_excluir:
            continue
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def cargar_archivo(ruta: Path) -> tuple[pd.DataFrame, dict]:
    """
    Carga un archivo SINCA (CSV, XLSX o XLS).
    Detecta automáticamente el formato y retorna el DataFrame
    junto con la información del formato detectado.

    Args:
        ruta: Path al archivo

    Returns:
        Tupla (DataFrame, dict_info_formato)

    Raises:
        ValueError: Si la extensión no es soportada
        Exception: Si falla la lectura
    """
    ext = ruta.suffix.lower()
    logger.info(f"Cargando: {ruta.name} (formato: {ext})")

    try:
        if ext == ".csv":
            encoding   = detectar_encoding(ruta)
            info_fmt   = detectar_formato_sinca(ruta, encoding)
            df         = cargar_csv_sinca(ruta, encoding, info_fmt["skiprows"])

        elif ext in (".xlsx", ".xls"):
            engine   = "openpyxl" if ext == ".xlsx" else "xlrd"
            encoding = "utf-8"
            # Para Excel siempre intentamos primero sin skiprows y luego con 3
            df_raw = pd.read_excel(ruta, engine=engine)
            primera = str(df_raw.columns[0]).lower()
            skiprows = 3 if any(kw in primera for kw in _META_KEYWORDS) else 0
            info_fmt = {
                "formato":   "A" if skiprows > 0 else "B",
                "skiprows":  skiprows,
                "metadatos": {},
            }
            if skiprows > 0:
                df_raw = pd.read_excel(ruta, engine=engine, skiprows=skiprows)
            df = df_raw.copy()
            df.columns = [str(c).strip() for c in df.columns]
            df = df.dropna(how="all").reset_index(drop=True)

        else:
            raise ValueError(f"Extensión '{ext}' no soportada. Usa .csv, .xlsx o .xls.")

        logger.info(f"Cargado: {len(df):,} filas × {df.shape[1]} columnas")
        return df, info_fmt

    except Exception as e:
        logger.error(f"Error al cargar el archivo: {e}")
        raise


# ---------------------------------------------------------------------------
# Búsqueda de archivo
# ---------------------------------------------------------------------------

def encontrar_archivo_sinca(ruta_env: str) -> Path:
    """
    Devuelve la ruta al archivo SINCA.
    Primero usa la ruta de .env; si no existe, busca en data/raw/.

    Args:
        ruta_env: Valor de SINCA_RAW_PATH en el .env

    Returns:
        Path al archivo encontrado

    Raises:
        FileNotFoundError: Si no hay ningún archivo válido
    """
    ruta = Path(ruta_env)
    if ruta.exists():
        logger.info(f"Archivo encontrado: {ruta}")
        return ruta

    logger.warning(f"'{ruta_env}' no existe. Buscando en data/raw/ ...")
    carpeta   = Path("data/raw")
    candidatos = []
    for ext in [".csv", ".xlsx", ".xls"]:
        candidatos.extend(carpeta.glob(f"*{ext}"))

    if not candidatos:
        raise FileNotFoundError(
            f"No hay archivos .csv/.xlsx/.xls en '{carpeta}'.\n"
            "Descarga datos desde https://sinca.mma.gob.cl/ y "
            f"colócalos en '{carpeta}/'."
        )

    if len(candidatos) > 1:
        logger.warning(
            f"Múltiples archivos encontrados: {[str(c) for c in candidatos]}\n"
            f"Usando el primero: {candidatos[0]}\n"
            "Para elegir otro, define SINCA_RAW_PATH en tu .env"
        )
    return candidatos[0]


# ---------------------------------------------------------------------------
# Análisis de columnas y fechas
# ---------------------------------------------------------------------------

def detectar_columnas_relevantes(df: pd.DataFrame) -> dict:
    """
    Detecta columnas relevantes para el proyecto buscando palabras clave.

    Args:
        df: DataFrame cargado

    Returns:
        dict con listas de columnas por categoría
    """
    cols_lower = [c.lower() for c in df.columns]
    cols_orig  = list(df.columns)

    def buscar(keywords: list) -> list:
        return [
            cols_orig[i]
            for i, col in enumerate(cols_lower)
            if any(kw in col for kw in keywords)
        ]

    return {
        "fecha":            buscar(["fecha", "date", "hora", "time", "yymmdd", "hhmm"]),
        "mp25":             buscar(["mp 2", "mp2", "pm2", "pm 2", "2,5", "2.5"]),
        "mp10":             buscar(["mp 10", "mp10", "pm10", "pm 10"]),
        "estado_registro":  buscar(["validado", "preliminar", "no valid"]),
        "estacion":         buscar(["estacion", "estación", "station"]),
        "comuna":           buscar(["comuna", "municipio", "localidad"]),
        "otros_gases":      buscar(["so2", "no2", "co", "o3"]),
    }


def analizar_fechas(df: pd.DataFrame, cols_fecha: list) -> dict:
    """
    Para cada columna de fecha detectada, obtiene el rango disponible.

    Args:
        df: DataFrame
        cols_fecha: Columnas candidatas a fecha

    Returns:
        dict con info de rango por columna
    """
    info = {}
    for col in cols_fecha:
        try:
            # Formato YYMMDD (ej: 260101 → 2026-01-01)
            if df[col].dtype in ["int64", "float64"]:
                serie = pd.to_datetime(
                    df[col].astype(str).str.zfill(6),
                    format="%y%m%d",
                    errors="coerce",
                )
            else:
                serie = pd.to_datetime(df[col], errors="coerce")

            n_ok = int(serie.notna().sum())
            if n_ok > 0:
                info[col] = {
                    "n_validas":   n_ok,
                    "n_invalidas": int(serie.isna().sum()),
                    "fecha_min":   str(serie.min().date()),
                    "fecha_max":   str(serie.max().date()),
                }
            else:
                info[col] = {"error": "No parseable como fecha"}
        except Exception as e:
            info[col] = {"error": str(e)}
    return info


# ---------------------------------------------------------------------------
# Generación del reporte
# ---------------------------------------------------------------------------

def generar_reporte(
    df: pd.DataFrame,
    ruta: Path,
    info_fmt: dict,
    cols_relevantes: dict,
    info_fechas: dict,
) -> str:
    """
    Genera el reporte Markdown completo de la exploración.

    Args:
        df: DataFrame cargado
        ruta: Path al archivo original
        info_fmt: Dict con info del formato detectado
        cols_relevantes: Dict de columnas por categoría
        info_fechas: Dict con rangos de fechas

    Returns:
        str con el contenido Markdown del reporte
    """
    ahora  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nulos  = df.isnull().sum()
    pct_nu = (nulos / len(df) * 100).round(2) if len(df) > 0 else nulos * 0

    meta = info_fmt.get("metadatos", {})

    L = [
        "# Reporte de exploración — Archivo SINCA",
        "",
        f"**Generado:** {ahora}",
        f"**Archivo:** `{ruta.name}`",
        f"**Formato SINCA detectado:** {info_fmt.get('formato', '?')}",
        "",
    ]

    if meta:
        L += ["## Metadatos del archivo", ""]
        for k, v in meta.items():
            L.append(f"- **{k}:** {v}")
        L.append("")

    L += [
        "---", "",
        "## 1. Resumen general", "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Filas totales | {len(df):,} |",
        f"| Columnas totales | {df.shape[1]} |",
        f"| Filas duplicadas | {int(df.duplicated().sum()):,} |",
        f"| Columnas con algún nulo | {int((nulos > 0).sum())} |",
        f"| Granularidad | Horaria (requiere agregación diaria) |",
        "",
        "---", "",
        "## 2. Columnas disponibles", "",
        "| Columna | Tipo pandas | Nulos | % Nulos | Ejemplo |",
        "|---|---|---|---|---|",
    ]

    for col in df.columns:
        tipo = str(df[col].dtype)
        n_nu = int(nulos[col])
        pct  = float(pct_nu[col])
        ej   = df[col].dropna().iloc[0] if df[col].notna().any() else "—"
        ej_s = str(ej)[:35].replace("|", "\\|")
        L.append(f"| `{col}` | {tipo} | {n_nu:,} | {pct}% | {ej_s} |")

    L += ["", "---", "", "## 3. Columnas relevantes para el proyecto", ""]

    etiquetas = {
        "fecha":           "Fecha / hora",
        "mp25":            "MP 2,5 — material particulado fino",
        "mp10":            "MP 10 — material particulado grueso",
        "estado_registro": "Estado de validación del registro",
        "estacion":        "Estación de monitoreo",
        "comuna":          "Comuna / localidad",
        "otros_gases":     "Otros gases (SO₂, NO₂, CO, O₃)",
    }

    for cat, label in etiquetas.items():
        cols = cols_relevantes.get(cat, [])
        L.append(f"### {label}")
        L += ([f"- `{c}`" for c in cols] if cols else ["_No detectada._"])
        L.append("")

    L += ["---", "", "## 4. Rango de fechas disponibles", ""]

    if info_fechas:
        for col, info in info_fechas.items():
            L.append(f"### `{col}`")
            if "error" in info:
                L.append(f"- ⚠️ {info['error']}")
            else:
                L += [
                    f"- Registros válidos: **{info['n_validas']:,}**",
                    f"- Desde: **{info['fecha_min']}**",
                    f"- Hasta: **{info['fecha_max']}**",
                ]
            L.append("")
    else:
        L += ["_No se detectaron columnas de fecha._", ""]

    L += [
        "---", "",
        "## 5. Estadísticas descriptivas (columnas numéricas)", "",
        "```",
        df.describe().round(2).to_string(),
        "```",
        "",
        "---", "",
        "## 6. Primeras 5 filas", "",
        "```",
        df.head(5).to_string(),
        "```",
        "",
        "---", "",
        "## 7. Notas importantes para el ETL", "",
        f"### Formato detectado: {info_fmt.get('formato')}",
    ]

    if info_fmt.get("formato") == "A":
        L += [
            "- Una sola columna de fecha+hora: `Fecha y hora`",
            "- Múltiples parámetros en columnas separadas (MP 10, MP 2,5, SO2, etc.)",
            "- El nombre `MP 2,5` contiene una coma → se maneja con `engine='python'`",
        ]
    elif info_fmt.get("formato") == "B":
        L += [
            "- Fecha y hora en columnas **separadas**: `FECHA (YYMMDD)` y `HORA (HHMM)`",
            "- Formato de fecha: YYMMDD (ej: 260101 = 2026-01-01)",
            "- Formato de hora: HHMM (ej: 0000 = 00:00)",
            "- Los valores de MP 2,5 están en 3 columnas según validación:",
            "  - `Registros validados` — usar estos preferentemente",
            "  - `Registros preliminares` — usar si validados es nulo",
            "  - `Registros no validados` — evitar salvo que no haya alternativa",
            "- El ETL deberá combinar las 3 columnas en una sola `mp25`",
        ]

    L += [
        "",
        "### Tareas antes de escribir extract_sinca.py",
        "- [ ] Descargar datos históricos 2022–2025 desde SINCA",
        "- [ ] Verificar si todos los archivos históricos tienen el mismo formato",
        "- [ ] Confirmar función de agregación diaria (promedio vs máximo)",
        "- [ ] Decidir prioridad entre registros validados / preliminares / no validados",
    ]

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def main():
    """Función principal del explorador de datos SINCA."""
    logger.info("=" * 60)
    logger.info("AireChile Analytics — Explorador de datos SINCA")
    logger.info("=" * 60)

    try:
        ruta = encontrar_archivo_sinca(SINCA_RAW_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        df, info_fmt = cargar_archivo(ruta)
    except Exception as e:
        logger.error(f"No se pudo cargar: {e}")
        sys.exit(1)

    meta = info_fmt.get("metadatos", {})

    # Resumen en consola
    print("\n" + "=" * 62)
    print(f"  FORMATO   : SINCA Formato {info_fmt.get('formato')}")
    if meta.get("estacion"):
        print(f"  ESTACIÓN  : {meta['estacion']}")
    print(f"  ARCHIVO   : {ruta.name}")
    print(f"  TAMAÑO    : {len(df):,} filas × {df.shape[1]} columnas")
    print("=" * 62)

    nulos = df.isnull().sum()
    print("\n┌─ COLUMNAS ──────────────────────────────────────────────────┐")
    for col in df.columns:
        n   = int(nulos[col])
        pct = round(n / len(df) * 100, 1) if len(df) > 0 else 0
        print(f"│  {col:<35} {str(df[col].dtype):<10}  nulos: {n:>4} ({pct}%)")
    print("└─────────────────────────────────────────────────────────────┘")

    cols_relevantes = detectar_columnas_relevantes(df)
    print("\n┌─ COLUMNAS RELEVANTES ────────────────────────────────────────┐")
    for cat, cols in cols_relevantes.items():
        estado = str(cols) if cols else "(no detectada)"
        print(f"│  {cat.upper():<22}  {estado}")
    print("└─────────────────────────────────────────────────────────────┘")

    info_fechas = {}
    if cols_relevantes["fecha"]:
        info_fechas = analizar_fechas(df, cols_relevantes["fecha"])
        print("\n┌─ RANGO DE FECHAS ────────────────────────────────────────────┐")
        for col, info in info_fechas.items():
            if "error" not in info:
                print(f"│  {col}: {info['fecha_min']} → {info['fecha_max']}")
                print(f"│  ({info['n_validas']:,} registros válidos)")
            else:
                print(f"│  {col}: ERROR — {info['error']}")
        print("└──────────────────────────────────────────────────────────────┘")

    # Nota especial para Formato B
    if info_fmt.get("formato") == "B":
        print("\n┌─ NOTA FORMATO B ─────────────────────────────────────────────┐")
        print("│  Los valores MP 2,5 están en 3 columnas según validación:     ")
        print("│  · Registros validados     → usar primero                     ")
        print("│  · Registros preliminares  → usar si validados es nulo        ")
        print("│  · Registros no validados  → evitar                           ")
        print("│  El ETL combinará estas 3 en una sola columna mp25.           ")
        print("└──────────────────────────────────────────────────────────────┘")

    print("\n⚠️  Datos HORARIOS — el ETL agregará a nivel DIARIO.")

    DOCS_PATH.mkdir(exist_ok=True)
    reporte = generar_reporte(df, ruta, info_fmt, cols_relevantes, info_fechas)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(reporte)

    logger.info(f"Reporte guardado en: {REPORT_PATH}")
    print(f"\n✓ Reporte guardado en: {REPORT_PATH}\n")


if __name__ == "__main__":
    main()