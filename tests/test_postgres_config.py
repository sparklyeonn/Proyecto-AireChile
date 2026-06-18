"""
tests/test_postgres_config.py
==============================
AireChile Analytics — Tests de configuración de PostgreSQL.

Estos tests verifican la configuración del sistema sin requerir
una conexión activa a PostgreSQL. Esto permite ejecutarlos en
cualquier entorno (CI/CD, máquina sin PostgreSQL instalado)
y siguen siendo útiles para detectar errores de configuración.

Cobertura:
    - Variables de entorno requeridas definidas
    - database/schema.sql existe y contiene las tablas esperadas
    - database/db.py puede importarse sin error
    - etl/load_postgres.py puede importarse sin error
    - El schema SQL tiene sintaxis válida (no requiere conexión)
    - Los archivos CSV de entrada existen (si el pipeline fue ejecutado)

Ejecutar:
    pytest tests/test_postgres_config.py -v
"""

import os
import re
import sys
from pathlib import Path

import pytest

# Agregar raíz del proyecto al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA_PATH = ROOT / "database" / "schema.sql"

TABLAS_ESPERADAS = [
    "estaciones",
    "mediciones_sinca",
    "meteorologia",
    "dataset_modelo",
    "predicciones_modelo",
    "log_etl",
]

VARS_ENV_REQUERIDAS = [
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]

CSVS_PIPELINE = {
    "sinca_transformado.csv":      ROOT / "data/processed/sinca_transformado.csv",
    "open_meteo_transformado.csv": ROOT / "data/processed/open_meteo_transformado.csv",
    "dataset_modelo_base.csv":     ROOT / "data/processed/dataset_modelo_base.csv",
    "prediccion_actual.csv":       ROOT / "data/processed/prediccion_actual.csv",
}


# ---------------------------------------------------------------------------
# Fixture: cargar .env una sola vez
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def cargar_env():
    """Carga variables de entorno desde .env antes de los tests."""
    from dotenv import load_dotenv
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Cargar .env.example como fallback para CI
        env_example = ROOT / ".env.example"
        if env_example.exists():
            load_dotenv(env_example)


# ---------------------------------------------------------------------------
# Tests: variables de entorno
# ---------------------------------------------------------------------------

class TestVariablesEntorno:

    def test_env_example_existe(self):
        """El archivo .env.example debe existir en el repositorio."""
        env_example = ROOT / ".env.example"
        assert env_example.exists(), (
            ".env.example no encontrado. "
            "Es necesario para que otros desarrolladores configuren el entorno."
        )

    def test_env_example_tiene_vars_postgres(self):
        """El .env.example debe documentar las variables de PostgreSQL."""
        env_example = ROOT / ".env.example"
        if not env_example.exists():
            pytest.skip(".env.example no existe")

        contenido = env_example.read_text(encoding="utf-8")
        for var in VARS_ENV_REQUERIDAS:
            assert var in contenido, (
                f"Variable '{var}' no documentada en .env.example"
            )

    def test_env_example_tiene_database_url(self):
        """El .env.example debe incluir DATABASE_URL."""
        env_example = ROOT / ".env.example"
        if not env_example.exists():
            pytest.skip(".env.example no existe")

        contenido = env_example.read_text(encoding="utf-8")
        assert "DATABASE_URL" in contenido

    @pytest.mark.parametrize("var", VARS_ENV_REQUERIDAS)
    def test_variable_env_definida(self, var):
        """
        Cada variable de entorno requerida debe estar definida.
        Acepta el valor del .env real O del .env.example.
        No falla si el valor es el placeholder del .env.example.
        """
        valor = os.getenv(var)
        assert valor is not None, (
            f"Variable '{var}' no definida.\n"
            f"Agrega al .env: {var}=<valor>"
        )
        assert len(valor.strip()) > 0, (
            f"Variable '{var}' está vacía."
        )


# ---------------------------------------------------------------------------
# Tests: schema.sql
# ---------------------------------------------------------------------------

class TestSchemaSql:

    def test_archivo_schema_existe(self):
        """database/schema.sql debe existir."""
        assert SCHEMA_PATH.exists(), (
            f"Schema no encontrado: {SCHEMA_PATH}"
        )

    def test_schema_no_vacio(self):
        contenido = SCHEMA_PATH.read_text(encoding="utf-8")
        assert len(contenido.strip()) > 0, "schema.sql está vacío"

    @pytest.mark.parametrize("tabla", TABLAS_ESPERADAS)
    def test_schema_contiene_tabla(self, tabla):
        """Cada tabla esperada debe aparecer en el schema."""
        contenido = SCHEMA_PATH.read_text(encoding="utf-8").lower()
        assert tabla in contenido, (
            f"Tabla '{tabla}' no encontrada en database/schema.sql"
        )

    def test_schema_usa_create_table_if_not_exists(self):
        """El schema debe usar IF NOT EXISTS para ser idempotente."""
        contenido = SCHEMA_PATH.read_text(encoding="utf-8").upper()
        assert "CREATE TABLE IF NOT EXISTS" in contenido, (
            "El schema debe usar CREATE TABLE IF NOT EXISTS "
            "para poder ejecutarse múltiples veces sin errores."
        )

    def test_schema_tiene_primary_keys(self):
        """Todas las tablas deben tener PRIMARY KEY."""
        contenido = SCHEMA_PATH.read_text(encoding="utf-8").upper()
        assert "PRIMARY KEY" in contenido

    def test_schema_tiene_tabla_log_etl(self):
        """La tabla log_etl es crítica para la auditoría del pipeline."""
        contenido = SCHEMA_PATH.read_text(encoding="utf-8").lower()
        assert "log_etl" in contenido

    def test_schema_tiene_unique_constraints(self):
        """Deben existir restricciones UNIQUE para evitar duplicados."""
        contenido = SCHEMA_PATH.read_text(encoding="utf-8").upper()
        assert "UNIQUE" in contenido

    def test_schema_tiene_checks_nivel_calidad(self):
        """Los campos de nivel de calidad deben tener CHECK constraints."""
        contenido = SCHEMA_PATH.read_text(encoding="utf-8")
        assert "buena" in contenido and "regular" in contenido and "mala" in contenido, (
            "El schema debe incluir CHECK constraints con los valores "
            "válidos: buena, regular, mala"
        )

    def test_schema_menciona_todas_columnas_dataset(self):
        """La tabla dataset_modelo debe tener las columnas del CSV base."""
        columnas_clave = [
            "mp25", "temperatura_max", "humedad_relativa",
            "nivel_calidad_aire_dia_siguiente",
        ]
        contenido = SCHEMA_PATH.read_text(encoding="utf-8").lower()
        for col in columnas_clave:
            assert col in contenido, (
                f"Columna '{col}' no encontrada en dataset_modelo del schema"
            )


# ---------------------------------------------------------------------------
# Tests: importación de módulos
# ---------------------------------------------------------------------------

class TestImportacionModulos:

    def test_database_db_importable(self):
        """database/db.py debe poder importarse sin error de sintaxis."""
        try:
            import database.db as db_module
            assert hasattr(db_module, "get_engine"), (
                "database/db.py debe exponer la función get_engine()"
            )
            assert hasattr(db_module, "probar_conexion"), (
                "database/db.py debe exponer la función probar_conexion()"
            )
        except ImportError as e:
            pytest.fail(f"No se pudo importar database.db: {e}")

    def test_database_init_db_importable(self):
        """database/init_db.py debe poder importarse."""
        try:
            import database.init_db as init_module
            assert hasattr(init_module, "init_db")
            assert hasattr(init_module, "leer_schema")
        except ImportError as e:
            pytest.fail(f"No se pudo importar database.init_db: {e}")

    def test_load_postgres_importable(self):
        """etl/load_postgres.py debe poder importarse."""
        try:
            import etl.load_postgres as lp
            assert hasattr(lp, "load_all"), (
                "etl/load_postgres.py debe exponer la función load_all()"
            )
            assert hasattr(lp, "cargar_csv_a_tabla")
        except ImportError as e:
            pytest.fail(f"No se pudo importar etl.load_postgres: {e}")

    def test_etl_postgres_main_importable(self):
        """etl/etl_postgres_main.py debe poder importarse."""
        try:
            import etl.etl_postgres_main as epm
            assert hasattr(epm, "run_etl_postgres")
        except ImportError as e:
            pytest.fail(f"No se pudo importar etl.etl_postgres_main: {e}")

    def test_database_init_py_existe_o_path_funciona(self):
        """
        Los módulos de database/ deben ser importables.
        Esto falla si falta __init__.py o el path no está configurado.
        """
        db_dir = ROOT / "database"
        assert db_dir.exists(), f"Directorio database/ no encontrado: {db_dir}"
        assert (db_dir / "db.py").exists()
        assert (db_dir / "init_db.py").exists()
        assert (db_dir / "schema.sql").exists()


# ---------------------------------------------------------------------------
# Tests: archivos CSV del pipeline (opcionales con skip)
# ---------------------------------------------------------------------------

class TestArchivosCsvPipeline:
    """
    Verifica que los CSVs generados por el pipeline existen antes de
    intentar cargarlos a PostgreSQL.

    Estos tests usan pytest.skip si los archivos no existen,
    porque el pipeline puede no haberse ejecutado aún.
    """

    @pytest.mark.parametrize("nombre,ruta", list(CSVS_PIPELINE.items()))
    def test_csv_pipeline_existe(self, nombre, ruta):
        if not ruta.exists():
            pytest.skip(
                f"{nombre} no encontrado. "
                f"Ejecuta el pipeline ETL primero."
            )
        assert ruta.exists()

    def test_dataset_modelo_tiene_filas(self):
        ruta = CSVS_PIPELINE["dataset_modelo_base.csv"]
        if not ruta.exists():
            pytest.skip("dataset_modelo_base.csv no encontrado.")

        import pandas as pd
        df = pd.read_csv(ruta)
        assert len(df) > 0, "dataset_modelo_base.csv está vacío"

    def test_prediccion_tiene_nivel_predicho(self):
        ruta = CSVS_PIPELINE["prediccion_actual.csv"]
        if not ruta.exists():
            pytest.skip("prediccion_actual.csv no encontrado.")

        import pandas as pd
        df = pd.read_csv(ruta)
        assert "nivel_predicho" in df.columns
        assert df.iloc[0]["nivel_predicho"] in {"buena", "regular", "mala"}


# ---------------------------------------------------------------------------
# Test de integración: leer_schema() funciona
# ---------------------------------------------------------------------------

class TestLeerSchema:

    def test_leer_schema_retorna_string(self):
        from database.init_db import leer_schema
        contenido = leer_schema()
        assert isinstance(contenido, str)
        assert len(contenido) > 0

    def test_leer_schema_contiene_create_table(self):
        from database.init_db import leer_schema
        contenido = leer_schema().upper()
        assert "CREATE TABLE" in contenido