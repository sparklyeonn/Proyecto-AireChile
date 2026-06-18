"""
tests/test_docker_config.py
===========================
AireChile Analytics — Tests de validación de la configuración Docker.

Verifica que los archivos de Docker tienen el contenido correcto
sin necesitar Docker instalado ni contenedores corriendo.

Cobertura:
    - Dockerfile existe y tiene instrucciones clave
    - docker-compose.yml existe y tiene los servicios esperados
    - .dockerignore existe y excluye los archivos sensibles
    - Variables de entorno documentadas en .env.example
    - El módulo dashboard puede importar sus dependencias básicas

Ejecutar:
    pytest tests/test_docker_config.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Tests: Dockerfile
# ---------------------------------------------------------------------------

class TestDockerfile:

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = ROOT / "Dockerfile"
        if not ruta.exists():
            pytest.skip("Dockerfile no encontrado")
        return ruta.read_text(encoding="utf-8")

    def test_dockerfile_existe(self):
        assert (ROOT / "Dockerfile").exists(), "Dockerfile no encontrado en la raíz"

    def test_usa_python_311(self, contenido):
        assert "python:3.11" in contenido, (
            "Dockerfile debe usar Python 3.11 como imagen base"
        )

    def test_define_workdir(self, contenido):
        assert "WORKDIR" in contenido

    def test_copia_requirements(self, contenido):
        assert "requirements.txt" in contenido

    def test_instala_dependencias(self, contenido):
        assert "pip install" in contenido

    def test_expone_puerto_8501(self, contenido):
        assert "8501" in contenido, "Dockerfile debe exponer el puerto 8501 de Streamlit"

    def test_cmd_incluye_streamlit(self, contenido):
        assert "streamlit" in contenido.lower(), (
            "CMD del Dockerfile debe lanzar Streamlit"
        )

    def test_cmd_incluye_dashboard_app(self, contenido):
        assert "dashboards/app.py" in contenido

    def test_server_address_0000(self, contenido):
        """Streamlit debe escuchar en 0.0.0.0 para ser accesible desde fuera."""
        assert "0.0.0.0" in contenido, (
            "Streamlit debe usar --server.address=0.0.0.0 en el Dockerfile"
        )


# ---------------------------------------------------------------------------
# Tests: docker-compose.yml
# ---------------------------------------------------------------------------

class TestDockerCompose:

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = ROOT / "docker-compose.yml"
        if not ruta.exists():
            pytest.skip("docker-compose.yml no encontrado")
        return ruta.read_text(encoding="utf-8")

    def test_compose_existe(self):
        assert (ROOT / "docker-compose.yml").exists()

    def test_servicio_postgres_presente(self, contenido):
        assert "postgres" in contenido, (
            "docker-compose.yml debe incluir el servicio 'postgres'"
        )

    def test_servicio_dashboard_presente(self, contenido):
        assert "dashboard" in contenido, (
            "docker-compose.yml debe incluir el servicio 'dashboard'"
        )

    def test_imagen_postgres_15(self, contenido):
        assert "postgres:15" in contenido, (
            "Debe usar postgres:15 (versión estable LTS)"
        )

    def test_puerto_5432_mapeado(self, contenido):
        assert "5432" in contenido

    def test_puerto_8501_mapeado(self, contenido):
        assert "8501" in contenido

    def test_healthcheck_postgres(self, contenido):
        assert "healthcheck" in contenido, (
            "PostgreSQL debe tener healthcheck para que el dashboard "
            "espere a que esté listo"
        )

    def test_depends_on_presente(self, contenido):
        assert "depends_on" in contenido, (
            "El dashboard debe depender del servicio postgres"
        )

    def test_schema_montado_como_initdb(self, contenido):
        """El schema.sql debe ejecutarse automáticamente al iniciar PostgreSQL."""
        assert "docker-entrypoint-initdb.d" in contenido, (
            "schema.sql debe montarse en /docker-entrypoint-initdb.d/ "
            "para ejecutarse al inicializar la base de datos"
        )

    def test_volumen_data_montado(self, contenido):
        assert "./data:/app/data" in contenido, (
            "El directorio data/ debe montarse como volumen"
        )

    def test_volumen_models_montado(self, contenido):
        assert "./models:/app/models" in contenido, (
            "El directorio models/ debe montarse como volumen"
        )

    def test_postgres_host_es_nombre_servicio(self, contenido):
        """
        Dentro de Docker, el host de PostgreSQL debe ser 'postgres'
        (nombre del servicio), no 'localhost'.
        """
        assert "POSTGRES_HOST:     postgres" in contenido or \
               "POSTGRES_HOST: postgres" in contenido, (
            "POSTGRES_HOST debe ser 'postgres' (nombre del servicio) "
            "para la comunicación entre contenedores"
        )

    def test_tiene_red_interna(self, contenido):
        assert "networks" in contenido

    def test_tiene_volumen_nombrado_postgres(self, contenido):
        assert "postgres_data" in contenido, (
            "Debe existir un volumen nombrado para persistir los datos de PostgreSQL"
        )

    def test_parse_yaml_valido(self, contenido):
        """El archivo YAML debe tener sintaxis válida."""
        try:
            import yaml
            datos = yaml.safe_load(contenido)
            assert datos is not None
            assert "services" in datos
        except ImportError:
            pytest.skip("PyYAML no instalado — instala con: pip install pyyaml")
        except Exception as e:
            pytest.fail(f"docker-compose.yml tiene sintaxis YAML inválida: {e}")


# ---------------------------------------------------------------------------
# Tests: .dockerignore
# ---------------------------------------------------------------------------

class TestDockIgnore:

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = ROOT / ".dockerignore"
        if not ruta.exists():
            pytest.skip(".dockerignore no encontrado")
        return ruta.read_text(encoding="utf-8")

    def test_dockerignore_existe(self):
        assert (ROOT / ".dockerignore").exists()

    def test_excluye_env_real(self, contenido):
        assert ".env" in contenido, ".dockerignore debe excluir .env"

    def test_excluye_venv(self, contenido):
        assert ".venv" in contenido

    def test_excluye_datos_raw(self, contenido):
        assert "data/raw" in contenido, (
            "Los datos crudos no deben copiarse al contenedor"
        )

    def test_excluye_datos_procesados(self, contenido):
        assert "data/processed" in contenido

    def test_excluye_model_pkl(self, contenido):
        assert "model.pkl" in contenido

    def test_excluye_git(self, contenido):
        assert ".git" in contenido

    def test_excluye_pycache(self, contenido):
        assert "__pycache__" in contenido


# ---------------------------------------------------------------------------
# Tests: .env.example actualizado para Docker
# ---------------------------------------------------------------------------

class TestEnvExampleDocker:

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = ROOT / ".env.example"
        if not ruta.exists():
            pytest.skip(".env.example no encontrado")
        return ruta.read_text(encoding="utf-8")

    def test_tiene_postgres_host(self, contenido):
        assert "POSTGRES_HOST" in contenido

    def test_tiene_postgres_db(self, contenido):
        assert "POSTGRES_DB" in contenido

    def test_tiene_database_url(self, contenido):
        assert "DATABASE_URL" in contenido

    def test_tiene_comentario_docker(self, contenido):
        """Debe haber alguna mención a Docker o al nombre del servicio."""
        assert "postgres" in contenido.lower() or "docker" in contenido.lower()


# ---------------------------------------------------------------------------
# Tests: importación del dashboard
# ---------------------------------------------------------------------------

class TestDashboardImportable:

    def test_streamlit_importable(self):
        """Streamlit debe estar instalado."""
        try:
            import streamlit
        except ImportError:
            pytest.skip(
                "Streamlit no está instalado en este entorno. "
                "Ejecuta: pip install streamlit"
            )

    def test_plotly_importable(self):
        """Plotly debe estar instalado."""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip(
                "Plotly no está instalado en este entorno. "
                "Ejecuta: pip install plotly"
            )

    def test_pandas_importable(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.fail("Pandas no está instalado.")

    def test_dashboard_app_existe(self):
        assert (ROOT / "dashboards" / "app.py").exists(), (
            "dashboards/app.py no encontrado"
        )

    def test_dashboard_sintaxis_valida(self):
        """El archivo app.py no debe tener errores de sintaxis."""
        import ast
        ruta = ROOT / "dashboards" / "app.py"
        if not ruta.exists():
            pytest.skip("dashboards/app.py no encontrado")
        try:
            ast.parse(ruta.read_text(encoding="utf-8"))
        except SyntaxError as e:
            pytest.fail(f"dashboards/app.py tiene error de sintaxis: {e}")

    def test_requirements_txt_existe(self):
        assert (ROOT / "requirements.txt").exists()

    def test_requirements_incluye_streamlit(self):
        ruta = ROOT / "requirements.txt"
        if not ruta.exists():
            pytest.skip("requirements.txt no encontrado")
        contenido = ruta.read_text(encoding="utf-8").lower()
        assert "streamlit" in contenido

    def test_requirements_incluye_sqlalchemy(self):
        ruta = ROOT / "requirements.txt"
        if not ruta.exists():
            pytest.skip("requirements.txt no encontrado")
        contenido = ruta.read_text(encoding="utf-8").lower()
        assert "sqlalchemy" in contenido

    def test_requirements_incluye_psycopg2(self):
        ruta = ROOT / "requirements.txt"
        if not ruta.exists():
            pytest.skip("requirements.txt no encontrado")
        contenido = ruta.read_text(encoding="utf-8").lower()
        assert "psycopg2" in contenido