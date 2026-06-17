"""
tests/test_merge_sinca_meteo.py
===============================
AireChile Analytics — Tests unitarios para merge_sinca_meteo.py

Valida el proceso de unión de datasets SINCA y meteorología usando
DataFrames sintéticos (no requieren archivos reales ni conexión a internet).

Cobertura:
    - Merge conserva columna 'fecha'
    - Dataset final contiene columnas SINCA y meteorológicas
    - Variable objetivo 'nivel_calidad_aire_dia_siguiente' no se pierde
    - Dataset final no queda vacío
    - Left join conserva filas SINCA sin datos meteorológicos
    - No se generan columnas duplicadas (_x/_y)
    - Merge por fecha exacta funciona correctamente

Ejecutar:
    pytest tests/test_merge_sinca_meteo.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.merge_sinca_meteo import (
    _ejecutar_merge,
    _validar_compatibilidad,
    _validar_resultado,
    _seleccionar_columnas,
    COLUMNAS_FINALES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_sinca_ejemplo():
    """
    DataFrame que simula la salida del ETL SINCA.
    5 días con todas las columnas del dataset SINCA limpio.
    """
    return pd.DataFrame({
        "fecha": pd.date_range("2024-06-01", periods=5, freq="D"),
        "estacion":      "Puente Alto",
        "comuna":        "Puente Alto",
        "mp25":          [15.0, 28.0, 55.0, 20.0, 42.0],
        "estado_registro": ["validado"] * 5,
        "nivel_calidad_aire": ["buena", "regular", "mala", "buena", "regular"],
        "mes":           [6, 6, 6, 6, 6],
        "dia_semana":    [5, 6, 0, 1, 2],
        "mp25_dia_anterior":  [None, 15.0, 28.0, 55.0, 20.0],
        "mp25_promedio_7d":   [None, None, None, 32.67, 29.5],
        "nivel_calidad_aire_dia_siguiente": [
            "regular", "mala", "buena", "regular", None
        ],
    })


@pytest.fixture
def df_meteo_ejemplo():
    """
    DataFrame que simula la salida del ETL Open-Meteo.
    5 días con todas las columnas meteorológicas.
    """
    return pd.DataFrame({
        "fecha": pd.date_range("2024-06-01", periods=5, freq="D"),
        "temperatura_max":      [18.0, 17.5, 16.0, 20.0, 19.5],
        "temperatura_min":      [ 8.0,  7.5,  6.0, 10.0,  9.5],
        "temperatura_promedio": [13.0, 12.5, 11.0, 15.0, 14.5],
        "humedad_relativa":     [65.0, 70.0, 75.0, 60.0, 68.0],
        "velocidad_viento":     [12.0, 15.0, 18.0, 10.0, 14.0],
        "precipitacion":        [ 0.0,  0.0,  2.5,  0.0,  0.0],
    })


@pytest.fixture
def df_sinca_4_dias():
    """Dataset SINCA con 4 días para tests de solapamiento."""
    return pd.DataFrame({
        "fecha": pd.date_range("2024-06-01", periods=4, freq="D"),
        "estacion": "Test", "comuna": "Test",
        "mp25": [10.0, 20.0, 30.0, 40.0],
        "estado_registro": ["validado"] * 4,
        "nivel_calidad_aire": ["buena", "buena", "regular", "regular"],
        "mes": [6] * 4, "dia_semana": [0, 1, 2, 3],
        "mp25_dia_anterior": [None, 10.0, 20.0, 30.0],
        "mp25_promedio_7d": [None, None, None, None],
        "nivel_calidad_aire_dia_siguiente": ["buena", "regular", "regular", None],
    })


@pytest.fixture
def df_meteo_3_dias():
    """Dataset meteorológico con solo 3 de los 4 días del SINCA."""
    return pd.DataFrame({
        "fecha": pd.date_range("2024-06-01", periods=3, freq="D"),
        "temperatura_max":      [18.0, 17.5, 16.0],
        "temperatura_min":      [ 8.0,  7.5,  6.0],
        "temperatura_promedio": [13.0, 12.5, 11.0],
        "humedad_relativa":     [65.0, 70.0, 75.0],
        "velocidad_viento":     [12.0, 15.0, 18.0],
        "precipitacion":        [ 0.0,  0.0,  2.5],
    })


# ---------------------------------------------------------------------------
# Tests: _ejecutar_merge
# ---------------------------------------------------------------------------

class TestEjecutarMerge:

    def test_merge_conserva_columna_fecha(self, df_sinca_ejemplo, df_meteo_ejemplo):
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        assert "fecha" in resultado.columns

    def test_merge_contiene_columnas_sinca(self, df_sinca_ejemplo, df_meteo_ejemplo):
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        cols_sinca = ["mp25", "estacion", "comuna", "nivel_calidad_aire",
                      "mp25_dia_anterior", "mp25_promedio_7d"]
        for col in cols_sinca:
            assert col in resultado.columns, f"Columna SINCA faltante: {col}"

    def test_merge_contiene_columnas_meteorologicas(self, df_sinca_ejemplo, df_meteo_ejemplo):
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        cols_meteo = ["temperatura_max", "temperatura_min", "temperatura_promedio",
                      "humedad_relativa", "velocidad_viento", "precipitacion"]
        for col in cols_meteo:
            assert col in resultado.columns, f"Columna meteorológica faltante: {col}"

    def test_merge_conserva_variable_objetivo(self, df_sinca_ejemplo, df_meteo_ejemplo):
        """La columna target no debe perderse durante el merge."""
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        assert "nivel_calidad_aire_dia_siguiente" in resultado.columns

    def test_merge_no_vacio(self, df_sinca_ejemplo, df_meteo_ejemplo):
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        assert not resultado.empty

    def test_merge_conserva_filas_sinca(self, df_sinca_ejemplo, df_meteo_ejemplo):
        """Left join: el número de filas debe ser igual al de SINCA."""
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        assert len(resultado) == len(df_sinca_ejemplo)

    def test_merge_left_join_preserva_filas_sin_meteo(
        self, df_sinca_4_dias, df_meteo_3_dias
    ):
        """
        Con 4 días en SINCA y solo 3 en meteo, el resultado debe tener 4 filas.
        El día 4 (sin datos meteo) debe tener NaN en columnas meteorológicas.
        """
        resultado = _ejecutar_merge(df_sinca_4_dias, df_meteo_3_dias)
        assert len(resultado) == 4

        # La última fila (día 4) debe tener NaN en temperatura_max
        ultima = resultado.iloc[-1]
        assert pd.isna(ultima["temperatura_max"])

    def test_merge_sin_columnas_duplicadas(self, df_sinca_ejemplo, df_meteo_ejemplo):
        """No deben aparecer columnas con sufijos _x o _y."""
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        cols = list(resultado.columns)
        assert not any("_x" in c or "_y" in c for c in cols), (
            f"Columnas duplicadas encontradas: {[c for c in cols if '_x' in c or '_y' in c]}"
        )

    def test_merge_une_por_fecha_correcta(self, df_sinca_ejemplo, df_meteo_ejemplo):
        """
        Verifica que los valores meteorológicos se asignaron al día correcto.
        El primer día (2024-06-01) debe tener temperatura_max=18.0.
        """
        resultado = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        primera_fila = resultado.iloc[0]
        assert primera_fila["temperatura_max"] == pytest.approx(18.0)
        assert primera_fila["precipitacion"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _validar_compatibilidad
# ---------------------------------------------------------------------------

class TestValidarCompatibilidad:

    def test_rangos_solapados_no_lanza_error(self, df_sinca_ejemplo, df_meteo_ejemplo):
        """Cuando los rangos se solapan, no debe lanzar excepción."""
        _validar_compatibilidad(df_sinca_ejemplo, df_meteo_ejemplo)  # sin error

    def test_rangos_sin_solape_lanza_error(self):
        """Si los rangos no se solapan, debe lanzar ValueError."""
        df_sinca = pd.DataFrame({
            "fecha": pd.date_range("2022-01-01", periods=5, freq="D")
        })
        df_meteo = pd.DataFrame({
            "fecha": pd.date_range("2025-01-01", periods=5, freq="D")
        })
        with pytest.raises(ValueError, match="solapamiento"):
            _validar_compatibilidad(df_sinca, df_meteo)


# ---------------------------------------------------------------------------
# Tests: _validar_resultado
# ---------------------------------------------------------------------------

class TestValidarResultado:

    def test_dataset_valido_retorna_true(self, df_sinca_ejemplo, df_meteo_ejemplo):
        merged = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        assert _validar_resultado(merged, len(df_sinca_ejemplo)) is True

    def test_dataset_vacio_retorna_false(self):
        df_vacio = pd.DataFrame()
        assert _validar_resultado(df_vacio, 100) is False

    def test_dataset_sin_fecha_retorna_false(self):
        df_sin_fecha = pd.DataFrame({
            "mp25": [10.0, 20.0],
            "nivel_calidad_aire_dia_siguiente": ["buena", "regular"],
        })
        assert _validar_resultado(df_sin_fecha, 2) is False

    def test_dataset_sin_target_retorna_false(self):
        df_sin_target = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=2, freq="D"),
            "mp25": [10.0, 20.0],
        })
        assert _validar_resultado(df_sin_target, 2) is False


# ---------------------------------------------------------------------------
# Tests: _seleccionar_columnas
# ---------------------------------------------------------------------------

class TestSeleccionarColumnas:

    def test_columnas_en_orden_correcto(self, df_sinca_ejemplo, df_meteo_ejemplo):
        merged = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        resultado = _seleccionar_columnas(merged)
        # Las columnas del resultado deben ser un subconjunto de COLUMNAS_FINALES
        assert all(c in COLUMNAS_FINALES for c in resultado.columns)

    def test_fecha_es_primera_columna(self, df_sinca_ejemplo, df_meteo_ejemplo):
        merged = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        resultado = _seleccionar_columnas(merged)
        assert resultado.columns[0] == "fecha"

    def test_target_es_ultima_columna(self, df_sinca_ejemplo, df_meteo_ejemplo):
        merged = _ejecutar_merge(df_sinca_ejemplo, df_meteo_ejemplo)
        resultado = _seleccionar_columnas(merged)
        assert resultado.columns[-1] == "nivel_calidad_aire_dia_siguiente"


# ---------------------------------------------------------------------------
# Test de integración: pipeline completo con datos sintéticos
# ---------------------------------------------------------------------------

class TestIntegracion:

    def test_pipeline_completo_genera_dataset_correcto(
        self, df_sinca_ejemplo, df_meteo_ejemplo, tmp_path
    ):
        """
        Test end-to-end: simula el pipeline merge completo con datos
        sintéticos y verifica el resultado sin tocar archivos reales.
        """
        from etl.merge_sinca_meteo import merge_sinca_meteo

        # Guardar datasets sintéticos en carpeta temporal
        ruta_sinca = tmp_path / "sinca_transformado.csv"
        ruta_meteo = tmp_path / "open_meteo_transformado.csv"
        ruta_salida = tmp_path / "dataset_modelo_base.csv"

        df_sinca_ejemplo.to_csv(ruta_sinca, index=False)
        df_meteo_ejemplo.to_csv(ruta_meteo, index=False)

        # Ejecutar merge
        df_final = merge_sinca_meteo(
            ruta_sinca=ruta_sinca,
            ruta_meteo=ruta_meteo,
            ruta_salida=ruta_salida,
        )

        # Verificaciones
        assert not df_final.empty, "El dataset final no debe estar vacío"
        assert "fecha" in df_final.columns
        assert "nivel_calidad_aire_dia_siguiente" in df_final.columns
        assert "temperatura_max" in df_final.columns
        assert "mp25" in df_final.columns
        assert ruta_salida.exists(), "El archivo CSV final debe existir"

        # Verificar que el CSV fue guardado correctamente
        df_leido = pd.read_csv(ruta_salida)
        assert len(df_leido) == len(df_final)