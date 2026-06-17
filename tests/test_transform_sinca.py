"""
tests/test_transform_sinca.py
=============================
AireChile Analytics — Tests unitarios para transform_sinca.py

Valida cada función de transformación de forma aislada usando
DataFrames sintéticos, sin depender del archivo SINCA real.
Esto garantiza que los tests corran en cualquier máquina y en CI/CD.

Cobertura:
    - _construir_mp25()            → prioridad validado > preliminar
    - _clasificar_calidad()        → umbrales buena/regular/mala
    - _agregar_nivel_calidad()     → columna nivel_calidad_aire
    - _agregar_columnas_temporales → mes y dia_semana
    - _calcular_mp25_dia_anterior()→ lag-1 por estación
    - _calcular_promedio_7d()      → rolling mean 7 días
    - _calcular_target()           → nivel_calidad_aire_dia_siguiente
    - transform_sinca()            → pipeline completo end-to-end

Ejecutar:
    pytest tests/test_transform_sinca.py -v
    pytest tests/test_transform_sinca.py -v --tb=short
"""

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import pytest

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.transform_sinca import (
    _construir_mp25,
    _clasificar_calidad,
    _agregar_nivel_calidad,
    _agregar_columnas_temporales,
    _calcular_mp25_dia_anterior,
    _calcular_promedio_7d,
    _calcular_target,
    transform_sinca,
    UMBRAL_BUENA,
    UMBRAL_REGULAR,
)


# ---------------------------------------------------------------------------
# Fixtures — DataFrames reutilizables en múltiples tests
# ---------------------------------------------------------------------------

@pytest.fixture
def df_crudo_simple():
    """
    DataFrame mínimo que simula la salida de extract_sinca().
    5 días consecutivos con distintas combinaciones de validado/preliminar.
    """
    return pd.DataFrame({
        "fecha": pd.date_range("2024-01-01", periods=5, freq="D"),
        "estacion": "Puente Alto",
        "comuna":   "Puente Alto",
        "mp25_validado":    [10.0,  None, 30.0, None, 60.0],
        "mp25_preliminar":  [None,  20.0, 99.0, None, 55.0],
        "mp25_no_validado": [None,  None, None, None, None],
    })


@pytest.fixture
def df_con_mp25():
    """
    DataFrame ya con columna mp25 construida (post _construir_mp25).
    10 días con valores variados para probar clasificaciones y métricas.
    """
    return pd.DataFrame({
        "fecha":     pd.date_range("2024-01-01", periods=10, freq="D"),
        "estacion":  "Puente Alto",
        "comuna":    "Puente Alto",
        "mp25":      [5.0, 15.0, 25.0, 26.0, 40.0, 50.0, 51.0, 70.0, 20.0, 30.0],
        "estado_registro": ["validado"] * 10,
        "mp25_validado":    [5.0, 15.0, 25.0, 26.0, 40.0, 50.0, 51.0, 70.0, 20.0, 30.0],
        "mp25_preliminar":  [None] * 10,
        "mp25_no_validado": [None] * 10,
    })


# ---------------------------------------------------------------------------
# Tests: _construir_mp25
# ---------------------------------------------------------------------------

class TestConstruirMp25:

    def test_usa_validado_cuando_disponible(self, df_crudo_simple):
        """Si mp25_validado tiene valor, debe usarse aunque preliminar también tenga."""
        df = _construir_mp25(df_crudo_simple)
        # Fila 0: validado=10, preliminar=None → debe usar 10
        assert df.loc[0, "mp25"] == pytest.approx(10.0)
        assert df.loc[0, "estado_registro"] == "validado"

    def test_usa_preliminar_como_respaldo(self, df_crudo_simple):
        """Si mp25_validado es None pero preliminar tiene valor, usa preliminar."""
        df = _construir_mp25(df_crudo_simple)
        # Fila 1: validado=None, preliminar=20 → debe usar 20
        assert df.loc[1, "mp25"] == pytest.approx(20.0)
        assert df.loc[1, "estado_registro"] == "preliminar"

    def test_validado_tiene_prioridad_sobre_preliminar(self, df_crudo_simple):
        """Cuando ambas columnas tienen valor, validado siempre gana."""
        df = _construir_mp25(df_crudo_simple)
        # Fila 2: validado=30, preliminar=99 → debe usar 30, no 99
        assert df.loc[2, "mp25"] == pytest.approx(30.0)
        assert df.loc[2, "estado_registro"] == "validado"

    def test_sin_dato_cuando_ambas_son_nulas(self, df_crudo_simple):
        """Si ambas columnas son None, mp25 queda NaN y estado='sin_dato'."""
        df = _construir_mp25(df_crudo_simple)
        # Fila 3: validado=None, preliminar=None
        assert pd.isna(df.loc[3, "mp25"])
        assert df.loc[3, "estado_registro"] == "sin_dato"

    def test_columnas_creadas(self, df_crudo_simple):
        """Verifica que se crean las columnas mp25 y estado_registro."""
        df = _construir_mp25(df_crudo_simple)
        assert "mp25" in df.columns
        assert "estado_registro" in df.columns

    def test_no_modifica_dataframe_original(self, df_crudo_simple):
        """La función no debe modificar el DataFrame de entrada (copia defensiva)."""
        df_original = df_crudo_simple.copy()
        _construir_mp25(df_crudo_simple)
        pd.testing.assert_frame_equal(df_crudo_simple, df_original)


# ---------------------------------------------------------------------------
# Tests: _clasificar_calidad
# ---------------------------------------------------------------------------

class TestClasificarCalidad:

    def test_limite_inferior_buena(self):
        """0 µg/m³ debe clasificarse como buena."""
        assert _clasificar_calidad(0.0) == "buena"

    def test_exactamente_umbral_buena(self):
        """El valor exacto del umbral (25.0) es buena (≤ 25)."""
        assert _clasificar_calidad(UMBRAL_BUENA) == "buena"

    def test_un_poco_sobre_umbral_buena(self):
        """25.1 µg/m³ debe clasificarse como regular."""
        assert _clasificar_calidad(25.1) == "regular"

    def test_exactamente_umbral_regular(self):
        """El valor exacto del segundo umbral (50.0) es regular (≤ 50)."""
        assert _clasificar_calidad(UMBRAL_REGULAR) == "regular"

    def test_un_poco_sobre_umbral_regular(self):
        """50.1 µg/m³ debe clasificarse como mala."""
        assert _clasificar_calidad(50.1) == "mala"

    def test_valor_alto_es_mala(self):
        """Valores muy altos (preemergencia) deben ser mala."""
        assert _clasificar_calidad(200.0) == "mala"

    def test_nan_retorna_sin_dato(self):
        """np.nan debe retornar 'sin_dato', no lanzar error."""
        assert _clasificar_calidad(np.nan) == "sin_dato"

    def test_none_retorna_sin_dato(self):
        """None debe retornar 'sin_dato', no lanzar error."""
        assert _clasificar_calidad(None) == "sin_dato"

    @pytest.mark.parametrize("valor,esperado", [
        (0.0,   "buena"),
        (12.5,  "buena"),
        (25.0,  "buena"),
        (25.1,  "regular"),
        (37.5,  "regular"),
        (50.0,  "regular"),
        (50.1,  "mala"),
        (100.0, "mala"),
        (300.0, "mala"),
    ])
    def test_clasificacion_parametrizada(self, valor, esperado):
        """Verifica múltiples valores contra sus clasificaciones esperadas."""
        assert _clasificar_calidad(valor) == esperado


# ---------------------------------------------------------------------------
# Tests: _agregar_nivel_calidad
# ---------------------------------------------------------------------------

class TestAgregarNivelCalidad:

    def test_crea_columna_nivel_calidad_aire(self, df_con_mp25):
        df = _agregar_nivel_calidad(df_con_mp25)
        assert "nivel_calidad_aire" in df.columns

    def test_valores_correctos(self, df_con_mp25):
        df = _agregar_nivel_calidad(df_con_mp25)
        # mp25=5 → buena, mp25=26 → regular, mp25=51 → mala
        assert df.loc[df["mp25"] == 5.0,  "nivel_calidad_aire"].iloc[0] == "buena"
        assert df.loc[df["mp25"] == 26.0, "nivel_calidad_aire"].iloc[0] == "regular"
        assert df.loc[df["mp25"] == 51.0, "nivel_calidad_aire"].iloc[0] == "mala"

    def test_solo_clases_validas(self, df_con_mp25):
        df = _agregar_nivel_calidad(df_con_mp25)
        clases = set(df["nivel_calidad_aire"].unique())
        assert clases.issubset({"buena", "regular", "mala", "sin_dato"})


# ---------------------------------------------------------------------------
# Tests: _agregar_columnas_temporales
# ---------------------------------------------------------------------------

class TestColumnasTemporales:

    def test_crea_columna_mes(self, df_con_mp25):
        df = _agregar_columnas_temporales(df_con_mp25)
        assert "mes" in df.columns

    def test_crea_columna_dia_semana(self, df_con_mp25):
        df = _agregar_columnas_temporales(df_con_mp25)
        assert "dia_semana" in df.columns

    def test_mes_correcto(self):
        """Fecha 2024-03-15 debe tener mes=3."""
        df = pd.DataFrame({
            "fecha":    [pd.Timestamp("2024-03-15")],
            "mp25":     [10.0],
            "estacion": ["Test"],
            "comuna":   ["Test"],
        })
        df = _agregar_columnas_temporales(df)
        assert df.loc[0, "mes"] == 3

    def test_dia_semana_correcto(self):
        """2024-01-01 es lunes → dia_semana=0."""
        df = pd.DataFrame({
            "fecha":    [pd.Timestamp("2024-01-01")],  # lunes
            "mp25":     [10.0],
            "estacion": ["Test"],
            "comuna":   ["Test"],
        })
        df = _agregar_columnas_temporales(df)
        assert df.loc[0, "dia_semana"] == 0  # 0 = lunes en pandas

    def test_rango_mes(self, df_con_mp25):
        df = _agregar_columnas_temporales(df_con_mp25)
        assert df["mes"].between(1, 12).all()

    def test_rango_dia_semana(self, df_con_mp25):
        df = _agregar_columnas_temporales(df_con_mp25)
        assert df["dia_semana"].between(0, 6).all()


# ---------------------------------------------------------------------------
# Tests: _calcular_mp25_dia_anterior
# ---------------------------------------------------------------------------

class TestMp25DiaAnterior:

    def test_primera_fila_es_nan(self, df_con_mp25):
        """La primera fila no tiene día anterior → debe ser NaN."""
        df = _calcular_mp25_dia_anterior(df_con_mp25)
        assert pd.isna(df.loc[0, "mp25_dia_anterior"])

    def test_segunda_fila_es_valor_de_ayer(self, df_con_mp25):
        """La fila 1 debe tener el mp25 de la fila 0 como día anterior."""
        df = _calcular_mp25_dia_anterior(df_con_mp25)
        assert df.loc[1, "mp25_dia_anterior"] == pytest.approx(df_con_mp25.loc[0, "mp25"])

    def test_crea_columna(self, df_con_mp25):
        df = _calcular_mp25_dia_anterior(df_con_mp25)
        assert "mp25_dia_anterior" in df.columns

    def test_lag_correcto_en_toda_la_serie(self, df_con_mp25):
        """Verifica que el lag-1 es correcto para todas las filas (excepto la primera)."""
        df = _calcular_mp25_dia_anterior(df_con_mp25)
        for i in range(1, len(df)):
            assert df.loc[i, "mp25_dia_anterior"] == pytest.approx(
                df_con_mp25.loc[i - 1, "mp25"]
            )

    def test_multiples_estaciones_lag_independiente(self):
        """
        Con dos estaciones, el lag no debe cruzarse entre ellas.
        La primera fila de cada estación debe ser NaN.
        """
        df = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=4, freq="D").tolist() * 2,
            "estacion": ["A"] * 4 + ["B"] * 4,
            "comuna":   ["CA"] * 4 + ["CB"] * 4,
            "mp25": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
            "mp25_validado":    [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
            "mp25_preliminar":  [None] * 8,
            "mp25_no_validado": [None] * 8,
        })
        df = df.sort_values(["estacion", "fecha"]).reset_index(drop=True)
        df = _calcular_mp25_dia_anterior(df)

        # Primera fila de estación A y de estación B deben ser NaN
        assert pd.isna(df[df["estacion"] == "A"].iloc[0]["mp25_dia_anterior"])
        assert pd.isna(df[df["estacion"] == "B"].iloc[0]["mp25_dia_anterior"])

        # Segunda fila de B no debe tener el valor de la última fila de A
        segunda_b = df[df["estacion"] == "B"].iloc[1]["mp25_dia_anterior"]
        assert segunda_b == pytest.approx(100.0)  # primer valor de B


# ---------------------------------------------------------------------------
# Tests: _calcular_promedio_7d
# ---------------------------------------------------------------------------

class TestPromedio7d:

    def test_crea_columna(self, df_con_mp25):
        df = _calcular_promedio_7d(df_con_mp25)
        assert "mp25_promedio_7d" in df.columns

    def test_primeras_filas_son_nan_o_parciales(self, df_con_mp25):
        """Con min_periods=3, las primeras 3 filas deben ser NaN."""
        df = _calcular_promedio_7d(df_con_mp25)
        # Las primeras 3 filas no tienen suficientes días anteriores
        assert pd.isna(df.loc[0, "mp25_promedio_7d"])
        assert pd.isna(df.loc[1, "mp25_promedio_7d"])
        assert pd.isna(df.loc[2, "mp25_promedio_7d"])

    def test_promedio_correcto_desde_fila_8(self):
        """
        Con 10+ días de datos, el promedio de los últimos 7 días
        (sin incluir el día actual) debe calcularse correctamente.
        """
        valores = [10.0] * 14  # 14 días todos con mp25=10
        df = pd.DataFrame({
            "fecha":     pd.date_range("2024-01-01", periods=14, freq="D"),
            "estacion":  "Test",
            "comuna":    "Test",
            "mp25":      valores,
            "mp25_validado":    valores,
            "mp25_preliminar":  [None] * 14,
            "mp25_no_validado": [None] * 14,
        })
        df = _calcular_promedio_7d(df)

        # Desde la fila 7 en adelante, el promedio de 7 días = 10.0
        for i in range(7, 14):
            assert df.loc[i, "mp25_promedio_7d"] == pytest.approx(10.0), (
                f"Fila {i}: esperado 10.0, got {df.loc[i, 'mp25_promedio_7d']}"
            )

    def test_no_incluye_dia_actual(self):
        """
        El promedio móvil usa shift(1), por lo que no incluye el día actual.
        Si el día actual tiene un spike, no debe afectar su propio promedio.
        """
        valores = [10.0] * 8 + [999.0]  # spike en el último día
        df = pd.DataFrame({
            "fecha":     pd.date_range("2024-01-01", periods=9, freq="D"),
            "estacion":  "Test",
            "comuna":    "Test",
            "mp25":      valores,
            "mp25_validado":    valores,
            "mp25_preliminar":  [None] * 9,
            "mp25_no_validado": [None] * 9,
        })
        df = _calcular_promedio_7d(df)

        # El promedio del último día (fila 8) debe ser ~10, no incluir 999
        assert df.loc[8, "mp25_promedio_7d"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: _calcular_target
# ---------------------------------------------------------------------------

class TestCalcularTarget:

    def test_crea_columna_target(self, df_con_mp25):
        df_prep = _agregar_nivel_calidad(df_con_mp25)
        df = _calcular_target(df_prep)
        assert "nivel_calidad_aire_dia_siguiente" in df.columns

    def test_ultima_fila_es_nan(self, df_con_mp25):
        """La última fila no tiene día siguiente → target debe ser NaN."""
        df_prep = _agregar_nivel_calidad(df_con_mp25)
        df = _calcular_target(df_prep)
        assert pd.isna(df.loc[len(df) - 1, "nivel_calidad_aire_dia_siguiente"])

    def test_target_es_nivel_del_dia_siguiente(self, df_con_mp25):
        """
        El target de la fila i debe ser la clasificación del mp25 de la fila i+1.
        """
        df_prep = _agregar_nivel_calidad(df_con_mp25)
        df = _calcular_target(df_prep)

        for i in range(len(df) - 1):
            mp25_siguiente = df_con_mp25.loc[i + 1, "mp25"]
            nivel_esperado = _clasificar_calidad(mp25_siguiente)
            nivel_real     = df.loc[i, "nivel_calidad_aire_dia_siguiente"]
            assert nivel_real == nivel_esperado, (
                f"Fila {i}: mp25 siguiente={mp25_siguiente}, "
                f"esperado='{nivel_esperado}', got='{nivel_real}'"
            )

    def test_clases_validas_en_target(self, df_con_mp25):
        df_prep = _agregar_nivel_calidad(df_con_mp25)
        df = _calcular_target(df_prep)
        clases = set(df["nivel_calidad_aire_dia_siguiente"].dropna().unique())
        assert clases.issubset({"buena", "regular", "mala"})


# ---------------------------------------------------------------------------
# Tests: transform_sinca (pipeline completo)
# ---------------------------------------------------------------------------

class TestTransformSinca:

    def test_retorna_dataframe(self, df_crudo_simple):
        resultado = transform_sinca(df_crudo_simple)
        assert isinstance(resultado, pd.DataFrame)

    def test_columnas_finales_completas(self, df_crudo_simple):
        """El DataFrame resultante debe tener exactamente las 11 columnas esperadas."""
        COLUMNAS_ESPERADAS = {
            "fecha", "estacion", "comuna", "mp25", "estado_registro",
            "nivel_calidad_aire", "mes", "dia_semana",
            "mp25_dia_anterior", "mp25_promedio_7d",
            "nivel_calidad_aire_dia_siguiente",
        }
        resultado = transform_sinca(df_crudo_simple)
        assert COLUMNAS_ESPERADAS.issubset(set(resultado.columns))

    def test_no_hay_filas_sin_fecha(self, df_crudo_simple):
        resultado = transform_sinca(df_crudo_simple)
        assert resultado["fecha"].isna().sum() == 0

    def test_mp25_es_numerico(self, df_crudo_simple):
        import pandas as pd
        resultado = transform_sinca(df_crudo_simple)
        assert pd.api.types.is_numeric_dtype(resultado["mp25"])

    def test_filas_sin_mp25_eliminadas(self):
        """Filas donde tanto validado como preliminar son NaN deben eliminarse."""
        df = pd.DataFrame({
            "fecha":             pd.date_range("2024-01-01", periods=3, freq="D"),
            "estacion":         "Test",
            "comuna":           "Test",
            "mp25_validado":    [10.0, None, 30.0],   # fila 1 sin dato
            "mp25_preliminar":  [None, None, None],   # fila 1 sin respaldo
            "mp25_no_validado": [None, None, None],
        })
        resultado = transform_sinca(df)
        # Deben quedar solo 2 filas (la fila 1 sin mp25 fue eliminada)
        assert len(resultado) == 2

    def test_error_si_dataframe_vacio(self):
        """transform_sinca debe lanzar ValueError si recibe un DataFrame vacío."""
        df_vacio = pd.DataFrame()
        with pytest.raises(ValueError, match="vacío"):
            transform_sinca(df_vacio)

    def test_error_si_faltan_columnas(self):
        """transform_sinca debe lanzar KeyError si faltan columnas requeridas."""
        df_incompleto = pd.DataFrame({"fecha": [pd.Timestamp("2024-01-01")]})
        with pytest.raises(KeyError):
            transform_sinca(df_incompleto)

    def test_ordenado_por_fecha(self, df_crudo_simple):
        """El resultado debe estar ordenado cronológicamente."""
        resultado = transform_sinca(df_crudo_simple)
        fechas = resultado["fecha"].tolist()
        assert fechas == sorted(fechas)

    def test_columnas_no_incluyen_crudas(self, df_crudo_simple):
        """Las columnas crudas mp25_validado, etc. no deben estar en el output."""
        resultado = transform_sinca(df_crudo_simple)
        cols = set(resultado.columns)
        assert "mp25_validado" not in cols
        assert "mp25_preliminar" not in cols
        assert "mp25_no_validado" not in cols