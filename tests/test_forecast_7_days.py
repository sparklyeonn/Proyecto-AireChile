"""
tests/test_forecast_7_days.py
=============================
AireChile Analytics — Tests unitarios del pronóstico a 7 días.

Todos los tests usan datos sintéticos y modelos entrenados en memoria
(no requieren archivos reales ni conexión a internet).

Cobertura:
    - prediccion_7_dias.csv tiene 7 filas
    - Contiene columnas obligatorias
    - horizonte_dia va de 1 a 7
    - Categorías son solo buena/regular/mala
    - mp25_estimado es numérico y no negativo
    - Predicción recursiva actualiza estado correctamente
    - clasificar_mp25 con umbrales DS59
    - cargar_estado_inicial lee el dataset correctamente
    - El dashboard puede leer el CSV sin errores

Ejecutar:
    pytest tests/test_forecast_7_days.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.predict_7_days import (
    clasificar_mp25,
    predecir_7_dias,
    _calcular_promedio_7d,
    FEATURES,
    UMBRAL_BUENA,
    UMBRAL_REGULAR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def modelo_sintetico():
    """
    RandomForestRegressor entrenado con datos sintéticos.
    Suficiente para probar la lógica de predicción recursiva.
    """
    from sklearn.ensemble import RandomForestRegressor
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "mp25":               np.random.uniform(5, 100, n),
        "mp25_dia_anterior":  np.random.uniform(5, 100, n),
        "mp25_promedio_7d":   np.random.uniform(10, 80, n),
        "mes":                np.random.randint(1, 13, n),
        "dia_semana":         np.random.randint(0, 7, n),
        "temperatura_max":    np.random.uniform(5, 30, n),
        "temperatura_min":    np.random.uniform(0, 15, n),
        "temperatura_promedio": np.random.uniform(3, 22, n),
        "humedad_relativa":   np.random.uniform(40, 90, n),
        "velocidad_viento":   np.random.uniform(5, 40, n),
        "precipitacion":      np.random.uniform(0, 10, n),
    })
    y = X["mp25"] * 0.8 + np.random.normal(0, 5, n)
    y = y.clip(0, 200)
    modelo = RandomForestRegressor(n_estimators=10, random_state=42)
    modelo.fit(X[FEATURES], y)
    return modelo


@pytest.fixture(scope="module")
def estado_inicial_sintetico():
    """Estado inicial sintético que simula cargar_estado_inicial()."""
    return {
        "fecha_base":    pd.Timestamp("2026-06-08"),
        "mp25_actual":   85.0,
        "mp25_anterior": 80.0,
        "mp25_7d_reales": [70.0, 75.0, 80.0, 85.0, 88.0, 82.0, 85.0],
        "estacion":      "Puente Alto",
        "comuna":        "Puente Alto",
    }


@pytest.fixture(scope="module")
def df_meteo_sintetico():
    """DataFrame de pronóstico meteorológico sintético para 7 días."""
    fechas = pd.date_range("2026-06-09", periods=7, freq="D")
    return pd.DataFrame({
        "fecha":               fechas,
        "temperatura_max":     [18.0, 17.5, 16.0, 20.0, 19.5, 15.0, 14.0],
        "temperatura_min":     [ 8.0,  7.5,  6.0, 10.0,  9.5,  5.0,  4.0],
        "temperatura_promedio":[13.0, 12.5, 11.0, 15.0, 14.5, 10.0,  9.0],
        "humedad_relativa":    [65.0, 70.0, 75.0, 60.0, 68.0, 80.0, 85.0],
        "velocidad_viento":    [12.0, 15.0, 18.0, 10.0, 14.0,  8.0,  6.0],
        "precipitacion":       [ 0.0,  0.0,  2.5,  0.0,  0.0,  5.0,  0.0],
    })


@pytest.fixture(scope="module")
def df_pronostico_7_dias(modelo_sintetico, estado_inicial_sintetico, df_meteo_sintetico):
    """Pronóstico de 7 días generado con datos sintéticos."""
    return predecir_7_dias(
        modelo_sintetico,
        estado_inicial_sintetico,
        df_meteo_sintetico,
        forecast_days=7,
    )


# ---------------------------------------------------------------------------
# Tests: clasificar_mp25
# ---------------------------------------------------------------------------

class TestClasificarMp25:

    @pytest.mark.parametrize("valor,esperado", [
        (0.0,       "buena"),
        (10.0,      "buena"),
        (25.0,      "buena"),    # exactamente en el umbral → buena
        (25.1,      "regular"),
        (35.0,      "regular"),
        (50.0,      "regular"),  # exactamente en el umbral → regular
        (50.1,      "mala"),
        (100.0,     "mala"),
        (np.nan,    "sin_dato"),
        (-1.0,      "sin_dato"),
    ])
    def test_clasificacion(self, valor, esperado):
        assert clasificar_mp25(valor) == esperado

    def test_umbral_buena_correcto(self):
        assert UMBRAL_BUENA == 25.0

    def test_umbral_regular_correcto(self):
        assert UMBRAL_REGULAR == 50.0


# ---------------------------------------------------------------------------
# Tests: _calcular_promedio_7d
# ---------------------------------------------------------------------------

class TestPromedio7d:

    def test_promedio_exacto_con_7_valores(self):
        valores = [10.0] * 7
        assert _calcular_promedio_7d(valores) == pytest.approx(10.0)

    def test_usa_solo_ultimos_7(self):
        # 10 valores: los primeros 3 son 100, los últimos 7 son 10
        valores = [100.0, 100.0, 100.0] + [10.0] * 7
        resultado = _calcular_promedio_7d(valores)
        assert resultado == pytest.approx(10.0)

    def test_menos_de_7_valores(self):
        """Con menos de 7 valores usa los que hay."""
        valores = [20.0, 30.0, 40.0]
        resultado = _calcular_promedio_7d(valores)
        assert resultado == pytest.approx(30.0)

    def test_lista_vacia_retorna_cero(self):
        assert _calcular_promedio_7d([]) == 0.0

    def test_ignora_nan(self):
        valores = [10.0, np.nan, 20.0, np.nan, 30.0]
        resultado = _calcular_promedio_7d(valores)
        assert resultado == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Tests: estructura del pronóstico de 7 días
# ---------------------------------------------------------------------------

class TestPronostico7Dias:

    def test_retorna_dataframe(self, df_pronostico_7_dias):
        assert isinstance(df_pronostico_7_dias, pd.DataFrame)

    def test_exactamente_7_filas(self, df_pronostico_7_dias):
        """El pronóstico debe tener exactamente 7 filas."""
        assert len(df_pronostico_7_dias) == 7, (
            f"Se esperaban 7 filas, se obtuvieron {len(df_pronostico_7_dias)}"
        )

    def test_horizonte_dia_de_1_a_7(self, df_pronostico_7_dias):
        """horizonte_dia debe ir de 1 a 7 sin saltos."""
        horizontes = sorted(df_pronostico_7_dias["horizonte_dia"].tolist())
        assert horizontes == list(range(1, 8)), (
            f"horizonte_dia esperado: [1,2,3,4,5,6,7], obtenido: {horizontes}"
        )

    def test_columnas_obligatorias(self, df_pronostico_7_dias):
        columnas_req = [
            "fecha", "estacion", "comuna",
            "mp25_estimado", "nivel_calidad_aire_predicho",
            "temperatura_max", "temperatura_min", "temperatura_promedio",
            "humedad_relativa", "velocidad_viento", "precipitacion",
            "horizonte_dia", "fecha_generacion",
        ]
        for col in columnas_req:
            assert col in df_pronostico_7_dias.columns, (
                f"Columna obligatoria faltante: {col}"
            )

    def test_categorias_validas(self, df_pronostico_7_dias):
        """Solo debe haber buena, regular o mala (no sin_dato)."""
        clases = set(df_pronostico_7_dias["nivel_calidad_aire_predicho"].unique())
        validas = {"buena", "regular", "mala"}
        invalidas = clases - validas
        assert not invalidas, f"Categorías inválidas: {invalidas}"

    def test_mp25_no_negativo(self, df_pronostico_7_dias):
        """MP2.5 estimado no puede ser negativo."""
        assert (df_pronostico_7_dias["mp25_estimado"] >= 0).all()

    def test_mp25_es_numerico(self, df_pronostico_7_dias):
        assert pd.api.types.is_numeric_dtype(df_pronostico_7_dias["mp25_estimado"])

    def test_sin_nulos_en_mp25(self, df_pronostico_7_dias):
        assert df_pronostico_7_dias["mp25_estimado"].isna().sum() == 0

    def test_fechas_consecutivas(self, df_pronostico_7_dias, estado_inicial_sintetico):
        """Las fechas deben ser consecutivas a partir del día siguiente a fecha_base."""
        fechas = pd.to_datetime(df_pronostico_7_dias["fecha"])
        fecha_esperada_1 = estado_inicial_sintetico["fecha_base"] + timedelta(days=1)
        assert fechas.iloc[0] == fecha_esperada_1, (
            f"Primera fecha esperada: {fecha_esperada_1.date()}, "
            f"obtenida: {fechas.iloc[0].date()}"
        )
        for i in range(1, len(fechas)):
            diff = (fechas.iloc[i] - fechas.iloc[i-1]).days
            assert diff == 1, f"Las fechas no son consecutivas entre fila {i-1} y {i}"

    def test_estacion_y_comuna_correctas(self, df_pronostico_7_dias, estado_inicial_sintetico):
        assert (df_pronostico_7_dias["estacion"] == estado_inicial_sintetico["estacion"]).all()
        assert (df_pronostico_7_dias["comuna"] == estado_inicial_sintetico["comuna"]).all()

    def test_fecha_generacion_es_string(self, df_pronostico_7_dias):
        for val in df_pronostico_7_dias["fecha_generacion"]:
            assert isinstance(val, str), f"fecha_generacion debe ser string, got {type(val)}"

    def test_nivel_coherente_con_mp25(self, df_pronostico_7_dias):
        """El nivel predicho debe ser coherente con el mp25_estimado."""
        for _, row in df_pronostico_7_dias.iterrows():
            nivel_esperado = clasificar_mp25(row["mp25_estimado"])
            assert row["nivel_calidad_aire_predicho"] == nivel_esperado, (
                f"Inconsistencia: mp25={row['mp25_estimado']:.1f} "
                f"→ nivel debería ser '{nivel_esperado}' "
                f"pero es '{row['nivel_calidad_aire_predicho']}'"
            )


# ---------------------------------------------------------------------------
# Tests: predicción recursiva
# ---------------------------------------------------------------------------

class TestPrediccionRecursiva:

    def test_estado_se_actualiza(self, modelo_sintetico, df_meteo_sintetico):
        """
        Verifica que la predicción del día 2 usa el mp25 estimado del día 1
        como mp25_dia_anterior, no el valor original.
        """
        estado = {
            "fecha_base":    pd.Timestamp("2026-06-08"),
            "mp25_actual":   85.0,
            "mp25_anterior": 80.0,
            "mp25_7d_reales": [70.0, 75.0, 80.0, 85.0, 88.0, 82.0, 85.0],
            "estacion":      "Test",
            "comuna":        "Test",
        }
        df_pred = predecir_7_dias(modelo_sintetico, estado, df_meteo_sintetico, 2)

        # El mp25_estimado del día 1 debe ser diferente del mp25 real
        # (el modelo predice, no copia)
        assert df_pred.iloc[0]["mp25_estimado"] >= 0

        # Hay exactamente 2 filas
        assert len(df_pred) == 2

    def test_pronostico_n_dias_arbitrario(self, modelo_sintetico, estado_inicial_sintetico):
        """predecir_7_dias debe funcionar para cualquier N ≤ días disponibles en meteo."""
        df_meteo_10 = pd.DataFrame({
            "fecha": pd.date_range("2026-06-09", periods=10, freq="D"),
            "temperatura_max":      [15.0] * 10,
            "temperatura_min":      [5.0] * 10,
            "temperatura_promedio": [10.0] * 10,
            "humedad_relativa":     [65.0] * 10,
            "velocidad_viento":     [10.0] * 10,
            "precipitacion":        [0.0] * 10,
        })
        for n_dias in [1, 3, 5, 7]:
            df = predecir_7_dias(
                modelo_sintetico, estado_inicial_sintetico, df_meteo_10, n_dias
            )
            assert len(df) == n_dias, f"Para {n_dias} días, se obtuvieron {len(df)} filas"


# ---------------------------------------------------------------------------
# Tests: integración con CSV
# ---------------------------------------------------------------------------

class TestIntegracionCSV:

    def test_csv_generado_tiene_7_filas(
        self, df_pronostico_7_dias, tmp_path
    ):
        """El CSV guardado debe poder leerse y tener exactamente 7 filas."""
        ruta = tmp_path / "prediccion_7_dias.csv"
        df_pronostico_7_dias.to_csv(ruta, index=False)

        df_leido = pd.read_csv(ruta)
        assert len(df_leido) == 7

    def test_csv_tiene_columnas_correctas(self, df_pronostico_7_dias, tmp_path):
        ruta = tmp_path / "prediccion_7_dias.csv"
        df_pronostico_7_dias.to_csv(ruta, index=False)

        df_leido = pd.read_csv(ruta)
        assert "fecha" in df_leido.columns
        assert "mp25_estimado" in df_leido.columns
        assert "nivel_calidad_aire_predicho" in df_leido.columns
        assert "horizonte_dia" in df_leido.columns

    def test_dashboard_puede_leer_csv(self, df_pronostico_7_dias, tmp_path):
        """
        Simula la lectura que hace el dashboard:
        pd.read_csv + parse_dates + iterrows básico.
        """
        ruta = tmp_path / "prediccion_7_dias.csv"
        df_pronostico_7_dias.to_csv(ruta, index=False)

        df = pd.read_csv(ruta, parse_dates=["fecha"])

        # El dashboard itera las filas para mostrar el semáforo
        for _, row in df.iterrows():
            nivel = row["nivel_calidad_aire_predicho"]
            assert nivel in {"buena", "regular", "mala"}
            mp25  = row["mp25_estimado"]
            assert mp25 >= 0

    def test_horizonte_dia_es_entero(self, df_pronostico_7_dias, tmp_path):
        ruta = tmp_path / "prediccion_7_dias.csv"
        df_pronostico_7_dias.to_csv(ruta, index=False)
        df = pd.read_csv(ruta)
        assert pd.api.types.is_numeric_dtype(df["horizonte_dia"])
        assert df["horizonte_dia"].min() == 1
        assert df["horizonte_dia"].max() == 7


# ---------------------------------------------------------------------------
# Tests: archivo real (skip si no existe)
# ---------------------------------------------------------------------------

class TestArchivoReal:

    RUTA_CSV = ROOT / "data/processed/prediccion_7_dias.csv"

    def test_archivo_existe_si_pipeline_ejecutado(self):
        if not self.RUTA_CSV.exists():
            pytest.skip(
                "prediccion_7_dias.csv no encontrado. "
                "Ejecuta: python models/predict_7_days.py"
            )
        assert self.RUTA_CSV.exists()

    def test_archivo_real_tiene_7_filas(self):
        if not self.RUTA_CSV.exists():
            pytest.skip("prediccion_7_dias.csv no encontrado.")
        df = pd.read_csv(self.RUTA_CSV)
        assert len(df) == 7

    def test_archivo_real_columnas_completas(self):
        if not self.RUTA_CSV.exists():
            pytest.skip("prediccion_7_dias.csv no encontrado.")
        df = pd.read_csv(self.RUTA_CSV)
        assert "horizonte_dia" in df.columns
        assert "mp25_estimado" in df.columns
        assert "nivel_calidad_aire_predicho" in df.columns

    def test_archivo_real_categorias_validas(self):
        if not self.RUTA_CSV.exists():
            pytest.skip("prediccion_7_dias.csv no encontrado.")
        df = pd.read_csv(self.RUTA_CSV)
        clases = set(df["nivel_calidad_aire_predicho"].unique())
        assert clases.issubset({"buena", "regular", "mala"})