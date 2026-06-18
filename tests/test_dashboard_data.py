"""
tests/test_dashboard_data.py
============================
AireChile Analytics — Tests de integridad de datos para el dashboard.

Verifica que los archivos generados por el pipeline tienen la estructura
correcta antes de que el dashboard intente leerlos. Detecta problemas
antes de abrir el navegador.

Estos tests NO verifican la interfaz visual (eso requeriría Selenium).
Verifican que los archivos fuente del dashboard son válidos.

Ejecutar:
    pytest tests/test_dashboard_data.py -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Rutas de archivos (relativas a la raíz del proyecto)
ROOT = Path(__file__).parent.parent

PATHS = {
    "dataset":    ROOT / "data/processed/dataset_modelo_base.csv",
    "prediccion": ROOT / "data/processed/prediccion_actual.csv",
    "metrics":    ROOT / "models/metrics/model_metrics.json",
    "fi":         ROOT / "models/metrics/feature_importance.csv",
    "cm":         ROOT / "models/metrics/confusion_matrix.csv",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    """Carga el dataset base del modelo una vez para todos los tests del módulo."""
    if not PATHS["dataset"].exists():
        pytest.skip(
            f"dataset_modelo_base.csv no encontrado en {PATHS['dataset']}. "
            "Ejecuta: python etl/etl_meteo_main.py"
        )
    return pd.read_csv(PATHS["dataset"], parse_dates=["fecha"])


@pytest.fixture(scope="module")
def prediccion():
    """Carga el CSV de predicción si existe."""
    if not PATHS["prediccion"].exists():
        pytest.skip(
            "prediccion_actual.csv no encontrado. "
            "Ejecuta: python models/predict.py"
        )
    return pd.read_csv(PATHS["prediccion"])


@pytest.fixture(scope="module")
def metricas():
    """Carga las métricas del modelo si existen."""
    if not PATHS["metrics"].exists():
        pytest.skip(
            "model_metrics.json no encontrado. "
            "Ejecuta: python models/train_model.py"
        )
    with open(PATHS["metrics"], encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def feature_importance():
    """Carga feature_importance.csv si existe."""
    if not PATHS["fi"].exists():
        pytest.skip("feature_importance.csv no encontrado.")
    return pd.read_csv(PATHS["fi"])


@pytest.fixture(scope="module")
def confusion_matrix():
    """Carga confusion_matrix.csv si existe."""
    if not PATHS["cm"].exists():
        pytest.skip("confusion_matrix.csv no encontrado.")
    return pd.read_csv(PATHS["cm"], index_col=0)


# ---------------------------------------------------------------------------
# Tests: dataset_modelo_base.csv
# ---------------------------------------------------------------------------

class TestDatasetModeloBase:

    def test_archivo_existe(self):
        """El archivo dataset_modelo_base.csv debe existir."""
        assert PATHS["dataset"].exists(), (
            f"Archivo no encontrado: {PATHS['dataset']}\n"
            "Ejecuta: python etl/etl_meteo_main.py"
        )

    def test_dataset_no_vacio(self, dataset):
        assert not dataset.empty
        assert len(dataset) > 0

    def test_columna_fecha_existe(self, dataset):
        assert "fecha" in dataset.columns

    def test_columna_fecha_es_datetime(self, dataset):
        assert pd.api.types.is_datetime64_any_dtype(dataset["fecha"])

    def test_columna_mp25_existe(self, dataset):
        assert "mp25" in dataset.columns

    def test_columna_mp25_es_numerica(self, dataset):
        assert pd.api.types.is_numeric_dtype(dataset["mp25"])

    def test_columna_mp25_no_negativa(self, dataset):
        """MP2.5 no puede ser negativo físicamente."""
        mp25_validos = dataset["mp25"].dropna()
        assert (mp25_validos >= 0).all(), (
            f"Hay {(mp25_validos < 0).sum()} valores negativos en mp25."
        )

    def test_variable_objetivo_existe(self, dataset):
        assert "nivel_calidad_aire_dia_siguiente" in dataset.columns, (
            "Falta la variable objetivo. "
            "Verifica etl/transform_sinca.py"
        )

    def test_variable_objetivo_clases_validas(self, dataset):
        clases_validas = {"buena", "regular", "mala"}
        clases_presentes = set(
            dataset["nivel_calidad_aire_dia_siguiente"].dropna().unique()
        )
        invalidas = clases_presentes - clases_validas
        assert not invalidas, f"Clases inválidas encontradas: {invalidas}"

    def test_columnas_sinca_presentes(self, dataset):
        """Columnas producidas por el ETL SINCA."""
        cols_sinca = [
            "estacion", "comuna", "estado_registro", "nivel_calidad_aire",
            "mes", "dia_semana", "mp25_dia_anterior", "mp25_promedio_7d",
        ]
        for col in cols_sinca:
            assert col in dataset.columns, f"Columna SINCA faltante: {col}"

    def test_columnas_meteo_presentes(self, dataset):
        """Columnas producidas por el ETL Open-Meteo."""
        cols_meteo = [
            "temperatura_max", "temperatura_min", "temperatura_promedio",
            "humedad_relativa", "velocidad_viento", "precipitacion",
        ]
        for col in cols_meteo:
            assert col in dataset.columns, f"Columna meteorológica faltante: {col}"

    def test_rango_fechas_razonable(self, dataset):
        """El dataset debe cubrir al menos 1 año de datos."""
        rango = (dataset["fecha"].max() - dataset["fecha"].min()).days
        assert rango >= 365, (
            f"El dataset solo cubre {rango} días. "
            "Se necesita al menos 1 año para entrenar el modelo."
        )

    def test_sin_filas_completamente_vacias(self, dataset):
        n_vacias = dataset.isnull().all(axis=1).sum()
        assert n_vacias == 0, f"Hay {n_vacias} filas completamente vacías."

    def test_fechas_sin_duplicados(self, dataset):
        """No debe haber dos filas para la misma fecha y estación."""
        duplicados = dataset.duplicated(subset=["fecha", "estacion"]).sum()
        assert duplicados == 0, (
            f"Hay {duplicados} fechas duplicadas por estación."
        )


# ---------------------------------------------------------------------------
# Tests: prediccion_actual.csv
# ---------------------------------------------------------------------------

class TestPrediccionActual:

    def test_archivo_existe(self):
        """Si predict.py fue ejecutado, el archivo debe existir."""
        if not PATHS["prediccion"].exists():
            pytest.skip("prediccion_actual.csv aún no generado.")
        assert PATHS["prediccion"].exists()

    def test_tiene_nivel_predicho(self, prediccion):
        assert "nivel_predicho" in prediccion.columns

    def test_nivel_predicho_es_clase_valida(self, prediccion):
        nivel = prediccion.iloc[0]["nivel_predicho"]
        assert nivel in {"buena", "regular", "mala"}, (
            f"Nivel predicho inválido: '{nivel}'"
        )

    def test_tiene_probabilidad(self, prediccion):
        assert "probabilidad_predicho" in prediccion.columns

    def test_probabilidad_entre_0_y_1(self, prediccion):
        prob = float(prediccion.iloc[0]["probabilidad_predicho"])
        assert 0 <= prob <= 1, f"Probabilidad fuera de rango: {prob}"

    def test_tiene_fecha_predicha(self, prediccion):
        assert "fecha_predicha" in prediccion.columns

    def test_tiene_fecha_base(self, prediccion):
        assert "fecha_base" in prediccion.columns

    def test_probabilidades_suman_uno(self, prediccion):
        cols_prob = ["prob_buena", "prob_regular", "prob_mala"]
        if all(c in prediccion.columns for c in cols_prob):
            total = (
                float(prediccion.iloc[0]["prob_buena"]) +
                float(prediccion.iloc[0]["prob_regular"]) +
                float(prediccion.iloc[0]["prob_mala"])
            )
            assert abs(total - 1.0) < 0.01, (
                f"Las probabilidades no suman 1: {total}"
            )


# ---------------------------------------------------------------------------
# Tests: métricas del modelo
# ---------------------------------------------------------------------------

class TestMetricasModelo:

    def test_archivo_metrics_existe(self):
        if not PATHS["metrics"].exists():
            pytest.skip("model_metrics.json aún no generado.")
        assert PATHS["metrics"].exists()

    def test_metricas_se_leen_como_json(self, metricas):
        assert isinstance(metricas, dict)

    def test_accuracy_presente(self, metricas):
        assert "accuracy" in metricas

    def test_accuracy_entre_0_y_1(self, metricas):
        acc = metricas["accuracy"]
        assert 0 <= acc <= 1, f"Accuracy fuera de rango: {acc}"

    def test_accuracy_mejor_que_azar(self, metricas):
        """Accuracy debe superar el azar para 3 clases (33%)."""
        assert metricas["accuracy"] > 0.33, (
            f"Accuracy {metricas['accuracy']:.2f} no supera el azar (0.33)."
        )

    def test_f1_presente(self, metricas):
        assert "f1_weighted" in metricas

    def test_metricas_por_clase_presentes(self, metricas):
        assert "metricas_por_clase" in metricas
        clases = set(metricas["metricas_por_clase"].keys())
        assert clases == {"buena", "regular", "mala"}

    def test_confusion_matrix_presente(self, metricas):
        assert "confusion_matrix" in metricas

    def test_feature_importance_legible(self, feature_importance):
        assert "feature" in feature_importance.columns
        assert "importancia" in feature_importance.columns
        assert len(feature_importance) > 0

    def test_importancias_suman_uno(self, feature_importance):
        total = feature_importance["importancia"].sum()
        assert abs(total - 1.0) < 0.01, (
            f"Las importancias suman {total}, no 1.0"
        )

    def test_confusion_matrix_legible(self, confusion_matrix):
        assert not confusion_matrix.empty
        # Debe ser cuadrada (n×n donde n = número de clases)
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]