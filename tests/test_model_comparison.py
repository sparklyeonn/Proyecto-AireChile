"""
tests/test_model_comparison.py
===============================
AireChile Analytics — Tests de comparación de modelos ML.

Verifica que los scripts de comparación existen, se pueden importar,
tienen las funciones esperadas, y generan los archivos correctos
cuando se ejecutan con un dataset sintético pequeño.

Los tests usan datasets temporales (tmp_path de pytest) para no
depender de archivos pesados ni rutas absolutas.

Ejecutar:
    pytest tests/test_model_comparison.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixture: dataset sintético pequeño y rápido
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset_sintetico(tmp_path_factory):
    """
    Dataset sintético de 200 filas con las columnas del proyecto.
    Suficientemente pequeño para que los tests corran rápido.
    """
    tmp = tmp_path_factory.mktemp("data")
    np.random.seed(42)
    n = 200

    fechas = pd.date_range("2024-01-01", periods=n, freq="D")
    mp25 = np.array([
        np.random.uniform(50, 100) if 5 <= f.month <= 8
        else np.random.uniform(5, 45)
        for f in fechas
    ])

    def clasificar(v):
        if v <= 25: return "buena"
        if v <= 50: return "regular"
        return "mala"

    mp25_ant  = np.roll(mp25, 1); mp25_ant[0] = np.nan
    mp25_7d   = pd.Series(mp25).shift(1).rolling(7, min_periods=3).mean().values
    target    = [clasificar(mp25[i+1]) if i+1 < n else None for i in range(n)]

    df = pd.DataFrame({
        "fecha":             fechas,
        "estacion":          "Puente Alto",
        "comuna":            "Puente Alto",
        "mp25":              mp25.round(2),
        "estado_registro":   "validado",
        "nivel_calidad_aire": [clasificar(v) for v in mp25],
        "mes":               [f.month for f in fechas],
        "dia_semana":        [f.dayofweek for f in fechas],
        "mp25_dia_anterior": mp25_ant.round(2),
        "mp25_promedio_7d":  mp25_7d.round(2),
        "temperatura_max":   np.random.uniform(8, 28, n).round(1),
        "temperatura_min":   np.random.uniform(2, 15, n).round(1),
        "temperatura_promedio": np.random.uniform(5, 22, n).round(1),
        "humedad_relativa":  np.random.uniform(40, 90, n).round(1),
        "velocidad_viento":  np.random.uniform(5, 35, n).round(1),
        "precipitacion":     np.where(np.random.random(n) > 0.85,
                                      np.random.uniform(0, 15, n), 0).round(1),
        "nivel_calidad_aire_dia_siguiente": target,
    })

    ruta = tmp / "dataset_modelo_base.csv"
    df.to_csv(ruta, index=False)
    return ruta


# ---------------------------------------------------------------------------
# Tests: existencia e importación de archivos
# ---------------------------------------------------------------------------

class TestExistenciaArchivos:

    def test_compare_classification_existe(self):
        ruta = ROOT / "models" / "compare_classification_models.py"
        assert ruta.exists(), (
            "models/compare_classification_models.py no encontrado. "
            "Asegúrate de haber creado el archivo."
        )

    def test_compare_regression_existe(self):
        ruta = ROOT / "models" / "compare_regression_models.py"
        assert ruta.exists(), (
            "models/compare_regression_models.py no encontrado."
        )

    def test_compare_classification_importable(self):
        """El módulo debe importarse sin errores de sintaxis o dependencias."""
        try:
            import models.compare_classification_models as m
            assert m is not None
        except ImportError as e:
            pytest.fail(f"No se pudo importar compare_classification_models: {e}")

    def test_compare_regression_importable(self):
        try:
            import models.compare_regression_models as m
            assert m is not None
        except ImportError as e:
            pytest.fail(f"No se pudo importar compare_regression_models: {e}")


# ---------------------------------------------------------------------------
# Tests: funciones principales existen
# ---------------------------------------------------------------------------

class TestFuncionesClasificacion:

    def test_funcion_cargar_dataset_existe(self):
        from models.compare_classification_models import cargar_dataset
        assert callable(cargar_dataset)

    def test_funcion_comparar_modelos_existe(self):
        from models.compare_classification_models import comparar_modelos
        assert callable(comparar_modelos)

    def test_funcion_definir_modelos_existe(self):
        from models.compare_classification_models import definir_modelos
        assert callable(definir_modelos)

    def test_funcion_tuning_existe(self):
        from models.compare_classification_models import tuning_random_forest
        assert callable(tuning_random_forest)

    def test_funcion_principal_existe(self):
        from models.compare_classification_models import compare_classification_models
        assert callable(compare_classification_models)

    def test_modelos_definidos_son_4(self):
        from models.compare_classification_models import definir_modelos
        modelos = definir_modelos()
        assert len(modelos) == 4

    def test_nombres_modelos_clasificacion(self):
        from models.compare_classification_models import definir_modelos
        modelos = definir_modelos()
        esperados = {
            "LogisticRegression",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
        }
        assert set(modelos.keys()) == esperados


class TestFuncionesRegresion:

    def test_funcion_cargar_dataset_existe(self):
        from models.compare_regression_models import cargar_dataset
        assert callable(cargar_dataset)

    def test_funcion_comparar_modelos_existe(self):
        from models.compare_regression_models import comparar_modelos
        assert callable(comparar_modelos)

    def test_funcion_definir_modelos_existe(self):
        from models.compare_regression_models import definir_modelos
        assert callable(definir_modelos)

    def test_funcion_tuning_existe(self):
        from models.compare_regression_models import tuning_random_forest_regressor
        assert callable(tuning_random_forest_regressor)

    def test_funcion_principal_existe(self):
        from models.compare_regression_models import compare_regression_models
        assert callable(compare_regression_models)

    def test_modelos_definidos_son_4(self):
        from models.compare_regression_models import definir_modelos
        modelos = definir_modelos()
        assert len(modelos) == 4

    def test_nombres_modelos_regresion(self):
        from models.compare_regression_models import definir_modelos
        modelos = definir_modelos()
        esperados = {
            "LinearRegression",
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
        }
        assert set(modelos.keys()) == esperados


# ---------------------------------------------------------------------------
# Tests: ejecución con dataset sintético
# ---------------------------------------------------------------------------

class TestEjecucionClasificacion:

    def test_pipeline_completo_genera_csv(self, dataset_sintetico, tmp_path):
        """El pipeline de clasificación debe generar el CSV de comparación."""
        from models.compare_classification_models import compare_classification_models

        df = compare_classification_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics",
            run_tuning=False,  # sin tuning para velocidad en tests
        )

        assert not df.empty
        csv = tmp_path / "metrics" / "classification_model_comparison.csv"
        assert csv.exists(), "classification_model_comparison.csv no generado"

    def test_csv_tiene_columnas_correctas(self, dataset_sintetico, tmp_path):
        from models.compare_classification_models import compare_classification_models

        df = compare_classification_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_cols",
            run_tuning=False,
        )

        columnas_requeridas = {"modelo", "accuracy", "precision_weighted",
                               "recall_weighted", "f1_weighted"}
        assert columnas_requeridas.issubset(set(df.columns))

    def test_csv_tiene_4_modelos(self, dataset_sintetico, tmp_path):
        from models.compare_classification_models import compare_classification_models

        df = compare_classification_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_4",
            run_tuning=False,
        )
        assert len(df) == 4

    def test_genera_summary_json(self, dataset_sintetico, tmp_path):
        from models.compare_classification_models import compare_classification_models

        compare_classification_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_summary",
            run_tuning=False,
        )

        json_path = tmp_path / "metrics_summary" / "classification_model_comparison_summary.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "mejor_modelo" in data
        assert "metrica_seleccion" in data
        assert data["metrica_seleccion"] == "f1_weighted"
        assert "justificacion" in data
        assert data["cantidad_modelos_comparados"] == 4

    def test_metricas_en_rango_valido(self, dataset_sintetico, tmp_path):
        """Accuracy y F1 deben estar entre 0 y 1."""
        from models.compare_classification_models import compare_classification_models

        df = compare_classification_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_rango",
            run_tuning=False,
        )

        assert (df["accuracy"].between(0, 1)).all()
        assert (df["f1_weighted"].between(0, 1)).all()

    def test_mejor_modelo_tiene_mayor_f1(self, dataset_sintetico, tmp_path):
        """El primer modelo del DataFrame debe tener el mayor F1."""
        from models.compare_classification_models import compare_classification_models

        df = compare_classification_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_orden",
            run_tuning=False,
        )

        assert df.iloc[0]["f1_weighted"] == df["f1_weighted"].max()

    def test_error_sin_dataset(self, tmp_path):
        """Debe lanzar FileNotFoundError si el dataset no existe."""
        from models.compare_classification_models import compare_classification_models

        with pytest.raises(FileNotFoundError):
            compare_classification_models(
                dataset_path=tmp_path / "no_existe.csv",
                metrics_dir=tmp_path / "metrics",
                run_tuning=False,
            )


class TestEjecucionRegresion:

    def test_pipeline_completo_genera_csv(self, dataset_sintetico, tmp_path):
        """El pipeline de regresión debe generar el CSV de comparación."""
        from models.compare_regression_models import compare_regression_models

        df = compare_regression_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_reg",
            run_tuning=False,
        )

        assert not df.empty
        csv = tmp_path / "metrics_reg" / "regression_model_comparison.csv"
        assert csv.exists()

    def test_csv_tiene_columnas_correctas(self, dataset_sintetico, tmp_path):
        from models.compare_regression_models import compare_regression_models

        df = compare_regression_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_reg_cols",
            run_tuning=False,
        )

        columnas_requeridas = {"modelo", "mae", "rmse", "r2"}
        assert columnas_requeridas.issubset(set(df.columns))

    def test_csv_tiene_4_modelos(self, dataset_sintetico, tmp_path):
        from models.compare_regression_models import compare_regression_models

        df = compare_regression_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_reg_4",
            run_tuning=False,
        )
        assert len(df) == 4

    def test_genera_summary_json(self, dataset_sintetico, tmp_path):
        from models.compare_regression_models import compare_regression_models

        compare_regression_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_reg_summary",
            run_tuning=False,
        )

        json_path = tmp_path / "metrics_reg_summary" / "regression_model_comparison_summary.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "mejor_modelo" in data
        assert data["metrica_seleccion"] == "rmse"
        assert data["cantidad_modelos_comparados"] == 4

    def test_mae_rmse_positivos(self, dataset_sintetico, tmp_path):
        """MAE y RMSE siempre deben ser positivos."""
        from models.compare_regression_models import compare_regression_models

        df = compare_regression_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_reg_pos",
            run_tuning=False,
        )

        assert (df["mae"] >= 0).all()
        assert (df["rmse"] >= 0).all()

    def test_mejor_modelo_tiene_menor_rmse(self, dataset_sintetico, tmp_path):
        """El primer modelo del DataFrame debe tener el menor RMSE."""
        from models.compare_regression_models import compare_regression_models

        df = compare_regression_models(
            dataset_path=dataset_sintetico,
            metrics_dir=tmp_path / "metrics_reg_orden",
            run_tuning=False,
        )

        assert df.iloc[0]["rmse"] == df["rmse"].min()

    def test_construye_target_si_no_existe(self, tmp_path):
        """Si mp25_dia_siguiente no existe, debe construirla con shift(-1)."""
        from models.compare_regression_models import compare_regression_models

        # Dataset sin la columna target
        df = pd.DataFrame({
            "mp25":              np.random.uniform(10, 80, 150),
            "mp25_dia_anterior": np.random.uniform(10, 80, 150),
            "mp25_promedio_7d":  np.random.uniform(10, 80, 150),
            "mes":               np.random.randint(1, 13, 150),
            "dia_semana":        np.random.randint(0, 7, 150),
            "temperatura_max":   np.random.uniform(5, 30, 150),
            "temperatura_min":   np.random.uniform(0, 15, 150),
            "temperatura_promedio": np.random.uniform(3, 22, 150),
            "humedad_relativa":  np.random.uniform(40, 90, 150),
            "velocidad_viento":  np.random.uniform(5, 35, 150),
            "precipitacion":     np.zeros(150),
        })
        ruta = tmp_path / "sin_target.csv"
        df.to_csv(ruta, index=False)

        # No debe lanzar error
        result = compare_regression_models(
            dataset_path=ruta,
            metrics_dir=tmp_path / "metrics_shift",
            run_tuning=False,
        )
        assert not result.empty

    def test_error_sin_dataset(self, tmp_path):
        from models.compare_regression_models import compare_regression_models

        with pytest.raises(FileNotFoundError):
            compare_regression_models(
                dataset_path=tmp_path / "no_existe.csv",
                metrics_dir=tmp_path / "metrics",
                run_tuning=False,
            )