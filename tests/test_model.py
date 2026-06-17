"""
tests/test_model.py
===================
AireChile Analytics — Tests unitarios para train_model.py y predict.py

Todos los tests usan datasets sintéticos pequeños (no el dataset real
ni el modelo .pkl) para garantizar velocidad y reproducibilidad.

Cobertura:
    - Dataset no está vacío
    - Variable objetivo presente
    - Features principales presentes
    - El modelo se entrena con datos pequeños
    - El split temporal es correcto (no aleatorio)
    - El modelo predice clases válidas
    - predict_proba devuelve probabilidades que suman 1
    - guardar_artefactos crea los archivos esperados

Ejecutar:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v --tb=short
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.train_model import (
    FEATURES,
    TARGET,
    RF_PARAMS,
    cargar_dataset,
    preparar_datos,
    split_temporal,
    entrenar_modelo,
    evaluar_modelo,
    calcular_feature_importance,
    guardar_artefactos,
)
from models.predict import (
    cargar_modelo,
    obtener_ultimo_registro,
    generar_prediccion,
    FEATURES as PREDICT_FEATURES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset_sintetico():
    """
    Dataset sintético pequeño pero realista.
    120 filas (suficiente para train/test split) con distribución
    de clases similar a los datos reales de Santiago.
    """
    np.random.seed(42)
    n = 120
    fechas = pd.date_range("2024-01-01", periods=n, freq="D")

    # MP2.5 con patrón estacional (mayor en invierno: meses 5-8)
    mp25 = np.array([
        np.random.uniform(50, 100) if 5 <= f.month <= 8
        else np.random.uniform(5, 45)
        for f in fechas
    ])

    def clasificar(v):
        if v <= 25: return "buena"
        if v <= 50: return "regular"
        return "mala"

    mp25_ant = np.roll(mp25, 1); mp25_ant[0] = np.nan
    mp25_7d  = pd.Series(mp25).shift(1).rolling(7, min_periods=3).mean().values
    target   = [clasificar(mp25[i+1]) if i+1 < n else None for i in range(n)]

    return pd.DataFrame({
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
        "precipitacion":     np.where(
            np.random.random(n) > 0.85,
            np.random.uniform(0, 15, n), 0
        ).round(1),
        "nivel_calidad_aire_dia_siguiente": target,
    })


@pytest.fixture
def modelo_entrenado(dataset_sintetico):
    """Modelo RandomForest entrenado con el dataset sintético."""
    X, y = preparar_datos(dataset_sintetico)
    X_train, _, y_train, _ = split_temporal(X, y)
    return entrenar_modelo(X_train, y_train)


# ---------------------------------------------------------------------------
# Tests: validación del dataset
# ---------------------------------------------------------------------------

class TestValidacionDataset:

    def test_dataset_no_vacio(self, dataset_sintetico):
        """El dataset sintético no debe estar vacío."""
        assert not dataset_sintetico.empty
        assert len(dataset_sintetico) > 0

    def test_variable_objetivo_presente(self, dataset_sintetico):
        """La columna target debe existir."""
        assert TARGET in dataset_sintetico.columns

    def test_features_principales_presentes(self, dataset_sintetico):
        """Todas las features del modelo deben estar presentes."""
        for feature in FEATURES:
            assert feature in dataset_sintetico.columns, (
                f"Feature faltante: {feature}"
            )

    def test_fecha_presente(self, dataset_sintetico):
        assert "fecha" in dataset_sintetico.columns

    def test_columna_target_tiene_clases_validas(self, dataset_sintetico):
        clases_validas = {"buena", "regular", "mala"}
        clases_presentes = set(
            dataset_sintetico[TARGET].dropna().unique()
        )
        assert clases_presentes.issubset(clases_validas), (
            f"Clases inválidas: {clases_presentes - clases_validas}"
        )

    def test_mp25_es_numerico(self, dataset_sintetico):
        assert pd.api.types.is_numeric_dtype(dataset_sintetico["mp25"])

    def test_cargar_dataset_lanza_error_si_no_existe(self, tmp_path):
        """cargar_dataset() debe lanzar FileNotFoundError si el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            cargar_dataset(tmp_path / "no_existe.csv")

    def test_cargar_dataset_lanza_error_si_faltan_columnas(self, tmp_path):
        """cargar_dataset() debe lanzar ValueError si faltan columnas."""
        df_incompleto = pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=5)})
        ruta = tmp_path / "incompleto.csv"
        df_incompleto.to_csv(ruta, index=False)
        with pytest.raises(ValueError, match="faltantes"):
            cargar_dataset(ruta)


# ---------------------------------------------------------------------------
# Tests: preparar_datos
# ---------------------------------------------------------------------------

class TestPrepararDatos:

    def test_retorna_x_e_y(self, dataset_sintetico):
        X, y = preparar_datos(dataset_sintetico)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_tiene_features_correctas(self, dataset_sintetico):
        X, _ = preparar_datos(dataset_sintetico)
        assert list(X.columns) == FEATURES

    def test_filas_con_nulos_eliminadas(self):
        """
        Las filas con NaN en features deben eliminarse.
        Se usa un dataset de 110 filas (>100 mínimo requerido).
        """
        n = 110
        vals_mp25 = [float(i + 1) for i in range(n)]
        vals_mp25[50] = None  # esta fila debe eliminarse

        df = pd.DataFrame({
            **{f: ([1.0] * n) for f in FEATURES},
            TARGET: ["buena"] * n,
            "fecha": pd.date_range("2024-01-01", periods=n),
        })
        df["mp25"] = vals_mp25

        X, y = preparar_datos(df)
        assert len(X) == n - 1  # la fila con None fue eliminada

    def test_no_hay_nulos_en_x(self, dataset_sintetico):
        X, _ = preparar_datos(dataset_sintetico)
        assert X.isnull().sum().sum() == 0

    def test_no_hay_nulos_en_y(self, dataset_sintetico):
        _, y = preparar_datos(dataset_sintetico)
        assert y.isnull().sum() == 0


# ---------------------------------------------------------------------------
# Tests: split temporal
# ---------------------------------------------------------------------------

class TestSplitTemporal:

    def test_proporciones_correctas(self, dataset_sintetico):
        """El split debe respetar la proporción 80/20."""
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, y_train, y_test = split_temporal(X, y, test_size=0.2)

        n_total = len(X)
        # Tolerancia de 1 fila por redondeo
        assert abs(len(X_train) - int(n_total * 0.8)) <= 1
        assert abs(len(X_test)  - int(n_total * 0.2)) <= 1

    def test_sin_solapamiento(self, dataset_sintetico):
        """Train y test no deben compartir índices."""
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, _, _ = split_temporal(X, y)

        indices_train = set(X_train.index)
        indices_test  = set(X_test.index)
        assert len(indices_train & indices_test) == 0

    def test_train_precede_a_test(self, dataset_sintetico):
        """
        El índice máximo de train debe ser menor que el mínimo de test.
        Esto garantiza que no hay data leakage temporal.
        """
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, _, _ = split_temporal(X, y)

        assert X_train.index.max() < X_test.index.min(), (
            "DATA LEAKAGE: hay índices de test que son anteriores al "
            "máximo índice de train. El split no es temporal."
        )

    def test_cubre_todos_los_datos(self, dataset_sintetico):
        """Train + test debe cubrir exactamente todos los datos."""
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, _, _ = split_temporal(X, y)
        assert len(X_train) + len(X_test) == len(X)


# ---------------------------------------------------------------------------
# Tests: entrenamiento del modelo
# ---------------------------------------------------------------------------

class TestEntrenamientoModelo:

    def test_modelo_se_entrena(self, dataset_sintetico):
        """El modelo debe entrenarse sin lanzar excepciones."""
        X, y = preparar_datos(dataset_sintetico)
        X_train, _, y_train, _ = split_temporal(X, y)
        modelo = entrenar_modelo(X_train, y_train)
        assert modelo is not None

    def test_modelo_es_random_forest(self, modelo_entrenado):
        assert isinstance(modelo_entrenado, RandomForestClassifier)

    def test_modelo_tiene_clases(self, modelo_entrenado):
        clases = set(modelo_entrenado.classes_)
        assert clases.issubset({"buena", "regular", "mala"})

    def test_modelo_tiene_n_estimators_correctos(self, modelo_entrenado):
        assert modelo_entrenado.n_estimators == RF_PARAMS["n_estimators"]

    def test_modelo_predice_sin_error(self, modelo_entrenado, dataset_sintetico):
        X, _ = preparar_datos(dataset_sintetico)
        _, X_test, _, _ = split_temporal(X, _)
        predicciones = modelo_entrenado.predict(X_test)
        assert len(predicciones) == len(X_test)

    def test_predicciones_son_clases_validas(self, modelo_entrenado, dataset_sintetico):
        """Todas las predicciones deben ser buena, regular o mala."""
        X, _ = preparar_datos(dataset_sintetico)
        _, X_test, _, _ = split_temporal(X, _)
        predicciones = modelo_entrenado.predict(X_test)
        for pred in predicciones:
            assert pred in {"buena", "regular", "mala"}, (
                f"Predicción inválida: '{pred}'"
            )

    def test_predict_proba_suma_uno(self, modelo_entrenado, dataset_sintetico):
        """Las probabilidades de cada fila deben sumar 1 (con tolerancia)."""
        X, _ = preparar_datos(dataset_sintetico)
        _, X_test, _, _ = split_temporal(X, _)
        probs = modelo_entrenado.predict_proba(X_test)
        sumas = probs.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-6 for s in sumas)

    def test_accuracy_razonable(self, modelo_entrenado, dataset_sintetico):
        """El accuracy debe ser mejor que azar (>33% para 3 clases)."""
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, y_train, y_test = split_temporal(X, y)
        modelo_entrenado.fit(X_train, y_train)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, modelo_entrenado.predict(X_test))
        assert acc > 0.33, (
            f"Accuracy {acc:.2f} es menor que azar (0.33). "
            "El modelo no está aprendiendo."
        )


# ---------------------------------------------------------------------------
# Tests: feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:

    def test_todas_features_tienen_importancia(self, modelo_entrenado):
        fi = calcular_feature_importance(modelo_entrenado, FEATURES)
        assert len(fi) == len(FEATURES)

    def test_importancias_suman_uno(self, modelo_entrenado):
        fi = calcular_feature_importance(modelo_entrenado, FEATURES)
        total = fi["importancia"].sum()
        assert abs(total - 1.0) < 1e-4

    def test_importancias_no_negativas(self, modelo_entrenado):
        fi = calcular_feature_importance(modelo_entrenado, FEATURES)
        assert (fi["importancia"] >= 0).all()

    def test_mp25_es_feature_importante(self, modelo_entrenado):
        """
        mp25 debería ser una de las features más importantes, dada la
        naturaleza del problema (predice calidad del aire con datos del aire).
        """
        fi = calcular_feature_importance(modelo_entrenado, FEATURES)
        top3 = set(fi.head(3)["feature"].tolist())
        features_mp25 = {"mp25", "mp25_dia_anterior", "mp25_promedio_7d"}
        # Al menos una de las features de mp25 debe estar en el top 3
        assert len(top3 & features_mp25) >= 1, (
            f"Ninguna feature de MP2.5 está en el top 3: {top3}"
        )


# ---------------------------------------------------------------------------
# Tests: guardar artefactos
# ---------------------------------------------------------------------------

class TestGuardarArtefactos:

    def test_archivos_creados(self, modelo_entrenado, dataset_sintetico, tmp_path):
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, y_train, y_test = split_temporal(X, y)
        modelo_entrenado.fit(X_train, y_train)
        metricas = evaluar_modelo(modelo_entrenado, X_test, y_test)
        fi = calcular_feature_importance(modelo_entrenado, FEATURES)

        model_path  = tmp_path / "model.pkl"
        metrics_dir = tmp_path / "metrics"

        guardar_artefactos(modelo_entrenado, metricas, fi, model_path, metrics_dir)

        assert model_path.exists(),            "model.pkl no fue creado"
        assert (metrics_dir / "model_metrics.json").exists()
        assert (metrics_dir / "feature_importance.csv").exists()
        assert (metrics_dir / "confusion_matrix.csv").exists()

    def test_modelo_pkl_es_cargable(self, modelo_entrenado, dataset_sintetico, tmp_path):
        """El modelo guardado debe poder cargarse y generar predicciones."""
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, y_train, y_test = split_temporal(X, y)
        modelo_entrenado.fit(X_train, y_train)
        metricas = evaluar_modelo(modelo_entrenado, X_test, y_test)
        fi = calcular_feature_importance(modelo_entrenado, FEATURES)

        model_path  = tmp_path / "model.pkl"
        metrics_dir = tmp_path / "metrics"
        guardar_artefactos(modelo_entrenado, metricas, fi, model_path, metrics_dir)

        # Cargar y predecir
        modelo_cargado = joblib.load(model_path)
        pred = modelo_cargado.predict(X_test.iloc[:1])
        assert pred[0] in {"buena", "regular", "mala"}

    def test_metrics_json_contiene_accuracy(self, modelo_entrenado, dataset_sintetico, tmp_path):
        X, y = preparar_datos(dataset_sintetico)
        X_train, X_test, y_train, y_test = split_temporal(X, y)
        modelo_entrenado.fit(X_train, y_train)
        metricas = evaluar_modelo(modelo_entrenado, X_test, y_test)
        fi = calcular_feature_importance(modelo_entrenado, FEATURES)

        model_path  = tmp_path / "model.pkl"
        metrics_dir = tmp_path / "metrics"
        guardar_artefactos(modelo_entrenado, metricas, fi, model_path, metrics_dir)

        with open(metrics_dir / "model_metrics.json") as f:
            datos = json.load(f)

        assert "accuracy" in datos
        assert 0 <= datos["accuracy"] <= 1


# ---------------------------------------------------------------------------
# Tests: predicción
# ---------------------------------------------------------------------------

class TestPrediccion:

    def test_prediccion_clase_valida(self, modelo_entrenado, dataset_sintetico):
        """La predicción debe ser una de las 3 clases válidas."""
        X, y = preparar_datos(dataset_sintetico)
        X_train, _, y_train, _ = split_temporal(X, y)
        modelo_entrenado.fit(X_train, y_train)

        # Tomar el último registro completo
        df_completo = dataset_sintetico[FEATURES + ["fecha"]].dropna()
        ultimo = df_completo.iloc[-1]
        fecha_base = pd.Timestamp(ultimo["fecha"])

        resultado = generar_prediccion(modelo_entrenado, ultimo, fecha_base)

        assert resultado["nivel_predicho"] in {"buena", "regular", "mala"}

    def test_probabilidades_suman_uno(self, modelo_entrenado, dataset_sintetico):
        X, y = preparar_datos(dataset_sintetico)
        X_train, _, y_train, _ = split_temporal(X, y)
        modelo_entrenado.fit(X_train, y_train)

        df_completo = dataset_sintetico[FEATURES + ["fecha"]].dropna()
        ultimo = df_completo.iloc[-1]
        fecha_base = pd.Timestamp(ultimo["fecha"])

        resultado = generar_prediccion(modelo_entrenado, ultimo, fecha_base)

        total_prob = (
            resultado.get("prob_buena", 0) +
            resultado.get("prob_regular", 0) +
            resultado.get("prob_mala", 0)
        )
        assert abs(total_prob - 1.0) < 0.01, (
            f"Las probabilidades suman {total_prob}, esperado ~1.0"
        )

    def test_fecha_predicha_es_dia_siguiente(self, modelo_entrenado, dataset_sintetico):
        from datetime import timedelta
        X, y = preparar_datos(dataset_sintetico)
        X_train, _, y_train, _ = split_temporal(X, y)
        modelo_entrenado.fit(X_train, y_train)

        df_completo = dataset_sintetico[FEATURES + ["fecha"]].dropna()
        ultimo = df_completo.iloc[-1]
        fecha_base = pd.Timestamp(ultimo["fecha"])

        resultado = generar_prediccion(modelo_entrenado, ultimo, fecha_base)

        fecha_esperada = (fecha_base + timedelta(days=1)).date()
        assert resultado["fecha_predicha"] == str(fecha_esperada)

    def test_cargar_modelo_lanza_error_si_no_existe(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cargar_modelo(tmp_path / "inexistente.pkl")