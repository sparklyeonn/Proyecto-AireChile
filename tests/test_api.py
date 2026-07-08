"""
tests/test_api.py
=================
AireChile Analytics — Tests automatizados de la API REST.

Usa TestClient de FastAPI para hacer peticiones HTTP reales
contra la aplicación sin levantar un servidor externo.

Los tests están diseñados para pasar incluso cuando los archivos
de datos no existen (escenario de CI/CD sin datos generados),
verificando que la API maneja esos casos con errores controlados
en lugar de crashear con traceback de Python.

Cobertura:
    - GET /health → siempre debe responder 200
    - GET /stations → siempre debe responder 200
    - GET /prediction/current → 200 con datos o 404 sin archivo
    - GET /forecast/7-days → 200 con datos o 404 sin archivo
    - GET /metrics → 200 con datos o 404 sin archivo
    - Errores controlados devuelven JSON con campo "error"
    - Estructura de respuesta válida cuando hay datos

Ejecutar:
    pytest tests/test_api.py -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests: /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_responde_200(self):
        """El endpoint de salud siempre debe responder 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_tiene_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_tiene_proyecto(self):
        data = client.get("/health").json()
        assert data["project"] == "AireChile Analytics"

    def test_health_tiene_version(self):
        data = client.get("/health").json()
        assert "version" in data


# ---------------------------------------------------------------------------
# Tests: /stations
# ---------------------------------------------------------------------------

class TestStations:

    def test_stations_responde_200(self):
        """El endpoint de estaciones siempre responde 200."""
        resp = client.get("/stations")
        assert resp.status_code == 200

    def test_stations_retorna_lista(self):
        data = client.get("/stations").json()
        assert isinstance(data, list)

    def test_stations_tiene_al_menos_una_estacion(self):
        data = client.get("/stations").json()
        assert len(data) >= 1

    def test_stations_tiene_campos_obligatorios(self):
        estacion = client.get("/stations").json()[0]
        campos = ["estacion", "comuna", "latitud", "longitud",
                  "fuente_calidad_aire", "fuente_meteorologia"]
        for campo in campos:
            assert campo in estacion, f"Campo faltante: {campo}"

    def test_stations_coordenadas_correctas(self):
        """Puente Alto debe tener coordenadas aproximadas de Santiago."""
        estacion = client.get("/stations").json()[0]
        assert -35 < estacion["latitud"] < -30
        assert -72 < estacion["longitud"] < -68


# ---------------------------------------------------------------------------
# Tests: /prediction/current
# ---------------------------------------------------------------------------

class TestPredictionCurrent:

    def test_sin_archivo_devuelve_404_no_crash(self):
        """
        Si prediccion_actual.csv no existe, la API debe devolver 404
        con mensaje controlado, no un error 500 de Python.
        """
        resp = client.get("/prediction/current")
        # Puede ser 200 (si hay archivo) o 404 (si no hay)
        assert resp.status_code in (200, 404)

    def test_404_tiene_campo_error(self):
        """Si es 404, el body debe tener campo 'error'."""
        resp = client.get("/prediction/current")
        if resp.status_code == 404:
            body = resp.json()
            assert "detail" in body
            assert "error" in body["detail"]

    def test_200_tiene_campos_obligatorios(self, tmp_path):
        """
        Si el archivo existe con formato correcto, la respuesta
        debe tener los campos obligatorios.
        """
        # Crear archivo temporal con datos mínimos válidos
        csv = tmp_path / "prediccion_actual.csv"
        pd.DataFrame([{
            "fecha_base": "2026-06-08",
            "fecha_predicha": "2026-06-09",
            "nivel_predicho": "mala",
            "probabilidad_predicho": 0.97,
            "prob_buena": 0.01,
            "prob_regular": 0.02,
            "prob_mala": 0.97,
            "mp25_base": 87.86,
        }]).to_csv(csv, index=False)

        # Parchear la ruta en services
        import api.services as svc
        ruta_original = svc.RUTA_PREDICCION
        svc.RUTA_PREDICCION = csv
        try:
            resp = client.get("/prediction/current")
            assert resp.status_code == 200
            data = resp.json()
            assert "nivel_calidad_aire_predicho" in data
            assert "recomendacion" in data
            assert "fuente" in data
            assert data["fuente"] == "modelo_clasificador"
        finally:
            svc.RUTA_PREDICCION = ruta_original

    def test_recomendacion_no_vacia(self, tmp_path):
        """La recomendación nunca debe ser un string vacío."""
        csv = tmp_path / "prediccion_actual.csv"
        pd.DataFrame([{
            "fecha_base": "2026-06-08",
            "fecha_predicha": "2026-06-09",
            "nivel_predicho": "buena",
            "probabilidad_predicho": 0.8,
            "prob_buena": 0.8, "prob_regular": 0.1, "prob_mala": 0.1,
            "mp25_base": 12.0,
        }]).to_csv(csv, index=False)

        import api.services as svc
        ruta_original = svc.RUTA_PREDICCION
        svc.RUTA_PREDICCION = csv
        try:
            data = client.get("/prediction/current").json()
            assert data["recomendacion"] != ""
            assert len(data["recomendacion"]) > 10
        finally:
            svc.RUTA_PREDICCION = ruta_original


# ---------------------------------------------------------------------------
# Tests: /forecast/7-days
# ---------------------------------------------------------------------------

class TestForecast7Days:

    def test_sin_archivo_devuelve_404_no_crash(self):
        resp = client.get("/forecast/7-days")
        assert resp.status_code in (200, 404)

    def test_404_tiene_campo_error(self):
        resp = client.get("/forecast/7-days")
        if resp.status_code == 404:
            body = resp.json()
            assert "detail" in body
            assert "error" in body["detail"]

    def test_200_tiene_7_dias(self, tmp_path):
        """Con archivo válido, el pronóstico debe tener exactamente 7 días."""
        csv = tmp_path / "prediccion_7_dias.csv"
        pd.DataFrame([{
            "fecha": f"2026-06-0{i+1}",
            "estacion": "Puente Alto",
            "comuna": "Puente Alto",
            "mp25_estimado": 40.0 + i * 5,
            "nivel_calidad_aire_predicho": "regular",
            "temperatura_max": 15.0,
            "temperatura_min": 5.0,
            "velocidad_viento": 12.0,
            "precipitacion": 0.0,
            "horizonte_dia": i + 1,
            "fecha_generacion": "2026-06-08 20:00:00",
        } for i in range(7)]).to_csv(csv, index=False)

        import api.services as svc
        ruta_original = svc.RUTA_FORECAST
        svc.RUTA_FORECAST = csv
        try:
            resp = client.get("/forecast/7-days")
            assert resp.status_code == 200
            data = resp.json()
            assert "dias" in data
            assert len(data["dias"]) == 7
        finally:
            svc.RUTA_FORECAST = ruta_original

    def test_cada_dia_tiene_campos_obligatorios(self, tmp_path):
        csv = tmp_path / "prediccion_7_dias.csv"
        pd.DataFrame([{
            "fecha": f"2026-06-0{i+1}",
            "estacion": "Puente Alto",
            "comuna": "Puente Alto",
            "mp25_estimado": 55.0,
            "nivel_calidad_aire_predicho": "mala",
            "temperatura_max": 10.0,
            "temperatura_min": 3.0,
            "velocidad_viento": 8.0,
            "precipitacion": 0.0,
            "horizonte_dia": i + 1,
            "fecha_generacion": "2026-06-08 20:00:00",
        } for i in range(7)]).to_csv(csv, index=False)

        import api.services as svc
        ruta_original = svc.RUTA_FORECAST
        svc.RUTA_FORECAST = csv
        try:
            dias = client.get("/forecast/7-days").json()["dias"]
            campos = ["horizonte_dia", "fecha", "mp25_estimado",
                      "nivel_calidad_aire_predicho", "recomendacion"]
            for dia in dias:
                for campo in campos:
                    assert campo in dia, f"Falta campo '{campo}' en día {dia.get('horizonte_dia')}"
        finally:
            svc.RUTA_FORECAST = ruta_original

    def test_horizonte_dia_de_1_a_7(self, tmp_path):
        csv = tmp_path / "prediccion_7_dias.csv"
        pd.DataFrame([{
            "fecha": f"2026-06-{i+1:02d}",
            "estacion": "Puente Alto",
            "comuna": "Puente Alto",
            "mp25_estimado": 30.0,
            "nivel_calidad_aire_predicho": "regular",
            "horizonte_dia": i + 1,
            "fecha_generacion": "2026-06-08 20:00:00",
        } for i in range(7)]).to_csv(csv, index=False)

        import api.services as svc
        ruta_original = svc.RUTA_FORECAST
        svc.RUTA_FORECAST = csv
        try:
            dias = client.get("/forecast/7-days").json()["dias"]
            horizontes = [d["horizonte_dia"] for d in dias]
            assert sorted(horizontes) == list(range(1, 8))
        finally:
            svc.RUTA_FORECAST = ruta_original


# ---------------------------------------------------------------------------
# Tests: /metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_sin_archivo_devuelve_404_no_crash(self):
        resp = client.get("/metrics")
        assert resp.status_code in (200, 404)

    def test_404_tiene_campo_error(self):
        resp = client.get("/metrics")
        if resp.status_code == 404:
            assert "detail" in resp.json()

    def test_200_tiene_accuracy(self, tmp_path):
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        metrics_file = metrics_dir / "model_metrics.json"
        metrics_file.write_text(json.dumps({
            "accuracy": 0.679,
            "f1_weighted": 0.671,
            "precision_weighted": 0.674,
            "recall_weighted": 0.679,
            "n_test": 324,
            "clases": ["buena", "mala", "regular"],
            "metricas_por_clase": {
                "buena":   {"precision": 0.55, "recall": 0.66, "f1_score": 0.60},
                "regular": {"precision": 0.45, "recall": 0.33, "f1_score": 0.38},
                "mala":    {"precision": 0.99, "recall": 0.99, "f1_score": 0.99},
            },
        }))

        import api.services as svc
        ruta_original = svc.RUTA_METRICS
        svc.RUTA_METRICS = metrics_file
        try:
            resp = client.get("/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert "accuracy" in data
            assert 0 <= data["accuracy"] <= 1
        finally:
            svc.RUTA_METRICS = ruta_original


# ---------------------------------------------------------------------------
# Tests: manejo de errores generales
# ---------------------------------------------------------------------------

class TestManejoErrores:

    def test_endpoint_inexistente_devuelve_404(self):
        resp = client.get("/endpoint_que_no_existe")
        assert resp.status_code == 404

    def test_metodo_post_no_permitido(self):
        """La API solo expone GET; POST debe devolver 405."""
        resp = client.post("/health")
        assert resp.status_code == 405

    def test_recomendacion_buena(self):
        from api.services import get_recomendacion
        rec = get_recomendacion("buena")
        assert "favorables" in rec.lower() or "normal" in rec.lower()

    def test_recomendacion_regular(self):
        from api.services import get_recomendacion
        rec = get_recomendacion("regular")
        assert "sensibles" in rec.lower() or "reducir" in rec.lower()

    def test_recomendacion_mala(self):
        from api.services import get_recomendacion
        rec = get_recomendacion("mala")
        assert "evitar" in rec.lower() or "reducir" in rec.lower()

    def test_recomendacion_nivel_desconocido(self):
        from api.services import get_recomendacion
        rec = get_recomendacion("desconocido")
        assert len(rec) > 0  # siempre retorna algo