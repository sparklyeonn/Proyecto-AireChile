"""
api/main.py
===========
AireChile Analytics — API REST con FastAPI.

Expone los resultados del pipeline de ciencia de datos como endpoints
HTTP consumibles por cualquier cliente (web, móvil, dashboard u otro sistema).

Endpoints disponibles:
    GET /                   → portada simple de la API
    GET /health             → estado del servicio
    GET /prediction/current → predicción del día siguiente
    GET /forecast/7-days    → pronóstico de los próximos 7 días
    GET /metrics            → métricas del modelo RandomForest
    GET /stations           → estaciones de monitoreo disponibles

Documentación interactiva automática:
    Local:
        http://localhost:8000/docs
        http://localhost:8000/redoc
        http://localhost:8000/openapi.json

    Render:
        https://proyecto-airechile.onrender.com/docs
        https://proyecto-airechile.onrender.com/redoc
        https://proyecto-airechile.onrender.com/openapi.json

Ejecución local:
    uvicorn api.main:app --reload --port 8000

Ejecución en Render:
    uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    PrediccionActualResponse,
    Forecast7DiasResponse,
    MetricasResponse,
    EstacionResponse,
    ErrorResponse,
)
from api import services


# ---------------------------------------------------------------------------
# Inicialización de la aplicación
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AireChile Analytics API",
    description=(
        "API REST del proyecto AireChile Analytics. "
        "Expone predicciones de calidad del aire, pronósticos a 7 días, "
        "métricas del modelo y datos de estaciones de monitoreo en Santiago de Chile.\n\n"
        "**Datos fuente:** SINCA (Ministerio del Medio Ambiente de Chile) + Open-Meteo API\n\n"
        "**Modelo:** RandomForestClassifier + RandomForestRegressor"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Equipo AireChile Analytics",
        "url": "https://github.com/sparklyeonn/Proyecto-AireChile",
    },
    license_info={
        "name": "MIT",
    },
)

# Permitir peticiones desde el dashboard Streamlit y clientes externos.
# Para un ambiente productivo real, se recomienda restringir allow_origins
# a los dominios del dashboard y servicios autorizados.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper para errores controlados
# ---------------------------------------------------------------------------

def _not_found(detalle: str) -> HTTPException:
    """Retorna un error 404 con formato uniforme."""
    return HTTPException(
        status_code=404,
        detail={"error": "Archivo no encontrado", "detalle": detalle},
    )


def _server_error(detalle: str) -> HTTPException:
    """Retorna un error 500 con formato uniforme."""
    return HTTPException(
        status_code=500,
        detail={"error": "Error interno del servidor", "detalle": detalle},
    )


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Portada de la API",
    tags=["Sistema"],
)
def root() -> dict:
    """
    Endpoint raíz para evitar el mensaje 404 al abrir la URL principal.

    Sirve como portada simple de la API y entrega accesos rápidos
    a la documentación y endpoints principales.
    """
    return {
        "proyecto": "AireChile Analytics",
        "estado": "API funcionando",
        "version": "1.0.0",
        "documentacion": "/docs",
        "openapi": "/openapi.json",
        "endpoints": [
            "/health",
            "/prediction/current",
            "/forecast/7-days",
            "/metrics",
            "/stations",
        ],
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    tags=["Sistema"],
)
def health_check() -> HealthResponse:
    """
    Verifica que la API está activa y responde correctamente.

    Útil para monitoreo, Docker healthchecks y depuración rápida.
    """
    return HealthResponse(
        status="ok",
        project="AireChile Analytics",
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# GET /prediction/current
# ---------------------------------------------------------------------------

@app.get(
    "/prediction/current",
    response_model=PrediccionActualResponse,
    summary="Predicción de calidad del aire para mañana",
    tags=["Predicción"],
    responses={
        404: {"model": ErrorResponse, "description": "Archivo de predicción no encontrado"},
        500: {"model": ErrorResponse, "description": "Error al procesar el archivo"},
    },
)
def get_prediccion_actual() -> PrediccionActualResponse:
    """
    Retorna la predicción más reciente del modelo clasificador para el día siguiente.

    Lee el archivo `data/processed/prediccion_actual.csv` generado por
    `python models/predict.py`.

    La predicción incluye:
    - Nivel esperado: buena, regular o mala.
    - Probabilidades por clase.
    - Recomendación operativa.
    """
    try:
        return services.get_prediccion_actual()
    except FileNotFoundError as e:
        raise _not_found(str(e))
    except Exception as e:
        raise _server_error(str(e))


# ---------------------------------------------------------------------------
# GET /forecast/7-days
# ---------------------------------------------------------------------------

@app.get(
    "/forecast/7-days",
    response_model=Forecast7DiasResponse,
    summary="Pronóstico de calidad del aire para los próximos 7 días",
    tags=["Predicción"],
    responses={
        404: {"model": ErrorResponse, "description": "Archivo de pronóstico no encontrado"},
        500: {"model": ErrorResponse, "description": "Error al procesar el archivo"},
    },
)
def get_forecast_7_dias() -> Forecast7DiasResponse:
    """
    Retorna el pronóstico de MP2.5 y nivel de calidad del aire para
    cada uno de los próximos 7 días.

    Usa predicción recursiva con un RandomForestRegressor:
    cada MP2.5 estimado se incorpora como insumo para el día siguiente.

    Lee el archivo `data/processed/prediccion_7_dias.csv` generado por
    `python models/predict_7_days.py`.
    """
    try:
        return services.get_forecast_7_dias()
    except FileNotFoundError as e:
        raise _not_found(str(e))
    except Exception as e:
        raise _server_error(str(e))


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@app.get(
    "/metrics",
    response_model=MetricasResponse,
    summary="Métricas del modelo predictivo",
    tags=["Modelo"],
    responses={
        404: {"model": ErrorResponse, "description": "Archivo de métricas no encontrado"},
        500: {"model": ErrorResponse, "description": "Error al leer métricas"},
    },
)
def get_metricas() -> MetricasResponse:
    """
    Retorna las métricas de evaluación del modelo RandomForestClassifier.

    Incluye:
    - Accuracy global en el conjunto de test.
    - F1-score ponderado.
    - Precision y recall por clase.
    - Top 5 variables más importantes.

    Lee `models/metrics/model_metrics.json` y `models/metrics/feature_importance.csv`
    generados por `python models/train_model.py`.
    """
    try:
        return services.get_metricas()
    except FileNotFoundError as e:
        raise _not_found(str(e))
    except Exception as e:
        raise _server_error(str(e))


# ---------------------------------------------------------------------------
# GET /stations
# ---------------------------------------------------------------------------

@app.get(
    "/stations",
    response_model=list[EstacionResponse],
    summary="Estaciones de monitoreo disponibles",
    tags=["Datos"],
)
def get_estaciones() -> list[EstacionResponse]:
    """
    Retorna la lista de estaciones de monitoreo de calidad del aire
    incluidas en el proyecto.

    La versión actual trabaja con la estación Puente Alto (SINCA),
    con datos desde 2022. La arquitectura permite escalar a más estaciones.
    """
    try:
        return services.get_estaciones()
    except Exception as e:
        raise _server_error(str(e))


# ---------------------------------------------------------------------------
# Punto de entrada para ejecución directa en desarrollo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )