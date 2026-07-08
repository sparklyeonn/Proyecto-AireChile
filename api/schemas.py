"""
api/schemas.py
==============
AireChile Analytics — Modelos de datos (schemas) de la API REST.

Define las estructuras de respuesta de cada endpoint usando Pydantic.
Esto garantiza validación automática, serialización JSON correcta
y documentación generada por FastAPI en /docs.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Respuesta del endpoint de verificación de estado."""
    status: str = Field(..., example="ok")
    project: str = Field(..., example="AireChile Analytics")
    version: str = Field(..., example="1.0.0")


# ---------------------------------------------------------------------------
# Predicción día siguiente
# ---------------------------------------------------------------------------

class PrediccionActualResponse(BaseModel):
    """Predicción del clasificador para el día siguiente."""
    fecha_base: str = Field(..., example="2026-06-08")
    fecha_prediccion: str = Field(..., example="2026-06-09")
    nivel_calidad_aire_predicho: str = Field(..., example="mala")
    probabilidad: Optional[float] = Field(None, example=0.97)
    prob_buena: Optional[float] = Field(None, example=0.01)
    prob_regular: Optional[float] = Field(None, example=0.02)
    prob_mala: Optional[float] = Field(None, example=0.97)
    mp25_base: Optional[float] = Field(None, example=87.86)
    recomendacion: str = Field(..., example="Evitar actividad física al aire libre.")
    fuente: str = Field(default="modelo_clasificador")


# ---------------------------------------------------------------------------
# Pronóstico 7 días
# ---------------------------------------------------------------------------

class DiaPronostico(BaseModel):
    """Un día del pronóstico de 7 días."""
    horizonte_dia: int = Field(..., ge=1, le=7, example=1)
    fecha: str = Field(..., example="2026-06-09")
    mp25_estimado: float = Field(..., example=65.4)
    nivel_calidad_aire_predicho: str = Field(..., example="mala")
    temperatura_max: Optional[float] = Field(None, example=12.5)
    temperatura_min: Optional[float] = Field(None, example=4.2)
    velocidad_viento: Optional[float] = Field(None, example=18.3)
    precipitacion: Optional[float] = Field(None, example=0.0)
    recomendacion: str = Field(..., example="Evitar actividad física al aire libre.")


class Forecast7DiasResponse(BaseModel):
    """Pronóstico completo de los próximos 7 días."""
    estacion: str = Field(..., example="Puente Alto")
    fecha_generacion: Optional[str] = Field(None, example="2026-06-08 21:00:00")
    dias: list[DiaPronóstico]


# ---------------------------------------------------------------------------
# Métricas del modelo
# ---------------------------------------------------------------------------

class MetricasPorClase(BaseModel):
    precision: float
    recall: float
    f1_score: float


class MetricasResponse(BaseModel):
    """Métricas del modelo clasificador RandomForest."""
    accuracy: float = Field(..., example=0.679)
    f1_weighted: float = Field(..., example=0.671)
    precision_weighted: float = Field(..., example=0.674)
    recall_weighted: float = Field(..., example=0.679)
    n_test: Optional[int] = Field(None, example=324)
    clases: Optional[list[str]] = Field(None, example=["buena", "mala", "regular"])
    metricas_por_clase: Optional[dict[str, MetricasPorClase]] = None
    top_features: Optional[list[dict]] = Field(
        None,
        example=[{"feature": "mp25", "importancia": 0.26}]
    )


# ---------------------------------------------------------------------------
# Estaciones
# ---------------------------------------------------------------------------

class EstacionResponse(BaseModel):
    """Estación de monitoreo de calidad del aire."""
    estacion: str = Field(..., example="Puente Alto")
    comuna: str = Field(..., example="Puente Alto")
    latitud: float = Field(..., example=-33.6117)
    longitud: float = Field(..., example=-70.5758)
    fuente_calidad_aire: str = Field(..., example="SINCA")
    fuente_meteorologia: str = Field(..., example="Open-Meteo")
    periodo_datos: Optional[str] = Field(None, example="2022-01-01 / 2026-06-13")


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Respuesta de error controlado."""
    error: str = Field(..., example="Archivo no encontrado")
    detalle: str = Field(
        ...,
        example="No existe data/processed/prediccion_7_dias.csv. "
                "Ejecute primero: python models/predict_7_days.py"
    )