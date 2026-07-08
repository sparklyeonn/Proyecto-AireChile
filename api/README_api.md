# API REST — AireChile Analytics

API REST construida con **FastAPI** que expone los resultados del pipeline
de ciencia de datos de AireChile Analytics como endpoints HTTP.

Permite que cualquier cliente (otra aplicación, dashboard externo, script)
consulte predicciones, pronósticos y métricas sin necesidad de acceder
directamente a los archivos CSV o al código Python.

---

## Cómo ejecutarla localmente

```bash
# Desde la raíz del proyecto
uvicorn api.main:app --reload --port 8000
```

Documentación interactiva (Swagger UI):
```
http://localhost:8000/docs
```

Documentación alternativa (ReDoc):
```
http://localhost:8000/redoc
```

---

## Endpoints disponibles

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| GET | `/prediction/current` | Predicción del día siguiente |
| GET | `/forecast/7-days` | Pronóstico de los próximos 7 días |
| GET | `/metrics` | Métricas del modelo RandomForest |
| GET | `/stations` | Estaciones de monitoreo disponibles |

---

## Ejemplos de respuesta

### GET /health

```json
{
  "status": "ok",
  "project": "AireChile Analytics",
  "version": "1.0.0"
}
```

### GET /prediction/current

```json
{
  "fecha_base": "2026-06-08",
  "fecha_prediccion": "2026-06-09",
  "nivel_calidad_aire_predicho": "mala",
  "probabilidad": 0.97,
  "prob_buena": 0.01,
  "prob_regular": 0.02,
  "prob_mala": 0.97,
  "mp25_base": 87.86,
  "recomendacion": "Evitar actividad física al aire libre. Niños, adultos mayores y personas con enfermedades respiratorias deben reducir la exposición al máximo.",
  "fuente": "modelo_clasificador"
}
```

### GET /forecast/7-days

```json
{
  "estacion": "Puente Alto",
  "fecha_generacion": "2026-06-08 20:00:00",
  "dias": [
    {
      "horizonte_dia": 1,
      "fecha": "2026-06-09",
      "mp25_estimado": 65.4,
      "nivel_calidad_aire_predicho": "mala",
      "temperatura_max": 12.5,
      "temperatura_min": 4.2,
      "velocidad_viento": 18.3,
      "precipitacion": 0.0,
      "recomendacion": "Evitar actividad física al aire libre..."
    }
  ]
}
```

### GET /metrics

```json
{
  "accuracy": 0.679,
  "f1_weighted": 0.671,
  "precision_weighted": 0.674,
  "recall_weighted": 0.679,
  "n_test": 324,
  "clases": ["buena", "mala", "regular"],
  "metricas_por_clase": {
    "mala": {"precision": 0.99, "recall": 0.99, "f1_score": 0.99}
  },
  "top_features": [
    {"feature": "mp25", "importancia": 0.26},
    {"feature": "mp25_promedio_7d", "importancia": 0.24}
  ]
}
```

### GET /stations

```json
[
  {
    "estacion": "Puente Alto",
    "comuna": "Puente Alto",
    "latitud": -33.6117,
    "longitud": -70.5758,
    "fuente_calidad_aire": "SINCA",
    "fuente_meteorologia": "Open-Meteo",
    "periodo_datos": "2022-01-01 / 2026-06-13"
  }
]
```

---

## Archivos que necesita

| Endpoint | Archivo requerido | Generado por |
|---|---|---|
| `/prediction/current` | `data/processed/prediccion_actual.csv` | `python models/predict.py` |
| `/forecast/7-days` | `data/processed/prediccion_7_dias.csv` | `python models/predict_7_days.py` |
| `/metrics` | `models/metrics/model_metrics.json` | `python models/train_model.py` |
| `/stations` | *(ninguno — datos fijos)* | — |

Si un archivo no existe, el endpoint devuelve **404** con un mensaje
indicando qué comando ejecutar para generarlo.

---

## Errores comunes

### 404 — Archivo no encontrado

```json
{
  "detail": {
    "error": "Archivo no encontrado",
    "detalle": "No existe data/processed/prediccion_7_dias.csv. Ejecute primero: python models/predict_7_days.py"
  }
}
```

**Solución:** ejecutar el script indicado en el mensaje.

### 422 — Error de validación

Ocurre si el archivo CSV tiene columnas faltantes o tipos incorrectos.
Revisar que el pipeline ETL y el modelo se ejecutaron correctamente.

### CORS

Si se consume la API desde un navegador en otro dominio, verificar que el
origen esté permitido en la configuración de CORS en `api/main.py`.

---

## Estructura del paquete

```
api/
├── __init__.py      ← paquete Python
├── main.py          ← aplicación FastAPI y endpoints
├── schemas.py       ← modelos Pydantic de request/response
├── services.py      ← lógica de negocio (lectura de archivos)
└── README_api.md    ← este archivo
```

---

## Tests

```bash
pytest tests/test_api.py -v
```

Los tests cubren todos los endpoints y verifican que los errores se manejan
correctamente sin generar excepciones no controladas.