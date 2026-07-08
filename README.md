# AireChile Analytics

## 1. Descripción general

**AireChile Analytics** es un proyecto de ciencia de datos orientado a anticipar la calidad del aire en Santiago, tomando como caso piloto la estación de Puente Alto.

El sistema integra datos históricos de calidad del aire provenientes de SINCA con variables meteorológicas de Open-Meteo. A partir de estos datos se construye un pipeline completo que permite limpiar, transformar, integrar, modelar, predecir y visualizar información ambiental.

El objetivo principal es predecir la calidad del aire del día siguiente y entregar una estimación de MP2.5 para los próximos días, apoyando la toma de decisiones en municipalidades, colegios, clínicas, empresas y ciudadanía.

---

## 2. Problema abordado

Santiago presenta episodios recurrentes de contaminación atmosférica, especialmente durante los meses de otoño e invierno. Estos episodios pueden afectar la salud de la población, generar restricciones vehiculares, suspender actividades físicas al aire libre y aumentar el riesgo para grupos sensibles.

El problema principal es que muchas decisiones se toman de forma reactiva, cuando el episodio de contaminación ya ocurrió.

AireChile Analytics busca anticipar estos escenarios mediante modelos predictivos entrenados con datos históricos de MP2.5 y variables meteorológicas.

---

## 3. Fuentes de datos

El proyecto utiliza dos fuentes principales:

### 3.1 SINCA

Fuente utilizada para obtener datos históricos de calidad del aire.

Variable principal:

```text
MP2.5
```

El MP2.5 corresponde al material particulado fino medido en microgramos por metro cúbico de aire.

### 3.2 Open-Meteo

Fuente utilizada para obtener variables meteorológicas históricas y de pronóstico.

Variables utilizadas:

```text
temperatura_max
temperatura_min
temperatura_promedio
humedad_relativa
velocidad_viento
precipitacion
```

---

## 4. Alcance del proyecto

El proyecto se implementa como piloto funcional para la estación de Puente Alto.

El flujo actual considera:

```text
SINCA Puente Alto + Open-Meteo Puente Alto → Dataset integrado → Modelos ML → Dashboard/API
```

Aunque el proyecto trabaja con una estación piloto, la arquitectura puede escalar a múltiples estaciones agregando coordenadas, estación, comuna y lógica de integración por fecha + estación.

---

## 5. Estructura general del proyecto

```text
Proyecto-AireChile/
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── services.py
│   └── README_api.md
│
├── dashboards/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── documentacion_tecnica.md
│
├── etl/
│   ├── extract_sinca.py
│   ├── transform_sinca.py
│   ├── etl_sinca_main.py
│   ├── extract_meteo.py
│   ├── transform_meteo.py
│   ├── merge_sinca_meteo.py
│   └── etl_meteo_main.py
│
├── models/
│   ├── train_model.py
│   ├── predict.py
│   ├── train_forecast_model.py
│   ├── predict_7_days.py
│   ├── compare_classification_models.py
│   ├── compare_regression_models.py
│   └── metrics/
│
├── tests/
│   ├── test_api.py
│   ├── test_dashboard_data.py
│   ├── test_model_comparison.py
│   └── otros tests del proyecto
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 6. Variables principales del dataset

El dataset final se genera en:

```text
data/processed/dataset_modelo_base.csv
```

Variables principales:

```text
fecha
estacion
comuna
mp25
nivel_calidad_aire
nivel_calidad_aire_dia_siguiente
mp25_dia_anterior
mp25_promedio_7d
mes
dia_semana
temperatura_max
temperatura_min
temperatura_promedio
humedad_relativa
velocidad_viento
precipitacion
```

---

## 7. Clasificación de calidad del aire

El proyecto clasifica la calidad del aire en tres niveles:

```text
buena
regular
mala
```

Criterios utilizados según concentración de MP2.5:

```text
buena:   MP2.5 <= 25 µg/m³
regular: MP2.5 > 25 y <= 50 µg/m³
mala:    MP2.5 > 50 µg/m³
```

---

## 8. Pipeline ETL

El pipeline ETL realiza la extracción, transformación e integración de datos.

### 8.1 ETL SINCA

Ejecutar:

```powershell
python etl/etl_sinca_main.py
```

Este proceso:

```text
- lee el archivo SINCA crudo
- detecta el formato del archivo
- limpia y normaliza columnas
- convierte fechas
- filtra registros válidos
- genera variables derivadas
- guarda datos procesados
```

### 8.2 ETL Open-Meteo + Merge

Ejecutar:

```powershell
python etl/etl_meteo_main.py
```

Este proceso:

```text
- obtiene datos meteorológicos históricos
- transforma variables meteorológicas
- une datos SINCA y meteorología por fecha
- genera el dataset base para modelamiento
```

---

## 9. Modelos predictivos principales

El proyecto trabaja con dos enfoques principales:

### 9.1 Clasificación

Predice la categoría de calidad del aire del día siguiente:

```text
nivel_calidad_aire_dia_siguiente
```

Modelo principal utilizado:

```text
RandomForestClassifier
```

### 9.2 Regresión

Estima el valor numérico de MP2.5 del día siguiente:

```text
mp25_dia_siguiente
```

Modelo principal utilizado:

```text
RandomForestRegressor
```

Este modelo se usa como base para el pronóstico de 7 días.

---

## 10. Entrenamiento del modelo

Entrenar el modelo clasificador:

```powershell
python models/train_model.py
```

Archivos esperados:

```text
models/model.pkl
models/metrics/model_metrics.json
models/metrics/feature_importance.csv
models/metrics/confusion_matrix.csv
```

Generar predicción del día siguiente:

```powershell
python models/predict.py
```

Archivo esperado:

```text
data/processed/prediccion_actual.csv
```

---

## 11. Pronóstico de 7 días

El proyecto incluye un pronóstico de MP2.5 para los próximos 7 días.

Ejecutar:

```powershell
python etl/extract_meteo_forecast.py
python models/train_forecast_model.py
python models/predict_7_days.py
```

Archivo esperado:

```text
data/processed/prediccion_7_dias.csv
```

---

## 12. Comparación de modelos y tuning

El proyecto incorpora una comparación formal de modelos de Machine Learning para justificar técnicamente la selección del modelo final.

La comparación se divide en dos tareas:

```text
1. Clasificación de calidad del aire del día siguiente.
2. Regresión para estimar el valor numérico de MP2.5 del día siguiente.
```

---

### 12.1 Modelos de clasificación comparados

Archivo:

```text
models/compare_classification_models.py
```

Modelos comparados:

```text
LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
GradientBoostingClassifier
```

Variable objetivo:

```text
nivel_calidad_aire_dia_siguiente
```

Métricas utilizadas:

```text
accuracy
precision_weighted
recall_weighted
f1_weighted
```

La métrica principal es:

```text
f1_weighted
```

Se usa `f1_weighted` porque considera el equilibrio entre precision y recall, ponderando el peso de cada clase. Esto es relevante porque la calidad del aire puede presentar clases desbalanceadas.

---

### 12.2 Modelos de regresión comparados

Archivo:

```text
models/compare_regression_models.py
```

Modelos comparados:

```text
LinearRegression
DecisionTreeRegressor
RandomForestRegressor
GradientBoostingRegressor
```

Variable objetivo:

```text
mp25_dia_siguiente
```

Métricas utilizadas:

```text
MAE
RMSE
R2
```

La métrica principal es:

```text
RMSE
```

Se usa `RMSE` porque penaliza con mayor fuerza los errores grandes en la estimación de MP2.5, lo que es relevante para escenarios de mala calidad del aire.

---

### 12.3 Tuning

Se aplica tuning con:

```text
GridSearchCV
```

El tuning se realiza sobre RandomForest para clasificación y regresión.

Además, se utiliza:

```text
TimeSeriesSplit
```

Esto permite respetar el orden temporal de los datos y reducir el riesgo de fuga de información futura durante la validación.

---

### 12.4 Ejecutar comparación de modelos

```powershell
python models/compare_classification_models.py
python models/compare_regression_models.py
```

Archivos generados:

```text
models/metrics/classification_model_comparison.csv
models/metrics/classification_model_comparison_summary.json
models/metrics/classification_tuning_results.json
models/metrics/regression_model_comparison.csv
models/metrics/regression_model_comparison_summary.json
models/metrics/regression_tuning_results.json
```

---

### 12.5 Visualización en dashboard

Los resultados de comparación se visualizan en el dashboard Streamlit, dentro de la sección:

```text
Modelo → Comparación
```

Esta vista permite revisar:

```text
- tabla comparativa de modelos de clasificación
- tabla comparativa de modelos de regresión
- mejor modelo según la métrica seleccionada
- justificación técnica
- resultados del tuning
```

---

## 13. Dashboard Streamlit

El dashboard se encuentra en:

```text
dashboards/app.py
```

Ejecutar:

```powershell
streamlit run dashboards/app.py --server.port 8502
```

Secciones principales:

```text
Inicio
Histórico
Meteorología
Predicción
Modelo
Vista técnica
```

La sección Modelo incluye:

```text
- métricas del modelo principal
- métricas por clase
- importancia de variables
- matriz de confusión
- metodología
- comparación de modelos
```

---

## 14. API REST

El proyecto incluye una API REST desarrollada con FastAPI.

Ejecutar:

```powershell
uvicorn api.main:app --reload --port 8000
```

Documentación interactiva:

```text
http://localhost:8000/docs
```

Endpoints principales:

```text
GET /health
GET /prediction/current
GET /forecast/7-days
GET /metrics
GET /stations
```

---

## 15. Docker

El proyecto puede ejecutarse mediante Docker.

Construir imagen:

```powershell
docker build -t airechile-analytics .
```

Validar Docker Compose:

```powershell
docker compose config
```

Levantar servicios:

```powershell
docker compose up --build
```

Detener servicios:

```powershell
docker compose down
```

---

## 16. Tests

Ejecutar tests de comparación de modelos:

```powershell
python -m pytest tests/test_model_comparison.py -v
```

Ejecutar tests del dashboard:

```powershell
python -m pytest tests/test_dashboard_data.py -v
```

Ejecutar tests de API:

```powershell
python -m pytest tests/test_api.py -v
```

Ejecutar todos los tests:

```powershell
python -m pytest tests/ -v
```

---

## 17. CI/CD

El proyecto incluye integración continua con GitHub Actions.

El workflow valida:

```text
- instalación de dependencias
- importación de la API
- ejecución de tests
- validación de Docker Compose
- construcción de imagen Docker
```

Archivo principal:

```text
.github/workflows/ci.yml
```

---

## 18. Orden recomendado de ejecución local

```powershell
# 1. Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar ETL
python etl/etl_sinca_main.py
python etl/etl_meteo_main.py

# 4. Entrenar modelo principal
python models/train_model.py

# 5. Generar predicción día siguiente
python models/predict.py

# 6. Generar pronóstico 7 días
python etl/extract_meteo_forecast.py
python models/train_forecast_model.py
python models/predict_7_days.py

# 7. Comparar modelos
python models/compare_classification_models.py
python models/compare_regression_models.py

# 8. Ejecutar dashboard
streamlit run dashboards/app.py --server.port 8502

# 9. Ejecutar tests
python -m pytest tests/ -v
```

---

## 19. Decisiones técnicas principales

### 19.1 Uso de RandomForest

RandomForest se utiliza porque:

```text
- captura relaciones no lineales
- funciona bien con variables de distinta escala
- es robusto ante outliers
- permite obtener importancia de variables
- tiene buen desempeño en problemas tabulares
```

### 19.2 Split temporal

Se usa separación temporal porque los datos de calidad del aire tienen dependencia cronológica.

El modelo entrena con datos más antiguos y se evalúa con datos más recientes.

Esto evita que el modelo aprenda información futura.

### 19.3 TimeSeriesSplit

En el tuning se usa `TimeSeriesSplit` para mantener la lógica temporal del problema durante la validación cruzada.

### 19.4 Métrica F1 weighted

En clasificación se prioriza `f1_weighted` porque considera precision y recall, ponderando el peso de cada clase.

### 19.5 Métrica RMSE

En regresión se prioriza `RMSE` porque penaliza más los errores grandes, relevantes en episodios de contaminación alta.

---

## 20. Mejoras futuras

Posibles mejoras futuras:

```text
- agregar más estaciones de Santiago
- incorporar más contaminantes como MP10, NO2 u O3
- usar modelos de series temporales
- crear alertas automáticas por correo o WhatsApp
- desplegar API y dashboard en la nube
- incorporar base de datos productiva con actualización programada
```

---

## 21. Estado actual del proyecto

El proyecto cuenta con:

```text
- ETL funcional
- integración de datos SINCA y Open-Meteo
- modelo clasificador
- modelo regresor para pronóstico
- predicción del día siguiente
- pronóstico de 7 días
- comparación de modelos
- tuning con validación temporal
- dashboard Streamlit
- API REST
- Docker
- PostgreSQL
- tests automatizados
- CI/CD
```

AireChile Analytics representa una solución completa de ciencia de datos aplicada a un problema ambiental real.
