# Documentación técnica — AireChile Analytics

## 1. Introducción

AireChile Analytics es una solución de ciencia de datos desarrollada para predecir y visualizar la calidad del aire en Santiago, usando como caso piloto la estación de Puente Alto.

El proyecto integra datos ambientales y meteorológicos, construye un dataset modelable, entrena modelos predictivos, expone resultados mediante dashboard y API, e incorpora pruebas automatizadas, Docker y CI/CD.

El propósito técnico es implementar un flujo completo de ciencia de datos:

```text
Extracción → Limpieza → Transformación → Integración → Modelamiento → Predicción → Visualización → API → Testing → Despliegue
```

---

## 2. Objetivo técnico del sistema

El objetivo técnico es construir un sistema capaz de:

```text
- leer datos históricos de calidad del aire
- limpiar y transformar registros crudos
- integrar datos meteorológicos
- generar variables predictoras
- entrenar modelos de Machine Learning
- predecir la calidad del aire del día siguiente
- estimar MP2.5 para los próximos 7 días
- visualizar resultados en un dashboard
- exponer resultados mediante una API REST
- validar el funcionamiento mediante tests
- facilitar despliegue mediante Docker y CI/CD
```

---

## 3. Arquitectura general

El sistema se organiza en módulos:

```text
data/raw/           → datos crudos
data/processed/     → datos limpios y predicciones
etl/                → extracción, transformación e integración
models/             → entrenamiento, predicción y comparación de modelos
models/metrics/     → métricas, resultados y evidencia de modelos
dashboards/         → dashboard Streamlit
api/                → API REST FastAPI
tests/              → pruebas automatizadas
docs/               → documentación técnica
.github/workflows/  → integración continua
```

---

## 4. Flujo de datos

El flujo principal del proyecto es:

```text
SINCA CSV
   ↓
extract_sinca.py
   ↓
transform_sinca.py
   ↓
sinca_transformado.csv
   ↓
Open-Meteo API
   ↓
extract_meteo.py
   ↓
transform_meteo.py
   ↓
open_meteo_transformado.csv
   ↓
merge_sinca_meteo.py
   ↓
dataset_modelo_base.csv
   ↓
train_model.py / train_forecast_model.py
   ↓
predicciones y métricas
   ↓
dashboard Streamlit + API FastAPI
```

---

## 5. Datos utilizados

### 5.1 Datos de calidad del aire

Fuente:

```text
SINCA
```

Variable principal:

```text
mp25
```

El MP2.5 corresponde a partículas finas en suspensión con diámetro menor o igual a 2,5 micrómetros. En el proyecto se mide en:

```text
µg/m³
```

### 5.2 Datos meteorológicos

Fuente:

```text
Open-Meteo
```

Variables utilizadas:

```text
temperatura_max
temperatura_min
temperatura_promedio
humedad_relativa
velocidad_viento
precipitacion
```

Estas variables se incorporan porque influyen en la dispersión o acumulación de contaminantes.

---

## 6. Dataset base

Archivo principal:

```text
data/processed/dataset_modelo_base.csv
```

Este archivo contiene la integración entre datos SINCA y Open-Meteo.

Columnas principales:

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

## 7. Limpieza y transformación de datos

El pipeline de transformación realiza:

```text
- normalización de nombres de columnas
- conversión de fechas
- validación de registros
- eliminación de nulos críticos
- ordenamiento temporal
- creación de variables derivadas
- clasificación de MP2.5 en niveles de calidad del aire
- generación de target del día siguiente
- integración de meteorología por fecha
```

---

## 8. Variables derivadas

### 8.1 MP2.5 del día anterior

```text
mp25_dia_anterior
```

Representa el valor de MP2.5 observado el día anterior. Permite que el modelo capture persistencia temporal.

### 8.2 Promedio móvil de 7 días

```text
mp25_promedio_7d
```

Resume la tendencia reciente de contaminación.

### 8.3 Mes

```text
mes
```

Captura estacionalidad. En Santiago, los episodios críticos suelen concentrarse en otoño e invierno.

### 8.4 Día de la semana

```text
dia_semana
```

Permite capturar patrones semanales asociados a actividad urbana.

### 8.5 Nivel de calidad del aire

```text
nivel_calidad_aire
```

Clasifica el estado actual de calidad del aire.

### 8.6 Nivel de calidad del aire del día siguiente

```text
nivel_calidad_aire_dia_siguiente
```

Corresponde a la variable objetivo de clasificación.

### 8.7 MP2.5 del día siguiente

```text
mp25_dia_siguiente
```

Corresponde a la variable objetivo de regresión. Si no existe en el dataset, se construye desplazando `mp25` un día hacia adelante.

---

## 9. Clasificación ambiental

El proyecto utiliza tres clases:

```text
buena
regular
mala
```

Reglas:

```text
buena:   MP2.5 <= 25 µg/m³
regular: MP2.5 > 25 y <= 50 µg/m³
mala:    MP2.5 > 50 µg/m³
```

Esta clasificación transforma una variable numérica en una variable categórica útil para comunicación y toma de decisiones.

---

## 10. Modelamiento predictivo

El proyecto implementa dos tareas supervisadas:

```text
1. Clasificación
2. Regresión
```

---

## 11. Modelo de clasificación

### 11.1 Objetivo

Predecir la categoría de calidad del aire del día siguiente:

```text
nivel_calidad_aire_dia_siguiente
```

Clases:

```text
buena
regular
mala
```

### 11.2 Modelo principal

```text
RandomForestClassifier
```

### 11.3 Justificación

RandomForestClassifier es adecuado porque:

```text
- maneja relaciones no lineales
- funciona bien con datos tabulares
- no requiere escalamiento de variables
- es robusto ante outliers
- permite calcular importancia de variables
- puede trabajar con clases desbalanceadas usando class_weight
```

---

## 12. Modelo de regresión

### 12.1 Objetivo

Estimar el valor numérico de MP2.5 del día siguiente:

```text
mp25_dia_siguiente
```

### 12.2 Modelo principal

```text
RandomForestRegressor
```

### 12.3 Uso dentro del proyecto

El modelo regresor se utiliza para estimar valores de MP2.5 y apoyar el pronóstico de 7 días.

### 12.4 Justificación

RandomForestRegressor es adecuado porque:

```text
- captura relaciones no lineales entre clima y contaminación
- es robusto ante valores extremos
- no requiere normalización de variables
- permite trabajar con variables en distintas unidades
- funciona bien en problemas tabulares con tamaño de datos moderado
```

---

## 13. Separación temporal de datos

El proyecto utiliza split temporal, no aleatorio.

La lógica es:

```text
80% de datos más antiguos → entrenamiento
20% de datos más recientes → test
```

Esta decisión es importante porque la calidad del aire tiene dependencia temporal.

Usar una separación aleatoria podría generar fuga de información, ya que el modelo podría aprender patrones del futuro y obtener métricas artificialmente altas.

---

## 14. Comparación de modelos

Para justificar la selección del modelo final, se implementó una comparación formal de modelos de clasificación y regresión.

Archivos:

```text
models/compare_classification_models.py
models/compare_regression_models.py
```

La comparación permite demostrar que la selección del modelo final se basa en evidencia cuantitativa y no en una elección arbitraria.

---

## 15. Comparación de clasificación

### 15.1 Modelos comparados

```text
LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
GradientBoostingClassifier
```

### 15.2 Métricas utilizadas

```text
accuracy
precision_weighted
recall_weighted
f1_weighted
```

### 15.3 Métrica principal

```text
f1_weighted
```

### 15.4 Justificación de la métrica

`f1_weighted` combina precision y recall, ponderando el peso de cada clase.

Esta métrica es adecuada porque los datos de calidad del aire pueden estar desbalanceados. Por ejemplo, puede haber más días buenos o regulares que días malos.

### 15.5 Resultado esperado

El script genera:

```text
models/metrics/classification_model_comparison.csv
models/metrics/classification_model_comparison_summary.json
models/metrics/classification_tuning_results.json
```

---

## 16. Comparación de regresión

### 16.1 Modelos comparados

```text
LinearRegression
DecisionTreeRegressor
RandomForestRegressor
GradientBoostingRegressor
```

### 16.2 Métricas utilizadas

```text
MAE
RMSE
R2
```

### 16.3 Métrica principal

```text
RMSE
```

### 16.4 Justificación de la métrica

`RMSE` penaliza más los errores grandes. Esto es importante en calidad del aire, porque un error alto podría significar no detectar correctamente un día con contaminación elevada.

### 16.5 Resultado esperado

El script genera:

```text
models/metrics/regression_model_comparison.csv
models/metrics/regression_model_comparison_summary.json
models/metrics/regression_tuning_results.json
```

---

## 17. Tuning de hiperparámetros

Se aplica tuning mediante:

```text
GridSearchCV
```

El tuning se aplica sobre RandomForest en clasificación y regresión.

Parámetros evaluados:

```python
{
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}
```

---

## 18. Validación temporal en tuning

Para la validación cruzada se utiliza:

```text
TimeSeriesSplit
```

Esto permite respetar la estructura temporal del dataset.

La ventaja de esta técnica es que cada fold mantiene el orden cronológico, evitando que el modelo sea validado con datos que temporalmente pertenecen al pasado después de haber entrenado con datos futuros.

---

## 19. Archivos de métricas

La carpeta:

```text
models/metrics/
```

contiene archivos de evaluación y evidencia.

Archivos principales del modelo inicial:

```text
model_metrics.json
feature_importance.csv
confusion_matrix.csv
```

Archivos de comparación de modelos:

```text
classification_model_comparison.csv
classification_model_comparison_summary.json
classification_tuning_results.json
regression_model_comparison.csv
regression_model_comparison_summary.json
regression_tuning_results.json
```

---

## 20. Dashboard

El dashboard está desarrollado con Streamlit.

Archivo:

```text
dashboards/app.py
```

Ejecutar:

```powershell
streamlit run dashboards/app.py --server.port 8502
```

Secciones:

```text
Inicio
Histórico
Meteorología
Predicción
Modelo
Vista técnica
```

---

## 21. Sección Modelo del dashboard

La sección Modelo permite revisar:

```text
- rendimiento del modelo clasificador
- métricas por clase
- importancia de variables
- matriz de confusión
- explicación metodológica
- comparación de modelos
```

La pestaña de comparación muestra:

```text
- modelos de clasificación comparados
- modelos de regresión comparados
- métricas principales
- modelo ganador
- justificación técnica
- resultados de tuning
```

---

## 22. API REST

La API está desarrollada con FastAPI.

Archivo principal:

```text
api/main.py
```

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

## 23. Docker

El proyecto incorpora Docker para facilitar ejecución y despliegue.

Archivos principales:

```text
Dockerfile
docker-compose.yml
```

Comandos:

```powershell
docker build -t airechile-analytics .
docker compose config
docker compose up --build
docker compose down
```

---

## 24. Base de datos

El proyecto incorpora soporte para PostgreSQL.

Componentes relacionados:

```text
database/schema.sql
etl/load_postgres.py
etl/etl_postgres_main.py
```

La base de datos permite persistir información procesada del pipeline y facilitar una arquitectura más cercana a un entorno productivo.

---

## 25. Tests automatizados

El proyecto incorpora pruebas automatizadas con Pytest.

Tests principales:

```text
tests/test_api.py
tests/test_dashboard_data.py
tests/test_model_comparison.py
```

### 25.1 Tests de comparación de modelos

Archivo:

```text
tests/test_model_comparison.py
```

Valida:

```text
- existencia de scripts de comparación
- importación de scripts sin errores
- existencia de funciones principales
- definición de 4 modelos de clasificación
- definición de 4 modelos de regresión
- generación de CSV y JSON
- métricas en rangos válidos
- manejo de error cuando falta el dataset
```

Comando:

```powershell
python -m pytest tests/test_model_comparison.py -v
```

### 25.2 Tests del dashboard

Comando:

```powershell
python -m pytest tests/test_dashboard_data.py -v
```

### 25.3 Tests de API

Comando:

```powershell
python -m pytest tests/test_api.py -v
```

### 25.4 Todos los tests

```powershell
python -m pytest tests/ -v
```

---

## 26. CI/CD

El proyecto utiliza GitHub Actions para integración continua.

Archivo:

```text
.github/workflows/ci.yml
```

El workflow valida:

```text
- checkout del repositorio
- instalación de Python
- instalación de dependencias
- importación de la API
- ejecución de tests
- validación de Docker Compose
- construcción de imagen Docker
```

Esto permite detectar errores antes de integrar cambios a la rama principal.

---

## 27. Ejecución local recomendada

```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar ETL
python etl/etl_sinca_main.py
python etl/etl_meteo_main.py

# Entrenar modelo principal
python models/train_model.py

# Generar predicción
python models/predict.py

# Generar pronóstico de 7 días
python etl/extract_meteo_forecast.py
python models/train_forecast_model.py
python models/predict_7_days.py

# Comparar modelos
python models/compare_classification_models.py
python models/compare_regression_models.py

# Ejecutar dashboard
streamlit run dashboards/app.py --server.port 8502

# Ejecutar API
uvicorn api.main:app --reload --port 8000

# Ejecutar tests
python -m pytest tests/ -v
```

---

## 28. Manejo de errores frecuentes

### 28.1 Falta dataset base

Si falta:

```text
data/processed/dataset_modelo_base.csv
```

Ejecutar:

```powershell
python etl/etl_sinca_main.py
python etl/etl_meteo_main.py
```

### 28.2 Faltan métricas del modelo

Si falta:

```text
models/metrics/model_metrics.json
```

Ejecutar:

```powershell
python models/train_model.py
```

### 28.3 Faltan archivos de comparación

Ejecutar:

```powershell
python models/compare_classification_models.py
python models/compare_regression_models.py
```

### 28.4 Faltan predicciones

Ejecutar:

```powershell
python models/predict.py
python etl/extract_meteo_forecast.py
python models/train_forecast_model.py
python models/predict_7_days.py
```

### 28.5 El dashboard muestra información antigua

Ejecutar:

```powershell
streamlit cache clear
streamlit run dashboards/app.py --server.port 8502
```

---

## 29. Decisiones técnicas defendibles

### 29.1 Uso de una estación piloto

El proyecto utiliza Puente Alto como caso piloto para asegurar un flujo completo end-to-end.

La arquitectura puede escalar a múltiples estaciones agregando:

```text
- tabla de estaciones
- coordenadas por estación
- unión por fecha + estación
- variable estación/comuna en el modelo
- filtro de estación en dashboard y API
```

### 29.2 Uso de RandomForest

RandomForest es defendible porque entrega buen rendimiento, robustez, interpretabilidad mediante importancia de variables y facilidad de implementación en datos tabulares.

### 29.3 Uso de comparación de modelos

La comparación permite justificar que el modelo final fue seleccionado mediante métricas y no por preferencia.

### 29.4 Uso de validación temporal

La validación temporal es necesaria porque el problema tiene naturaleza cronológica.

### 29.5 Uso de dashboard

El dashboard permite transformar el análisis técnico en una herramienta visual para usuarios no técnicos.

### 29.6 Uso de API

La API permite exponer resultados para integración con otros sistemas.

### 29.7 Uso de Docker y CI/CD

Docker y CI/CD acercan el proyecto a una solución reproducible y mantenible.

---

## 30. Mejora futura: múltiples estaciones

Una mejora futura relevante es agregar más estaciones de Santiago.

Para implementarlo sería necesario:

```text
- agregar más archivos SINCA
- crear catálogo de estaciones con latitud y longitud
- consultar Open-Meteo por coordenadas de cada estación
- integrar datos por fecha + estación
- reentrenar modelos con variable estación o comuna
- agregar filtros por estación en dashboard
- extender endpoints de API para estación específica
```

Esta mejora no se incluye en la versión actual para priorizar un flujo completo y funcional.

---

## 31. Conclusión técnica

AireChile Analytics implementa una solución integral de ciencia de datos aplicada a calidad del aire.

El proyecto incluye:

```text
- datos reales
- pipeline ETL
- variables derivadas
- modelos supervisados
- comparación de modelos
- tuning con validación temporal
- predicción del día siguiente
- pronóstico de 7 días
- dashboard interactivo
- API REST
- Docker
- PostgreSQL
- tests automatizados
- CI/CD
```

La incorporación de comparación de modelos y visualización en dashboard fortalece la justificación técnica del sistema, ya que permite demostrar que la selección del modelo final está respaldada por métricas, validación y evidencia reproducible.
