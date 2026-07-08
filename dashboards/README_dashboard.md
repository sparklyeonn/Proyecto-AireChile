# Dashboard — AireChile Analytics

## 1. Descripción

El dashboard de **AireChile Analytics** es una aplicación interactiva desarrollada con Streamlit para visualizar datos históricos, variables meteorológicas, predicciones de calidad del aire, pronóstico de 7 días, métricas del modelo y comparación de modelos de Machine Learning.

El dashboard transforma los resultados técnicos del proyecto en una herramienta visual para apoyar la toma de decisiones en municipalidades, colegios, clínicas, empresas y ciudadanía.

Archivo principal:

```text
dashboards/app.py
```

---

## 2. Objetivo del dashboard

El objetivo del dashboard es mostrar de forma clara:

```text
- evolución histórica de MP2.5
- clasificación de calidad del aire
- variables meteorológicas asociadas
- predicción del día siguiente
- pronóstico de MP2.5 para 7 días
- métricas del modelo principal
- matriz de confusión
- importancia de variables
- comparación de modelos de clasificación y regresión
- estado técnico de archivos del pipeline
```

---

## 3. Requisitos previos

Antes de ejecutar el dashboard, se recomienda tener activado el entorno virtual e instaladas las dependencias del proyecto.

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 4. Archivos requeridos

El dashboard necesita leer archivos generados previamente por el pipeline.

### 4.1 Dataset base

```text
data/processed/dataset_modelo_base.csv
```

Este archivo contiene el dataset final integrado entre SINCA y Open-Meteo.

Se genera con:

```powershell
python etl/etl_sinca_main.py
python etl/etl_meteo_main.py
```

---

### 4.2 Predicción del día siguiente

```text
data/processed/prediccion_actual.csv
```

Se genera con:

```powershell
python models/predict.py
```

---

### 4.3 Pronóstico de 7 días

```text
data/processed/prediccion_7_dias.csv
```

Se genera con:

```powershell
python etl/extract_meteo_forecast.py
python models/train_forecast_model.py
python models/predict_7_days.py
```

---

### 4.4 Métricas del modelo principal

```text
models/metrics/model_metrics.json
models/metrics/feature_importance.csv
models/metrics/confusion_matrix.csv
```

Se generan con:

```powershell
python models/train_model.py
```

---

### 4.5 Archivos de comparación de modelos

```text
models/metrics/classification_model_comparison.csv
models/metrics/classification_model_comparison_summary.json
models/metrics/classification_tuning_results.json
models/metrics/regression_model_comparison.csv
models/metrics/regression_model_comparison_summary.json
models/metrics/regression_tuning_results.json
```

Se generan con:

```powershell
python models/compare_classification_models.py
python models/compare_regression_models.py
```

Estos archivos son opcionales para abrir el dashboard, pero necesarios para visualizar la sección de comparación de modelos.

---

## 5. Ejecución del dashboard

Desde la raíz del proyecto:

```powershell
streamlit run dashboards/app.py --server.port 8502
```

Luego abrir en el navegador:

```text
http://localhost:8502
```

Si se usa el puerto por defecto de Streamlit:

```powershell
streamlit run dashboards/app.py
```

El navegador abrirá normalmente:

```text
http://localhost:8501
```

---

## 6. Secciones del dashboard

El dashboard contiene las siguientes secciones principales:

```text
🏠 Inicio
📈 Histórico
🌡️ Meteorología
🔮 Predicción
🤖 Modelo
⚙️ Vista técnica
```

---

## 7. Sección Inicio

La sección **Inicio** muestra una vista general del producto.

Incluye:

```text
- descripción del problema ambiental
- propuesta de valor del sistema
- sectores beneficiados
- estado general del sistema
- cantidad de registros
- rango temporal del dataset
- último valor de MP2.5
- predicción del día siguiente
- confianza de la predicción
- gráfico mensual resumido de MP2.5
```

Esta sección está pensada para explicar el proyecto desde un enfoque comercial y no solo técnico.

---

## 8. Sección Histórico

La sección **Histórico** permite analizar la evolución de MP2.5 en el tiempo.

Incluye:

```text
- filtro por rango de fechas
- serie temporal diaria de MP2.5
- promedio móvil de 7 días
- umbrales de calidad del aire
- distribución de días por clase
- distribución porcentual
- resumen estadístico
- tabla de datos históricos
```

Clases utilizadas:

```text
buena
regular
mala
```

Criterios:

```text
buena:   MP2.5 <= 25 µg/m³
regular: MP2.5 > 25 y <= 50 µg/m³
mala:    MP2.5 > 50 µg/m³
```

---

## 9. Sección Meteorología

La sección **Meteorología** muestra variables climáticas obtenidas desde Open-Meteo.

Incluye:

```text
- temperatura máxima
- temperatura mínima
- temperatura promedio
- humedad relativa
- velocidad del viento
- precipitación
- relación entre MP2.5 y variables meteorológicas
```

También permite seleccionar año y revisar gráficos separados por variable.

---

## 10. Sección Predicción

La sección **Predicción** muestra dos bloques principales.

### 10.1 Pronóstico para los próximos 7 días

Muestra:

```text
- tarjetas tipo semáforo por día
- fecha de cada predicción
- MP2.5 estimado
- nivel de calidad del aire predicho
- gráfico de línea de MP2.5 estimado
- tabla de pronóstico
- recomendación semanal
```

Este bloque usa:

```text
data/processed/prediccion_7_dias.csv
```

---

### 10.2 Predicción del día siguiente

Muestra:

```text
- fecha base
- fecha predicha
- nivel predicho
- confianza del modelo
- probabilidades por clase
```

Este bloque usa:

```text
data/processed/prediccion_actual.csv
```

---

## 11. Sección Modelo

La sección **Modelo** muestra el rendimiento y la explicación técnica del modelo predictivo.

Incluye pestañas para:

```text
📊 Por clase
🎯 Feature importance
🔲 Confusión
📚 Metodología
🔬 Comparación
```

---

### 11.1 Pestaña Por clase

Muestra las métricas por clase del modelo principal:

```text
precision
recall
f1-score
```

Permite revisar cómo se comporta el modelo para cada categoría:

```text
buena
regular
mala
```

---

### 11.2 Pestaña Feature importance

Muestra la importancia de las variables utilizadas por el modelo.

Archivo utilizado:

```text
models/metrics/feature_importance.csv
```

Esta vista ayuda a explicar qué variables influyen más en la predicción.

---

### 11.3 Pestaña Confusión

Muestra la matriz de confusión del modelo.

Archivo utilizado:

```text
models/metrics/confusion_matrix.csv
```

La diagonal principal representa predicciones correctas. Los valores fuera de la diagonal corresponden a errores de clasificación.

---

### 11.4 Pestaña Metodología

Explica decisiones técnicas del modelo, incluyendo:

```text
- por qué se usa RandomForest
- por qué se usa class_weight="balanced"
- por qué se usa split temporal
- por qué no se usa shuffle aleatorio con series temporales
```

---

### 11.5 Pestaña Comparación

Esta pestaña incorpora la comparación formal de modelos.

Permite visualizar:

```text
- comparación de modelos de clasificación
- comparación de modelos de regresión
- mejor modelo según la métrica seleccionada
- justificación técnica
- resultados del tuning
```

#### Clasificación

Archivo utilizado:

```text
models/metrics/classification_model_comparison.csv
```

Modelos comparados:

```text
LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
GradientBoostingClassifier
```

Métricas:

```text
accuracy
precision_weighted
recall_weighted
f1_weighted
```

Métrica principal:

```text
f1_weighted
```

#### Regresión

Archivo utilizado:

```text
models/metrics/regression_model_comparison.csv
```

Modelos comparados:

```text
LinearRegression
DecisionTreeRegressor
RandomForestRegressor
GradientBoostingRegressor
```

Métricas:

```text
MAE
RMSE
R2
```

Métrica principal:

```text
RMSE
```

#### Tuning

Archivos utilizados:

```text
models/metrics/classification_tuning_results.json
models/metrics/regression_tuning_results.json
```

El tuning se realiza con:

```text
GridSearchCV
TimeSeriesSplit
```

El uso de `TimeSeriesSplit` permite respetar el orden temporal de los datos y evitar fuga de información futura.

---

## 12. Sección Vista técnica

La sección **Vista técnica** muestra el estado del pipeline y los archivos requeridos.

Incluye:

```text
- diagrama textual del flujo del pipeline
- estado de archivos requeridos
- estado de archivos de comparación de modelos
- detalles del dataset base
- columnas y tipos de datos
- distribución de la variable objetivo
- comandos para ejecutar el pipeline completo
```

Esta sección sirve para demostrar que el dashboard no es una vista aislada, sino parte de un sistema completo.

---

## 13. Manejo de archivos faltantes

El dashboard está diseñado para no romperse si falta algún archivo opcional.

Si falta un archivo requerido, muestra mensajes como:

```text
Ejecuta: python etl/etl_meteo_main.py
Ejecuta: python models/train_model.py
Ejecuta: python models/compare_classification_models.py
Ejecuta: python models/compare_regression_models.py
```

Esto permite identificar rápidamente qué parte del pipeline falta ejecutar.

---

## 14. Orden recomendado antes de abrir el dashboard

Para ver todas las secciones completas, ejecutar:

```powershell
python etl/etl_sinca_main.py
python etl/etl_meteo_main.py

python models/train_model.py
python models/predict.py

python etl/extract_meteo_forecast.py
python models/train_forecast_model.py
python models/predict_7_days.py

python models/compare_classification_models.py
python models/compare_regression_models.py

streamlit run dashboards/app.py --server.port 8502
```

---

## 15. Pruebas relacionadas con el dashboard

Ejecutar pruebas del dashboard:

```powershell
python -m pytest tests/test_dashboard_data.py -v
```

Ejecutar pruebas de comparación de modelos:

```powershell
python -m pytest tests/test_model_comparison.py -v
```

Ejecutar todos los tests:

```powershell
python -m pytest tests/ -v
```

---

## 16. Solución de problemas

### 16.1 El dashboard no abre

Verificar instalación de Streamlit:

```powershell
pip install streamlit
```

Ejecutar desde la raíz del proyecto:

```powershell
streamlit run dashboards/app.py --server.port 8502
```

---

### 16.2 No aparece el dataset

Ejecutar:

```powershell
python etl/etl_sinca_main.py
python etl/etl_meteo_main.py
```

---

### 16.3 No aparece la predicción del día siguiente

Ejecutar:

```powershell
python models/predict.py
```

---

### 16.4 No aparece el pronóstico de 7 días

Ejecutar:

```powershell
python etl/extract_meteo_forecast.py
python models/train_forecast_model.py
python models/predict_7_days.py
```

---

### 16.5 No aparecen métricas del modelo

Ejecutar:

```powershell
python models/train_model.py
```

---

### 16.6 No aparece comparación de modelos

Ejecutar:

```powershell
python models/compare_classification_models.py
python models/compare_regression_models.py
```

---

### 16.7 Streamlit muestra información antigua

Limpiar caché:

```powershell
streamlit cache clear
streamlit run dashboards/app.py --server.port 8502
```

---

### 16.8 El puerto 8502 está ocupado

Usar otro puerto:

```powershell
streamlit run dashboards/app.py --server.port 8503
```

---

## 17. Decisiones técnicas del dashboard

### 17.1 Uso de Streamlit

Se usa Streamlit porque permite crear aplicaciones de datos de forma rápida, clara y adecuada para una presentación académica.

### 17.2 Uso de Plotly

Se usa Plotly para gráficos interactivos, especialmente series temporales, barras, tortas, dispersión y heatmaps.

### 17.3 Uso de caché

El dashboard utiliza caché de Streamlit para evitar recargar archivos en cada interacción.

### 17.4 Separación por secciones

La navegación lateral permite separar el análisis en bloques:

```text
producto
histórico
meteorología
predicción
modelo
vista técnica
```

Esto facilita la defensa del proyecto.

### 17.5 Comparación de modelos en dashboard

Agregar la comparación al dashboard permite demostrar visualmente que el modelo final fue seleccionado con evidencia cuantitativa.

---

## 18. Relación con la evaluación

El dashboard aporta evidencia para los siguientes aspectos del proyecto:

```text
- visualización de datos
- análisis histórico
- predicción con Machine Learning
- interpretación de métricas
- comparación de modelos
- justificación técnica
- producto funcional
- demo para presentación
```

---

## 19. Comando rápido

Para abrir el dashboard después de tener todos los archivos generados:

```powershell
streamlit run dashboards/app.py --server.port 8502
```

---

## 20. Estado esperado

Cuando el dashboard está completamente cargado, debe mostrar:

```text
- dataset histórico
- predicción del día siguiente
- pronóstico de 7 días
- métricas del modelo
- importancia de variables
- matriz de confusión
- comparación de modelos
- estado técnico del pipeline
```
