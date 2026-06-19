# Documentación técnica — AireChile Analytics

## 1. Descripción general del sistema

**AireChile Analytics** es una solución de ciencia de datos orientada al análisis, visualización y predicción de calidad del aire en Santiago de Chile.

El sistema trabaja principalmente con datos de material particulado fino **MP2.5** de la estación **Puente Alto**, provenientes de SINCA, y los complementa con información meteorológica obtenida desde Open-Meteo.

La solución permite:

* Procesar datos históricos de calidad del aire.
* Integrar variables meteorológicas.
* Construir un dataset consolidado.
* Entrenar modelos predictivos.
* Generar una predicción del día siguiente.
* Generar un pronóstico de calidad del aire para los próximos 7 días.
* Visualizar resultados en un dashboard interactivo.
* Persistir información en PostgreSQL.
* Ejecutar el sistema localmente o mediante Docker.
* Validar componentes mediante tests automatizados.

---

## 2. Arquitectura general

El flujo técnico del proyecto sigue una arquitectura de tipo pipeline:

```text
SINCA CSV
   ↓
ETL SINCA
   ↓
Datos MP2.5 procesados
   ↓
Open-Meteo Historical API
   ↓
ETL Meteorológico
   ↓
Merge SINCA + Meteorología
   ↓
Dataset modelo base
   ↓
Modelos predictivos
   ↓
Predicción día siguiente + Pronóstico 7 días
   ↓
Dashboard Streamlit
   ↓
PostgreSQL
   ↓
Docker
```

El sistema separa responsabilidades en carpetas:

```text
etl/          → extracción, transformación y carga de datos
models/       → entrenamiento y predicción de modelos
dashboards/   → aplicación Streamlit
database/     → conexión, schema e inicialización PostgreSQL
tests/        → pruebas automatizadas
docker/       → documentación Docker
data/         → archivos crudos y procesados
docs/         → documentación técnica
```

---

## 3. Fuentes de datos

El proyecto utiliza tres tipos de fuentes de datos:

| Tipo     | Fuente                    | Descripción                                     |
| -------- | ------------------------- | ----------------------------------------------- |
| CSV      | SINCA / MMA Chile         | Datos históricos de MP2.5                       |
| API REST | Open-Meteo Historical API | Datos meteorológicos históricos                 |
| API REST | Open-Meteo Forecast API   | Datos meteorológicos futuros                    |
| SQL      | PostgreSQL                | Persistencia de datos procesados y predicciones |

---

## 4. Datos SINCA

El archivo principal de calidad del aire se ubica en:

```text
data/raw/sinca_puente_alto_mp25_2022_2026.csv
```

Actualmente se trabaja con la estación:

```text
Estación: Puente Alto
Comuna: Puente Alto
Variable principal: MP2.5
```

El ETL SINCA procesa los datos originales y genera columnas limpias para el modelo.

Columnas principales generadas:

| Columna                          | Descripción                        |
| -------------------------------- | ---------------------------------- |
| fecha                            | Fecha del registro                 |
| estacion                         | Nombre de la estación              |
| comuna                           | Comuna asociada                    |
| mp25                             | Valor de material particulado fino |
| estado_registro                  | Tipo de registro usado             |
| nivel_calidad_aire               | Clasificación del día actual       |
| mes                              | Mes del año                        |
| dia_semana                       | Día de la semana                   |
| mp25_dia_anterior                | Valor MP2.5 del día anterior       |
| mp25_promedio_7d                 | Promedio móvil de 7 días           |
| nivel_calidad_aire_dia_siguiente | Variable objetivo del clasificador |

Clasificación utilizada:

| Nivel   | Rango MP2.5                 |
| ------- | --------------------------- |
| Buena   | 0 a 25 µg/m³                |
| Regular | Mayor a 25 y hasta 50 µg/m³ |
| Mala    | Mayor a 50 µg/m³            |

---

## 5. ETL SINCA

Archivo principal:

```text
etl/etl_sinca_main.py
```

Este pipeline ejecuta:

1. Extracción del archivo CSV.
2. Limpieza de columnas.
3. Construcción de la variable `mp25`.
4. Clasificación de calidad del aire.
5. Creación de variables temporales.
6. Creación de variables rezagadas.
7. Generación del target del día siguiente.
8. Guardado del archivo procesado.

Comando de ejecución:

```powershell
python etl/etl_sinca_main.py
```

En la versión actual, el proceso puede generar el archivo:

```text
data/processed/sinca_puente_alto_mp25_limpio.csv
```

El merge meteorológico espera el archivo:

```text
data/processed/sinca_transformado.csv
```

Por lo tanto, si es necesario, se debe copiar el archivo limpio con el nombre esperado:

```powershell
Copy-Item data\processed\sinca_puente_alto_mp25_limpio.csv data\processed\sinca_transformado.csv -Force
```

---

## 6. ETL meteorológico

El sistema obtiene datos históricos desde Open-Meteo Historical API.

Archivo principal:

```text
etl/etl_meteo_main.py
```

Este pipeline ejecuta:

1. Extracción meteorológica histórica.
2. Transformación y normalización.
3. Merge con datos SINCA.
4. Generación del dataset final para el modelo.

Comando:

```powershell
python etl/etl_meteo_main.py
```

Variables meteorológicas utilizadas:

| Variable             | Descripción                 |
| -------------------- | --------------------------- |
| temperatura_max      | Temperatura máxima diaria   |
| temperatura_min      | Temperatura mínima diaria   |
| temperatura_promedio | Temperatura promedio diaria |
| humedad_relativa     | Humedad relativa promedio   |
| velocidad_viento     | Velocidad del viento        |
| precipitacion        | Precipitación diaria        |

Archivo final generado:

```text
data/processed/dataset_modelo_base.csv
```

Este archivo es la base principal para entrenar los modelos predictivos.

---

## 7. Dataset modelo base

El archivo consolidado se ubica en:

```text
data/processed/dataset_modelo_base.csv
```

Este dataset contiene información de calidad del aire y meteorología en una sola tabla.

Variables principales:

```text
fecha
estacion
comuna
mp25
nivel_calidad_aire
mp25_dia_anterior
mp25_promedio_7d
nivel_calidad_aire_dia_siguiente
mes
dia_semana
temperatura_max
temperatura_min
temperatura_promedio
humedad_relativa
velocidad_viento
precipitacion
```

Este dataset se utiliza para:

* Entrenar el modelo clasificador.
* Entrenar el modelo regresor.
* Generar predicciones.
* Alimentar visualizaciones del dashboard.
* Cargar información a PostgreSQL.

---

## 8. Modelo clasificador del día siguiente

Archivo de entrenamiento:

```text
models/train_model.py
```

Archivo de predicción:

```text
models/predict.py
```

Modelo utilizado:

```text
RandomForestClassifier
```

Objetivo:

```text
Predecir si la calidad del aire del día siguiente será buena, regular o mala.
```

Variable objetivo:

```text
nivel_calidad_aire_dia_siguiente
```

Variables utilizadas:

```text
mp25
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

Comandos:

```powershell
python models/train_model.py
python models/predict.py
```

Archivo generado:

```text
data/processed/prediccion_actual.csv
```

---

## 9. Modelo regresor para pronóstico a 7 días

Para mejorar el resultado final del proyecto, se agregó un modelo adicional para pronosticar los próximos 7 días.

Archivos principales:

```text
models/train_forecast_model.py
models/predict_7_days.py
etl/extract_meteo_forecast.py
```

Modelo utilizado:

```text
RandomForestRegressor
```

Objetivo:

```text
Predecir el valor numérico estimado de MP2.5 para los próximos días.
```

El uso de un modelo regresor permite generar valores numéricos de MP2.5 y luego clasificarlos como:

```text
buena
regular
mala
```

El pronóstico se realiza de forma recursiva:

1. Se toma el último valor real disponible de MP2.5.
2. Se descarga meteorología futura desde Open-Meteo Forecast API.
3. Se predice el MP2.5 del día 1.
4. El valor estimado del día 1 se usa como rezago para el día 2.
5. El proceso se repite hasta completar 7 días.
6. Cada valor estimado se clasifica según los umbrales definidos.

Comandos:

```powershell
python models/train_forecast_model.py
python etl/extract_meteo_forecast.py
python models/predict_7_days.py
```

Archivo generado:

```text
data/processed/prediccion_7_dias.csv
```

Columnas principales:

| Columna                     | Descripción                        |
| --------------------------- | ---------------------------------- |
| fecha                       | Fecha pronosticada                 |
| estacion                    | Estación usada                     |
| comuna                      | Comuna asociada                    |
| mp25_estimado               | Valor estimado de MP2.5            |
| nivel_calidad_aire_predicho | Categoría predicha                 |
| temperatura_max             | Temperatura máxima pronosticada    |
| temperatura_min             | Temperatura mínima pronosticada    |
| temperatura_promedio        | Temperatura promedio pronosticada  |
| humedad_relativa            | Humedad relativa pronosticada      |
| velocidad_viento            | Velocidad del viento pronosticada  |
| precipitacion               | Precipitación pronosticada         |
| horizonte_dia               | Día del pronóstico, de 1 a 7       |
| fecha_generacion            | Fecha de generación del pronóstico |

---

## 10. Dashboard Streamlit

Archivo principal:

```text
dashboards/app.py
```

El dashboard transforma los resultados técnicos en una interfaz visual e interactiva.

Secciones disponibles:

| Sección       | Descripción                                              |
| ------------- | -------------------------------------------------------- |
| Inicio        | Presentación general del producto                        |
| Histórico     | Evolución histórica del MP2.5                            |
| Meteorología  | Visualización de variables climáticas                    |
| Predicción    | Predicción del día siguiente y pronóstico 7 días         |
| Modelo        | Métricas, matriz de confusión e importancia de variables |
| Vista técnica | Estado de archivos y pipeline                            |

Ejecución local:

```powershell
streamlit cache clear
streamlit run dashboards/app.py --server.port 8502
```

URL local recomendada:

```text
http://localhost:8502
```

Se usa el puerto `8502` en ejecución local para evitar conflicto con Docker, que usa `8501`.

La sección de predicción muestra:

* Pronóstico para los próximos 7 días.
* Tarjetas tipo semáforo.
* Gráfico de línea de MP2.5 estimado.
* Tabla de pronóstico.
* Recomendación semanal.
* Predicción del día siguiente con el modelo clasificador.

---

## 11. PostgreSQL

El proyecto incorpora PostgreSQL como capa de persistencia.

Archivos principales:

```text
database/schema.sql
database/db.py
database/init_db.py
etl/load_postgres.py
etl/etl_postgres_main.py
```

Tablas principales:

| Tabla               | Descripción                          |
| ------------------- | ------------------------------------ |
| estaciones          | Información de estación y comuna     |
| mediciones_sinca    | Datos históricos de calidad del aire |
| meteorologia        | Datos meteorológicos históricos      |
| dataset_modelo      | Dataset consolidado                  |
| predicciones_modelo | Predicción del día siguiente         |
| predicciones_7_dias | Pronóstico de 7 días                 |
| log_etl             | Registro de ejecución del pipeline   |

Inicializar base de datos:

```powershell
python database/init_db.py
```

Cargar datos:

```powershell
python etl/load_postgres.py
```

Ejecutar pipeline PostgreSQL completo:

```powershell
python etl/etl_postgres_main.py
```

Verificar tabla de predicciones:

```powershell
python -c "from database.db import get_engine; import pandas as pd; engine=get_engine(); print(pd.read_sql('SELECT COUNT(*) AS total FROM predicciones_7_dias', engine))"
```

---

## 12. Docker

El proyecto puede ejecutarse con Docker Compose.

Archivos principales:

```text
Dockerfile
docker-compose.yml
.dockerignore
docker/README_docker.md
```

Servicios:

| Servicio  | Descripción              |
| --------- | ------------------------ |
| dashboard | Aplicación Streamlit     |
| postgres  | Base de datos PostgreSQL |

Puertos utilizados:

| Servicio                  | Puerto |
| ------------------------- | ------ |
| Streamlit Docker          | 8501   |
| PostgreSQL Docker externo | 5433   |
| PostgreSQL interno Docker | 5432   |

Validar configuración:

```powershell
docker compose config
```

Ejecutar con Docker:

```powershell
docker compose up --build
```

Dashboard Docker:

```text
http://localhost:8501
```

Ejecutar comandos dentro del contenedor:

```powershell
docker compose exec dashboard python models/train_forecast_model.py
docker compose exec dashboard python etl/extract_meteo_forecast.py
docker compose exec dashboard python models/predict_7_days.py
docker compose exec dashboard python etl/load_postgres.py
```

Verificar PostgreSQL en Docker:

```powershell
docker compose exec postgres psql -U postgres -d airechile -c "\dt"
docker compose exec postgres psql -U postgres -d airechile -c "SELECT COUNT(*) FROM predicciones_7_dias;"
```

Detener contenedores:

```powershell
docker compose down
```

No se recomienda usar:

```powershell
docker compose down -v
```

a menos que se quiera borrar el volumen de PostgreSQL.

---

## 13. Variables de entorno

El proyecto utiliza `.env` para definir rutas, credenciales y parámetros.

Archivo base:

```text
.env.example
```

El archivo real debe crearse localmente:

```powershell
Copy-Item .env.example .env
```

Variables principales:

```env
SINCA_RAW_PATH=data/raw/sinca_puente_alto_mp25_2022_2026.csv
SINCA_PROCESSED_PATH=data/processed/sinca_transformado.csv

OPENMETEO_BASE_URL=https://archive-api.open-meteo.com/v1/archive
METEO_LATITUDE=-33.6117
METEO_LONGITUDE=-70.5758
METEO_START_DATE=2022-01-01
METEO_END_DATE=2026-06-16

METEO_RAW_PATH=data/raw/open_meteo_puente_alto_2022_2026.csv
METEO_PROCESSED_PATH=data/processed/open_meteo_transformado.csv

DATASET_MODELO_BASE_PATH=data/processed/dataset_modelo_base.csv

MODEL_OUTPUT_PATH=models/model.pkl
MODEL_METRICS_DIR=models/metrics
PREDICTION_OUTPUT_PATH=data/processed/prediccion_actual.csv

FORECAST_DAYS=7
OPENMETEO_FORECAST_URL=https://api.open-meteo.com/v1/forecast
METEO_FORECAST_RAW_PATH=data/raw/open_meteo_forecast_7dias.csv
PREDICTION_7_DAYS_OUTPUT_PATH=data/processed/prediccion_7_dias.csv
FORECAST_MODEL_OUTPUT_PATH=models/model_mp25_regressor.pkl
FORECAST_METRICS_DIR=models/metrics

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airechile
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_SCHEMA=public
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/airechile
```

La contraseña de PostgreSQL debe ajustarse a la configuración local del equipo.

---

## 14. Testing

Los tests se ubican en:

```text
tests/
```

Archivos principales:

```text
test_transform_sinca.py
test_merge_sinca_meteo.py
test_model.py
test_dashboard_data.py
test_postgres_config.py
test_docker_config.py
test_forecast_7_days.py
```

Ejecutar todos los tests:

```powershell
python -m pytest tests/ -v
```

Ejecutar solo tests del pronóstico:

```powershell
python -m pytest tests/test_forecast_7_days.py -v
```

Los tests validan:

* Estructura de datos procesados.
* Transformación SINCA.
* Merge con meteorología.
* Funcionamiento del modelo.
* Lectura de datos en dashboard.
* Configuración PostgreSQL.
* Configuración Docker.
* Generación del pronóstico de 7 días.

---

## 15. Flujo completo recomendado

Para regenerar todo el proyecto desde datos actualizados:

```powershell
python etl/etl_sinca_main.py
Copy-Item data\processed\sinca_puente_alto_mp25_limpio.csv data\processed\sinca_transformado.csv -Force
python etl/etl_meteo_main.py
python models/train_model.py
python models/predict.py
python models/train_forecast_model.py
python etl/extract_meteo_forecast.py
python models/predict_7_days.py
python database/init_db.py
python etl/load_postgres.py
python -m pytest tests/ -v
streamlit run dashboards/app.py --server.port 8502
```

Para probar con Docker:

```powershell
docker compose down
docker compose config
docker compose up --build
```

---

## 16. Archivos generados que no deben subirse

Los siguientes archivos son generados localmente o pueden contener datos pesados/sensibles:

```text
.env
data/raw/
data/processed/
models/model.pkl
models/model_mp25_regressor.pkl
models/metrics/
__pycache__/
.pytest_cache/
```

Estos archivos deben estar considerados en `.gitignore`.

---

## 17. Limitaciones técnicas

El sistema actual presenta algunas limitaciones:

* Se trabaja principalmente con una estación: Puente Alto.
* La predicción depende de la calidad y actualización del CSV SINCA.
* El pronóstico a 7 días usa predicción recursiva, por lo que el error puede acumularse.
* La meteorología futura depende de una API externa.
* El modelo no reemplaza alertas oficiales de la autoridad ambiental.
* El pipeline SINCA y el merge deben mantener nombres de archivos consistentes.

---

## 18. Mejoras futuras

Posibles mejoras técnicas:

* Incorporar más estaciones de Santiago.
* Comparar resultados entre comunas.
* Agregar nuevos contaminantes como MP10.
* Automatizar la descarga directa desde SINCA.
* Incorporar modelos alternativos como XGBoost o LightGBM.
* Publicar el dashboard en la nube.
* Agregar alertas automáticas.
* Incorporar mapas geográficos interactivos.
* Mejorar la consistencia de nombres de archivos en el pipeline.

---

## 19. Estado actual del proyecto

El proyecto cuenta con:

* ETL SINCA.
* ETL Open-Meteo.
* Dataset consolidado.
* Modelo clasificador.
* Modelo regresor para pronóstico de 7 días.
* Dashboard Streamlit.
* PostgreSQL.
* Docker.
* Tests automatizados.
* Flujo de trabajo con Git y GitHub.

La solución es ejecutable tanto de forma local como mediante Docker.

---
