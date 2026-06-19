# 🌿 AireChile Analytics

**AireChile Analytics** es una solución de análisis y predicción de calidad del aire para Santiago de Chile. El sistema integra datos históricos de material particulado MP2.5 desde SINCA, datos meteorológicos desde Open-Meteo, modelos predictivos de Machine Learning, PostgreSQL, Docker y un dashboard interactivo desarrollado en Streamlit.

El objetivo principal es transformar datos ambientales en información útil para anticipar condiciones de calidad del aire y apoyar la toma de decisiones en municipalidades, colegios, clínicas, empresas y ciudadanía.

---

## 📌 Descripción del problema

En Santiago, especialmente durante los meses de invierno, se presentan episodios críticos de contaminación atmosférica asociados principalmente al material particulado fino MP2.5.

El problema es que gran parte de la información disponible es reactiva: permite revisar lo que ya ocurrió, pero no anticipar de forma simple lo que puede ocurrir en los próximos días.

AireChile Analytics busca resolver ese problema mediante un sistema que:

* Procesa datos históricos de calidad del aire.
* Integra variables meteorológicas.
* Entrena modelos predictivos.
* Genera una predicción diaria.
* Entrega un pronóstico de calidad del aire para los próximos 7 días.
* Visualiza los resultados en un dashboard interactivo.
* Permite persistir datos en PostgreSQL.
* Puede ejecutarse de forma local o mediante Docker.

---

## 🎯 Objetivo general

Desarrollar una solución de ciencia de datos capaz de analizar, visualizar y predecir condiciones de calidad del aire en Santiago, usando datos reales de MP2.5 y variables meteorológicas.

---

## 🎯 Objetivos específicos

* Construir un pipeline ETL para datos SINCA.
* Integrar datos meteorológicos históricos desde Open-Meteo.
* Generar un dataset consolidado para modelamiento.
* Entrenar modelos predictivos de Machine Learning.
* Clasificar la calidad del aire como `buena`, `regular` o `mala`.
* Generar pronóstico de calidad del aire para los próximos 7 días.
* Crear un dashboard interactivo en Streamlit.
* Persistir información en PostgreSQL.
* Contenerizar la solución mediante Docker y Docker Compose.
* Validar la solución mediante tests automatizados.

---

## 🧩 Fuentes de datos

El proyecto utiliza tres tipos de fuentes de datos:

| Tipo de fuente | Fuente                    | Uso                                                      |
| -------------- | ------------------------- | -------------------------------------------------------- |
| CSV            | SINCA / MMA Chile         | Datos históricos de MP2.5                                |
| API REST       | Open-Meteo Historical API | Meteorología histórica                                   |
| API REST       | Open-Meteo Forecast API   | Meteorología futura para pronóstico                      |
| SQL            | PostgreSQL                | Persistencia de datos procesados, modelos y predicciones |

---

## 🏭 Datos SINCA

El archivo principal de calidad del aire corresponde a la estación **Puente Alto**.

Ruta esperada:

```text
data/raw/sinca_puente_alto_mp25_2022_2026.csv
```

El pipeline procesa las columnas originales de SINCA, construye la variable `mp25` y clasifica la calidad del aire según los siguientes criterios:

| Nivel   | Rango MP2.5                 |
| ------- | --------------------------- |
| Buena   | 0 a 25 µg/m³                |
| Regular | Mayor a 25 y hasta 50 µg/m³ |
| Mala    | Mayor a 50 µg/m³            |

---

## 🌦️ Datos meteorológicos

Los datos meteorológicos se obtienen desde Open-Meteo e incluyen:

* Temperatura máxima.
* Temperatura mínima.
* Temperatura promedio.
* Humedad relativa.
* Velocidad del viento.
* Precipitación.

Estos datos se integran con los registros de MP2.5 para crear el dataset final del modelo.

---

## 🤖 Modelos predictivos

El proyecto utiliza dos enfoques de Machine Learning:

### 1. Modelo clasificador

Archivo principal:

```text
models/train_model.py
```

Modelo utilizado:

```text
RandomForestClassifier
```

Objetivo:

```text
Predecir la categoría de calidad del aire del día siguiente:
buena, regular o mala.
```

Archivo de predicción generado:

```text
data/processed/prediccion_actual.csv
```

---

### 2. Modelo regresor para pronóstico a 7 días

Archivos principales:

```text
models/train_forecast_model.py
models/predict_7_days.py
```

Modelo utilizado:

```text
RandomForestRegressor
```

Objetivo:

```text
Estimar el valor numérico de MP2.5 para los próximos 7 días.
```

Luego, cada valor estimado de MP2.5 se clasifica como:

```text
buena
regular
mala
```

Archivo generado:

```text
data/processed/prediccion_7_dias.csv
```

La predicción se realiza de forma recursiva:

1. Se toma el último valor real de MP2.5 disponible.
2. Se obtiene meteorología futura desde Open-Meteo.
3. Se predice el MP2.5 del día 1.
4. Ese valor estimado se usa como variable rezagada para el día 2.
5. El proceso se repite hasta completar 7 días.

---

## 📊 Dashboard interactivo

El dashboard fue desarrollado con **Streamlit**.

Archivo principal:

```text
dashboards/app.py
```

El dashboard incluye las siguientes secciones:

| Sección       | Descripción                         |
| ------------- | ----------------------------------- |
| Inicio        | Presentación comercial del producto |
| Histórico     | Evolución histórica de MP2.5        |
| Meteorología  | Variables climáticas asociadas      |
| Predicción    | Pronóstico de calidad del aire      |
| Modelo        | Métricas y explicación del modelo   |
| Vista técnica | Estado de archivos y pipeline       |

La sección de predicción muestra:

* Pronóstico para los próximos 7 días.
* Tarjetas tipo semáforo.
* Tabla con valores estimados.
* Gráfico de línea de MP2.5 estimado.
* Recomendación semanal según la peor condición esperada.
* Predicción del día siguiente con el modelo clasificador.

---

## 🗄️ Base de datos PostgreSQL

El proyecto utiliza PostgreSQL para persistir los resultados del pipeline.

Archivos principales:

```text
database/schema.sql
database/db.py
database/init_db.py
etl/load_postgres.py
etl/etl_postgres_main.py
```

Tablas principales:

| Tabla               | Descripción                             |
| ------------------- | --------------------------------------- |
| estaciones          | Información de estación y comuna        |
| mediciones_sinca    | Mediciones históricas de MP2.5          |
| meteorologia        | Datos meteorológicos históricos         |
| dataset_modelo      | Dataset consolidado para modelamiento   |
| predicciones_modelo | Predicción del día siguiente            |
| predicciones_7_dias | Pronóstico de calidad del aire a 7 días |
| log_etl             | Registro de ejecución del pipeline      |

---

## 🐳 Docker

El proyecto puede ejecutarse mediante Docker Compose.

Archivos principales:

```text
Dockerfile
docker-compose.yml
.dockerignore
docker/README_docker.md
```

Servicios definidos:

| Servicio  | Descripción              |
| --------- | ------------------------ |
| dashboard | Aplicación Streamlit     |
| postgres  | Base de datos PostgreSQL |

Puertos:

| Servicio                  | Puerto                |
| ------------------------- | --------------------- |
| Dashboard Docker          | http://localhost:8501 |
| PostgreSQL Docker externo | localhost:5433        |
| PostgreSQL interno Docker | postgres:5432         |

Se usa el puerto externo `5433` para PostgreSQL en Docker, con el fin de evitar conflicto con una instalación local de PostgreSQL en Windows que usa el puerto `5432`.

---

## 📁 Estructura del proyecto

```text
Proyecto-AireChile/
├── dashboards/
│   ├── app.py
│   └── README_dashboard.md
├── data/
│   ├── raw/
│   └── processed/
├── database/
│   ├── __init__.py
│   ├── db.py
│   ├── init_db.py
│   └── schema.sql
├── docker/
│   └── README_docker.md
├── docs/
├── etl/
│   ├── extract_sinca.py
│   ├── transform_sinca.py
│   ├── etl_sinca_main.py
│   ├── extract_meteo.py
│   ├── transform_meteo.py
│   ├── merge_sinca_meteo.py
│   ├── etl_meteo_main.py
│   ├── extract_meteo_forecast.py
│   ├── load_postgres.py
│   └── etl_postgres_main.py
├── models/
│   ├── train_model.py
│   ├── predict.py
│   ├── train_forecast_model.py
│   └── predict_7_days.py
├── tests/
│   ├── test_transform_sinca.py
│   ├── test_merge_sinca_meteo.py
│   ├── test_model.py
│   ├── test_dashboard_data.py
│   ├── test_postgres_config.py
│   ├── test_docker_config.py
│   └── test_forecast_7_days.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## ⚙️ Requisitos

Para ejecutar el proyecto localmente se requiere:

* Python 3.11 o superior.
* PostgreSQL instalado localmente, si se desea usar base de datos local.
* Docker Desktop, si se desea ejecutar con contenedores.
* Git.
* PowerShell en Windows.

---

## 📦 Instalación local

Clonar el repositorio:

```powershell
git clone https://github.com/sparklyeonn/Proyecto-AireChile.git
cd Proyecto-AireChile
```

Crear entorno virtual:

```powershell
python -m venv .venv
```

Activar entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔐 Variables de entorno

Crear un archivo `.env` a partir de `.env.example`.

```powershell
Copy-Item .env.example .env
```

Variables principales:

```env
PROJECT_NAME=AireChile Analytics
LOG_LEVEL=INFO

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

MODEL_DATASET_PATH=data/processed/dataset_modelo_base.csv
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

> Nota: en ejecución local, la contraseña de PostgreSQL debe coincidir con la configurada en el equipo.

---

## 📥 Preparación de datos

Ubicar el archivo SINCA en:

```text
data/raw/sinca_puente_alto_mp25_2022_2026.csv
```

Si se actualiza el CSV de SINCA, se debe revisar la última fecha disponible y actualizar en `.env`:

```env
METEO_END_DATE=YYYY-MM-DD
```

Ejemplo:

```env
METEO_END_DATE=2026-06-16
```

---

## 🔄 Ejecución del pipeline ETL

### 1. ETL SINCA

```powershell
python etl/etl_sinca_main.py
```

Este proceso genera el archivo procesado de SINCA.

Si el pipeline genera un archivo con nombre distinto, asegurar que exista:

```text
data/processed/sinca_transformado.csv
```

En caso necesario, se puede copiar el archivo generado:

```powershell
Copy-Item data\processed\sinca_puente_alto_mp25_limpio.csv data\processed\sinca_transformado.csv -Force
```

---

### 2. ETL meteorológico + merge

```powershell
python etl/etl_meteo_main.py
```

Este proceso:

* Descarga meteorología histórica desde Open-Meteo.
* Transforma variables meteorológicas.
* Realiza merge con datos SINCA.
* Genera el dataset final:

```text
data/processed/dataset_modelo_base.csv
```

---

## 🤖 Entrenamiento y predicción

### Modelo clasificador del día siguiente

Entrenar modelo:

```powershell
python models/train_model.py
```

Generar predicción:

```powershell
python models/predict.py
```

Archivo generado:

```text
data/processed/prediccion_actual.csv
```

---

### Modelo regresor para pronóstico a 7 días

Entrenar modelo regresor:

```powershell
python models/train_forecast_model.py
```

Descargar meteorología futura:

```powershell
python etl/extract_meteo_forecast.py
```

Generar pronóstico de 7 días:

```powershell
python models/predict_7_days.py
```

Archivo generado:

```text
data/processed/prediccion_7_dias.csv
```

Ver resultado:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/prediccion_7_dias.csv'); print(df.to_string())"
```

---

## 🗄️ Inicializar y cargar PostgreSQL

Inicializar tablas:

```powershell
python database/init_db.py
```

Cargar datos a PostgreSQL:

```powershell
python etl/load_postgres.py
```

También se puede ejecutar el flujo PostgreSQL completo:

```powershell
python etl/etl_postgres_main.py
```

Verificar tabla de predicción de 7 días:

```powershell
python -c "from database.db import get_engine; import pandas as pd; engine=get_engine(); print(pd.read_sql('SELECT COUNT(*) AS total FROM predicciones_7_dias', engine))"
```

---

## 📊 Ejecutar dashboard local

En Windows, si el puerto `8501` está ocupado por Docker, se recomienda ejecutar Streamlit en el puerto `8502`.

```powershell
streamlit cache clear
streamlit run dashboards/app.py --server.port 8502
```

Abrir en navegador:

```text
http://localhost:8502
```

---

## 🐳 Ejecutar con Docker

Validar configuración:

```powershell
docker compose config
```

Construir y levantar servicios:

```powershell
docker compose up --build
```

Abrir dashboard:

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

Verificar PostgreSQL dentro de Docker:

```powershell
docker compose exec postgres psql -U postgres -d airechile -c "\dt"
docker compose exec postgres psql -U postgres -d airechile -c "SELECT COUNT(*) FROM predicciones_7_dias;"
```

Detener contenedores sin borrar datos:

```powershell
docker compose down
```

> No usar `docker compose down -v` salvo que se quiera borrar el volumen de PostgreSQL.

---

## 🧪 Testing

Ejecutar tests del pronóstico:

```powershell
python -m pytest tests/test_forecast_7_days.py -v
```

Ejecutar todos los tests:

```powershell
python -m pytest tests/ -v
```

Los tests validan:

* Transformación de datos SINCA.
* Merge SINCA + meteorología.
* Entrenamiento y predicción.
* Dashboard.
* Configuración PostgreSQL.
* Configuración Docker.
* Pronóstico de 7 días.

---

## 🚀 Flujo completo local recomendado

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

---

## 📈 Resultado esperado

Al ejecutar el sistema, se obtiene:

* Dataset histórico consolidado de calidad del aire y meteorología.
* Modelo clasificador para predicción del día siguiente.
* Modelo regresor para estimar MP2.5 futuro.
* Pronóstico de calidad del aire para los próximos 7 días.
* Dashboard interactivo con visualizaciones.
* Base de datos PostgreSQL con información persistida.
* Proyecto ejecutable localmente y con Docker.

---

## 💼 Valor del proyecto

AireChile Analytics puede apoyar decisiones en:

* Municipalidades.
* Colegios.
* Clínicas.
* Empresas.
* Actividades deportivas.
* Ciudadanía general.

Ejemplos de uso:

* Anticipar días con mala calidad del aire.
* Recomendar reducción de actividad física al aire libre.
* Apoyar medidas preventivas en colegios.
* Informar a grupos sensibles.
* Preparar protocolos ante episodios críticos.

---

## ⚠️ Limitaciones

* Actualmente se trabaja principalmente con la estación Puente Alto.
* El modelo depende de la calidad y actualización del CSV SINCA.
* La predicción de 7 días usa un enfoque recursivo, por lo que el error puede acumularse.
* La meteorología futura proviene de una API externa.
* El sistema no reemplaza alertas oficiales de la autoridad ambiental.

---

## 🔮 Mejoras futuras

* Incorporar más estaciones de Santiago.
* Comparar resultados por comuna.
* Agregar MP10 u otros contaminantes.
* Automatizar la descarga directa desde SINCA.
* Agregar modelos alternativos como XGBoost, LightGBM o series temporales.
* Publicar el dashboard en la nube.
* Crear alertas automáticas por correo o mensajería.
* Incorporar mapas geográficos interactivos.

---

## 🧾 Archivos que no deben subirse al repositorio

Los siguientes archivos son generados localmente o contienen información sensible:

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

## 🧑‍💻 Comandos Git útiles

Ver estado:

```powershell
git status
```

Agregar cambios de código:

```powershell
git add dashboards/app.py
git add etl/
git add models/
git add tests/
git add database/
git add README.md
```

Crear commit:

```powershell
git commit -m "docs(readme): actualizar documentación principal del proyecto"
```

Subir cambios:

```powershell
git push origin feature/setup
```

---

## 👥 Equipo

Proyecto desarrollado para la evaluación de **SCY1101**.

Integrantes:

* Génesis Baeza
* Jimena Galicia
* Alejandra González
* Constanza González

---

## 📌 Estado del proyecto

El proyecto cuenta con:

* ETL automatizado.
* Datos SINCA.
* Datos Open-Meteo.
* Modelo clasificador.
* Modelo regresor.
* Predicción a 7 días.
* Dashboard Streamlit.
* PostgreSQL.
* Docker.
* Tests automatizados.
* Flujo de trabajo con Git y GitHub.

---
