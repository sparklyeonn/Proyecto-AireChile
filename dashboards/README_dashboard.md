# Dashboard — AireChile Analytics

Dashboard interactivo construido con Streamlit que transforma el pipeline
técnico en un producto comercial demostrable.

-----

## Qué muestra el dashboard

|Sección            |Contenido                                                         |
|-------------------|------------------------------------------------------------------|
|🏠 **Inicio**       |Propuesta de valor, KPIs principales, serie mensual de MP2.5      |
|📈 **Histórico**    |Serie diaria de MP2.5, distribución por clase, filtro de fechas   |
|🌡️ **Meteorología** |Temperatura, humedad, viento, precipitación y correlaciones       |
|🔮 **Predicción**   |Semáforo visual, probabilidades por clase, recomendación operativa|
|🤖 **Modelo**       |Métricas por clase, feature importance, matriz de confusión       |
|⚙️ **Vista técnica**|Estado de archivos, columnas del dataset, comandos del pipeline   |

-----

## Archivos que necesita

El dashboard lee solo archivos locales ya generados por el pipeline:

|Archivo                                 |Generado por                  |
|----------------------------------------|------------------------------|
|`data/processed/dataset_modelo_base.csv`|`python etl/etl_meteo_main.py`|
|`data/processed/prediccion_actual.csv`  |`python models/predict.py`    |
|`models/metrics/model_metrics.json`     |`python models/train_model.py`|
|`models/metrics/feature_importance.csv` |`python models/train_model.py`|
|`models/metrics/confusion_matrix.csv`   |`python models/train_model.py`|

Si falta algún archivo, el dashboard muestra un mensaje de error con
el comando exacto para generarlo.

-----

## Cómo ejecutarlo

```bash
# 1. Activar entorno virtual
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows PowerShell

# 2. Instalar dependencias (si no están instaladas)
pip install streamlit plotly

# 3. Lanzar el dashboard
streamlit run dashboards/app.py
```

El dashboard se abre automáticamente en `http://localhost:8501`.

-----

## Qué hacer si falta un archivo

Si el dashboard muestra un error de archivo no encontrado, ejecuta el
pipeline en este orden desde la raíz del proyecto:

```bash
# Paso 1: ETL de datos SINCA
python etl/etl_sinca_main.py

# Paso 2: ETL meteorológico y merge
python etl/etl_meteo_main.py

# Paso 3: Entrenamiento del modelo
python models/train_model.py

# Paso 4: Generar predicción actual
python models/predict.py

# Paso 5: Lanzar dashboard
streamlit run dashboards/app.py
```

-----

## Cómo se conecta con el ETL y el modelo

```
data/raw/sinca_*.csv
        ↓
etl/etl_sinca_main.py → data/processed/sinca_transformado.csv
        ↓
etl/etl_meteo_main.py → data/processed/dataset_modelo_base.csv
        ↓
models/train_model.py → models/model.pkl + models/metrics/
        ↓
models/predict.py     → data/processed/prediccion_actual.csv
        ↓
dashboards/app.py     ← lee todos los archivos anteriores
```

El dashboard no escribe ningún archivo — solo lee. Esto garantiza
que el pipeline ETL y el modelo son la única fuente de verdad.

-----

## Navegación

El dashboard usa una barra lateral con `st.sidebar.radio` para navegar
entre secciones. Puedes ejecutar la aplicación sin PostgreSQL ni Docker:
todo corre con archivos CSV locales.