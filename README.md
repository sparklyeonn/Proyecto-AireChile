# Modelo Predictivo — AireChile Analytics

## Descripción general

**AireChile Analytics** es una plataforma de análisis y alerta predictiva de calidad del aire enfocada en Santiago de Chile.

El módulo predictivo tiene como objetivo estimar si la calidad del aire del **día siguiente** será **buena**, **regular** o **mala**, usando datos históricos de contaminación y variables meteorológicas.

Este modelo permite que el sistema no solo muestre lo que ya ocurrió, sino que también entregue una alerta anticipada para apoyar la toma de decisiones en instituciones como municipalidades, colegios, clínicas, empresas o entidades públicas.

---

## Objetivo del modelo

Predecir la calidad del aire esperada para el día siguiente en Santiago de Chile a partir de las condiciones del día actual.

La predicción se entrega en tres categorías:

```text
buena
regular
mala
```

---

## Variable objetivo

La variable objetivo del modelo es:

```text
nivel_calidad_aire_dia_siguiente
```

Esta variable representa la calidad del aire esperada para el día siguiente.

### Construcción de la variable objetivo

Para cada registro correspondiente al día `t`, se toma el valor promedio de MP2.5 del día `t+1` y se clasifica según rangos definidos para el proyecto.

De esta forma, el modelo aprende la siguiente lógica:

```text
Con las condiciones de contaminación y clima de hoy,
¿qué nivel de calidad del aire se espera para mañana?
```

---

## Criterio de clasificación

Los niveles de calidad del aire se construyen a partir del valor de MP2.5, siguiendo criterios basados en referencias normativas chilenas para material particulado fino.

| Clase     | Rango MP2.5 µg/m³ | Interpretación                                       |
| --------- | ----------------: | ---------------------------------------------------- |
| `buena`   |            0 – 25 | Calidad aceptable, sin restricciones relevantes      |
| `regular` |         25.1 – 50 | Riesgo moderado, especialmente para grupos sensibles |
| `mala`    |              > 50 | Riesgo alto, posible episodio crítico ambiental      |

Si MP2.5 no se encuentra disponible en alguna fuente, el sistema puede utilizar MP10 como variable secundaria de respaldo.

---

## Algoritmo utilizado

El modelo seleccionado es:

```python
RandomForestClassifier
```

de la librería `scikit-learn`.

### Justificación técnica

Se eligió Random Forest porque:

* Funciona bien con datos tabulares.
* Tolera valores extremos o episodios puntuales de alta contaminación.
* No requiere escalar variables como temperatura, humedad o MP2.5.
* Permite calcular la importancia de cada variable.
* Es más fácil de explicar en una defensa académica que modelos más complejos.
* Puede manejar clases desbalanceadas usando `class_weight='balanced'`.

---

## Variables de entrada

El modelo utiliza variables ambientales, meteorológicas y temporales.

| Variable            | Descripción                                   | Fuente               |
| ------------------- | --------------------------------------------- | -------------------- |
| `mp25`              | Material particulado fino del día actual      | SINCA                |
| `mp10`              | Material particulado grueso del día actual    | SINCA                |
| `temperatura_max`   | Temperatura máxima diaria                     | Open-Meteo           |
| `humedad_relativa`  | Humedad relativa diaria                       | Open-Meteo           |
| `velocidad_viento`  | Velocidad media del viento                    | Open-Meteo           |
| `precipitacion`     | Precipitación acumulada diaria                | Open-Meteo           |
| `mes`               | Mes del año                                   | Derivada de la fecha |
| `dia_semana`        | Día de la semana                              | Derivada de la fecha |
| `mp25_dia_anterior` | MP2.5 registrado el día anterior              | SINCA procesado      |
| `mp25_promedio_7d`  | Promedio móvil de MP2.5 de los últimos 7 días | SINCA procesado      |

---

## Flujo de entrenamiento

El entrenamiento sigue el siguiente proceso:

```text
1. Cargar datos procesados desde PostgreSQL.
2. Eliminar registros con valores nulos en variables críticas.
3. Separar variables predictoras y variable objetivo.
4. Aplicar división temporal:
   - 80% datos de entrenamiento.
   - 20% datos de prueba.
5. Entrenar un RandomForestClassifier.
6. Evaluar el modelo.
7. Guardar el modelo entrenado como model.pkl.
```

---

## Configuración del modelo

Configuración inicial propuesta:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
```

### Explicación de los parámetros

| Parámetro                 | Función                           |
| ------------------------- | --------------------------------- |
| `n_estimators=200`        | Número de árboles del bosque      |
| `max_depth=10`            | Profundidad máxima de cada árbol  |
| `class_weight='balanced'` | Compensa diferencias entre clases |
| `random_state=42`         | Permite reproducir resultados     |

---

## Métricas de evaluación

El modelo se evalúa con las siguientes métricas:

| Métrica             | Uso                                             |
| ------------------- | ----------------------------------------------- |
| Accuracy            | Mide el porcentaje general de aciertos          |
| Precision           | Evalúa qué tan confiable es cada clase predicha |
| Recall              | Mide qué tan bien detecta cada clase real       |
| F1-score            | Balance entre precision y recall                |
| Matriz de confusión | Permite ver en qué clases se equivoca el modelo |

Desde el punto de vista del producto, la clase más importante es:

```text
mala
```

Esto se debe a que un falso negativo puede ser riesgoso. Es decir, si el modelo predice calidad `buena` o `regular`, pero realmente el día siguiente resulta `mala`, el sistema no estaría alertando correctamente un posible episodio crítico.

---

## Resultado esperado

El modelo debe entregar una predicción simple y accionable:

```text
Predicción para mañana: REGULAR
```

También puede mostrar la probabilidad asociada:

```text
Probabilidad estimada: 72%
```

Ejemplo de salida:

| Fecha de predicción | Nivel predicho | Probabilidad |
| ------------------- | -------------- | -----------: |
| 2025-06-15          | regular        |         0.72 |

---

## Integración con el dashboard

La predicción se visualizará en el dashboard de Streamlit dentro de una sección llamada:

```text
Predicción de calidad del aire para mañana
```

Esta sección mostrará:

* Nivel esperado: buena, regular o mala.
* Probabilidad del resultado.
* Variables más influyentes.
* Recomendación operativa simple.

Ejemplo:

```text
Calidad del aire esperada para mañana: MALA

Recomendación:
Evitar actividad física intensa al aire libre.
Priorizar medidas preventivas para grupos sensibles.
```

---

## Interpretación del modelo

Random Forest permite obtener la importancia de cada variable mediante `feature_importances_`.

Se espera que las variables más relevantes sean:

1. `mp25`
2. `mp25_dia_anterior`
3. `mp25_promedio_7d`
4. `velocidad_viento`
5. `humedad_relativa`
6. `mes`

Esta información será mostrada en la vista técnica del dashboard para explicar qué factores influyen más en la predicción.

---

## Limitaciones

El modelo tiene algunas limitaciones importantes:

* Predice un nivel general para Santiago, no necesariamente para cada comuna.
* Depende de la calidad y continuidad de los datos históricos.
* No reemplaza los sistemas oficiales de alerta ambiental.
* El horizonte de predicción inicial es solo de un día.
* No considera todas las variables posibles, como dirección del viento, tráfico vehicular o fuentes industriales específicas.
* Puede verse afectado por eventos atípicos, como incendios, restricciones sanitarias o cambios bruscos en movilidad.

---

## Archivos principales del módulo

```text
models/
├── train_model.py      # Entrena el modelo predictivo
├── predict.py          # Genera predicciones usando el modelo guardado
└── model.pkl           # Modelo entrenado serializado con joblib
```

---

## Ejecución esperada

### Entrenar el modelo

```bash
python models/train_model.py
```

### Generar una predicción

```bash
python models/predict.py
```

---

## Rol dentro de AireChile Analytics

El modelo predictivo es una capa de valor agregado dentro del sistema completo.

El flujo general del producto es:

```text
Datos SINCA + Open-Meteo
        ↓
Pipeline ETL
        ↓
Base de datos PostgreSQL
        ↓
Modelo Random Forest
        ↓
Dashboard Streamlit
        ↓
Alerta visual para el usuario
```

---

## Enfoque comercial

AireChile Analytics no busca ser solo un tablero de gráficos. Su objetivo es transformar datos ambientales en una herramienta de decisión.

La predicción permite anticipar riesgos y entregar alertas claras para:

* Municipalidades.
* Colegios.
* Clínicas.
* Empresas con trabajadores en terreno.
* Instituciones públicas.

Mensaje principal del producto:

```text
AireChile Analytics transforma datos de calidad del aire en alertas predictivas para tomar mejores decisiones en Chile.
```
