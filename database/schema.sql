-- =============================================================
-- database/schema.sql
-- AireChile Analytics — Definición del esquema PostgreSQL
--
-- Crea todas las tablas del sistema con sus restricciones.
-- Ejecutar con: python database/init_db.py
-- O manualmente: psql -U postgres -d airechile -f database/schema.sql
-- =============================================================


-- -------------------------------------------------------------
-- Tabla: estaciones
-- Catálogo de estaciones de monitoreo de calidad del aire.
-- Referenciada por mediciones_sinca.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS estaciones (
    id        SERIAL PRIMARY KEY,
    nombre    VARCHAR(100) NOT NULL,
    comuna    VARCHAR(80)  NOT NULL,
    latitud   DECIMAL(9, 6),
    longitud  DECIMAL(9, 6),
    activa    BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (nombre, comuna)
);

-- -------------------------------------------------------------
-- Tabla: mediciones_sinca
-- Datos históricos de calidad del aire del SINCA por estación.
-- Una fila por día por estación.
-- Corresponde al CSV: data/processed/sinca_transformado.csv
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mediciones_sinca (
    id                               SERIAL PRIMARY KEY,
    fecha                            DATE         NOT NULL,
    estacion                         VARCHAR(100) NOT NULL,
    comuna                           VARCHAR(80)  NOT NULL,
    mp25                             DECIMAL(8, 2),
    estado_registro                  VARCHAR(20)
        CHECK (estado_registro IN ('validado', 'preliminar', 'sin_dato')),
    nivel_calidad_aire               VARCHAR(10)
        CHECK (nivel_calidad_aire IN ('buena', 'regular', 'mala', 'sin_dato')),
    mes                              SMALLINT CHECK (mes BETWEEN 1 AND 12),
    dia_semana                       SMALLINT CHECK (dia_semana BETWEEN 0 AND 6),
    mp25_dia_anterior                DECIMAL(8, 2),
    mp25_promedio_7d                 DECIMAL(8, 2),
    nivel_calidad_aire_dia_siguiente VARCHAR(10)
        CHECK (nivel_calidad_aire_dia_siguiente IN ('buena', 'regular', 'mala')),
    UNIQUE (fecha, estacion)
);

-- -------------------------------------------------------------
-- Tabla: meteorologia
-- Datos meteorológicos diarios de Open-Meteo.
-- Una fila por día (para una ubicación fija: Puente Alto/Santiago).
-- Corresponde al CSV: data/processed/open_meteo_transformado.csv
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS meteorologia (
    id                   SERIAL PRIMARY KEY,
    fecha                DATE         NOT NULL UNIQUE,
    temperatura_max      DECIMAL(5, 2),
    temperatura_min      DECIMAL(5, 2),
    temperatura_promedio DECIMAL(5, 2),
    humedad_relativa     DECIMAL(5, 2),
    velocidad_viento     DECIMAL(6, 2),
    precipitacion        DECIMAL(6, 2)
);

-- -------------------------------------------------------------
-- Tabla: dataset_modelo
-- Dataset consolidado listo para entrenar el RandomForest.
-- Combina mediciones_sinca + meteorologia por fecha.
-- Corresponde al CSV: data/processed/dataset_modelo_base.csv
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dataset_modelo (
    id                               SERIAL PRIMARY KEY,
    fecha                            DATE         NOT NULL,
    estacion                         VARCHAR(100) NOT NULL,
    comuna                           VARCHAR(80)  NOT NULL,
    mp25                             DECIMAL(8, 2),
    estado_registro                  VARCHAR(20),
    nivel_calidad_aire               VARCHAR(10),
    mes                              SMALLINT,
    dia_semana                       SMALLINT,
    mp25_dia_anterior                DECIMAL(8, 2),
    mp25_promedio_7d                 DECIMAL(8, 2),
    temperatura_max                  DECIMAL(5, 2),
    temperatura_min                  DECIMAL(5, 2),
    temperatura_promedio             DECIMAL(5, 2),
    humedad_relativa                 DECIMAL(5, 2),
    velocidad_viento                 DECIMAL(6, 2),
    precipitacion                    DECIMAL(6, 2),
    nivel_calidad_aire_dia_siguiente VARCHAR(10),
    UNIQUE (fecha, estacion)
);

-- -------------------------------------------------------------
-- Tabla: predicciones_modelo
-- Predicciones generadas por predict.py.
-- Corresponde al CSV: data/processed/prediccion_actual.csv
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predicciones_modelo (
    id                   SERIAL PRIMARY KEY,
    fecha_base           DATE         NOT NULL,
    fecha_predicha       DATE         NOT NULL UNIQUE,
    nivel_predicho       VARCHAR(10)  NOT NULL
        CHECK (nivel_predicho IN ('buena', 'regular', 'mala')),
    probabilidad_predicho DECIMAL(6, 4) CHECK (probabilidad_predicho BETWEEN 0 AND 1),
    prob_buena           DECIMAL(6, 4),
    prob_regular         DECIMAL(6, 4),
    prob_mala            DECIMAL(6, 4),
    mp25_base            DECIMAL(8, 2),
    generado_en          TIMESTAMP NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------
-- Tabla: log_etl
-- Registro de ejecuciones del pipeline ETL.
-- Permite auditar cuándo y cómo se actualizaron los datos.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS log_etl (
    id                   SERIAL PRIMARY KEY,
    proceso              VARCHAR(100) NOT NULL,
    fecha_ejecucion      TIMESTAMP    NOT NULL DEFAULT NOW(),
    estado               VARCHAR(20)  NOT NULL
        CHECK (estado IN ('OK', 'ERROR', 'ADVERTENCIA')),
    registros_procesados INTEGER,
    mensaje              TEXT
);

-- -------------------------------------------------------------
-- Comentarios de tablas (documentación en la base de datos)
-- -------------------------------------------------------------
COMMENT ON TABLE estaciones           IS 'Catálogo de estaciones de monitoreo SINCA';
COMMENT ON TABLE mediciones_sinca     IS 'Mediciones históricas de calidad del aire por estación';
COMMENT ON TABLE meteorologia         IS 'Datos meteorológicos diarios Open-Meteo para Puente Alto';
COMMENT ON TABLE dataset_modelo       IS 'Dataset consolidado SINCA+Meteo listo para RandomForest';
COMMENT ON TABLE predicciones_modelo  IS 'Predicciones del modelo de calidad del aire';
COMMENT ON TABLE log_etl              IS 'Log de ejecuciones del pipeline ETL';

-- -------------------------------------------------------------
-- Tabla: predicciones_7_dias
-- Pronóstico de calidad del aire para los próximos 7 días.
-- Corresponde al CSV: data/processed/prediccion_7_dias.csv
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predicciones_7_dias (
    id                          SERIAL PRIMARY KEY,
    fecha                       DATE        NOT NULL,
    estacion                    VARCHAR(100) NOT NULL,
    comuna                      VARCHAR(80)  NOT NULL,
    mp25_estimado               DECIMAL(8, 2),
    nivel_calidad_aire_predicho VARCHAR(10)
        CHECK (nivel_calidad_aire_predicho IN ('buena', 'regular', 'mala')),
    temperatura_max             DECIMAL(5, 2),
    temperatura_min             DECIMAL(5, 2),
    temperatura_promedio        DECIMAL(5, 2),
    humedad_relativa            DECIMAL(5, 2),
    velocidad_viento            DECIMAL(6, 2),
    precipitacion               DECIMAL(6, 2),
    horizonte_dia               SMALLINT CHECK (horizonte_dia BETWEEN 1 AND 7),
    fecha_generacion            TIMESTAMP,
    UNIQUE (fecha, estacion)
);

COMMENT ON TABLE predicciones_7_dias IS 'Pronóstico recursivo de calidad del aire a 7 días';