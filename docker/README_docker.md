# Docker — AireChile Analytics

Guía para levantar el proyecto completo con Docker: base de datos
PostgreSQL y dashboard Streamlit en contenedores reproducibles.

---

## Servicios incluidos

| Servicio | Imagen | Puerto | Descripción |
|---|---|---|---|
| `postgres` | postgres:15-alpine | 5432 | Base de datos PostgreSQL |
| `dashboard` | Dockerfile local | 8501 | Dashboard Streamlit |

---

## Requisitos previos

- Docker Desktop instalado y corriendo
- PowerShell en Windows
- Archivo `.env` configurado (copiar desde `.env.example`)

---

## Configuración inicial

```powershell
# 1. Copiar el archivo de variables de entorno
Copy-Item .env.example .env

# 2. Editar .env con tus valores si es necesario
# (los valores por defecto funcionan para desarrollo local)
notepad .env
```

---

## Diferencia entre ejecución local y con Docker

| Aspecto | Local | Docker |
|---|---|---|
| PostgreSQL host | `localhost` | `postgres` (nombre del servicio) |
| PostgreSQL puerto | `5432` | `5432` (mapeado al host) |
| Archivos de datos | `data/` local | Montados como volumen en `/app/data` |
| Modelos | `models/` local | Montados como volumen en `/app/models` |
| Schema SQL | Ejecutar `python database/init_db.py` | Ejecutado automáticamente al iniciar PostgreSQL |

La diferencia clave es el host de PostgreSQL. Localmente usas
`POSTGRES_HOST=localhost`. Con Docker, el contenedor del dashboard
se comunica con el contenedor de PostgreSQL usando el nombre del
servicio: `POSTGRES_HOST=postgres`.

El archivo `docker-compose.yml` ya maneja esto automáticamente —
sobreescribe `POSTGRES_HOST=postgres` para el servicio dashboard.
Tu `.env` puede seguir teniendo `POSTGRES_HOST=localhost` para uso local.

---

## Comandos de uso

### Primera vez

```powershell
# Construir imágenes y levantar servicios
docker compose up --build

# O en background (sin ver logs en tiempo real)
docker compose up --build -d
```

### Uso diario

```powershell
# Levantar servicios (sin reconstruir)
docker compose up -d

# Ver logs de todos los servicios
docker compose logs -f

# Ver logs solo del dashboard
docker compose logs -f dashboard

# Ver logs solo de PostgreSQL
docker compose logs -f postgres

# Detener servicios (mantiene los datos)
docker compose stop

# Detener y eliminar contenedores (mantiene el volumen de datos)
docker compose down

# Detener, eliminar contenedores Y volumen de datos
# ⚠️ Esto borra la base de datos
docker compose down -v
```

### Ejecutar scripts dentro del contenedor

```powershell
# Inicializar la base de datos manualmente
docker compose exec dashboard python database/init_db.py

# Ejecutar el ETL SINCA
docker compose exec dashboard python etl/etl_sinca_main.py

# Ejecutar el ETL meteorológico y merge
docker compose exec dashboard python etl/etl_meteo_main.py

# Entrenar el modelo
docker compose exec dashboard python models/train_model.py

# Generar predicción
docker compose exec dashboard python models/predict.py

# Cargar datos a PostgreSQL
docker compose exec dashboard python etl/load_postgres.py

# Pipeline PostgreSQL completo
docker compose exec dashboard python etl/etl_postgres_main.py

# Ejecutar todos los tests
docker compose exec dashboard python -m pytest tests/ -v
```

### Verificar el estado

```powershell
# Ver contenedores corriendo
docker compose ps

# Conectarse a PostgreSQL directamente
docker compose exec postgres psql -U postgres -d airechile

# Ver tablas creadas
docker compose exec postgres psql -U postgres -d airechile -c "\dt"

# Ver cantidad de registros
docker compose exec postgres psql -U postgres -d airechile -c "SELECT COUNT(*) FROM dataset_modelo;"
```

### Abrir el dashboard

Una vez que los servicios están corriendo, abre en tu navegador:

```
http://localhost:8501
```

---

## Flujo completo recomendado

```
1. docker compose up --build -d
   → PostgreSQL inicia y ejecuta schema.sql automáticamente
   → Dashboard inicia esperando a que PostgreSQL esté listo

2. (Si tienes datos CSV en data/raw/)
   docker compose exec dashboard python etl/etl_sinca_main.py
   docker compose exec dashboard python etl/etl_meteo_main.py
   docker compose exec dashboard python models/train_model.py
   docker compose exec dashboard python models/predict.py
   docker compose exec dashboard python etl/load_postgres.py

3. Abrir http://localhost:8501
```

---

## Solución de errores comunes

**`Cannot connect to the Docker daemon`**
Docker Desktop no está corriendo. Ábrelo antes de ejecutar comandos.

**`port is already allocated`**
El puerto 5432 o 8501 ya está en uso.
```powershell
# Detener PostgreSQL local si está corriendo
# O cambiar el puerto en .env:
# POSTGRES_PORT=5433
```

**`no such service: dashboard`**
Asegúrate de estar en la carpeta raíz del proyecto donde está `docker-compose.yml`.

**El dashboard muestra archivos no encontrados**
Los archivos CSV y modelos deben existir en `data/processed/` y `models/`
en tu máquina local (se montan como volúmenes). Ejecuta el pipeline primero.