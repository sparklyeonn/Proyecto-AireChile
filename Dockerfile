# =============================================================
# Dockerfile — AireChile Analytics
# =============================================================
# Imagen base: Python 3.11 slim (menor tamaño, sin extras innecesarios)
FROM python:3.11-slim

# Metadatos del proyecto
LABEL maintainer="AireChile Analytics"
LABEL description="Plataforma de análisis y alerta predictiva de calidad del aire"

# Evitar prompts interactivos durante la instalación de paquetes del sistema
ENV DEBIAN_FRONTEND=noninteractive

# Variables de entorno de Python:
#   PYTHONDONTWRITEBYTECODE=1 → no genera archivos .pyc
#   PYTHONUNBUFFERED=1        → logs salen directamente a consola (no se bufferean)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para psycopg2
# libpq-dev: cabeceras de PostgreSQL para compilar psycopg2
# gcc: compilador para paquetes con extensiones C
# Después de instalar, limpiar cache de apt para reducir el tamaño de la imagen
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (aprovecha la caché de capas de Docker:
# si requirements.txt no cambia, esta capa no se reconstruye)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente del proyecto
# (los archivos excluidos en .dockerignore no se copian)
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Puerto de healthcheck para el servicio de dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Comando por defecto: lanzar el dashboard Streamlit
# --server.address=0.0.0.0 → accesible desde fuera del contenedor
# --server.port=8501        → puerto estándar de Streamlit
# --server.headless=true    → no abrir navegador automáticamente
CMD ["streamlit", "run", "dashboards/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]