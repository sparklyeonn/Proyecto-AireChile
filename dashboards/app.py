"""
dashboards/app.py
=================
AireChile Analytics — Dashboard interactivo Streamlit.

Transforma el pipeline técnico en un producto comercial demostrable
para municipalidades, colegios, clínicas y empresas de Santiago.

Secciones:
    1. Inicio / Producto     → propuesta de valor, KPIs principales
    2. Histórico             → serie temporal MP2.5, distribución por clase
    3. Meteorología          → temperatura, humedad, viento, precipitación
    4. Predicción            → semáforo, probabilidades, recomendación
    5. Modelo                → métricas, feature importance, confusión
    6. Vista técnica         → estado del pipeline, rutas, columnas

Ejecución:
    streamlit run dashboards/app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AireChile Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Rutas de archivos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent

PATHS = {
    "dataset":     ROOT / "data/processed/dataset_modelo_base.csv",
    "prediccion":  ROOT / "data/processed/prediccion_actual.csv",
    "metrics":     ROOT / "models/metrics/model_metrics.json",
    "fi":          ROOT / "models/metrics/feature_importance.csv",
    "cm":          ROOT / "models/metrics/confusion_matrix.csv",
}

# Colores del semáforo y gráficos
COLORES_NIVEL = {
    "buena":   "#27ae60",
    "regular": "#f39c12",
    "mala":    "#e74c3c",
}
COLOR_MP25    = "#2980b9"
COLOR_METEO   = "#8e44ad"

# ---------------------------------------------------------------------------
# Funciones de carga (con caché para no releer en cada interacción)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def cargar_dataset() -> pd.DataFrame | None:
    """Carga el dataset base del modelo."""
    if not PATHS["dataset"].exists():
        return None
    df = pd.read_csv(PATHS["dataset"], parse_dates=["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


@st.cache_data(ttl=300)
def cargar_prediccion() -> pd.DataFrame | None:
    """Carga la predicción actual."""
    if not PATHS["prediccion"].exists():
        return None
    return pd.read_csv(PATHS["prediccion"])


@st.cache_data(ttl=300)
def cargar_metricas() -> dict | None:
    """Carga las métricas del modelo desde JSON."""
    if not PATHS["metrics"].exists():
        return None
    with open(PATHS["metrics"], encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def cargar_feature_importance() -> pd.DataFrame | None:
    if not PATHS["fi"].exists():
        return None
    return pd.read_csv(PATHS["fi"])


@st.cache_data(ttl=300)
def cargar_confusion_matrix() -> pd.DataFrame | None:
    if not PATHS["cm"].exists():
        return None
    return pd.read_csv(PATHS["cm"], index_col=0)


# ---------------------------------------------------------------------------
# CSS personalizado: tipografía limpia y colores sobrios
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Fondo sidebar */
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #ecf0f1 !important; }

    /* Métricas con fondo sutil */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Tarjeta de predicción */
    .semaforo-card {
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        color: white;
        font-size: 1.1rem;
    }

    /* Sección de inicio */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        background: #2980b9;
        color: white;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Barra lateral — navegación
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 AireChile Analytics")
    st.markdown("---")
    seccion = st.radio(
        "Navegar",
        options=[
            "🏠 Inicio",
            "📈 Histórico",
            "🌡️ Meteorología",
            "🔮 Predicción",
            "🤖 Modelo",
            "⚙️ Vista técnica",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Datos: SINCA MMA Chile<br>"
        "Clima: Open-Meteo API<br>"
        "Modelo: RandomForest</small>",
        unsafe_allow_html=True,
    )

# Cargar datos una vez para todas las secciones
df        = cargar_dataset()
pred_df   = cargar_prediccion()
metricas  = cargar_metricas()
fi_df     = cargar_feature_importance()
cm_df     = cargar_confusion_matrix()


# ===========================================================================
# SECCIÓN 1 — INICIO / PRODUCTO
# ===========================================================================
if seccion == "🏠 Inicio":

    # Encabezado hero
    st.markdown(
        '<p class="hero-title">🌿 AireChile Analytics</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">'
        "Transformamos datos ambientales en alertas predictivas para tomar "
        "mejores decisiones en Chile."
        "</p>",
        unsafe_allow_html=True,
    )

    # Badges de sectores
    sectores = ["🏛️ Municipalidades", "🏫 Colegios", "🏥 Clínicas",
                "🏭 Empresas", "🚴 Deporte", "👨‍👩‍👧 Ciudadanos"]
    st.markdown(
        " ".join(f'<span class="badge">{s}</span>' for s in sectores),
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # El problema
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("El problema")
        st.markdown("""
Santiago es una de las ciudades más contaminadas de América Latina.
Cada invierno, los episodios de **preemergencia y emergencia ambiental**
obligan a restricciones vehiculares, cierres de colegios y alertas de salud.

**El problema:** la información disponible es reactiva — dice lo que ya
pasó, no lo que viene.

**La solución:** AireChile Analytics analiza datos históricos de calidad
del aire (SINCA), los combina con condiciones meteorológicas (Open-Meteo)
y predice si el día siguiente será **buena, regular o mala** calidad del
aire, con horas de anticipación.
        """)

    with col2:
        st.subheader("Cómo funciona")
        pasos = [
            ("📥", "Datos SINCA",     "MP2.5, MP10 históricos por estación"),
            ("🌤️", "Open-Meteo",      "Temperatura, viento, humedad, lluvia"),
            ("⚙️", "ETL Pipeline",    "Limpieza, merge y enriquecimiento"),
            ("🤖", "RandomForest",    "Predicción del día siguiente"),
            ("📊", "Dashboard",       "Alertas y visualización en tiempo real"),
        ]
        for icono, titulo, desc in pasos:
            with st.container():
                st.markdown(f"**{icono} {titulo}** — {desc}")

    st.markdown("---")

    # KPIs principales
    st.subheader("Estado actual del sistema")

    if df is None:
        st.error(
            "⚠️ No se encontró `dataset_modelo_base.csv`. "
            "Ejecuta: `python etl/etl_meteo_main.py`"
        )
    else:
        pred_nivel = "—"
        pred_prob  = "—"
        pred_fecha = "—"
        if pred_df is not None:
            pred_nivel = pred_df.iloc[0]["nivel_predicho"].upper()
            pred_prob  = f"{pred_df.iloc[0]['probabilidad_predicho']:.0%}"
            pred_fecha = pred_df.iloc[0]["fecha_predicha"]

        ultimo_mp25 = df["mp25"].dropna().iloc[-1] if df["mp25"].notna().any() else None
        emoji_pred  = {"BUENA": "🟢", "REGULAR": "🟡", "MALA": "🔴"}.get(pred_nivel, "❓")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📋 Registros totales", f"{len(df):,}")
        c2.metric(
            "📅 Rango de datos",
            f"{df['fecha'].min().year} – {df['fecha'].max().year}"
        )
        c3.metric(
            "🏭 Último MP2.5",
            f"{ultimo_mp25:.1f} µg/m³" if ultimo_mp25 else "N/D",
            help="Material particulado fino, µg/m³"
        )
        c4.metric(
            f"Mañana ({pred_fecha})",
            f"{emoji_pred} {pred_nivel}",
        )
        c5.metric("Confianza predicción", pred_prob)

        # Mini gráfico resumen MP2.5
        st.markdown("#### Serie temporal MP2.5 — resumen anual")
        df_anual = (
            df.set_index("fecha")["mp25"]
            .resample("ME")
            .mean()
            .reset_index()
            .rename(columns={"fecha": "Mes", "mp25": "MP2.5 promedio (µg/m³)"})
        )
        fig_resumen = px.line(
            df_anual, x="Mes", y="MP2.5 promedio (µg/m³)",
            title="Evolución mensual de MP2.5 en Puente Alto",
            color_discrete_sequence=[COLOR_MP25],
        )
        fig_resumen.add_hline(y=25, line_dash="dash", line_color="#27ae60",
                              annotation_text="Umbral buena (25 µg/m³)")
        fig_resumen.add_hline(y=50, line_dash="dash", line_color="#e74c3c",
                              annotation_text="Umbral mala (50 µg/m³)")
        fig_resumen.update_layout(height=320, margin=dict(t=50, b=20))
        st.plotly_chart(fig_resumen, use_container_width=True)


# ===========================================================================
# SECCIÓN 2 — HISTÓRICO
# ===========================================================================
elif seccion == "📈 Histórico":
    st.title("📈 Histórico de calidad del aire")
    st.caption("Estación Puente Alto — Datos SINCA MMA Chile")

    if df is None:
        st.error("Archivo no encontrado. Ejecuta: `python etl/etl_meteo_main.py`")
        st.stop()

    # Filtro de fechas
    col1, col2 = st.columns(2)
    fecha_min = df["fecha"].min().date()
    fecha_max = df["fecha"].max().date()

    with col1:
        fecha_inicio = st.date_input(
            "Desde", value=fecha_min, min_value=fecha_min, max_value=fecha_max
        )
    with col2:
        fecha_fin = st.date_input(
            "Hasta", value=fecha_max, min_value=fecha_min, max_value=fecha_max
        )

    mask = (df["fecha"].dt.date >= fecha_inicio) & (df["fecha"].dt.date <= fecha_fin)
    df_filtrado = df[mask].copy()

    if df_filtrado.empty:
        st.warning("No hay datos en el rango seleccionado.")
        st.stop()

    st.caption(f"Mostrando {len(df_filtrado):,} registros")
    st.markdown("---")

    # Gráfico de línea MP2.5
    st.subheader("Concentración diaria de MP2.5")

    # Colorear puntos según nivel
    df_filtrado["color"] = df_filtrado["nivel_calidad_aire"].map(COLORES_NIVEL)

    fig_linea = go.Figure()
    fig_linea.add_trace(go.Scatter(
        x=df_filtrado["fecha"], y=df_filtrado["mp25"],
        mode="lines", name="MP2.5",
        line=dict(color=COLOR_MP25, width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>MP2.5: %{y:.1f} µg/m³<extra></extra>",
    ))
    # Promedio móvil 7 días
    if "mp25_promedio_7d" in df_filtrado.columns:
        fig_linea.add_trace(go.Scatter(
            x=df_filtrado["fecha"], y=df_filtrado["mp25_promedio_7d"],
            mode="lines", name="Promedio 7 días",
            line=dict(color="#e67e22", width=2, dash="dot"),
        ))
    fig_linea.add_hline(y=25, line_dash="dash", line_color="#27ae60",
                        annotation_text="Buena (≤25)")
    fig_linea.add_hline(y=50, line_dash="dash", line_color="#e74c3c",
                        annotation_text="Mala (>50)")
    fig_linea.update_layout(
        height=380, margin=dict(t=30, b=20),
        legend=dict(orientation="h", y=-0.15),
        yaxis_title="µg/m³",
        hovermode="x unified",
    )
    st.plotly_chart(fig_linea, use_container_width=True)

    # Distribución por clase
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Días por calidad del aire")
        conteo = df_filtrado["nivel_calidad_aire"].value_counts().reset_index()
        conteo.columns = ["Nivel", "Días"]
        conteo["Nivel"] = pd.Categorical(
            conteo["Nivel"], categories=["buena", "regular", "mala"]
        )
        conteo = conteo.sort_values("Nivel")
        fig_barras = px.bar(
            conteo, x="Nivel", y="Días",
            color="Nivel",
            color_discrete_map=COLORES_NIVEL,
            text="Días",
        )
        fig_barras.update_traces(textposition="outside")
        fig_barras.update_layout(
            height=320, showlegend=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_barras, use_container_width=True)

    with col2:
        st.subheader("Distribución porcentual")
        fig_pie = px.pie(
            conteo, names="Nivel", values="Días",
            color="Nivel",
            color_discrete_map=COLORES_NIVEL,
            hole=0.4,
        )
        fig_pie.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Resumen estadístico
    st.subheader("Resumen estadístico")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MP2.5 promedio", f"{df_filtrado['mp25'].mean():.1f} µg/m³")
    col2.metric("MP2.5 máximo",   f"{df_filtrado['mp25'].max():.1f} µg/m³")
    col3.metric("MP2.5 mínimo",   f"{df_filtrado['mp25'].min():.1f} µg/m³")
    dias_mala = (df_filtrado["nivel_calidad_aire"] == "mala").sum()
    col4.metric("Días calidad mala", f"{dias_mala:,}")

    # Tabla resumen
    with st.expander("Ver tabla de datos"):
        st.dataframe(
            df_filtrado[["fecha", "mp25", "nivel_calidad_aire",
                         "mp25_dia_anterior", "mp25_promedio_7d"]]
            .rename(columns={
                "fecha": "Fecha", "mp25": "MP2.5",
                "nivel_calidad_aire": "Nivel",
                "mp25_dia_anterior": "MP2.5 ayer",
                "mp25_promedio_7d": "Promedio 7d",
            }),
            use_container_width=True,
        )

    # Leyenda de niveles
    st.markdown("---")
    st.markdown("#### Clasificación de calidad del aire (Norma DS59 Chile)")
    c1, c2, c3 = st.columns(3)
    c1.success("🟢 **Buena** — MP2.5 ≤ 25 µg/m³\nCondiciones saludables para todas las personas.")
    c2.warning("🟡 **Regular** — MP2.5 26–50 µg/m³\nRiesgo moderado. Precaución para grupos sensibles.")
    c3.error("🔴 **Mala** — MP2.5 > 50 µg/m³\nRiesgo alto. Puede activar preemergencia o emergencia.")


# ===========================================================================
# SECCIÓN 3 — METEOROLOGÍA
# ===========================================================================
elif seccion == "🌡️ Meteorología":
    st.title("🌡️ Variables meteorológicas")
    st.caption("Datos históricos desde Open-Meteo API — Santiago, Puente Alto")

    if df is None:
        st.error("Archivo no encontrado. Ejecuta: `python etl/etl_meteo_main.py`")
        st.stop()

    cols_meteo = ["temperatura_max", "temperatura_min", "temperatura_promedio",
                  "humedad_relativa", "velocidad_viento", "precipitacion"]
    cols_faltantes = [c for c in cols_meteo if c not in df.columns]
    if cols_faltantes:
        st.warning(f"Columnas meteorológicas faltantes: {cols_faltantes}")

    # Filtro de año para no sobrecargar los gráficos
    anios = sorted(df["fecha"].dt.year.unique(), reverse=True)
    anio_sel = st.selectbox("Seleccionar año", options=["Todos"] + [str(a) for a in anios])

    df_m = df.copy()
    if anio_sel != "Todos":
        df_m = df_m[df_m["fecha"].dt.year == int(anio_sel)]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🌡️ Temperatura", "💧 Humedad", "💨 Viento",
         "🌧️ Precipitación", "🔗 MP2.5 vs Clima"]
    )

    with tab1:
        if "temperatura_max" in df_m.columns:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=df_m["fecha"], y=df_m["temperatura_max"],
                name="Máxima", line=dict(color="#e74c3c", width=1.5),
            ))
            fig_t.add_trace(go.Scatter(
                x=df_m["fecha"], y=df_m["temperatura_min"],
                name="Mínima", line=dict(color="#3498db", width=1.5),
            ))
            fig_t.add_trace(go.Scatter(
                x=df_m["fecha"], y=df_m["temperatura_promedio"],
                name="Promedio", line=dict(color="#e67e22", width=1.5, dash="dot"),
            ))
            fig_t.update_layout(
                title="Temperatura diaria (°C)", height=380,
                yaxis_title="°C", hovermode="x unified",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Datos de temperatura no disponibles.")

    with tab2:
        if "humedad_relativa" in df_m.columns:
            fig_h = px.line(
                df_m, x="fecha", y="humedad_relativa",
                title="Humedad relativa promedio (%)",
                color_discrete_sequence=[COLOR_METEO],
            )
            fig_h.update_layout(height=350, yaxis_title="%")
            st.plotly_chart(fig_h, use_container_width=True)
            col1, col2 = st.columns(2)
            col1.metric("Humedad promedio", f"{df_m['humedad_relativa'].mean():.1f}%")
            col2.metric("Humedad máxima",   f"{df_m['humedad_relativa'].max():.1f}%")
        else:
            st.info("Datos de humedad no disponibles.")

    with tab3:
        if "velocidad_viento" in df_m.columns:
            fig_v = px.line(
                df_m, x="fecha", y="velocidad_viento",
                title="Velocidad del viento (km/h)",
                color_discrete_sequence=["#1abc9c"],
            )
            fig_v.update_layout(height=350, yaxis_title="km/h")
            st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.info("Datos de viento no disponibles.")

    with tab4:
        if "precipitacion" in df_m.columns:
            df_prec = df_m[df_m["precipitacion"] > 0]
            fig_p = px.bar(
                df_prec, x="fecha", y="precipitacion",
                title="Días con precipitación (mm)",
                color_discrete_sequence=["#3498db"],
            )
            fig_p.update_layout(height=350, yaxis_title="mm")
            st.plotly_chart(fig_p, use_container_width=True)
            col1, col2 = st.columns(2)
            col1.metric("Días con lluvia", f"{len(df_prec):,}")
            col2.metric("Precipitación máxima", f"{df_m['precipitacion'].max():.1f} mm")
        else:
            st.info("Datos de precipitación no disponibles.")

    with tab5:
        st.markdown("#### Relación entre MP2.5 y variables meteorológicas")
        st.caption(
            "Una correlación negativa con el viento indica que días "
            "con más viento tienden a tener menor MP2.5 (mayor dispersión)."
        )
        var_x = st.selectbox(
            "Variable meteorológica",
            options=[c for c in cols_meteo if c in df_m.columns and c != "precipitacion"]
        )
        fig_scatter = px.scatter(
            df_m.dropna(subset=["mp25", var_x]),
            x=var_x, y="mp25",
            color="nivel_calidad_aire",
            color_discrete_map=COLORES_NIVEL,
            opacity=0.6,
            title=f"MP2.5 vs {var_x}",
            trendline="ols",
        )
        fig_scatter.update_layout(height=420, yaxis_title="MP2.5 (µg/m³)")
        st.plotly_chart(fig_scatter, use_container_width=True)


# ===========================================================================
# SECCIÓN 4 — PREDICCIÓN
# ===========================================================================
elif seccion == "🔮 Predicción":
    st.title("🔮 Predicción de calidad del aire")

    if pred_df is None:
        st.error(
            "No se encontró `prediccion_actual.csv`. "
            "Ejecuta: `python models/predict.py`"
        )
        st.stop()

    pred = pred_df.iloc[0]
    nivel = pred["nivel_predicho"]
    color = COLORES_NIVEL.get(nivel, "#7f8c8d")
    prob  = float(pred["probabilidad_predicho"])
    emoji = {"buena": "🟢", "regular": "🟡", "mala": "🔴"}.get(nivel, "❓")

    # Tarjeta semáforo
    st.markdown(
        f"""
        <div class="semaforo-card" style="background-color:{color}; margin-bottom:24px;">
            <div style="font-size:3rem">{emoji}</div>
            <div style="font-size:2rem; font-weight:700; text-transform:uppercase;">
                {nivel}
            </div>
            <div style="font-size:1.1rem; margin-top:8px;">
                Calidad del aire esperada para mañana
            </div>
            <div style="font-size:1.4rem; margin-top:8px; font-weight:600;">
                {pred['fecha_predicha']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Datos de contexto
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fecha base (hoy)", pred["fecha_base"])
    col2.metric("Fecha predicha", pred["fecha_predicha"])
    col3.metric("Nivel predicho", f"{emoji} {nivel.upper()}")
    col4.metric("Confianza", f"{prob:.0%}")

    st.markdown("---")

    # Probabilidades por clase
    col_izq, col_der = st.columns([1, 1.5])

    with col_izq:
        st.subheader("Probabilidades por clase")
        prob_data = pd.DataFrame({
            "Clase": ["Buena", "Regular", "Mala"],
            "Probabilidad": [
                float(pred.get("prob_buena", 0)),
                float(pred.get("prob_regular", 0)),
                float(pred.get("prob_mala", 0)),
            ],
            "Color": ["#27ae60", "#f39c12", "#e74c3c"],
        })
        fig_prob = go.Figure(go.Bar(
            x=prob_data["Probabilidad"],
            y=prob_data["Clase"],
            orientation="h",
            marker_color=prob_data["Color"],
            text=[f"{p:.1%}" for p in prob_data["Probabilidad"]],
            textposition="outside",
        ))
        fig_prob.update_layout(
            height=220, margin=dict(t=10, b=10, l=10, r=60),
            xaxis=dict(range=[0, 1.1], tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown(f"**MP2.5 base (hoy):** {pred.get('mp25_base', 'N/D')} µg/m³")

    with col_der:
        st.subheader("Recomendación operativa")
        if nivel == "buena":
            st.success(
                "✅ **Condiciones aceptables**\n\n"
                "La calidad del aire mañana se espera dentro de rangos normales. "
                "No se anticipan restricciones ambientales. "
                "Actividades al aire libre son seguras para toda la población."
            )
        elif nivel == "regular":
            st.warning(
                "⚠️ **Precaución — grupos sensibles**\n\n"
                "Se esperan niveles moderados de contaminación. "
                "Se recomienda precaución para niños, adultos mayores y personas "
                "con enfermedades respiratorias. Evitar ejercicio físico intenso "
                "al aire libre en horas de alta contaminación (mañana temprana)."
            )
        else:
            st.error(
                "🚨 **Alerta — riesgo alto**\n\n"
                "Se esperan niveles altos de MP2.5. Posible preemergencia ambiental. "
                "Se recomienda:\n"
                "- Limitar actividad física intensa al aire libre\n"
                "- Mantener ventanas cerradas en horas pico\n"
                "- Evitar encender chimeneas o calefactores a leña\n"
                "- Mantener informados a colegios y centros de salud"
            )

    # Historial reciente
    if df is not None:
        st.markdown("---")
        st.subheader("Contexto: últimos 30 días")
        df_rec = df.tail(30).copy()
        fig_rec = px.bar(
            df_rec, x="fecha", y="mp25",
            color="nivel_calidad_aire",
            color_discrete_map=COLORES_NIVEL,
            title="MP2.5 últimos 30 días",
        )
        fig_rec.add_hline(y=25, line_dash="dash", line_color="#27ae60")
        fig_rec.add_hline(y=50, line_dash="dash", line_color="#e74c3c")
        fig_rec.update_layout(height=300, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_rec, use_container_width=True)


# ===========================================================================
# SECCIÓN 5 — MODELO
# ===========================================================================
elif seccion == "🤖 Modelo":
    st.title("🤖 Modelo predictivo — RandomForestClassifier")

    if metricas is None:
        st.error(
            "No se encontraron métricas. "
            "Ejecuta: `python models/train_model.py`"
        )
        st.stop()

    # KPIs del modelo
    st.subheader("Rendimiento en conjunto de test (20% más reciente)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",      f"{metricas['accuracy']:.1%}")
    col2.metric("F1 weighted",   f"{metricas['f1_weighted']:.1%}")
    col3.metric("Precision",     f"{metricas['precision_weighted']:.1%}")
    col4.metric("Recall",        f"{metricas['recall_weighted']:.1%}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Por clase", "🎯 Feature importance", "🔲 Confusión", "📚 Metodología"]
    )

    with tab1:
        st.subheader("Métricas por clase")
        mpc = metricas.get("metricas_por_clase", {})
        if mpc:
            df_mpc = pd.DataFrame(mpc).T.reset_index()
            df_mpc.columns = ["Clase", "Precision", "Recall", "F1-Score"]
            df_mpc[["Precision", "Recall", "F1-Score"]] = \
                df_mpc[["Precision", "Recall", "F1-Score"]].astype(float).round(3)

            fig_mpc = go.Figure()
            for metrica, color in [("Precision", "#3498db"), ("Recall", "#e74c3c"),
                                    ("F1-Score", "#2ecc71")]:
                fig_mpc.add_trace(go.Bar(
                    name=metrica, x=df_mpc["Clase"], y=df_mpc[metrica],
                    marker_color=color,
                    text=df_mpc[metrica].map(lambda x: f"{x:.3f}"),
                    textposition="outside",
                ))
            fig_mpc.update_layout(
                barmode="group", height=380,
                yaxis=dict(range=[0, 1.1]),
                margin=dict(t=30, b=20),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_mpc, use_container_width=True)
            st.dataframe(df_mpc, use_container_width=True, hide_index=True)

            st.info(
                "💡 La clase **'mala'** tiene el mayor recall: el modelo está "
                "diseñado para no dejar pasar días de preemergencia. "
                "Gracias a `class_weight='balanced'`, se penaliza más el "
                "error en clases menos frecuentes."
            )

    with tab2:
        st.subheader("Importancia de variables (Feature Importance)")
        if fi_df is not None:
            fig_fi = px.bar(
                fi_df.sort_values("importancia"),
                x="importancia", y="feature",
                orientation="h",
                color="importancia",
                color_continuous_scale="Blues",
                title="Contribución de cada variable al modelo",
            )
            fig_fi.update_layout(
                height=420, margin=dict(t=40, b=20),
                coloraxis_showscale=False,
                xaxis_title="Importancia (reducción Gini promedio)",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption(
                "Las variables MP2.5 del día actual, el promedio de 7 días "
                "y el valor del día anterior dominan la predicción. "
                "El mes captura la estacionalidad del invierno santiaguino."
            )
        else:
            st.warning("Archivo feature_importance.csv no encontrado.")

    with tab3:
        st.subheader("Matriz de confusión")
        if cm_df is not None:
            # Limpiar nombres de índices para el heatmap
            clases = [c.replace("real_", "") for c in cm_df.index]
            cols   = [c.replace("pred_", "") for c in cm_df.columns]
            values = cm_df.values

            fig_cm = go.Figure(go.Heatmap(
                z=values,
                x=[f"Pred: {c}" for c in cols],
                y=[f"Real: {c}" for c in clases],
                colorscale="Blues",
                text=values,
                texttemplate="%{text}",
                textfont={"size": 16},
                hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Casos: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                title="Matriz de confusión — conjunto de test",
                height=400,
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(
                "La diagonal principal son las predicciones correctas. "
                "Idealmente los valores fuera de la diagonal deben ser bajos."
            )
        else:
            st.warning("Archivo confusion_matrix.csv no encontrado.")

    with tab4:
        st.subheader("Decisiones metodológicas")

        st.markdown("#### ¿Por qué RandomForestClassifier?")
        razones_rf = [
            ("🌳 Robusto ante outliers",
             "Los episodios de preemergencia generan picos extremos de MP2.5. "
             "RF maneja outliers mejor que modelos lineales porque cada árbol "
             "vota independientemente."),
            ("📏 Sin necesidad de escalar",
             "Las features tienen unidades distintas (µg/m³, °C, km/h). "
             "RF no es sensible a la escala — no se requiere StandardScaler."),
            ("⚖️ class_weight=balanced",
             "Santiago tiene más días con buena calidad que días malos. "
             "Sin balanceo, el modelo aprendería a predecir siempre 'buena'. "
             "El parámetro balanced ajusta los pesos inversamente a la frecuencia."),
            ("📊 Feature importance nativa",
             "RF entrega directamente la importancia de cada variable sin "
             "necesitar herramientas adicionales (SHAP, LIME, etc.)."),
        ]
        for titulo, desc in razones_rf:
            with st.expander(titulo):
                st.write(desc)

        st.markdown("#### ¿Por qué split temporal y no aleatorio?")
        st.error(
            "🚫 **No se debe usar `shuffle=True` con datos de series temporales.**\n\n"
            "Un split aleatorio mezclaría fechas futuras en el entrenamiento — "
            "el modelo vería el futuro durante el aprendizaje (*data leakage*) "
            "y sus métricas en test serían infladas artificialmente."
        )
        st.success(
            "✅ **Split temporal correcto:**\n\n"
            "- **Entrenamiento:** 80% de datos más *antiguos* (primeros ~4 años)\n"
            "- **Test:** 20% de datos más *recientes* (últimos ~10 meses)\n\n"
            "Este esquema replica el escenario real de uso: "
            "el modelo predice el futuro usando solo el pasado."
        )


# ===========================================================================
# SECCIÓN 6 — VISTA TÉCNICA
# ===========================================================================
elif seccion == "⚙️ Vista técnica":
    st.title("⚙️ Vista técnica del pipeline")
    st.caption("Estado del sistema y archivos requeridos")

    # Diagrama del pipeline en texto
    st.subheader("Flujo del pipeline")
    st.code(
        "SINCA CSV → extract_sinca.py → transform_sinca.py\n"
        "                                       ↓\n"
        "Open-Meteo API → extract_meteo.py → transform_meteo.py\n"
        "                                       ↓\n"
        "                              merge_sinca_meteo.py\n"
        "                                       ↓\n"
        "                         dataset_modelo_base.csv\n"
        "                                       ↓\n"
        "                           train_model.py → model.pkl\n"
        "                                       ↓\n"
        "                           predict.py → prediccion_actual.csv\n"
        "                                       ↓\n"
        "                           dashboards/app.py (este dashboard)",
        language="text",
    )

    st.markdown("---")

    # Estado de archivos
    st.subheader("Estado de archivos del sistema")

    archivos_requeridos = [
        ("dataset_modelo_base.csv", PATHS["dataset"],
         "python etl/etl_meteo_main.py"),
        ("prediccion_actual.csv",   PATHS["prediccion"],
         "python models/predict.py"),
        ("model_metrics.json",      PATHS["metrics"],
         "python models/train_model.py"),
        ("feature_importance.csv",  PATHS["fi"],
         "python models/train_model.py"),
        ("confusion_matrix.csv",    PATHS["cm"],
         "python models/train_model.py"),
    ]

    for nombre, ruta, cmd in archivos_requeridos:
        if ruta.exists():
            size_kb = ruta.stat().st_size / 1024
            st.success(f"✅ `{nombre}` — {size_kb:.1f} KB — `{ruta}`")
        else:
            st.error(f"❌ `{nombre}` no encontrado → Ejecuta: `{cmd}`")

    st.markdown("---")

    # Detalles del dataset
    if df is not None:
        st.subheader("Detalles del dataset base")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas totales", f"{len(df):,}")
        col2.metric("Columnas", f"{df.shape[1]}")
        col3.metric(
            "Rango fechas",
            f"{df['fecha'].min().date()} → {df['fecha'].max().date()}"
        )

        with st.expander("Ver columnas y tipos de datos"):
            df_cols = pd.DataFrame({
                "Columna": df.columns,
                "Tipo":    df.dtypes.astype(str).values,
                "Nulos":   df.isnull().sum().values,
                "% Nulos": (df.isnull().mean() * 100).round(1).values,
            })
            st.dataframe(df_cols, use_container_width=True, hide_index=True)

        # Distribución del target
        st.subheader("Distribución de la variable objetivo")
        dist = df["nivel_calidad_aire_dia_siguiente"].value_counts().reset_index()
        dist.columns = ["Clase", "Días"]
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig_dist = px.bar(
                dist, x="Clase", y="Días",
                color="Clase",
                color_discrete_map=COLORES_NIVEL,
                text="Días",
            )
            fig_dist.update_traces(textposition="outside")
            fig_dist.update_layout(
                height=280, showlegend=False, margin=dict(t=20, b=10)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            st.dataframe(dist, use_container_width=True, hide_index=True)
            total = dist["Días"].sum()
            for _, row in dist.iterrows():
                pct = row["Días"] / total * 100
                st.markdown(f"**{row['Clase']}:** {row['Días']:,} días ({pct:.1f}%)")

    else:
        st.warning("Dataset no disponible. Ejecuta el pipeline ETL primero.")

    # Scripts para ejecutar el pipeline completo
    st.markdown("---")
    st.subheader("Comandos para ejecutar el pipeline completo")
    st.code(
        "# 1. ETL SINCA\n"
        "python etl/etl_sinca_main.py\n\n"
        "# 2. ETL Open-Meteo + Merge\n"
        "python etl/etl_meteo_main.py\n\n"
        "# 3. Entrenar modelo\n"
        "python models/train_model.py\n\n"
        "# 4. Generar predicción\n"
        "python models/predict.py\n\n"
        "# 5. Lanzar dashboard\n"
        "streamlit run dashboards/app.py\n\n"
        "# 6. Ejecutar todos los tests\n"
        "pytest tests/ -v",
        language="bash",
    )