import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import altair as alt

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Predicción de Energía Activa",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: #0d0d0d;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .stTab {
        font-size: 1.2rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

DATA_PATH = "energia_limpio.csv"
MODEL_RF_PATH = "models/modelo_rf_simple.pkl"
MODEL_GB_PATH = "models/modelo_gb_simple.pkl"
FEATURES_PATH = "models/features_simple.pkl"

# =========================================================
# CARGA DE MODELOS Y DATOS
# =========================================================
@st.cache_resource
def load_models():
    try:
        rf = joblib.load(MODEL_RF_PATH)
        gb = joblib.load(MODEL_GB_PATH)
        features = joblib.load(FEATURES_PATH)
        return rf, gb, features, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

rf_model, gb_model, FEATURES, load_error = load_models()

df = None
try:
    df = load_data()
    df["ratio_reactiva_potencia"] = df["ENERGÍA REACTIVA"] / (df["POTENCIA MÁXIMA"] + 1)
except:
    df = None

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/electricity.png", width=80)
    st.title("⚙️ Configuración")
    st.markdown("---")
    
    if load_error:
        st.error(f"❌ Error cargando modelos: {load_error}")
    else:
        st.success("✅ Modelos cargados correctamente")
    
    st.markdown("---")
    st.subheader("📊 Variables del Modelo")
    
    with st.expander("Ver detalles", expanded=True):
        st.markdown("""
        - 🔋 **Energía Reactiva** (kVAR)
        - ⚡ **Potencia Máxima** (kW)
        - 📐 **Ratio Reactiva/Potencia** (calculado)
        """)
    
    st.metric("Total Variables", "3", delta="Optimizado")
    
    st.markdown("---")
    model_choice = st.selectbox(
        "🤖 Modelo a usar",
        ["Random Forest", "Gradient Boosting", "Promedio"],
        help="Selecciona el modelo de predicción"
    )
    
    st.markdown("---")
    if df is not None:
        st.subheader("📈 Estadísticas Rápidas")
        st.metric("Registros", f"{len(df):,}")
        st.metric("Energía Promedio", f"{df['ENERGÍA ACTIVA'].mean():,.0f} kWh")

# =========================================================
# HEADER PRINCIPAL
# =========================================================
st.markdown('<h1 class="main-header">⚡ Sistema de Predicción de Energía Activa</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Análisis predictivo de consumo energético basado en Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predicción", "📊 Datos", "📈 Métricas", "ℹ️ Información"])

# =========================================================
# TAB 1: PREDICCIÓN
# =========================================================
with tab1:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.header("🔮 Realizar Predicción")
        
        st.markdown("""
        <div class="info-box">
        <strong>ℹ️ Instrucciones:</strong><br>
        Ingresa los valores de Energía Reactiva y Potencia Máxima para obtener una predicción 
        del consumo de Energía Activa usando modelos de Machine Learning entrenados.
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("form_pred", clear_on_submit=False):
            st.subheader("📝 Parámetros de Entrada")
            
            c1, c2 = st.columns(2)
            with c1:
                energia_reactiva = st.number_input(
                    "🔋 ENERGÍA REACTIVA (kVAR)",
                    min_value=0.0,
                    value=5000.0,
                    step=100.0,
                    help="Ingresa la energía reactiva en kVAR"
                )
            with c2:
                potencia_max = st.number_input(
                    "⚡ POTENCIA MÁXIMA (kW)",
                    min_value=0.0,
                    value=8000.0,
                    step=100.0,
                    help="Ingresa la potencia máxima en kW"
                )
            
            # Mostrar el ratio calculado
            if potencia_max > 0:
                ratio_preview = energia_reactiva / (potencia_max + 1)
                st.info(f"📐 Ratio calculado: {ratio_preview:.4f}")
            
            submit = st.form_submit_button("🔮 Predecir Energía Activa", use_container_width=True)
        
        if submit:
            if rf_model is None:
                st.error("❌ Modelos no cargados. Verifica los archivos.")
            else:
                with st.spinner("⏳ Realizando predicción..."):
                    ratio = energia_reactiva / (potencia_max + 1)
                    entrada = pd.DataFrame([[energia_reactiva, potencia_max, ratio]], columns=FEATURES)
                    
                    pred_rf = rf_model.predict(entrada)[0]
                    pred_gb = gb_model.predict(entrada)[0]
                    pred_prom = (pred_rf + pred_gb) / 2
                    
                    st.markdown("---")
                    st.subheader("📊 Resultados de Predicción")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("### 🌲 Random Forest")
                        st.markdown(f"<h2 style='color: white;'>{pred_rf:,.2f} kWh</h2>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("### 📈 Gradient Boosting")
                        st.markdown(f"<h2 style='color: white;'>{pred_gb:,.2f} kWh</h2>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Promedio")
                        st.markdown(f"<h2 style='color: white;'>{pred_prom:,.2f} kWh</h2>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("✅ Predicción completada correctamente")
                    
                    # Gráfico de comparación
                    st.subheader("📊 Comparación de Modelos")
                    comparacion_df = pd.DataFrame({
                        'Modelo': ['Random Forest', 'Gradient Boosting', 'Promedio'],
                        'Predicción (kWh)': [pred_rf, pred_gb, pred_prom]
                    })
                    
                    chart = alt.Chart(comparacion_df).mark_bar().encode(
                        x=alt.X('Modelo:N', title='Modelo'),
                        y=alt.Y('Predicción (kWh):Q', title='Energía Activa (kWh)'),
                        color=alt.Color('Modelo:N', scale=alt.Scale(scheme='viridis')),
                        tooltip=['Modelo', 'Predicción (kWh)']
                    ).properties(
                        height=300
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
    
    with col_b:
        st.header("💡 Consejos")
        st.markdown("""
        <div class="info-box">
        <strong>🎯 Valores Típicos:</strong><br><br>
        <strong>Energía Reactiva:</strong><br>
        • Bajo: 1,000 - 3,000 kVAR<br>
        • Medio: 3,000 - 7,000 kVAR<br>
        • Alto: 7,000+ kVAR<br><br>
        
        <strong>Potencia Máxima:</strong><br>
        • Bajo: 2,000 - 5,000 kW<br>
        • Medio: 5,000 - 10,000 kW<br>
        • Alto: 10,000+ kW
        </div>
        """, unsafe_allow_html=True)
        
        if df is not None:
            st.markdown("---")
            st.subheader("📊 Rangos en Dataset")
            st.metric("Max Energía Reactiva", f"{df['ENERGÍA REACTIVA'].max():,.0f} kVAR")
            st.metric("Max Potencia", f"{df['POTENCIA MÁXIMA'].max():,.0f} kW")
            st.metric("Max Energía Activa", f"{df['ENERGÍA ACTIVA'].max():,.0f} kWh")

# =========================================================
# TAB 2: DATOS
# =========================================================
with tab2:
    st.header("📊 Exploración del Dataset")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Registros", f"{len(df):,}")
        with col2:
            st.metric("📊 Variables", f"{len(df.columns)}")
        with col3:
            st.metric("💾 Tamaño", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        with col4:
            st.metric("✅ Completos", f"{100 - (df.isnull().sum().sum() / df.size * 100):.1f}%")
        
        st.markdown("---")
        
        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            num_rows = st.slider("Número de filas a mostrar", 10, 500, 100)
        with col_f2:
            sort_column = st.selectbox("Ordenar por", df.columns.tolist())
        
        df_display = df.head(num_rows).sort_values(by=sort_column, ascending=False)
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "📥 Descargar CSV Completo",
                df.to_csv(index=False).encode("utf-8"),
                "energia_limpio.csv",
                "text/csv",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                "📥 Descargar Vista Actual",
                df_display.to_csv(index=False).encode("utf-8"),
                "energia_filtrado.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        st.subheader("📈 Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
        
    else:
        st.error("❌ No se pudo cargar el CSV.")

# =========================================================
# TAB 3: MÉTRICAS
# =========================================================
with tab3:
    st.header("📈 Análisis de Rendimiento de Modelos")
    
    if df is not None and rf_model is not None:
        X = df[FEATURES]
        y = df["ENERGÍA ACTIVA"]
        
        pred_rf_all = rf_model.predict(X)
        pred_gb_all = gb_model.predict(X)
        
        r2_rf = r2_score(y, pred_rf_all)
        r2_gb = r2_score(y, pred_gb_all)
        
        rmse_rf = mean_squared_error(y, pred_rf_all) ** 0.5
        rmse_gb = mean_squared_error(y, pred_gb_all) ** 0.5
        
        mae_rf = mean_absolute_error(y, pred_rf_all)
        mae_gb = mean_absolute_error(y, pred_gb_all)
        
        # Métricas principales
        st.subheader("🎯 Métricas de Precisión")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌲 RF - R²", f"{r2_rf:.4f}", delta=f"{(r2_rf - r2_gb):.4f}")
        with col2:
            st.metric("🌲 RF - RMSE", f"{rmse_rf:,.0f}")
        with col3:
            st.metric("📈 GB - R²", f"{r2_gb:.4f}", delta=f"{(r2_gb - r2_rf):.4f}")
        with col4:
            st.metric("📈 GB - RMSE", f"{rmse_gb:,.0f}")
        
        st.markdown("---")
        
        # Tabla comparativa
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.subheader("📊 Comparación Detallada")
            metricas_df = pd.DataFrame({
                'Métrica': ['R² Score', 'RMSE', 'MAE'],
                'Random Forest': [f"{r2_rf:.4f}", f"{rmse_rf:,.2f}", f"{mae_rf:,.2f}"],
                'Gradient Boosting': [f"{r2_gb:.4f}", f"{rmse_gb:,.2f}", f"{mae_gb:,.2f}"]
            })
            st.dataframe(metricas_df, use_container_width=True, hide_index=True)
        
        with col_t2:
            st.subheader("🏆 Mejor Modelo")
            if r2_rf > r2_gb:
                st.success("✅ Random Forest tiene mejor R²")
            else:
                st.success("✅ Gradient Boosting tiene mejor R²")
            
            if rmse_rf < rmse_gb:
                st.info("✅ Random Forest tiene menor RMSE")
            else:
                st.info("✅ Gradient Boosting tiene menor RMSE")
        
        st.markdown("---")
        
        # Visualizaciones
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.subheader("📦 Distribución de Energía Activa")
            box = alt.Chart(df).mark_boxplot(color='#667eea').encode(
                y=alt.Y("ENERGÍA ACTIVA:Q", title="Energía Activa (kWh)")
            ).properties(
                height=400,
                title="Distribución de Energía Activa"
            )
            st.altair_chart(box, use_container_width=True)
        
        with col_v2:
            st.subheader("📊 Histograma de Energía Activa")
            hist = alt.Chart(df).mark_bar(color='#764ba2').encode(
                alt.X("ENERGÍA ACTIVA:Q", bin=alt.Bin(maxbins=30), title="Energía Activa (kWh)"),
                y=alt.Y('count()', title='Frecuencia'),
                tooltip=['count()']
            ).properties(
                height=400,
                title="Distribución de Frecuencias"
            )
            st.altair_chart(hist, use_container_width=True)
        
        # Gráfico de dispersión
        st.subheader("🎯 Predicciones vs Valores Reales")
        
        scatter_df = pd.DataFrame({
            'Real': y[:500],  # Primeros 500 para mejor visualización
            'RF Predicción': pred_rf_all[:500],
            'GB Predicción': pred_gb_all[:500]
        })
        
        tab_rf, tab_gb = st.tabs(["Random Forest", "Gradient Boosting"])
        
        with tab_rf:
            scatter_rf = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X('Real:Q', title='Valor Real (kWh)'),
                y=alt.Y('RF Predicción:Q', title='Predicción (kWh)'),
                color=alt.value('#667eea'),
                tooltip=['Real', 'RF Predicción']
            ).properties(
                height=400
            )
            
            line = alt.Chart(pd.DataFrame({'x': [scatter_df['Real'].min(), scatter_df['Real'].max()]})).mark_line(color='red', strokeDash=[5, 5]).encode(
                x='x:Q',
                y='x:Q'
            )
            
            st.altair_chart(scatter_rf + line, use_container_width=True)
        
        with tab_gb:
            scatter_gb = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X('Real:Q', title='Valor Real (kWh)'),
                y=alt.Y('GB Predicción:Q', title='Predicción (kWh)'),
                color=alt.value('#764ba2'),
                tooltip=['Real', 'GB Predicción']
            ).properties(
                height=400
            )
            
            st.altair_chart(scatter_gb + line, use_container_width=True)
        
    else:
        st.error("❌ No se pueden calcular métricas sin datos o modelos.")

# =========================================================
# TAB 4: INFORMACIÓN
# =========================================================
with tab4:
    st.header("ℹ️ Información del Sistema")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.subheader("🤖 Sobre los Modelos")
        st.markdown("""
        Este sistema utiliza dos algoritmos de Machine Learning:
        
        **🌲 Random Forest:**
        - Ensemble de árboles de decisión
        - Robusto ante outliers
        - Buena capacidad de generalización
        
        **📈 Gradient Boosting:**
        - Boosting secuencial
        - Alta precisión
        - Optimización iterativa
        
        **🎯 Promedio:**
        - Combina ambos modelos
        - Reduce el sesgo individual
        - Mayor estabilidad
        """)
    
    with col_i2:
        st.subheader("📊 Variables Utilizadas")
        st.markdown("""
        **Entrada:**
        1. **Energía Reactiva (kVAR):** Energía que no realiza trabajo útil
        2. **Potencia Máxima (kW):** Máxima demanda de potencia
        3. **Ratio Reactiva/Potencia:** Variable derivada calculada automáticamente
        
        **Salida:**
        - **Energía Activa (kWh):** Energía consumida efectivamente
        
        **💡 Nota:** El ratio ayuda al modelo a capturar la relación entre 
        energía reactiva y potencia, mejorando la precisión de las predicciones.
        """)
    
    st.markdown("---")
    
    st.subheader("📚 Métricas de Evaluación")
    st.markdown("""
    - **R² Score:** Coeficiente de determinación (0-1). Más cercano a 1 = mejor ajuste
    - **RMSE:** Error cuadrático medio. Menor valor = mejor precisión
    - **MAE:** Error absoluto medio. Menor valor = mejor precisión
    """)
    
    st.markdown("---")
    
    st.info("💡 **Tip:** Para mejores resultados, utiliza el modelo 'Promedio' que combina las fortalezas de ambos algoritmos.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>⚡ Sistema de Predicción de Energía Activa | Powered by Machine Learning</p>", unsafe_allow_html=True)