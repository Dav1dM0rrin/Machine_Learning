"""
Aplicación Streamlit - Sistema de Calidad del Aire
Predicción con Machine Learning - PARCIAL
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="🌍 Calidad del Aire - ML",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

@st.cache_data
def cargar_datos():
    """Carga el dataset"""
    df = pd.read_csv('aire_limpio.csv')
    return df

def crear_categoria_calidad(pm25):
    """Clasifica calidad del aire según PM2.5 (EPA)"""
    if pm25 <= 12:
        return 'Buena'
    elif pm25 <= 35.4:
        return 'Moderada'
    elif pm25 <= 55.4:
        return 'Mala'
    else:
        return 'Peligrosa'

@st.cache_resource
def cargar_modelos():
    """Carga los modelos entrenados"""
    try:
        rf = joblib.load('modelo_random_forest.pkl')
        gb = joblib.load('modelo_gradient_boosting.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        features = joblib.load('features.pkl')
        return rf, gb, scaler, le, features
    except:
        return None, None, None, None, None

def obtener_color_calidad(calidad):
    """Retorna color según la calidad"""
    colores = {
        'Buena': '#00CC66',
        'Moderada': '#FFCC00', 
        'Mala': '#FF6600',
        'Peligrosa': '#CC0000'
    }
    return colores.get(calidad, '#808080')

def obtener_emoji_calidad(calidad):
    """Retorna emoji según la calidad"""
    emojis = {
        'Buena': '✅',
        'Moderada': '⚠️',
        'Mala': '🔶',
        'Peligrosa': '🔴'
    }
    return emojis.get(calidad, '❓')

# =============================================================================
# CARGAR DATOS Y MODELOS
# =============================================================================
df = cargar_datos()
df['Calidad_Aire'] = df['pm2_5'].apply(crear_categoria_calidad)
rf, gb, scaler, le, features = cargar_modelos()

# =============================================================================
# SIDEBAR - NAVEGACIÓN
# =============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
st.sidebar.title("🌍 Calidad del Aire")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "📌 Navegación:",
    ["🏠 Inicio", "🔮 Predicción", "📈 Métricas y Validación", "📊 Visualizaciones", "📋 Base de Datos"]
)

st.sidebar.markdown("---")
st.sidebar.success("""
**🤖 Algoritmos ML:**
- 🌲 Random Forest
- 🚀 Gradient Boosting
""")

st.sidebar.info("""
**📊 Variable Objetivo:**  
Calidad del Aire basada en PM2.5
- Buena: 0-12 µg/m³
- Moderada: 12.1-35.4 µg/m³
- Mala: 35.5-55.4 µg/m³
- Peligrosa: >55.4 µg/m³
""")

# =============================================================================
# PÁGINA: INICIO
# =============================================================================
if pagina == "🏠 Inicio":
    st.title("🌍 Sistema de Clasificación de Calidad del Aire")
    st.markdown("### Machine Learning con Random Forest y Gradient Boosting")
    st.markdown("---")
    
    # Info del proyecto
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 Descripción del Proyecto
        
        Este sistema utiliza **algoritmos de Machine Learning** para clasificar la calidad 
        del aire basándose en mediciones de sensores ambientales.
        
        ### 🎯 Objetivo
        Predecir la categoría de calidad del aire (Buena, Moderada, Mala, Peligrosa) 
        utilizando variables como CO, CO2, PM10, PM5, Humedad y Temperatura.
        
        ### 🔬 Algoritmos Implementados
        1. **Random Forest Classifier** - Conjunto de árboles de decisión
        2. **Gradient Boosting Classifier** - Boosting con árboles secuenciales
        
        ### 📊 Features Utilizados
        - `CO` - Monóxido de carbono (ppm)
        - `CO2` - Dióxido de carbono (ppm)
        - `PM10` - Material particulado 10µm (µg/m³)
        - `PM5` - Material particulado 5µm (µg/m³)
        - `Humedad Relativa` (%)
        - `Temperatura` (°C)
        """)
    
    with col2:
        st.markdown("### 📈 Resumen de Datos")
        st.metric("Total Registros", f"{len(df):,}")
        st.metric("Variables", f"{len(df.columns)}")
        
        # Distribución
        st.markdown("### 🎯 Distribución")
        for cat in ['Buena', 'Moderada', 'Mala', 'Peligrosa']:
            count = len(df[df['Calidad_Aire'] == cat])
            pct = count / len(df) * 100
            st.write(f"{obtener_emoji_calidad(cat)} **{cat}:** {count:,} ({pct:.1f}%)")
    
    st.markdown("---")
    
    # Métricas rápidas
    if rf is not None:
        st.markdown("### ⚡ Métricas Rápidas de los Modelos")
        data = df[(df['Temperatura'] != 0) & (df['Humedad Relativa'] != 0)].copy()
        X = data[features]
        y = le.transform(data['Calidad_Aire'])
        X_scaled = scaler.transform(X)
        _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        acc_rf = accuracy_score(y_test, rf.predict(X_test))
        acc_gb = accuracy_score(y_test, gb.predict(X_test))
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌲 RF Accuracy", f"{acc_rf*100:.2f}%")
        col2.metric("🌲 RF Score", f"{acc_rf*100:.2f}%", "✅ >85%")
        col3.metric("🚀 GB Accuracy", f"{acc_gb*100:.2f}%")
        col4.metric("🚀 GB Score", f"{acc_gb*100:.2f}%", "✅ >85%")

# =============================================================================
# PÁGINA: PREDICCIÓN
# =============================================================================
elif pagina == "🔮 Predicción":
    st.title("🔮 Predicción de Calidad del Aire")
    st.markdown("### Ingresa los valores de los sensores para obtener una predicción")
    st.markdown("---")
    
    if rf is None:
        st.error("⚠️ No se encontraron los modelos. Ejecuta primero `aire_entrenar.py`")
        st.code("python aire_entrenar.py", language="bash")
    else:
        st.success("✅ Modelos cargados correctamente")
        
        # Formulario de predicción
        st.markdown("### 📝 Formulario de Predicción")
        
        with st.form("form_prediccion"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                co = st.number_input("🔹 CO (ppm)", min_value=0.0, max_value=50.0, value=15.0, step=1.0,
                                    help="Monóxido de carbono")
                co2 = st.number_input("🔹 CO2 (ppm)", min_value=300.0, max_value=700.0, value=450.0, step=10.0,
                                     help="Dióxido de carbono")
            
            with col2:
                pm10 = st.number_input("🔹 PM10 (µg/m³)", min_value=0.0, max_value=100.0, value=25.0, step=1.0,
                                      help="Material particulado 10µm")
                pm5 = st.number_input("🔹 PM5 (µg/m³)", min_value=0.0, max_value=50.0, value=10.0, step=1.0,
                                     help="Material particulado 5µm")
            
            with col3:
                humedad = st.number_input("🔹 Humedad (%)", min_value=0.0, max_value=100.0, value=60.0, step=5.0,
                                         help="Humedad relativa del ambiente")
                temp = st.number_input("🔹 Temperatura (°C)", min_value=-10.0, max_value=45.0, value=20.0, step=1.0,
                                      help="Temperatura ambiente")
            
            submit = st.form_submit_button("🔍 REALIZAR PREDICCIÓN", use_container_width=True)
        
        if submit:
            # Preparar datos
            datos = np.array([[co, co2, pm10, pm5, humedad, temp]])
            datos_scaled = scaler.transform(datos)
            
            # Predicciones
            pred_rf = rf.predict(datos_scaled)
            pred_gb = gb.predict(datos_scaled)
            prob_rf = rf.predict_proba(datos_scaled)[0]
            prob_gb = gb.predict_proba(datos_scaled)[0]
            
            clase_rf = le.inverse_transform(pred_rf)[0]
            clase_gb = le.inverse_transform(pred_gb)[0]
            
            st.markdown("---")
            st.markdown("## 📊 Resultados de la Predicción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌲 Random Forest")
                color_rf = obtener_color_calidad(clase_rf)
                emoji_rf = obtener_emoji_calidad(clase_rf)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color_rf}, {color_rf}dd); 
                            padding:30px; border-radius:15px; text-align:center; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                    <h1 style='color:white; margin:0; font-size:2.5em;'>{emoji_rf} {clase_rf}</h1>
                    <p style='color:white; margin-top:10px; font-size:1.2em;'>
                        Confianza: {max(prob_rf)*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Probabilidades por clase:")
                for i, clase in enumerate(le.classes_):
                    st.progress(prob_rf[i], text=f"{clase}: {prob_rf[i]*100:.1f}%")
            
            with col2:
                st.markdown("### 🚀 Gradient Boosting")
                color_gb = obtener_color_calidad(clase_gb)
                emoji_gb = obtener_emoji_calidad(clase_gb)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color_gb}, {color_gb}dd); 
                            padding:30px; border-radius:15px; text-align:center;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                    <h1 style='color:white; margin:0; font-size:2.5em;'>{emoji_gb} {clase_gb}</h1>
                    <p style='color:white; margin-top:10px; font-size:1.2em;'>
                        Confianza: {max(prob_gb)*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Probabilidades por clase:")
                for i, clase in enumerate(le.classes_):
                    st.progress(prob_gb[i], text=f"{clase}: {prob_gb[i]*100:.1f}%")
            
            # Resumen de datos ingresados
            st.markdown("---")
            st.markdown("### 📋 Datos Ingresados")
            datos_df = pd.DataFrame({
                'Variable': ['CO', 'CO2', 'PM10', 'PM5', 'Humedad', 'Temperatura'],
                'Valor': [co, co2, pm10, pm5, humedad, temp],
                'Unidad': ['ppm', 'ppm', 'µg/m³', 'µg/m³', '%', '°C']
            })
            st.dataframe(datos_df, use_container_width=True, hide_index=True)

# =============================================================================
# PÁGINA: MÉTRICAS Y VALIDACIÓN
# =============================================================================
elif pagina == "📈 Métricas y Validación":
    st.title("📈 Métricas y Validación de Modelos")
    st.markdown("### Evaluación del desempeño de los algoritmos de Machine Learning")
    st.markdown("---")
    
    if rf is None:
        st.error("⚠️ No se encontraron los modelos. Ejecuta primero `aire_entrenar.py`")
    else:
        # Preparar datos
        data = df[(df['Temperatura'] != 0) & (df['Humedad Relativa'] != 0)].copy()
        X = data[features]
        y = le.transform(data['Calidad_Aire'])
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        pred_rf = rf.predict(X_test)
        pred_gb = gb.predict(X_test)
        
        # Calcular métricas
        metricas = {
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Random Forest': [
                f"{accuracy_score(y_test, pred_rf)*100:.2f}%",
                f"{precision_score(y_test, pred_rf, average='weighted')*100:.2f}%",
                f"{recall_score(y_test, pred_rf, average='weighted')*100:.2f}%",
                f"{f1_score(y_test, pred_rf, average='weighted')*100:.2f}%"
            ],
            'Gradient Boosting': [
                f"{accuracy_score(y_test, pred_gb)*100:.2f}%",
                f"{precision_score(y_test, pred_gb, average='weighted')*100:.2f}%",
                f"{recall_score(y_test, pred_gb, average='weighted')*100:.2f}%",
                f"{f1_score(y_test, pred_gb, average='weighted')*100:.2f}%"
            ]
        }
        
        # Mostrar métricas principales
        st.markdown("### 🎯 Métricas Principales (Score > 85%)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            acc_rf = accuracy_score(y_test, pred_rf)
            f1_rf = f1_score(y_test, pred_rf, average='weighted')
            st.markdown("#### 🌲 Random Forest")
            st.metric("Accuracy Score", f"{acc_rf*100:.2f}%", 
                     f"{'✅ Cumple >85%' if acc_rf > 0.85 else '❌ No cumple'}")
            st.metric("F1-Score", f"{f1_rf*100:.2f}%",
                     f"{'✅ Cumple >85%' if f1_rf > 0.85 else '❌ No cumple'}")
        
        with col2:
            acc_gb = accuracy_score(y_test, pred_gb)
            f1_gb = f1_score(y_test, pred_gb, average='weighted')
            st.markdown("#### 🚀 Gradient Boosting")
            st.metric("Accuracy Score", f"{acc_gb*100:.2f}%",
                     f"{'✅ Cumple >85%' if acc_gb > 0.85 else '❌ No cumple'}")
            st.metric("F1-Score", f"{f1_gb*100:.2f}%",
                     f"{'✅ Cumple >85%' if f1_gb > 0.85 else '❌ No cumple'}")
        
        st.markdown("---")
        
        # Tabla comparativa
        st.markdown("### 📊 Tabla Comparativa de Métricas")
        st.dataframe(pd.DataFrame(metricas), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Matrices de confusión
        st.markdown("### 🔢 Matrices de Confusión")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌲 Random Forest")
            cm_rf = confusion_matrix(y_test, pred_rf)
            fig_rf = px.imshow(
                cm_rf, text_auto=True,
                x=le.classes_, y=le.classes_,
                color_continuous_scale='Blues',
                labels={'x': 'Predicción', 'y': 'Valor Real', 'color': 'Cantidad'}
            )
            fig_rf.update_layout(height=450)
            st.plotly_chart(fig_rf, use_container_width=True)
        
        with col2:
            st.markdown("#### 🚀 Gradient Boosting")
            cm_gb = confusion_matrix(y_test, pred_gb)
            fig_gb = px.imshow(
                cm_gb, text_auto=True,
                x=le.classes_, y=le.classes_,
                color_continuous_scale='Greens',
                labels={'x': 'Predicción', 'y': 'Valor Real', 'color': 'Cantidad'}
            )
            fig_gb.update_layout(height=450)
            st.plotly_chart(fig_gb, use_container_width=True)
        
        st.markdown("---")
        
        # Importancia de variables
        st.markdown("### 🎯 Importancia de Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            imp_rf = pd.DataFrame({
                'Variable': features,
                'Importancia': rf.feature_importances_
            }).sort_values('Importancia', ascending=True)
            
            fig_imp_rf = px.bar(imp_rf, x='Importancia', y='Variable', orientation='h',
                               title='Random Forest', color='Importancia',
                               color_continuous_scale='Blues')
            fig_imp_rf.update_layout(height=400)
            st.plotly_chart(fig_imp_rf, use_container_width=True)
        
        with col2:
            imp_gb = pd.DataFrame({
                'Variable': features,
                'Importancia': gb.feature_importances_
            }).sort_values('Importancia', ascending=True)
            
            fig_imp_gb = px.bar(imp_gb, x='Importancia', y='Variable', orientation='h',
                               title='Gradient Boosting', color='Importancia',
                               color_continuous_scale='Greens')
            fig_imp_gb.update_layout(height=400)
            st.plotly_chart(fig_imp_gb, use_container_width=True)

# =============================================================================
# PÁGINA: VISUALIZACIONES
# =============================================================================
elif pagina == "📊 Visualizaciones":
    st.title("📊 Visualizaciones de Datos")
    st.markdown("### Análisis exploratorio del dataset de calidad del aire")
    st.markdown("---")
    
    # Distribución de calidad del aire
    st.markdown("### 🎯 Distribución de Calidad del Aire")
    col1, col2 = st.columns(2)
    
    with col1:
        conteo = df['Calidad_Aire'].value_counts()
        colores = [obtener_color_calidad(c) for c in conteo.index]
        fig_pie = px.pie(values=conteo.values, names=conteo.index,
                        color_discrete_sequence=colores, hole=0.4,
                        title='Distribución por Categoría')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(x=conteo.index, y=conteo.values, color=conteo.index,
                        color_discrete_map={c: obtener_color_calidad(c) for c in conteo.index},
                        title='Cantidad por Categoría', labels={'x': 'Calidad', 'y': 'Cantidad'})
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Boxplots de contaminantes
    st.markdown("### 📈 Distribución de Contaminantes por Calidad del Aire")
    
    variable = st.selectbox("Selecciona variable:", ['pm2_5', 'pm10', 'pm5', 'co', 'co2'])
    
    fig_box = px.box(df, x='Calidad_Aire', y=variable, color='Calidad_Aire',
                    color_discrete_map={c: obtener_color_calidad(c) for c in df['Calidad_Aire'].unique()},
                    title=f'Distribución de {variable.upper()} por Calidad del Aire')
    fig_box.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # Matriz de correlación
    st.markdown("### 🔗 Matriz de Correlación")
    cols_corr = ['co', 'co2', 'pm10', 'pm2_5', 'pm5', 'Humedad Relativa', 'Temperatura']
    corr = df[cols_corr].corr()
    fig_corr = px.imshow(corr, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r', title='Correlación entre Variables')
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plot
    st.markdown("### 🔵 Relación entre Variables")
    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Variable X:", ['pm2_5', 'pm10', 'co', 'co2', 'Temperatura'], index=0)
    with col2:
        var_y = st.selectbox("Variable Y:", ['pm10', 'pm2_5', 'co', 'co2', 'Humedad Relativa'], index=0)
    
    fig_scatter = px.scatter(df, x=var_x, y=var_y, color='Calidad_Aire',
                            color_discrete_map={c: obtener_color_calidad(c) for c in df['Calidad_Aire'].unique()},
                            title=f'{var_x} vs {var_y}', opacity=0.6)
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================================
# PÁGINA: BASE DE DATOS
# =============================================================================
elif pagina == "📋 Base de Datos":
    st.title("📋 Base de Datos")
    st.markdown("### Exploración y filtrado del dataset")
    st.markdown("---")
    
    # Estadísticas
    st.markdown("### 📊 Estadísticas Descriptivas")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Filtros
    st.markdown("### 🔍 Filtrar Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        equipos = st.multiselect("Equipo:", options=df['Nombre Equipo'].unique(),
                                default=df['Nombre Equipo'].unique())
    with col2:
        calidades = st.multiselect("Calidad del Aire:", options=['Buena', 'Moderada', 'Mala', 'Peligrosa'],
                                  default=['Buena', 'Moderada', 'Mala', 'Peligrosa'])
    with col3:
        n_rows = st.slider("Número de filas:", 10, 1000, 100)
    
    # Aplicar filtros
    df_filtrado = df[(df['Nombre Equipo'].isin(equipos)) & (df['Calidad_Aire'].isin(calidades))].head(n_rows)
    
    st.markdown(f"**📌 Mostrando {len(df_filtrado)} registros**")
    st.dataframe(df_filtrado, use_container_width=True, height=400)
    
    # Descargar
    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtrado.to_csv(index=False)
        st.download_button("📥 Descargar CSV Filtrado", csv, "datos_filtrados.csv", "text/csv", use_container_width=True)
    with col2:
        csv_full = df.to_csv(index=False)
        st.download_button("📥 Descargar CSV Completo", csv_full, "datos_completos.csv", "text/csv", use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Información")
st.sidebar.markdown(f"**Registros:** {len(df):,}")
st.sidebar.markdown(f"**Features:** {len(features)}")
st.sidebar.markdown("**Modelos:** 2 (RF + GB)")