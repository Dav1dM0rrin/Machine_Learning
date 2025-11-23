import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="Predicción de Pasajeros - Transporte ",
    page_icon="🚌",
    layout="wide",
)

# -----------------------------
# Utilidades: carga y helpers
# -----------------------------
@st.cache_resource
def load_models():
    """Carga los modelos y transformadores guardados con joblib.
    Asegúrate de tener en la misma carpeta los archivos:
     - random_forest_model.pkl
     - gradient_boosting_model.pkl
     - scaler.pkl
     - le_ciudad.pkl
     - le_sistema.pkl
    """
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        gb_model = joblib.load('gradient_boosting_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_ciudad = joblib.load('le_ciudad.pkl')
        le_sistema = joblib.load('le_sistema.pkl')
        return rf_model, gb_model, scaler, le_ciudad, le_sistema
    except Exception as e:
        # Re-lanzamos la excepción para que la app la muestre y la UX sea clara
        raise RuntimeError(f"Error cargando modelos: {e}")

@st.cache_data
def load_data(path: str = 'transporte_limpio.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalizaciones / arreglos mínimos
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    # Estandarizar coma decimal en "Variación Transmilenio"
    if 'Variación Transmilenio' in df.columns:
        df['Variación Transmilenio'] = df['Variación Transmilenio'].astype(str).str.replace(',', '.').replace('nan', np.nan)
        df['Variación Transmilenio'] = pd.to_numeric(df['Variación Transmilenio'], errors='coerce')
        df['Variación Transmilenio'].fillna(df['Variación Transmilenio'].median(), inplace=True)
    return df


def compute_model_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# -----------------------------
# Cargar recursos
# -----------------------------
models_loaded = True
try:
    rf_model, gb_model, scaler, le_ciudad, le_sistema = load_models()
except Exception as e:
    models_loaded = False
    load_error = str(e)

try:
    df = load_data('transporte_limpio.csv')
except Exception as e:
    st.error(f"No se pudo cargar el archivo 'transporte_limpio.csv': {e}")
    st.stop()


# Mapeo Ciudad → Sistemas válidos según el CSV 
sistemas_por_ciudad = {
    "Cali/Valle": ["MIO"],
    "Cartagena": ["TRANSCARIBE"],
    "Barranquilla": ["TRANSMETRO"],
    "Bucaramanga": ["METROLINEA"],
    "Pereira": ["MEGABUS"],
    "Medellin": ["SITVA"],
    "Bogotá": ["TRANSMILENIO/SITP", "TRONCAL", "ZONAL Y DUAL"],
    "Bogotá y Soacha": ["TRANSMILENIO/SITP", "TRONCAL", "ZONAL Y DUAL"],
}


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["Inicio", "Exploración", "Modelos ML", "Predicción", "Métricas Avanzadas", "Descargas / Evidencias"]) 

# -----------------------------
# Página: Inicio (Hero + KPIs)
# -----------------------------
if page == "Inicio":
    st.title("🚌 Predicción de Pasajeros - Transporte Limpio")
    st.write("App ° resultados, métricas y predicciones.")

    # KPIs principales en fila
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", f"{len(df):,}")
    col2.metric("Ciudades", df['Ciudad'].nunique())
    col3.metric("Sistemas", df['Sistema'].nunique())
    if 'Pasajeros/dia' in df.columns:
        col4.metric("Promedio pasajeros/día", f"{df['Pasajeros/dia'].mean():,.0f}")

    st.markdown("---")
    st.subheader("Descripción rápida del proyecto")
    st.write("Usamos Random Forest y Gradient Boosting para predecir pasajeros diarios. La app permite explorar el dataset, validar métricas y generar predicciones interactivas.")

    if not models_loaded:
        st.warning("Los modelos no fueron cargados: asegúrate de ejecutar primero el entrenamiento y que los .pkl estén en la carpeta.\n\nDetalle: " + load_error)

# -----------------------------
# Página: Exploración
# -----------------------------
elif page == "Exploración":
    st.header("Exploración de datos")

    # Filtros rápidos
    with st.expander("Filtros rápidos"):
        ciudades = st.multiselect("Ciudades", options=sorted(df['Ciudad'].unique()), default=sorted(df['Ciudad'].unique())[:3])
        sistemas = st.multiselect("Sistemas", options=sorted(df['Sistema'].unique()), default=sorted(df['Sistema'].unique())[:2])
        fecha_range = None
        if 'Fecha' in df.columns:
            min_date = df['Fecha'].min()
            max_date = df['Fecha'].max()
            fecha_range = st.date_input("Rango de fechas", value=(min_date, max_date))

    df_f = df.copy()
    if ciudades:
        df_f = df_f[df_f['Ciudad'].isin(ciudades)]
    if sistemas:
        df_f = df_f[df_f['Sistema'].isin(sistemas)]
    if fecha_range and len(fecha_range) == 2 and 'Fecha' in df_f.columns:
        df_f = df_f[(df_f['Fecha'] >= pd.to_datetime(fecha_range[0])) & (df_f['Fecha'] <= pd.to_datetime(fecha_range[1]))]

    st.subheader("Vista de datos (primeras 100 filas)")
    st.dataframe(df_f.head(100), use_container_width=True)

    st.markdown("---")
    st.subheader("Gráficos interactivos")
    # Box pasajeros por ciudad
    if 'Pasajeros/dia' in df.columns:
        fig_box = px.box(df_f, x='Ciudad', y='Pasajeros/dia', title='Distribución de pasajeros por ciudad')
        st.plotly_chart(fig_box, use_container_width=True)

    # Tendencia temporal
    if 'Fecha' in df.columns and 'Pasajeros/dia' in df.columns:
        df_time = df_f.groupby('Fecha', as_index=False)['Pasajeros/dia'].sum()
        fig_line = px.line(df_time, x='Fecha', y='Pasajeros/dia', title='Tendencia temporal (suma diaria)')
        st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# Página: Modelos ML
# -----------------------------
elif page == "Modelos ML":
    st.header("🤖 Evaluación de los Modelos de Machine Learning")

    if not models_loaded:
        st.error("Los modelos no están cargados. Asegúrate de ejecutar el script de entrenamiento primero.")
    else:
        st.write("""
        En esta sección puedes ver el rendimiento real de ambos modelos de Machine Learning.
        Las métricas se calculan sobre una muestra representativa del dataset.
        """)

        # Preprocesamiento
        df_p = df.copy()
        df_p["Año"] = df_p["Fecha"].dt.year
        df_p["Mes"] = df_p["Fecha"].dt.month
        df_p["Dia"] = df_p["Fecha"].dt.day

        df_p["Ciudad_encoded"] = le_ciudad.transform(df_p["Ciudad"])
        df_p["Sistema_encoded"] = le_sistema.transform(df_p["Sistema"])

        features = [
            "Ciudad_encoded","Sistema_encoded","Variación Transmilenio",
            "Pasajeros día típico laboral","Pasajeros día sábado","Pasajeros día festivo",
            "DíaSemana","Año","Mes","Dia"
        ]

        sample = df_p.sample(n=min(500, len(df_p)), random_state=42)
        X = sample[features]
        y = sample["Pasajeros/dia"]

        Xs = scaler.transform(X)
        y_pred_rf = rf_model.predict(Xs)
        y_pred_gb = gb_model.predict(Xs)

        r2_rf, rmse_rf, mae_rf = compute_model_metrics(y, y_pred_rf)
        r2_gb, rmse_gb, mae_gb = compute_model_metrics(y, y_pred_gb)

        # ---------------------------------------------------------------------
        # DISEÑO: Contenedores con tarjetas para ambos modelos
        # ---------------------------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("### 🌲 Random Forest")
                st.metric("R² (Coeficiente de Determinación)", f"{r2_rf:.4f}")
                st.caption("**Qué significa R²:** Indica qué tanto el modelo explica la variabilidad real de los datos. 1.0 = predicción perfecta.")

                st.metric("RMSE (Error cuadrático medio)", f"{rmse_rf:,.2f}")
                st.caption("**RMSE:** Es el error promedio entre las predicciones y los valores reales. Entre más bajo, mejor.")

                st.metric("MAE (Error absoluto medio)", f"{mae_rf:,.2f}")
                st.caption("**MAE:** Error promedio absoluto. Mide cuánto se equivoca el modelo en promedio.")

                if r2_rf >= 0.85:
                    st.success("Cumple con el requisito del parcial (R² ≥ 0.85)")
                else:
                    st.warning("⚠️ No cumple con el requisito del parcial")

        with col2:
            with st.container(border=True):
                st.markdown("### 🚀 Gradient Boosting")
                st.metric("R² (Coeficiente de Determinación)", f"{r2_gb:.4f}")
                st.caption("**R²:** Mide qué tanto el modelo captura la relación real entre las variables de entrada y la salida.")

                st.metric("RMSE (Error cuadrático medio)", f"{rmse_gb:,.2f}")
                st.caption("**RMSE:** Error cuadrático promedio entre los valores reales y las predicciones.")

                st.metric("MAE (Error absoluto medio)", f"{mae_gb:,.2f}")
                st.caption("**MAE:** Indica en promedio cuántos pasajeros se equivoca el modelo.")

                if r2_gb >= 0.85:
                    st.success("Cumple con el requisito del parcial (R² ≥ 0.85)")
                else:
                    st.warning("⚠️ No cumple con el requisito del parcial")

        st.markdown("---")

        # Comparación gráfica
        st.subheader("📈 Comparación: Predicciones vs Valores Reales (Random Forest)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, y=y_pred_rf, mode="markers",
            name="Predicción", marker=dict(opacity=0.6, color="skyblue")
        ))
        fig.add_trace(go.Scatter(
            x=[y.min(), y.max()], y=[y.min(), y.max()],
            mode="lines", name="Línea perfecta", line=dict(color="red", dash="dash")
        ))
        fig.update_layout(xaxis_title="Real", yaxis_title="Predicción")
        st.plotly_chart(fig, use_container_width=True)
        
# -----------------------------
# Página: Predicción (form)
# -----------------------------
elif page == "Predicción":
    st.header("Hacer una predicción")

    if not models_loaded:
        st.error("Modelos no cargados. No es posible predecir hasta que estén disponibles.")
    else:

        # ---------------------------------------------------
        # FILTRO INTELIGENTE Ciudad → Sistema (con session_state)
        # ---------------------------------------------------
        if "ciudad_sel" not in st.session_state:
            st.session_state["ciudad_sel"] = le_ciudad.classes_[0]

        if "sistema_sel" not in st.session_state:
            st.session_state["sistema_sel"] = le_sistema.classes_[0]

        ciudad = st.selectbox(
            "Ciudad",
            options=le_ciudad.classes_,
            index=list(le_ciudad.classes_).index(st.session_state["ciudad_sel"]),
            key="ciudad_sel"
        )

        # Filtrar sistemas válidos según la ciudad
        sistemas_validos = sistemas_por_ciudad.get(ciudad, [])

        # Si el sistema actual no pertenece a la ciudad → resetear
        if st.session_state["sistema_sel"] not in sistemas_validos:
            st.session_state["sistema_sel"] = sistemas_validos[0]

        sistema = st.selectbox(
            "Sistema",
            options=sistemas_validos,
            index=sistemas_validos.index(st.session_state["sistema_sel"]),
            key="sistema_sel"
        )

        # ---------------------------------------------------
        # Resto del formulario
        # ---------------------------------------------------
        with st.form("pred_form"):
            st.subheader("Datos de entrada")

            c1, c2 = st.columns(2)

            with c1:
                variacion = st.number_input("Variación Transmilenio", value=-0.7, step=0.01)
                pasajeros_laboral = st.number_input(
                    "Pasajeros día típico laboral",
                    value=int(df['Pasajeros día típico laboral'].median()),
                    step=1000
                )

            with c2:
                pasajeros_sabado = st.number_input("Pasajeros día sábado",
                    value=int(df['Pasajeros día sábado'].median()),
                    step=1000)
                pasajeros_festivo = st.number_input("Pasajeros día festivo",
                    value=int(df['Pasajeros día festivo'].median()),
                    step=1000)
                dia_semana = st.selectbox(
                    "Día de la semana",
                    options=[1,2,3,4,5,6,7],
                    format_func=lambda x: {1:'Lun',2:'Mar',3:'Mie',4:'Jue',5:'Vie',6:'Sab',7:'Dom'}[x]
                )

            año = st.number_input("Año", min_value=2000, max_value=2100, value=2024)
            mes = st.number_input("Mes", min_value=1, max_value=12, value=8)
            dia = st.number_input("Día", min_value=1, max_value=31, value=15)

            submitted = st.form_submit_button("Predecir 🚀")

        # ---------------------------------------------------
        # Cuando se presiona el botón
        # ---------------------------------------------------
        if submitted:
            try:
                ciudad_enc = le_ciudad.transform([ciudad])[0]
                sistema_enc = le_sistema.transform([sistema])[0]
            except Exception as e:
                st.error(f"Error codificando ciudad/sistema: {e}")
            else:
                X_in = np.array([[ciudad_enc, sistema_enc, variacion,
                                  pasajeros_laboral, pasajeros_sabado, pasajeros_festivo,
                                  dia_semana, año, mes, dia]])
                Xs = scaler.transform(X_in)

                pred_rf = rf_model.predict(Xs)[0]
                pred_gb = gb_model.predict(Xs)[0]
                pred_mean = (pred_rf + pred_gb) / 2

                st.success("Predicción completada")

                c1, c2, c3 = st.columns(3)
                c1.metric("Random Forest", f"{pred_rf:,.0f}")
                c2.metric("Gradient Boosting", f"{pred_gb:,.0f}")
                c3.metric("Promedio", f"{pred_mean:,.0f}")

                fig_comp = go.Figure(data=[
                    go.Bar(x=['RF','GB','Promedio'], y=[pred_rf, pred_gb, pred_mean])
                ])
                fig_comp.update_layout(title='Comparación de predicciones', yaxis_title='Pasajeros')
                st.plotly_chart(fig_comp, use_container_width=True)

# -----------------------------
# Página: Métricas Avanzadas
# -----------------------------
elif page == "Métricas Avanzadas":

    
    st.header("Análisis detallado de errores y residuales")

    if not models_loaded:
        st.error("Modelos no disponibles.")
    else:
        # Sample para análisis
        df_p = df.copy()
        df_p['Año'] = df_p['Fecha'].dt.year if 'Fecha' in df_p.columns else df_p.get('Año', 2020)
        df_p['Mes'] = df_p['Fecha'].dt.month if 'Fecha' in df_p.columns else df_p.get('Mes', 1)
        df_p['Dia'] = df_p['Fecha'].dt.day if 'Fecha' in df_p.columns else df_p.get('Dia', 1)
        df_p['Ciudad_encoded'] = le_ciudad.transform(df_p['Ciudad'])
        df_p['Sistema_encoded'] = le_sistema.transform(df_p['Sistema'])

        features = ['Ciudad_encoded','Sistema_encoded','Variación Transmilenio',
                    'Pasajeros día típico laboral','Pasajeros día sábado','Pasajeros día festivo',
                    'DíaSemana','Año','Mes','Dia']

        sample = df_p.sample(n=min(500, len(df_p)), random_state=42)
        X = sample[features]
        y = sample['Pasajeros/dia']
        Xs = scaler.transform(X)
        y_pred_rf = rf_model.predict(Xs)
        y_pred_gb = gb_model.predict(Xs)

        errors_rf = y - y_pred_rf
        errors_gb = y - y_pred_gb

        st.subheader("Distribución de errores")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(errors_rf, nbins=50, title='Errores RF'), use_container_width=True)
        c2.plotly_chart(px.histogram(errors_gb, nbins=50, title='Errores GB'), use_container_width=True)

        st.subheader("Residuales vs Predicción (RF)")
        fig_res = px.scatter(x=y_pred_rf, y=errors_rf, labels={'x':'Predicciones','y':'Residuales'})
        fig_res.add_hline(y=0, line_dash='dash')
        st.plotly_chart(fig_res, use_container_width=True)

        # ------------------------------------------
        # Importancia de Variables
        # ------------------------------------------
        st.subheader("📊 Importancia de Variables")

        # Obtener importancia de variables
        # Usar los modelos cargados y la lista de features definida arriba
        importancia_rf = rf_model.feature_importances_
        importancia_gb = gb_model.feature_importances_
        variables = features

        # Graficar importancia con plotly (RF positiva, GB negativa para separar)
        fig_imp = go.Figure()
        # Invertir orden para que la variable más importante aparezca arriba
        vars_rev = variables[::-1]
        rf_rev = importancia_rf[::-1]
        gb_rev = importancia_gb[::-1]

        fig_imp.add_trace(go.Bar(
            x=rf_rev, y=vars_rev, orientation='h', name='Random Forest',
            marker=dict(color='rgba(50,150,200,0.8)')
        ))
        fig_imp.add_trace(go.Bar(
            x=[-v for v in gb_rev], y=vars_rev, orientation='h', name='Gradient Boosting',
            marker=dict(color='rgba(200,100,100,0.8)')
        ))

        fig_imp.update_layout(
            title="Importancia de Variables (RF arriba / GB abajo)",
            xaxis_title="Importancia",
            barmode='relative',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------
# Página: Descargas / Evidencias
# -----------------------------
elif page == "Descargas / Evidencias":
    st.header("Evidencias y descargas")
    st.write("En esta sección puedes descargar los artefactos generados y dejar evidencia para el parcial.")

    # Link local al CSV subido
    st.markdown("**Archivo de datos (local):**")
    st.write("transporte_limpio.csv")
    st.download_button("Descargar CSV original", data=open('transporte_limpio.csv','rb'), file_name='transporte_limpio.csv')

    st.markdown("**Modelos y transformadores:**")
    for f in ['random_forest_model.pkl','gradient_boosting_model.pkl','scaler.pkl','le_ciudad.pkl','le_sistema.pkl']:
        try:
            st.write(f)
            st.download_button(f"Descargar {f}", data=open(f,'rb'), file_name=f)
        except Exception:
            st.info(f"Archivo {f} no disponible en la carpeta")

    st.markdown("---")

# Footer
st.markdown('---')
st.caption('🚌 Sistema de Predicción de Pasajeros - Transporte Limpio')
