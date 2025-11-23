import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CARGAR Y PREPARAR DATOS
# ============================================

def cargar_datos(ruta_csv='transporte_limpio.csv'):
    """Carga y prepara el dataset"""
    df = pd.read_csv( ruta_csv, encoding='latin1')
    

    # Convertir fecha a datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
    
    # Extraer características de fecha
    df['Mes'] = df['Fecha'].dt.month
    df['Año'] = df['Fecha'].dt.year
    
    # Eliminar columnas no útiles
    df = df.drop(['id', 'Fecha'], axis=1)
    
    return df

# ============================================
# ALGORITMO 1: RANDOM FOREST REGRESSOR
# Predice número de pasajeros diarios
# ============================================
def modelo_random_forest(df):
    """
    Random Forest para predecir pasajeros diarios
    """
    print("\n" + "="*60)
    print("ALGORITMO 1: RANDOM FOREST REGRESSOR")
    print("="*60)
    
    # Preparar datos
    df_rf = df.copy()
    
    # Eliminar filas con valores nulos en la variable objetivo
    df_rf = df_rf.dropna(subset=['Pasajeros/dia'])
    
    # Codificar variables categóricas
    le_ciudad = LabelEncoder()
    le_sistema = LabelEncoder()
    
    df_rf['Ciudad_encoded'] = le_ciudad.fit_transform(df_rf['Ciudad'])
    df_rf['Sistema_encoded'] = le_sistema.fit_transform(df_rf['Sistema'])
    
    # Seleccionar características
    features = ['Ciudad_encoded', 'Sistema_encoded', 'DiaSemana', 
                'Pasajeros dia tipico laboral', 'Pasajeros dia sabado', 
                'Pasajeros dia festivo', 'Mes', 'Año']
    
    X = df_rf[features]
    y = df_rf['Pasajeros/dia']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo
    print("\n🌳 Entrenando Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = rf_model.predict(X_test)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    score = rf_model.score(X_test, y_test) * 100
    
    print(f"\n📊 RESULTADOS:")
    print(f"   • Score (R²): {score:.2f}%")
    print(f"   • R² Score: {r2:.4f}")
    print(f"   • RMSE: {rmse:.2f}")
    
    # Importancia de características
    importancias = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 Importancia de Variables:")
    print(importancias.to_string(index=False))
    
    # Guardar encoders para predicción
    return rf_model, le_ciudad, le_sistema, features, score

# ============================================
# ALGORITMO 2: GRADIENT BOOSTING CLASSIFIER
# Clasifica demanda (Alta/Media/Baja)
# ============================================
def modelo_gradient_boosting(df):
    """
    Gradient Boosting para clasificar nivel de demanda
    """
    print("\n" + "="*60)
    print("ALGORITMO 2: GRADIENT BOOSTING CLASSIFIER")
    print("="*60)
    
    # Preparar datos
    df_gb = df.copy()
    df_gb = df_gb.dropna(subset=['Pasajeros/dia'])
    
    # Crear categorías de demanda basadas en percentiles
    percentil_33 = df_gb['Pasajeros/dia'].quantile(0.33)
    percentil_66 = df_gb['Pasajeros/dia'].quantile(0.66)
    
    def categorizar_demanda(pasajeros):
        if pasajeros < percentil_33:
            return 0  # Baja
        elif pasajeros < percentil_66:
            return 1  # Media
        else:
            return 2  # Alta
    
    df_gb['Demanda_Categoria'] = df_gb['Pasajeros/dia'].apply(categorizar_demanda)
    
    # Codificar variables
    le_ciudad = LabelEncoder()
    le_sistema = LabelEncoder()
    
    df_gb['Ciudad_encoded'] = le_ciudad.fit_transform(df_gb['Ciudad'])
    df_gb['Sistema_encoded'] = le_sistema.fit_transform(df_gb['Sistema'])
    
    # Características
    features = ['Ciudad_encoded', 'Sistema_encoded', 'DiaSemana',
                'Pasajeros dia tipico laboral', 'Pasajeros dia sabado',
                'Pasajeros dia festivo', 'Mes', 'Año']
    
    X = df_gb[features]
    y = df_gb['Demanda_Categoria']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    print("\n⚡ Entrenando Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = gb_model.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    print(f"\n📊 RESULTADOS:")
    print(f"   • Accuracy: {accuracy:.2f}%")
    
    print(f"\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Baja', 'Media', 'Alta']))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Matriz de Confusión:")
    print(cm)
    
    return gb_model, le_ciudad, le_sistema, features, accuracy, percentil_33, percentil_66

# ============================================
# FUNCIÓN DE PREDICCIÓN INTERACTIVA
# ============================================
def predecir_interactivo(rf_model, gb_model, le_ciudad_rf, le_sistema_rf, 
                         le_ciudad_gb, le_sistema_gb, features, 
                         percentil_33, percentil_66):
    """
    Permite hacer predicciones ingresando datos por consola
    """
    print("\n" + "="*60)
    print("🎯 PREDICCIÓN INTERACTIVA")
    print("="*60)
    
    # Obtener ciudades y sistemas disponibles
    ciudades = le_ciudad_rf.classes_
    sistemas = le_sistema_rf.classes_
    
    print("\n📍 Ciudades disponibles:")
    for i, ciudad in enumerate(ciudades):
        print(f"   {i}: {ciudad}")
    
    print("\n🚌 Sistemas disponibles:")
    for i, sistema in enumerate(sistemas):
        print(f"   {i}: {sistema}")
    
    # Input del usuario
    ciudad_idx = int(input("\nSeleccione ciudad (número): "))
    sistema_idx = int(input("Seleccione sistema (número): "))
    dia_semana = int(input("Día de la semana (1=Lunes, 7=Domingo): "))
    pas_laboral = float(input("Pasajeros día típico laboral: "))
    pas_sabado = float(input("Pasajeros día sábado: "))
    pas_festivo = float(input("Pasajeros día festivo: "))
    mes = int(input("Mes (1-12): "))
    año = int(input("Año: "))
    
    # Preparar datos para predicción
    input_data = pd.DataFrame({
        'Ciudad_encoded': [ciudad_idx],
        'Sistema_encoded': [sistema_idx],
        'DiaSemana': [dia_semana],
        'Pasajeros dia tipico laboral': [pas_laboral],
        'Pasajeros dia sabado': [pas_sabado],
        'Pasajeros dia festivo': [pas_festivo],
        'Mes': [mes],
        'Año': [año]
    })
    
    # Predicción Random Forest
    pred_rf = rf_model.predict(input_data)[0]
    
    # Predicción Gradient Boosting
    pred_gb = gb_model.predict(input_data)[0]
    demanda_labels = ['Baja', 'Media', 'Alta']
    
    print("\n" + "="*60)
    print("📈 RESULTADOS DE PREDICCIÓN")
    print("="*60)
    print(f"\n🌳 Random Forest:")
    print(f"   Pasajeros estimados: {pred_rf:,.0f}")
    
    print(f"\n⚡ Gradient Boosting:")
    print(f"   Nivel de demanda: {demanda_labels[pred_gb]}")
    print(f"   (Baja: <{percentil_33:,.0f} | Media: {percentil_33:,.0f}-{percentil_66:,.0f} | Alta: >{percentil_66:,.0f})")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # Cargar datos
    print("📂 Cargando datos...")
    df = cargar_datos('transporte_limpio.csv')
    
    print(f"✅ Dataset cargado: {len(df)} registros")
    print(f"📊 Columnas: {list(df.columns)}")
    
    # Ejecutar algoritmos
    rf_model, le_ciudad_rf, le_sistema_rf, features_rf, score_rf = modelo_random_forest(df)
    gb_model, le_ciudad_gb, le_sistema_gb, features_gb, accuracy_gb, p33, p66 = modelo_gradient_boosting(df)
    
    # Verificar si cumplen con el 85%
    print("\n" + "="*60)
    print("✅ VALIDACIÓN DE REQUISITOS")
    print("="*60)
    
    if score_rf >= 85:
        print(f"✅ Random Forest: {score_rf:.2f}% - CUMPLE (>85%)")
    else:
        print(f"⚠️ Random Forest: {score_rf:.2f}% - NO CUMPLE (>85%)")
    
    if accuracy_gb >= 85:
        print(f"✅ Gradient Boosting: {accuracy_gb:.2f}% - CUMPLE (>85%)")
    else:
        print(f"⚠️ Gradient Boosting: {accuracy_gb:.2f}% - NO CUMPLE (>85%)")
    
    # Predicción interactiva
    continuar = input("\n¿Desea hacer una predicción? (s/n): ")
    if continuar.lower() == 's':
        predecir_interactivo(rf_model, gb_model, le_ciudad_rf, le_sistema_rf,
                           le_ciudad_gb, le_sistema_gb, features_rf, p33, p66)