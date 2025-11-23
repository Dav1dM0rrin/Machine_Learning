"""
Modelo de Machine Learning para Clasificación de Calidad del Aire
Algoritmos: Random Forest y Gradient Boosting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGAR Y PREPARAR DATOS
# =============================================================================

def cargar_datos(ruta='aire_limpio.csv'):
    """Carga el dataset de calidad del aire"""
    df = pd.read_csv(ruta)
    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    return df

def crear_categoria_calidad(pm25):
    """
    Clasifica la calidad del aire según PM2.5 (estándar EPA)
    - Buena: 0-12
    - Moderada: 12.1-35.4
    - Mala: 35.5-55.4
    - Peligrosa: >55.4
    """
    if pm25 <= 12:
        return 'Buena'
    elif pm25 <= 35.4:
        return 'Moderada'
    elif pm25 <= 55.4:
        return 'Mala'
    else:
        return 'Peligrosa'

def preparar_datos(df):
    """Prepara los datos para el modelo"""
    # Crear copia
    data = df.copy()
    
    # Crear variable objetivo (clasificación de calidad del aire)
    data['Calidad_Aire'] = data['pm2_5'].apply(crear_categoria_calidad)
    
    # Mostrar distribución de clases
    print("\n📊 Distribución de Calidad del Aire:")
    print(data['Calidad_Aire'].value_counts())
    
    # Features para el modelo
    features = ['co', 'co2', 'pm10', 'pm5', 'Humedad Relativa', 'Temperatura']
    
    # Eliminar filas con valores faltantes o cero en temperatura/humedad
    data = data[(data['Temperatura'] != 0) & (data['Humedad Relativa'] != 0)]
    
    X = data[features]
    y = data['Calidad_Aire']
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\n✅ Datos preparados: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"   Clases: {list(le.classes_)}")
    
    return X, y_encoded, le, features

# =============================================================================
# 2. ENTRENAR MODELOS
# =============================================================================

def entrenar_random_forest(X_train, X_test, y_train, y_test):
    """Entrena y evalúa Random Forest"""
    print("\n" + "="*60)
    print("🌲 RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📈 MÉTRICAS:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   F1-Score: {f1*100:.2f}%")
    print(f"\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    return rf, accuracy, y_pred

def entrenar_gradient_boosting(X_train, X_test, y_train, y_test):
    """Entrena y evalúa Gradient Boosting"""
    print("\n" + "="*60)
    print("🚀 GRADIENT BOOSTING CLASSIFIER")
    print("="*60)
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📈 MÉTRICAS:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   F1-Score: {f1*100:.2f}%")
    print(f"\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    return gb, accuracy, y_pred

# =============================================================================
# 3. GUARDAR MODELOS
# =============================================================================

def guardar_modelos(rf, gb, scaler, le, features):
    """Guarda los modelos entrenados"""
    joblib.dump(rf, 'modelo_random_forest.pkl')
    joblib.dump(gb, 'modelo_gradient_boosting.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(features, 'features.pkl')
    print("\n💾 Modelos guardados exitosamente!")

# =============================================================================
# 4. FUNCIÓN DE PREDICCIÓN
# =============================================================================

def predecir_calidad(co, co2, pm10, pm5, humedad, temperatura):
    """Realiza predicción con los modelos guardados"""
    # Cargar modelos
    rf = joblib.load('modelo_random_forest.pkl')
    gb = joblib.load('modelo_gradient_boosting.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Preparar datos
    datos = np.array([[co, co2, pm10, pm5, humedad, temperatura]])
    datos_scaled = scaler.transform(datos)
    
    # Predicciones
    pred_rf = le.inverse_transform(rf.predict(datos_scaled))[0]
    pred_gb = le.inverse_transform(gb.predict(datos_scaled))[0]
    
    # Probabilidades
    prob_rf = rf.predict_proba(datos_scaled)[0]
    prob_gb = gb.predict_proba(datos_scaled)[0]
    
    return {
        'random_forest': {'prediccion': pred_rf, 'probabilidades': prob_rf},
        'gradient_boosting': {'prediccion': pred_gb, 'probabilidades': prob_gb}
    }

# =============================================================================
# 5. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("🌍 SISTEMA DE CLASIFICACIÓN DE CALIDAD DEL AIRE")
    print("="*60)
    
    # Cargar datos
    df = cargar_datos('aire_limpio.csv')
    
    # Preparar datos
    X, y, le, features = preparar_datos(df)
    
    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📦 División de datos:")
    print(f"   Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   Prueba: {X_test.shape[0]} muestras")
    
    # Entrenar modelos
    rf, acc_rf, pred_rf = entrenar_random_forest(X_train, X_test, y_train, y_test)
    gb, acc_gb, pred_gb = entrenar_gradient_boosting(X_train, X_test, y_train, y_test)
    
    # Guardar modelos
    guardar_modelos(rf, gb, scaler, le, features)
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL")
    print("="*60)
    print(f"   Random Forest:      {acc_rf*100:.2f}% accuracy")
    print(f"   Gradient Boosting:  {acc_gb*100:.2f}% accuracy")
    print("="*60)
    
    # Ejemplo de predicción por consola
    print("\n🔮 PREDICCIÓN DE EJEMPLO:")
    print("-"*40)
    resultado = predecir_calidad(
        co=15, co2=450, pm10=25, pm5=10, 
        humedad=60, temperatura=20
    )
    print(f"   Random Forest: {resultado['random_forest']['prediccion']}")
    print(f"   Gradient Boosting: {resultado['gradient_boosting']['prediccion']}")