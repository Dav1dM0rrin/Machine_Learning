"""
Sistema de Machine Learning para Predicción de Transporte Público
Proyecto: Minería de Datos 2025-2
Algoritmos: Random Forest Regressor y XGBoost Classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, classification_report, confusion_matrix,
                            precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# CARGAR Y PREPARAR DATOS
# ============================================
def cargar_y_preparar_datos(ruta_csv='transporte_limpio.csv'):
    """
    Carga y realiza Feature Engineering avanzado
    """
    print("\n" + "="*70)
    print("📂 CARGANDO Y PREPARANDO DATOS")
    print("="*70)
    
    df = pd.read_csv(ruta_csv)
    print(f"✅ Registros cargados: {len(df)}")
    print(f"✅ Columnas: {list(df.columns)}")
    
    # Convertir fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
    
    # Feature Engineering: Extraer características temporales
    df['Mes'] = df['Fecha'].dt.month
    df['Año'] = df['Fecha'].dt.year
    df['DiaMes'] = df['Fecha'].dt.day
    df['Trimestre'] = df['Fecha'].dt.quarter
    
    # Características categóricas mejoradas
    df['EsFinDeSemana'] = df['DiaSemana'].isin([6, 7]).astype(int)
    df['EsLaboralPico'] = df['DiaSemana'].isin([1, 2, 3, 4, 5]).astype(int)
    
    # Temporada (basado en análisis del dataset)
    def obtener_temporada(mes):
        if mes in [3, 4, 5, 9, 10, 11]:  # Meses de pandemia y restricciones
            return 'Restriccion'
        elif mes in [6, 7, 8]:  # Meses de reapertura
            return 'Reapertura'
        else:
            return 'Normal'
    
    df['Temporada'] = df['Mes'].apply(obtener_temporada)
    
    # Calcular variación respecto a día típico
    df['VariacionLaboral'] = (df['Pasajeros/dia'] / df['Pasajeros dia tipico laboral']) - 1
    df['VariacionSabado'] = (df['Pasajeros/dia'] / df['Pasajeros dia sabado']) - 1
    
    # Llenar NaN con 0
    df['VariacionLaboral'].fillna(0, inplace=True)
    df['VariacionSabado'].fillna(0, inplace=True)
    
    print(f"\n✨ Features creados:")
    print(f"   • Temporales: Mes, Año, DiaMes, Trimestre")
    print(f"   • Categóricos: EsFinDeSemana, EsLaboralPico, Temporada")
    print(f"   • Variaciones: VariacionLaboral, VariacionSabado")
    
    return df

# ============================================
# ALGORITMO 1: RANDOM FOREST REGRESSOR (OPTIMIZADO)
# ============================================
def entrenar_random_forest(df):
    """
    Random Forest optimizado para predecir pasajeros diarios
    """
    print("\n" + "="*70)
    print("🌳 ALGORITMO 1: RANDOM FOREST REGRESSOR")
    print("="*70)
    
    # Preparar datos
    df_rf = df.copy()
    df_rf = df_rf.dropna(subset=['Pasajeros/dia'])
    
    # Encoders para variables categóricas
    le_ciudad = LabelEncoder()
    le_sistema = LabelEncoder()
    le_temporada = LabelEncoder()
    
    df_rf['Ciudad_encoded'] = le_ciudad.fit_transform(df_rf['Ciudad'])
    df_rf['Sistema_encoded'] = le_sistema.fit_transform(df_rf['Sistema'])
    df_rf['Temporada_encoded'] = le_temporada.fit_transform(df_rf['Temporada'])
    
    # Características mejoradas
    features = [
        'Ciudad_encoded', 'Sistema_encoded', 'DiaSemana',
        'Pasajeros dia tipico laboral', 'Pasajeros dia sabado', 
        'Pasajeros dia festivo', 'Mes', 'Año', 'DiaMes', 'Trimestre',
        'EsFinDeSemana', 'EsLaboralPico', 'Temporada_encoded',
        'VariacionLaboral', 'VariacionSabado'
    ]
    
    X = df_rf[features]
    y = df_rf['Pasajeros/dia']
    
    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Dataset dividido:")
    print(f"   • Train: {len(X_train)} registros")
    print(f"   • Test: {len(X_test)} registros")
    
    # Modelo optimizado
    print("\n🌳 Entrenando Random Forest (optimizado)...")
    rf_model = RandomForestRegressor(
        n_estimators=200,           # Más árboles
        max_depth=20,               # Mayor profundidad
        min_samples_split=3,        # Más flexible
        min_samples_leaf=2,
        max_features='sqrt',        # Considera raíz cuadrada de features
        random_state=42,
        n_jobs=-1,                  # Usa todos los cores
        verbose=0
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Métricas
    r2_train = r2_score(y_train, y_pred_train) * 100
    r2_test = r2_score(y_test, y_pred_test) * 100
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n📊 RESULTADOS RANDOM FOREST:")
    print(f"   {'='*50}")
    print(f"   📈 R² Score (Train): {r2_train:.2f}%")
    print(f"   📈 R² Score (Test):  {r2_test:.2f}%")
    print(f"   📉 RMSE (Test):      {rmse_test:,.2f}")
    print(f"   📉 MAE (Test):       {mae_test:,.2f}")
    print(f"   {'='*50}")
    
    # Validación cruzada
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"   🔄 Cross-Validation (5-fold):")
    print(f"      R² promedio: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    # Verificar requisito del 85%
    if r2_test >= 85:
        print(f"\n   ✅ CUMPLE REQUISITO: {r2_test:.2f}% >= 85%")
    else:
        print(f"\n   ⚠️  NO CUMPLE: {r2_test:.2f}% < 85%")
    
    # Importancia de características
    importancias = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 TOP 10 Variables Más Importantes:")
    print(importancias.head(10).to_string(index=False))
    
    # Visualización de importancia
    plt.figure(figsize=(10, 6))
    top10 = importancias.head(10)
    plt.barh(top10['Feature'], top10['Importance'], color='skyblue')
    plt.xlabel('Importancia')
    plt.title('Top 10 Variables - Random Forest')
    plt.tight_layout()
    plt.savefig('rf_importancia.png', dpi=300, bbox_inches='tight')
    print("\n💾 Gráfico guardado: rf_importancia.png")
    
    # Visualización de predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Predicción Perfecta')
    plt.xlabel('Pasajeros Reales')
    plt.ylabel('Pasajeros Predichos')
    plt.title(f'Random Forest - Predicción vs Real (R²={r2_test:.2f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rf_predicciones.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: rf_predicciones.png")
    
    # Guardar modelo
    modelo_data = {
        'modelo': rf_model,
        'le_ciudad': le_ciudad,
        'le_sistema': le_sistema,
        'le_temporada': le_temporada,
        'features': features,
        'score': r2_test,
        'metricas': {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse': rmse_test,
            'mae': mae_test
        }
    }
    
    joblib.dump(modelo_data, 'modelo_random_forest.pkl')
    print("💾 Modelo guardado: modelo_random_forest.pkl")
    
    return modelo_data

# ============================================
# ALGORITMO 2: XGBOOST CLASSIFIER (OPTIMIZADO)
# ============================================
def entrenar_xgboost(df):
    """
    XGBoost (Gradient Boosting mejorado) para clasificar demanda
    """
    print("\n" + "="*70)
    print("⚡ ALGORITMO 2: XGBOOST CLASSIFIER")
    print("="*70)
    
    # Preparar datos
    df_xgb = df.copy()
    df_xgb = df_xgb.dropna(subset=['Pasajeros/dia'])
    
    # Crear categorías de demanda (más equilibradas)
    percentil_33 = df_xgb['Pasajeros/dia'].quantile(0.33)
    percentil_66 = df_xgb['Pasajeros/dia'].quantile(0.66)
    
    print(f"\n📊 Umbrales de Demanda:")
    print(f"   🔵 Baja:  < {percentil_33:,.0f} pasajeros")
    print(f"   🟠 Media: {percentil_33:,.0f} - {percentil_66:,.0f} pasajeros")
    print(f"   🔴 Alta:  > {percentil_66:,.0f} pasajeros")
    
    def categorizar_demanda(pasajeros):
        if pasajeros < percentil_33:
            return 0  # Baja
        elif pasajeros < percentil_66:
            return 1  # Media
        else:
            return 2  # Alta
    
    df_xgb['Demanda_Categoria'] = df_xgb['Pasajeros/dia'].apply(categorizar_demanda)
    
    # Verificar distribución
    distribucion = df_xgb['Demanda_Categoria'].value_counts().sort_index()
    print(f"\n📊 Distribución de Clases:")
    for cat, nombre in enumerate(['Baja', 'Media', 'Alta']):
        print(f"   {nombre}: {distribucion[cat]} registros ({distribucion[cat]/len(df_xgb)*100:.1f}%)")
    
    # Encoders
    le_ciudad = LabelEncoder()
    le_sistema = LabelEncoder()
    le_temporada = LabelEncoder()
    
    df_xgb['Ciudad_encoded'] = le_ciudad.fit_transform(df_xgb['Ciudad'])
    df_xgb['Sistema_encoded'] = le_sistema.fit_transform(df_xgb['Sistema'])
    df_xgb['Temporada_encoded'] = le_temporada.fit_transform(df_xgb['Temporada'])
    
    # Features
    features = [
        'Ciudad_encoded', 'Sistema_encoded', 'DiaSemana',
        'Pasajeros dia tipico laboral', 'Pasajeros dia sabado',
        'Pasajeros dia festivo', 'Mes', 'Año', 'DiaMes', 'Trimestre',
        'EsFinDeSemana', 'EsLaboralPico', 'Temporada_encoded',
        'VariacionLaboral', 'VariacionSabado'
    ]
    
    X = df_xgb[features]
    y = df_xgb['Demanda_Categoria']
    
    # División estratificada (mantiene proporciones de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset dividido:")
    print(f"   • Train: {len(X_train)} registros")
    print(f"   • Test: {len(X_test)} registros")
    
    # Modelo XGBoost optimizado
    print("\n⚡ Entrenando XGBoost (optimizado)...")
    xgb_model = XGBClassifier(
        n_estimators=200,           # Más iteraciones
        max_depth=6,                # Profundidad moderada
        learning_rate=0.1,          # Tasa de aprendizaje
        subsample=0.8,              # Submuestra de datos
        colsample_bytree=0.8,       # Submuestra de features
        random_state=42,
        eval_metric='mlogloss',     # Métrica para multiclase
        use_label_encoder=False,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # Predicciones
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # Métricas
    accuracy_train = accuracy_score(y_train, y_pred_train) * 100
    accuracy_test = accuracy_score(y_test, y_pred_test) * 100
    precision = precision_score(y_test, y_pred_test, average='weighted') * 100
    recall = recall_score(y_test, y_pred_test, average='weighted') * 100
    f1 = f1_score(y_test, y_pred_test, average='weighted') * 100
    
    print(f"\n📊 RESULTADOS XGBOOST:")
    print(f"   {'='*50}")
    print(f"   🎯 Accuracy (Train):  {accuracy_train:.2f}%")
    print(f"   🎯 Accuracy (Test):   {accuracy_test:.2f}%")
    print(f"   📈 Precision:         {precision:.2f}%")
    print(f"   📈 Recall:            {recall:.2f}%")
    print(f"   📈 F1-Score:          {f1:.2f}%")
    print(f"   {'='*50}")
    
    # Verificar requisito del 85%
    if accuracy_test >= 85:
        print(f"\n   ✅ CUMPLE REQUISITO: {accuracy_test:.2f}% >= 85%")
    else:
        print(f"\n   ⚠️  NO CUMPLE: {accuracy_test:.2f}% < 85%")
    
    # Reporte detallado
    print(f"\n📋 Reporte de Clasificación Detallado:")
    target_names = ['🔵 Baja', '🟠 Media', '🔴 Alta']
    print(classification_report(y_test, y_pred_test, target_names=target_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n🔢 Matriz de Confusión:")
    print(cm)
    
    # Visualización de matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Baja', 'Media', 'Alta'],
                yticklabels=['Baja', 'Media', 'Alta'])
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.title(f'Matriz de Confusión - XGBoost (Accuracy={accuracy_test:.2f}%)')
    plt.tight_layout()
    plt.savefig('xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n💾 Gráfico guardado: xgb_confusion_matrix.png")
    
    # Importancia de características
    importancias = pd.DataFrame({
        'Feature': features,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 TOP 10 Variables Más Importantes:")
    print(importancias.head(10).to_string(index=False))
    
    # Visualización de importancia
    plt.figure(figsize=(10, 6))
    top10 = importancias.head(10)
    plt.barh(top10['Feature'], top10['Importance'], color='coral')
    plt.xlabel('Importancia')
    plt.title('Top 10 Variables - XGBoost')
    plt.tight_layout()
    plt.savefig('xgb_importancia.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado: xgb_importancia.png")
    
    # Guardar modelo
    modelo_data = {
        'modelo': xgb_model,
        'le_ciudad': le_ciudad,
        'le_sistema': le_sistema,
        'le_temporada': le_temporada,
        'features': features,
        'accuracy': accuracy_test,
        'percentil_33': percentil_33,
        'percentil_66': percentil_66,
        'metricas': {
            'accuracy_train': accuracy_train,
            'accuracy_test': accuracy_test,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    joblib.dump(modelo_data, 'modelo_xgboost.pkl')
    print("💾 Modelo guardado: modelo_xgboost.pkl")
    
    return modelo_data

# ============================================
# PREDICCIÓN INTERACTIVA
# ============================================
def prediccion_interactiva():
    """
    Sistema de predicción por consola
    """
    print("\n" + "="*70)
    print("🎯 SISTEMA DE PREDICCIÓN INTERACTIVA")
    print("="*70)
    
    # Cargar modelos
    try:
        rf_data = joblib.load('modelo_random_forest.pkl')
        xgb_data = joblib.load('modelo_xgboost.pkl')
        print("✅ Modelos cargados exitosamente\n")
    except:
        print("❌ Error: Primero debes entrenar los modelos")
        return
    
    # Mostrar opciones
    print("📍 Ciudades disponibles:")
    ciudades = rf_data['le_ciudad'].classes_
    for i, ciudad in enumerate(ciudades):
        print(f"   {i}: {ciudad}")
    
    print("\n🚌 Sistemas disponibles:")
    sistemas = rf_data['le_sistema'].classes_
    for i, sistema in enumerate(sistemas):
        print(f"   {i}: {sistema}")
    
    print("\n🌤️  Temporadas:")
    print("   0: Normal")
    print("   1: Reapertura")
    print("   2: Restriccion")
    
    # Inputs del usuario
    try:
        ciudad_idx = int(input("\n▶️  Seleccione ciudad (número): "))
        sistema_idx = int(input("▶️  Seleccione sistema (número): "))
        dia_semana = int(input("▶️  Día de la semana (1=Lunes, 7=Domingo): "))
        mes = int(input("▶️  Mes (1-12): "))
        año = int(input("▶️  Año: "))
        pas_laboral = float(input("▶️  Pasajeros día típico laboral: "))
        pas_sabado = float(input("▶️  Pasajeros día sábado: "))
        pas_festivo = float(input("▶️  Pasajeros día festivo: "))
        temporada_idx = int(input("▶️  Temporada (0=Normal, 1=Reapertura, 2=Restriccion): "))
        
    except ValueError:
        print("❌ Error: Ingrese valores válidos")
        return
    
    # Preparar datos
    es_fin_semana = 1 if dia_semana in [6, 7] else 0
    es_laboral_pico = 1 if dia_semana in [1, 2, 3, 4, 5] else 0
    dia_mes = 15  # Asumimos día 15
    trimestre = (mes - 1) // 3 + 1
    variacion_laboral = 0
    variacion_sabado = 0
    
    input_data = pd.DataFrame({
        'Ciudad_encoded': [ciudad_idx],
        'Sistema_encoded': [sistema_idx],
        'DiaSemana': [dia_semana],
        'Pasajeros dia tipico laboral': [pas_laboral],
        'Pasajeros dia sabado': [pas_sabado],
        'Pasajeros dia festivo': [pas_festivo],
        'Mes': [mes],
        'Año': [año],
        'DiaMes': [dia_mes],
        'Trimestre': [trimestre],
        'EsFinDeSemana': [es_fin_semana],
        'EsLaboralPico': [es_laboral_pico],
        'Temporada_encoded': [temporada_idx],
        'VariacionLaboral': [variacion_laboral],
        'VariacionSabado': [variacion_sabado]
    })
    
    # Predicciones
    pred_rf = rf_data['modelo'].predict(input_data)[0]
    pred_xgb = xgb_data['modelo'].predict(input_data)[0]
    
    demanda_labels = ['🔵 BAJA', '🟠 MEDIA', '🔴 ALTA']
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("📈 RESULTADOS DE PREDICCIÓN")
    print("="*70)
    
    print(f"\n🌳 RANDOM FOREST (Regresión):")
    print(f"   Pasajeros estimados: {pred_rf:,.0f}")
    print(f"   Confianza: R²={rf_data['score']:.2f}%")
    
    print(f"\n⚡ XGBOOST (Clasificación):")
    print(f"   Nivel de demanda: {demanda_labels[pred_xgb]}")
    print(f"   Umbrales:")
    print(f"      • Baja:  < {xgb_data['percentil_33']:,.0f}")
    print(f"      • Media: {xgb_data['percentil_33']:,.0f} - {xgb_data['percentil_66']:,.0f}")
    print(f"      • Alta:  > {xgb_data['percentil_66']:,.0f}")
    print(f"   Confianza: Accuracy={xgb_data['accuracy']:.2f}%")
    
    print("\n" + "="*70)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("   SISTEMA DE MACHINE LEARNING - TRANSPORTE PÚBLICO")
    print("   Proyecto: Minería de Datos 2025-2")
    print("🚀"*35)
    
    # 1. Cargar datos
    df = cargar_y_preparar_datos('transporte_limpio.csv')
    
    # 2. Entrenar modelos
    rf_modelo = entrenar_random_forest(df)
    xgb_modelo = entrenar_xgboost(df)
    
    # 3. Resumen final
    print("\n" + "="*70)
    print("✅ RESUMEN FINAL - VALIDACIÓN DE REQUISITOS")
    print("="*70)
    
    print(f"\n🌳 Random Forest Regressor:")
    print(f"   • Score: {rf_modelo['score']:.2f}%")
    if rf_modelo['score'] >= 85:
        print(f"   • Estado: ✅ CUMPLE (≥85%)")
    else:
        print(f"   • Estado: ⚠️  NO CUMPLE (<85%)")
    
    print(f"\n⚡ XGBoost Classifier:")
    print(f"   • Accuracy: {xgb_modelo['accuracy']:.2f}%")
    if xgb_modelo['accuracy'] >= 85:
        print(f"   • Estado: ✅ CUMPLE (≥85%)")
    else:
        print(f"   • Estado: ⚠️  NO CUMPLE (<85%)")
    
    # 4. Predicción interactiva
    print("\n" + "="*70)
    continuar = input("\n¿Desea hacer una predicción interactiva? (s/n): ")
    if continuar.lower() == 's':
        prediccion_interactiva()
    
    print("\n✅ Proceso completado exitosamente")
    print("💾 Archivos generados:")
    print("   • modelo_random_forest.pkl")
    print("   • modelo_xgboost.pkl")
    print("   • rf_importancia.png")
    print("   • rf_predicciones.png")
    print("   • xgb_confusion_matrix.png")
    print("   • xgb_importancia.png")
    
    print("\n🚀 Siguiente paso: Crear aplicación Streamlit")