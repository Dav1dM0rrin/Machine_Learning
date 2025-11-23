import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_csv('transporte_limpio.csv')

print("=" * 60)
print("DATASET DE TRANSPORTE LIMPIO - ANÁLISIS INICIAL")
print("=" * 60)
print(f"\nDimensiones: {df.shape}")
print(f"\nColumnas: {df.columns.tolist()}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nInformación del dataset:\n{df.info()}")
print(f"\nEstadísticas:\n{df.describe()}")

# ============================================================================
# PREPROCESAMIENTO DE DATOS
# ============================================================================

# Eliminar columna ID
df = df.drop('ID', axis=1)

# Convertir fecha a datetime y extraer características
df['Fecha'] = pd.to_datetime(df['Fecha'])
df['Año'] = df['Fecha'].dt.year
df['Mes'] = df['Fecha'].dt.month
df['Dia'] = df['Fecha'].dt.day

# Limpiar columna "Variación Transmilenio" (reemplazar comas por puntos)
df['Variación Transmilenio'] = df['Variación Transmilenio'].replace('', np.nan)
df['Variación Transmilenio'] = df['Variación Transmilenio'].astype(str).str.replace(',', '.')
df['Variación Transmilenio'] = pd.to_numeric(df['Variación Transmilenio'], errors='coerce')

# Rellenar valores nulos
df['Variación Transmilenio'].fillna(df['Variación Transmilenio'].median(), inplace=True)

# Codificar variables categóricas
le_ciudad = LabelEncoder()
le_sistema = LabelEncoder()

df['Ciudad_encoded'] = le_ciudad.fit_transform(df['Ciudad'])
df['Sistema_encoded'] = le_sistema.fit_transform(df['Sistema'])

# Guardar encoders
joblib.dump(le_ciudad, 'le_ciudad.pkl')
joblib.dump(le_sistema, 'le_sistema.pkl')

print("\n" + "=" * 60)
print("ENCODERS GUARDADOS")
print("=" * 60)
print(f"Ciudades: {list(le_ciudad.classes_)}")
print(f"Sistemas: {list(le_sistema.classes_)}")

# ============================================================================
# PREPARAR DATOS PARA MODELADO
# ============================================================================

# Seleccionar características
features = ['Ciudad_encoded', 'Sistema_encoded', 'Variación Transmilenio', 
            'Pasajeros día típico laboral', 'Pasajeros día sábado', 
            'Pasajeros día festivo', 'DíaSemana', 'Año', 'Mes', 'Dia']

X = df[features]
y = df['Pasajeros/dia']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

print("\n" + "=" * 60)
print("DATOS PREPARADOS")
print("=" * 60)
print(f"Conjunto de entrenamiento: {X_train.shape}")
print(f"Conjunto de prueba: {X_test.shape}")

# ============================================================================
# MODELO 1: RANDOM FOREST REGRESSOR
# ============================================================================

print("\n" + "=" * 60)
print("MODELO 1: RANDOM FOREST REGRESSOR")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Métricas Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
accuracy_rf = r2_rf * 100

print(f"\n✓ Entrenamiento completado")
print(f"\nMÉTRICAS DEL MODELO:")
print(f"  • R² Score: {r2_rf:.4f}")
print(f"  • Precisión: {accuracy_rf:.2f}%")
print(f"  • RMSE: {rmse_rf:.2f}")
print(f"  • MAE: {mae_rf:.2f}")
print(f"  • MSE: {mse_rf:.2f}")

# Guardar modelo
joblib.dump(rf_model, 'random_forest_model.pkl')
print(f"\n✓ Modelo guardado: random_forest_model.pkl")

# ============================================================================
# MODELO 2: GRADIENT BOOSTING REGRESSOR
# ============================================================================

print("\n" + "=" * 60)
print("MODELO 2: GRADIENT BOOSTING REGRESSOR")
print("=" * 60)

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

# Métricas Gradient Boosting
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
accuracy_gb = r2_gb * 100

print(f"\n✓ Entrenamiento completado")
print(f"\nMÉTRICAS DEL MODELO:")
print(f"  • R² Score: {r2_gb:.4f}")
print(f"  • Precisión: {accuracy_gb:.2f}%")
print(f"  • RMSE: {rmse_gb:.2f}")
print(f"  • MAE: {mae_gb:.2f}")
print(f"  • MSE: {mse_gb:.2f}")

# Guardar modelo
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
print(f"\n✓ Modelo guardado: gradient_boosting_model.pkl")

# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================

print("\n" + "=" * 60)
print("COMPARACIÓN DE MODELOS")
print("=" * 60)

comparison = pd.DataFrame({
    'Modelo': ['Random Forest', 'Gradient Boosting'],
    'R² Score': [r2_rf, r2_gb],
    'Precisión (%)': [accuracy_rf, accuracy_gb],
    'RMSE': [rmse_rf, rmse_gb],
    'MAE': [mae_rf, mae_gb],
    'MSE': [mse_rf, mse_gb]
})

print(f"\n{comparison.to_string(index=False)}")

# Determinar mejor modelo
if accuracy_rf > accuracy_gb:
    print(f"\n🏆 MEJOR MODELO: Random Forest con {accuracy_rf:.2f}% de precisión")
else:
    print(f"\n🏆 MEJOR MODELO: Gradient Boosting con {accuracy_gb:.2f}% de precisión")

# ============================================================================
# PREDICCIÓN DE EJEMPLO CON DATOS INGRESADOS
# ============================================================================

print("\n" + "=" * 60)
print("EJEMPLO DE PREDICCIÓN")
print("=" * 60)

# Ejemplo de predicción
ejemplo_input = {
    'Ciudad': 'Bogotá',
    'Sistema': 'TRANSMILENIO/SITP',
    'Variación Transmilenio': -0.7,
    'Pasajeros día típico laboral': 3860061,
    'Pasajeros día sábado': 2499019,
    'Pasajeros día festivo': 1188607,
    'DíaSemana': 1,
    'Año': 2020,
    'Mes': 8,
    'Dia': 15
}

# Preparar datos para predicción
ciudad_encoded = le_ciudad.transform([ejemplo_input['Ciudad']])[0]
sistema_encoded = le_sistema.transform([ejemplo_input['Sistema']])[0]

ejemplo_features = np.array([[
    ciudad_encoded,
    sistema_encoded,
    ejemplo_input['Variación Transmilenio'],
    ejemplo_input['Pasajeros día típico laboral'],
    ejemplo_input['Pasajeros día sábado'],
    ejemplo_input['Pasajeros día festivo'],
    ejemplo_input['DíaSemana'],
    ejemplo_input['Año'],
    ejemplo_input['Mes'],
    ejemplo_input['Dia']
]])

ejemplo_scaled = scaler.transform(ejemplo_features)

pred_rf = rf_model.predict(ejemplo_scaled)[0]
pred_gb = gb_model.predict(ejemplo_scaled)[0]

print(f"\nDatos de entrada:")
for key, value in ejemplo_input.items():
    print(f"  • {key}: {value}")

print(f"\nPredicciones:")
print(f"  • Random Forest: {pred_rf:,.0f} pasajeros")
print(f"  • Gradient Boosting: {pred_gb:,.0f} pasajeros")

print("\n" + "=" * 60)
print("✓ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 60)
print("\nArchivos generados:")
print("  • random_forest_model.pkl")
print("  • gradient_boosting_model.pkl")
print("  • scaler.pkl")
print("  • le_ciudad.pkl")
print("  • le_sistema.pkl")

# ============================================================================
# SISTEMA DE PREDICCIÓN INTERACTIVO POR CONSOLA
# ============================================================================

def hacer_prediccion_consola():
    """Función para realizar predicciones interactivas por consola"""
    
    print("\n" + "=" * 60)
    print("🔮 SISTEMA DE PREDICCIÓN INTERACTIVO")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        print("Ingresa los datos para la predicción:")
        print("-" * 60)
        
        try:
            # Mostrar opciones de ciudades
            print("\n📍 CIUDADES DISPONIBLES:")
            for i, ciudad in enumerate(le_ciudad.classes_, 1):
                print(f"  {i}. {ciudad}")
            
            ciudad_idx = int(input("\nSelecciona el número de ciudad: ")) - 1
            if ciudad_idx < 0 or ciudad_idx >= len(le_ciudad.classes_):
                print("❌ Opción inválida. Intenta de nuevo.")
                continue
            ciudad = le_ciudad.classes_[ciudad_idx]
            
            # Mostrar opciones de sistemas
            print("\n🚌 SISTEMAS DISPONIBLES:")
            for i, sistema in enumerate(le_sistema.classes_, 1):
                print(f"  {i}. {sistema}")
            
            sistema_idx = int(input("\nSelecciona el número de sistema: ")) - 1
            if sistema_idx < 0 or sistema_idx >= len(le_sistema.classes_):
                print("❌ Opción inválida. Intenta de nuevo.")
                continue
            sistema = le_sistema.classes_[sistema_idx]
            
            # Solicitar datos numéricos
            print("\n📊 DATOS NUMÉRICOS:")
            variacion = float(input("Variación Transmilenio (ej: -0.7): "))
            pasajeros_laboral = int(input("Pasajeros día típico laboral (ej: 3860061): "))
            pasajeros_sabado = int(input("Pasajeros día sábado (ej: 2499019): "))
            pasajeros_festivo = int(input("Pasajeros día festivo (ej: 1188607): "))
            
            # Día de la semana
            print("\n📅 DÍA DE LA SEMANA:")
            print("  1=Lunes, 2=Martes, 3=Miércoles, 4=Jueves, 5=Viernes, 6=Sábado, 7=Domingo")
            dia_semana = int(input("Día de la semana (1-7): "))
            if dia_semana < 1 or dia_semana > 7:
                print("❌ Día inválido. Debe ser entre 1 y 7.")
                continue
            
            # Fecha
            print("\n📆 FECHA:")
            año = int(input("Año (ej: 2024): "))
            mes = int(input("Mes (1-12): "))
            dia = int(input("Día (1-31): "))
            
            if mes < 1 or mes > 12 or dia < 1 or dia > 31:
                print("❌ Fecha inválida.")
                continue
            
            # Preparar datos para predicción
            ciudad_encoded = le_ciudad.transform([ciudad])[0]
            sistema_encoded = le_sistema.transform([sistema])[0]
            
            input_features = np.array([[
                ciudad_encoded,
                sistema_encoded,
                variacion,
                pasajeros_laboral,
                pasajeros_sabado,
                pasajeros_festivo,
                dia_semana,
                año,
                mes,
                dia
            ]])
            
            input_scaled = scaler.transform(input_features)
            
            # Hacer predicciones
            pred_rf = rf_model.predict(input_scaled)[0]
            pred_gb = gb_model.predict(input_scaled)[0]
            pred_promedio = (pred_rf + pred_gb) / 2
            
            # Mostrar resultados
            print("\n" + "=" * 60)
            print("📊 RESULTADOS DE LA PREDICCIÓN")
            print("=" * 60)
            
            print("\n📋 DATOS INGRESADOS:")
            print(f"  • Ciudad: {ciudad}")
            print(f"  • Sistema: {sistema}")
            print(f"  • Variación Transmilenio: {variacion}")
            print(f"  • Pasajeros día laboral: {pasajeros_laboral:,}")
            print(f"  • Pasajeros sábado: {pasajeros_sabado:,}")
            print(f"  • Pasajeros festivo: {pasajeros_festivo:,}")
            dias = {1: 'Lunes', 2: 'Martes', 3: 'Miércoles', 4: 'Jueves', 
                   5: 'Viernes', 6: 'Sábado', 7: 'Domingo'}
            print(f"  • Día de la semana: {dias[dia_semana]}")
            print(f"  • Fecha: {dia:02d}/{mes:02d}/{año}")
            
            print("\n🎯 PREDICCIONES:")
            print(f"  🌲 Random Forest:      {pred_rf:>12,.0f} pasajeros")
            print(f"  🚀 Gradient Boosting:  {pred_gb:>12,.0f} pasajeros")
            print(f"  📊 Promedio:           {pred_promedio:>12,.0f} pasajeros")
            
            # Calcular diferencia porcentual entre modelos
            diff_percent = abs(pred_rf - pred_gb) / pred_promedio * 100
            print(f"\n📈 Diferencia entre modelos: {diff_percent:.2f}%")
            
            if diff_percent < 5:
                print("✅ Ambos modelos están muy alineados en la predicción.")
            elif diff_percent < 10:
                print("⚠️  Hay una diferencia moderada entre los modelos.")
            else:
                print("❌ Hay una diferencia significativa entre los modelos.")
            
        except ValueError as e:
            print(f"\n❌ Error: Entrada inválida. Por favor ingresa valores correctos.")
            print(f"   Detalle: {e}")
            continue
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            continue
        
        # Preguntar si desea hacer otra predicción
        print("\n" + "-" * 60)
        otra = input("\n¿Deseas hacer otra predicción? (s/n): ").lower()
        
        if otra != 's' and otra != 'si' and otra != 'sí':
            print("\n" + "=" * 60)
            print("👋 ¡Gracias por usar el sistema de predicción!")
            print("=" * 60)
            break

# Preguntar si desea usar el sistema de predicción
print("\n" + "=" * 60)
respuesta = input("\n¿Deseas realizar predicciones por consola? (s/n): ").lower()

if respuesta == 's' or respuesta == 'si' or respuesta == 'sí':
    hacer_prediccion_consola()
else:
    print("\n✓ Puedes hacer predicciones más tarde ejecutando la aplicación Streamlit.")
    print("  Comando: streamlit run app.py")

print("\n" + "=" * 60)
print("🎉 PROGRAMA FINALIZADO")
print("=" * 60)