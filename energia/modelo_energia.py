# modelo_energia_simple.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ========================================================
# 1. Cargar datos (SIN LIMPIEZA – ya viene limpio de KNIME)
# ========================================================
df = pd.read_csv("energia_limpio.csv")

# Crear la feature adicional explicable
df["ratio_reactiva_potencia"] = df["ENERGÍA REACTIVA"] / (df["POTENCIA MÁXIMA"] + 1)

# Features finales
X = df[["ENERGÍA REACTIVA", "POTENCIA MÁXIMA", "ratio_reactiva_potencia"]]
y = df["ENERGÍA ACTIVA"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ========================================================
# 2. Modelos simples pero sólidos
# ========================================================
rf = RandomForestRegressor(
    n_estimators=400,
    random_state=42
)

gb = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.1,
    random_state=42
)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# ========================================================
# 3. Métricas
# ========================================================
def metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return r2, rmse, mae

pred_rf = rf.predict(X_test)
pred_gb = gb.predict(X_test)

r2_rf, rmse_rf, mae_rf = metrics(y_test, pred_rf)
r2_gb, rmse_gb, mae_gb = metrics(y_test, pred_gb)

print("\n===== RANDOM FOREST =====")
print(f"R²:   {r2_rf:.4f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"MAE:  {mae_rf:.2f}")

print("\n===== GRADIENT BOOSTING =====")
print(f"R²:   {r2_gb:.4f}")
print(f"RMSE: {rmse_gb:.2f}")
print(f"MAE:  {mae_gb:.2f}")

# ========================================================
# 4. Guardar modelos
# ========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/modelo_rf_simple.pkl")
joblib.dump(gb, "models/modelo_gb_simple.pkl")
joblib.dump(list(X.columns), "models/features_simple.pkl")

print("\nModelos guardados en carpeta /models")

# ========================================================
# 5. Predicción por consola
# ========================================================
def prediccion_consola():
    print("\n=== PREDICCIÓN POR CONSOLA (ENERGÍA ACTIVA) ===")

    try:
        er = float(input("ENERGÍA REACTIVA (kVAR): "))
        pmax = float(input("POTENCIA MÁXIMA (kW): "))
    except:
        print("Valor inválido.")
        return

    ratio = er / (pmax + 1)

    datos = pd.DataFrame([[er, pmax, ratio]],
                         columns=["ENERGÍA REACTIVA", "POTENCIA MÁXIMA", "ratio_reactiva_potencia"])

    prf = rf.predict(datos)[0]
    pgb = gb.predict(datos)[0]
    promedio = (prf + pgb) / 2

    print("\n===== RESULTADO =====")
    print(f"Random Forest:      {prf:,.2f} kWh")
    print(f"Gradient Boosting:  {pgb:,.2f} kWh")
    print(f"Promedio:           {promedio:,.2f} kWh")

run = input("\n¿Deseas hacer una predicción por consola? (s/n): ").strip().lower()
if run == "s":
    prediccion_consola()
else:
    print("Finalizado.")
