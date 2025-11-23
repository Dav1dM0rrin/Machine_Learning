import pandas as pd

df = pd.read_csv("aire_limpio.csv")

# Tomar 500 filas aleatorias
df_muestra = df.sample(n=1000, random_state=42)

df_muestra.to_csv("muestra_1000.csv", index=False)
