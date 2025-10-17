import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Lee la bd
archivo = "BD_M_VyC_C.xlsx"

# extrae los primeros 10 datos
df = pd.read_excel(archivo, header=1, nrows=10)

print("Datos originales:\n", df.head(), "\n")

# Filtrar solo columnas numéricas
df_num = df.select_dtypes(include=[np.number])

# Elimina la columna ID 
if "%id" in df_num.columns:
    df_num = df_num.drop(columns=["%id"])

print("Datos numéricos usados (sin ID):\n", df_num.head(), "\n")

# 1) Matriz de varianza-covarianza
S = df_num.cov()
traza = np.trace(S)
det = np.linalg.det(S)

print("Matriz de varianza-covarianza (S):\n", S, "\n")
print(f"Traza(S): {traza:.4f}")
print(f"Determinante(S): {det:.4e}\n")

# 2) Matriz de correlación
R = df_num.corr()
print("Matriz de correlación (R):\n", R, "\n")

# 3) Exporta los resultados a Excel
with pd.ExcelWriter("resultados_VyC.xlsx") as writer:
    S.to_excel(writer, sheet_name="Matriz_Var_Cov")
    R.to_excel(writer, sheet_name="Matriz_Correlacion")
    resumen = pd.DataFrame({
        "Traza(S)": [traza],
        "Determinante(S)": [det]
    })
    resumen.to_excel(writer, sheet_name="Resumen", index=False)

print("Resultados guardados en 'resultados_VyC.xlsx'")

# 4) Visualización Gráfica de matriz de correlacion 
plt.figure(figsize=(10,10))
sns.heatmap(R, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor - Matriz de Correlación (R)")
plt.tight_layout()
plt.show()
