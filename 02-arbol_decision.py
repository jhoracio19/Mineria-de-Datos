import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Cargar y limpiar la BD
archivo = "BD_Ejercicio.xlsx"  # Ubicación de la BD 
df = pd.read_excel(archivo, sheet_name="Hoja1", skiprows=1)

# Renombrar columnas
df.columns = ["clase", "elemento", "A1", "A2", "A3", "A4", "A5"]

# Rellenar clases faltantes
df["clase"] = df["clase"].ffill()

# Seleccionar variables y convertir a int
X = df[["A1", "A2", "A3", "A4", "A5"]].astype(int)
y = df["clase"]


# Entrenar al modelo ID3
modelo = DecisionTreeClassifier(criterion="entropy", random_state=0)  # ID3 = entropía
modelo.fit(X, y)

# 3. Visualizar graficamente el arbol
plt.figure(figsize=(12, 8))  # tamaño más grande
plot_tree(modelo,
            feature_names=["A1", "A2", "A3", "A4", "A5"],
            class_names=modelo.classes_,
            filled=True,
            rounded=True)
plt.title("Árbol de Decisión - Algoritmo ID3", fontsize=14)
plt.show()
