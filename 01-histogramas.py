# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Cargar el archivo Excel
ruta = "BD_CancerM.xlsx"  # Cambia por la ruta de tu archivo
df = pd.read_excel(ruta)

# 2️⃣ Renombrar columnas para facilidad
df.columns = ["ID", "diagnosis", "radius", "texture"]

# 3️⃣ Separar benignos y malignos
benignos = df[df["diagnosis"] == "B"]
malignos = df[df["diagnosis"] == "M"]

# -----------------------------
# 4️⃣ HISTOGRAMA GENERAL
# -----------------------------
plt.figure(figsize=(7, 5))
plt.hist(df["radius"], bins=20, edgecolor="black", color="skyblue")
plt.title("Histograma general - Radio del tumor")
plt.xlabel("Radio medio")
plt.ylabel("Frecuencia")
plt.savefig("histograma_general.png", dpi=300)  # Guarda imagen
plt.show()

# -----------------------------
# 5️⃣ HISTOGRAMA BENIGNOS
# -----------------------------
plt.figure(figsize=(7, 5))
plt.hist(benignos["radius"], bins=20, edgecolor="black", color="green")
plt.title("Histograma - Tumores Benignos")
plt.xlabel("Radio medio")
plt.ylabel("Frecuencia")
plt.savefig("histograma_benigno.png", dpi=300)  # Guarda imagen
plt.show()

# -----------------------------
# 6️⃣ HISTOGRAMA MALIGNOS
# -----------------------------
plt.figure(figsize=(7, 5))
plt.hist(malignos["radius"], bins=20, edgecolor="black", color="red")
plt.title("Histograma - Tumores Malignos")
plt.xlabel("Radio medio")
plt.ylabel("Frecuencia")
plt.savefig("histograma_maligno.png", dpi=300)  # Guarda imagen
plt.show()
