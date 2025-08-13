# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt

# Cargar la bd
ruta = "BD_CancerM.xlsx"
df = pd.read_excel(ruta)

# renombrar columnas
df.columns = ["ID", "diagnosis", "radius", "texture"]

# Separar benignos y malignos
benignos = df[df["diagnosis"] == "B"]
malignos = df[df["diagnosis"] == "M"]


# Histogramas General
plt.figure(figsize=(7, 5))
plt.hist(df["radius"], bins=20, edgecolor="black", color="skyblue")
plt.title("Histograma general - Radio del tumor")
plt.xlabel("Radio medio")
plt.ylabel("Frecuencia")
plt.savefig("histograma_general.png", dpi=300)  
plt.show()

# Histograma Benignos
plt.figure(figsize=(7, 5))
plt.hist(benignos["radius"], bins=20, edgecolor="black", color="green")
plt.title("Histograma - Tumores Benignos")
plt.xlabel("Radio medio")
plt.ylabel("Frecuencia")
plt.savefig("histograma_benigno.png", dpi=300)  
plt.show()

# Histograma Malignos
plt.figure(figsize=(7, 5))
plt.hist(malignos["radius"], bins=20, edgecolor="black", color="red")
plt.title("Histograma - Tumores Malignos")
plt.xlabel("Radio medio")
plt.ylabel("Frecuencia")
plt.savefig("histograma_maligno.png", dpi=300)  
plt.show()
