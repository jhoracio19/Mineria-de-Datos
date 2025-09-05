# Distribucción binomial (simulación)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# --- Entradas del usuario ---
n = int(input("¿Cuántas veces se quieres lanzar la moneda? (ejemplo: 10): "))
p = float(input("¿Probabilidad de cara? (ejemplo: 0.5): "))
if not (0 <= p <= 1):
    raise ValueError("La probabilidad debe estar entre 0 y 1")

# --- Valores posibles ---
k = np.arange(0, n+1)

# --- Distribución binomial ---
pmf = binom.pmf(k, n, p)

# --- Gráfico ---
plt.figure(figsize=(8,5))
plt.stem(k, pmf, basefmt=" ")
plt.title(f"Distribución Binomial: n={n}, p={p}")
plt.xlabel("Número de caras (k)")
plt.ylabel("Probabilidad P(X=k)")
plt.grid(alpha=0.3)
plt.show()

