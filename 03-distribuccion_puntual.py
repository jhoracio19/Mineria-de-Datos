import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- Entrada del usuario ---
lam = float(input("¿Cuál es el valor de lambda (tasa promedio)? (ejemplo: 3): "))

# --- Valores posibles (0 a 3*lambda para ver buena parte de la masa) ---
k = np.arange(0, int(lam*3)+1)

# --- PMF ---
pmf = poisson.pmf(k, lam)

# --- Gráfico ---
plt.figure(figsize=(8,5))
plt.stem(k, pmf, basefmt=" ")
plt.title(f"Distribución Poisson (λ={lam})")
plt.xlabel("Número de eventos (k)")
plt.ylabel("Probabilidad P(X=k)")
plt.grid(alpha=0.3)
plt.show()
