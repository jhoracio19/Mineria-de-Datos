import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Entradas del usuario ---
mu = float(input("Introduce la media (μ), ejemplo 0: "))
sigma = float(input("Introduce la desviación estándar (σ>0), ejemplo 1: "))

if sigma <= 0:
    raise ValueError("La desviación estándar debe ser mayor que 0")

# --- Rango de valores ---
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)

# --- PDF ---
pdf = norm.pdf(x, mu, sigma)

# --- Gráfico ---
plt.figure(figsize=(8,5))
plt.plot(x, pdf, label=f"N({mu}, {sigma**2})")
plt.title(f"Distribución Normal (μ={mu}, σ={sigma})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axvline(mu, color="red", linestyle="--", label="Media")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
