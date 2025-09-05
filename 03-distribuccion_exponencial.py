import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# --- Entrada del usuario ---
lam = float(input("Introduce el valor de lambda (λ>0), ejemplo 0.5: "))

if lam <= 0:
    raise ValueError("Lambda debe ser mayor que 0")

# --- Rango de valores ---
x = np.linspace(0, 10/lam, 500)  # hasta ~10 veces la media

# --- PDF ---
pdf = expon.pdf(x, scale=1/lam)

# --- Gráfico ---
plt.figure(figsize=(8,5))
plt.plot(x, pdf, label=f"Exp(λ={lam})")
plt.title(f"Distribución Exponencial (λ={lam})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
