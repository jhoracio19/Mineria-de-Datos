import numpy as np
import matplotlib.pyplot as plt

# --- Configuración ---
rng = np.random.default_rng(0)  # semilla fija
N = int(input("Número de iteraciones (ejemplo: 5000): "))
burn_in = int(input("Número de iteraciones de burn-in (ejemplo: 500): "))

# --- Muestreo de Y dado X ---
def sample_y_given_x(x):
    # f(y|x) = (2x+3y+2)/(2(2x+5)), y en (0,2)
    a = (2*x + 2) / (2*(2*x + 5))
    b = 3 / (2*(2*x + 5))

    u = rng.uniform()
    # Resolver (b/2)y^2 + a y - u = 0
    A = b/2; B = a; C = -u
    disc = B*B - 4*A*C
    y = (-B + np.sqrt(disc)) / (2*A)
    return float(np.clip(y, 1e-6, 2-1e-6))

# --- Muestreo de X dado Y ---
def sample_x_given_y(y):
    # f(x|y) = (2x+3y+2)/(2(3y+4)), x en (0,2)
    a = (3*y + 2) / (2*(3*y + 4))
    b = 1 / (3*y + 4)
    u = rng.uniform()
    # Resolver (b/2)x^2 + a x - u = 0
    A = b/2; B = a; C = -u
    disc = B*B - 4*A*C
    x = (-B + np.sqrt(disc)) / (2*A)
    return float(np.clip(x, 1e-6, 2-1e-6))

# --- Algoritmo de Gibbs ---
def gibbs(N, burn_in, x0=1.0, y0=1.0):
    x, y = x0, y0
    xs, ys = [], []
    for t in range(N + burn_in):
        y = sample_y_given_x(x)
        x = sample_x_given_y(y)
        if t >= burn_in:
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

# --- Ejecutar Gibbs ---
xs, ys = gibbs(N, burn_in)

print("Promedio muestral: E[X]≈", xs.mean(), " E[Y]≈", ys.mean())

# --- Graficar ---
plt.figure(figsize=(6,5))
plt.scatter(xs[::10], ys[::10], s=6, alpha=0.5)
plt.title("Simulación Gibbs de f(x,y) = (2x+3y+2)/28 en (0,2)x(0,2)")
plt.xlabel("X"); plt.ylabel("Y")
plt.grid(alpha=0.3)
plt.show()
