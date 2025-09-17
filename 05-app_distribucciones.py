import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import binom, norm, expon, bernoulli, poisson

st.set_page_config(page_title="Proyecto 2 | Simulador de Distribuciones", page_icon="📊", layout="wide")

# Ejecutar con:
# streamlit run (nombre_del_archivo).py

# ===== Sidebar principal =====
st.sidebar.title("📊 Menú de opciones")
opcion = st.sidebar.selectbox(
    "Elige una distribución / método:",
    ["Binomial", "Normal", "Exponencial", "Puntual (Bernoulli / Poisson)", "Normal Bivariada (Gibbs)"]
)

# ============================================================
#  BINOMIAL
# ============================================================
if opcion == "Binomial":
    st.title("⚪ Distribución Binomial")
    n = st.sidebar.number_input("Número de ensayos (n)", value=10, min_value=1, step=1)
    p = st.sidebar.slider("Probabilidad de éxito (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    k = np.arange(0, n+1)
    pmf = binom.pmf(k, n, p)

    fig, ax = plt.subplots()
    ax.stem(k, pmf, basefmt=" ")
    ax.set_title(f"Binomial(n={n}, p={p})")
    ax.set_xlabel("Número de éxitos (k)")
    ax.set_ylabel("P(X=k)")
    st.pyplot(fig)

# ============================================================
#  NORMAL
# ============================================================
elif opcion == "Normal":
    st.title("📈 Distribución Normal")
    mu = st.sidebar.number_input("Media (μ)", value=0.0, step=0.1)
    sigma = st.sidebar.number_input("Desviación estándar (σ)", value=1.0, min_value=0.01, step=0.1)

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    pdf = norm.pdf(x, mu, sigma)

    fig, ax = plt.subplots()
    ax.plot(x, pdf, label=f"N({mu}, {sigma**2:.2f})")
    ax.axvline(mu, color="red", ls="--", label="Media")
    ax.set_title("Distribución Normal")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(0, max(pdf) * 1.2)  # fija el eje Y para ver cambios al variar σ
    ax.legend()
    st.pyplot(fig)

# ============================================================
#  EXPONENCIAL
# ============================================================
elif opcion == "Exponencial":
    st.title("📉 Distribución Exponencial")
    lam = st.sidebar.number_input("Lambda (λ)", value=0.5, min_value=0.01, step=0.1)

    x = np.linspace(0, 10/lam, 500)
    pdf = expon.pdf(x, scale=1/lam)

    fig, ax = plt.subplots()
    ax.plot(x, pdf, label=f"Exp(λ={lam})")
    ax.set_title("Distribución Exponencial")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    st.pyplot(fig)

# ============================================================
#  PUNTUAL: Bernoulli / Poisson
# ============================================================
elif opcion == "Puntual (Bernoulli / Poisson)":
    st.title("🎲 Distribuciones Puntuales")
    tipo = st.sidebar.radio("Elige el tipo:", ["Bernoulli", "Poisson"])

    if tipo == "Bernoulli":
        p = st.sidebar.slider("Probabilidad de éxito (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        k = [0, 1]
        pmf = bernoulli.pmf(k, p)

        fig, ax = plt.subplots()
        ax.stem(k, pmf, basefmt=" ")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fracaso (0)", "Éxito (1)"])
        ax.set_title(f"Bernoulli(p={p})")
        ax.set_ylabel("P(X=k)")
        st.pyplot(fig)

    else:  # Poisson
        lam = st.sidebar.number_input("Lambda (λ)", value=3.0, min_value=0.1, step=0.5)
        k = np.arange(0, int(lam*5)+1)  # rango más amplio
        pmf = poisson.pmf(k, lam)

        fig, ax = plt.subplots()
        ax.stem(k, pmf, basefmt=" ")
        ax.set_title(f"Poisson(λ={lam})")
        ax.set_xlabel("Número de eventos (k)")
        ax.set_ylabel("P(X=k)")
        st.pyplot(fig)

# ============================================================
# 🔹 NORMAL BIVARIADA con Gibbs
# ============================================================
elif opcion == "Normal Bivariada (Gibbs)":
    st.title("🔄 Normal Bivariada usando Gibbs Sampling")

    mu1 = st.sidebar.number_input("μ1 (media X)", value=0.0, step=0.1)
    mu2 = st.sidebar.number_input("μ2 (media Y)", value=0.0, step=0.1)
    sigma1 = st.sidebar.number_input("σ1 (desv. X)", value=1.0, min_value=0.01, step=0.1)
    sigma2 = st.sidebar.number_input("σ2 (desv. Y)", value=1.0, min_value=0.01, step=0.1)
    rho = st.sidebar.slider("ρ (correlación)", min_value=-0.98, max_value=0.98, value=0.5, step=0.02)

    N = st.sidebar.number_input("Iteraciones", value=5000, step=1000)
    burn_in = st.sidebar.number_input("Burn-in", value=500, step=100)
    thin = st.sidebar.number_input("Thinning", value=1, min_value=1)
    skip = st.sidebar.slider("Submuestreo para gráfico", 1, 50, 10)

    # Gibbs Sampling
    def gibbs_bivar_normal(mu1, mu2, s1, s2, rho, N, burn_in, thin, x0=0, y0=0, seed=42):
        rng = np.random.default_rng(seed)
        x, y = x0, y0
        out_x, out_y = [], []

        var_x = (1 - rho**2) * s1**2
        var_y = (1 - rho**2) * s2**2
        sd_x = np.sqrt(var_x)
        sd_y = np.sqrt(var_y)

        for t in range(N + burn_in):
            mean_x = mu1 + rho * (s1 / s2) * (y - mu2)
            x = rng.normal(mean_x, sd_x)

            mean_y = mu2 + rho * (s2 / s1) * (x - mu1)
            y = rng.normal(mean_y, sd_y)

            if t >= burn_in and ((t - burn_in) % thin == 0):
                out_x.append(x)
                out_y.append(y)

        return np.array(out_x), np.array(out_y)

    # Ejecutar Gibbs
    X, Y = gibbs_bivar_normal(mu1, mu2, sigma1, sigma2, rho, int(N), int(burn_in), int(thin))

    # --- Gráfico 2D Scatter ---
    fig, ax = plt.subplots()
    ax.scatter(X[::skip], Y[::skip], s=8, alpha=0.4)
    ax.set_title("Muestras Gibbs - Normal Bivariada")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)

    # --- Superficie 3D (densidad) ---
    from mpl_toolkits.mplot3d import Axes3D  # necesario para proyección 3D

    fig3d = plt.figure(figsize=(7, 5))
    ax3d = fig3d.add_subplot(111, projection="3d")

    # Histograma 2D de densidad
    hist, xedges, yedges = np.histogram2d(X, Y, bins=40, density=True)

    xpos, ypos = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2
    )

    # Superficie
    ax3d.plot_surface(xpos, ypos, hist.T, cmap="viridis", edgecolor="none", alpha=0.9)

    ax3d.set_title("Superficie 3D - Densidad Gibbs")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Densidad")

    st.pyplot(fig3d)
