# gibbs_normal_bivariada_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Normal Bivariada por Gibbs", page_icon="📈", layout="wide")

# ====== Sidebar (Parámetros) ======
st.sidebar.title("🎛️ Parámetros")
mu1 = st.sidebar.number_input("μ1 (media X)", value=0.0, step=0.1)
mu2 = st.sidebar.number_input("μ2 (media Y)", value=0.0, step=0.1)
sigma1 = st.sidebar.number_input("σ1 (desv. X)", value=1.0, min_value=0.01, step=0.1)
sigma2 = st.sidebar.number_input("σ2 (desv. Y)", value=1.2, min_value=0.01, step=0.1)
rho = st.sidebar.slider("ρ (correlación)", min_value=-0.98, max_value=0.98, value=0.6, step=0.02)

N = st.sidebar.number_input("Iteraciones totales", value=10000, step=1000)
burn_in = st.sidebar.number_input("Burn-in", value=1000, step=500, help="Muestras descartadas al inicio")
thin = st.sidebar.number_input("Thinning (cada k)", value=1, min_value=1, step=1)
seed = st.sidebar.number_input("Semilla", value=42, step=1)

col1, col2 = st.columns([1, 1])
with col1:
    x0 = st.number_input("X inicial", value=0.0, step=0.1)
with col2:
    y0 = st.number_input("Y inicial", value=0.0, step=0.1)

st.sidebar.markdown("---")
run = st.sidebar.button("▶️ Ejecutar Gibbs")
clear = st.sidebar.button("🧹 Limpiar")

st.title("📈 Normal Bivariada por Gibbs Sampling")
st.caption("Modelo:  (X, Y) ~ N( (μ1, μ2),  [[σ1², ρσ1σ2], [ρσ1σ2, σ2²]] )")

# ====== Utilidades ======
def gibbs_bivar_normal(mu1, mu2, s1, s2, rho, N, burn_in, thin, x0, y0, seed=0):
    """
    Gibbs para una Normal Bivariada usando condicionales:
    X|Y=y ~ N( μ1 + ρ (σ1/σ2) (y-μ2), (1-ρ²) σ1² )
    Y|X=x ~ N( μ2 + ρ (σ2/σ1) (x-μ1), (1-ρ²) σ2² )
    """
    rng = np.random.default_rng(seed)

    x, y = x0, y0
    out_x, out_y = [], []

    var_x = (1 - rho**2) * s1**2
    var_y = (1 - rho**2) * s2**2
    sd_x = np.sqrt(var_x)
    sd_y = np.sqrt(var_y)

    total_iters = int(N)
    b_in = int(burn_in)
    k = int(thin)

    for t in range(total_iters + b_in):
        # X | Y = y
        mean_x = mu1 + rho * (s1 / s2) * (y - mu2)
        x = rng.normal(mean_x, sd_x)

        # Y | X = x
        mean_y = mu2 + rho * (s2 / s1) * (x - mu1)
        y = rng.normal(mean_y, sd_y)

        if t >= b_in and ((t - b_in) % k == 0):
            out_x.append(x)
            out_y.append(y)

    return np.array(out_x), np.array(out_y)

def teoricos(mu1, mu2, s1, s2, rho):
    cov = rho * s1 * s2
    Sigma = np.array([[s1**2, cov], [cov, s2**2]])
    return Sigma

# ====== Acción ======
if run and not clear:
    X, Y = gibbs_bivar_normal(mu1, mu2, sigma1, sigma2, rho, N, burn_in, thin, x0, y0, seed)
    n = len(X)

    # Métricas muestrales
    mean_x, mean_y = X.mean(), Y.mean()
    std_x, std_y = X.std(ddof=1), Y.std(ddof=1)
    corr = np.corrcoef(X, Y)[0, 1]

    Sigma = teoricos(mu1, mu2, sigma1, sigma2, rho)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("E[X] (muestral)", f"{mean_x:.3f}", delta=f"vs μ1 {mu1:.2f}")
    mc2.metric("E[Y] (muestral)", f"{mean_y:.3f}", delta=f"vs μ2 {mu2:.2f}")
    mc3.metric("σ[X] (muestral)", f"{std_x:.3f}", delta=f"vs σ1 {sigma1:.2f}")
    mc4.metric("σ[Y] (muestral)", f"{std_y:.3f}", delta=f"vs σ2 {sigma2:.2f}")
    st.metric("ρ (muestral)", f"{corr:.3f}", delta=f"vs ρ {rho:.2f}")

    st.subheader("Nube de puntos y marginales")
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.25)

    # Scatter / densidad
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(X, Y, s=6, alpha=0.35)
    ax0.set_title("Muestras Gibbs")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    ax0.grid(alpha=0.2)

    # Histogramas verticales
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(X, bins=40, alpha=0.6, orientation='horizontal', label='X', color="#1f77b4")
    ax1.hist(Y, bins=40, alpha=0.6, orientation='horizontal', label='Y', color="#ff7f0e")
    ax1.set_title("Marginales")
    ax1.set_xlabel("Frecuencia")
    ax1.legend()
    ax1.grid(alpha=0.2)

    st.pyplot(fig)

    st.subheader("Trazas (convergencia)")
    fig2, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax[0].plot(X, lw=0.7)
    ax[0].axhline(mu1, color='r', ls='--', lw=1, label='μ1')
    ax[0].set_ylabel("X")
    ax[0].legend(); ax[0].grid(alpha=0.2)

    ax[1].plot(Y, lw=0.7, color="#ff7f0e")
    ax[1].axhline(mu2, color='r', ls='--', lw=1, label='μ2')
    ax[1].set_ylabel("Y")
    ax[1].set_xlabel("Iteración")
    ax[1].legend(); ax[1].grid(alpha=0.2)

    st.pyplot(fig2)

    st.subheader("Matriz de covarianzas teórica")
    st.write(Sigma)

elif clear:
    st.experimental_rerun()
else:
    st.info("Ajusta los parámetros en el panel izquierdo y presiona **▶️ Ejecutar Gibbs**.")
