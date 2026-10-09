import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from joblib import load
import shap

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
logo_path = os.path.join("assets", "logo.png")
st.set_page_config(
    page_title="ChurnShield | Customer Churn Prediction",
    page_icon=logo_path if os.path.exists(logo_path) else "🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS — Dark Premium Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 60%, #0d1117 100%);
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(120deg, #1e3a5f 0%, #1a1f3a 50%, #0f2027 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 2rem;
    display: flex; align-items: center; gap: 1.8rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.55);
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -50%; left: -30%;
    width: 600px; height: 400px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon { font-size: 3.5rem; line-height: 1; flex-shrink: 0; }
.hero-logo-img {
    width: 68px; height: 68px;
    border-radius: 16px;
    object-fit: cover;
    border: 1px solid rgba(99,179,237,0.35);
    box-shadow: 0 4px 20px rgba(0,0,0,0.45);
    flex-shrink: 0;
}
.hero-title {
    font-size: 2.1rem; font-weight: 800;
    background: linear-gradient(90deg, #63b3ed, #a78bfa, #f687b3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle { color: #94a3b8; font-size: 0.98rem; margin-top: 0.35rem; }

/* ── Section Cards ── */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(4px);
}
.section-title {
    font-size: 1.15rem; font-weight: 700; letter-spacing: 0.02em;
    color: #60a5fa;
    margin-bottom: 1.1rem; display: flex; align-items: center; gap: 0.7rem;
}

/* ── Vector Picture Badge ── */
.heading-icon {
    width: 32px; height: 32px;
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(139, 92, 246, 0.25));
    border: 1px solid rgba(99, 179, 237, 0.35);
    border-radius: 9px;
    padding: 6px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    flex-shrink: 0;
}
.heading-icon svg {
    width: 100%; height: 100%;
    stroke: #60a5fa;
    stroke-width: 2;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.04);
    border-radius: 12px; padding: 4px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; color: #94a3b8;
    font-weight: 500; padding: 0.55rem 1.4rem;
    font-size: 0.9rem; transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4);
}

/* ── Inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] input,
.stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s ease;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
}
label, .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important; font-weight: 700 !important;
    width: 100% !important; transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.5) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important; font-size: 0.78rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 2rem !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #63b3ed, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899) !important;
    border-radius: 99px !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 99px !important; height: 10px !important;
}

/* ── Alerts ── */
.stSuccess, .stWarning, .stError {
    border-radius: 12px !important; border: none !important; font-weight: 600 !important;
}

/* ── Headings ── */
h2, h3 {
    background: linear-gradient(90deg, #e2e8f0, #94a3b8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper: Section Header with Vector Picture Icon
# ─────────────────────────────────────────────
def icon_title(label: str, svg_inner: str) -> str:
    return (
        f'<div class="section-title">'
        f'<span class="heading-icon">'
        f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{svg_inner}'
        f'</svg>'
        f'</span>'
        f'<span>{label}</span>'
        f'</div>'
    )

# ─────────────────────────────────────────────
# Load Models  (backend — untouched)
# ─────────────────────────────────────────────
logistic_model = load("logistic_model.pkl")
rf_model = load("rf_model.pkl")

preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

# ─────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────
def get_base64_image(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_b64 = get_base64_image(logo_path)
logo_markup = f'<img src="data:image/png;base64,{logo_b64}" class="hero-logo-img" alt="Logo">' if logo_b64 else '<div class="hero-icon">🛡️</div>'

st.markdown(f"""
<div class="hero-banner">
    {logo_markup}
    <div>
        <div class="hero-title">ChurnShield</div>
        <div class="hero-subtitle">AI-powered customer churn prediction &nbsp;·&nbsp; Logistic Regression &amp; Random Forest</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["Prediction Engine", "Model Intelligence & Insights"])

# ══════════════════════════════════════════════
# TAB 1 — Prediction
# ══════════════════════════════════════════════
with tab1:

    # Model selector
    st.markdown('<div class="section-card">' + icon_title(
        "Model Configuration",
        '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>'
    ), unsafe_allow_html=True)
    model_choice = st.radio(
        "Choose Model",
        ["Logistic Regression", "Random Forest"],
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Customer Profile
    st.markdown('<div class="section-card">' + icon_title(
        "Customer Profile",
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>'
    ), unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    with col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col4:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Tenure
    st.markdown('<div class="section-card">' + icon_title(
        "Tenure & Duration",
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/><circle cx="12" cy="15" r="2"/>'
    ), unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72)
    st.markdown('</div>', unsafe_allow_html=True)

    # Phone Services
    st.markdown('<div class="section-card">' + icon_title(
        "Phone Services",
        '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>'
    ), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    with col2:
        multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Internet & Add-ons
    st.markdown('<div class="section-card">' + icon_title(
        "Internet & Value-Added Services",
        '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'
    ), unsafe_allow_html=True)
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with col2:
        onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col3:
        deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with col4:
        techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    col1, col2 = st.columns(2)
    with col1:
        streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    with col2:
        streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Billing & Contract
    st.markdown('<div class="section-card">' + icon_title(
        "Billing & Contract",
        '<rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/><line x1="5" y1="15" x2="9" y2="15"/>'
    ), unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col3:
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
    col1, col2 = st.columns(2)
    with col1:
        monthly = st.number_input("Monthly Charges", 0.0, 200.0)
    with col2:
        total = st.number_input("Total Charges", 0.0, 10000.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Backend data assembly (untouched logic) ──
    data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phoneservice],
        "MultipleLines": [multiplelines],
        "InternetService": [internet],
        "OnlineSecurity": [onlinesecurity],
        "OnlineBackup": [onlinebackup],
        "DeviceProtection": [deviceprotection],
        "TechSupport": [techsupport],
        "StreamingTV": [streamingtv],
        "StreamingMovies": [streamingmovies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    X_transformed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    explainer = shap.TreeExplainer(rf_classifier)

    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = rf_model

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Run Churn Prediction")

    if predict_clicked:
        prob = model.predict_proba(data)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card">' + icon_title(
            "Prediction Result",
            '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
        ), unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(prob))
        st.markdown("<br>", unsafe_allow_html=True)
        if prob > 0.6:
            st.error("High Risk Customer — Immediate retention action recommended")
        elif prob > 0.3:
            st.warning("Medium Risk Customer — Monitor and engage proactively")
        else:
            st.success("Low Risk Customer — Customer is likely to remain active")
        st.caption(f"Model used: **{model_choice}**")
        st.markdown('</div>', unsafe_allow_html=True)

        # SHAP Explanation
        st.markdown('<div class="section-card">' + icon_title(
            "Prediction Explanation (SHAP Waterfall)",
            '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
        ), unsafe_allow_html=True)
        X_transformed = preprocessor.transform(data)
        explainer = shap.Explainer(rf_classifier)
        shap_values = explainer(X_transformed_df)
        fig = plt.figure()
        plt.style.use("dark_background")
        fig.patch.set_facecolor("#111827")
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — Model Insights
# ══════════════════════════════════════════════
with tab2:

    preprocessor = rf_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    rf_classifier = rf_model.named_steps["model"]
    feature_importance = rf_classifier.feature_importances_

    # ── Helper: strip sklearn pipeline prefixes (num__, category__) ──
    def clean_name(n):
        n = n.replace("num__", "").replace("category__", "")
        n = n.replace("_", " ")
        return n

    clean_feature_names = [clean_name(f) for f in feature_names]

    feat_imp = pd.DataFrame({
        "feature": clean_feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)

    top_features = feat_imp.head(15)

    # ── 1. Feature Importance bar chart ─────────────────────────────
    st.markdown('<div class="section-card">' + icon_title(
        "Top Churn Drivers — Random Forest Feature Importance",
        '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
    ), unsafe_allow_html=True)
    st.caption("Gini-based global feature importance extracted from the trained Random Forest ensemble.")
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    n = len(top_features)
    bar_colors = ["#63b3ed" if i < n // 3 else "#a78bfa" if i < 2 * n // 3 else "#f687b3"
                  for i in range(n)]
    bars = ax.barh(top_features["feature"], top_features["importance"],
                   color=bar_colors, edgecolor="none", height=0.65)
    ax.invert_yaxis()
    for bar, val in zip(bars, top_features["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color="#cbd5e1", fontsize=10, fontweight="500")
    ax.set_title("Top Drivers of Customer Churn (Random Forest)", color="#e2e8f0",
                 fontsize=15, fontweight="bold", pad=16)
    ax.tick_params(axis="y", colors="#e2e8f0", labelsize=11)
    ax.tick_params(axis="x", colors="#64748b", labelsize=9)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xlabel("Importance Score", color="#64748b", fontsize=10)
    ax.xaxis.grid(True, color=(1, 1, 1, 0.05), linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Cached SHAP Computation on Real Dataset Sample ──────────────
    @st.cache_data(show_spinner="Computing Random Forest SHAP distributions from dataset...")
    def get_rf_shap_data():
        csv_path = os.path.join("dataset", "Telco_Customer_Churn.csv")
        if os.path.exists(csv_path):
            df_real = pd.read_csv(csv_path)
            df_real["TotalCharges"] = pd.to_numeric(df_real["TotalCharges"], errors="coerce")
            # Representative sample across diverse customers
            sample_df = df_real.sample(min(120, len(df_real)), random_state=42)
        else:
            sample_df = pd.DataFrame()

        X_sample_transformed = preprocessor.transform(sample_df)
        raw_feature_names = preprocessor.get_feature_names_out()
        cols = [clean_name(f) for f in raw_feature_names]
        X_sample_df = pd.DataFrame(X_sample_transformed, columns=cols)
        explainer = shap.TreeExplainer(rf_classifier)
        shap_vals = explainer(X_sample_df)
        return shap_vals, cols

    shap_values, clean_cols = get_rf_shap_data()

    # ── 2. SHAP Beeswarm ─────────────────────────────────────────────
    st.markdown('<div class="section-card">' + icon_title(
        "SHAP Summary — Beeswarm Distribution (Random Forest)",
        '<circle cx="6" cy="12" r="2"/><circle cx="12" cy="7" r="2"/><circle cx="18" cy="14" r="2"/><circle cx="12" cy="17" r="2"/><circle cx="20" cy="8" r="2"/><circle cx="4" cy="7" r="2"/><circle cx="15" cy="11" r="2"/>'
    ), unsafe_allow_html=True)
    st.caption("Distribution of SHAP impact values for the Random Forest model across a representative sample of customers. Red/Pink indicates higher feature values, blue indicates lower feature values.")
    shap_cmap = mcolors.LinearSegmentedColormap.from_list(
        "shap_custom", ["#38bdf8", "#818cf8", "#f472b6"], N=256
    )
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    shap.plots.beeswarm(
        shap_values[:, :, 1],
        max_display=15,
        color=shap_cmap,
        show=False,
        plot_size=None,
    )
    cur_ax = plt.gca()
    cur_ax.set_facecolor("#111827")
    cur_ax.tick_params(axis="y", colors="#e2e8f0", labelsize=11)
    cur_ax.tick_params(axis="x", colors="#94a3b8", labelsize=10)
    cur_ax.xaxis.label.set_color("#94a3b8")
    cur_ax.xaxis.label.set_fontsize(11)
    cur_ax.spines[["top", "right"]].set_visible(False)
    cur_ax.spines[["left", "bottom"]].set_color("#374151")
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)
    plt.style.use("default")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. SHAP Bar chart (custom, not shap.plots.bar) ───────────────
    st.markdown('<div class="section-card">' + icon_title(
        "SHAP Global Impact — Mean |SHAP| (Random Forest)",
        '<line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="16" y2="12"/><line x1="4" y1="18" x2="12" y2="18"/>'
    ), unsafe_allow_html=True)
    st.caption("Mean absolute SHAP values indicating overall magnitude of influence on churn predictions.")
    mean_shap = np.abs(shap_values[:, :, 1].values).mean(axis=0)
    shap_imp = pd.DataFrame({"feature": clean_cols, "shap": mean_shap})\
                 .sort_values("shap", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    n2 = len(shap_imp)
    shap_colors = plt.cm.cool(np.linspace(0.2, 0.9, n2))
    bars2 = ax.barh(shap_imp["feature"], shap_imp["shap"],
                    color=shap_colors, edgecolor="none", height=0.65)
    for bar, val in zip(bars2, shap_imp["shap"]):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"+{val:.3f}", va="center", ha="left",
                color="#cbd5e1", fontsize=10, fontweight="500")
    ax.set_title("Mean |SHAP Value| per Feature (Random Forest)", color="#e2e8f0",
                 fontsize=15, fontweight="bold", pad=16)
    ax.tick_params(axis="y", colors="#e2e8f0", labelsize=11)
    ax.tick_params(axis="x", colors="#64748b", labelsize=9)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xlabel("Mean |SHAP value|", color="#64748b", fontsize=10)
    ax.xaxis.grid(True, color=(1, 1, 1, 0.05), linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    performance_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "ROC AUC": [0.86, 0.85],
        "F1 Score": [0.64, 0.65],
        "Precision": [0.52, 0.56],
        "Recall": [0.84, 0.78]
    })

    st.markdown('<div class="section-card">' + icon_title(
        "Model Performance Comparison",
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
    ), unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("LR — ROC AUC", "0.86")
    with col2:
        st.metric("LR — F1 Score", "0.64")
    with col3:
        st.metric("RF — ROC AUC", "0.85")
    with col4:
        st.metric("RF — F1 Score", "0.65")
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">' + icon_title(
        "Business Insights & Strategic Recommendations",
        '<path d="M9 18h6m-4 4h2m-4-7a7 7 0 1 1 6 0c-.8.8-1 1.8-1 3H10c0-1.2-.2-2.2-1-3z"/>'
    ), unsafe_allow_html=True)
    st.markdown("""
### Key Drivers of Customer Churn

Based on the model analysis and feature importance results, several factors significantly influence customer churn:

**1. Customer Tenure**
- Customers with shorter tenure are much more likely to churn.
- New customers have a higher probability of leaving compared to long-term subscribers.

**2. Contract Type**
- Customers on **month-to-month contracts** show the highest churn risk.
- Long-term contracts such as **one-year or two-year agreements significantly reduce churn**.

**3. Monthly Charges**
- Higher monthly charges correlate with increased churn probability.
- Customers paying more are more likely to switch providers if they perceive better value elsewhere.

**4. Internet Service Type**
- Customers using **fiber optic internet services** show relatively higher churn rates compared to DSL users.

**5. Lack of Value-Added Services**
- Customers without services like **online security, tech support, or device protection** are more likely to churn.

---

### Business Recommendations

• Encourage **long-term contracts** through discounts or loyalty rewards.  
• Offer **bundled services (security, tech support)** to increase customer retention.  
• Provide **special retention offers for high-charge customers** to reduce churn risk.  
• Focus retention campaigns on **new customers with low tenure**.
""")
    st.markdown('</div>', unsafe_allow_html=True)
