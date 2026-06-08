import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, pickle, datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PCA · Dimensionality Reduction", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#080b12; color:#dde1f0; }
.main .block-container { background:#080b12; padding:2rem 3rem; max-width:1400px; }
[data-testid="stSidebar"] { background:#0c0f1a !important; border-right:1px solid #1a2040; }
[data-testid="stSidebar"] .block-container { padding:1.4rem 1rem; }

.hero {
    background:linear-gradient(135deg,#0a0d1e 0%,#0f1428 60%,#080b18 100%);
    border:1px solid #1a3040; border-radius:14px;
    padding:2rem 2.5rem; margin-bottom:1.8rem; position:relative; overflow:hidden;
}
.hero::after {
    content:''; position:absolute; bottom:-80px; right:-40px;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(56,189,248,.09) 0%,transparent 70%);
    border-radius:50%;
}
.hero h1 {
    font-size:2.3rem; font-weight:800;
    background:linear-gradient(90deg,#38bdf8 0%,#2dd4bf 45%,#34d399 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 .3rem 0; font-family:'JetBrains Mono',monospace; letter-spacing:-1px;
}
.hero p { color:#4a5580; font-size:.9rem; margin:0; }

.ct {
    font-size:.68rem; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:#38bdf8; margin-bottom:.5rem;
    font-family:'JetBrains Mono',monospace;
}

.metric-row { display:flex; gap:.9rem; flex-wrap:wrap; margin-bottom:1.4rem; }
.metric-box {
    flex:1; min-width:120px; background:#0c0f1a; border:1px solid #1a2040;
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
}
.metric-box .val {
    font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace;
    background:linear-gradient(90deg,#38bdf8,#2dd4bf);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.metric-box .lbl { font-size:.68rem; color:#3a4560; text-transform:uppercase;
                   letter-spacing:.09em; margin-top:.2rem; }

.train-wrap {
    background:#0c0f1a; border:1px solid #1a3040; border-radius:10px;
    padding:1.1rem 1.6rem; margin:1.1rem 0; display:flex;
    align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
}
.train-info { color:#4a5580; font-size:.84rem; }
.train-info strong { color:#8892b0; }

.pred-card {
    background:#0c0f1a; border:1px solid #1a3040; border-radius:10px;
    padding:1.4rem 1.8rem; margin-top:1rem;
}
.pred-badge {
    display:inline-block; padding:.35rem 1.1rem; border-radius:6px;
    font-family:'JetBrains Mono',monospace; font-size:.9rem; font-weight:600;
    background:linear-gradient(135deg,#0369a1,#0d9488); color:#fff; margin-top:.5rem;
}

.stButton > button {
    background:linear-gradient(135deg,#0369a1,#0d9488) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.82rem !important;
    font-weight:600 !important; letter-spacing:.05em !important;
    padding:.55rem 1.5rem !important; transition:opacity .2s !important;
}
.stButton > button:hover { opacity:.82 !important; }

[data-testid="stSlider"] > div > div > div { background:#0ea5e9 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color:#0c0f1a !important; border-color:#1a2040 !important; color:#dde1f0 !important;
}
[data-testid="stFileUploader"] section {
    background:#0c0f1a !important; border:1px dashed #1a3040 !important; border-radius:10px !important;
}
[data-testid="stDataFrame"] { border:1px solid #1a2040; border-radius:8px; }
hr { border-color:#1a2040 !important; }

.saved-banner {
    background:#071318; border:1px solid #0c3040; border-radius:10px;
    padding:1rem 1.5rem; margin-top:1.5rem;
    font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#38bdf8;
}
.saved-banner span { color:#3a4560; }

.js-plotly-plot .plotly .modebar { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Dirs ──────────────────────────────────────────────────────────────────────
RAW_DIR, OUT_DIR, MODEL_DIR = "data/raw", "data/pca", "models"
for d in [RAW_DIR, OUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["pca_components","pca_X","pca_Xr","pca_feature_cols",
            "pca_df_result","pca_n","pca_scale","pca_model_path",
            "pca_out_csv","pca_model","pca_pred_result","pca_pred_vals"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
PALETTE = ["#38bdf8","#2dd4bf","#818cf8","#f472b6","#34d399",
           "#fb923c","#facc15","#a78bfa","#4ade80","#f87171"]

def mpl_dark(figsize=(8,5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#080b12")
    ax.set_facecolor("#0c0f1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2040")
    ax.tick_params(colors="#4a5580", labelsize=8)
    ax.xaxis.label.set_color("#4a5580"); ax.yaxis.label.set_color("#4a5580")
    ax.title.set_color("#c0c8e0")
    return fig, ax

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔬 PCA</h1>
  <p>Principal Component Analysis · Variance explained · Loadings · Interactive 3-D · Transform</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"],
                                label_visibility="collapsed")
    st.divider()
    st.markdown("**Model Parameters**")
    n_components  = st.slider("Number of components", 2, 20, 2)
    scale_data    = st.checkbox("Standardise features", value=True)
    st.divider()
    st.markdown("**Colour by**")
    color_col_choice = st.text_input("Column name to colour scatter (optional)",
                                     placeholder="e.g. species, label …")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_raw = None
if uploaded:
    raw_path = os.path.join(RAW_DIR, uploaded.name)
    with open(raw_path, "wb") as f: f.write(uploaded.getbuffer())
    df_raw = pd.read_csv(raw_path) if uploaded.name.endswith(".csv") \
             else pd.read_excel(raw_path)
    st.success(f"✔ **{uploaded.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

if df_raw is None:
    st.info("⬆ Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) < 2:
    st.error("Need at least 2 numeric columns."); st.stop()

with st.expander("🔍 Preview data", expanded=False):
    st.dataframe(df_raw.head(200), use_container_width=True)

feature_cols = st.multiselect("", options=num_cols, default=num_cols,
                              placeholder="Select feature columns…",
                              label_visibility="collapsed")
if len(feature_cols) < 2:
    st.warning("Select at least 2 feature columns."); st.stop()

n_comp_capped = min(n_components, len(feature_cols), len(df_raw.dropna(subset=feature_cols)))

X_raw  = df_raw[feature_cols].dropna().values
sc_tmp = StandardScaler()
X      = sc_tmp.fit_transform(X_raw) if scale_data else X_raw.astype(float)

# ── Train button ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="train-wrap">
  <div class="train-info">
    <strong>n_components={n_comp_capped}</strong> &nbsp;·&nbsp;
    <strong>{len(X):,}</strong> samples &nbsp;·&nbsp;
    <strong>{len(feature_cols)}</strong> features &nbsp;·&nbsp;
    scaled={scale_data}
  </div>
""", unsafe_allow_html=True)
train_btn = st.button("🔬 Fit PCA", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# ── Fit ───────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Computing principal components …"):
        pca = PCA(n_components=n_comp_capped)
        Xr  = pca.fit_transform(X)

    model_path = os.path.join(MODEL_DIR, f"pca_n{n_comp_capped}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": pca, "scaler": sc_tmp,
                     "features": feature_cols, "scaled": scale_data,
                     "n_components": n_comp_capped}, f)

    valid_idx = df_raw[feature_cols].dropna().index
    df_result = df_raw.loc[valid_idx].copy()
    for i in range(n_comp_capped):
        df_result[f"PC{i+1}"] = np.round(Xr[:, i], 5)
    out_csv = os.path.join(OUT_DIR, f"pca_n{n_comp_capped}.csv")
    df_result.to_csv(out_csv, index=False)

    st.session_state.pca_components  = pca
    st.session_state.pca_X           = X
    st.session_state.pca_Xr          = Xr
    st.session_state.pca_feature_cols= feature_cols
    st.session_state.pca_df_result   = df_result
    st.session_state.pca_n           = n_comp_capped
    st.session_state.pca_scale       = scale_data
    st.session_state.pca_model_path  = model_path
    st.session_state.pca_out_csv     = out_csv
    st.session_state.pca_model       = pca
    st.session_state.pca_pred_result = None

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.pca_components is None:
    st.stop()

pca          = st.session_state.pca_model
X_ss         = st.session_state.pca_X
Xr           = st.session_state.pca_Xr
feature_cols = st.session_state.pca_feature_cols
df_result    = st.session_state.pca_df_result
n_ss         = st.session_state.pca_n
model_path   = st.session_state.pca_model_path
out_csv      = st.session_state.pca_out_csv

evr          = pca.explained_variance_ratio_
cumvar       = np.cumsum(evr)
n_90         = int(np.searchsorted(cumvar, 0.90)) + 1

st.divider()
st.markdown("### 📊 Results")

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{n_ss}</div><div class="lbl">Components</div></div>
  <div class="metric-box"><div class="val">{cumvar[-1]*100:.1f}%</div><div class="lbl">Variance Explained</div></div>
  <div class="metric-box"><div class="val">{evr[0]*100:.1f}%</div><div class="lbl">PC1 Variance</div></div>
  <div class="metric-box"><div class="val">{n_90}</div><div class="lbl">PCs for 90%</div></div>
  <div class="metric-box"><div class="val">{len(X_ss):,}</div><div class="lbl">Samples</div></div>
  <div class="metric-box"><div class="val">{len(feature_cols)}</div><div class="lbl">Features</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 1 — Scree plot (bar + cumulative line)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ct">Scree Plot — Explained Variance per Component</div>',
            unsafe_allow_html=True)

fig_sc, ax_sc = mpl_dark((12, 3.5))
x_idx = np.arange(1, n_ss + 1)
ax_sc.bar(x_idx, evr * 100,
          color=[PALETTE[i % len(PALETTE)] for i in range(n_ss)],
          width=0.55, alpha=0.88, label="Individual")
ax2 = ax_sc.twinx()
ax2.plot(x_idx, cumvar * 100, color="#f472b6", lw=2,
         marker="o", markersize=4, label="Cumulative")
ax2.axhline(90, color="#f472b6", lw=0.8, linestyle="--", alpha=0.5)
ax2.set_ylabel("Cumulative %", color="#f472b6", fontsize=8)
ax2.tick_params(colors="#f472b6", labelsize=7)
ax2.set_ylim(0, 105)
ax_sc.set_xlabel("Principal Component"); ax_sc.set_ylabel("Variance Explained (%)")
ax_sc.set_xticks(x_idx)
ax_sc.set_xticklabels([f"PC{i}" for i in x_idx], fontsize=8, color="#c0c8e0")
ax_sc.set_title("Scree Plot", fontsize=10)
lines1, labels1 = ax_sc.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_sc.legend(lines1 + lines2, labels1 + labels2,
             fontsize=7, facecolor="#0c0f1a", edgecolor="#1a2040", labelcolor="#c0c8e0")
plt.tight_layout(); st.pyplot(fig_sc); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 2 — Loadings heatmap
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ct">Component Loadings Heatmap</div>', unsafe_allow_html=True)

loadings = pca.components_   # (n_components, n_features)
fig_lh, ax_lh = plt.subplots(figsize=(12, max(2.5, n_ss * 0.55 + 1.2)),
                              facecolor="#080b12")
ax_lh.set_facecolor("#080b12")
im = ax_lh.imshow(loadings, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_lh, fraction=0.02, pad=0.01, label="Loading")
ax_lh.set_xticks(range(len(feature_cols)))
ax_lh.set_xticklabels(feature_cols, rotation=40, ha="right",
                       fontsize=8, color="#c0c8e0")
ax_lh.set_yticks(range(n_ss))
ax_lh.set_yticklabels([f"PC{i+1}" for i in range(n_ss)],
                       fontsize=8, color="#c0c8e0")
ax_lh.set_title("Feature Loadings per Principal Component",
                color="#c0c8e0", fontsize=10)
for sp in ax_lh.spines.values(): sp.set_edgecolor("#1a2040")
ax_lh.tick_params(colors="#4a5580")
# annotate cells
for i in range(n_ss):
    for j in range(len(feature_cols)):
        ax_lh.text(j, i, f"{loadings[i,j]:.2f}",
                   ha="center", va="center",
                   fontsize=6.5, color="#dde1f0",
                   fontfamily="monospace")
plt.tight_layout(); st.pyplot(fig_lh); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 3 — 2-D PC scatter (PC1 vs PC2)  +  optional colour
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="ct">PC1 vs PC2 Scatter</div>', unsafe_allow_html=True)

# resolve colour column
color_vals  = None
color_label = None
if color_col_choice and color_col_choice in df_result.columns:
    raw_cv = df_result[color_col_choice].values
    if pd.api.types.is_numeric_dtype(raw_cv):
        color_vals  = raw_cv.astype(float)
        color_label = color_col_choice
    else:
        cats        = pd.Categorical(raw_cv)
        color_vals  = cats.codes.astype(float)
        color_label = color_col_choice

fig_2d, ax_2d = mpl_dark((12, 5))
if color_vals is not None:
    sc = ax_2d.scatter(Xr[:, 0], Xr[:, 1],
                       c=color_vals, cmap="plasma",
                       s=14, alpha=0.65, linewidths=0)
    plt.colorbar(sc, ax=ax_2d, label=color_label, fraction=0.02, pad=0.01)
else:
    ax_2d.scatter(Xr[:, 0], Xr[:, 1],
                  color=PALETTE[0], s=14, alpha=0.55, linewidths=0)

ax_2d.axhline(0, color="#1a2040", lw=0.8); ax_2d.axvline(0, color="#1a2040", lw=0.8)
ax_2d.set_xlabel(f"PC1  ({evr[0]*100:.1f}% var)")
ax_2d.set_ylabel(f"PC2  ({evr[1]*100:.1f}% var)" if n_ss > 1 else "PC2")
ax_2d.set_title("Principal Component Scatter", fontsize=10)
plt.tight_layout(); st.pyplot(fig_2d); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIZ 4 — Interactive 3-D (PC1 / PC2 / PC3)
# ══════════════════════════════════════════════════════════════════════════════
if n_ss >= 3:
    st.markdown('<div class="ct">Interactive 3-D Scatter — PC1 × PC2 × PC3 (rotate me!)</div>',
                unsafe_allow_html=True)
    st.caption("Drag to rotate · Scroll to zoom · Double-click to reset")

    marker_color = color_vals if color_vals is not None else PALETTE[0]
    colorscale   = "Plasma" if color_vals is not None else None

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=Xr[:, 0], y=Xr[:, 1], z=Xr[:, 2],
        mode="markers",
        marker=dict(
            size=3,
            color=marker_color,
            colorscale=colorscale if colorscale else [[0, PALETTE[0]], [1, PALETTE[0]]],
            opacity=0.70,
            line=dict(width=0),
            colorbar=dict(title=color_label, thickness=10,
                          tickfont=dict(color="#4a5580", size=9))
            if color_vals is not None else dict()
        ),
        text=[f"PC1:{Xr[i,0]:.2f} PC2:{Xr[i,1]:.2f} PC3:{Xr[i,2]:.2f}"
              for i in range(len(Xr))],
        hoverinfo="text", name="Samples"
    )])

    fig_3d.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0c0f1a",
        font=dict(family="JetBrains Mono", color="#8892b0", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        height=560,
        title=dict(text=f"3-D PCA Scatter  ({cumvar[2]*100:.1f}% total variance)",
                   font=dict(color="#8892b0", size=13)),
        scene=dict(
            bgcolor="#0c0f1a",
            xaxis=dict(backgroundcolor="#0c0f1a", gridcolor="#1a2040",
                       showbackground=True, tickfont=dict(color="#4a5580"),
                       title=f"PC1 ({evr[0]*100:.1f}%)"),
            yaxis=dict(backgroundcolor="#0c0f1a", gridcolor="#1a2040",
                       showbackground=True, tickfont=dict(color="#4a5580"),
                       title=f"PC2 ({evr[1]*100:.1f}%)"),
            zaxis=dict(backgroundcolor="#0c0f1a", gridcolor="#1a2040",
                       showbackground=True, tickfont=dict(color="#4a5580"),
                       title=f"PC3 ({evr[2]*100:.1f}%)"),
        ),
        legend=dict(bgcolor="#0c0f1a", bordercolor="#1a2040", borderwidth=1,
                    font=dict(color="#8892b0", size=10)),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🏷️ Transformed Data")
st.markdown('<div class="ct">Original dataset with PC coordinates appended</div>',
            unsafe_allow_html=True)
st.dataframe(df_result, use_container_width=True, height=320)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM NEW POINT
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="ct">Project a new data point into PCA space</div>',
            unsafe_allow_html=True)
with st.expander("🔮 Enter feature values", expanded=True):
    col_inputs = st.columns(min(len(feature_cols), 4))
    new_vals   = {}
    for i, col in enumerate(feature_cols):
        col_min  = float(df_raw[col].min())
        col_max  = float(df_raw[col].max())
        col_mean = float(df_raw[col].mean())
        prev     = (st.session_state.pca_pred_vals or {}).get(col, round(col_mean, 4))
        new_vals[col] = col_inputs[i % len(col_inputs)].number_input(
            col, value=prev, min_value=col_min, max_value=col_max,
            step=round((col_max - col_min) / 100, 6), format="%.4f",
            key=f"pca_pred_{col}")

    if st.button("🔍 Project Point", key="pca_predict"):
        new_row    = np.array([[new_vals[c] for c in feature_cols]])
        new_scaled = sc_tmp.transform(new_row) if st.session_state.pca_scale else new_row
        projected  = pca.transform(new_scaled)[0]
        st.session_state.pca_pred_result = projected
        st.session_state.pca_pred_vals   = new_vals

    if st.session_state.pca_pred_result is not None:
        proj = st.session_state.pca_pred_result
        badges = "".join([
            f'<span style="background:#0c0f1a; border:1px solid #1a3040;'
            f'border-radius:6px; padding:.25rem .7rem;'
            f'font-family:JetBrains Mono,monospace; font-size:.78rem;'
            f'color:{PALETTE[i % len(PALETTE)]};">'
            f'PC{i+1}: {v:.4f}</span>'
            for i, v in enumerate(proj)
        ])
        st.markdown(f"""
        <div class="pred-card">
          <div style="color:#3a4560;font-size:.78rem;font-family:'JetBrains Mono',monospace;
                      text-transform:uppercase;letter-spacing:.1em;">Projected coordinates</div>
          <div class="pred-badge">🔬 {n_ss}-D projection ready</div>
          <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.9rem;">{badges}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADS + SAVED BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇ Download Transformed CSV",
                       df_result.to_csv(index=False).encode(),
                       file_name=f"pca_n{n_ss}.csv",
                       mime="text/csv", use_container_width=True)
with c2:
    with open(model_path, "rb") as f:
        st.download_button("⬇ Download Model (.pkl)", f.read(),
                           file_name=os.path.basename(model_path),
                           mime="application/octet-stream",
                           use_container_width=True)

abs_model = os.path.abspath(model_path)
abs_csv   = os.path.abspath(out_csv)
ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div class="saved-banner">
  ✔ Model saved to &nbsp;<strong>{abs_model}</strong><br>
  ✔ Transformed data saved to &nbsp;<strong>{abs_csv}</strong><br>
  <span>Saved at {ts} &nbsp;·&nbsp; n_components={n_ss}
  &nbsp;·&nbsp; variance explained={cumvar[-1]*100:.2f}%
  &nbsp;·&nbsp; PCs for 90% variance={n_90}</span>
</div>
""", unsafe_allow_html=True)