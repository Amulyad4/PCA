import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page config
st.set_page_config(page_title="PCA Visualization", page_icon="📊", layout="wide")

st.title("📊 Principal Component Analysis (PCA)")

# Upload dataset
uploaded = st.file_uploader("Upload CSV/Excel File", type=["csv", "xlsx", "xls"])

if uploaded:
    # Read file
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Select numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns")
        st.stop()

    st.subheader("Select Features")
    feature_cols = st.multiselect(
        "Choose numeric columns",
        options=num_cols,
        default=num_cols
    )

    if len(feature_cols) < 2:
        st.warning("Select at least 2 columns")
        st.stop()

    # Data
    X = df[feature_cols].dropna()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    n_components = st.slider("Number of Components", 2, min(len(feature_cols), 10), 2)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    st.divider()

    # ==========================
    # Visualization 1
    # Explained Variance
    # ==========================
    st.subheader("1️⃣ Explained Variance Ratio")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_
    )
    ax1.set_xlabel("Principal Components")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("Explained Variance Ratio")
    st.pyplot(fig1)

    # ==========================
    # Visualization 2
    # 2D PCA Scatter
    # ==========================
    st.subheader("2️⃣ 2D PCA Scatter Plot")

    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("2D PCA Projection")
    st.pyplot(fig2)

    # ==========================
    # Visualization 3
    # 3D PCA Scatter
    # ==========================
    if len(feature_cols) >= 3:
        st.subheader("3️⃣ Interactive 3D PCA Scatter")

        pca_3d = PCA(n_components=3)
        X_3d = pca_3d.fit_transform(X_scaled)

        fig3 = px.scatter_3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            labels={"x": "PC1", "y": "PC2", "z": "PC3"},
            title="3D PCA Projection"
        )

        st.plotly_chart(fig3, use_container_width=True)