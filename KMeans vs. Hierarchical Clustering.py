import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Streamlit app
st.title("KMeans vs. Hierarchical Clustering Comparison")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Let user select two numerical columns for clustering
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_columns) < 2:
        st.error("Dataset must have at least two numerical columns for clustering.")
    else:
        st.subheader("Select Columns for Clustering")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis column", numerical_columns, index=0)
        with col2:
            y_col = st.selectbox("Y-axis column", numerical_columns, index=1 if len(numerical_columns) > 1 else 0)

        # Preprocess dataset
        df_cluster = df[[x_col, y_col]].copy()
        df_cluster.columns = ['x', 'y']  # Rename for consistency

        # Handle missing values
        if df_cluster.isna().any().any():
            st.warning("Missing values detected. Filling with column means.")
            df_cluster['x'] = df_cluster['x'].fillna(df_cluster['x'].mean())
            df_cluster['y'] = df_cluster['y'].fillna(df_cluster['y'].mean())

        # Scale features
        X = df_cluster[['x', 'y']].values
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
        df_scaled = pd.DataFrame(X_scaled, columns=['x_scaled', 'y_scaled'])

        # --- KMeans Implementation ---
        st.subheader("KMeans Clustering (k=3 and k=4)")

        def kmeans(X, k, max_iters=100):
            idx = np.random.choice(X.shape[0], k, replace=False)
            centroids = X[idx]
            for _ in range(max_iters):
                distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
                labels = np.argmin(distances, axis=0)
                new_centroids = np.array([
                    X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
                    for i in range(k)
                ])
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids
            return labels, centroids

        k_values = [3, 4]
        kmeans_df = df_scaled.copy()
        col_km1, col_km2 = st.columns(2)

        for i, k in enumerate(k_values):
            labels, centroids = kmeans(X_scaled, k)
            kmeans_df[f'kmeans_cluster_k{k}'] = labels

            with [col_km1, col_km2][i]:
                fig_km, ax_km = plt.subplots(figsize=(6, 5))
                for c in range(k):
                    cluster_data = kmeans_df[kmeans_df[f'kmeans_cluster_k{k}'] == c]
                    ax_km.scatter(cluster_data['x_scaled'], cluster_data['y_scaled'], label=f'KMeans Cluster {c}', alpha=0.6)
                ax_km.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
                ax_km.set_title(f'KMeans with k={k}')
                ax_km.set_xlabel("Scaled " + x_col)
                ax_km.set_ylabel("Scaled " + y_col)
                ax_km.legend()
                ax_km.grid(True)
                st.pyplot(fig_km)

        # --- Hierarchical Clustering Implementation ---
        st.subheader("Hierarchical Clustering")
        num_clusters_hc = st.slider("Number of clusters for Hierarchical Clustering", min_value=2, max_value=10, value=3)
        linkage_method = st.selectbox("Linkage Method", ['ward', 'average', 'complete', 'single'])

        agg_clustering = AgglomerativeClustering(n_clusters=num_clusters_hc, linkage=linkage_method)
        agg_labels = agg_clustering.fit_predict(X_scaled)
        hierarchical_df = df_scaled.copy()
        hierarchical_df['hierarchical_cluster'] = agg_labels

        fig_hc, ax_hc = plt.subplots(figsize=(6, 5))
        for c in np.unique(agg_labels):
            cluster_data = hierarchical_df[hierarchical_df['hierarchical_cluster'] == c]
            ax_hc.scatter(cluster_data['x_scaled'], cluster_data['y_scaled'], label=f'Cluster {c}', alpha=0.6)
        ax_hc.set_title(f'Hierarchical Clustering (k={num_clusters_hc}, Linkage={linkage_method})')
        ax_hc.set_xlabel("Scaled " + x_col)
        ax_hc.set_ylabel("Scaled " + y_col)
        ax_hc.legend()
        ax_hc.grid(True)
        st.pyplot(fig_hc)

        # --- Dendrogram ---
        st.subheader("Hierarchical Clustering Dendrogram")

        dendro_threshold = st.slider("Max rows for Dendrogram display", min_value=10, max_value=500, value=50)

        if df.shape[0] <= dendro_threshold:
            linked = linkage(X_scaled, method=linkage_method)
            fig_dendrogram, ax_dendrogram = plt.subplots(figsize=(10, 5))
            dendrogram(linked,
                       orientation='top',
                       distance_sort='descending',
                       show_leaf_counts=True,
                       ax=ax_dendrogram)
            ax_dendrogram.set_title(f"Dendrogram (Linkage={linkage_method})")
            ax_dendrogram.set_xlabel("Data Points")
            ax_dendrogram.set_ylabel("Cluster Distance")
            st.pyplot(fig_dendrogram)
        else:
            st.info(f"Dendrogram not shown for datasets with more than {dendro_threshold} rows. You can adjust the threshold above.")

else:
    st.info("Please upload a CSV file to proceed.")
